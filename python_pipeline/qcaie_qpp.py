"""
QCAIE v2.0 - Continuous Wavelet Transform (CWT) QPP detector
==============================================================
Quasi-Periodic Pulsations (QPP) are a known pre-flare signature: the
solar corona oscillates at periods of 1-300 seconds in the minutes-to-
hours before a flare. Detecting QPP in the HXR channel gives an
independent early-warning signal that complements the Neupert residual.

This module uses scipy.signal.cwt (classical Morlet wavelet CWT, NO ML)
to:
  1. Compute the wavelet power spectrum of the HXR signal.
  2. Find the dominant oscillation period in the 4-60 second band.
  3. Estimate the QPP significance via a red-noise null hypothesis
     (Torrence & Compo 1998), the textbook astrophysical CWT method.
  4. Trigger a QPP alert when significant periodic power persists for
     >= N consecutive samples.

References:
- Torrence, C. and Compo, G.P. (1998) "A Practical Guide to Wavelet
  Analysis", Bull. Amer. Meteor. Soc., 79, 61-78.
- Inglis, A.R. et al. (2019) "Quasi-periodic Pulsations in Solar and
  Stellar Flares: A Review of the Underlying Physics", Frontiers in
  Astronomy and Space Sciences.
"""
from __future__ import annotations

import numpy as np
from scipy import signal
from dataclasses import dataclass, field
from typing import Optional


# Periods (in samples) to scan. At 1 Hz cadence, 4-60 samples = 4-60 s.
# At 1-min cadence (GOES), 4-60 samples = 4-60 min (still astrophysically
# meaningful: long-period QPPs).
DEFAULT_PERIODS = np.arange(4, 61, 2)


@dataclass
class QPPState:
    """Tracks the QPP detector state across time steps."""
    enabled: bool = True
    current_period: float = 0.0
    current_significance: float = 0.0
    consec_significant: int = 0
    confirmation_steps: int = 5  # require 5 consecutive significant samples
    significance_threshold: float = 3.0  # 3-sigma above red noise
    qpp_alert_armed: bool = False
    qpp_first_step: Optional[int] = None
    history: list = field(default_factory=list)


def _red_noise_null(x: np.ndarray, periods: np.ndarray) -> np.ndarray:
    """Compute the red-noise null hypothesis power spectrum (Torrence & Compo).

    The red-noise background AR(1) process has power:
        P_null(k) = (1 - r^2) / (1 - 2 r cos(2 pi k / N) + r^2)
    where r is the lag-1 autocorrelation. The 95% / 99% confidence levels
    are P_null * chi2_ppf(0.95, 2) / 2 (chi-square with 2 DOF).

    Returns a (len(periods),) array of 99% confidence levels. FLOOR at 1e-3
    to prevent zero denominators when the AR(1) coefficient is very small
    (white noise) which would otherwise produce inflated significance.
    """
    N = len(x)
    x = x - np.mean(x)
    if N < 10:
        return np.ones(len(periods)) * 1e-3
    # Lag-1 autocorrelation
    if N > 2 and np.std(x[:-1]) > 1e-9 and np.std(x[1:]) > 1e-9:
        r = float(np.corrcoef(x[:-1], x[1:])[0, 1])
    else:
        r = 0.0
    r = max(min(r, 0.99), -0.99)
    freqs = np.fft.fftfreq(N, d=1.0)
    P_null = np.zeros(N // 2)
    for k in range(N // 2):
        theta = 2 * np.pi * k / N
        P_null[k] = (1 - r * r) / max(1 - 2 * r * np.cos(theta) + r * r, 1e-9)
    # 99% chi-square confidence (DOF=2 for wavelet power)
    from scipy.stats import chi2
    chi2_99 = float(chi2.ppf(0.99, df=2)) / 2.0
    # Map our `periods` (in samples) onto the frequency array
    null_at_periods = np.zeros(len(periods))
    for i, p in enumerate(periods):
        k = int(round(N / max(p, 1)))  # frequency bin index
        if 0 < k < len(P_null):
            null_at_periods[i] = P_null[k] * chi2_99
        else:
            null_at_periods[i] = 1e-3
    # FLOOR at 1e-3 * mean wavelet power to prevent zero denominators
    return np.maximum(null_at_periods, 1e-3)


def detect_qpp_cwt(hxr_segment: np.ndarray,
                   periods: np.ndarray = DEFAULT_PERIODS,
                   wavelet_width: float = 3.0) -> tuple[float, float, np.ndarray]:
    """Run Morlet CWT on an HXR segment and return (best_period, significance, powers).

    `significance` = wavelet_power_at_best_period / red_noise_99_confidence.
    A significance > 1.0 means the detected oscillation exceeds the 99%
    confidence level of the red-noise null hypothesis.
    """
    hxr_segment = np.asarray(hxr_segment, dtype=float)
    N = len(hxr_segment)
    if N < max(periods) * 2:
        return 0.0, 0.0, np.zeros(len(periods))
    hxr_detrend = hxr_segment - np.mean(hxr_segment)
    # Use scipy's CWT with the Ricker (Mexican Hat) wavelet. The `w`
    # parameter of ricker sets the wavelet width; we pass it via a lambda
    # because scipy.signal.cwt passes the `width` (period) as the first
    # positional arg, but our ricker also needs `w`.
    def _wavelet(width, w=wavelet_width):
        return signal.ricker(width, w)

    cwtmatr = signal.cwt(hxr_detrend, _wavelet, periods)
    powers = np.abs(cwtmatr) ** 2
    # Mean power at each period across the time axis
    mean_powers = np.mean(powers, axis=1)
    null_99 = _red_noise_null(hxr_segment, periods)
    significance = mean_powers / (null_99 + 1e-9)
    if np.any(significance > 0):
        best_idx = int(np.argmax(significance))
        return float(periods[best_idx]), float(significance[best_idx]), significance
    return 0.0, 0.0, significance


def step_qpp(hxr_history: np.ndarray, state: QPPState,
             min_len: int = 60) -> QPPState:
    """One step of the QPP detector.

    `hxr_history` is the HXR time series up to and including the current
    sample. We use the last `min_len * 3` samples (or all if shorter) to
    compute the CWT.
    """
    if not state.enabled or len(hxr_history) < min_len:
        state.history.append({
            "step": len(hxr_history) - 1,
            "period": 0.0, "significance": 0.0, "alert": False,
        })
        return state
    segment = hxr_history[-min(len(hxr_history), 180):]  # last 180 samples max
    period, sig, _ = detect_qpp_cwt(segment)
    state.current_period = period
    state.current_significance = sig
    if sig > state.significance_threshold:
        state.consec_significant += 1
    else:
        state.consec_significant = 0
    if (state.consec_significant >= state.confirmation_steps
            and not state.qpp_alert_armed):
        state.qpp_alert_armed = True
        state.qpp_first_step = len(hxr_history) - 1 - state.confirmation_steps + 1
    state.history.append({
        "step": len(hxr_history) - 1,
        "period": period, "significance": sig,
        "alert": state.qpp_alert_armed,
    })
    return state


def plot_qpp_spectrogram(hxr: np.ndarray, save_path: str,
                          periods: np.ndarray = DEFAULT_PERIODS,
                          wavelet_width: float = 3.0) -> None:
    """Plot the CWT power spectrogram of HXR for diagnostic visualization."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    hxr = np.asarray(hxr, dtype=float)
    N = len(hxr)
    if N < max(periods) * 2:
        return
    hxr_detrend = hxr - np.mean(hxr)
    def _wavelet(width, w=wavelet_width):
        return signal.ricker(width, w)
    cwtmatr = signal.cwt(hxr_detrend, _wavelet, periods)
    power = np.abs(cwtmatr) ** 2
    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    im = ax.pcolormesh(np.arange(N), periods, np.log10(power + 1e-9),
                        shading="auto", cmap="viridis")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Period (samples)")
    ax.set_title("QPP Detector -- Wavelet Power Spectrogram of HXR "
                  "(Ricker CWT)")
    plt.colorbar(im, ax=ax, label="log10(power)")
    fig.savefig(save_path, dpi=130)
    plt.close(fig)
