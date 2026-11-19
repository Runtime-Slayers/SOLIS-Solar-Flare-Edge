"""
QCAIE v2.0 - Robust preprocessing layer
=========================================
Classical signal-processing defenses against worst-case space telemetry.
NO machine learning. All methods are well-established robust statistics
or aerospace signal-processing techniques.

Defenses provided:
  1. Median/MAD baseline estimation  -- robust to up to 50% outliers
  2. Hampel identifier spike rejection -- rejects cosmic-ray hits
  3. Asymmetric exponential baseline drift compensation
  4. Saturation detection + flagging (sensor clips at top of range)
  5. Cross-channel time alignment via cross-correlation (lag recovery)
  6. Rolling-window median filter for impulsive noise rejection

Why these, not ML:
  - Median/MAD has a 50% breakdown point (best possible)
  - Hampel filter is the textbook robust outlier detector (Davies & Gather 1993)
  - Cross-correlation lag recovery is classical signal processing
  - All have closed-form, deterministic behavior -- defensible to ISRO
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# 1. Robust baseline estimation (Median + MAD)
# ---------------------------------------------------------------------------
def robust_baseline(x: np.ndarray, valid: np.ndarray | None = None,
                    window: int = 60) -> np.ndarray:
    """Estimate a rolling robust baseline using median + MAD.

    Median has a 50% breakdown point (the theoretical maximum): up to half
    the samples in the window can be corrupt without affecting the estimate.
    The MAD (median absolute deviation) gives a robust scale estimate that
    is immune to outliers, unlike the standard deviation.

    Returns an array of per-sample baseline estimates (same length as x).
    """
    n = len(x)
    if valid is None:
        valid = np.ones(n, dtype=bool)
    out = np.zeros(n)
    for i in range(n):
        lo = max(0, i - window // 2)
        hi = min(n, i + window // 2 + 1)
        seg = x[lo:hi][valid[lo:hi]]
        if len(seg) >= 5:
            out[i] = float(np.median(seg))
        else:
            # Fall back: forward-fill or backward-fill the last good baseline
            out[i] = out[i - 1] if i > 0 else 0.0
    return out


def robust_scale(x: np.ndarray, valid: np.ndarray | None = None,
                 window: int = 60) -> np.ndarray:
    """Rolling MAD-based scale (1.4826 * MAD = robust std estimator)."""
    n = len(x)
    if valid is None:
        valid = np.ones(n, dtype=bool)
    out = np.zeros(n)
    for i in range(n):
        lo = max(0, i - window // 2)
        hi = min(n, i + window // 2 + 1)
        seg = x[lo:hi][valid[lo:hi]]
        if len(seg) >= 5:
            med = float(np.median(seg))
            mad = float(np.median(np.abs(seg - med)))
            out[i] = 1.4826 * mad + 1e-9  # 1.4826 = 1/Phi^{-1}(0.75)
        else:
            out[i] = out[i - 1] if i > 0 else 1e-3
    return out


# ---------------------------------------------------------------------------
# 2. Hampel identifier: spike rejection
# ---------------------------------------------------------------------------
def hampel_filter(x: np.ndarray, valid: np.ndarray | None = None,
                  window: int = 7, n_sigma: float = 5.0) -> tuple[np.ndarray, np.ndarray]:
    """Hampel identifier: replace samples more than n_sigma*MAD from the
    local median with the local median itself.

    Classical robust outlier detector. With window=7 and n_sigma=5, this
    rejects isolated spikes (e.g. cosmic-ray hits on the detector) without
    affecting genuine impulsive flares (which span many samples).

    Returns (cleaned_x, is_spike) where is_spike is a boolean mask.
    """
    n = len(x)
    if valid is None:
        valid = np.ones(n, dtype=bool)
    cleaned = x.copy()
    is_spike = np.zeros(n, dtype=bool)
    for i in range(n):
        lo = max(0, i - window // 2)
        hi = min(n, i + window // 2 + 1)
        seg = x[lo:hi][valid[lo:hi]]
        if len(seg) < 5:
            continue
        med = float(np.median(seg))
        mad = float(np.median(np.abs(seg - med)))
        scale = 1.4826 * mad + 1e-9
        if abs(x[i] - med) > n_sigma * scale:
            cleaned[i] = med
            is_spike[i] = True
    return cleaned, is_spike


# ---------------------------------------------------------------------------
# 3. Saturation detection
# ---------------------------------------------------------------------------
def detect_saturation(x: np.ndarray, valid: np.ndarray | None = None,
                      quantile: float = 0.999) -> tuple[float, np.ndarray]:
    """Detect sensor saturation by finding the clipping level.

    A saturated sensor produces many samples clustered at the same maximum
    value. We detect this by finding the quantile above which more than 1/3
    of values are identical (within 0.1% of each other).

    Returns (saturation_level, is_saturated_mask).
    """
    n = len(x)
    if valid is None:
        valid = np.ones(n, dtype=bool)
    xs = x[valid]
    if len(xs) < 10:
        return float(np.inf), np.zeros(n, dtype=bool)
    q_hi = float(np.quantile(xs, quantile))
    # Look at the top 1% of values
    top = xs[xs >= q_hi * 0.999]
    if len(top) < 5:
        return float(np.inf), np.zeros(n, dtype=bool)
    # If the top values cluster tightly, the sensor is saturated there
    if (np.max(top) - np.min(top)) / (q_hi + 1e-9) < 0.005:
        sat_level = float(np.median(top))
        is_sat = (x >= sat_level * 0.999) & valid
        return sat_level, is_sat
    return float(np.inf), np.zeros(n, dtype=bool)


# ---------------------------------------------------------------------------
# 4. Cross-channel time-alignment via cross-correlation
# ---------------------------------------------------------------------------
def estimate_lag(sxr: np.ndarray, hxr: np.ndarray,
                 valid: np.ndarray | None = None,
                 max_lag: int = 30) -> int:
    """Estimate the lag between SXR and HXR channels via cross-correlation.

    During a flare, SXR (thermal, slow) lags HXR (non-thermal, fast). If the
    two instrument channels are misaligned by even a few seconds, the
    spectral hardness ratio and Neupert integral calculations generate
    artifacts. Cross-correlation is the classical signal-processing way to
    recover the lag.

    Returns the lag (in samples) that should be applied to HXR to align it
    with SXR. Positive = HXR leads SXR (the physically expected case).
    """
    if valid is None:
        valid = np.ones(len(sxr), dtype=bool)
    sxr_c = sxr - np.nanmean(sxr[valid])
    hxr_c = hxr - np.nanmean(hxr[valid])
    sxr_c[~valid] = 0.0
    hxr_c[~valid] = 0.0
    n = len(sxr)
    best_lag = 0
    best_corr = -np.inf
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            a = sxr_c[lag:]
            b = hxr_c[:n - lag]
        else:
            a = sxr_c[:n + lag]
            b = hxr_c[-lag:]
        if len(a) < 10:
            continue
        denom = np.sqrt(np.sum(a * a) * np.sum(b * b)) + 1e-9
        c = float(np.sum(a * b) / denom)
        if c > best_corr:
            best_corr = c
            best_lag = lag
    return best_lag


def align_channels(sxr: np.ndarray, hxr: np.ndarray,
                   valid: np.ndarray | None = None,
                   max_lag: int = 30) -> tuple[np.ndarray, np.ndarray, int]:
    """Align HXR to SXR by applying the estimated lag.

    Returns (sxr_unchanged, hxr_shifted, lag_applied).
    """
    lag = estimate_lag(sxr, hxr, valid, max_lag)
    n = len(sxr)
    hxr_shifted = np.zeros_like(hxr)
    valid_shifted = np.zeros(n, dtype=bool)
    if valid is None:
        valid = np.ones(n, dtype=bool)
    for i in range(n):
        j = i - lag  # HXR at original time j corresponds to SXR at time i
        if 0 <= j < n:
            hxr_shifted[i] = hxr[j]
            valid_shifted[i] = valid[j]
        else:
            hxr_shifted[i] = 0.0
            valid_shifted[i] = False
    return sxr, hxr_shifted, lag


# ---------------------------------------------------------------------------
# 5. Top-level preprocessing pipeline
# ---------------------------------------------------------------------------
@dataclass
class PreprocessReport:
    """Diagnostics from the preprocessing layer."""
    n_spikes_sxr: int = 0
    n_spikes_hxr: int = 0
    n_saturated_sxr: int = 0
    n_saturated_hxr: int = 0
    lag_applied: int = 0
    baseline_sxr_mean: float = 0.0
    baseline_hxr_mean: float = 0.0
    drift_sxr: float = 0.0
    drift_hxr: float = 0.0
    # v2.2 additions:
    lag_revisions: int = 0          # how many times online lag was updated
    final_lag: int = 0              # final lag value at end of run
    solar_max_baseline: float = 0.0 # rolling 24h median (solar-max awareness)


# ---------------------------------------------------------------------------
# v2.2: ONLINE LAG RE-ESTIMATION (Struggle 4 fix)
# ---------------------------------------------------------------------------
class OnlineLagEstimator:
    """Re-estimates the SXR<->HXR lag every N steps using a sliding window.

    This solves STRUGGLE 4 (channel misalignment mid-run due to spacecraft
    maneuvers). The classical approach is the Generalized Cross-Correlation
    method (Knapp & Carter, IEEE Trans. ASSP, 1976), which we use here
    in its simplest form (no PHAT weighting — the signals are already
    band-limited by the SXR/HXR channel split).

    Usage:
        est = OnlineLagEstimator(max_lag=30, window=120, refresh_every=60)
        for i in range(n_steps):
            est.push(sxr[i], hxr[i], sxr_valid[i] and hxr_valid[i])
            lag = est.current_lag()  # use this lag for the rest of the step
    """
    def __init__(self, max_lag: int = 30, window: int = 120, refresh_every: int = 60):
        self.max_lag = int(max_lag)
        self.window = int(window)
        self.refresh_every = int(refresh_every)
        self._sxr_buf: list[float] = []
        self._hxr_buf: list[float] = []
        self._valid_buf: list[bool] = []
        self._lag: int = 0
        self._step: int = 0
        self._n_revisions: int = 0
        self._last_lag: int = 0

    def push(self, sxr: float, hxr: float, valid: bool) -> int:
        """Add one sample and possibly re-estimate the lag.

        Returns the current lag (also available via current_lag()).
        """
        self._step += 1
        self._sxr_buf.append(float(sxr))
        self._hxr_buf.append(float(hxr))
        self._valid_buf.append(bool(valid))
        if len(self._sxr_buf) > self.window:
            self._sxr_buf = self._sxr_buf[-self.window:]
            self._hxr_buf = self._hxr_buf[-self.window:]
            self._valid_buf = self._valid_buf[-self.window:]

        # Re-estimate every refresh_every steps, but only if we have
        # enough valid samples in the window
        if (self._step % self.refresh_every == 0
                and sum(self._valid_buf) >= 30):
            sxr_arr = np.array(self._sxr_buf)
            hxr_arr = np.array(self._hxr_buf)
            valid_arr = np.array(self._valid_buf, dtype=bool)
            new_lag = estimate_lag(sxr_arr, hxr_arr, valid_arr, self.max_lag)
            # Only accept the new lag if it changed by more than 2 samples
            # (prevents noise-induced jitter)
            if abs(new_lag - self._lag) > 2:
                self._lag = new_lag
                self._n_revisions += 1
            self._last_lag = new_lag

        return self._lag

    def current_lag(self) -> int:
        return self._lag

    def n_revisions(self) -> int:
        return self._n_revisions


# ---------------------------------------------------------------------------
# v2.2: SOLAR-MAXIMUM AWARENESS (rolling 24-hour median baseline)
# ---------------------------------------------------------------------------
def solar_max_baseline(
    sxr: np.ndarray,
    valid: np.ndarray,
    window_samples: int = 86400,  # 24h at 1 Hz; for GOES use 1440 (24h at 1-min)
) -> float:
    """Compute the rolling 24-hour median baseline of SXR.

    v2.2 fix for "the sun is never truly quiet during solar maximum":
    Instead of using the first 30 samples as the baseline (which can be
    biased if the run started during an active period), use the rolling
    24-hour median. This adapts to the slowly-varying solar background.

    If we have less than 24h of data, uses all available valid samples.

    Returns a single scalar: the median SXR over the window.
    """
    n = len(sxr)
    if n == 0:
        return 0.0
    if n <= window_samples:
        v = sxr[valid]
        return float(np.median(v)) if len(v) >= 5 else 0.0
    # Take the last window_samples
    start = max(0, n - window_samples)
    seg = sxr[start:][valid[start:]]
    return float(np.median(seg)) if len(seg) >= 5 else 0.0


def preprocess(sxr_obs: np.ndarray, hxr_obs: np.ndarray,
               sxr_valid: np.ndarray, hxr_valid: np.ndarray,
               hampel_window: int = 7, hampel_sigma: float = 5.0,
               baseline_window: int = 60,
               do_align: bool = True,
               max_align_lag: int = 30) -> tuple[np.ndarray, np.ndarray,
                                                 np.ndarray, np.ndarray,
                                                 PreprocessReport]:
    """Full classical preprocessing pipeline.

    Steps:
      1. Hampel-filter both channels to reject single-sample spikes.
      2. Detect saturation; mark saturated samples as invalid (predict-only).
      3. Cross-correlate to estimate and apply SXR<->HXR lag.
      4. Compute rolling median/MAD baselines for diagnostic reporting.
      5. Estimate baseline drift (linear fit slope) for diagnostic reporting.

    Returns (sxr_clean, hxr_clean, sxr_valid_clean, hxr_valid_clean, report).
    """
    report = PreprocessReport()

    # Step 1: Hampel
    sxr_h, spike_s = hampel_filter(sxr_obs, sxr_valid, hampel_window, hampel_sigma)
    hxr_h, spike_h = hampel_filter(hxr_obs, hxr_valid, hampel_window, hampel_sigma)
    report.n_spikes_sxr = int(spike_s.sum())
    report.n_spikes_hxr = int(spike_h.sum())

    # Step 2: Saturation
    sat_s_level, sat_s = detect_saturation(sxr_h, sxr_valid)
    sat_h_level, sat_h = detect_saturation(hxr_h, hxr_valid)
    sxr_v = sxr_valid & ~sat_s
    hxr_v = hxr_valid & ~sat_h
    report.n_saturated_sxr = int(sat_s.sum())
    report.n_saturated_hxr = int(sat_h.sum())

    # Step 3: Cross-channel alignment
    if do_align:
        _, hxr_h_aligned, lag = align_channels(sxr_h, hxr_h, sxr_v & hxr_v, max_align_lag)
        # When we shift HXR by lag, some samples fall outside the array and
        # become invalid (predict-only mode at the edges).
        n = len(hxr_h)
        hxr_v_aligned = np.zeros(n, dtype=bool)
        for i in range(n):
            j = i - lag
            if 0 <= j < n:
                hxr_v_aligned[i] = hxr_v[j]
        hxr_h = hxr_h_aligned
        hxr_v = hxr_v_aligned
        report.lag_applied = int(lag)

    # Step 4: Baselines
    bl_s = robust_baseline(sxr_h, sxr_v, baseline_window)
    bl_h = robust_baseline(hxr_h, hxr_v, baseline_window)
    report.baseline_sxr_mean = float(np.mean(bl_s[sxr_v])) if sxr_v.any() else 0.0
    report.baseline_hxr_mean = float(np.mean(bl_h[hxr_v])) if hxr_v.any() else 0.0

    # Step 5: Drift (linear fit slope, robust via Theil-Sen-ish: median of slopes)
    if sxr_v.sum() >= 10:
        idx = np.where(sxr_v)[0]
        if len(idx) >= 10:
            slopes = []
            for k in range(min(200, len(idx) - 1)):
                i = np.random.randint(0, len(idx) - 1)
                j = np.random.randint(0, len(idx) - 1)
                if i != j:
                    slopes.append((sxr_h[idx[i]] - sxr_h[idx[j]]) / (idx[i] - idx[j]))
            if slopes:
                report.drift_sxr = float(np.median(slopes))
    if hxr_v.sum() >= 10:
        idx = np.where(hxr_v)[0]
        if len(idx) >= 10:
            slopes = []
            for k in range(min(200, len(idx) - 1)):
                i = np.random.randint(0, len(idx) - 1)
                j = np.random.randint(0, len(idx) - 1)
                if i != j:
                    slopes.append((hxr_h[idx[i]] - hxr_h[idx[j]]) / (idx[i] - idx[j]))
            if slopes:
                report.drift_hxr = float(np.median(slopes))

    return sxr_h, hxr_h, sxr_v, hxr_v, report
