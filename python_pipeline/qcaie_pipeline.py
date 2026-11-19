"""
QCAIE v2.0 - Data synchronization pipeline
============================================
Aligns SoLEXS (1s SXR) and HEL1OS (~1s HXR) onto a common 100ms master clock.
Three-step pipeline:
    1. Zero-Order Hold (ZOH)         -- align HXR to master clock (sharp features)
    2. Linear interpolation           -- align SXR to master clock (smooth signals)
    3. Rolling Neupert integral       -- np.trapz over a 60-step sliding window

Also handles data dropouts: missing records produce None measurements, which the
UKF gracefully absorbs in predict-only mode.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class SyncedStream:
    t_master: np.ndarray         # (T,) master clock seconds
    sxr: np.ndarray              # (T,) aligned SXR
    hxr: np.ndarray              # (T,) aligned HXR
    sxr_valid: np.ndarray        # (T,) bool, True if measurement present
    hxr_valid: np.ndarray        # (T,) bool, True if measurement present
    dropout_steps: np.ndarray    # (K,) indices where BOTH channels dropped


def build_master_clock(t_start: float, t_end: float, dt_master: float = 0.1) -> np.ndarray:
    """Master clock at 100 ms cadence (10x the sensor rate)."""
    return np.arange(t_start, t_end + dt_master / 2, dt_master)


def zoh_align(t_master: np.ndarray, t_sensor: np.ndarray, x_sensor: np.ndarray,
              valid_sensor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Zero-Order Hold alignment: nearest preceding value.

    Sharp impulsive features in HXR should NOT be smeared by linear interp,
    so we use ZOH (np.interp with left/right fill).
    """
    # Use np.interp in 'nearest' manner via searchsorted
    aligned = np.zeros_like(t_master)
    aligned_valid = np.zeros_like(t_master, dtype=bool)
    if len(t_sensor) == 0:
        return aligned, aligned_valid
    # Find, for each master time, the index of the last sensor time <= t_master
    idx = np.searchsorted(t_sensor, t_master, side="right") - 1
    for i, j in enumerate(idx):
        if j >= 0 and j < len(t_sensor) and valid_sensor[j]:
            aligned[i] = x_sensor[j]
            aligned_valid[i] = True
        else:
            # Look ahead to the next valid sample (forward fill)
            j_fwd = j + 1
            while j_fwd < len(t_sensor) and not valid_sensor[j_fwd]:
                j_fwd += 1
            if j_fwd < len(t_sensor):
                aligned[i] = x_sensor[j_fwd]
                aligned_valid[i] = True
    return aligned, aligned_valid


def linear_align(t_master: np.ndarray, t_sensor: np.ndarray, x_sensor: np.ndarray,
                 valid_sensor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Linear interpolation alignment for smooth thermal SXR signals."""
    aligned = np.zeros_like(t_master)
    aligned_valid = np.zeros_like(t_master, dtype=bool)
    if len(t_sensor) == 0:
        return aligned, aligned_valid
    valid_t = t_sensor[valid_sensor]
    valid_x = x_sensor[valid_sensor]
    if len(valid_t) < 2:
        return aligned, aligned_valid
    # Interpolate only over the range where we have valid samples on both sides
    in_range = (t_master >= valid_t[0]) & (t_master <= valid_t[-1])
    aligned[in_range] = np.interp(t_master[in_range], valid_t, valid_x)
    aligned_valid[in_range] = True
    return aligned, aligned_valid


def rolling_neupert_integral(hxr: np.ndarray, dt: float, window: int = 60) -> np.ndarray:
    """Rolling trapezoidal integral of HXR over a sliding 60-step window.

    Avoids memory overflow from accumulating from t=0 and gives a local
    energy-deposition rate.
    """
    T = len(hxr)
    integral = np.zeros(T)
    for i in range(T):
        lo = max(0, i - window + 1)
        seg = hxr[lo:i + 1]
        integral[i] = np.trapz(seg, dx=dt)
    return integral


def sync_pipeline(
    t_sxr: np.ndarray, sxr: np.ndarray, sxr_valid: np.ndarray,
    t_hxr: np.ndarray, hxr: np.ndarray, hxr_valid: np.ndarray,
    dt_master: float = 0.1,
) -> SyncedStream:
    """End-to-end sync: build master clock, ZOH HXR, linear-interp SXR."""
    t_start = max(t_sxr[0] if len(t_sxr) else 0, t_hxr[0] if len(t_hxr) else 0)
    t_end = min(t_sxr[-1] if len(t_sxr) else 1e9, t_hxr[-1] if len(t_hxr) else 1e9)
    t_master = build_master_clock(t_start, t_end, dt_master)

    sxr_a, sxr_v = linear_align(t_master, t_sxr, sxr, sxr_valid)
    hxr_a, hxr_v = zoh_align(t_master, t_hxr, hxr, hxr_valid)

    # Dropout = both channels missing at the same master step
    both_missing = (~sxr_v) & (~hxr_v)
    dropout_steps = np.where(both_missing)[0]

    return SyncedStream(
        t_master=t_master,
        sxr=sxr_a,
        hxr=hxr_a,
        sxr_valid=sxr_v,
        hxr_valid=hxr_v,
        dropout_steps=dropout_steps,
    )


def measurement_at_step(stream: SyncedStream, i: int) -> Optional[np.ndarray]:
    """Build the measurement vector z = [SXR, HXR] at step i, or None if dropout.

    If only one channel is missing, we keep the other (the UKF handles a 2D
    measurement with one channel set to NaN by treating NaN as dropout). For
    simplicity, if EITHER channel is invalid we mark this step as a dropout and
    let the UKF predict-only -- this is the mission-safe choice per the
    methodology.
    """
    if not (stream.sxr_valid[i] and stream.hxr_valid[i]):
        return None
    return np.array([stream.sxr[i], stream.hxr[i]])
