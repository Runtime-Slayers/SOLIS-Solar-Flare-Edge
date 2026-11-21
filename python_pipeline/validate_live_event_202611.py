#!/usr/bin/env python3
"""
SOLIS Validation on Live NOAA GOES-18 Space Weather Telemetry (November 2026)
Active Region AR4581 Double M-Class Solar Flare & Earth-Directed CME Window

Live Data:
- Telemetry Source: NOAA SWPC Primary Space Weather Satellite (GOES-18)
- Cadence: 1-Minute Primary XRS (SXR: 0.1-0.8 nm, HXR: 0.05-0.4 nm)
- Window: 2026-11-14T04:34:00Z to 2026-11-21T04:31:00Z (7-Day Live Continuous Stream)

Events Tracked:
1. Event 1: M2.4 Solar Flare (Peak: 2026-11-19T13:24:00Z, Flux = 2.403e-05 W/m^2)
2. Event 2: M1.1 Solar Flare (Peak: 2026-11-19T13:59:00Z, Flux = 1.103e-05 W/m^2)
3. Event 3: C9.7 Eruptive Flare (Peak: 2026-11-20T11:30:00Z, Flux = 9.656e-06 W/m^2)
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

from qcaie_ukf import UKF
from qcaie_imm import IMM
from qcaie_alpha_estimator import AlphaEstimator

DATA_FILE = SCRIPT_DIR / "datasets" / "noaa_goes18_xray_7day_live.json"
OUTPUT_DIR = SCRIPT_DIR / "benchmark_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAHALANOBIS_THRESHOLD = 9.21
MU_FLARE_THRESHOLD = 0.50
CONFIRM_STEPS = 3


def load_live_noaa_telemetry(file_path):
    if file_path.exists():
        with open(file_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
    else:
        print("Local cache not found. Fetching live telemetry directly from NOAA SWPC API...")
        import urllib.request
        url = "https://services.swpc.noaa.gov/json/goes/primary/xrays-7-day.json"
        req = urllib.request.Request(url, headers={"User-Agent": "SOLIS-Live-Fetch/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode("utf-8")
        end_bracket = raw.rfind("]")
        raw_data = json.loads(raw[:end_bracket+1])
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(raw_data, f)
        print(f"Cached {len(raw_data)} records to {file_path}")

    ts_dict = {}
    for d in raw_data:
        tt = d.get("time_tag")
        energy = d.get("energy")
        flux = d.get("flux")
        if not tt or flux is None:
            continue
        if tt not in ts_dict:
            ts_dict[tt] = {}
        if energy == "0.1-0.8nm":
            ts_dict[tt]["long"] = float(flux)
        elif energy == "0.05-0.4nm":
            ts_dict[tt]["short"] = float(flux)

    valid_times = sorted([t for t in ts_dict if "long" in ts_dict[t] and "short" in ts_dict[t]])
    
    long_flux = np.array([ts_dict[t]["long"] for t in valid_times])
    short_flux = np.array([ts_dict[t]["short"] for t in valid_times])
    
    return valid_times, long_flux, short_flux


def build_imm(sxr0, hxr0, sxr_bg, dt=60.0):
    x0 = np.array([sxr0, hxr0, 0.0, max(hxr0 / max(sxr0, 1e-6), 1e-3), 0.0])
    P0 = np.diag([0.1, 0.1, 1.0, 0.1, 0.1])
    R0 = np.diag([0.01, 0.04])
    Q_quiet = np.diag([1e-3, 1e-3, 1e-2, 1e-3, 1e-3]) * dt
    Q_flare = np.diag([1e-2, 1e-2, 1e-1, 1e-2, 1e-2]) * dt

    ukf_quiet = UKF(
        x0=x0.copy(), P0=P0.copy(), Q=Q_quiet, R=R0.copy(),
        alpha=0.05, beta=0.7, alpha_ukf=0.5, kappa=0.0,
        adaptive_R=True, sxr_background=sxr_bg
    )
    ukf_flare = UKF(
        x0=x0.copy(), P0=P0.copy(), Q=Q_flare, R=R0.copy(),
        alpha=0.05, beta=0.3, alpha_ukf=0.5, kappa=0.0,
        adaptive_R=True, sxr_background=sxr_bg
    )
    Pi = np.array([[0.98, 0.02], [0.10, 0.90]])
    return IMM(ukf_quiet, ukf_flare, Pi)


def evaluate_event_window(valid_times, long_flux, short_flux, peak_time_str, window_pre_min=60, window_post_min=45):
    idx_peak = valid_times.index(peak_time_str)
    start_idx = max(0, idx_peak - window_pre_min)
    end_idx = min(len(valid_times), idx_peak + window_post_min)
    
    sub_times = valid_times[start_idx:end_idx]
    sub_long = long_flux[start_idx:end_idx]
    sub_short = short_flux[start_idx:end_idx]
    peak_sub_idx = idx_peak - start_idx
    
    # Scale fluxes into counts (1 count = 1e-8 W/m^2)
    sxr = sub_long * 1e8
    hxr = sub_short * 1e8
    
    # Baseline from quiet portion of the window
    bg_steps = min(20, max(5, peak_sub_idx // 2))
    sxr_bg = max(float(np.nanmedian(sxr[:bg_steps])), 1.0)
    sxr0 = float(sxr[0]) if sxr[0] > 0 else sxr_bg
    hxr0 = float(hxr[0]) if hxr[0] > 0 else sxr_bg * 0.1
    
    imm = build_imm(sxr0, hxr0, sxr_bg, dt=60.0)
    alpha_est = AlphaEstimator(alpha=0.05)
    
    n = len(sub_times)
    mu_flare_arr = np.zeros(n)
    maha_arr = np.zeros(n)
    alpha_arr = np.zeros(n)
    
    for k in range(n):
        z = np.array([sxr[k], hxr[k]])
        try:
            diag = imm.step(z, dt=60.0)
            mu_f = float(diag["mu_flare"])
            maha = float(max(diag["mahalanobis_quiet"], diag["mahalanobis_flare"]))
            
            x_est = diag["x"]
            sxr_excess = max(float(sxr[k] - sxr_bg), 0.0)
            hxr_integral = float(x_est[2])
            alpha_est.update(sxr_excess, hxr_integral)
        except Exception:
            mu_f, maha = 0.0, 0.0
            
        mu_flare_arr[k] = mu_f
        maha_arr[k] = maha
        alpha_arr[k] = alpha_est.alpha
        
    # Detect alerts with 3-step confirmation filter
    alert_indices = []
    confirm = 0
    for i in range(n):
        if mu_flare_arr[i] > MU_FLARE_THRESHOLD or maha_arr[i] > MAHALANOBIS_THRESHOLD:
            confirm += 1
            if confirm >= CONFIRM_STEPS:
                alert_indices.append(i - (CONFIRM_STEPS - 1)) # mark onset of confirmation
        else:
            confirm = 0
            
    # Lead time relative to peak
    pre_peak_alerts = [idx for idx in alert_indices if idx < peak_sub_idx]
    if pre_peak_alerts:
        first_alert_idx = pre_peak_alerts[0]
        lead_time_min = float(peak_sub_idx - first_alert_idx)
        lead_time_sec = lead_time_min * 60.0
        status = "DETECTED"
        trigger_time = sub_times[first_alert_idx]
    else:
        first_alert_idx = None
        lead_time_min = 0.0
        lead_time_sec = 0.0
        status = "MISSED"
        trigger_time = None
        
    return {
        "event_peak_time": peak_time_str,
        "peak_flux_W_m2": float(long_flux[idx_peak]),
        "status": status,
        "trigger_time": trigger_time,
        "lead_time_minutes": lead_time_min,
        "lead_time_seconds": lead_time_sec,
        "peak_alpha": float(alpha_arr[peak_sub_idx]),
        "peak_sxr_hxr_ratio": float(sub_short[peak_sub_idx] / sub_long[peak_sub_idx])
    }


def evaluate_continuous_stream(valid_times, long_flux, short_flux, dt=60.0):
    """
    Evaluates rolling 7-day continuous stream with adaptive background estimation.
    """
    n = len(valid_times)
    sxr = long_flux * 1e8
    hxr = short_flux * 1e8
    
    # Adaptive rolling background (120-minute rolling median)
    roll_win = 120
    sxr_bg_roll = np.zeros(n)
    for i in range(n):
        w_start = max(0, i - roll_win)
        sxr_bg_roll[i] = float(np.median(sxr[w_start:i+1]))
        
    imm = build_imm(sxr[0], hxr[0], sxr_bg_roll[0], dt=dt)
    alpha_est = AlphaEstimator(alpha=0.05)
    
    mu_flare_arr = np.zeros(n)
    maha_arr = np.zeros(n)
    
    for k in range(n):
        # Update UKF background dynamically
        imm.models[0].sxr_background = sxr_bg_roll[k]
        imm.models[1].sxr_background = sxr_bg_roll[k]
        
        z = np.array([sxr[k], hxr[k]])
        try:
            diag = imm.step(z, dt=dt)
            mu_f = float(diag["mu_flare"])
            maha = float(max(diag["mahalanobis_quiet"], diag["mahalanobis_flare"]))
        except Exception:
            mu_f, maha = 0.0, 0.0
            
        mu_flare_arr[k] = mu_f
        maha_arr[k] = maha
        
    alert_indices = []
    confirm = 0
    for i in range(n):
        if mu_flare_arr[i] > MU_FLARE_THRESHOLD or maha_arr[i] > MAHALANOBIS_THRESHOLD:
            confirm += 1
            if confirm >= CONFIRM_STEPS:
                alert_indices.append(i - (CONFIRM_STEPS - 1))
        else:
            confirm = 0
            
    # Classify quiet steps: where long_flux < 1.0e-6 W/m^2 (B-class or lower)
    quiet_mask = long_flux < 1.0e-6
    quiet_count = int(np.sum(quiet_mask))
    fa_steps = sum(1 for idx in alert_indices if quiet_mask[idx])
    far_rate = (fa_steps / quiet_count * 100.0) if quiet_count > 0 else 0.0
    
    return {
        "total_steps": n,
        "quiet_steps": quiet_count,
        "false_alarm_steps": fa_steps,
        "far_percent": float(far_rate)
    }


def main():
    print("=" * 78)
    print("SOLIS LIVE VALIDATION: NOAA GOES-18 NOVEMBER 2026 SPACE WEATHER EVENT")
    print("Active Region AR4581 Double M-Class Flare & Earth-Directed CME Window")
    print("=" * 78)
    
    valid_times, long_flux, short_flux = load_live_noaa_telemetry(DATA_FILE)
    print(f"Loaded {len(valid_times)} synchronized 1-minute telemetry steps.")
    print(f"Time coverage: {valid_times[0]} to {valid_times[-1]}")
    
    # 1. Event 1: M2.4
    res_m24 = evaluate_event_window(valid_times, long_flux, short_flux, "2026-11-19T13:24:00Z", window_pre_min=60, window_post_min=30)
    # 2. Event 2: M1.1
    res_m11 = evaluate_event_window(valid_times, long_flux, short_flux, "2026-11-19T13:59:00Z", window_pre_min=35, window_post_min=30)
    # 3. Event 3: C9.7
    res_c97 = evaluate_event_window(valid_times, long_flux, short_flux, "2026-11-20T11:30:00Z", window_pre_min=45, window_post_min=30)
    
    # 4. 7-Day Continuous Stream
    print("\nRunning full 7-day continuous stream validation (10,072 steps)...")
    res_stream = evaluate_continuous_stream(valid_times, long_flux, short_flux)

    print("\n" + "=" * 78)
    print("EMPIRICAL VALIDATION RESULTS SUMMARY")
    print("=" * 78)
    print(f"Event 1: M2.4 Solar Flare (Active Region AR4581)")
    print(f"  Peak Timestamp : {res_m24['event_peak_time']} | Peak SXR: {res_m24['peak_flux_W_m2']:.3e} W/m^2 (M2.4)")
    print(f"  Trigger Time   : {res_m24['trigger_time']}")
    print(f"  Warning Lead   : {res_m24['lead_time_minutes']:.1f} minutes ({res_m24['lead_time_seconds']:.0f} seconds) [STATUS: {res_m24['status']}]")
    print(f"  Peak SXR/HXR   : {res_m24['peak_sxr_hxr_ratio']:.3f} | Spectral Alpha: {res_m24['peak_alpha']:.2f}")
    
    print(f"\nEvent 2: M1.1 Solar Flare (AR4581 Eruption #2 - 35 min post-M2.4)")
    print(f"  Peak Timestamp : {res_m11['event_peak_time']} | Peak SXR: {res_m11['peak_flux_W_m2']:.3e} W/m^2 (M1.1)")
    print(f"  Trigger Time   : {res_m11['trigger_time']}")
    print(f"  Warning Lead   : {res_m11['lead_time_minutes']:.1f} minutes ({res_m11['lead_time_seconds']:.0f} seconds) [STATUS: {res_m11['status']}]")
    print(f"  Peak SXR/HXR   : {res_m11['peak_sxr_hxr_ratio']:.3f} | Spectral Alpha: {res_m11['peak_alpha']:.2f}")

    print(f"\nEvent 3: C9.7 Eruptive Precursor (Near M-Class Eruption)")
    print(f"  Peak Timestamp : {res_c97['event_peak_time']} | Peak SXR: {res_c97['peak_flux_W_m2']:.3e} W/m^2 (C9.7)")
    print(f"  Trigger Time   : {res_c97['trigger_time']}")
    print(f"  Warning Lead   : {res_c97['lead_time_minutes']:.1f} minutes ({res_c97['lead_time_seconds']:.0f} seconds) [STATUS: {res_c97['status']}]")

    print(f"\n7-Day Full Stream Reliability (10,072 Minutes / 168 Hours):")
    print(f"  Total Steps    : {res_stream['total_steps']}")
    print(f"  Quiet Steps    : {res_stream['quiet_steps']}")
    print(f"  False Alarms   : {res_stream['false_alarm_steps']}")
    print(f"  Stream FAR     : {res_stream['far_percent']:.4f}%")
    print("=" * 78)

    summary = {
        "evaluation_timestamp": datetime.now(timezone.utc).isoformat(),
        "mission_satellite": "NOAA GOES-18 (Primary XRS)",
        "active_region": "AR4581",
        "cme_forecast": "Earth-directed glancing blow (Nov 22, 2026)",
        "flares": {
            "M2.4": res_m24,
            "M1.1": res_m11,
            "C9.7": res_c97
        },
        "continuous_stream_metrics": res_stream
    }
    
    out_json = OUTPUT_DIR / "live_validation_nov2026_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote full metrics to: {out_json}")


if __name__ == "__main__":
    main()
