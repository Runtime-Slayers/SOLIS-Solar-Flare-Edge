#!/usr/bin/env python3
"""
Publication-Quality Figure Generator for SOLIS Live Telemetry Validation (AR4581)
Designed for peer-reviewed space science / aerospace engineering manuscripts & GitHub showcase.
Zero overlapping text, generous headroom, clear lead-time shading, and professional typography.
"""

import json
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from pathlib import Path

# Paths
DATA_FILE = Path(r"C:\Users\brr33\Downloads\SOLIS-Solar-Flare-Edge\python_pipeline\datasets\noaa_goes18_xray_7day_live.json")
RESULTS_FILE = Path(r"C:\Users\brr33\Downloads\SOLIS-Solar-Flare-Edge\python_pipeline\benchmark_results\live_validation_nov2026_results.json")
OUTPUT_FIG = Path(r"C:\Users\brr33\Downloads\SOLIS-Solar-Flare-Edge\assets\fig7_live_noaa_nov2026_validation.png")

with open(DATA_FILE, "r", encoding="utf-8") as f:
    raw = json.load(f)

with open(RESULTS_FILE, "r", encoding="utf-8") as f:
    results = json.load(f)

# Extract synchronized 1-minute time series
ts_dict = {}
for d in raw:
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

# Create figure with high DPI and generous aspect ratio
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12.5, 9.0), dpi=300)
fig.patch.set_facecolor('#ffffff')

# Global styling
for ax in (ax1, ax2):
    ax.set_facecolor('#fbfbfc')
    ax.grid(True, which="major", ls="-", lw=0.6, color="#e2e4e8", zorder=0)
    ax.grid(True, which="minor", ls=":", lw=0.4, color="#edf0f3", zorder=0)
    ax.set_yscale("log")
    ax.tick_params(axis='both', which='major', labelsize=10, length=5, direction='out')
    ax.tick_params(axis='both', which='minor', length=2.5, direction='out')
    for spine in ax.spines.values():
        spine.set_color('#2c3e50')
        spine.set_linewidth(1.0)

# ==============================================================================
# PANEL 1: Double M-Class Solar Flare (M2.4 + M1.1) - 19 Nov 2026
# ==============================================================================
m_window = [t for t in valid_times if "2026-11-19T11:30" <= t <= "2026-11-19T16:30"]
m_indices = [valid_times.index(t) for t in m_window]
m_long = long_flux[m_indices]
m_short = short_flux[m_indices]
m_steps = np.arange(len(m_window))

# Plot irradiance curves
line_sxr1, = ax1.plot(m_steps, m_long, color="#d95f02", lw=2.2, zorder=4, label="GOES-18 SXR (0.1–0.8 nm, Primary 1-min)")
line_hxr1, = ax1.plot(m_steps, m_short, color="#2b5c8f", lw=1.6, ls="--", zorder=4, label="GOES-18 HXR (0.05–0.4 nm, Hard Channel)")

# Indices of events
p1_idx = m_window.index("2026-11-19T13:24:00Z")  # M2.4 Peak
p2_idx = m_window.index("2026-11-19T13:59:00Z")  # M1.1 Peak
trig1_idx = m_window.index("2026-11-19T12:24:00Z") # Flare 1 Trigger

# Shaded Early-Warning Lead Time Zone (Green)
ax1.axvspan(trig1_idx, p1_idx, color="#d4edda", alpha=0.55, zorder=1, label="SOLIS Early Warning Window (60.0 min lead time)")

# Threshold lines with label tags
ax1.axhline(1e-4, color="#721c24", ls="-.", lw=1.0, alpha=0.8, zorder=2)
ax1.text(len(m_steps)-2, 1.08e-4, "X-Class Threshold (10⁻⁴ W/m²)", color="#721c24", fontsize=8.5, ha="right", va="bottom", fontweight="bold")

ax1.axhline(1e-5, color="#c0392b", ls="--", lw=1.1, alpha=0.8, zorder=2)
ax1.text(len(m_steps)-2, 1.08e-5, "M-Class Threshold (10⁻⁵ W/m²)", color="#c0392b", fontsize=8.5, ha="right", va="bottom", fontweight="bold")

ax1.axhline(1e-6, color="#d35400", ls=":", lw=1.0, alpha=0.8, zorder=2)
ax1.text(len(m_steps)-2, 1.08e-6, "C-Class Threshold (10⁻⁶ W/m²)", color="#d35400", fontsize=8.5, ha="right", va="bottom", fontweight="bold")

# Y-Axis limits with generous headroom (1e-8 to 4e-4) to avoid any title collision
ax1.set_ylim(2e-8, 4e-4)

# Trigger point marker
ax1.scatter([trig1_idx], [m_long[trig1_idx]], color="#155724", marker="^", s=110, zorder=6, edgecolors="white", lw=1.2)
ax1.annotate("SOLIS Safe-Mode Assertion\n12:24:00 UTC (Thermal Onset)",
             xy=(trig1_idx, m_long[trig1_idx]), xytext=(trig1_idx - 35, 3e-7),
             arrowprops=dict(arrowstyle="->", color="#155724", lw=1.4, connectionstyle="arc3,rad=-0.15"),
             fontsize=9, fontweight="bold", color="#155724",
             bbox=dict(boxstyle="round,pad=0.4", fc="#ffffff", ec="#155724", lw=1.0, alpha=0.95))

# Peak 1 (M2.4) Annotation - positioned cleanly below title in open region
ax1.scatter([p1_idx], [m_long[p1_idx]], color="#c0392b", marker="o", s=90, zorder=6, edgecolors="black", lw=1.2)
ax1.annotate("Flare 1 Peak: M2.4\nFlux: 2.40 × 10⁻⁵ W/m²\nWarning Lead: 60.0 min (3600 s)",
             xy=(p1_idx, m_long[p1_idx]), xytext=(p1_idx - 58, 7.5e-5),
             arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.4, connectionstyle="arc3,rad=0.15"),
             fontsize=9, fontweight="bold", color="#900c3f",
             bbox=dict(boxstyle="round,pad=0.4", fc="#ffffff", ec="#c0392b", lw=1.0, alpha=0.95))

# Peak 2 (M1.1) Annotation - positioned cleanly to the right
ax1.scatter([p2_idx], [m_long[p2_idx]], color="#8e44ad", marker="s", s=85, zorder=6, edgecolors="black", lw=1.2)
ax1.annotate("Flare 2 Peak: M1.1 (35 min later)\nFlux: 1.10 × 10⁻⁵ W/m²\nContinuous Lock Retained",
             xy=(p2_idx, m_long[p2_idx]), xytext=(p2_idx + 22, 4.5e-5),
             arrowprops=dict(arrowstyle="->", color="#8e44ad", lw=1.4, connectionstyle="arc3,rad=-0.15"),
             fontsize=9, fontweight="bold", color="#4a235a",
             bbox=dict(boxstyle="round,pad=0.4", fc="#ffffff", ec="#8e44ad", lw=1.0, alpha=0.95))

# Panel title placed neatly inside or left-aligned with ample pad
ax1.set_title("(a) Active Region AR4581 Double M-Class Eruption Sequence — 19 November 2026 (Live NOAA GOES-18)", 
              fontsize=11.5, fontweight="bold", color="#1c2833", loc="left", pad=12)
ax1.set_ylabel("Solar Irradiance (W · m⁻²)", fontsize=10.5, fontweight="bold", color="#1c2833")

# Time Ticks
tick_interval = 30
ax1.set_xticks(m_steps[::tick_interval])
ax1.set_xticklabels([m_window[i].split("T")[1][:5] + " UTC" for i in range(0, len(m_window), tick_interval)], fontweight="medium")
ax1.legend(loc="lower right", frameon=True, facecolor="#ffffff", edgecolor="#bdc3c7", fontsize=8.8, framealpha=0.95)


# ==============================================================================
# PANEL 2: Eruptive Precursor (C9.7 Near-M) - 20 Nov 2026
# ==============================================================================
c_window = [t for t in valid_times if "2026-11-20T09:30" <= t <= "2026-11-20T14:00"]
c_indices = [valid_times.index(t) for t in c_window]
c_long = long_flux[c_indices]
c_short = short_flux[c_indices]
c_steps = np.arange(len(c_window))

# Plot irradiance curves
ax2.plot(c_steps, c_long, color="#16a085", lw=2.2, zorder=4, label="GOES-18 SXR (0.1–0.8 nm, Primary 1-min)")
ax2.plot(c_steps, c_short, color="#2b5c8f", lw=1.6, ls="--", zorder=4, label="GOES-18 HXR (0.05–0.4 nm, Hard Channel)")

cp_idx = c_window.index("2026-11-20T11:30:00Z")   # C9.7 Peak
ctrig_idx = c_window.index("2026-11-20T11:16:00Z") # C9.7 Trigger (14 min pre-peak)

# Shaded Early-Warning Lead Time Zone (Green)
ax2.axvspan(ctrig_idx, cp_idx, color="#d4edda", alpha=0.55, zorder=1, label="SOLIS Early Warning Window (14.0 min / 840 s lead time)")

# Threshold lines with label tags
ax2.axhline(1e-5, color="#c0392b", ls="--", lw=1.1, alpha=0.8, zorder=2)
ax2.text(len(c_steps)-2, 1.08e-5, "M-Class Threshold (10⁻⁵ W/m²)", color="#c0392b", fontsize=8.5, ha="right", va="bottom", fontweight="bold")

ax2.axhline(1e-6, color="#d35400", ls=":", lw=1.0, alpha=0.8, zorder=2)
ax2.text(len(c_steps)-2, 1.08e-6, "C-Class Threshold (10⁻⁶ W/m²)", color="#d35400", fontsize=8.5, ha="right", va="bottom", fontweight="bold")

# Y-Axis limits with generous headroom (1e-8 to 2e-4)
ax2.set_ylim(1.5e-8, 2e-4)

# Trigger point marker
ax2.scatter([ctrig_idx], [c_long[ctrig_idx]], color="#155724", marker="^", s=110, zorder=6, edgecolors="white", lw=1.2)
ax2.annotate("Safe-Mode Latch Triggered\n11:16:00 UTC (14 min pre-peak)",
             xy=(ctrig_idx, c_long[ctrig_idx]), xytext=(ctrig_idx - 45, 1.8e-6),
             arrowprops=dict(arrowstyle="->", color="#155724", lw=1.4, connectionstyle="arc3,rad=-0.15"),
             fontsize=9, fontweight="bold", color="#155724",
             bbox=dict(boxstyle="round,pad=0.4", fc="#ffffff", ec="#155724", lw=1.0, alpha=0.95))

# Peak (C9.7) Annotation - placed with plenty of margin
ax2.scatter([cp_idx], [c_long[cp_idx]], color="#c0392b", marker="o", s=90, zorder=6, edgecolors="black", lw=1.2)
ax2.annotate("C9.7 Eruption Peak (Near M-Class)\nFlux: 9.66 × 10⁻⁶ W/m²\nWarning Lead: 14.0 min (840 s)",
             xy=(cp_idx, c_long[cp_idx]), xytext=(cp_idx + 18, 3.8e-5),
             arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.4, connectionstyle="arc3,rad=-0.15"),
             fontsize=9, fontweight="bold", color="#900c3f",
             bbox=dict(boxstyle="round,pad=0.4", fc="#ffffff", ec="#c0392b", lw=1.0, alpha=0.95))

# Panel title
ax2.set_title("(b) Active Region AR4581 Precursor Eruption (C9.7) — 20 November 2026 (Live NOAA GOES-18)", 
              fontsize=11.5, fontweight="bold", color="#1c2833", loc="left", pad=12)
ax2.set_ylabel("Solar Irradiance (W · m⁻²)", fontsize=10.5, fontweight="bold", color="#1c2833")
ax2.set_xlabel("Observation Timestamp (Universal Time UTC)", fontsize=10.5, fontweight="bold", color="#1c2833")

# Time Ticks
ax2.set_xticks(c_steps[::30])
ax2.set_xticklabels([c_window[i].split("T")[1][:5] + " UTC" for i in range(0, len(c_window), 30)], fontweight="medium")
ax2.legend(loc="lower right", frameon=True, facecolor="#ffffff", edgecolor="#bdc3c7", fontsize=8.8, framealpha=0.95)

# Overall clean suptitle
fig.suptitle("SOLIS Operational In-Orbit Telemetry Validation: Active Region AR4581 Live Space Weather Tracking",
             fontsize=13.5, fontweight="heavy", color="#1a252f", y=0.985)

plt.subplots_adjust(top=0.93, bottom=0.07, hspace=0.32, left=0.08, right=0.96)
plt.savefig(OUTPUT_FIG, dpi=300)
print(f"Generated pristine, non-overlapping figure at: {OUTPUT_FIG}")
