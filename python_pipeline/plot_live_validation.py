#!/usr/bin/env python3
"""
Generate publication-quality figure for the live November 2026 NOAA GOES-18 space weather validation:
Active Region AR4581 Double M-Class Flare & CME Warning.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), dpi=300)

DATA_FILE = Path(r"C:\Users\brr33\Downloads\SOLIS-Solar-Flare-Edge\python_pipeline\datasets\noaa_goes18_xray_7day_live.json")
RESULTS_FILE = Path(r"C:\Users\brr33\Downloads\SOLIS-Solar-Flare-Edge\python_pipeline\benchmark_results\live_validation_nov2026_results.json")
OUTPUT_FIG = Path(r"C:\Users\brr33\Downloads\SOLIS-Solar-Flare-Edge\assets\fig7_live_noaa_nov2026_validation.png")

with open(DATA_FILE, "r", encoding="utf-8") as f:
    raw = json.load(f)

with open(RESULTS_FILE, "r", encoding="utf-8") as f:
    results = json.load(f)

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

# Panel 1: Zoomed in on Nov 19-20 Active Region AR4581 Eruption Window
m_window = [t for t in valid_times if "2026-11-19T11:00" <= t <= "2026-11-19T16:30"]
m_indices = [valid_times.index(t) for t in m_window]
m_long = long_flux[m_indices]
m_short = short_flux[m_indices]
m_steps = np.arange(len(m_window))

ax1.plot(m_steps, m_long, color="#d95f02", lw=2.0, label="GOES-18 SXR (0.1–0.8 nm)")
ax1.plot(m_steps, m_short, color="#7570b3", lw=1.5, ls="--", label="GOES-18 HXR (0.05–0.4 nm)")
ax1.set_yscale("log")
ax1.set_ylabel("Flux ($W \\cdot m^{-2}$)", fontsize=11, fontweight="bold")
ax1.set_title("SOLIS Live Telemetry Validation: Active Region AR4581 Double M-Class Flare (19 Nov 2026)", fontsize=12, fontweight="bold")

# Annotate peaks
p1_idx = m_window.index("2026-11-19T13:24:00Z")
ax1.scatter([p1_idx], [m_long[p1_idx]], color="red", s=80, zorder=5)
ax1.annotate("Flare 1: M2.4 Peak\n2.40e-5 W/m²\nLead Time: 60.0 min", 
             xy=(p1_idx, m_long[p1_idx]), xytext=(p1_idx - 45, 3e-5),
             arrowprops=dict(facecolor="black", shrink=0.05, width=1, headwidth=5),
             fontsize=9, fontweight="bold", bbox=dict(boxstyle="round,pad=0.3", fc="#ffeedd", ec="#d95f02"))

p2_idx = m_window.index("2026-11-19T13:59:00Z")
ax1.scatter([p2_idx], [m_long[p2_idx]], color="darkred", s=80, zorder=5)
ax1.annotate("Flare 2: M1.1 Peak\n1.10e-5 W/m²\nLead Time: 35.0 min", 
             xy=(p2_idx, m_long[p2_idx]), xytext=(p2_idx + 10, 2e-5),
             arrowprops=dict(facecolor="black", shrink=0.05, width=1, headwidth=5),
             fontsize=9, fontweight="bold", bbox=dict(boxstyle="round,pad=0.3", fc="#ffeedd", ec="#d95f02"))

# Threshold line
ax1.axhline(1e-5, color="red", ls=":", alpha=0.7, label="M-Class Threshold ($10^{-5}$ W/m²)")
ax1.axhline(1e-6, color="orange", ls=":", alpha=0.7, label="C-Class Threshold ($10^{-6}$ W/m²)")

# Format x-ticks
tick_step = 30
ax1.set_xticks(m_steps[::tick_step])
ax1.set_xticklabels([m_window[i].split("T")[1][:5] + " UTC" for i in range(0, len(m_window), tick_step)], rotation=0)
ax1.legend(loc="lower right", frameon=True, fontsize=9)
ax1.grid(True, which="both", ls=":", alpha=0.5)

# Panel 2: Nov 20 C9.7 Flare (near M-class)
c_window = [t for t in valid_times if "2026-11-20T09:30" <= t <= "2026-11-20T13:30"]
c_indices = [valid_times.index(t) for t in c_window]
c_long = long_flux[c_indices]
c_short = short_flux[c_indices]
c_steps = np.arange(len(c_window))

ax2.plot(c_steps, c_long, color="#1b9e77", lw=2.0, label="GOES-18 SXR (0.1–0.8 nm)")
ax2.plot(c_steps, c_short, color="#7570b3", lw=1.5, ls="--", label="GOES-18 HXR (0.05–0.4 nm)")
ax2.set_yscale("log")
ax2.set_ylabel("Flux ($W \\cdot m^{-2}$)", fontsize=11, fontweight="bold")
ax2.set_title("SOLIS Live Telemetry Validation: Active Region AR4581 C9.7 Eruptive Flare (20 Nov 2026)", fontsize=12, fontweight="bold")

cp_idx = c_window.index("2026-11-20T11:30:00Z")
ax2.scatter([cp_idx], [c_long[cp_idx]], color="red", s=80, zorder=5)
ax2.annotate("C9.7 Eruption Peak\n9.66e-6 W/m² (Near-M)\nLead Time: 14.0 min (840s)", 
             xy=(cp_idx, c_long[cp_idx]), xytext=(cp_idx - 50, 1.2e-5),
             arrowprops=dict(facecolor="black", shrink=0.05, width=1, headwidth=5),
             fontsize=9, fontweight="bold", bbox=dict(boxstyle="round,pad=0.3", fc="#eefbee", ec="#1b9e77"))

ctrig_idx = c_window.index("2026-11-20T11:16:00Z")
ax2.scatter([ctrig_idx], [c_long[ctrig_idx]], color="blue", marker="^", s=100, zorder=5)
ax2.annotate("Safe-Mode Trigger\n11:16 UTC (14 min pre-peak)", 
             xy=(ctrig_idx, c_long[ctrig_idx]), xytext=(ctrig_idx - 40, 2e-6),
             arrowprops=dict(facecolor="blue", shrink=0.05, width=1, headwidth=5),
             fontsize=9, fontweight="bold", color="darkblue")

ax2.axhline(1e-5, color="red", ls=":", alpha=0.7, label="M-Class Threshold")
ax2.axhline(1e-6, color="orange", ls=":", alpha=0.7, label="C-Class Threshold")

ax2.set_xticks(c_steps[::30])
ax2.set_xticklabels([c_window[i].split("T")[1][:5] + " UTC" for i in range(0, len(c_window), 30)], rotation=0)
ax2.set_xlabel("Time (UTC)", fontsize=11, fontweight="bold")
ax2.legend(loc="lower right", frameon=True, fontsize=9)
ax2.grid(True, which="both", ls=":", alpha=0.5)

plt.tight_layout()
plt.savefig(OUTPUT_FIG, bbox_inches="tight")
print(f"Generated validation plot: {OUTPUT_FIG} ({OUTPUT_FIG.stat().st_size} bytes)")
