# SOLIS: Solar Observation via Linear Inference Systems

> **Edge-Computing Solar Flare Nowcasting & Autonomous Safe-Mode Trigger for SmallSats**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Platform: ESP32](https://img.shields.io/badge/Platform-ESP32-green.svg)](https://www.espressif.com/)
[![Target: Aditya--L1](https://img.shields.io/badge/Mission-ISRO%20Aditya--L1-orange.svg)](https://www.isro.gov.in/Aditya_L1.html)
[![Validation: NOAA GOES--18](https://img.shields.io/badge/Validation-NOAA%20GOES--18-blue.svg)](https://www.ngdc.noaa.gov/stp/satellite/goes-r.html)

[Coming soon: 30-second physical ESP32 hardware demo]

SOLIS is an **open-source student engineering prototype** developed by undergraduates at Amrita Vishwa Vidyapeetham (Coimbatore). The system investigates real-time, low-power onboard solar flare early warning and autonomous payload shutter protection for resource-constrained SmallSat and CubeSat platforms.

We welcome feedback, issue reports, peer review, and collaboration from space systems engineers, solar physicists, and embedded developers.

---

## 1. System Architecture

SOLIS operates a 4-stage pipeline designed for deterministic execution on microcontrollers:

`
[ Telemetry Ingestion ] (1 Hz SXR / HXR Flux Streams)
          │
          ▼
┌────────────────────────────────────────────────────────┐
│ 1. Robust Signal Conditioning                          │
│    • Hampel rolling filter (3σ outlier rejection)      │
│    • Huber M-estimator for cosmic-ray spike immunity   │
└────────────────────────────────────────────────────────┘
          │
          ▼
┌────────────────────────────────────────────────────────┐
│ 2. 6D State Estimation & Multi-Regime IMM              │
│    • 6D Unscented Kalman Filter (UKF) tracking:        │
│      x = [F_s, dF_s/dt, d²F_s/dt², F_h, α, T_e]ᵀ       │
│    • 4-Regime Interacting Multiple Model (IMM):        │
│      Quiet-Sun, Pre-Flare, Impulsive, Gradual Decay    │
│    • Topological Case-Based Reasoning (CBR) lookup     │
└────────────────────────────────────────────────────────┘
          │
          ▼
┌────────────────────────────────────────────────────────┐
│ 3. Fault-Tolerant Software TMR                         │
│    • Triple Modular Redundancy (TMR) across 3 copies   │
│    • 2-out-of-3 bitwise majority voting                │
│    • Background SRAM scrubbing against radiation SEUs │
└────────────────────────────────────────────────────────┘
          │
          ▼
┌────────────────────────────────────────────────────────┐
│ 4. Autonomous Safe-Mode Assertion                      │
│    • Hardware GPIO 26 latch (< 4.1 ms latency)         │
│    • Commands instrument shutter closure & safe-mode   │
└────────────────────────────────────────────────────────┘
`

---

## 2. Hardware Profiling vs. Simulation Boundary

To ensure absolute engineering transparency, the testing boundary is strictly defined:

| Dimension | Physical ESP32 Silicon (Benchtop) | Cycle-Accurate Wokwi Simulation |
| :--- | :--- | :--- |
| **Execution Timing** | **4.058 ms** active compute @ 80 MHz (internal 64-bit hardware timers) | Validated cycle alignment with physical measurements |
| **Power Dissipation** | **33.28 mW** board total / **0.415 mW** core compute (3.3V DC current shunt) | Not modeled |
| **Memory Footprint** | **28.4 KB** SRAM heap / **291.7 KB** Flash partition | Verified binary layout matching compiled ELF |
| **SEU Fault Injection** | Benchtop testing reserved for power and timing longevity | **1,000-trial automated bit-flip testbench** (
un_authentic_hil.py) on compiled ELF binary |
| **Fault Recovery** | Manual pin injection verified GPIO latching | **100% single-bit recovery** via TMR scrub; fail-safe lock on fatal corruptions |

---

## 3. Verified Performance & Metrics

Metrics derived directly from the archival telemetry validation suite (atch_analysis.py, ml_baseline_compare.py):

### Mission-Wide Operational Validation (84 Archival Flares, 881,278 Telemetry Steps)
* **Dataset Period:** Solar Cycle 25 (May 2024 to July 2026 for SoLEXS; June to July 2026 for HEL1OS).
* **Probability of Detection (POD):** **97.6%** (82 of 84 flare events detected).
* **Operational False Alarm Rate (FAR):** **0.0039%** (34 false positives across 881,278 continuous steps).
* **Operational True Skill Statistic (TSS):** **0.9761**.
* **Severe Flare Detection (M/X Class):** **100.0% POD** (23 of 23 events), **0.0001% FAR**, **TSS = 0.9999**.
* **Median Warning Lead Time:** **7.60 minutes** before GOES peak flux (up to 18.2 minutes on gradual eruptive events).

### Machine Learning Baseline Comparison (Identical 10,000-Step Benchmark Partition)
Evaluated across 45 independent evaluation windows (23 flares and 22 quiet-Sun windows):

| Model Architecture | POD (%) | FAR (%) | TSS | Precision (%) | Inference Latency (ms) | Peak RAM |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | 82.6% | 4.5% | 0.781 | 95.0% | 1.82 ms | 3.4 MB |
| **Support Vector Machine (RBF)**| 78.3% | 9.1% | 0.692 | 90.0% | 0.94 ms | 1.8 MB |
| **Multilayer Perceptron (MLP)** | 87.0% | 4.5% | 0.825 | 95.2% | 0.42 ms | 512 KB |
| **Lightweight 1D-CNN** | 91.3% | 4.5% | 0.868 | 95.5% | 2.15 ms | 1.2 MB |
| **SOLIS (UKF-IMM + TMR)** | **100.0%** | **0.0%** | **1.000** | **100.0%** | **4.06 ms** | **28.4 KB** |

*Note: While SOLIS achieves TSS = 1.000 on this standardized 45-window ML benchmark subset, its operational performance across the full 84-flare mission dataset is TSS = 0.9761.*

---

## 4. Quick Start Guide

### A. Python Pipeline & Validation
1. Clone the repository and install requirements:
   `ash
   git clone https://github.com/Runtime-Slayers/SOLIS-Solar-Flare-Edge.git
   cd SOLIS-Solar-Flare-Edge/python_pipeline
   pip install -r requirements.txt
   `
2. Public Data Sources:
   * **ISRO Aditya-L1 SoLEXS & HEL1OS:** Download public Level-1 FITS data from [ISDC PRADAN](https://pradan.issdc.gov.in/ch1/).
   * **NOAA GOES-18 XRS:** Available from [NOAA NCEI Space Weather Data](https://www.ngdc.noaa.gov/stp/satellite/goes-r.html).
3. Run historical validation:
   `ash
   python validate_xclass_historical.py
   python ml_baseline_compare.py
   `

### B. Wokwi Simulation
To run the automated Hardware-in-the-Loop testbench:
`ash
cd simulation_wokwi
python run_authentic_hil.py
`
Alternatively, open simulation_wokwi/diagram.json and irmware/SOLIS_V3_1_TMR_SafeMode.ino in the [Wokwi Web Simulator](https://wokwi.com/).

### C. ESP32 Firmware Flashing
1. Open irmware/SOLIS_V3_1_TMR_SafeMode.ino in Arduino IDE or use rduino-cli.
2. Select target board: ESP32 Dev Module (CPU Frequency: 80 MHz for power optimization).
3. Connect GPIO 26 to an oscilloscope or LED to observe safe-mode triggering.
4. Upload firmware:
   `ash
   arduino-cli compile --fqbn esp32:esp32:esp32 firmware/SOLIS_V3_1_TMR_SafeMode.ino
   arduino-cli upload -p <PORT> --fqbn esp32:esp32:esp32 firmware/SOLIS_V3_1_TMR_SafeMode.ino
   `

---

## 5. Repository Structure

`	ext
├── assets/             # Architecture and validation figures
├── docs/               # System architecture diagrams and technical documentation
├── firmware/           # ESP32 C++ flight code, TMR module, and benchtop power profiler
├── python_pipeline/    # UKF-IMM pipeline, FITS ingestion, and benchmark suite
└── simulation_wokwi/   # Wokwi HIL configuration and automated SEU test scripts
`

---

## 6. Team & Contact

Developed by student researchers at the **Department of Artificial Intelligence, Amrita Vishwa Vidyapeetham, Coimbatore, India**:

* **Rajendra Reddy Bhavanam** ([brr1154@gmail.com](mailto:brr1154@gmail.com)) — Lead Researcher & Embedded Systems
* **Saran Boddu** — Signal Processing & Mathematical Modeling
* **Likith Palakurthi** — Data Ingestion & Statistical Validation
* **Muthuraman Ramanathan** — Domain Analysis & Flare Physics

We actively welcome technical reviews, constructive criticism, SmallSat mission collaborations, and research internship inquiries.
