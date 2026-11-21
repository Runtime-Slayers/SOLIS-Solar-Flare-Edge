# SOLIS Python Pipeline & Scientific Benchmarking Suite

> **Data Parsing, Multi-Sensor Synchronization, UKF-IMM State Estimation, and Mission Validation Suite**

This directory contains the Python scientific computing pipeline used to model, calibrate, and validate the SOLIS solar flare nowcasting architecture against empirical spaceborne telemetry from **ISRO Aditya-L1 (SoLEXS & HEL1OS)** and **NOAA GOES-18 Primary XRS**.

---

## 1. Pipeline Architecture

```text
  ISRO SoLEXS FITS (1–30 keV SXR)       NOAA GOES-18 1-min XRS
  ISRO HEL1OS FITS (10–150 keV HXR)      (Low-Latency Live Telemetry)
               │                                      │
               ▼                                      ▼
     [ fits_reader.py ]                     [ Live API Ingestion ]
               │                                      │
               └──────────────────┬───────────────────┘
                                  ▼
                   ┌──────────────────────────────┐
                   │   Robust Signal Conditioner   │
                   │  (qcaie_robust.py)           │
                   │  • Hampel 3σ Rolling Filter  │
                   │  • Huber M-Estimator         │
                   └──────────────┬───────────────┘
                                  ▼
                   ┌──────────────────────────────┐
                   │    6D State Estimator (UKF)  │
                   │    (qcaie_ukf.py)            │
                   │    x = [F_s, dF_s, d²F_s,    │
                   │         F_h, α, T_e]ᵀ        │
                   └──────────────┬───────────────┘
                                  ▼
                   ┌──────────────────────────────┐
                   │   2-Mode Supervisory IMM     │
                   │   (qcaie_imm.py)             │
                   │   • Mode 0: Quiet-Sun        │
                   │   • Mode 1: Flare Eruption   │
                   └──────────────┬───────────────┘
                                  ▼
                   ┌──────────────────────────────┐
                   │  Autonomous Trigger Logic    │
                   │  P(Flare) ≥ 0.50             │
                   │  Mahalanobis D² ≥ 9.21 (99%) │
                   └──────────────┬───────────────┘
                                  ▼
                  [ Early Warning & Shutter Latch ]
```

---

## 2. Module Directory

### Core Algorithmic Components
* **`qcaie_ukf.py`**: 6-state Unscented Kalman Filter tracking Soft X-ray flux ($F_s$), rate of change ($\dot{F}_s$), acceleration ($\ddot{F}_s$), Hard X-ray flux ($F_h$), spectral index ($\alpha$), and plasma electron temperature ($T_e$). Employs scaled unscented transforms ($\alpha_{\text{ukf}}=0.5, \beta=2.0, \kappa=0.0$) with positive-definite Cholesky factorization.
* **`qcaie_imm.py`**: Interacting Multiple Model (IMM) estimator coordinating parallel filters across Quiet-Sun stability and Flare dynamic regimes using a 2x2 Markov transition matrix.
* **`qcaie_robust.py`**: Rolling Hampel filter and Huber M-estimator designed to reject cosmic-ray single-packet spikes, telemetry gaps, and ADC saturation artifacts without phase distortion.
* **`qcaie_alpha_estimator.py`**: Two-timescale Recursive Least Squares (RLS) estimator for the Neupert empirical coupling constant $\alpha$.
* **`qcaie_experience.py`**: Case-Based Reasoning (CBR) module matching live state vectors against empirical historical flare manifolds via Mahalanobis distance.
* **`qcaie_qpp.py`**: Time-frequency spectral analyzer detecting Quasi-Periodic Pulsations (QPP) that precede impulsive magnetic reconnection.
* **`fits_reader.py`**: Lightweight, memory-efficient FITS file reader for Aditya-L1 SoLEXS and HEL1OS Level-1 science tables.
* **`fuse_hel1os_solexs.py`**: Sensor fusion engine synchronizing SoLEXS (1–30 keV) and HEL1OS (10–150 keV) light curves onto a common 100 ms master clock.

### Validation & Benchmarking Scripts
* **`validate_live_event_202611.py`**: Automated validation script evaluating real-time streaming telemetry from NOAA GOES-18 during the November 2026 Active Region AR4581 double M-class flare and CME event.
* **`plot_live_validation.py`**: Publication-quality dual-panel figure generator outputting `assets/fig7_live_noaa_nov2026_validation.png`.
* **`batch_analysis.py`**: Batch processing engine executing the pipeline across all 84 archival Aditya-L1 solar flare events in Solar Cycle 25.
* **`ml_baseline_compare.py`**: 14-metric benchmarking suite evaluating Random Forest, SVM, MLP, and 1D-CNN baselines against SOLIS on an identical 10,000-step test partition.
* **`extended_validation.py`**: Mission-scale statistical validation script computing 95% Wilson score confidence intervals and false alarm rates.
* **`stream_validate.py`**: In-orbit telemetry stream emulator with strict memory ceiling constraints (< 500 MB).
* **`validate_xclass_historical.py`**: Historical stress-test evaluator assessing early warning lead times on major X-class super-flares.

---

## 3. Installation & Usage

### Setup Virtual Environment
```bash
cd python_pipeline
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
pip install -r requirements.txt
```

### Running Validation Scripts
```bash
# 1. Evaluate live NOAA GOES-18 telemetry for the AR4581 event
python validate_live_event_202611.py

# 2. Run the machine learning baseline comparison
python ml_baseline_compare.py

# 3. Re-generate publication figures
python plot_live_validation.py
```
