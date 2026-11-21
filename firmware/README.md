# SOLIS Embedded Flight Firmware

> **SmallSat / CubeSat Microcontroller Implementation for Real-Time Solar Flare Nowcasting & Autonomous Safe-Mode Assertion**

This directory contains the embedded C++ flight software and benchtop hardware profiling sketches for the SOLIS payload safety subsystem.

---

## 1. Hardware Architecture & Constraints

* **Target Microcontroller:** Espressif ESP32-WROOM-32 (Dual-Core Xtensa 32-bit LX6).
* **Operating Core Frequency:** **80 MHz** (downclocked from 240 MHz to optimize thermal and electrical power dissipation).
* **Deterministic Memory Model (Zero Dynamic Allocation):**
  * **Dynamic Heap Footprint:** **0 Bytes** (strict flight safety constraint; no `malloc`, `new`, or dynamic `String` reallocations in the filter cycle).
  * **Static SRAM Footprint:** **22.9 KB** allocated at compile-time in `.bss` and `.data` sections to hold triplicate TMR state buffers, sigma-point matrices, and IMM covariance arrays.
  * **On-Disk Flash Storage:** **291.5 KB** (< 23% of the default 1.3 MB application partition).
* **Timing Budget:** **4.058 ms** active computation per 1-second telemetry step (0.41% CPU duty cycle), leaving > 99.5% of MCU resources available for platform telemetry handling or sleep states.

---

## 2. GPIO Pinout & Hardware Interfacing

| Pin | Direction | Signal Type | Electrical Specification | Subsystem Description |
| :--- | :---: | :---: | :---: | :--- |
| **GPIO 26** | **Output** | Digital Output (Latch) | 3.3V CMOS Logic (Active HIGH) | **Payload Safe-Mode Trigger:** Commands optical shutter closure and instrument power isolation (< 4.1 ms latency). |
| **GPIO 2** | **Output** | Digital Status | 3.3V CMOS Logic | **Heartbeat / Diagnostic LED:** Toggles on valid filter cycle; pulses rapidly on SEU scrub event. |
| **GPIO 34** | **Input** | Analog / Digital | 0 – 3.3V ADC1_CH6 | Optional analog solar flux sensor telemetry channel. |
| **GPIO 1 (TX0)** | **Output** | UART Serial | 115200 Baud, 8N1 | Telemetry output stream (SXR flux estimate, Mahalanobis innovation distance, TMR scrub status). |
| **GPIO 3 (RX0)** | **Input** | UART Serial | 115200 Baud, 8N1 | In-orbit telemetry stream input from detector front-end electronics (FEE). |

---

## 3. Directory Contents

* **`SOLIS_V3_1_TMR_SafeMode.ino`**: Primary microcontroller entry point running the deterministic filter loop, majority voting verification, and GPIO 26 safe-mode assertion.
* **`solis_v3_tmr.cpp` / `solis_v3_tmr.h`**: Software Triple Modular Redundancy (TMR) engine that maintains three isolated memory copies of the filter state in SRAM and performs background bitwise scrubbing against radiation-induced Single Event Upsets (SEUs).
* **`QCAIE_v2_ESP32.ino`**: Monolithic flight sketch integrating the 6D state estimator, IMM supervisory filter, embedded topological case base, and live NOAA Wi-Fi polling for benchtop demonstration.
* **`embedded_cases.h`**: Pre-compiled static flash memory array (`PROGMEM`) containing empirical solar flare topological trajectories for zero-allocation case retrieval.
* **`power_profiler/`**: Dedicated benchtop hardware sketch and documentation for empirical electrical power dissipation and 64-bit hardware timer execution profiling (`power_profiler.ino`).

---

## 4. Build & Flashing Instructions

### Prerequisites
* [Arduino IDE](https://www.arduino.cc/en/software) (version 2.0+) or [arduino-cli](https://arduino.github.io/arduino-cli/)
* ESP32 Board Package: `esp32 by Espressif Systems` (version 2.0.11 or later)

### Compilation via `arduino-cli`
```bash
# Compile flight firmware with O3 optimization
arduino-cli compile --fqbn esp32:esp32:esp32 --build-property "build.extra_flags=-O3" firmware/SOLIS_V3_1_TMR_SafeMode.ino

# Flash to target board over USB/UART
arduino-cli upload -p /dev/ttyUSB0 --fqbn esp32:esp32:esp32 firmware/SOLIS_V3_1_TMR_SafeMode.ino
```

### Serial Monitor Verification
Open a serial terminal at **115200 baud**. The firmware outputs telemetry packets in structured CSV format:
```text
[TIME_MS],[STEP],[SXR_EST],[MAHALANOBIS],[IMM_P_FLARE],[TMR_STATUS],[SAFE_MODE_LATCH]
1000,1,10.24,0.1245,0.0210,NOMINAL,0
2000,2,10.28,0.1412,0.0215,NOMINAL,0
...
45000,45,1245.80,12.4500,0.8540,NOMINAL,1
```
