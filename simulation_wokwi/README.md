# SOLIS Hardware-in-the-Loop (HIL) Simulation Testbed

> **Cycle-Accurate Wokwi Virtual Testbed for ESP32 Firmware Execution & Radiation SEU Fault-Injection Testing**

This directory contains the simulation configurations, circuit wiring descriptions, and automated test runners used to validate the compiled SOLIS firmware binary (`.elf`) in a simulated space radiation environment.

---

## 1. Simulation Philosophy & Testing Boundary

To maintain rigorous aerospace engineering integrity, SOLIS explicitly separates benchtop physical hardware testing from virtual cycle-accurate simulation:

* **Physical ESP32 Silicon (Benchtop):** Reserved for high-precision electrical power dissipation (33.28 mW total board, 0.415 mW core compute @ 80 MHz on 3.3V DC rail) and deterministic execution latency (4.058 ms via 64-bit APB hardware timers).
* **Cycle-Accurate Wokwi Simulation:** Utilized for automated, high-throughput **Single Event Upset (SEU) radiation fault-injection** (1,000 automated bit-flip trials) executing the compiled Xtensa ELF binary over virtual UART pipes to preserve physical hardware longevity.

---

## 2. Directory Contents

* **`diagram.json`**: Wokwi graphical wiring schema connecting the ESP32 microcontroller to:
  * GPIO 26: Safe-Mode Latch Indicator LED (Red)
  * GPIO 2: Telemetry Heartbeat LED (Blue)
  * UART0 (TX/RX): Serial telemetry communication pipeline at 115200 baud
* **`wokwi.toml`**: Simulation build manifest defining the ELF binary path, firmware architecture, and gdb debugger port.
* **`scenario.yaml`**: Wokwi scenario specification for automated input stream feeding and timing assertion verification.
* **`run_authentic_hil.py`**: Automated HIL testbench executing 1,000 pseudo-random SRAM bit-flip trials into the running binary. Evaluates the Software TMR majority voting engine and verifies 100% recovery from single-bit corruptions.
* **`run_hil_test.py`**: Fast smoke test runner for local verification of serial telemetry exchange and GPIO safe-mode latching.

---

## 3. Running the Simulation

### Option A: Automated CLI Testbench
Ensure `wokwi-cli` is installed:
```bash
npm install -g @wokwi/cli
export WOKWI_CLI_TOKEN="your_token_here"

cd simulation_wokwi
python run_authentic_hil.py
```

### Option B: Interactive Browser Simulation
1. Navigate to the [Wokwi ESP32 Web Simulator](https://wokwi.com/projects/new/esp32).
2. Paste the contents of `simulation_wokwi/diagram.json` into the **diagram.json** tab.
3. Paste the contents of `firmware/SOLIS_V3_1_TMR_SafeMode.ino` into the **sketch.ino** tab.
4. Click **Play (Start Simulation)** to observe real-time telemetry output in the virtual serial monitor and verify GPIO 26 LED activation when a flare condition is recognized.
