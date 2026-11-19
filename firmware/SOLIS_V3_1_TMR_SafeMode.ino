#include "solis_v3_tmr.h"

SolisV3_TMR solis;

void setup() {
    Serial.begin(115200);
    solis.begin();
    Serial.println("SOLIS_V3_1_READY");
}

void loop() {
    if (Serial.available() > 0) {
        String line = Serial.readStringUntil('\n');
        line.trim();
        if (line.length() == 0) return;

        if (line.startsWith("PING")) {
            Serial.println("PONG");
        } 
        else if (line.startsWith("INJECT,")) {
            // Command format: INJECT,<block_0_1_2>,<byte_offset>,<bit_mask>
            int block = 0, offset = 0, mask = 0;
            if (sscanf(line.c_str(), "INJECT,%d,%d,%d", &block, &offset, &mask) == 3) {
                solis.inject_raw_sram_fault((uint8_t)block, (uint8_t)offset, (uint8_t)mask);
                Serial.println("INJECT_OK");
            } else {
                Serial.println("INJECT_ERR_PARSE");
            }
        } 
        else if (line.startsWith("STEP,")) {
            // Command format: STEP,<sxr_flux>,<hxr_flux>,<dt>
            float sxr = 0.0f, hxr = 0.0f, dt = 1.0f;
            if (sscanf(line.c_str(), "STEP,%f,%f,%f", &sxr, &hxr, &dt) >= 2) {
                float mahalanobis = 0.0f;
                bool seu_repaired = false;
                unsigned long exec_us = 0;

                bool alert = solis.step(sxr, hxr, dt, mahalanobis, seu_repaired, exec_us);

                // Response format: DATA,<ALERT|NOMINAL>,<mahalanobis>,<SEU_REPAIRED|CLEAN>,<exec_us>
                Serial.print("DATA,");
                Serial.print(alert ? "ALERT" : "NOMINAL");
                Serial.print(",");
                Serial.print(mahalanobis, 6);
                Serial.print(",");
                Serial.print(seu_repaired ? "SEU_REPAIRED" : "CLEAN");
                Serial.print(",");
                Serial.println(exec_us);
            } else {
                Serial.println("STEP_ERR_PARSE");
            }
        } 
        else if (line.startsWith("QUERY_STATE")) {
            float x[STATE_DIM];
            float P[STATE_DIM][STATE_DIM];
            solis.get_active_state(x, P);
            Serial.print("STATE,");
            for (int i = 0; i < STATE_DIM; i++) {
                Serial.print(x[i], 8);
                if (i < STATE_DIM - 1) Serial.print(",");
            }
            Serial.println();
        }
    }
}
