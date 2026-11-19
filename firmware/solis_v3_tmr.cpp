#include "solis_v3_tmr.h"

SolisV3_TMR::SolisV3_TMR() {
    init_ukf_parameters();
    init_state(state_A);
    init_state(state_B);
    init_state(state_C);
    baseline_flux = 1e-7f;
    dynamic_threshold = 3.0f;
}

void SolisV3_TMR::begin() {
    init_state(state_A);
    init_state(state_B);
    init_state(state_C);
}

void SolisV3_TMR::init_ukf_parameters() {
    alpha_ukf = 0.5f;
    beta_ukf = 2.0f;
    kappa_ukf = 0.0f;

    lambda_ukf = alpha_ukf * alpha_ukf * (STATE_DIM + kappa_ukf) - STATE_DIM;
    gamma_ukf = sqrtf(STATE_DIM + lambda_ukf);

    // Weights for mean and covariance
    Wm[0] = lambda_ukf / (STATE_DIM + lambda_ukf);
    Wc[0] = Wm[0] + (1.0f - alpha_ukf * alpha_ukf + beta_ukf);

    for (int i = 1; i < NUM_SIGMA; i++) {
        Wm[i] = 1.0f / (2.0f * (STATE_DIM + lambda_ukf));
        Wc[i] = Wm[i];
    }

    // Process noise covariance Q (6x6)
    for (int i = 0; i < STATE_DIM; i++) {
        for (int j = 0; j < STATE_DIM; j++) {
            Q[i][j] = 0.0f;
        }
    }
    Q[0][0] = 1e-12f; // SXR variance
    Q[1][1] = 1e-13f; // HXR variance
    Q[2][2] = 1e-10f; // dSXR/dt variance
    Q[3][3] = 1e-11f; // dHXR/dt variance
    Q[4][4] = 1e-4f;  // SHR variance
    Q[5][5] = 1e-6f;  // Beta decay variance

    // Measurement noise covariance R (2x2)
    R[0][0] = 1e-14f; // SXR measurement noise
    R[0][1] = 0.0f;
    R[1][0] = 0.0f;
    R[1][1] = 1e-15f; // HXR measurement noise
}

void SolisV3_TMR::init_state(UKFState& state) {
    state.x[0] = 1e-7f; // SXR baseline
    state.x[1] = 1e-8f; // HXR baseline
    state.x[2] = 0.0f;  // dSXR/dt
    state.x[3] = 0.0f;  // dHXR/dt
    state.x[4] = 0.1f;  // Initial SHR (HXR/SXR)
    state.x[5] = 0.7f;  // Thermal decay rate beta

    for (int i = 0; i < STATE_DIM; i++) {
        for (int j = 0; j < STATE_DIM; j++) {
            state.P[i][j] = (i == j) ? 1e-14f : 0.0f;
        }
    }
    state.P[2][2] = 1e-12f; // dSXR/dt variance
    state.P[3][3] = 1e-13f; // dHXR/dt variance
    state.P[4][4] = 0.01f;  // SHR variance
    state.P[5][5] = 0.01f;  // Beta variance
}

bool SolisV3_TMR::cholesky_decompose(const float A[STATE_DIM][STATE_DIM], float L[STATE_DIM][STATE_DIM]) {
    for (int i = 0; i < STATE_DIM; i++) {
        for (int j = 0; j < STATE_DIM; j++) {
            L[i][j] = 0.0f;
        }
    }

    for (int i = 0; i < STATE_DIM; i++) {
        for (int j = 0; j <= i; j++) {
            float sum = 0.0f;
            for (int k = 0; k < j; k++) {
                sum += L[i][k] * L[j][k];
            }

            if (i == j) {
                float val = A[i][i] - sum;
                if (val <= 0.0f) {
                    val = 1e-12f; // Numerical regularization
                }
                L[i][j] = sqrtf(val);
            } else {
                if (fabsf(L[j][j]) < 1e-12f) {
                    L[i][j] = 0.0f;
                } else {
                    L[i][j] = (A[i][j] - sum) / L[j][j];
                }
            }
        }
    }
    return true;
}

void SolisV3_TMR::state_transition(const float x_in[STATE_DIM], float x_out[STATE_DIM], float dt) {
    // Non-linear Solar Plasma State Transition:
    // SXR evolves according to thermal decay + Neupert evaporation integral derivative
    float beta = x_in[5];
    if (beta < 0.01f) beta = 0.01f;
    if (beta > 2.0f) beta = 2.0f;

    // x[0]: SXR
    x_out[0] = x_in[0] + (x_in[2] - beta * (x_in[0] - baseline_flux)) * dt;
    if (x_out[0] < 1e-9f) x_out[0] = 1e-9f;

    // x[1]: HXR (Impulsive non-thermal injection)
    x_out[1] = x_in[1] + x_in[3] * dt;
    if (x_out[1] < 1e-10f) x_out[1] = 1e-10f;

    // x[2]: dSXR/dt
    x_out[2] = x_in[2] * 0.95f; // Damped derivative
    
    // x[3]: dHXR/dt
    x_out[3] = x_in[3] * 0.90f;

    // x[4]: SHR = HXR / SXR
    x_out[4] = x_out[1] / x_out[0];

    // x[5]: Beta parameter random walk
    x_out[5] = x_in[5];
}

void SolisV3_TMR::measurement_model(const float x_in[STATE_DIM], float z_out[MEAS_DIM]) {
    // Measurement matrix maps state to direct SXR (0) and HXR (1)
    z_out[0] = x_in[0];
    z_out[1] = x_in[1];
}

void SolisV3_TMR::ukf_predict_update(UKFState& state, float z_meas[MEAS_DIM], float dt, float& mahalanobis_out) {
    float L[STATE_DIM][STATE_DIM];
    cholesky_decompose(state.P, L);

    // 1. Generate Sigma Points (13 points)
    float sigma_points[NUM_SIGMA][STATE_DIM];
    for (int i = 0; i < STATE_DIM; i++) {
        sigma_points[0][i] = state.x[i];
    }

    for (int i = 0; i < STATE_DIM; i++) {
        for (int j = 0; j < STATE_DIM; j++) {
            sigma_points[i + 1][j] = state.x[j] + gamma_ukf * L[j][i];
            sigma_points[i + 1 + STATE_DIM][j] = state.x[j] - gamma_ukf * L[j][i];
        }
    }

    // 2. Propagate Sigma Points through Non-linear State Transition
    float sigma_prop[NUM_SIGMA][STATE_DIM];
    for (int i = 0; i < NUM_SIGMA; i++) {
        state_transition(sigma_points[i], sigma_prop[i], dt);
    }

    // 3. Predicted State Mean x_pred
    float x_pred[STATE_DIM] = {0.0f};
    for (int i = 0; i < NUM_SIGMA; i++) {
        for (int j = 0; j < STATE_DIM; j++) {
            x_pred[j] += Wm[i] * sigma_prop[i][j];
        }
    }

    // 4. Predicted State Covariance P_pred
    float P_pred[STATE_DIM][STATE_DIM] = {{0.0f}};
    for (int i = 0; i < NUM_SIGMA; i++) {
        float dx[STATE_DIM];
        for (int j = 0; j < STATE_DIM; j++) {
            dx[j] = sigma_prop[i][j] - x_pred[j];
        }
        for (int j = 0; j < STATE_DIM; j++) {
            for (int k = 0; k < STATE_DIM; k++) {
                P_pred[j][k] += Wc[i] * dx[j] * dx[k];
            }
        }
    }
    for (int j = 0; j < STATE_DIM; j++) {
        for (int k = 0; k < STATE_DIM; k++) {
            P_pred[j][k] += Q[j][k];
        }
    }

    // 5. Predict Measurements through Measurement Model
    float gamma_meas[NUM_SIGMA][MEAS_DIM];
    for (int i = 0; i < NUM_SIGMA; i++) {
        measurement_model(sigma_prop[i], gamma_meas[i]);
    }

    float z_pred[MEAS_DIM] = {0.0f};
    for (int i = 0; i < NUM_SIGMA; i++) {
        for (int j = 0; j < MEAS_DIM; j++) {
            z_pred[j] += Wm[i] * gamma_meas[i][j];
        }
    }

    // 6. Innovation Covariance S (2x2) and Cross-Covariance Pxz (6x2)
    float S[MEAS_DIM][MEAS_DIM] = {{0.0f}};
    float Pxz[STATE_DIM][MEAS_DIM] = {{0.0f}};

    for (int i = 0; i < NUM_SIGMA; i++) {
        float dz[MEAS_DIM];
        for (int j = 0; j < MEAS_DIM; j++) {
            dz[j] = gamma_meas[i][j] - z_pred[j];
        }

        float dx[STATE_DIM];
        for (int j = 0; j < STATE_DIM; j++) {
            dx[j] = sigma_prop[i][j] - x_pred[j];
        }

        for (int j = 0; j < MEAS_DIM; j++) {
            for (int k = 0; k < MEAS_DIM; k++) {
                S[j][k] += Wc[i] * dz[j] * dz[k];
            }
        }

        for (int j = 0; j < STATE_DIM; j++) {
            for (int k = 0; k < MEAS_DIM; k++) {
                Pxz[j][k] += Wc[i] * dx[j] * dz[k];
            }
        }
    }

    S[0][0] += R[0][0];
    S[0][1] += R[0][1];
    S[1][0] += R[1][0];
    S[1][1] += R[1][1];

    // 7. Invert 2x2 Matrix S
    float detS = S[0][0] * S[1][1] - S[0][1] * S[1][0];
    if (fabsf(detS) < 1e-24f) detS = 1e-24f;
    float invDetS = 1.0f / detS;

    float S_inv[MEAS_DIM][MEAS_DIM];
    S_inv[0][0] =  S[1][1] * invDetS;
    S_inv[0][1] = -S[0][1] * invDetS;
    S_inv[1][0] = -S[1][0] * invDetS;
    S_inv[1][1] =  S[0][0] * invDetS;

    // 8. Kalman Gain K = Pxz * S_inv (6x2)
    float K[STATE_DIM][MEAS_DIM];
    for (int i = 0; i < STATE_DIM; i++) {
        K[i][0] = Pxz[i][0] * S_inv[0][0] + Pxz[i][1] * S_inv[1][0];
        K[i][1] = Pxz[i][0] * S_inv[0][1] + Pxz[i][1] * S_inv[1][1];
    }

    // 9. Innovation Vector y = z_meas - z_pred
    float y[MEAS_DIM];
    y[0] = z_meas[0] - z_pred[0];
    y[1] = z_meas[1] - z_pred[1];

    // 10. Mahalanobis Distance d_M = sqrt(y^T * S_inv * y)
    float maha_sq = y[0] * (S_inv[0][0] * y[0] + S_inv[0][1] * y[1]) +
                    y[1] * (S_inv[1][0] * y[0] + S_inv[1][1] * y[1]);
    if (maha_sq < 0.0f) maha_sq = 0.0f;
    mahalanobis_out = sqrtf(maha_sq);

    // 11. State Mean Update: x = x_pred + K * y
    for (int i = 0; i < STATE_DIM; i++) {
        state.x[i] = x_pred[i] + (K[i][0] * y[0] + K[i][1] * y[1]);
    }

    // 12. State Covariance Update: P = P_pred - K * S * K^T
    float KS[STATE_DIM][MEAS_DIM];
    for (int i = 0; i < STATE_DIM; i++) {
        KS[i][0] = K[i][0] * S[0][0] + K[i][1] * S[1][0];
        KS[i][1] = K[i][0] * S[0][1] + K[i][1] * S[1][1];
    }

    for (int i = 0; i < STATE_DIM; i++) {
        for (int j = 0; j < STATE_DIM; j++) {
            float KSKt = KS[i][0] * K[j][0] + KS[i][1] * K[j][1];
            state.P[i][j] = P_pred[i][j] - KSKt;
            // Force symmetry and regularize diagonal
            if (i == j && state.P[i][j] < 1e-15f) {
                state.P[i][j] = 1e-15f;
            }
        }
    }
}

bool SolisV3_TMR::check_and_repair_state() {
    bool A_eq_B = (memcmp(&state_A, &state_B, sizeof(UKFState)) == 0);
    bool B_eq_C = (memcmp(&state_B, &state_C, sizeof(UKFState)) == 0);
    bool A_eq_C = (memcmp(&state_A, &state_C, sizeof(UKFState)) == 0);

    if (A_eq_B && B_eq_C) {
        return false; // All nominal
    }

    // TMR Majority Voting
    if (A_eq_B && !B_eq_C) {
        // State C corrupted -> overwrite with A
        memcpy(&state_C, &state_A, sizeof(UKFState));
    } else if (B_eq_C && !A_eq_B) {
        // State A corrupted -> overwrite with B
        memcpy(&state_A, &state_B, sizeof(UKFState));
    } else if (A_eq_C && !A_eq_B) {
        // State B corrupted -> overwrite with A
        memcpy(&state_B, &state_A, sizeof(UKFState));
    } else {
        // Unrecoverable multi-bit catastrophic disruption: reinitialize
        init_state(state_A);
        init_state(state_B);
        init_state(state_C);
    }
    return true;
}

bool SolisV3_TMR::step(float sxr_meas, float hxr_meas, float dt, float& out_mahalanobis, bool& seu_repaired, unsigned long& execution_time_us) {
    unsigned long start_micros = micros();

    // 1. TMR Health Check & Repair
    seu_repaired = check_and_repair_state();

    // 2. Adaptive Baseline & Dynamic Mahalanobis Threshold
    baseline_flux = 0.999f * baseline_flux + 0.001f * sxr_meas;
    dynamic_threshold = 3.0f + (baseline_flux / 1e-6f) * 0.5f;

    // 3. Execute Authentic UKF on Redundant Blocks
    float z[MEAS_DIM] = {sxr_meas, hxr_meas};
    float m_A = 0.0f, m_B = 0.0f, m_C = 0.0f;

    ukf_predict_update(state_A, z, dt, m_A);
    ukf_predict_update(state_B, z, dt, m_B);
    ukf_predict_update(state_C, z, dt, m_C);

    out_mahalanobis = m_A;
    execution_time_us = micros() - start_micros;

    bool alert = (out_mahalanobis > dynamic_threshold);
    return alert;
}

void SolisV3_TMR::inject_raw_sram_fault(uint8_t target_block, uint8_t byte_offset, uint8_t bit_mask) {
    uint8_t* base_ptr = nullptr;
    if (target_block == 0) base_ptr = reinterpret_cast<uint8_t*>(&state_A);
    else if (target_block == 1) base_ptr = reinterpret_cast<uint8_t*>(&state_B);
    else base_ptr = reinterpret_cast<uint8_t*>(&state_C);

    if (byte_offset < sizeof(UKFState)) {
        *(base_ptr + byte_offset) ^= bit_mask;
    }
}

void SolisV3_TMR::get_active_state(float out_x[STATE_DIM], float out_P[STATE_DIM][STATE_DIM]) {
    for (int i = 0; i < STATE_DIM; i++) {
        out_x[i] = state_A.x[i];
        for (int j = 0; j < STATE_DIM; j++) {
            out_P[i][j] = state_A.P[i][j];
        }
    }
}
