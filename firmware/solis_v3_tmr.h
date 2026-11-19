#ifndef SOLIS_V3_TMR_H
#define SOLIS_V3_TMR_H

#include <Arduino.h>
#include <string.h>
#include <math.h>

#define STATE_DIM 6
#define MEAS_DIM 2
#define NUM_SIGMA (2 * STATE_DIM + 1) // 13 Sigma Points

// 6D State Vector:
// x[0] = SXR Flux (W/m^2)
// x[1] = HXR Flux (W/m^2)
// x[2] = SXR Derivative d(SXR)/dt
// x[3] = HXR Derivative d(HXR)/dt
// x[4] = Spectral Hardness Ratio (HXR / SXR)
// x[5] = Thermal Decay Constant Beta (s^-1)
struct UKFState {
    float x[STATE_DIM];
    float P[STATE_DIM][STATE_DIM];
};

class SolisV3_TMR {
private:
    // Triple Modular Redundant States in physical SRAM
    UKFState state_A;
    UKFState state_B;
    UKFState state_C;

    // Process & Measurement Noise Covariances
    float Q[STATE_DIM][STATE_DIM];
    float R[MEAS_DIM][MEAS_DIM];

    // Unscented Transform Parameters & Weights
    float alpha_ukf;
    float beta_ukf;
    float kappa_ukf;
    float lambda_ukf;
    float gamma_ukf;
    float Wm[NUM_SIGMA];
    float Wc[NUM_SIGMA];

    // Adaptive Background and Dynamic Threshold
    float baseline_flux;
    float dynamic_threshold;

    void init_state(UKFState& state);
    void init_ukf_parameters();
    
    // Matrix & Cholesky Decomposition Helpers
    bool cholesky_decompose(const float A[STATE_DIM][STATE_DIM], float L[STATE_DIM][STATE_DIM]);
    void state_transition(const float x_in[STATE_DIM], float x_out[STATE_DIM], float dt);
    void measurement_model(const float x_in[STATE_DIM], float z_out[MEAS_DIM]);

    // Full 6D Unscented Kalman Filter Predict & Update
    void ukf_predict_update(UKFState& state, float z_meas[MEAS_DIM], float dt, float& mahalanobis_out);

    // TMR Fault Detection, Isolation, and Voting
    bool check_and_repair_state();

public:
    SolisV3_TMR();
    void begin();

    // Main execution step: takes measurements, runs TMR voting, advances UKF, calculates Mahalanobis
    bool step(float sxr_meas, float hxr_meas, float dt, float& out_mahalanobis, bool& seu_repaired, unsigned long& execution_time_us);

    // Physical memory fault injector (for authentic verification)
    void inject_raw_sram_fault(uint8_t target_block, uint8_t byte_offset, uint8_t bit_mask);

    // State Inspection
    void get_active_state(float out_x[STATE_DIM], float out_P[STATE_DIM][STATE_DIM]);
};

#endif // SOLIS_V3_TMR_H
