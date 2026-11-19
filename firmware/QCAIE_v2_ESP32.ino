/*
 * QCAIE v2.2 — ESP32 Real-Time Solar Flare Nowcasting System
 * ==========================================================
 * 
 * "Logics and mathematics won't work in space as we thought.
 *  Data can say a little, but what says more? Experience."
 * 
 * Hardware: ESP32 WROOM DA Module
 * Power:   USB charger (5V, 500mA minimum)
 * Network: WiFi (2.4GHz)
 * Output:  Serial Monitor (115200 baud) + Built-in LED + Web Dashboard
 * 
 * Data Sources:
 *   - NOAA SWPC GOES X-ray (real-time, 1-min cadence, no API key)
 *     https://services.swpc.noaa.gov/json/goes/primary/xrays-1-day.json
 *   - NASA DONKI Solar Flare Events (for cross-confirmation, uses API key)
 *     https://api.nasa.gov/DONKI/FLR
 * 
 * v2.2 Features:
 *   1. 5D UKF ( Unscented Kalman Filter) with Joseph-form covariance update
 *   2. IMM (Interacting Multiple Model) with 2 UKFs (Quiet + Pre-Flare)
 *   3. Neupert residual + Mahalanobis 3-sigma gating
 *   4. Simple alpha (Neupert constant) estimator with adaptive ceiling
 *   5. Experience Layer (CBR) — retrieves similar past cases from SPIFFS
 *   6. Cadence-specific confirmation threshold (2 steps at 1-min cadence)
 *   7. Bayesian prior shift on P(Pre-Flare) threshold
 *   8. Alert clearing (debounced, 5-step cooldown)
 * 
 * NO ML / NO Deep Learning — Pure Classical Kalman Filtering + CBR
 * 
 * HOW TO USE:
 *   1. Open this file in Arduino IDE (embedded_cases.h must be in the same folder)
 *   2. Select Board: "ESP32 WROOM DA Module"
 *   3. Set the correct COM port
 *   4. Edit WIFI_SSID and WIFI_PASSWORD below
 *   5. Upload this firmware (NO SPIFFS upload needed — cases are embedded)
 *   6. Open Serial Monitor at 115200 baud
 *
 * FILES NEEDED (just 2):
 *   - QCAIE_v2_ESP32.ino    (this file — the firmware)
 *   - embedded_cases.h       (the 15 experience cases as C++ data)
 *   Both files MUST be in the same folder. The Arduino IDE will auto-compile
 *   both when you upload.
 */
#include <WiFi.h>
#include <WiFiClientSecure.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include <WebServer.h>
#include <ESPmDNS.h>
#include <math.h>
// Note: SPIFFS not needed — experience cases are embedded in embedded_cases.h
// ==================== CONFIGURATION ====================
const char* WIFI_SSID     = "realme P1 Pro 5G";
const char* WIFI_PASSWORD = "Sepas0609";
const char* NASA_API_KEY  = "CuOQTnWbnklGXU4dHMRhie84OVB5jjn2uvydiNdr";
const char* NOAAXRAY_URL  = "https://services.swpc.noaa.gov/json/goes/primary/xrays-6-hour.json";
// NASA DONKI URL with API key PRE-APPENDED — no runtime concatenation needed
const char* NASA_DONKI_URL = "https://api.nasa.gov/DONKI/FLR?startDate=2026-06-14&api_key=CuOQTnWbnklGXU4dHMRhie84OVB5jjn2uvydiNdr";

// mDNS hostname — access the dashboard at http://solis.local
// Works on iPhone, Mac, and most Android phones on the same WiFi
const char* MDNS_HOSTNAME = "solis";

#define FETCH_INTERVAL_MS   60000   // Fetch new data every 60 seconds
#define LED_PIN             2       // Built-in LED on most ESP32 boards
#define LED_ALERT           HIGH
#define LED_QUIET           LOW
// ==================== EXPERIENCE LAYER TYPES (must be before functions) ====================
// Arduino IDE auto-generates function prototypes at the top of the file.
// If these structs are defined after functions that use them, the auto-generated
// prototypes will reference an unknown type. So we define them here, at the top.
#define N_FEATURES 14
#define MAX_CASES 20
#define MAX_LESSON_LEN 200
struct ExperienceCase {
  char case_id[40];
  char source[28];        // "synthetic-adversarial" is 21 chars + null = 22, so 28 is safe
  char date[12];
  char flare_class[8];
  float cadence_seconds;
  float feature[N_FEATURES];
  float alpha_final;
  bool false_alarm;
  bool missed;
  bool correct;
  int lead_time_steps;
  char lesson[MAX_LESSON_LEN];
};
struct ExperienceAdvice {
  int n_similar;
  float prior_P_flare;
  float recommended_alpha_ceiling;
  int recommended_confirm_steps;
  float false_alarm_risk;
  char top_lesson[MAX_LESSON_LEN];
  char matched_case_ids[120];
};
const float FEATURE_WEIGHTS[N_FEATURES] = {
  1.0f, 0.8f, 1.2f, 0.7f, 0.9f, 0.6f, 1.0f, 1.0f,
  0.8f, 0.5f, 0.6f, 0.5f, 0.5f, 0.3f
};
const float FEATURE_SCALES[N_FEATURES] = {
  0.5f, 0.05f, 5.0f, 0.5f, 0.3f, 0.5f, 2.0f, 1.0f,
  1.0f, 0.5f, 5.0f, 0.3f, 0.3f, 30.0f
};
ExperienceCase case_base[MAX_CASES];
int n_cases_loaded = 0;
// Include the embedded experience database (15 cases as C++ initializers).
// This eliminates the need for SPIFFS upload — the cases are compiled
// directly into the firmware.
#include "embedded_cases.h"
// ==================== UKF + IMM ====================
#define N_STATE 5   // [SXR, HXR, integral, SHR, dSHR]
#define N_SIGMA 11  // 2*N_STATE + 1
// ---- 5x5 Matrix helper ----
struct Matrix5 {
  float m[N_STATE][N_STATE];
  void zero() {
    for (int i = 0; i < N_STATE; i++)
      for (int j = 0; j < N_STATE; j++)
        m[i][j] = 0.0f;
  }
  void identity() {
    zero();
    for (int i = 0; i < N_STATE; i++) m[i][i] = 1.0f;
  }
  void symmetrize() {
    for (int i = 0; i < N_STATE; i++)
      for (int j = i + 1; j < N_STATE; j++) {
        float avg = (m[i][j] + m[j][i]) * 0.5f;
        m[i][j] = avg;
        m[j][i] = avg;
      }
  }
};
// ---- State transition f(x, dt, alpha, beta, sxr_bg) ----
// State: [SXR, HXR, integral, SHR, dSHR]
// SXR evolves as: sxr_bg + beta*(sxr-sxr_bg) + (1-beta)*alpha*integral
// HXR is random walk (held constant in prediction)
// integral is leaky integrator: leak*x[2] + max(x[1],0)*dt
// SHR = HXR/SXR
// dSHR = (SHR_new - SHR_old) / dt
void stateTransition(const float* x, float dt, float alpha, float beta,
                     float sxr_bg, float* x_next) {
  float sxr_excess_old = x[0] - sxr_bg;
  float sxr_excess_new = beta * sxr_excess_old + (1.0f - beta) * alpha * x[2];
  float sxr_new = sxr_bg + sxr_excess_new;
  if (sxr_new < 1e-12f) sxr_new = 1e-12f;
  float hxr_new = x[1];
  if (hxr_new < 1e-12f) hxr_new = 1e-12f;
  float leak = expf(-dt / 300.0f);  // tau = 300s
  float integ_new = leak * x[2] + fmaxf(x[1], 0.0f) * dt;
  float shr_new = hxr_new / sxr_new;
  float dshr_new = (shr_new - x[3]) / dt;
  if (dshr_new > 100.0f) dshr_new = 100.0f;
  if (dshr_new < -100.0f) dshr_new = -100.0f;
  x_next[0] = sxr_new;
  x_next[1] = hxr_new;
  x_next[2] = integ_new;
  x_next[3] = shr_new;
  x_next[4] = dshr_new;
}
// ---- Measurement function h(x) = [x[0], x[1]] ----
void measurementFunc(const float* x, float* z) {
  z[0] = x[0];
  z[1] = x[1];
}
// ==================== UKF CLASS ====================
class UKF {
public:
  float x[N_STATE];
  Matrix5 P, Q;
  float R[2][2], R0[2][2];
  float alpha_neupert;
  float beta;
  float sxr_background;
  float alpha_ceil;       // v2.2: adaptive ceiling (raise-only)
  float alpha_ukf, kappa, lambda_;
  float Wm[N_SIGMA], Wc[N_SIGMA];
  float sigmas[N_SIGMA][N_STATE];
  float sigmas_pred[N_SIGMA][N_STATE];
  float S[2][2];
  float last_residual[2];
  float last_mahalanobis;
  void init(float alpha_n, float beta_n, float sxr_bg,
            float sxr_init, float hxr_init) {
    alpha_neupert = alpha_n;
    beta = beta_n;
    sxr_background = sxr_bg;
    alpha_ceil = 2.0f;
    last_mahalanobis = 0.0f;
    last_residual[0] = 0.0f;
    last_residual[1] = 0.0f;
    for (int i = 0; i < N_STATE; i++) x[i] = 0.0f;
    x[0] = sxr_init;
    x[1] = hxr_init;
    x[2] = 0.0f;
    x[3] = hxr_init / fmaxf(sxr_init, 1e-6f);
    x[4] = 0.0f;
    P.identity();
    P.m[0][0] = 0.1f; P.m[1][1] = 0.1f;
    P.m[2][2] = 1.0f; P.m[3][3] = 0.1f; P.m[4][4] = 0.1f;
    Q.zero();
    Q.m[0][0] = 1e-3f; Q.m[1][1] = 1e-3f;
    Q.m[2][2] = 1e-2f; Q.m[3][3] = 1e-3f; Q.m[4][4] = 1e-3f;
    R[0][0] = 0.01f; R[0][1] = 0.0f;
    R[1][0] = 0.0f;  R[1][1] = 0.04f;
    R0[0][0] = 0.01f; R0[0][1] = 0.0f;
    R0[1][0] = 0.0f;  R0[1][1] = 0.04f;
    alpha_ukf = 0.5f;
    kappa = 0.0f;
    lambda_ = alpha_ukf * alpha_ukf * (N_STATE + kappa) - N_STATE;
    float w0 = lambda_ / (N_STATE + lambda_);
    float wi = 1.0f / (2.0f * (N_STATE + lambda_));
    Wm[0] = w0; Wc[0] = w0 + (1.0f - alpha_ukf * alpha_ukf + 2.0f);
    for (int i = 1; i < N_SIGMA; i++) { Wm[i] = wi; Wc[i] = wi; }
  }
  // Cholesky-based sigma point generation
  void computeSigmaPoints() {
    Matrix5 P_sym;
    for (int i = 0; i < N_STATE; i++)
      for (int j = 0; j < N_STATE; j++)
        P_sym.m[i][j] = (P.m[i][j] + P.m[j][i]) * 0.5f;
    float L[N_STATE][N_STATE];
    for (int i = 0; i < N_STATE; i++)
      for (int j = 0; j < N_STATE; j++)
        L[i][j] = 0.0f;
    for (int i = 0; i < N_STATE; i++) {
      for (int j = 0; j <= i; j++) {
        float sum = P_sym.m[i][j];
        for (int k = 0; k < j; k++)
          sum -= L[i][k] * L[j][k];
        if (i == j) {
          L[i][j] = sqrtf(fmaxf(sum, 1e-6f));
        } else {
          L[i][j] = sum / fmaxf(L[j][j], 1e-6f);
        }
      }
    }
    float scale = sqrtf(N_STATE + lambda_);
    for (int i = 0; i < N_STATE; i++)
      sigmas[0][i] = x[i];
    for (int i = 0; i < N_STATE; i++) {
      for (int j = 0; j < N_STATE; j++)
        sigmas[i + 1][j] = x[j] + scale * L[j][i];
      for (int j = 0; j < N_STATE; j++)
        sigmas[N_STATE + i + 1][j] = x[j] - scale * L[j][i];
    }
  }
  void predict(float dt) {
    computeSigmaPoints();
    for (int i = 0; i < N_SIGMA; i++)
      stateTransition(sigmas[i], dt, alpha_neupert, beta, sxr_background, sigmas_pred[i]);
    for (int i = 0; i < N_STATE; i++) {
      x[i] = 0.0f;
      for (int j = 0; j < N_SIGMA; j++)
        x[i] += Wm[j] * sigmas_pred[j][i];
    }
    for (int i = 0; i < N_STATE; i++)
      for (int j = 0; j < N_STATE; j++) {
        float sum = 0.0f;
        for (int k = 0; k < N_SIGMA; k++) {
          float dx_i = sigmas_pred[k][i] - x[i];
          float dx_j = sigmas_pred[k][j] - x[j];
          sum += Wc[k] * dx_i * dx_j;
        }
        P.m[i][j] = sum + Q.m[i][j];
      }
    P.symmetrize();
  }
  void update(const float* z, float dt) {
    if (z == nullptr) {
      last_residual[0] = 0.0f;
      last_residual[1] = 0.0f;
      last_mahalanobis = 0.0f;
      return;
    }
    float z_sigmas[N_SIGMA][2];
    for (int i = 0; i < N_SIGMA; i++)
      measurementFunc(sigmas_pred[i], z_sigmas[i]);
    float z_pred[2] = {0.0f, 0.0f};
    for (int i = 0; i < N_SIGMA; i++)
      for (int j = 0; j < 2; j++)
        z_pred[j] += Wm[i] * z_sigmas[i][j];
    // Innovation covariance S = H*P*H^T + R (2x2)
    for (int i = 0; i < 2; i++)
      for (int j = 0; j < 2; j++) {
        float sum = 0.0f;
        for (int k = 0; k < N_SIGMA; k++) {
          float dz_i = z_sigmas[k][i] - z_pred[i];
          float dz_j = z_sigmas[k][j] - z_pred[j];
          sum += Wc[k] * dz_i * dz_j;
        }
        S[i][j] = sum + R[i][j];
      }
    S[0][1] = (S[0][1] + S[1][0]) * 0.5f;
    S[1][0] = S[0][1];
    // Cross-covariance C = P_xz (N_STATE x 2)
    float C[N_STATE][2];
    for (int i = 0; i < N_STATE; i++)
      for (int j = 0; j < 2; j++) {
        float sum = 0.0f;
        for (int k = 0; k < N_SIGMA; k++) {
          float dx_i = sigmas_pred[k][i] - x[i];
          float dz_j = z_sigmas[k][j] - z_pred[j];
          sum += Wc[k] * dx_i * dz_j;
        }
        C[i][j] = sum;
      }
    // S inverse (2x2)
    float detS = S[0][0] * S[1][1] - S[0][1] * S[1][0];
    if (fabsf(detS) < 1e-12f) detS = 1e-12f;
    float S_inv[2][2];
    S_inv[0][0] =  S[1][1] / detS;
    S_inv[0][1] = -S[0][1] / detS;
    S_inv[1][0] = -S[1][0] / detS;
    S_inv[1][1] =  S[0][0] / detS;
    // Kalman gain K = C * S^-1
    float K[N_STATE][2];
    for (int i = 0; i < N_STATE; i++)
      for (int j = 0; j < 2; j++)
        K[i][j] = C[i][0] * S_inv[0][j] + C[i][1] * S_inv[1][j];
    // Residual
    float residual[2] = {z[0] - z_pred[0], z[1] - z_pred[1]};
    last_residual[0] = residual[0];
    last_residual[1] = residual[1];
    // Mahalanobis = sqrt(r^T * S^-1 * r)
    float rS_inv_r = residual[0] * S_inv[0][0] * residual[0] +
                     residual[0] * S_inv[0][1] * residual[1] +
                     residual[1] * S_inv[1][0] * residual[0] +
                     residual[1] * S_inv[1][1] * residual[1];
    last_mahalanobis = sqrtf(fmaxf(rS_inv_r, 0.0f));
    // State update: x = x + K * r
    for (int i = 0; i < N_STATE; i++)
      x[i] += K[i][0] * residual[0] + K[i][1] * residual[1];
    // v2.2 FIX: Correct Joseph-form covariance update
    // H = [[1,0,0,0,0],[0,1,0,0,0]]  (picks rows 0 and 1)
    // P_new = (I-KH)*P*(I-KH)^T + K*R*K^T
    //       = P - K*H*P - (K*H*P)^T + K*(H*P*H^T+R)*K^T
    //       = P - K*H*P - P*H^T*K^T + K*S*K^T
    //
    // K*H*P:  K[i][0]*P[0][j] + K[i][1]*P[1][j]
    // P*H^T*K^T: P[i][0]*K[j][0] + P[i][1]*K[j][1]
    // Save old P values that will be needed (rows 0,1 and cols 0,1)
    float P_row0[N_STATE], P_row1[N_STATE];
    float P_col0[N_STATE], P_col1[N_STATE];
    for (int j = 0; j < N_STATE; j++) {
      P_row0[j] = P.m[0][j];
      P_row1[j] = P.m[1][j];
    }
    for (int i = 0; i < N_STATE; i++) {
      P_col0[i] = P.m[i][0];
      P_col1[i] = P.m[i][1];
    }
    for (int i = 0; i < N_STATE; i++) {
      for (int j = 0; j < N_STATE; j++) {
        float P_ij = P.m[i][j];
        float KHP  = K[i][0] * P_row0[j] + K[i][1] * P_row1[j];
        float PHtKt = P_col0[i] * K[j][0] + P_col1[i] * K[j][1];
        // v2.2 FIX: Use S (innovation covariance), NOT R.
        // Joseph form: P_new = P - KHP - PHtKt + K*S*K^T
        //   where S = H*P*H^T + R (already computed above)
        // Using R here would make P go negative → NaN.
        float KSKt = K[i][0] * S[0][0] * K[j][0] + K[i][0] * S[0][1] * K[j][1] +
                     K[i][1] * S[1][0] * K[j][0] + K[i][1] * S[1][1] * K[j][1];
        P.m[i][j] = P_ij - KHP - PHtKt + KSKt;
      }
    }
    P.symmetrize();
    // Floor diagonal for numerical stability
    for (int i = 0; i < N_STATE; i++) {
      if (P.m[i][i] < 1e-8f) P.m[i][i] = 1e-8f;
    }
  }
  float neupertResidual() {
    return x[0] - sxr_background - alpha_neupert * x[2];
  }
};
// ==================== IMM CLASS ====================
class IMM {
public:
  UKF ukf_quiet;
  UKF ukf_flare;
  float mu[2];
  float Pi[2][2];
  int step_count;
  bool primary_alert;
  int primary_first_step;
  int primary_consec;
  int primary_confirm;
  float mu_threshold;
  int clear_counter;
  int total_alerts;
  int total_false_alarms;
  int total_quiet_steps;
  int total_active_steps;
  float x_combined[N_STATE];
  void init(float sxr_bg, float sxr_init, float hxr_init) {
    ukf_quiet.init(0.05f, 0.7f, sxr_bg, sxr_init, hxr_init);
    ukf_flare.init(0.05f, 0.3f, sxr_bg, sxr_init, hxr_init);
    mu[0] = 0.95f;
    mu[1] = 0.05f;
    Pi[0][0] = 0.98f; Pi[0][1] = 0.02f;
    Pi[1][0] = 0.10f; Pi[1][1] = 0.90f;
    step_count = 0;
    primary_alert = false;
    primary_first_step = -1;
    primary_consec = 0;
    primary_confirm = 2;        // v2.2: 2 steps at 1-min cadence
    mu_threshold = 0.5f;
    clear_counter = 0;
    total_alerts = 0;
    total_false_alarms = 0;
    total_quiet_steps = 0;
    total_active_steps = 0;
  }
  float step(float sxr, float hxr, float dt, bool has_data) {
    step_count++;
    // 1. Interaction (mixing)
    float c[2];
    c[0] = Pi[0][0] * mu[0] + Pi[1][0] * mu[1];
    c[1] = Pi[0][1] * mu[0] + Pi[1][1] * mu[1];
    if (c[0] < 1e-12f) c[0] = 1e-12f;
    if (c[1] < 1e-12f) c[1] = 1e-12f;
    float mu_mix[2][2];
    mu_mix[0][0] = Pi[0][0] * mu[0] / c[0];
    mu_mix[0][1] = Pi[1][0] * mu[1] / c[0];
    mu_mix[1][0] = Pi[0][1] * mu[0] / c[1];
    mu_mix[1][1] = Pi[1][1] * mu[1] / c[1];
    float x_mix_q[N_STATE], x_mix_f[N_STATE];
    for (int i = 0; i < N_STATE; i++) {
      x_mix_q[i] = mu_mix[0][0] * ukf_quiet.x[i] + mu_mix[1][0] * ukf_flare.x[i];
      x_mix_f[i] = mu_mix[0][1] * ukf_quiet.x[i] + mu_mix[1][1] * ukf_flare.x[i];
      ukf_quiet.x[i] = x_mix_q[i];
      ukf_flare.x[i] = x_mix_f[i];
    }
    // 2. Predict
    ukf_quiet.predict(dt);
    ukf_flare.predict(dt);
    // 3. Update
    float z[2] = {sxr, hxr};
    float* z_ptr = has_data ? z : nullptr;
    ukf_quiet.update(z_ptr, dt);
    ukf_flare.update(z_ptr, dt);
    // 4. Combine (model probability update)
    if (has_data) {
      float detS_q = ukf_quiet.S[0][0] * ukf_quiet.S[1][1] - ukf_quiet.S[0][1] * ukf_quiet.S[1][0];
      float detS_f = ukf_flare.S[0][0] * ukf_flare.S[1][1] - ukf_flare.S[0][1] * ukf_flare.S[1][0];
      if (fabsf(detS_q) < 1e-12f) detS_q = 1e-12f;
      if (fabsf(detS_f) < 1e-12f) detS_f = 1e-12f;
      float nis_q = ukf_quiet.last_mahalanobis * ukf_quiet.last_mahalanobis;
      float nis_f = ukf_flare.last_mahalanobis * ukf_flare.last_mahalanobis;
      float L_q = expf(-0.5f * fminf(nis_q, 50.0f)) / sqrtf(fmaxf(detS_q, 1e-12f));
      float L_f = expf(-0.5f * fminf(nis_f, 50.0f)) / sqrtf(fmaxf(detS_f, 1e-12f));
      if (L_q < 1e-300f) L_q = 1e-300f;
      if (L_f < 1e-300f) L_f = 1e-300f;
      float joint0 = c[0] * L_q;
      float joint1 = c[1] * L_f;
      float denom = joint0 + joint1;
      if (denom > 1e-300f) {
        mu[0] = joint0 / denom;
        mu[1] = joint1 / denom;
        if (mu[0] < 1e-6f) mu[0] = 1e-6f;
        if (mu[1] < 1e-6f) mu[1] = 1e-6f;
        float s = mu[0] + mu[1];
        mu[0] /= s;
        mu[1] /= s;
      }
    }
    // Combined state
    for (int i = 0; i < N_STATE; i++)
      x_combined[i] = mu[0] * ukf_quiet.x[i] + mu[1] * ukf_flare.x[i];
    // 5. Alert logic
    float maha = fmaxf(ukf_quiet.last_mahalanobis, ukf_flare.last_mahalanobis);
    bool raw_alert = (mu[1] > mu_threshold) && (maha > 0.5f);
    if (raw_alert) {
      primary_consec++;
    } else {
      primary_consec = 0;
    }
    if (primary_consec >= primary_confirm && !primary_alert) {
      primary_alert = true;
      primary_first_step = step_count - primary_confirm + 1;
      total_alerts++;
    }
    // Clear alert when signal drops for 5+ steps
    if (primary_alert && mu[1] < mu_threshold * 0.5f && primary_consec == 0) {
      clear_counter++;
      if (clear_counter >= 5) {
        primary_alert = false;
        clear_counter = 0;
      }
    } else {
      clear_counter = 0;
    }
    if (mu[1] > mu_threshold) total_active_steps++;
    else total_quiet_steps++;
    return mu[1];
  }
  float getMahalanobis() {
    return fmaxf(ukf_quiet.last_mahalanobis, ukf_flare.last_mahalanobis);
  }
  // v2.2: Simple alpha estimator (gradient descent on Neupert relationship)
  // alpha = alpha + lr * (sxr_excess - alpha*integral) * integral / (integral^2 + eps)
  void updateAlpha(float dt) {
    float sxr_excess = x_combined[0] - ukf_quiet.sxr_background;
    float integral = x_combined[2];
    float integ_sq = integral * integral;
    if (integ_sq < 1.0f) return;  // skip when integral is too small
    float lr = 0.01f * dt;  // learning rate scales with dt
    float innov = sxr_excess - ukf_quiet.alpha_neupert * integral;
    float new_alpha = ukf_quiet.alpha_neupert + lr * innov * integral / (integ_sq + 1e-6f);
    // Clamp to [0.001, alpha_ceil]
    float ceil = ukf_quiet.alpha_ceil;
    if (new_alpha < 0.001f) new_alpha = 0.001f;
    if (new_alpha > ceil) new_alpha = ceil;
    ukf_quiet.alpha_neupert = new_alpha;
    ukf_flare.alpha_neupert = new_alpha;
  }
};
// ==================== EXPERIENCE LAYER (CBR) ====================
// Types and globals (ExperienceCase, ExperienceAdvice, FEATURE_WEIGHTS,
// FEATURE_SCALES, case_base, n_cases_loaded) are defined at the TOP of
// this file to prevent Arduino IDE auto-prototype errors.
// Rolling history for feature extraction
float exp_alpha_hist[60];
float exp_maha_hist[60];
float exp_mu_hist[60];
float exp_sxr_hist[60];
float exp_hxr_hist[60];
float exp_integ_hist[60];
float exp_qpp_hist[60];
int exp_health_degraded_count = 0;
int exp_dropout_count = 0;
int exp_hist_idx = 0;
int exp_hist_count = 0;
// ==================== EXPERIENCE CASE LOADING ====================
//
// The 15 experience cases are EMBEDDED in the firmware (see embedded_cases.h,
// included near the top of this file). This means:
//   - No SPIFFS upload required
//   - No experience_db.json file to manage
//   - Just upload this .ino + embedded_cases.h and you're done
//
// If you want to update the case base later, edit embedded_cases.h directly
// (or regenerate it from experience_db.json using gen_embedded.py).
bool loadExperienceCaseBase() {
  // Copy the embedded cases into the runtime case_base[] array.
  // The EMBEDDED_CASES array is in flash (const), case_base[] is in RAM
  // so the CBR retrieval code can access it quickly.
  n_cases_loaded = 0;
  int n_to_copy = (EMBEDDED_CASE_COUNT < MAX_CASES) ? EMBEDDED_CASE_COUNT : MAX_CASES;
  for (int i = 0; i < n_to_copy; i++) {
    case_base[i] = EMBEDDED_CASES[i];
    n_cases_loaded++;
  }
  Serial.print(F("[EXP] Loaded "));
  Serial.print(n_cases_loaded);
  Serial.println(F(" embedded cases (no SPIFFS needed)"));
  return n_cases_loaded > 0;
}
// ==================== FEATURE EXTRACTION ====================
void observeExperience(float alpha, float maha, float mu_flare,
                       float sxr, float hxr, float integral,
                       float background, float qpp_signif,
                       const char* health, bool is_dropout) {
  exp_alpha_hist[exp_hist_idx] = alpha;
  exp_maha_hist[exp_hist_idx] = maha;
  exp_mu_hist[exp_hist_idx] = mu_flare;
  exp_sxr_hist[exp_hist_idx] = sxr;
  exp_hxr_hist[exp_hist_idx] = hxr;
  exp_integ_hist[exp_hist_idx] = integral;
  exp_qpp_hist[exp_hist_idx] = qpp_signif;
  if (strcmp(health, "UNHEALTHY") == 0) exp_health_degraded_count++;
  if (is_dropout) exp_dropout_count++;
  exp_hist_idx = (exp_hist_idx + 1) % 60;
  if (exp_hist_count < 60) exp_hist_count++;
}
void extractFeatureSignature(float* out) {
  if (exp_hist_count < 5) {
    for (int i = 0; i < N_FEATURES; i++) out[i] = 0.0f;
    return;
  }
  int w = (exp_hist_count < 30) ? exp_hist_count : 30;
  float alpha_now = exp_alpha_hist[(exp_hist_idx - 1 + 60) % 60];
  float alpha_first = exp_alpha_hist[(exp_hist_idx - w + 60) % 60];
  float alpha_slope = (alpha_now - alpha_first) / fmaxf(w - 1, 1) * 30.0f;
  float maha_max = -1e9f;
  float maha_first = exp_maha_hist[(exp_hist_idx - w + 60) % 60];
  float maha_now = exp_maha_hist[(exp_hist_idx - 1 + 60) % 60];
  for (int i = 0; i < w; i++) {
    float v = exp_maha_hist[(exp_hist_idx - w + i + 60) % 60];
    if (v > maha_max) maha_max = v;
  }
  float maha_slope = (maha_now - maha_first) / fmaxf(w - 1, 1) * 30.0f;
  float mu_sum = 0, mu_max = -1e9f;
  for (int i = 0; i < w; i++) {
    float v = exp_mu_hist[(exp_hist_idx - w + i + 60) % 60];
    mu_sum += v;
    if (v > mu_max) mu_max = v;
  }
  float mu_mean = mu_sum / w;
  float sxr_sum = 0, hxr_sum = 0;
  for (int i = 0; i < w; i++) {
    sxr_sum += exp_sxr_hist[(exp_hist_idx - w + i + 60) % 60];
    hxr_sum += exp_hxr_hist[(exp_hist_idx - w + i + 60) % 60];
  }
  float sxr_mean = sxr_sum / w;
  float hxr_mean = hxr_sum / w;
  float sxr_hxr_ratio = sxr_mean / fmaxf(hxr_mean, 1e-6f);
  float integ_first = exp_integ_hist[(exp_hist_idx - w + 60) % 60];
  float integ_now = exp_integ_hist[(exp_hist_idx - 1 + 60) % 60];
  float integ_growth = (integ_now - integ_first) / fmaxf(w - 1, 1);
  float qpp_max = -1e9f;
  for (int i = 0; i < w; i++) {
    float v = exp_qpp_hist[(exp_hist_idx - w + i + 60) % 60];
    if (v > qpp_max) qpp_max = v;
  }
  float degraded_frac = (float)exp_health_degraded_count / fmaxf(exp_hist_count, 1);
  float dropout_frac = (float)exp_dropout_count / fmaxf(exp_hist_count, 1);
  out[0]  = alpha_now;
  out[1]  = alpha_slope;
  out[2]  = maha_max;
  out[3]  = maha_slope;
  out[4]  = mu_mean;
  out[5]  = mu_max;
  out[6]  = sxr_hxr_ratio;
  out[7]  = integ_growth;
  out[8]  = sxr_mean;
  out[9]  = 0.0f;
  out[10] = qpp_max;
  out[11] = degraded_frac;
  out[12] = dropout_frac;
  out[13] = 60.0f;
}
// ==================== CBR RETRIEVAL ====================
float weightedDistance(const float* a, const float* b) {
  float sum = 0;
  for (int i = 0; i < N_FEATURES; i++) {
    float na = a[i] / (FEATURE_SCALES[i] + 1e-9f);
    float nb = b[i] / (FEATURE_SCALES[i] + 1e-9f);
    float diff = na - nb;
    sum += FEATURE_WEIGHTS[i] * diff * diff;
  }
  return sqrtf(sum);
}
ExperienceAdvice adviseFromExperience() {
  ExperienceAdvice advice;
  advice.n_similar = 0;
  advice.prior_P_flare = 0.5f;
  advice.recommended_alpha_ceiling = 2.0f;
  advice.recommended_confirm_steps = 4;
  advice.false_alarm_risk = 0.0f;
  advice.top_lesson[0] = '\0';
  advice.matched_case_ids[0] = '\0';
  if (n_cases_loaded == 0) return advice;
  float query[N_FEATURES];
  extractFeatureSignature(query);
  const int K = 3;
  const float MAX_DIST = 12.0f;
  int matched_idx[K];
  float matched_dist[K];
  for (int k = 0; k < K; k++) { matched_idx[k] = -1; matched_dist[k] = 1e9f; }
  for (int c = 0; c < n_cases_loaded; c++) {
    float d = weightedDistance(query, case_base[c].feature);
    for (int k = 0; k < K; k++) {
      if (d < matched_dist[k]) {
        for (int j = K - 1; j > k; j--) {
          matched_dist[j] = matched_dist[j - 1];
          matched_idx[j] = matched_idx[j - 1];
        }
        matched_dist[k] = d;
        matched_idx[k] = c;
        break;
      }
    }
  }
  float weights[K];
  float total_w = 0;
  int n_valid = 0;
  for (int k = 0; k < K; k++) {
    if (matched_idx[k] >= 0 && matched_dist[k] <= MAX_DIST) {
      weights[k] = expf(-(matched_dist[k] * matched_dist[k]) / (2.0f * 4.0f * 4.0f));
      total_w += weights[k];
      n_valid++;
    } else {
      weights[k] = 0;
    }
  }
  advice.n_similar = n_valid;
  if (n_valid == 0 || total_w <= 0) return advice;
  float fa_rate = 0, flare_rate = 0;
  for (int k = 0; k < K; k++) {
    if (matched_idx[k] < 0) continue;
    ExperienceCase& c = case_base[matched_idx[k]];
    bool is_flare = !c.false_alarm && !c.missed && (c.lead_time_steps != 0);
    fa_rate += weights[k] * (c.false_alarm ? 1.0f : 0.0f);
    flare_rate += weights[k] * (is_flare ? 1.0f : 0.0f);
  }
  advice.prior_P_flare = flare_rate / total_w;
  advice.false_alarm_risk = fa_rate / total_w;
  float max_alpha = 0;
  for (int k = 0; k < K; k++) {
    if (matched_idx[k] >= 0 && case_base[matched_idx[k]].alpha_final > max_alpha)
      max_alpha = case_base[matched_idx[k]].alpha_final;
  }
  advice.recommended_alpha_ceiling = fmaxf(2.0f, max_alpha * 1.5f);
  if (advice.false_alarm_risk > 0.4f) {
    advice.recommended_confirm_steps = 6;
  } else if (advice.prior_P_flare > 0.6f) {
    advice.recommended_confirm_steps = 3;
  } else {
    advice.recommended_confirm_steps = 4;
  }
  if (matched_idx[0] >= 0) {
    strncpy(advice.top_lesson, case_base[matched_idx[0]].lesson, MAX_LESSON_LEN - 1);
    advice.top_lesson[MAX_LESSON_LEN - 1] = '\0';
    strncpy(advice.matched_case_ids, case_base[matched_idx[0]].case_id, 119);
    advice.matched_case_ids[119] = '\0';
  }
  return advice;
}
// ==================== GLOBAL STATE ====================
IMM imm;
WebServer server(80);
#define MAX_SAMPLES 1440
float sxr_buffer[MAX_SAMPLES];
float hxr_buffer[MAX_SAMPLES];
float mu_buffer[MAX_SAMPLES];
float maha_buffer[MAX_SAMPLES];
int buffer_idx = 0;
int buffer_count = 0;
float current_sxr = 0.0f;
float current_hxr = 0.0f;
float current_mu_flare = 0.0f;
float current_maha = 0.0f;
float current_alpha = 0.05f;
bool alert_active = false;
int total_steps_processed = 0;
int total_false_alarms = 0;
int total_alerts = 0;
unsigned long last_fetch_time = 0;
float sxr_baseline = 1.0f;
float hxr_baseline = 0.5f;
bool baseline_established = false;
int baseline_samples = 0;
float sxr_baseline_sum = 0.0f;
float hxr_baseline_sum = 0.0f;
String donki_flare_info = "No DONKI data fetched yet";
// ==================== WiFi ====================
unsigned long last_wifi_retry_time = 0;
const unsigned long WIFI_RETRY_INTERVAL_MS = 15000;
IPAddress saved_ip;
bool wifi_connected = false;

void connectWiFi() {
  Serial.println(F("\n############################################"));
  Serial.println(F("#                                          #"));
  Serial.println(F("#   SOLIS - Solar Flare Nowcasting System  #"));
  Serial.println(F("#   Team Runtime Slayers | BAH 2026 PS-15  #"));
  Serial.println(F("#                                          #"));
  Serial.println(F("############################################"));
  Serial.println();
  Serial.print(F("[WiFi] Connecting to: "));
  Serial.println(WIFI_SSID);
  Serial.println();

  WiFi.disconnect(true);
  delay(200);
  WiFi.mode(WIFI_STA);
  WiFi.setAutoReconnect(true);
  WiFi.persistent(true);

  Serial.print(F("[WiFi] Status: Connecting"));
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  // Try for 60 seconds (120 x 500ms)
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 120) {
    delay(500);
    Serial.print(".");
    attempts++;
    if (attempts % 20 == 0) {
      Serial.println();
      Serial.print(F("[WiFi] Still trying... status = "));
      Serial.println(WiFi.status());
    }
  }

  if (WiFi.status() == WL_CONNECTED) {
    wifi_connected = true;
    saved_ip = WiFi.localIP();
    Serial.println();
    Serial.println();
    Serial.println(F("============================================"));
    Serial.println(F("  WIFI CONNECTED SUCCESSFULLY!"));
    Serial.println(F("============================================"));
    Serial.println();
    Serial.println(F("--- DASHBOARD ADDRESS (3 formats) ---"));
    Serial.println();
    Serial.print(F("  URL:      http://"));
    Serial.println(saved_ip);
    Serial.print(F("  IP:       "));
    Serial.println(saved_ip);
    Serial.print(F("  Hostname: http://"));
    Serial.print(MDNS_HOSTNAME);
    Serial.println(F(".local"));
    Serial.println();
    Serial.print(F("  Signal:   "));
    Serial.print(WiFi.RSSI());
    Serial.println(F(" dBm"));
    Serial.print(F("  MAC:      "));
    Serial.println(WiFi.macAddress());
    Serial.println();
    Serial.println(F("  Open any of the above in your browser."));
    Serial.println(F("  All 3 team members can view simultaneously"));
    Serial.println(F("  on the same WiFi."));
    Serial.println(F("============================================"));
    Serial.println();

    // Start mDNS
    if (MDNS.begin(MDNS_HOSTNAME)) {
      Serial.print(F("[mDNS] Started: http://"));
      Serial.print(MDNS_HOSTNAME);
      Serial.println(F(".local"));
      MDNS.addService("http", "tcp", 80);
    } else {
      Serial.println(F("[mDNS] Failed - use IP address instead"));
    }
    Serial.println();
  } else {
    wifi_connected = false;
    Serial.println();
    Serial.println();
    Serial.println(F("============================================"));
    Serial.println(F("  WIFI CONNECTION FAILED!"));
    Serial.println(F("============================================"));
    Serial.println();
    Serial.print(F("  Status code: "));
    Serial.println(WiFi.status());
    Serial.println();
    Serial.println(F("  Troubleshooting:"));
    Serial.println(F("  1. Check hotspot is ON"));
    Serial.println(F("  2. Check SSID and password"));
    Serial.println(F("  3. Must be 2.4GHz (ESP32 does not support 5GHz)"));
    Serial.println(F("  4. Move closer to WiFi source"));
    Serial.println(F("  5. System will retry every 15 seconds"));
    Serial.println(F("  6. Press RST button after fixing WiFi"));
    Serial.println(F("============================================"));
    Serial.println();
    last_wifi_retry_time = millis();
  }
}

void keepWiFiConnected() {
  if (WiFi.status() == WL_CONNECTED) {
    if (!wifi_connected) {
      wifi_connected = true;
      saved_ip = WiFi.localIP();
      Serial.println();
      Serial.println(F("============================================"));
      Serial.println(F("  WIFI RECONNECTED!"));
      Serial.println(F("============================================"));
      Serial.print(F("  Dashboard: http://"));
      Serial.println(saved_ip);
      Serial.print(F("  Or:        http://"));
      Serial.print(MDNS_HOSTNAME);
      Serial.println(F(".local"));
      Serial.println(F("============================================"));
      Serial.println();
    }
    return;
  }

  if (wifi_connected) {
    wifi_connected = false;
    Serial.println(F("[WiFi] Connection lost! Will retry..."));
  }

  unsigned long now = millis();
  if (last_wifi_retry_time == 0 || now - last_wifi_retry_time >= WIFI_RETRY_INTERVAL_MS) {
    last_wifi_retry_time = now;
    Serial.print(F("[WiFi] Retrying (status="));
    Serial.print(WiFi.status());
    Serial.println(F(")..."));
    WiFi.disconnect();
    delay(100);
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  }
}
// ==================== DATA FETCHING ====================
//
// Uses NOAA's 6-hour X-ray endpoint (~163 KB) instead of 1-day (~656 KB).
// The 6-hour data fits in ESP32 RAM. We use simple String search to find
// the last SXR (0.1-0.8nm) and HXR (0.05-0.4nm) flux values.
//
bool fetchXrayData() {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println(F("[NOAA] WiFi not connected. Skipping fetch."));
    return false;
  }

  WiFiClientSecure client;
  client.setInsecure();
  client.setTimeout(15);

  HTTPClient http;
  http.setConnectTimeout(10000);
  http.setTimeout(15000);
  http.begin(client, NOAAXRAY_URL);

  Serial.print(F("[NOAA] Fetching GOES X-ray (6-hour)... "));
  int httpCode = http.GET();
  Serial.print(F("HTTP "));
  Serial.println(httpCode);

  if (httpCode != 200) {
    Serial.printf("[NOAA] HTTP error: %d\n", httpCode);
    http.end();
    return false;
  }

  // 6-hour data is ~163KB — fits in ESP32 RAM (~320KB free heap)
  String payload = http.getString();
  http.end();

  int payloadLen = payload.length();
  Serial.print(F("[NOAA] Payload: "));
  Serial.print(payloadLen);
  Serial.println(F(" bytes"));

  if (payloadLen < 100) {
    Serial.println(F("[NOAA] Payload too small — RAM allocation may have failed"));
    return false;
  }

  // Simple parsing: search backwards for the last SXR and HXR flux values
  // JSON format: ...,"flux":1.84e-06,...,"energy":"0.1-0.8nm"},...
  // Note: "flux" comes BEFORE "energy" in each object

  float latest_sxr = -1, latest_hxr = -1;

  // Find the LAST occurrence of "0.1-0.8nm" (SXR channel)
  int sxr_pos = payload.lastIndexOf("0.1-0.8nm");
  if (sxr_pos > 0) {
    // Search backwards from this position to find the nearest "flux": value
    int flux_pos = payload.lastIndexOf("\"flux\":", sxr_pos);
    if (flux_pos > 0) {
      // Extract the number after "flux":
      String fluxStr = payload.substring(flux_pos + 7, flux_pos + 30);
      // Trim to just the number (stop at comma or space)
      int comma = fluxStr.indexOf(',');
      if (comma > 0) fluxStr = fluxStr.substring(0, comma);
      latest_sxr = fluxStr.toFloat();
    }
  }

  // Find the LAST occurrence of "0.05-0.4nm" (HXR channel)
  int hxr_pos = payload.lastIndexOf("0.05-0.4nm");
  if (hxr_pos > 0) {
    int flux_pos = payload.lastIndexOf("\"flux\":", hxr_pos);
    if (flux_pos > 0) {
      String fluxStr = payload.substring(flux_pos + 7, flux_pos + 30);
      int comma = fluxStr.indexOf(',');
      if (comma > 0) fluxStr = fluxStr.substring(0, comma);
      latest_hxr = fluxStr.toFloat();
    }
  }

  // Free the payload memory ASAP
  payload = "";

  Serial.printf("[NOAA] Parsed: SXR=%.2e HXR=%.2e\n", latest_sxr, latest_hxr);

  if (latest_sxr > 0 && latest_hxr > 0) {
    current_sxr = latest_sxr * 1e7f;
    current_hxr = latest_hxr * 1e7f;
    Serial.printf("[NOAA] Data: SXR=%.4f HXR=%.4f\n", current_sxr, current_hxr);
    if (!baseline_established) {
      sxr_baseline_sum += current_sxr;
      hxr_baseline_sum += current_hxr;
      baseline_samples++;
      if (baseline_samples >= 5) {
        sxr_baseline = sxr_baseline_sum / baseline_samples;
        hxr_baseline = hxr_baseline_sum / baseline_samples;
        baseline_established = true;
        Serial.printf("[NOAA] Baseline: SXR=%.4f HXR=%.4f (n=%d)\n",
                      sxr_baseline, hxr_baseline, baseline_samples);
        imm.init(sxr_baseline, current_sxr, current_hxr);
      }
    }
    return true;
  }

  Serial.println(F("[NOAA] No valid SXR/HXR pair found"));
  return false;
}
void fetchDONKIFlares() {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println(F("[DONKI] WiFi not connected. Skipping fetch."));
    return;
  }
  WiFiClientSecure client;
  client.setInsecure();
  client.setTimeout(15);

  HTTPClient http;
  http.setConnectTimeout(10000);
  http.setTimeout(15000);

  // URL already contains the API key (pre-appended at top of file)
  // No concatenation needed — eliminates API_KEY_MISSING error
  Serial.print(F("[DONKI] Fetching NASA flare data... "));
  http.begin(client, NASA_DONKI_URL);

  int httpCode = http.GET();
  Serial.print(F("HTTP "));
  Serial.println(httpCode);

  if (httpCode == 200) {
    String payload = http.getString();
    Serial.print(F("[DONKI] Payload size: "));
    Serial.print(payload.length());
    Serial.println(F(" bytes"));

    if (payload.length() > 10 && payload[0] == '[') {
#if ARDUINOJSON_VERSION_MAJOR >= 7
      JsonDocument doc;
#else
      DynamicJsonDocument doc(50000);
#endif
      DeserializationError error = deserializeJson(doc, payload);
      if (!error) {
        JsonArray arr = doc.as<JsonArray>();
        if (arr.size() > 0) {
          donki_flare_info = "NASA DONKI: " + String(arr.size()) + " flare(s) recently.";
          int n_show = (arr.size() < 3) ? arr.size() : 3;
          for (int i = 0; i < n_show; i++) {
            JsonObject flr = arr[i];
            String classType = flr["classType"];
            String beginTime = flr["beginTime"];
            donki_flare_info += "\n  " + classType + " at " + beginTime.substring(0, 16);
          }
        } else {
          donki_flare_info = "NASA DONKI: No recent flares.";
        }
      }
    }
  } else {
    Serial.print(F("[DONKI] Error! HTTP code: "));
    Serial.println(httpCode);
    // Read the error response body for debugging
    String errPayload = http.getString();
    Serial.print(F("[DONKI] Error response: "));
    Serial.println(errPayload);

    if (httpCode == 403) {
      donki_flare_info = "NASA DONKI: API key rejected (HTTP 403).\nCheck API key is valid.";
    } else if (httpCode == -1) {
      donki_flare_info = "NASA DONKI: Connection failed.\nCheck WiFi or DNS.";
    } else {
      donki_flare_info = "NASA DONKI: HTTP " + String(httpCode) + "\n" + errPayload.substring(0, 100);
    }
  }
  http.end();
}
// ============================================================================
// PROFESSIONAL ISRO-QUALITY WEB DASHBOARD
// Replace the existing handleRoot() and handleData() functions with these.
// This dashboard is designed for ISRO officials to understand:
//   1. What the system is doing (methodology)
//   2. Real-time results (data + detection)
//   3. System health and statistics
// ============================================================================
// ==================== WEB DASHBOARD ====================
void handleRoot() {
  // Build a professional dashboard HTML page
  // Design: Dark theme with ISRO-style colors, card-based layout
  // Shows: Status banner, real-time data, methodology, results, system info
  String html = "<!DOCTYPE html><html><head>\n";
  html += "<meta charset='utf-8'>\n";
  html += "<meta name='viewport' content='width=device-width, initial-scale=1'>\n";
  html += "<meta http-equiv='refresh' content='10'>\n";  // Refresh every 10 seconds
  html += "<title>SOLIS - Solar Flare Nowcasting System</title>\n";
  html += "<style>\n";
  // ---- CSS STYLES ----
  // Body: Dark space-themed background
  html += "*{box-sizing:border-box;margin:0;padding:0;}\n";
  html += "body{font-family:'Segoe UI',Arial,sans-serif;background:#0B1426;color:#E0E0E0;"
          "margin:0;padding:0;min-height:100vh;}\n";
  // Header
  html += ".header{background:linear-gradient(135deg,#0F2027,#203A43,#2C5364);"
          "padding:20px 30px;border-bottom:3px solid #37DCF2;}\n";
  html += ".header h1{color:#FFFFFF;font-size:24px;margin-bottom:4px;}\n";
  html += ".header .subtitle{color:#37DCF2;font-size:13px;}\n";
  html += ".header .team{color:#888;font-size:11px;margin-top:6px;}\n";
  // Container
  html += ".container{max-width:1200px;margin:0 auto;padding:20px;}\n";
  // Status banner (alert or quiet)
  html += ".status-banner{padding:20px;border-radius:10px;margin-bottom:20px;"
          "text-align:center;font-size:18px;font-weight:bold;}\n";
  html += ".status-alert{background:#E74C3C;color:#FFFFFF;"
          "border:2px solid #C0392B;animation:blink 1.5s infinite;}\n";
  html += ".status-quiet{background:#1E8449;color:#FFFFFF;"
          "border:2px solid #196F3D;}\n";
  html += "@keyframes blink{0%,100%{opacity:1;}50%{opacity:0.7;}}\n";
  // Grid layout for cards — 2 columns, compact
  html += ".grid{display:grid;grid-template-columns:repeat(2,1fr);"
          "gap:10px;margin-bottom:15px;}\n";
  // Cards
  html += ".card{background:#13243B;border-radius:8px;padding:12px;"
          "border:1px solid #1E3A5F;}\n";
  html += ".card h2{color:#37DCF2;font-size:12px;text-transform:uppercase;"
          "letter-spacing:1px;border-bottom:1px solid #1E3A5F;"
          "padding-bottom:6px;margin-bottom:8px;}\n";
  // Data rows — compact
  html += ".data-row{display:flex;justify-content:space-between;"
          "padding:5px 0;border-bottom:1px solid #1A2F4A;}\n";
  html += ".data-row:last-child{border-bottom:none;}\n";
  html += ".data-label{color:#8899AA;font-size:12px;}\n";
  html += ".data-value{color:#FFFFFF;font-weight:bold;font-size:13px;"
          "font-family:'Courier New',monospace;}\n";
  html += ".data-value-highlight{color:#37DCF2;}\n";
  html += ".data-value-alert{color:#E74C3C;}\n";
  html += ".data-value-safe{color:#2ECC71;}\n";
  // Progress bar
  html += ".progress-bar{width:100%;height:18px;background:#1A2F4A;"
          "border-radius:9px;overflow:hidden;margin-top:4px;margin-bottom:4px;}\n";
  html += ".progress-fill{height:100%;border-radius:9px;}\n";
  html += ".progress-safe{background:linear-gradient(90deg,#27AE60,#2ECC71);}\n";
  html += ".progress-warn{background:linear-gradient(90deg,#F39C12,#E67E22);}\n";
  html += ".progress-alert{background:linear-gradient(90deg,#E74C3C,#C0392B);}\n";
  // Methodology section — compact
  html += ".methodology{background:#13243B;border-radius:8px;padding:12px;"
          "border:1px solid #1E3A5F;margin-bottom:12px;}\n";
  html += ".methodology h2{color:#37DCF2;font-size:12px;text-transform:uppercase;"
          "letter-spacing:1px;border-bottom:1px solid #1E3A5F;"
          "padding-bottom:6px;margin-bottom:8px;}\n";
  html += ".methodology-step{display:flex;align-items:flex-start;"
          "padding:4px 0;border-bottom:1px solid #1A2F4A;}\n";
  html += ".methodology-step:last-child{border-bottom:none;}\n";
  html += ".step-num{background:#37DCF2;color:#0B1426;font-weight:bold;"
          "width:20px;height:20px;border-radius:50%;text-align:center;"
          "line-height:20px;font-size:10px;margin-right:8px;flex-shrink:0;}\n";
  html += ".step-content{flex:1;}\n";
  html += ".step-title{color:#FFFFFF;font-size:12px;font-weight:bold;}\n";
  html += ".step-desc{color:#8899AA;font-size:11px;margin-top:1px;}\n";
  // Stats grid — compact
  html += ".stats-grid{display:grid;grid-template-columns:repeat(4,1fr);"
          "gap:8px;margin-bottom:12px;}\n";
  html += ".stat-box{background:#13243B;border-radius:6px;padding:8px;"
          "text-align:center;border:1px solid #1E3A5F;}\n";
  html += ".stat-value{font-size:20px;font-weight:bold;color:#37DCF2;"
          "font-family:'Courier New',monospace;}\n";
  html += ".stat-label{font-size:9px;color:#8899AA;text-transform:uppercase;"
          "letter-spacing:1px;margin-top:2px;}\n";
  // Footer
  html += ".footer{text-align:center;padding:10px;color:#445566;"
          "font-size:10px;border-top:1px solid #1A2F4A;}\n";
  html += ".footer a{color:#37DCF2;text-decoration:none;}\n";
  // DONKI section
  html += ".donki{background:#13243B;border-radius:8px;padding:10px;"
          "border:1px solid #1E3A5F;margin-bottom:12px;}\n";
  html += ".donki h2{color:#37DCF2;font-size:12px;text-transform:uppercase;"
          "letter-spacing:1px;margin-bottom:6px;}\n";
  html += ".donki-content{color:#B0C4DE;font-size:11px;"
          "white-space:pre-wrap;font-family:monospace;}\n";
  html += "</style></head><body>\n";
  // ---- HEADER ----
  html += "<div class='header'>\n";
  html += "<h1>SOLIS - Solar Observation via Linear Inference Systems</h1>\n";
  html += "<div class='subtitle'>Solar Flare Nowcasting & Forecasting System | "
          "Aditya-L1 SoLEXS + HEL1OS</div>\n";
  html += "<div class='team'>Team Runtime Slayers | "
          "Bharatiya Antariksh Hackathon 2026 | Problem Statement 15</div>\n";
  html += "</div>\n";
  html += "<div class='container'>\n";
  // ---- STATUS BANNER ----
  if (alert_active) {
    html += "<div class='status-banner status-alert'>";
    html += "&#9888; PRE-FLARE ALERT ACTIVE!<br>";
    html += "<span style='font-size:14px;font-weight:normal;'>P(Pre-Flare) = ";
    html += String(current_mu_flare, 4);
    html += " | Mahalanobis = ";
    html += String(current_maha, 2);
    html += " sigma</span></div>\n";
  } else {
    html += "<div class='status-banner status-quiet'>";
    html += "&#10004; STATUS: QUIET SUN<br>";
    html += "<span style='font-size:14px;font-weight:normal;'>No pre-flare activity detected - System monitoring normally</span></div>\n";
  }
  // ---- STATS GRID (4 boxes) ----
  html += "<div class='stats-grid'>\n";
  // Stat 1: Steps
  html += "<div class='stat-box'>";
  html += "<div class='stat-value'>" + String(total_steps_processed) + "</div>";
  html += "<div class='stat-label'>Steps Processed</div></div>\n";
  // Stat 2: Alerts
  html += "<div class='stat-box'>";
  html += "<div class='stat-value'>" + String(total_alerts) + "</div>";
  html += "<div class='stat-label'>Total Alerts</div></div>\n";
  // Stat 3: False Alarms
  html += "<div class='stat-box'>";
  html += "<div class='stat-value'>" + String(total_false_alarms) + "</div>";
  html += "<div class='stat-label'>False Alarms</div></div>\n";
  // Stat 4: Cases Loaded
  html += "<div class='stat-box'>";
  html += "<div class='stat-value'>" + String(n_cases_loaded) + "</div>";
  html += "<div class='stat-label'>Experience Cases</div></div>\n";
  html += "</div>\n";  // end stats-grid
  // ---- MAIN GRID (2 columns) ----
  html += "<div class='grid'>\n";
  // ---- CARD 1: Real-Time X-ray Data ----
  html += "<div class='card'>\n";
  html += "<h2>&#128292; Real-Time X-ray Data</h2>\n";
  html += "<div class='data-row'><span class='data-label'>SXR Flux (0.1-0.8 nm)</span>"
          "<span class='data-value'>" + String(current_sxr, 4) + "</span></div>\n";
  html += "<div class='data-row'><span class='data-label'>HXR Flux (0.05-0.4 nm)</span>"
          "<span class='data-value'>" + String(current_hxr, 4) + "</span></div>\n";
  html += "<div class='data-row'><span class='data-label'>SXR Baseline</span>"
          "<span class='data-value'>" + String(sxr_baseline, 4) + "</span></div>\n";
  html += "</div>\n";  // end card 1
  // ---- CARD 2: Detection Status ----
  html += "<div class='card'>\n";
  html += "<h2>&#128269; Detection Status</h2>\n";
  // P(Pre-Flare) with progress bar
  html += "<div class='data-row'><span class='data-label'>P(Pre-Flare)</span>"
          "<span class='data-value ";
  if (current_mu_flare > 0.5) html += "data-value-alert";
  else if (current_mu_flare > 0.2) html += "data-value-highlight";
  else html += "data-value-safe";
  html += "'>" + String(current_mu_flare, 4) + "</span></div>\n";
  // Progress bar
  int pct = (int)(current_mu_flare * 100);
  if (pct > 100) pct = 100;
  html += "<div class='progress-bar'><div class='progress-";
  if (current_mu_flare > 0.5) html += "alert";
  else if (current_mu_flare > 0.2) html += "warn";
  else html += "safe";
  html += "' style='width:" + String(pct) + "%'></div></div>\n";
  html += "<div class='data-row'><span class='data-label'>Mahalanobis Distance</span>"
          "<span class='data-value ";
  if (current_maha > 3.0) html += "data-value-alert";
  else html += "data-value-safe";
  html += "'>" + String(current_maha, 2) + " sigma</span></div>\n";
  html += "<div class='data-row'><span class='data-label'>Alert Status</span>";
  if (alert_active) {
    html += "<span class='data-value data-value-alert'>ACTIVE</span></div>\n";
  } else {
    html += "<span class='data-value data-value-safe'>QUIET</span></div>\n";
  }
  html += "</div>\n";  // end card 2
  // ---- CARD 3: Kalman Filter Parameters ----
  html += "<div class='card'>\n";
  html += "<h2>&#9881; Kalman Filter Parameters</h2>\n";
  html += "<div class='data-row'><span class='data-label'>Alpha (Neupert constant)</span>"
          "<span class='data-value'>" + String(current_alpha, 4) + "</span></div>\n";
  html += "<div class='data-row'><span class='data-label'>Alpha Ceiling</span>"
          "<span class='data-value'>" + String(imm.ukf_flare.alpha_ceil, 2) + "</span></div>\n";
  html += "<div class='data-row'><span class='data-label'>Confirm Threshold</span>"
          "<span class='data-value'>" + String(imm.primary_confirm) + " steps</span></div>\n";
  html += "<div class='data-row'><span class='data-label'>Mu Threshold (P-flare trigger)</span>"
          "<span class='data-value'>" + String(imm.mu_threshold, 2) + "</span></div>\n";
  html += "</div>\n";  // end card 3
  // ---- CARD 4: System Health ----
  html += "<div class='card'>\n";
  html += "<h2>&#128241; System Health</h2>\n";
  html += "<div class='data-row'><span class='data-label'>Uptime</span>"
          "<span class='data-value'>" + String(millis() / 1000) + " sec</span></div>\n";
  html += "<div class='data-row'><span class='data-label'>WiFi Signal</span>"
          "<span class='data-value'>" + String(WiFi.RSSI()) + " dBm</span></div>\n";
  html += "<div class='data-row'><span class='data-label'>Free Heap Memory</span>"
          "<span class='data-value'>" + String(ESP.getFreeHeap()) + " bytes</span></div>\n";
  html += "<div class='data-row'><span class='data-label'>Data Source</span>"
          "<span class='data-value'>NOAA SWPC GOES-18</span></div>\n";
  html += "</div>\n";  // end card 4
  html += "</div>\n";  // end grid
  // ---- METHODOLOGY SECTION ----
  html += "<div class='methodology'>\n";
  html += "<h2>&#128218; How SOLIS Works - Methodology</h2>\n";
  html += "<div class='methodology-step'>";
  html += "<div class='step-num'>1</div>";
  html += "<div class='step-content'>";
  html += "<div class='step-title'>Data Collection</div>";
  html += "<div class='step-desc'>Receives Soft X-ray (SoLEXS) and Hard X-ray (HEL1OS) data from Aditya-L1 satellite. Currently using NOAA GOES-18 for live validation.</div>";
  html += "</div></div>\n";
  html += "<div class='methodology-step'>";
  html += "<div class='step-num'>2</div>";
  html += "<div class='step-content'>";
  html += "<div class='step-title'>Preprocessing - Hampel Filter</div>";
  html += "<div class='step-desc'>Removes cosmic-ray spikes and noise. Aligns the two X-ray channels in time using cross-correlation.</div>";
  html += "</div></div>\n";
  html += "<div class='methodology-step'>";
  html += "<div class='step-num'>3</div>";
  html += "<div class='step-content'>";
  html += "<div class='step-title'>Kalman Filter (UKF) - 5D State Estimation</div>";
  html += "<div class='step-desc'>Tracks 5 quantities: SXR, HXR, cumulative HXR integral, Spectral Hardness Ratio, and its rate of change. Uses 11 sigma points for optimal nonlinear estimation.</div>";
  html += "</div></div>\n";
  html += "<div class='methodology-step'>";
  html += "<div class='step-num'>4</div>";
  html += "<div class='step-content'>";
  html += "<div class='step-title'>IMM - Interacting Multiple Model</div>";
  html += "<div class='step-desc'>Runs two Kalman Filters in parallel: Quiet Sun model and Pre-Flare model. Automatically switches between them. Output: P(Pre-Flare) probability.</div>";
  html += "</div></div>\n";
  html += "<div class='methodology-step'>";
  html += "<div class='step-num'>5</div>";
  html += "<div class='step-content'>";
  html += "<div class='step-title'>Neupert Effect - Physics Check</div>";
  html += "<div class='step-desc'>Uses the known physics that Hard X-rays come BEFORE Soft X-rays during a flare. If HXR does not predict SXR, a flare is building up.</div>";
  html += "</div></div>\n";
  html += "<div class='methodology-step'>";
  html += "<div class='step-num'>6</div>";
  html += "<div class='step-content'>";
  html += "<div class='step-title'>Mahalanobis 3-Sigma Gate</div>";
  html += "<div class='step-desc'>Statistical test: is the current measurement genuinely unusual (beyond 3 standard deviations) or just normal noise?</div>";
  html += "</div></div>\n";
  html += "<div class='methodology-step'>";
  html += "<div class='step-num'>7</div>";
  html += "<div class='step-content'>";
  html += "<div class='step-title'>CBR - Case-Based Reasoning (Experience Layer)</div>";
  html += "<div class='step-desc'>Searches 15 past solar events for similar situations. Adjusts thresholds based on what worked before. Classical AI - no machine learning.</div>";
  html += "</div></div>\n";
  html += "<div class='methodology-step'>";
  html += "<div class='step-num'>8</div>";
  html += "<div class='step-content'>";
  html += "<div class='step-title'>Adaptive Confirmation and Alert</div>";
  html += "<div class='step-desc'>Waits 2-6 steps to confirm before firing alert (prevents false alarms). Output: LED + this Web Dashboard + JSON API.</div>";
  html += "</div></div>\n";
  html += "</div>\n";  // end methodology
  // ---- NASA DONKI SECTION ----
  html += "<div class='donki'>\n";
  html += "<h2>&#128640; NASA DONKI - Official Flare Events (Cross-Confirmation)</h2>\n";
  html += "<div class='donki-content'>" + donki_flare_info + "</div>\n";
  html += "</div>\n";
  // ---- FOOTER ----
  html += "<div class='footer'>\n";
  html += "IP: http://" + WiFi.localIP().toString() + " | Local WiFi only | ";
  html += "JSON: <a href='/data'>/data</a> | Auto-refresh: 10s<br>\n";
  html += "SOLIS | ESP32 WROOM DA | " + String(__DATE__) + " " + String(__TIME__) + " | ";
  html += "NO ML - Pure Classical Kalman + CBR\n";
  html += "</div>\n";
  html += "</div>\n";  // end container
  html += "</body></html>\n";
  server.send(200, "text/html", html);
}
void handleData() {
  // JSON API endpoint - returns all metrics as JSON
  String json = "{\n";
  json += "  \"system\": \"SOLIS - Solar Observation via Linear Inference Systems\",\n";
  json += "  \"team\": \"Runtime Slayers\",\n";
  json += "  \"problem_statement\": \"PS-15: Solar Flare Nowcasting using Aditya-L1\",\n";
  json += "  \"status\": \"" + String(alert_active ? "ALERT" : "QUIET") + "\",\n";
  json += "  \"realtime\": {\n";
  json += "    \"sxr\": " + String(current_sxr, 6) + ",\n";
  json += "    \"hxr\": " + String(current_hxr, 6) + ",\n";
  json += "    \"p_pre_flare\": " + String(current_mu_flare, 4) + ",\n";
  json += "    \"mahalanobis_sigma\": " + String(current_maha, 4) + ",\n";
  json += "    \"alpha_neupert\": " + String(current_alpha, 4) + "\n";
  json += "  },\n";
  json += "  \"adaptive_params\": {\n";
  json += "    \"alpha_ceiling\": " + String(imm.ukf_flare.alpha_ceil, 2) + ",\n";
  json += "    \"confirm_threshold_steps\": " + String(imm.primary_confirm) + ",\n";
  json += "    \"mu_threshold\": " + String(imm.mu_threshold, 2) + "\n";
  json += "  },\n";
  json += "  \"experience_layer\": {\n";
  json += "    \"cases_loaded\": " + String(n_cases_loaded) + ",\n";
  json += "    \"method\": \"Case-Based Reasoning (CBR) - classical AI, no ML\"\n";
  json += "  },\n";
  json += "  \"statistics\": {\n";
  json += "    \"steps_processed\": " + String(total_steps_processed) + ",\n";
  json += "    \"total_alerts\": " + String(total_alerts) + ",\n";
  json += "    \"false_alarms\": " + String(total_false_alarms) + ",\n";
  json += "    \"uptime_seconds\": " + String(millis() / 1000) + ",\n";
  json += "    \"free_heap_bytes\": " + String(ESP.getFreeHeap()) + ",\n";
  json += "    \"wifi_rssi_dbm\": " + String(WiFi.RSSI()) + "\n";
  json += "  },\n";
  json += "  \"data_source\": \"NOAA SWPC GOES-18 XRS (1-min cadence)\",\n";
  json += "  \"cross_confirmation\": \"NASA DONKI FLR API\"\n";
  json += "}\n";
  server.send(200, "application/json", json);
}
// ==================== MAIN ====================
void setup() {
  Serial.begin(115200);
  // Wait 3 seconds for the ESP32 boot ROM messages (74880 baud) to finish
  // Those garbled characters (JJJJJj!ABBAB) at the top are normal boot ROM output
  delay(3000);
  // Flush any remaining garbage
  while (Serial.available()) Serial.read();

  Serial.println();
  Serial.println(F("############################################"));
  Serial.println(F("#                                          #"));
  Serial.println(F("#   SOLIS - Solar Flare Nowcasting System  #"));
  Serial.println(F("#   Team Runtime Slayers | BAH 2026 PS-15  #"));
  Serial.println(F("#                                          #"));
  Serial.println(F("############################################"));
  Serial.println();

  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LED_QUIET);

  Serial.println(F("[SETUP] Step 1: Connecting to WiFi..."));
  connectWiFi();

  Serial.println(F("[SETUP] Step 2: Loading experience database..."));
  loadExperienceCaseBase();

  Serial.println(F("[SETUP] Step 3: Starting web server..."));
  server.on("/", handleRoot);
  server.on("/data", handleData);
  server.begin();
  Serial.println(F("[SETUP] Web server started on port 80"));

  Serial.println(F("[SETUP] Step 4: Fetching initial solar data..."));
  if (wifi_connected) {
    fetchXrayData();
    fetchDONKIFlares();
  } else {
    Serial.println(F("[SETUP] WiFi not connected - will fetch once available"));
  }

  Serial.println();
  Serial.println(F("============================================"));
  Serial.println(F("  SOLIS SYSTEM READY!"));
  Serial.println(F("============================================"));
  if (wifi_connected) {
    Serial.println();
    Serial.println(F("  +--------------------------------------+"));
    Serial.println(F("  |  DASHBOARD ACCESS ADDRESS:           |"));
    Serial.println(F("  |                                      |"));
    Serial.print(F("  |  http://"));
    Serial.print(saved_ip);
    int pad = 30 - String(saved_ip.toString().c_str()).length();
    for (int i = 0; i < pad; i++) Serial.print(F(" "));
    Serial.println(F("      |"));
    Serial.println(F("  +--------------------------------------+"));
    Serial.println();
    Serial.println(F("  HOW TO SHARE WITH YOUR TEAM:"));
    Serial.println(F("  1. This is a LOCAL IP (same WiFi only)"));
    Serial.println(F("  2. All 3 team members must connect to"));
    Serial.println(F("     the SAME WiFi hotspot"));
    Serial.println(F("  3. Open the address above in any browser"));
    Serial.println(F("  4. All 3 can view simultaneously"));
    Serial.println();
    Serial.println(F("  NOTE: This IP is NOT public on the"));
    Serial.println(F("  internet. It only works on the local"));
    Serial.println(F("  WiFi network. For internet access, use"));
    Serial.println(F("  port forwarding or a tunnel service."));
    Serial.println(F("============================================"));
  } else {
    Serial.println();
    Serial.println(F("  WiFi not connected yet."));
    Serial.println(F("  Will retry every 15 seconds."));
  }
  Serial.println();
  Serial.println(F("  Dashboard auto-refresh: 10 seconds"));
  Serial.println(F("  Data fetch: every 60 seconds"));
  Serial.println(F("  Experience layer: every 10 steps"));
  Serial.println(F("============================================"));
  Serial.println();
}
void loop() {
  server.handleClient();
  keepWiFiConnected();

  if (!wifi_connected) {
    delay(100);
    return;
  }

  unsigned long now = millis();
  if (now - last_fetch_time > FETCH_INTERVAL_MS || last_fetch_time == 0) {
    last_fetch_time = now;
    Serial.println(F("\n--- Fetching solar data ---"));
    bool got_data = fetchXrayData();
    if (total_steps_processed % 10 == 0 && total_steps_processed > 0) {
      fetchDONKIFlares();
    }
    if (got_data && baseline_established) {
      // v2.2: Query experience layer BEFORE stepping
      ExperienceAdvice advice;
      bool have_advice = false;
      if (n_cases_loaded > 0 && total_steps_processed > 30
          && total_steps_processed % 10 == 0) {
        advice = adviseFromExperience();
        have_advice = (advice.n_similar > 0);
      }
      // v2.2: Set adaptive confirmation threshold (BEFORE imm.step)
      float prev_maha = current_maha;
      int confirm_threshold;
      if (prev_maha > 5.0f) confirm_threshold = 2;
      else if (prev_maha > 3.0f) confirm_threshold = 2;
      else confirm_threshold = 3;
      if (have_advice) {
        confirm_threshold = (int)(0.7f * (float)advice.recommended_confirm_steps
                                  + 0.3f * (float)confirm_threshold);
        if (confirm_threshold < 2) confirm_threshold = 2;
      }
      imm.primary_confirm = confirm_threshold;
      // v2.2: Bayesian prior shift on mu_threshold
      float new_mu_threshold = 0.5f;
      if (have_advice) {
        new_mu_threshold = 0.5f + 0.3f * (0.5f - advice.prior_P_flare);
        if (new_mu_threshold < 0.30f) new_mu_threshold = 0.30f;
        if (new_mu_threshold > 0.70f) new_mu_threshold = 0.70f;
      }
      imm.mu_threshold = new_mu_threshold;
      // v2.2: Apply adaptive alpha ceiling (raise-only)
      if (have_advice && advice.recommended_alpha_ceiling > 2.0f) {
        float new_ceil = advice.recommended_alpha_ceiling;
        if (new_ceil > 10.0f) new_ceil = 10.0f;
        if (new_ceil > imm.ukf_flare.alpha_ceil) {
          imm.ukf_flare.alpha_ceil = new_ceil;
          imm.ukf_quiet.alpha_ceil = new_ceil;
          Serial.printf("[EXP] Raised alpha ceiling to %.2f (%d cases)\n",
                        new_ceil, advice.n_similar);
        }
      }
      // Step the filter
      float dt = 60.0f;
      float mu_flare = imm.step(current_sxr, current_hxr, dt, true);
      // v2.2: Update alpha (simple gradient estimator)
      imm.updateAlpha(dt);
      current_mu_flare = mu_flare;
      current_maha = imm.getMahalanobis();
      current_alpha = imm.ukf_flare.alpha_neupert;
      total_steps_processed++;
      // Feed observation to experience layer
      observeExperience(
        current_alpha, current_maha, current_mu_flare,
        current_sxr, current_hxr, imm.ukf_flare.x[2],
        sxr_baseline, 0.0f,
        "HEALTHY", false
      );
      // Print experience advice
      if (have_advice) {
        Serial.printf("[EXP] n=%d prior=%.2f FA_risk=%.2f ceil=%.2f confirm=%d mu_t=%.2f\n",
                      advice.n_similar, advice.prior_P_flare,
                      advice.false_alarm_risk, advice.recommended_alpha_ceiling,
                      advice.recommended_confirm_steps, new_mu_threshold);
        if (advice.top_lesson[0] != '\0') {
          char tmp[101];
          strncpy(tmp, advice.top_lesson, 100);
          tmp[100] = '\0';
          Serial.print(F("[EXP] Lesson: "));
          Serial.println(tmp);
        }
      }
      // Store in ring buffer
      sxr_buffer[buffer_idx] = current_sxr;
      hxr_buffer[buffer_idx] = current_hxr;
      mu_buffer[buffer_idx] = current_mu_flare;
      maha_buffer[buffer_idx] = current_maha;
      buffer_idx = (buffer_idx + 1) % MAX_SAMPLES;
      if (buffer_count < MAX_SAMPLES) buffer_count++;
      // Alert logic
      bool new_alert = imm.primary_alert;
      if (new_alert && !alert_active) {
        alert_active = true;
        total_alerts++;
        digitalWrite(LED_PIN, LED_ALERT);
        Serial.println(F("========================================"));
        Serial.println(F("PRE-FLARE ALERT TRIGGERED!"));
        Serial.printf("  P(Pre-Flare): %.4f\n", current_mu_flare);
        Serial.printf("  Mahalanobis: %.2f sigma\n", current_maha);
        Serial.printf("  SXR: %.4f, HXR: %.4f\n", current_sxr, current_hxr);
        Serial.printf("  Alpha: %.4f (ceil=%.2f)\n", current_alpha, imm.ukf_flare.alpha_ceil);
        Serial.printf("  Confirm: %d steps, Mu threshold: %.2f\n",
                      confirm_threshold, new_mu_threshold);
        if (have_advice) {
          Serial.printf("  [EXP] %d similar, prior=%.2f\n",
                        advice.n_similar, advice.prior_P_flare);
        }
        Serial.println(F("========================================"));
      } else if (!new_alert && alert_active) {
        alert_active = false;
        digitalWrite(LED_PIN, LED_QUIET);
        Serial.println(F("Alert cleared. Quiet monitoring."));
      }
      // Print status
      Serial.printf("[Step %d] SXR=%.4f HXR=%.4f P=%.4f Maha=%.2f Alpha=%.4f",
                    total_steps_processed, current_sxr, current_hxr,
                    current_mu_flare, current_maha, current_alpha);
      if (alert_active) Serial.print(F(" *** ALERT ***"));
      Serial.println();
      // False alarm check
      if (alert_active && current_mu_flare < 0.3f) {
        total_false_alarms++;
        Serial.printf("  [FALSE ALARM] P(Flare) dropped to %.3f\n",
                      current_mu_flare);
      }
    }
  }
}