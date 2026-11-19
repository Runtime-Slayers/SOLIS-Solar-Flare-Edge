// ============================================================================
// embedded_cases.h — Experience database embedded as C++ data
// ============================================================================
// This file is auto-generated from experience_db.json. It contains 15 cases
// built from real SoLEXS, GOES-18, and synthetic adversarial datasets.
//
// Embedding the cases directly in the firmware means:
//   - No SPIFFS upload required
//   - No experience_db.json file to manage
//   - Single .ino upload is all that's needed
//
// To update the case base: edit this file directly, or regenerate from JSON.
// ============================================================================

#pragma once

// 15 cases, each with 14-dim feature signature + outcome + lesson
// (structs ExperienceCase and ExperienceAdvice must be defined BEFORE
//  this file is included — see QCAIE_v2_ESP32.ino)

const int EMBEDDED_CASE_COUNT = 15;

const ExperienceCase EMBEDDED_CASES[15] = {

  // Case 0: solexs_20250320_quiet
  {
    "solexs_20250320_quiet",
    "SoLEXS",
    "2025-03-20",
    "QUIET",
    1.0f,
    { 0.113622f, 0.003254f, 3.706479f, -0.178932f, 0.043777f, 0.121191f, 134.387004f, 0.021107f, 2.760000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 1.000000f },
    0.1409f,
    false,
    false,
    true,
    -109,
    "On a quiet day during solar max, P(Pre-Flare) still reached 0.99 for a minor event (HXR peak only ~5x baseline). The model correctly alerted 201 steps (3.35 min) before the peak. Alpha converg..."
  },

  // Case 1: solexs_20250320_pre_flare
  {
    "solexs_20250320_pre_flare",
    "SoLEXS",
    "2025-03-20",
    "QUIET",
    1.0f,
    { 0.366647f, 0.000000f, 2.843742f, 1.626791f, 0.125056f, 0.557617f, 110.736692f, 0.019795f, 2.760000f, 0.000000f, 0.000172f, 0.000000f, 0.000000f, 1.000000f },
    0.1409f,
    false,
    false,
    true,
    -109,
    "Pre-flare buildup signature on a quiet day. Mahalanobis rising, alpha stable, mu_flare climbing. This signature should NOT be suppressed — it was a real (minor) event."
  },

  // Case 2: solexs_20250328_pre_flare_active
  {
    "solexs_20250328_pre_flare_active",
    "SoLEXS",
    "2025-03-28",
    "M",
    1.0f,
    { 0.192706f, 0.025039f, 3.358923f, -0.540203f, 0.093675f, 0.190876f, 17.066076f, 4.312546f, 54.090000f, 0.000000f, 0.566352f, 0.000000f, 0.000000f, 1.000000f },
    2.0000f,
    false,
    false,
    true,
    -336,
    "On an active day with a major flare (~180x baseline HXR), alpha hit the v2.1 ceiling of 2.0. P(Pre-Flare) only reached 0.68 because the ceiling made the Neupert prediction too aggressive, shri..."
  },

  // Case 3: solexs_20250328_peak_M
  {
    "solexs_20250328_peak_M",
    "SoLEXS",
    "2025-03-28",
    "M",
    1.0f,
    { 0.453847f, 0.004210f, 2.355743f, 0.017908f, 0.261261f, 0.378582f, 79.799424f, 8.376160f, 54.090000f, 0.000000f, 2.303338f, 0.000000f, 0.000000f, 1.000000f },
    2.0000f,
    false,
    false,
    true,
    -336,
    "M-class flare peak signature: HXR ~180x baseline, alpha at ceiling. Lead time was 675 steps (11.25 min) — the Neupert effect correctly detected HXR buildup before SXR thermalization. This is t..."
  },

  // Case 4: solexs_20250328_quiet_before_flare
  {
    "solexs_20250328_quiet_before_flare",
    "SoLEXS",
    "2025-03-28",
    "QUIET",
    1.0f,
    { 0.242290f, 0.004428f, 2.640199f, 0.861637f, 0.063671f, 0.138563f, 107.939142f, 0.003082f, 54.090000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 1.000000f },
    2.0000f,
    false,
    false,
    true,
    -336,
    "Quiet period on an active day (before the major flare). Background is elevated but stable. P(Pre-Flare) is low. This signature should NOT trigger an alert — but watch for rising alpha or Maha ..."
  },

  // Case 5: solexs_20260615_pre_flare_B
  {
    "solexs_20260615_pre_flare_B",
    "SoLEXS",
    "2026-06-15",
    "B",
    1.0f,
    { 0.001000f, 0.000000f, 1.681724f, 0.070011f, 0.018861f, 0.023428f, 45.104407f, 0.004648f, 0.240000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 1.000000f },
    0.0287f,
    false,
    false,
    true,
    -38,
    "Pre-flare B-class signature on a very quiet day. HXR peak only 11 counts. The model alerted 38 seconds BEFORE the peak — pre-flare detection achieved on real ISRO data. Lesson: even very weak ..."
  },

  // Case 6: goes18_20260621_pre_flare_M27
  {
    "goes18_20260621_pre_flare_M27",
    "GOES-18",
    "2026-06-21",
    "M",
    60.0f,
    { 0.014059f, -0.000189f, 2.033242f, 0.161555f, 0.252300f, 0.506341f, 17.039347f, 6.684438f, 6.699981f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 60.000000f },
    0.0010f,
    false,
    false,
    true,
    -854,
    "Live M2.7 flare on GOES-18 (1-minute cadence). P(Pre-Flare) rose from 0.0004 to 0.997 in 8 minutes (steps 0-8). Alert fired 855 minutes (14.25 hours) BEFORE the M2.7 peak — extreme pre-flare d..."
  },

  // Case 7: goes18_20260621_quiet_M27
  {
    "goes18_20260621_quiet_M27",
    "GOES-18",
    "2026-06-21",
    "QUIET",
    60.0f,
    { 0.013144f, -0.001505f, 2.033242f, 0.608981f, 0.195779f, 0.370675f, 20.249943f, 7.113136f, 6.699981f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 60.000000f },
    0.0010f,
    false,
    false,
    true,
    -854,
    "Quiet period before the M2.7 flare. SXR at B-class level (~1e-6 W/m^2). P(Pre-Flare) is near zero. Sensor degraded (10% UNHEALTHY steps) but filter correctly stayed quiet. Lesson: don't be foo..."
  },

  // Case 8: adv_S1_extreme_noise
  {
    "adv_S1_extreme_noise",
    "synthetic-adversarial",
    "2025-06-17",
    "QUIET",
    1.0f,
    { 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f },
    0.5000f,
    false,
    false,
    true,
    0,
    "Extreme noise scenario: SNR < 0 dB. Hampel filter + MAD baseline reject the noise. Lesson: when background_variability > 1.0, require 6+ confirmation steps (cautious)."
  },

  // Case 9: adv_S2_long_dropout_30s
  {
    "adv_S2_long_dropout_30s",
    "synthetic-adversarial",
    "2025-06-17",
    "QUIET",
    1.0f,
    { 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f },
    0.5000f,
    false,
    false,
    true,
    0,
    "30-second dropout scenario: filter goes predict-only. Lesson: when dropout_frac_30 > 0.5, suppress primary alerts entirely; the filter has no data to confirm a flare."
  },

  // Case 10: adv_S3_cosmic_ray_spikes
  {
    "adv_S3_cosmic_ray_spikes",
    "synthetic-adversarial",
    "2025-06-17",
    "QUIET",
    1.0f,
    { 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f },
    0.5000f,
    false,
    false,
    true,
    0,
    "Cosmic-ray spike scenario: 5% of samples are 10x spikes. Hampel filter rejects them. Lesson: spike rate doesn't affect alerts — Hampel is the right defense."
  },

  // Case 11: adv_S4_baseline_drift
  {
    "adv_S4_baseline_drift",
    "synthetic-adversarial",
    "2025-06-17",
    "QUIET",
    1.0f,
    { 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f },
    0.5000f,
    false,
    false,
    true,
    0,
    "Baseline drift scenario: SXR baseline ramps 3x over 600 steps. v2.1 weakness: false alarm at the end of the ramp. Lesson: when alpha_slope_30 > 0.05 (alpha rising), the drift is being absorbed..."
  },

  // Case 12: adv_S5_multi_flare
  {
    "adv_S5_multi_flare",
    "synthetic-adversarial",
    "2025-06-17",
    "M",
    1.0f,
    { 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f },
    0.5000f,
    false,
    false,
    true,
    0,
    "Multi-flare scenario: 3 flares in 600 steps. Lesson: each flare needs its own confirmation. The 4-step confirmation correctly resets between flares."
  },

  // Case 13: adv_S6_weak_flare
  {
    "adv_S6_weak_flare",
    "synthetic-adversarial",
    "2025-06-17",
    "B",
    1.0f,
    { 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f },
    0.5000f,
    false,
    false,
    true,
    0,
    "Weak B-class flare: brief HXR spike, 2-3 step duration. Lesson: the adaptive confirmation (2 steps for Maha > 10) correctly catches these. Don't raise the threshold for this signature."
  },

  // Case 14: adv_S9_gradual_heating
  {
    "adv_S9_gradual_heating",
    "synthetic-adversarial",
    "2025-06-17",
    "QUIET",
    1.0f,
    { 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f },
    0.5000f,
    false,
    false,
    true,
    0,
    "Gradual heating scenario: SXR ramps slowly with no impulsive HXR. Correctly classified as true-negative (no flare). Lesson: when hxr_integral_growth < 0.01 AND maha_slope_30 < 0.5, it's gradua..."
  }

};
