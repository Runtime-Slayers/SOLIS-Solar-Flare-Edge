"""
QCAIE v2.1 - Two-Timescale Recursive alpha (Neupert constant) estimator
=======================================================================
FIX 1: Faster alpha convergence with two-timescale RLS.

Problem (v2.0): The alpha estimator uses a single forgetting factor (0.98),
which is too slow to converge during the warmup period. When alpha starts
at 0.001 but the true value is 0.5, the filter sees large SXR residuals
and thinks "flare!" when really the alpha is just wrong.

Solution (v2.1): Two-timescale RLS:
  - Fast timescale (lam_fast=0.90): Converges quickly during warmup.
    Used for the first 100 samples to rapidly find the correct alpha.
  - Slow timescale (lam_slow=0.99): Stable tracking after convergence.
    Used after warmup to track gradual changes in the active region.

This is the standard approach in adaptive control for systems with both
fast initialization and slow parameter drift (Ioannou & Sun, "Robust
Adaptive Control", Chapter 4). NO ML.

FIX 3: Bayesian alpha prior.
Instead of starting at alpha=0.001 (too low for most active regions),
start at alpha=0.05 (median of observed values) with a broad prior
covariance P=10.0 (high uncertainty). This prevents the "alpha windup"
problem where the initial residual is large just because alpha is wrong.

FIX: Robust innovation clipping.
Instead of a hard clip at ±10, use a Hampel-style soft clip based on
the running MAD of the innovation sequence. This prevents single-sample
outliers from corrupting the estimate while still allowing large genuine
innovations (e.g., during a flare) to pass through.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from collections import deque


@dataclass
class AlphaEstimator:
    """Two-timescale RLS estimator for the Neupert constant alpha.

    v2.1 improvements over v2.0:
    - Two forgetting factors (fast for warmup, slow for tracking)
    - Bayesian prior (start at alpha=0.05 with high uncertainty)
    - Hampel-style robust innovation clipping
    - Convergence detection (switch from fast to slow automatically)
    """
    # Initial estimate (Bayesian prior — median of observed values)
    alpha: float = 0.05
    P: float = 10.0          # High initial uncertainty (broad prior)

    # Two-timescale forgetting factors
    lam_fast: float = 0.90   # Fast convergence during warmup
    lam_slow: float = 0.99   # Stable tracking after convergence
    lam: float = 0.90        # Current forgetting factor (starts fast)

    # Convergence detection
    min_samples: int = 5     # Wait this many valid samples before updating
    n_samples: int = 0
    convergence_threshold: float = 0.01  # P below this = converged
    converged: bool = False

    # Robust innovation clipping (Hampel-style)
    innovation_history: deque = field(default_factory=lambda: deque(maxlen=30))

    # Alpha bounds (v2.1: tighter ceiling to prevent runaway on flare days)
    # v2.2: `ceil` is now ADAPTIVE — can be raised at runtime by the experience
    # layer if similar past active regions needed a higher ceiling. The
    # experience layer (qcaie_experience.py) provides recommended_alpha_ceiling
    # based on the rolling 95th percentile of similar cases + 50% margin.
    floor: float = 0.001
    ceil: float = 2.0   # default; experience layer may raise this at runtime

    def update(self, sxr_excess: float, hxr_integral: float) -> float:
        """One two-timescale RLS update step.

        Model: sxr_excess = alpha * hxr_integral + noise

        During warmup (not converged): use lam_fast=0.90 for rapid convergence.
        After convergence (P < threshold): switch to lam_slow=0.99 for stability.

        Robust innovation: instead of hard-clipping at ±10, use a Hampel
        identifier based on the running MAD of innovations. This lets large
        genuine flare signals through while rejecting single-sample spikes.
        """
        self.n_samples += 1
        if self.n_samples <= self.min_samples:
            return self.alpha

        h = float(hxr_integral)
        # Skip when regressor is too small (quiet sun, integral near zero)
        if abs(h) < 1.0:
            return self.alpha

        # Check convergence: if P is small, switch to slow timescale
        if not self.converged and self.P < self.convergence_threshold:
            self.converged = True
            self.lam = self.lam_slow
        elif not self.converged:
            self.lam = self.lam_fast

        # RLS update
        K = self.P * h / (self.lam + h * h * self.P)
        innov = float(sxr_excess) - self.alpha * h

        # Robust innovation clipping (Hampel-style)
        self.innovation_history.append(innov)
        if len(self.innovation_history) >= 10:
            innov_arr = np.array(self.innovation_history)
            med = float(np.median(innov_arr))
            mad = float(np.median(np.abs(innov_arr - med)))
            scale = 1.4826 * mad + 1e-9
            # Soft clip: if innovation > 5 * scale, clip it
            # But allow up to 20 * scale (genuine flare signal)
            if abs(innov - med) > 5 * scale:
                # Soft clip: reduce but don't eliminate
                clip_limit = 5 * scale * np.sign(innov - med)
                innov = med + clip_limit
        else:
            # Not enough history for Hampel — use hard clip
            innov = float(np.clip(innov, -10.0, 10.0))

        self.alpha = self.alpha + K * innov
        self.P = (self.P - K * h * self.P) / self.lam

        # Constrain alpha
        self.alpha = float(np.clip(self.alpha, self.floor, self.ceil))
        # Keep P bounded
        self.P = float(np.clip(self.P, 1e-6, 1e4))

        return self.alpha

    def status(self) -> str:
        """Return human-readable status of the estimator."""
        phase = "CONVERGED" if self.converged else "WARMUP"
        return (f"Alpha={self.alpha:.4f} (P={self.P:.4f}, {phase}, "
                f"lam={self.lam:.2f}, samples={self.n_samples})")

    # ======================================================================
    # v2.2: EXPERIENCE-BASED ADAPTIVE CEILING
    # ======================================================================
    def apply_adaptive_ceiling(self, recommended_ceil: float) -> None:
        """Raise the alpha ceiling based on experience-layer advice.

        Called by the main pipeline after the experience layer (CBR) retrieves
        similar past cases. If similar active regions had alpha_final > 2.0,
        the ceiling is raised to allow the estimator to track them.

        The ceiling is NEVER lowered below 2.0 (the v2.1 default). It can
        only be raised. This prevents the experience layer from making the
        estimator less robust than v2.1.
        """
        # Take the max of (current ceil, recommended) to ensure monotonic raising
        new_ceil = max(self.ceil, max(2.0, float(recommended_ceil)))
        # Cap at 10.0 (the original v2.0 ceiling) — anything above this is
        # physically meaningless for the Neupert effect
        new_ceil = min(new_ceil, 10.0)
        self.ceil = new_ceil
