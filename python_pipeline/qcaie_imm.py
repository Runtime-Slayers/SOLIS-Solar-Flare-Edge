"""
QCAIE v2.0 - Interacting Multiple Model (IMM) estimator
========================================================
Two UKFs run in parallel, each tuned for a different solar regime:
    Model 1 (Quiet Sun) : small Q, beta=0.7  (physics-driven, expects stability)
    Model 2 (Pre-Flare) : large Q, beta=0.3  (data-driven, tracks rapid changes)

A Markov transition matrix governs switching:
    Pi = | 0.95  0.05 |    quiet -> quiet = 95%, quiet -> flare = 5%
         | 0.10  0.90 |    flare  -> quiet = 10%, flare -> flare = 90%

Each IMM cycle: Interaction -> Predict -> Update -> Combine.
The model probabilities are updated by the measurement likelihood under each
model's innovation covariance, providing a statistically rigorous mode switch.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import multivariate_normal

from qcaie_ukf import UKF


class IMM:
    """Interacting Multiple Model with two UKFs.

    Parameters
    ----------
    ukf_quiet, ukf_flare : UKF
        Pre-instantiated UKFs for the two regimes.
    Pi : ndarray of shape (2, 2)
        Markov transition matrix. Pi[i, j] = P(model j at t+1 | model i at t).
    """

    def __init__(self, ukf_quiet: UKF, ukf_flare: UKF,
                 Pi: np.ndarray | None = None):
        self.models = [ukf_quiet, ukf_flare]
        if Pi is None:
            Pi = np.array([[0.95, 0.05], [0.10, 0.90]])
        self.Pi = np.asarray(Pi, dtype=float)
        # Initial model probabilities: 95% quiet, 5% flare
        self.mu = np.array([0.95, 0.05])
        self._step = 0

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
    def step(self, z: np.ndarray | None, dt: float) -> dict:
        """Run one IMM cycle: Interaction -> Predict -> Update -> Combine.

        Returns a dict of diagnostics: combined state, model probabilities,
        per-model likelihoods, Mahalanobis distances, Neupert residual.
        """
        # ----------------------------------------------------------------
        # 1. INTERACTION (mixing)
        # ----------------------------------------------------------------
        # c_j = sum_i mu_i * Pi[i, j]  (normalization constants)
        c = self.Pi.T @ self.mu
        c = np.maximum(c, 1e-12)
        # mixing probabilities mu_ij = Pi[i, j] * mu_i / c_j
        mu_mix = np.zeros((2, 2))
        for j in range(2):
            for i in range(2):
                mu_mix[i, j] = self.Pi[i, j] * self.mu[i] / c[j]

        # Mixed initial conditions for each model
        mixed_states = []
        mixed_covs = []
        for j in range(2):
            x_mix = sum(mu_mix[i, j] * self.models[i].x for i in range(2))
            P_mix = np.zeros_like(self.models[0].P)
            for i in range(2):
                dx = self.models[i].x - x_mix
                P_mix += mu_mix[i, j] * (self.models[i].P + np.outer(dx, dx))
            mixed_states.append(x_mix)
            mixed_covs.append(P_mix)

        # Apply mixed initial conditions
        for j in range(2):
            self.models[j].x = mixed_states[j].copy()
            self.models[j].P = mixed_covs[j].copy()

        # ----------------------------------------------------------------
        # 2. PREDICT (each model independently)
        # ----------------------------------------------------------------
        for m in self.models:
            m.predict(dt)

        # ----------------------------------------------------------------
        # 3. UPDATE (each model processes the measurement; dropout-aware)
        # ----------------------------------------------------------------
        likelihoods = np.zeros(2)
        for k, m in enumerate(self.models):
            m.update(z, dt)
            # Compute Gaussian likelihood of measurement under model's innovation
            if z is None:
                # Dropout: no likelihood update; keep probabilities
                likelihoods[k] = 1.0
            else:
                residual = m.last_residual
                S = m.S
                try:
                    # Use multivariate normal pdf of the residual under N(0, S)
                    rv = multivariate_normal(mean=np.zeros(2), cov=S,
                                             allow_singular=True)
                    likelihoods[k] = max(rv.pdf(residual), 1e-300)
                except Exception:
                    likelihoods[k] = 1e-300

        # ----------------------------------------------------------------
        # 4. COMBINE (update model probabilities, fuse state)
        # ----------------------------------------------------------------
        if z is not None:
            # mu_new_j = c_j * L_j / sum_k (c_k * L_k)
            joint = c * likelihoods
            denom = np.sum(joint)
            if denom < 1e-300:
                # Degenerate: keep current probabilities
                pass
            else:
                self.mu = joint / denom
                # Floor to avoid lock-in
                self.mu = np.maximum(self.mu, 1e-6)
                self.mu = self.mu / np.sum(self.mu)

        # Combined state and covariance
        x_comb = sum(self.mu[k] * self.models[k].x for k in range(2))
        P_comb = np.zeros_like(self.models[0].P)
        for k in range(2):
            dx = self.models[k].x - x_comb
            P_comb += self.mu[k] * (self.models[k].P + np.outer(dx, dx))

        # Diagnostics
        diag = {
            "step": self._step,
            "x": x_comb.copy(),
            "P": P_comb.copy(),
            "mu_quiet": float(self.mu[0]),
            "mu_flare": float(self.mu[1]),
            "likelihood_quiet": float(likelihoods[0]),
            "likelihood_flare": float(likelihoods[1]),
            "mahalanobis_quiet": float(self.models[0].last_mahalanobis),
            "mahalanobis_flare": float(self.models[1].last_mahalanobis),
            "neupert_residual_quiet": float(self.models[0].neupert_residual()),
            "neupert_residual_flare": float(self.models[1].neupert_residual()),
            "x_quiet": self.models[0].x.copy(),
            "x_flare": self.models[1].x.copy(),
            "R_quiet": self.models[0].R.copy(),
            "R_flare": self.models[1].R.copy(),
        }
        self._step += 1
        return diag
