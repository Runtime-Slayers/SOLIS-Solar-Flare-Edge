"""
QCAIE v2.0 - Unscented Kalman Filter (UKF) core
================================================
5-dimensional state vector tracking solar X-ray energy balance:
    x[0] : SXR(t)                 -- Soft X-ray flux (thermal), W/m^2
    x[1] : HXR(t)                 -- Hard X-ray flux (non-thermal), W/m^2
    x[2] : integral(HXR, 0, t)    -- Cumulative hard X-ray energy, J/m^2
    x[3] : SHR = HXR/SXR           -- Spectral Hardness Ratio (dimensionless)
    x[4] : d(SHR)/dt               -- Rate of change of hardness, 1/s

State transition encodes the Neupert effect constraint:
    x[2](t+dt) = x[2](t) + max(x[1], 0) * dt                       (HXR integral)
    x[3](t+dt) = x[1](t+dt) / max(x[0](t+dt), eps)                  (SHR)
    x[4](t+dt) = (x[3](t+dt) - x[3](t)) / dt                        (SHR derivative)
    x[0](t+dt) = beta * x[0](t) + (1-beta) * alpha * x[2](t)        (Neupert SXR)
    x[1](t+dt) = x[1](t)                                            (random walk)

The blending parameter `beta` controls physics-vs-data trust:
    Quiet Sun  model : beta = 0.7  (physics-driven, expects stability)
    Pre-Flare  model : beta = 0.3  (data-driven, tracks rapid changes)

References:
- Neupert, W.M. (1968), ApJ 153, L59 -- classic Neupert effect
- Julier & Uhlmann (2004), "Unscented Filtering and Nonlinear Estimation"
- Bar-Shalom et al., "Estimation with Applications to Tracking and Navigation"
"""
from __future__ import annotations

import numpy as np

EPS = 1e-12  # numerical floor for divisions


# ---------------------------------------------------------------------------
# State transition and measurement functions
# ---------------------------------------------------------------------------
def state_transition(x: np.ndarray, dt: float, alpha: float, beta: float,
                     sxr_background: float = 1.0,
                     integral_tau: float = 300.0) -> np.ndarray:
    """Neupert-constrained non-linear state transition f(x, dt).

    Physical model:
        SXR(t) = SXR_bg + excess_thermal(t)
        excess_thermal(t+dt) = beta * excess_thermal(t) + (1-beta) * alpha * integral(HXR, 0, t)

    The integral uses a LEAKY INTEGRATOR with time constant `integral_tau`
    (in SECONDS, default 300s = 5 minutes). This implements the methodology's
    "rolling 60-step window" for the Neupert integral: instead of accumulating
    from t=0 (which grows unboundedly and causes numerical issues on long
    runs), the integral decays exponentially with time constant integral_tau.

    The leak factor per step is exp(-dt / integral_tau), so the same time
    constant works regardless of cadence. At dt=1s, leak=exp(-1/300)=0.9967.
    At dt=60s, leak=exp(-60/300)=0.819. The 300s time constant is long
    enough to capture the full duration of a typical flare impulsive phase
    (~60s) plus its decay (~300s), but short enough to prevent unbounded
    growth on multi-hour observations.
    """
    x_next = x.copy()
    # 1. Neupert-constrained SXR prediction (excess over background)
    sxr_excess_old = x[0] - sxr_background
    sxr_excess_new = beta * sxr_excess_old + (1.0 - beta) * alpha * x[2]
    sxr_new = sxr_background + sxr_excess_new
    sxr_new = max(sxr_new, EPS)
    # 2. HXR is propagated as random walk (the measurement will correct it).
    hxr_new = max(x[1], EPS)
    # 3. Leaky cumulative HXR energy integral. Time constant is in seconds
    # so the leak factor is dt-independent.
    leak = float(np.exp(-dt / max(integral_tau, 1e-6)))
    integ_new = leak * x[2] + max(x[1], 0.0) * dt
    # 4. Spectral Hardness Ratio
    shr_new = hxr_new / max(sxr_new, EPS)
    # 5. SHR rate of change (clamped)
    dshr_new = (shr_new - x[3]) / dt
    dshr_new = float(np.clip(dshr_new, -100.0, 100.0))

    x_next[0] = sxr_new
    x_next[1] = hxr_new
    x_next[2] = integ_new
    x_next[3] = shr_new
    x_next[4] = dshr_new
    return x_next


def measurement_function(x: np.ndarray) -> np.ndarray:
    """h(x) = [x[0], x[1]] -- we directly observe SXR and HXR."""
    return np.array([x[0], x[1]])


# ---------------------------------------------------------------------------
# UKF implementation
# ---------------------------------------------------------------------------
class UKF:
    """Unscented Kalman Filter with Neupert-constrained state transition.

    Parameters
    ----------
    x0 : ndarray of shape (5,)
        Initial state vector.
    P0 : ndarray of shape (5, 5)
        Initial covariance.
    Q : ndarray of shape (5, 5)
        Process noise covariance. Use a small Q for the Quiet Sun model and a
        larger Q for the Pre-Flare model.
    R : ndarray of shape (2, 2)
        Measurement noise covariance (SXR, HXR). Will be adapted online if
        `adaptive_R` is True.
    alpha : float
        Neupert proportionality constant linking SXR to the HXR integral.
        SXR ~ alpha * integral(HXR).
    beta : float
        Neupert blending factor (0.7 for Quiet Sun, 0.3 for Pre-Flare).
    alpha_ukf : float
        UKF sigma-point spread parameter (typical 1e-3 to 1).
    kappa : float
        UKF secondary sigma-point parameter (typical 0).
    adaptive_R : bool
        If True, dynamically update R using an EMA of recent residuals.
    """

    def __init__(
        self,
        x0: np.ndarray,
        P0: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        alpha: float = 1.0,
        beta: float = 0.7,
        alpha_ukf: float = 1.0,
        kappa: float = 0.0,
        adaptive_R: bool = True,
        sxr_background: float = 1.0,
    ):
        self.n = len(x0)
        self.x = np.asarray(x0, dtype=float).copy()
        self.P = np.asarray(P0, dtype=float).copy()
        self.Q = np.asarray(Q, dtype=float).copy()
        self.R = np.asarray(R, dtype=float).copy()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.alpha_ukf = float(alpha_ukf)
        self.kappa = float(kappa)
        self.adaptive_R = bool(adaptive_R)
        self.sxr_background = float(sxr_background)

        # Save initial R for use as a floor during adaptive R updates
        self._R0 = np.asarray(R, dtype=float).copy()

        # UKF lambda and weights
        self.lam = self.alpha_ukf**2 * (self.n + self.kappa) - self.n
        self.Wm, self.Wc = self._compute_weights()

        # Bookkeeping for adaptive R and residual diagnostics
        self._residual_history: list[np.ndarray] = []
        self._residual_window = 20
        self._residual_ema_beta = 0.1  # slow adaptation

        # Latest innovation covariance (set in update step)
        self.S = self.R.copy()
        self.last_residual = np.zeros(2)
        self.last_mahalanobis = 0.0

    # ------------------------------------------------------------------
    # Sigma points and weights
    # ------------------------------------------------------------------
    def _compute_weights(self) -> tuple[np.ndarray, np.ndarray]:
        n = self.n
        lam = self.lam
        Wm = np.full(2 * n + 1, 1.0 / (2.0 * (n + lam)))
        Wc = np.full(2 * n + 1, 1.0 / (2.0 * (n + lam)))
        Wm[0] = lam / (n + lam)
        Wc[0] = lam / (n + lam) + (1.0 - self.alpha_ukf**2 + 2.0)  # beta_ukf=2 optimal for Gaussian
        return Wm, Wc

    def _sigma_points(self) -> np.ndarray:
        """Generate 2n+1 sigma points using eigendecomposition (numerically robust).

        Cholesky can fail when P becomes ill-conditioned during IMM mixing;
        eigendecomposition with negative-eigenvalue flooring enforces PSD.
        """
        n = self.n
        # Symmetrize for safety
        P_sym = 0.5 * (self.P + self.P.T)
        # Replace NaN/Inf with safe values (defensive; should never trigger
        # but if upstream alpha estimator or health monitor produces a NaN
        # we don't want a hard crash)
        if not np.all(np.isfinite(P_sym)):
            P_sym = np.eye(n) * 1.0
        # Clip extreme values to prevent eigendecomposition convergence issues
        P_sym = np.clip(P_sym, -1e8, 1e8)
        try:
            eigvals, eigvecs = np.linalg.eigh(P_sym)
        except np.linalg.LinAlgError:
            # Fall back to identity covariance if eigendecomposition fails
            eigvals = np.ones(n)
            eigvecs = np.eye(n)
        eigvals = np.maximum(eigvals, 1e-6)  # floor negatives
        sqrt_P = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

        sigmas = np.zeros((2 * n + 1, n))
        sigmas[0] = self.x
        scale = np.sqrt(n + self.lam)
        for i in range(n):
            col = sqrt_P[:, i] * scale
            sigmas[i + 1] = self.x + col
            sigmas[i + n + 1] = self.x - col
        return sigmas

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------
    def predict(self, dt: float) -> None:
        sigmas = self._sigma_points()
        # Propagate each sigma point through the non-linear state transition
        sigmas_pred = np.array(
            [state_transition(s, dt, self.alpha, self.beta, self.sxr_background,
                              integral_tau=300.0)
             for s in sigmas]
        )
        # Predicted mean (weighted)
        self.x = np.sum(self.Wm[:, None] * sigmas_pred, axis=0)
        # Predicted covariance
        dx = sigmas_pred - self.x[None, :]
        self.P = dx.T @ np.diag(self.Wc) @ dx + self.Q
        # Symmetrize
        self.P = 0.5 * (self.P + self.P.T)
        # Save predicted sigma points for the update step
        self._sigmas_pred = sigmas_pred

    # ------------------------------------------------------------------
    # Update (with optional adaptive R and dropout support)
    # ------------------------------------------------------------------
    def update(self, z: np.ndarray | None, dt: float) -> None:
        """Update state with measurement z, or skip update if z is None (dropout).

        If z is None (data dropout / quality flag failure), the filter stays in
        predict-only mode: state evolves via physics, covariance grows naturally,
        and no NaN is produced. This is mission-safe by design.
        """
        if z is None:
            # Predict-only: covariance grows by Q (already added in predict),
            # no measurement correction. No NaN, no special handling.
            self.last_residual = np.zeros(2)
            self.last_mahalanobis = 0.0
            return

        z = np.asarray(z, dtype=float)
        # Measurement sigma points
        z_sigmas = np.array([measurement_function(s) for s in self._sigmas_pred])
        z_pred = np.sum(self.Wm[:, None] * z_sigmas, axis=0)
        # Innovation covariance
        dz = z_sigmas - z_pred[None, :]
        S = dz.T @ np.diag(self.Wc) @ dz + self.R
        S = 0.5 * (S + S.T)
        # Cross-covariance
        dx = self._sigmas_pred - self.x[None, :]
        C = dx.T @ np.diag(self.Wc) @ dz
        # Kalman gain
        K = C @ np.linalg.inv(S)
        # Residual / innovation
        residual = z - z_pred
        # State update
        self.x = self.x + K @ residual
        # Joseph-form covariance update (numerically stable, guarantees PSD)
        I = np.eye(self.n)
        P_joseph = (I - K @ np.eye(2) @ self._H_jacobian())  # placeholder, replaced below
        # For non-linear h(x) = [x0, x1], the linearized H is constant:
        H = np.zeros((2, self.n))
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        P_joseph = (I - K @ H) @ self.P @ (I - K @ H).T + K @ self.R @ K.T
        self.P = 0.5 * (P_joseph + P_joseph.T)

        # Save diagnostics
        self.S = S
        self.last_residual = residual
        try:
            self.last_mahalanobis = float(
                np.sqrt(max(residual @ np.linalg.inv(S) @ residual, 0.0))
            )
        except np.linalg.LinAlgError:
            self.last_mahalanobis = float(np.linalg.norm(residual))

        # Adaptive R: EMA on recent residual covariance. Floor at 50% of the
        # initial R to prevent covariance collapse during quiet periods (which
        # would otherwise make normal noise fluctuations look statistically
        # anomalous and inflate the false-alarm rate).
        if self.adaptive_R:
            self._residual_history.append(residual.copy())
            if len(self._residual_history) > self._residual_window:
                self._residual_history.pop(0)
            if len(self._residual_history) >= 5:
                # EMA of residual covariance; beta=0.1 -> slow adaptation
                recent = np.array(self._residual_history)
                R_ema = recent.T @ recent / len(recent)
                R_new = (
                    (1.0 - self._residual_ema_beta) * self.R
                    + self._residual_ema_beta * R_ema
                )
                # Floor at 50% of the initial R (saved at construction)
                R_floor = 0.5 * self._R0
                self.R = np.maximum(R_new, R_floor)

    def _H_jacobian(self) -> np.ndarray:
        """Linearized measurement matrix H = d h / d x for h = [x0, x1]."""
        H = np.zeros((2, self.n))
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        return H

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def neupert_residual(self) -> float:
        """D(t) = SXR(t) - SXR_bg - alpha * integral(HXR).

        This is the innovation sequence of the Neupert constraint. In a correct
        model it should be zero-mean white noise; deviation indicates that
        hard X-ray energy deposition is NOT matching SXR thermalization -- the
        pre-flare signal. SXR_bg is subtracted because the thermal background
        is independent of HXR energy deposition.
        """
        return float(self.x[0] - self.sxr_background - self.alpha * self.x[2])

    def mahalanobis_distance(self) -> float:
        return self.last_mahalanobis
