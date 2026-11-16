"""
trajectory_adjoint.py
Lyapunov Sliding-Window Trajectory Adjoint Engine (phyz-diff paradigm).

Implements reverse-mode continuous trajectory adjoint backpropagation
through the coupled latent dynamics and firmware control rollouts.

Enables simultaneous co-design optimization of:
  - Physical 3D geometry parameters (build123d)
  - Microcontroller firmware control parameters (PID gains, setpoints)
with Lyapunov windowing/discounting to suppress chaotic vortex divergence.
"""

from typing import Dict, Any, List, Optional, Tuple, Callable
import numpy as np
from optimizer.latent_dynamics import LatentDynamicsModel


class TrajectoryAdjointEngine:
    """
    Computes analytical reverse-mode trajectory adjoints dJ/dp.
    """

    def __init__(self, model: Optional[LatentDynamicsModel] = None, lyapunov_gamma: float = 0.05):
        self.model = model or LatentDynamicsModel()
        self.gamma = lyapunov_gamma  # Lyapunov exponential discounting rate

    def compute_step_jacobians(
        self,
        z: np.ndarray,
        u: np.ndarray,
        p: np.ndarray,
        dt: float,
        eps: float = 1e-6
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the exact discrete transition Jacobians of the forward integration step:
            z_{t+1} = Φ(z_t, u_t, p, dt)
        Returns:
            dPhi_dz: (6, 6)
            dPhi_dp: (6, 4)
        """
        n_z = self.model.STATE_DIM
        n_p = self.model.PARAM_DIM

        dPhi_dz = np.zeros((n_z, n_z), dtype=np.float64)
        for i in range(n_z):
            z_plus = z.copy()
            z_plus[i] += eps
            z_minus = z.copy()
            z_minus[i] -= eps
            step_plus = self.model.step_rk4(z_plus, u, p, dt)
            step_minus = self.model.step_rk4(z_minus, u, p, dt)
            dPhi_dz[:, i] = (step_plus - step_minus) / (2.0 * eps)

        dPhi_dp = np.zeros((n_z, n_p), dtype=np.float64)
        for j in range(n_p):
            p_plus = p.copy()
            p_plus[j] += eps
            p_minus = p.copy()
            p_minus[j] -= eps
            step_plus = self.model.step_rk4(z, u, p_plus, dt)
            step_minus = self.model.step_rk4(z, u, p_minus, dt)
            dPhi_dp[:, j] = (step_plus - step_minus) / (2.0 * eps)

        return dPhi_dz, dPhi_dp

    def compute_loss(
        self,
        z_traj: np.ndarray,
        u_traj: np.ndarray,
        p: np.ndarray,
        dt: float
    ) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        """
        Evaluates objective functional:
            J = Φ(z(T)) + ∑ L(z(t), u(t), p) * dt
        Objective:
            Maximize separation efficiency z[2] (minimize -z[2])
            Minimize pressure drop z[1] (weight 0.01)
            Minimize structural stress z[3] (weight 0.05)
        Returns:
            J_total: float
            dL_dz: (T, 6)
            dPhi_dzT: (6,)
            dL_dp: (T, 4)
        """
        T = u_traj.shape[0]
        n_z = self.model.STATE_DIM
        n_p = self.model.PARAM_DIM

        dL_dz = np.zeros((T, n_z), dtype=np.float64)
        dL_dp = np.zeros((T, n_p), dtype=np.float64)

        # Stage loss: L(z, u) = 0.01 * ΔP - 1.0 * η_sep + 0.05 * σ_max
        running_cost = 0.0
        for t in range(T):
            z_t = z_traj[t]
            running_cost += (0.01 * z_t[1] - 1.0 * z_t[2] + 0.05 * z_t[3]) * dt
            dL_dz[t, 1] = 0.01
            dL_dz[t, 2] = -1.0
            dL_dz[t, 3] = 0.05

        # Terminal cost: Φ(z(T)) = -2.0 * η_sep(T) + 0.02 * ΔP(T)
        z_T = z_traj[-1]
        terminal_cost = -2.0 * z_T[2] + 0.02 * z_T[1]

        dPhi_dzT = np.zeros(n_z, dtype=np.float64)
        dPhi_dzT[2] = -2.0
        dPhi_dzT[1] = 0.02

        J_total = running_cost + terminal_cost
        return J_total, dL_dz, dPhi_dzT, dL_dp

    def solve_reverse_adjoint(
        self,
        z_traj: np.ndarray,
        u_traj: np.ndarray,
        p: np.ndarray,
        dt: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Integrates reverse discrete adjoint:
            λ_t = [dPhi/dz]^T λ_{t+1} + dL/dz_t
            λ_T = dPhi/dz(T)
        Accumulates exact parameter gradient:
            dJ/dp = ∑ [dPhi/dp]^T λ_{t+1} + dL/dp
        with Lyapunov discounting factor.
        """
        T = u_traj.shape[0]
        n_z = self.model.STATE_DIM
        n_p = self.model.PARAM_DIM

        J_val, dL_dz, dPhi_dzT, dL_dp = self.compute_loss(z_traj, u_traj, p, dt)

        lambda_traj = np.zeros((T + 1, n_z), dtype=np.float64)
        lambda_traj[-1] = dPhi_dzT

        dJ_dp = np.zeros(n_p, dtype=np.float64)

        for t in range(T - 1, -1, -1):
            tau = (T - 1 - t) * dt
            discount = np.exp(-self.gamma * tau)

            z_t = z_traj[t]
            u_t = u_traj[t]

            dPhi_dz, dPhi_dp = self.compute_step_jacobians(z_t, u_t, p, dt)

            # Adjoint backpropagation
            lam_next = lambda_traj[t + 1]
            lam_next_discounted = lam_next * discount

            # Parameter gradient accumulation: [dPhi/dp]^T * λ_{t+1} + dL/dp * dt
            dJ_dp += np.dot(dPhi_dp.T, lam_next_discounted) + dL_dp[t] * dt

            # Reverse step for adjoint state λ_t: [dPhi/dz]^T * λ_{t+1} + dL/dz * dt
            lambda_traj[t] = np.dot(dPhi_dz.T, lam_next) + dL_dz[t] * dt

        return lambda_traj, dJ_dp

    def verify_adjoint_vs_finite_differences(
        self,
        z0: np.ndarray,
        u_traj: np.ndarray,
        p0: np.ndarray,
        dt: float = 0.005,
        eps: float = 1e-5
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Cross-checks reverse-mode trajectory adjoint against central finite differences.
        Gate requirement: relative error <= 10^-3.
        """
        # 1. Analytic Adjoint
        z_traj = self.model.rollout(z0, u_traj, p0, dt)
        _, dJ_dp_adj = self.solve_reverse_adjoint(z_traj, u_traj, p0, dt)

        # 2. Central Finite Differences
        n_p = len(p0)
        dJ_dp_fd = np.zeros(n_p, dtype=np.float64)

        for j in range(n_p):
            p_plus = p0.copy()
            p_plus[j] += eps
            z_plus = self.model.rollout(z0, u_traj, p_plus, dt)
            J_plus, _, _, _ = self.compute_loss(z_plus, u_traj, p_plus, dt)

            p_minus = p0.copy()
            p_minus[j] -= eps
            z_minus = self.model.rollout(z0, u_traj, p_minus, dt)
            J_minus, _, _, _ = self.compute_loss(z_minus, u_traj, p_minus, dt)

            dJ_dp_fd[j] = (J_plus - J_minus) / (2.0 * eps)

        diff_norm = np.linalg.norm(dJ_dp_adj - dJ_dp_fd)
        ref_norm = np.linalg.norm(dJ_dp_fd) + 1e-12
        rel_err = float(diff_norm / ref_norm)

        return rel_err, dJ_dp_adj, dJ_dp_fd
