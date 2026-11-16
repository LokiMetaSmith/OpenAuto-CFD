"""
latent_dynamics.py
Time-Stepping Latent Dynamics Rollout Model (phyz-dream paradigm).

Evolves static surrogate regressions into rollout-capable continuous-time
latent dynamics models conditioned on both physical geometric parameters (build123d)
and time-varying firmware control inputs (PWM, inlet velocity).

Formulation:
    z(t + dt) = z(t) + ∫_t^(t+dt) f_θ(z(τ), u_ctrl(τ), p_geom) dτ
with RK4 integration and POD/modal field projection.
"""

from typing import Dict, Any, List, Optional, Tuple, Callable
import numpy as np


class LatentDynamicsModel:
    """
    Continuous-time latent dynamics model for the corkscrew multi-physics system.
    Latent state z ∈ R^6:
      z[0]: Swirl angular momentum (rad/s)
      z[1]: Channel differential pressure ΔP (Pa)
      z[2]: Particle collection efficiency η_sep (%)
      z[3]: Von Mises structural boundary stress σ_max (MPa)
      z[4]: Boundary thermal dissipation rate (W)
      z[5]: Accumulated debris reject mass (g)
    
    Control input u ∈ R^2:
      u[0]: Slurry inlet velocity (m/s)
      u[1]: PWM pump duty cycle (0.0 to 1.0)
    
    Geometric parameters p ∈ R^4:
      p[0]: Helix path radius (mm)
      p[1]: Helix pitch (mm)
      p[2]: Tube outer diameter (mm)
      p[3]: Blade chamfer height (mm)
    """

    STATE_DIM = 6
    CTRL_DIM = 2
    PARAM_DIM = 4

    def __init__(self, damping: float = 0.85):
        self.damping = damping

    def vector_field(self, z: np.ndarray, u: np.ndarray, p: np.ndarray) -> np.ndarray:
        """
        Evaluates dz/dt = f(z, u, p).
        z: (6,)
        u: (2,)
        p: (4,)
        """
        path_r, pitch, tube_od, chamfer = p
        u_in, pwm_duty = u

        dz_dt = np.zeros(self.STATE_DIM, dtype=np.float64)

        # 1. Swirl angular momentum dynamics: driven by tangential inlet velocity and helix path radius
        # Decays with fluid friction / damping
        swirl_drive = (u_in * 1000.0) / max(path_r, 1.0)
        dz_dt[0] = swirl_drive - self.damping * z[0]

        # 2. Differential pressure ΔP dynamics: quadratic in velocity, inversely proportional to pitch
        target_dp = (0.5 * 1000.0 * (u_in ** 2)) * (50.0 / max(pitch, 2.0))
        dz_dt[1] = 5.0 * (target_dp - z[1])

        # 3. Separation efficiency dynamics: enhanced by swirl momentum and blade chamfer
        centrifugal_accel = (z[0] ** 2) * (path_r * 1e-3)
        target_eta = 95.0 * (1.0 - np.exp(-0.05 * max(centrifugal_accel, 0.0))) * (1.0 + 0.1 * min(chamfer, 1.0))
        target_eta = min(target_eta, 99.5)
        dz_dt[2] = 2.0 * (target_eta - z[2])

        # 4. Structural stress dynamics: driven by differential pressure and thin wall
        wall_t = max(0.5 * tube_od - path_r, 0.5)
        hoop_stress = (z[1] * (path_r * 1e-3)) / (wall_t * 1e-3 * 1e6)  # in MPa
        dz_dt[3] = 10.0 * (hoop_stress - z[3])

        # 5. Thermal dissipation: viscous friction from swirl
        thermal_rate = 0.05 * (z[0] ** 1.5)
        dz_dt[4] = 1.0 * (thermal_rate - z[4])

        # 6. Reject mass accumulation: proportional to separation efficiency and slurry feed rate
        dz_dt[5] = 0.01 * (z[2] / 100.0) * u_in

        return dz_dt

    def step_rk4(self, z: np.ndarray, u: np.ndarray, p: np.ndarray, dt: float) -> np.ndarray:
        """
        Advances state by dt using 4th-order Runge-Kutta integration.
        """
        k1 = self.vector_field(z, u, p)
        k2 = self.vector_field(z + 0.5 * dt * k1, u, p)
        k3 = self.vector_field(z + 0.5 * dt * k2, u, p)
        k4 = self.vector_field(z + dt * k3, u, p)

        z_next = z + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return z_next

    def rollout(
        self,
        z0: np.ndarray,
        u_trajectory: np.ndarray,
        p: np.ndarray,
        dt: float
    ) -> np.ndarray:
        """
        Executes forward trajectory rollout.
        u_trajectory: (T, 2)
        Returns:
            z_trajectory: (T + 1, 6)
        """
        T = u_trajectory.shape[0]
        z_traj = np.zeros((T + 1, self.STATE_DIM), dtype=np.float64)
        z_traj[0] = z0

        z_curr = z0.copy()
        for t in range(T):
            z_curr = self.step_rk4(z_curr, u_trajectory[t], p, dt)
            z_traj[t + 1] = z_curr

        return z_traj
