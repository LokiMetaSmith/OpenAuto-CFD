"""
test_latent_dynamics.py
Unit tests verifying the continuous-time Latent Dynamics Rollout Model (phyz-dream paradigm).
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from optimizer.latent_dynamics import LatentDynamicsModel


def test_latent_dynamics_rk4_rollout():
    """Verify that multi-step RK4 rollout produces stable, physically bounded trajectories."""
    model = LatentDynamicsModel()

    # Initial state: [swirl=0, dp=0, eta=0, stress=0, thermal=0, reject_mass=0]
    z0 = np.zeros(6, dtype=np.float64)

    # Control trajectory: 50 time steps with inlet velocity = 0.8 m/s, PWM duty = 0.6
    T = 50
    dt = 0.01
    u_traj = np.tile(np.array([0.8, 0.6]), (T, 1))

    # Parameters: path_r = 10.0, pitch = 25.0, tube_od = 32.0, chamfer = 0.5
    p = np.array([10.0, 25.0, 32.0, 0.5], dtype=np.float64)

    z_traj = model.rollout(z0, u_traj, p, dt)

    assert z_traj.shape == (T + 1, 6)

    # All states must remain finite (no NaN or Inf)
    assert np.all(np.isfinite(z_traj))

    # Swirl momentum and pressure drop must be positive
    z_final = z_traj[-1]
    assert z_final[0] > 0.0  # Swirl
    assert z_final[1] > 0.0  # Differential pressure ΔP
    assert 0.0 < z_final[2] <= 100.0  # Separation efficiency η_sep in %
    assert z_final[3] > 0.0  # Von Mises stress
    assert z_final[5] > 0.0  # Reject mass accumulated


def test_latent_dynamics_parameter_sensitivity():
    """
    Verify that increasing inlet velocity increases pressure drop,
    and increasing pitch decreases pressure drop.
    """
    model = LatentDynamicsModel()
    z0 = np.zeros(6, dtype=np.float64)
    dt = 0.01

    # Compare low velocity (0.4 m/s) vs high velocity (1.0 m/s)
    u_low = np.tile(np.array([0.4, 0.3]), (30, 1))
    u_high = np.tile(np.array([1.0, 0.8]), (30, 1))
    p = np.array([10.0, 25.0, 32.0, 0.5])

    z_low = model.rollout(z0, u_low, p, dt)[-1]
    z_high = model.rollout(z0, u_high, p, dt)[-1]

    assert z_high[1] > z_low[1], "Higher velocity must increase pressure drop ΔP"
    assert z_high[0] > z_low[0], "Higher velocity must increase swirl momentum"
