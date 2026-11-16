"""
test_trajectory_adjoint.py
Unit tests verifying the Lyapunov-Windowed Trajectory Adjoint Engine against finite differences.
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from optimizer.trajectory_adjoint import TrajectoryAdjointEngine
from optimizer.latent_dynamics import LatentDynamicsModel


def test_trajectory_adjoint_vs_finite_differences_gate():
    """
    Verify that reverse-mode trajectory adjoint matches central finite differences
    within <= 10^-3 relative error across all geometric parameters.
    """
    model = LatentDynamicsModel()
    engine = TrajectoryAdjointEngine(model=model, lyapunov_gamma=0.0)

    z0 = np.array([5.0, 100.0, 50.0, 2.0, 1.0, 0.1], dtype=np.float64)
    p0 = np.array([10.0, 25.0, 32.0, 0.5], dtype=np.float64)

    # 20 time steps
    T = 20
    dt = 0.005
    u_traj = np.tile(np.array([0.7, 0.5]), (T, 1))

    rel_err, dJ_adj, dJ_fd = engine.verify_adjoint_vs_finite_differences(
        z0=z0,
        u_traj=u_traj,
        p0=p0,
        dt=dt,
        eps=1e-5
    )

    print(f"Trajectory Adjoint: {dJ_adj}")
    print(f"Central FD:         {dJ_fd}")
    print(f"Relative Error:     {rel_err:.2e}")

    # Phase Gate criterion: rel_err <= 10^-3
    assert rel_err <= 1e-3, f"Trajectory adjoint rel error {rel_err:.2e} exceeded gate threshold 1e-3"
