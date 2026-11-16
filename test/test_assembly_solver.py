"""
test_assembly_solver.py
Unit tests verifying the Levenberg-Marquardt assembly layout solver and
analytic Implicit Function Theorem (IFT) derivatives against central finite differences.
"""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from optimizer.assembly_solver import UpstreamAssemblySolver


def test_assembly_solver_convergence():
    """Verify LM solver converges to F(x, p) = 0 for standard assembly parameters."""
    solver = UpstreamAssemblySolver()
    p_dict = {
        "tube_od_mm": 32.0,
        "tube_wall_mm": 1.5,
        "insert_length_mm": 50.0,
        "inlet_port_dia_mm": 6.35,
        "sensor_standoff_mm": 3.0,
        "oring_cs_mm": 1.78
    }

    solved_dict, x_sol = solver.solve(p_dict)

    # Check key assembly mates
    # r_inner = 16.0 - 1.5 = 14.5; r_port = 3.175; y_inlet = 14.5 - 3.175 = 11.325
    assert abs(solved_dict["x_inlet"] - 0.0) < 1e-6
    assert abs(solved_dict["y_inlet"] - 11.325) < 1e-6
    assert abs(solved_dict["z_inlet"] - 45.0) < 1e-6
    assert abs(solved_dict["r_sensor_tap"] - 19.0) < 1e-6
    assert abs(solved_dict["gland_depth_mm"] - (0.75 * 1.78)) < 1e-6

    # Verify residual norm is zero within tolerance
    p0 = np.array([p_dict[k] for k in solver.param_names], dtype=np.float64)
    res = solver.residual_system(x_sol, p0)
    assert np.linalg.norm(res) < 1e-8


def test_assembly_solver_ift_gradient_gate():
    """
    Verify that analytic IFT Jacobian matches central finite differences
    within <= 10^-6 relative error across all assembly parameters.
    """
    solver = UpstreamAssemblySolver()
    p_dict = {
        "tube_od_mm": 32.0,
        "tube_wall_mm": 1.5,
        "insert_length_mm": 50.0,
        "inlet_port_dia_mm": 6.35,
        "sensor_standoff_mm": 3.0,
        "oring_cs_mm": 1.78
    }

    rel_err, J_ift, J_fd = solver.verify_ift_vs_finite_differences(p_dict, eps=1e-6)

    print(f"IFT vs Central FD relative error: {rel_err:.2e}")
    # Gate check: relative error must be <= 10^-6
    assert rel_err <= 1e-6, f"IFT relative error {rel_err:.2e} exceeded gate threshold 1e-6"
