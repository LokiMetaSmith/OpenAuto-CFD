"""
test_phyz_validate_gate.py
Fail-Closed Benchmark Gatekeeper Suite (phyz-validate standard).

Enforces strict verification gates:
  Gate 1: Finite Terminal State Enforcement (no NaNs, no unphysical infinities).
  Gate 2: Analytic & IFT Gradient Accuracy (<= 10^-6 rel error vs central FD).
  Gate 3: Continuous Hadamard BRep Shape Gradient Accuracy (<= 10^-3 rel error vs central FD).
  Gate 4: Lyapunov-Windowed Trajectory Adjoint Accuracy (<= 10^-3 rel error vs central FD).
  Gate 5: Fail-Closed Receipt Integrity (Violated receipt immediately blocks downstream dispatch).
  Gate 6: Hardware-Aware Compute Dispatch Law (N < 128 CPU break-even).
  Gate 7: MCP Server Phyz & Vcad Tools Contract.
"""

import sys
import os
import json
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from optimizer.receipt_engine import ClaimReceiptGraph, ReceiptState
from optimizer.assembly_solver import UpstreamAssemblySolver
from optimizer.brep_sensitivity import SpatialBRepSensitivityEngine
from optimizer.guardian import PhyzConservationGuardian, SolverTier
from optimizer.latent_dynamics import LatentDynamicsModel
from optimizer.trajectory_adjoint import TrajectoryAdjointEngine
from optimizer.compute_router import HardwareComputeRouter
import optimizer.mcp_server as mcp_server


def test_gate_1_finite_terminal_state():
    """Gate 1: Every simulation and surrogate rollout must produce finite, non-NaN terminal states."""
    model = LatentDynamicsModel()
    z0 = np.array([2.0, 50.0, 20.0, 1.0, 0.5, 0.0], dtype=np.float64)
    p0 = np.array([10.0, 25.0, 32.0, 0.5], dtype=np.float64)
    u_traj = np.tile(np.array([0.75, 0.55]), (30, 1))

    z_traj = model.rollout(z0, u_traj, p0, dt=0.01)

    assert np.all(np.isfinite(z_traj)), "Gate 1 Violation: Latent dynamics produced NaN or Inf states"
    assert z_traj[-1, 1] > 0.0, "Gate 1 Violation: Non-positive pressure drop"
    assert 0.0 <= z_traj[-1, 2] <= 100.0, "Gate 1 Violation: Separation efficiency out of [0, 100] bounds"


def test_gate_2_analytic_ift_gradient_accuracy():
    """Gate 2: Upstream Assembly LM solver IFT derivatives must match central FD <= 10^-6 rel error."""
    solver = UpstreamAssemblySolver()
    p_dict = {
        "tube_od_mm": 32.0,
        "tube_wall_mm": 1.5,
        "insert_length_mm": 50.0,
        "inlet_port_dia_mm": 6.35,
        "sensor_standoff_mm": 3.0,
        "oring_cs_mm": 1.78
    }

    rel_err, _, _ = solver.verify_ift_vs_finite_differences(p_dict, eps=1e-6)
    assert rel_err <= 1e-6, f"Gate 2 Violation: IFT rel error {rel_err:.2e} exceeded 1e-6"


def test_gate_3_hadamard_brep_gradient_accuracy():
    """Gate 3: Continuous Hadamard BRep boundary gradient must match central FD <= 10^-3 rel error."""
    import trimesh
    engine = SpatialBRepSensitivityEngine()

    def geom_builder(r: float) -> trimesh.Trimesh:
        return trimesh.creation.cylinder(radius=r, height=20.0, sections=48)

    rel_err, _, _ = engine.verify_shape_gradient_vs_fd(
        geom_builder=geom_builder,
        p0=8.0,
        functional_fn=lambda m: float(m.volume),
        adjoint_density_fn=lambda m: np.ones(len(m.faces), dtype=np.float64),
        delta_p=1e-3
    )
    assert rel_err <= 1e-3, f"Gate 3 Violation: Hadamard BRep rel error {rel_err:.2e} exceeded 1e-3"


def test_gate_4_trajectory_adjoint_accuracy():
    """Gate 4: Reverse-mode trajectory adjoint must match central FD <= 10^-3 rel error."""
    engine = TrajectoryAdjointEngine(lyapunov_gamma=0.0)
    z0 = np.array([5.0, 100.0, 50.0, 2.0, 1.0, 0.1], dtype=np.float64)
    p0 = np.array([10.0, 25.0, 32.0, 0.5], dtype=np.float64)
    u_traj = np.tile(np.array([0.7, 0.5]), (20, 1))

    rel_err, _, _ = engine.verify_adjoint_vs_finite_differences(z0, u_traj, p0, dt=0.005, eps=1e-5)
    assert rel_err <= 1e-3, f"Gate 4 Violation: Trajectory adjoint rel error {rel_err:.2e} exceeded 1e-3"


def test_gate_5_fail_closed_receipt_blocks_downstream():
    """Gate 5: Instability can never masquerade as speed; Violated receipt must hard-block downstream steps."""
    # Violate center clearance: profile_r (12.0) > path_r (10.0)
    bad_params = {
        "insert_length_mm": 50.0,
        "twist_deg": 720.0,
        "helix_path_radius_mm": 10.0,
        "helix_profile_radius_mm": 12.0,
        "helix_void_profile_radius_mm": 2.5,
        "tube_od_mm": 32.0,
        "tube_wall_mm": 1.5,
        "blade_chamfer_mm": 0.5,
        "barrel_length_mm": 60.0
    }

    graph = ClaimReceiptGraph()
    graph.resolve(bad_params)

    assert graph.is_valid() is False, "Gate 5 Violation: Invalid parameter set evaluated as valid"
    assert len(graph.get_violations()) > 0

    # Test MCP tool fail-closed behavior
    res = mcp_server.evaluate_coupled_design(
        helix_path_radius_mm=10.0,
        helix_profile_radius_mm=12.0  # Violation
    )
    assert res["status"] == "rejected_fail_closed", "Gate 5 Violation: evaluate_coupled_design did not reject Violated design"


def test_gate_6_hardware_compute_dispatch_threshold():
    """Gate 6: Enforce CPU dispatch for N < 128 (< 10 ms) and GPU batching for N >= 128."""
    router = HardwareComputeRouter()

    # Interactive slider scrub (N = 1)
    results_single, meta_single = router.dispatch([np.array([10.0, 25.0, 32.0, 0.5])], lambda p: float(np.sum(p)))
    assert meta_single["device"] == "CPU"
    assert meta_single["wall_clock_ms"] < 10.0, "Gate 6 Violation: Interactive scrub took >= 10 ms"

    # Batch optimization (N = 150)
    results_batch, meta_batch = router.dispatch([np.array([10.0, 25.0, 32.0, 0.5]) for _ in range(150)], lambda p: float(np.sum(p)))
    assert meta_batch["device"] == "GPU_BATCH"


def test_gate_7_mcp_tools_contract():
    """Gate 7: Verify all 4 new MCP tools execute and return structured contracts."""
    # 1. get_receipt_graph_status
    status_res = mcp_server.get_receipt_graph_status()
    assert "is_valid" in status_res
    assert "receipts" in status_res
    assert status_res["is_valid"] is True

    # 2. get_guardian_telemetry
    guardian_res = mcp_server.get_guardian_telemetry(current_energy=1025.0, initial_energy=1000.0)
    assert guardian_res["active_tier"] == "Tier 1 (Coarse Physics)"
    assert guardian_res["fallback_triggered"] is True

    # 3. run_trajectory_adjoint_sensitivity
    sens_res = mcp_server.run_trajectory_adjoint_sensitivity()
    assert sens_res["status"] == "success"
    assert "ranking_by_leverage" in sens_res
    assert len(sens_res["ranking_by_leverage"]) == 4

    # 4. evaluate_coupled_design
    eval_res = mcp_server.evaluate_coupled_design()
    assert eval_res["status"] == "success"
    assert "predicted_metrics" in eval_res
    assert eval_res["predicted_metrics"]["separation_efficiency_pct"] > 0.0
