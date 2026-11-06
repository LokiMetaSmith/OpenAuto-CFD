"""
verify_benchmark_suite.py

Verification test suite for Automated Multi-Algorithm Benchmark & Pareto Front Analysis.
Tests:
  1. Multi-Objective Physical Ground Truth Oracle & Geometric Invalidation
  2. Non-Dominated Pareto Front Identification (Dominance Logic)
  3. 2D Hypervolume Indicator Properties
  4. Multi-Algorithm Benchmark Execution (Random vs L-BFGS-B vs PINN vs CAD Agent)
"""

import sys
import os
import json
import time

sys.path.insert(0, os.path.abspath("optimizer"))
sys.path.insert(0, os.path.abspath("viewer"))

from benchmark_orchestrator import (
    evaluate_filter_physics,
    compute_non_dominated_front,
    compute_2d_hypervolume,
    BenchmarkOrchestrator
)


def test_physical_oracle():
    print("\n--- Test 1: Ground-Truth Physical Oracle & Geometric Bounds ---")
    # Valid design
    valid_p = {
        "number_of_complete_revolutions": 2.2,
        "helix_path_radius_mm": 2.1,
        "helix_profile_radius_mm": 1.3,
        "blade_chamfer_mm": 0.5
    }
    res_valid = evaluate_filter_physics(valid_p)
    assert res_valid["valid"] is True
    assert 1500.0 <= res_valid["delta_p"] <= 5000.0, f"Unexpected delta_p: {res_valid['delta_p']}"
    assert 80.0 <= res_valid["separation_efficiency"] <= 99.0, f"Unexpected eff: {res_valid['separation_efficiency']}"
    print(f"  Valid design: delta_p={res_valid['delta_p']:.1f} Pa, eff={res_valid['separation_efficiency']:.2f}%, FOM={res_valid['figure_of_merit']:.2f}")

    # Invalid geometry (profile >= path radius)
    invalid_p = {
        "number_of_complete_revolutions": 2.0,
        "helix_path_radius_mm": 1.5,
        "helix_profile_radius_mm": 1.8, # Invalid!
        "blade_chamfer_mm": 0.5
    }
    res_invalid = evaluate_filter_physics(invalid_p)
    assert res_invalid["valid"] is False
    assert res_invalid["delta_p"] == 9999.0
    print("  Invalid design correctly penalized with valid=False.")
    print("[PASS] Test 1: Physical Oracle & Geometric Bounds verified.")


def test_pareto_front_identification():
    print("\n--- Test 2: Non-Dominated Pareto Front Identification ---")
    candidates = [
        # (eff: maximize, dp: minimize)
        {"id": "A", "separation_efficiency": 95.0, "delta_p": 2500.0, "valid": True}, # Pareto
        {"id": "B", "separation_efficiency": 96.0, "delta_p": 2800.0, "valid": True}, # Pareto (higher eff, higher dp)
        {"id": "C", "separation_efficiency": 94.0, "delta_p": 2200.0, "valid": True}, # Pareto (lower dp, lower eff)
        {"id": "D", "separation_efficiency": 93.0, "delta_p": 3000.0, "valid": True}, # Dominated by A, B, and C
        {"id": "E", "separation_efficiency": 97.0, "delta_p": 4500.0, "valid": False} # Invalid, excluded
    ]

    front = compute_non_dominated_front(candidates)
    front_ids = {c["id"] for c in front}
    print(f"  Identified Pareto Front: {front_ids}")

    assert "A" in front_ids, "Point A should be on Pareto front"
    assert "B" in front_ids, "Point B should be on Pareto front"
    assert "C" in front_ids, "Point C should be on Pareto front"
    assert "D" not in front_ids, "Point D is dominated and must NOT be on Pareto front"
    assert "E" not in front_ids, "Invalid point E must NOT be on Pareto front"
    assert len(front) == 3, f"Expected 3 Pareto points, got {len(front)}"
    print("[PASS] Test 2: Non-Dominated Pareto Front Identification verified.")


def test_hypervolume_metric():
    print("\n--- Test 3: 2D Hypervolume Indicator Properties ---")
    # Front 1: High performance
    front_high = [
        {"separation_efficiency": 96.0, "delta_p": 2000.0},
        {"separation_efficiency": 94.0, "delta_p": 1600.0}
    ]
    # Front 2: Inferior performance
    front_low = [
        {"separation_efficiency": 82.0, "delta_p": 4500.0}
    ]

    hv_high = compute_2d_hypervolume(front_high)
    hv_low = compute_2d_hypervolume(front_low)

    print(f"  Hypervolume (High Front): {hv_high:.4f}")
    print(f"  Hypervolume (Low Front):  {hv_low:.4f}")

    assert 0.0 <= hv_low < hv_high <= 1.0, f"Expected 0 <= {hv_low} < {hv_high} <= 1.0"
    print("[PASS] Test 3: Hypervolume metric monotonicity verified.")


def test_benchmark_orchestration():
    print("\n--- Test 4: Multi-Algorithm Benchmark Orchestration (10 iters) ---")
    orchestrator = BenchmarkOrchestrator()
    report = orchestrator.run_full_benchmark(n_iterations=10)

    summary = report["summary"]
    assert "Random Search" in summary
    assert "L-BFGS-B" in summary
    assert "PINN Surrogate" in summary
    assert "Autonomous CAD Agent" in summary

    print("\n" + orchestrator.format_markdown_table(report) + "\n")

    # Verify best FOM and pareto points
    rand_fom = summary["Random Search"]["best_fom"]
    pinn_fom = summary["PINN Surrogate"]["best_fom"]
    cad_fom = summary["Autonomous CAD Agent"]["best_fom"]
    print(f"  Random Search Best FOM:    {rand_fom:.2f}")
    print(f"  PINN Surrogate Best FOM:   {pinn_fom:.2f}")
    print(f"  CAD Agent Best FOM:        {cad_fom:.2f}")

    assert pinn_fom >= rand_fom or cad_fom >= rand_fom, "Physics-informed/agent optimization should match or beat random search"
    assert report["global_pareto_front_size"] >= 2, "Should identify multiple Pareto-optimal trade-offs"
    print("[PASS] Test 4: Multi-algorithm benchmark successfully executed and evaluated.")


if __name__ == "__main__":
    print("================================================================")
    print("      RUNNING PHASE 3: MULTI-ALGORITHM BENCHMARK SUITE          ")
    print("================================================================")
    test_physical_oracle()
    test_pareto_front_identification()
    test_hypervolume_metric()
    test_benchmark_orchestration()
    print("\n>>> ALL PHASE 3 BENCHMARK TESTS PASSED SUCCESSFULLY! <<<")
