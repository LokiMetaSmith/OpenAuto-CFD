"""
verify_multifidelity.py

Verification suite for Phase 2: Multi-Fidelity Mesh Pyramid (Co-Kriging & Two-Stage Screening).
Tests:
  1. MultiFidelitySurrogate cross-fidelity scaling rho and discrepancy delta(x) calibration.
  2. MultiFidelityPhysicsDriver resolution control (4.8mm coarse vs 1.5mm fine).
  3. Two-Stage Candidate Screening prunes unviable geometries while promoting champions.
  4. End-to-end active multi-fidelity optimization loop with compute speedup tracking.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath("optimizer"))

from surrogate_multifidelity import MultiFidelitySurrogate
from multifidelity_driver import MultiFidelityPhysicsDriver
from model_fusion_multifidelity import MultiFidelityModelFusionOptimizer


def test_multifidelity_surrogate_calibration():
    print("\n--- Test 1: Multi-Fidelity Surrogate Calibration (Co-Kriging) ---")
    param_names = ["helix_path_radius_mm", "number_of_complete_revolutions"]
    surrogate = MultiFidelitySurrogate(domain="cfd", param_names=param_names)

    # 1. Add 6 coarse simulation checkpoints (representing fast 15s runs)
    for r in [1.5, 2.5, 3.5]:
        for n in [1.0, 3.0]:
            p = {"helix_path_radius_mm": r, "number_of_complete_revolutions": n}
            # Coarse pressure drop has a downward bias of ~20%
            p_coarse = 1200.0 + 400.0 * n + 80.0 * r
            eff_coarse = 82.0 + 3.0 * n - 1.0 * r
            surrogate.add_low_fidelity_sample(p, {"delta_p": p_coarse, "separation_efficiency": eff_coarse})

    assert len(surrogate.low_fi_history) == 6
    print(f"  Added 6 coarse samples. Low-fi surrogate fitted: {surrogate.low_fi_surrogate.is_fitted}")

    # 2. Add only 2 sparse fine simulation checkpoints (representing expensive 15m runs)
    # Ground truth fine values: fine = 1.25 * coarse + 50 Pa
    for r, n in [(1.5, 1.0), (3.5, 3.0)]:
        p = {"helix_path_radius_mm": r, "number_of_complete_revolutions": n}
        p_coarse = 1200.0 + 400.0 * n + 80.0 * r
        eff_coarse = 82.0 + 3.0 * n - 1.0 * r
        p_fine = p_coarse * 1.25 + 50.0
        eff_fine = eff_coarse * 1.05 + 1.0
        surrogate.add_high_fidelity_sample(p, {"delta_p": p_fine, "separation_efficiency": eff_fine})

    assert len(surrogate.high_fi_history) == 2
    assert "delta_p" in surrogate.rho
    rho_dp = surrogate.rho["delta_p"]
    print(f"  Calibrated Cross-Fidelity Scaling rho for delta_p: {rho_dp:.3f} (True ratio ~ 1.25)")
    assert 1.20 <= rho_dp <= 1.30, f"rho={rho_dp} should be near 1.25"

    # 3. Test prediction at intermediate point (never evaluated with fine mesh)
    query_p = {"helix_path_radius_mm": 2.5, "number_of_complete_revolutions": 2.0}
    pred_H, unc = surrogate.predict_metrics(query_p)

    # Calculate expected coarse and fine values
    expected_coarse_dp = 1200.0 + 400.0 * 2.0 + 80.0 * 2.5  # 2200.0
    expected_fine_dp = expected_coarse_dp * 1.25 + 50.0      # 2800.0

    print(f"  Query at intermediate point: {query_p}")
    print(f"  Coarse value: {expected_coarse_dp:.1f} Pa")
    print(f"  Multi-Fidelity Fused Prediction: {pred_H['delta_p']:.1f} Pa (Target fine: {expected_fine_dp:.1f} Pa)")
    rel_err = abs(pred_H["delta_p"] - expected_fine_dp) / expected_fine_dp
    print(f"  Multi-Fidelity relative error: {rel_err*100:.2f}%")
    assert rel_err < 0.05, f"Multi-fidelity prediction error {rel_err} exceeds 5%"

    # 4. Serialization
    db_test = "artifacts/test_mf_save.json"
    surrogate.save(db_test)
    loaded = MultiFidelitySurrogate.load(db_test)
    pred_loaded, _ = loaded.predict_metrics(query_p)
    assert np.isclose(pred_H["delta_p"], pred_loaded["delta_p"], atol=1e-2)
    print("  [PASS] Multi-Fidelity surrogate calibration and serialization verified.")


def test_multifidelity_driver_execution():
    print("\n--- Test 2: Multi-Fidelity Driver Mesh Resolution Control ---")
    driver = MultiFidelityPhysicsDriver(base_driver=None, domain="cfd", verbose=False)

    coarse_settings = driver.get_mesh_settings("coarse")
    fine_settings = driver.get_mesh_settings("fine")

    assert coarse_settings["target_cell_size"] > fine_settings["target_cell_size"]
    assert coarse_settings["relative_cost"] < 0.1
    print(f"  Coarse target_cell_size: {coarse_settings['target_cell_size']} mm (Rel Cost: {coarse_settings['relative_cost']*100:.1f}%)")
    print(f"  Fine target_cell_size:   {fine_settings['target_cell_size']} mm (Rel Cost: {fine_settings['relative_cost']*100:.1f}%)")

    p = {"number_of_complete_revolutions": 2.0, "helix_path_radius_mm": 1.8}
    res_coarse = driver.execute(p, fidelity="coarse", mock_run=True)
    res_fine = driver.execute(p, fidelity="fine", mock_run=True)

    assert res_coarse["fidelity"] == "coarse"
    assert res_fine["fidelity"] == "fine"
    assert res_fine["metrics"]["delta_p"] > res_coarse["metrics"]["delta_p"]
    print(f"  Coarse delta_p: {res_coarse['metrics']['delta_p']:.1f} Pa")
    print(f"  Fine delta_p:   {res_fine['metrics']['delta_p']:.1f} Pa (Captures boundary layer)")
    print("  [PASS] Multi-Fidelity Driver execution verified.")


def test_two_stage_screening_pruning():
    print("\n--- Test 3: Two-Stage Active Screening Filter ---")
    param_defs = {
        "number_of_complete_revolutions": {"min": 1.0, "max": 4.0, "default": 2.0},
        "helix_path_radius_mm": {"min": 1.5, "max": 5.0, "default": 1.8}
    }

    db_screen = "artifacts/test_screening.json"
    for f in [db_screen, db_screen.replace(".json", "_multifidelity_meta.json"), db_screen.replace(".json", "_low_fi.json"), db_screen.replace(".json", "_high_fi.json")]:
        if os.path.exists(f): os.remove(f)

    opt = MultiFidelityModelFusionOptimizer(
        physics_driver=None,
        parameter_defs=param_defs,
        domain="cfd",
        surrogate_db_path=db_screen,
        screening_eff_threshold=89.0,
        screening_dp_threshold=4500.0,
        verbose=True
    )

    # 1. Good candidate: High revolutions (n=3.5), small radius (r=1.5) -> High efficiency, moderate pressure
    good_p = {"number_of_complete_revolutions": 3.5, "helix_path_radius_mm": 1.5}
    rec_good = opt.step(candidate_params=good_p, mock_run=True)
    assert rec_good["status"] == "PROMOTED_TO_FINE"
    print(f"  Good design status: {rec_good['status']} (Promoted to fine simulation)")

    # 2. Poor candidate: Low revolutions (n=1.0), large radius (r=4.5) -> Poor separation efficiency (~82%)
    bad_p = {"number_of_complete_revolutions": 1.0, "helix_path_radius_mm": 4.5}
    rec_bad = opt.step(candidate_params=bad_p, mock_run=True)
    assert rec_bad["status"] == "PRUNED_COARSE"
    assert "delta_p" not in rec_bad["fine_metrics"]
    print(f"  Poor design status: {rec_bad['status']} (Pruned after coarse run, saved compute!)")

    stats = opt.surrogate.get_fidelity_stats()
    print(f"  Diagnostics: {stats['low_fidelity_samples']} Coarse runs, {stats['high_fidelity_samples']} Fine runs")
    assert stats["speedup_factor"] > 1.0
    print(f"  Computational speedup: {stats['speedup_factor']}x")
    print("  [PASS] Two-Stage Candidate Screening verified.")


def test_end_to_end_multifidelity_cycle():
    print("\n--- Test 4: End-to-End Multi-Fidelity Optimization Cycle ---")
    param_defs = {
        "number_of_complete_revolutions": {"min": 1.0, "max": 4.0, "default": 2.0},
        "helix_path_radius_mm": {"min": 1.5, "max": 4.0, "default": 2.0}
    }

    db_path = "artifacts/test_e2e_mf.json"
    if os.path.exists(db_path):
        os.remove(db_path)

    opt = MultiFidelityModelFusionOptimizer(
        physics_driver=None,
        parameter_defs=param_defs,
        domain="cfd",
        surrogate_db_path=db_path,
        screening_eff_threshold=87.0,
        verbose=False
    )

    # Run 4 multi-fidelity iterations
    promoted_count = 0
    pruned_count = 0
    for i in range(4):
        rec = opt.step(mock_run=True)
        if rec["status"] == "PROMOTED_TO_FINE":
            promoted_count += 1
        else:
            pruned_count += 1

    stats = opt.surrogate.get_fidelity_stats()
    print(f"  Completed 4 iterations: {promoted_count} Promoted to Fine, {pruned_count} Pruned Coarse")
    print(f"  Final stats: {stats}")
    assert len(opt.history) == 4
    assert os.path.exists(opt.history[-1]["field_bin"])
    print("  [PASS] End-to-end multi-fidelity active optimization cycle verified.")


if __name__ == "__main__":
    print("=================================================================")
    print("Starting Multi-Fidelity Verification Suite...")
    print("=================================================================")
    test_multifidelity_surrogate_calibration()
    test_multifidelity_driver_execution()
    test_two_stage_screening_pruning()
    test_end_to_end_multifidelity_cycle()
    print("\n=================================================================")
    print("ALL MULTI-FIDELITY VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("=================================================================")
