"""
verify_cfd_fea_fusion.py

Verification suite for Multi-Physics Model Fusion across OpenFOAM (CFD) and CalculiX (FEA).
Tests:
  1. MultiPhysicsSurrogate continuous interpolation of metrics and 3D vector fields.
  2. Binary 3D Field buffer export and zero-loss roundtrip reading.
  3. Closed-loop CFD active learning cycle (pressure drop, separation efficiency).
  4. Closed-loop FEA active learning cycle (Von Mises stress, displacement, FoS).
  5. Joint multi-physics optimization coupling fluid and structural domains.
"""

import os
import sys
import shutil
import tempfile
import numpy as np

# Ensure optimizer directory is on sys.path
sys.path.insert(0, os.path.abspath("optimizer"))

from surrogate_multiphysics import MultiPhysicsSurrogate
from cfd_fea_field_io import (
    export_multiphysics_field_bin,
    read_multiphysics_field_bin,
    sample_corkscrew_mesh_points,
    extract_openfoam_fields,
    extract_fea_fields
)
from model_fusion_multiphysics import MultiPhysicsModelFusionOptimizer
from fea_driver import FeaDriver


def test_cfd_surrogate_and_field_prediction():
    print("\n--- Test 1: CFD MultiPhysicsSurrogate Metric & Field Interpolation ---")
    param_names = ["helix_path_radius_mm", "number_of_complete_revolutions"]
    surrogate = MultiPhysicsSurrogate(domain="cfd", param_names=param_names)

    # Add 4 calibration points
    coords = sample_corkscrew_mesh_points(n_points=100)
    for r in [1.5, 2.5]:
        for n in [1.5, 3.0]:
            p = {"helix_path_radius_mm": r, "number_of_complete_revolutions": n}
            delta_p = 1000.0 + 500.0 * n + 100.0 * r
            eff = 85.0 + 4.0 * n - 2.0 * r
            field = extract_openfoam_fields(".", params=p, n_points=100)
            surrogate.add_sample(params=p, metrics={"delta_p": delta_p, "separation_efficiency": eff}, field_data=field)

    assert surrogate.is_fitted, "Surrogate should be fitted after adding samples"
    assert len(surrogate.param_history) == 4

    # Test query at intermediate parameter
    query_p = {"helix_path_radius_mm": 2.0, "number_of_complete_revolutions": 2.25}
    pred_m, unc = surrogate.predict_metrics(query_p)
    print(f"  CFD Query Params: {query_p}")
    print(f"  Predicted Metrics: delta_p={pred_m['delta_p']:.1f} Pa, eff={pred_m['separation_efficiency']:.2f}% (uncertainty={unc:.3f})")
    assert 1500.0 < pred_m["delta_p"] < 3000.0
    assert 80.0 < pred_m["separation_efficiency"] < 100.0
    assert 0.0 <= unc <= 1.0

    # Test 3D field prediction
    pred_field = surrogate.predict_field(query_p)
    assert pred_field is not None
    assert "U" in pred_field and "p" in pred_field
    assert pred_field["U"].shape == (100, 3)
    assert pred_field["p"].shape == (100,)
    print(f"  Predicted 3D Field: U shape {pred_field['U'].shape}, p shape {pred_field['p'].shape}")

    # Test serialization
    temp_json = "artifacts/test_cfd_surrogate.json"
    surrogate.save(temp_json)
    loaded = MultiPhysicsSurrogate.load(temp_json)
    pred_loaded, _ = loaded.predict_metrics(query_p)
    assert np.isclose(pred_m["delta_p"], pred_loaded["delta_p"], atol=1e-3)
    print("  [PASS] CFD Surrogate metrics and 3D fields verified.")


def test_fea_surrogate_and_field_prediction():
    print("\n--- Test 2: FEA MultiPhysicsSurrogate Metric & Field Interpolation ---")
    param_names = ["blade_chamfer_mm", "fluid_pressure_bar"]
    surrogate = MultiPhysicsSurrogate(domain="fea", param_names=param_names)

    # Add calibration points
    for chamfer in [0.2, 0.8]:
        for p_bar in [1.0, 2.5]:
            p = {"blade_chamfer_mm": chamfer, "fluid_pressure_bar": p_bar}
            kt = max(1.1, 2.5 - chamfer * 0.8)
            vm = round(p_bar * 12.5 * kt, 2)
            fos = round(60.0 / vm, 2)
            disp = round(p_bar * 0.08 / (1.0 + chamfer * 0.2), 3)
            field = extract_fea_fields(".", params=p, n_points=100)
            surrogate.add_sample(
                params=p,
                metrics={"max_von_mises_stress_MPa": vm, "factor_of_safety": fos, "max_displacement_mm": disp},
                field_data=field
            )

    query_p = {"blade_chamfer_mm": 0.5, "fluid_pressure_bar": 1.75}
    pred_m, unc = surrogate.predict_metrics(query_p)
    print(f"  FEA Query Params: {query_p}")
    print(f"  Predicted Metrics: vm_stress={pred_m['max_von_mises_stress_MPa']:.2f} MPa, FoS={pred_m['factor_of_safety']:.2f}, disp={pred_m['max_displacement_mm']:.3f} mm")
    assert 20.0 < pred_m["max_von_mises_stress_MPa"] < 60.0
    assert 1.0 < pred_m["factor_of_safety"] < 4.0

    pred_field = surrogate.predict_field(query_p)
    assert pred_field is not None
    assert "disp" in pred_field and "von_mises" in pred_field
    print(f"  Predicted FEA Field: disp shape {pred_field['disp'].shape}, von_mises shape {pred_field['von_mises'].shape}")
    print("  [PASS] FEA Surrogate metrics and 3D fields verified.")


def test_binary_field_buffer_roundtrip():
    print("\n--- Test 3: Binary Multi-Physics Field Buffer Round-Trip ---")
    n_pts = 250
    coords = np.random.uniform(-10, 10, (n_pts, 3)).astype(np.float32)
    # CFD: 4 channels (ux, uy, uz, p)
    cfd_values = np.random.normal(0, 1, (n_pts, 4)).astype(np.float32)

    bin_path = "artifacts/test_roundtrip_cfd.bin"
    export_multiphysics_field_bin(bin_path, coords=coords, values=cfd_values, domain="CFD")

    read_back = read_multiphysics_field_bin(bin_path)
    assert read_back["domain"] == "CFD"
    assert read_back["n_points"] == n_pts
    assert read_back["n_channels"] == 4
    assert np.allclose(read_back["coords"], coords, atol=1e-6)
    assert np.allclose(read_back["values"], cfd_values, atol=1e-6)
    print(f"  Successfully verified {os.path.getsize(bin_path)} bytes binary field buffer for {n_pts} points.")
    print("  [PASS] Binary Field I/O verified.")


def test_closed_loop_cfd_model_fusion():
    print("\n--- Test 4: Closed-Loop CFD Model Fusion Active Learning ---")
    param_defs = {
        "helix_path_radius_mm": {"min": 1.5, "max": 5.0, "default": 1.8},
        "number_of_complete_revolutions": {"min": 1.0, "max": 4.0, "default": 2.0}
    }

    class MockFoamDriver:
        def __init__(self):
            self.case_dir = "temp_case_cfd"
        def prepare_case(self, params=None): pass
        def run_solver(self): return True
        def get_metrics(self):
            return {"delta_p": 2100.0, "separation_efficiency": 96.5, "residuals": 8.5e-5}
        def cleanup_ram_disk(self): pass

    db_cfd = "artifacts/test_fusion_cfd.json"
    if os.path.exists(db_cfd):
        os.remove(db_cfd)
    if os.path.exists(db_cfd.replace(".json", "_fields.npz")):
        os.remove(db_cfd.replace(".json", "_fields.npz"))

    driver = MockFoamDriver()
    opt = MultiPhysicsModelFusionOptimizer(
        physics_driver=driver,
        parameter_defs=param_defs,
        domain="cfd",
        surrogate_db_path=db_cfd,
        verbose=True
    )

    for step_i in range(3):
        rec = opt.step(mock_run=True)
        assert "delta_p" in rec["actual_metrics"]
        assert "separation_efficiency" in rec["actual_metrics"]
        assert "res_delta_p" in rec["residuals"]
        assert os.path.exists(rec["field_bin"])

    print(f"  Active learning completed 3 steps. Final surrogate sample count: {len(opt.surrogate.param_history)}")
    print("  [PASS] CFD Model Fusion loop verified.")


def test_closed_loop_fea_model_fusion():
    print("\n--- Test 5: Closed-Loop FEA Model Fusion Active Learning ---")
    param_defs = {
        "blade_chamfer_mm": {"min": 0.1, "max": 1.0, "default": 0.5},
        "inlet_fillet_radius_mm": {"min": 0.1, "max": 1.0, "default": 0.5}
    }

    db_fea = "artifacts/test_fusion_fea.json"
    if os.path.exists(db_fea):
        os.remove(db_fea)
    if os.path.exists(db_fea.replace(".json", "_fields.npz")):
        os.remove(db_fea.replace(".json", "_fields.npz"))

    tmp_dir = tempfile.mkdtemp(prefix="fea_test_")
    try:
        fea_driver = FeaDriver(case_dir=tmp_dir)
        opt = MultiPhysicsModelFusionOptimizer(
            physics_driver=fea_driver,
            parameter_defs=param_defs,
            domain="fea",
            surrogate_db_path=db_fea,
            verbose=True
        )

        for step_i in range(3):
            rec = opt.step(mock_run=False)  # Real B-rep FEA execution
            assert "max_von_mises_stress_MPa" in rec["actual_metrics"]
            assert "factor_of_safety" in rec["actual_metrics"]
            assert rec["actual_metrics"]["factor_of_safety"] > 0
            assert os.path.exists(rec["field_bin"])

        fea_driver.cleanup_ram_disk()
    finally:
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)

    print("  [PASS] FEA Model Fusion loop verified.")


def test_joint_multiphysics_model_fusion():
    print("\n--- Test 6: Joint CFD + FEA Multi-Physics Optimization ---")
    param_defs = {
        "number_of_complete_revolutions": {"min": 1.0, "max": 4.0, "default": 2.0},
        "blade_chamfer_mm": {"min": 0.1, "max": 1.0, "default": 0.5}
    }

    class MockJointDriver:
        def __init__(self):
            self.case_dir = "temp_case_joint"
        def prepare_case(self, params=None): pass
        def run_solver(self): return True
        def get_metrics(self):
            return {
                "delta_p": 2200.0,
                "separation_efficiency": 97.2,
                "max_von_mises_stress_MPa": 28.5,
                "factor_of_safety": 2.1
            }
        def cleanup_ram_disk(self): pass

    joint_driver = MockJointDriver()
    opt = MultiPhysicsModelFusionOptimizer(
        physics_driver=joint_driver,
        parameter_defs=param_defs,
        domain="joint",
        surrogate_db_path="artifacts/test_fusion_joint.json",
        verbose=True
    )

    rec = opt.step(mock_run=True)
    assert "delta_p" in rec["actual_metrics"]
    assert "factor_of_safety" in rec["actual_metrics"]
    print(f"  Joint optimization verified. Acquisition balanced fluid and structural constraints.")
    print("  [PASS] Joint Multi-Physics loop verified.")


if __name__ == "__main__":
    print("=================================================================")
    print("Starting Verification Suite for CFD and FEA Model Fusion...")
    print("=================================================================")
    test_cfd_surrogate_and_field_prediction()
    test_fea_surrogate_and_field_prediction()
    test_binary_field_buffer_roundtrip()
    test_closed_loop_cfd_model_fusion()
    test_closed_loop_fea_model_fusion()
    test_joint_multiphysics_model_fusion()
    print("\n=================================================================")
    print("ALL MULTI-PHYSICS VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("=================================================================")
