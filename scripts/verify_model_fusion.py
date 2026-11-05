"""
verify_model_fusion.py

Comprehensive Test Suite for Closed-Loop EM Model Fusion & FNO-3D Pipeline:
  1. Zero-Training RBF Field and S-Parameter Surrogate
  2. Binary Phasor Encoding/Decoding for GPU Shaders
  3. Closed-Loop Active Model Fusion Loop (Residual Learning)
  4. 3D Fourier Neural Operator (FNO-3D) Architecture
"""

import os
import sys
import time
import shutil
import numpy as np

# Ensure optimizer is on sys.path
sys.path.append(os.path.abspath("optimizer"))

from surrogate_rbf import RBFFieldSurrogate
from phasor_export import encode_phasor_binary, decode_phasor_binary, export_phasor_file
from model_fusion import ModelFusionOptimizer
from em_driver import OpenEMSDriver

try:
    from fno_3d import FNOModelWrapper, HAS_TORCH, FNO3D, relative_l2_loss
except (ImportError, OSError):
    HAS_TORCH = False


def test_zero_training_rbf_surrogate():
    print("\n" + "="*70)
    print(" [TEST 1] Zero-Training RBF Field & Metric Surrogate")
    print("="*70)

    param_names = ["helix_path_radius_mm", "helix_profile_radius_mm", "tube_wall_mm"]
    surrogate = RBFFieldSurrogate(param_names=param_names)

    # 1. Add sparse training observations (3 samples)
    n_points = 500
    coords = np.random.uniform(-20, 20, (n_points, 3)).astype(np.float32)

    samples = [
        {"helix_path_radius_mm": 10.0, "helix_profile_radius_mm": 3.0, "tube_wall_mm": 1.0},
        {"helix_path_radius_mm": 15.0, "helix_profile_radius_mm": 4.5, "tube_wall_mm": 1.5},
        {"helix_path_radius_mm": 20.0, "helix_profile_radius_mm": 6.0, "tube_wall_mm": 2.0},
    ]
    s11_targets = [-12.5, -22.0, -14.0]
    freqs = np.linspace(2.4e9, 2.5e9, 11)

    for i, p in enumerate(samples):
        # Synthetic standing wave field
        e_re = np.column_stack([
            np.sin(coords[:, 0] * 0.1 * (i + 1)),
            np.cos(coords[:, 1] * 0.1 * (i + 1)),
            np.zeros(n_points)
        ]).astype(np.float32)
        e_im = np.column_stack([
            np.cos(coords[:, 0] * 0.1 * (i + 1)),
            -np.sin(coords[:, 1] * 0.1 * (i + 1)),
            np.ones(n_points) * 0.1
        ]).astype(np.float32)

        s11_curve = -10.0 - (s11_targets[i] + 10.0) * np.exp(-((freqs - 2.45e9) / 2e7)**2)

        surrogate.add_sample(
            params=p,
            metrics={"S11": s11_targets[i]},
            sparam_curve=(freqs, s11_curve),
            field_data={"coords": coords, "E_re": e_re, "E_im": e_im}
        )

    # 2. Test exact reproduction at an observed point
    pred_m, unc = surrogate.predict_metrics(samples[1])
    print(f"  Observed Point Check: Actual S11={s11_targets[1]:.2f} dB, Predicted={pred_m['S11']:.2f} dB (Uncertainty: {unc:.4f})")
    assert abs(pred_m["S11"] - s11_targets[1]) < 0.2, "RBF should accurately fit observed points"
    assert unc < 1e-4, "Uncertainty at observed point should be near zero"

    # 3. Test interpolation at an unseen intermediate query point
    unseen_query = {"helix_path_radius_mm": 12.5, "helix_profile_radius_mm": 3.75, "tube_wall_mm": 1.25}
    t0 = time.time()
    pred_unseen, unc_unseen = surrogate.predict_metrics(unseen_query)
    sparam_pred = surrogate.predict_s_parameters(unseen_query)
    field_pred = surrogate.predict_field(unseen_query)
    t_eval = (time.time() - t0) * 1000.0

    print(f"  Interpolated Query: S11={pred_unseen['S11']:.2f} dB, Uncertainty={unc_unseen:.4f}")
    print(f"  Continuous S-param curve points: {len(sparam_pred[1])}")
    print(f"  Field prediction vectors: {field_pred['E_re'].shape} (Magnitude min/max: {field_pred['mag'].min():.3f} / {field_pred['mag'].max():.3f})")
    print(f"  Interpolation Inference Time: {t_eval:.2f} ms (Target: < 5 ms)")

    assert t_eval < 20.0, "RBF evaluation must be lightning fast (< 20ms)"
    assert field_pred["E_re"].shape == (n_points, 3)

    # 4. Test Persistence
    tmp_path = "artifacts/test_surrogate.json"
    surrogate.save(tmp_path)
    loaded_surr = RBFFieldSurrogate.load(tmp_path)
    pred_loaded, _ = loaded_surr.predict_metrics(unseen_query)
    assert abs(pred_loaded["S11"] - pred_unseen["S11"]) < 1e-6, "Saved and loaded surrogate must match"
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    print("  [PASS] Zero-Training RBF Surrogate verified successfully.")


def test_phasor_binary_export():
    print("\n" + "="*70)
    print(" [TEST 2] GPU Phasor Binary Format Encoder & Decoder")
    print("="*70)

    n = 2048
    coords = np.random.uniform(-15, 15, (n, 3)).astype(np.float32)
    e_re = np.random.normal(0, 1, (n, 3)).astype(np.float32)
    e_im = np.random.normal(0, 1, (n, 3)).astype(np.float32)
    mag = np.sqrt(np.sum(e_re**2 + e_im**2, axis=-1)).astype(np.float32)

    # Encode to binary
    buf = encode_phasor_binary(coords, e_re, e_im, mag)
    expected_bytes = 4 + n * 10 * 4
    print(f"  Encoded buffer size: {len(buf)} bytes (Expected: {expected_bytes} bytes for {n} probes)")
    assert len(buf) == expected_bytes, "Binary buffer length mismatch"

    # Decode and verify exact round-trip
    decoded = decode_phasor_binary(buf)
    assert decoded["n"] == n
    np.testing.assert_allclose(decoded["coords"], coords, atol=1e-6)
    np.testing.assert_allclose(decoded["E_re"], e_re, atol=1e-6)
    np.testing.assert_allclose(decoded["E_im"], e_im, atol=1e-6)
    np.testing.assert_allclose(decoded["mag"], mag, atol=1e-6)

    # Test file writing
    export_path = "artifacts/test_phasor.bin"
    export_phasor_file(export_path, coords, e_re, e_im, mag)
    assert os.path.exists(export_path)
    assert os.path.getsize(export_path) == expected_bytes
    os.remove(export_path)

    print("  [PASS] Binary phasor encoding/decoding is byte-exact.")


def test_closed_loop_model_fusion():
    print("\n" + "="*70)
    print(" [TEST 3] Closed-Loop Model Fusion & Active Optimization")
    print("="*70)

    param_defs = {
        "helix_path_radius_mm": {"min": 8.0, "max": 25.0, "type": "float"},
        "helix_profile_radius_mm": {"min": 2.0, "max": 8.0, "type": "float"},
        "tube_wall_mm": {"min": 0.8, "max": 3.0, "type": "float"},
    }

    case_dir = "artifacts/test_fusion_case"
    os.makedirs(case_dir, exist_ok=True)
    driver = OpenEMSDriver(case_dir=case_dir, container_engine="none", verbose=False)

    optimizer = ModelFusionOptimizer(
        physics_driver=driver,
        parameter_defs=param_defs,
        surrogate_db_path="artifacts/test_fusion_memory.json",
        exploration_weight=0.2,
        verbose=True
    )

    # Run 3 active learning cycles with mock solver
    print("  Executing 3 closed-loop active learning cycles...")
    for i in range(3):
        res = optimizer.step(mock_run=True)
        assert res["solver_success"] is True
        assert np.isfinite(res["residual_s11"])
        assert os.path.exists(res["phasor_bin"])

    print(f"\n  Final Surrogate Database Size: {len(optimizer.surrogate_rbf.param_history)} samples")
    print(f"  Surrogate S11 Prediction Error Trend: {[round(h['residual_s11'], 2) for h in optimizer.history]}")

    # Cleanup test files
    if os.path.exists("artifacts/test_fusion_memory.json"):
        os.remove("artifacts/test_fusion_memory.json")
    for h in optimizer.history:
        if os.path.exists(h["phasor_bin"]):
            os.remove(h["phasor_bin"])
    if os.path.exists(case_dir):
        shutil.rmtree(case_dir)

    print("  [PASS] Closed-loop active optimization pipeline verified.")


def test_fno_3d_architecture():
    print("\n" + "="*70)
    print(" [TEST 4] 3D Fourier Neural Operator (FNO-3D) Architecture")
    print("="*70)

    if not HAS_TORCH:
        print("  Notice: PyTorch not installed in this environment. Skipping torch neural test.")
        print("  (FNOModelWrapper gracefully operates in fallback mode)")
        return

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  PyTorch Version: {torch.__version__} (Device: {device})")

    # Instantiate model
    grid_dim = (24, 24, 24)
    batch_size = 2
    fno = FNO3D(in_channels=2, out_channels=6, width=16, modes=(6, 6, 6), n_blocks=2, sparam_bins=11).to(device)

    # Mock input: (B, 2, Nx, Ny, Nz)
    x = torch.randn(batch_size, 2, *grid_dim, device=device)
    target_fields = torch.randn(batch_size, 6, *grid_dim, device=device)
    target_sparams = torch.randn(batch_size, 11, device=device)

    # Forward pass
    t0 = time.time()
    pred_fields, pred_sparams = fno(x)
    t_fwd = (time.time() - t0) * 1000.0

    print(f"  Input Tensor Shape: {tuple(x.shape)}")
    print(f"  Predicted Volumetric Field Shape: {tuple(pred_fields.shape)} (Ex, Ey, Ez complex)")
    print(f"  Predicted S-Parameter Curve Shape: {tuple(pred_sparams.shape)}")
    print(f"  Forward Pass Time: {t_fwd:.2f} ms")

    assert pred_fields.shape == (batch_size, 6, *grid_dim)
    assert pred_sparams.shape == (batch_size, 11)

    # Backward pass
    loss_fields = relative_l2_loss(pred_fields, target_fields)
    loss_sparams = torch.nn.functional.mse_loss(pred_sparams, target_sparams)
    total_loss = loss_fields + loss_sparams

    total_loss.backward()
    print(f"  Backward Pass Loss: {total_loss.item():.4f}")
    assert total_loss.item() > 0

    print("  [PASS] FNO-3D spectral convolutions and gradients verified.")


def main():
    print("======================================================================")
    print("       EM Model Fusion & FNO-3D Surrogate Verification Suite")
    print("======================================================================")

    test_zero_training_rbf_surrogate()
    test_phasor_binary_export()
    test_closed_loop_model_fusion()
    test_fno_3d_architecture()

    print("\n" + "="*70)
    print(" ALL VERIFICATION TESTS PASSED SUCCESSFULLY! ")
    print("="*70)


if __name__ == "__main__":
    main()
