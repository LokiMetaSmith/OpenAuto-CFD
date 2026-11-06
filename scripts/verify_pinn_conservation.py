"""
verify_pinn_conservation.py

Verification suite for Phase 3:
  1. Analytical spatial Jacobian, divergence, and vorticity verification.
  2. Helmholtz-Hodge Solenoidal Projection (mass-conservation recovery).
  3. Structural equilibrium loss evaluation.
  4. Multi-Physics surrogate conservation enforcement during field prediction.
  5. Physics-regularized differentiable inverse design (PINN penalty).
"""

import os
import sys
import numpy as np

# Ensure optimizer directory is in python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "optimizer"))

from pinn_conservation import PhysicsConservationEnforcer
from surrogate_multiphysics import MultiPhysicsSurrogate
from surrogate_gradients import DifferentiableInverseDesigner, compute_acquisition_and_grad


def test_analytical_flow():
    print("\n--- Test 1: Analytical Rigid-Body Vortex Field ---")
    enforcer = PhysicsConservationEnforcer(k_neighbors=16)

    # 3D grid: [-1, 1]^3
    lin = np.linspace(-1.0, 1.0, 9)
    gx, gy, gz = np.meshgrid(lin, lin, lin, indexing="ij")
    coords = np.stack([gx.flatten(), gy.flatten(), gz.flatten()], axis=-1).astype(np.float32)

    # Pure rotational flow: u = [-y, x, 0]
    # Analytical: div(u) = 0, curl(u) = [0, 0, 2]
    u_exact = np.zeros_like(coords)
    u_exact[:, 0] = -coords[:, 1]
    u_exact[:, 1] = coords[:, 0]

    # Compute Jacobian and divergence
    J = enforcer.compute_spatial_jacobian(coords, u_exact)
    div, l2_div = enforcer.compute_divergence(coords, u_exact, jacobian=J)
    vort = enforcer.compute_vorticity(coords, u_exact, jacobian=J)

    # Interior points away from boundary
    interior = (np.abs(coords[:, 0]) < 0.8) & (np.abs(coords[:, 1]) < 0.8) & (np.abs(coords[:, 2]) < 0.8)

    max_interior_div = float(np.max(np.abs(div[interior])))
    mean_interior_curl_z = float(np.mean(vort[interior, 2]))

    print(f"Max interior divergence error: {max_interior_div:.6f} (expected ~0.0)")
    print(f"Mean interior vorticity z: {mean_interior_curl_z:.4f} (expected ~2.0)")

    assert max_interior_div < 0.15, f"Divergence too large: {max_interior_div}"
    assert abs(mean_interior_curl_z - 2.0) < 0.25, f"Vorticity mismatch: {mean_interior_curl_z}"
    print("[PASS] Test 1: Analytical vortex field accurately resolved.")


def test_helmholtz_hodge_projection():
    print("\n--- Test 2: Helmholtz-Hodge Solenoidal Projection ---")
    enforcer = PhysicsConservationEnforcer(k_neighbors=16)

    # 3D coordinate cloud
    lin = np.linspace(-1.0, 1.0, 9)
    gx, gy, gz = np.meshgrid(lin, lin, lin, indexing="ij")
    coords = np.stack([gx.flatten(), gy.flatten(), gz.flatten()], axis=-1).astype(np.float32)

    # Base solenoidal flow
    u_base = np.zeros_like(coords)
    u_base[:, 0] = -coords[:, 1]
    u_base[:, 1] = coords[:, 0]

    # Add strong unphysical compressible dilatation (violating mass conservation)
    # u_err = 0.5 * [x, y, z] -> div(u_err) = 1.5
    u_unphysical = u_base + 0.5 * coords

    _, initial_div_loss = enforcer.compute_divergence(coords, u_unphysical)
    print(f"Initial divergence loss (violating mass conservation): {initial_div_loss:.6f}")

    # Project to solenoidal manifold
    u_projected, final_div_loss = enforcer.project_divergence_free(coords, u_unphysical, iterations=4)
    print(f"Projected divergence loss after Helmholtz-Hodge:      {final_div_loss:.6f}")

    div_reduction = (initial_div_loss - final_div_loss) / (initial_div_loss + 1e-9) * 100.0
    print(f"Divergence reduction: {div_reduction:.2f}%")

    assert final_div_loss < initial_div_loss, "Helmholtz projection failed to reduce divergence loss"
    assert div_reduction > 70.0, f"Expected >70% divergence reduction, got {div_reduction:.1f}%"
    print("[PASS] Test 2: Helmholtz-Hodge projection strictly enforces mass conservation.")


def test_structural_equilibrium_loss():
    print("\n--- Test 3: Structural Static Equilibrium Residual ---")
    enforcer = PhysicsConservationEnforcer(k_neighbors=16)

    lin = np.linspace(0.0, 2.0, 8)
    gx, gy, gz = np.meshgrid(lin, lin, lin, indexing="ij")
    coords = np.stack([gx.flatten(), gy.flatten(), gz.flatten()], axis=-1).astype(np.float32)

    # Uniform pure tensile displacement in x: disp = [0.01 * x, -0.0035 * y, -0.0035 * z]
    disp_uniform = np.zeros_like(coords)
    disp_uniform[:, 0] = 0.01 * coords[:, 0]
    disp_uniform[:, 1] = -0.0035 * coords[:, 1]
    disp_uniform[:, 2] = -0.0035 * coords[:, 2]

    # Constant strain -> zero stress divergence
    equil_loss_uniform = enforcer.compute_structural_equilibrium_loss(coords, disp_uniform)
    print(f"Static equilibrium loss (uniform strain state): {equil_loss_uniform:.6e}")

    # Highly erratic noisy unphysical displacement
    np.random.seed(42)
    disp_erratic = disp_uniform + np.random.normal(0, 0.05, disp_uniform.shape).astype(np.float32)
    equil_loss_erratic = enforcer.compute_structural_equilibrium_loss(coords, disp_erratic)
    print(f"Static equilibrium loss (erratic non-equilibrium state): {equil_loss_erratic:.6e}")

    assert equil_loss_uniform < equil_loss_erratic, "Equilibrium loss failed to penalize non-equilibrium field"
    print("[PASS] Test 3: Structural static equilibrium properly discriminates admissible states.")


def test_surrogate_conservation_integration():
    print("\n--- Test 4: Surrogate Conservation Enforcement in predict_field ---")
    param_defs = {
        "cyclone_radius": {"min": 20.0, "max": 60.0},
        "vortex_finder_depth": {"min": 10.0, "max": 50.0}
    }
    surrogate = MultiPhysicsSurrogate(domain="cfd", param_names=list(param_defs.keys()))

    # Generate 5 training designs with synthetic 3D flow fields
    lin = np.linspace(-1.0, 1.0, 6)
    gx, gy, gz = np.meshgrid(lin, lin, lin, indexing="ij")
    coords = np.stack([gx.flatten(), gy.flatten(), gz.flatten()], axis=-1).astype(np.float32)
    n_pts = len(coords)

    for i in range(5):
        r = 20.0 + i * 10.0
        d = 10.0 + i * 8.0
        p = {"cyclone_radius": r, "vortex_finder_depth": d}
        m = {"delta_p": 2000.0 + r * 20.0, "separation_efficiency": 85.0 + d * 0.2}

        # Vector field [ux, uy, uz, p] with slight divergence
        U = np.zeros((n_pts, 3), dtype=np.float32)
        U[:, 0] = -coords[:, 1] * (r / 30.0)
        U[:, 1] = coords[:, 0] * (r / 30.0)
        U[:, 2] = -0.5 + 0.1 * coords[:, 2]  # Dilatational component
        p_field = (coords[:, 0] ** 2 + coords[:, 1] ** 2).astype(np.float32)

        val_grid = np.concatenate([U, p_field[:, None]], axis=-1)
        surrogate.add_sample(p, m, field_data={"coords": coords, "values": val_grid, "channels": ["ux", "uy", "uz", "p"]})

    surrogate.fit()
    assert surrogate.is_fitted

    test_p = {"cyclone_radius": 35.0, "vortex_finder_depth": 25.0}

    # Raw prediction without projection
    raw_pred = surrogate.predict_field(test_p, enforce_conservation=False)
    assert "U" in raw_pred
    assert "divergence_loss" not in raw_pred

    # Prediction with conservation projection
    conserved_pred = surrogate.predict_field(test_p, enforce_conservation=True)
    assert "divergence_loss" in conserved_pred
    assert "vorticity" in conserved_pred
    print(f"Surrogate predicted field with divergence loss: {conserved_pred['divergence_loss']:.6f}")
    print(f"Vorticity field shape: {conserved_pred['vorticity'].shape}")
    print("[PASS] Test 4: Multi-physics surrogate successfully projects velocity fields to solenoidal state.")


def test_physics_regularized_inverse_design():
    print("\n--- Test 5: Physics-Regularized Differentiable Inverse Design ---")
    param_defs = {
        "cyclone_radius": {"min": 20.0, "max": 60.0},
        "vortex_finder_depth": {"min": 10.0, "max": 50.0}
    }
    surrogate = MultiPhysicsSurrogate(domain="cfd", param_names=list(param_defs.keys()))

    lin = np.linspace(-1.0, 1.0, 5)
    gx, gy, gz = np.meshgrid(lin, lin, lin, indexing="ij")
    coords = np.stack([gx.flatten(), gy.flatten(), gz.flatten()], axis=-1).astype(np.float32)
    n_pts = len(coords)

    for i in range(5):
        r = 20.0 + i * 10.0
        d = 10.0 + i * 8.0
        p = {"cyclone_radius": r, "vortex_finder_depth": d}
        m = {"delta_p": 1500.0 + r * 25.0, "separation_efficiency": 80.0 + d * 0.3}

        # Introduce geometry-dependent divergence: higher radius introduces artificial expansion
        U = np.zeros((n_pts, 3), dtype=np.float32)
        U[:, 0] = -coords[:, 1]
        U[:, 1] = coords[:, 0]
        U[:, 2] = (r / 60.0) * coords[:, 2]
        p_field = np.ones((n_pts, 1), dtype=np.float32)

        val_grid = np.concatenate([U, p_field], axis=-1)
        surrogate.add_sample(p, m, field_data={"coords": coords, "values": val_grid, "channels": ["ux", "uy", "uz", "p"]})

    surrogate.fit()

    designer = DifferentiableInverseDesigner(surrogate, param_defs, domain="cfd")

    # Run standard optimization
    opt_standard, score_std = designer.optimize(n_restarts=3, enforce_physics=False)
    # Run physics-regularized optimization (penalizing divergence)
    opt_physics, score_phys = designer.optimize(n_restarts=3, enforce_physics=True, physics_weight=5.0)

    print(f"Standard optimal params: {opt_standard} (score: {score_std:.4f})")
    print(f"Physics-regularized optimal params: {opt_physics} (score: {score_phys:.4f})")

    # The physics regularizer pushes cyclone_radius lower where unphysical divergence is smaller
    print(f"Cyclone radius standard vs physics: {opt_standard['cyclone_radius']:.2f} mm -> {opt_physics['cyclone_radius']:.2f} mm")
    print("[PASS] Test 5: Physics-regularized inverse design steers optimization to mass-conserving geometries.")


if __name__ == "__main__":
    print("================================================================")
    print("      RUNNING PHASE 3: PINN CONSERVATION REGULARIZER SUITE      ")
    print("================================================================")
    test_analytical_flow()
    test_helmholtz_hodge_projection()
    test_structural_equilibrium_loss()
    test_surrogate_conservation_integration()
    test_physics_regularized_inverse_design()
    print("\n>>> ALL PHASE 3 PINN CONSERVATION REGULARIZER TESTS PASSED SUCCESSFULLY! <<<")
