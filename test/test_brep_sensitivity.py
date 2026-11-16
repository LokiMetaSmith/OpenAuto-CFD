"""
test_brep_sensitivity.py
Unit tests verifying the Spatial-Projected Boundary Velocity Extractor &
continuous Hadamard-Zolesio Shape Sensitivity Engine.
"""

import os
import sys
import numpy as np
import pytest
import trimesh

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from optimizer.brep_sensitivity import SpatialBRepSensitivityEngine


def make_cylinder_mesh(radius: float, height: float = 20.0, sections: int = 32) -> trimesh.Trimesh:
    """Helper to generate a clean closed cylinder mesh with given radius."""
    return trimesh.creation.cylinder(radius=radius, height=height, sections=sections)


def test_boundary_quadrature_extraction():
    """Verify face centroids, normals, and areas extraction."""
    engine = SpatialBRepSensitivityEngine()
    mesh = make_cylinder_mesh(radius=5.0, height=10.0)

    centroids, normals, areas = engine.extract_boundary_quadrature(mesh)

    assert centroids.shape[0] == len(mesh.faces)
    assert normals.shape[0] == len(mesh.faces)
    assert areas.shape[0] == len(mesh.faces)

    # Verify unit normal lengths
    lengths = np.linalg.norm(normals, axis=1)
    assert np.allclose(lengths, 1.0, atol=1e-5)

    # Total area matches trimesh area
    assert abs(np.sum(areas) - mesh.area) < 1e-5


def test_hadamard_shape_gradient_vs_fd_volume():
    """
    Verify Hadamard shape derivative against finite differences for a geometric functional:
    Functional J(r) = Volume(r) of a cylinder of radius r, height H.
    Analytic: V(r) = pi * r^2 * H  ==>  dV/dr = 2 * pi * r * H.
    Hadamard Theorem:
        dJ/dr = ∫_Γ G(x) (dx_Γ/dr · n) dΓ
    For volume, the boundary adjoint density G(x) = 1 everywhere on the boundary.
    Normal displacement on lateral wall: dx/dr · n = 1.
    Normal displacement on end caps: dx/dr · n = 0.
    Thus ∫_Γ 1 * (dx/dr · n) dΓ = Lateral Area = 2 * pi * r * H.
    """
    engine = SpatialBRepSensitivityEngine()
    H = 20.0
    r0 = 8.0

    def geom_builder(r: float) -> trimesh.Trimesh:
        return make_cylinder_mesh(radius=r, height=H, sections=48)

    def functional_fn(m: trimesh.Trimesh) -> float:
        return float(m.volume)

    def adjoint_density_fn(m: trimesh.Trimesh) -> np.ndarray:
        # G = 1 for volume functional
        return np.ones(len(m.faces), dtype=np.float64)

    rel_error, dJ_hadamard, dJ_fd = engine.verify_shape_gradient_vs_fd(
        geom_builder=geom_builder,
        p0=r0,
        functional_fn=functional_fn,
        adjoint_density_fn=adjoint_density_fn,
        delta_p=1e-3
    )

    print(f"Hadamard: {dJ_hadamard:.4f}, FD: {dJ_fd:.4f}, rel_error: {rel_error:.2e}")
    # Gate check: relative error must be <= 10^-3 for continuum boundary mesh
    assert rel_error <= 1e-3, f"Shape gradient rel_error {rel_error:.2e} exceeded gate threshold 1e-3"


def test_topological_invariance_against_face_shuffling():
    """
    Verify that boundary velocity extraction is completely invariant to
    internal face index reordering (the OpenCascade face naming problem).
    """
    engine = SpatialBRepSensitivityEngine()
    mesh_base = make_cylinder_mesh(radius=5.0, height=10.0, sections=24)
    mesh_pert = make_cylinder_mesh(radius=5.05, height=10.0, sections=24)

    # 1. Normal velocity on ordered mesh
    v_n_orig = engine.compute_normal_boundary_velocity(mesh_base, mesh_pert, delta_p=0.05)

    # 2. Permute the perturbed mesh's faces and vertices randomly (simulating OCCT re-indexing)
    rng = np.random.default_rng(42)
    faces_shuffled = mesh_pert.faces.copy()
    rng.shuffle(faces_shuffled)

    mesh_shuffled = trimesh.Trimesh(vertices=mesh_pert.vertices.copy(), faces=faces_shuffled, process=False)

    # 3. Normal velocity on shuffled mesh
    v_n_shuffled = engine.compute_normal_boundary_velocity(mesh_base, mesh_shuffled, delta_p=0.05)

    # Spatial closest-point projection must produce identical normal velocity vectors!
    max_diff = np.max(np.abs(v_n_orig - v_n_shuffled))
    assert max_diff < 1e-6, f"Max velocity diff {max_diff} exceeded 1e-6 under face reordering"
