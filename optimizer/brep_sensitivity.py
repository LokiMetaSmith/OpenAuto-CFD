"""
brep_sensitivity.py
Spatial-Projected Boundary Velocity Extractor & Hadamard-Zolesio Shape Sensitivity Engine.

Computes exact boundary sensitivities of build123d solids without depending on
OpenCascade (OCP) TopoDS_Face index stability:
    dJ/dp = ∫_Γ G(u, ψ) (dx_Γ/dp · n) dΓ

Evaluates the scalar normal boundary velocity (dx_Γ/dp · n) via spatial closest-point
projection / local Signed Distance Field (SDF) queries evaluated directly at the
unperturbed mesh boundary face centroids. This guarantees invariance against
OpenCascade face-splitting and topological renumbering.
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional, Callable, List
from scipy.spatial import cKDTree
import trimesh





class SpatialBRepSensitivityEngine:
    """
    Extracts topological-invariant boundary normal velocities and computes
    continuous Hadamard-Zolesio shape gradients across parameter perturbations.
    """

    def __init__(self, eps_p: float = 1e-4):
        self.eps_p = eps_p

    def extract_boundary_quadrature(self, mesh: trimesh.Trimesh) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extracts face centroids, outward unit normals, and surface areas.
        Returns:
            centroids: (M, 3)
            normals: (M, 3)
            areas: (M,)
        """
        centroids = mesh.triangles_center.astype(np.float64)
        normals = mesh.face_normals.astype(np.float64)
        areas = mesh.area_faces.astype(np.float64)

        # Normalize normals to unit length
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = np.where(norms > 1e-12, normals / norms, np.array([0.0, 0.0, 1.0]))

        return centroids, normals, areas

    def compute_normal_boundary_velocity(
        self,
        mesh_base: trimesh.Trimesh,
        mesh_perturbed: trimesh.Trimesh,
        delta_p: float
    ) -> np.ndarray:
        """
        Computes scalar normal boundary velocity v_n = (dx_Γ/dp · n) at base mesh centroids
        using spatial closest-point projection onto the perturbed surface.
        Insulated against OpenCascade internal face reordering.

        Returns:
            v_n: (M,) array of normal velocities at each face centroid of mesh_base.
        """
        if abs(delta_p) < 1e-12:
            return np.zeros(len(mesh_base.faces), dtype=np.float64)

        centroids_0, normals_0, _ = self.extract_boundary_quadrature(mesh_base)

        # Spatial closest point projection on the perturbed surface
        # Using trimesh ProximityQuery with BVH acceleration
        closest_points, distances, _ = mesh_perturbed.nearest.on_surface(centroids_0)

        # Displacement vector from base centroid to perturbed surface
        displacement = closest_points - centroids_0  # (M, 3)

        # Scalar projection onto outward unit normal
        # v_n = (displacement · n) / delta_p
        normal_displacement = np.sum(displacement * normals_0, axis=1)

        # For points outside the perturbed mesh, trimesh distance is positive;
        # if the surface expanded outward along n, displacement · n > 0.
        # Check inside/outside sign using contains query if available, or directional dot product
        v_n = normal_displacement / delta_p
        return v_n

    def compute_hadamard_shape_gradient(
        self,
        mesh_base: trimesh.Trimesh,
        v_n: np.ndarray,
        adjoint_density: np.ndarray
    ) -> float:
        """
        Evaluates the continuous boundary integral:
            dJ/dp = ∫_Γ G(u, ψ) * v_n dΓ ≈ ∑ G_i * v_{n, i} * ΔA_i
        adjoint_density: (M,) scalar density G on each face of mesh_base.
        """
        _, _, areas = self.extract_boundary_quadrature(mesh_base)
        dJ_dp = float(np.sum(adjoint_density * v_n * areas))
        return dJ_dp

    def verify_shape_gradient_vs_fd(
        self,
        geom_builder: Callable[[float], trimesh.Trimesh],
        p0: float,
        functional_fn: Callable[[trimesh.Trimesh], float],
        adjoint_density_fn: Callable[[trimesh.Trimesh], np.ndarray],
        delta_p: float = 1e-4
    ) -> Tuple[float, float, float]:
        """
        Cross-checks Hadamard shape derivative against central finite difference of the functional J(p):
            dJ/dp_FD = (J(p0 + delta_p) - J(p0 - delta_p)) / (2 * delta_p)
        Returns:
            (rel_error, dJ_dp_hadamard, dJ_dp_fd)
        """
        mesh_base = geom_builder(p0)
        mesh_plus = geom_builder(p0 + delta_p)
        mesh_minus = geom_builder(p0 - delta_p)

        # 1. Finite Difference
        J_plus = functional_fn(mesh_plus)
        J_minus = functional_fn(mesh_minus)
        dJ_dp_fd = (J_plus - J_minus) / (2.0 * delta_p)

        # 2. Hadamard Boundary Velocity (central difference on boundary points)
        v_n_plus = self.compute_normal_boundary_velocity(mesh_base, mesh_plus, delta_p)
        v_n_minus = self.compute_normal_boundary_velocity(mesh_base, mesh_minus, -delta_p)
        v_n_central = 0.5 * (v_n_plus + v_n_minus)

        # 3. Shape Integral
        adjoint_density = adjoint_density_fn(mesh_base)
        dJ_dp_hadamard = self.compute_hadamard_shape_gradient(mesh_base, v_n_central, adjoint_density)

        rel_error = abs(dJ_dp_hadamard - dJ_dp_fd) / (abs(dJ_dp_fd) + 1e-12)
        return rel_error, dJ_dp_hadamard, dJ_dp_fd
