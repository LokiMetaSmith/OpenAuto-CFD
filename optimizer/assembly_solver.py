"""
assembly_solver.py
Declarative Upstream Assembly Layout Solver with Implicit Function Theorem (IFT) Jacobians.

Resolves assembly mate constraints, sensor standoff distances, and O-ring gland envelopes
(from wiring/filter_interconnect.yaml and parametric specs) into scalar dimensions and transforms
before invoking imperative build123d scripts.

Extracts analytic parameter sensitivities via the Implicit Function Theorem:
    F(x, p) = 0  ==>  dx/dp = - [dF/dx]^(-1) * [dF/dp]
in a single linear solve upon Levenberg-Marquardt convergence, avoiding backprop through LM iterations.
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from scipy.optimize import least_squares


class UpstreamAssemblySolver:
    """
    Solves declarative assembly constraints F(x, p) = 0 for cyclone filter,
    tangential inlet port, differential pressure sensor tap, and outlet caps.
    """

    def __init__(self, tol: float = 1e-10):
        self.tol = tol
        # Parameter names p:
        # 0: tube_od_mm (outer diameter of barrel)
        # 1: tube_wall_mm (wall thickness)
        # 2: insert_length_mm (axial length of corkscrew insert)
        # 3: inlet_port_dia_mm (tangential inlet port diameter)
        # 4: sensor_standoff_mm (DP sensor standoff clearance)
        # 5: oring_cs_mm (O-ring cross section diameter)
        self.param_names = [
            "tube_od_mm",
            "tube_wall_mm",
            "insert_length_mm",
            "inlet_port_dia_mm",
            "sensor_standoff_mm",
            "oring_cs_mm"
        ]

        # Solved assembly variables x:
        # 0: x_inlet (tangential inlet X center)
        # 1: y_inlet (tangential inlet Y center)
        # 2: z_inlet (axial Z location of inlet)
        # 3: z_outlet (axial Z location of clean permeate outlet)
        # 4: r_sensor_tap (radial position of DP transducer port)
        # 5: gland_depth_mm (O-ring sealing gland depth)
        self.state_names = [
            "x_inlet",
            "y_inlet",
            "z_inlet",
            "z_outlet",
            "r_sensor_tap",
            "gland_depth_mm"
        ]

    def residual_system(self, x: np.ndarray, p: np.ndarray) -> np.ndarray:
        """
        Evaluates constraint residuals F(x, p) = 0.
        x: (6,) assembly states
        p: (6,) design parameters
        """
        tube_od, tube_wall, insert_len, port_dia, sensor_standoff, oring_cs = p
        x_in, y_in, z_in, z_out, r_sensor, gland_depth = x

        r_outer = 0.5 * tube_od
        r_inner = r_outer - tube_wall
        r_port = 0.5 * port_dia

        residuals = np.zeros(6, dtype=np.float64)

        # 1. Tangential inlet tangent condition:
        # The center of the tangential inlet must lie on the circle of radius (r_inner - r_port)
        # and be oriented along the tangent (x_in = 0, y_in = r_inner - r_port)
        residuals[0] = x_in
        residuals[1] = y_in - (r_inner - r_port)

        # 2. Inlet axial position: 5mm below top of insert
        residuals[2] = z_in - (insert_len - 5.0)

        # 3. Permeate outlet axial span: at the bottom cap, span = insert_len + 8mm
        residuals[3] = z_out - (z_in - insert_len - 8.0)

        # 4. Sensor tap standoff: located on outer barrel wall + standoff clearance
        residuals[4] = r_sensor - (r_outer + sensor_standoff)

        # 5. O-ring gland compression balance: depth = 0.75 * cross_section (25% squeeze)
        residuals[5] = gland_depth - (0.75 * oring_cs)

        return residuals

    def solve(self, p_dict: Dict[str, float], x0: Optional[np.ndarray] = None) -> Tuple[Dict[str, float], np.ndarray]:
        """
        Runs Levenberg-Marquardt to solve F(x, p) = 0.
        Returns:
            solved_dict: Dict mapping state names to solved scalar dimensions
            x_sol: np.ndarray (6,)
        """
        p = np.array([float(p_dict.get(name, self._default_param(name))) for name in self.param_names], dtype=np.float64)

        if x0 is None:
            # Smart initial guess
            r_outer = 0.5 * p[0]
            r_inner = r_outer - p[1]
            x0 = np.array([0.0, r_inner - 0.5 * p[3], p[2] - 5.0, -13.0, r_outer + p[4], 0.75 * p[5]], dtype=np.float64)

        res = least_squares(
            fun=self.residual_system,
            x0=x0,
            args=(p,),
            method='lm',
            ftol=self.tol,
            xtol=self.tol,
            gtol=self.tol
        )

        x_sol = res.x
        solved_dict = {name: float(x_sol[i]) for i, name in enumerate(self.state_names)}
        return solved_dict, x_sol

    def compute_ift_jacobian(self, x_sol: np.ndarray, p: np.ndarray) -> np.ndarray:
        """
        Computes analytic Implicit Function Theorem Jacobian:
            dx/dp = - [dF/dx]^(-1) * [dF/dp]
        Returns:
            dx_dp: (6, 6) matrix
        """
        # Formulate dF/dx (6x6) analytically
        # F0 = x_in
        # F1 = y_in - (0.5*tube_od - tube_wall - 0.5*port_dia)
        # F2 = z_in - (insert_len - 5.0)
        # F3 = z_out - (z_in - insert_len - 8.0)
        # F4 = r_sensor - (0.5*tube_od + sensor_standoff)
        # F5 = gland_depth - (0.75 * oring_cs)

        dF_dx = np.zeros((6, 6), dtype=np.float64)
        dF_dx[0, 0] = 1.0  # dF0/dx_in
        dF_dx[1, 1] = 1.0  # dF1/dy_in
        dF_dx[2, 2] = 1.0  # dF2/dz_in
        dF_dx[3, 2] = -1.0 # dF3/dz_in
        dF_dx[3, 3] = 1.0  # dF3/dz_out
        dF_dx[4, 4] = 1.0  # dF4/dr_sensor
        dF_dx[5, 5] = 1.0  # dF5/dgland_depth

        # Formulate dF/dp (6x6) analytically
        # p = [tube_od, tube_wall, insert_len, port_dia, sensor_standoff, oring_cs]
        dF_dp = np.zeros((6, 6), dtype=np.float64)
        dF_dp[1, 0] = -0.5  # dF1/dtube_od
        dF_dp[1, 1] = 1.0   # dF1/dtube_wall
        dF_dp[1, 3] = 0.5   # dF1/dport_dia

        dF_dp[2, 2] = -1.0  # dF2/dinsert_len

        dF_dp[3, 2] = 1.0   # dF3/dinsert_len

        dF_dp[4, 0] = -0.5  # dF4/dtube_od
        dF_dp[4, 4] = -1.0  # dF4/dsensor_standoff

        dF_dp[5, 5] = -0.75 # dF5/doring_cs

        # Solve linear system: (dF_dx) * (dx_dp) = - dF_dp
        dx_dp = -np.linalg.solve(dF_dx, dF_dp)
        return dx_dp

    def verify_ift_vs_finite_differences(
        self,
        p_dict: Dict[str, float],
        eps: float = 1e-6
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Cross-checks analytic IFT Jacobian against central finite differences.
        Returns (relative_error, J_ift, J_fd).
        Gate requirement: relative_error <= 10^-6.
        """
        p0 = np.array([float(p_dict.get(name, self._default_param(name))) for name in self.param_names], dtype=np.float64)
        _, x0 = self.solve(p_dict)
        J_ift = self.compute_ift_jacobian(x0, p0)

        n_x = len(self.state_names)
        n_p = len(self.param_names)
        J_fd = np.zeros((n_x, n_p), dtype=np.float64)

        for j in range(n_p):
            p_plus = p0.copy()
            p_plus[j] += eps
            p_plus_dict = {name: float(p_plus[i]) for i, name in enumerate(self.param_names)}
            _, x_plus = self.solve(p_plus_dict, x0=x0)

            p_minus = p0.copy()
            p_minus[j] -= eps
            p_minus_dict = {name: float(p_minus[i]) for i, name in enumerate(self.param_names)}
            _, x_minus = self.solve(p_minus_dict, x0=x0)

            J_fd[:, j] = (x_plus - x_minus) / (2.0 * eps)

        diff_norm = np.linalg.norm(J_ift - J_fd, ord='fro')
        ref_norm = np.linalg.norm(J_fd, ord='fro') + 1e-12
        rel_err = float(diff_norm / ref_norm)

        return rel_err, J_ift, J_fd

    def _default_param(self, name: str) -> float:
        defaults = {
            "tube_od_mm": 32.0,
            "tube_wall_mm": 1.5,
            "insert_length_mm": 50.0,
            "inlet_port_dia_mm": 6.35,  # 1/4" NPT
            "sensor_standoff_mm": 3.0,
            "oring_cs_mm": 1.78         # Standard AS568 cross section
        }
        return defaults.get(name, 1.0)
