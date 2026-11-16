"""
coupling_bus.py
Unified Solver Coupling Bus & Multi-Rate Flux Subcycler (phyz-coupling paradigm).

Couples fast time-domain loops (firmware SIL @ 100-1000 Hz, contact mechanics)
with slow continuum domains (CFD, FEA, thermal diffusion @ 10-50 Hz)
via formal boundary flux accounting and deterministic virtual-clock subcycling.
"""

from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
import numpy as np

from firmware_sil.virtual_clock import VirtualClockSubcycler


@dataclass
class BoundaryFluxLedger:
    """Tracks conserved quantities transferred across domain interfaces."""
    mass_injected_kg: float = 0.0
    mass_permeate_kg: float = 0.0
    mass_reject_kg: float = 0.0
    net_impulse_ns: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    thermal_work_joules: float = 0.0
    electrical_work_joules: float = 0.0

    def compute_mass_conservation_error(self) -> float:
        """Relative mass error = |m_in - (m_permeate + m_reject)| / (|m_in| + eps)."""
        m_out = self.mass_permeate_kg + self.mass_reject_kg
        return abs(self.mass_injected_kg - m_out) / (abs(self.mass_injected_kg) + 1e-9)


class PhyzCouplingBus:
    """
    Unified multi-rate simulation orchestrator.
    Drives subcycled fast domains and continuum solvers on a deterministic virtual clock.
    """

    def __init__(self, dt_slow: float = 0.01, dt_fast: float = 0.001):
        self.dt_slow = dt_slow
        self.dt_fast = dt_fast
        self.clock = VirtualClockSubcycler(dt_base=dt_fast)
        self.ledger = BoundaryFluxLedger()
        self.history: List[Dict[str, Any]] = []

    def reset(self):
        self.clock.reset()
        self.ledger = BoundaryFluxLedger()
        self.history.clear()

    def step_subcycled_coupling(
        self,
        continuum_solver_step: Callable[[float, float, float], Dict[str, Any]],
        fast_controller_step: Callable[[float, float, Dict[str, Any]], Dict[str, Any]],
        total_slow_steps: int = 10
    ) -> BoundaryFluxLedger:
        """
        Executes coupled multi-rate simulation.
        For each slow continuum step (dt_slow), subcycles M = dt_slow / dt_fast times in the fast controller.

        continuum_solver_step: fn(sim_time, dt_slow, mean_inlet_velocity) -> physics_state
            physics_state contains:
                'delta_p_pa': float
                'permeate_flux_kg_s': float
                'reject_flux_kg_s': float
                'wall_traction_n': np.ndarray (3,)
        fast_controller_step: fn(sim_time, dt_fast, physics_state) -> control_state
            control_state contains:
                'inlet_velocity_mps': float
                'pwm_duty': float
                'pump_electrical_power_w': float
        """
        physics_state = {
            "delta_p_pa": 1200.0,
            "permeate_flux_kg_s": 0.025,
            "reject_flux_kg_s": 0.005,
            "wall_traction_n": np.array([0.0, 0.0, -0.15], dtype=np.float64)
        }

        density = 1000.0  # kg/m^3 (water/slurry base)
        inlet_area = 3.167e-5  # m^2 (1/4" NPT)

        for step_idx in range(total_slow_steps):
            t_start = self.clock.sim_time
            velocities_in_subcycles = []

            # 1. Fast Subcycling Loop
            def subcycle_hook(t_curr: float, dt_sub: float):
                ctrl = fast_controller_step(t_curr, dt_sub, physics_state)
                u_in = float(ctrl.get("inlet_velocity_mps", 0.5))
                velocities_in_subcycles.append(u_in)

                # Mass flux during this microstep: dm = rho * u * A * dt
                dm_in = density * u_in * inlet_area * dt_sub
                self.ledger.mass_injected_kg += dm_in

                # Electrical work: dW = P * dt
                p_elec = float(ctrl.get("pump_electrical_power_w", 12.0))
                self.ledger.electrical_work_joules += p_elec * dt_sub

            # Execute M fast steps
            self.clock.subcycle(self.dt_slow, self.dt_fast, subcycle_hook)

            # 2. Advance Continuum Solver using averaged fast inputs
            mean_u_inlet = float(np.mean(velocities_in_subcycles))
            physics_state = continuum_solver_step(t_start, self.dt_slow, mean_u_inlet)

            # Accumulate continuum fluxes
            dm_perm = float(physics_state.get("permeate_flux_kg_s", 0.0)) * self.dt_slow
            dm_rej = float(physics_state.get("reject_flux_kg_s", 0.0)) * self.dt_slow
            self.ledger.mass_permeate_kg += dm_perm
            self.ledger.mass_reject_kg += dm_rej

            traction = np.asarray(physics_state.get("wall_traction_n", np.zeros(3)), dtype=np.float64)
            self.ledger.net_impulse_ns += traction * self.dt_slow

            self.history.append({
                "sim_time": self.clock.sim_time,
                "step_idx": step_idx,
                "mean_u_inlet": mean_u_inlet,
                "delta_p_pa": physics_state.get("delta_p_pa", 0.0),
                "mass_error": self.ledger.compute_mass_conservation_error()
            })

        return self.ledger
