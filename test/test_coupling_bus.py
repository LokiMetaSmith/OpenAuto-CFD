"""
test_coupling_bus.py
Unit tests verifying multi-rate flux subcycling and boundary accounting (phyz-coupling paradigm).
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from optimizer.coupling_bus import PhyzCouplingBus, BoundaryFluxLedger


def test_coupling_bus_multi_rate_subcycling():
    """
    Verify that fast controller subcycles 10 times per slow continuum step,
    and boundary fluxes (mass, impulse, electrical power) are accurately integrated.
    """
    bus = PhyzCouplingBus(dt_slow=0.01, dt_fast=0.001)

    fast_call_count = 0

    def mock_continuum_step(t, dt, mean_u):
        # Balanced mass flow: permeate + reject = inlet flow
        # inlet area = 3.167e-5 m^2, density = 1000 kg/m^3
        total_inflow_kg_s = 1000.0 * mean_u * 3.167e-5
        return {
            "delta_p_pa": 1400.0,
            "permeate_flux_kg_s": 0.85 * total_inflow_kg_s,
            "reject_flux_kg_s": 0.15 * total_inflow_kg_s,
            "wall_traction_n": np.array([0.0, 0.0, -0.2], dtype=np.float64)
        }

    def mock_fast_controller_step(t, dt, phys_state):
        nonlocal fast_call_count
        fast_call_count += 1
        return {
            "inlet_velocity_mps": 0.65,
            "pwm_duty": 0.55,
            "pump_electrical_power_w": 15.0
        }

    ledger = bus.step_subcycled_coupling(
        continuum_solver_step=mock_continuum_step,
        fast_controller_step=mock_fast_controller_step,
        total_slow_steps=10
    )

    # 10 slow steps with 10 fast subcycles each = 100 fast calls
    assert fast_call_count == 100
    assert abs(bus.clock.sim_time - 0.1) < 1e-9

    # Check mass conservation error across the interface
    mass_err = ledger.compute_mass_conservation_error()
    assert mass_err < 1e-6, f"Mass conservation error {mass_err:.2e} exceeded 1e-6"

    # Check impulse integration: 10 steps * 0.01 s * -0.2 N = -0.02 N*s
    assert abs(ledger.net_impulse_ns[2] - (-0.02)) < 1e-6

    # Check electrical work integration: 0.1 s * 15 W = 1.5 Joules
    assert abs(ledger.electrical_work_joules - 1.5) < 1e-6
