"""
test_firmware_sil.py
Unit tests verifying Software-in-the-Loop (SIL) Microcontroller Emulation:
  - Deterministic Virtual Clock Subcycler
  - Honeywell HSC-DIP-001PSI DP Sensor I2C emulator
  - 20 kHz PWM Slurry Pump Actuator emulator
  - Closed-loop Pressure Regulating PID controller at >= 100 Hz
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from firmware_sil.virtual_clock import VirtualClockSubcycler
from firmware_sil.hal.adc_honeywell_dp import HoneywellDPSensorEmulator
from firmware_sil.hal.pwm_pump_actuator import PWMPumpActuatorEmulator
from firmware_sil.controllers.native_pid_filter import PressureRegulatingPIDController


def test_virtual_clock_deterministic_stepping():
    """Verify that virtual clock steps deterministically without wall-clock sleep."""
    clock = VirtualClockSubcycler(dt_base=0.001)

    # 10 continuum steps of 0.01 s with 10 subcycles of 0.001 s
    subcycle_count = 0

    def subcycle_callback(t_sim, dt_sub):
        nonlocal subcycle_count
        subcycle_count += 1
        assert abs(dt_sub - 0.001) < 1e-9

    for _ in range(10):
        n = clock.subcycle(dt_slow=0.01, dt_fast=0.001, fast_callback=subcycle_callback)
        assert n == 10

    assert subcycle_count == 100
    assert abs(clock.sim_time - 0.1) < 1e-9


def test_honeywell_dp_sensor_i2c():
    """Verify Honeywell HSC-DIP-001PSI I2C quantization and pressure readback."""
    sensor = HoneywellDPSensorEmulator()

    # Test at zero pressure
    sensor.update_physics(0.0)
    p_zero = sensor.read_pressure_pa()
    assert abs(p_zero - 0.0) < 1.0

    # Test at midpoint (0.5 psi ~ 3447 Pa)
    sensor.update_physics(3447.38)
    p_mid = sensor.read_pressure_pa()
    assert abs(p_mid - 3447.38) < 2.0  # Within 14-bit quantization error

    # Test at full scale (1.0 psi = 6894.76 Pa)
    sensor.update_physics(6894.76)
    p_full = sensor.read_pressure_pa()
    assert abs(p_full - 6894.76) < 2.0


def test_pwm_pump_actuator():
    """Verify 20 kHz PWM translation to inlet velocity."""
    actuator = PWMPumpActuatorEmulator(max_flow_lpm=2.0, inlet_area_m2=3.167e-5)

    # 50% duty cycle
    actuator.set_duty_cycle(0.5)
    assert abs(actuator.duty_cycle - 0.5) < 1e-6
    assert abs(actuator.get_flow_rate_lpm() - 1.0) < 1e-6

    # Flow velocity = (1.0 L/min * 1e-3 / 60) / 3.167e-5 m^2 = 0.526 m/s
    u_in = actuator.get_inlet_velocity_mps()
    assert abs(u_in - 0.526) < 0.01


def test_closed_loop_pid_pressure_regulation():
    """
    Verify closed-loop pressure regulation running at >= 100 Hz in virtual time.
    Controller adjusts pump PWM to drive differential pressure toward target setpoint 1500 Pa.
    """
    clock = VirtualClockSubcycler(dt_base=0.005)  # 200 Hz
    sensor = HoneywellDPSensorEmulator()
    actuator = PWMPumpActuatorEmulator(max_flow_lpm=2.0)
    pid = PressureRegulatingPIDController(target_pressure_pa=1500.0)

    # Start with initial state: pressure = 800 Pa (below target)
    curr_pressure = 800.0

    # Run for 2.5 virtual seconds (500 steps at 200 Hz)
    dt_ctrl = 0.005
    for _ in range(500):
        sensor.update_physics(curr_pressure)
        measured_p = sensor.read_pressure_pa()

        duty, pwm_counts = pid.update(measured_p, dt=dt_ctrl)
        actuator.set_duty_cycle(duty)

        # First-order simulated plant response: higher duty cycle increases differential pressure
        plant_target = actuator.duty_cycle * 2500.0
        curr_pressure += (plant_target - curr_pressure) * 0.05
        clock.step(dt_ctrl)

    # After 2.5 seconds, update sensor with final pressure and verify within 25 Pa of 1500 Pa
    sensor.update_physics(curr_pressure)
    final_p = sensor.read_pressure_pa()
    assert abs(final_p - 1500.0) < 25.0, f"Final regulated pressure {final_p:.1f} Pa not within 25 Pa of 1500 Pa"
