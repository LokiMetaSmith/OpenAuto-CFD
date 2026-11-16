"""
native_pid_filter.py
Pure Python Digital Twin PID Controller for Corkscrew Filter Pressure Regulation.

Runs in deterministic virtual simulation time at >= 100 Hz.
Regulates differential pressure drop across the separation channel by modulating slurry pump PWM.
"""

from typing import Tuple


class PressureRegulatingPIDController:
    """
    Closed-loop discrete PID controller running inside the virtual-clock subcycler.
    Regulates differential pressure ΔP to a target setpoint (e.g. 1500 Pa)
    to maintain optimal particle separation velocity without clogging or blow-by.
    """

    def __init__(
        self,
        target_pressure_pa: float = 1500.0,
        kp: float = 0.001,
        ki: float = 0.01,
        kd: float = 0.0001,
        control_freq_hz: float = 100.0,
        output_min: float = 0.1,
        output_max: float = 1.0
    ):
        self.target_pressure_pa = target_pressure_pa
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.control_freq_hz = control_freq_hz
        self.output_min = output_min
        self.output_max = output_max

        self._integral: float = 0.0
        self._prev_error: float = 0.0
        self._last_control_output: float = output_min

    def reset(self):
        self._integral = 0.0
        self._prev_error = 0.0
        self._last_control_output = self.output_min

    def update(self, measured_pressure_pa: float, dt: float) -> Tuple[float, int]:
        """
        Executes one discrete PID step.
        measured_pressure_pa: Reading from Honeywell DPSensorEmulator
        dt: Virtual subcycle step size (s)
        Returns:
            duty_cycle: float in [output_min, output_max]
            pwm_counts: int in [0, 4095]
        """
        error = self.target_pressure_pa - measured_pressure_pa

        # Proportional term
        p_term = self.kp * error

        # Integral term with anti-windup clamping
        self._integral += error * dt
        self._integral = min(max(self._integral, -500.0), 500.0)
        i_term = self.ki * self._integral

        # Derivative term
        d_term = self.kd * (error - self._prev_error) / (dt + 1e-9)
        self._prev_error = error

        # Control output (duty cycle)
        raw_output = p_term + i_term + d_term + 0.5  # Baseline offset
        duty = min(max(raw_output, self.output_min), self.output_max)
        self._last_control_output = duty

        pwm_counts = int(round(duty * 4095))
        pwm_counts = min(max(pwm_counts, 0), 4095)

        return duty, pwm_counts
