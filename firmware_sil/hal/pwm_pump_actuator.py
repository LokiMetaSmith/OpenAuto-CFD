"""
pwm_pump_actuator.py
Hardware Abstraction Layer for 20 kHz PWM Slurry Feed Pump Actuator.
Translates 12-bit microcontroller PWM registers into physical volumetric flow and boundary velocity.
"""


class PWMPumpActuatorEmulator:
    """
    20 kHz PWM Feed Pump Actuator.
    Resolution: 12-bit (0 - 4095 counts).
    Maps duty cycle 0.0 to 1.0 to volumetric flow rate Q_max (L/min) and inlet face velocity (m/s).
    """

    PWM_FREQ_HZ = 20000.0
    MAX_COUNTS = 4095

    def __init__(self, max_flow_lpm: float = 2.0, inlet_area_m2: float = 3.167e-5):
        self.max_flow_lpm = max_flow_lpm
        self.inlet_area_m2 = inlet_area_m2
        self._raw_counts: int = 0
        self._duty_cycle: float = 0.0

    def set_pwm_counts(self, counts: int) -> None:
        """Sets raw 12-bit PWM register value from microcontroller."""
        self._raw_counts = min(max(int(counts), 0), self.MAX_COUNTS)
        self._duty_cycle = self._raw_counts / float(self.MAX_COUNTS)

    def set_duty_cycle(self, duty: float) -> None:
        """Sets duty cycle directly (0.0 to 1.0)."""
        self._duty_cycle = min(max(float(duty), 0.0), 1.0)
        self._raw_counts = int(round(self._duty_cycle * self.MAX_COUNTS))

    @property
    def duty_cycle(self) -> float:
        return self._duty_cycle

    def get_flow_rate_lpm(self) -> float:
        """Returns volumetric flow rate in Liters/minute."""
        return self._duty_cycle * self.max_flow_lpm

    def get_inlet_velocity_mps(self) -> float:
        """
        Calculates physical boundary velocity u_inlet = Q / A in m/s.
        1 L/min = (1e-3 / 60) m^3/s
        """
        flow_m3_s = (self.get_flow_rate_lpm() * 1e-3) / 60.0
        if self.inlet_area_m2 <= 1e-9:
            return 0.0
        return flow_m3_s / self.inlet_area_m2
