"""
adc_honeywell_dp.py
Hardware Abstraction Layer (HAL) for Honeywell HSC-DIP-001PSI Differential Pressure Sensor.

Models I2C telemetry, 14-bit ADC quantization, transfer functions, and status registers
extracted directly from wiring/filter_interconnect.yaml.
"""

from typing import Tuple


class HoneywellDPSensorEmulator:
    """
    Honeywell TruStability HSC Series Differential Pressure Sensor Emulator.
    Part Number: HSC-DIP-001PSI
    Full Scale: -1.0 to +1.0 psi (or 0.0 to 1.0 psi differential)
    Operating range: 0 to 6894.76 Pa (1.0 psi)
    I2C Address: 0x28
    ADC: 14-bit (16384 counts), 10% to 90% calibration curve:
        Count_min = 1638 (10% of 2^14)
        Count_max = 14745 (90% of 2^14)
    """

    I2C_ADDRESS = 0x28
    P_MIN_PA = 0.0
    P_MAX_PA = 6894.76  # 1 psi in Pascals
    COUNT_MIN = 1638
    COUNT_MAX = 14745

    def __init__(self, noise_std_pa: float = 0.0):
        self.noise_std_pa = noise_std_pa
        self._current_dp_pa: float = 0.0
        self._status_bits: int = 0  # 00 = Normal, 01 = Command Mode, 10 = Stale Data, 11 = Diagnostic

    def update_physics(self, delta_p_pa: float) -> None:
        """Updates the physical differential pressure input from CFD or surrogate."""
        self._current_dp_pa = max(0.0, float(delta_p_pa))

    def read_raw_i2c_bytes(self) -> Tuple[int, int, int, int]:
        """
        Simulates 4-byte I2C read from address 0x28:
            Byte 0: Status (bits 7:6) + Bridge Data MSB (bits 5:0)
            Byte 1: Bridge Data LSB (bits 7:0)
            Byte 2: Temperature Data MSB (bits 7:0)
            Byte 3: Temperature Data LSB (bits 7:5) + Diagnostic (bits 4:0)
        """
        # Linear conversion from Pa to counts
        clamped_p = min(max(self._current_dp_pa, self.P_MIN_PA), self.P_MAX_PA)
        fraction = (clamped_p - self.P_MIN_PA) / (self.P_MAX_PA - self.P_MIN_PA)
        count = int(round(self.COUNT_MIN + fraction * (self.COUNT_MAX - self.COUNT_MIN)))
        count = min(max(count, 0), 16383)

        b0 = ((self._status_bits & 0x03) << 6) | ((count >> 8) & 0x3F)
        b1 = count & 0xFF

        # Temperature ~25 C (counts ~1024)
        b2 = (1024 >> 3) & 0xFF
        b3 = (1024 & 0x07) << 5

        return b0, b1, b2, b3

    def read_pressure_pa(self) -> float:
        """Decodes raw I2C bytes back into calibrated engineering pressure in Pascals."""
        b0, b1, _, _ = self.read_raw_i2c_bytes()
        status = (b0 >> 6) & 0x03
        if status != 0:
            return 0.0

        count = ((b0 & 0x3F) << 8) | b1
        fraction = (count - self.COUNT_MIN) / (self.COUNT_MAX - self.COUNT_MIN)
        pressure_pa = self.P_MIN_PA + fraction * (self.P_MAX_PA - self.P_MIN_PA)
        return float(pressure_pa)
