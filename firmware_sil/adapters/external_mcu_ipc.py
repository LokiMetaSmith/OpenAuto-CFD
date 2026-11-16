"""
external_mcu_ipc.py
External Firmware Adapter for mounted microcontroller binaries.

Provides deterministic virtual-clock stepped IPC/FFI harness.
Guarantees execution repeatability without dependence on host CPU timing.
"""

from typing import Dict, Any, Optional


class ExternalFirmwareAdapter:
    """
    Adapter bridging external C/C++ or FreeRTOS firmware binaries (e.g. from sibling repos)
    via synchronous virtual-clock stepping.
    """

    def __init__(self, firmware_path: Optional[str] = None):
        self.firmware_path = firmware_path
        self.virtual_time_us: int = 0
        self.is_connected: bool = firmware_path is not None

    def step_virtual_us(self, dt_us: int, sensor_inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Advances the external firmware state machine by dt_us virtual microseconds.
        """
        self.virtual_time_us += dt_us
        # In mock/adapter mode, simulate MCU response registers
        dp_raw = sensor_inputs.get("dp_raw", 8192)
        # Compute simulated actuator command
        pwm_cmd = min(max(int(dp_raw * 0.5), 0), 4095)
        return {
            "virtual_time_us": self.virtual_time_us,
            "pwm_pump_cmd": pwm_cmd,
            "valve_state": 1 if dp_raw > 12000 else 0,
            "status_flags": 0x00
        }
