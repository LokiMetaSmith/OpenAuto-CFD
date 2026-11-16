"""
firmware_sil
Software-in-the-Loop (SIL) Microcontroller Emulation & Hardware-in-the-Loop Harness.
Provides deterministic virtual-clock stepping and hardware abstraction for fluidic sensors and actuators.
"""

from .virtual_clock import VirtualClockSubcycler

__all__ = ["VirtualClockSubcycler"]
