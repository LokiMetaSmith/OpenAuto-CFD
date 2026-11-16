"""
virtual_clock.py
Deterministic Virtual Clock Subcycler for Software-in-the-Loop (SIL) simulation.

Completely decoupled from host OS wall-clock time and system threads.
Guarantees 100% reproducible multi-rate subcycling.
"""

from typing import List, Callable, Tuple


class VirtualClockSubcycler:
    """
    Manages deterministic simulation time for fast control loops (firmware)
    and slow continuum solvers (CFD, FEA, thermal).
    """

    def __init__(self, dt_base: float = 1e-4):
        self.sim_time: float = 0.0
        self.dt_base: float = dt_base
        self.tick_count: int = 0
        self._subscribers: List[Tuple[float, Callable[[float, float], None]]] = []

    def reset(self, sim_time: float = 0.0):
        self.sim_time = sim_time
        self.tick_count = 0

    def step(self, dt: float) -> float:
        """Advance the master virtual simulation clock by dt."""
        self.sim_time += dt
        self.tick_count += 1
        return self.sim_time

    def subcycle(self, dt_slow: float, dt_fast: float, fast_callback: Callable[[float, float], None]) -> int:
        """
        Executes integer number of fast subcycles within one slow continuum step.
        dt_slow: Continuum time step (e.g. 0.01 s = 10 ms)
        dt_fast: Fast controller time step (e.g. 0.001 s = 1 ms, 1000 Hz)
        fast_callback: Function called each fast step with (current_time, dt_fast)
        Returns: total number of subcycles executed.
        """
        if dt_fast <= 0 or dt_slow <= 0:
            raise ValueError("Time steps must be strictly positive")

        n_subcycles = max(1, int(round(dt_slow / dt_fast)))
        actual_dt_fast = dt_slow / n_subcycles

        for _ in range(n_subcycles):
            fast_callback(self.sim_time, actual_dt_fast)
            self.sim_time += actual_dt_fast
            self.tick_count += 1

        return n_subcycles
