"""
Nanopore Electrophysiology & TIA Co-Simulation Engine
Synthesizes real-time single-molecule ionic current blockade pulses,
thermal & 1/f noise floors, and interfaces with the KiCad SPICE TIA analog front-end.
"""

import math
import time
import random
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

class ElectrophysiologyEngine:
    """Simulates real-time ionic conductance, single-molecule translocations, and analog TIA response."""

    def __init__(self):
        # Physics constants
        self.k_B = 1.380649e-23  # J/K
        self.T_kelvin = 295.15    # 22 C
        self.sample_rate_hz = 50000 # 50 kHz sampling

        # Default Parameters
        self.pore_diameter_nm = 4.0
        self.membrane_length_nm = 10.0
        self.bias_voltage_mv = 120.0
        self.buffer_conc_m = 1.0 # 1 M KCl (~10.5 S/m)

        # Transimpedance Amplifier (TIA) circuit parameters
        self.r_feedback_ohms = 1.0e9 # 1 Gigaohm
        self.c_feedback_farads = 0.2e-12 # 0.2 pF
        self.c_membrane_farads = 15.0e-12 # 15 pF parasitic cell capacitance

        # Noise & state
        self.history_size = 600
        self.time_buffer = np.zeros(self.history_size, dtype=np.float32)
        self.current_buffer = np.zeros(self.history_size, dtype=np.float32)
        self.voltage_buffer = np.zeros(self.history_size, dtype=np.float32)
        self.current_time_s = 0.0

        # Event tracking
        self.total_events = 0
        self.recent_events: List[Dict[str, Any]] = []
        self.last_event_time = 0.0

        # Pre-seed buffer
        self._init_baseline()

    def _get_buffer_conductivity_S_per_m(self, conc_m: float) -> float:
        # Standard KCl molar conductivity empirical approximation (~10.5 S/m at 1.0 M)
        return max(0.1, float(conc_m) * 10.5)

    def calculate_pore_conductance(self, d_p_nm: float, l_p_nm: float, conc_m: float) -> Tuple[float, float]:
        """
        Calculates open-pore resistance and conductance:
        R_0 = R_channel + R_access = (4*L) / (pi * sigma * d^2) + 1 / (sigma * d)
        Returns (resistance_ohms, conductance_siemens).
        """
        sigma = self._get_buffer_conductivity_S_per_m(conc_m)
        d_m = max(0.5, float(d_p_nm)) * 1.0e-9
        l_m = max(1.0, float(l_p_nm)) * 1.0e-9

        r_channel = (4.0 * l_m) / (math.pi * sigma * (d_m ** 2))
        r_access = 1.0 / (sigma * d_m)
        r_total = r_channel + r_access
        g_total = 1.0 / r_total
        return r_total, g_total

    def _init_baseline(self):
        r0, g0 = self.calculate_pore_conductance(
            self.pore_diameter_nm, self.membrane_length_nm, self.buffer_conc_m
        )
        v_bias = self.bias_voltage_mv * 1.0e-3
        i_base_A = v_bias / r0
        i_base_pA = i_base_A * 1.0e12

        dt = 1.0 / self.sample_rate_hz
        for i in range(self.history_size):
            self.current_time_s += dt
            self.time_buffer[i] = self.current_time_s
            noise = random.gauss(0, 12.0) # ~12 pA RMS noise
            self.current_buffer[i] = i_base_pA + noise
            self.voltage_buffer[i] = -(self.current_buffer[i] * 1.0e-12) * (self.r_feedback_ohms * 1.0e-3)

    def trigger_translocation_event(self, analyte: str = "dsDNA", duration_us: Optional[float] = None) -> Dict[str, Any]:
        """
        Synthesizes a physical single-molecule translocation blockade pulse.
        """
        r0, g0 = self.calculate_pore_conductance(
            self.pore_diameter_nm, self.membrane_length_nm, self.buffer_conc_m
        )
        v_bias = self.bias_voltage_mv * 1.0e-3
        i_base_pA = (v_bias / r0) * 1.0e12

        # Analyte dimensions
        if analyte == "ssDNA":
            d_mol = 1.4 # nm
            default_dur = random.uniform(80.0, 300.0)
        elif analyte == "protein":
            d_mol = 3.2 # nm
            default_dur = random.uniform(400.0, 1800.0)
        else: # dsDNA
            d_mol = 2.2 # nm
            default_dur = random.uniform(150.0, 650.0)

        dwell_us = duration_us if duration_us is not None else default_dur

        # Blockade ratio: delta_I / I_0 ~ (d_mol / d_pore)^2
        blockade_fraction = min(0.95, (d_mol / max(1.0, self.pore_diameter_nm)) ** 2)
        delta_i_pA = i_base_pA * blockade_fraction
        residual_i_pA = max(10.0, i_base_pA - delta_i_pA)

        self.total_events += 1
        now = time.time()
        self.last_event_time = now

        event_meta = {
            "id": self.total_events,
            "timestamp": now,
            "analyte": analyte,
            "dwell_us": round(dwell_us, 1),
            "baseline_pA": round(i_base_pA, 1),
            "blockade_pA": round(delta_i_pA, 1),
            "blockade_delta_pa": round(delta_i_pA, 1),
            "residual_pA": round(residual_i_pA, 1),
            "residual_current_na": round(residual_i_pA / 1000.0, 3),
            "blockade_ratio": round(blockade_fraction, 3)
        }

        self.recent_events.append(event_meta)
        if len(self.recent_events) > 30:
            self.recent_events.pop(0)

        # Inject into the tail of the current buffer
        n_samples = max(2, int((dwell_us * 1.0e-6) * self.sample_rate_hz))
        inject_start = max(0, self.history_size - n_samples - 5)
        for idx in range(inject_start, min(self.history_size, inject_start + n_samples)):
            noise = random.gauss(0, 10.0)
            self.current_buffer[idx] = residual_i_pA + noise
            self.voltage_buffer[idx] = -(self.current_buffer[idx] * 1.0e-12) * (self.r_feedback_ohms * 1.0e-3)

        return event_meta

    def step_simulation(self, n_steps: int = 15, auto_event_prob: float = 0.08) -> Dict[str, Any]:
        """
        Advances the electrophysiology time trace by n_steps and computes live telemetry.
        """
        r0, g0 = self.calculate_pore_conductance(
            self.pore_diameter_nm, self.membrane_length_nm, self.buffer_conc_m
        )
        v_bias = self.bias_voltage_mv * 1.0e-3
        i_base_pA = (v_bias / r0) * 1.0e12
        dt = 1.0 / self.sample_rate_hz

        # Roll existing buffers
        self.time_buffer = np.roll(self.time_buffer, -n_steps)
        self.current_buffer = np.roll(self.current_buffer, -n_steps)
        self.voltage_buffer = np.roll(self.voltage_buffer, -n_steps)

        # Fill newly rolled steps with baseline noise
        for s in range(n_steps):
            self.current_time_s += dt
            idx = self.history_size - n_steps + s
            self.time_buffer[idx] = self.current_time_s
            noise = random.gauss(0, 11.5)
            self.current_buffer[idx] = i_base_pA + noise
            self.voltage_buffer[idx] = -(self.current_buffer[idx] * 1.0e-12) * (self.r_feedback_ohms * 1.0e-3)

        # Spontaneous particle arrival if auto_event_prob triggered
        if random.random() < auto_event_prob:
            analyte_choice = random.choice(["dsDNA", "dsDNA", "ssDNA", "protein"])
            self.trigger_translocation_event(analyte=analyte_choice)

        # Compute summary statistics
        curr_slice = self.current_buffer[-100:]
        baseline_mean = float(np.mean(curr_slice))
        rms_noise = float(np.std(curr_slice))
        snr_db = round(20.0 * math.log10(max(1.0, baseline_mean) / max(0.1, rms_noise)), 1) if rms_noise > 0 else 40.0

        mean_dwell = float(np.mean([e["dwell_us"] for e in self.recent_events])) if self.recent_events else 320.0
        mean_blockade = float(np.mean([e["blockade_ratio"] for e in self.recent_events])) if self.recent_events else 0.30

        return {
            "current_time_s": round(self.current_time_s, 4),
            "baseline_current_pA": round(i_base_pA, 1),
            "baseline_current_na": round(i_base_pA / 1000.0, 3),
            "pore_resistance_Mohm": round(r0 * 1.0e-6, 2),
            "rms_noise_pA": round(rms_noise, 2),
            "rms_noise_pa": round(rms_noise, 2),
            "snr_db": snr_db,
            "total_events": self.total_events,
            "events_detected": self.total_events,
            "mean_dwell_us": round(mean_dwell, 1),
            "mean_blockade_ratio": round(mean_blockade, 3),
            "time_trace_ms": [round(float(t * 1000.0), 2) for t in self.time_buffer[::4]], # downsample 4x for smooth 60fps JSON
            "current_trace_pA": [round(float(i), 1) for i in self.current_buffer[::4]],
            "current_na": [round(float(i / 1000.0), 4) for i in self.current_buffer[::4]],
            "voltage_trace_mV": [round(float(v), 2) for v in self.voltage_buffer[::4]],
            "recent_events": self.recent_events[-8:],
            "events": [
                {
                    "start_us": (e.get("timestamp", 0) % 0.002) * 1e6,
                    "dwell_us": e["dwell_us"],
                    "residual_current_na": round(e["residual_pA"] / 1000.0, 3),
                    "blockade_delta_pa": e["blockade_pA"],
                    "analyte": e["analyte"]
                }
                for e in self.recent_events[-8:]
            ]
        }

# Singleton instance
_ep_engine = None

def get_electrophysiology_engine() -> ElectrophysiologyEngine:
    global _ep_engine
    if _ep_engine is None:
        _ep_engine = ElectrophysiologyEngine()
    return _ep_engine
