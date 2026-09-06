"""
fluidic_cosim_engine.py

End-to-End Fluidics-to-Nanopore Multi-Physics Co-Simulation Engine.
Couples upstream Corkscrew Vortex Particle Separator fluid dynamics (efficiency, flow rate, pressure drop)
directly into the downstream Nanopore Electrophysiology and Transimpedance Amplifier frontend.
"""

import math
import random
from typing import Dict, List, Any, Optional, Tuple

try:
    from .nanopore_engine import NanoporeElectrophysiologyEngine
except (ImportError, ValueError):
    from nanopore_engine import NanoporeElectrophysiologyEngine


class FluidicNanoporeCosimEngine:
    """
    Couples:
    1. Upstream cyclone separator collection efficiency eta and pressure drop Delta P.
    2. Filtrate fluid delivery through microcapillary tubing into Delrin flowcell chamber H1.
    3. Analyte delivery and DNA translocation event arrival rate J_trans.
    4. Debris breakthrough kinetics: uncaptured particulate matter causing stochastic baseline drift,
       current spikes, and pore clogging / blockage saturation when eta < 98%.
    """

    R_HYD_CHAMBER = 1.45e9  # Pa * s / m^3 (fluidic resistance of capillary + chamber)
    PORE_JAM_THRESHOLD_DIAM_NM = 4.0  # Nanopore constriction diameter

    def __init__(self, nano_engine: Optional[NanoporeElectrophysiologyEngine] = None):
        self.nano_engine = nano_engine or NanoporeElectrophysiologyEngine()

    def simulate_cosimulation(
        self,
        filter_efficiency_pct: float = 99.96,
        pressure_drop_psi: float = 0.42,
        bias_voltage_mv: float = 100.0,
        pore_diameter_nm: float = 4.0,
        inlet_analyte_pico_molar: float = 250.0,
        inlet_debris_particles_per_ml: float = 1.0e6,
        duration_ms: float = 4.0,
        sample_rate_khz: float = 250.0
    ) -> Dict[str, Any]:
        """
        Executes coupled multi-physics simulation connecting vortex separation to nanopore stream.
        """
        eta = max(50.0, min(99.999, float(filter_efficiency_pct))) / 100.0
        dp_pa = max(100.0, float(pressure_drop_psi) * 6894.76)
        v_bias = float(bias_voltage_mv)
        d_pore = float(pore_diameter_nm)

        # 1. Microfluidic Flow Delivery
        # Volumetric flow rate Q = Delta P / R_hyd (m^3/s -> uL/min)
        flow_rate_m3_s = dp_pa / self.R_HYD_CHAMBER
        flow_rate_ul_min = round(flow_rate_m3_s * 1e9 * 60.0, 2)  # Typically 10 - 150 uL/min

        # 2. Particulate Debris Breakthrough vs Analyte Transmission
        # Target DNA analyte passes through into filtrate
        analyte_transmission = 0.94  # 94% analyte recovery
        c_analyte_pm = inlet_analyte_pico_molar * analyte_transmission

        # Debris breakthrough fraction = (1 - eta)
        debris_breakthrough_pct = round((1.0 - eta) * 100.0, 4)
        c_debris_out_ml = inlet_debris_particles_per_ml * (1.0 - eta)

        # Translocation capture rate scales with analyte concentration and electric field
        base_rate_hz = 3000.0
        flow_factor = math.sqrt(max(0.1, flow_rate_ul_min / 50.0))
        bias_factor = max(0.2, v_bias / 100.0)
        conc_factor = max(0.1, c_analyte_pm / 250.0)
        eff_event_rate_hz = round(base_rate_hz * flow_factor * bias_factor * conc_factor, 1)

        # Debris clogging arrival rate
        # When eta is high (99.96%), lambda_clog is negligible (< 0.05 Hz)
        # When eta drops (< 95%), lambda_clog surges (> 50 Hz)
        clog_factor = (1.0 - eta) / 0.0004
        clog_rate_hz = round(0.04 * (clog_factor ** 1.8) * (flow_rate_ul_min / 50.0), 2)
        clogging_risk = "NEGLIGIBLE (Ultra-Pure)" if eta >= 0.999 else ("LOW (Clean)" if eta >= 0.98 else ("MODERATE (Intermittent Jams)" if eta >= 0.93 else "CRITICAL (Pore Saturated)"))

        # 3. Simulate Base Electrophysiology Stream
        stream_res = self.nano_engine.simulate_translocation_stream(
            pore_diameter_nm=d_pore,
            bias_voltage_mv=v_bias,
            target_event_rate_hz=eff_event_rate_hz,
            duration_ms=duration_ms,
            sample_rate_khz=sample_rate_khz
        )

        i0_na = stream_res["baseline_current_na"]
        current_na = list(stream_res["current_na"])
        time_us = stream_res["time_us"]
        n_samples = len(current_na)

        # 4. Inject Debris Clogging & Jamming Events if Breakthrough Exists
        debris_events = []
        expected_clogs = (duration_ms * 1e-3) * clog_rate_hz
        num_clogs = 1 if (expected_clogs > 0.3 and random.random() < expected_clogs) else (int(expected_clogs) if expected_clogs >= 1.0 else 0)

        if eta < 0.95 and num_clogs == 0 and random.random() < 0.65:
            num_clogs = 1  # Ensure visible clogging demo when filter efficiency is degraded

        for c_idx in range(num_clogs):
            jam_us = random.uniform(350.0, 950.0)
            t_start_us = random.uniform(200.0, max(300.0, duration_ms * 1000.0 - jam_us - 100.0))
            t_end_us = t_start_us + jam_us

            debris_events.append({
                "id": c_idx + 1,
                "type": "debris_blockade",
                "start_us": round(t_start_us, 1),
                "duration_us": round(jam_us, 1),
                "severity": "CRITICAL_JAM",
                "message": "Unfiltered micro-particulate jammed in nanopore constriction"
            })

            # Force current to near-zero jam level with recovery
            for idx, t in enumerate(time_us):
                if t_start_us <= t <= t_end_us:
                    jam_depth = 0.94 + random.gauss(0.0, 0.02)
                    jam_depth = max(0.85, min(0.98, jam_depth))
                    current_na[idx] = round(i0_na * (1.0 - jam_depth) + random.gauss(0.0, 0.015), 4)

        return {
            "upstream_filter": {
                "efficiency_percent": round(eta * 100.0, 3),
                "pressure_drop_psi": round(pressure_drop_psi, 3),
                "pressure_drop_pa": round(dp_pa, 1),
                "filtrate_flow_rate_ul_min": flow_rate_ul_min,
                "debris_breakthrough_percent": debris_breakthrough_pct,
                "filtrate_debris_concentration_ml": round(c_debris_out_ml, 1)
            },
            "nanopore_coupling": {
                "pore_diameter_nm": d_pore,
                "bias_voltage_mv": v_bias,
                "effective_translocation_rate_hz": eff_event_rate_hz,
                "debris_clogging_rate_hz": clog_rate_hz,
                "clogging_risk": clogging_risk,
                "is_pore_clogged": len(debris_events) > 0,
                "clogging_events_detected": len(debris_events),
                "clogging_events": debris_events
            },
            "stream": {
                "baseline_current_na": i0_na,
                "rms_noise_pa": stream_res["rms_noise_pa"],
                "tia_bandwidth_khz": stream_res["tia_bandwidth_khz"],
                "snr_db": stream_res["snr_db"],
                "dna_translocations_detected": stream_res["events_detected"],
                "dna_events": stream_res["events"],
                "time_us": time_us,
                "current_na": current_na
            }
        }
