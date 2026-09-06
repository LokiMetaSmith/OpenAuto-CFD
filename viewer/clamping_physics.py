"""
Clamping Force, Gasket Compression, and Hermetic Sealing Physics Engine
Calculates mechanical clamping physics, silicone & TPU deformation,
and hermetic sealing safety margins for the Daemon Pore cartridge stack.
"""

import math
from typing import Dict, Any

class ClampingPhysicsEngine:
    """Calculates torque-to-clamp force conversion, elastic gasket deformation, and sealing pressure."""

    def __init__(self):
        # Mechanical constants
        self.friction_k = 0.20        # Friction coefficient of dry plastic threads
        self.neck_radius_m = 0.0375   # 75mm thread OD / 2 = 37.5mm
        self.e_silicone_mpa = 1.50    # Young's modulus for 40A cast silicone (MPa)
        self.e_tpu_mpa = 25.0         # Young's modulus for 85A TPU pusher (MPa)

        # Baseline geometry
        self.hex_diam_mm = 60.0
        self.hex_side_mm = self.hex_diam_mm / 2.0
        self.a_hex_gross_mm2 = (3.0 * math.sqrt(3.0) / 2.0) * (self.hex_side_mm ** 2)
        self.a_ports_mm2 = 4.0 * math.pi * (2.0 ** 2)
        self.a_center_mm2 = math.pi * (10.0 ** 2)
        self.a_stack_mm2 = self.a_hex_gross_mm2 - self.a_ports_mm2 - self.a_center_mm2 # ~1973 mm^2

    def calculate_clamping(self,
                           torque_nm: float = 0.50,
                           gasket_thick_mm: float = 1.5875,
                           pusher_thick_mm: float = 4.0,
                           puck_thick_mm: float = 6.0) -> Dict[str, Any]:
        """
        Computes clamping force, compression of all elastic stack components, and sealing status.
        """
        t_nm = max(0.01, float(torque_nm))
        g_thick = max(0.2, float(gasket_thick_mm))
        pp_thick = max(0.5, float(pusher_thick_mm))
        puck_thick = max(0.5, float(puck_thick_mm))

        # 1. Clamping Normal Force F_clamp = Torque / (Friction * r_thread)
        f_clamp_N = t_nm / (self.friction_k * (self.neck_radius_m * 2.0))

        # 2. Average Sealing Pressure P_seal = F_clamp / Area (in MPa = N/mm^2)
        p_seal_mpa = f_clamp_N / self.a_stack_mm2

        # 3. Elastic Compression of Silicone Gaskets (Top & Bottom Gaskets)
        # delta_h = (F * h0) / (E * Area)
        delta_h_silicone_mm = (f_clamp_N * g_thick) / (self.e_silicone_mpa * self.a_stack_mm2)
        strain_silicone = min(0.60, delta_h_silicone_mm / g_thick)

        # 4. Elastic Compression of TPU Plates
        delta_h_tpu_bot_mm = (f_clamp_N * pp_thick) / (self.e_tpu_mpa * self.a_stack_mm2)
        delta_h_tpu_top_mm = (f_clamp_N * puck_thick) / (self.e_tpu_mpa * self.a_stack_mm2)

        # 5. Total Stack Compression
        # Uncompressed Stack Height = Pusher(4) + Plate1(4) + GasketBot(1.6) + PCB(0.8) + GasketTop(1.6) + Plate3(4) + Puck(6) = 22.0mm
        h_uncompressed_mm = pp_thick + 4.0 + g_thick + 0.8 + g_thick + 4.0 + puck_thick
        total_compression_mm = (2.0 * delta_h_silicone_mm) + delta_h_tpu_bot_mm + delta_h_tpu_top_mm
        h_compressed_mm = max(10.0, h_uncompressed_mm - total_compression_mm)

        # 6. Sealing Threshold Classification
        if p_seal_mpa < 0.12:
            status = "LEAK_RISK"
            status_text = "Insufficient Clamping (Leak Risk)"
            status_color = "#ef4444"
            recommendation = "Increase cap torque to at least 0.35 N·m to ensure airtight fluidic seal."
        elif p_seal_mpa <= 0.45:
            status = "OPTIMAL"
            status_text = "Hermetic Seal (Optimal)"
            status_color = "#10b981"
            recommendation = "Sealing pressure is ideal for microfluidic buffer containment without channel deformation."
        else:
            status = "OVERCLAMP_RISK"
            status_text = "Overclamp Warning (Deformation Risk)"
            status_color = "#f59e0b"
            recommendation = "High torque risks stripping plastic threads or occluding microfluidic channels."

        return {
            "torque_nm": round(t_nm, 3),
            "clamp_force_N": round(f_clamp_N, 1),
            "clamp_force_n": round(f_clamp_N, 1),
            "sealing_pressure_mpa": round(p_seal_mpa, 4),
            "sealing_pressure_psi": round(p_seal_mpa * 145.038, 1),
            "silicone_compression_mm": round(delta_h_silicone_mm, 3),
            "silicone_strain_pct": round(strain_silicone * 100.0, 1),
            "tpu_bottom_compression_mm": round(delta_h_tpu_bot_mm, 3),
            "tpu_top_compression_mm": round(delta_h_tpu_top_mm, 3),
            "total_stack_compression_mm": round(total_compression_mm, 3),
            "uncompressed_height_mm": round(h_uncompressed_mm, 2),
            "compressed_height_mm": round(h_compressed_mm, 2),
            "effective_gasket_thickness_mm": round(max(0.1, g_thick - delta_h_silicone_mm), 3),
            "status": status,
            "status_text": status_text,
            "status_color": status_color,
            "recommendation": recommendation
        }

# Singleton instance
_clamping_engine = None

def get_clamping_physics_engine() -> ClampingPhysicsEngine:
    global _clamping_engine
    if _clamping_engine is None:
        _clamping_engine = ClampingPhysicsEngine()
    return _clamping_engine
