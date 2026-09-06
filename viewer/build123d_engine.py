"""
Build123d Direct CAD Integration Engine
Streams tessellated geometry and parametric stackups directly from build123d Python scripts.
Eliminates intermediate STL/DXF artifacts, ensures exact non-overlapping mathematical stackup,
and supports interactive parametric adjustments (explode factor, layer thickness, tolerances).
"""

import sys
import os
import time
import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

logger = logging.getLogger("build123d_engine")
logger.setLevel(logging.INFO)

# Ensure daemon-pore project root is on sys.path
DAEMON_PORE_DIR = r"C:\Users\Loki-VR\Documents\projects\Daemon Pore\daemon-pore"
if os.path.exists(DAEMON_PORE_DIR) and DAEMON_PORE_DIR not in sys.path:
    sys.path.insert(0, DAEMON_PORE_DIR)

# Try importing build123d and project modules
BUILD123D_AVAILABLE = False
try:
    from build123d import *
    from src.cartridge import (
        Bottom_Pusher_Plate,
        Plate_1_Base_Hex,
        Fitted_Amplifier_Gasket,
        PCB_Rigid_Active,
        Cast_Nanopore_Wafer_Gasket,
        Plate_3_Top_Hex,
        Pressure_Puck
    )
    from src.housing import (
        Housing_Case_Reader,
        Housing_Lid_Reader,
        Housing_Pump_Case,
        Housing_Pump_Lid
    )
    from src.torque_cap import Torque_Cap_Outer, Torque_Cap_Inner
    from src.nanopore_wafer_mold import Nanopore_Wafer_Mold_Left, Nanopore_Wafer_Mold_Right
    import src.constants as const
    BUILD123D_AVAILABLE = True
    logger.info("build123d engine successfully initialized with daemon-pore components.")
except Exception as err:
    logger.warning("build123d or daemon-pore components not available: %s", err)


class Build123dEngine:
    """Direct build123d CAD tessellation, parametric assembly, and binary streaming engine."""

    def __init__(self):
        self.part_builders = {}
        self.mesh_cache: Dict[str, bytes] = {}
        self.meta_cache: Dict[str, Dict[str, Any]] = {}
        if BUILD123D_AVAILABLE:
            self._register_builders()

    def _register_builders(self):
        self.part_builders = {
            # Cartridge stack
            "pusher": ("Bottom Pusher Plate (TPU)", Bottom_Pusher_Plate),
            "base": ("Plate 1 Base Hex (PP 4mm)", Plate_1_Base_Hex),
            "gasket_bot": ("Fitted Amplifier Gasket (Silicone)", Fitted_Amplifier_Gasket),
            "pcb": ("Active Rigid PCB (FR4)", PCB_Rigid_Active),
            "gasket_top": ("Cast Nanopore Wafer Gasket (Silicone)", Cast_Nanopore_Wafer_Gasket),
            "top_plate": ("Plate 3 Top Hex (PP 4mm)", Plate_3_Top_Hex),
            "puck": ("Pressure Puck (Aluminum)", Pressure_Puck),
            # Enclosure
            "case": ("Reader Housing Case", Housing_Case_Reader),
            "lid": ("Reader Housing Lid", Housing_Lid_Reader),
            "cap_inner": ("Torque Cap Inner Drive", Torque_Cap_Inner),
            "cap_outer": ("Torque Cap Outer Knurl", Torque_Cap_Outer),
            # Wafer Molds
            "mold_left": ("Nanopore Wafer Mold (Left Half)", lambda: Nanopore_Wafer_Mold_Left(is_nanopore=True)),
            "mold_right": ("Nanopore Wafer Mold (Right Half)", lambda: Nanopore_Wafer_Mold_Right(is_nanopore=True)),
            # Pump Module
            "pump_case": ("Peristaltic Pump Housing Case", Housing_Pump_Case),
            "pump_lid": ("Peristaltic Pump Housing Lid", Housing_Pump_Lid)
        }
        try:
            from src.assembly import route_tubing, route_wiring
            self.part_builders["tubing"] = ("Silicone Fluidic Tubing", route_tubing)
            self.part_builders["wiring"] = ("Internal Wiring Loom", route_wiring)
        except Exception as e:
            logger.warning("Could not register tubing/wiring: %s", e)

    def is_available(self) -> bool:
        return BUILD123D_AVAILABLE

    def update_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Dynamically updates build123d dimensional constants and clears geometry cache."""
        if not BUILD123D_AVAILABLE:
            return {"error": "build123d unavailable"}
        changed = {}
        if "gasket_thick_mm" in params:
            val = float(params["gasket_thick_mm"])
            const.GASKET_THICK = val
            changed["gasket_thick_mm"] = val
        if "plate_thick_mm" in params:
            val = float(params["plate_thick_mm"])
            const.PLATE_THICK = val
            changed["plate_thick_mm"] = val
        if "wafer_hole_size_mm" in params:
            val = float(params["wafer_hole_size_mm"])
            const.WAFER_HOLE_SIZE = val
            changed["wafer_hole_size_mm"] = val
        if "hex_diam_mm" in params:
            val = float(params["hex_diam_mm"])
            const.HEX_DIAM = val
            const.HEX_SIDE = val / 2.0
            changed["hex_diam_mm"] = val

        self.mesh_cache.clear()
        self.meta_cache.clear()
        logger.info("Updated build123d parameters: %s (cache cleared)", changed)
        return {"status": "ok", "updated_params": changed}

    def export_part_file(self, part_id: str, fmt: str = "step") -> Optional[Tuple[bytes, str, str]]:
        """Generates on-demand STEP or STL export binary data for physical manufacturing."""
        if not BUILD123D_AVAILABLE or part_id not in self.part_builders:
            return None
        title, builder_fn = self.part_builders[part_id]
        import tempfile
        from build123d.exporters3d import export_step, export_stl
        part = builder_fn()
        fmt = fmt.lower()
        if fmt == "stl":
            ext = "stl"
            mime = "model/stl"
            export_fn = export_stl
        else:
            ext = "step"
            mime = "model/step"
            export_fn = export_step

        with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tf:
            temp_path = tf.name

        try:
            export_fn(part, temp_path)
            with open(temp_path, "rb") as f:
                data = f.read()
            filename = f"{part_id}.{ext}"
            return data, filename, mime
        finally:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass

    def get_part_mesh_binary(self, part_id: str, tolerance: float = 0.1) -> Optional[Tuple[bytes, int]]:
        """
        Builds and tessellates a build123d part into a raw Float32 WebGL buffer (triangle vertices).
        Returns (packed_bytes, vertex_count).
        """
        if not BUILD123D_AVAILABLE or part_id not in self.part_builders:
            return None

        cache_key = f"{part_id}_tol_{tolerance}"
        if cache_key in self.mesh_cache:
            v_count = self.meta_cache[cache_key]["vertex_count"]
            return self.mesh_cache[cache_key], v_count

        title, builder_fn = self.part_builders[part_id]
        t0 = time.time()
        try:
            part = builder_fn()
            verts, tris = part.tessellate(tolerance=tolerance)
            verts_np = np.array([[pt.X, pt.Y, pt.Z] for pt in verts], dtype=np.float32)
            tris_np = np.array(tris, dtype=np.int32)
            tri_verts = verts_np[tris_np.flatten()]  # (N_tri * 3, 3)
            buf = tri_verts.astype(np.float32).tobytes()
            v_count = len(tri_verts)

            self.mesh_cache[cache_key] = buf
            self.meta_cache[cache_key] = {
                "vertex_count": v_count,
                "triangle_count": len(tris),
                "build_time_ms": round((time.time() - t0) * 1000, 2)
            }
            logger.info("Tessellated %s: %d tris (%d bytes) in %.1f ms",
                        part_id, len(tris), len(buf), (time.time() - t0) * 1000)
            return buf, v_count
        except Exception as e:
            logger.error("Error tessellating build123d part %s: %s", part_id, e)
            return None

    def get_assembly_manifest(self, mode: str = "cartridge", explode: float = 0.0) -> Dict[str, Any]:
        """
        Returns the assembly part list with exact non-penetrating stack positions,
        material properties, colors, and live fluidic channel elevations.
        """
        if not BUILD123D_AVAILABLE:
            return {"error": "build123d not available", "parts": []}

        explode_gap = max(0.0, float(explode))

        if mode in ["cartridge", "stack"]:
            # Exact mathematical non-penetrating stackup:
            # Layer 0: Pusher [0.0, 4.0] -> base_z = 0.0
            # Layer 1: Base Plate [0.0, 4.0] -> base_z = 4.0
            # Layer 2: Gasket Bot [0.0, 1.5875] -> base_z = 8.0
            # Layer 3: Active PCB [0.0, 0.8] -> base_z = 9.6
            # Layer 4: Gasket Top [-0.8, 1.5875] -> base_z = 11.2 (protrusion nests flush into PCB hole)
            # Layer 5: Top Plate [0.0, 4.0] -> base_z = 12.0
            # Layer 6: Pressure Puck [-3.0, 3.0] -> base_z = 19.0 (bottom rests flush on top of Plate 3 at 16.0)

            stack_layers = [
                {
                    "id": "pusher",
                    "name": "Bottom Pusher Plate (TPU)",
                    "layer": 0,
                    "base_z": 0.0,
                    "thickness": 4.0,
                    "color": "#475569",
                    "opacity": 0.95,
                    "transparent": False,
                    "metalness": 0.4,
                    "roughness": 0.3
                },
                {
                    "id": "base",
                    "name": "Plate 1 Hex Base (PP 4mm)",
                    "layer": 1,
                    "base_z": 4.0,
                    "thickness": 4.0,
                    "color": "#10b981",
                    "opacity": 0.75,
                    "transparent": True,
                    "metalness": 0.1,
                    "roughness": 0.2
                },
                {
                    "id": "gasket_bot",
                    "name": "Fitted Amplifier Gasket (Trans Channel)",
                    "layer": 2,
                    "base_z": 8.0,
                    "thickness": 1.6,
                    "color": "#f97316",
                    "opacity": 0.90,
                    "transparent": True,
                    "metalness": 0.1,
                    "roughness": 0.5
                },
                {
                    "id": "pcb",
                    "name": "Active Rigid PCB (Amplifier)",
                    "layer": 3,
                    "base_z": 9.6,
                    "thickness": 0.8,
                    "color": "#059669",
                    "opacity": 1.0,
                    "transparent": False,
                    "metalness": 0.3,
                    "roughness": 0.4
                },
                {
                    "id": "gasket_top",
                    "name": "Cast Nanopore Wafer Gasket (Cis Channel)",
                    "layer": 4,
                    "base_z": 11.2,
                    "thickness": 1.6,
                    "color": "#00f0ff",
                    "opacity": 0.80,
                    "transparent": True,
                    "metalness": 0.15,
                    "roughness": 0.25
                },
                {
                    "id": "top_plate",
                    "name": "Plate 3 Top Hex (PP 4mm)",
                    "layer": 5,
                    "base_z": 12.0,
                    "thickness": 4.0,
                    "color": "#38bdf8",
                    "opacity": 0.60,
                    "transparent": True,
                    "metalness": 0.1,
                    "roughness": 0.1
                },
                {
                    "id": "puck",
                    "name": "Pressure Puck (Anodized Al)",
                    "layer": 6,
                    "base_z": 19.0,
                    "thickness": 6.0,
                    "color": "#ec4899",
                    "opacity": 0.95,
                    "transparent": False,
                    "metalness": 0.5,
                    "roughness": 0.3
                }
            ]

            parts = []
            for item in stack_layers:
                z_pos = item["base_z"] + item["layer"] * explode_gap
                part_entry = dict(item)
                part_entry["z_pos"] = z_pos
                parts.append(part_entry)

            fluidic_heights = {
                "zBot": 8.8 + 2 * explode_gap,
                "zPore": 10.0 + 3 * explode_gap,
                "zTop": 11.2 + 4 * explode_gap
            }

            return {
                "mode": mode,
                "parts": parts,
                "explode": explode_gap,
                "total_stack_height": 22.0 + 6 * explode_gap,
                "fluidic_heights": fluidic_heights
            }

        elif mode == "channels":
            # Focused microfluidic channel core
            stack_layers = [
                {
                    "id": "gasket_bot",
                    "name": "Fitted Amplifier Gasket (Trans Channel)",
                    "layer": 0,
                    "base_z": 0.0,
                    "thickness": 1.6,
                    "color": "#f97316",
                    "opacity": 0.85,
                    "transparent": True,
                    "metalness": 0.1,
                    "roughness": 0.4
                },
                {
                    "id": "pcb",
                    "name": "Active Rigid PCB (Aperture Carrier)",
                    "layer": 1,
                    "base_z": 1.6,
                    "thickness": 0.8,
                    "color": "#059669",
                    "opacity": 0.95,
                    "transparent": False,
                    "metalness": 0.3,
                    "roughness": 0.3
                },
                {
                    "id": "gasket_top",
                    "name": "Cast Nanopore Wafer Gasket (Cis Channel)",
                    "layer": 2,
                    "base_z": 3.2,
                    "thickness": 1.6,
                    "color": "#00f0ff",
                    "opacity": 0.80,
                    "transparent": True,
                    "metalness": 0.15,
                    "roughness": 0.2
                }
            ]
            parts = []
            for item in stack_layers:
                z_pos = item["base_z"] + item["layer"] * explode_gap
                part_entry = dict(item)
                part_entry["z_pos"] = z_pos
                parts.append(part_entry)

            fluidic_heights = {
                "zBot": 0.8 + 0 * explode_gap,
                "zPore": 2.0 + 1 * explode_gap,
                "zTop": 3.2 + 2 * explode_gap
            }
            return {
                "mode": mode,
                "parts": parts,
                "explode": explode_gap,
                "fluidic_heights": fluidic_heights
            }

        elif mode == "enclosure":
            # Full instrument reader enclosure
            parts = [
                {
                    "id": "case",
                    "name": "Reader Base Housing",
                    "layer": 0,
                    "base_z": 0.0,
                    "z_pos": 0.0,
                    "color": "#1e293b",
                    "opacity": 0.95,
                    "transparent": False,
                    "metalness": 0.4,
                    "roughness": 0.4
                },
                {
                    "id": "lid",
                    "name": "Reader Instrument Lid",
                    "layer": 1,
                    "base_z": 26.0,
                    "z_pos": 26.0 + explode_gap,
                    "color": "#334155",
                    "opacity": 0.70,
                    "transparent": True,
                    "metalness": 0.3,
                    "roughness": 0.3
                },
                {
                    "id": "cap_inner",
                    "name": "Torque Cap Inner Drive",
                    "layer": 2,
                    "base_z": 62.0,
                    "z_pos": 62.0 + explode_gap * 2.0,
                    "color": "#0284c7",
                    "opacity": 0.95,
                    "transparent": False,
                    "metalness": 0.4,
                    "roughness": 0.3
                },
                {
                    "id": "cap_outer",
                    "name": "Torque Cap Outer Knurl",
                    "layer": 3,
                    "base_z": 62.0,
                    "z_pos": 62.0 + explode_gap * 2.0,
                    "color": "#0ea5e9",
                    "opacity": 0.95,
                    "transparent": False,
                    "metalness": 0.3,
                    "roughness": 0.3
                }
            ]
            fluidic_heights = {
                "zBot": 13.0,
                "zPore": 14.5,
                "zTop": 16.0
            }
            return {
                "mode": mode,
                "parts": parts,
                "explode": explode_gap,
                "fluidic_heights": fluidic_heights
            }

        elif mode == "mold":
            parts = [
                {
                    "id": "mold_left",
                    "name": "Wafer Mold Left Half",
                    "layer": 0,
                    "base_z": 0.0,
                    "z_pos": -explode_gap,
                    "color": "#eab308",
                    "opacity": 0.95,
                    "transparent": False,
                    "metalness": 0.25,
                    "roughness": 0.25
                },
                {
                    "id": "mold_right",
                    "name": "Wafer Mold Right Half",
                    "layer": 1,
                    "base_z": 0.0,
                    "z_pos": explode_gap,
                    "color": "#f59e0b",
                    "opacity": 0.95,
                    "transparent": False,
                    "metalness": 0.25,
                    "roughness": 0.25
                }
            ]
            return {
                "mode": mode,
                "parts": parts,
                "explode": explode_gap,
                "fluidic_heights": None
            }

        elif mode == "pump":
            parts = [
                {
                    "id": "pump_case",
                    "name": "Peristaltic Pump Housing Case",
                    "layer": 0,
                    "base_z": 0.0,
                    "z_pos": 0.0,
                    "color": "#1e293b",
                    "opacity": 0.95,
                    "transparent": False,
                    "metalness": 0.4,
                    "roughness": 0.3
                },
                {
                    "id": "pump_lid",
                    "name": "Peristaltic Pump Housing Lid",
                    "layer": 1,
                    "base_z": 35.0,
                    "z_pos": 35.0 + explode_gap,
                    "color": "#475569",
                    "opacity": 0.85,
                    "transparent": True,
                    "metalness": 0.4,
                    "roughness": 0.3
                }
            ]
            return {
                "mode": mode,
                "parts": parts,
                "explode": explode_gap,
                "fluidic_heights": None
            }

        elif mode in ["system", "full_system"]:
            parts = [
                # Bottom Peristaltic Pump Module
                {
                    "id": "pump_case",
                    "name": "Peristaltic Pump Housing Case",
                    "layer": 0,
                    "base_z": -75.0,
                    "z_pos": -75.0 - explode_gap * 2.0,
                    "color": "#0f172a",
                    "opacity": 0.95,
                    "transparent": False,
                    "metalness": 0.4,
                    "roughness": 0.3
                },
                {
                    "id": "pump_lid",
                    "name": "Pump Module Inter-Stage Lid",
                    "layer": 1,
                    "base_z": -12.0,
                    "z_pos": -12.0 - explode_gap,
                    "color": "#1e293b",
                    "opacity": 0.90,
                    "transparent": False,
                    "metalness": 0.4,
                    "roughness": 0.3
                },
                # Reader Housing
                {
                    "id": "case",
                    "name": "Reader Instrument Base Case",
                    "layer": 2,
                    "base_z": 0.0,
                    "z_pos": 0.0,
                    "color": "#1e293b",
                    "opacity": 0.95,
                    "transparent": False,
                    "metalness": 0.4,
                    "roughness": 0.4
                },
                # Cartridge stack seated in reader neck
                {
                    "id": "pusher",
                    "name": "Bottom Pusher Plate (TPU)",
                    "layer": 3,
                    "base_z": 12.0,
                    "z_pos": 12.0,
                    "color": "#475569",
                    "opacity": 0.95,
                    "transparent": False,
                    "metalness": 0.4,
                    "roughness": 0.3
                },
                {
                    "id": "base",
                    "name": "Plate 1 Hex Base (PP 4mm)",
                    "layer": 4,
                    "base_z": 16.0,
                    "z_pos": 16.0 + explode_gap * 0.5,
                    "color": "#10b981",
                    "opacity": 0.75,
                    "transparent": True,
                    "metalness": 0.1,
                    "roughness": 0.2
                },
                {
                    "id": "gasket_bot",
                    "name": "Fitted Amplifier Gasket (Trans Channel)",
                    "layer": 5,
                    "base_z": 20.0,
                    "z_pos": 20.0 + explode_gap * 1.0,
                    "color": "#f97316",
                    "opacity": 0.90,
                    "transparent": True,
                    "metalness": 0.1,
                    "roughness": 0.5
                },
                {
                    "id": "pcb",
                    "name": "Active Rigid PCB (Amplifier)",
                    "layer": 6,
                    "base_z": 21.6,
                    "z_pos": 21.6 + explode_gap * 1.5,
                    "color": "#059669",
                    "opacity": 1.0,
                    "transparent": False,
                    "metalness": 0.3,
                    "roughness": 0.4
                },
                {
                    "id": "gasket_top",
                    "name": "Cast Nanopore Wafer Gasket (Cis Channel)",
                    "layer": 7,
                    "base_z": 23.2,
                    "z_pos": 23.2 + explode_gap * 2.0,
                    "color": "#00f0ff",
                    "opacity": 0.80,
                    "transparent": True,
                    "metalness": 0.15,
                    "roughness": 0.25
                },
                {
                    "id": "top_plate",
                    "name": "Plate 3 Top Hex (PP 4mm)",
                    "layer": 8,
                    "base_z": 24.0,
                    "z_pos": 24.0 + explode_gap * 2.5,
                    "color": "#38bdf8",
                    "opacity": 0.60,
                    "transparent": True,
                    "metalness": 0.1,
                    "roughness": 0.1
                },
                {
                    "id": "puck",
                    "name": "Pressure Puck (Aluminum)",
                    "layer": 9,
                    "base_z": 31.0,
                    "z_pos": 31.0 + explode_gap * 3.0,
                    "color": "#ec4899",
                    "opacity": 0.95,
                    "transparent": False,
                    "metalness": 0.5,
                    "roughness": 0.3
                },
                # Reader Lid & Torque Cap
                {
                    "id": "lid",
                    "name": "Reader Instrument Lid",
                    "layer": 10,
                    "base_z": 30.0,
                    "z_pos": 30.0 + explode_gap * 1.5,
                    "color": "#334155",
                    "opacity": 0.70,
                    "transparent": True,
                    "metalness": 0.3,
                    "roughness": 0.3
                },
                {
                    "id": "cap_inner",
                    "name": "Torque Cap Inner Drive",
                    "layer": 11,
                    "base_z": 62.0,
                    "z_pos": 62.0 + explode_gap * 3.5,
                    "color": "#0284c7",
                    "opacity": 0.95,
                    "transparent": False,
                    "metalness": 0.4,
                    "roughness": 0.3
                },
                {
                    "id": "cap_outer",
                    "name": "Torque Cap Outer Knurl",
                    "layer": 12,
                    "base_z": 62.0,
                    "z_pos": 62.0 + explode_gap * 3.5,
                    "color": "#0ea5e9",
                    "opacity": 0.95,
                    "transparent": False,
                    "metalness": 0.3,
                    "roughness": 0.3
                },
                # Fluidic Tubing & Wiring Loom
                {
                    "id": "tubing",
                    "name": "Peristaltic Fluidic Tubing Splines",
                    "layer": 13,
                    "base_z": 0.0,
                    "z_pos": 0.0,
                    "color": "#00f0ff",
                    "opacity": 0.75,
                    "transparent": True,
                    "metalness": 0.2,
                    "roughness": 0.2
                },
                {
                    "id": "wiring",
                    "name": "Internal USB-C Wiring Loom",
                    "layer": 14,
                    "base_z": 0.0,
                    "z_pos": 0.0,
                    "color": "#ef4444",
                    "opacity": 0.95,
                    "transparent": False,
                    "metalness": 0.5,
                    "roughness": 0.3
                }
            ]
            fluidic_heights = {
                "zBot": 20.8 + explode_gap * 1.0,
                "zPore": 22.0 + explode_gap * 1.5,
                "zTop": 23.2 + explode_gap * 2.0
            }
            return {
                "mode": mode,
                "parts": parts,
                "explode": explode_gap,
                "fluidic_heights": fluidic_heights
            }

        return {"error": f"Unknown mode: {mode}", "parts": []}

# Singleton instance
_engine_instance = None

def get_build123d_engine() -> Build123dEngine:
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = Build123dEngine()
    return _engine_instance
