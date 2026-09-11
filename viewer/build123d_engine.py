"""
Build123d Direct CAD Integration Engine
Streams tessellated geometry and parametric stackups directly from build123d Python scripts.
Eliminates intermediate STL/DXF artifacts, ensures exact non-overlapping mathematical stackup,
and supports interactive parametric adjustments (explode factor, layer thickness, tolerances).
"""

import sys
import os
import math
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
        Pressure_Puck,
        Nanopore_Wafer_Silicon
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


def Peristaltic_Pump_Motor_Body():
    """Diitao peristaltic pump motor cylinder body (h=38, d=27.6)."""
    with BuildPart(mode=Mode.PRIVATE) as p:
        with Locations((0, 0, 38.0 / 2.0)):
            Cylinder(radius=27.6 / 2.0, height=38.0)
    return p.part

def Peristaltic_Pump_Motor_Head():
    """Diitao peristaltic pump head with tubing stubs (h=20, d=31.7) at z=38."""
    with BuildPart(mode=Mode.PRIVATE) as p:
        with Locations((0, 0, 38.0 + 20.0 / 2.0)):
            Cylinder(radius=31.7 / 2.0, height=20.0)
        with Locations((12.0, 5.0, 38.0 + 10.0)):
            Cylinder(radius=2.5, height=15.0, rotation=(0, 90, 30))
        with Locations((12.0, -5.0, 38.0 + 10.0)):
            Cylinder(radius=2.5, height=15.0, rotation=(0, 90, -30))
    return p.part

def External_Tubing_Loop_Right():
    """External blue peristaltic tubing loop connecting pump out to reader in."""
    p0 = (50.0 + 10.0, 20.0, 32.5)
    p1 = (50.0 + 25.0, 20.0, 32.5)
    p2 = (50.0 + 25.0, 20.0, 65.0 + 5.0 + 25.0 - 5.0)
    p3 = (50.0 + 10.0, 20.0, 65.0 + 5.0 + 25.0 - 5.0)
    with BuildPart(mode=Mode.PRIVATE) as p:
        with BuildLine():
            Spline([p0, p1, p2, p3])
        v0 = Vector(p1) - Vector(p0)
        plane = Plane(origin=p0, z_dir=v0)
        with BuildSketch(plane):
            Circle(radius=2.0)
        sweep()
    return p.part

def Luer_Adapters_Cluster():
    """4 Threaded Luer Adapters and 4 Elbow Adapters fitted under Bottom_Pusher_Plate."""
    from src.adapters import Threaded_Luer_Adapter, Elbow_Adapter
    parts = []
    for a in [const.ANGLE_FLUID_IN_BOT, const.ANGLE_FLUID_IN_TOP, const.ANGLE_FLUID_OUT_BOT, const.ANGLE_FLUID_OUT_TOP]:
        ang_rad = math.radians(a)
        px = const.PORT_RADIUS * math.cos(ang_rad)
        py = const.PORT_RADIUS * math.sin(ang_rad)
        rot_z = 90.0 if a > 180 else -90.0
        adapter = Threaded_Luer_Adapter().rotate(Axis.X, 180).move(Location((px, py, 0)))
        elbow = Elbow_Adapter().rotate(Axis.X, 180).rotate(Axis.Z, rot_z).move(Location((px, py, -20.2)))
        parts.append(adapter)
        parts.append(elbow)
    return Compound(children=parts)


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
            "base": ("Plate 1 Base Hex (Clear Acrylic)", Plate_1_Base_Hex),
            "gasket_bot": ("Fitted Amplifier Gasket (Trans Channel)", Fitted_Amplifier_Gasket),
            "pcb": ("Active Rigid PCB (FR4)", PCB_Rigid_Active),
            "wafer": ("Silicon Nanopore Wafer (4x4mm 45°)", Nanopore_Wafer_Silicon),
            "gasket_top": ("Cast Nanopore Wafer Gasket (Cis Channel)", Cast_Nanopore_Wafer_Gasket),
            "top_plate": ("Plate 3 Top Hex (Clear Acrylic)", Plate_3_Top_Hex),
            "puck": ("Pressure Puck (Anodized Charcoal Al)", Pressure_Puck),
            # Enclosure & Torque-Lock Cap
            "case": ("Reader Housing Case (White)", Housing_Case_Reader),
            "lid": ("Reader Housing Lid with M75 Thread (White)", Housing_Lid_Reader),
            "cap_inner": ("Torque Cap Inner Drive (Silver)", Torque_Cap_Inner),
            "cap_outer": ("Torque Cap Outer Knurl (Orange)", Torque_Cap_Outer),
            # Standalone Wafer Molds & Cast Part
            "mold_left": ("Nanopore Wafer Mold (Left Half)", lambda: Nanopore_Wafer_Mold_Left(is_nanopore=True)),
            "mold_right": ("Nanopore Wafer Mold (Right Half)", lambda: Nanopore_Wafer_Mold_Right(is_nanopore=True)),
            "mold_gasket": ("Cast PDMS Gasket in Mold Cavity", lambda: Cast_Nanopore_Wafer_Gasket().move(Location((0, 0, -1.5875 / 2.0))).rotate(Axis.Y, 90).move(Location((0, 0, 35.0)))),
            "mold_wafer": ("Silicon Nanopore Wafer in Mold", lambda: Nanopore_Wafer_Silicon().move(Location((0, 0, -0.4))).move(Location((0, 0, -1.5875 / 2.0))).rotate(Axis.Y, 90).move(Location((0, 0, 35.0)))),
            # Pump Module Components
            "pump_case": ("Peristaltic Pump Housing Case", Housing_Pump_Case),
            "pump_lid": ("Peristaltic Pump Housing Lid", Housing_Pump_Lid),
            "pump_motor_body_1": ("Peristaltic Pump Motor 1 Body", lambda: Peristaltic_Pump_Motor_Body().move(Location((-25.0, 20.0, 0.0)))),
            "pump_motor_head_1": ("Peristaltic Pump Motor 1 Head", lambda: Peristaltic_Pump_Motor_Head().move(Location((-25.0, 20.0, 0.0)))),
            "pump_motor_body_2": ("Peristaltic Pump Motor 2 Body", lambda: Peristaltic_Pump_Motor_Body().move(Location((25.0, 20.0, 0.0)))),
            "pump_motor_head_2": ("Peristaltic Pump Motor 2 Head", lambda: Peristaltic_Pump_Motor_Head().move(Location((25.0, 20.0, 0.0)))),
            "tubing_loop_right": ("External Blue Tubing Loop (Right)", External_Tubing_Loop_Right),
            "luer_adapters": ("Threaded Luer & Elbow Adapters", Luer_Adapters_Cluster),
            # Dual Mold Visualizer at X = -130 (Multi_Mold_Visualizer in OpenSCAD)
            "sys_mold_nanopore_left": ("Wafer Mold Left (Teal)", lambda: Nanopore_Wafer_Mold_Left(is_nanopore=True).move(Location((-130.0, -55.0, 0.0)))),
            "sys_mold_nanopore_right": ("Wafer Mold Right (Teal)", lambda: Nanopore_Wafer_Mold_Right(is_nanopore=True).move(Location((-130.0, -55.0, 0.0)))),
            "sys_mold_nanopore_gasket": ("Cast Wafer Gasket (Red)", lambda: Cast_Nanopore_Wafer_Gasket().move(Location((0, 0, -1.5875 / 2.0))).rotate(Axis.Y, 90).move(Location((-130.0, -55.0, 35.0)))),
            "sys_mold_nanopore_wafer": ("Silicon Wafer (45°)", lambda: Nanopore_Wafer_Silicon().move(Location((0, 0, -0.4))).move(Location((0, 0, -1.5875 / 2.0))).rotate(Axis.Y, 90).move(Location((-130.0, -55.0, 35.0)))),
            "sys_mold_amp_left": ("Amplifier Mold Left (Steel Blue)", lambda: Nanopore_Wafer_Mold_Left(is_nanopore=False).move(Location((-130.0, 55.0, 0.0)))),
            "sys_mold_amp_right": ("Amplifier Mold Right (Steel Blue)", lambda: Nanopore_Wafer_Mold_Right(is_nanopore=False).move(Location((-130.0, 55.0, 0.0)))),
            "sys_mold_amp_gasket": ("Cast Amplifier Gasket (Red)", lambda: Fitted_Amplifier_Gasket().move(Location((0, 0, -1.5875 / 2.0))).rotate(Axis.Y, 90).move(Location((-130.0, 55.0, 35.0))))
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
        material properties, colors, and live fluidic channel elevations matching OpenSCAD truth.
        """
        if not BUILD123D_AVAILABLE:
            return {"error": "build123d not available", "parts": []}

        explode_gap = max(0.0, float(explode))

        if mode in ["cartridge", "stack"]:
            # Mathematical non-penetrating stackup with Torque-Lock Cap:
            # Layer 0: Pusher [0.0, 3.0] -> base_z = 0.0
            # Layer 1: Base Plate [0.0, 4.0] -> base_z = 3.0
            # Layer 2: Gasket Bot [0.0, 1.6] -> base_z = 7.0
            # Layer 3: Active PCB [0.0, 0.8] -> base_z = 8.6
            # Layer 3.5: Silicon Nanopore Wafer [-0.25, 0.25] -> base_z = 9.0
            # Layer 4: Gasket Top [0.0, 1.6] -> base_z = 9.4 (protrusion -0.8 nests into PCB cutout down to 8.6)
            # Layer 5: Top Plate [0.0, 4.0] -> base_z = 11.0
            # Layer 6: Pressure Puck [-3.0, 3.0] -> base_z = 18.0 (bottom rests flush on Plate 3 at 15.0, top at 21.0)
            # Layer 7: Torque Cap Outer [0.0, 30.2] -> base_z = 25.0
            # Layer 8: Torque Cap Inner [0.0, 20.2] -> base_z = 29.0

            stack_layers = [
                {
                    "id": "pusher",
                    "name": "Bottom Pusher Plate (TPU)",
                    "layer": 0,
                    "base_z": 0.0,
                    "slide_factor": 0.0,
                    "thickness": 3.0,
                    "color": "#475569",
                    "opacity": 0.95,
                    "transparent": False,
                    "metalness": 0.4,
                    "roughness": 0.3
                },
                {
                    "id": "base",
                    "name": "Plate 1 Hex Base (Clear Acrylic)",
                    "layer": 1,
                    "base_z": 3.0,
                    "slide_factor": 0.5,
                    "thickness": 4.0,
                    "color": "#f1f5f9",
                    "opacity": 0.65,
                    "transparent": True,
                    "metalness": 0.1,
                    "roughness": 0.1
                },
                {
                    "id": "gasket_bot",
                    "name": "Fitted Amplifier Gasket (Trans Channel)",
                    "layer": 2,
                    "base_z": 7.0,
                    "slide_factor": 1.0,
                    "thickness": 1.6,
                    "color": "#ef4444",
                    "opacity": 0.85,
                    "transparent": True,
                    "metalness": 0.1,
                    "roughness": 0.3
                },
                {
                    "id": "pcb",
                    "name": "Active Rigid PCB (Amplifier)",
                    "layer": 3,
                    "base_z": 8.6,
                    "slide_factor": 1.5,
                    "thickness": 0.8,
                    "color": "#10b981",
                    "opacity": 1.0,
                    "transparent": False,
                    "metalness": 0.3,
                    "roughness": 0.4
                },
                {
                    "id": "wafer",
                    "name": "Silicon Nanopore Wafer (4x4mm 45°)",
                    "layer": 3.5,
                    "base_z": 9.0,
                    "slide_factor": 1.75,
                    "thickness": 0.5,
                    "color": "#64748b",
                    "opacity": 0.95,
                    "transparent": False,
                    "metalness": 0.85,
                    "roughness": 0.2
                },
                {
                    "id": "gasket_top",
                    "name": "Cast Nanopore Wafer Gasket (Cis Channel)",
                    "layer": 4,
                    "base_z": 9.4,
                    "slide_factor": 2.0,
                    "thickness": 1.6,
                    "color": "#ef4444",
                    "opacity": 0.85,
                    "transparent": True,
                    "metalness": 0.1,
                    "roughness": 0.3
                },
                {
                    "id": "top_plate",
                    "name": "Plate 3 Top Hex (Clear Acrylic)",
                    "layer": 5,
                    "base_z": 11.0,
                    "slide_factor": 2.5,
                    "thickness": 4.0,
                    "color": "#f1f5f9",
                    "opacity": 0.65,
                    "transparent": True,
                    "metalness": 0.1,
                    "roughness": 0.1
                },
                {
                    "id": "puck",
                    "name": "Pressure Puck (Anodized Charcoal Al)",
                    "layer": 6,
                    "base_z": 18.0,
                    "slide_factor": 3.0,
                    "thickness": 6.0,
                    "color": "#1e293b",
                    "opacity": 0.95,
                    "transparent": False,
                    "metalness": 0.6,
                    "roughness": 0.3
                },
                {
                    "id": "cap_outer",
                    "name": "Torque Cap Outer Knurl (Vivid Orange)",
                    "layer": 7,
                    "base_z": 25.0,
                    "slide_factor": 3.5,
                    "thickness": 30.2,
                    "color": "#f59e0b",
                    "opacity": 0.95,
                    "transparent": False,
                    "metalness": 0.3,
                    "roughness": 0.3
                },
                {
                    "id": "cap_inner",
                    "name": "Torque Cap Inner Drive (Metallic Silver)",
                    "layer": 8,
                    "base_z": 29.0,
                    "slide_factor": 4.0,
                    "thickness": 20.2,
                    "color": "#94a3b8",
                    "opacity": 0.95,
                    "transparent": False,
                    "metalness": 0.8,
                    "roughness": 0.25
                }
            ]

            parts = []
            for item in stack_layers:
                z_pos = item["base_z"] + item["slide_factor"] * explode_gap
                part_entry = dict(item)
                part_entry["z_pos"] = z_pos
                parts.append(part_entry)

            fluidic_heights = {
                "zBot": 7.8 + 1.0 * explode_gap,
                "zPore": 9.0 + 1.75 * explode_gap,
                "zTop": 10.2 + 2.0 * explode_gap
            }

            return {
                "mode": mode,
                "parts": parts,
                "explode": explode_gap,
                "total_stack_height": 29.0 + 4.0 * explode_gap,
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
                    "slide_factor": 0.0,
                    "thickness": 1.6,
                    "color": "#ef4444",
                    "opacity": 0.85,
                    "transparent": True,
                    "metalness": 0.1,
                    "roughness": 0.3
                },
                {
                    "id": "pcb",
                    "name": "Active Rigid PCB (Aperture Carrier)",
                    "layer": 1,
                    "base_z": 1.6,
                    "slide_factor": 1.0,
                    "thickness": 0.8,
                    "color": "#10b981",
                    "opacity": 1.0,
                    "transparent": False,
                    "metalness": 0.3,
                    "roughness": 0.4
                },
                {
                    "id": "wafer",
                    "name": "Silicon Nanopore Wafer (4x4mm 45°)",
                    "layer": 1.5,
                    "base_z": 2.0,
                    "slide_factor": 1.5,
                    "thickness": 0.5,
                    "color": "#64748b",
                    "opacity": 0.95,
                    "transparent": False,
                    "metalness": 0.85,
                    "roughness": 0.2
                },
                {
                    "id": "gasket_top",
                    "name": "Cast Nanopore Wafer Gasket (Cis Channel)",
                    "layer": 2,
                    "base_z": 2.4,
                    "slide_factor": 2.0,
                    "thickness": 1.6,
                    "color": "#ef4444",
                    "opacity": 0.85,
                    "transparent": True,
                    "metalness": 0.1,
                    "roughness": 0.3
                }
            ]
            parts = []
            for item in stack_layers:
                z_pos = item["base_z"] + item["slide_factor"] * explode_gap
                part_entry = dict(item)
                part_entry["z_pos"] = z_pos
                parts.append(part_entry)

            fluidic_heights = {
                "zBot": 0.8 + 0 * explode_gap,
                "zPore": 2.0 + 1.5 * explode_gap,
                "zTop": 3.2 + 2 * explode_gap
            }
            return {
                "mode": mode,
                "parts": parts,
                "explode": explode_gap,
                "fluidic_heights": fluidic_heights
            }

        elif mode == "enclosure":
            # Reader Instrument Enclosure (Case + Lid with Threaded Neck + Torque Cap)
            parts = [
                {
                    "id": "case",
                    "name": "Reader Base Housing (Solid White)",
                    "layer": 0,
                    "base_z": 0.0,
                    "slide_factor": 0.0,
                    "z_pos": 0.0,
                    "color": "#ffffff",
                    "opacity": 1.0,
                    "transparent": False,
                    "metalness": 0.2,
                    "roughness": 0.35
                },
                {
                    "id": "lid",
                    "name": "Reader Instrument Lid with M75 Thread (Solid White)",
                    "layer": 1,
                    "base_z": 25.0,
                    "slide_factor": 1.0,
                    "z_pos": 25.0 + explode_gap,
                    "color": "#f8fafc",
                    "opacity": 1.0,
                    "transparent": False,
                    "metalness": 0.2,
                    "roughness": 0.35
                },
                {
                    "id": "cap_outer",
                    "name": "Torque Cap Outer Knurl (Vivid Orange)",
                    "layer": 2,
                    "base_z": 55.0,
                    "slide_factor": 2.0,
                    "z_pos": 55.0 + explode_gap * 2.0,
                    "color": "#f59e0b",
                    "opacity": 0.95,
                    "transparent": False,
                    "metalness": 0.3,
                    "roughness": 0.3
                },
                {
                    "id": "cap_inner",
                    "name": "Torque Cap Inner Drive (Metallic Silver)",
                    "layer": 3,
                    "base_z": 62.0,
                    "slide_factor": 2.5,
                    "z_pos": 62.0 + explode_gap * 2.5,
                    "color": "#94a3b8",
                    "opacity": 0.95,
                    "transparent": False,
                    "metalness": 0.8,
                    "roughness": 0.25
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
                    "name": "Nanopore Wafer Mold Left Half (Teal)",
                    "layer": 0,
                    "base_x": 0.0,
                    "base_y": 0.0,
                    "base_z": 0.0,
                    "x_pos": -explode_gap,
                    "y_pos": 0.0,
                    "z_pos": 0.0,
                    "slide_dir": "x",
                    "slide_factor": -1.0,
                    "color": "#14b8a6",
                    "opacity": 0.95,
                    "transparent": False,
                    "metalness": 0.25,
                    "roughness": 0.25
                },
                {
                    "id": "mold_gasket",
                    "name": "Cast PDMS Gasket (Molded Red Silicone)",
                    "layer": 0.5,
                    "base_x": 0.0,
                    "base_y": 0.0,
                    "base_z": 0.0,
                    "x_pos": 0.0,
                    "y_pos": 0.0,
                    "z_pos": 0.0,
                    "slide_dir": "none",
                    "slide_factor": 0.0,
                    "color": "#ef4444",
                    "opacity": 0.85,
                    "transparent": True,
                    "metalness": 0.15,
                    "roughness": 0.3
                },
                {
                    "id": "mold_wafer",
                    "name": "Silicon Nanopore Wafer (4x4mm 45°)",
                    "layer": 0.6,
                    "base_x": 0.0,
                    "base_y": 0.0,
                    "base_z": 0.0,
                    "x_pos": 0.0,
                    "y_pos": 0.0,
                    "z_pos": 0.0,
                    "slide_dir": "none",
                    "slide_factor": 0.0,
                    "color": "#64748b",
                    "opacity": 0.95,
                    "transparent": False,
                    "metalness": 0.85,
                    "roughness": 0.2
                },
                {
                    "id": "mold_right",
                    "name": "Nanopore Wafer Mold Right Half (Teal)",
                    "layer": 1,
                    "base_x": 0.0,
                    "base_y": 0.0,
                    "base_z": 0.0,
                    "x_pos": explode_gap,
                    "y_pos": 0.0,
                    "z_pos": 0.0,
                    "slide_dir": "x",
                    "slide_factor": 1.0,
                    "color": "#14b8a6",
                    "opacity": 0.95,
                    "transparent": False,
                    "metalness": 0.25,
                    "roughness": 0.25
                }
            ]
            mold_cavity_data = {
                "cavity_z": 35.0,
                "mold_h": 90.0,
                "mold_w": 80.0,
                "res_d": 15.0,
                "sprue_z": 61.0,
                "gasket_diam": 60.0,
                "gasket_thick": 1.6
            }
            return {
                "mode": mode,
                "parts": parts,
                "explode": explode_gap,
                "mold_cavity": mold_cavity_data,
                "fluidic_heights": None
            }

        elif mode == "pump":
            parts = [
                {
                    "id": "pump_case",
                    "name": "Peristaltic Pump Housing Case (Dim Gray)",
                    "layer": 0,
                    "base_z": 0.0,
                    "slide_factor": 0.0,
                    "z_pos": 0.0,
                    "color": "#1e293b",
                    "opacity": 0.95,
                    "transparent": False,
                    "metalness": 0.4,
                    "roughness": 0.3
                },
                {
                    "id": "pump_motor_body_1",
                    "name": "Peristaltic Pump Motor 1 Body (Silver)",
                    "layer": 0.2,
                    "base_z": 5.0,
                    "slide_factor": 0.0,
                    "z_pos": 5.0,
                    "color": "#cbd5e1",
                    "opacity": 0.95,
                    "transparent": False,
                    "metalness": 0.85,
                    "roughness": 0.2
                },
                {
                    "id": "pump_motor_head_1",
                    "name": "Peristaltic Pump Motor 1 Head (Dodger Blue)",
                    "layer": 0.3,
                    "base_z": 5.0,
                    "slide_factor": 0.0,
                    "z_pos": 5.0,
                    "color": "#0284c7",
                    "opacity": 0.95,
                    "transparent": False,
                    "metalness": 0.3,
                    "roughness": 0.3
                },
                {
                    "id": "pump_motor_body_2",
                    "name": "Peristaltic Pump Motor 2 Body (Silver)",
                    "layer": 0.2,
                    "base_z": 5.0,
                    "slide_factor": 0.0,
                    "z_pos": 5.0,
                    "color": "#cbd5e1",
                    "opacity": 0.95,
                    "transparent": False,
                    "metalness": 0.85,
                    "roughness": 0.2
                },
                {
                    "id": "pump_motor_head_2",
                    "name": "Peristaltic Pump Motor 2 Head (Dodger Blue)",
                    "layer": 0.3,
                    "base_z": 5.0,
                    "slide_factor": 0.0,
                    "z_pos": 5.0,
                    "color": "#0284c7",
                    "opacity": 0.95,
                    "transparent": False,
                    "metalness": 0.3,
                    "roughness": 0.3
                },
                {
                    "id": "pump_lid",
                    "name": "Peristaltic Pump Housing Lid (Light Gray)",
                    "layer": 1,
                    "base_z": 65.0,
                    "slide_factor": 1.0,
                    "z_pos": 65.0 + explode_gap,
                    "color": "#94a3b8",
                    "opacity": 0.90,
                    "transparent": False,
                    "metalness": 0.7,
                    "roughness": 0.25
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
                # 1. Bottom Peristaltic Pump Module
                {
                    "id": "pump_case",
                    "name": "Peristaltic Pump Housing Case (Dim Gray)",
                    "layer": 0,
                    "base_z": -75.0,
                    "slide_factor": -2.0,
                    "z_pos": -75.0 - explode_gap * 2.0,
                    "color": "#1e293b",
                    "opacity": 0.95,
                    "transparent": False,
                    "metalness": 0.4,
                    "roughness": 0.3
                },
                {
                    "id": "pump_motor_body_1",
                    "name": "Pump Motor 1 Body (Silver)",
                    "layer": 0.2,
                    "base_z": -70.0,
                    "slide_factor": -2.0,
                    "z_pos": -70.0 - explode_gap * 2.0,
                    "color": "#cbd5e1",
                    "opacity": 0.95,
                    "transparent": False,
                    "metalness": 0.85,
                    "roughness": 0.2
                },
                {
                    "id": "pump_motor_head_1",
                    "name": "Pump Motor 1 Head (Blue)",
                    "layer": 0.3,
                    "base_z": -70.0,
                    "slide_factor": -2.0,
                    "z_pos": -70.0 - explode_gap * 2.0,
                    "color": "#0284c7",
                    "opacity": 0.95,
                    "transparent": False,
                    "metalness": 0.3,
                    "roughness": 0.3
                },
                {
                    "id": "pump_motor_body_2",
                    "name": "Pump Motor 2 Body (Silver)",
                    "layer": 0.2,
                    "base_z": -70.0,
                    "slide_factor": -2.0,
                    "z_pos": -70.0 - explode_gap * 2.0,
                    "color": "#cbd5e1",
                    "opacity": 0.95,
                    "transparent": False,
                    "metalness": 0.85,
                    "roughness": 0.2
                },
                {
                    "id": "pump_motor_head_2",
                    "name": "Pump Motor 2 Head (Blue)",
                    "layer": 0.3,
                    "base_z": -70.0,
                    "slide_factor": -2.0,
                    "z_pos": -70.0 - explode_gap * 2.0,
                    "color": "#0284c7",
                    "opacity": 0.95,
                    "transparent": False,
                    "metalness": 0.3,
                    "roughness": 0.3
                },
                {
                    "id": "pump_lid",
                    "name": "Pump Module Inter-Stage Lid (Light Gray)",
                    "layer": 1,
                    "base_z": -10.0,
                    "slide_factor": -1.0,
                    "z_pos": -10.0 - explode_gap,
                    "color": "#94a3b8",
                    "opacity": 0.90,
                    "transparent": False,
                    "metalness": 0.7,
                    "roughness": 0.25
                },
                # 2. External Tubing Loop
                {
                    "id": "tubing_loop_right",
                    "name": "External Peristaltic Tubing Loop (Sky Blue)",
                    "layer": 1.5,
                    "base_z": -75.0,
                    "slide_factor": 0.0,
                    "z_pos": -75.0,
                    "color": "#00bfff",
                    "opacity": 0.85,
                    "transparent": True,
                    "metalness": 0.2,
                    "roughness": 0.2
                },
                # 3. Reader Enclosure
                {
                    "id": "case",
                    "name": "Reader Instrument Base Case (Solid White)",
                    "layer": 2,
                    "base_z": 0.0,
                    "slide_factor": 0.0,
                    "z_pos": 0.0,
                    "color": "#ffffff",
                    "opacity": 1.0,
                    "transparent": False,
                    "metalness": 0.2,
                    "roughness": 0.35
                },
                {
                    "id": "lid",
                    "name": "Reader Instrument Lid with M75 Thread (Solid White)",
                    "layer": 3,
                    "base_z": 5.0,
                    "slide_factor": 0.4,
                    "z_pos": 5.0 + explode_gap * 0.4,
                    "color": "#f8fafc",
                    "opacity": 1.0,
                    "transparent": False,
                    "metalness": 0.2,
                    "roughness": 0.35
                },
                # 4. Cartridge Stack (seated atop Lid ceiling inside M75 neck)
                {
                    "id": "luer_adapters",
                    "name": "Threaded Luer & Elbow Adapters",
                    "layer": 3.8,
                    "base_z": 30.0,
                    "slide_factor": 0.7,
                    "z_pos": 30.0 + explode_gap * 0.7,
                    "color": "#94a3b8",
                    "opacity": 0.95,
                    "transparent": False,
                    "metalness": 0.6,
                    "roughness": 0.3
                },
                {
                    "id": "pusher",
                    "name": "Bottom Pusher Plate (TPU)",
                    "layer": 4,
                    "base_z": 30.0,
                    "slide_factor": 0.9,
                    "z_pos": 30.0 + explode_gap * 0.9,
                    "color": "#475569",
                    "opacity": 0.95,
                    "transparent": False,
                    "metalness": 0.4,
                    "roughness": 0.3
                },
                {
                    "id": "base",
                    "name": "Plate 1 Hex Base (Clear Acrylic)",
                    "layer": 5,
                    "base_z": 33.0,
                    "slide_factor": 1.2,
                    "z_pos": 33.0 + explode_gap * 1.2,
                    "color": "#f1f5f9",
                    "opacity": 0.65,
                    "transparent": True,
                    "metalness": 0.1,
                    "roughness": 0.1
                },
                {
                    "id": "gasket_bot",
                    "name": "Fitted Amplifier Gasket (Trans Channel)",
                    "layer": 6,
                    "base_z": 37.0,
                    "slide_factor": 1.5,
                    "z_pos": 37.0 + explode_gap * 1.5,
                    "color": "#ef4444",
                    "opacity": 0.85,
                    "transparent": True,
                    "metalness": 0.1,
                    "roughness": 0.3
                },
                {
                    "id": "pcb",
                    "name": "Active Rigid PCB (Amplifier)",
                    "layer": 7,
                    "base_z": 38.6,
                    "slide_factor": 1.8,
                    "z_pos": 38.6 + explode_gap * 1.8,
                    "color": "#10b981",
                    "opacity": 1.0,
                    "transparent": False,
                    "metalness": 0.3,
                    "roughness": 0.4
                },
                {
                    "id": "wafer",
                    "name": "Silicon Nanopore Wafer (4x4mm 45°)",
                    "layer": 7.5,
                    "base_z": 39.0,
                    "slide_factor": 1.95,
                    "z_pos": 39.0 + explode_gap * 1.95,
                    "color": "#64748b",
                    "opacity": 0.95,
                    "transparent": False,
                    "metalness": 0.85,
                    "roughness": 0.2
                },
                {
                    "id": "gasket_top",
                    "name": "Cast Nanopore Wafer Gasket (Cis Channel)",
                    "layer": 8,
                    "base_z": 39.4,
                    "slide_factor": 2.1,
                    "z_pos": 39.4 + explode_gap * 2.1,
                    "color": "#ef4444",
                    "opacity": 0.85,
                    "transparent": True,
                    "metalness": 0.1,
                    "roughness": 0.3
                },
                {
                    "id": "top_plate",
                    "name": "Plate 3 Top Hex (Clear Acrylic)",
                    "layer": 9,
                    "base_z": 41.0,
                    "slide_factor": 2.4,
                    "z_pos": 41.0 + explode_gap * 2.4,
                    "color": "#f1f5f9",
                    "opacity": 0.65,
                    "transparent": True,
                    "metalness": 0.1,
                    "roughness": 0.1
                },
                {
                    "id": "puck",
                    "name": "Pressure Puck (Anodized Charcoal Al)",
                    "layer": 10,
                    "base_z": 48.0,
                    "slide_factor": 2.7,
                    "z_pos": 48.0 + explode_gap * 2.7,
                    "color": "#1e293b",
                    "opacity": 0.95,
                    "transparent": False,
                    "metalness": 0.6,
                    "roughness": 0.3
                },
                # 5. Torque-Lock Cap Assembly
                {
                    "id": "cap_outer",
                    "name": "Torque Cap Outer Knurl (Vivid Orange)",
                    "layer": 11,
                    "base_z": 58.0,
                    "slide_factor": 3.1,
                    "z_pos": 58.0 + explode_gap * 3.1,
                    "color": "#f59e0b",
                    "opacity": 0.95,
                    "transparent": False,
                    "metalness": 0.3,
                    "roughness": 0.3
                },
                {
                    "id": "cap_inner",
                    "name": "Torque Cap Inner Drive (Metallic Silver)",
                    "layer": 12,
                    "base_z": 64.0,
                    "slide_factor": 3.5,
                    "z_pos": 64.0 + explode_gap * 3.5,
                    "color": "#94a3b8",
                    "opacity": 0.95,
                    "transparent": False,
                    "metalness": 0.8,
                    "roughness": 0.25
                },
                # 6. Side Mold Visualizer (Multi_Mold_Visualizer in OpenSCAD at X = -130)
                {
                    "id": "sys_mold_nanopore_left",
                    "name": "Wafer Mold Left Half (Teal)",
                    "layer": 20,
                    "base_x": -130.0,
                    "base_y": -55.0,
                    "base_z": -30.0,
                    "x_pos": -130.0 - explode_gap * 0.6,
                    "y_pos": -55.0,
                    "z_pos": -30.0,
                    "slide_dir": "x",
                    "slide_factor": -0.6,
                    "color": "#14b8a6",
                    "opacity": 0.95,
                    "transparent": False,
                    "metalness": 0.25,
                    "roughness": 0.25
                },
                {
                    "id": "sys_mold_nanopore_gasket",
                    "name": "Cast Wafer Gasket (Red Silicone)",
                    "layer": 20.5,
                    "base_x": -130.0,
                    "base_y": -55.0,
                    "base_z": -30.0,
                    "x_pos": -130.0,
                    "y_pos": -55.0,
                    "z_pos": -30.0,
                    "slide_dir": "none",
                    "slide_factor": 0.0,
                    "color": "#ef4444",
                    "opacity": 0.85,
                    "transparent": True,
                    "metalness": 0.15,
                    "roughness": 0.3
                },
                {
                    "id": "sys_mold_nanopore_right",
                    "name": "Wafer Mold Right Half (Teal)",
                    "layer": 21,
                    "base_x": -130.0,
                    "base_y": -55.0,
                    "base_z": -30.0,
                    "x_pos": -130.0 + explode_gap * 0.6,
                    "y_pos": -55.0,
                    "z_pos": -30.0,
                    "slide_dir": "x",
                    "slide_factor": 0.6,
                    "color": "#14b8a6",
                    "opacity": 0.95,
                    "transparent": False,
                    "metalness": 0.25,
                    "roughness": 0.25
                },
                {
                    "id": "sys_mold_amp_left",
                    "name": "Amplifier Mold Left Half (Steel Blue)",
                    "layer": 22,
                    "base_x": -130.0,
                    "base_y": 55.0,
                    "base_z": -30.0,
                    "x_pos": -130.0 - explode_gap * 0.6,
                    "y_pos": 55.0,
                    "z_pos": -30.0,
                    "slide_dir": "x",
                    "slide_factor": -0.6,
                    "color": "#0284c7",
                    "opacity": 0.95,
                    "transparent": False,
                    "metalness": 0.25,
                    "roughness": 0.25
                },
                {
                    "id": "sys_mold_amp_gasket",
                    "name": "Cast Amplifier Gasket (Red Silicone)",
                    "layer": 22.5,
                    "base_x": -130.0,
                    "base_y": 55.0,
                    "base_z": -30.0,
                    "x_pos": -130.0,
                    "y_pos": 55.0,
                    "z_pos": -30.0,
                    "slide_dir": "none",
                    "slide_factor": 0.0,
                    "color": "#ef4444",
                    "opacity": 0.85,
                    "transparent": True,
                    "metalness": 0.15,
                    "roughness": 0.3
                },
                {
                    "id": "sys_mold_amp_right",
                    "name": "Amplifier Mold Right Half (Steel Blue)",
                    "layer": 23,
                    "base_x": -130.0,
                    "base_y": 55.0,
                    "base_z": -30.0,
                    "x_pos": -130.0 + explode_gap * 0.6,
                    "y_pos": 55.0,
                    "z_pos": -30.0,
                    "slide_dir": "x",
                    "slide_factor": 0.6,
                    "color": "#0284c7",
                    "opacity": 0.95,
                    "transparent": False,
                    "metalness": 0.25,
                    "roughness": 0.25
                }
            ]
            fluidic_heights = {
                "zBot": 37.8 + explode_gap * 1.5,
                "zPore": 39.0 + explode_gap * 1.95,
                "zTop": 40.2 + explode_gap * 2.1
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
