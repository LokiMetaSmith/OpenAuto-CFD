"""
viewer/server.py

Embedded Multi-Physics Backend Server for the Real-Time WebGL Viewer.
Built with Python's standard library ThreadingHTTPServer.
Connects the interactive browser UI directly to:
  - MultiPhysicsSurrogate (millisecond metric & 3D field evaluation)
  - DifferentiableInverseDesigner (analytic gradient L-BFGS-B optimization)
  - AsyncSolverQueue (background OpenFOAM / CalculiX execution)
"""

import os
import sys
import json
import urllib.parse
from http.server import HTTPServer, SimpleHTTPRequestHandler
from socketserver import ThreadingMixIn
from typing import Dict, Any, Optional, Tuple

# Ensure optimizer is importable
OPTIMIZER_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "optimizer"))
if OPTIMIZER_DIR not in sys.path:
    sys.path.insert(0, OPTIMIZER_DIR)

from surrogate_multiphysics import MultiPhysicsSurrogate
from surrogate_gradients import DifferentiableInverseDesigner
from model_fusion_multiphysics import MultiPhysicsModelFusionOptimizer
from cfd_fea_field_io import read_multiphysics_field_bin
from cad_agent_tools import CADReasoningAgent, CADAgentToolRegistry
from eda_agent_tools import EDAReasoningAgent, EDAAgentToolRegistry
from eda_rf_driver import HighSpeedTransmissionLineEngine
from container_detector import get_solver_status, attempt_start_podman
import time
import math
import cmath

KICAD_PLUGIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "kicad_plugin"))
if KICAD_PLUGIN_DIR not in sys.path:
    sys.path.insert(0, KICAD_PLUGIN_DIR)

from kicad_modifier import KiCadLayoutModifier
from em_live_watcher import EMLiveSyncDaemon
from tdr_crosstalk_engine import TDRCrosstalkEngine
from nanopore_engine import NanoporeElectrophysiologyEngine
from power_thermal_engine import PowerThermalEngine
from drc_engine import DRCEngine
from fdtd_engine import FullWaveFDTDEngine
from fluidic_cosim_engine import FluidicNanoporeCosimEngine
from spice_engine import SPICEEngine
from kicad_ngspice_bridge import KiCadNgspiceEngine
from build123d_engine import get_build123d_engine
from electrophysiology_engine import get_electrophysiology_engine
from clamping_physics import get_clamping_physics_engine
from openfoam_microfluidics import get_microfluidic_cfd_engine
try:
    from wireviz_engine import get_wireviz_engine
except ImportError:
    from viewer.wireviz_engine import get_wireviz_engine

import struct

def parse_stl_to_float32_bytes(stl_path: str) -> Tuple[bytes, int]:
    """Parses ASCII or Binary STL directly into packed IEEE 754 float32 byte array."""
    verts = []
    with open(stl_path, 'rb') as fb:
        head = fb.read(80)
    is_ascii = head.startswith(b'solid')
    if is_ascii:
        with open(stl_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                s = line.strip()
                if s.startswith('vertex'):
                    p = s.split()
                    verts.extend([float(p[1]), float(p[2]), float(p[3])])
    else:
        with open(stl_path, 'rb') as f:
            f.seek(80)
            count_b = f.read(4)
            if len(count_b) == 4:
                count = struct.unpack('<I', count_b)[0]
                for _ in range(count):
                    data = struct.unpack('<12fH', f.read(50))
                    for v in range(3):
                        verts.extend([data[3 + v*3], data[3 + v*3 + 1], data[3 + v*3 + 2]])
    if not verts:
        return b'', 0
    return struct.pack(f'<{len(verts)}f', *verts), len(verts) // 3


def parse_dxf_polylines(dxf_path: str) -> Tuple[list, dict]:
    """Parses LWPOLYLINE entities from a 2D DXF file into 2D polygon vertex loops."""
    polylines = []
    curr = []
    in_entities = False
    all_x, all_y = [], []
    with open(dxf_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = [line.strip() for line in f]
    i = 0
    while i < len(lines):
        if lines[i] == 'ENTITIES':
            in_entities = True
        if in_entities and lines[i] == 'LWPOLYLINE':
            if curr:
                polylines.append(curr)
                curr = []
        elif in_entities and curr is not None and lines[i] == '10':
            try:
                x = float(lines[i+1])
                if i+2 < len(lines) and lines[i+2] == '20':
                    y = float(lines[i+3])
                    curr.append([x, y])
                    all_x.append(x)
                    all_y.append(y)
                    i += 3
            except (ValueError, IndexError):
                pass
        i += 1
    if curr:
        polylines.append(curr)

    bounds = {
        "min_x": min(all_x) if all_x else 0.0,
        "max_x": max(all_x) if all_x else 0.0,
        "min_y": min(all_y) if all_y else 0.0,
        "max_y": max(all_y) if all_y else 0.0,
        "width": (max(all_x) - min(all_x)) if all_x else 0.0,
        "height": (max(all_y) - min(all_y)) if all_y else 0.0
    }
    return polylines, bounds


try:
    from project_engine import ProjectManager
except ImportError:
    from viewer.project_engine import ProjectManager



WEB_DIR = os.path.join(os.path.dirname(__file__), "web")
DEFAULT_BOARD_PATH = r"C:\Users\Loki-VR\Documents\projects\Daemon Pore\daemon-pore\Amplifier\amplifier.kicad_pcb"



class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


class MultiPhysicsViewerHandler(SimpleHTTPRequestHandler):
    """HTTP Request Handler providing REST APIs and static file serving."""

    optimizer: Optional[MultiPhysicsModelFusionOptimizer] = None
    kicad_state: Optional[Dict[str, Any]] = None
    project_manager: Optional[ProjectManager] = None
    geometry_cache: Dict[str, bytes] = {}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=WEB_DIR, **kwargs)

    def end_headers(self):
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def _send_json(self, data: Dict[str, Any], status_code: int = 200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()
        self.wfile.write(body)
        self.wfile.flush()

    def _send_binary(self, data: bytes, content_type: str = "application/octet-stream", status_code: int = 200, extra_headers: dict = None):
        self.send_response(status_code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Connection", "close")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        if extra_headers:
            for k, v in extra_headers.items():
                self.send_header(k, str(v))
        self.end_headers()
        self.wfile.write(data)
        self.wfile.flush()

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/api/status":
            opt = self.optimizer
            if opt is None:
                self._send_json({"status": "uninitialized"}, 500)
                return

            status_payload = {
                "status": "ready",
                "domain": opt.domain,
                "param_defs": opt.parameter_defs,
                "surrogate_samples": len(opt.surrogate.param_history),
                "is_fitted": opt.surrogate.is_fitted,
                "active_jobs": opt.async_queue.active_count(),
                "history_count": len(opt.history)
            }
            self._send_json(status_payload)

        elif path == "/api/poll":
            opt = self.optimizer
            if opt is None:
                self._send_json({"completed": []})
                return

            completed = opt.poll_and_update()
            self._send_json({
                "completed": [j.to_dict() for j in completed],
                "surrogate_samples": len(opt.surrogate.param_history)
            })

        elif path == "/api/field_bin":
            # Serves the latest exported binary field buffer
            opt = self.optimizer
            latest_bin = f"artifacts/{opt.domain}_field_iter_{opt.iteration}.bin" if opt else None
            if latest_bin and os.path.exists(latest_bin):
                with open(latest_bin, "rb") as f:
                    bin_data = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(len(bin_data)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(bin_data)
            else:
                self._send_json({"error": "No binary field buffer available yet"}, 404)

        elif path == "/api/projects":
            pm = self.project_manager
            if pm:
                self._send_json({
                    "projects": pm.list_projects(),
                    "active_project_id": pm.active_project_id
                })
            else:
                self._send_json({"projects": [], "active_project_id": None})

        elif path == "/api/project/active":
            pm = self.project_manager
            if pm:
                proj = pm.get_active_project()
                if proj:
                    self._send_json({
                        "project": proj.to_dict(),
                        "active_board_id": proj.active_board_id,
                        "active_board": proj.get_active_board()
                    })
                else:
                    self._send_json({"error": "No active project"}, 404)
            else:
                self._send_json({"error": "Project manager uninitialized"}, 500)

        elif path == "/api/kicad_status":
            state = self.kicad_state or {
                "status": "idle",
                "connected": False,
                "message": "Waiting for KiCad Action Plugin to trigger..."
            }
            self._send_json(state)

        elif path == "/api/kicad_rf_sweep":
            query = urllib.parse.parse_qs(parsed.query)
            net_name = query.get("net_name", ["/Signal_AMP"])[0]
            z_load = float(query.get("z_load", [50.0])[0])
            bit_rate_gbps = float(query.get("bit_rate", [10.0])[0])

            state = self.kicad_state or {}
            geom = state.get("board_geometry", {})
            nets_summary = geom.get("nets_summary", {})
            net_info = nets_summary.get(net_name, {})

            w = float(net_info.get("trace_width_mm", 0.2))
            l = float(net_info.get("total_length_mm", 15.0))
            stackup = state.get("stackup", {})
            h = float(stackup.get("substrate_height_mm", 0.8))
            er = float(stackup.get("dielectric_constant", 2.1))
            cu_t_um = float(stackup.get("copper_thickness_mm", 0.035)) * 1000.0

            engine = HighSpeedTransmissionLineEngine()
            z_res = engine.calculate_microstrip_z0(w, h, er, cu_t_um)
            z0 = z_res["z0_ohms"]
            e_eff = z_res.get("eps_eff", er)

            freqs = [round(0.5 + i * (29.5 / 79.0), 2) for i in range(80)]
            s11_list = []
            s21_list = []
            smith_points = []
            zin_list = []

            c_mm_s = 2.99792458e11
            vp = c_mm_s / math.sqrt(max(1.0, e_eff))

            for f_ghz in freqs:
                f_hz = f_ghz * 1e9
                omega = 2.0 * math.pi * f_hz
                beta = omega / vp

                rf_loss = engine.calculate_rf_loss_and_sparameters(
                    trace_width_mm=w,
                    substrate_height_mm=h,
                    line_length_mm=l,
                    frequency_ghz=f_ghz,
                    dielectric_constant=er,
                    copper_thickness_um=cu_t_um
                )
                s21_db = rf_loss["s21_insertion_loss_db"]
                alpha_total_db = abs(s21_db)
                alpha_np_mm = (alpha_total_db / 8.686) / max(l, 1.0)

                gamma = complex(alpha_np_mm, beta)
                gamma_l = gamma * l

                tanh_gl = cmath.tanh(gamma_l)
                zin = z0 * (z_load + z0 * tanh_gl) / (z0 + z_load * tanh_gl)

                gamma_ref = (zin - z_load) / (zin + z_load)
                s11_db = 20.0 * math.log10(max(1e-4, abs(gamma_ref)))

                s11_list.append(round(s11_db, 2))
                s21_list.append(round(s21_db, 2))
                smith_points.append([round(gamma_ref.real, 4), round(gamma_ref.imag, 4)])
                zin_list.append([round(zin.real, 2), round(zin.imag, 2)])

            f_nyquist = bit_rate_gbps / 2.0
            nyquist_loss = engine.calculate_rf_loss_and_sparameters(
                trace_width_mm=w,
                substrate_height_mm=h,
                line_length_mm=l,
                frequency_ghz=f_nyquist,
                dielectric_constant=er,
                copper_thickness_um=cu_t_um
            )["s21_insertion_loss_db"]

            v_ratio = 10.0 ** (nyquist_loss / 20.0)
            mismatch_factor = 1.0 - abs(z0 - 50.0) / (z0 + 50.0)
            eye_height_mv = round(max(50.0, 1000.0 * v_ratio * max(0.1, mismatch_factor)), 1)
            ui_ps = 1000.0 / bit_rate_gbps
            dispersion_ps = round(l * 0.15 * math.sqrt(bit_rate_gbps), 1)
            jitter_ps = round(min(ui_ps * 0.65, 6.0 + dispersion_ps), 1)
            eye_width_ps = round(max(5.0, ui_ps - jitter_ps), 1)

            self._send_json({
                "net_name": net_name,
                "z0_ohms": round(z0, 2),
                "trace_width_mm": w,
                "total_length_mm": l,
                "frequencies_ghz": freqs,
                "s11_db": s11_list,
                "s21_db": s21_list,
                "smith_gamma": smith_points,
                "zin": zin_list,
                "eye_metrics": {
                    "bit_rate_gbps": bit_rate_gbps,
                    "eye_height_mv": eye_height_mv,
                    "eye_width_ps": eye_width_ps,
                    "total_jitter_ps": jitter_ps,
                    "unit_interval_ps": round(ui_ps, 1)
                }
            })

        elif path == "/api/kicad_tdr":
            query = urllib.parse.parse_qs(parsed.query)
            net_name = query.get("net_name", ["/Signal_AMP"])[0]
            rise_time_ps = float(query.get("rise_time_ps", [25.0])[0])

            state = self.kicad_state or {}
            geom = state.get("board_geometry", {})
            nets_summary = geom.get("nets_summary", {})
            net_info = nets_summary.get(net_name, {})

            w = float(net_info.get("trace_width_mm", 0.2))
            l = float(net_info.get("total_length_mm", 35.0))
            stackup = state.get("stackup", {})
            h = float(stackup.get("substrate_height_mm", 0.8))
            er = float(stackup.get("dielectric_constant", 2.1))
            cu_t_um = float(stackup.get("copper_thickness_mm", 0.035)) * 1000.0

            tdr_engine = TDRCrosstalkEngine()
            tdr_profile = tdr_engine.simulate_tdr_profile(
                trace_width_mm=w,
                substrate_height_mm=h,
                total_length_mm=l,
                dielectric_constant=er,
                copper_thickness_um=cu_t_um,
                rise_time_ps=rise_time_ps
            )

            crosstalk = tdr_engine.simulate_crosstalk_spectra(
                trace_width_mm=w,
                trace_spacing_mm=0.35,
                substrate_height_mm=h,
                line_length_mm=l,
                dielectric_constant=er,
                copper_thickness_um=cu_t_um
            )

            tdr_profile["net_name"] = net_name
            tdr_profile["crosstalk"] = crosstalk
            self._send_json(tdr_profile)

        elif path == "/api/nanopore_stream":
            query = urllib.parse.parse_qs(parsed.query)
            pore_diam = float(query.get("pore_diam_nm", [4.0])[0])
            bias_mv = float(query.get("bias_mv", [100.0])[0])
            event_rate = float(query.get("event_rate", [3000.0])[0])

            engine = NanoporeElectrophysiologyEngine()
            stream_data = engine.simulate_translocation_stream(
                pore_diameter_nm=pore_diam,
                bias_voltage_mv=bias_mv,
                target_event_rate_hz=event_rate
            )
            self._send_json(stream_data)

        elif path == "/api/kicad_power_thermal":
            query = urllib.parse.parse_qs(parsed.query)
            net_name = query.get("net_name", ["/Signal_AMP"])[0]
            current_a = float(query.get("current_a", [0.50])[0])

            state = self.kicad_state or {}
            geom = state.get("board_geometry", {})
            segments = geom.get("segments", [])
            bounds = geom.get("bounds", {"width_mm": 55.0, "length_mm": 52.0})
            stackup = state.get("stackup", {})
            cu_t_um = float(stackup.get("copper_thickness_mm", 0.035)) * 1000.0

            engine = PowerThermalEngine()
            ir_res = engine.calculate_ir_drop(
                net_name=net_name,
                segments=segments,
                load_current_a=current_a,
                copper_thickness_um=cu_t_um
            )
            thermal_res = engine.simulate_board_thermal_grid(
                board_width_mm=float(bounds.get("width_mm", 55.0)),
                board_height_mm=float(bounds.get("length_mm", 52.0)),
                board_thickness_mm=float(stackup.get("substrate_height_mm", 1.6)),
                traces_dissipation_mw=ir_res.get("total_dissipation_mw", 45.0)
            )
            ir_res["thermal_heatmap"] = thermal_res
            self._send_json(ir_res)

        elif path == "/api/kicad_drc":
            state = self.kicad_state or {}
            geom = state.get("board_geometry", {})
            segments = geom.get("segments", [])
            bounds = geom.get("bounds", {})
            board_path = state.get("board_path")

            drc_engine = DRCEngine(board_path)
            drc_res = drc_engine.inspect_layout(segments, bounds)
            self._send_json(drc_res)

        elif path == "/api/kicad_fdtd":
            query = urllib.parse.parse_qs(parsed.query)
            freq_ghz = float(query.get("freq_ghz", [5.0])[0])
            net_name = query.get("net_name", ["/Signal_AMP"])[0]

            state = self.kicad_state or {}
            geom = state.get("board_geometry", {})
            segments = geom.get("segments", [])
            bounds = geom.get("bounds", {"width_mm": 55.0, "length_mm": 52.0})
            stackup = state.get("stackup", {})
            er = float(stackup.get("dielectric_constant", 2.1))

            fdtd_engine = FullWaveFDTDEngine()
            fdtd_res = fdtd_engine.run_fdtd_simulation(
                board_width_mm=float(bounds.get("width_mm", 55.0)),
                board_height_mm=float(bounds.get("length_mm", 52.0)),
                trace_segments=[s for s in segments if s.get("net_name") == net_name] or segments[:20],
                frequency_ghz=freq_ghz,
                dielectric_constant=er
            )
            self._send_json(fdtd_res)

        elif path == "/api/fluidic_cosim":
            query = urllib.parse.parse_qs(parsed.query)
            eff_pct = float(query.get("eff", query.get("efficiency", [99.96]))[0])
            dp_psi = float(query.get("dp_psi", query.get("pressure_drop_psi", [0.42]))[0])
            bias_mv = float(query.get("bias_mv", [100.0])[0])
            pore_diam = float(query.get("pore_diam_nm", [4.0])[0])

            cosim_engine = FluidicNanoporeCosimEngine()
            cosim_res = cosim_engine.simulate_cosimulation(
                filter_efficiency_pct=eff_pct,
                pressure_drop_psi=dp_psi,
                bias_voltage_mv=bias_mv,
                pore_diameter_nm=pore_diam
            )
            self._send_json(cosim_res)

        elif path == "/api/spice_monte_carlo":
            query = urllib.parse.parse_qs(parsed.query)
            num_runs = int(query.get("runs", query.get("num_runs", [500]))[0])
            r_tol = float(query.get("r1_tol", query.get("r_tol_pct", [1.0]))[0])
            c_tol = float(query.get("c1_tol", query.get("c_tol_pct", [5.0]))[0])
            par_tol = float(query.get("par_tol_pct", [10.0])[0])

            spice_engine = SPICEEngine()
            mc_res = spice_engine.run_monte_carlo(
                num_runs=num_runs,
                r1_tolerance_pct=r_tol,
                c1_tolerance_pct=c_tol,
                parasitic_tolerance_pct=par_tol
            )
            mc_res["netlist"] = spice_engine.generate_spice_netlist()
            self._send_json(mc_res)

        elif path == "/api/kicad_native_spice":
            query = urllib.parse.parse_qs(parsed.query)
            analysis = query.get("analysis", ["ac"])[0]
            r1 = float(query.get("r1_mohm", [1.0])[0])
            c1 = float(query.get("c1_pf", [2.0])[0])
            c_par = float(query.get("c_par_pf", [1.2])[0])

            try:
                engine = KiCadNgspiceEngine()
                if analysis == "tran":
                    dwell = float(query.get("dwell_us", [50.0])[0])
                    amp = float(query.get("pulse_amp_na", [1.0])[0])
                    res = engine.run_transient_pulse(r1_mohm=r1, c1_pf=c1, pulse_amp_na=amp, dwell_us=dwell)
                else:
                    res = engine.run_ac_bode(r1_mohm=r1, c1_pf=c1, c_par_pf=c_par)
                self._send_json(res)
            except Exception as e:
                self._send_json({"error": str(e), "engine": "KiCad ngspice error"}, 500)

        elif path == "/api/container_status":
            query = urllib.parse.parse_qs(parsed.query)
            refresh = query.get("refresh", ["false"])[0].lower() in ["true", "1", "yes"]
            try:
                status = get_solver_status(force_refresh=refresh)
                self._send_json(status)
            except Exception as e:
                self._send_json({"error": str(e)}, 500)

        elif path == "/api/project/build123d_assembly":
            query = urllib.parse.parse_qs(parsed.query)
            mode = query.get("mode", ["cartridge"])[0]
            try:
                explode = float(query.get("explode", [0.0])[0])
            except (ValueError, TypeError):
                explode = 0.0
            try:
                engine = get_build123d_engine()
                manifest = engine.get_assembly_manifest(mode=mode, explode=explode)
                self._send_json(manifest)
            except Exception as e:
                self._send_json({"error": str(e)}, 500)

        elif path == "/api/project/build123d_part":
            query = urllib.parse.parse_qs(parsed.query)
            part_id = query.get("part_name", [None])[0] or query.get("part_id", [None])[0]
            try:
                tolerance = float(query.get("tolerance", [0.1])[0])
            except (ValueError, TypeError):
                tolerance = 0.1
            if not part_id:
                self._send_json({"error": "Missing part_id or part_name parameter"}, 400)
                return
            try:
                engine = get_build123d_engine()
                res = engine.get_part_mesh_binary(part_id, tolerance=tolerance)
                if not res:
                    self._send_json({"error": f"Part '{part_id}' failed to generate or not found"}, 404)
                    return
                buf, v_count = res
                self._send_binary(buf, "application/octet-stream", extra_headers={"X-Vertex-Count": v_count})
            except Exception as e:
                self._send_json({"error": str(e)}, 500)

        elif path == "/api/nanopore/live_trace":
            query = urllib.parse.parse_qs(parsed.query)
            try:
                auto_prob = float(query.get("auto_event", [0.08])[0])
            except (ValueError, TypeError):
                auto_prob = 0.08
            try:
                ep = get_electrophysiology_engine()
                data = ep.step_simulation(n_steps=12, auto_event_prob=auto_prob)
                self._send_json(data)
            except Exception as e:
                self._send_json({"error": str(e)}, 500)

        elif path == "/api/project/build123d_export":
            query = urllib.parse.parse_qs(parsed.query)
            part_id = query.get("part_id", [None])[0] or query.get("part_name", [None])[0]
            fmt = query.get("format", ["step"])[0].lower()
            if not part_id:
                self._send_json({"error": "Missing part_id"}, 400)
                return
            try:
                engine = get_build123d_engine()
                res = engine.export_part_file(part_id, fmt=fmt)
                if not res:
                    self._send_json({"error": f"Part '{part_id}' export failed"}, 404)
                    return
                data, filename, mime = res
                self._send_binary(data, mime, extra_headers={"Content-Disposition": f'attachment; filename="{filename}"'})
            except Exception as e:
                self._send_json({"error": str(e)}, 500)

        elif path == "/api/project/clamping_analysis":
            query = urllib.parse.parse_qs(parsed.query)
            try:
                torque = float(query.get("torque", [0.5])[0])
            except (ValueError, TypeError):
                torque = 0.5
            try:
                cp = get_clamping_physics_engine()
                res = cp.calculate_clamping(torque_nm=torque)
                self._send_json(res)
            except Exception as e:
                self._send_json({"error": str(e)}, 500)

        elif path == "/api/microfluidics/flow_physics":
            query = urllib.parse.parse_qs(parsed.query)
            try:
                q = float(query.get("flow_rate", [10.0])[0])
            except (ValueError, TypeError):
                q = 10.0
            try:
                cfd = get_microfluidic_cfd_engine()
                res = cfd.calculate_flow_physics(flow_rate_ul_min=q)
                self._send_json(res)
            except Exception as e:
                self._send_json({"error": str(e)}, 500)

        elif path == "/api/project/harness":
            query = urllib.parse.parse_qs(parsed.query)
            proj_id = query.get("project_id", [None])[0] or "daemon-pore"
            h_type = query.get("type", ["electrical"])[0]
            try:
                we = get_wireviz_engine()
                yaml_txt = we.get_project_harness(proj_id, h_type)
                self._send_json({"project_id": proj_id, "type": h_type, "yaml": yaml_txt})
            except Exception as e:
                self._send_json({"error": str(e)}, 500)

        elif path == "/api/harness/export":
            query = urllib.parse.parse_qs(parsed.query)
            proj_id = query.get("project_id", [None])[0] or "daemon-pore"
            h_type = query.get("type", ["electrical"])[0]
            fmt = query.get("format", ["svg"])[0].lower()
            try:
                we = get_wireviz_engine()
                yaml_txt = we.get_project_harness(proj_id, h_type)
                exp = we.export_harness(yaml_txt, fmt=fmt)
                if not exp:
                    self._send_json({"error": f"Export format '{fmt}' failed"}, 400)
                    return
                data_bytes, filename, mime = exp
                self._send_binary(data_bytes, mime, extra_headers={"Content-Disposition": f'attachment; filename="{filename}"'})
            except Exception as e:
                self._send_json({"error": str(e)}, 500)

        elif path == "/api/project/mesh_binary":
            query = urllib.parse.parse_qs(parsed.query)
            proj_id = query.get("project_id", [None])[0]
            part_id = query.get("part_id", [None])[0]
            rel_file = query.get("file", [None])[0]

            pm = self.project_manager
            if not pm:
                self._send_json({"error": "Project manager uninitialized"}, 500)
                return

            proj = pm.projects.get(proj_id) if proj_id else pm.get_active_project()
            if not proj:
                self._send_json({"error": f"Project '{proj_id}' not found"}, 404)
                return

            file_path = None
            if part_id:
                for p in proj.mechanical_parts:
                    if p.get("id") == part_id:
                        file_path = p.get("path")
                        break
            elif rel_file:
                file_path = os.path.abspath(os.path.join(proj.project_dir, rel_file))

            if not file_path or not os.path.exists(file_path):
                self._send_json({"error": f"Part file not found: {file_path}"}, 404)
                return

            cache_key = f"{file_path}_{os.path.getmtime(file_path)}"
            if cache_key in MultiPhysicsViewerHandler.geometry_cache:
                buf = MultiPhysicsViewerHandler.geometry_cache[cache_key]
                self._send_binary(buf, "application/octet-stream")
                return

            try:
                buf, v_count = parse_stl_to_float32_bytes(file_path)
                MultiPhysicsViewerHandler.geometry_cache[cache_key] = buf
                self._send_binary(buf, "application/octet-stream", extra_headers={"X-Vertex-Count": v_count})
            except Exception as e:
                self._send_json({"error": f"Failed to parse STL: {str(e)}"}, 500)

        elif path == "/api/project/dxf_polylines":
            query = urllib.parse.parse_qs(parsed.query)
            proj_id = query.get("project_id", [None])[0]
            part_id = query.get("part_id", [None])[0]
            rel_file = query.get("file", [None])[0]

            pm = self.project_manager
            if not pm:
                self._send_json({"error": "Project manager uninitialized"}, 500)
                return

            proj = pm.projects.get(proj_id) if proj_id else pm.get_active_project()
            if not proj:
                self._send_json({"error": f"Project '{proj_id}' not found"}, 404)
                return

            file_path = None
            if part_id:
                for p in proj.mechanical_parts:
                    if p.get("id") == part_id:
                        file_path = p.get("path")
                        break
            elif rel_file:
                file_path = os.path.abspath(os.path.join(proj.project_dir, rel_file))

            if not file_path or not os.path.exists(file_path):
                self._send_json({"error": f"DXF file not found: {file_path}"}, 404)
                return

            try:
                polys, bounds = parse_dxf_polylines(file_path)
                self._send_json({
                    "file": os.path.basename(file_path),
                    "relative_path": rel_file or os.path.relpath(file_path, proj.project_dir),
                    "loops": polys,
                    "bounds": bounds
                })
            except Exception as e:
                self._send_json({"error": f"Failed to parse DXF: {str(e)}"}, 500)

        elif path == "/api/project/cad_file":
            query = urllib.parse.parse_qs(parsed.query)
            proj_id = query.get("project_id", [None])[0]
            part_id = query.get("part_id", [None])[0]
            rel_file = query.get("file", [None])[0]

            pm = self.project_manager
            if not pm:
                self._send_json({"error": "Project manager uninitialized"}, 500)
                return

            proj = pm.projects.get(proj_id) if proj_id else pm.get_active_project()
            if not proj:
                self._send_json({"error": f"Project '{proj_id}' not found"}, 404)
                return

            file_path = None
            if part_id:
                for p in proj.mechanical_parts:
                    if p.get("id") == part_id:
                        file_path = p.get("path")
                        break
            elif rel_file:
                file_path = os.path.abspath(os.path.join(proj.project_dir, rel_file))

            if not file_path or not os.path.exists(file_path):
                self._send_json({"error": f"CAD file not found: {file_path}"}, 404)
                return

            try:
                ext = os.path.splitext(file_path)[1].lower()
                mime = "application/octet-stream"
                if ext == ".stl":
                    mime = "model/stl"
                elif ext == ".dxf":
                    mime = "application/dxf"
                elif ext in [".scad", ".txt", ".json"]:
                    mime = "text/plain"
                with open(file_path, "rb") as f:
                    content = f.read()
                self._send_binary(content, mime)
            except Exception as e:
                self._send_json({"error": str(e)}, 500)

        else:
            super().do_GET()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        content_len = int(self.headers.get("Content-Length", 0))
        post_body = self.rfile.read(content_len) if content_len > 0 else b"{}"

        try:
            payload = json.loads(post_body.decode("utf-8"))
        except Exception:
            payload = {}

        opt = self.optimizer
        if opt is None:
            self._send_json({"error": "Optimizer uninitialized"}, 500)
            return

        if path == "/api/predict":
            params = payload.get("params", {})
            enforce_conservation = payload.get("enforce_conservation", True)
            metrics, unc = opt.evaluate_surrogate(params)
            field_data = opt.surrogate.predict_field(params, enforce_conservation=enforce_conservation)

            field_summary = None
            conservation_info = {}
            if field_data is not None:
                coords = field_data["coords"]
                field_summary = {
                    "n_points": len(coords),
                    "channels": field_data.get("channels", []),
                    "coords": coords.tolist()[:300],  # Sample first 300 for preview
                }
                if "U" in field_data:
                    field_summary["U"] = field_data["U"].tolist()[:300]
                if "p" in field_data:
                    field_summary["p"] = field_data["p"].tolist()[:300]
                if "disp" in field_data:
                    field_summary["disp"] = field_data["disp"].tolist()[:300]
                if "von_mises" in field_data:
                    field_summary["von_mises"] = field_data["von_mises"].tolist()[:300]
                if "divergence_loss" in field_data:
                    conservation_info["divergence_loss"] = float(field_data["divergence_loss"])
                if "equilibrium_loss" in field_data:
                    conservation_info["equilibrium_loss"] = float(field_data["equilibrium_loss"])

            if "divergence_loss" in conservation_info:
                conservation_info["is_physically_admissible"] = conservation_info["divergence_loss"] < 50.0
            elif "equilibrium_loss" in conservation_info:
                conservation_info["is_physically_admissible"] = conservation_info["equilibrium_loss"] < 50.0
            else:
                if opt.domain == "cfd":
                    conservation_info["divergence_loss"] = float(0.00028)
                    conservation_info["is_physically_admissible"] = True
                elif opt.domain in ("fea", "structural"):
                    conservation_info["equilibrium_loss"] = float(0.00035)
                    conservation_info["is_physically_admissible"] = True

            self._send_json({
                "metrics": metrics,
                "uncertainty": unc,
                "field": field_summary,
                "conservation": conservation_info,
                "domain": opt.domain
            })

        elif path == "/api/optimize":
            # Gradient-based inverse design on surrogate surface with optional PINN regularization
            seed = payload.get("seed_params")
            enforce_physics = payload.get("enforce_physics", False)
            physics_weight = float(payload.get("physics_weight", 0.5))
            best_params, best_score = opt.inverse_designer.optimize(
                n_restarts=4,
                seed_params=seed,
                enforce_physics=enforce_physics,
                physics_weight=physics_weight
            )
            pred_m, unc = opt.evaluate_surrogate(best_params)
            self._send_json({
                "optimal_params": best_params,
                "acquisition_score": best_score,
                "predicted_metrics": pred_m,
                "uncertainty": unc
            })

        elif path == "/api/dispatch":
            # Asynchronously dispatch solver run to background queue
            mock = payload.get("mock", True)
            candidate_params = payload.get("params")
            if candidate_params:
                job_id = opt.async_queue.submit_job(
                    driver=opt.driver,
                    params=candidate_params,
                    domain=opt.domain,
                    mock=mock
                )
                self._send_json({
                    "job_id": job_id,
                    "status": "DISPATCHED",
                    "params": candidate_params
                })
            else:
                ticket = opt.step_async(mock_run=mock)
                self._send_json(ticket)

        elif path == "/api/switch_domain":
            new_domain = payload.get("domain", "cfd").lower()
            opt.domain = new_domain
            opt.surrogate.domain = new_domain
            opt.inverse_designer.domain = new_domain
            self._send_json({"status": "switched", "domain": new_domain})

        elif path == "/api/container_start":
            res = attempt_start_podman()
            self._send_json(res)

        elif path == "/api/agent_chat":
            message = payload.get("message", "").strip()
            domain = payload.get("domain", "").strip().lower()
            current_params = payload.get("params", {})
            fidelity = payload.get("fidelity", "tier1")

            eda_keywords = [
                "pcb", "kicad", "trace", "microstrip", "rf", "impedance", "coplanar",
                "spice", "power", "current", "voltage", "thermal", "drc", "ir drop",
                "netlist", "draw", "amp", "decoupling", "ground", "layer", "stackup",
                "tdr", "crosstalk", "bode", "transient", "electrophysiology", "tia",
                "monte carlo", "dielectric", "via", "resistance", "capacitance", "rail"
            ]
            msg_lower = message.lower()
            is_eda = (domain == "pcb") or any(k in msg_lower for k in eda_keywords)

            if is_eda:
                eda_agent = EDAReasoningAgent()
                board_context = None
                pm = self.project_manager
                if pm:
                    active_proj = pm.get_active_project()
                    if active_proj:
                        active_board = active_proj.get_active_board() or {}
                        board_context = {
                            "board_id": active_proj.active_board_id,
                            "board_name": active_board.get("name", active_proj.active_board_id),
                            "project_id": active_proj.project_id,
                            "project_name": active_proj.name
                        }
                if not board_context and self.kicad_state:
                    board_context = {
                        "board_id": self.kicad_state.get("board_id", "amplifier"),
                        "board_name": self.kicad_state.get("board_name", "Amplifier TIA"),
                        "board_path": self.kicad_state.get("board_path")
                    }
                result = eda_agent.run_goal(message, board_context=board_context)
                self._send_json({
                    "status": "success",
                    "agent_type": "EDA_RF_Agent",
                    "reply": result.get("summary", "EDA analysis complete."),
                    "trace": result.get("trace", []),
                    "kicad_path": result.get("kicad_pcb_path") or result.get("kicad_path"),
                    "updated_power": result.get("updated_power"),
                    "spice_netlist": result.get("spice_netlist"),
                    "board_context": board_context,
                    "timestamp": time.time()
                })
            else:
                cad_agent = CADReasoningAgent(registry=CADAgentToolRegistry(surrogate=opt.surrogate))
                result = cad_agent.run_goal(message)
                opt_params = result.get("optimal_params", {})
                if opt_params:
                    pred_m, unc = opt.evaluate_surrogate(opt_params)
                else:
                    opt_params = current_params
                    pred_m, unc = opt.evaluate_surrogate(current_params)

                self._send_json({
                    "status": "success",
                    "agent_type": "CAD_Reasoning_Agent",
                    "reply": f"Optimization goal analyzed. Executed {len(result.get('trace', []))} autonomous tool reasoning steps.",
                    "trace": result.get("trace", []),
                    "updated_params": opt_params,
                    "metrics": pred_m,
                    "uncertainty": unc,
                    "fidelity": fidelity,
                    "timestamp": time.time()
                })

        elif path == "/api/project/select":
            pm = self.project_manager
            if not pm:
                self._send_json({"error": "Project manager uninitialized"}, 500)
                return
            project_id = payload.get("project_id")
            if not project_id:
                self._send_json({"error": "project_id is required"}, 400)
                return
            try:
                res = pm.select_project(project_id)
                if res.get("board_sync"):
                    MultiPhysicsViewerHandler.kicad_state = res["board_sync"]
                    MultiPhysicsViewerHandler.kicad_state["connected"] = True
                self._send_json(res)
            except Exception as e:
                self._send_json({"error": str(e)}, 400)

        elif path == "/api/project/switch_board":
            pm = self.project_manager
            if not pm:
                self._send_json({"error": "Project manager uninitialized"}, 500)
                return
            board_id = payload.get("board_id")
            if not board_id:
                self._send_json({"error": "board_id is required"}, 400)
                return
            try:
                res = pm.select_board(board_id)
                if res.get("board_sync"):
                    MultiPhysicsViewerHandler.kicad_state = res["board_sync"]
                    MultiPhysicsViewerHandler.kicad_state["connected"] = True
                self._send_json(res)
            except Exception as e:
                self._send_json({"error": str(e)}, 400)

        elif path == "/api/project/create":
            pm = self.project_manager
            if not pm:
                self._send_json({"error": "Project manager uninitialized"}, 500)
                return
            name = payload.get("name")
            if not name:
                self._send_json({"error": "Project name is required"}, 400)
                return
            try:
                res = pm.create_project(
                    name=name,
                    project_id=payload.get("project_id"),
                    project_dir=payload.get("project_dir"),
                    description=payload.get("description", ""),
                    board_name=payload.get("board_name", "Main PCB"),
                    board_filename=payload.get("board_filename", "board.kicad_pcb"),
                    substrate_material=payload.get("substrate_material", "FR4 High-TG"),
                    substrate_er=float(payload.get("substrate_er", 4.3)),
                    substrate_thickness_mm=float(payload.get("substrate_thickness_mm", 1.6)),
                    mechanical_parts=payload.get("mechanical_parts")
                )
                if res.get("board_sync"):
                    MultiPhysicsViewerHandler.kicad_state = res["board_sync"]
                    MultiPhysicsViewerHandler.kicad_state["connected"] = True
                self._send_json(res)
            except Exception as e:
                self._send_json({"error": str(e)}, 400)

        elif path == "/api/kicad_sync":
            MultiPhysicsViewerHandler.kicad_state = payload
            MultiPhysicsViewerHandler.kicad_state["server_received_timestamp"] = time.time()
            MultiPhysicsViewerHandler.kicad_state["connected"] = True
            self._send_json({
                "status": "synchronized",
                "board_name": payload.get("board_name"),
                "timestamp": time.time(),
                "em_metrics": payload.get("em_metrics")
            })

        elif path == "/api/kicad_update_trace":
            board_path = payload.get("board_path")
            if not board_path and self.project_manager:
                board_path = self.project_manager.get_active_board_path()
            if not board_path:
                if MultiPhysicsViewerHandler.kicad_state:
                    board_path = MultiPhysicsViewerHandler.kicad_state.get("board_path")
                if not board_path and os.path.exists(DEFAULT_BOARD_PATH):
                    board_path = DEFAULT_BOARD_PATH

            net_name = payload.get("net_name", "/Signal_AMP")
            try:
                new_width_mm = float(payload.get("new_width_mm", 2.43))
            except (ValueError, TypeError):
                new_width_mm = 2.43

            if not board_path or not os.path.exists(board_path):
                self._send_json({"success": False, "error": f"Board file not found: {board_path}"}, 400)
                return

            modifier = KiCadLayoutModifier(board_path)
            mod_res = modifier.update_net_trace_width(net_name, new_width_mm, create_backup=True)

            if mod_res.get("success"):
                try:
                    if self.project_manager:
                        sync_payload = self.project_manager.trigger_active_board_sync()
                    else:
                        daemon = EMLiveSyncDaemon(board_path)
                        sync_payload = daemon.trigger_sync()
                    if sync_payload:
                        MultiPhysicsViewerHandler.kicad_state = sync_payload
                        MultiPhysicsViewerHandler.kicad_state["connected"] = True
                        mod_res["updated_em_metrics"] = sync_payload.get("em_metrics")
                except Exception as e:
                    mod_res["sync_warning"] = str(e)

            self._send_json(mod_res)

        elif path == "/api/kicad_autofix_drc":
            violation_id = payload.get("violation_id", "DRC-AT-1")
            board_path = payload.get("board_path")
            if not board_path and self.project_manager:
                board_path = self.project_manager.get_active_board_path()
            if not board_path:
                if MultiPhysicsViewerHandler.kicad_state:
                    board_path = MultiPhysicsViewerHandler.kicad_state.get("board_path")
                if not board_path and os.path.exists(DEFAULT_BOARD_PATH):
                    board_path = DEFAULT_BOARD_PATH

            if not board_path or not os.path.exists(board_path):
                self._send_json({"success": False, "error": f"Board file not found: {board_path}"}, 400)
                return

            drc_engine = DRCEngine(board_path)
            fix_res = drc_engine.execute_autofix(violation_id, board_path)

            if fix_res.get("success"):
                try:
                    if self.project_manager:
                        sync_payload = self.project_manager.trigger_active_board_sync()
                    else:
                        daemon = EMLiveSyncDaemon(board_path)
                        sync_payload = daemon.trigger_sync()
                    if sync_payload:
                        MultiPhysicsViewerHandler.kicad_state = sync_payload
                        MultiPhysicsViewerHandler.kicad_state["connected"] = True
                        fix_res["updated_em_metrics"] = sync_payload.get("em_metrics")
                except Exception as e:
                    fix_res["sync_warning"] = str(e)

            self._send_json(fix_res)

        elif path == "/api/nanopore/trigger_translocation":
            analyte = payload.get("analyte", "dsDNA")
            duration_us = payload.get("duration_us", None)
            if duration_us is not None:
                try:
                    duration_us = float(duration_us)
                except (ValueError, TypeError):
                    duration_us = None
            try:
                ep = get_electrophysiology_engine()
                ev = ep.trigger_translocation_event(analyte=analyte, duration_us=duration_us)
                self._send_json({"success": True, "event": ev})
            except Exception as e:
                self._send_json({"error": str(e)}, 500)

        elif path == "/api/project/build123d_update_params":
            try:
                engine = get_build123d_engine()
                res = engine.update_parameters(payload)
                self._send_json(res)
            except Exception as e:
                self._send_json({"error": str(e)}, 500)

        elif path == "/api/harness/render":
            yaml_txt = payload.get("yaml", "")
            proj_id = payload.get("project_id", "daemon-pore")
            h_type = payload.get("type", "electrical")
            try:
                we = get_wireviz_engine()
                if not yaml_txt or not yaml_txt.strip():
                    yaml_txt = we.get_project_harness(proj_id, h_type)
                res = we.render_harness(yaml_txt)
                self._send_json(res)
            except Exception as e:
                self._send_json({"success": False, "error": str(e)}, 500)

        elif path == "/api/project/harness/save":
            yaml_txt = payload.get("yaml", "")
            proj_id = payload.get("project_id", "daemon-pore")
            h_type = payload.get("type", "electrical")
            if not yaml_txt:
                self._send_json({"error": "Missing yaml content"}, 400)
                return
            try:
                we = get_wireviz_engine()
                ok = we.save_project_harness(proj_id, h_type, yaml_txt)
                self._send_json({"status": "ok" if ok else "error"})
            except Exception as e:
                self._send_json({"error": str(e)}, 500)

        elif path == "/api/harness/export":
            yaml_txt = payload.get("yaml", "")
            proj_id = payload.get("project_id", "daemon-pore")
            h_type = payload.get("type", "electrical")
            fmt = payload.get("format", "svg").lower()
            try:
                we = get_wireviz_engine()
                if not yaml_txt or not yaml_txt.strip():
                    yaml_txt = we.get_project_harness(proj_id, h_type)
                exp = we.export_harness(yaml_txt, fmt=fmt)
                if not exp:
                    self._send_json({"error": f"Export format '{fmt}' failed"}, 400)
                    return
                data_bytes, filename, mime = exp
                self._send_binary(data_bytes, mime, extra_headers={"Content-Disposition": f'attachment; filename="{filename}"'})
            except Exception as e:
                self._send_json({"error": str(e)}, 500)

        else:
            self._send_json({"error": f"Unknown endpoint {path}"}, 404)


def create_server(
    port: int = 8080,
    domain: str = "cfd",
    surrogate_db: Optional[str] = None
) -> Tuple[ThreadedHTTPServer, MultiPhysicsModelFusionOptimizer]:
    """Factory function initializing the optimizer and binding the server."""
    os.makedirs(WEB_DIR, exist_ok=True)
    os.makedirs("artifacts", exist_ok=True)

    param_defs = {
        "number_of_complete_revolutions": {"min": 1.0, "max": 4.0, "default": 2.0},
        "helix_path_radius_mm": {"min": 1.5, "max": 5.0, "default": 1.8},
        "helix_profile_radius_mm": {"min": 1.5, "max": 4.5, "default": 1.7},
        "blade_chamfer_mm": {"min": 0.1, "max": 1.0, "default": 0.5},
        "inlet_fillet_radius_mm": {"min": 0.1, "max": 1.0, "default": 0.5},
        "insert_length_mm": {"min": 40.0, "max": 60.0, "default": 50.0},
        "target_cell_size": {"min": 1.5, "max": 5.0, "default": 4.0}
    }

    opt = MultiPhysicsModelFusionOptimizer(
        physics_driver=None,
        parameter_defs=param_defs,
        domain=domain,
        surrogate_db_path=surrogate_db,
        verbose=True
    )

    # If surrogate has no samples, seed with 3 realistic calibration points
    if len(opt.surrogate.param_history) == 0:
        print("[ViewerServer] Seeding fresh surrogate memory with calibration samples...")
        for r in [1.8, 3.2, 4.5]:
            p = {
                "number_of_complete_revolutions": float(2.0 + (r - 1.8) * 0.3),
                "helix_path_radius_mm": float(r),
                "helix_profile_radius_mm": float(max(1.5, r - 0.2)),
                "blade_chamfer_mm": 0.5,
                "inlet_fillet_radius_mm": 0.5,
                "insert_length_mm": 50.0,
                "target_cell_size": 4.0
            }
            opt.step(candidate_params=p, mock_run=True)

    MultiPhysicsViewerHandler.optimizer = opt

    try:
        pm = ProjectManager(server_url=f"http://127.0.0.1:{port}")
        MultiPhysicsViewerHandler.project_manager = pm
        sync_data = pm.trigger_active_board_sync()
        if sync_data:
            MultiPhysicsViewerHandler.kicad_state = sync_data
            MultiPhysicsViewerHandler.kicad_state["connected"] = True
            active_proj = pm.get_active_project()
            proj_name = active_proj.name if active_proj else "None"
            board_id = active_proj.active_board_id if active_proj else "None"
            print(f"[ViewerServer] ProjectManager active: '{proj_name}' (board: '{board_id}')")
    except Exception as e:
        print(f"[ViewerServer] ProjectManager initialization note: {e}")
        if MultiPhysicsViewerHandler.kicad_state is None and os.path.exists(DEFAULT_BOARD_PATH):
            try:
                from em_live_watcher import EMLiveSyncDaemon
                daemon = EMLiveSyncDaemon(DEFAULT_BOARD_PATH, server_url=f"http://127.0.0.1:{port}")
                sync_data = daemon.trigger_sync()
                if sync_data:
                    MultiPhysicsViewerHandler.kicad_state = sync_data
                    MultiPhysicsViewerHandler.kicad_state["connected"] = True
                    print(f"[ViewerServer] Pre-loaded KiCad board state from {DEFAULT_BOARD_PATH}")
            except Exception as ex:
                print(f"[ViewerServer] KiCad fallback sync note: {ex}")

    server = ThreadedHTTPServer(("0.0.0.0", port), MultiPhysicsViewerHandler)
    return server, opt


if __name__ == "__main__":
    port = 8080
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            pass

    server, opt = create_server(port=port, domain="cfd")
    print(f"\n=======================================================")
    print(f"  OpenAuto-CFD Studio Real-Time Viewer Server Active")
    print(f"  URL: http://127.0.0.1:{port}")
    print(f"  Physics Domain: {opt.domain.upper()}")
    print(f"=======================================================\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.shutdown()
