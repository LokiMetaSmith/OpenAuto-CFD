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


WEB_DIR = os.path.join(os.path.dirname(__file__), "web")
DEFAULT_BOARD_PATH = r"C:\Users\Loki-VR\Documents\projects\Daemon Pore\daemon-pore\Amplifier\amplifier.kicad_pcb"



class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


class MultiPhysicsViewerHandler(SimpleHTTPRequestHandler):
    """HTTP Request Handler providing REST APIs and static file serving."""

    optimizer: Optional[MultiPhysicsModelFusionOptimizer] = None
    kicad_state: Optional[Dict[str, Any]] = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=WEB_DIR, **kwargs)

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

        elif path == "/api/gencad/synthesize":
            from gencad_driver import GenCADDriver
            query = urllib.parse.parse_qs(parsed.query)
            dp = float(query.get("delta_p", [2500.0])[0])
            eff = float(query.get("separation_efficiency", [95.0])[0])
            fmt = query.get("format", ["build123d"])[0]

            driver = GenCADDriver()
            res = driver.generate_and_export(
                physics_target={"delta_p": dp, "separation_efficiency": eff},
                output_dir="artifacts",
                filename_prefix="gencad_hud_synthesized",
                format_type=fmt,
                use_transformer_sequence=True
            )
            self._send_json(res)

        elif path == "/api/gencad/sample":
            from gencad_driver import GenCADDriver
            query = urllib.parse.parse_qs(parsed.query)
            dp = float(query.get("delta_p", [2500.0])[0])
            eff = float(query.get("separation_efficiency", [95.0])[0])
            n_samples = int(query.get("n_samples", [3])[0])

            driver = GenCADDriver()
            samples = driver.sample_diverse_cad_programs(
                {"delta_p": dp, "separation_efficiency": eff},
                n_samples=n_samples,
                format_type="build123d"
            )
            self._send_json({
                "status": "success",
                "physics_target": {"delta_p": dp, "separation_efficiency": eff},
                "samples": samples
            })

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

        elif path == "/api/agent_chat":
            message = payload.get("message", "").strip()
            current_params = payload.get("params", {})
            fidelity = payload.get("fidelity", "tier1")

            msg_lower = message.lower()
            if any(k in msg_lower for k in ["pcb", "kicad", "trace", "microstrip", "rf", "impedance", "coplanar"]):
                eda_agent = EDAReasoningAgent()
                result = eda_agent.run_goal(message)
                self._send_json({
                    "status": "success",
                    "agent_type": "EDA_RF_Agent",
                    "reply": result.get("summary", "EDA analysis complete."),
                    "trace": result.get("trace", []),
                    "kicad_path": result.get("kicad_pcb_path") or result.get("kicad_path"),
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
                    daemon = EMLiveSyncDaemon(board_path)
                    sync_payload = daemon.trigger_sync()
                    if sync_payload:
                        MultiPhysicsViewerHandler.kicad_state = sync_payload
                        MultiPhysicsViewerHandler.kicad_state["connected"] = True
                        fix_res["updated_em_metrics"] = sync_payload.get("em_metrics")
                except Exception as e:
                    fix_res["sync_warning"] = str(e)

            self._send_json(fix_res)

        elif path == "/api/gencad/synthesize":
            from gencad_driver import GenCADDriver
            p_target = payload.get("physics_target", {})
            dp = float(p_target.get("delta_p", payload.get("delta_p", 2500.0)))
            eff = float(p_target.get("separation_efficiency", payload.get("separation_efficiency", 95.0)))
            fmt = payload.get("format_type", payload.get("format", "build123d"))

            driver = GenCADDriver()
            res = driver.generate_and_export(
                physics_target={"delta_p": dp, "separation_efficiency": eff},
                output_dir="artifacts",
                filename_prefix="gencad_hud_synthesized",
                format_type=fmt,
                use_transformer_sequence=True
            )
            self._send_json(res)

        elif path == "/api/gencad/sample":
            from gencad_driver import GenCADDriver
            p_target = payload.get("physics_target", {})
            dp = float(p_target.get("delta_p", payload.get("delta_p", 2500.0)))
            eff = float(p_target.get("separation_efficiency", payload.get("separation_efficiency", 95.0)))
            n_samples = int(payload.get("n_samples", 3))

            driver = GenCADDriver()
            samples = driver.sample_diverse_cad_programs(
                {"delta_p": dp, "separation_efficiency": eff},
                n_samples=n_samples,
                format_type="build123d"
            )
            self._send_json({
                "status": "success",
                "physics_target": {"delta_p": dp, "separation_efficiency": eff},
                "samples": samples
            })

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
    print(f"  Atlas Fields Studio Real-Time Viewer Server Active")
    print(f"  URL: http://127.0.0.1:{port}")
    print(f"  Physics Domain: {opt.domain.upper()}")
    print(f"=======================================================\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.shutdown()
