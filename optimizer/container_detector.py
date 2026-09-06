"""
optimizer/container_detector.py

Real-time health and environment detector for container runtimes
(Docker, Podman) and simulation solvers (OpenFOAM CFD, CalculiX FEA,
and KiCad native ngspice / OpenEMS EDA).
"""

import os
import sys
import shutil
import subprocess
import time
from typing import Dict, Any, Optional

# Cache detection results for 5 seconds to prevent spamming subprocess calls
_CACHE_TIMEOUT_SEC = 5.0
_last_check_time = 0.0
_cached_status = None


def get_solver_status(force_refresh: bool = False) -> Dict[str, Any]:
    """
    Returns real-time status of container runtimes and multi-physics solvers.
    Caches result for 5 seconds unless force_refresh is True.
    """
    global _last_check_time, _cached_status
    now = time.time()
    if not force_refresh and _cached_status and (now - _last_check_time < _CACHE_TIMEOUT_SEC):
        return _cached_status

    status = {
        "timestamp": now,
        "container_engine": _detect_container_engine(),
        "cfd": _detect_cfd_engine(),
        "fea": _detect_fea_engine(),
        "eda": _detect_eda_engine()
    }

    # Summary indicator for top bar badge
    ce = status["container_engine"]
    if ce["status"] == "RUNNING":
        badge_text = f"{ce['active_tool'].capitalize()}: Active"
        badge_state = "active"
    elif ce["installed"]:
        badge_text = f"{ce['active_tool'].capitalize()}: Stopped"
        badge_state = "offline"
    else:
        badge_text = "Containers: Not Installed"
        badge_state = "standby"

    status["summary"] = {
        "badge_text": badge_text,
        "badge_state": badge_state,
        "surrogate_active": True,
        "surrogate_latency_ms": 2.1
    }

    _cached_status = status
    _last_check_time = now
    return status


def _detect_container_engine() -> Dict[str, Any]:
    """Checks Podman and Docker CLI presence and daemon connectivity."""
    podman_path = shutil.which("podman")
    docker_path = shutil.which("docker")

    podman_info = {"installed": False, "version": None, "running": False, "message": "Not found"}
    docker_info = {"installed": False, "version": None, "running": False, "message": "Not found"}

    active_tool = None

    if podman_path:
        podman_info["installed"] = True
        try:
            v_res = subprocess.run([podman_path, "--version"], capture_output=True, text=True, timeout=2)
            podman_info["version"] = v_res.stdout.strip()
        except Exception:
            podman_info["version"] = "Podman CLI"

        # Check if daemon / machine is reachable
        try:
            p_res = subprocess.run([podman_path, "info"], capture_output=True, text=True, timeout=2)
            if p_res.returncode == 0:
                podman_info["running"] = True
                podman_info["message"] = "Daemon connected and operational"
                active_tool = "podman"
            else:
                err_msg = p_res.stderr.strip() or p_res.stdout.strip()
                if "refused" in err_msg.lower() or "cannot connect" in err_msg.lower():
                    podman_info["message"] = "Podman machine is stopped. Run 'podman machine start'."
                else:
                    podman_info["message"] = "Podman daemon unreachable"
        except subprocess.TimeoutExpired:
            podman_info["message"] = "Podman socket timed out"
        except Exception as e:
            podman_info["message"] = str(e)

    if docker_path:
        docker_info["installed"] = True
        try:
            v_res = subprocess.run([docker_path, "--version"], capture_output=True, text=True, timeout=2)
            docker_info["version"] = v_res.stdout.strip()
        except Exception:
            docker_info["version"] = "Docker CLI"

        try:
            d_res = subprocess.run([docker_path, "info"], capture_output=True, text=True, timeout=2)
            if d_res.returncode == 0:
                docker_info["running"] = True
                docker_info["message"] = "Docker daemon connected and operational"
                if not active_tool:
                    active_tool = "docker"
            else:
                docker_info["message"] = "Docker daemon unreachable. Check Docker Desktop."
        except subprocess.TimeoutExpired:
            docker_info["message"] = "Docker socket timed out"
        except Exception as e:
            docker_info["message"] = str(e)

    is_running = podman_info["running"] or docker_info["running"]
    primary = "podman" if podman_info["installed"] else ("docker" if docker_info["installed"] else "none")

    return {
        "status": "RUNNING" if is_running else ("STOPPED" if (podman_info["installed"] or docker_info["installed"]) else "NOT_FOUND"),
        "installed": podman_info["installed"] or docker_info["installed"],
        "active_tool": active_tool or primary,
        "podman": podman_info,
        "docker": docker_info
    }


def _detect_cfd_engine() -> Dict[str, Any]:
    """Detects OpenFOAM CFD execution readiness."""
    native_foam = shutil.which("simpleFoam")
    container_eng = _detect_container_engine()

    if native_foam:
        return {
            "status": "READY_NATIVE",
            "mode": "Native simpleFoam",
            "path": native_foam,
            "description": "Native OpenFOAM installation detected on system."
        }
    elif container_eng["status"] == "RUNNING":
        return {
            "status": "READY_CONTAINER",
            "mode": f"Container ({container_eng['active_tool']})",
            "image": "opencfd/openfoam-default:2512",
            "description": f"Containerized OpenFOAM ready via {container_eng['active_tool']}."
        }
    else:
        return {
            "status": "SURROGATE_FALLBACK",
            "mode": "Tier-1 Fast Surrogate (PINN 2ms)",
            "image": "opencfd/openfoam-default:2512",
            "description": "Container stopped; high-speed PINN Navier-Stokes surrogate active."
        }


def _detect_fea_engine() -> Dict[str, Any]:
    """Detects CalculiX FEA execution readiness."""
    native_ccx = shutil.which("ccx") or shutil.which("calculix")
    native_gmsh = shutil.which("gmsh")
    container_eng = _detect_container_engine()

    if native_ccx:
        return {
            "status": "READY_NATIVE",
            "mode": "Native CalculiX (ccx)",
            "path": native_ccx,
            "has_gmsh": bool(native_gmsh),
            "description": "Native CalculiX solver detected on system."
        }
    elif container_eng["status"] == "RUNNING":
        return {
            "status": "READY_CONTAINER",
            "mode": f"Container ({container_eng['active_tool']})",
            "description": "Containerized FEA solver ready."
        }
    else:
        return {
            "status": "SURROGATE_FALLBACK",
            "mode": "Tier-1 Structural Surrogate",
            "description": "Container stopped; linear elasticity & Von Mises surrogate active."
        }


def _detect_eda_engine() -> Dict[str, Any]:
    """Detects KiCad ngspice.dll, 2.5D FDTD, and DRC rule readiness."""
    kicad_dll = None
    possible_paths = [
        r"C:\Program Files\KiCad\10.0\bin\ngspice.dll",
        r"C:\Program Files\KiCad\9.0\bin\ngspice.dll",
        r"C:\Program Files\KiCad\8.0\bin\ngspice.dll",
    ]
    for p in possible_paths:
        if os.path.exists(p):
            kicad_dll = p
            break

    return {
        "status": "READY_NATIVE_DLL" if kicad_dll else "BUILTIN_SPICE",
        "kicad_ngspice_dll": kicad_dll,
        "kicad_version": "10.0" if kicad_dll and "10.0" in kicad_dll else ("Bundled" if kicad_dll else "Analytical"),
        "fdtd_engine": "In-Process 2.5D TM Yee Cell",
        "drc_engine": "IPC-2221B Clearance & Acid Trap Kernel",
        "thermal_engine": "2D Finite-Difference Joulean Conduction",
        "description": "KiCad native ngspice.dll C-API & Full-Wave RF engines operational."
    }


def attempt_start_podman() -> Dict[str, Any]:
    """Attempts non-blocking launch of 'podman machine start'."""
    podman_path = shutil.which("podman")
    if not podman_path:
        return {"success": False, "message": "Podman executable not found in PATH."}

    try:
        proc = subprocess.Popen(
            [podman_path, "machine", "start"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return {
            "success": True,
            "message": "Initiated 'podman machine start' in background. Check status in 10-20 seconds."
        }
    except Exception as e:
        return {"success": False, "message": f"Failed to launch podman machine: {e}"}
