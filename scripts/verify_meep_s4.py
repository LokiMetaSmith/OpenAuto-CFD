"""
verify_meep_s4.py

Comprehensive Verification Suite for Photonic Physics Simulation:
  1. Meep FDTD Solver (waveguides, ring resonators, nanopore metasurfaces)
  2. Stanford S4 RCWA Solver (periodic photonic crystal slabs, gratings, diffraction)
  3. PhysicsEngineFactory dynamic driver resolution
  4. CADAgentToolRegistry LLM tool-calling execution
  5. FastMCP server tool execution
  6. WebGL Viewer REST API endpoints (/api/photonic/meep_simulate, /api/photonic/s4_simulate, /api/photonic/simulate)
"""

import os
import sys
import json
import tempfile
import urllib.request
import urllib.parse
import threading
import time

repo_root = os.path.abspath(os.path.dirname(__file__))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

optimizer_dir = os.path.join(repo_root, "optimizer")
if optimizer_dir not in sys.path:
    sys.path.insert(0, optimizer_dir)

viewer_dir = os.path.join(repo_root, "viewer")
if viewer_dir not in sys.path:
    sys.path.insert(0, viewer_dir)

from optimizer.physics_factory import PhysicsEngineFactory
from optimizer.meep_driver import MeepDriver, export_meep_vtk_field
from optimizer.s4_driver import S4Driver, export_s4_vtk_field
from optimizer.cad_agent_tools import CADAgentToolRegistry
import optimizer.mcp_server as mcp_server


def test_meep_driver():
    print("\n--- 1. Testing Meep FDTD Driver ---")
    with tempfile.TemporaryDirectory() as td:
        # Test A: Waveguide
        driver_wg = MeepDriver(
            case_dir=td,
            config={
                "meep": {
                    "geometry_type": "waveguide",
                    "wavelength_min_um": 1.4,
                    "wavelength_max_um": 1.7,
                    "resolution": 20
                }
            }
        )
        driver_wg.prepare_case()
        assert driver_wg.run_meshing() is True, "Meshing validation failed for waveguide"
        assert driver_wg.run_solver() is True, "Solver execution failed for waveguide"
        metrics_wg = driver_wg.get_metrics()
        print("  [Waveguide Metrics]:", metrics_wg)
        assert "transmission" in metrics_wg and metrics_wg["transmission"] > 0.0
        assert "insertion_loss_db" in metrics_wg
        assert "resonant_wavelength_nm" in metrics_wg

        spectrum_wg = driver_wg.get_spectrum()
        assert len(spectrum_wg["wavelength_nm"]) > 0, "Empty wavelength spectrum"
        assert len(spectrum_wg["transmission"]) == len(spectrum_wg["wavelength_nm"])

        vtk_dir = driver_wg.generate_vtk()
        assert vtk_dir is not None and os.path.exists(vtk_dir), "VTK generation failed"
        vtk_file = os.path.join(vtk_dir, "meep_field.vtk")
        assert os.path.exists(vtk_file), "meep_field.vtk not found"
        with open(vtk_file, "r") as f:
            header = f.readline()
            assert "# vtk DataFile Version" in header

        # Test B: Ring Resonator
        driver_ring = MeepDriver(
            case_dir=td,
            config={
                "meep": {
                    "geometry_type": "ring_resonator",
                    "ring_radius_um": 3.0,
                    "ring_gap_um": 0.15
                }
            }
        )
        driver_ring.prepare_case()
        assert driver_ring.run_solver() is True
        metrics_ring = driver_ring.get_metrics()
        print("  [Ring Resonator Metrics]:", metrics_ring)
        assert metrics_ring["q_factor"] > 0.0, "Expected non-zero Q-factor"
        assert metrics_ring["extinction_ratio_db"] > 1.0, "Expected extinction ratio"

        # Test C: Nanopore Metasurface
        driver_pore = MeepDriver(
            case_dir=td,
            config={
                "meep": {
                    "geometry_type": "nanopore_metasurface",
                    "pore_diam_nm": 80.0
                }
            }
        )
        driver_pore.prepare_case()
        assert driver_pore.run_solver() is True
        metrics_pore = driver_pore.get_metrics()
        print("  [Nanopore Metasurface Metrics]:", metrics_pore)
        assert metrics_pore["transmission"] > 0.0

        driver_wg.cleanup_ram_disk()
    print("Meep Driver tests PASSED.")


def test_s4_driver():
    print("\n--- 2. Testing Stanford S4 RCWA Driver ---")
    with tempfile.TemporaryDirectory() as td:
        driver = S4Driver(
            case_dir=td,
            config={
                "s4": {
                    "lattice_period_um": 0.8,
                    "num_harmonics": 49,
                    "wavelength_min_um": 0.7,
                    "wavelength_max_um": 1.6,
                    "slab_thickness_um": 0.22,
                    "hole_radius_um": 0.18,
                    "polarization": "TE"
                }
            }
        )
        driver.prepare_case()
        assert driver.run_meshing() is True, "Harmonic validation failed"
        assert driver.run_solver() is True, "S4 solver execution failed"
        metrics = driver.get_metrics()
        print("  [S4 RCWA Metrics]:", metrics)
        assert "zero_order_transmission" in metrics
        assert "diffraction_efficiency" in metrics
        assert "resonant_wavelength_nm" in metrics
        assert metrics["q_factor"] > 0.0

        spectrum = driver.get_spectrum()
        assert len(spectrum["wavelength_nm"]) > 0
        assert len(spectrum["transmission_0th"]) == len(spectrum["wavelength_nm"])
        assert len(spectrum["diffraction_efficiency"]) == len(spectrum["wavelength_nm"])

        vtk_dir = driver.generate_vtk()
        assert vtk_dir is not None and os.path.exists(vtk_dir)
        vtk_file = os.path.join(vtk_dir, "s4_field.vtk")
        assert os.path.exists(vtk_file)

        driver.cleanup_ram_disk()
    print("S4 RCWA Driver tests PASSED.")


def test_physics_factory():
    print("\n--- 3. Testing PhysicsEngineFactory ---")
    with tempfile.TemporaryDirectory() as td:
        d1 = PhysicsEngineFactory.get_driver(td, {"physics": {"type": "meep"}})
        assert d1.__class__.__name__ == "MeepDriver"

        d2 = PhysicsEngineFactory.get_driver(td, {"physics": {"type": "fdtd"}})
        assert d2.__class__.__name__ == "MeepDriver"

        d3 = PhysicsEngineFactory.get_driver(td, {"physics": {"type": "s4"}})
        assert d3.__class__.__name__ == "S4Driver"

        d4 = PhysicsEngineFactory.get_driver(td, {"physics": {"type": "rcwa"}})
        assert d4.__class__.__name__ == "S4Driver"

        d5 = PhysicsEngineFactory.get_driver(td, {"physics": {"type": "photonic"}})
        assert d5.__class__.__name__ == "MeepDriver"

        d6 = PhysicsEngineFactory.get_driver(td, {"physics": {"type": "photonic"}, "photonic": {"solver": "s4"}})
        assert d6.__class__.__name__ == "S4Driver" 

        d1.cleanup_ram_disk()
        d2.cleanup_ram_disk()
        d3.cleanup_ram_disk()
        d4.cleanup_ram_disk()
        d5.cleanup_ram_disk()
        d6.cleanup_ram_disk()
    print("PhysicsEngineFactory tests PASSED.")


def test_cad_agent_tools():
    print("\n--- 4. Testing CADAgentToolRegistry Photonic Tools ---")
    registry = CADAgentToolRegistry()
    spec = registry.get_tools_spec()
    tool_names = [t["function"]["name"] for t in spec]
    assert "simulate_meep_fdtd" in tool_names, "simulate_meep_fdtd not found in tools"
    assert "simulate_s4_rcwa" in tool_names, "simulate_s4_rcwa not found in tools"

    # Execute simulate_meep_fdtd
    res_meep = registry.execute_tool(
        "simulate_meep_fdtd",
        {"geometry_type": "waveguide", "wavelength_min_um": 1.4, "wavelength_max_um": 1.7}
    )
    assert res_meep["status"] == "success", f"Meep tool failed: {res_meep}"
    assert "metrics" in res_meep
    assert "spectrum_sample" in res_meep
    print("  [CAD Agent Meep Tool Result]:", res_meep["metrics"])

    # Execute simulate_s4_rcwa
    res_s4 = registry.execute_tool(
        "simulate_s4_rcwa",
        {"lattice_period_um": 0.8, "wavelength_min_um": 0.7, "wavelength_max_um": 1.6}
    )
    assert res_s4["status"] == "success", f"S4 tool failed: {res_s4}"
    assert "metrics" in res_s4
    assert "spectrum_sample" in res_s4
    print("  [CAD Agent S4 Tool Result]:", res_s4["metrics"])
    print("CADAgentToolRegistry photonic tests PASSED.")


def test_mcp_photonic_tools():
    print("\n--- 5. Testing FastMCP Photonic Tools ---")
    with tempfile.TemporaryDirectory() as td:
        m1 = mcp_server.run_meep_simulation_tool(
            geometry_type="ring_resonator",
            ring_radius_um=2.5,
            case_dir=td
        )
        assert m1["status"] == "success"
        assert m1["geometry_type"] == "ring_resonator"
        assert "metrics" in m1
        assert "spectrum" in m1
        print("  [MCP Meep]:", m1["metrics"])

        m2 = mcp_server.run_s4_simulation_tool(
            lattice_period_um=0.9,
            case_dir=td
        )
        assert m2["status"] == "success"
        assert "metrics" in m2
        assert "spectrum" in m2
        print("  [MCP S4]:", m2["metrics"])
    print("FastMCP Photonic Tools tests PASSED.")


def test_viewer_api_photonic():
    print("\n--- 6. Testing WebGL Viewer REST API Photonic Endpoints ---")
    from viewer.server import create_server
    port = 8192
    server, opt = create_server(port=port)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    time.sleep(1.0)

    try:
        base_url = f"http://127.0.0.1:{port}"

        # GET /api/photonic/meep_simulate
        url_meep = f"{base_url}/api/photonic/meep_simulate?geometry=waveguide&wl_min=1.4&wl_max=1.7"
        req = urllib.request.Request(url_meep)
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            assert data["status"] == "success"
            assert data["solver"] == "Meep_FDTD"
            assert "metrics" in data
            assert "spectrum" in data
            print("  [GET /api/photonic/meep_simulate]: OK (Transmission =", data["metrics"]["transmission"], ")")

        # GET /api/photonic/s4_simulate
        url_s4 = f"{base_url}/api/photonic/s4_simulate?lattice_period=0.8&wl_min=0.8&wl_max=1.5"
        req = urllib.request.Request(url_s4)
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            assert data["status"] == "success"
            assert data["solver"] == "S4_RCWA"
            assert "metrics" in data
            assert "spectrum" in data
            print("  [GET /api/photonic/s4_simulate]: OK (Resonant WL =", data["metrics"]["resonant_wavelength_nm"], "nm)")

        # POST /api/photonic/simulate (Meep)
        url_post = f"{base_url}/api/photonic/simulate"
        post_body = json.dumps({
            "solver": "meep",
            "config": {"geometry_type": "ring_resonator", "ring_radius_um": 3.0}
        }).encode("utf-8")
        req = urllib.request.Request(url_post, data=post_body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            assert data["status"] == "success"
            assert "metrics" in data
            print("  [POST /api/photonic/simulate (Meep)]: OK (Q-factor =", data["metrics"]["q_factor"], ")")

        # POST /api/photonic/simulate (S4)
        post_body_s4 = json.dumps({
            "solver": "s4",
            "config": {"lattice_period_um": 0.8, "slab_thickness_um": 0.22}
        }).encode("utf-8")
        req = urllib.request.Request(url_post, data=post_body_s4, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            assert data["status"] == "success"
            assert "metrics" in data
            print("  [POST /api/photonic/simulate (S4)]: OK (Zero-order Transmission =", data["metrics"]["zero_order_transmission"], ")")

    finally:
        server.shutdown()
        server.server_close()
    print("WebGL Viewer REST API Photonic tests PASSED.")


if __name__ == "__main__":
    test_meep_driver()
    test_s4_driver()
    test_physics_factory()
    test_cad_agent_tools()
    test_mcp_photonic_tools()
    test_viewer_api_photonic()
    print("\n=======================================================")
    print("*** ALL MEEP & S4 PHOTONIC VERIFICATION TESTS PASSED! ***")
    print("=======================================================\n")
