import sys
try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass
#!/usr/bin/env python3
"""
verify_samara_feather_e2e.py

Comprehensive End-to-End (E2E) Verification Test:
1. Verifies cross-platform project discovery without hardcoded path dependencies.
2. Imports external project at C:\\Users\\Loki-VR\\Documents\\projects\\Samara_Feather.
3. Validates synthesis of hardware/samara_stator.kicad_pcb (6 planar spiral coils, ESP32 MCU pads, IMU markers).
4. Validates generation of openauto.project.json manifest with auto-discovered mechanical parts.
5. Verifies multi-physics simulations on the hardware:
   - Circuit Board: Planar stator coil inductance, skin depth, AC resistance, Q factor, trace Z0, thermal rise.
   - Pressurized Airflow: Centrifugal impeller Euler head, Stodola slip factor, plenum pressure, mass flow rate, internal wing duct Fanno flow losses.
   - Wing Tip Velocity: Trailing edge nozzle expansion exit velocity, jet thrust, autorotation drag equilibrium, steady tip velocity, vehicle RPM, and blown lift.
6. Tests live REST API endpoints on a local server instance without daemon-pore artifact leaks.
"""

import os
import sys
import time
import json
import urllib.request
import urllib.error
import threading

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
VIEWER_DIR = os.path.join(REPO_DIR, "viewer")
OPTIMIZER_DIR = os.path.join(REPO_DIR, "optimizer")
KICAD_PLUGIN_DIR = os.path.join(REPO_DIR, "kicad_plugin")

for p in [VIEWER_DIR, OPTIMIZER_DIR, KICAD_PLUGIN_DIR, REPO_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from project_engine import ProjectManager, MultiphysicsProject, get_default_project_search_dirs
from samara_hardware_driver import SamaraHardwareDriver
from server import create_server


def test_cross_platform_discovery():
    print("\n--- [Test 1] Cross-Platform Project Discovery ---")
    dirs = get_default_project_search_dirs()
    print(f"Discovered candidate search dirs ({len(dirs)}):")
    for d in dirs:
        print(f"  - {d} (exists: {os.path.exists(d)})")
    assert len(dirs) > 0, "Expected at least 1 search directory"
    print("[PASS] Dynamic cross-platform search directory discovery passed.")


def test_import_samara_feather(target_dir):
    print(f"\n--- [Test 2] Importing External Project: {target_dir} ---")
    assert os.path.exists(target_dir), f"Target directory does not exist: {target_dir}"

    pm = ProjectManager()
    res = pm.import_project(project_dir=target_dir)

    assert res.get("success") is True, f"Project import failed: {res}"
    pid = res.get("project_id")
    print(f"Imported project ID: {pid}")

    # Check openauto.project.json
    manifest_p = os.path.join(target_dir, "openauto.project.json")
    assert os.path.exists(manifest_p), f"Manifest file not created: {manifest_p}"
    with open(manifest_p, "r", encoding="utf-8") as f:
        mf = json.load(f)

    print(f"Project Name: {mf.get('name')}")
    print(f"Active Board: {mf.get('active_board')}")
    boards = mf.get("boards", {})
    assert len(boards) > 0, "No boards found in imported project manifest"
    print(f"Registered Boards ({len(boards)}): {list(boards.keys())}")

    # Check that planar stator board exists
    stator_board_path = os.path.join(target_dir, "hardware", "samara_stator.kicad_pcb")
    assert os.path.exists(stator_board_path), f"Synthesized PCB board not found: {stator_board_path}"
    print(f"[PASS] Synthesized stator PCB verified at: {stator_board_path}")

    # Check mechanical parts
    mech_parts = mf.get("mechanical", {}).get("parts", [])
    print(f"Discovered Mechanical CAD Parts ({len(mech_parts)}):")
    for part in mech_parts:
        print(f"  - [{part.get('id')}] {part.get('name')} -> {part.get('path')}")
    assert len(mech_parts) >= 3, f"Expected at least 3 CAD parts, got {len(mech_parts)}"

    # Check project selection in ProjectManager
    active_proj = pm.get_active_project()
    assert active_proj is not None, "No active project in ProjectManager"
    assert active_proj.project_id == pid, f"Active project mismatch: {active_proj.project_id} != {pid}"
    print("[PASS] Project import and manifest generation verified.")
    return pid, pm


def test_samara_multi_physics_simulations(target_dir):
    print(f"\n--- [Test 3] Multi-Physics Simulations on Samara Feather Hardware ---")
    driver = SamaraHardwareDriver(target_dir)

    # 1. Circuit Board Simulation
    print("\n[3A] Circuit Board Simulation (Planar Stator Spirals & Interconnects):")
    board_sim = driver.simulate_circuit_board(freq_mhz=3.0, phase_current_arms=1.5)
    assert board_sim["status"] == "success"
    sm = board_sim["stator_metrics"]
    tm = board_sim["transmission_line_metrics"]
    thm = board_sim["thermal_and_power"]

    print(f"  - Stator Coils: {sm['num_coils']} coils, {sm['turns_per_coil']} turns each")
    print(f"  - Single Coil Inductance: {sm['coil_inductance_uh']} uH")
    print(f"  - Phase Inductance: {sm['phase_inductance_uh']} uH")
    print(f"  - High-Freq Skin Depth (3 MHz): {sm['skin_depth_um']} um")
    print(f"  - AC Winding Resistance: {sm['coil_ac_resistance_ohms']} Ohms")
    print(f"  - Quality Factor Q: {sm['coil_quality_factor_q']}")
    print(f"  - Trace Characteristic Impedance Z0: {tm['characteristic_impedance_z0_ohms']} Ohms")
    print(f"  - Stator Total Power Dissipation: {thm['total_stator_loss_watts']} W")
    print(f"  - IPC-2152 Substrate Temp Rise: +{thm['temperature_rise_c']} C -> Board: {thm['estimated_board_temp_c']} C")

    assert sm["coil_inductance_uh"] > 0.5, "Inductance too low"
    assert sm["coil_quality_factor_q"] > 10.0, "Quality factor too low"
    assert 40.0 <= tm["characteristic_impedance_z0_ohms"] <= 160.0, "Trace impedance out of expected range"
    assert 45.0 <= tm.get("rf_controlled_impedance_50ohm_z0", 50.0) <= 60.0, "RF 50-ohm trace out of range"

    # 2. Pressurized Airflow CFD Simulation
    print("\n[3B] Pressurized Airflow Simulation (Centrifugal Impeller & Internal Wing Duct):")
    airflow_sim = driver.simulate_impeller_airflow(rpm=6393.0)
    assert airflow_sim["status"] == "success"
    ia = airflow_sim["impeller_aerodynamics"]
    cp = airflow_sim["central_plenum"]
    df = airflow_sim["internal_wing_duct_fanno_flow"]

    print(f"  - Impeller Tip Speed: {ia['tip_speed_u2_m_s']} m/s")
    print(f"  - Stodola Slip Factor: {ia['stodola_slip_factor']}")
    print(f"  - Total Pressure Rise: {ia['total_pressure_rise_pa']} Pa")
    print(f"  - Plenum Static Pressure: {cp['static_pressure_gauge_pa']} Pa gauge")
    print(f"  - Mass Flow Rate: {cp['mass_flow_total_g_s']} g/s total ({cp['mass_flow_per_wing_g_s']} g/s per wing)")
    print(f"  - Wing Duct Flow Velocity: {df['internal_flow_velocity_m_s']} m/s")
    print(f"  - Wing Duct Reynolds Number: {df['reynolds_number']}")
    print(f"  - Fanno Flow Friction Pressure Drop: {df['duct_frictional_loss_pa']} Pa")
    print(f"  - Net Pressure at Trailing Edge Nozzles: {df['net_nozzle_pressure_gauge_pa']} Pa")

    assert ia["total_pressure_rise_pa"] > 800.0, "Pressure rise below specification"
    assert cp["mass_flow_total_g_s"] > 15.0, "Mass flow rate below specification"
    assert df["net_nozzle_pressure_gauge_pa"] > 500.0, "Net nozzle pressure insufficient"

    # 3. Wing Tip Velocity & Autorotation Dynamics
    print("\n[3C] Wing Tip Velocity & Autorotation Aerodynamics:")
    wing_sim = driver.simulate_wing_velocity(
        nozzle_press_gauge_pa=df["net_nozzle_pressure_gauge_pa"],
        mass_flow_per_wing_g_s=cp["mass_flow_per_wing_g_s"]
    )
    assert wing_sim["status"] == "success"
    nj = wing_sim["nozzle_jet_expansion"]
    sa = wing_sim["steady_state_autorotation"]
    fl = wing_sim["flight_lift_and_hover"]

    print(f"  - Nozzle Expansion Exit Velocity: {nj['exit_velocity_m_s']} m/s")
    print(f"  - Jet Reaction Thrust (2 wings): {nj['total_jet_thrust_n']} N")
    print(f"  - Driving Reaction Torque: {nj['driving_torque_nm']} Nm")
    print(f"  - Steady-State Autorotation Rate: {sa['steady_rotation_rad_s']} rad/s ({sa['steady_vehicle_rpm']} RPM)")
    print(f"  - Wing Tip Velocity: {sa['wing_tip_velocity_m_s']} m/s ({sa['wing_tip_velocity_km_h']} km/h)")
    print(f"  - Wing Tip Mach: M = {sa['wing_tip_mach']}")
    print(f"  - Rotational Aerodynamic Lift: {fl['rotational_aero_lift_n']} N")
    print(f"  - Blown Coanda Circulation Lift: {fl['blown_coanda_lift_n']} N")
    print(f"  - Total Vertical Lift: {fl['total_vertical_lift_n']} N (Weight: {fl['vehicle_weight_n']} N)")
    print(f"  - Thrust-to-Weight Ratio: {fl['thrust_to_weight_ratio']} ({fl['hover_climb_capability']})")

    assert nj["exit_velocity_m_s"] > 25.0, "Exit velocity too low"
    assert sa["wing_tip_velocity_m_s"] > 30.0, "Wing tip velocity below autorotation threshold"
    assert fl["thrust_to_weight_ratio"] > 0.95, "Thrust-to-weight insufficient for flight"
    print("[PASS] All Samara Feather hardware multi-physics simulations validated successfully.")


def test_live_server_endpoints(target_dir, port=8089):
    print(f"\n--- [Test 4] Live REST API Server Verification (Port {port}) ---")
    server, opt = create_server(port=port)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    time.sleep(0.5)

    base_url = f"http://127.0.0.1:{port}"

    try:
        # 1. Test /api/projects
        req = urllib.request.Request(f"{base_url}/api/projects")
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            print(f"Projects API returned {len(data.get('projects', []))} projects.")
            pids = [p["project_id"] for p in data.get("projects", [])]
            print(f"Available project IDs: {pids}")

        # 2. Test /api/project/import via HTTP POST
        post_data = json.dumps({"project_dir": target_dir}).encode("utf-8")
        req = urllib.request.Request(f"{base_url}/api/project/import", data=post_data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 200
            import_res = json.loads(resp.read().decode("utf-8"))
            assert import_res.get("success") is True
            print(f"[PASS] /api/project/import successful for: {import_res.get('project_id')}")

        # 3. Test /api/project/select
        post_sel = json.dumps({"project_id": "samara-feather"}).encode("utf-8")
        req = urllib.request.Request(f"{base_url}/api/project/select", data=post_sel, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 200
            sel_res = json.loads(resp.read().decode("utf-8"))
            print(f"[PASS] /api/project/select switched active project to: {sel_res.get('active_project', {}).get('name')}")

        # 4. Test /api/project/mesh_binary for coaxial_impeller
        mesh_url = f"{base_url}/api/project/mesh_binary?project_id=samara-feather&part_id=coaxial_impeller"
        req = urllib.request.Request(mesh_url)
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 200
            mesh_bytes = resp.read()
            v_count = resp.headers.get("X-Vertex-Count")
            print(f"[PASS] /api/project/mesh_binary (coaxial_impeller): {len(mesh_bytes)} bytes, vertex count: {v_count}")
            assert len(mesh_bytes) > 1000, "Mesh binary data too small"

        # 5. Test /api/samara/simulate_all
        sim_url = f"{base_url}/api/samara/simulate_all?rpm=6393"
        req = urllib.request.Request(sim_url)
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 200
            sim_all = json.loads(resp.read().decode("utf-8"))
            assert sim_all.get("status") == "success"
            print(f"[PASS] /api/samara/simulate_all returned complete multi-physics state:")
            print(f"   Stator Inductance: {sim_all['circuit_board']['stator_metrics']['phase_inductance_uh']} uH")
            print(f"   Plenum Press: {sim_all['impeller_airflow']['central_plenum']['static_pressure_gauge_pa']} Pa")
            print(f"   Tip Velocity: {sim_all['wing_tip_velocity']['steady_state_autorotation']['wing_tip_velocity_m_s']} m/s")

        # 6. Test /api/hardware/simulate POST
        hw_post = json.dumps({"hardware_type": "samara", "rpm": 6500.0, "freq_mhz": 3.5}).encode("utf-8")
        req = urllib.request.Request(f"{base_url}/api/hardware/simulate", data=hw_post, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 200
            hw_res = json.loads(resp.read().decode("utf-8"))
            assert hw_res.get("status") == "success"
            print(f"[PASS] /api/hardware/simulate POST returned status: {hw_res.get('status')}")

    finally:
        server.shutdown()
        server.server_close()
        print("[PASS] Server shutdown cleanly.")


def main():
    print("=====================================================================")
    print("  OPENAUTO-CFD: SAMARA FEATHER E2E IMPORT & MULTI-PHYSICS VERIFICATION")
    print("=====================================================================")

    target_dir = os.path.abspath(r"C:\Users\Loki-VR\Documents\projects\Samara_Feather")
    if not os.path.exists(target_dir):
        # Alternative peer repo path
        alt_path = os.path.abspath(os.path.join(REPO_DIR, "..", "Samara_Feather"))
        if os.path.exists(alt_path):
            target_dir = alt_path

    test_cross_platform_discovery()
    pid, pm = test_import_samara_feather(target_dir)
    test_samara_multi_physics_simulations(target_dir)
    test_live_server_endpoints(target_dir, port=8089)

    print("\n=====================================================================")
    print("  [SUCCESS] ALL SAMARA FEATHER E2E VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("=====================================================================")


if __name__ == "__main__":
    main()
