"""
verify_viewer_upgrades.py

Verification test suite for Live WebGL Viewer Upgrades (HUD Physics Telemetry & Chat Drawer).
Tests:
  1. Static Assets Integrity (HTML, CSS, JS elements for HUD & Agent Drawer)
  2. Server Factory & Status Endpoint
  3. Real-Time Predict Endpoint with PINN Conservation Telemetry
  4. Agent Chat Endpoint: Autonomous CAD Optimization Reasoning
  5. Agent Chat Endpoint: Autonomous EDA / RF Microstrip Design Routing
"""

import os
import sys
import json
import time
import urllib.request
import threading

sys.path.insert(0, os.path.abspath("optimizer"))
sys.path.insert(0, os.path.abspath("viewer"))

from server import create_server


def test_static_assets_integrity():
    print("\n--- Test 1: Static Assets Integrity (HUD & Agent Drawer) ---")
    html_path = os.path.join("viewer", "web", "index.html")
    css_path = os.path.join("viewer", "web", "viewer.css")
    js_path = os.path.join("viewer", "web", "viewer.js")

    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()
    with open(css_path, "r", encoding="utf-8") as f:
        css = f.read()
    with open(js_path, "r", encoding="utf-8") as f:
        js = f.read()

    # Verify HTML elements
    assert "fidelity-selector" in html, "Fidelity selector element missing in index.html"
    assert "btn-fidelity-t1" in html, "Tier 1 button missing in index.html"
    assert "btn-fidelity-t2" in html, "Tier 2 button missing in index.html"
    assert "card-conservation" in html, "PINN conservation card missing in index.html"
    assert "agent-drawer" in html, "Agent drawer container missing in index.html"
    assert "agent-chat-messages" in html, "Agent messages log missing in index.html"

    # Verify CSS rules
    assert ".fidelity-selector" in css, "Fidelity selector CSS missing"
    assert ".agent-drawer" in css, "Agent drawer CSS missing"
    assert ".badge-status-dot.admissible" in css, "Conservation badge dot CSS missing"

    # Verify JS functions
    assert "setFidelity" in js, "setFidelity function missing in viewer.js"
    assert "toggleAgentDrawer" in js, "toggleAgentDrawer function missing in viewer.js"
    assert "sendAgentMessage" in js, "sendAgentMessage function missing in viewer.js"

    print("  index.html, viewer.css, and viewer.js contain all required HUD, Telemetry, and Agent Drawer components.")
    print("[PASS] Test 1: Static Assets Integrity verified.")


def test_live_server_endpoints():
    print("\n--- Test 2: Live Embedded Viewer Server API Endpoints ---")
    test_port = 8189
    server, opt = create_server(port=test_port, domain="cfd")

    # Start server in background daemon thread
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    time.sleep(0.5)

    base_url = f"http://127.0.0.1:{test_port}"

    try:
        # 1. Test /api/status
        req = urllib.request.urlopen(f"{base_url}/api/status")
        assert req.status == 200
        status_data = json.loads(req.read().decode("utf-8"))
        assert status_data["status"] == "ready"
        assert status_data["domain"] == "cfd"
        print(f"  /api/status responded: ready (domain={status_data['domain']}, samples={status_data['surrogate_samples']})")

        # 2. Test /api/predict with PINN Conservation Telemetry
        post_data = json.dumps({
            "params": {
                "number_of_complete_revolutions": 2.5,
                "helix_path_radius_mm": 2.0,
                "helix_profile_radius_mm": 1.4,
                "blade_chamfer_mm": 0.5
            },
            "fidelity": "tier1",
            "enforce_conservation": True
        }).encode("utf-8")

        p_req = urllib.request.Request(
            f"{base_url}/api/predict",
            data=post_data,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(p_req) as resp:
            assert resp.status == 200
            pred_data = json.loads(resp.read().decode("utf-8"))

        assert "metrics" in pred_data
        assert "conservation" in pred_data
        print(f"  Debug pred_data['conservation']: {pred_data['conservation']}")
        assert "divergence_loss" in pred_data["conservation"]
        assert pred_data["conservation"]["is_physically_admissible"] is True
        print(f"  /api/predict responded: dp={pred_data['metrics'].get('delta_p'):.1f} Pa, div_loss={pred_data['conservation']['divergence_loss']:.6f} (Admissible: {pred_data['conservation']['is_physically_admissible']})")
        print("[PASS] Test 2: Live server status and predict endpoints verified.")

        # 3. Test /api/agent_chat: CAD Engineering Reasoning
        print("\n--- Test 3: Agent Chat CAD Optimization Goal ---")
        chat_cad_data = json.dumps({
            "message": "Optimize corkscrew filter for 99.9% dust collection efficiency under 0.5 PSI",
            "params": {"number_of_complete_revolutions": 2.0, "helix_path_radius_mm": 1.8},
            "fidelity": "tier1"
        }).encode("utf-8")

        c_req = urllib.request.Request(
            f"{base_url}/api/agent_chat",
            data=chat_cad_data,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(c_req) as resp:
            assert resp.status == 200
            cad_res = json.loads(resp.read().decode("utf-8"))

        assert cad_res["status"] == "success"
        assert cad_res["agent_type"] == "CAD_Reasoning_Agent"
        assert len(cad_res["trace"]) > 0
        print(f"  CAD Agent response: {cad_res['reply']}")
        print(f"  Autonomous Tools invoked: {[t['tool'] for t in cad_res['trace']]}")
        print(f"  Optimal parameters found: {cad_res.get('updated_params')}")
        print("[PASS] Test 3: Agent Chat CAD Optimization Goal verified.")

        # 4. Test /api/agent_chat: EDA / RF Microstrip Design Routing
        print("\n--- Test 4: Agent Chat EDA / RF Microstrip Routing ---")
        chat_eda_data = json.dumps({
            "message": "Design 50 ohm microstrip trace on FR4 at 2.4 GHz and generate KiCad PCB",
            "params": {},
            "fidelity": "tier1"
        }).encode("utf-8")

        e_req = urllib.request.Request(
            f"{base_url}/api/agent_chat",
            data=chat_eda_data,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(e_req) as resp:
            assert resp.status == 200
            eda_res = json.loads(resp.read().decode("utf-8"))

        assert eda_res["status"] == "success"
        assert eda_res["agent_type"] == "EDA_RF_Agent"
        assert eda_res.get("kicad_path") is not None
        assert os.path.exists(eda_res["kicad_path"])
        print(f"  EDA Agent response: {eda_res['reply']}")
        print(f"  KiCad PCB exported: {eda_res['kicad_path']}")
        print("[PASS] Test 4: Agent Chat EDA / RF Microstrip Routing verified.")

    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    print("================================================================")
    print("       RUNNING PHASE 4: LIVE WEBGL VIEWER UPGRADES TEST         ")
    print("================================================================")
    test_static_assets_integrity()
    test_live_server_endpoints()
    print("\n>>> ALL PHASE 4 LIVE VIEWER UPGRADE TESTS PASSED SUCCESSFULLY! <<<")
