"""
verify_viewer_api.py

Automated programmatic test suite for Atlas Fields Studio WebGL Viewer Server.
Verifies REST API endpoints (including GenCAD endpoints) and static asset delivery.
"""

import os
import sys
import time
import json
import threading
import urllib.request
import urllib.error

sys.path.insert(0, os.path.abspath("viewer"))
sys.path.insert(0, os.path.abspath("optimizer"))

from server import create_server


def test_viewer_server_apis():
    print("\n--- Test: Viewer Server REST APIs & Static Assets ---")
    port = 8899
    server, opt = create_server(port=port, domain="cfd")

    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    time.sleep(0.3)

    base_url = f"http://127.0.0.1:{port}"

    try:
        # 1. Test Static Index Delivery
        req = urllib.request.Request(f"{base_url}/")
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 200
            html = resp.read().decode("utf-8")
            assert "Atlas Fields Studio" in html
            print("  [PASS] GET / served HTML successfully.")

        # 2. Test CSS & JS Assets
        with urllib.request.urlopen(f"{base_url}/viewer.css") as resp:
            assert resp.status == 200
            print("  [PASS] GET /viewer.css served successfully.")

        with urllib.request.urlopen(f"{base_url}/viewer.js") as resp:
            assert resp.status == 200
            print("  [PASS] GET /viewer.js served successfully.")

        # 3. Test GET /api/status
        with urllib.request.urlopen(f"{base_url}/api/status") as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            assert data["status"] == "ready"
            assert data["domain"] == "cfd"
            assert "number_of_complete_revolutions" in data["param_defs"]
            print(f"  [PASS] GET /api/status: domain={data['domain']}, samples={data['surrogate_samples']}")

        # 4. Test POST /api/predict (Sub-20ms surrogate evaluation)
        p_query = {
            "number_of_complete_revolutions": 2.5,
            "helix_path_radius_mm": 2.2,
            "blade_chamfer_mm": 0.6
        }
        t0 = time.time()
        req = urllib.request.Request(
            f"{base_url}/api/predict",
            data=json.dumps({"params": p_query}).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            t_ms = (time.time() - t0) * 1000.0
            assert "metrics" in data
            assert "delta_p" in data["metrics"]
            assert "separation_efficiency" in data["metrics"]
            print(f"  [PASS] POST /api/predict in {t_ms:.2f} ms: delta_p={data['metrics']['delta_p']:.1f}, eff={data['metrics']['separation_efficiency']:.2f}%")

        # 5. Test GET /api/gencad/synthesize
        req = urllib.request.Request(f"{base_url}/api/gencad/synthesize?delta_p=2500&separation_efficiency=95")
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            assert data["status"] == "success"
            assert "script_code" in data
            print(f"  [PASS] GET /api/gencad/synthesize: generated script from prompt ({data['retrieved_nearest_match']}).")

        # 6. Test GET /api/gencad/sample
        req = urllib.request.Request(f"{base_url}/api/gencad/sample?delta_p=2500&separation_efficiency=95&n_samples=2")
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            assert data["status"] == "success"
            assert len(data["samples"]) == 2
            print(f"  [PASS] GET /api/gencad/sample: sampled {len(data['samples'])} diverse CAD sequences.")

    finally:
        opt.shutdown()
        server.shutdown()
        print("  Server stopped cleanly.")


if __name__ == "__main__":
    print("=================================================================")
    print("Starting Viewer Server Verification...")
    print("=================================================================")
    test_viewer_server_apis()
    print("\n=================================================================")
    print("ALL VIEWER API TESTS PASSED SUCCESSFULLY!")
    print("=================================================================")
