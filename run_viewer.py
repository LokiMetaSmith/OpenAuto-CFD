"""
run_viewer.py

CLI Launcher for OpenAuto-CFD Interactive WebGL Viewer.
Usage:
  .venv\Scripts\python.exe run_viewer.py --port 8080 --domain cfd
"""

import os
import sys
import argparse
import webbrowser
import time

# Add viewer and optimizer to sys.path
sys.path.insert(0, os.path.abspath("viewer"))
sys.path.insert(0, os.path.abspath("optimizer"))

from server import create_server


def main():
    parser = argparse.ArgumentParser(description="OpenAuto-CFD Real-Time Viewer")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind server (default: 8080)")
    parser.add_argument("--domain", type=str, default="cfd", choices=["cfd", "fea", "joint", "em"], help="Physics domain (default: cfd)")
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser automatically")
    args = parser.parse_args()

    server, opt = create_server(port=args.port, domain=args.domain)
    url = f"http://127.0.0.1:{args.port}"

    print(f"\n=================================================================")
    print(f"       OPENAUTO-CFD STUDIO — REAL-TIME MULTI-PHYSICS STUDIO")
    print(f"=================================================================")
    print(f"  Local WebGL URL : {url}")
    print(f"  Active Physics  : {args.domain.upper()}")
    print(f"  Surrogate Model : Zero-Training RBF + FNO-3D")
    print(f"  Inverse Design  : Differentiable L-BFGS-B (Analytic Jacobian)")
    print(f"  Background Queue: Active ({opt.async_queue.max_workers} workers)")
    print(f"=================================================================\n")
    print("Press Ctrl+C in terminal to stop the server.\n")

    if not args.no_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[OpenAutoViewer] Shutting down server gracefully...")
        opt.shutdown()
        server.shutdown()
        print("[OpenAutoViewer] Server stopped.")


if __name__ == "__main__":
    main()
