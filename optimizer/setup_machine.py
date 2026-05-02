import subprocess
import argparse
import sys
import shutil
import json

def run_cmd(cmd, description=None, check=True):
    if description:
        print(f"Running: {description}...")
    else:
        print(f"Executing: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, check=check, text=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: Command failed: {e}")
        return False
    except FileNotFoundError:
        print(f"Error: Command not found: {cmd[0]}")
        return False

def check_podman():
    if not shutil.which("podman"):
        print("Error: 'podman' executable not found in PATH.")
        return False
    return True

def get_machine_info():
    try:
        result = subprocess.run(
            ["podman", "machine", "info", "--format", "json"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if result.returncode == 0:
            return json.loads(result.stdout)
    except Exception:
        pass
    return None

def resize_podman_machine(memory_mb, cpus=None, machine_name="podman-machine-default", yes=False):
    if not check_podman():
        return

    print(f"Preparing to resize Podman machine '{machine_name}' to {memory_mb}MB RAM" + (f" and {cpus} CPUs" if cpus else "") + ".")
    print("Warning: This requires restarting the Podman machine. Active containers will be stopped.")

    if not yes:
        confirm = input("Do you want to continue? (y/n) ").strip().lower()
        if confirm != 'y':
            print("Aborted.")
            return

    # 1. Stop Machine
    print("Stopping Podman machine...")
    # Try stopping generic default or specific name
    # 'podman machine stop' stops the default
    if not run_cmd(["podman", "machine", "stop"], check=False):
        print("Warning: Failed to stop machine (maybe it's already stopped or doesn't exist?). Continuing...")

    # 2. Set Resources
    cmd = ["podman", "machine", "set", "--memory", str(memory_mb)]
    if cpus:
        cmd.extend(["--cpus", str(cpus)])

    # Target specific machine if needed, but 'set' usually applies to default
    # If the user has multiple machines, they might need to be specific, but standard setup is one.

    print("Applying new settings...")
    if not run_cmd(cmd):
        print("Failed to set machine resources. Ensure 'podman machine set' is supported (Podman v4+).")
        return

    # 3. Start Machine
    print("Starting Podman machine...")
    if run_cmd(["podman", "machine", "start"]):
        print(f"Successfully resized Podman machine to {memory_mb}MB.")
    else:
        print("Failed to restart Podman machine. Please check logs.")

def main():
    parser = argparse.ArgumentParser(description="Configure Podman Machine Resources")
    parser.add_argument("--memory", type=int, default=16384, help="Target memory in MB (default: 16384)")
    parser.add_argument("--cpus", type=int, help="Target CPU count")
    parser.add_argument("--machine", type=str, default="podman-machine-default", help="Machine name (default: podman-machine-default)")
    parser.add_argument("-y", "--yes", action="store_true", help="Automatically confirm all prompts")

    args = parser.parse_args()

    # Simple check if running on Linux (native) vs Windows/Mac
    if sys.platform == "linux":
        print("Note: On Linux, Podman typically uses host memory/CPUs natively.")
        print("This script is intended for Podman Machine (Windows/Mac/WSL2).")
        print("To limit memory on Linux, use 'podman run --memory ...'.")
        if not args.yes:
            confirm = input("Are you sure you want to proceed? (y/n) ").strip().lower()
            if confirm != 'y':
                return

    resize_podman_machine(args.memory, args.cpus, args.machine, yes=args.yes)

if __name__ == "__main__":
    main()
