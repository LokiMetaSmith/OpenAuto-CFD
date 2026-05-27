import os
import sys
import yaml
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), 'optimizer'))
from physics_factory import PhysicsEngineFactory
from scad_driver import ScadDriver
from simulation_runner import run_simulation

def run_em_test(config_file):
    print(f"Running EM proof-of-concept using {config_file}...")

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    # Base parameters for testing
    params = {
        "length": 50.0,
        "thickness": 2.0
    }

    scad_file = config.get('geometry', {}).get('parametric_model', 'dipole_antenna.scad')
    scad = ScadDriver(scad_file)

    physics_driver = PhysicsEngineFactory.get_driver(
        case_dir="em_test_case",
        config=config,
        container_engine="auto",
        verbose=True
    )

    metrics, png_paths, solid_stl, fluid_stl, vtk = run_simulation(
        scad,
        physics_driver,
        params,
        output_stl_name="dipole.stl"
    )

    print("\n=== Final EM Metrics ===")
    print(metrics)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test openEMS integration")
    parser.add_argument("--config", type=str, default="configs/example_config.yaml", help="Path to config file")
    args = parser.parse_args()

    run_em_test(args.config)
