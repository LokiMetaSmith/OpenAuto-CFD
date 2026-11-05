import math

import os
import sys
import yaml
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), 'optimizer'))
from physics_factory import PhysicsEngineFactory

def calculate_analytical_cfd_physics():
    """Legacy analytical CFD calculator for the corkscrew filter."""
    # --- Parameters ---
    rho_air = 1.225  # kg/m^3
    mu_air = 1.81e-5 # Pa·s
    d_tube_mm = 19.05 # mm
    d_tube = d_tube_mm / 1000.0 # m
    r_coil_mm = 25.0
    r_coil = r_coil_mm / 1000.0
    rho_particle = 1590.0 # kg/m^3
    d_particle_microns = 10.0
    d_particle = d_particle_microns * 1e-6 # m
    v_high = 40.0 # m/s
    v_low_min = 5.0 # m/s
    v_low_max = 10.0 # m/s

    print("--- Analytical CFD Calculation Parameters ---")
    print(f"Tube Diameter (D): {d_tube_mm:.2f} mm")
    print(f"Coil Radius (Rc): {r_coil_mm:.2f} mm")
    print(f"Particle Diameter (dp): {d_particle_microns} microns")
    print(f"Fluid Density: {rho_air} kg/m^3")
    print(f"Fluid Viscosity: {mu_air} Pa·s")
    print("-------------------------------------------\n")

    def calc_re(v):
        return (rho_air * v * d_tube) / mu_air

    def calc_de(re):
        return re * math.sqrt(d_tube / (2 * r_coil))

    def calc_stk(v):
        tau_p = (rho_particle * (d_particle**2)) / (18 * mu_air)
        tau_f = d_tube / v
        return tau_p / tau_f

    re_high = calc_re(v_high)
    de_high = calc_de(re_high)
    stk_high = calc_stk(v_high)
    print(f"--- High Velocity ~{v_high} m/s ---")
    print(f"Re: {re_high:.0f}, De: {de_high:.0f}, Stk: {stk_high:.2f}")

    re_low_min, re_low_max = calc_re(v_low_min), calc_re(v_low_max)
    de_low_min, de_low_max = calc_de(re_low_min), calc_de(re_low_max)
    stk_low_min, stk_low_max = calc_stk(v_low_min), calc_stk(v_low_max)

    print(f"\n--- Low Velocity {v_low_min}-{v_low_max} m/s ---")
    print(f"Re: {re_low_min:.0f} - {re_low_max:.0f}")
    print(f"De: {de_low_min:.0f} - {de_low_max:.0f}")
    print(f"Stk: {stk_low_min:.2f} - {stk_low_max:.2f}")


def extract_metrics_from_case(config_file, case_dir):
    """
    Acts as a bridge/factory to extract metrics from an existing simulation case
    using the appropriate physics engine driver based on the config.
    """
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading configuration file {config_file}: {e}")
        return None

    # Instantiate physics driver using the factory
    try:
        physics_driver = PhysicsEngineFactory.get_driver(case_dir, config=config)
    except ValueError as e:
        print(f"Factory Error: {e}")
        return None

    print(f"Extracting metrics from {case_dir} using {physics_driver.__class__.__name__}...")
    metrics = physics_driver.get_metrics()

    import json
    print("\n--- Extracted Metrics ---")
    print(json.dumps(metrics, indent=2))
    print("-------------------------\n")
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Physics Calculation & Extraction Bridge")
    parser.add_argument("--config", type=str, help="Path to the problem definition YAML file")
    parser.add_argument("--case-dir", type=str, help="Path to the simulation case directory")
    args = parser.parse_args()

    if args.config and args.case_dir:
        extract_metrics_from_case(args.config, args.case_dir)
    else:
        # Fallback to legacy analytical CFD calculations if no args provided
        calculate_analytical_cfd_physics()
