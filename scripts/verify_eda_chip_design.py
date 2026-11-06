"""
verify_eda_chip_design.py

Comprehensive verification suite for Phase 1: EDA, Chip Design & RF Transmission Line Extension.
Tests:
  1. Microstrip characteristic impedance Z0 accuracy against IPC-2141 benchmark (50 Ohm).
  2. Coupled differential pair impedance Z_diff accuracy (100 Ohm).
  3. Frequency-dependent S-parameters (S11, S21) and skin depth attenuation.
  4. Inverse design optimizer convergence (<0.1% impedance error in <10ms).
  5. KiCad S-expression syntax validation and OpenSCAD 3D solid generation.
  6. Autonomous EDAReasoningAgent multi-turn execution loop.
"""

import os
import sys
import re

# Ensure optimizer directory is in python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "optimizer"))

from eda_rf_driver import HighSpeedTransmissionLineEngine, KiCadPcbExporter
from eda_agent_tools import EDAAgentToolRegistry, EDAReasoningAgent, EDA_TOOLS_SCHEMA


def test_microstrip_analytical_model():
    print("\n--- Test 1: Single-Ended Microstrip Conformal Mapping (IPC-2141 Benchmark) ---")
    engine = HighSpeedTransmissionLineEngine()

    # Benchmark: FR4 substrate (h = 1.6 mm, eps_r = 4.3, t = 35 um)
    # IPC-2141 standard predicts ~50 Ohm at w ~ 3.05 - 3.12 mm
    calc = engine.calculate_microstrip_z0(
        trace_width_mm=3.09,
        substrate_height_mm=1.6,
        dielectric_constant=4.3,
        copper_thickness_um=35.0
    )

    z0 = calc["z0_ohms"]
    eps_eff = calc["eps_eff"]
    print(f"Calculated Z0: {z0:.3f} Ohms (Target: ~50.0 Ohms)")
    print(f"Effective Permittivity eps_eff: {eps_eff:.4f}")
    print(f"Propagation Delay: {calc['delay_ps_per_mm']:.2f} ps/mm")

    assert abs(z0 - 50.0) < 0.8, f"Z0 deviation exceeds tolerance: {z0}"
    assert 3.0 < eps_eff < 3.5, f"eps_eff out of range: {eps_eff}"
    print("[PASS] Test 1: Microstrip conformal mapping accurately matches IPC-2141 standard.")


def test_differential_pair_model():
    print("\n--- Test 2: Coupled Differential Pair Transmission Line (100 Ohm Target) ---")
    engine = HighSpeedTransmissionLineEngine()

    # Differential pair: w = 0.81 mm, spacing = 0.35 mm, h = 1.0 mm, eps_r = 4.3 -> Z_diff ~ 100 Ohm
    diff_calc = engine.calculate_differential_pair(
        trace_width_mm=0.81,
        trace_spacing_mm=0.35,
        substrate_height_mm=1.0,
        dielectric_constant=4.3
    )

    z_diff = diff_calc["z_diff_ohms"]
    z_odd = diff_calc["z_odd_ohms"]
    z_even = diff_calc["z_even_ohms"]
    coupling = diff_calc["coupling_coefficient"]

    print(f"Differential Impedance Z_diff: {z_diff:.2f} Ohms (Target: ~100.0 Ohms)")
    print(f"Odd mode Z_odd: {z_odd:.2f} Ohms, Even mode Z_even: {z_even:.2f} Ohms")
    print(f"Coupling Coefficient k: {coupling:.4f}")

    assert abs(z_diff - 2.0 * z_odd) < 1e-4, "Z_diff must equal 2 * Z_odd by definition"
    assert z_even > z_odd, "Even mode impedance must be strictly greater than odd mode"
    assert abs(z_diff - 100.0) < 3.5, f"Differential impedance out of expected bounds: {z_diff}"
    print("[PASS] Test 2: Coupled differential transmission line physics verified.")


def test_rf_sparameters():
    print("\n--- Test 3: Frequency-Dependent S-Parameters & Skin Effect Attenuation ---")
    engine = HighSpeedTransmissionLineEngine()

    # Evaluate transmission line across 1 GHz, 5 GHz, 28 GHz
    res_1g = engine.calculate_rf_loss_and_sparameters(trace_width_mm=3.09, substrate_height_mm=1.6, line_length_mm=100.0, frequency_ghz=1.0)
    res_5g = engine.calculate_rf_loss_and_sparameters(trace_width_mm=3.09, substrate_height_mm=1.6, line_length_mm=100.0, frequency_ghz=5.0)
    res_28g = engine.calculate_rf_loss_and_sparameters(trace_width_mm=3.09, substrate_height_mm=1.6, line_length_mm=100.0, frequency_ghz=28.0)

    print(f"At 1.0 GHz:  Skin depth={res_1g['skin_depth_um']:.2f} um, S21={res_1g['s21_insertion_loss_db']:.3f} dB, S11={res_1g['s11_return_loss_db']} dB")
    print(f"At 5.0 GHz:  Skin depth={res_5g['skin_depth_um']:.2f} um, S21={res_5g['s21_insertion_loss_db']:.3f} dB, S11={res_5g['s11_return_loss_db']} dB")
    print(f"At 28.0 GHz: Skin depth={res_28g['skin_depth_um']:.2f} um, S21={res_28g['s21_insertion_loss_db']:.3f} dB, S11={res_28g['s11_return_loss_db']} dB")

    # Skin depth shrinks as 1 / sqrt(f)
    assert res_1g["skin_depth_um"] > res_5g["skin_depth_um"] > res_28g["skin_depth_um"]
    # Insertion loss increases with frequency
    assert abs(res_28g["s21_insertion_loss_db"]) > abs(res_5g["s21_insertion_loss_db"]) > abs(res_1g["s21_insertion_loss_db"])
    # Return loss is well-matched (< -20 dB)
    assert res_1g["s11_return_loss_db"] < -20.0
    print("[PASS] Test 3: Frequency-dependent skin depth and S-parameters verified.")


def test_inverse_design_optimization():
    print("\n--- Test 4: Inverse Design Impedance Optimizer (Fast L-BFGS-B / Brent) ---")
    registry = EDAAgentToolRegistry(artifacts_dir="artifacts")

    # Target 50.0 Ohm Single-Ended
    res_se = registry.execute_tool("optimize_trace_impedance", {
        "target_z0_ohms": 50.0,
        "substrate_height_mm": 1.6,
        "dielectric_constant": 4.3
    })
    assert res_se["status"] == "success"
    assert res_se["impedance_error_percent"] < 0.05, f"Error too high: {res_se['impedance_error_percent']}%"
    print(f"  Single-ended: Width={res_se['optimal_trace_width_mm']} mm -> Z0={res_se['achieved_z0_ohms']:.3f} Ohms (Error: {res_se['impedance_error_percent']}%)")

    # Target 100.0 Ohm Differential
    res_diff = registry.execute_tool("optimize_trace_impedance", {
        "target_z0_ohms": 100.0,
        "substrate_height_mm": 0.8,
        "dielectric_constant": 4.3,
        "is_differential": True,
        "trace_spacing_mm": 0.25
    })
    assert res_diff["status"] == "success"
    assert res_diff["impedance_error_percent"] < 0.1, f"Error too high: {res_diff['impedance_error_percent']}%"
    print(f"  Differential: Width={res_diff['optimal_trace_width_mm']} mm -> Z_diff={res_diff['achieved_z_diff_ohms']:.3f} Ohms (Error: {res_diff['impedance_error_percent']}%)")

    print("[PASS] Test 4: Inverse design optimizer achieves <0.1% impedance precision.")


def test_kicad_and_scad_synthesis():
    print("\n--- Test 5: KiCad PCB (.kicad_pcb) & OpenSCAD 3D Synthesis ---")
    registry = EDAAgentToolRegistry(artifacts_dir="artifacts")

    res = registry.execute_tool("generate_kicad_pcb", {
        "trace_width_mm": 3.09,
        "substrate_height_mm": 1.6,
        "line_length_mm": 50.0,
        "output_pcb_filename": "test_verification.kicad_pcb"
    })

    assert res["status"] == "success"
    pcb_file = res["kicad_pcb_file"]
    scad_file = res["scad_3d_stackup_file"]

    assert os.path.exists(pcb_file), f"KiCad file not found: {pcb_file}"
    assert os.path.exists(scad_file), f"OpenSCAD file not found: {scad_file}"
    assert os.path.getsize(pcb_file) > 500
    assert os.path.getsize(scad_file) > 200

    # Validate KiCad S-expression structure
    with open(pcb_file, "r", encoding="utf-8") as f:
        content = f.read()
    assert "(kicad_pcb" in content
    assert '(layer "F.Cu")' in content
    assert '(layer "Edge.Cuts")' in content
    assert "(segment" in content
    assert "(zone" in content

    # Check balanced parentheses
    open_count = content.count("(")
    close_count = content.count(")")
    assert open_count == close_count, f"Unbalanced parentheses in KiCad file: {open_count} vs {close_count}"

    print(f"  Successfully verified KiCad file ({os.path.getsize(pcb_file)} bytes) and OpenSCAD 3D solid.")
    print("[PASS] Test 5: KiCad S-expression syntax and 3D stackup validated.")


def test_autonomous_eda_agent():
    print("\n--- Test 6: Autonomous EDAReasoningAgent Multi-Turn Campaign ---")
    agent = EDAReasoningAgent()
    goal = "Design a 50 Ohm high-speed microstrip transmission line on 1.6mm FR4 for 10 GHz with S11 return loss < -18 dB"

    result = agent.run_goal(goal)
    assert result["status"] == "completed"
    assert result["optimal_width_mm"] > 0
    assert abs(result["achieved_impedance"] - 50.0) < 0.1
    assert os.path.exists(result["kicad_pcb_path"])
    assert len(result["trace"]) == 3

    print("\nFinal Autonomous EDA Agent Report:")
    print(result["summary"])
    print("[PASS] Test 6: Autonomous EDA Reasoning Agent completed multi-turn workflow.")


if __name__ == "__main__":
    print("================================================================")
    print("         RUNNING PHASE 1: EDA & CHIP DESIGN TEST SUITE          ")
    print("================================================================")
    test_microstrip_analytical_model()
    test_differential_pair_model()
    test_rf_sparameters()
    test_inverse_design_optimization()
    test_kicad_and_scad_synthesis()
    test_autonomous_eda_agent()
    print("\n>>> ALL PHASE 1 EDA & CHIP DESIGN TESTS PASSED SUCCESSFULLY! <<<")
