"""
verify_gencad.py

Verification suite for GenCAD (Physics-Driven CAD Embedding & Retrieval):
  1. Test Physics Encoding & Contrastive Latent Retrieval (top-k nearest CAD programs).
  2. Test Continuous CAD Parameter Synthesis conditioned on CFD target feature vectors.
  3. Test Executable CAD Script Generation (OpenSCAD .scad & build123d .py formats).
  4. Test CAD Engine Factory integration & STL export to confirm clean connection to downstream CFD meshing.
  5. Test CADAgentToolRegistry GenCAD tool integration.
"""

import os
import sys

# Ensure optimizer directory is in python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "optimizer"))

from gencad_driver import GenCADDriver
from cad_agent_tools import CADAgentToolRegistry
from cad_factory import CadEngineFactory


def test_gencad_encoding_and_retrieval():
    print("\n--- Test 1: GenCAD Physics Encoding & Latent Space Retrieval ---")
    driver = GenCADDriver()

    # Target physics profile 1: High efficiency filter requirement
    target_cfd_1 = {
        "delta_p": 2500.0,
        "separation_efficiency": 95.0,
        "flow_rate_m3s": 0.012,
        "drag_coefficient": 0.65
    }

    retrieved = driver.retrieve_cad_program(target_cfd_1, top_k=3)
    assert len(retrieved) == 3, f"Expected 3 matches, got {len(retrieved)}"
    assert retrieved[0]["similarity_score"] >= retrieved[1]["similarity_score"]
    assert retrieved[1]["similarity_score"] >= retrieved[2]["similarity_score"]

    top_match_name = retrieved[0]["name"]
    print(f"  CFD Target 1 (95% efficiency, 2500 Pa) -> Top match: '{top_match_name}' (sim: {retrieved[0]['similarity_score']})")

    # Target physics profile 2: Aerofoil high lift requirement
    target_cfd_2 = {
        "lift_coefficient": 1.4,
        "drag_coefficient": 0.09,
        "delta_p": 400.0,
        "separation_efficiency": 0.0
    }

    retrieved_2 = driver.retrieve_cad_program(target_cfd_2, top_k=1)
    top_match_2 = retrieved_2[0]["name"]
    print(f"  CFD Target 2 (High lift airfoil) -> Top match: '{top_match_2}' (sim: {retrieved_2[0]['similarity_score']})")
    assert "Airfoil" in top_match_2 or "High Lift" in top_match_2

    print("[PASS] Test 1: Physics latent space encoding and top-k retrieval validated.")


def test_gencad_synthesis_and_script_generation():
    print("\n--- Test 2: GenCAD Parameter Synthesis & CAD Code Export ---")
    driver = GenCADDriver()

    target_cfd = {
        "delta_p": 2200.0,
        "separation_efficiency": 93.5,
        "flow_rate_m3s": 0.011
    }

    # Synthesize continuous parameters
    synth_params = driver.synthesize_cad_parameters(target_cfd)
    assert "number_of_complete_revolutions" in synth_params
    assert "helix_path_radius_mm" in synth_params
    assert "helix_profile_radius_mm" in synth_params
    assert "blade_chamfer_mm" in synth_params

    print(f"  Synthesized CAD Parameters: {synth_params}")

    # Generate OpenSCAD code
    scad_script = driver.generate_cad_script(synth_params, format_type="openscad")
    assert "number_of_complete_revolutions" in scad_script
    assert "module corkscrew_vane()" in scad_script
    assert len(scad_script) > 200

    # Generate build123d code
    py_script = driver.generate_cad_script(synth_params, format_type="build123d")
    assert "from build123d import *" in py_script
    assert "Helix(pitch=pitch" in py_script
    assert len(py_script) > 200

    # Full export workflow
    export_res = driver.generate_and_export(
        physics_target=target_cfd,
        output_dir="artifacts",
        filename_prefix="test_gencad_export",
        format_type="build123d"
    )
    assert export_res["status"] == "success"
    assert os.path.exists(export_res["output_file"])
    print(f"  Successfully exported build123d script to {export_res['output_file']}")

    print("[PASS] Test 2: Parameter synthesis and CAD script generation validated.")


def test_cad_factory_downstream_meshing_connection():
    print("\n--- Test 3: Downstream CAD Meshing Connection ---")
    driver = GenCADDriver()

    target_cfd = {
        "delta_p": 2100.0,
        "separation_efficiency": 94.0
    }

    # Generate build123d script
    res = driver.generate_and_export(
        physics_target=target_cfd,
        output_dir="artifacts",
        filename_prefix="gencad_mesh_test",
        format_type="build123d"
    )

    py_path = res["output_file"]
    assert os.path.exists(py_path)

    # Validate instantiation via CadEngineFactory
    cad_driver = CadEngineFactory.get_driver(
        scad_file_path=py_path,
        cad_engine="build123d"
    )
    assert cad_driver is not None

    # Test generating STL for downstream CFD meshing
    stl_out = os.path.join("artifacts", "gencad_mesh_test.stl")
    synth_params = res["synthesized_parameters"]
    gen_ok = cad_driver.generate_stl(synth_params, stl_out)
    assert gen_ok, "Failed to generate STL from synthesized parameters"
    assert os.path.exists(stl_out)

    min_pt, max_pt = cad_driver.get_bounds(stl_out)
    assert min_pt is not None and max_pt is not None
    assert max_pt[0] > min_pt[0]  # x_max > x_min
    assert max_pt[2] > min_pt[2]  # z_max > z_min

    print(f"  CadEngineFactory initialized build123d driver from GenCAD output successfully.")
    print(f"  Generated STL at '{stl_out}' ({os.path.getsize(stl_out)} bytes)")
    print(f"  Bounding box calculated for downstream meshing: min={min_pt}, max={max_pt}")

    print("[PASS] Test 3: Downstream CFD meshing connection confirmed.")


def test_cad_agent_tool_registry_gencad_integration():
    print("\n--- Test 4: CADAgentToolRegistry GenCAD Tool Integration ---")
    registry = CADAgentToolRegistry(artifacts_dir="artifacts")

    # 1. Test retrieve_cad_from_physics_target tool
    ret_res = registry.execute_tool(
        "retrieve_cad_from_physics_target",
        {"physics_target": {"delta_p": 2600.0, "separation_efficiency": 96.0}, "top_k": 2}
    )
    assert ret_res["status"] == "success"
    assert len(ret_res["matches"]) == 2
    print(f"  retrieve_cad_from_physics_target: top match = '{ret_res['matches'][0]['name']}'")

    # 2. Test synthesize_gencad_script tool
    syn_res = registry.execute_tool(
        "synthesize_gencad_script",
        {"physics_target": {"delta_p": 2600.0, "separation_efficiency": 96.0}, "format_type": "build123d"}
    )
    assert syn_res["status"] == "success"
    assert os.path.exists(syn_res["output_file"])
    print(f"  synthesize_gencad_script: generated {syn_res['output_file']}")

    print("[PASS] Test 4: GenCAD agent tools executed cleanly via registry.")


if __name__ == "__main__":
    print("================================================================")
    print("         RUNNING GENCAD PHYSICS-DRIVEN CAD RETRIEVAL SUITE      ")
    print("================================================================")
    test_gencad_encoding_and_retrieval()
    test_gencad_synthesis_and_script_generation()
    test_cad_factory_downstream_meshing_connection()
    test_cad_agent_tool_registry_gencad_integration()
    print("\n>>> ALL GENCAD TESTS PASSED SUCCESSFULLY! <<<")
