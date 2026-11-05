"""
verify_gencad.py

Verification suite for GenCAD (Physics & Cross-Modal Vision-Driven CAD Embedding, AST Sequence Tokenization & Transformer Model):
  1. Test Physics Encoding & Contrastive Latent Retrieval (top-k nearest CAD programs).
  2. Test CAD Sequence Tokenizer (AST Token encoding and Python build123d deserialization).
  3. Test PyTorch GenCADTransformerModel autoregressive sequence inference.
  4. Test Phase 2 Cross-Modal Vision Rendering & Image-Conditioned CAD Generation.
  5. Test CAD Engine Factory integration & STL export to confirm clean connection to downstream CFD meshing.
  6. Test CADAgentToolRegistry GenCAD tool integration (physics & vision tools).
"""

import os
import sys

# Ensure optimizer directory is in python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "optimizer"))

from gencad_driver import (
    GenCADDriver,
    CADSequenceTokenizer,
    GenCADTransformerModel,
    render_stl_to_image_array,
    HAS_TORCH
)
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


def test_cad_sequence_tokenizer_and_ast():
    print("\n--- Test 2: CAD Sequence Tokenizer & AST Generator ---")
    tokenizer = CADSequenceTokenizer()
    assert tokenizer.vocab_size > 30

    test_params = {
        "number_of_complete_revolutions": 3.0,
        "helix_path_radius_mm": 2.5,
        "helix_profile_radius_mm": 1.5,
        "blade_chamfer_mm": 0.8
    }

    # Encode to AST token IDs
    token_ids = tokenizer.encode_parameters(test_params)
    assert len(token_ids) > 10
    assert token_ids[0] == tokenizer.bos_id
    assert token_ids[-1] == tokenizer.eos_id

    # Decode back to parameters
    decoded_params = tokenizer.decode_tokens_to_parameters(token_ids)
    assert decoded_params["number_of_complete_revolutions"] == 3.0
    assert decoded_params["blade_chamfer_mm"] == 0.8

    # Deserialize to executable build123d code
    script_code = tokenizer.decode_tokens_to_script(token_ids, format_type="build123d")
    assert "from build123d import *" in script_code
    assert "Helix(pitch=pitch" in script_code

    print(f"  Tokenizer vocabulary size: {tokenizer.vocab_size}")
    print(f"  Encoded {len(token_ids)} AST sequence tokens -> Decoded parameters cleanly.")
    print("[PASS] Test 2: CAD sequence tokenization and AST deserialization validated.")


def test_gencad_transformer_model():
    print("\n--- Test 3: PyTorch GenCAD Transformer Sequence Generation ---")
    assert HAS_TORCH, "PyTorch is required for Transformer model test"

    driver = GenCADDriver()
    target_cfd = {
        "delta_p": 2300.0,
        "separation_efficiency": 94.5,
        "flow_rate_m3s": 0.011
    }

    # Execute transformer sequence generation
    tok_ids, synth_params, script_code = driver.generate_transformer_ast_sequence(target_cfd)
    assert len(tok_ids) > 5
    assert "number_of_complete_revolutions" in synth_params
    assert "from build123d import *" in script_code

    print(f"  Generated AST Token Sequence IDs ({len(tok_ids)} tokens): {tok_ids[:8]}...")
    print(f"  Transformer generated parameters: {synth_params}")
    print("[PASS] Test 3: PyTorch GenCAD Transformer model sequence generation validated.")


def test_gencad_cross_modal_vision():
    print("\n--- Test 4: Phase 2 Cross-Modal Vision Rendering & Image Generation ---")
    driver = GenCADDriver()
    stl_path = "artifacts/gencad_mesh_test.stl"
    img_out = "artifacts/test_wireframe.png"

    # 1. Render 3D STL to 2D wireframe PNG image
    img_arr = render_stl_to_image_array(stl_path, output_path=img_out)
    assert img_arr.shape == (224, 224, 3)
    assert os.path.exists(img_out)
    print(f"  Rendered 3D STL to 2D image '{img_out}' ({os.path.getsize(img_out)} bytes)")

    # 2. Retrieve CAD program from 2D image input
    matches = driver.retrieve_cad_from_image(img_out, top_k=2)
    assert len(matches) == 2
    top_match = matches[0]["name"]
    print(f"  Retrieved CAD program from 2D image: '{top_match}' (sim: {matches[0]['similarity_score']})")

    # 3. Generate executable build123d script from 2D image
    gen_res = driver.generate_cad_from_image(img_out, format_type="build123d")
    assert gen_res["status"] == "success"
    assert "from build123d import *" in gen_res["script_code"]
    print(f"  Generated CAD script from image successfully.")

    print("[PASS] Test 4: Phase 2 Cross-Modal Vision rendering and image-conditioned generation validated.")


def test_cad_factory_downstream_meshing_connection():
    print("\n--- Test 5: Downstream CAD Meshing Connection ---")
    driver = GenCADDriver()

    target_cfd = {
        "delta_p": 2100.0,
        "separation_efficiency": 94.0
    }

    # Generate build123d script via Transformer AST sequence workflow
    res = driver.generate_and_export(
        physics_target=target_cfd,
        output_dir="artifacts",
        filename_prefix="gencad_mesh_test",
        format_type="build123d",
        use_transformer_sequence=True
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

    print("[PASS] Test 5: Downstream CFD meshing connection confirmed.")


def test_cad_agent_tool_registry_gencad_integration():
    print("\n--- Test 6: CADAgentToolRegistry GenCAD Tool Integration ---")
    registry = CADAgentToolRegistry(artifacts_dir="artifacts")

    # 1. Test retrieve_cad_from_physics_target tool
    ret_res = registry.execute_tool(
        "retrieve_cad_from_physics_target",
        {"physics_target": {"delta_p": 2600.0, "separation_efficiency": 96.0}, "top_k": 2}
    )
    assert ret_res["status"] == "success"
    assert len(ret_res["matches"]) == 2
    print(f"  retrieve_cad_from_physics_target: top match = '{ret_res['matches'][0]['name']}'")

    # 2. Test retrieve_cad_from_image tool
    ret_img_res = registry.execute_tool(
        "retrieve_cad_from_image",
        {"image_path": "artifacts/test_wireframe.png", "top_k": 2}
    )
    assert ret_img_res["status"] == "success"
    assert len(ret_img_res["matches"]) == 2
    print(f"  retrieve_cad_from_image: top match = '{ret_img_res['matches'][0]['name']}'")

    # 3. Test generate_cad_from_image tool
    gen_img_res = registry.execute_tool(
        "generate_cad_from_image",
        {"image_path": "artifacts/test_wireframe.png", "format_type": "build123d"}
    )
    assert gen_img_res["status"] == "success"
    assert "from build123d import *" in gen_img_res["script_code"]
    print(f"  generate_cad_from_image: generated code successfully.")

    print("[PASS] Test 6: GenCAD agent physics and vision tools executed cleanly via registry.")


if __name__ == "__main__":
    print("================================================================")
    print("  RUNNING GENCAD TRANSFORMER & CROSS-MODAL VISION SUITE        ")
    print("================================================================")
    test_gencad_encoding_and_retrieval()
    test_cad_sequence_tokenizer_and_ast()
    test_gencad_transformer_model()
    test_gencad_cross_modal_vision()
    test_cad_factory_downstream_meshing_connection()
    test_cad_agent_tool_registry_gencad_integration()
    print("\n>>> ALL GENCAD TRANSFORMER & CROSS-MODAL VISION TESTS PASSED SUCCESSFULLY! <<<")
