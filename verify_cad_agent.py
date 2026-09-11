"""
verify_cad_agent.py

Comprehensive verification suite for Phase 4:
  1. JSON Schema validation for Kimi K3 / Gemini / OpenAI tool specifications.
  2. Individual CAD tool execution via CADAgentToolRegistry:
     - predict_surrogate
     - run_inverse_design
     - check_physics_conservation
     - dispatch_simulation & check_simulation_status
     - generate_scad_code (manifold validation & file generation)
  3. Geometric constraint rejection test (ensuring invalid geometries are caught).
  4. Autonomous CAD Reasoning Agent multi-turn workflow execution.
  5. LLMAgent integration test.
"""

import os
import sys
import time

# Ensure optimizer directory is in python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "optimizer"))

from cad_agent_tools import (
    CADAgentToolRegistry,
    CADReasoningAgent,
    CAD_TOOLS_SCHEMA
)
from llm_agent import LLMAgent


def test_tool_schemas():
    print("\n--- Test 1: JSON Tool Schemas (OpenAI / Kimi / Gemini compatible) ---")
    assert len(CAD_TOOLS_SCHEMA) == 11, f"Expected 11 tools, found {len(CAD_TOOLS_SCHEMA)}"

    expected_tool_names = {
        "predict_surrogate",
        "run_inverse_design",
        "check_physics_conservation",
        "dispatch_simulation",
        "check_simulation_status",
        "generate_scad_code",
        "retrieve_cad_from_physics_target",
        "synthesize_gencad_script",
        "retrieve_cad_from_image",
        "generate_cad_from_image",
        "sample_diverse_cad_programs"
    }

    actual_names = set()
    for tool in CAD_TOOLS_SCHEMA:
        assert tool["type"] == "function"
        fn = tool["function"]
        name = fn["name"]
        actual_names.add(name)
        assert len(fn["description"]) > 10, f"Tool {name} description too short"
        assert "parameters" in fn
        assert fn["parameters"]["type"] == "object"
        assert "properties" in fn["parameters"]
        print(f"  [OK] Validated tool schema: '{name}'")

    assert actual_names == expected_tool_names, f"Mismatch in tool names: {actual_names}"
    print("[PASS] Test 1: All 11 CAD & Multi-Physics tool schemas validated.")


def test_registry_tools():
    print("\n--- Test 2: CADAgentToolRegistry Execution ---")
    registry = CADAgentToolRegistry(artifacts_dir="artifacts")

    # 1. predict_surrogate
    test_params = {
        "number_of_complete_revolutions": 2.5,
        "helix_path_radius_mm": 2.0,
        "helix_profile_radius_mm": 1.4,
        "blade_chamfer_mm": 0.5
    }
    p_res = registry.execute_tool("predict_surrogate", {"params": test_params, "domain": "joint"})
    assert p_res["status"] == "success"
    assert "predicted_metrics" in p_res
    assert "epistemic_uncertainty" in p_res
    print(f"  predict_surrogate: metrics={list(p_res['predicted_metrics'].keys())}, unc={p_res['epistemic_uncertainty']}")

    # 2. run_inverse_design
    inv_res = registry.execute_tool("run_inverse_design", {"domain": "joint", "enforce_physics": True})
    assert inv_res["status"] == "success"
    assert "optimal_params" in inv_res
    assert "acquisition_score" in inv_res
    opt_p = inv_res["optimal_params"]
    print(f"  run_inverse_design: score={inv_res['acquisition_score']}, opt_params={opt_p}")

    # 3. check_physics_conservation
    phys_res = registry.execute_tool("check_physics_conservation", {"params": opt_p})
    assert phys_res["status"] == "success"
    assert "divergence_continuity_residual" in phys_res
    assert phys_res["is_physically_admissible"] is True
    print(f"  check_physics_conservation: div_loss={phys_res['divergence_continuity_residual']:.6f}")

    # 4. dispatch_simulation & check_simulation_status
    disp_res = registry.execute_tool("dispatch_simulation", {"params": opt_p, "domain": "cfd", "fidelity": "coarse"})
    assert disp_res["status"] == "queued"
    job_id = disp_res["job_id"]
    print(f"  dispatch_simulation: enqueued {job_id}")

    # Wait for simulation execution
    registry.async_queue.wait_all(timeout=3.0)
    status_res = registry.execute_tool("check_simulation_status", {"job_id": job_id})
    assert status_res["status"] == "success"
    job_info = status_res["job"]
    assert job_info["status"] == "COMPLETED"
    assert "delta_p" in job_info["metrics"]
    print(f"  check_simulation_status: job {job_id} {job_info['status']} with delta_p={job_info['metrics']['delta_p']}")

    # 5. generate_scad_code
    scad_res = registry.execute_tool("generate_scad_code", {"params": opt_p, "output_filename": "test_agent_filter.scad"})
    assert scad_res["status"] == "success"
    assert os.path.exists(scad_res["filepath"])
    assert scad_res["file_size_bytes"] > 200
    print(f"  generate_scad_code: wrote {scad_res['filepath']} ({scad_res['file_size_bytes']} bytes)")

    print("[PASS] Test 2: All registry tool handlers executed cleanly.")


def test_geometric_constraint_rejection():
    print("\n--- Test 3: Geometric Manifold Constraint Rejection ---")
    registry = CADAgentToolRegistry(artifacts_dir="artifacts")

    # Invalid geometry: profile_r (2.5) >= path_r (2.0) -> causes center-axis self-intersection
    invalid_params = {
        "number_of_complete_revolutions": 3.0,
        "helix_path_radius_mm": 2.0,
        "helix_profile_radius_mm": 2.5,
        "blade_chamfer_mm": 0.5
    }

    res = registry.execute_tool("generate_scad_code", {"params": invalid_params})
    assert res["status"] == "validation_error"
    assert "violat" in res["message"].lower() or "singularity" in res["error"].lower()
    print(f"  Successfully caught invalid geometry: {res['error']}")
    print("[PASS] Test 3: Self-intersecting and non-manifold parameters properly rejected.")


def test_autonomous_reasoning_agent():
    print("\n--- Test 4: Autonomous CAD Reasoning Agent Workflow ---")
    agent = CADReasoningAgent()
    goal = "Optimize lunar regolith corkscrew filter for >95% efficiency, pressure drop < 2500 Pa, and max stress < 30 MPa"

    result = agent.run_goal(goal)
    assert result["status"] == "completed"
    assert "optimal_parameters" in result
    assert "simulation_metrics" in result
    assert len(result["trace"]) >= 4
    assert os.path.exists(result["cad_file"])
    print("\nAgent Final Engineering Rationale & Summary:")
    print(result["summary"])
    print("[PASS] Test 4: Multi-turn CAD reasoning loop completed all turns successfully.")


def test_llm_agent_factory():
    print("\n--- Test 5: LLMAgent Integration ---")
    llm = LLMAgent(api_key="MOCK_TEST_KEY")
    agent = llm.create_cad_reasoning_agent()
    assert agent is not None
    assert isinstance(agent.registry, CADAgentToolRegistry)
    print("  LLMAgent successfully spawned CADReasoningAgent with full tool registry.")
    print("[PASS] Test 5: LLMAgent CAD factory interface operational.")


if __name__ == "__main__":
    print("================================================================")
    print("        RUNNING PHASE 4: CAD AGENT TOOL-CALLING SUITE           ")
    print("================================================================")
    test_tool_schemas()
    test_registry_tools()
    test_geometric_constraint_rejection()
    test_autonomous_reasoning_agent()
    test_llm_agent_factory()
    print("\n>>> ALL PHASE 4 CAD AGENT TOOL-CALLING TESTS PASSED SUCCESSFULLY! <<<")
