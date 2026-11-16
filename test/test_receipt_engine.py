"""
test_receipt_engine.py
Unit tests verifying the fail-closed Claim & Receipt DAG Engine (vcad verification paradigm).
"""

import time
import os
import sys
import json
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from optimizer.receipt_engine import ClaimReceiptGraph, ReceiptState, ClaimReceipt, PredicateRegistry


def test_receipt_graph_resolution_speed():
    """Verify that resolving the entire Claim & Receipt DAG executes in < 50 ms."""
    graph = ClaimReceiptGraph(log_path="/tmp/test_opt_log.jsonl")
    params = {
        "insert_length_mm": 50.0,
        "twist_deg": 720.0,
        "helix_path_radius_mm": 10.0,
        "helix_profile_radius_mm": 4.0,
        "helix_void_profile_radius_mm": 2.5,
        "tube_od_mm": 32.0,
        "tube_wall_mm": 1.5,
        "blade_chamfer_mm": 0.5,
        "barrel_length_mm": 60.0
    }

    t0 = time.perf_counter()
    receipts = graph.resolve(params)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    assert elapsed_ms < 50.0, f"Resolution took {elapsed_ms:.2f} ms (expected < 50 ms)"
    assert graph.is_valid() is True
    assert len(graph.get_violations()) == 0


def test_receipt_graph_fail_closed_on_violation():
    """Verify that any violated constraint marks the graph invalid and captures margin."""
    graph = ClaimReceiptGraph(log_path="/tmp/test_opt_log.jsonl")
    # Violate center clearance: profile_r (12.0) > path_r (10.0) -> center self-intersection
    params = {
        "insert_length_mm": 50.0,
        "twist_deg": 720.0,
        "helix_path_radius_mm": 10.0,
        "helix_profile_radius_mm": 12.0,  # VIOLATION
        "helix_void_profile_radius_mm": 2.5,
        "tube_od_mm": 32.0,
        "tube_wall_mm": 1.5,
        "blade_chamfer_mm": 0.5,
        "barrel_length_mm": 60.0
    }

    graph.resolve(params)
    assert graph.is_valid() is False

    violations = graph.get_violations()
    assert len(violations) >= 1
    v_ids = [v.claim_id for v in violations]
    assert "claim_helix_center_clearance" in v_ids

    center_v = next(v for v in violations if v.claim_id == "claim_helix_center_clearance")
    assert center_v.state == ReceiptState.VIOLATED
    assert center_v.quantitative_margin < 0.0


def test_receipt_graph_stale_tracking():
    """Verify dirty tracking: mutating an upstream parameter marks dependent receipts STALE."""
    graph = ClaimReceiptGraph(log_path="/tmp/test_opt_log.jsonl")
    params = {
        "insert_length_mm": 50.0,
        "twist_deg": 720.0,
        "helix_path_radius_mm": 10.0,
        "helix_profile_radius_mm": 4.0,
        "helix_void_profile_radius_mm": 2.5,
        "tube_od_mm": 32.0,
        "tube_wall_mm": 1.5,
        "blade_chamfer_mm": 0.5,
        "barrel_length_mm": 60.0
    }
    graph.resolve(params)
    assert graph.is_valid() is True

    # Mutate only blade_chamfer_mm
    graph.mark_stale_for_params({"blade_chamfer_mm"})
    stale_receipts = graph.get_stale()
    assert len(stale_receipts) == 1
    assert stale_receipts[0].claim_id == "claim_blade_chamfer"

    # Re-resolving clears stale state
    graph.resolve(params)
    assert len(graph.get_stale()) == 0


def test_receipt_graph_atomic_log_recording(tmp_path):
    """Verify atomic write-locking to JSONL log."""
    log_file = tmp_path / "test_receipts.jsonl"
    graph = ClaimReceiptGraph(log_path=str(log_file))
    params = {
        "insert_length_mm": 50.0,
        "twist_deg": 720.0,
        "helix_path_radius_mm": 10.0,
        "helix_profile_radius_mm": 4.0,
        "helix_void_profile_radius_mm": 2.5,
        "tube_od_mm": 32.0,
        "tube_wall_mm": 1.5,
        "blade_chamfer_mm": 0.5,
        "barrel_length_mm": 60.0
    }
    graph.resolve(params)
    graph.record_to_log(run_id="run_test_001", extra_meta={"agent": "agent_1_test"})

    assert os.path.exists(log_file)
    with open(log_file, "r") as f:
        lines = f.readlines()
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["id"] == "run_test_001"
        assert data["receipt_summary"]["valid"] is True
        assert len(data["receipts"]) >= 7
