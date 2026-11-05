"""
verify_distributed_swarm.py

Verification suite for Phase 2: Distributed Worker Swarm & Git-Based Job Queue Integration.
Tests:
  1. Concurrent multi-worker claim & atomic race resolution without double-processing.
  2. Heartbeat tracking & dead-worker stale job recovery.
  3. Distributed two-stage active screening (pruning sub-par designs).
  4. Automatic multi-physics surrogate synchronization and refitting from distributed log.
"""

import os
import sys
import time
import shutil
import tempfile
import threading
from typing import Dict, Any, List

# Ensure optimizer directory is in python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "optimizer"))

from data_store import DataStore
from job_manager import JobManager
from surrogate_multiphysics import MultiPhysicsSurrogate
from model_fusion_multifidelity import TwoStageScreeningFilter


def test_concurrent_worker_claiming():
    print("\n--- Test 1: Concurrent Worker Claiming & Race Condition Resolution ---")
    tmp_dir = tempfile.mkdtemp(prefix="swarm_test_")
    log_file = os.path.join(tmp_dir, "test_swarm.jsonl")

    store = DataStore(log_file=log_file)
    manager = JobManager(store)

    # Queue 12 jobs
    job_ids = []
    for i in range(12):
        p = {"number_of_complete_revolutions": 1.5 + (i * 0.2), "helix_path_radius_mm": 2.0}
        job_ids.append(manager.create_job(p))

    print(f"Created {len(job_ids)} jobs in distributed queue.")

    # 3 Concurrent Workers
    worker_names = ["Worker-Alpha", "Worker-Beta", "Worker-Gamma"]
    claimed_by = {w: [] for w in worker_names}
    lock = threading.Lock()

    def worker_thread(w_name):
        while True:
            job = manager.claim_next_job(w_name)
            if not job:
                break
            with lock:
                claimed_by[w_name].append(job["id"])
            # Simulate short execution
            time.sleep(0.01)
            # Complete job with synthetic metrics
            n = job["parameters"]["number_of_complete_revolutions"]
            manager.complete_job(job["id"], {
                "delta_p": float(1500.0 + n * 600.0),
                "separation_efficiency": float(88.0 + n * 2.5)
            })

    threads = [threading.Thread(target=worker_thread, args=(w,)) for w in worker_names]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Verify each job was claimed exactly once
    all_claimed = []
    for w, j_list in claimed_by.items():
        print(f"  {w} processed {len(j_list)} jobs: {j_list[:3]}...")
        all_claimed.extend(j_list)

    assert len(all_claimed) == 12, f"Expected 12 total processed jobs, got {len(all_claimed)}"
    assert len(set(all_claimed)) == 12, "Duplicate job execution detected! Race condition occurred."

    # Verify no pending jobs remain
    pending = manager.get_pending_jobs()
    assert len(pending) == 0, f"Expected 0 pending jobs, got {len(pending)}"
    print("[PASS] Test 1: Concurrent worker claiming resolved all 12 jobs without collisions.")

    shutil.rmtree(tmp_dir, ignore_errors=True)


def test_dead_worker_requeue():
    print("\n--- Test 2: Dead-Worker Stale Job Reclamation ---")
    tmp_dir = tempfile.mkdtemp(prefix="dead_worker_test_")
    log_file = os.path.join(tmp_dir, "test_stale.jsonl")

    store = DataStore(log_file=log_file)
    manager = JobManager(store)

    # Worker-Dead claims job 500 seconds ago
    job_id = manager.create_job({"helix_path_radius_mm": 2.2})
    claimed = manager.claim_job(job_id, worker_id="Worker-Dead", timestamp=time.time() - 500.0)
    assert claimed is True

    # Record old heartbeat from 500 seconds ago
    store.append_result({
        "id": "hb_Worker-Dead_old",
        "status": "heartbeat",
        "worker_id": "Worker-Dead",
        "timestamp": time.time() - 500.0,
        "parameters": {"_heartbeat": True}
    })

    # Worker-Live runs stale check (threshold: 300s)
    manager.record_heartbeat("Worker-Live")
    requeued = manager.requeue_stale_jobs(stale_threshold_sec=300.0)
    print(f"  Requeued count: {requeued}")
    assert requeued == 1, f"Expected 1 requeued job, got {requeued}"

    # Worker-Live can now claim the orphaned job
    reclaimed_job = manager.claim_next_job("Worker-Live")
    assert reclaimed_job is not None
    assert reclaimed_job["id"] == job_id
    print(f"  Worker-Live successfully reclaimed orphaned job: {job_id}")

    manager.complete_job(job_id, {"delta_p": 2450.0, "separation_efficiency": 93.5})
    print("[PASS] Test 2: Dead-worker stale job reclamation verified.")

    shutil.rmtree(tmp_dir, ignore_errors=True)


def test_distributed_two_stage_screening():
    print("\n--- Test 3: Distributed Two-Stage Screening ---")
    screening = TwoStageScreeningFilter(
        min_efficiency_threshold=89.0,
        max_pressure_drop_threshold=4500.0
    )

    # Design A: High performance candidate
    cand_a = {"delta_p": 2600.0, "separation_efficiency": 94.0}
    # Design B: Non-viable candidate (poor efficiency)
    cand_b = {"delta_p": 4900.0, "separation_efficiency": 84.0}

    pass_a, msg_a = screening.evaluate_coarse_candidate(cand_a)
    pass_b, msg_b = screening.evaluate_coarse_candidate(cand_b)

    print(f"  Candidate A: Passed={pass_a} ({msg_a})")
    print(f"  Candidate B: Passed={pass_b} ({msg_b})")

    assert pass_a is True, "Candidate A should have passed screening"
    assert pass_b is False, "Candidate B should have been rejected"
    print("[PASS] Test 3: Distributed two-stage candidate screening verified.")


def test_surrogate_sync_from_swarm():
    print("\n--- Test 4: Automatic Surrogate Sync and Refitting from Swarm Log ---")
    tmp_dir = tempfile.mkdtemp(prefix="sync_test_")
    log_file = os.path.join(tmp_dir, "test_sync.jsonl")

    store = DataStore(log_file=log_file)
    manager = JobManager(store)

    # Initialize empty surrogate
    param_names = ["number_of_complete_revolutions", "helix_path_radius_mm"]
    surrogate = MultiPhysicsSurrogate(domain="cfd", param_names=param_names)
    assert surrogate.is_fitted is False

    # Simulate 5 completed jobs from different workers
    for i in range(5):
        n = 2.0 + i * 0.4
        r = 1.8 + i * 0.3
        j_id = manager.create_job({"number_of_complete_revolutions": n, "helix_path_radius_mm": r})
        manager.claim_job(j_id, f"Worker-{i}")
        manager.complete_job(j_id, {
            "delta_p": float(1800.0 + n * 500.0),
            "separation_efficiency": float(89.0 + n * 2.0)
        })

    # Sync surrogate
    added = manager.sync_surrogate_from_completed_jobs(surrogate)
    print(f"  Synced {added} samples from distributed log.")
    assert added == 5
    assert surrogate.is_fitted is True

    # Test prediction on synced model
    pred, unc = surrogate.predict_metrics({"number_of_complete_revolutions": 2.8, "helix_path_radius_mm": 2.4})
    print(f"  Surrogate prediction: delta_p={pred['delta_p']:.1f} Pa, eff={pred['separation_efficiency']:.2f}% (unc={unc:.3f})")
    assert "delta_p" in pred
    assert pred["delta_p"] > 2000.0

    print("[PASS] Test 4: Automatic surrogate synchronization and refitting verified.")

    shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    print("================================================================")
    print("      RUNNING PHASE 2: DISTRIBUTED WORKER SWARM TEST SUITE      ")
    print("================================================================")
    test_concurrent_worker_claiming()
    test_dead_worker_requeue()
    test_distributed_two_stage_screening()
    test_surrogate_sync_from_swarm()
    print("\n>>> ALL PHASE 2 DISTRIBUTED SWARM TESTS PASSED SUCCESSFULLY! <<<")
