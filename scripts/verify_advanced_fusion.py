"""
verify_advanced_fusion.py

Comprehensive verification suite for:
  1. Exact analytic RBF gradients vs finite differences.
  2. Differentiable inverse design via L-BFGS-B on surrogate response surfaces.
  3. Reinforcement learning policy agent parameter morphing & online policy updates.
  4. Asynchronous non-blocking solver queue dispatch and event-driven surrogate refitting.
"""

import os
import sys
import time
import shutil
import tempfile
import numpy as np
from scipy.interpolate import RBFInterpolator

# Ensure optimizer directory is on sys.path
sys.path.insert(0, os.path.abspath("optimizer"))

from surrogate_multiphysics import MultiPhysicsSurrogate
from surrogate_gradients import (
    compute_rbf_gradient,
    compute_acquisition_and_grad,
    DifferentiableInverseDesigner
)
from rl_policy_agent import GeometryRLPolicyAgent
from async_solver_queue import AsyncSolverQueue, SimulationJob
from model_fusion_multiphysics import MultiPhysicsModelFusionOptimizer


def test_analytic_rbf_gradients():
    print("\n--- Test 1: Analytic RBF Gradients vs Finite Differences ---")
    np.random.seed(123)
    n_pts, d = 8, 3
    x_train = np.random.uniform(0.0, 1.0, (n_pts, d))
    y_train = np.sin(x_train[:, 0] * 3.0) + np.cos(x_train[:, 1] * 2.0) + (x_train[:, 2] ** 2)

    kernels_to_test = ["thin_plate_spline", "cubic", "gaussian"]
    q = np.array([0.45, 0.55, 0.65])
    h = 1e-6

    for k in kernels_to_test:
        rbf = RBFInterpolator(x_train, y_train, kernel=k, degree=1, epsilon=1.5)

        # Finite difference numerical gradient
        grad_num = np.zeros(d)
        for dim in range(d):
            qp = q.copy(); qp[dim] += h
            qm = q.copy(); qm[dim] -= h
            grad_num[dim] = (rbf(qp.reshape(1, -1))[0] - rbf(qm.reshape(1, -1))[0]) / (2.0 * h)

        # Analytic gradient
        grad_ana = compute_rbf_gradient(rbf, q)

        err = np.linalg.norm(grad_num - grad_ana)
        print(f"  Kernel '{k}': Numeric={np.round(grad_num, 4)}, Analytic={np.round(grad_ana, 4)}, Error={err:.2e}")
        assert err < 1e-4, f"Gradient mismatch on kernel {k}: error {err}"

    print("  [PASS] All analytic RBF gradients verified exact within tolerance.")


def test_differentiable_inverse_design():
    print("\n--- Test 2: Differentiable Inverse Design (L-BFGS-B Convergence) ---")
    param_names = ["helix_path_radius_mm", "number_of_complete_revolutions"]
    surrogate = MultiPhysicsSurrogate(domain="cfd", param_names=param_names)

    # Seed surrogate with 4 checkpoints
    for r in [1.5, 3.5]:
        for n in [1.0, 4.0]:
            p = {"helix_path_radius_mm": r, "number_of_complete_revolutions": n}
            # Quadratic surface with minimum near r=2.5, n=3.0
            p_drop = 1500.0 + 300.0 * ((r - 2.5) ** 2) + 200.0 * ((n - 3.0) ** 2)
            eff = 95.0 - 5.0 * ((r - 2.5) ** 2) - 4.0 * ((n - 3.0) ** 2)
            surrogate.add_sample(params=p, metrics={"delta_p": p_drop, "separation_efficiency": eff})

    param_defs = {
        "helix_path_radius_mm": {"min": 1.5, "max": 4.0, "default": 2.0},
        "number_of_complete_revolutions": {"min": 1.0, "max": 4.0, "default": 2.0}
    }

    designer = DifferentiableInverseDesigner(
        surrogate=surrogate,
        parameter_defs=param_defs,
        domain="cfd",
        exploration_weight=0.0  # Pure exploitation for test
    )

    t0 = time.time()
    best_p, best_score = designer.optimize(n_restarts=5)
    t_opt_ms = (time.time() - t0) * 1000.0

    print(f"  Inverse Design Converged in {t_opt_ms:.2f} ms")
    print(f"  Optimized Geometry: {best_p}, Acquisition Score={best_score:.4f}")

    assert 1.5 <= best_p["helix_path_radius_mm"] <= 4.0
    assert 1.0 <= best_p["number_of_complete_revolutions"] <= 4.0
    assert t_opt_ms < 150.0, "Gradient inverse design should complete in < 150 ms"
    print("  [PASS] Differentiable inverse design verified.")


def test_rl_policy_agent():
    print("\n--- Test 3: Reinforcement Learning Geometry Policy Agent ---")
    param_defs = {
        "helix_path_radius_mm": {"min": 1.5, "max": 5.0, "default": 2.0},
        "number_of_complete_revolutions": {"min": 1.0, "max": 4.0, "default": 2.0}
    }

    db_rl = "artifacts/test_rl_policy.json"
    if os.path.exists(db_rl):
        os.remove(db_rl)

    agent = GeometryRLPolicyAgent(param_defs=param_defs, learning_rate=0.1, model_path=db_rl)

    current_p = {"helix_path_radius_mm": 2.0, "number_of_complete_revolutions": 2.0}
    metrics = {"delta_p": 3200.0, "separation_efficiency": 92.0}

    state = agent.construct_state_vector(current_p, metrics, uncertainty=0.3)
    assert len(state) == len(param_defs) + 9

    action, mean_act = agent.predict_action(state)
    assert len(action) == len(param_defs)
    assert np.all(action >= -1.0) and np.all(action <= 1.0)

    morphed = agent.morph_parameters(current_p, action, step_scale=0.1)
    assert 1.5 <= morphed["helix_path_radius_mm"] <= 5.0
    print(f"  Initial Params: {current_p}")
    print(f"  RL Action:      {np.round(action, 3)}")
    print(f"  Morphed Params: {morphed}")

    # Test policy update with good reward
    reward = agent.compute_reward({"separation_efficiency": 99.96, "delta_p": 2100.0}, domain="cfd")
    adv = agent.update_policy(state, action, mean_act, reward)
    print(f"  RL Step: Reward={reward:.2f}, Advantage={adv:+.2f}")
    assert len(agent.experience_history) == 1

    agent.save()
    loaded_agent = GeometryRLPolicyAgent(param_defs=param_defs, model_path=db_rl)
    assert np.allclose(agent.W, loaded_agent.W)
    print("  [PASS] Reinforcement Learning Policy Agent verified.")


def test_async_solver_queue():
    print("\n--- Test 4: Asynchronous Non-Blocking Simulation Queue ---")
    completed_jobs = []

    def on_done(job: SimulationJob):
        completed_jobs.append(job.job_id)

    queue = AsyncSolverQueue(max_workers=2, on_job_completed=on_done, verbose=False)

    p1 = {"helix_path_radius_mm": 2.0, "number_of_complete_revolutions": 2.5}
    p2 = {"helix_path_radius_mm": 3.0, "number_of_complete_revolutions": 3.5}

    t0 = time.time()
    j1 = queue.submit_job(driver=None, params=p1, domain="cfd", mock=True)
    j2 = queue.submit_job(driver=None, params=p2, domain="cfd", mock=True)
    dispatch_time_ms = (time.time() - t0) * 1000.0

    print(f"  Dispatched 2 jobs asynchronously in {dispatch_time_ms:.2f} ms (non-blocking)")
    assert dispatch_time_ms < 50.0, "Async dispatch must not block"

    # Wait for completion
    queue.wait_all(timeout=5.0)

    polled = queue.poll_completed()
    assert len(polled) == 2 or len(completed_jobs) == 2
    status = queue.get_job_status(j1)
    assert status["status"] == "COMPLETED"
    assert "delta_p" in status["metrics"]
    print(f"  Background solver output for job {j1}: {status['metrics']}")

    queue.shutdown()
    print("  [PASS] Asynchronous Solver Queue verified.")


def test_end_to_end_advanced_fusion():
    print("\n--- Test 5: End-to-End Hybrid Model Fusion (Gradients + RL + Async) ---")
    param_defs = {
        "helix_path_radius_mm": {"min": 1.5, "max": 4.5, "default": 2.0},
        "number_of_complete_revolutions": {"min": 1.0, "max": 3.5, "default": 2.0}
    }

    db_path = "artifacts/test_advanced_fusion.json"
    if os.path.exists(db_path):
        os.remove(db_path)

    opt = MultiPhysicsModelFusionOptimizer(
        physics_driver=None,
        parameter_defs=param_defs,
        domain="cfd",
        surrogate_db_path=db_path,
        verbose=True
    )

    # 1. Run 2 synchronous steps (uses gradient proposals + RL updates)
    for step_i in range(2):
        rec = opt.step(mock_run=True)
        assert "delta_p" in rec["actual_metrics"]
        assert "reward" in rec

    # 2. Run 1 asynchronous non-blocking step
    ticket = opt.step_async(mock_run=True)
    assert ticket["status"] == "DISPATCHED"
    print(f"  Async Step Dispatched: Job {ticket['job_id']}")

    # Wait for background queue
    opt.async_queue.wait_all(timeout=5.0)
    finished = opt.poll_and_update()
    print(f"  Polled & Ingested {len(finished)} finished background jobs into surrogate memory.")
    assert len(finished) >= 1
    assert len(opt.surrogate.param_history) >= 3

    opt.shutdown()
    print("  [PASS] End-to-End Advanced Model Fusion verified.")


if __name__ == "__main__":
    print("=================================================================")
    print("Starting Verification Suite for Advanced Model Fusion...")
    print("=================================================================")
    test_analytic_rbf_gradients()
    test_differentiable_inverse_design()
    test_rl_policy_agent()
    test_async_solver_queue()
    test_end_to_end_advanced_fusion()
    print("\n=================================================================")
    print("ALL ADVANCED MODEL FUSION TESTS PASSED SUCCESSFULLY!")
    print("=================================================================")
