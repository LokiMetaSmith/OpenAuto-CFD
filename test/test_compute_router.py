"""
test_compute_router.py
Unit tests verifying the Hardware-Aware Compute Router & N = 128 break-even dispatch law.
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from optimizer.compute_router import HardwareComputeRouter


def dummy_evaluator(p: np.ndarray) -> float:
    """Mock surrogate evaluation."""
    return float(np.sum(p ** 2))


def test_compute_router_selection_threshold():
    """Verify device selection: CPU for N < 128; GPU_BATCH for N >= 128."""
    router = HardwareComputeRouter()

    # Small interactive batches (< 128) must select CPU
    assert router.select_device(1) == "CPU"
    assert router.select_device(16) == "CPU"
    assert router.select_device(64) == "CPU"
    assert router.select_device(127) == "CPU"

    # Large parallel sweeps (>= 128) must select GPU_BATCH
    assert router.select_device(128) == "GPU_BATCH"
    assert router.select_device(256) == "GPU_BATCH"
    assert router.select_device(1000) == "GPU_BATCH"


def test_compute_router_dispatch_execution():
    """Verify dispatch execution and profiling telemetry."""
    router = HardwareComputeRouter()

    # Interactive slider scrub (N = 1)
    inputs_single = [np.array([10.0, 25.0, 32.0, 0.5])]
    results_single, meta_single = router.dispatch(inputs_single, dummy_evaluator)

    assert len(results_single) == 1
    assert meta_single["device"] == "CPU"
    assert meta_single["is_gpu_eligible"] is False
    assert meta_single["wall_clock_ms"] < 10.0  # sub-10 ms interactive execution

    # Large batch optimization (N = 150)
    inputs_batch = [np.array([10.0, 25.0, 32.0, 0.5]) for _ in range(150)]
    results_batch, meta_batch = router.dispatch(inputs_batch, dummy_evaluator)

    assert len(results_batch) == 150
    assert meta_batch["device"] == "GPU_BATCH"
    assert meta_batch["is_gpu_eligible"] is True


def test_compute_router_benchmark_law():
    """Verify benchmark suite across batch sizes."""
    router = HardwareComputeRouter()
    bench = router.benchmark_dispatch_law(dummy_evaluator, batch_sizes=[1, 16, 128, 256])

    assert 1 in bench and bench[1]["device"] == "CPU"
    assert 16 in bench and bench[16]["device"] == "CPU"
    assert 128 in bench and bench[128]["device"] == "GPU_BATCH"
    assert 256 in bench and bench[256]["device"] == "GPU_BATCH"
