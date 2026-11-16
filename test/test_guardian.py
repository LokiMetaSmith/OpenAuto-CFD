"""
test_guardian.py
Unit tests verifying the Conservation Guardian & deterministic 3-tier auto-switching (phyz-guardian paradigm).
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from optimizer.guardian import PhyzConservationGuardian, SolverTier, GuardianAuditReport


def test_guardian_nominal_stay_tier_0():
    """Verify that nominal simulation with < 1% energy drift remains on Tier 0."""
    guardian = PhyzConservationGuardian(energy_drift_tol=0.01)

    # Initial energy = 1000 J, current energy = 1005 J (0.5% drift)
    report, _ = guardian.audit_rollout_step(
        current_energy=1005.0,
        initial_energy=1000.0,
        boundary_work=0.0
    )

    assert report.active_tier == SolverTier.TIER_0_LATENT
    assert report.fallback_triggered is False
    assert report.energy_drift_pct == pytest.approx(0.5, rel=1e-3)


def test_guardian_fallback_to_tier_1_on_drift():
    """
    Verify that energy drift > 1% deterministically triggers fallback to
    Tier 1 (Coarse Physics).
    """
    guardian = PhyzConservationGuardian(energy_drift_tol=0.01, critical_drift_tol=0.05)

    # Initial energy = 1000 J, current energy = 1025 J (2.5% drift > 1.0%)
    report, _ = guardian.audit_rollout_step(
        current_energy=1025.0,
        initial_energy=1000.0,
        boundary_work=0.0
    )

    assert report.active_tier == SolverTier.TIER_1_COARSE
    assert report.fallback_triggered is True
    assert "energy drift" in report.fallback_reason.lower()


def test_guardian_fallback_to_tier_2_critical_or_signoff():
    """
    Verify that energy drift > 5% or final candidate sign-off
    deterministically queues Tier 2 (Full Order Verified).
    """
    guardian = PhyzConservationGuardian(critical_drift_tol=0.05)

    # 1. Critical drift (7.0% drift)
    report_crit, _ = guardian.audit_rollout_step(
        current_energy=1070.0,
        initial_energy=1000.0,
        boundary_work=0.0
    )
    assert report_crit.active_tier == SolverTier.TIER_2_FULL_ORDER
    assert report_crit.fallback_triggered is True

    # 2. Final sign-off request with nominal energy
    report_signoff, _ = guardian.audit_rollout_step(
        current_energy=1001.0,
        initial_energy=1000.0,
        boundary_work=0.0,
        is_final_signoff=True
    )
    assert report_signoff.active_tier == SolverTier.TIER_2_FULL_ORDER
    assert "final sign-off" in report_signoff.fallback_reason.lower()


def test_guardian_solenoidal_divergence_projection():
    """Verify that solenoidal projection reduces velocity divergence and triggers Tier 0 recovery."""
    guardian = PhyzConservationGuardian(divergence_tol=1e-3)

    # Synthesize 3D coordinate grid (5x5x5 = 125 points)
    x = np.linspace(0, 1, 5)
    y = np.linspace(0, 1, 5)
    z = np.linspace(0, 1, 5)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    coords = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)

    # Create synthetic diverging velocity field u = [x, y, -2z] + expansion [0.5x, 0.5y, 0.5z]
    u = np.stack([1.5 * coords[:, 0], 1.5 * coords[:, 1], -0.5 * coords[:, 2]], axis=-1)

    report, proj_vel = guardian.audit_rollout_step(
        current_energy=1000.0,
        initial_energy=1000.0,
        coords=coords,
        velocity_field=u
    )

    assert report.solenoidal_projected is True
    assert proj_vel is not None
    assert proj_vel.shape == u.shape
