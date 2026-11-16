"""
guardian.py
Runtime Conservation Watchdog & Solver Auto-Switcher (phyz-guardian paradigm).

Monitors energy drift, mass continuity error, and solenoidal velocity divergence.
Enforces deterministic 3-tier solver auto-switching:
  Tier 0 (Sub-10 ms): Neural latent dynamics + DEC Helmholtz-Hodge projection
  Tier 1 (Sub-second): Coarse mesh / compliant penalty contact solver
  Tier 2 (Asynchronous): Full-order OpenFOAM / CalculiX / openEMS verification solve
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

from optimizer.pinn_conservation import PhysicsConservationEnforcer


class SolverTier(str, Enum):
    TIER_0_LATENT = "Tier 0 (Latent Dynamics)"
    TIER_1_COARSE = "Tier 1 (Coarse Physics)"
    TIER_2_FULL_ORDER = "Tier 2 (Full Order Verified)"


@dataclass
class GuardianAuditReport:
    active_tier: SolverTier
    energy_drift_pct: float
    solenoidal_div_l2: float
    mass_continuity_error: float
    fallback_triggered: bool
    fallback_reason: str = ""
    solenoidal_projected: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "active_tier": self.active_tier.value,
            "energy_drift_pct": float(self.energy_drift_pct),
            "solenoidal_div_l2": float(self.solenoidal_div_l2),
            "mass_continuity_error": float(self.mass_continuity_error),
            "fallback_triggered": bool(self.fallback_triggered),
            "fallback_reason": str(self.fallback_reason),
            "solenoidal_projected": bool(self.solenoidal_projected)
        }


class PhyzConservationGuardian:
    """
    Watches system energy, momentum, and divergence during simulation rollouts.
    Deterministically triggers solver tier degradation when conservation drifts.
    """

    def __init__(
        self,
        energy_drift_tol: float = 0.01,       # 1.0% drift triggers Tier 1 fallback
        critical_drift_tol: float = 0.05,     # 5.0% drift triggers Tier 2 fallback
        divergence_tol: float = 1e-4,          # Solenoidal divergence tolerance
        enable_dec_projection: bool = True
    ):
        self.energy_drift_tol = energy_drift_tol
        self.critical_drift_tol = critical_drift_tol
        self.divergence_tol = divergence_tol
        self.enable_dec_projection = enable_dec_projection
        self.dec_enforcer = PhysicsConservationEnforcer(k_neighbors=12)
        self.active_tier: SolverTier = SolverTier.TIER_0_LATENT
        self.audit_history: List[GuardianAuditReport] = []

    def reset(self):
        self.active_tier = SolverTier.TIER_0_LATENT
        self.audit_history.clear()

    def audit_rollout_step(
        self,
        current_energy: float,
        initial_energy: float,
        boundary_work: float = 0.0,
        coords: Optional[np.ndarray] = None,
        velocity_field: Optional[np.ndarray] = None,
        mass_error: float = 0.0,
        is_final_signoff: bool = False
    ) -> Tuple[GuardianAuditReport, Optional[np.ndarray]]:
        """
        Audits one simulation step or neural surrogate prediction.
        Returns:
            report: GuardianAuditReport
            projected_velocity: np.ndarray if DEC projection applied, else None
        """
        # 1. Energy drift audit
        net_drift = abs(current_energy - initial_energy - boundary_work)
        ref_energy = abs(initial_energy) + 1e-6
        energy_drift_ratio = net_drift / ref_energy
        energy_drift_pct = energy_drift_ratio * 100.0

        # 2. Solenoidal velocity divergence audit
        div_l2 = 0.0
        projected_vel = None
        solenoidal_applied = False

        if coords is not None and velocity_field is not None and len(coords) >= 12:
            _, div_l2 = self.dec_enforcer.compute_divergence(coords, velocity_field)

            # If divergence breaches tolerance, attempt DEC solenoidal projection
            if div_l2 > self.divergence_tol and self.enable_dec_projection:
                projected_vel, post_div_l2 = self.dec_enforcer.project_divergence_free(coords, velocity_field)
                div_l2 = post_div_l2
                solenoidal_applied = True

        # 3. Deterministic Tier Switching Logic
        fallback_triggered = False
        fallback_reason = ""

        if is_final_signoff:
            self.active_tier = SolverTier.TIER_2_FULL_ORDER
            fallback_triggered = True
            fallback_reason = "Final sign-off requested; queued Tier 2 full-order solve"
        elif energy_drift_ratio > self.critical_drift_tol:
            self.active_tier = SolverTier.TIER_2_FULL_ORDER
            fallback_triggered = True
            fallback_reason = f"Critical energy drift {energy_drift_pct:.2f}% exceeded {self.critical_drift_tol*100:.1f}%"
        elif energy_drift_ratio > self.energy_drift_tol or div_l2 > self.divergence_tol:
            self.active_tier = SolverTier.TIER_1_COARSE
            fallback_triggered = True
            fallback_reason = f"Energy drift {energy_drift_pct:.2f}% or div(u) {div_l2:.2e} breached Tier 0 threshold"
        else:
            self.active_tier = SolverTier.TIER_0_LATENT

        report = GuardianAuditReport(
            active_tier=self.active_tier,
            energy_drift_pct=energy_drift_pct,
            solenoidal_div_l2=div_l2,
            mass_continuity_error=mass_error,
            fallback_triggered=fallback_triggered,
            fallback_reason=fallback_reason,
            solenoidal_projected=solenoidal_applied
        )
        self.audit_history.append(report)
        return report, projected_vel
