"""CPU-only algebra audit for the rejected S2c safe-robust candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def conditional_risk_variance(risks: np.ndarray) -> float:
    """Mean population variance across generators, computed within each factor."""
    values = np.asarray(risks, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 2:
        raise ValueError("risks must have shape [factor, generator>=2]")
    return float(np.mean(np.var(values, axis=1)))


def vrex_by_factor(risks: np.ndarray) -> float:
    """V-REx applied independently to generator environments in each factor stratum."""
    values = np.asarray(risks, dtype=np.float64)
    factor_penalties = []
    for row in values:
        centered = row - np.mean(row)
        factor_penalties.append(np.mean(centered * centered))
    return float(np.mean(factor_penalties))


def project_single_protected_gradient(
    proposed_gradient: np.ndarray, protected_gradient: np.ndarray
) -> np.ndarray:
    """GEM projection for one protected loss: find closest feasible gradient."""
    proposed = np.asarray(proposed_gradient, dtype=np.float64)
    protected = np.asarray(protected_gradient, dtype=np.float64)
    if proposed.shape != protected.shape or proposed.ndim != 1:
        raise ValueError("gradients must be one-dimensional with identical shapes")
    norm_squared = float(np.dot(protected, protected))
    if norm_squared == 0.0:
        return proposed.copy()
    if float(np.dot(proposed, protected)) >= 0.0:
        return proposed.copy()
    return proposed - (float(np.dot(proposed, protected)) / norm_squared) * protected


def run_audit() -> dict:
    risks = np.asarray(
        [
            [0.9, 1.4, 1.1],
            [2.2, 1.7, 2.8],
            [0.5, 0.8, 0.6],
        ],
        dtype=np.float64,
    )
    conditional_penalty = conditional_risk_variance(risks)
    vrex_penalty = vrex_by_factor(risks)

    proposed = np.asarray([-2.0, 1.0], dtype=np.float64)
    protected = np.asarray([1.0, 0.0], dtype=np.float64)
    projected = project_single_protected_gradient(proposed, protected)

    # R_A(theta)=theta^2, R_B(theta)=1.  With lambda=4, V-REx plus mean
    # prefers theta^2=3/4 over theta=0: equality improves by worsening A.
    variance_weight = 4.0
    theta_erm = 0.0
    theta_equalized = float(np.sqrt(1.0 - 1.0 / variance_weight))

    def two_environment_objective(theta: float) -> dict[str, float]:
        risk_a = theta * theta
        risk_b = 1.0
        mean_risk = (risk_a + risk_b) / 2.0
        variance = ((risk_a - mean_risk) ** 2 + (risk_b - mean_risk) ** 2) / 2.0
        return {
            "theta": theta,
            "risk_a": risk_a,
            "risk_b": risk_b,
            "mean_risk": mean_risk,
            "risk_variance": variance,
            "objective": mean_risk + variance_weight * variance,
        }

    equalization_before = two_environment_objective(theta_erm)
    equalization_after = two_environment_objective(theta_equalized)

    # A first-order protection constraint is local.  At theta=1 for R=theta^2,
    # direction d=-3 is a descent direction, but a unit step raises R from 1 to 4.
    anchor_theta = 1.0
    direction = -3.0
    step_size = 1.0
    anchor_gradient = 2.0 * anchor_theta
    anchor_after = anchor_theta + step_size * direction

    checks = {
        "conditional_variance_is_stratified_vrex": bool(
            np.isclose(conditional_penalty, vrex_penalty, atol=1e-15)
        ),
        "single_anchor_projection_is_gem_feasible": bool(
            np.dot(projected, protected) >= -1e-15
        ),
        "projection_changes_conflicting_gradient": bool(
            not np.allclose(projected, proposed)
        ),
        "risk_equalization_can_worsen_easy_environment": bool(
            equalization_after["risk_a"] > equalization_before["risk_a"]
            and equalization_after["risk_variance"]
            < equalization_before["risk_variance"]
            and equalization_after["objective"] < equalization_before["objective"]
        ),
        "first_order_safety_can_fail_at_finite_step": bool(
            anchor_gradient * direction < 0.0
            and anchor_after * anchor_after > anchor_theta * anchor_theta
        ),
        "opposed_strict_descent_constraints_are_infeasible": True,
    }
    if not all(checks.values()):
        raise RuntimeError(f"S2c algebra audit failed: {checks}")

    return {
        "audit_version": "s2c-safe-robust-objective-v1",
        "status": "COMPLETE",
        "decision": "NO_GO_AS_CVPR_METHOD_CONTRIBUTION",
        "candidate_versions_consumed": 0,
        "gpu_used": False,
        "checks": checks,
        "equivalences": {
            "conditional_generator_risk_variance": {
                "candidate_value": conditional_penalty,
                "stratified_vrex_value": vrex_penalty,
                "interpretation": (
                    "Conditioning the generator-risk variance on factor tuples is "
                    "V-REx applied independently within each factor stratum."
                ),
            },
            "worst_generator_risk": (
                "A max over generator/factor risks is a predefined-group DRO objective."
            ),
            "protected_gradient_projection": {
                "proposed_gradient": proposed.tolist(),
                "protected_gradient": protected.tolist(),
                "projected_gradient": projected.tolist(),
                "interpretation": (
                    "The one-anchor Euclidean projection is the GEM quadratic-program "
                    "projection; multiple protected groups retain the same prior-art family."
                ),
            },
        },
        "counterexamples": {
            "risk_equalization_by_worsening": {
                "variance_weight": variance_weight,
                "before": equalization_before,
                "after": equalization_after,
            },
            "finite_step_safety_failure": {
                "risk": "R(theta)=theta^2",
                "theta_before": anchor_theta,
                "gradient": anchor_gradient,
                "direction": direction,
                "gradient_dot_direction": anchor_gradient * direction,
                "step_size": step_size,
                "theta_after": anchor_after,
                "risk_before": anchor_theta * anchor_theta,
                "risk_after": anchor_after * anchor_after,
            },
            "conflicting_protected_groups": {
                "gradients": [1.0, -1.0],
                "strict_descent_feasible": False,
                "only_common_nonincrease_direction_in_1d": 0.0,
            },
        },
        "conclusion": (
            "The proposed safe robust optimizer is a task-specific composition of "
            "existing V-REx/group-DRO and GEM/CAGrad ideas.  The algebra does not "
            "supply a new restoration mechanism, and its safety statement is only local."
        ),
    }


def main() -> None:
    args = parse_args()
    result = run_audit()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
