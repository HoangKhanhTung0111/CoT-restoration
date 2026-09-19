"""Numerically audit candidate interaction-supervision losses.

This is a CPU-only sanity check.  It does not train a restoration model; it
verifies which candidate objectives are algebraically equivalent to ordinary
composite reconstruction and which ones add an independent training signal.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _maximum_absolute_difference(left: Tensor, right: Tensor) -> float:
    return float((left - right).abs().max().item())


def _gradient(
    loss: Tensor, parameter: Tensor, *, retain_graph: bool = False
) -> Optional[Tensor]:
    result = torch.autograd.grad(
        loss,
        parameter,
        retain_graph=retain_graph,
        allow_unused=True,
    )[0]
    return result


def run_audit(seed: int) -> Dict[str, object]:
    generator = torch.Generator().manual_seed(seed)
    shape = (2, 3, 7, 5)
    y = torch.rand(shape, generator=generator, dtype=torch.float64)
    x_a = torch.rand(shape, generator=generator, dtype=torch.float64)
    x_b = torch.rand(shape, generator=generator, dtype=torch.float64)
    x_ab = torch.rand(shape, generator=generator, dtype=torch.float64)

    r_a = y - x_a
    r_b = y - x_b
    r_ab = y - x_ab
    u_ab = r_ab - r_a - r_b

    # Candidate 1: define the predicted interaction from the predicted final
    # residual and ground-truth primitive residuals.  This cancels exactly to
    # ordinary composite reconstruction.
    predicted_r_ab = torch.randn(
        shape, generator=generator, dtype=torch.float64, requires_grad=True
    )
    reconstruction_loss = F.mse_loss(predicted_r_ab, r_ab)
    rewritten_interaction_loss = F.mse_loss(
        predicted_r_ab - r_a - r_b,
        u_ab,
    )
    reconstruction_gradient = _gradient(
        reconstruction_loss, predicted_r_ab, retain_graph=True
    )
    rewritten_gradient = _gradient(rewritten_interaction_loss, predicted_r_ab)
    assert reconstruction_gradient is not None
    assert rewritten_gradient is not None

    # Candidate 2: an independent interaction head is a genuinely different
    # auxiliary target.  If the deployed restoration does not consume this
    # head, ordinary reconstruction supplies no gradient to it.
    predicted_output = torch.randn(
        shape, generator=generator, dtype=torch.float64, requires_grad=True
    )
    predicted_u = torch.randn(
        shape, generator=generator, dtype=torch.float64, requires_grad=True
    )
    output_loss = F.mse_loss(predicted_output, y)
    independent_interaction_loss = F.mse_loss(predicted_u, u_ab)
    output_gradient_on_u = _gradient(output_loss, predicted_u)
    interaction_gradient_on_u = _gradient(independent_interaction_loss, predicted_u)
    assert interaction_gradient_on_u is not None

    # Candidate 3: make the decomposition part of the deployed residual.  A
    # reconstruction loss sees only the sum, while component supervision
    # identifies a particular (generator-defined) decomposition.
    predicted_r_a = torch.randn(
        shape, generator=generator, dtype=torch.float64, requires_grad=True
    )
    predicted_r_b = torch.randn(
        shape, generator=generator, dtype=torch.float64, requires_grad=True
    )
    predicted_u_composed = torch.randn(
        shape, generator=generator, dtype=torch.float64, requires_grad=True
    )
    composed_reconstruction_loss = F.mse_loss(
        predicted_r_a + predicted_r_b + predicted_u_composed,
        r_ab,
    )
    component_loss = (
        F.mse_loss(predicted_r_a, r_a)
        + F.mse_loss(predicted_r_b, r_b)
        + F.mse_loss(predicted_u_composed, u_ab)
    )
    composed_gradients = [
        _gradient(composed_reconstruction_loss, parameter, retain_graph=True)
        for parameter in (predicted_r_a, predicted_r_b, predicted_u_composed)
    ]
    component_gradients = [
        _gradient(component_loss, parameter, retain_graph=True)
        for parameter in (predicted_r_a, predicted_r_b, predicted_u_composed)
    ]
    assert all(item is not None for item in composed_gradients)
    assert all(item is not None for item in component_gradients)

    # The consistency identity contains no new label information once all
    # component targets are met: it and its gradients are zero at that point.
    target_r_ab = r_ab.detach().clone().requires_grad_(True)
    target_r_a = r_a.detach().clone().requires_grad_(True)
    target_r_b = r_b.detach().clone().requires_grad_(True)
    target_u = u_ab.detach().clone().requires_grad_(True)
    consistency_at_targets = F.mse_loss(
        target_r_ab,
        target_r_a + target_r_b + target_u,
    )
    consistency_gradients = torch.autograd.grad(
        consistency_at_targets,
        (target_r_ab, target_r_a, target_r_b, target_u),
    )

    tolerance = 1e-12
    rewrite_loss_difference = abs(
        float(reconstruction_loss.item() - rewritten_interaction_loss.item())
    )
    rewrite_gradient_difference = _maximum_absolute_difference(
        reconstruction_gradient, rewritten_gradient
    )
    composed_gradient_difference = max(
        _maximum_absolute_difference(composed_gradients[0], item)
        for item in composed_gradients[1:]
        if item is not None
    )
    component_vs_composed_difference = max(
        _maximum_absolute_difference(composed, component)
        for composed, component in zip(composed_gradients, component_gradients)
        if composed is not None and component is not None
    )

    return {
        "seed": seed,
        "dtype": "float64",
        "naive_rewrite": {
            "loss_absolute_difference": rewrite_loss_difference,
            "gradient_max_absolute_difference": rewrite_gradient_difference,
            "equivalent_within_tolerance": (
                rewrite_loss_difference <= tolerance
                and rewrite_gradient_difference <= tolerance
            ),
        },
        "independent_auxiliary_head": {
            "reconstruction_gradient_on_interaction_head": (
                0.0
                if output_gradient_on_u is None
                else float(output_gradient_on_u.norm().item())
            ),
            "interaction_loss_gradient_norm": float(
                interaction_gradient_on_u.norm().item()
            ),
            "is_distinct_training_signal": interaction_gradient_on_u.norm().item() > 0.0,
        },
        "deployed_additive_decomposition": {
            "max_difference_between_reconstruction_gradients_of_heads": (
                composed_gradient_difference
            ),
            "max_difference_component_vs_reconstruction_gradients": (
                component_vs_composed_difference
            ),
            "interpretation": (
                "Reconstruction constrains only the sum; component targets select "
                "one generator-defined decomposition."
            ),
        },
        "consistency_identity_at_component_targets": {
            "loss": float(consistency_at_targets.item()),
            "maximum_gradient_norm": max(
                float(item.norm().item()) for item in consistency_gradients
            ),
            "adds_new_target_information": False,
        },
    }


def main() -> None:
    args = parse_args()
    result = run_audit(args.seed)
    payload = json.dumps(result, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
