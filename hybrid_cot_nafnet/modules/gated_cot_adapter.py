"""Memory-efficient degradation reasoning and skip-feature modulation."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch
from torch import Tensor, nn


class SimpleGate(nn.Module):
    """NAFNet-style multiplicative gate."""

    def forward(self, x: Tensor) -> Tensor:
        first, second = x.chunk(2, dim=1)
        return first * second


class LayerNorm2d(nn.Module):
    """Layer normalization over channels independently at every pixel."""

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        input_dtype = x.dtype
        x_float = x.float()
        mean = x_float.mean(dim=1, keepdim=True)
        variance = (x_float - mean).square().mean(dim=1, keepdim=True)
        output = (
            (x_float - mean)
            * torch.rsqrt(variance + self.eps)
            * self.weight.float()
            + self.bias.float()
        )
        return output.to(input_dtype)


class GatedCoTAdapter(nn.Module):
    """Infer degradation attributes and modulate bottleneck/skip features.

    The module implements a compact Thinking -> Planning interface:

    * Thinking extracts content and degradation embeddings from the bottleneck.
    * Planning maps the degradation embedding to zero-initialized affine gates.
    * Action is performed by the unchanged NAFNet decoder on gated features.

    Zero initialization makes a newly attached adapter start as an identity map,
    which preserves a loaded NAFNet checkpoint at initialization.
    """

    def __init__(
        self,
        bottleneck_channels: int,
        skip_channels: Sequence[int],
        hidden_channels: int = 64,
        num_degradations: int = 4,
        modulation_limit: float = 0.1,
        use_skip_gates: bool = True,
    ) -> None:
        super().__init__()
        if hidden_channels < 4:
            raise ValueError("hidden_channels must be at least 4")
        self.bottleneck_channels = int(bottleneck_channels)
        self.skip_channels = tuple(int(c) for c in skip_channels)
        self.hidden_channels = int(hidden_channels)
        self.modulation_limit = float(modulation_limit)
        self.use_skip_gates = bool(use_skip_gates)

        self.norm = LayerNorm2d(self.bottleneck_channels)
        self.local_features = nn.Sequential(
            nn.Conv2d(self.bottleneck_channels, 2 * hidden_channels, 1),
            nn.Conv2d(
                2 * hidden_channels,
                2 * hidden_channels,
                3,
                padding=1,
                groups=2 * hidden_channels,
            ),
            SimpleGate(),
        )
        self.content_projection = nn.Conv2d(hidden_channels, hidden_channels, 1)
        self.degradation_projection = nn.Conv2d(hidden_channels, hidden_channels, 1)
        self.degradation_head = nn.Linear(hidden_channels, num_degradations)

        self.planner = nn.Sequential(
            nn.Linear(hidden_channels, 2 * hidden_channels),
            _VectorSimpleGate(),
        )
        all_channels = (self.bottleneck_channels,) + self.skip_channels
        self.modulation_head = nn.Linear(hidden_channels, 2 * sum(all_channels))
        self.bottleneck_residual = nn.Conv2d(hidden_channels, self.bottleneck_channels, 1)

        # Identity initialization: no bottleneck residual and no affine change.
        nn.init.zeros_(self.modulation_head.weight)
        nn.init.zeros_(self.modulation_head.bias)
        nn.init.zeros_(self.bottleneck_residual.weight)
        nn.init.zeros_(self.bottleneck_residual.bias)

    @staticmethod
    def _pool(x: Tensor) -> Tensor:
        return x.mean(dim=(-2, -1))

    def _split_affine(self, affine: Tensor) -> List[Tuple[Tensor, Tensor]]:
        channels = (self.bottleneck_channels,) + self.skip_channels
        scales_and_biases = affine.split([2 * c for c in channels], dim=1)
        outputs: List[Tuple[Tensor, Tensor]] = []
        for item, channel_count in zip(scales_and_biases, channels):
            scale, bias = item.split(channel_count, dim=1)
            outputs.append((scale[:, :, None, None], bias[:, :, None, None]))
        return outputs

    def _modulate(self, x: Tensor, scale: Tensor, bias: Tensor) -> Tensor:
        limit = self.modulation_limit
        return x * (1.0 + limit * torch.tanh(scale)) + limit * torch.tanh(bias)

    def forward(
        self, bottleneck: Tensor, skips: Sequence[Tensor]
    ) -> Tuple[Tensor, List[Tensor], Dict[str, Tensor]]:
        if len(skips) != len(self.skip_channels):
            raise ValueError(
                f"Expected {len(self.skip_channels)} skip tensors, received {len(skips)}"
            )

        local = self.local_features(self.norm(bottleneck))
        content_map = self.content_projection(local)
        degradation_map = self.degradation_projection(local)
        content_embedding = self._pool(content_map)
        degradation_embedding = self._pool(degradation_map)
        degradation_logits = self.degradation_head(degradation_embedding)

        plan = self.planner(degradation_embedding)
        affine = self.modulation_head(plan)
        affine_groups = self._split_affine(affine)

        bottleneck = bottleneck + self.bottleneck_residual(content_map)
        bottleneck = self._modulate(bottleneck, *affine_groups[0])
        if self.use_skip_gates:
            modulated_skips = [
                self._modulate(skip, *parameters)
                for skip, parameters in zip(skips, affine_groups[1:])
            ]
            active_affine = affine
        else:
            modulated_skips = list(skips)
            active_affine = affine[:, : 2 * self.bottleneck_channels]
        auxiliary = {
            "degradation_logits": degradation_logits,
            "degradation_embedding": degradation_embedding,
            "content_embedding": content_embedding,
            "plan_embedding": plan,
            # Keep scalar statistics one-dimensional so DataParallel can gather
            # values from two Kaggle T4 GPUs without scalar-gather warnings.
            "gate_regularization": torch.tanh(active_affine).abs().mean().reshape(1),
            "gate_mean_abs": torch.tanh(active_affine).abs().detach().mean().reshape(1),
            "gate_max_abs": torch.tanh(active_affine).abs().detach().amax().reshape(1),
        }
        return bottleneck, modulated_skips, auxiliary


class _VectorSimpleGate(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        first, second = x.chunk(2, dim=-1)
        return first * second
