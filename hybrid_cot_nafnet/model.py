# NAFNet portions are derived from megvii-research/NAFNet (MIT License).
# See the repository-level LICENSE file for attribution and license terms.
"""Self-contained NAFNet backbone and CoT-inspired hybrid model.

The implementation intentionally avoids importing BasicSR so the Kaggle pipeline only
needs PyTorch, Pillow and NumPy. Backbone parameter names match the official
NAFNet implementation, allowing compatible checkpoints to be loaded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .modules import GatedCoTAdapter


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        # Shape deliberately matches the official NAFNet checkpoint.
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        # Deep GoPro checkpoints can overflow when variance is accumulated in
        # FP16. Keep normalization statistics in FP32 and cast back afterwards.
        input_dtype = x.dtype
        x_float = x.float()
        mean = x_float.mean(dim=1, keepdim=True)
        variance = (x_float - mean).square().mean(dim=1, keepdim=True)
        normalized = (x_float - mean) * torch.rsqrt(variance + self.eps)
        output = (
            normalized * self.weight.float()[None, :, None, None]
            + self.bias.float()[None, :, None, None]
        )
        return output.to(input_dtype)


class SimpleGate(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        first, second = x.chunk(2, dim=1)
        return first * second


class NAFBlock(nn.Module):
    """Official NAFNet block expressed without a BasicSR dependency."""

    def __init__(self, channels: int, dw_expand: int = 2, ffn_expand: int = 2) -> None:
        super().__init__()
        dw_channels = channels * dw_expand
        self.conv1 = nn.Conv2d(channels, dw_channels, 1)
        self.conv2 = nn.Conv2d(
            dw_channels, dw_channels, 3, padding=1, groups=dw_channels
        )
        self.conv3 = nn.Conv2d(dw_channels // 2, channels, 1)
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channels // 2, dw_channels // 2, 1),
        )
        self.sg = SimpleGate()

        ffn_channels = channels * ffn_expand
        self.conv4 = nn.Conv2d(channels, ffn_channels, 1)
        self.conv5 = nn.Conv2d(ffn_channels // 2, channels, 1)
        self.norm1 = LayerNorm2d(channels)
        self.norm2 = LayerNorm2d(channels)
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, inp: Tensor) -> Tensor:
        x = self.conv1(self.norm1(inp))
        x = self.sg(self.conv2(x))
        x = self.conv3(x * self.sca(x))
        y = inp + x * self.beta
        x = self.conv5(self.sg(self.conv4(self.norm2(y))))
        return y + x * self.gamma


class NAFNet(nn.Module):
    """NAFNet baseline with a checkpoint-compatible state dictionary."""

    def __init__(
        self,
        img_channel: int = 3,
        width: int = 32,
        middle_blk_num: int = 1,
        enc_blk_nums: Sequence[int] = (1, 1, 1, 28),
        dec_blk_nums: Sequence[int] = (1, 1, 1, 1),
    ) -> None:
        super().__init__()
        if len(enc_blk_nums) != len(dec_blk_nums):
            raise ValueError("Encoder and decoder must have the same number of stages")
        self.intro = nn.Conv2d(img_channel, width, 3, padding=1)
        self.ending = nn.Conv2d(width, img_channel, 3, padding=1)
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        channels = width
        for block_count in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(*[NAFBlock(channels) for _ in range(block_count)])
            )
            self.downs.append(nn.Conv2d(channels, 2 * channels, 2, stride=2))
            channels *= 2
        self.middle_blks = nn.Sequential(
            *[NAFBlock(channels) for _ in range(middle_blk_num)]
        )
        for block_count in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(channels, 2 * channels, 1, bias=False),
                    nn.PixelShuffle(2),
                )
            )
            channels //= 2
            self.decoders.append(
                nn.Sequential(*[NAFBlock(channels) for _ in range(block_count)])
            )
        self.padder_size = 2 ** len(self.encoders)

    def check_image_size(self, x: Tensor) -> Tensor:
        _, _, height, width = x.shape
        pad_h = (self.padder_size - height % self.padder_size) % self.padder_size
        pad_w = (self.padder_size - width % self.padder_size) % self.padder_size
        return F.pad(x, (0, pad_w, 0, pad_h))

    def _encode(self, x: Tensor) -> Tuple[Tensor, List[Tensor]]:
        skips: List[Tensor] = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            skips.append(x)
            x = down(x)
        return x, skips

    def _decode(self, x: Tensor, skips: Sequence[Tensor]) -> Tensor:
        for decoder, up, skip in zip(self.decoders, self.ups, reversed(skips)):
            x = decoder(up(x) + skip)
        return x

    def forward(self, inp: Tensor) -> Tensor:
        original_height, original_width = inp.shape[-2:]
        padded = self.check_image_size(inp)
        x, skips = self._encode(self.intro(padded))
        x = self._decode(self.middle_blks(x), skips)
        output = self.ending(x) + padded
        return output[:, :, :original_height, :original_width]


class CoTNAFNet(NAFNet):
    """NAFNet with one bottleneck reasoner and gates on every skip level."""

    def __init__(
        self,
        adapter_hidden: int = 64,
        use_skip_gates: bool = True,
        use_multiscale_degradation: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        width = int(kwargs.get("width", 32))
        skip_channels = [width * (2**level) for level in range(len(self.encoders))]
        bottleneck_channels = width * (2 ** len(self.encoders))
        self.cot_adapter = GatedCoTAdapter(
            bottleneck_channels=bottleneck_channels,
            skip_channels=skip_channels,
            hidden_channels=adapter_hidden,
            num_degradations=4,
            use_skip_gates=use_skip_gates,
            use_multiscale_degradation=use_multiscale_degradation,
        )

    def forward(self, inp: Tensor, return_aux: bool = False):
        original_height, original_width = inp.shape[-2:]
        padded = self.check_image_size(inp)
        x, skips = self._encode(self.intro(padded))
        x = self.middle_blks(x)
        x, skips, auxiliary = self.cot_adapter(x, skips)
        x = self._decode(x, skips)
        output = (self.ending(x) + padded)[:, :, :original_height, :original_width]
        if return_aux:
            return output, auxiliary
        return output


class OrderConditioner(nn.Module):
    """Zero-initialized affine modulation from a privileged binary order code."""

    def __init__(
        self,
        bottleneck_channels: int,
        skip_channels: Sequence[int],
        hidden_channels: int = 32,
        modulation_limit: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_channels < 2:
            raise ValueError("hidden_channels must be at least 2")
        self.bottleneck_channels = int(bottleneck_channels)
        self.skip_channels = tuple(int(value) for value in skip_channels)
        self.modulation_limit = float(modulation_limit)
        self.embedding = nn.Embedding(2, hidden_channels)
        self.planner = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.GELU(),
        )
        channels = (self.bottleneck_channels,) + self.skip_channels
        self.affine = nn.Linear(hidden_channels, 2 * sum(channels))
        # Both true-code and fixed-code models are exactly the pretrained NAFNet
        # function at initialization. Only training can activate conditioning.
        nn.init.zeros_(self.affine.weight)
        nn.init.zeros_(self.affine.bias)

    def _modulate(self, feature: Tensor, scale: Tensor, bias: Tensor) -> Tensor:
        limit = self.modulation_limit
        return feature * (1.0 + limit * torch.tanh(scale)) + limit * torch.tanh(bias)

    def forward(
        self,
        bottleneck: Tensor,
        skips: Sequence[Tensor],
        order_ids: Tensor,
    ) -> Tuple[Tensor, List[Tensor]]:
        if order_ids.ndim != 1 or order_ids.shape[0] != bottleneck.shape[0]:
            raise ValueError(
                "order_ids must have shape [batch] matching the image batch"
            )
        if torch.any((order_ids < 0) | (order_ids > 1)):
            raise ValueError("order_ids must contain only 0 (A) or 1 (B)")
        plan = self.planner(self.embedding(order_ids.long()))
        affine = self.affine(plan)
        channels = (self.bottleneck_channels,) + self.skip_channels
        groups = affine.split([2 * value for value in channels], dim=1)
        scale, bias = groups[0].split(self.bottleneck_channels, dim=1)
        bottleneck = self._modulate(
            bottleneck, scale[:, :, None, None], bias[:, :, None, None]
        )
        conditioned_skips: List[Tensor] = []
        for skip, item, channel_count in zip(skips, groups[1:], self.skip_channels):
            scale, bias = item.split(channel_count, dim=1)
            conditioned_skips.append(
                self._modulate(
                    skip, scale[:, :, None, None], bias[:, :, None, None]
                )
            )
        return bottleneck, conditioned_skips


class OrderConditionedNAFNet(NAFNet):
    """NAFNet with a matched tiny conditioner for the privileged-order oracle."""

    def __init__(self, order_hidden: int = 32, **kwargs) -> None:
        super().__init__(**kwargs)
        width = int(kwargs.get("width", 32))
        skip_channels = [width * (2**level) for level in range(len(self.encoders))]
        bottleneck_channels = width * (2 ** len(self.encoders))
        self.order_conditioner = OrderConditioner(
            bottleneck_channels,
            skip_channels,
            hidden_channels=order_hidden,
        )

    def forward(self, inp: Tensor, order_ids: Tensor | None = None) -> Tensor:
        original_height, original_width = inp.shape[-2:]
        padded = self.check_image_size(inp)
        x, skips = self._encode(self.intro(padded))
        x = self.middle_blks(x)
        if order_ids is None:
            order_ids = torch.zeros(inp.shape[0], dtype=torch.long, device=inp.device)
        x, skips = self.order_conditioner(x, skips, order_ids)
        x = self._decode(x, skips)
        output = self.ending(x) + padded
        return output[:, :, :original_height, :original_width]


@dataclass(frozen=True)
class ModelPreset:
    width: int
    enc_blk_nums: Tuple[int, ...]
    middle_blk_num: int
    dec_blk_nums: Tuple[int, ...]


PRESETS: Dict[str, ModelPreset] = {
    # Exact topologies used by the four official checkpoints available on Kaggle.
    "gopro32": ModelPreset(32, (1, 1, 1, 28), 1, (1, 1, 1, 1)),
    "gopro64": ModelPreset(64, (1, 1, 1, 28), 1, (1, 1, 1, 1)),
    "sidd32": ModelPreset(32, (2, 2, 4, 8), 12, (2, 2, 2, 2)),
    "sidd64": ModelPreset(64, (2, 2, 4, 8), 12, (2, 2, 2, 2)),
    # Backward-compatible name used by the first MVP checkpoints.
    "nafnet32": ModelPreset(32, (1, 1, 1, 28), 1, (1, 1, 1, 1)),
    # Faster option for smoke tests and constrained Kaggle sessions.
    "compact": ModelPreset(24, (1, 1, 2, 4), 2, (1, 1, 1, 1)),
}


def build_model(
    model_type: str = "hybrid",
    preset: str = "nafnet32",
    adapter_hidden: int = 64,
    use_skip_gates: bool = True,
    use_multiscale_degradation: bool = False,
    order_hidden: int = 32,
) -> nn.Module:
    if preset not in PRESETS:
        raise KeyError(f"Unknown preset {preset!r}; choose from {sorted(PRESETS)}")
    config = PRESETS[preset]
    kwargs = dict(
        img_channel=3,
        width=config.width,
        enc_blk_nums=config.enc_blk_nums,
        middle_blk_num=config.middle_blk_num,
        dec_blk_nums=config.dec_blk_nums,
    )
    if model_type == "baseline":
        return NAFNet(**kwargs)
    if model_type == "hybrid":
        return CoTNAFNet(
            adapter_hidden=adapter_hidden,
            use_skip_gates=use_skip_gates,
            use_multiscale_degradation=use_multiscale_degradation,
            **kwargs,
        )
    if model_type == "order_conditioned":
        return OrderConditionedNAFNet(order_hidden=order_hidden, **kwargs)
    raise KeyError("model_type must be 'baseline', 'hybrid', or 'order_conditioned'")


def count_parameters(model: nn.Module) -> Dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    adapter = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if name.startswith(("cot_adapter.", "order_conditioner."))
    )
    return {"total": total, "adapter": adapter, "backbone": total - adapter}
