from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class LoRASpec:
    name: str
    in_features: int
    out_features: int
    rank: int

    @property
    def num_params(self) -> int:
        return self.rank * self.in_features + self.out_features * self.rank


@dataclass(frozen=True)
class AffineSpec:
    """A per-sample scale and bias update for a normalized feature vector."""

    name: str
    features: int

    @property
    def num_params(self) -> int:
        return 2 * self.features


class HyperLoRALinear(nn.Module):
    """Linear layer with externally supplied LoRA deltas.

    The base linear layer keeps its original parameters. The LoRA tensors are
    regular tensors, not Parameters, so gradients can flow back into a
    hypernetwork that produced them.
    """

    def __init__(self, base: nn.Linear, rank: int, scale: float = 1.0):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}")
        self.base = base
        self.rank = int(rank)
        self.scale = float(scale)
        self.lora_A = None
        self.lora_B = None

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def clear_lora(self):
        self.lora_A = None
        self.lora_B = None

    def set_lora(self, A: torch.Tensor, B: torch.Tensor):
        if A.shape[-2:] != (self.rank, self.in_features):
            raise ValueError(
                f"Expected A shape (..., {self.rank}, {self.in_features}), got {tuple(A.shape)}"
            )
        if B.shape[-2:] != (self.out_features, self.rank):
            raise ValueError(
                f"Expected B shape (..., {self.out_features}, {self.rank}), got {tuple(B.shape)}"
            )
        self.lora_A = A
        self.lora_B = B

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        if self.lora_A is None or self.lora_B is None:
            return y

        A = self.lora_A.to(device=x.device, dtype=x.dtype)
        B = self.lora_B.to(device=x.device, dtype=x.dtype)

        if A.dim() == 2:
            delta = F.linear(F.linear(x, A), B)
        elif A.dim() == 3:
            if x.shape[0] != A.shape[0] or A.shape[0] != B.shape[0]:
                raise ValueError(
                    f"Batched LoRA expects batch {x.shape[0]}, got A={tuple(A.shape)}, B={tuple(B.shape)}"
                )
            hidden = torch.einsum("b...i,bri->b...r", x, A)
            delta = torch.einsum("b...r,bor->b...o", hidden, B)
        else:
            raise ValueError(f"Unsupported LoRA tensor rank: {A.dim()}")

        return y + self.scale * delta


class HyperLayerNorm(nn.Module):
    """LayerNorm with an externally supplied, per-sample affine correction."""

    def __init__(self, base: nn.LayerNorm):
        super().__init__()
        if len(base.normalized_shape) != 1:
            raise ValueError("HyperLayerNorm currently requires a one-dimensional normalized shape")
        self.base = base
        self.affine_delta_weight = None
        self.affine_delta_bias = None

    @property
    def features(self) -> int:
        return int(self.base.normalized_shape[0])

    def clear_affine(self):
        self.affine_delta_weight = None
        self.affine_delta_bias = None

    def set_affine(self, delta_weight: torch.Tensor, delta_bias: torch.Tensor):
        expected = self.features
        if delta_weight.shape[-1] != expected or delta_bias.shape[-1] != expected:
            raise ValueError(
                f"Expected affine tensors ending in {expected}, got "
                f"{tuple(delta_weight.shape)} and {tuple(delta_bias.shape)}"
            )
        if delta_weight.shape != delta_bias.shape:
            raise ValueError("LayerNorm affine weight and bias updates must have matching shapes")
        self.affine_delta_weight = delta_weight
        self.affine_delta_bias = delta_bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.affine_delta_weight is None or self.affine_delta_bias is None:
            return self.base(x)

        delta_weight = self.affine_delta_weight.to(device=x.device, dtype=x.dtype)
        delta_bias = self.affine_delta_bias.to(device=x.device, dtype=x.dtype)
        if delta_weight.dim() == 1:
            weight = self.base.weight.to(dtype=x.dtype) + delta_weight
            bias = self.base.bias.to(dtype=x.dtype) + delta_bias
            return F.layer_norm(x, self.base.normalized_shape, weight, bias, self.base.eps)
        if delta_weight.dim() != 2 or x.shape[0] != delta_weight.shape[0]:
            raise ValueError(
                f"Batched LayerNorm affine expects batch {x.shape[0]}, got {tuple(delta_weight.shape)}"
            )

        normalized = F.layer_norm(x, self.base.normalized_shape, None, None, self.base.eps)
        view_shape = (x.shape[0],) + (1,) * (x.dim() - 2) + (self.features,)
        weight = self.base.weight.to(dtype=x.dtype).view((1,) * (x.dim() - 1) + (self.features,))
        bias = self.base.bias.to(dtype=x.dtype).view((1,) * (x.dim() - 1) + (self.features,))
        return normalized * (weight + delta_weight.view(view_shape)) + bias + delta_bias.view(view_shape)


def get_parent_module(root: nn.Module, module_name: str) -> tuple[nn.Module, str]:
    parts = module_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def replace_linear_with_lora(
    root: nn.Module,
    module_name: str,
    rank: int,
    scale: float,
) -> HyperLoRALinear:
    parent, child_name = get_parent_module(root, module_name)
    child = getattr(parent, child_name)
    if isinstance(child, HyperLoRALinear):
        return child
    if not isinstance(child, nn.Linear):
        raise TypeError(f"Can only wrap nn.Linear modules, got {module_name}: {type(child)}")
    wrapped = HyperLoRALinear(child, rank=rank, scale=scale)
    setattr(parent, child_name, wrapped)
    return wrapped


def replace_layer_norm_with_hyper(root: nn.Module, module_name: str) -> HyperLayerNorm:
    parent, child_name = get_parent_module(root, module_name)
    child = getattr(parent, child_name)
    if isinstance(child, HyperLayerNorm):
        return child
    if not isinstance(child, nn.LayerNorm):
        raise TypeError(f"Can only wrap nn.LayerNorm modules, got {module_name}: {type(child)}")
    wrapped = HyperLayerNorm(child)
    setattr(parent, child_name, wrapped)
    return wrapped
