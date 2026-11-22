"""Correction shapes other than LoRA for the predictor's linear layers.

A *shape* is the parameterization of a correction: which numbers there are, and how
they change a frozen linear layer.  ``models/lora.py`` has LoRA.  This module adds two
more, so a method can change the shape and keep everything else (hook sites, fitting
rule, hypernetwork, selection) the same:

- **(IA)^3** (Liu et al., 2022, arXiv:2205.05638): learned per-feature rescaling of the
  attention keys and values and of the MLP hidden activation.  It is multiplicative and
  it cannot add a direction the frozen layer does not already produce.
- **DoRA** (Liu et al., 2024, arXiv:2402.09353): the weight is split into a
  per-output-channel magnitude and a direction; LoRA changes the direction and a
  separate vector changes the magnitude.

Two conventions hold for every shape here, because the methods rely on them:

1. **All-zero tensors give exactly the frozen layer.**  (IA)^3 stores ``s`` and applies
   ``1 + s``; DoRA stores ``B`` and a magnitude *delta* ``dm`` on top of the frozen
   magnitude.  AdaJEPA v2's brake resets a correction by zeroing tensors, and the
   hypernetworks start their output layer near zero, so a shape that broke this would
   break both without an error.  The forward is written as ``base(x) + delta`` with a
   delta that is exactly zero at zero, so the result is bit-identical to the frozen
   layer, not only close to it.
2. **The same method names as ``HyperLoRALinear``** -- ``set_lora(*tensors)`` and
   ``clear_lora()`` -- and the same leading batch dimension convention (unbatched
   tensors are shared, a leading batch dimension is per sample).  Code that installs,
   applies, or temporarily removes a LoRA correction then works unchanged.

The LayerNorm affine delta that the LoRA methods also adapt is not a linear-layer
correction and is kept as it is; see ``install_predictor_shape_targets``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from .hyper_lora import (
    LoRATarget,
    NormTarget,
    install_predictor_hyper_targets,
    select_predictor_lora_module_names,
)
from .lora import get_parent_module, replace_layer_norm_with_hyper, AffineSpec

SHAPES = ("lora", "ia3", "dora")

_NORM_EPS = 1e-12


# -- specs ---------------------------------------------------------------------


@dataclass(frozen=True)
class IA3Spec:
    """A rescaling of ``length`` features, starting at ``offset``, on one side of a linear.

    ``side='out'`` scales output features (the keys and values of a fused QKV
    projection); ``side='in'`` scales input features (the MLP hidden activation, as
    the input of the second MLP linear).
    """

    name: str
    in_features: int
    out_features: int
    side: str
    offset: int
    length: int

    @property
    def tensor_shapes(self) -> tuple[tuple[int, ...], ...]:
        return ((self.length,),)

    @property
    def num_params(self) -> int:
        return self.length


@dataclass(frozen=True)
class DoRASpec:
    """LoRA factors for the direction plus a per-output-channel magnitude delta."""

    name: str
    in_features: int
    out_features: int
    rank: int

    @property
    def tensor_shapes(self) -> tuple[tuple[int, ...], ...]:
        return (
            (self.rank, self.in_features),
            (self.out_features, self.rank),
            (self.out_features,),
        )

    @property
    def num_params(self) -> int:
        return self.rank * self.in_features + self.out_features * self.rank + self.out_features


# -- modules -------------------------------------------------------------------


def _feature_view(vector: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Broadcast a (F,) or per-sample (B, F) vector against x of shape (B, ..., F)."""
    if vector.dim() == 1:
        return vector
    if vector.dim() != 2 or vector.shape[0] != x.shape[0]:
        raise ValueError(
            f"Batched correction expects batch {x.shape[0]}, got {tuple(vector.shape)}"
        )
    return vector.view((x.shape[0],) + (1,) * (x.dim() - 2) + (vector.shape[-1],))


class HyperIA3Linear(nn.Module):
    """Linear layer with an externally supplied (IA)^3 rescaling ``1 + s``."""

    def __init__(self, base: nn.Linear, side: str, offset: int, length: int):
        super().__init__()
        if side not in ("in", "out"):
            raise ValueError(f"(IA)^3 side must be 'in' or 'out', got {side!r}")
        width = base.in_features if side == "in" else base.out_features
        if offset < 0 or length <= 0 or offset + length > width:
            raise ValueError(
                f"(IA)^3 slice [{offset}, {offset + length}) does not fit {side} width {width}"
            )
        self.base = base
        self.side = side
        self.offset = int(offset)
        self.length = int(length)
        self.ia3_s = None

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def clear_lora(self):
        self.ia3_s = None

    def set_lora(self, s: torch.Tensor):
        if s.shape[-1] != self.length or s.dim() not in (1, 2):
            raise ValueError(f"Expected (IA)^3 tensor (..., {self.length}), got {tuple(s.shape)}")
        self.ia3_s = s

    def correction_tensors(self):
        return (self.ia3_s,)

    def _full_scale(self, s: torch.Tensor, width: int) -> torch.Tensor:
        """``s`` padded with zeros to the full feature width (zero = unscaled)."""
        if self.offset == 0 and self.length == width:
            return s
        pad = (self.offset, width - self.offset - self.length)
        return F.pad(s, pad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.ia3_s is None:
            return self.base(x)
        s = self.ia3_s.to(device=x.device, dtype=x.dtype)
        if self.side == "in":
            s_full = _feature_view(self._full_scale(s, self.in_features), x)
            # base(x * (1 + s)) == base(x) + W (x * s): the delta form keeps s = 0 exact.
            return self.base(x) + F.linear(x * s_full, self.base.weight)
        y = self.base(x)
        s_full = _feature_view(self._full_scale(s, self.out_features), y)
        return y + y * s_full


class HyperDoRALinear(nn.Module):
    """Linear layer with an externally supplied DoRA correction ``(A, B, dm)``.

    ``W' = (m0 + dm) * (W + scale*B A) / ||W + scale*B A||``, with norms taken per output
    channel and ``m0 = ||W||``.  The row norms are computed in closed form from ``A``,
    ``B`` and ``A W^T`` so a per-sample correction never materializes a weight matrix
    per sample.  The norm is *not* detached (the DoRA paper's memory shortcut); the
    layers here are small.
    """

    def __init__(self, base: nn.Linear, rank: int, scale: float = 1.0):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"DoRA rank must be positive, got {rank}")
        self.base = base
        self.rank = int(rank)
        self.scale = float(scale)
        self.lora_A = None
        self.lora_B = None
        self.dora_dm = None

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def clear_lora(self):
        self.lora_A = None
        self.lora_B = None
        self.dora_dm = None

    def set_lora(self, A: torch.Tensor, B: torch.Tensor, dm: torch.Tensor):
        if A.shape[-2:] != (self.rank, self.in_features):
            raise ValueError(f"Expected A (..., {self.rank}, {self.in_features}), got {tuple(A.shape)}")
        if B.shape[-2:] != (self.out_features, self.rank):
            raise ValueError(f"Expected B (..., {self.out_features}, {self.rank}), got {tuple(B.shape)}")
        if dm.shape[-1] != self.out_features or dm.dim() != A.dim() - 1:
            raise ValueError(f"Expected dm (..., {self.out_features}) matching A, got {tuple(dm.shape)}")
        self.lora_A = A
        self.lora_B = B
        self.dora_dm = dm

    def correction_tensors(self):
        return (self.lora_A, self.lora_B, self.dora_dm)

    def _factor(self, A, B, dm):
        """Per-output-channel ``(m0 + dm) / ||W + scale*B A||`` in float32."""
        W = self.base.weight.float()
        A, B, dm = A.float(), B.float(), dm.float()
        w_sq = (W * W).sum(dim=1)
        m0 = w_sq.sqrt()
        s = self.scale
        if A.dim() == 2:
            AW = A @ W.t()                                   # (r, out)
            cross = (B * AW.t()).sum(dim=1)                  # (out,)
            quad = ((B @ (A @ A.t())) * B).sum(dim=1)        # (out,)
        else:
            AW = torch.einsum("bri,oi->bor", A, W)           # (b, out, r)
            cross = (B * AW).sum(dim=-1)                     # (b, out)
            AAt = torch.einsum("bri,bsi->brs", A, A)         # (b, r, r)
            quad = (torch.einsum("bor,brs->bos", B, AAt) * B).sum(dim=-1)
        norm_sq = w_sq + 2.0 * s * cross + (s * s) * quad
        norm = norm_sq.clamp_min(_NORM_EPS).sqrt()
        return (m0 + dm) / norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        if self.lora_A is None:
            return y
        A, B, dm = self.lora_A, self.lora_B, self.dora_dm
        factor = self._factor(A, B, dm).to(dtype=x.dtype)
        A = A.to(device=x.device, dtype=x.dtype)
        B = B.to(device=x.device, dtype=x.dtype)
        if A.dim() == 2:
            lora = F.linear(F.linear(x, A), B)
        else:
            if x.shape[0] != A.shape[0]:
                raise ValueError(f"Batched DoRA expects batch {x.shape[0]}, got {A.shape[0]}")
            lora = torch.einsum("b...r,bor->b...o", torch.einsum("b...i,bri->b...r", x, A), B)
        factor = _feature_view(factor, x)
        # W' x = factor * (W x + scale * B A x).  Written as base(x) plus a delta that is
        # exactly zero when B = 0 and dm = 0 (then factor == m0 / m0 == 1 bit for bit).
        return y + (factor - 1.0) * F.linear(x, self.base.weight) + factor * (self.scale * lora)


# -- installation --------------------------------------------------------------


def _replace(root: nn.Module, name: str, build) -> nn.Module:
    parent, child_name = get_parent_module(root, name)
    child = getattr(parent, child_name)
    if not isinstance(child, nn.Linear):
        raise TypeError(f"Can only wrap nn.Linear modules, got {name}: {type(child)}")
    wrapped = build(child)
    setattr(parent, child_name, wrapped)
    return wrapped


def _ia3_site(name: str, linear: nn.Linear):
    """Where (IA)^3 scales a predictor linear, or None if it leaves it alone.

    Original (IA)^3 placement: keys, values, and the MLP hidden activation.  The fused
    ``to_qkv`` is chunked as (q, k, v) by ``models/vit.py``, so keys and values are its
    last two thirds; the hidden activation is the input of the second MLP linear.
    """
    if name.endswith(".to_qkv"):
        if linear.out_features % 3:
            raise ValueError(f"{name}: fused QKV width {linear.out_features} is not divisible by 3")
        inner = linear.out_features // 3
        return "out", inner, 2 * inner
    if name.endswith(".1.net.4"):
        return "in", 0, linear.in_features
    return None


def install_predictor_shape_targets(
    predictor: nn.Module,
    shape: str = "lora",
    scope: str = "predlast_all",
    rank: int = 2,
    scale: float = 1.0,
) -> tuple[list[LoRATarget], list[NormTarget]]:
    """Install a correction of the given shape on the predictor.

    ``shape='lora'`` defers to ``install_predictor_hyper_targets`` unchanged.  The other
    shapes use the same scope and keep the same final-LayerNorm affine target, so the
    only difference between rows is the parameterization of the linear layers.
    """
    if shape == "lora":
        return install_predictor_hyper_targets(predictor, scope=scope, rank=rank, scale=scale)
    if shape not in SHAPES:
        raise ValueError(f"Unknown correction shape {shape!r}; expected one of {SHAPES}")

    targets = []
    for name in select_predictor_lora_module_names(predictor, scope):
        linear = predictor.get_submodule(name)
        if shape == "ia3":
            site = _ia3_site(name, linear)
            if site is None:
                continue
            side, offset, length = site
            module = _replace(
                predictor, name, lambda base: HyperIA3Linear(base, side, offset, length)
            )
            spec = IA3Spec(
                name=name, in_features=linear.in_features, out_features=linear.out_features,
                side=side, offset=offset, length=length,
            )
        else:
            module = _replace(predictor, name, lambda base: HyperDoRALinear(base, rank, scale))
            spec = DoRASpec(
                name=name, in_features=linear.in_features, out_features=linear.out_features,
                rank=int(rank),
            )
        targets.append(LoRATarget(name=name, module=module, spec=spec))
    if not targets:
        raise ValueError(f"No {shape} targets for scope {scope!r}")

    norm_targets = []
    if scope.startswith(("predlast", "predall", "predfirstlast")):
        norm_name = "transformer.norm"
        norm = replace_layer_norm_with_hyper(predictor, norm_name)
        norm_targets.append(
            NormTarget(name=norm_name, module=norm, spec=AffineSpec(name=norm_name, features=norm.features))
        )
    return targets, norm_targets


def init_correction_tensors(spec, generator: torch.Generator | None = None):
    """Initial online-fitted tensors for one target: random where the shape needs it,
    zero everywhere that controls the size of the correction.

    Returns ``(tensors, zeroable)``: zeroing every tensor in ``zeroable`` returns the
    layer to exactly frozen whatever the others hold.
    """
    if isinstance(spec, IA3Spec):
        s = torch.zeros(spec.length)
        return [s], [0]
    if isinstance(spec, DoRASpec):
        A = torch.empty(spec.rank, spec.in_features)
        nn.init.normal_(A, mean=0.0, std=1.0 / max(spec.in_features, 1) ** 0.5, generator=generator)
        B = torch.zeros(spec.out_features, spec.rank)
        dm = torch.zeros(spec.out_features)
        return [A, B, dm], [1, 2]
    raise TypeError(f"init_correction_tensors does not handle {type(spec).__name__}")
