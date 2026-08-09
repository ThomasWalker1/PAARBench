from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Union

import torch
from torch import nn

from .lora import (
    AffineSpec,
    HyperLayerNorm,
    HyperLoRALinear,
    LoRASpec,
    replace_layer_norm_with_hyper,
    replace_linear_with_lora,
)

log = logging.getLogger(__name__)


@dataclass
class LoRATarget:
    name: str
    module: HyperLoRALinear
    spec: LoRASpec


@dataclass
class NormTarget:
    name: str
    module: HyperLayerNorm
    spec: AffineSpec


def _layer_indices(depth: int, scope: str) -> set[int]:
    if scope.startswith("predfirst"):
        return {0}
    if scope.startswith("predlast"):
        return {max(depth - 1, 0)}
    if scope.startswith("predall"):
        return set(range(depth))
    raise ValueError(f"Unknown LoRA target scope '{scope}'")


def _target_kinds(scope: str) -> set[str]:
    if scope.endswith("_attn"):
        return {"attn"}
    if scope.endswith("_mlp"):
        return {"mlp"}
    if scope.endswith("_all"):
        return {"attn", "mlp"}
    if scope in {"predfirst", "predlast", "predall"}:
        return {"attn", "mlp"}
    raise ValueError(f"Unknown LoRA target scope '{scope}'")


def select_predictor_lora_module_names(predictor: nn.Module, scope: str) -> list[str]:
    layers = getattr(getattr(predictor, "transformer", None), "layers", None)
    if layers is None:
        raise ValueError("HyperLoRA LoRA selection currently expects predictor.transformer.layers")

    target_layers = _layer_indices(len(layers), scope)
    kinds = _target_kinds(scope)
    names = []
    for idx in sorted(target_layers):
        if "attn" in kinds:
            names.extend(
                [
                    f"transformer.layers.{idx}.0.to_qkv",
                    f"transformer.layers.{idx}.0.to_out.0",
                ]
            )
        if "mlp" in kinds:
            names.extend(
                [
                    f"transformer.layers.{idx}.1.net.1",
                    f"transformer.layers.{idx}.1.net.4",
                ]
            )
    return names


def full_rank_for_scope(predictor: nn.Module, scope: str) -> int:
    """Smallest rank at which every targeted matrix's correction is unconstrained.

    ``B A`` with ``rank = min(d_in, d_out)`` spans all of ``R^{d_out x d_in}``, so
    this is the rank at which the generated correction stops being a low-rank
    restriction and becomes an arbitrary dense weight delta.
    """
    ranks = []
    for name in select_predictor_lora_module_names(predictor, scope):
        module = predictor.get_submodule(name)
        ranks.append(min(int(module.in_features), int(module.out_features)))
    if not ranks:
        raise ValueError(f"No LoRA target modules for scope '{scope}'")
    return max(ranks)


def install_predictor_lora(
    predictor: nn.Module,
    scope: str = "predlast_all",
    rank: int = 4,
    scale: float = 1.0,
) -> list[LoRATarget]:
    targets = []
    for name in select_predictor_lora_module_names(predictor, scope):
        module = replace_linear_with_lora(predictor, name, rank=rank, scale=scale)
        spec = LoRASpec(
            name=name,
            in_features=module.in_features,
            out_features=module.out_features,
            rank=module.rank,
        )
        targets.append(LoRATarget(name=name, module=module, spec=spec))
    log.info(
        "Installed HyperLoRA LoRA scope=%s rank=%s on %s predictor modules",
        scope,
        rank,
        len(targets),
    )
    return targets


def install_predictor_hyper_targets(
    predictor: nn.Module,
    scope: str = "predlast_all",
    rank: int = 4,
    scale: float = 1.0,
) -> tuple[list[LoRATarget], list[NormTarget]]:
    """Install LoRA targets and the final predictor LayerNorm when applicable."""
    lora_targets = install_predictor_lora(predictor, scope=scope, rank=rank, scale=scale)
    norm_targets = []
    if scope.startswith(("predlast", "predall", "predfirstlast")):
        name = "transformer.norm"
        module = replace_layer_norm_with_hyper(predictor, name)
        norm_targets.append(
            NormTarget(name=name, module=module, spec=AffineSpec(name=name, features=module.features))
        )
    log.info("Installed %s adaptive predictor LayerNorm modules", len(norm_targets))
    return lora_targets, norm_targets


class _PackedCorrectionMixin:
    """Shared packing conventions for generators that emit one flat vector.

    Both generator families produce the same layout -- for every LoRA spec an
    ``A`` block followed by a ``B`` block, and for every affine spec a weight
    block followed by a bias block -- so only the map from context to that
    vector differs between them.
    """

    def _init_packing(
        self,
        context_dim: int,
        specs: list[Union[LoRASpec, AffineSpec]],
        freeze_lora_A: bool,
        fixed_a_scale: float,
    ):
        self.context_dim = int(context_dim)
        self.specs = list(specs)
        self.freeze_lora_A = bool(freeze_lora_A)
        self.fixed_a_scale = float(fixed_a_scale)
        self._fixed_a_buffers = {}
        self.total_params = sum(self._emitted_params(spec) for spec in self.specs)
        for idx, spec in enumerate(self.specs):
            if isinstance(spec, LoRASpec) and self.freeze_lora_A:
                buffer_name = f"fixed_lora_A_{idx}"
                init = torch.randn(spec.rank, spec.in_features)
                init = init * (self.fixed_a_scale / max(spec.in_features, 1) ** 0.5)
                self.register_buffer(buffer_name, init)
                self._fixed_a_buffers[spec.name] = buffer_name

    def _emitted_params(self, spec: Union[LoRASpec, AffineSpec]) -> int:
        if isinstance(spec, AffineSpec):
            return spec.num_params
        if self.freeze_lora_A:
            return spec.out_features * spec.rank
        return spec.num_params

    def _unpack(self, packed: torch.Tensor) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        if packed.shape[-1] != self.total_params:
            raise RuntimeError(
                f"HyperLoRA output has {packed.shape[-1]} dims, expected {self.total_params}"
            )
        offset = 0
        result = {}
        batch = packed.shape[0]
        for spec in self.specs:
            if isinstance(spec, AffineSpec):
                weight = packed[:, offset : offset + spec.features]
                offset += spec.features
                bias = packed[:, offset : offset + spec.features]
                offset += spec.features
                result[spec.name] = (weight, bias)
                continue
            a_num = 0 if self.freeze_lora_A else spec.rank * spec.in_features
            b_num = spec.out_features * spec.rank
            if self.freeze_lora_A:
                A = getattr(self, self._fixed_a_buffers[spec.name]).unsqueeze(0).expand(
                    batch, -1, -1
                )
            else:
                A = packed[:, offset : offset + a_num].reshape(
                    batch, spec.rank, spec.in_features
                )
                offset += a_num
            B = packed[:, offset : offset + b_num].reshape(
                batch, spec.out_features, spec.rank
            )
            offset += b_num
            result[spec.name] = (A, B)
        return result


class HyperLoRAGenerator(_PackedCorrectionMixin, nn.Module):
    """Map context embeddings to packed per-sample LoRA tensors."""

    def __init__(
        self,
        context_dim: int,
        specs: list[Union[LoRASpec, AffineSpec]],
        hidden_dim: int = 512,
        depth: int = 2,
        init_scale: float = 0.0,
        freeze_lora_A: bool = False,
        fixed_a_scale: float = 1.0,
    ):
        super().__init__()
        if depth < 1:
            raise ValueError("HyperLoRAGenerator depth must be >= 1")
        self._init_packing(context_dim, specs, freeze_lora_A, fixed_a_scale)
        layers = []
        in_dim = self.context_dim
        for _ in range(depth - 1):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.GELU()])
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, self.total_params))
        self.net = nn.Sequential(*layers)
        self.reset_output(init_scale)

    def reset_output(self, init_scale: float):
        last = self.net[-1]
        if not isinstance(last, nn.Linear):
            return
        nn.init.normal_(last.weight, mean=0.0, std=float(init_scale))
        nn.init.zeros_(last.bias)

    def forward(self, context: torch.Tensor) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        return self._unpack(self.net(context))


class ChunkedHyperGenerator(_PackedCorrectionMixin, nn.Module):
    """Emit the packed correction in fixed-size chunks from a shared head.

    ``HyperLoRAGenerator``'s output layer is ``hidden_dim x total_params``, which
    is 10.5 M parameters for the rank-2 correction but 2.0 B once the emitted
    correction is full rank. Following the chunked hypernetwork of von Oswald et
    al., a single small head is instead queried once per chunk with a learned
    chunk embedding, so generator size grows with the chunk count only through
    that embedding table. The head is deliberately two layers deep: a single
    linear layer on ``[context; chunk]`` is additively separable, which would
    make every chunk the same function of the context up to a fixed offset.
    """

    def __init__(
        self,
        context_dim: int,
        specs: list[Union[LoRASpec, AffineSpec]],
        hidden_dim: int = 512,
        depth: int = 2,
        init_scale: float = 0.0,
        chunk_size: int = 512,
        chunk_embed_dim: int = 64,
        head_hidden_dim: int = 128,
        freeze_lora_A: bool = False,
        fixed_a_scale: float = 1.0,
    ):
        super().__init__()
        if depth < 1:
            raise ValueError("ChunkedHyperGenerator depth must be >= 1")
        if chunk_size < 1 or chunk_embed_dim < 1 or head_hidden_dim < 1:
            raise ValueError("ChunkedHyperGenerator chunk/head dimensions must be positive")
        self._init_packing(context_dim, specs, freeze_lora_A, fixed_a_scale)
        self.chunk_size = int(chunk_size)
        self.num_chunks = -(-self.total_params // self.chunk_size)  # ceil
        trunk = []
        in_dim = self.context_dim
        for _ in range(depth - 1):
            trunk.extend([nn.Linear(in_dim, hidden_dim), nn.GELU()])
            in_dim = hidden_dim
        trunk.append(nn.Linear(in_dim, hidden_dim))
        self.trunk = nn.Sequential(*trunk)
        self.chunk_embedding = nn.Parameter(torch.empty(self.num_chunks, int(chunk_embed_dim)))
        nn.init.normal_(self.chunk_embedding, std=0.02)
        # The first head layer is applied to the context branch and the chunk
        # branch separately and summed, which is exactly a linear layer on the
        # concatenation but never materializes (batch, chunks, hidden+embed).
        self.head_context = nn.Linear(hidden_dim, int(head_hidden_dim))
        self.head_chunk = nn.Linear(int(chunk_embed_dim), int(head_hidden_dim), bias=False)
        self.head_out = nn.Linear(int(head_hidden_dim), self.chunk_size)
        self.reset_output(init_scale)

    def reset_output(self, init_scale: float):
        nn.init.normal_(self.head_out.weight, mean=0.0, std=float(init_scale))
        nn.init.zeros_(self.head_out.bias)

    def forward(self, context: torch.Tensor) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        z = self.trunk(context)
        hidden = self.head_context(z).unsqueeze(1) + self.head_chunk(self.chunk_embedding).unsqueeze(0)
        chunks = self.head_out(torch.nn.functional.gelu(hidden))
        packed = chunks.reshape(context.shape[0], -1)[:, : self.total_params]
        return self._unpack(packed)


def build_generator(
    kind: str,
    context_dim: int,
    specs: list[Union[LoRASpec, AffineSpec]],
    hidden_dim: int = 512,
    depth: int = 2,
    init_scale: float = 0.0,
    freeze_lora_A: bool = False,
    fixed_a_scale: float = 1.0,
    chunk_size: int = 512,
    chunk_embed_dim: int = 64,
    head_hidden_dim: int = 128,
):
    """Build the generator named by ``kind`` ('mlp' or 'chunked')."""
    if kind == "mlp":
        return HyperLoRAGenerator(
            context_dim=context_dim,
            specs=specs,
            hidden_dim=hidden_dim,
            depth=depth,
            init_scale=init_scale,
            freeze_lora_A=freeze_lora_A,
            fixed_a_scale=fixed_a_scale,
        )
    if kind == "chunked":
        return ChunkedHyperGenerator(
            context_dim=context_dim,
            specs=specs,
            hidden_dim=hidden_dim,
            depth=depth,
            init_scale=init_scale,
            chunk_size=chunk_size,
            chunk_embed_dim=chunk_embed_dim,
            head_hidden_dim=head_hidden_dim,
            freeze_lora_A=freeze_lora_A,
            fixed_a_scale=fixed_a_scale,
        )
    raise ValueError(f"Unknown hypernetwork generator kind '{kind}'")


class HyperAdapterGate(nn.Module):
    """Learn a conservative per-context scale for generated corrections."""

    def __init__(self, context_dim: int, hidden_dim: int = 64, init_bias: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(context_dim),
            nn.Linear(context_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, float(init_bias))

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(context))


def apply_lora_tensors(
    targets: list[LoRATarget],
    tensors: dict[str, tuple[torch.Tensor, torch.Tensor]],
):
    for target in targets:
        A, B = tensors[target.name]
        target.module.set_lora(A, B)


def apply_hyper_tensors(
    lora_targets: list[LoRATarget],
    norm_targets: list[NormTarget],
    tensors: dict[str, tuple[torch.Tensor, torch.Tensor]],
    gate: torch.Tensor | None = None,
):
    if gate is None:
        apply_lora_tensors(lora_targets, tensors)
    else:
        if gate.ndim != 2 or gate.shape[-1] != 1:
            raise ValueError(f"Expected gate shape (B, 1), got {tuple(gate.shape)}")
        gated_lora = {}
        for target in lora_targets:
            A, B = tensors[target.name]
            if A.shape[0] != gate.shape[0]:
                raise ValueError("Gate and generated LoRA batch sizes must match")
            gated_lora[target.name] = (A, B * gate.view(gate.shape[0], 1, 1))
        apply_lora_tensors(lora_targets, gated_lora)
    for target in norm_targets:
        delta_weight, delta_bias = tensors[target.name]
        if gate is not None:
            if delta_weight.shape[0] != gate.shape[0]:
                raise ValueError("Gate and generated LayerNorm batch sizes must match")
            delta_weight = delta_weight * gate
            delta_bias = delta_bias * gate
        target.module.set_affine(delta_weight, delta_bias)


def snapshot_base_weights(
    lora_targets: list[LoRATarget],
    norm_targets: list[NormTarget],
) -> dict:
    """Clone the pristine base weights of every adapted module.

    Folding mutates those weights, and the correction is recomputed from scratch at
    every replan, so the pristine values must be restored before each fold or the
    corrections would accumulate across replans -- exactly the cross-replan parameter
    drift that HyperLoRA is supposed to avoid.
    """
    snapshot = {}
    for target in lora_targets:
        snapshot[("lora", target.name)] = target.module.base.weight.detach().clone()
    for target in norm_targets:
        snapshot[("norm", target.name)] = (
            target.module.base.weight.detach().clone(),
            target.module.base.bias.detach().clone(),
        )
    return snapshot


@torch.no_grad()
def restore_base_weights(
    lora_targets: list[LoRATarget],
    norm_targets: list[NormTarget],
    snapshot: dict,
):
    for target in lora_targets:
        target.module.base.weight.copy_(snapshot[("lora", target.name)])
    for target in norm_targets:
        weight, bias = snapshot[("norm", target.name)]
        target.module.base.weight.copy_(weight)
        target.module.base.bias.copy_(bias)


@torch.no_grad()
def fold_hyper_tensors(
    lora_targets: list[LoRATarget],
    norm_targets: list[NormTarget],
    tensors: dict[str, tuple[torch.Tensor, torch.Tensor]],
    gate: torch.Tensor | None = None,
):
    """Fold the emitted correction into the base weights instead of wrapping the forward.

    ``apply_hyper_tensors`` routes every planner rollout through the LoRA path
    (``y = Wx + scale * B(Ax)``), which costs two extra matmuls per adapted linear per
    call.  With a 100-step GD-MPC inner loop that overhead dominates the adaptation
    saving (+0.80 s/replan, making HyperLoRA *slower* end to end than the
    online baseline it beats on the adaptation component).

    Since ``y = Wx + scale * x(BA)^T = x(W + scale*BA)^T``, the correction can be folded
    into ``W`` once per replan, after which the planner runs at frozen-model speed.

    Only valid for a single episode: one weight matrix cannot carry per-episode
    corrections, so this raises on a batched context.  That is the deployment setting
    (one robot, one episode) and the setting the latency table measures.
    """
    for target in lora_targets:
        A, B = tensors[target.name]
        if gate is not None:
            B = B * gate.view(gate.shape[0], 1, 1)
        if A.dim() == 3:
            if A.shape[0] != 1 or B.shape[0] != 1:
                raise ValueError(
                    "fold_weights requires a single episode (batch 1); got batch "
                    f"{A.shape[0]}. Per-episode corrections cannot be folded into one "
                    "weight matrix -- run the batched evaluator without fold_weights."
                )
            A, B = A[0], B[0]
        weight = target.module.base.weight
        weight.add_((target.module.scale * (B @ A)).to(dtype=weight.dtype, device=weight.device))
        target.module.clear_lora()
    for target in norm_targets:
        delta_weight, delta_bias = tensors[target.name]
        if gate is not None:
            delta_weight = delta_weight * gate
            delta_bias = delta_bias * gate
        if delta_weight.dim() == 2:
            if delta_weight.shape[0] != 1:
                raise ValueError(
                    "fold_weights requires a single episode (batch 1); got batch "
                    f"{delta_weight.shape[0]}."
                )
            delta_weight, delta_bias = delta_weight[0], delta_bias[0]
        base = target.module.base
        base.weight.add_(delta_weight.to(dtype=base.weight.dtype, device=base.weight.device))
        base.bias.add_(delta_bias.to(dtype=base.bias.dtype, device=base.bias.device))
        target.module.clear_affine()


def lora_parameter_budget(
    specs: list[Union[LoRASpec, AffineSpec]],
    freeze_lora_A: bool = False,
) -> int:
    """Number of adaptive coefficients emitted for one context sample."""
    total = 0
    for spec in specs:
        if isinstance(spec, AffineSpec) or not freeze_lora_A:
            total += spec.num_params
        else:
            total += spec.out_features * spec.rank
    return total


def module_parameter_count(module: nn.Module) -> int:
    """Count stored module parameters, excluding per-sample generated tensors."""
    return sum(parameter.numel() for parameter in module.parameters())


def clear_lora(targets: list[LoRATarget]):
    for target in targets:
        target.module.clear_lora()


def clear_hyper_tensors(lora_targets: list[LoRATarget], norm_targets: list[NormTarget]):
    clear_lora(lora_targets)
    for target in norm_targets:
        target.module.clear_affine()
