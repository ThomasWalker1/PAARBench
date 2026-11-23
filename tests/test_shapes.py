"""Correction shapes in models/shapes.py: exact identity at zero, and the right math."""

import pytest
import torch
from torch import nn

from models.shapes import (
    DoRASpec,
    HyperDoRALinear,
    HyperIA3Linear,
    IA3Spec,
    init_correction_tensors,
    install_predictor_shape_targets,
)
from models.vit import ViTPredictor


def _linear(i=12, o=9, bias=True, seed=0):
    torch.manual_seed(seed)
    return nn.Linear(i, o, bias=bias)


@pytest.mark.parametrize("bias", [True, False])
def test_dora_zero_is_bit_identical(bias):
    base = _linear(bias=bias)
    wrapped = HyperDoRALinear(base, rank=2)
    x = torch.randn(3, 5, 12)
    A = torch.randn(2, 12)
    wrapped.set_lora(A, torch.zeros(9, 2), torch.zeros(9))
    assert torch.equal(wrapped(x), base(x))


def test_dora_matches_dense_formula():
    base = _linear()
    wrapped = HyperDoRALinear(base, rank=2, scale=0.7)
    A, B, dm = torch.randn(2, 12), torch.randn(9, 2), torch.randn(9) * 0.1
    wrapped.set_lora(A, B, dm)
    x = torch.randn(4, 12)
    W = base.weight
    V = W + 0.7 * B @ A
    W_new = ((W.norm(dim=1) + dm) / V.norm(dim=1)).unsqueeze(1) * V
    expected = x @ W_new.t() + base.bias
    torch.testing.assert_close(wrapped(x), expected, rtol=1e-5, atol=1e-5)


def test_dora_batched_matches_unbatched():
    base = _linear()
    wrapped = HyperDoRALinear(base, rank=2)
    A, B, dm = torch.randn(3, 2, 12), torch.randn(3, 9, 2), torch.randn(3, 9)
    x = torch.randn(3, 5, 12)
    wrapped.set_lora(A, B, dm)
    batched = wrapped(x)
    for b in range(3):
        wrapped.set_lora(A[b], B[b], dm[b])
        torch.testing.assert_close(batched[b], wrapped(x[b]), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("side,offset,length", [("out", 3, 6), ("in", 0, 12)])
def test_ia3_zero_is_bit_identical_and_scales(side, offset, length):
    base = _linear()
    wrapped = HyperIA3Linear(base, side, offset, length)
    x = torch.randn(2, 5, 12)
    wrapped.set_lora(torch.zeros(length))
    assert torch.equal(wrapped(x), base(x))

    s = torch.randn(length)
    wrapped.set_lora(s)
    if side == "out":
        scale = torch.ones(9)
        scale[offset: offset + length] += s
        expected = base(x) * scale
    else:
        expected = base(x * (1 + s))
    torch.testing.assert_close(wrapped(x), expected, rtol=1e-5, atol=1e-5)

    wrapped.set_lora(s.expand(2, -1).clone())
    torch.testing.assert_close(wrapped(x), expected, rtol=1e-5, atol=1e-5)


def _predictor():
    torch.manual_seed(0)
    return ViTPredictor(
        num_patches=4, num_frames=2, dim=16, depth=2, heads=2, mlp_dim=32, dim_head=8,
    )


@pytest.mark.parametrize("shape", ["ia3", "dora"])
def test_installed_predictor_is_frozen_at_init(shape):
    predictor = _predictor()
    x = torch.randn(2, 8, 16)
    with torch.no_grad():
        reference = predictor(x)
    targets, norms = install_predictor_shape_targets(predictor, shape=shape, rank=2)
    assert len(norms) == 1
    generator = torch.Generator().manual_seed(0)
    for target in targets:
        tensors, _ = init_correction_tensors(target.spec, generator=generator)
        target.module.set_lora(*tensors)
    for norm in norms:
        norm.module.set_affine(torch.zeros(16), torch.zeros(16))
    with torch.no_grad():
        assert torch.equal(predictor(x), reference)


def test_ia3_sites_are_keys_values_and_mlp_hidden():
    predictor = _predictor()
    targets, _ = install_predictor_shape_targets(predictor, shape="ia3", rank=2)
    sites = {t.name: (t.spec.side, t.spec.offset, t.spec.length) for t in targets}
    assert sites == {
        "transformer.layers.1.0.to_qkv": ("out", 16, 32),
        "transformer.layers.1.1.net.4": ("in", 0, 32),
    }
    assert all(isinstance(t.spec, IA3Spec) for t in targets)


def test_dora_param_count():
    spec = DoRASpec(name="x", in_features=404, out_features=3072, rank=2)
    assert spec.num_params == 2 * 404 + 3072 * 2 + 3072
