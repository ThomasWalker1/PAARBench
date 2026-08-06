"""Tests for the benchmark-standard tunable hyperparameter protocol."""

from types import SimpleNamespace

import pytest

from paarbench import methods
from paarbench.tunable import (
    AxisMapsTo,
    StandardSelection,
    TunableAxis,
    TunableSpec,
    parse_tunable,
)


class _StubHarness:
    def __init__(self, scores):
        self.scores = scores
        self.seen = []
        self.columns_used = 0

    def run(self, **params):
        self.columns_used += 1
        self.seen.append(params)
        key = tuple(sorted(params.items()))
        return SimpleNamespace(success=self.scores.get(key, 0.0),
                               median_distance_delta=0.0,
                               catastrophe_rate=0.0,
                               compounding_slope=0.0)


def _score_map(mapping):
    return {tuple(sorted({k: v}.items())): score
            for (k, v), score in mapping.items()}


def test_parse_tunable_rejects_more_than_two_axes(tmp_path):
    method_dir = tmp_path / "m"
    method_dir.mkdir()
    manifest = method_dir / "method.yaml"
    manifest.write_text(
        "name: m\ndisplay_name: x\ndescription: x\nadapter: a:A\nsettings: [pushobj]\n"
        "tunable:\n  axes:\n    a:\n      initial: [1, 2, 3]\n    b:\n      initial: [1, 2, 3]\n"
        "    c:\n      initial: [1, 2, 3]\n"
    )
    with pytest.raises(methods.MethodError, match="at most 2"):
        methods.load("m", tmp_path)


def test_standard_selection_runs_initial_three_point_grid():
    spec = TunableSpec(axes=[TunableAxis("lr", [1e-4, 1e-3, 1e-2], scale="log")])
    harness = _StubHarness(_score_map({
        ("lr", 1e-4): 0.1,
        ("lr", 1e-3): 0.5,
        ("lr", 1e-2): 0.2,
    }))
    chosen = StandardSelection(spec, "pushobj").select(harness)
    assert chosen == {"lr": 1e-3}
    assert harness.columns_used == 3


def test_standard_selection_expands_when_best_is_on_boundary():
    spec = TunableSpec(
        axes=[TunableAxis("lr", [1e-4, 1e-3, 1e-2], scale="log")],
        max_expansions=1,
    )
    harness = _StubHarness(_score_map({
        ("lr", 1e-4): 0.1,
        ("lr", 1e-3): 0.2,
        ("lr", 1e-2): 0.5,
        ("lr", 1e-1): 1.0,
    }))
    chosen = StandardSelection(spec, "pushobj").select(harness)
    assert chosen == {"lr": 1e-1}
    assert harness.columns_used == 4


def test_standard_selection_maps_discrete_epoch_to_checkpoint_path():
    spec = TunableSpec(axes=[TunableAxis(
        "training_epoch",
        [1, 2, 3],
        scale="discrete",
        pool_by_setting={"pushobj": [1, 2, 3, 4, 5]},
        maps_to=AxisMapsTo(
            param="checkpoint_path",
            by_setting={
                "pushobj": "checkpoints/pushobj_adapters/hyper_r2_distill0/hyper_lora_epoch_{}.pth",
            },
        ),
    )])
    harness = _StubHarness({
        (("checkpoint_path",
          "checkpoints/pushobj_adapters/hyper_r2_distill0/hyper_lora_epoch_1.pth"),): 0.1,
        (("checkpoint_path",
          "checkpoints/pushobj_adapters/hyper_r2_distill0/hyper_lora_epoch_2.pth"),): 0.9,
        (("checkpoint_path",
          "checkpoints/pushobj_adapters/hyper_r2_distill0/hyper_lora_epoch_3.pth"),): 0.5,
    })
    chosen = StandardSelection(spec, "pushobj").select(harness)
    assert chosen["checkpoint_path"].endswith("hyper_lora_epoch_2.pth")


def test_shipped_tunable_methods_validate():
    for method in methods.discover():
        if method.tunable is None:
            continue
        assert methods.validate(method) == [], f"{method.name} tunable declaration invalid"


def test_shipped_methods_do_not_mix_tunable_and_custom_selection():
    for method in methods.discover():
        if method.tunable is not None:
            assert method.selection_ref is None, method.name
