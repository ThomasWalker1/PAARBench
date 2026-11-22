# AdaJEPA v2 · DoRA — v2 fitting rule, magnitude + direction

[`methods/adajepa_v2`](../adajepa_v2/) with one change: the correction is DoRA instead of
a plain rank-2 LoRA. The fitting rule, optimizer, clipping, brake, buffer and tuned axes
are v2's, at v2's selection cost. Read the v2 README for the fitting rule.

| correction | sites on the predictor's last block | parameters |
|---|---|---|
| v2 (LoRA r=2) | `to_qkv`, `to_out.0`, `net.1`, `net.4` + final LayerNorm delta | 20,424 |
| **this row (DoRA r=2)** | same four linears: rank-2 direction `B·A` + per-output magnitude delta `dm` (5,928) + final LayerNorm delta | **26,352** |

Each adapted linear computes `W' = (m0 + dm) · (W + B·A) / ‖W + B·A‖` (row norms), with
`m0 = ‖W‖` frozen. The forward uses the closed-form row norm, so no dense `W'` is built.
`B = 0, dm = 0` gives a bit-identical frozen layer, so the first replan of every episode
plans with the frozen model. Implementation: `models/shapes.py` (`HyperDoRALinear`,
`install_predictor_shape_targets`).

## Results

All seven settings, held-out test cohorts (seeds 100/200/300). Paired metrics are against
the batched Frozen reference on the same episodes.

| setting | success | v2 LoRA | frozen | median dist Δ | catastrophe | compounding | regret | adapt s/replan |
|---|---|---|---|---|---|---|---|---|
| `pushobj` | **0.707** | 0.697 | 0.500 | +8 [+2, +20] | 21.0% vs 14.4% | +0.36 [-0.13, +1.47] | 61% / +279 | 0.461 |
| `pushobj_shift` | **0.413** | 0.409 | 0.300 | +1 [-8, +12] | **12.3%** vs 15.9% | +0.19 [-0.19, +0.64] | 52% / +178 | 0.379 |
| `pusht` | 0.471 | 0.478 | 0.349 | +52 [+21, +72] | 25.6% vs 22.2% | +3.25 [+2.17, +4.54] | 62% / +419 | 0.421 |
| `maze_medium` | 0.780 | 0.847 | 0.773 | +0 [+0, +0] | 5.0% vs 0.0% | -0.00 [-0.02, +0.01] | 75% / +1 | 0.642 |
| `maze_medium_low_density` | 0.727 | 0.773 | 0.747 | -1 [-2, +0] | 8.3% vs 6.2% | +0.02 [-0.04, +0.07] | **33%** / +3 | 0.609 |
| `maze_medium_high_damping` | 0.693 | 0.740 | 0.567 | +0 [-0, +0] | 2.8% vs 0.0% | +0.00 [-0.01, +0.01] | 58% / +2 | 0.621 |
| `maze_diverse` | 0.613 | 0.727 | 0.593 | -1 [-2, +0] | 10.5% vs 7.7% | -0.06 [-0.09, -0.02] | 34% / +3 | 0.595 |

Pooled over the 2,100 held-out episodes: **0.592** success (v2 LoRA 0.610, frozen 0.473).

- Highest success of any row on `pushobj` (0.707) and `pushobj_shift` (0.413); both are
  within one SE of v2.
- Within one SE of v2 on `pusht` (0.471 vs 0.478).
- Lower catastrophe than v2 on `pushobj_shift` (12.3% vs 15.9%).
- About 2× v2's adaptation cost per replan (the row-norm term in every forward).

## Selection

Same two axes and grid as v2 (`pred_lr` 5e-4, 2e-3, 1e-2, log expansion; `horizon` 1, 3,
5, pool 1, 2, 3, 5, 8). 12 columns on `pushobj` and `maze_medium`, 14 on `pusht`.

| setting | froze | best grid cells (selection cohort, seed 0) |
|---|---|---|
| `pushobj` | `pred_lr 2e-3`, `horizon 5` | 0.685 at (2e-3, 5); expansion to horizon 8 ≤ 0.675 |
| `pusht` | `pred_lr 1e-2`, `horizon 5` | 0.447 at (1e-2, 5), (1e-2, 8); expansions to 4e-2 and horizon 8 ≤ 0.447 |
| `maze_medium` | `pred_lr 2e-3`, `horizon 5` | 0.820 at (2e-3, 5), (1e-2, 1); expansion to horizon 8 ≤ 0.800 |

`pushobj_shift` inherits from `pushobj`; the three maze shift settings inherit from
`maze_medium`.

## Cost and determinism

Episode-isolated (`requires_episode_isolation: true`), for the same reason as v2. The only
stochastic element is the `A` initialization, normal with std `1/√in` from a CPU generator seeded by
`lora_init_seed: 0`; `B` and `dm` start at zero.

## Reported per replan

The v2 keys, under the prefix `adajepa_v2_dora/` (`loss`, `rollout_depth`, `fresh_loss`,
`fresh_loss_frozen`, `fresh_loss_ratio`, `braked`, `brakes`, `buffer_size`).

## Reproduce

```bash
.venv/bin/python scripts/evaluate.py adajepa_v2_dora --setting pushobj --gpus 0,1,2,3 --per-gpu 4
```
