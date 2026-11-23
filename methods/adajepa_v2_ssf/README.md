# AdaJEPA v2 · SSF — v2 fitting rule, per-channel scale and shift

[`methods/adajepa_v2`](../adajepa_v2/) with one change: the correction is SSF instead of a
rank-2 LoRA. The fitting rule, optimizer, clipping, brake, buffer and tuned axes are v2's,
at v2's selection cost. Read the v2 README for the fitting rule.

| correction | sites on the predictor's last block | parameters |
|---|---|---|
| v2 (LoRA r=2) | `to_qkv`, `to_out.0`, `net.1`, `net.4` + final LayerNorm delta | 20,424 |
| **this row (SSF)** | output `y (1 + g) + b` on the same four linears (11,856) + final LayerNorm delta (808) | **12,664** |

`g = b = 0` gives a bit-identical frozen layer, so the first replan of every episode plans
with the frozen model. The correction can only rescale and shift the output channels the
frozen block already computes. Implementation: `models/shapes.py` (`HyperSSFLinear`,
`install_predictor_shape_targets`).

## Results

All seven settings, held-out test cohorts (seeds 100/200/300). Paired metrics are against
the batched Frozen reference on the same episodes.

| setting | success | v2 LoRA | frozen | median dist Δ | catastrophe | compounding | regret | adapt s/replan |
|---|---|---|---|---|---|---|---|---|
| `pushobj` | 0.685 | 0.697 | 0.500 | +6 [-0, +18] | 17.0% vs 14.4% | +0.41 [-0.06, +1.41] | 56% / +357 | 0.231 |
| `pushobj_shift` | 0.402 | 0.409 | 0.300 | +13 [+4, +22] | **14.3%** vs 15.9% | +0.93 [+0.50, +1.53] | 59% / +178 | 0.206 |
| `pusht` | 0.462 | 0.478 | 0.349 | +22 [+9, +32] | **13.8%** vs 22.2% | +1.12 [+0.46, +1.85] | 62% / +245 | 0.299 |
| `maze_medium` | 0.827 | 0.847 | 0.773 | +0 [-1, +0] | 0.0% vs 0.0% | -0.00 [-0.07, +0.02] | 53% / +1 | 0.137 |
| `maze_medium_low_density` | 0.760 | 0.773 | 0.747 | -1 [-2, +0] | 12.5% vs 6.2% | +0.02 [-0.04, +0.07] | 38% / +7 | 0.112 |
| `maze_medium_high_damping` | 0.740 | 0.740 | 0.567 | +0 [-0, +0] | 0.0% vs 0.0% | +0.00 [-0.01, +0.02] | 50% / +2 | 0.134 |
| `maze_diverse` | 0.620 | 0.727 | 0.593 | -1 [-1, +0] | 9.1% vs 7.7% | -0.06 [-0.09, -0.02] | 39% / +2 | 0.153 |

Pooled over the 2,100 held-out episodes: **0.591** success (v2 LoRA 0.610, frozen 0.473),
with 62% of v2's correction parameters.

- Above frozen on all seven settings.
- Within one SE of v2 on six of seven settings; equal to v2 on `maze_medium_high_damping`
  (0.740).
- Lowest PushT catastrophe rate of the v2 family: 13.8% (v2 LoRA 22.2%, (IA)³ 24.5%,
  DoRA 25.6%).

## Selection

Same two axes and grid as v2 (`pred_lr` 5e-4, 2e-3, 1e-2, log expansion; `horizon` 1, 3,
5, pool 1, 2, 3, 5, 8).

| setting | froze | best grid cells (selection cohort, seed 0) | cost |
|---|---|---|---|
| `pushobj` | `pred_lr 2e-3`, `horizon 3` | 0.675 at (2e-3, 3); 0.670 at (5e-4, 5), (2e-3, 5) | 9 |
| `pusht` | `pred_lr 2e-3`, `horizon 5` | 0.427 at (2e-3, 5), (2e-3, 8); expansion to horizon 8 ≤ 0.427 | 12 |
| `maze_medium` | `pred_lr 1e-2`, `horizon 1` | 0.900 at (1e-2, 1); expansion to 4e-2 ≤ 0.840 | 12 |

`pushobj_shift` inherits from `pushobj`; the three maze shift settings inherit from
`maze_medium`.

## Cost and determinism

Episode-isolated (`requires_episode_isolation: true`), for the same reason as v2. The SSF
vectors start at zero; there is no random initialization in the correction.

## Reported per replan

The v2 keys, under the prefix `adajepa_v2_ssf/` (`loss`, `rollout_depth`, `fresh_loss`,
`fresh_loss_frozen`, `fresh_loss_ratio`, `braked`, `brakes`, `buffer_size`).

## Reproduce

```bash
.venv/bin/python scripts/evaluate.py adajepa_v2_ssf --setting pushobj --gpus 0,1,2,3 --per-gpu 4
```
