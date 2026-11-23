# AdaJEPA v2 · AdaLN — v2 fitting rule, adaptive LayerNorms

[`methods/adajepa_v2`](../adajepa_v2/) with one change: the correction is a set of affine
deltas on every LayerNorm of the predictor instead of a rank-2 LoRA on the last block. No
linear layer is changed. The fitting rule, optimizer, clipping, brake, buffer and tuned
axes are v2's, at v2's selection cost. Read the v2 README for the fitting rule.

| correction | sites on the predictor | parameters |
|---|---|---|
| v2 (LoRA r=2) | `to_qkv`, `to_out.0`, `net.1`, `net.4` of the last block + final LayerNorm delta | 20,424 |
| **this row (AdaLN)** | weight and bias deltas on all 13 LayerNorms: attention and MLP norm at each of 6 depths, plus the final norm (13 × 808) | **10,504** |

Each LayerNorm computes `LN(x) · (γ + dγ) + (β + dβ)`. All-zero deltas give a bit-identical
frozen predictor, so the first replan of every episode plans with the frozen model. Unlike
the other shapes, the correction reaches every depth of the predictor. Implementation:
`models/shapes.py` (`_install_all_layer_norms`, `install_predictor_shape_targets`).

## Results

All seven settings, held-out test cohorts (seeds 100/200/300). Paired metrics are against
the batched Frozen reference on the same episodes.

| setting | success | v2 LoRA | frozen | median dist Δ | catastrophe | compounding | regret | adapt s/replan |
|---|---|---|---|---|---|---|---|---|
| `pushobj` | 0.688 | 0.697 | 0.500 | +3 [-4, +12] | **8.7%** vs 14.4% | +0.12 [-0.34, +0.88] | 53% / +158 | **0.106** |
| `pushobj_shift` | 0.378 | 0.409 | 0.300 | +21 [+7, +30] | 16.7% vs 15.9% | +1.05 [+0.34, +1.68] | 59% / +291 | 0.107 |
| `pusht` | 0.436 | 0.478 | 0.349 | +21 [+6, +48] | 22.5% vs 22.2% | +1.56 [+0.75, +2.50] | 58% / +305 | 0.323 |
| `maze_medium` | **0.853** | 0.847 | 0.773 | +0 [+0, +2] | 6.7% vs 0.0% | +0.02 [-0.01, +0.05] | 73% / +4 | 0.269 |
| `maze_medium_low_density` | 0.753 | 0.773 | 0.747 | -1 [-3, +4] | 31.6% vs 6.2% | -0.03 [-0.08, +0.05] | 42% / +9 | 0.385 |
| `maze_medium_high_damping` | **0.767** | 0.740 | 0.567 | -0 [-1, +0] | 0.0% vs 0.0% | +0.00 [-0.02, +0.02] | 46% / +1 | 0.396 |
| `maze_diverse` | 0.707 | 0.727 | 0.593 | -0 [-1, +1] | **3.8%** vs 7.7% | -0.06 [-0.09, -0.01] | 42% / +1 | 0.256 |

Pooled over the 2,100 held-out episodes: **0.591** success (v2 LoRA 0.610, frozen 0.473),
with 51% of v2's correction parameters.

- Highest `maze_medium_high_damping` success of every method on the leaderboard (0.767),
  and highest on `maze_medium` (0.853, equal with AdaJEPA).
- Above frozen on all seven settings.
- Lowest catastrophe rate of the v2 family on `pushobj` (8.7%) and `maze_diverse` (3.8%).
- Lowest adaptation cost of the v2 family on `pushobj`: 0.106 s per replan (v2 LoRA 0.239).

## Selection

Same two axes and grid as v2 (`pred_lr` 5e-4, 2e-3, 1e-2, log expansion; `horizon` 1, 3,
5, pool 1, 2, 3, 5, 8).

| setting | froze | best grid cells (selection cohort, seed 0) | cost |
|---|---|---|---|
| `pushobj` | `pred_lr 5e-4`, `horizon 1` | 0.695 at (5e-4, 1); 0.690 at (5e-4, 3); 0.685 at (2e-3, 5); expansion to 1.25e-4 ≤ 0.665 | 12 |
| `pusht` | `pred_lr 5e-4`, `horizon 5` | 0.447 at (5e-4, 5), (5e-4, 8); expansion to 1.25e-4 and horizon 8 ≤ 0.447 | 14 |
| `maze_medium` | `pred_lr 2e-3`, `horizon 3` | 0.880 at (2e-3, 3); 0.840 at (2e-3, 5), (1e-2, 3) | 9 |

`pushobj_shift` inherits from `pushobj`; the three maze shift settings inherit from
`maze_medium`. In the first run, one episode in the (2e-3, 5) `pushobj` cell and one in the
(2e-3, 1) `maze_medium` cell stopped on a CUDA "device busy" error. The re-run completed
both cells (0.685 and 0.800). Neither changes the selection.

## Cost and determinism

Episode-isolated (`requires_episode_isolation: true`), for the same reason as v2. The
deltas start at zero; there is no random initialization in the correction.

## Reported per replan

The v2 keys, under the prefix `adajepa_v2_adaln/` (`loss`, `rollout_depth`, `fresh_loss`,
`fresh_loss_frozen`, `fresh_loss_ratio`, `braked`, `brakes`, `buffer_size`).

## Reproduce

```bash
.venv/bin/python scripts/evaluate.py adajepa_v2_adaln --setting pushobj --gpus 0,1,2,3 --per-gpu 4
```
