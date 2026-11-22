# AdaJEPA v2 · (IA)³ — v2 fitting rule, multiplicative rescaling

[`methods/adajepa_v2`](../adajepa_v2/) with one change: the correction is (IA)³ instead
of a rank-2 LoRA. The fitting rule, optimizer, clipping, brake, buffer and tuned axes are
v2's, at v2's selection cost. Read the v2 README for the fitting rule.

| correction | sites on the predictor's last block | parameters |
|---|---|---|
| v2 (LoRA r=2) | `to_qkv`, `to_out.0`, `net.1`, `net.4` + final LayerNorm delta | 20,424 |
| **this row ((IA)³)** | keys and values of `to_qkv` (scale `1 + s`, 2,048), input of `net.4` = MLP hidden activation (2,048) + final LayerNorm delta (808) | **4,904** |

`s = 0` gives a bit-identical frozen layer, so the first replan of every episode plans with
the frozen model. The correction can only rescale features the frozen block already
computes. Implementation: `models/shapes.py` (`HyperIA3Linear`, `install_predictor_shape_targets`).

## Results

All seven settings, held-out test cohorts (seeds 100/200/300). Paired metrics are against
the batched Frozen reference on the same episodes.

| setting | success | v2 LoRA | frozen | median dist Δ | catastrophe | compounding | regret | adapt s/replan |
|---|---|---|---|---|---|---|---|---|
| `pushobj` | 0.660 | 0.697 | 0.500 | +10 [+4, +19] | **12.6%** vs 14.4% | +0.62 [+0.31, +1.17] | 61% / +250 | 0.220 |
| `pushobj_shift` | 0.353 | 0.409 | 0.300 | +7 [+2, +14] | **7.1%** vs 15.9% | +0.50 [+0.20, +0.83] | 59% / +118 | 0.189 |
| `pusht` | 0.433 | 0.478 | 0.349 | +37 [+19, +50] | 24.5% vs 22.2% | +2.29 [+1.27, +3.34] | 62% / +308 | **0.094** |
| `maze_medium` | 0.840 | 0.847 | 0.773 | +0 [-0, +1] | 0.0% vs 0.0% | -0.00 [-0.04, +0.01] | 65% / +1 | 0.289 |
| `maze_medium_low_density` | 0.747 | 0.773 | 0.747 | +1 [-0, +7] | 30.0% vs 6.2% | -0.03 [-0.11, +0.03] | 70% / +14 | 0.283 |
| `maze_medium_high_damping` | 0.720 | 0.740 | 0.567 | +0 [-0, +1] | 0.0% vs 0.0% | -0.00 [-0.01, +0.01] | 65% / +2 | 0.303 |
| `maze_diverse` | 0.653 | 0.727 | 0.593 | -0 [-1, +0] | 6.1% vs 7.7% | -0.06 [-0.09, -0.02] | 36% / +2 | 0.266 |

Pooled over the 2,100 held-out episodes: **0.569** success (v2 LoRA 0.610, frozen 0.473),
with 24% of v2's correction parameters.

- Above frozen on six of seven settings; ties frozen on `maze_medium_low_density`.
- Within one SE of v2 on `maze_medium` (0.840 vs 0.847) and `maze_medium_high_damping`
  (0.720 vs 0.740).
- Lowest catastrophe rate of the v2 family on `pushobj` (12.6%) and `pushobj_shift` (7.1%).
- On `pusht` the correction compounds faster than v2's (+2.29 vs +1.23 per replan).

## Selection

Same two axes and grid as v2 (`pred_lr` 5e-4, 2e-3, 1e-2, log expansion; `horizon` 1, 3,
5, pool 1, 2, 3, 5, 8). 12 columns on each selectable setting.

| setting | froze | best grid cells (selection cohort, seed 0) |
|---|---|---|
| `pushobj` | `pred_lr 5e-4`, `horizon 3` | 0.660 at (5e-4, 3), (2e-3, 1), (2e-3, 3); expansion to 1.25e-4 ≤ 0.500 |
| `pusht` | `pred_lr 1e-2`, `horizon 1` | 0.413 at (1e-2, 1), (1e-2, 3); expansion to 4e-2 ≤ 0.407 |
| `maze_medium` | `pred_lr 1e-2`, `horizon 3` | 0.860 at (1e-2, 3); expansion to 4e-2 ≤ 0.840 |

`pushobj_shift` inherits from `pushobj`; the three maze shift settings inherit from
`maze_medium`.

## Cost and determinism

Episode-isolated (`requires_episode_isolation: true`), for the same reason as v2. The
(IA)³ vectors start at zero; there is no random initialization in the correction.

## Reported per replan

The v2 keys, under the prefix `adajepa_v2_ia3/` (`loss`, `rollout_depth`, `fresh_loss`,
`fresh_loss_frozen`, `fresh_loss_ratio`, `braked`, `brakes`, `buffer_size`).

## Reproduce

```bash
.venv/bin/python scripts/evaluate.py adajepa_v2_ia3 --setting pushobj --gpus 0,1,2,3 --per-gpu 4
```
