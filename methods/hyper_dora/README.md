# Hyper-DoRA — amortized hypernetwork, magnitude + direction

[`methods/hyperlora`](../hyperlora/) with one change: the generator emits a rank-2 DoRA
correction instead of a rank-2 LoRA. The context encoder (`transition_buffer`, 5
transitions, `transformer_query`), the training recipe and the selection axis are
HyperLoRA's. Its control is [`methods/static_dora`](../static_dora/): the same correction
shape, the same data and recipe, no context.

| | correction emitted per replan | generator output | trainable parameters |
|---|---|---|---|
| HyperLoRA | rank-2 LoRA on 4 linears + final LayerNorm delta | 20,424 | 11.3M |
| **Hyper-DoRA** | rank-2 direction `B·A` + per-output magnitude delta `dm` on the same 4 linears + final LayerNorm delta | **26,352** | **14.3M** |

Each adapted linear computes `W' = (m0 + dm) · (W + B·A) / ‖W + B·A‖` (row norms, closed
form). An all-zero `B` and `dm` gives a bit-identical frozen layer; the generator's output
layer is initialised at scale 1e-4. Implementation: `models/shapes.py`, `models/hyper_lora.py`.

## Results

Held-out test cohorts (seeds 100/200/300). In brackets: Hyper-DoRA minus Static-DoRA.

| setting | Hyper-DoRA | Static-DoRA | HyperLoRA | frozen | catastrophe | compounding | adapt s/replan |
|---|---|---|---|---|---|---|---|
| `pushobj` | **0.628** (+0.081) | 0.547 | 0.607 | 0.500 | 6.8% | +0.04 [-0.11, +0.15] | 0.142 |
| `pushobj_shift` | 0.322 (+0.035) | 0.287 | 0.322 | 0.300 | 12.3% | -0.04 [-0.36, +0.19] | 0.140 |
| `pusht` | 0.427 (+0.063) | 0.364 | 0.453 | 0.349 | 9.2% | -0.26 [-0.52, +0.05] | 0.151 |
| `maze_medium` | 0.773 (+0.020) | 0.753 | 0.793 | 0.773 | 4.0% | +0.01 [-0.01, +0.02] | 0.154 |
| `maze_medium_low_density` | 0.800 (-0.013) | 0.813 | 0.807 | 0.747 | 35.3% | +0.02 [-0.01, +0.07] | 0.165 |
| `maze_medium_high_damping` | 0.587 (+0.040) | 0.547 | 0.533 | 0.567 | 2.1% | +0.01 [-0.00, +0.02] | 0.148 |
| `maze_diverse` | **0.647** (+0.020) | 0.627 | 0.587 | 0.593 | 9.4% | -0.00 [-0.03, +0.02] | 0.142 |

Pooled over the 2,100 held-out episodes: **0.540** success (Static-DoRA 0.492, HyperLoRA
0.534, frozen 0.473).

- Highest success of any amortized row on `pushobj` (0.628) and `maze_diverse` (0.647).
- Matched minus static is +0.049 pooled (HyperLoRA minus Static LoRA: +0.041), and
  positive on six of seven settings.
- Compounding slope within ±0.04 per replan on six settings; -0.26 on `pusht`.
- Adaptation cost per replan within 0.03 s of HyperLoRA.

## Selection

One axis, `training_epoch`, initial 1, 2, 3, pool 1–5, on the selection cohort.

| setting | froze | selection cohort | cost |
|---|---|---|---|
| `pushobj` | epoch 3 | e1 0.555, e2 0.570, e3 0.590, e4 0.550 | 4 |
| `pusht` | epoch 1 | e1 0.440, e2 0.433, e3 0.433 | 3 |
| `maze_medium` | epoch 4 | e1 0.740, e2 0.700, e3 0.800, e4 0.820, e5 0.800 | 5 |

`pushobj_shift` inherits from `pushobj`; the three maze shift settings inherit from
`maze_medium`.

## Offline training

```bash
.venv/bin/python methods/hyperlora/train.py \
  --ckpt-dir checkpoints/pushobj_shape_shift --model-epoch latest \
  --data-path data/pushobj_multishape \
  --output-dir checkpoints/pushobj_adapters/hyper_dora_r2 \
  --correction-shape dora --no-distill-adapter --rank 2 \
  --target-scope predlast_all --context-mode transition_buffer \
  --context-aggregator transformer_query --context-transitions 5 \
  --epochs 5 --device cuda:0
```

PushT: `--ckpt-dir checkpoints/pusht_visual_shift`, output `checkpoints/pvs_adapters/hyper_dora_r2`.
MediumMaze: `--ckpt-dir checkpoints/mediummaze_dynamics_shift --data-path data/point_maze_medium`,
output `checkpoints/maze_medium_adapters/hyper_dora_r2`. All five epoch files are
required (`hyper_lora_epoch_{1..5}.pth`); they are listed in `CHECKPOINTS.sha256`.
