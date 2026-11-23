# Hyper-AdaLN — amortized hypernetwork, adaptive LayerNorms

[`methods/hyperlora`](../hyperlora/) with one change: the generator emits affine deltas for
every LayerNorm in the predictor instead of a rank-2 LoRA. No linear layer is changed. The
context encoder (`transition_buffer`, 5 transitions, `transformer_query`), the training
recipe and the selection axis are HyperLoRA's. Its control is
[`methods/static_adaln`](../static_adaln/): the same correction shape, the same data and
recipe, no context.

| | correction emitted per replan | generator output | trainable parameters |
|---|---|---|---|
| HyperLoRA | rank-2 LoRA on 4 linears of the last block + final LayerNorm delta | 20,424 | 11.3M |
| **Hyper-AdaLN** | weight and bias deltas on all 13 LayerNorms (attention and MLP norm at each of 6 depths + final norm) | **10,504** | **6.22M** |

Each LayerNorm computes `LN(x) · (γ + dγ) + (β + dβ)`. All-zero deltas give a bit-identical
frozen predictor; the generator's output layer is initialised at scale 1e-4. Unlike the
other shapes, the correction reaches every depth of the predictor. Implementation:
`models/shapes.py` (`_install_all_layer_norms`), `models/hyper_lora.py`.

## Results

Held-out test cohorts (seeds 100/200/300). In brackets: Hyper-AdaLN minus Static-AdaLN.

| setting | Hyper-AdaLN | Static-AdaLN | HyperLoRA | frozen | catastrophe | compounding | adapt s/replan |
|---|---|---|---|---|---|---|---|
| `pushobj` | **0.648** (+0.023) | 0.625 | 0.607 | 0.500 | 15.5% | -0.07 [-0.29, +0.06] | 0.132 |
| `pushobj_shift` | **0.349** (+0.022) | 0.327 | 0.322 | 0.300 | 13.4% | +0.35 [+0.02, +0.73] | 0.133 |
| `pusht` | **0.509** (+0.133) | 0.376 | 0.453 | 0.349 | 11.8% | -0.08 [-0.59, +0.35] | 0.133 |
| `maze_medium` | 0.740 (-0.020) | 0.760 | 0.793 | 0.773 | 0.0% | -0.01 [-0.04, +0.01] | 0.241 |
| `maze_medium_low_density` | 0.800 (-0.013) | 0.813 | 0.807 | 0.747 | 9.5% | +0.00 [-0.03, +0.07] | 0.254 |
| `maze_medium_high_damping` | 0.520 (-0.053) | 0.573 | 0.533 | 0.567 | 0.0% | +0.00 [-0.01, +0.02] | 0.254 |
| `maze_diverse` | 0.613 (+0.013) | 0.600 | 0.587 | 0.593 | 26.3% | -0.00 [-0.02, +0.01] | 0.272 |

Pooled over the 2,100 held-out episodes: **0.560** success (Static-AdaLN 0.525, HyperLoRA
0.534, frozen 0.473). This is the highest pooled success of the five Hyper rows.

- Highest PushT success of every method on the leaderboard, online or amortized: 0.509
  (AdaJEPA v2 0.478), with 11.8% catastrophe (AdaJEPA v2 22.2%).
- Highest success of any amortized row on `pushobj` (0.648) and `pushobj_shift` (0.349).
- Matched minus static is +0.035 pooled and +0.133 on `pusht`.
- 55% of HyperLoRA's trainable parameters.

## Selection

One axis, `training_epoch`, initial 1, 2, 3, pool 1–5, on the selection cohort.

| setting | froze | selection cohort | cost |
|---|---|---|---|
| `pushobj` | epoch 4 | e1 0.595, e2 0.590, e3 0.600, e4 0.640, e5 0.610 | 5 |
| `pusht` | epoch 3 | e1 0.407, e2 0.453, e3 0.487, e4 0.480 | 4 |
| `maze_medium` | epoch 1 | e1 0.740, e2 0.740, e3 0.680 | 3 |

`pushobj_shift` inherits from `pushobj`; the three maze shift settings inherit from
`maze_medium`.

## Offline training

```bash
.venv/bin/python methods/hyperlora/train.py \
  --ckpt-dir checkpoints/pushobj_shape_shift --model-epoch latest \
  --data-path data/pushobj_multishape \
  --output-dir checkpoints/pushobj_adapters/hyper_adaln_r2 \
  --correction-shape adaln --no-distill-adapter --rank 2 \
  --target-scope predlast_all --context-mode transition_buffer \
  --context-aggregator transformer_query --context-transitions 5 \
  --epochs 5 --device cuda:0
```

PushT: `--ckpt-dir checkpoints/pusht_visual_shift`, output `checkpoints/pvs_adapters/hyper_adaln_r2`.
MediumMaze: `--ckpt-dir checkpoints/mediummaze_dynamics_shift --data-path data/point_maze_medium`,
output `checkpoints/maze_medium_adapters/hyper_adaln_r2`. All five epoch files are
required (`hyper_lora_epoch_{1..5}.pth`); they are listed in `CHECKPOINTS.sha256`.
`--rank` and `--target-scope` are recorded in the checkpoint metadata but do not change
the AdaLN correction, which always covers every LayerNorm.
