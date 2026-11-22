# Hyper-(IA)³ — amortized hypernetwork, multiplicative rescaling

[`methods/hyperlora`](../hyperlora/) with one change: the generator emits an (IA)³
correction instead of a rank-2 LoRA. The context encoder (`transition_buffer`, 5
transitions, `transformer_query`), the training recipe and the selection axis are
HyperLoRA's. Its control is [`methods/static_ia3`](../static_ia3/): the same correction
shape, the same data and recipe, no context.

| | correction emitted per replan | generator output | trainable parameters |
|---|---|---|---|
| HyperLoRA | rank-2 LoRA on 4 linears + final LayerNorm delta | 20,424 | 11.3M |
| **Hyper-(IA)³** | scale `1 + s` on `to_qkv` keys and values (2,048) and on the MLP hidden activation (2,048) + final LayerNorm delta (808) | **4,904** | **3.34M** |

`s = 0` gives a bit-identical frozen layer; the generator's output layer is initialised
at scale 1e-4. Implementation: `models/shapes.py`, `models/hyper_lora.py`.

## Results

Held-out test cohorts (seeds 100/200/300). In brackets: Hyper-(IA)³ minus Static-(IA)³.

| setting | Hyper-(IA)³ | Static-(IA)³ | HyperLoRA | frozen | catastrophe | compounding | adapt s/replan |
|---|---|---|---|---|---|---|---|
| `pushobj` | 0.605 (+0.117) | 0.488 | 0.607 | 0.500 | 6.4% | -0.01 [-0.10, +0.09] | 0.147 |
| `pushobj_shift` | **0.329** (+0.031) | 0.298 | 0.322 | 0.300 | 6.5% | -0.06 [-0.25, +0.13] | 0.136 |
| `pusht` | 0.431 (+0.051) | 0.380 | 0.453 | 0.349 | 12.1% | +0.05 [-0.14, +0.22] | 0.128 |
| `maze_medium` | 0.787 (-0.013) | 0.800 | 0.793 | 0.773 | 5.0% | -0.00 [-0.02, +0.01] | 0.171 |
| `maze_medium_low_density` | 0.760 (+0.013) | 0.747 | 0.807 | 0.747 | 31.6% | -0.02 [-0.06, +0.02] | 0.135 |
| `maze_medium_high_damping` | **0.607** (+0.080) | 0.527 | 0.533 | 0.567 | 0.0% | +0.00 [-0.01, +0.02] | 0.138 |
| `maze_diverse` | 0.640 (+0.020) | 0.620 | 0.587 | 0.593 | 8.3% | +0.00 [-0.01, +0.03] | 0.151 |

Pooled over the 2,100 held-out episodes: **0.535** success (Static-(IA)³ 0.477,
HyperLoRA 0.534, frozen 0.473).

- Above frozen on all seven settings.
- Matched minus static is +0.058 pooled (HyperLoRA minus Static LoRA: +0.041), and
  positive on six of seven settings.
- Highest success of any amortized row on `pushobj_shift` and `maze_medium_high_damping`.
- Compounding slope within ±0.06 per replan on every setting.

## Selection

One axis, `training_epoch`, initial 1, 2, 3, pool 1–5, on the selection cohort.

| setting | froze | selection cohort | cost |
|---|---|---|---|
| `pushobj` | epoch 3 | e1 0.575, e2 0.545, e3 0.575, e4 0.570 | 4 |
| `pusht` | epoch 1 | e1 0.380, e2 0.360, e3 0.373 | 3 |
| `maze_medium` | epoch 1 | e1 0.820, e2 0.720, e3 0.680 | 3 |

`pushobj_shift` inherits from `pushobj`; the three maze shift settings inherit from
`maze_medium`.

## Offline training

```bash
.venv/bin/python methods/hyperlora/train.py \
  --ckpt-dir checkpoints/pushobj_shape_shift --model-epoch latest \
  --data-path data/pushobj_multishape \
  --output-dir checkpoints/pushobj_adapters/hyper_ia3_r2 \
  --correction-shape ia3 --no-distill-adapter --rank 2 \
  --target-scope predlast_all --context-mode transition_buffer \
  --context-aggregator transformer_query --context-transitions 5 \
  --epochs 5 --device cuda:0
```

PushT: `--ckpt-dir checkpoints/pusht_visual_shift`, output `checkpoints/pvs_adapters/hyper_ia3_r2`.
MediumMaze: `--ckpt-dir checkpoints/mediummaze_dynamics_shift --data-path data/point_maze_medium`,
output `checkpoints/maze_medium_adapters/hyper_ia3_r2`. All five epoch files are
required (`hyper_lora_epoch_{1..5}.pth`); they are listed in `CHECKPOINTS.sha256`.
`--correction-shape` other than `lora` does not combine with `--distill-adapter`,
`--dense-finetune`, `--adapter-gate` or `--freeze-lora-a`.
