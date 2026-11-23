# Hyper-SSF — amortized hypernetwork, per-channel scale and shift

[`methods/hyperlora`](../hyperlora/) with one change: the generator emits an SSF correction
instead of a rank-2 LoRA. The context encoder (`transition_buffer`, 5 transitions,
`transformer_query`), the training recipe and the selection axis are HyperLoRA's. Its
control is [`methods/static_ssf`](../static_ssf/): the same correction shape, the same data
and recipe, no context.

| | correction emitted per replan | generator output | trainable parameters |
|---|---|---|---|
| HyperLoRA | rank-2 LoRA on 4 linears + final LayerNorm delta | 20,424 | 11.3M |
| **Hyper-SSF** | output `y (1 + g) + b` on the same 4 linears (11,856) + final LayerNorm delta (808) | **12,664** | **7.32M** |

`g = b = 0` gives a bit-identical frozen layer; the generator's output layer is initialised
at scale 1e-4. Implementation: `models/shapes.py` (`HyperSSFLinear`), `models/hyper_lora.py`.

## Results

Held-out test cohorts (seeds 100/200/300). In brackets: Hyper-SSF minus Static-SSF.

| setting | Hyper-SSF | Static-SSF | HyperLoRA | frozen | catastrophe | compounding | adapt s/replan |
|---|---|---|---|---|---|---|---|
| `pushobj` | 0.628 (+0.130) | 0.498 | 0.607 | 0.500 | 7.3% | -0.06 [-0.19, +0.06] | 0.134 |
| `pushobj_shift` | 0.327 (+0.049) | 0.278 | 0.322 | 0.300 | 6.2% | -0.17 [-0.36, +0.07] | 0.131 |
| `pusht` | 0.453 (+0.077) | 0.376 | 0.453 | 0.349 | 8.5% | -0.11 [-0.39, +0.11] | 0.131 |
| `maze_medium` | 0.773 (-0.007) | 0.780 | 0.793 | 0.773 | 0.0% | -0.02 [-0.04, +0.00] | 0.258 |
| `maze_medium_low_density` | 0.713 (-0.100) | 0.813 | 0.807 | 0.747 | 20.0% | +0.01 [-0.02, +0.07] | 0.267 |
| `maze_medium_high_damping` | 0.573 (-0.047) | 0.620 | 0.533 | 0.567 | 0.0% | -0.00 [-0.02, +0.01] | 0.192 |
| `maze_diverse` | **0.667** (+0.054) | 0.613 | 0.587 | 0.593 | 18.2% | -0.00 [-0.02, +0.02] | 0.161 |

Pooled over the 2,100 held-out episodes: **0.541** success (Static-SSF 0.484, HyperLoRA
0.534, frozen 0.473).

- Matched minus static is +0.057 pooled (HyperLoRA minus Static LoRA: +0.041), and +0.130
  on `pushobj`.
- Highest success of any amortized row on `maze_diverse` (0.667).
- Lowest mean catastrophe rate of the five Hyper rows: 8.0% over the seven settings
  (HyperLoRA 9.4%).
- Compounding slope negative or within ±0.02 per replan on every setting.
- 65% of HyperLoRA's trainable parameters.

## Selection

One axis, `training_epoch`, initial 1, 2, 3, pool 1–5, on the selection cohort.

| setting | froze | selection cohort | cost |
|---|---|---|---|
| `pushobj` | epoch 2 | e1 0.595, e2 0.615, e3 0.560 | 3 |
| `pusht` | epoch 4 | e1 0.380, e2 0.393, e3 0.433, e4 0.447, e5 0.440 | 5 |
| `maze_medium` | epoch 4 | e1 0.740, e2 0.740, e3 0.760, e4 0.780, e5 0.680 | 5 |

`pushobj_shift` inherits from `pushobj`; the three maze shift settings inherit from
`maze_medium`.

## Offline training

```bash
.venv/bin/python methods/hyperlora/train.py \
  --ckpt-dir checkpoints/pushobj_shape_shift --model-epoch latest \
  --data-path data/pushobj_multishape \
  --output-dir checkpoints/pushobj_adapters/hyper_ssf_r2 \
  --correction-shape ssf --no-distill-adapter --rank 2 \
  --target-scope predlast_all --context-mode transition_buffer \
  --context-aggregator transformer_query --context-transitions 5 \
  --epochs 5 --device cuda:0
```

PushT: `--ckpt-dir checkpoints/pusht_visual_shift`, output `checkpoints/pvs_adapters/hyper_ssf_r2`.
MediumMaze: `--ckpt-dir checkpoints/mediummaze_dynamics_shift --data-path data/point_maze_medium`,
output `checkpoints/maze_medium_adapters/hyper_ssf_r2`. All five epoch files are
required (`hyper_lora_epoch_{1..5}.pth`); they are listed in `CHECKPOINTS.sha256`.
