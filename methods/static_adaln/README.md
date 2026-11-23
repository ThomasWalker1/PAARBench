# Static AdaLN — unconditioned correction

The control for [`methods/hyper_adaln`](../hyper_adaln/). One learned set of affine deltas
on all 13 LayerNorms of the predictor (attention and MLP norm at each of 6 depths, plus the
final norm; 10,504 parameters), applied to every episode and never updated. No linear
layer is changed. Same offline trainer, data and recipe as Hyper-AdaLN, with no context
and no generator.

## Results

Held-out test cohorts (seeds 100/200/300).

| setting | Static-AdaLN | Hyper-AdaLN | Static LoRA | frozen | catastrophe |
|---|---|---|---|---|---|
| `pushobj` | **0.625** | 0.648 | 0.547 | 0.500 | 7.9% |
| `pushobj_shift` | 0.327 | 0.349 | 0.296 | 0.300 | 14.7% |
| `pusht` | 0.376 | 0.509 | 0.360 | 0.349 | 13.8% |
| `maze_medium` | 0.760 | 0.740 | 0.780 | 0.773 | 0.0% |
| `maze_medium_low_density` | **0.813** | 0.800 | 0.787 | 0.747 | 17.6% |
| `maze_medium_high_damping` | 0.573 | 0.520 | 0.573 | 0.567 | 0.0% |
| `maze_diverse` | 0.600 | 0.613 | 0.607 | 0.593 | 19.4% |

Pooled over the 2,100 held-out episodes: **0.525** (Hyper-AdaLN 0.560, Static LoRA 0.493,
frozen 0.473). This is the highest pooled success of the five Static rows, and above
HyperLoRA on `pushobj` (0.625 vs 0.607).

## Selection

`training_epoch`, initial 1, 2, 3, pool 1–5. Froze epoch 3 on `pushobj` (e1 0.565, e2
0.590, e3 0.615, e4 0.610; 4 columns), epoch 2 on `pusht` (0.327, 0.373, 0.333; 3
columns), epoch 1 on `maze_medium` (0.780, 0.720, 0.720; 3 columns).

## Offline training

```bash
.venv/bin/python methods/static_lora/train.py \
  --ckpt-dir checkpoints/pushobj_shape_shift --model-epoch latest \
  --data-path data/pushobj_multishape \
  --output-dir checkpoints/pushobj_adapters/static_adaln_r2 \
  --correction-shape adaln --no-distill-adapter --rank 2 --epochs 5 --device cuda:0
```

PushT and MediumMaze: as for [`hyper_adaln`](../hyper_adaln/README.md), with output
directories `static_adaln_r2`.
