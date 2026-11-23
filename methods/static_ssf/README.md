# Static SSF — unconditioned correction

The control for [`methods/hyper_ssf`](../hyper_ssf/). One learned SSF correction (output
`y (1 + g) + b` on the four linears of the predictor's last block, plus the final LayerNorm
delta; 12,664 parameters), applied to every episode and never updated. Same offline
trainer, data and recipe as Hyper-SSF, with no context and no generator.

## Results

Held-out test cohorts (seeds 100/200/300).

| setting | Static-SSF | Hyper-SSF | Static LoRA | frozen | catastrophe |
|---|---|---|---|---|---|
| `pushobj` | 0.498 | 0.628 | 0.547 | 0.500 | 2.7% |
| `pushobj_shift` | 0.278 | 0.327 | 0.296 | 0.300 | 7.6% |
| `pusht` | 0.376 | 0.453 | 0.360 | 0.349 | 9.4% |
| `maze_medium` | 0.780 | 0.773 | 0.780 | 0.773 | 0.0% |
| `maze_medium_low_density` | **0.813** | 0.713 | 0.787 | 0.747 | 25.0% |
| `maze_medium_high_damping` | **0.620** | 0.573 | 0.573 | 0.567 | 2.3% |
| `maze_diverse` | 0.613 | 0.667 | 0.607 | 0.593 | 8.8% |

Pooled over the 2,100 held-out episodes: 0.484 (Hyper-SSF 0.541, Static LoRA 0.493,
frozen 0.473).

## Selection

`training_epoch`, initial 1, 2, 3, pool 1–5. Froze epoch 3 on `pushobj` (e1 0.400, e2
0.435, e3 0.445, e4 0.435; 4 columns), epoch 2 on `pusht` (0.320, 0.360, 0.333; 3
columns), epoch 1 on `maze_medium` (0.780, 0.720, 0.760; 3 columns).

## Offline training

```bash
.venv/bin/python methods/static_lora/train.py \
  --ckpt-dir checkpoints/pushobj_shape_shift --model-epoch latest \
  --data-path data/pushobj_multishape \
  --output-dir checkpoints/pushobj_adapters/static_ssf_r2 \
  --correction-shape ssf --no-distill-adapter --rank 2 --epochs 5 --device cuda:0
```

PushT and MediumMaze: as for [`hyper_ssf`](../hyper_ssf/README.md), with output
directories `static_ssf_r2`.
