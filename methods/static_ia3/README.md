# Static (IA)³ — unconditioned correction

The control for [`methods/hyper_ia3`](../hyper_ia3/). One learned (IA)³ correction (scale
`1 + s` on the `to_qkv` keys and values and on the MLP hidden activation of the
predictor's last block, plus the final LayerNorm delta; 4,904 parameters), applied to
every episode and never updated. Same offline trainer, data and recipe as Hyper-(IA)³,
with no context and no generator.

## Results

Held-out test cohorts (seeds 100/200/300).

| setting | Static-(IA)³ | Hyper-(IA)³ | Static LoRA | frozen | catastrophe |
|---|---|---|---|---|---|
| `pushobj` | 0.488 | 0.605 | 0.547 | 0.500 | 2.2% |
| `pushobj_shift` | 0.298 | 0.329 | 0.296 | 0.300 | 4.7% |
| `pusht` | 0.380 | 0.431 | 0.360 | 0.349 | 11.0% |
| `maze_medium` | **0.800** | 0.787 | 0.780 | 0.773 | 0.0% |
| `maze_medium_low_density` | 0.747 | 0.760 | 0.787 | 0.747 | 22.7% |
| `maze_medium_high_damping` | 0.527 | 0.607 | 0.573 | 0.567 | 1.9% |
| `maze_diverse` | 0.620 | 0.640 | 0.607 | 0.593 | 12.8% |

Pooled over the 2,100 held-out episodes: 0.477 (Hyper-(IA)³ 0.535, Static LoRA 0.493,
frozen 0.473).

## Selection

`training_epoch`, initial 1, 2, 3, pool 1–5. Froze epoch 2 on `pushobj` (e1 0.405, e2
0.410, e3 0.410), `pusht` (0.340, 0.360, 0.347) and `maze_medium` (0.620, 0.780, 0.760).
3 columns each.

## Offline training

```bash
.venv/bin/python methods/static_lora/train.py \
  --ckpt-dir checkpoints/pushobj_shape_shift --model-epoch latest \
  --data-path data/pushobj_multishape \
  --output-dir checkpoints/pushobj_adapters/static_ia3_r2 \
  --correction-shape ia3 --no-distill-adapter --rank 2 --epochs 5 --device cuda:0
```

PushT and MediumMaze: as for [`hyper_ia3`](../hyper_ia3/README.md), with output
directories `static_ia3_r2`.
