# Static DoRA — unconditioned correction

The control for [`methods/hyper_dora`](../hyper_dora/). One learned rank-2 DoRA correction
(direction `B·A` and magnitude delta `dm` on the four linears of the predictor's last
block, plus the final LayerNorm delta; 26,352 parameters), applied to every episode and
never updated. Same offline trainer, data and recipe as Hyper-DoRA, with no context and
no generator.

## Results

Held-out test cohorts (seeds 100/200/300).

| setting | Static-DoRA | Hyper-DoRA | Static LoRA | frozen | catastrophe |
|---|---|---|---|---|---|
| `pushobj` | 0.547 | 0.628 | 0.547 | 0.500 | 2.6% |
| `pushobj_shift` | 0.287 | 0.322 | 0.296 | 0.300 | 9.2% |
| `pusht` | 0.364 | 0.427 | 0.360 | 0.349 | 12.7% |
| `maze_medium` | 0.753 | 0.773 | 0.780 | 0.773 | 0.0% |
| `maze_medium_low_density` | **0.813** | 0.800 | 0.787 | 0.747 | 16.7% |
| `maze_medium_high_damping` | 0.547 | 0.587 | 0.573 | 0.567 | 0.0% |
| `maze_diverse` | 0.627 | 0.647 | 0.607 | 0.593 | 11.8% |

Pooled over the 2,100 held-out episodes: 0.492 (Hyper-DoRA 0.540, Static LoRA 0.493,
frozen 0.473).

## Selection

`training_epoch`, initial 1, 2, 3, pool 1–5; 3 columns each. Froze epoch 1 on `pushobj`
(e1 0.530, e2 0.525, e3 0.525) and `pusht` (0.360, 0.353, 0.347), epoch 2 on
`maze_medium` (0.720, 0.740, 0.720).

## Offline training

```bash
.venv/bin/python methods/static_lora/train.py \
  --ckpt-dir checkpoints/pushobj_shape_shift --model-epoch latest \
  --data-path data/pushobj_multishape \
  --output-dir checkpoints/pushobj_adapters/static_dora_r2 \
  --correction-shape dora --no-distill-adapter --rank 2 --epochs 5 --device cuda:0
```

PushT and MediumMaze: as for [`hyper_dora`](../hyper_dora/README.md), with output
directories `static_dora_r2`.
