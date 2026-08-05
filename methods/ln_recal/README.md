# LN-Recal — closed-form output-LayerNorm refit

**What it adapts.** The predictor's final `LayerNorm` (`predictor.transformer.norm`), and
nothing else. Its output is the last thing that happens to every predicted latent:

```
y = LayerNorm(x) * w + b
```

**What it is trying to fix.** The hypothesis that the base world model is *miscalibrated*
rather than wrong under shift — that its one-step predictions land in the right region of
latent space with a systematic per-feature offset and gain error. If that is true, no
gradients are needed: the affine that best maps the normalized predictor features onto the
next latents actually observed this episode is a two-parameter-per-feature least-squares
problem with a closed-form solution.

At every replan, LN-Recal solves

```
min over (gamma_j, beta_j)   sum_i ( gamma_j * n_ij + beta_j - t_ij )^2
                           + lambda * ( (gamma_j - w_j)^2 + (beta_j - b_j)^2 )
```

per feature `j`, where `n_ij` is the normalized pre-affine predictor feature for buffered
transition `i`, `t_ij` is the corresponding *observed* next latent, and
`lambda = ridge * (number of samples)`. It installs `gamma - w`, `beta - b` as a
per-episode delta on the LayerNorm. Only the visual and proprioceptive feature blocks are
fitted; the trailing action block is overwritten from the action encoder at every rollout
step, so it has no target and its delta is held at zero.

## Why it is on the board

Not because it is expected to win. It is the **cheap analytic point on the frontier**, and
the board did not have one: every existing adaptive arm either walks an optimizer
trajectory (AdaJEPA, Restore TTA, PAD) or deploys weights trained offline for the purpose
(HyperJEPA, Static LoRA). LN-Recal does neither, so it measures something the other rows
cannot: **how much of online gradient TTA's gain is available with no gradients, no trained
parameters, and no checkpoint at all.**

Three properties hold by construction rather than by tuning:

| property | why it is structural |
|---|---|
| no accumulation | the fit is recomputed from the frozen affine every replan, from statistics that are provably invariant to the installed correction — the correction acts on the LayerNorm *output*, while the fit reads its *input* and an encoder target |
| batched | every buffered statistic and the emitted delta carry a leading episode axis, so `requires_episode_isolation: false` is correct and a cohort costs one process per shape |
| no artifacts | nothing is loaded from disk, so there is no checkpoint to mis-transcribe and no epoch to select |

The memorylessness claim is tested rather than asserted:
`methods/ln_recal/test_ln_recal_reset.py::test_the_fit_is_memoryless` requires the emitted delta to
be **bit-identical** whether the fit runs from the frozen model or with a different
correction already installed.

## What it reports

Per-replan scalars, all outcome-free:

| metric | meaning |
|---|---|
| `ln_recal/delta_rms_rel` | RMS of the emitted affine delta relative to the pretrained affine. The size of the correction, in units of the thing it perturbs. |
| `ln_recal/fit_residual_ratio` | one-step latent error after the fit over the error before it, on the fitted transitions. ≤ 1 by construction; near 1 means the episode's evidence does not support recalibration at this `ridge`. |
| `ln_recal/delta_dispersion` | how much the emitted delta varies *between* episodes, relative to its own scale. Near 0 would mean the fit is picking up something shared rather than episode-specific. |
| `ln_recal/base_residual` | the one-step latent error being fitted. |
| `ln_recal/samples`, `ln_recal/buffer_transitions` | how much evidence the fit had. |

`fit_residual_ratio` is the one worth watching: it separates "the correction is small
because the model is already calibrated" from "the correction is small because it was
shrunk", which the success rate cannot.

## Hyperparameters and how they were chosen

Declared rule: `selection:RidgeGrid` — 8 columns on the selection cohort,
`ridge ∈ {3, 10, 30, 100} × fit_weight ∈ {false, true}`, scored on paired
`median_distance_delta`. The grid's *bounds* were set from the adapter's own
`delta_rms_rel` diagnostic at a reduced `n_evals=3` on the selection cohort, so that the
axis brackets both the near-frozen end and the end where the closed loop deteriorates;
that reasoning, and the measurements behind it, are in `selection.py`.

`buffer_size: 20` is not swept. One model-resolution transition is executed per replan
(`n_taken_actions // frameskip = 1`) and an episode is at most 20 replans, so a 20-entry
buffer is "the whole episode" — the fit uses all the evidence there is, and a shorter
window would be a second shrinkage knob confounded with `ridge`. It is authored, not
selected, and it is inert in the sense that no reachable value makes the buffer the binding
constraint.

## Result

See the setting sections of [`../../LEADERBOARD.md`](../../LEADERBOARD.md) for the full
frontier. In brief:

| setting | LN-Recal | frozen | catastrophe | compounding |
|---|---|---|---|---|
| `pushobj` | **0.597** | 0.485 | **5.3%** | −0.02 [−0.08, +0.01] |
| `pushobj_shift` | **0.320** | 0.293 | **1.0%** | −0.04 [−0.31, +0.17] |
| `pusht` | 0.340 | 0.350 | 4.9% | −0.04 [−0.11, +0.01] |

Three things are worth saying about that, in decreasing order of how comfortable they are.

**On `pushobj` a gradient-free correction buys most of what gradients buy.** +0.112 over
frozen puts it above Static LoRA and PAD and within one SE of HyperJEPA — with no trained
parameters, no checkpoint, and 8 batched selection columns against HyperJEPA's trained
hypernetwork or AdaJEPA's 16 episode-isolated ones. It is also the *safest* adaptive arm on
the board: the lowest catastrophe rate, the lowest 90th-percentile regret, and a compounding
slope indistinguishable from zero — the memorylessness claim showing up in the metric
designed to detect its absence.

**On the held-out shapes it ties the trained hypernetwork** (0.320 both) at an eighth of its
catastrophe rate (1.0% vs 7.5%) and a fifteenth of the frozen model's memory overhead.
Whatever `pushobj_shift` breaks, the part of it a *recalibration* can fix is apparently the
same part an amortized LoRA generator learns to fix.

**On `pusht` it does nothing** — 0.340 against frozen's 0.350, a difference well inside one
SE. This is the honest limit of the hypothesis: `pusht_visual_shift` is an appearance shift,
and the whole selection sweep there is flat (every cell within ±0.03 of frozen, see
[`SWEEP.md`](SWEEP.md)), so there is no ridge at which the closed-form refit helps. A shift
that changes what the encoder *sees* is not a miscalibration of the predictor's output, and
the method has no way to represent it. HyperJEPA gains +0.090 there; that gap is the part of
adaptation that genuinely requires learning something offline.

The sweep is worth reading for a reason unrelated to this method: on `pushobj`, across eight
cells of one scalar, the catastrophe rate ranges from **1.0% to 31.1%** while success moves
only 0.455 → 0.595. The same mechanism, tuned differently, is either the safest arm on the
board or the most dangerous — the benchmark's thesis with a worked example, and why the
selected cell is reported with its rule attached.

## Limitations, stated plainly

- **One token per frame.** In these bases the encoder emits a single pooled visual token,
  so the fit gets one sample per executed transition: at most 20 per episode, for 2
  parameters per feature. `ridge` is doing real work, not cosmetics.
- **One-step fit, multi-step use.** The affine is fitted on one-step transitions and then
  applied at every step of a 5-step planning rollout, so both the correction *and the
  error in estimating it* accumulate across the horizon. This is a property of any additive
  latent correction, and it is the mechanism that makes an aggressive `ridge` harmful long
  before it makes the one-step fit bad.
- **Peak memory is an implementation choice, not a property of the method.** The reported
  ~890 MB comes from encoding the whole cohort's executed transitions in one forward pass
  inside `on_transition` (50 episodes × one frame through the visual encoder). The
  arithmetic itself is four running sums per feature. Chunking that encode would trade the
  memory back for latency; nothing about the fit requires either.
- **It cannot represent a change of dynamics**, only a recalibration of the output. A shift
  that alters *which* latent direction the object moves in is outside its hypothesis class
  by construction. Comparing it against HyperJEPA on the same LoRA surface is therefore
  informative about which of the two the shift actually is.
