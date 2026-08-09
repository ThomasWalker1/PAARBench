# AdaJEPA — online gradient TTA

Takes gradient steps on the world model's own prediction loss over a buffer of
transitions the controller has executed. The correction is an optimizer trajectory: it
accumulates across replans and is never recomputed from the frozen weights, which is
what the compounding-slope column measures. Rank-2 LoRA plus a LayerNorm delta on the
predictor's last block; nothing offline, no extra checkpoint.

This row is the paired reference several methods are compared against and its record is
not to be re-derived casually — see CONTRIBUTING. **Nothing about the method or its
numbers changed when this file was added.** What follows is provenance that was only in
the adapter's docstring, and it matters for reading
[`methods/adajepa_v2/`](../adajepa_v2/), which is a second version of *this* arm.

## Provenance: what this row is a port of

`method.yaml` names its source honestly — the predecessor project
`~/HyperJEPA`, `planning/adaptation.py` @ `628dff7` — and the port is faithful to it.
That predecessor was itself a re-implementation of
[AdaJEPA](https://github.com/agentic-learning-ai-lab/adajepa), and it simplified the
fitting rule. So the leaderboard label "AdaJEPA" names a family, not a reproduction of
the published implementation. The differences are not small:

| | published `adajepa` (`planning/adajepa.py`) | this row |
|---|---|---|
| fitting window | sliding windows of `min(num_hist, T)` observed frames, teacher-forced (`_prediction_loss`) — 3 frames on these bases | **one** frame: `encode(obs_0, action)` on a single-frame `obs_0`, one `predict`, one target |
| buffer | every executed segment, **merged into one contiguous trajectory** (`_merge_segments`; `replay_buffer=false` is the default) | 5 **isolated** single-step pairs |
| adapted parameters | predictor's last transformer layer **densely**, plus `transformer.norm`, plus the encoder's non-backbone head (`finetune_encoder=True` by default) | rank-2 LoRA + LayerNorm delta on the last block; encoder never touched |
| optimizer | rebuilt **inside every** `finetune()` call, i.e. fresh per replan | one AdamW per episode, moments persisting across replans |
| defaults | `steps=1`, `lr=5e-4`, Adam | `steps=5`, `pred_lr=2e-3` (selected), AdamW, `grad_clip_norm=1.0` |

Two of these are worth being explicit about, because they change what a reader should
conclude from the row:

- **The one-frame window is the largest deviation.** These bases were trained with
  `num_hist: 3`, and the planner's own rollout accumulates context up to 3 frames, so
  the published rule fits the predictor in the regime it is used in and this row does
  not. `methods/adajepa_v2` generalizes the window in the other direction (open-loop
  multi-step); the published rule — 1-step-ahead, teacher-forced over a `num_hist`
  window — sits between the two and **no row on this board measures it**. Any claim of
  the form "multi-step beats single-step for AdaJEPA" is, until that arm is run, a claim
  about this port rather than about the method.
- **Persisting Adam moments across replans** is what makes the correction an optimizer
  *trajectory* in the sense the compounding column describes. Upstream's weights still
  accumulate, but its moment estimates do not.

## A latent alignment hazard (not currently triggered)

`on_transition` truncates to the chunk's first action (`action[:, :1]`) and pairs it
with the chunk's *final* observation. That is exactly correct in every setting on the
board, because `n_taken_actions // frameskip == 1` in all of them (5 // 5), so a chunk
*is* one model step. If a setting ever executed a longer chunk, this would silently
train on a mis-specified transition — one action labelled with the endpoint of several —
rather than fail. Upstream slices the executed chunk at the model stride
(`start = iter_idx * T * fs`, `_extract_adajepa_data`) and handles `T > 1` correctly;
`methods/adajepa_v2` counts the same boundaries backwards from the end of the rollout,
with tests. The adapter's docstring flags the truncation as deliberately preserved
predecessor semantics; this is the note saying what it would cost if a setting changed.

## Reported per replan

`adajepa/loss`, `adajepa/visual_loss`, `adajepa/proprio_loss`, `adajepa/buffer_size`,
`adajepa/pre_update_loss`, `adajepa/refresh_skipped`, `adajepa/refresh_interval`.

## Selection

Two axes (`pred_lr`, `steps`), 9 initial cells plus boundary expansion. Selected
`steps=5` on all three settings it was tuned on, with `pred_lr` 2e-3 on `pushobj`,
1e-2 on `pusht`, 5e-4 on `maze_medium`.

## Note on `wm.train()` during fitting

`step()` puts the world model into train mode for its gradient steps. On these
checkpoints that enables the predictor's dropout (`train_predictor: true` in the base
config), so the fitted gradients carry dropout noise and depend on the global RNG state.
There is no BatchNorm anywhere in these models — normalization is LayerNorm and
GroupNorm — so no running statistics move, and the episode-boundary reset is unaffected.
