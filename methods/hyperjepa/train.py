#!/usr/bin/env python3
"""Train HyperJEPA (or Static LoRA) adapter checkpoints for PAARBench.

Ported from HyperJEPA's ``train_hyper_lora.py``. World-model + offline dataset
loading goes through ``paarbench.offline``; AdaJEPA distillation teachers use
``methods.adajepa.adapter.AdaJEPAAdapter``. Maze recipes pass
``--no-distill-adapter``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import custom_resolvers  # noqa: F401
from models.hyper_lora import (
    HyperAdapterGate,
    apply_hyper_tensors,
    build_generator,
    clear_hyper_tensors,
    full_rank_for_scope,
    install_predictor_hyper_targets,
    lora_parameter_budget,
    module_parameter_count,
    select_predictor_lora_module_names,
)
from models.hyper_context import OrderedTransitionContext, transition_buffer_features
from methods.adajepa.adapter import AdaJEPAAdapter
from paarbench.offline import load_training_bundle
from paarbench.world_model import load_model, load_train_config
from utils import move_to_device, seed


log = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train HyperJEPA LoRA generator.")
    parser.add_argument("--ckpt-dir", required=True)
    parser.add_argument("--model-epoch", default="latest")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument(
        "--min-epochs",
        type=int,
        default=0,
        help="Minimum epochs to run before validation-loss early stopping may trigger.",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=0,
        help=(
            "Stop after this many consecutive validation epochs without an "
            "improvement of at least --early-stop-min-delta. 0 disables early stopping."
        ),
    )
    parser.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=0.0,
        help="Minimum validation-loss decrease counted as an early-stopping improvement.",
    )
    # ---- few-shot transfer (Thread B): finetune a pretrained adapter, or the
    # base predictor itself, on K episodes of a held-out shape at matched compute.
    parser.add_argument(
        "--data-path",
        default=None,
        help="Override the base checkpoint's dataset path (e.g. a held-out-shape pool).",
    )
    parser.add_argument(
        "--n-rollout",
        type=int,
        default=None,
        help="Train on only the first K episodes of the dataset (the few-shot budget).",
    )
    parser.add_argument(
        "--init-adapter",
        default=None,
        help=(
            "Initialize the hypernetwork (or static-LoRA parameters) from a trained "
            "adapter checkpoint instead of from scratch. The few-shot starting point."
        ),
    )
    parser.add_argument(
        "--dense-finetune",
        default="none",
        choices=["none", "predlast_all", "predictor_all"],
        help=(
            "Full-model control: instead of a hypernetwork, directly finetune base "
            "predictor weights. 'predlast_all' matches the modules the LoRA targets; "
            "'predictor_all' finetunes the whole predictor."
        ),
    )
    parser.add_argument(
        "--steps-per-epoch",
        type=int,
        default=None,
        help=(
            "Force each epoch to be exactly this many gradient steps, cycling the "
            "loader as needed. Makes checkpoint spacing (and compute) identical "
            "across few-shot budgets K, which otherwise give very different epochs."
        ),
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Total gradient-step budget; training stops once it is reached.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--context-mode",
        choices=["first_frame", "hist_frames", "transition_buffer", "state_only"],
        default="transition_buffer",
        help=(
            "What the hypernetwork reads. 'transition_buffer' is the method: an "
            "ordered buffer of executed-transition residual/action features. "
            "'state_only' drops the buffer entirely and conditions on the pooled "
            "latent of the current observation -- the ablation that asks whether "
            "the transition evidence is needed at all, given that the mechanism "
            "analysis finds the correction to be state-conditional."
        ),
    )
    parser.add_argument("--context-frames", type=int, default=3)
    parser.add_argument(
        "--context-aggregator",
        choices=["mean", "transformer", "transformer_query"],
        default="transformer_query",
        help="Aggregate transition features by their mean or an ordered transformer.",
    )
    parser.add_argument(
        "--context-feature-kind",
        choices=["residual_action", "latent_residual_action"],
        default="residual_action",
        help="Transition-buffer token contents for HyperJEPA context.",
    )
    parser.add_argument(
        "--context-transitions",
        type=int,
        default=5,
        help=(
            "Number of preceding one-action transitions. If omitted, preserve the "
            "legacy window derived from the world-model training slice."
        ),
    )
    parser.add_argument("--context-model-dim", type=int, default=128)
    parser.add_argument("--context-depth", type=int, default=2)
    parser.add_argument("--context-heads", type=int, default=4)
    parser.add_argument("--context-dropout", type=float, default=0.0)
    parser.add_argument("--adapter-gate", action="store_true")
    parser.add_argument("--gate-hidden-dim", type=int, default=64)
    parser.add_argument("--gate-init-bias", type=float, default=0.0)
    parser.add_argument("--target-scope", default="predlast_all")
    parser.add_argument(
        "--static-lora",
        action="store_true",
        help=(
            "Control: train a single context-independent LoRA (+ final LayerNorm "
            "delta) offline instead of a hypernetwork. No context encoder or "
            "generator is built; the shared LoRA parameters are optimized directly."
        ),
    )
    parser.add_argument(
        "--rank",
        default="4",
        help=(
            "Rank of the generated correction, or 'full' for the smallest rank at "
            "which every targeted matrix's correction is an unconstrained dense "
            "weight delta (404 for this predictor). 'full' requires "
            "--generator chunked; the MLP generator's output layer would be 2.0 B "
            "parameters at that width."
        ),
    )
    parser.add_argument(
        "--generator",
        choices=["mlp", "chunked"],
        default="mlp",
        help=(
            "How the packed correction is produced from the context. 'mlp' is one "
            "linear map onto the whole correction (the method as reported); "
            "'chunked' queries a small shared head once per fixed-size chunk with "
            "a learned chunk embedding, which is what makes high and full rank "
            "affordable."
        ),
    )
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--chunk-embed-dim", type=int, default=64)
    parser.add_argument("--head-hidden-dim", type=int, default=128)
    parser.add_argument("--lora-scale", type=float, default=1.0)
    parser.add_argument(
        "--freeze-lora-a",
        action="store_true",
        help="Use fixed random LoRA A matrices and generate only B matrices.",
    )
    parser.add_argument("--fixed-a-scale", type=float, default=1.0)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--init-scale", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=50)
    parser.add_argument(
        "--prediction-mode",
        choices=["one_step", "rollout", "ranking"],
        default="one_step",
        help=(
            "Phase-two primary objective. 'one_step' is the legacy teacher-forced "
            "loss from wm.forward(). 'rollout' unrolls the corrected predictor "
            "autoregressively with wm.rollout(), matching how the correction is "
            "actually used inside the MPC planner. 'ranking' goes further and "
            "supervises the goal-cost ordering over action sequences, which is the "
            "only thing the planner reads off the model; it is added to the "
            "one-step term rather than replacing it, so the latents stay "
            "calibrated."
        ),
    )
    parser.add_argument(
        "--ranking-weight",
        type=float,
        default=1.0,
        help="Weight on the goal-cost ranking term of --prediction-mode ranking.",
    )
    parser.add_argument(
        "--rollout-frames",
        type=int,
        default=4,
        help=(
            "Number of future frames to predict/supervise in --prediction-mode "
            "rollout. Only used when prediction-mode=rollout."
        ),
    )
    parser.add_argument(
        "--objective",
        choices=["prediction", "relative", "hinge"],
        default="prediction",
        help=(
            "Shape of the primary (teacher-free) loss over the latent prediction "
            "error. 'prediction' is the plain mean squared error the base model "
            "was trained with. 'relative' divides each sample's error by the "
            "frozen model's error on the same sample, so every sample contributes "
            "its fractional improvement instead of its absolute error -- the "
            "offline analogue of an online update, which descends the residual of "
            "whatever episode it is in, at that episode's own scale. 'hinge' keeps "
            "the plain error and adds a penalty for exceeding the frozen model's "
            "error, which supervises the regressions the plain loss is indifferent "
            "to."
        ),
    )
    parser.add_argument(
        "--hinge-weight",
        type=float,
        default=1.0,
        help="Weight on the regression penalty of --objective hinge.",
    )
    parser.add_argument(
        "--prediction-loss-weight",
        type=float,
        default=1.0,
        help=(
            "Weight on the primary prediction term. Set to 0 with "
            "--distill-space weights to train purely on how well the emitted "
            "correction matches the teacher's, which asks whether the "
            "hypernetwork can represent that correction at all, separately from "
            "whether representing it helps."
        ),
    )
    distill_group = parser.add_mutually_exclusive_group()
    distill_group.add_argument("--distill-adapter", dest="distill_adapter", action="store_true")
    distill_group.add_argument("--no-distill-adapter", dest="distill_adapter", action="store_false")
    parser.set_defaults(distill_adapter=True)
    parser.add_argument("--distill-weight", type=float, default=1.0)
    parser.add_argument(
        "--teacher-update-scope",
        default="predlast_encfrozen",
        help=(
            "AdaJEPA update scope used for offline distillation. Use "
            "'lora_predlast_all' for a LoRA-AdaJEPA teacher matched to HyperJEPA."
        ),
    )
    parser.add_argument("--teacher-lora-rank", type=int, default=None)
    parser.add_argument("--teacher-lora-scale", type=float, default=1.0)
    parser.add_argument(
        "--teacher-steps",
        type=int,
        default=1,
        help=(
            "Gradient steps the online teacher takes per sample. The published "
            "teacher takes one; the tuned optimum of the matched rank-2 online arm "
            "is ten, so distilling the configuration that actually wins requires "
            "this. Only the batched LoRA fast path implements more than one step."
        ),
    )
    parser.add_argument(
        "--teacher-lr",
        type=float,
        default=5e-4,
        help="Step size of the online teacher's AdamW update.",
    )
    parser.add_argument(
        "--distill-space",
        choices=["function", "weights", "both"],
        default="function",
        help=(
            "Where the teacher is matched. 'function' matches its predicted latent, "
            "which is what a distillation loss normally means and what leaves the "
            "student free to reach the same predictions by any correction. 'weights' "
            "matches the teacher's correction itself, elementwise on the induced "
            "weight delta rather than on its factors, which are only defined up to "
            "an invertible reparameterization. Weight-space matching removes the "
            "credit-assignment problem in function-space matching and is the "
            "sharpest available test of whether the hypernetwork can represent the "
            "correction online adaptation finds."
        ),
    )
    parser.add_argument(
        "--weight-distill-weight",
        type=float,
        default=1.0,
        help="Weight on the weight-space matching term of --distill-space.",
    )
    parser.add_argument(
        "--no-distill-cache",
        dest="distill_cache",
        action="store_false",
        help="Disable the in-memory cache for independently computed AdaJEPA targets.",
    )
    parser.set_defaults(distill_cache=True)
    parser.add_argument(
        "--distill-cache-dir",
        default=None,
        help=(
            "Optional persistent directory for episode-isolated teacher targets. "
            "Cache keys include teacher scope/rank/scale and the sample tensors."
        ),
    )
    parser.add_argument(
        "--max-train-slices",
        type=int,
        default=None,
        help=(
            "Restrict training to the first N slices of the (deterministically "
            "shuffled) train split. Required to make a precomputed teacher cache "
            "hit: precompute and training must cover the same pool, and the full "
            "split is far too large to precompute a reference teacher over. The "
            "slice permutation is seeded by --seed, so N slices is a reproducible "
            "random subset -- keep --seed identical between the two runs."
        ),
    )
    parser.add_argument(
        "--precompute-teacher-cache",
        action="store_true",
        help=(
            "Compute and store distillation targets for the whole training pool, then "
            "exit without training. This is what makes a NON-PROXY (dense-weight) "
            "AdaJEPA teacher affordable: the reference teacher path costs one "
            "episode-isolated gradient step per sample, but the targets are a pure "
            "function of (sample, teacher config), so they can be built once in "
            "parallel and reused by every subsequent epoch and run."
        ),
    )
    parser.add_argument("--cache-shard", type=int, default=0)
    parser.add_argument(
        "--cache-shards",
        type=int,
        default=1,
        help="Shard the precompute over this many processes (batch index modulo).",
    )
    parser.add_argument(
        "--cache-prefixes",
        default="all",
        help=(
            "Context-suffix lengths to precompute, comma-separated, or 'all'. "
            "Training draws a random suffix length per batch (an episode-prefix "
            "augmentation), so every length the trainer can draw must be cached or "
            "it will fall back to the slow path. Use with --fixed-context-prefix to "
            "cache a single length instead."
        ),
    )
    parser.add_argument(
        "--fixed-context-prefix",
        type=int,
        default=None,
        help=(
            "Disable the random context-suffix augmentation and always use this many "
            "transitions. At eval the buffer is full every MPC step, so a fixed full "
            "prefix is the eval-matched setting and needs 1/K of the teacher cache."
        ),
    )
    parser.add_argument(
        "--teacher-target-mode",
        choices=["auto", "reference", "batched_lora"],
        default="auto",
        help=(
            "How to compute uncached teacher targets. 'batched_lora' is an "
            "episode-isolated fast path for one-step LoRA-AdaJEPA teachers."
        ),
    )
    return parser.parse_args()


def move_batch(obs, act, device):
    return move_to_device(obs, device), act.to(device)


@torch.no_grad()
def encode_context(wm, obs, mode: str, context_frames: int):
    z = wm.encode_obs(obs)
    visual = z["visual"]
    proprio = z["proprio"]
    if mode == "first_frame":
        visual = visual[:, :1]
        proprio = proprio[:, :1]
    elif mode == "hist_frames":
        n = max(1, min(int(context_frames), visual.shape[1]))
        visual = visual[:, :n]
        proprio = proprio[:, :n]
    else:
        raise ValueError(f"Unsupported context mode: {mode}")
    return torch.cat([visual.mean(dim=(1, 2)), proprio.mean(dim=1)], dim=-1)


def transition_context_features(
    wm, obs, act, mode, context_frames, feature_kind="residual_action", current_obs=None
):
    if mode == "transition_buffer":
        return transition_buffer_features(wm, obs, act, feature_kind=feature_kind)
    if mode == "state_only":
        # One token, carrying the same pooled current-observation latent that the
        # transformer_query aggregator uses as its query -- so this arm keeps the
        # state pathway and drops the transition pathway.
        if current_obs is None:
            raise ValueError("context_mode='state_only' needs the current observation")
        return state_query_feature(wm, current_obs).unsqueeze(1)
    return encode_context(wm, obs, mode, context_frames).unsqueeze(1)


@torch.no_grad()
def state_query_feature(wm, obs):
    z = wm.encode_obs(obs)
    return torch.cat([z["visual"].mean(dim=(1, 2)), z["proprio"].mean(dim=1)], dim=-1)


def aggregate_context(features, aggregator, context_encoder=None, query=None):
    if aggregator == "mean":
        return features.mean(dim=1)
    if aggregator in {"transformer", "transformer_query"}:
        if context_encoder is None:
            raise ValueError("Transformer context aggregation requires a context encoder")
        return context_encoder(features, query=query)
    raise ValueError(f"Unknown context aggregator: {aggregator}")


def split_context_and_prediction_window(
    obs, act, model_num_hist, model_num_pred, context_transitions, prediction_frames=None
):
    """Separate preceding adaptation transitions from the base-model loss window.

    ``prediction_frames`` overrides the length of the prediction window. It
    defaults to the legacy ``num_hist + num_pred``; the rollout objective passes
    a longer window (``num_hist + rollout_frames``).
    """
    if context_transitions is None:
        return obs, act, obs, act
    context_transitions = int(context_transitions)
    if prediction_frames is None:
        prediction_frames = int(model_num_hist) + int(model_num_pred)
    prediction_frames = int(prediction_frames)
    expected = context_transitions + prediction_frames
    if obs["visual"].shape[1] < expected or act.shape[1] < expected:
        raise ValueError(
            f"Need at least {expected} frames for {context_transitions} context transitions "
            f"and a {prediction_frames}-frame prediction window"
        )
    context_obs = {key: value[:, : context_transitions + 1] for key, value in obs.items()}
    context_act = act[:, :context_transitions]
    prediction_obs = {key: value[:, context_transitions : context_transitions + prediction_frames] for key, value in obs.items()}
    prediction_act = act[:, context_transitions : context_transitions + prediction_frames]
    return context_obs, context_act, prediction_obs, prediction_act


def rollout_prediction_loss(wm, prediction_obs, prediction_act, model_num_hist):
    """Autoregressive rollout loss matching the deployment (MPC) predictor path.

    The first ``model_num_hist`` frames seed ``wm.rollout``; the corrected
    predictor is then unrolled over the remaining actions and each predicted
    observation latent is compared to the encoded observed future frame. This is
    the objective the correction is actually used under at deployment, unlike the
    teacher-forced one-step ``wm.forward`` loss.
    """
    num_hist = int(model_num_hist)
    obs_0 = {key: value[:, :num_hist] for key, value in prediction_obs.items()}
    z_obses, _ = wm.rollout(obs_0, prediction_act)
    available = prediction_obs["visual"].shape[1] - num_hist
    predicted = z_obses["visual"].shape[1] - num_hist
    future = min(available, predicted)
    if future < 1:
        raise ValueError(
            "Rollout objective produced no supervised future frames "
            f"(available={available}, predicted={predicted})."
        )
    target_obs = {
        key: value[:, num_hist : num_hist + future] for key, value in prediction_obs.items()
    }
    with torch.no_grad():
        z_tgt = wm.encode_obs(target_obs)
    pred_visual = z_obses["visual"][:, num_hist : num_hist + future]
    pred_proprio = z_obses["proprio"][:, num_hist : num_hist + future]
    visual_loss = wm.emb_criterion(pred_visual, z_tgt["visual"])
    proprio_loss = wm.emb_criterion(pred_proprio, z_tgt["proprio"])
    return visual_loss + proprio_loss


RELATIVE_LOSS_EPS = 1e-6


def one_step_prediction_terms(wm, obs, act, latents=None):
    """Return the corrected one-step prediction and its per-sample latent error.

    ``wm.forward`` reports one scalar and folds in terms (curvature, decoder
    reconstruction) that are computed from frozen quantities and therefore
    contribute no gradient to a phase-two correction. Computing the prediction
    term directly gives the same gradient, exposes the per-sample error the
    relative and hinge objectives need, and lets an already-encoded window be
    reused so a reference pass costs one predictor call rather than one encoder
    call.
    """
    if latents is None:
        with torch.no_grad():
            latents = wm.encode(obs, act)
    z_src = latents[:, : wm.num_hist]
    z_tgt = latents[:, wm.num_pred :]
    if wm.stop_grad:
        z_tgt = z_tgt.detach()
    z_pred = wm.predict(z_src)
    if wm.concat_dim == 0:
        pred, target = z_pred[:, :, :-1, :], z_tgt[:, :, :-1, :]
    else:
        pred, target = z_pred[..., : -wm.action_dim], z_tgt[..., : -wm.action_dim]
    per_sample = (pred - target).square().flatten(start_dim=1).mean(dim=1)
    return z_pred, per_sample


def correction_distance(
    student,
    teacher,
    lora_targets,
    norm_targets,
    student_scale=1.0,
    teacher_scale=1.0,
):
    """Mean squared distance between two corrections, in weight space.

    Matching the generated factors ``A`` and ``B`` directly would be wrong: any
    invertible ``M`` gives ``(BM)(M^-1 A) = BA``, so the factors are only defined
    up to a reparameterization and two identical corrections can have arbitrarily
    different factors. The induced weight delta is the invariant, and its squared
    distance expands into traces of small matrices,

        ||B_s A_s - B_t A_t||_F^2 = tr[(B_s^T B_s)(A_s A_s^T)]
                                  - 2 tr[(B_s^T B_t)(A_s A_t^T)]
                                  + tr[(B_t^T B_t)(A_t A_t^T)],

    so no (d_out x d_in) delta is ever materialized -- which is what makes this
    affordable at full rank, where the deltas would be 3.3 M entries per sample.

    Each module's distance is reported *relative* to the teacher's own squared
    norm and then averaged over modules. Without that normalization the term is
    unusable: a one-step update at 5e-4 gives per-entry deltas of order 1e-5, so
    the absolute distance is ~1e-9 against a prediction loss of ~3e-2 and adding
    it changes nothing. Normalized, 1.0 is the distance achieved by emitting no
    correction at all and 0.0 is an exact match, so the value is directly
    readable as the fraction of the teacher's correction left unexplained.
    """
    terms = []
    for target in lora_targets:
        A_s, B_s = student[target.name]
        A_t, B_t = teacher[target.name]
        B_s = B_s * float(student_scale)
        B_t = B_t * float(teacher_scale)
        gram_ss = torch.einsum("bor,bos->brs", B_s, B_s)
        gram_st = torch.einsum("bor,bos->brs", B_s, B_t)
        gram_tt = torch.einsum("bor,bos->brs", B_t, B_t)
        cov_ss = torch.einsum("bri,bsi->brs", A_s, A_s)
        cov_st = torch.einsum("bri,bsi->brs", A_s, A_t)
        cov_tt = torch.einsum("bri,bsi->brs", A_t, A_t)
        teacher_norm = (gram_tt * cov_tt).sum(dim=(1, 2))
        squared = (
            (gram_ss * cov_ss).sum(dim=(1, 2))
            - 2.0 * (gram_st * cov_st).sum(dim=(1, 2))
            + teacher_norm
        )
        terms.append((squared / (teacher_norm.detach() + RELATIVE_LOSS_EPS)).mean())
    for target in norm_targets:
        weight_s, bias_s = student[target.name]
        weight_t, bias_t = teacher[target.name]
        student_affine = torch.cat([weight_s, bias_s], dim=-1)
        teacher_affine = torch.cat([weight_t, bias_t], dim=-1)
        teacher_norm = teacher_affine.square().sum(dim=-1)
        squared = (student_affine - teacher_affine).square().sum(dim=-1)
        terms.append((squared / (teacher_norm.detach() + RELATIVE_LOSS_EPS)).mean())
    if not terms:
        raise ValueError("correction_distance needs at least one target module")
    return torch.stack(terms).mean()


def action_ranking_loss(wm, prediction_obs, prediction_act, model_num_hist):
    """Supervise the goal-cost ordering the planner actually consumes.

    Every other phase-two objective regresses latents, but the planner never
    reads a latent: it reads the goal distance of Equation (mpc), and only that
    distance's *ordering* over candidate action sequences decides which action is
    executed. A correction could reduce latent error everywhere and still reorder
    two candidates wrongly, or raise latent error and rank perfectly.

    This objective takes the window's observed final frame as the goal, rolls the
    corrected predictor out under the actions that demonstrably reached it and
    under another window's actions from the same history, and asks that the first
    score better. The score is the normalized comparison
    ``d+ / (d+ + d-)``, which needs no margin hyperparameter and is invariant to
    the latent scale: ``0.5`` is no discrimination and ``0`` is perfect.
    """
    num_hist = int(model_num_hist)
    obs_0 = {key: value[:, :num_hist] for key, value in prediction_obs.items()}
    with torch.no_grad():
        z_goal = wm.encode_obs({key: value[:, -1:] for key, value in prediction_obs.items()})

    def terminal_distance(actions):
        z_obses, _ = wm.rollout(obs_0, actions)
        return sum(
            _per_sample_mse(z_obses[key][:, -1:], z_goal[key])
            for key in ("visual", "proprio")
        )

    # The counterfactual negates the *future* actions only. ``wm.rollout`` pairs
    # the first num_hist actions with the observed history frames, so leaving
    # those alone makes both rollouts start from an identical initial latent and
    # differ only in the actions being ranked. Negation rather than a
    # batch-derangement negative because the validation loader is deliberately
    # unshuffled: there, neighbouring batch entries are neighbouring windows of
    # the same episode, so a cross-batch counterfactual degenerates to the real
    # one and the ratio collapses to 0.5 for reasons that have nothing to do with
    # the model. Negation is within-sample, never degenerate, and in
    # distribution, since the pool contains motion in every direction.
    counterfactual = torch.cat(
        [prediction_act[:, :num_hist], -prediction_act[:, num_hist:]], dim=1
    )
    positive = terminal_distance(prediction_act)
    negative = terminal_distance(counterfactual)
    return (positive / (positive + negative + RELATIVE_LOSS_EPS)).mean()


class TensorPreprocessor:
    """Dataset tensors are already transformed; AdaJEPA only needs identity preprocessing."""

    @staticmethod
    def transform_obs(obs):
        return obs


def transition_items(obs, act):
    items = []
    for idx in range(obs["visual"].shape[1] - 1):
        obs_0 = {key: value[:, idx : idx + 1] for key, value in obs.items()}
        obs_1 = {key: value[:, idx + 1 : idx + 2] for key, value in obs.items()}
        items.append((obs_0, act[:, idx : idx + 1], obs_1))
    return items


def slice_batch(obs, act, idx):
    return {key: value[idx : idx + 1] for key, value in obs.items()}, act[idx : idx + 1]


def suffix_context_window(obs, act, num_transitions):
    num_transitions = int(num_transitions)
    if num_transitions < 1 or num_transitions > act.shape[1]:
        raise ValueError(
            f"Expected 1 <= num_transitions <= {act.shape[1]}, got {num_transitions}"
        )
    return (
        {key: value[:, -(num_transitions + 1) :] for key, value in obs.items()},
        act[:, -num_transitions:],
    )


def _add_tensor_to_digest(digest, name, tensor):
    tensor = tensor.detach().contiguous().cpu()
    digest.update(name.encode("utf-8"))
    digest.update(str(tuple(tensor.shape)).encode("utf-8"))
    digest.update(str(tensor.dtype).encode("utf-8"))
    digest.update(tensor.numpy().tobytes())


def teacher_cache_namespace(
    teacher_update_scope="predlast_encfrozen",
    teacher_lora_rank=4,
    teacher_lora_scale=1.0,
    teacher_steps=1,
    teacher_lr=5e-4,
):
    namespace = {
        "version": 2,
        "teacher_update_scope": str(teacher_update_scope),
        "teacher_lora_rank": int(teacher_lora_rank),
        "teacher_lora_scale": float(teacher_lora_scale),
        "lora_init": "normal_cpu_seeded_A_zero_B_v1",
        "teacher_steps": int(teacher_steps),
        "teacher_optimizer": "adamw_first_step_default",
    }
    # Keep the version-2 key set byte-identical for the published one-step
    # teacher at its published step size, so the existing caches still hit.
    if int(teacher_steps) != 1 or float(teacher_lr) != 5e-4:
        namespace["teacher_lr"] = float(teacher_lr)
    return namespace


def teacher_cache_key(context_obs, context_act, prediction_obs, prediction_act, namespace=None):
    digest = hashlib.sha256()
    if namespace is not None:
        digest.update(json.dumps(namespace, sort_keys=True).encode("utf-8"))
    for prefix, obs in (("context", context_obs), ("prediction", prediction_obs)):
        for key in sorted(obs):
            _add_tensor_to_digest(digest, f"{prefix}.{key}", obs[key])
    _add_tensor_to_digest(digest, "context.action", context_act)
    _add_tensor_to_digest(digest, "prediction.action", prediction_act)
    return digest.hexdigest()


def teacher_seed_from_key(cache_key):
    return int(cache_key[:16], 16) % (2**31)


class TeacherTargetCache:
    def __init__(self, root=None):
        self.root = Path(root).resolve() if root is not None else None
        self.memory = {}
        self.hits = 0
        self.misses = 0
        self.disk_hits = 0
        self.writes = 0
        if self.root is not None:
            self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key):
        if self.root is None:
            return None
        return self.root / key[:2] / f"{key}.pt"

    def get(self, key):
        cached = self.memory.get(key)
        if cached is not None:
            self.hits += 1
            return cached
        path = self._path(key)
        if path is not None and path.exists():
            cached = torch.load(path, map_location="cpu")
            self.memory[key] = cached
            self.hits += 1
            self.disk_hits += 1
            return cached
        self.misses += 1
        return None

    def __setitem__(self, key, value):
        tensor = value.detach().cpu()
        self.memory[key] = tensor
        path = self._path(key)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(f".{os.getpid()}.tmp")
            torch.save(tensor, tmp)
            os.replace(tmp, path)
            self.writes += 1

    def stats(self):
        return {
            "entries": len(self.memory),
            "hits": self.hits,
            "misses": self.misses,
            "disk_hits": self.disk_hits,
            "writes": self.writes,
            "root": str(self.root) if self.root is not None else None,
        }


def _cache_get(target_cache, key):
    if target_cache is None:
        return None
    return target_cache.get(key)


def _cache_set(target_cache, key, value):
    if target_cache is None:
        return
    target_cache[key] = value


def _single_teacher_prediction(
    teacher,
    teacher_initial,
    context_obs,
    context_act,
    prediction_obs,
    prediction_act,
    teacher_update_scope="predlast_encfrozen",
    teacher_lora_rank=4,
    teacher_lora_scale=1.0,
    teacher_lora_init_seed=None,
):
    """One predictor-only AdaJEPA update for one episode-local sample."""
    if context_act.shape[0] != 1:
        raise ValueError("_single_teacher_prediction expects batch size 1")
    with torch.no_grad():
        for name, parameter in teacher.predictor.named_parameters():
            initial = teacher_initial.get(name)
            if initial is not None:
                parameter.copy_(initial)
    rng_devices = [context_act.device] if context_act.device.type == "cuda" else []
    with torch.random.fork_rng(devices=rng_devices):
        if teacher_lora_init_seed is not None:
            torch.manual_seed(int(teacher_lora_init_seed))
        adapter = AdaJEPAAdapter(
            wm=teacher,
            preprocessor=TensorPreprocessor(),
            update_scope=teacher_update_scope,
            lora_rank=teacher_lora_rank,
            lora_scale=teacher_lora_scale,
            lora_init_seed=teacher_lora_init_seed,
            steps=1,
            buffer_size=5,
            buffer_strategy="recent",
        )
    for obs_0, action, obs_1 in transition_items(context_obs, context_act):
        adapter.append(obs_0, action, obs_1)
    adapter.step()
    with torch.no_grad():
        prediction = teacher(prediction_obs, prediction_act)[0].detach()
    return prediction


def _per_sample_mse(pred, target):
    return (pred - target).square().flatten(start_dim=1).mean(dim=1)


class _BatchedAdamW:
    """AdamW over plain tensors, so a whole batch of teachers can step at once.

    ``torch.optim.AdamW`` owns its parameters, which the batched teacher path
    cannot use: every batch element carries its own correction inside one
    stacked tensor and the tensors are rebuilt each batch. The update here is the
    same one ``AdaJEPAAdapter`` applies (decoupled weight decay, default betas),
    and at ``t = 1`` it collapses to ``p (1 - lr w) - lr g / (|g| + eps)``, the
    closed form the one-step path used before -- so previously cached one-step
    targets stay valid.
    """

    def __init__(self, lr=5e-4, betas=(0.9, 0.999), weight_decay=0.01, eps=1e-8):
        self.lr = float(lr)
        self.beta1, self.beta2 = (float(b) for b in betas)
        self.weight_decay = float(weight_decay)
        self.eps = float(eps)
        self.step_count = 0
        self._m = None
        self._v = None

    def step(self, params, grads):
        if self._m is None:
            self._m = [torch.zeros_like(p) for p in params]
            self._v = [torch.zeros_like(p) for p in params]
        self.step_count += 1
        bias1 = 1.0 - self.beta1**self.step_count
        bias2 = 1.0 - self.beta2**self.step_count
        updated = []
        for idx, (param, grad) in enumerate(zip(params, grads)):
            self._m[idx] = self.beta1 * self._m[idx] + (1.0 - self.beta1) * grad
            self._v[idx] = self.beta2 * self._v[idx] + (1.0 - self.beta2) * grad.square()
            denom = (self._v[idx] / bias2).sqrt() + self.eps
            step = (self._m[idx] / bias1) / denom
            updated.append(param * (1.0 - self.lr * self.weight_decay) - self.lr * step)
        return updated


def _batched_lora_teacher(
    teacher,
    teacher_initial,
    context_obs,
    context_act,
    prediction_obs,
    prediction_act,
    cache_keys,
    teacher_update_scope="lora_predlast_all",
    teacher_lora_rank=4,
    teacher_lora_scale=1.0,
    teacher_steps=1,
    teacher_lr=5e-4,
    return_correction=False,
):
    """Episode-isolated LoRA-AdaJEPA teacher for a whole batch.

    The LoRA modules support batched A/B tensors.  Since each batch element
    only reads its own adapter slice, differentiating a sum of per-sample
    losses gives independent per-sample gradients without mixing episodes.
    Returns the teacher's prediction and, when asked, the correction it reached,
    which is what a weight-space distillation loss is matched against.
    """
    if not teacher_update_scope.startswith("lora_"):
        raise ValueError("Batched teacher fast path is only valid for LoRA-AdaJEPA scopes.")
    batch = context_act.shape[0]
    if len(cache_keys) != batch:
        raise ValueError("Expected one cache key per fast-path teacher sample.")
    with torch.no_grad():
        for name, parameter in teacher.predictor.named_parameters():
            initial = teacher_initial.get(name)
            if initial is not None:
                parameter.copy_(initial)

    target_scope = teacher_update_scope.removeprefix("lora_")
    lora_targets, norm_targets = install_predictor_hyper_targets(
        teacher.predictor,
        scope=target_scope,
        rank=teacher_lora_rank,
        scale=teacher_lora_scale,
    )
    device = context_act.device
    sample_generators = []
    for key in cache_keys:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(teacher_seed_from_key(key))
        sample_generators.append(generator)

    params = []
    for target in lora_targets:
        A_values = []
        for generator in sample_generators:
            tensor = torch.empty(target.spec.rank, target.spec.in_features, device="cpu")
            torch.nn.init.normal_(
                tensor,
                mean=0.0,
                std=1.0 / max(target.spec.in_features, 1) ** 0.5,
                generator=generator,
            )
            A_values.append(tensor)
        A = torch.stack(A_values, dim=0).to(device).requires_grad_(True)
        B = torch.zeros(
            batch,
            target.spec.out_features,
            target.spec.rank,
            device=device,
            requires_grad=True,
        )
        target.module.set_lora(A, B)
        params.extend([A, B])

    for target in norm_targets:
        delta_weight = torch.zeros(batch, target.spec.features, device=device, requires_grad=True)
        delta_bias = torch.zeros(batch, target.spec.features, device=device, requires_grad=True)
        target.module.set_affine(delta_weight, delta_bias)
        params.extend([delta_weight, delta_bias])

    was_training = teacher.training
    teacher.train()
    optimizer = _BatchedAdamW(lr=teacher_lr)
    transitions = transition_items(context_obs, context_act)
    for _ in range(max(int(teacher_steps), 1)):
        losses = []
        for obs_0, action, obs_1 in transitions:
            z_src = teacher.encode(obs_0, action)
            z_pred = teacher.predict(z_src)
            z_pred_obs, _ = teacher.separate_emb(z_pred)
            with torch.no_grad():
                z_tgt_obs = teacher.encode_obs(obs_1)
            visual_loss = _per_sample_mse(
                z_pred_obs["visual"][:, -1:],
                z_tgt_obs["visual"][:, -1:],
            )
            proprio_loss = _per_sample_mse(
                z_pred_obs["proprio"][:, -1:],
                z_tgt_obs["proprio"][:, -1:],
            )
            losses.append(visual_loss + proprio_loss)
        per_sample_loss = torch.stack(losses, dim=0).mean(dim=0)
        grads = torch.autograd.grad(per_sample_loss.sum(), params)
        with torch.no_grad():
            stepped = optimizer.step(params, grads)
        # The next step differentiates the loss at the updated correction, so the
        # stepped tensors become the new leaves of the teacher's graph.
        params = [tensor.detach().requires_grad_(True) for tensor in stepped]
        offset = 0
        for target in lora_targets:
            target.module.set_lora(params[offset], params[offset + 1])
            offset += 2
        for target in norm_targets:
            target.module.set_affine(params[offset], params[offset + 1])
            offset += 2

    correction = None
    with torch.no_grad():
        if return_correction:
            correction = {}
            offset = 0
            for target in lora_targets:
                correction[target.name] = (
                    params[offset].detach().clone(),
                    params[offset + 1].detach().clone(),
                )
                offset += 2
            for target in norm_targets:
                correction[target.name] = (
                    params[offset].detach().clone(),
                    params[offset + 1].detach().clone(),
                )
                offset += 2
        prediction = teacher(prediction_obs, prediction_act)[0].detach()
    teacher.train(was_training)
    return prediction, correction


def teacher_prediction(
    teacher,
    teacher_initial,
    context_obs,
    context_act,
    prediction_obs,
    prediction_act,
    target_cache=None,
    teacher_update_scope="predlast_encfrozen",
    teacher_lora_rank=4,
    teacher_lora_scale=1.0,
    teacher_target_mode="auto",
    teacher_steps=1,
    teacher_lr=5e-4,
    need_correction=False,
):
    """Episode-isolated AdaJEPA distillation targets for an offline batch.

    Offline batches contain unrelated trajectory slices, so each element must
    receive a fresh base teacher, optimizer, and transition buffer.  This
    reference path intentionally loops over samples; cached targets or a
    functional equivalent can replace it later without changing semantics.

    Returns ``(prediction, correction)``; ``correction`` is ``None`` unless
    ``need_correction``, which weight-space distillation sets. The cache stores
    predictions only, so asking for the correction bypasses it -- rather than
    silently returning a correction for some samples and not others.
    """
    namespace = teacher_cache_namespace(
        teacher_update_scope=teacher_update_scope,
        teacher_lora_rank=teacher_lora_rank,
        teacher_lora_scale=teacher_lora_scale,
        teacher_steps=teacher_steps,
        teacher_lr=teacher_lr,
    )
    if need_correction:
        if not teacher_update_scope.startswith("lora_"):
            raise ValueError("Weight-space distillation requires a lora_* teacher scope.")
        return _batched_lora_teacher(
            teacher,
            teacher_initial,
            context_obs,
            context_act,
            prediction_obs,
            prediction_act,
            [
                teacher_cache_key(
                    *slice_batch(context_obs, context_act, idx),
                    *slice_batch(prediction_obs, prediction_act, idx),
                    namespace=namespace,
                )
                for idx in range(context_act.shape[0])
            ],
            teacher_update_scope=teacher_update_scope,
            teacher_lora_rank=teacher_lora_rank,
            teacher_lora_scale=teacher_lora_scale,
            teacher_steps=teacher_steps,
            teacher_lr=teacher_lr,
            return_correction=True,
        )
    predictions = [None] * context_act.shape[0]
    misses = []
    miss_keys = []
    for idx in range(context_act.shape[0]):
        sample_context_obs, sample_context_act = slice_batch(context_obs, context_act, idx)
        sample_prediction_obs, sample_prediction_act = slice_batch(prediction_obs, prediction_act, idx)
        cache_key = teacher_cache_key(
            sample_context_obs,
            sample_context_act,
            sample_prediction_obs,
            sample_prediction_act,
            namespace=namespace,
        )
        if target_cache is not None:
            cached = _cache_get(target_cache, cache_key)
            if cached is not None:
                predictions[idx] = cached.to(context_act.device)
                continue
        misses.append(idx)
        miss_keys.append(cache_key)

    if misses:
        use_batched_lora = teacher_update_scope.startswith("lora_") and teacher_target_mode in {
            "auto",
            "batched_lora",
        }
        if teacher_target_mode == "batched_lora" and not teacher_update_scope.startswith("lora_"):
            raise ValueError("--teacher-target-mode=batched_lora requires a lora_* teacher scope.")
        if use_batched_lora:
            miss_context_obs = {key: value[misses] for key, value in context_obs.items()}
            miss_context_act = context_act[misses]
            miss_prediction_obs = {key: value[misses] for key, value in prediction_obs.items()}
            miss_prediction_act = prediction_act[misses]
            miss_prediction, _ = _batched_lora_teacher(
                teacher,
                teacher_initial,
                miss_context_obs,
                miss_context_act,
                miss_prediction_obs,
                miss_prediction_act,
                miss_keys,
                teacher_update_scope=teacher_update_scope,
                teacher_lora_rank=teacher_lora_rank,
                teacher_lora_scale=teacher_lora_scale,
                teacher_steps=teacher_steps,
                teacher_lr=teacher_lr,
            )
            for local_idx, batch_idx in enumerate(misses):
                prediction = miss_prediction[local_idx : local_idx + 1]
                _cache_set(target_cache, miss_keys[local_idx], prediction)
                predictions[batch_idx] = prediction
        else:
            for batch_idx, cache_key in zip(misses, miss_keys):
                sample_context_obs, sample_context_act = slice_batch(context_obs, context_act, batch_idx)
                sample_prediction_obs, sample_prediction_act = slice_batch(prediction_obs, prediction_act, batch_idx)
                prediction = _single_teacher_prediction(
                    teacher,
                    teacher_initial,
                    sample_context_obs,
                    sample_context_act,
                    sample_prediction_obs,
                    sample_prediction_act,
                    teacher_update_scope=teacher_update_scope,
                    teacher_lora_rank=teacher_lora_rank,
                    teacher_lora_scale=teacher_lora_scale,
                    teacher_lora_init_seed=teacher_seed_from_key(cache_key),
                )
                _cache_set(target_cache, cache_key, prediction)
                predictions[batch_idx] = prediction
    if any(prediction is None for prediction in predictions):
        raise RuntimeError("Internal error: failed to compute every teacher prediction.")
    return torch.cat(predictions, dim=0), None


def precompute_teacher_cache(
    teacher,
    teacher_initial,
    dataset,
    distill_cache,
    device,
    model_cfg,
    args,
    teacher_lora_rank,
):
    """Fill the distillation-target cache for a training pool, then return.

    The reference (non-proxy) teacher runs one episode-isolated AdaJEPA gradient
    step per sample, which is far too slow to redo every epoch.  Targets are a
    deterministic function of (sample tensors, teacher config), so this walks the
    pool once with a *fixed order* and writes every target to the shared on-disk
    cache; training then reads instead of recomputing.  Shards are disjoint by
    batch index, so N processes on N GPUs cover the pool cooperatively.

    Training samples a random context-suffix length per batch, so unless
    ``--fixed-context-prefix`` pins it, every reachable length must be cached or
    training silently falls back to the slow path.
    """
    if args.cache_prefixes == "all":
        prefixes = list(range(1, int(args.context_transitions) + 1))
    else:
        prefixes = [int(tok) for tok in args.cache_prefixes.split(",") if tok.strip()]
    if args.fixed_context_prefix is not None:
        prefixes = [int(args.fixed_context_prefix)]

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,  # deterministic coverage; shards must not overlap
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
    )
    namespace = teacher_cache_namespace(
        teacher_update_scope=args.teacher_update_scope,
        teacher_lora_rank=teacher_lora_rank,
        teacher_lora_scale=args.teacher_lora_scale,
        teacher_steps=args.teacher_steps,
        teacher_lr=args.teacher_lr,
    )
    print(
        f"[precompute] shard {args.cache_shard}/{args.cache_shards} "
        f"prefixes={prefixes} batch={args.batch_size} pool={len(dataset)} "
        f"namespace={namespace}",
        flush=True,
    )
    started = time.time()
    done_batches = 0
    samples = 0
    for batch_idx, (obs, act, _state) in enumerate(loader):
        if batch_idx % int(args.cache_shards) != int(args.cache_shard):
            continue
        if args.max_train_batches is not None and done_batches >= int(args.max_train_batches):
            break
        obs, act = move_batch(obs, act, device)
        context_obs, context_act, prediction_obs, prediction_act = (
            split_context_and_prediction_window(
                obs,
                act,
                model_num_hist=model_cfg.num_hist,
                model_num_pred=model_cfg.num_pred,
                context_transitions=args.context_transitions,
            )
        )
        for prefix in prefixes:
            target_context_obs, target_context_act = suffix_context_window(
                context_obs, context_act, prefix
            )
            teacher_prediction(
                teacher,
                teacher_initial,
                target_context_obs,
                target_context_act,
                prediction_obs,
                prediction_act,
                target_cache=distill_cache,
                teacher_update_scope=args.teacher_update_scope,
                teacher_lora_rank=teacher_lora_rank,
                teacher_lora_scale=args.teacher_lora_scale,
                teacher_target_mode=args.teacher_target_mode,
                teacher_steps=args.teacher_steps,
                teacher_lr=args.teacher_lr,
            )
            samples += act.shape[0]
        done_batches += 1
        if done_batches % 10 == 0:
            elapsed = time.time() - started
            print(
                f"[precompute] shard {args.cache_shard}: {done_batches} batches, "
                f"{samples} targets, {elapsed / max(samples, 1):.3f} s/target, "
                f"{elapsed / 60.0:.1f} min elapsed, {distill_cache.stats()}",
                flush=True,
            )
    elapsed = time.time() - started
    print(
        f"[precompute] DONE shard {args.cache_shard}: {done_batches} batches, "
        f"{samples} targets in {elapsed / 60.0:.1f} min "
        f"({elapsed / max(samples, 1):.3f} s/target), {distill_cache.stats()}",
        flush=True,
    )


def cycle_batches(dataloader, n_batches):
    """Yield exactly ``n_batches`` batches, restarting the loader when exhausted."""
    seen = 0
    while seen < n_batches:
        for batch in dataloader:
            yield batch
            seen += 1
            if seen >= n_batches:
                return


def run_epoch(
    wm,
    generator,
    lora_targets,
    norm_targets,
    dataloader,
    optimizer,
    device,
    context_mode,
    context_frames,
    context_aggregator,
    context_encoder,
    adapter_gate,
    model_num_hist,
    model_num_pred,
    context_transitions,
    teacher=None,
    teacher_initial=None,
    distill_weight=0.0,
    distill_cache=None,
    context_feature_kind="residual_action",
    teacher_update_scope="predlast_encfrozen",
    teacher_lora_rank=4,
    teacher_lora_scale=1.0,
    teacher_target_mode="auto",
    max_batches=None,
    prediction_mode="one_step",
    rollout_frames=None,
    ranking_weight=1.0,
    static_lora=False,
    train=True,
    steps_per_epoch=None,
    step_state=None,
    max_total_steps=None,
    fixed_context_prefix=None,
    objective="prediction",
    hinge_weight=1.0,
    prediction_loss_weight=1.0,
    teacher_steps=1,
    teacher_lr=5e-4,
    distill_space="function",
    weight_distill_weight=1.0,
    lora_scale=1.0,
):
    prediction_frames = None
    if prediction_mode in {"rollout", "ranking"}:
        prediction_frames = int(model_num_hist) + int(rollout_frames)
    wm.eval()
    if generator is not None:
        generator.train(train)
    if context_encoder is not None:
        context_encoder.train(train)
    if adapter_gate is not None:
        adapter_gate.train(train)
    losses = []
    if train and steps_per_epoch is not None:
        # A few-shot budget of K episodes yields far fewer slices than the full
        # dataset, so a fixed step count per epoch (cycling the loader) is what
        # keeps compute -- and checkpoint spacing -- comparable across K.
        budget = int(steps_per_epoch)
        if max_total_steps is not None and step_state is not None:
            budget = min(budget, int(max_total_steps) - int(step_state["steps"]))
        iterator = tqdm(cycle_batches(dataloader, budget), total=max(budget, 0), desc="train")
    else:
        iterator = tqdm(dataloader, desc="train" if train else "valid")
    for batch_idx, (obs, act, _state) in enumerate(iterator):
        if max_batches is not None and batch_idx >= max_batches:
            break
        if (
            train
            and max_total_steps is not None
            and step_state is not None
            and int(step_state["steps"]) >= int(max_total_steps)
        ):
            break
        obs, act = move_batch(obs, act, device)
        context_obs, context_act, prediction_obs, prediction_act = split_context_and_prediction_window(
            obs,
            act,
            model_num_hist=model_num_hist,
            model_num_pred=model_num_pred,
            context_transitions=context_transitions,
            prediction_frames=prediction_frames,
        )
        # Static-LoRA control: a single context-independent correction is held on
        # the targets as persistent trainable parameters, so there is no context
        # to build, no generator to query, and nothing to clear per batch.
        target_context_obs = context_obs
        target_context_act = context_act
        frozen_error = None
        prediction_latents = None
        if not static_lora:
            with torch.no_grad():
                # Offline batches are unrelated slices; derive a stable frozen-model
                # residual rather than carrying the prior batch's generated adapter.
                clear_hyper_tensors(lora_targets, norm_targets)
                context_features = transition_context_features(
                    wm,
                    context_obs,
                    context_act,
                    context_mode,
                    context_frames,
                    feature_kind=context_feature_kind,
                    current_obs={key: value[:, :1] for key, value in prediction_obs.items()},
                )
                # The relative and hinge objectives are both defined against the
                # uncorrected model's error on the same sample, and this is the
                # only point in the batch where the correction is off. The encoded
                # window is kept and reused, so measuring the reference costs one
                # predictor pass rather than a second encoder pass.
                if objective in {"relative", "hinge"} and prediction_mode == "one_step":
                    prediction_latents = wm.encode(prediction_obs, prediction_act)
                    _, frozen_error = one_step_prediction_terms(
                        wm, prediction_obs, prediction_act, latents=prediction_latents
                    )
            # Episode prefixes at deployment contain between one and K transitions.
            # Random suffix lengths make this explicit during phase-two training.
            if fixed_context_prefix is not None and context_features.shape[1] > 1:
                prefix = max(1, min(int(fixed_context_prefix), context_features.shape[1]))
                context_features = context_features[:, -prefix:]
                target_context_obs, target_context_act = suffix_context_window(
                    context_obs,
                    context_act,
                    prefix,
                )
            elif train and context_aggregator in {"transformer", "transformer_query"} and context_features.shape[1] > 1:
                prefix = int(torch.randint(1, context_features.shape[1] + 1, ()).item())
                context_features = context_features[:, -prefix:]
                target_context_obs, target_context_act = suffix_context_window(
                    context_obs,
                    context_act,
                    prefix,
                )
            query = None
            if context_aggregator == "transformer_query":
                with torch.no_grad():
                    current_obs = {key: value[:, :1] for key, value in prediction_obs.items()}
                    query = state_query_feature(wm, current_obs)
            context = aggregate_context(context_features, context_aggregator, context_encoder, query=query)
            tensors = generator(context)
            gate = adapter_gate(context) if adapter_gate is not None else None
            apply_hyper_tensors(lora_targets, norm_targets, tensors, gate=gate)

        if train:
            optimizer.zero_grad(set_to_none=True)
        # The distillation teacher is a one-step predictor; its prediction window
        # is the first (num_hist + num_pred) frames of the (possibly longer)
        # rollout window, which share the same history and actions.
        one_step_frames = int(model_num_hist) + int(model_num_pred)
        if prediction_mode in {"rollout", "ranking"}:
            one_step_obs = {key: value[:, :one_step_frames] for key, value in prediction_obs.items()}
            one_step_act = prediction_act[:, :one_step_frames]
            if prediction_mode == "rollout":
                loss = rollout_prediction_loss(wm, prediction_obs, prediction_act, model_num_hist)
                z_pred = None
            else:
                # The ranking term alone constrains only an ordering, which leaves
                # the latents free to drift anywhere consistent with it, so it is
                # added to the one-step error rather than replacing it.
                z_pred, per_sample_error = one_step_prediction_terms(
                    wm, one_step_obs, one_step_act
                )
                loss = per_sample_error.mean() + float(ranking_weight) * action_ranking_loss(
                    wm, prediction_obs, prediction_act, model_num_hist
                )
        else:
            z_pred, per_sample_error = one_step_prediction_terms(
                wm, prediction_obs, prediction_act, latents=prediction_latents
            )
            if objective == "prediction" or frozen_error is None:
                loss = per_sample_error.mean()
            elif objective == "relative":
                # Every sample contributes its fractional improvement, so the
                # objective is not dominated by whichever slices the frozen model
                # already predicts worst.
                loss = (per_sample_error / (frozen_error + RELATIVE_LOSS_EPS)).mean()
            else:
                loss = per_sample_error.mean() + float(hinge_weight) * torch.clamp(
                    per_sample_error - frozen_error, min=0.0
                ).mean()
            loss = float(prediction_loss_weight) * loss
            one_step_obs = prediction_obs
            one_step_act = prediction_act
        if train and teacher is not None:
            if z_pred is None:
                z_pred = one_step_prediction_terms(wm, one_step_obs, one_step_act)[0]
            target_prediction, target_correction = teacher_prediction(
                teacher,
                teacher_initial,
                target_context_obs,
                target_context_act,
                one_step_obs,
                one_step_act,
                target_cache=distill_cache,
                teacher_update_scope=teacher_update_scope,
                teacher_lora_rank=teacher_lora_rank,
                teacher_lora_scale=teacher_lora_scale,
                teacher_target_mode=teacher_target_mode,
                teacher_steps=teacher_steps,
                teacher_lr=teacher_lr,
                need_correction=distill_space in {"weights", "both"},
            )
            if distill_space in {"function", "both"}:
                loss = loss + float(distill_weight) * torch.nn.functional.mse_loss(
                    z_pred, target_prediction
                )
            if distill_space in {"weights", "both"}:
                loss = loss + float(weight_distill_weight) * correction_distance(
                    tensors,
                    target_correction,
                    lora_targets,
                    norm_targets,
                    student_scale=lora_scale,
                    teacher_scale=teacher_lora_scale,
                )
        if train:
            loss.backward()
            optimizer.step()
            if step_state is not None:
                step_state["steps"] = int(step_state["steps"]) + 1
        losses.append(float(loss.detach().cpu()))
        iterator.set_postfix(loss=sum(losses) / len(losses))
    return sum(losses) / max(len(losses), 1)


def main():
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(levelname)s] %(message)s")
    args = parse_args()
    seed(args.seed)

    ckpt_dir = Path(args.ckpt_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Peek at hydra.yaml only for hist/pred geometry; dataset + WM load via offline.
    model_cfg = load_train_config(ckpt_dir)
    device = torch.device(args.device)
    if args.context_transitions is not None and args.context_transitions < 1:
        raise ValueError("--context-transitions must be positive")
    dataset_num_hist = (
        model_cfg.num_hist
        if args.context_transitions is None
        else int(args.context_transitions) + model_cfg.num_hist
    )
    if args.prediction_mode in {"rollout", "ranking"}:
        if args.context_transitions is None:
            raise ValueError(f"--prediction-mode {args.prediction_mode} requires --context-transitions")
        if int(args.rollout_frames) < int(model_cfg.num_pred):
            raise ValueError("--rollout-frames must be >= model num_pred")
        # Reserve the extra observed future frames the rollout objective supervises.
        dataset_num_hist += int(args.rollout_frames) - int(model_cfg.num_pred)

    bundle = load_training_bundle(
        ckpt_dir,
        model_epoch=args.model_epoch,
        data_path=args.data_path,
        n_rollout=args.n_rollout,
        num_hist=int(dataset_num_hist),
        num_pred=int(model_cfg.num_pred),
        frameskip=int(model_cfg.frameskip),
        device=str(device),
        freeze=True,
    )
    model_cfg = bundle.train_cfg
    datasets = bundle.datasets
    wm = bundle.wm
    model_ckpt = bundle.model_ckpt

    dataloader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    if args.num_workers > 0:
        dataloader_kwargs["prefetch_factor"] = args.prefetch_factor
    if args.max_train_slices is not None:
        n_pool = min(int(args.max_train_slices), len(datasets["train"]))
        datasets["train"] = torch.utils.data.Subset(datasets["train"], range(n_pool))
        log.info(
            "Restricted train pool to %s of %s slices (seed %s)",
            n_pool,
            n_pool if not hasattr(datasets["train"], "dataset") else len(datasets["train"].dataset),
            args.seed,
        )
    loaders = {}
    for split, dset in datasets.items():
        if split not in {"train", "valid"}:
            continue
        # When max_train_batches is used, a new random order each epoch is
        # essential: otherwise every epoch repeats the same prefix of slices.
        loaders[split] = torch.utils.data.DataLoader(
            dset,
            shuffle=(split == "train"),
            **dataloader_kwargs,
        )

    # 'full' is resolved against the predictor, so a scope change cannot silently
    # leave some matrix's correction rank-restricted.
    if str(args.rank).lower() == "full":
        args.rank = full_rank_for_scope(wm.predictor, args.target_scope)
        # The static control optimizes the correction directly, so it has no
        # generator to size; only a hypernetwork is constrained here.
        if args.generator != "chunked" and not args.static_lora:
            raise ValueError(
                f"--rank full resolves to {args.rank} on this predictor; the mlp "
                "generator's output layer would be ~2 B parameters. Use "
                "--generator chunked."
            )
        log.info("Resolved --rank full to %s for scope %s", args.rank, args.target_scope)
    else:
        args.rank = int(args.rank)

    # The reference error these objectives are defined against is measured in the
    # window where the correction is off, which only the hypernetwork path has.
    if args.objective != "prediction" and (args.static_lora or args.dense_finetune != "none"):
        raise ValueError(
            f"--objective {args.objective} needs the frozen model's error on the same "
            "sample, which is only measurable on the hypernetwork path."
        )
    if args.objective != "prediction" and args.prediction_mode != "one_step":
        raise ValueError(
            f"--objective {args.objective} is defined on the one-step prediction error."
        )

    dense_finetune = args.dense_finetune != "none"
    if dense_finetune:
        # Full-model control for the few-shot comparison: no hypernetwork and no
        # LoRA wrappers -- the base predictor's own weights are the adapted
        # parameters, trained on the same data, loss and step budget.
        lora_targets, norm_targets = [], []
    else:
        lora_targets, norm_targets = install_predictor_hyper_targets(
            wm.predictor,
            scope=args.target_scope,
            rank=args.rank,
            scale=args.lora_scale,
        )
    specs = [target.spec for target in lora_targets] + [target.spec for target in norm_targets]

    teacher = None
    teacher_initial = None
    if args.distill_adapter:
        teacher = load_model(
            model_ckpt=model_ckpt,
            train_cfg=model_cfg,
            num_action_repeat=model_cfg.num_action_repeat,
            device=device,
        )
        teacher.eval()
        teacher_initial = {
            name: parameter.detach().clone()
            for name, parameter in teacher.predictor.named_parameters()
        }
    distill_cache = None
    if args.distill_adapter and args.distill_cache:
        distill_cache = TeacherTargetCache(args.distill_cache_dir)
    teacher_lora_rank = args.rank if args.teacher_lora_rank is None else args.teacher_lora_rank

    if args.precompute_teacher_cache:
        if teacher is None or distill_cache is None:
            raise ValueError(
                "--precompute-teacher-cache needs --distill-adapter and --distill-cache-dir"
            )
        precompute_teacher_cache(
            teacher=teacher,
            teacher_initial=teacher_initial,
            dataset=datasets["train"],
            distill_cache=distill_cache,
            device=device,
            model_cfg=model_cfg,
            args=args,
            teacher_lora_rank=teacher_lora_rank,
        )
        return

    first_obs, first_act, _ = next(iter(loaders["train"]))
    first_obs, _ = move_batch(first_obs, first_act, device)
    first_act = first_act.to(device)
    with torch.no_grad():
        first_context_obs, first_context_act, _first_prediction_obs, _first_prediction_act = (
            split_context_and_prediction_window(
                first_obs,
                first_act,
                model_num_hist=model_cfg.num_hist,
                model_num_pred=model_cfg.num_pred,
                context_transitions=args.context_transitions,
            )
        )
        context_dim = transition_context_features(
            wm,
            first_context_obs,
            first_context_act,
            args.context_mode,
            args.context_frames,
            feature_kind=args.context_feature_kind,
            current_obs={key: value[:, :1] for key, value in _first_prediction_obs.items()},
        ).shape[-1]
    context_encoder = None
    generator = None
    adapter_gate = None
    static_lora_state = None
    if dense_finetune:
        if args.dense_finetune == "predictor_all":
            dense_modules = {"": wm.predictor}
        else:
            names = list(select_predictor_lora_module_names(wm.predictor, args.target_scope))
            if args.target_scope.startswith(("predlast", "predall", "predfirstlast")):
                names.append("transformer.norm")
            dense_modules = {name: wm.predictor.get_submodule(name) for name in names}
        trainable_parameters = []
        for module in dense_modules.values():
            for parameter in module.parameters():
                parameter.requires_grad = True
                trainable_parameters.append(parameter)
        log.info(
            "Dense finetune scope=%s: %s modules, %s trainable parameters",
            args.dense_finetune,
            len(dense_modules),
            sum(p.numel() for p in trainable_parameters),
        )
    elif args.static_lora:
        # Context-independent control: shared LoRA (+ LayerNorm delta) parameters
        # trained directly, initialized to the identity correction (B = 0).
        static_lora_state = {}
        trainable_parameters = []
        init_generator = torch.Generator(device="cpu")
        init_generator.manual_seed(int(args.seed))
        for target in lora_targets:
            A_init = torch.empty(target.spec.rank, target.spec.in_features)
            torch.nn.init.normal_(
                A_init,
                mean=0.0,
                std=1.0 / max(target.spec.in_features, 1) ** 0.5,
                generator=init_generator,
            )
            A = torch.nn.Parameter(A_init.to(device))
            B = torch.nn.Parameter(
                torch.zeros(target.spec.out_features, target.spec.rank, device=device)
            )
            target.module.set_lora(A, B)
            static_lora_state[target.name] = ("lora", A, B)
            trainable_parameters += [A, B]
        for target in norm_targets:
            delta_weight = torch.nn.Parameter(torch.zeros(target.spec.features, device=device))
            delta_bias = torch.nn.Parameter(torch.zeros(target.spec.features, device=device))
            target.module.set_affine(delta_weight, delta_bias)
            static_lora_state[target.name] = ("norm", delta_weight, delta_bias)
            trainable_parameters += [delta_weight, delta_bias]
    else:
        if args.context_aggregator in {"transformer", "transformer_query"}:
            if args.context_mode != "transition_buffer":
                raise ValueError("Transformer aggregation is currently supported for transition_buffer only")
            context_encoder = OrderedTransitionContext(
                input_dim=context_dim,
                model_dim=args.context_model_dim,
                depth=args.context_depth,
                heads=args.context_heads,
                max_context_length=max(
                    int(args.context_transitions or model_cfg.num_hist),
                    1,
                ),
                dropout=args.context_dropout,
                query_dim=(
                    state_query_feature(wm, first_context_obs).shape[-1]
                    if args.context_aggregator == "transformer_query"
                    else None
                ),
            ).to(device)
        generator = build_generator(
            args.generator,
            context_dim=context_dim,
            specs=specs,
            hidden_dim=args.hidden_dim,
            depth=args.depth,
            init_scale=args.init_scale,
            freeze_lora_A=args.freeze_lora_a,
            fixed_a_scale=args.fixed_a_scale,
            chunk_size=args.chunk_size,
            chunk_embed_dim=args.chunk_embed_dim,
            head_hidden_dim=args.head_hidden_dim,
        ).to(device)
        adapter_gate = (
            HyperAdapterGate(
                context_dim=context_dim,
                hidden_dim=args.gate_hidden_dim,
                init_bias=args.gate_init_bias,
            ).to(device)
            if args.adapter_gate
            else None
        )
        trainable_parameters = list(generator.parameters())
        if context_encoder is not None:
            trainable_parameters += list(context_encoder.parameters())
        if adapter_gate is not None:
            trainable_parameters += list(adapter_gate.parameters())

    if args.init_adapter is not None:
        # Few-shot starting point: the adapter pretrained on the in-distribution
        # shapes.  Loading is strict, so an architecture/rank mismatch with the
        # checkpoint fails loudly instead of silently training from scratch.
        if dense_finetune:
            raise ValueError("--init-adapter is not applicable to --dense-finetune")
        init_payload = torch.load(args.init_adapter, map_location=device)
        if args.static_lora:
            if "static_lora" not in init_payload:
                raise ValueError(f"{args.init_adapter} holds no static_lora entry")
            with torch.no_grad():
                for name, entry in init_payload["static_lora"].items():
                    kind, *tensors = static_lora_state[name]
                    if entry["kind"] != kind:
                        raise ValueError(f"static-LoRA kind mismatch for {name}")
                    if kind == "lora":
                        tensors[0].copy_(entry["A"].to(device))
                        tensors[1].copy_(entry["B"].to(device))
                    else:
                        tensors[0].copy_(entry["delta_weight"].to(device))
                        tensors[1].copy_(entry["delta_bias"].to(device))
        else:
            generator.load_state_dict(init_payload["hyper_lora"])
            if context_encoder is not None:
                context_encoder.load_state_dict(init_payload["context_encoder"])
            if adapter_gate is not None:
                adapter_gate.load_state_dict(init_payload["adapter_gate"])
        log.info(
            "Initialized adapter from %s (epoch %s)",
            args.init_adapter,
            init_payload.get("epoch"),
        )

    optimizer = torch.optim.AdamW(trainable_parameters, lr=args.lr)

    metadata = vars(args).copy()
    metadata.update(
        {
            "ckpt_dir": str(ckpt_dir),
            "context_dim": int(context_dim),
            "dataset_num_hist": int(dataset_num_hist),
            "context_encoder_parameters": (
                module_parameter_count(context_encoder) if context_encoder is not None else 0
            ),
            "adapter_gate_parameters": (
                module_parameter_count(adapter_gate) if adapter_gate is not None else 0
            ),
            "target_specs": [spec.__dict__ for spec in specs],
            "world_model_parameters": module_parameter_count(wm),
            "hypernetwork_parameters": (
                module_parameter_count(generator) if generator is not None else 0
            ),
            "static_lora": bool(args.static_lora),
            "freeze_lora_A": bool(args.freeze_lora_a),
            "fixed_a_scale": float(args.fixed_a_scale),
            "per_sample_lora_parameters": lora_parameter_budget(
                specs,
                freeze_lora_A=args.freeze_lora_a,
            ),
            "train_loader_shuffle": True,
            "distill_cache_enabled": distill_cache is not None,
            "distill_cache_dir": args.distill_cache_dir,
            "teacher_target_mode": args.teacher_target_mode,
            "teacher_lora_rank_resolved": int(teacher_lora_rank),
            "max_epochs": int(args.epochs),
            "min_epochs": int(args.min_epochs),
            "early_stop_patience": int(args.early_stop_patience),
            "early_stop_min_delta": float(args.early_stop_min_delta),
        }
    )
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    best_val = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    min_epochs = max(0, int(args.min_epochs))
    early_stop_patience = max(0, int(args.early_stop_patience))
    early_stop_min_delta = float(args.early_stop_min_delta)
    # Gradient-step accounting is the compute currency of the few-shot comparison:
    # the hypernetwork and the dense control get the same step budget, and the
    # measured training seconds go into the log next to it.
    step_state = {"steps": 0}
    total_train_seconds = 0.0
    max_epochs = args.epochs
    if args.max_steps is not None and args.steps_per_epoch:
        max_epochs = max(
            1, -(-int(args.max_steps) // int(args.steps_per_epoch))
        )  # ceil
    for epoch in range(1, max_epochs + 1):
        if args.max_steps is not None and step_state["steps"] >= int(args.max_steps):
            break
        epoch_start = time.time()
        train_loss = run_epoch(
            wm,
            generator,
            lora_targets,
            norm_targets,
            loaders["train"],
            optimizer,
            device,
            args.context_mode,
            args.context_frames,
            args.context_aggregator,
            context_encoder,
            adapter_gate,
            model_cfg.num_hist,
            model_cfg.num_pred,
            args.context_transitions,
            teacher=teacher,
            teacher_initial=teacher_initial,
            distill_weight=args.distill_weight,
            distill_cache=distill_cache,
            context_feature_kind=args.context_feature_kind,
            teacher_update_scope=args.teacher_update_scope,
            teacher_lora_rank=teacher_lora_rank,
            teacher_lora_scale=args.teacher_lora_scale,
            teacher_target_mode=args.teacher_target_mode,
            max_batches=args.max_train_batches,
            prediction_mode=args.prediction_mode,
            rollout_frames=args.rollout_frames,
            ranking_weight=args.ranking_weight,
            static_lora=args.static_lora or dense_finetune,
            train=True,
            steps_per_epoch=args.steps_per_epoch,
            step_state=step_state,
            max_total_steps=args.max_steps,
            fixed_context_prefix=args.fixed_context_prefix,
            objective=args.objective,
            hinge_weight=args.hinge_weight,
            prediction_loss_weight=args.prediction_loss_weight,
            teacher_steps=args.teacher_steps,
            teacher_lr=args.teacher_lr,
            distill_space=args.distill_space,
            weight_distill_weight=args.weight_distill_weight,
            lora_scale=args.lora_scale,
        )
        total_train_seconds += time.time() - epoch_start
        with torch.no_grad():
            val_loss = run_epoch(
                wm,
                generator,
                lora_targets,
                norm_targets,
                loaders["valid"],
                optimizer,
                device,
                args.context_mode,
                args.context_frames,
                args.context_aggregator,
                context_encoder,
                adapter_gate,
                model_cfg.num_hist,
                model_cfg.num_pred,
                args.context_transitions,
                teacher=None,
                teacher_initial=None,
                context_feature_kind=args.context_feature_kind,
                max_batches=args.max_val_batches,
                prediction_mode=args.prediction_mode,
                rollout_frames=args.rollout_frames,
                ranking_weight=args.ranking_weight,
                static_lora=args.static_lora or dense_finetune,
                train=False,
                fixed_context_prefix=args.fixed_context_prefix,
                objective=args.objective,
                hinge_weight=args.hinge_weight,
                prediction_loss_weight=args.prediction_loss_weight,
                lora_scale=args.lora_scale,
            )
        improved = val_loss < best_val - early_stop_min_delta
        if improved:
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        should_stop = (
            early_stop_patience > 0
            and epoch >= min_epochs
            and epochs_without_improvement >= early_stop_patience
        )
        record = {
            "epoch": epoch,
            "grad_steps": int(step_state["steps"]),
            "train_seconds": round(total_train_seconds, 2),
            "trainable_parameters": int(sum(p.numel() for p in trainable_parameters)),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "best_val_loss": min(best_val, val_loss),
            "best_epoch": epoch if improved else best_epoch,
            "early_stop/improved": improved,
            "early_stop/epochs_without_improvement": epochs_without_improvement,
            "early_stop/should_stop": should_stop,
        }
        if distill_cache is not None and hasattr(distill_cache, "stats"):
            record["distill_cache"] = distill_cache.stats()
        with (output_dir / "logs.json").open("a") as f:
            f.write(json.dumps(record) + "\n")
        print(record, flush=True)

        payload = {
            "epoch": epoch,
            "grad_steps": int(step_state["steps"]),
            "train_seconds": round(total_train_seconds, 2),
            "metadata": metadata,
        }
        if dense_finetune:
            # Only the predictor is adapted, so its state_dict is the whole delta.
            # scripts/materialize_dense_ckpt.py turns it into an eval-ready
            # checkpoint directory for the ordinary frozen planner path.
            payload["predictor_state"] = {
                name: parameter.detach().cpu()
                for name, parameter in wm.predictor.state_dict().items()
            }
            payload["dense_finetune"] = args.dense_finetune
        elif args.static_lora:
            static_payload = {}
            for name, entry in static_lora_state.items():
                kind = entry[0]
                if kind == "lora":
                    static_payload[name] = {
                        "kind": "lora",
                        "A": entry[1].detach().cpu(),
                        "B": entry[2].detach().cpu(),
                    }
                else:
                    static_payload[name] = {
                        "kind": "norm",
                        "delta_weight": entry[1].detach().cpu(),
                        "delta_bias": entry[2].detach().cpu(),
                    }
            payload["static_lora"] = static_payload
            payload["lora_scale"] = float(args.lora_scale)
        else:
            payload["hyper_lora"] = generator.state_dict()
            if context_encoder is not None:
                payload["context_encoder"] = context_encoder.state_dict()
            if adapter_gate is not None:
                payload["adapter_gate"] = adapter_gate.state_dict()
        torch.save(payload, output_dir / "hyper_lora_latest.pth")
        torch.save(payload, output_dir / f"hyper_lora_epoch_{epoch}.pth")
        if improved or val_loss <= best_val:
            best_val = val_loss
            best_epoch = epoch
            torch.save(payload, output_dir / "hyper_lora_best.pth")
        if should_stop:
            print(
                {
                    "early_stop": True,
                    "epoch": epoch,
                    "best_epoch": best_epoch,
                    "best_val_loss": best_val,
                    "patience": early_stop_patience,
                    "min_delta": early_stop_min_delta,
                },
                flush=True,
            )
            break


if __name__ == "__main__":
    main()
