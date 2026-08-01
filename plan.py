import os
import gym
import json
import time
import hydra
import random
import torch
import pickle
import wandb
import logging
import warnings
import numpy as np
import submitit
from itertools import product
from pathlib import Path
from einops import rearrange
from omegaconf import OmegaConf, open_dict

from env.venv import SubprocVectorEnv
from custom_resolvers import replace_slash
from preprocessor import Preprocessor
from planning.evaluator import PlanEvaluator
from planning.image_corruption import corrupt_obs_dict
from utils import cfg_to_dict, seed

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)


ALL_MODEL_KEYS = [
    "encoder",
    "predictor",
    "decoder",
    "proprio_encoder",
    "action_encoder",
]

def planning_main_in_dir(working_dir, cfg_dict):
    os.chdir(working_dir)
    return planning_main(cfg_dict=cfg_dict)

def launch_plan_jobs(
    epoch,
    cfg_dicts,
    plan_output_dir,
):
    with submitit.helpers.clean_env():
        jobs = []
        for cfg_dict in cfg_dicts:
            subdir_name = f"{cfg_dict['planner']['name']}_goal_source={cfg_dict['goal_source']}_goal_H={cfg_dict['goal_H']}_alpha={cfg_dict['objective']['alpha']}"
            subdir_path = os.path.join(plan_output_dir, subdir_name)
            executor = submitit.AutoExecutor(
                folder=subdir_path, slurm_max_num_timeout=20
            )
            executor.update_parameters(
                **{
                    k: v
                    for k, v in cfg_dict["hydra"]["launcher"].items()
                    if k != "submitit_folder"
                }
            )
            cfg_dict["saved_folder"] = subdir_path
            cfg_dict["wandb_logging"] = False  # don't init wandb
            job = executor.submit(planning_main_in_dir, subdir_path, cfg_dict)
            jobs.append((epoch, subdir_name, job))
            print(
                f"Submitted evaluation job for checkpoint: {subdir_path}, job id: {job.job_id}"
            )
        return jobs


def build_plan_cfg_dicts(
    plan_cfg_path="",
    ckpt_base_path="",
    model_name="",
    model_epoch="final",
    planner=["gd", "cem"],
    goal_source=["dset"],
    goal_H=[1, 5, 10],
    alpha=[0, 0.1, 1],
):
    """
    Return a list of plan overrides, for model_path, add a key in the dict {"model_path": model_path}.
    """
    config_path = os.path.dirname(plan_cfg_path)
    overrides = [
        {
            "planner": p,
            "goal_source": g_source,
            "goal_H": g_H,
            "ckpt_base_path": ckpt_base_path,
            "model_name": model_name,
            "model_epoch": model_epoch,
            "objective": {"alpha": a},
        }
        for p, g_source, g_H, a in product(planner, goal_source, goal_H, alpha)
    ]
    cfg = OmegaConf.load(plan_cfg_path)
    cfg_dicts = []
    for override_args in overrides:
        planner = override_args["planner"]
        planner_cfg = OmegaConf.load(
            os.path.join(config_path, f"planner/{planner}.yaml")
        )
        cfg["planner"] = OmegaConf.merge(cfg.get("planner", {}), planner_cfg)
        override_args.pop("planner")
        cfg = OmegaConf.merge(cfg, OmegaConf.create(override_args))
        cfg_dict = OmegaConf.to_container(cfg)
        cfg_dict["planner"]["horizon"] = cfg_dict["goal_H"]  # assume planning horizon equals to goal horizon
        cfg_dicts.append(cfg_dict)
    return cfg_dicts


class PlanWorkspace:
    def __init__(
        self,
        cfg_dict: dict,
        wm: torch.nn.Module,
        dset,
        env: SubprocVectorEnv,
        env_name: str,
        frameskip: int,
        wandb_run: wandb.run,
    ):
        self.cfg_dict = cfg_dict
        self.wm = wm
        self.dset = dset
        self.env = env
        self.env_name = env_name
        self.frameskip = frameskip
        self.wandb_run = wandb_run
        self.device = next(wm.parameters()).device

        # `eval_episode_index` runs a single episode out of the cohort a batched
        # run of `eval_episode_total` episodes would have evaluated, so that
        # methods requiring episode isolation (AdaJEPA online TTA owns mutable
        # parameters, hence n_evals=1) are scored on exactly the same segments
        # and environment seeds as the batched columns.
        self.eval_episode_total = (
            cfg_dict.get("eval_episode_total") or cfg_dict["n_evals"]
        )
        self.eval_episode_index = cfg_dict.get("eval_episode_index", None)
        if self.eval_episode_index is not None and cfg_dict["n_evals"] != 1:
            raise ValueError("eval_episode_index requires n_evals=1")

        # Independent environment seeds for every evaluation episode. The
        # previous multiplication by the episode index gave all-one seeds when
        # the outer seed was zero.
        if self.eval_episode_index is None:
            self.eval_seed = [
                cfg_dict["seed"] * self.eval_episode_total + n + 1
                for n in range(cfg_dict["n_evals"])
            ]
        else:
            self.eval_seed = [
                cfg_dict["seed"] * self.eval_episode_total
                + int(self.eval_episode_index)
                + 1
            ]
        print("eval_seed: ", self.eval_seed)
        self.n_evals = cfg_dict["n_evals"]
        self.goal_source = cfg_dict["goal_source"]
        self.goal_H = cfg_dict["goal_H"]
        # Eval-time observation shift (OOD). None => clean in-distribution eval.
        self.ood_corruption = cfg_dict.get("ood_corruption", None)
        self.ood_level = cfg_dict.get("ood_level", None)
        self.action_dim = self.dset.action_dim * self.frameskip
        self.debug_dset_init = cfg_dict["debug_dset_init"]

        objective_fn = hydra.utils.call(
            cfg_dict["objective"],
        )

        self.data_preprocessor = Preprocessor(
            action_mean=self.dset.action_mean,
            action_std=self.dset.action_std,
            state_mean=self.dset.state_mean,
            state_std=self.dset.state_std,
            proprio_mean=self.dset.proprio_mean,
            proprio_std=self.dset.proprio_std,
            transform=self.dset.transform,
        )

        if self.cfg_dict["goal_source"] == "file":
            self.prepare_targets_from_file(cfg_dict["goal_file_path"])
        elif self.cfg_dict["goal_source"] == "segments":
            self.prepare_targets_from_segments(cfg_dict["eval_data_path"])
        else:
            self.prepare_targets()

        self.evaluator = PlanEvaluator(
            obs_0=self.obs_0,
            obs_g=self.obs_g,
            state_0=self.state_0,
            state_g=self.state_g,
            env=self.env,
            wm=self.wm,
            frameskip=self.frameskip,
            seed=self.eval_seed,
            preprocessor=self.data_preprocessor,
            n_plot_samples=self.cfg_dict["n_plot_samples"],
            decode_for_viz=self.cfg_dict.get("decode_for_viz", True),
            ood_corruption=self.ood_corruption,
            ood_level=self.ood_level,
        )

        if self.wandb_run is None or isinstance(
            self.wandb_run, wandb.sdk.lib.disabled.RunDisabled
        ):
            self.wandb_run = DummyWandbRun()

        self.goal_H_model_steps = self.goal_H // self.frameskip
        self.log_filename = "logs.json"  # planner and final eval logs are dumped here
        planner_cfg = self.cfg_dict["planner"].copy()
        if "n_taken_actions" in planner_cfg:
            planner_cfg["n_taken_actions"] = planner_cfg["n_taken_actions"] // self.frameskip
        if "sub_planner" in planner_cfg and "horizon" in planner_cfg["sub_planner"]:
            planner_cfg["sub_planner"]["horizon"] = (
                planner_cfg["sub_planner"]["horizon"] // self.frameskip
            )
        elif "horizon" in planner_cfg:
            planner_cfg["horizon"] = planner_cfg["horizon"] // self.frameskip
        self.planner = hydra.utils.instantiate(
            planner_cfg,
            wm=self.wm,
            env=self.env,  # only for mpc
            action_dim=self.action_dim,
            objective_fn=objective_fn,
            preprocessor=self.data_preprocessor,
            evaluator=self.evaluator,
            wandb_run=self.wandb_run,
            log_filename=self.log_filename,
        )


        self.planner.horizon = self.goal_H_model_steps

        self.dump_targets()

    def prepare_targets(self):
        states = []
        actions = []
        observations = []
        
        if self.goal_source in ("random_state", "point_maze_cell_distance",
                                "point_maze_bfs"):
            # update env config from val trajs
            observations, states, actions, env_info = (
                self.sample_traj_segment_from_dset(traj_len=2)
            )
            self.env.update_env(env_info)

            # sample random states
            if self.goal_source == "point_maze_bfs":
                # BFS-distance-controlled reachable goals for held-out-layout eval.
                rand_init_state, rand_goal_state = (
                    self.env.sample_bfs_distance_init_goal_states(
                        self.eval_seed,
                        self.cfg_dict.get("point_maze_min_cell_distance", 3),
                        self.cfg_dict.get("point_maze_max_cell_distance", 5),
                        self.cfg_dict.get("point_maze_cell_jitter", 0.2),
                    )
                )
            elif self.goal_source == "point_maze_cell_distance":
                rand_init_state, rand_goal_state = (
                    self.env.sample_distant_cell_init_goal_states(
                        self.eval_seed,
                        self.cfg_dict.get("point_maze_min_cell_distance", 3.0),
                        self.cfg_dict.get("point_maze_cell_jitter", 0.25),
                    )
                )
            else:
                rand_init_state, rand_goal_state = self.env.sample_random_init_goal_states(
                    self.eval_seed
                )
            if self.env_name == "deformable_env": # take rand init state from dset for deformable envs
                rand_init_state = np.array([x[0] for x in states])

            obs_0, state_0 = self.env.prepare(self.eval_seed, rand_init_state)
            obs_g, state_g = self.env.prepare(self.eval_seed, rand_goal_state)

            # add dim for t
            for k in obs_0.keys():
                obs_0[k] = np.expand_dims(obs_0[k], axis=1)
                obs_g[k] = np.expand_dims(obs_g[k], axis=1)

            self.obs_0 = obs_0
            self.context_obs_0 = obs_0
            self.obs_g = obs_g
            self.state_0 = rand_init_state  # (b, d)
            self.state_g = rand_goal_state
            self.gt_actions = None
        else:
            # update env config from val trajs
            observations, states, actions, env_info = (
                self.sample_traj_segment_from_dset(traj_len=self.goal_H + 1)
            )
            self.env.update_env(env_info)

            # get states from val trajs
            init_state = [x[0] for x in states]
            init_state = np.array(init_state)
            actions = torch.stack(actions)
            if self.goal_source == "random_action":
                actions = torch.randn_like(actions)
            wm_actions = rearrange(actions, "b (t f) d -> b t (f d)", f=self.frameskip)
            exec_actions = self.data_preprocessor.denormalize_actions(actions)
            # replay actions in env to get gt obses
            rollout_obses, rollout_states = self.env.rollout(
                self.eval_seed, init_state, exec_actions.numpy()
            )
            self.obs_0 = {
                key: np.expand_dims(arr[:, 0], axis=1)
                for key, arr in rollout_obses.items()
            }
            context_len = max(1, int(self.cfg_dict.get("num_start_frames", 1)))
            self.context_obs_0 = {
                key: arr[:, :context_len]
                for key, arr in rollout_obses.items()
            }
            self.obs_g = {
                key: np.expand_dims(arr[:, -1], axis=1)
                for key, arr in rollout_obses.items()
            }
            self.state_0 = init_state  # (b, d)
            self.state_g = rollout_states[:, -1]  # (b, d)
            self.gt_actions = wm_actions

        # Eval-time OOD: corrupt the conditioning + goal frames the model
        # encodes, consistent with the closed-loop corruption in the evaluator.
        # No-op when ood_corruption is None (clean in-distribution eval).
        if self.ood_corruption:
            base_seed = (
                self.eval_seed[0]
                if isinstance(self.eval_seed, (list, tuple, np.ndarray))
                else self.eval_seed
            )
            self.obs_0 = corrupt_obs_dict(
                self.obs_0, self.ood_corruption, self.ood_level, seed=int(base_seed)
            )
            self.obs_g = corrupt_obs_dict(
                self.obs_g, self.ood_corruption, self.ood_level, seed=int(base_seed)
            )
            self.context_obs_0 = corrupt_obs_dict(
                self.context_obs_0, self.ood_corruption, self.ood_level,
                seed=int(base_seed),
            )

    def prepare_targets_from_segments(self, file_path):
        """Load AdaJEPA-format pre-sampled segments (init state + pixel-space actions),
        replay in our env to get obs/goal, matching adajepa/plan.py:422. Enables
        held-out-shape eval on our own base for the 4-method matrix."""
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        segments = data["segments"]
        self.goal_H = data["traj_len"]

        rng = random.Random(self.cfg_dict["seed"])
        chosen = rng.sample(segments, min(self.eval_episode_total, len(segments)))
        if self.eval_episode_index is not None:
            chosen = [chosen[int(self.eval_episode_index)]]

        # Pad 5-dim states to 7-dim (zero velocities) if needed
        raw_init = np.array([s["states"][0] for s in chosen])
        if raw_init.shape[-1] == 5:
            init_states = np.zeros((len(chosen), 7), dtype=np.float32)
            init_states[:, :5] = raw_init
        else:
            init_states = raw_init.astype(np.float32)
        seg_actions = torch.stack(
            [torch.tensor(s["actions"], dtype=torch.float32) for s in chosen]
        )

        env_info = [{"shape": s["shape"]} if "shape" in s else {} for s in chosen]
        self.env.update_env(env_info)

        # PushT/PushObj store actions in pixel-space; env expects pixel/action_scale.
        action_scale = 100.0 if self.env_name in ("pusht", "pushobj") else 1.0
        env_actions = seg_actions.numpy() / action_scale
        rollout_obses, rollout_states = self.env.rollout(
            self.eval_seed, init_states, env_actions
        )
        self.obs_0 = {
            key: np.expand_dims(arr[:, 0], axis=1) for key, arr in rollout_obses.items()
        }
        context_len = max(1, int(self.cfg_dict.get("num_start_frames", 1)))
        self.context_obs_0 = {
            key: arr[:, :context_len] for key, arr in rollout_obses.items()
        }
        self.obs_g = {
            key: np.expand_dims(arr[:, -1], axis=1) for key, arr in rollout_obses.items()
        }
        self.state_0 = init_states
        self.state_g = rollout_states[:, -1]
        # gt actions (normalized, frameskip-grouped) for planner warm-start
        denorm = seg_actions / action_scale
        normed = (denorm - self.dset.action_mean) / self.dset.action_std
        if normed.shape[1] % self.frameskip == 0:
            self.gt_actions = rearrange(normed, "b (t f) d -> b t (f d)", f=self.frameskip)
        else:
            self.gt_actions = None

    def sample_traj_segment_from_dset(self, traj_len):
        states = []
        actions = []
        observations = []
        env_info = []

        # Check if any trajectory is long enough
        valid_traj = [
            self.dset[i][0]["visual"].shape[0]
            for i in range(len(self.dset))
            if self.dset[i][0]["visual"].shape[0] >= traj_len
        ]
        if len(valid_traj) == 0:
            raise ValueError("No trajectory in the dataset is long enough.")

        # Sample init_states from dset.
        #
        # Draw `eval_episode_total` segments, not `n_evals`, and slice afterwards. The
        # draws come from the global `random` state, which is seeded by `seed` alone, so a
        # process running with n_evals=1 would otherwise redraw the *first* segment of the
        # sequence -- meaning every episode-isolated trial evaluates the SAME episode.
        # (Symptom: an entire 50-episode column scoring exactly 0.000 or exactly 1.000.)
        # Drawing the full sequence and taking element `eval_episode_index` makes isolated
        # trial i consume the identical draws as episode i of a batched run of the same
        # size, which is what makes the two comparable and the outcomes paired. When
        # `eval_episode_index` is None, `eval_episode_total` equals `n_evals` and the
        # behaviour is unchanged.
        for i in range(self.eval_episode_total):
            max_offset = -1
            while max_offset < 0:  # filter out traj that are not long enough
                traj_id = random.randint(0, len(self.dset) - 1)
                obs, act, state, e_info = self.dset[traj_id]
                max_offset = obs["visual"].shape[0] - traj_len
            state = state.numpy()
            offset = random.randint(0, max_offset)
            obs = {
                key: arr[offset : offset + traj_len]
                for key, arr in obs.items()
            }
            state = state[offset : offset + traj_len]
            act = act[offset : offset + self.goal_H]
            actions.append(act)
            states.append(state)
            observations.append(obs)
            env_info.append(e_info)
        if self.eval_episode_index is not None:
            idx = int(self.eval_episode_index)
            if idx >= len(observations):
                raise ValueError(
                    f"eval_episode_index={idx} is out of range for "
                    f"eval_episode_total={self.eval_episode_total}"
                )
            observations = [observations[idx]]
            states = [states[idx]]
            actions = [actions[idx]]
            env_info = [env_info[idx]]
        return observations, states, actions, env_info

    def prepare_targets_from_file(self, file_path):
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        self.obs_0 = data["obs_0"]
        self.obs_g = data["obs_g"]
        self.state_0 = data["state_0"]
        self.state_g = data["state_g"]
        self.gt_actions = data["gt_actions"]
        self.goal_H = data["goal_H"]
        self.context_obs_0 = data.get("context_obs_0", self.obs_0)

    def dump_targets(self):
        with open("plan_targets.pkl", "wb") as f:
            pickle.dump(
                {
                    "obs_0": self.obs_0,
                    "obs_g": self.obs_g,
                    "context_obs_0": getattr(self, "context_obs_0", self.obs_0),
                    "state_0": self.state_0,
                    "state_g": self.state_g,
                    "gt_actions": self.gt_actions,
                    "goal_H": self.goal_H,
                },
                f,
            )
        file_path = os.path.abspath("plan_targets.pkl")
        print(f"Dumped plan targets to {file_path}")

    def perform_planning(self):
        if self.debug_dset_init:
            actions_init = self.gt_actions
        else:
            actions_init = None
        actions, action_len = self.planner.plan(
            obs_0=self.obs_0,
            obs_g=self.obs_g,
            actions=actions_init,
            context_obs_0=getattr(self, "context_obs_0", self.obs_0),
        )
        logs, successes, _, _ = self.evaluator.eval_actions(
            actions.detach(), action_len, save_video=True, filename="output_final"
        )
        logs = {f"final_eval/{k}": v for k, v in logs.items()}
        self.wandb_run.log(logs)
        logs_entry = {
            key: (
                value.item()
                if isinstance(value, (np.float32, np.int32, np.int64))
                else value
            )
            for key, value in logs.items()
        }
        with open(self.log_filename, "a") as file:
            file.write(json.dumps(logs_entry) + "\n")
        return logs


def load_ckpt(snapshot_path, device):
    from models.dino import DinoV2Encoder
    _ = DinoV2Encoder('dinov2_vits14', 'x_norm_patchtokens')
    with snapshot_path.open("rb") as f:
        payload = torch.load(f, map_location=device)
    loaded_keys = []
    result = {}
    for k, v in payload.items():
        if k in ALL_MODEL_KEYS:
            loaded_keys.append(k)
            result[k] = v.to(device) if v is not None else None
    result["epoch"] = payload["epoch"]
    return result


def load_model(model_ckpt, train_cfg, num_action_repeat, device):
    result = {}
    if model_ckpt.exists():
        result = load_ckpt(model_ckpt, device)
        print(f"Resuming from epoch {result['epoch']}: {model_ckpt}")

    if "encoder" not in result:
        result["encoder"] = hydra.utils.instantiate(
            train_cfg.encoder,
        )
    if "predictor" not in result:
        raise ValueError("Predictor not found in model checkpoint")

    if train_cfg.has_decoder and "decoder" not in result:
        base_path = os.path.dirname(os.path.abspath(__file__))
        if train_cfg.env.decoder_path is not None:
            decoder_path = os.path.join(base_path, train_cfg.env.decoder_path)
            ckpt = torch.load(decoder_path)
            if isinstance(ckpt, dict):
                result["decoder"] = ckpt["decoder"]
            else:
                result["decoder"] = torch.load(decoder_path)
        else:
            raise ValueError(
                "Decoder path not found in model checkpoint \
                                and is not provided in config"
            )
    elif not train_cfg.has_decoder:
        result["decoder"] = None

    model = hydra.utils.instantiate(
        train_cfg.model,
        encoder=result["encoder"],
        proprio_encoder=result["proprio_encoder"],
        action_encoder=result["action_encoder"],
        predictor=result["predictor"],
        decoder=result["decoder"],
        proprio_dim=train_cfg.proprio_emb_dim,
        action_dim=train_cfg.action_emb_dim,
        concat_dim=train_cfg.concat_dim,
        num_action_repeat=num_action_repeat,
        num_proprio_repeat=train_cfg.num_proprio_repeat,
    )
    model.to(device)
    return model


class DummyWandbRun:
    def __init__(self):
        self.mode = "disabled"

    def log(self, *args, **kwargs):
        pass

    def watch(self, *args, **kwargs):
        pass

    def config(self, *args, **kwargs):
        pass

    def finish(self):
        pass


def planning_main(cfg_dict):
    t_start = time.perf_counter()
    t_after_model = None
    t_after_workspace = None
    t_after_planning = None

    output_dir = cfg_dict["saved_folder"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if cfg_dict["wandb_logging"]:
        wandb_run = wandb.init(
            project=f"plan_{cfg_dict['planner']['name']}", config=cfg_dict
        )
        wandb.run.name = "{}".format(output_dir.split("plan_outputs/")[-1])
    else:
        wandb_run = None

    ckpt_base_path = cfg_dict["ckpt_base_path"]
    model_name = cfg_dict.get("model_name")
    if ckpt_base_path.startswith("/"):
        model_path = ckpt_base_path
    else:
        model_path = f"{ckpt_base_path}/{cfg_dict['model_name']}/"
    model_path = os.path.abspath(model_path)
    with open(os.path.join(model_path, "hydra.yaml"), "r") as f:
        model_cfg = OmegaConf.load(f)

    # Optionally relocate the training dataset. The checkpoint records an absolute
    # `data_path`, which for the PointMaze bases is an SMB mount. Only small metadata
    # tensors are read from it (observations come from `hdf5_path`), but every eval
    # process loads them at startup, and a large fan-out is enough to take the mount
    # down. Staging those files locally and pointing here avoids that without editing
    # checkpoints. Affects only where data is read from, never what is read.
    data_path_override = cfg_dict.get("dataset_data_path", None)
    if data_path_override:
        model_cfg.env.dataset.data_path = str(data_path_override)
        print(f"[dataset] data_path -> {data_path_override}", flush=True)

    seed(cfg_dict["seed"])
    _, dset = hydra.utils.call(
        model_cfg.env.dataset,
        num_hist=model_cfg.num_hist,
        num_pred=model_cfg.num_pred,
        frameskip=model_cfg.frameskip,
    )
    dset = dset["valid"]

    num_action_repeat = model_cfg.num_action_repeat
    model_ckpt = (
        Path(model_path) / "checkpoints" / f"model_{cfg_dict['model_epoch']}.pth"
    )
    model = load_model(model_ckpt, model_cfg, num_action_repeat, device=device)
    t_after_model = time.perf_counter()
    print(f"[timing] setup_model_s={t_after_model - t_start:.3f}", flush=True)

    eval_env_name = cfg_dict.get("evaluation_env_name") or model_cfg.env.name
    eval_env_args = cfg_dict.get("evaluation_env_args", model_cfg.env.args)
    eval_env_kwargs = cfg_dict.get("evaluation_env_kwargs", model_cfg.env.kwargs)
    # use dummy vector env for wall and deformable envs
    if eval_env_name == "wall" or eval_env_name == "deformable_env":
        from env.serial_vector_env import SerialVectorEnv
        env = SerialVectorEnv(
            [
                gym.make(
                    eval_env_name, *eval_env_args, **eval_env_kwargs
                )
                for _ in range(cfg_dict["n_evals"])
            ]
        )
    else:
        env = SubprocVectorEnv(
            [
                lambda: gym.make(
                    eval_env_name, *eval_env_args, **eval_env_kwargs
                )
                for _ in range(cfg_dict["n_evals"])
            ]
        )

    plan_workspace = PlanWorkspace(
        cfg_dict=cfg_dict,
        wm=model,
        dset=dset,
        env=env,
        env_name=eval_env_name,
        frameskip=model_cfg.frameskip,
        wandb_run=wandb_run,
    )
    t_after_workspace = time.perf_counter()
    print(f"[timing] setup_workspace_s={t_after_workspace - t_after_model:.3f}", flush=True)

    logs = plan_workspace.perform_planning()
    t_after_planning = time.perf_counter()
    print(f"[timing] perform_planning_s={t_after_planning - t_after_workspace:.3f}", flush=True)
    print(f"[timing] total_planning_main_s={t_after_planning - t_start:.3f}", flush=True)
    return logs


@hydra.main(config_path="conf", config_name="plan_gd")
def main(cfg: OmegaConf):
    with open_dict(cfg):
        cfg["saved_folder"] = os.getcwd()
        log.info(f"Planning result saved dir: {cfg['saved_folder']}")
    cfg_dict = cfg_to_dict(cfg)
    cfg_dict["wandb_logging"] = bool(cfg_dict.get("wandb_logging", True))
    planning_main(cfg_dict)


if __name__ == "__main__":
    main()
