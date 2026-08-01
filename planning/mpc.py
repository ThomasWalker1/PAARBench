import time

import hydra
import numpy as np
import torch
from einops import rearrange

from paarbench.adapter import NullAdapter
from paarbench.planner_hooks import build_adapter
from utils import slice_trajdict_with_t

from .base_planner import BasePlanner


class MPCPlanner(BasePlanner):
    """
    an online planner so feedback from env is allowed

    Test-time adaptation enters through exactly one object, ``self.adapter``, which
    implements ``paarbench.adapter.TestTimeAdapter``.  There is deliberately no
    per-method branch here: adding a method must never require editing the control
    loop.  A frozen run uses ``NullAdapter``, whose hooks are all no-ops, so the
    frozen baseline is identical to running with no adaptation code at all.
    """

    def __init__(
        self,
        max_iter,
        n_taken_actions,
        sub_planner,
        wm,
        env,  # for online exec
        action_dim,
        objective_fn,
        preprocessor,
        evaluator,
        wandb_run,
        logging_prefix="mpc",
        log_filename="logs.json",
        adapter=None,
        **kwargs,
    ):
        super().__init__(
            wm,
            action_dim,
            objective_fn,
            preprocessor,
            evaluator,
            wandb_run,
            log_filename,
        )
        self.env = env
        self.max_iter = np.inf if max_iter is None else max_iter
        self.n_taken_actions = n_taken_actions
        self.logging_prefix = logging_prefix
        sub_planner["_target_"] = sub_planner["target"]
        self.sub_planner = hydra.utils.instantiate(
            sub_planner,
            wm=self.wm,
            action_dim=self.action_dim,
            objective_fn=self.objective_fn,
            preprocessor=self.preprocessor,
            evaluator=self.evaluator,  # evaluator is shared for mpc and sub_planner
            wandb_run=self.wandb_run,
            log_filename=None,
        )
        self.is_success = None
        self.action_len = None  # keep track of the step each traj reaches success
        self.iter = 0
        self.planned_actions = []
        self.adapter = build_adapter(
            adapter, wm=self.wm, preprocessor=self.preprocessor
        ) or NullAdapter()

    def _apply_success_mask(self, actions):
        device = actions.device
        mask = torch.tensor(self.is_success).bool()
        actions[mask] = 0
        masked_actions = rearrange(
            actions[mask], "... (f d) -> ... f d", f=self.evaluator.frameskip
        )
        masked_actions = self.preprocessor.normalize_actions(masked_actions.cpu())
        masked_actions = rearrange(masked_actions, "... f d -> ... (f d)")
        actions[mask] = masked_actions.to(device)
        return actions

    @staticmethod
    def _peak_memory_mb():
        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.max_memory_allocated() / (1024 ** 2)

    def plan(self, obs_0, obs_g, actions=None, context_obs_0=None):
        """
        actions is NOT used
        Returns:
            actions: (B, T, action_dim) torch.Tensor
        """
        n_evals = obs_0["visual"].shape[0]
        self.is_success = np.zeros(n_evals, dtype=bool)
        self.action_len = np.full(n_evals, np.inf)
        init_obs_0, init_state_0 = self.evaluator.get_init_cond()

        cur_obs_0 = obs_0
        memo_actions = actions

        # Reset all per-episode adapter state and restore the base model.  Adaptation
        # cost is measured from here on, separately from planner cost.
        episode_start = time.perf_counter()
        episode_logs = dict(
            self.adapter.on_episode_start(
                context_obs_0 if context_obs_0 is not None else cur_obs_0,
                obs_g,
            )
        )
        episode_logs["timing/adapt_episode_start_s"] = time.perf_counter() - episode_start

        while not np.all(self.is_success) and self.iter < self.max_iter:
            self.sub_planner.logging_prefix = f"plan_{self.iter}"
            replan_start = time.perf_counter()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            # --- adaptation: apply or refresh the correction before solving -------
            adapt_start = time.perf_counter()
            adapt_logs = dict(self.adapter.before_plan(cur_obs_0))
            adapt_seconds = time.perf_counter() - adapt_start
            adapt_peak_mb = self._peak_memory_mb()

            # --- planning ---------------------------------------------------------
            planner_start = time.perf_counter()
            actions, _ = self.sub_planner.plan(
                obs_0=cur_obs_0,
                obs_g=obs_g,
                actions=memo_actions,
                step=self.iter,
            )  # (b, t, act_dim)
            planner_seconds = time.perf_counter() - planner_start
            taken_actions = actions.detach()[:, : self.n_taken_actions]
            self._apply_success_mask(taken_actions)
            memo_actions = actions.detach()[:, self.n_taken_actions :]
            self.planned_actions.append(taken_actions)

            # --- execution --------------------------------------------------------
            print(f"MPC iter {self.iter} Eval ------- ")
            action_so_far = torch.cat(self.planned_actions, dim=1)
            self.evaluator.assign_init_cond(
                obs_0=init_obs_0,
                state_0=init_state_0,
            )
            evaluator_start = time.perf_counter()
            logs, successes, e_obses, e_states = self.evaluator.eval_actions(
                action_so_far,
                self.action_len,
                filename=f"plan{self.iter}",
                save_video=True,
            )
            evaluator_seconds = time.perf_counter() - evaluator_start
            new_successes = successes & ~self.is_success  # Identify new successes
            self.is_success = (
                self.is_success | successes
            )  # Update overall success status
            self.action_len[new_successes] = (
                (self.iter + 1) * self.n_taken_actions
            )  # Update only for the newly successful trajectories

            print("self.is_success: ", self.is_success)
            logs = {f"{self.logging_prefix}/{k}": v for k, v in logs.items()}
            logs.update({"step": self.iter + 1})
            logs.update(episode_logs)
            episode_logs = {}
            logs.update(adapt_logs)

            # --- adaptation: observe what the environment did ---------------------
            # The whole rollout is handed over, not just its final frame: a method may
            # build one feature per executed action, and that needs the frame
            # boundaries.  Methods wanting only the endpoint slice it themselves.
            observe_start = time.perf_counter()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            logs.update(
                self.adapter.on_transition(
                    cur_obs_0,
                    taken_actions.detach(),
                    e_obses,
                    self.evaluator.frameskip,
                )
            )
            adapt_seconds += time.perf_counter() - observe_start
            adapt_peak_mb = max(adapt_peak_mb, self._peak_memory_mb())

            # update evaluator's init conditions with new env feedback
            e_final_obs = slice_trajdict_with_t(e_obses, start_idx=-1)
            cur_obs_0 = e_final_obs
            e_final_state = e_states[:, -1]
            self.evaluator.assign_init_cond(
                obs_0=e_final_obs,
                state_0=e_final_state,
            )

            logs.update(self.adapter.metrics())
            logs.update(
                {
                    "timing/planner_s": planner_seconds,
                    "timing/evaluator_s": evaluator_seconds,
                    "timing/adapt_s": adapt_seconds,
                    "timing/model_replan_s": planner_seconds + adapt_seconds,
                    "timing/replan_total_s": time.perf_counter() - replan_start,
                    "memory/adapt_peak_mb": adapt_peak_mb,
                }
            )
            self.wandb_run.log(logs)
            self.dump_logs(logs)
            self.iter += 1
            self.sub_planner.logging_prefix = f"plan_{self.iter}"

        planned_actions = torch.cat(self.planned_actions, dim=1)
        self.evaluator.assign_init_cond(
            obs_0=init_obs_0,
            state_0=init_state_0,
        )

        return planned_actions, self.action_len
