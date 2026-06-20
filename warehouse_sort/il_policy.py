"""Imitation-learning policy entrypoints for eval.py / the judge.

Each function satisfies the policy contract:
    policy.act(obs, deterministic=True) -> Tensor (num_envs, action_dim) in [-1, 1]

Wire one in via the config `policy` field:
    pixi run python eval.py difficulty=easy \\
        policy=warehouse_sort.il_policy:load_dp_rgb \\
        checkpoint=<path> eval_config=conf/eval/default.yaml
"""

import torch


def _add_baseline_path(rel):
    import os, sys
    p = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "il", "baselines", rel))
    if p not in sys.path:
        sys.path.insert(0, p)


# --------------------------------------------------------------------------- #
# RGB Diffusion Policy — image + robot proprioception, NO privileged state.
# Same fixed image input shape at every difficulty; same checkpoint runs across configs.
# Template only — image IL is not yet solving this task.
# --------------------------------------------------------------------------- #
class _DPRgbPolicy:
    def __init__(self, agent, obs_horizon, device, num_inference_steps=16,
                 execution_horizon=4):
        self.agent = agent.to(device).eval()
        self.agent.noise_scheduler.set_timesteps(num_inference_steps)
        self.obs_horizon = obs_horizon
        self.device = device
        self.execution_horizon = int(execution_horizon)
        if self.execution_horizon < 1:
            raise ValueError("execution_horizon must be at least 1")
        self.prev = None
        self.action_cache = None
        self.action_index = 0

    def reset(self):
        """Clear temporal state at an environment reset."""
        self.prev = None
        self.action_cache = None
        self.action_index = 0

    @torch.no_grad()
    def act(self, obs, deterministic=True):
        state = obs["state"].float().to(self.device)
        rgb = obs["rgb"].to(self.device)
        cur = {"state": state, "rgb": rgb}
        if self.prev is None or self.prev["state"].shape != state.shape:
            self.prev = cur

        if self.action_cache is None or self.action_index >= self.action_cache.shape[1]:
            obs_seq = {
                "state": torch.stack([self.prev["state"], state], dim=1),
                "rgb": torch.stack([self.prev["rgb"], rgb], dim=1),
            }
            predicted = self.agent.get_action(obs_seq)
            cache_len = min(self.execution_horizon, predicted.shape[1])
            self.action_cache = predicted[:, :cache_len]
            self.action_index = 0

        action = self.action_cache[:, self.action_index]
        self.action_index += 1
        self.prev = cur
        return action.clamp(-1.0, 1.0)


def load_dp_rgb(checkpoint, sample_obs, action_space, device,
                obs_horizon=2, act_horizon=8, pred_horizon=16,
                diffusion_step_embed_dim=64, unet_dims=(64, 128, 256), n_groups=8,
                num_inference_steps=16, execution_horizon=4,
                visual_encoder="resnet18", num_kp=32, input_resolution=160,
                aux_color_weight=0.0):
    """Load an RGB Diffusion Policy checkpoint (vendored train_rgbd; uses EMA weights).

    Template implementation — image IL is not yet solving this task.
    """
    import types
    import numpy as np
    import gymnasium.spaces as spaces
    _add_baseline_path("diffusion_policy")
    from train_rgbd import Agent

    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    saved = ckpt.get("model_config", {})
    obs_horizon = saved.get("obs_horizon", obs_horizon)
    act_horizon = saved.get("act_horizon", act_horizon)
    pred_horizon = saved.get("pred_horizon", pred_horizon)
    diffusion_step_embed_dim = saved.get(
        "diffusion_step_embed_dim", diffusion_step_embed_dim
    )
    unet_dims = saved.get("unet_dims", unet_dims)
    n_groups = saved.get("n_groups", n_groups)
    visual_encoder = saved.get("visual_encoder", visual_encoder)
    num_kp = saved.get("num_kp", num_kp)
    input_resolution = saved.get("input_resolution", input_resolution)
    aux_color_weight = saved.get("aux_color_weight", aux_color_weight)

    h, w, c = sample_obs["rgb"].shape[1:]
    state_dim = sample_obs["state"].shape[1]
    stub = types.SimpleNamespace(
        single_observation_space=spaces.Dict({
            "state": spaces.Box(-np.inf, np.inf, (obs_horizon, state_dim), np.float32),
            "rgb": spaces.Box(0, 255, (obs_horizon, h, w, c), np.uint8),
        }),
        single_action_space=spaces.Box(-1.0, 1.0, (action_space.shape[0],), np.float32),
    )
    args = types.SimpleNamespace(
        obs_horizon=obs_horizon, act_horizon=act_horizon, pred_horizon=pred_horizon,
        diffusion_step_embed_dim=diffusion_step_embed_dim, unet_dims=list(unet_dims),
        n_groups=n_groups, visual_encoder=visual_encoder, num_kp=num_kp,
        input_resolution=input_resolution,
        aux_color_weight=aux_color_weight,
    )
    agent = Agent(stub, args)
    agent.load_state_dict(ckpt.get("ema_agent", ckpt.get("agent")))
    return _DPRgbPolicy(
        agent, obs_horizon, device,
        num_inference_steps=num_inference_steps,
        execution_horizon=execution_horizon,
    )


def load_dp_rgb_exec2(*args, **kwargs):
    return load_dp_rgb(*args, execution_horizon=2, **kwargs)


def load_dp_rgb_exec8(*args, **kwargs):
    return load_dp_rgb(*args, execution_horizon=8, **kwargs)
