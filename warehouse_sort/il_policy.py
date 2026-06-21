"""Imitation-learning policy entrypoints for eval.py / the judge.

Each function satisfies the policy contract:
    policy.act(obs, deterministic=True) -> Tensor (num_envs, action_dim) in [-1, 1]

Wire one in via the config `policy` field:
    pixi run python eval.py difficulty=easy \\
        policy=warehouse_sort.il_policy:load_dp \\
        checkpoint=<path> eval_config=conf/eval/default.yaml

  load_dp      — state Diffusion Policy (main track; one checkpoint PER level)
  load_dp_rgb  — RGB Diffusion Policy (optional image track; template)
"""

import torch


def _add_baseline_path(rel):
    import os, sys
    p = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "il", "baselines", rel))
    if p not in sys.path:
        sys.path.insert(0, p)


# --------------------------------------------------------------------------- #
# State Diffusion Policy (MAIN track) — privileged low-dim state obs. The execution horizon
# controls how often the policy replans. The state vector is parcel-count-specific, so a
# checkpoint is trained PER difficulty level.
# --------------------------------------------------------------------------- #
class _DPPolicy:
    def __init__(self, net, scheduler, obs_horizon, pred_horizon, act_dim, device,
                 act_horizon=1, num_inference_steps=100, obs_mean=None, obs_std=None):
        self.net = net.to(device).eval()
        self.scheduler = scheduler
        self.scheduler.set_timesteps(num_inference_steps)
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.act_horizon = act_horizon
        self.act_dim = act_dim
        self.device = device
        self.obs_mean = obs_mean.to(device) if obs_mean is not None else None
        self.obs_std = obs_std.to(device) if obs_std is not None else None
        self.prev = None
        self.action_chunk = None
        self.action_index = 0

    def reset(self):
        self.prev = None
        self.action_chunk = None
        self.action_index = 0

    @torch.no_grad()
    def act(self, obs, deterministic=True):
        cur = (obs["state"] if isinstance(obs, dict) else obs).float().to(self.device)
        if self.obs_mean is not None:
            cur = (cur - self.obs_mean) / self.obs_std
        if self.prev is None or self.prev.shape != cur.shape:
            self.prev = cur
        previous = self.prev
        self.prev = cur

        if self.action_chunk is None or self.action_index >= self.action_chunk.shape[1]:
            hist = [previous, cur][-self.obs_horizon:]
            while len(hist) < self.obs_horizon:
                hist = [hist[0]] + hist
            obs_cond = torch.stack(hist, dim=1).flatten(start_dim=1)
            B = cur.shape[0]
            naction = torch.randn((B, self.pred_horizon, self.act_dim), device=self.device)
            for k in self.scheduler.timesteps:
                noise_pred = self.net(sample=naction, timestep=k, global_cond=obs_cond)
                naction = self.scheduler.step(model_output=noise_pred, timestep=k, sample=naction).prev_sample
            start = self.obs_horizon - 1
            self.action_chunk = naction[:, start:start + self.act_horizon]
            self.action_index = 0

        action = self.action_chunk[:, self.action_index]
        self.action_index += 1
        return action.clamp(-1.0, 1.0)


def load_dp(checkpoint, sample_obs, action_space, device,
            obs_horizon=2, act_horizon=1, pred_horizon=16, diffusion_step_embed_dim=64,
            unet_dims=(64, 128, 256), n_groups=8, num_diffusion_iters=100,
            num_inference_steps=100):
    """Load a state Diffusion Policy checkpoint (uses EMA weights)."""
    _add_baseline_path("diffusion_policy")
    from diffusion_policy.conditional_unet1d import ConditionalUnet1D
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    saved = ckpt.get("model_config", {})
    obs_horizon = saved.get("obs_horizon", obs_horizon)
    pred_horizon = saved.get("pred_horizon", pred_horizon)
    diffusion_step_embed_dim = saved.get("diffusion_step_embed_dim", diffusion_step_embed_dim)
    unet_dims = saved.get("unet_dims", unet_dims)
    n_groups = saved.get("n_groups", n_groups)
    num_diffusion_iters = saved.get("num_diffusion_iters", num_diffusion_iters)

    state = sample_obs["state"] if isinstance(sample_obs, dict) else sample_obs
    obs_dim = state.shape[1]
    act_dim = action_space.shape[0]
    net = ConditionalUnet1D(
        input_dim=act_dim, global_cond_dim=obs_horizon * obs_dim,
        diffusion_step_embed_dim=diffusion_step_embed_dim,
        down_dims=list(unet_dims), n_groups=n_groups,
    )
    sd = ckpt.get("ema_agent", ckpt.get("agent"))
    net_sd = {k.replace("noise_pred_net.", "", 1): v for k, v in sd.items()
              if k.startswith("noise_pred_net.")}
    net.load_state_dict(net_sd)
    scheduler = DDPMScheduler(num_train_timesteps=num_diffusion_iters,
                              beta_schedule="squaredcos_cap_v2", clip_sample=True,
                              prediction_type="epsilon")
    return _DPPolicy(
        net, scheduler, obs_horizon, pred_horizon, act_dim, device, act_horizon,
        num_inference_steps=num_inference_steps,
        obs_mean=ckpt.get("obs_mean"), obs_std=ckpt.get("obs_std"),
    )


def load_dp_exec2(*args, **kwargs):
    kwargs["act_horizon"] = 2
    return load_dp(*args, **kwargs)


def load_dp_exec4(*args, **kwargs):
    kwargs["act_horizon"] = 4
    return load_dp(*args, **kwargs)


def load_dp_exec8(*args, **kwargs):
    kwargs["act_horizon"] = 8
    return load_dp(*args, **kwargs)


# --------------------------------------------------------------------------- #
# RGB Diffusion Policy (OPTIONAL image track) — image + robot proprioception, NO privileged state.
# Same fixed image input shape at every difficulty; same checkpoint runs across configs.
# Template only — image IL is not yet solving this task.
# --------------------------------------------------------------------------- #
class _DPRgbPolicy:
    def __init__(self, agent, obs_horizon, device, num_inference_steps=16):
        self.agent = agent.to(device).eval()
        self.agent.noise_scheduler.set_timesteps(num_inference_steps)
        self.obs_horizon = obs_horizon
        self.device = device
        self.prev = None

    @torch.no_grad()
    def act(self, obs, deterministic=True):
        state = obs["state"].float().to(self.device)
        rgb = obs["rgb"].to(self.device)
        cur = {"state": state, "rgb": rgb}
        if self.prev is None or self.prev["state"].shape != state.shape:
            self.prev = cur
        obs_seq = {
            "state": torch.stack([self.prev["state"], state], dim=1),
            "rgb": torch.stack([self.prev["rgb"], rgb], dim=1),
        }
        self.prev = cur
        aseq = self.agent.get_action(obs_seq)
        return aseq[:, 0].clamp(-1.0, 1.0)


def load_dp_rgb(checkpoint, sample_obs, action_space, device,
                obs_horizon=2, act_horizon=8, pred_horizon=16,
                diffusion_step_embed_dim=64, unet_dims=(64, 128, 256), n_groups=8,
                num_inference_steps=16, visual_encoder="resnet18", num_kp=32):
    """Load an RGB Diffusion Policy checkpoint (vendored train_rgbd; uses EMA weights).

    Template implementation — image IL is not yet solving this task.
    """
    import types
    import numpy as np
    import gymnasium.spaces as spaces
    _add_baseline_path("diffusion_policy")
    from train_rgbd import Agent

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
    )
    agent = Agent(stub, args)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    agent.load_state_dict(ckpt.get("ema_agent", ckpt.get("agent")))
    return _DPRgbPolicy(agent, obs_horizon, device, num_inference_steps=num_inference_steps)
