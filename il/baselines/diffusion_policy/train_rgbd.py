ALGO_NAME = "BC_Diffusion_rgbd_UNet"

import os
import random
import time
import types
from collections import defaultdict
from dataclasses import dataclass, field
from functools import partial
from typing import List, Optional

import gymnasium as gym
from gymnasium.vector.vector_env import VectorEnv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import tyro
from diffusers.optimization import get_scheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from gymnasium import spaces
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import BatchSampler, RandomSampler, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

from diffusion_policy.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.evaluate import evaluate
from diffusion_policy.make_env import make_eval_envs
from diffusion_policy.plain_conv import PlainConv
from diffusion_policy.utils import (IterationBasedBatchSampler,
                                    build_state_obs_extractor, convert_obs,
                                    worker_init_fn)


@dataclass
class Args:
    exp_name: Optional[str] = None
    """the name of this experiment"""
    unique_run_name: bool = True
    """Append seed and timestamp to exp_name so repeated runs never overwrite each other."""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "ManiSkill"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    env_id: str = "PegInsertionSide-v1"
    """the id of the environment"""
    demo_path: str = (
        "demos/PegInsertionSide-v1/trajectory.state.pd_ee_delta_pose.physx_cpu.h5"
    )
    """the path of demo dataset, it is expected to be a ManiSkill dataset h5py format file"""
    num_demos: Optional[int] = None
    """number of trajectories to load from the demo dataset"""
    total_iters: int = 1_000_000
    """total timesteps of the experiment"""
    batch_size: int = 256
    """the batch size of sample from the replay memory"""

    # Diffusion Policy specific arguments
    lr: float = 1e-4
    """the learning rate of the diffusion policy"""
    learning_rate: Optional[float] = None
    """Readable alias for --lr; when set, this value wins."""
    obs_horizon: int = 2  # Seems not very important in ManiSkill, 1, 2, 4 work well
    act_horizon: int = 8  # Seems not very important in ManiSkill, 4, 8, 15 work well
    pred_horizon: int = (
        16  # 16->8 leads to worse performance, maybe it is like generate a half image; 16->32, improvement is very marginal
    )
    diffusion_step_embed_dim: int = 64  # not very important
    unet_dims: List[int] = field(
        default_factory=lambda: [64, 128, 256]
    )  # default setting is about ~4.5M params
    n_groups: int = (
        8  # jigu says it is better to let each group have at least 8 channels; it seems 4 and 8 are simila
    )

    # Environment/experiment specific arguments
    obs_mode: str = "rgb+depth"
    """The observation mode to use for the environment, which dictates what visual inputs to pass to the model. Can be "rgb", "depth", or "rgb+depth"."""
    obs_camera: str = "scene"
    """Accepted for demo-replay compat; WarehouseSort only uses the fixed third-person scene camera."""
    visual_encoder: str = "plain_conv"
    """RGB encoder: "plain_conv" (vendored) or "resnet18" (ResNet18 + SpatialSoftmax keypoints)."""
    num_kp: int = 32
    """SpatialSoftmax keypoints (resnet18 encoder); 2*num_kp coords localise parcels+bins+gripper."""
    input_resolution: int = 160
    """Resize encoder inputs; larger values give SpatialSoftmax a finer feature grid."""
    aux_color_weight: float = 0.1
    """Weight for image-derived red/blue centroid supervision; set to 0 to disable."""
    aug_shift_px: float = 4.0
    """Maximum random image translation during training, in pixels. Set to 0 to disable."""
    aug_brightness: float = 0.10
    """Uniform brightness jitter strength during training. Set to 0 to disable."""
    aug_contrast: float = 0.10
    """Uniform contrast jitter strength during training. Set to 0 to disable."""
    # WarehouseSort scene knobs (so the eval env matches the demos). Defaults = easy.
    num_parcels: int = 2
    max_episode_steps: Optional[int] = None
    """Change the environments' max_episode_steps to this value. Sometimes necessary if the demonstrations being imitated are too short. Typically the default
    max episode steps of environments in ManiSkill are tuned lower so reinforcement learning agents can learn faster."""
    log_freq: int = 1000
    """the frequency of logging the training metrics"""
    eval_freq: int = 5000
    """the frequency of evaluating the agent on the evaluation environments"""
    save_freq: Optional[int] = None
    """the frequency of saving the model checkpoints. By default this is None and will only save checkpoints based on the best evaluation metrics."""
    num_eval_episodes: int = 100
    """the number of episodes to evaluate the agent on"""
    num_eval_envs: int = 10
    """the number of parallel environments to evaluate the agent on"""
    sim_backend: str = "physx_cpu"
    """the simulation backend to use for evaluation environments. can be "cpu" or "gpu"""
    num_dataload_workers: int = 0
    """the number of workers to use for loading the training data in the torch dataloader"""
    control_mode: str = "pd_joint_delta_pos"
    """the control mode to use for the evaluation environments. Must match the control mode of the demonstration dataset."""

    # additional tags/configs for logging purposes to wandb and shared comparisons with other algorithms
    demo_type: Optional[str] = None


def reorder_keys(d, ref_dict):
    out = dict()
    for k, v in ref_dict.items():
        if isinstance(v, dict) or isinstance(v, spaces.Dict):
            out[k] = reorder_keys(d[k], ref_dict[k])
        else:
            out[k] = d[k]
    return out


class SmallDemoDataset_DiffusionPolicy(Dataset):  # Load everything into memory
    def __init__(self, data_path, obs_process_fn, obs_space, include_rgb, include_depth, device, num_traj):
        self.include_rgb = include_rgb
        self.include_depth = include_depth
        from diffusion_policy.utils import load_demo_dataset
        data_paths = data_path if isinstance(data_path, (list, tuple)) else [data_path]
        trajectories = {"observations": [], "actions": []}
        trajectory_sources = []
        for source_idx, path in enumerate(data_paths):
            source = load_demo_dataset(path, num_traj=num_traj, concat=False)
            trajectories["observations"].extend(source["observations"])
            trajectories["actions"].extend(source["actions"])
            trajectory_sources.extend([source_idx] * len(source["actions"]))
            print(f"Loaded source {source_idx + 1}/{len(data_paths)}: {path}", flush=True)
        # trajectories['observations'] is a list of dict, each dict is a traj, with keys in obs_space, values with length L+1
        # trajectories['actions'] is a list of np.ndarray (L, act_dim)
        print("Raw trajectory loaded, beginning observation pre-processing...")

        # Pre-process the observations, make them align with the obs returned by the obs_wrapper
        obs_traj_dict_list = []
        for obs_traj_dict in trajectories["observations"]:
            _obs_traj_dict = (
                reorder_keys(obs_traj_dict, obs_space)
                if obs_space is not None else obs_traj_dict
            )
            _obs_traj_dict = obs_process_fn(_obs_traj_dict)
            if self.include_depth:
                _obs_traj_dict["depth"] = torch.Tensor(
                    _obs_traj_dict["depth"].astype(np.float32)
                ).to(device=device, dtype=torch.float16)
            if self.include_rgb:
                _obs_traj_dict["rgb"] = torch.from_numpy(_obs_traj_dict["rgb"]).to(
                    device
                )  # still uint8
            _obs_traj_dict["state"] = torch.from_numpy(_obs_traj_dict["state"]).to(
                device
            )
            obs_traj_dict_list.append(_obs_traj_dict)
        trajectories["observations"] = obs_traj_dict_list
        self.obs_keys = list(_obs_traj_dict.keys())
        # Pre-process the actions
        for i in range(len(trajectories["actions"])):
            trajectories["actions"][i] = torch.Tensor(trajectories["actions"][i]).to(
                device=device
            )
        print(
            "Obs/action pre-processing is done, start to pre-compute the slice indices..."
        )

        # Pre-compute all possible (traj_idx, start, end) tuples, this is very specific to Diffusion Policy
        if (
            "delta_pos" in args.control_mode
            or args.control_mode == "base_pd_joint_vel_arm_pd_joint_vel"
        ):
            print("Detected a delta controller type, padding with a zero action to ensure the arm stays still after solving tasks.")
            self.pad_action_arm = torch.zeros(
                (trajectories["actions"][0].shape[1] - 1,), device=device
            )
            # to make the arm stay still, we pad the action with 0 in 'delta_pos' control mode
            # gripper action needs to be copied from the last action
        else:
            # NOTE for absolute joint pos control probably should pad with the final joint position action.
            raise NotImplementedError(f"Control Mode {args.control_mode} not supported")
        self.obs_horizon, self.pred_horizon = obs_horizon, pred_horizon = (
            args.obs_horizon,
            args.pred_horizon,
        )
        self.slices = []
        self.slice_sources = []
        num_traj = len(trajectories["actions"])
        total_transitions = 0
        for traj_idx in range(num_traj):
            L = trajectories["actions"][traj_idx].shape[0]
            assert trajectories["observations"][traj_idx]["state"].shape[0] == L + 1
            total_transitions += L

            # |o|o|                             observations: 2
            # | |a|a|a|a|a|a|a|a|               actions executed: 8
            # |p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 16
            pad_before = obs_horizon - 1
            # Pad before the trajectory, so the first action of an episode is in "actions executed"
            # obs_horizon - 1 is the number of "not used actions"
            pad_after = pred_horizon - obs_horizon
            # Pad after the trajectory, so all the observations are utilized in training
            # Note that in the original code, pad_after = act_horizon - 1, but I think this is not the best choice
            traj_slices = [
                (traj_idx, start, start + pred_horizon)
                for start in range(-pad_before, L - pred_horizon + pad_after)
            ]  # slice indices follow convention [start, end)
            self.slices += traj_slices
            self.slice_sources += [trajectory_sources[traj_idx]] * len(traj_slices)

        source_counts = np.bincount(self.slice_sources, minlength=len(data_paths))
        self.sample_weights = [1.0 / source_counts[source] for source in self.slice_sources]
        print(f"Balanced source slice counts: {source_counts.tolist()}", flush=True)

        print(
            f"Total transitions: {total_transitions}, Total obs sequences: {len(self.slices)}"
        )

        self.trajectories = trajectories

    def __getitem__(self, index):
        traj_idx, start, end = self.slices[index]
        L, act_dim = self.trajectories["actions"][traj_idx].shape

        obs_traj = self.trajectories["observations"][traj_idx]
        obs_seq = {}
        for k, v in obs_traj.items():
            obs_seq[k] = v[
                max(0, start) : start + self.obs_horizon
            ]  # start+self.obs_horizon is at least 1
            if start < 0:  # pad before the trajectory
                pad_obs_seq = torch.stack([obs_seq[k][0]] * abs(start), dim=0)
                obs_seq[k] = torch.cat((pad_obs_seq, obs_seq[k]), dim=0)
            # don't need to pad obs after the trajectory, see the above char drawing

        act_seq = self.trajectories["actions"][traj_idx][max(0, start) : end]
        if start < 0:  # pad before the trajectory
            act_seq = torch.cat([act_seq[0].repeat(-start, 1), act_seq], dim=0)
        if end > L:  # pad after the trajectory
            gripper_action = act_seq[-1, -1]  # assume gripper is with pos controller
            pad_action = torch.cat((self.pad_action_arm, gripper_action[None]), dim=0)
            act_seq = torch.cat([act_seq, pad_action.repeat(end - L, 1)], dim=0)
            # making the robot (arm and gripper) stay still
        assert (
            obs_seq["state"].shape[0] == self.obs_horizon
            and act_seq.shape[0] == self.pred_horizon
        )
        return {
            "observations": obs_seq,
            "actions": act_seq,
        }

    def __len__(self):
        return len(self.slices)


class Agent(nn.Module):
    def __init__(self, env: VectorEnv, args: Args):
        super().__init__()
        self.obs_horizon = args.obs_horizon
        self.act_horizon = args.act_horizon
        self.pred_horizon = args.pred_horizon
        assert (
            len(env.single_observation_space["state"].shape) == 2
        )  # (obs_horizon, obs_dim)
        assert len(env.single_action_space.shape) == 1  # (act_dim, )
        assert (env.single_action_space.high == 1).all() and (
            env.single_action_space.low == -1
        ).all()
        # denoising results will be clipped to [-1,1], so the action should be in [-1,1] as well
        self.act_dim = env.single_action_space.shape[0]
        obs_state_dim = env.single_observation_space["state"].shape[1]
        total_visual_channels = 0
        self.include_rgb = "rgb" in env.single_observation_space.keys()
        self.include_depth = "depth" in env.single_observation_space.keys()

        if self.include_rgb:
            total_visual_channels += env.single_observation_space["rgb"].shape[-1]
        if self.include_depth:
            total_visual_channels += env.single_observation_space["depth"].shape[-1]

        visual_feature_dim = 256
        enc = getattr(args, "visual_encoder", "plain_conv")
        if enc in ("resnet18", "resnet50"):
            # ResNet18 + SpatialSoftmax: encodes object/gripper LOCATIONS as keypoint coords
            # (see lerobot_encoder). Best for spatial pick-and-place.
            from diffusion_policy.lerobot_encoder import ResNetSpatialSoftmax
            self.visual_encoder = ResNetSpatialSoftmax(
                in_channels=total_visual_channels, out_dim=visual_feature_dim,
                num_kp=getattr(args, "num_kp", 32), backbone=enc,
            )
            self.normalize_rgb = True
        else:
            # pool_feature_map=False: flatten the full 8x8 conv feature map instead of global
            # max-pooling it to 1x1 (which discards WHERE objects are and makes the policy collapse).
            self.visual_encoder = PlainConv(
                in_channels=total_visual_channels, out_dim=visual_feature_dim, pool_feature_map=False
            )
            self.normalize_rgb = False
        self.aug_shift_px = float(getattr(args, "aug_shift_px", 0.0))
        self.aug_brightness = float(getattr(args, "aug_brightness", 0.0))
        self.aug_contrast = float(getattr(args, "aug_contrast", 0.0))
        self.input_resolution = int(getattr(args, "input_resolution", 128))
        self.aux_color_weight = float(getattr(args, "aux_color_weight", 0.0))
        self.color_head = (
            nn.Linear(visual_feature_dim, 4) if self.aux_color_weight > 0 else None
        )
        self.noise_pred_net = ConditionalUnet1D(
            input_dim=self.act_dim,  # act_horizon is not used (U-Net doesn't care)
            global_cond_dim=self.obs_horizon * (visual_feature_dim + obs_state_dim),
            diffusion_step_embed_dim=args.diffusion_step_embed_dim,
            down_dims=args.unet_dims,
            n_groups=args.n_groups,
        )
        self.num_diffusion_iters = 100
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.num_diffusion_iters,
            beta_schedule="squaredcos_cap_v2",  # has big impact on performance, try not to change
            clip_sample=True,  # clip output to [-1,1] to improve stability
            prediction_type="epsilon",  # predict noise (instead of denoised action)
        )

    def _augment_rgb(self, rgb):
        """Apply one random transform per sequence, shared by all history frames."""
        b, t, c, h, w = rgb.shape
        if self.aug_shift_px > 0:
            theta = torch.eye(2, 3, device=rgb.device, dtype=rgb.dtype)[None].repeat(b, 1, 1)
            shift = torch.empty((b, 2), device=rgb.device).uniform_(
                -self.aug_shift_px, self.aug_shift_px
            )
            theta[:, 0, 2] = 2.0 * shift[:, 0] / max(w - 1, 1)
            theta[:, 1, 2] = 2.0 * shift[:, 1] / max(h - 1, 1)
            theta = theta[:, None].expand(-1, t, -1, -1).reshape(b * t, 2, 3)
            flat = rgb.reshape(b * t, c, h, w)
            grid = F.affine_grid(theta, flat.shape, align_corners=False)
            rgb = F.grid_sample(
                flat, grid, mode="bilinear", padding_mode="border", align_corners=False
            ).reshape(b, t, c, h, w)

        if self.aug_brightness > 0:
            brightness = torch.empty((b, 1, 1, 1, 1), device=rgb.device).uniform_(
                1.0 - self.aug_brightness, 1.0 + self.aug_brightness
            )
            rgb = rgb * brightness
        if self.aug_contrast > 0:
            contrast = torch.empty((b, 1, 1, 1, 1), device=rgb.device).uniform_(
                1.0 - self.aug_contrast, 1.0 + self.aug_contrast
            )
            mean = rgb.mean(dim=(-3, -2, -1), keepdim=True)
            rgb = (rgb - mean) * contrast + mean
        return rgb.clamp(0.0, 1.0)

    @staticmethod
    def _color_centroids(rgb):
        """Return red and blue (x, y) centroids in [-1, 1] from RGB pixels."""
        red = (rgb[:, :, 0] > rgb[:, :, 1] * 1.35) & (rgb[:, :, 0] > rgb[:, :, 2] * 1.2)
        blue = (rgb[:, :, 2] > rgb[:, :, 1] * 1.2) & (rgb[:, :, 2] > rgb[:, :, 0] * 1.35)
        h, w = rgb.shape[-2:]
        ys, xs = torch.meshgrid(
            torch.linspace(-1.0, 1.0, h, device=rgb.device, dtype=rgb.dtype),
            torch.linspace(-1.0, 1.0, w, device=rgb.device, dtype=rgb.dtype),
            indexing="ij",
        )

        def centroid(mask):
            weight = mask.to(rgb.dtype)
            denom = weight.sum(dim=(-2, -1)).clamp_min(1.0)
            return torch.stack(
                ((weight * xs).sum(dim=(-2, -1)) / denom,
                 (weight * ys).sum(dim=(-2, -1)) / denom),
                dim=-1,
            )

        return torch.cat((centroid(red), centroid(blue)), dim=-1)

    def encode_obs(self, obs_seq, eval_mode, return_aux=False):
        color_target = None
        if self.include_rgb:
            rgb = obs_seq["rgb"].float() / 255.0  # (B, obs_horizon, 3*k, H, W)
            if not eval_mode:
                rgb = self._augment_rgb(rgb)
            if return_aux and self.color_head is not None:
                color_target = self._color_centroids(rgb)
            if self.normalize_rgb:
                mean = rgb.new_tensor((0.485, 0.456, 0.406)).view(1, 1, 3, 1, 1)
                std = rgb.new_tensor((0.229, 0.224, 0.225)).view(1, 1, 3, 1, 1)
                rgb = (rgb - mean) / std
            img_seq = rgb
        if self.include_depth:
            depth = obs_seq["depth"].float() / 1024.0  # (B, obs_horizon, 1*k, H, W)
            img_seq = depth
        if self.include_rgb and self.include_depth:
            img_seq = torch.cat([rgb, depth], dim=2)  # (B, obs_horizon, C, H, W), C=4*k
        batch_size = img_seq.shape[0]
        img_seq = img_seq.flatten(end_dim=1)  # (B*obs_horizon, C, H, W)
        if self.normalize_rgb and img_seq.shape[-2:] != (self.input_resolution, self.input_resolution):
            img_seq = F.interpolate(
                img_seq, size=(self.input_resolution, self.input_resolution),
                mode="bilinear", align_corners=False,
            )
        visual_feature = self.visual_encoder(img_seq)  # (B*obs_horizon, D)
        visual_feature = visual_feature.reshape(
            batch_size, self.obs_horizon, visual_feature.shape[1]
        )  # (B, obs_horizon, D)
        feature = torch.cat(
            (visual_feature, obs_seq["state"]), dim=-1
        )  # (B, obs_horizon, D+obs_state_dim)
        feature = feature.flatten(start_dim=1)  # (B, obs_horizon * (D+obs_state_dim))
        if return_aux:
            return feature, visual_feature, color_target
        return feature

    def compute_loss(self, obs_seq, action_seq):
        B = obs_seq["state"].shape[0]

        # observation as FiLM conditioning
        obs_cond, visual_feature, color_target = self.encode_obs(
            obs_seq, eval_mode=False, return_aux=True
        )  # (B, obs_horizon * obs_dim)

        # sample noise to add to actions
        noise = torch.randn((B, self.pred_horizon, self.act_dim), device=device)

        # sample a diffusion iteration for each data point
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (B,), device=device
        ).long()

        # add noise to the clean images(actions) according to the noise magnitude at each diffusion iteration
        # (this is the forward diffusion process)
        noisy_action_seq = self.noise_scheduler.add_noise(action_seq, noise, timesteps)

        # predict the noise residual
        noise_pred = self.noise_pred_net(
            noisy_action_seq, timesteps, global_cond=obs_cond
        )

        loss = F.mse_loss(noise_pred, noise)
        if self.color_head is not None and color_target is not None:
            color_pred = torch.tanh(self.color_head(visual_feature))
            loss = loss + self.aux_color_weight * F.mse_loss(color_pred, color_target)
        return loss

    def get_action(self, obs_seq):
        # init scheduler
        # self.noise_scheduler.set_timesteps(self.num_diffusion_iters)
        # set_timesteps will change noise_scheduler.timesteps is only used in noise_scheduler.step()
        # noise_scheduler.step() is only called during inference
        # if we use DDPM, and inference_diffusion_steps == train_diffusion_steps, then we can skip this

        # obs_seq['state']: (B, obs_horizon, obs_state_dim)
        B = obs_seq["state"].shape[0]
        with torch.no_grad():
            if self.include_rgb:
                obs_seq["rgb"] = obs_seq["rgb"].permute(0, 1, 4, 2, 3)
            if self.include_depth:
                obs_seq["depth"] = obs_seq["depth"].permute(0, 1, 4, 2, 3)

            obs_cond = self.encode_obs(
                obs_seq, eval_mode=True
            )  # (B, obs_horizon * obs_dim)

            # initialize action from Guassian noise
            noisy_action_seq = torch.randn(
                (B, self.pred_horizon, self.act_dim), device=obs_seq["state"].device
            )

            for k in self.noise_scheduler.timesteps:
                # predict noise
                noise_pred = self.noise_pred_net(
                    sample=noisy_action_seq,
                    timestep=k,
                    global_cond=obs_cond,
                )

                # inverse diffusion step (remove noise)
                noisy_action_seq = self.noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=noisy_action_seq,
                ).prev_sample

        # only take act_horizon number of actions
        start = self.obs_horizon - 1
        end = start + self.act_horizon
        return noisy_action_seq[:, start:end]  # (B, act_horizon, act_dim)


def save_ckpt(run_name, tag):
    os.makedirs(f"runs/{run_name}/checkpoints", exist_ok=True)
    ema.copy_to(ema_agent.parameters())
    torch.save(
        {
            "agent": agent.state_dict(),
            "ema_agent": ema_agent.state_dict(),
            "model_config": {
                "obs_horizon": args.obs_horizon,
                "act_horizon": args.act_horizon,
                "pred_horizon": args.pred_horizon,
                "diffusion_step_embed_dim": args.diffusion_step_embed_dim,
                "unet_dims": list(args.unet_dims),
                "n_groups": args.n_groups,
                "visual_encoder": args.visual_encoder,
                "num_kp": args.num_kp,
                "input_resolution": args.input_resolution,
                "aux_color_weight": args.aux_color_weight,
            },
        },
        f"runs/{run_name}/checkpoints/{tag}.pt",
    )


if __name__ == "__main__":
    args = tyro.cli(Args)

    timestamp = int(time.time())
    if args.exp_name is None:
        base_name = os.path.basename(__file__)[: -len(".py")]
    else:
        base_name = args.exp_name
    run_name = (f"{base_name}__seed{args.seed}__{timestamp}"
                if args.unique_run_name else base_name)

    demo_paths = [p.strip() for p in args.demo_path.split(",") if p.strip()]
    if not demo_paths or any(not path.endswith(".h5") for path in demo_paths):
        raise ValueError("--demo-path must be one or more comma-separated .h5 files")
    demo_infos = []
    for demo_path in demo_paths:
        import json

        json_file = demo_path[:-2] + "json"
        with open(json_file, "r") as f:
            demo_info = json.load(f)
            demo_infos.append(demo_info)
            if "control_mode" in demo_info["env_info"]["env_kwargs"]:
                control_mode = demo_info["env_info"]["env_kwargs"]["control_mode"]
            elif "control_mode" in demo_info["episodes"][0]:
                control_mode = demo_info["episodes"][0]["control_mode"]
            else:
                raise Exception("Control mode not found in json")
            assert (
                control_mode == args.control_mode
            ), f"Control mode mismatched. Dataset has control mode {control_mode}, but args has control mode {args.control_mode}"
    # Match the eval env to the demo distribution: pull the WarehouseSort scene kwargs straight
    # from the demo's recorded env_kwargs (num parcels, bins, distance, pose randomisation, camera)
    # so eval renders exactly what the policy was trained on (no manual flag duplication).
    _demo_scene_kwargs = {}
    if demo_infos and args.env_id.startswith("WarehouseSort"):
        # For mixed training, evaluate on the most difficult (last) dataset's scene.
        _dk = demo_infos[-1]["env_info"]["env_kwargs"]
        for _k in ("num_parcels", "fixed_poses", "randomization", "obs_camera"):
            if _k in _dk:
                _demo_scene_kwargs[_k] = _dk[_k]
    assert args.obs_horizon + args.act_horizon - 1 <= args.pred_horizon
    assert args.obs_horizon >= 1 and args.act_horizon >= 1 and args.pred_horizon >= 1

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    offline_training = os.name == "nt"

    # create evaluation environment
    env_kwargs = dict(
        control_mode=args.control_mode,
        reward_mode="sparse",
        obs_mode=args.obs_mode,
        obs_camera=args.obs_camera,
        render_mode="rgb_array",
        human_render_camera_configs=dict(shader_pack="default")
    )
    if args.env_id.startswith("WarehouseSort"):   # match the demo scene (num parcels, poses, rand)
        env_kwargs.update(num_parcels=args.num_parcels)
        env_kwargs.update(_demo_scene_kwargs)      # demo-recorded kwargs win (exact distribution match)
    assert args.max_episode_steps != None, "max_episode_steps must be specified as imitation learning algorithms task solve speed is dependent on the data you train on"
    env_kwargs["max_episode_steps"] = args.max_episode_steps
    other_kwargs = dict(obs_horizon=args.obs_horizon)
    envs = None
    if not offline_training:
        envs = make_eval_envs(
            args.env_id,
            args.num_eval_envs,
            args.sim_backend,
            env_kwargs,
            other_kwargs,
            video_dir=f"runs/{run_name}/videos" if args.capture_video else None,
            wrappers=[FlattenRGBDObservationWrapper],
        )
    else:
        print("[train] Windows native mode: offline training enabled; simulator eval disabled",
              flush=True)

    if args.track:
        import wandb
        config = vars(args)
        config["eval_env_cfg"] = dict(**env_kwargs, num_envs=args.num_eval_envs, env_id=args.env_id, env_horizon=args.max_episode_steps)
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=config,
            name=run_name,
            save_code=True,
            group="DiffusionPolicy",
            tags=["diffusion_policy"],
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    obs_process_fn = partial(
        convert_obs,
        concat_fn=partial(np.concatenate, axis=-1),
        transpose_fn=partial(
            np.transpose, axes=(0, 3, 1, 2)
        ),  # (B, H, W, C) -> (B, C, H, W)
        state_obs_extractor=build_state_obs_extractor(args.env_id),
        depth = any("rgbd" in path for path in demo_paths)
    )

    # create temporary env to get original observation space as AsyncVectorEnv (CPU parallelization) doesn't permit that
    # (use the SAME sim backend as the eval envs so this throwaway env doesn't spin up a CPU PhysX
    # system; eval itself runs on args.sim_backend, i.e. GPU here)
    if offline_training:
        orignal_obs_space = None
        include_rgb = "rgb" in args.obs_mode
        include_depth = "depth" in args.obs_mode
    else:
        tmp_env = gym.make(args.env_id, sim_backend=args.sim_backend, **env_kwargs)
        orignal_obs_space = tmp_env.observation_space
        include_rgb = tmp_env.unwrapped.obs_mode_struct.visual.rgb
        include_depth = tmp_env.unwrapped.obs_mode_struct.visual.depth
        tmp_env.close()

    dataset = SmallDemoDataset_DiffusionPolicy(
        data_path=demo_paths,
        obs_process_fn=obs_process_fn,
        obs_space=orignal_obs_space,
        include_rgb=include_rgb,
        include_depth=include_depth,
        device=torch.device("cpu"),
        num_traj=args.num_demos
    )
    if len(demo_paths) > 1:
        sampler = WeightedRandomSampler(
            dataset.sample_weights, num_samples=len(dataset), replacement=True
        )
    else:
        sampler = RandomSampler(dataset, replacement=False)
    batch_sampler = BatchSampler(sampler, batch_size=args.batch_size, drop_last=True)
    batch_sampler = IterationBasedBatchSampler(batch_sampler, args.total_iters)
    train_dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=args.num_dataload_workers,
        worker_init_fn=lambda worker_id: worker_init_fn(worker_id, base_seed=args.seed),
        persistent_workers=(args.num_dataload_workers > 0),
        pin_memory=(device.type == "cuda"),
    )

    if envs is None:
        sample = dataset[0]
        state_dim = sample["observations"]["state"].shape[-1]
        _, channels, height, width = sample["observations"]["rgb"].shape
        action_dim = sample["actions"].shape[-1]
        model_env = types.SimpleNamespace(
            single_observation_space=spaces.Dict({
                "state": spaces.Box(-np.inf, np.inf, (args.obs_horizon, state_dim), np.float32),
                "rgb": spaces.Box(0, 255, (args.obs_horizon, height, width, channels), np.uint8),
            }),
            single_action_space=spaces.Box(-1.0, 1.0, (action_dim,), np.float32),
        )
    else:
        model_env = envs

    agent = Agent(model_env, args).to(device)

    optimizer = optim.AdamW(
        params=agent.parameters(),
        lr=args.learning_rate if args.learning_rate is not None else args.lr,
        betas=(0.95, 0.999), weight_decay=1e-6,
    )

    # Cosine LR schedule with linear warmup
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=args.total_iters,
    )

    # Exponential Moving Average
    # accelerates training and improves stability
    # holds a copy of the model weights
    ema = EMAModel(parameters=agent.parameters(), power=0.75)
    ema_agent = Agent(model_env, args).to(device)

    best_eval_metrics = defaultdict(float)
    timings = defaultdict(float)

    # define evaluation and logging functions
    def evaluate_and_save_best(iteration):
        if iteration % args.eval_freq == 0:
            if envs is None:
                save_ckpt(run_name, "latest")
                print("[train] offline checkpoint saved (simulator eval unavailable on Windows)")
                return
            last_tick = time.time()
            ema.copy_to(ema_agent.parameters())
            eval_metrics = evaluate(
                args.num_eval_episodes, ema_agent, envs, device, args.sim_backend
            )
            timings["eval"] += time.time() - last_tick

            print(f"Evaluated {len(eval_metrics['success_at_end'])} episodes")
            for k in eval_metrics.keys():
                eval_metrics[k] = np.mean(eval_metrics[k])
                writer.add_scalar(f"eval/{k}", eval_metrics[k], iteration)
                print(f"{k}: {eval_metrics[k]:.4f}")

            save_on_best_metrics = ["sort_accuracy", "success_once", "success_at_end"]
            for k in save_on_best_metrics:
                if k in eval_metrics and eval_metrics[k] > best_eval_metrics[k]:
                    best_eval_metrics[k] = eval_metrics[k]
                    save_ckpt(run_name, f"best_eval_{k}")
                    print(
                        f"New best {k}_rate: {eval_metrics[k]:.4f}. Saving checkpoint."
                    )
    def log_metrics(iteration):
        if iteration % args.log_freq == 0:
            writer.add_scalar(
                "charts/learning_rate", optimizer.param_groups[0]["lr"], iteration
            )
            writer.add_scalar("losses/total_loss", total_loss.item(), iteration)
            for k, v in timings.items():
                writer.add_scalar(f"time/{k}", v, iteration)

    # ---------------------------------------------------------------------------- #
    # Training begins.
    # ---------------------------------------------------------------------------- #
    agent.train()
    pbar = tqdm(total=args.total_iters)
    last_tick = time.time()
    for iteration, data_batch in enumerate(train_dataloader):
        timings["data_loading"] += time.time() - last_tick

        # forward and compute loss
        last_tick = time.time()
        data_batch = {
            "observations": {k: v.to(device, non_blocking=True)
                             for k, v in data_batch["observations"].items()},
            "actions": data_batch["actions"].to(device, non_blocking=True),
        }
        total_loss = agent.compute_loss(
            obs_seq=data_batch["observations"],  # obs_batch_dict['state'] is (B, L, obs_dim)
            action_seq=data_batch["actions"],  # (B, L, act_dim)
        )
        timings["forward"] += time.time() - last_tick

        # backward
        last_tick = time.time()
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        lr_scheduler.step()  # step lr scheduler every batch, this is different from standard pytorch behavior
        timings["backward"] += time.time() - last_tick

        # ema step
        last_tick = time.time()
        ema.step(agent.parameters())
        timings["ema"] += time.time() - last_tick

        # Evaluation
        evaluate_and_save_best(iteration)
        log_metrics(iteration)

        # Checkpoint
        if args.save_freq is not None and iteration % args.save_freq == 0:
            save_ckpt(run_name, str(iteration))
        pbar.update(1)
        pbar.set_postfix({"loss": total_loss.item()})
        last_tick = time.time()

    evaluate_and_save_best(args.total_iters)
    log_metrics(args.total_iters)

    if envs is not None:
        envs.close()
    writer.close()
