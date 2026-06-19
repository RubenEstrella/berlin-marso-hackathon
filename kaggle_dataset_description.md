# WarehouseSort — Demonstration Datasets

Expert demonstrations for the **WarehouseSort** image-based pick-and-place challenge: a Franka
Panda robot sorts parcels into the bin matching the **colored tag on each parcel's top face**
(red tag → red bin, blue tag → blue bin), seen only through a fixed scene camera.

## What's included

**200 demonstration episodes per difficulty level**, organized by folder:

| folder | level | parcels | randomization |
|--------|-------|---------|---------------|
| `easy/`   | easy   | 2 | fully fixed layout |
| `medium/` | medium | 4 | small position jitter |
| `hard/`   | hard   | 6 | small position + slight orientation jitter; bins may swap sides |

Each folder contains one **ManiSkill trajectory dataset** (a pair — both files are required):

- `trajectory.rgb.pd_ee_delta_pos.physx_cuda.h5` — per-step **observations + actions**
- `trajectory.rgb.pd_ee_delta_pos.physx_cuda.json` — episode metadata (control mode, etc.)

**Observation** (per step): a fixed third-person scene-camera image `rgb` `(128, 128, 3)` uint8
plus robot **proprioception** `state` `(26,)` (joint/TCP state) — **no** privileged parcel or bin
coordinates. **Action**: `pd_ee_delta_pos`, 4 dims in `[-1, 1]` (end-effector Δxyz + gripper).

## How they were generated (ManiSkill 3)

A deterministic **scripted waypoint policy** solves the task in the GPU-accelerated
[ManiSkill 3](https://maniskill.readthedocs.io/en/latest/) simulator. We **record** each rollout
with ManiSkill's `RecordEpisode`, then **replay** it (`replay_trajectory`) to re-render the
scene-camera RGB observations — the standard ManiSkill *record → replay* pipeline. Trajectories
are clean (no action noise) and each episode ends the moment all parcels are correctly sorted.

> The scripted policy reads privileged simulator state to drive the arm — it is the **data
> generator only**. Submitted policies must act from the observation (image + proprioception).

## How to use them (imitation learning)

Train an **image policy** (e.g. a Diffusion Policy) by behavior cloning: predict the action
sequence from the recent `rgb` + `state` observations. The provided baseline loads these `.h5`
files directly and trains an RGB Diffusion Policy. Because the observation has the **same shape
at every difficulty**, a single trained policy can be evaluated on easy, medium, and hard.

See the challenge repository for the full training + evaluation pipeline.
