# 📦 WarehouseSort — Color-Matching Pick-and-Place Challenge

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/marso-robotics/berlin-marso-hackathon/blob/main/starter.ipynb)

A robotics imitation-learning challenge built on **[ManiSkill 3](https://maniskill.readthedocs.io/en/latest/)**.

A Franka Panda robot must sort parcels by color: pick each parcel from the inbound zone and
place it in the bin that matches the **colored tag on its top face** (red tag → red bin, blue
tag → blue bin). At harder levels the bin positions swap between episodes, so the robot must
read the colors rather than memorize a side.

### 🎬 What a solved episode looks like

The scripted policy (used only to generate the demonstrations) sorting parcels into the
color-matched bins — left panel is the scene view, right panel is the policy's camera:

| easy (2 parcels) | medium (4 parcels) | hard (6 parcels, bins may swap) |
|:---:|:---:|:---:|
| ![easy demo](media/easy_demo.gif) | ![medium demo](media/medium_demo.gif) | ![hard demo](media/hard_demo.gif) |

---

## 🏁 The challenge — image-based sorting

This is an **image challenge**: your policy sees only a fixed scene-camera **RGB image** plus the
robot's **proprioception** (joint/TCP state) — **no** privileged parcel or bin coordinates. It
must read the colors from pixels and act.

A complete **RGB Diffusion Policy pipeline** is provided as a **template** — download demos →
train → eval, runnable end-to-end. **It does not yet solve the task**; getting it to work is the
point of the challenge. **Any approach is welcome** — imitation learning on the provided demos,
reinforcement learning from the sparse `+1` reward, or anything else — as long as your policy
**acts from the observation** (see [SUBMISSION.md](SUBMISSION.md) for the contract).

The harder challenge is to generalize across the difficulty levels and to the unseen layouts
(new positions, bin side assignments) in the held-out configs.

> ⚠️ **Scripted / hard-coded controllers are not valid submissions.** We use a scripted policy
> only to *generate the demonstrations*; submitting one (or any policy that reads privileged env
> state directly instead of acting from the observation) is grounds for **disqualification**.

---

## 🎚️ Difficulty levels

| Level | Parcels | Randomization |
|-------|---------|--------------|
| **easy** | 2 | Fully fixed — same parcel positions and bin sides every episode |
| **medium** | 4 | Small parcel position jitter, fixed orientation, fixed bin sides |
| **hard** | 6 | Small parcel position jitter, slight orientation jitter, and bin sides may swap between episodes |

Held-out evaluation uses the same difficulty levels with different seeds and slightly wider
position randomization — build for generalization, not for the exact training layout.

---

## 🏆 Scoring

### Primary metric — Sort accuracy

**Sort accuracy** is the fraction of parcels placed in the correct-color bin by episode end.
It is the **only metric that determines your ranking**.

The check is **geometric and deterministic**: a parcel is correctly sorted when its body
rests inside the footprint of the matching-color bin and is settled low (below the rim).

### Final score — weighted average across all three levels

| level | weight |
|-------|--------|
| easy | 0.2 |
| medium | 0.3 |
| hard | 0.5 |

```
final_score = 0.2 × sort_accuracy_easy
            + 0.3 × sort_accuracy_medium
            + 0.5 × sort_accuracy_hard
```

Higher weights on harder levels reward generalisation.

---

## 🚀 Quick start

```bash
# 0. Get pixi (package manager)
curl -fsSL https://pixi.sh/install.sh | bash

# 1. Install dependencies
pixi install
pixi run install        # pip install -e .

# 2. Download the demonstrations (rgb datasets — the Kaggle competition data)
pixi run python il/download_demos.py

# 3. Train the RGB Diffusion Policy on the easy demos
pixi run python il/train.py method=dp_rgb demo_dir=easy

# 4. Evaluate
pixi run python eval.py difficulty=easy \
    policy=warehouse_sort.il_policy:load_dp_rgb \
    checkpoint=il/baselines/diffusion_policy/runs/warehouse_rgb_dp/checkpoints/best_eval_sort_accuracy.pt \
    eval_config=conf/eval/default.yaml
```

**Demonstrations are provided** for every level — **200 rgb episodes per level**, as the
[Kaggle competition data](https://www.kaggle.com/competitions/marso-hack-berlin-2026-robot-parcel-sorting-challenge/data).
On Kaggle the data is mounted automatically; elsewhere fetch it with
`pixi run python il/download_demos.py` (join the competition + set a Kaggle API token first).
Either way it stages into `il/demos/<level>/`. Generating your own is optional (see
[il/README.md](il/README.md)).

### 📊 What to expect & training time

The provided RGB Diffusion Policy is a **runnable template, not a solution** — out of the box it
**does not solve the task** (sort accuracy near 0). Getting an image policy to actually sort is
the challenge. Training time at default settings (`total_iters=30000`) is roughly **~30–90 min**
on a single modern GPU (e.g. Colab T4), scaling with iterations and image-encoder size.

One nice property: the rgb observation has a **fixed shape at every difficulty**, so a single
trained checkpoint can be evaluated on easy, medium, and hard (no per-level retraining required —
though training per level can still help).

For a guided walkthrough, open **[starter.ipynb](starter.ipynb)** — or click the badge at the top of this page to launch it directly in Google Colab (select a GPU runtime).

### What you'll submit

You hand us three things (full details in **[SUBMISSION.md](SUBMISSION.md)**):

1. **Your codebase** — a GitHub repo (fork of this one) containing your policy code.
2. **`submission.yaml`** — declares, per level, the checkpoint path.
3. **Your checkpoint(s) + a policy entrypoint** — a `module:function` that loads a checkpoint
   into a policy exposing `act(obs, deterministic=True)` (the provided
   `warehouse_sort.il_policy:load_dp_rgb` already does this).

**To submit:** [**fork**](https://github.com/marso-robotics/berlin-marso-hackathon/fork) this
repo, do all your work in your fork, and send us your fork's GitHub URL — you never push to this
repo. Read **[SUBMISSION.md](SUBMISSION.md)** for exactly how to package and submit your entry.

---

## 💡 Improving the baseline

Concrete things to try with the RGB Diffusion Policy (all via `il/train.py` flags or the loader):

- **Train longer / on more data** — raise `flags.total_iters`; record extra demos with
  `il/gen_demos.py`. Image policies are data-hungry, so this matters a lot here.
- **Visual encoder** — `flags.visual_encoder` (e.g. `resnet18`), `flags.num_kp` (SpatialSoftmax
  keypoints). The perception front-end is the likely bottleneck for an image policy.
- **Horizons** — `flags.pred_horizon` (how many actions are predicted), `flags.act_horizon`
  (how many are executed per inference), `flags.obs_horizon` (image history length).
- **Denoising steps at eval** — `num_inference_steps` in `load_dp_rgb` (more steps → better
  actions, slower inference). Eval-only, so safe to raise.
- **Network capacity** — `unet_dims`, `diffusion_step_embed_dim`.
- **Optimisation** — `flags.batch_size`, learning rate.
- **Generalisation** — the held-out configs use wider positions / more bin-swaps than training.
  Train across that variation (more demos, the hard config) rather than overfitting the exact
  training seeds — hard is weighted 0.5, so this is where the points are.

> ⚠️ If you change an **architecture/horizon** hyperparameter for training (`obs_horizon`,
> `act_horizon`, `pred_horizon`, `unet_dims`, `diffusion_step_embed_dim`, `n_groups`,
> `visual_encoder`, `num_kp`), pass the **same value** to your policy loader (`load_dp_rgb(...)`
> args, or your own `load_fn`) — or the checkpoint won't load. See
> [warehouse_sort/il_policy.py](warehouse_sort/il_policy.py).

For other algorithm ideas (other IL methods, RL), see
**[Where to find more approaches](#where-to-find-more-approaches)**.

---

## 👀 Observation

A fixed third-person **scene camera** plus robot proprioception — **no privileged parcel or bin
state**. With the `FlattenRGBDObservationWrapper` (applied for you), `obs` is a dict:

- `obs["rgb"]`: `(N, 128, 128, 3)` uint8 — the scene image, RGB channel order
- `obs["state"]`: `(N, 26)` float32 — **proprioception only** (joint positions/velocities, TCP
  pose, gripper); contains **no** parcel positions, tag colors, or bin locations

Your policy must read the colors and geometry from the image. The observation has the **same
shape at every difficulty**, so one trained policy can run on easy, medium, and hard.

---

## 🎮 Action space

`pd_ee_delta_pos`, 4 dims in `[-1, 1]`:

| dims | meaning |
|------|---------|
| `[0:3]` | end-effector delta xyz (±0.1 m/step) |
| `[3]` | gripper: +1 = open, −1 = close |

---

## 🎁 Reward

**Sparse only: `+1` per correctly placed parcel.** No dense reward is provided. If you choose
to train with reinforcement learning, designing a shaped reward is your job.

---

## ⚖️ How judging works

A judge evaluates your submission on all three levels using **held-out configs** and reports a
weighted aggregate (see **Scoring** above). The held-out configs use the same difficulty
levels, same colors, and same success check as training — only the seeds and position
randomization ranges differ (slightly wider than training). Evaluation runs the same `eval.py`
interface you have, so your policy must load and run from your submitted code with no changes.

For exactly how to package and submit your entry, see **[SUBMISSION.md](SUBMISSION.md)**.

---

## 📚 References

- **ManiSkill 3** — GPU-accelerated robot simulation: [docs](https://maniskill.readthedocs.io/en/latest/) / [arxiv.org/abs/2410.00425](https://arxiv.org/abs/2410.00425)
- **Diffusion Policy** — Chi et al. 2023: [diffusion-policy.cs.columbia.edu](https://diffusion-policy.cs.columbia.edu)
- The RGB template is built on the ManiSkill IL baselines and **[LeRobot](https://github.com/huggingface/lerobot)** conventions.

### Where to find more approaches

Our RGB Diffusion Policy template comes straight from the **[ManiSkill example
baselines](https://github.com/haosulab/ManiSkill/tree/main/examples/baselines)**. The exact
pipeline used here lives in `il/baselines/diffusion_policy/` — study `train_rgbd.py` (image DP)
to understand and extend it. ManiSkill ships many more reference
implementations (other IL methods, RL, motion planning); browse them in the
[examples](https://github.com/haosulab/ManiSkill/tree/main/examples) and the
[baselines docs](https://maniskill.readthedocs.io/en/latest/user_guide/learning_from_demos/index.html)
for inspiration on how to solve these environments.

---

## 📂 Repo layout

```
warehouse_sort/     # the ManiSkill environment + IL policy entrypoint
  env.py            # WarehouseSort-v1 (register, scene, obs, reward, evaluate)
  il_policy.py      # load_dp_rgb — wire into eval.py via policy=...
  utils.py          # env construction, rollout, metrics printing

conf/               # Hydra configs
  difficulty/       # easy.yaml / medium.yaml / hard.yaml
  eval/default.yaml # same-distribution eval (rehearse the judge interface)

il/                 # imitation learning
  gen_demos.py      # record + replay demonstrations (rgb)
  download_demos.py # fetch the provided rgb demo datasets (from Kaggle)
  train.py          # Hydra dispatcher -> vendored RGB Diffusion Policy trainer
  baselines/        # vendored ManiSkill DP baseline (diffusion_policy only)

examples/
  scripted_policy.py  # deterministic waypoint policy (demo source)

eval.py             # evaluate a checkpoint (same interface as judging)
judge/              # held-out eval configs (not distributed to competitors)
```
