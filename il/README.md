# Imitation Learning — WarehouseSort

This is an **image** challenge. The IL pipeline follows ManiSkill 3's standard approach:
**demos → train RGB Diffusion Policy → evaluate via eval.py**. The RGB DP is provided as a
runnable **template** — it does not yet solve the task; making an image policy work is the point.

---

## Step 1 — Demonstrations (provided)

**You don't need to record anything.** We provide pre-recorded **rgb** demos for every level
(**200 episodes per level**) as the [Kaggle competition
data](https://www.kaggle.com/competitions/marso-hack-berlin-2026-robot-parcel-sorting-challenge/data).

- **On Kaggle**: the competition data is mounted under `/kaggle/input/` automatically.
- **Elsewhere**: fetch it once (join the competition + set a Kaggle API token first):

```bash
pixi run python il/download_demos.py
```

Either way the files are staged into `il/demos/<level>/`. **Each dataset is a pair** — the
`trajectory.rgb.pd_ee_delta_pos.physx_cuda.h5` (data) **and** the matching
`trajectory.rgb.pd_ee_delta_pos.physx_cuda.json` (control-mode metadata). Both are required and
must sit together; the trainer finds the `.json` next to the `.h5`.

> ⚠️ The demos are recorded by rolling out a **scripted waypoint policy**
> (`examples/scripted_policy.py`). Using it to *collect data* is exactly how we built these
> datasets and is fine. **Submitting a scripted / hard-coded controller is not allowed and
> leads to disqualification** — your submitted policy must act from the observation.

### Optional — generate more demos

If you want extra data, record more with the same tool. Demos are clean scripted trajectories
(no action noise) and each episode ends the moment all parcels are sorted:

```bash
pixi run python il/gen_demos.py --difficulty easy   --num-episodes 200
pixi run python il/gen_demos.py --difficulty medium --num-episodes 200
pixi run python il/gen_demos.py --difficulty hard   --num-episodes 200
```

**How it works — record → replay → media** (the standard ManiSkill data pipeline):

1. **Record** — rolls the scripted waypoint policy across `--num-episodes` seeds (one env at a
   time) and writes the raw trajectories + env states to `il/demos/<level>/trajectory.h5` via
   ManiSkill's `RecordEpisode`. (The scripted policy reads privileged sim state to *control* the
   arm; this is only the data generator, not a submittable policy.)
2. **Replay** — runs ManiSkill's `replay_trajectory` to re-execute the recorded actions and
   render fresh **rgb** observations, producing the training-ready dataset the trainer loads:
   `trajectory.rgb.pd_ee_delta_pos.physx_cuda.h5`. Replay runs single-env (GPU) because it
   reproduces each trajectory from recorded env states.
3. **Media** — saves a demo `media/<level>_demo.mp4` and `media/<level>_demo.gif`
   (render + sensor views) so you can eyeball what a successful episode looks like.

Useful flags: `--no-replay` (raw demos only), `--no-media` (skip the mp4/gif), `--base-seed`
(shift the seed range so new demos differ from the provided set). Run
`pixi run python il/gen_demos.py --help` for all options.

---

## Step 2 — Train

```bash
pixi run python il/train.py method=dp_rgb demo_dir=all       # balanced easy + medium + hard
```

`demo_dir=all` samples each difficulty with equal probability, regardless of trajectory
length. To train on one level, pass `demo_dir=<level>`:
```bash
pixi run python il/train.py method=dp_rgb demo_dir=medium
```

Override any hyperparameter on the CLI:
```bash
pixi run python il/train.py method=dp_rgb flags.total_iters=50000 flags.eval_freq=5000
```

Checkpoints land at `il/baselines/diffusion_policy/runs/<run_name>/checkpoints/`. Run names
include the seed and timestamp by default, preventing repeated experiments from overwriting
checkpoints and TensorBoard logs.

**Datasets in a custom location** (e.g. the mounted Kaggle competition data, not `il/demos/`): pass `demo_path=`
the full path to the `.h5` — keep the matching `.json` in the same folder:

```bash
pixi run python il/train.py method=dp_rgb \
    demo_path=/kaggle/input/<dataset>/trajectory.rgb.pd_ee_delta_pos.physx_cuda.h5
```

---

## Step 3 — Evaluate

```bash
pixi run python eval.py difficulty=easy \
    policy=warehouse_sort.il_policy:load_dp_rgb \
    checkpoint=il/baselines/diffusion_policy/runs/warehouse_rgb_dp/checkpoints/best_eval_sort_accuracy.pt \
    eval_config=conf/eval/default.yaml
```

The rgb observation has the **same shape at every difficulty**, so one trained checkpoint can be
evaluated on easy, medium, and hard.

Evaluate all levels and calculate the weighted competition score:

```bash
pixi run python eval_all.py --checkpoint <checkpoint.pt>
```

Run keypoint-count and execution-horizon ablations:

```bash
pixi run python il/run_ablations.py train --num-kp 32 64 --total-iters 30000
pixi run python il/run_ablations.py eval --checkpoint <checkpoint.pt> --horizons 2 4 8
```

---

## Results

The provided RGB Diffusion Policy is a **runnable template, not a solution** — at default
settings it does **not** sort the parcels (sort accuracy ≈ 0). Getting an image policy to work
is the challenge.

| method | obs | default iters | approx. train time | status |
|--------|-----|:---:|:---:|---|
| Diffusion Policy | rgb (scene cam) | 30k | ~30–90 min | template — does not yet solve |

(Training time is for a single modern GPU, e.g. Colab T4.)

---

## Technical notes

- **Why Diffusion Policy?** A plain MLP behavior cloner collapses here due to compounding error;
  Diffusion Policy's action chunking helps. (It's still only a starting point — the image
  template does not yet solve the task.)
- **Image input = a single fixed third-person scene camera.** It keeps the whole workspace
  (robot + parcels + bins) in frame the entire episode and has the same shape at any parcel
  count, so one policy can run across difficulties.
- ManiSkill 3.0.1 pip wheel does not ship `examples/baselines`, so the DP baseline is
  vendored in `il/baselines/diffusion_policy/`.
- Set `HDF5_USE_FILE_LOCKING=FALSE` if replay/load races on the just-written `.h5`.
