"""Truncate a replayed trajectory dataset (h5 + json) to EXACTLY N trajectories.

The scripted demos are recorded with a margin and replayed; replay keeps only the episodes that
reproduce a full success, so the saved count overshoots the target slightly. This trims the
dataset to exactly N (keeps the first N traj groups + the first N json episodes).

  python il/_truncate_demos.py <path-to-trajectory.rgb.*.h5> <N>
"""
import json
import os
import sys

import h5py


def truncate(h5_path, n):
    json_path = h5_path[:-2] + "json"   # ".h5" -> ".json" (matches the trainer's lookup)
    with h5py.File(h5_path, "r") as f:
        keys = sorted((k for k in f.keys()), key=lambda x: int(x.split("_")[-1]))
        if len(keys) < n:
            sys.exit(f"only {len(keys)} trajectories saved (< {n}); generate more raw episodes")
        tmp = h5_path + ".tmp"
        with h5py.File(tmp, "w") as g:
            for ak, av in f.attrs.items():
                g.attrs[ak] = av
            for k in keys[:n]:
                f.copy(k, g)
    os.replace(tmp, h5_path)

    with open(json_path) as fh:
        data = json.load(fh)
    data["episodes"] = data["episodes"][:n]
    with open(json_path, "w") as fh:
        json.dump(data, fh)

    print(f"[truncate] {h5_path} -> {n} trajectories (json episodes -> {n})", flush=True)


if __name__ == "__main__":
    truncate(sys.argv[1], int(sys.argv[2]))
