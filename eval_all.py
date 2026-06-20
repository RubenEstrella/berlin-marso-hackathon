"""Evaluate one checkpoint on every difficulty and report the weighted score."""

import argparse
import os
import re
import subprocess
import sys


WEIGHTS = {"easy": 0.2, "medium": 0.3, "hard": 0.5}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--policy", default="warehouse_sort.il_policy:load_dp_rgb")
    parser.add_argument("--eval-config", default="conf/eval/default.yaml")
    parser.add_argument("overrides", nargs="*", help="additional Hydra overrides")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        parser.error(f"checkpoint not found: {args.checkpoint}")

    accuracies = {}
    for level in WEIGHTS:
        cmd = [
            sys.executable, "eval.py", f"difficulty={level}",
            f"policy={args.policy}", f"checkpoint={args.checkpoint}",
            f"eval_config={args.eval_config}", *args.overrides,
        ]
        result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        print(result.stdout, end="")
        if result.returncode:
            raise SystemExit(result.returncode)
        match = re.search(r"SORT ACCURACY:\s+([0-9.]+)\s*%", result.stdout)
        if not match:
            raise RuntimeError(f"could not read sort accuracy for {level}")
        accuracies[level] = float(match.group(1)) / 100.0

    score = sum(WEIGHTS[level] * accuracies[level] for level in WEIGHTS)
    print("\nWeighted score")
    for level in WEIGHTS:
        print(f"  {level:6s}: {accuracies[level]:.3f} x {WEIGHTS[level]:.1f}")
    print(f"  final : {score:.3f}")


if __name__ == "__main__":
    main()
