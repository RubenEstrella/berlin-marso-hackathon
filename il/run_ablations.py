"""Launch reproducible keypoint-count training and execution-horizon evaluation."""

import argparse
import os
import subprocess
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(cmd):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)

    train = sub.add_parser("train", help="train num_kp ablations on balanced data")
    train.add_argument("--num-kp", type=int, nargs="+", default=[32, 64])
    train.add_argument("--total-iters", type=int, default=30000)
    train.add_argument("--extra", nargs="*", default=[])

    evaluate = sub.add_parser("eval", help="compare execution horizons")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--horizons", type=int, nargs="+", default=[2, 4, 8])
    evaluate.add_argument("--eval-config", default="conf/eval/default.yaml")
    args = parser.parse_args()

    if args.mode == "train":
        for num_kp in args.num_kp:
            run([
                sys.executable, "il/train.py", "method=dp_rgb", "demo_dir=all",
                f"flags.num_kp={num_kp}", f"flags.total_iters={args.total_iters}",
                f"flags.exp_name=ablation_kp{num_kp}",
                "flags.unique_run_name=false", *args.extra,
            ])
        return

    policies = {
        2: "warehouse_sort.il_policy:load_dp_rgb_exec2",
        4: "warehouse_sort.il_policy:load_dp_rgb",
        8: "warehouse_sort.il_policy:load_dp_rgb_exec8",
    }
    for horizon in args.horizons:
        if horizon not in policies:
            parser.error("supported execution horizons are 2, 4, and 8")
        print(f"\n=== execution_horizon={horizon} ===", flush=True)
        run([
            sys.executable, "eval_all.py", "--checkpoint", args.checkpoint,
            "--policy", policies[horizon], "--eval-config", args.eval_config,
        ])


if __name__ == "__main__":
    main()
