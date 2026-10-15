"""Train and evaluate BDTF on one task.

    python train.py --task d4rl-halfcheetah --seed 10
    python train.py --task d4rl-hopper      --seed 10-100-1000-10000-100000
    python train.py --task d4rl-walker2d    --seed 10
    python train.py --task rolling          --seed 20

The task selects its dataset, environment, training/evaluation protocol and all
BDTF hyper-parameters (configs/default.yaml + configs/<task>.yaml).

Each run writes results/<task>-seed<seed>-<time>/:
    config.json     the full configuration that was used
    progress.csv    one row per evaluation (return, normalized score, diagnostics)
    summary.json    final score = mean normalized score of the last 10 evaluations
    models/final.pt the trained networks
"""

import argparse
import csv
import json
import random
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from bdtf.agent import BDTF
from bdtf.buffer import ReplayBuffer
from bdtf.config import TASKS, load_config
from bdtf.envs import make_task
from bdtf.evaluation import evaluate


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_seeds(text):
    return [int(p) for chunk in str(text).split(",") for p in chunk.split("-") if p.strip()]


def run(task, cfg, seed, device, out_root):
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(out_root) / f"{task}-seed{seed}-{stamp}"
    (run_dir / "models").mkdir(parents=True, exist_ok=True)

    set_seed(seed)
    torch.set_num_threads(1)
    env, data = make_task(cfg, seed)
    buffer = ReplayBuffer(env.state_dim, env.action_dim,
                          n_critics=cfg["agent"]["n_critics"],
                          bootstrap_p=cfg["agent"]["bootstrap_p"],
                          device=device, seed=seed).load(data)
    buffer.normalize_actions(env.action_low, env.action_high)
    buffer.finalize()
    agent = BDTF(env.state_dim, env.action_dim, device=device, **cfg["agent"])

    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump({"task": task, "seed": seed, "device": device, "config": cfg,
                   "dataset": buffer.stats()}, f, indent=2)
    print(f"\n=== {run_dir.name} ===\n    dataset {buffer.stats()}")

    t0 = time.perf_counter()
    recon = agent.pretrain_behavior(buffer, cfg["train"]["pretrain_steps"], cfg["train"]["batch_size"])
    print(f"    behaviour model: {cfg['train']['pretrain_steps']} steps, "
          f"reconstruction error {recon:.4f} ({time.perf_counter() - t0:.0f}s)")

    def policy_fn(obs):
        return agent.select_action(obs, deterministic=True)

    csv_file = open(run_dir / "progress.csv", "w", newline="", encoding="utf-8")
    writer, scores, running = None, [], defaultdict(list)
    wall0, t_train = time.perf_counter(), 0.0
    steps, every = cfg["train"]["steps"], cfg["train"]["eval_every"]
    for step in range(1, steps + 1):
        t1 = time.perf_counter()
        info = agent.train_step(buffer, cfg["train"]["batch_size"])
        t_train += time.perf_counter() - t1
        for key, value in info.items():
            running[key].append(value)

        if step % every == 0:
            stats = evaluate(policy_fn, env, cfg["eval"]["episodes"], cfg["eval"]["horizon"], seed)
            score = env.get_normalized_score(stats["return"])
            scores.append(score)
            row = {"step": step, "normalized_score": score,
                   "return_mean": stats["return"], "return_std": stats["return_std"]}
            for key in ("RMSE", "max_abs_dev", "in_tolerance", "move_variation"):
                if key in stats:
                    row[key] = stats[key]
            row.update({f"train_{k}": float(np.mean(v)) for k, v in running.items()})
            row.update({"ms_per_step": 1000.0 * t_train / step,
                        "wall_time_s": time.perf_counter() - wall0})
            running.clear()
            if writer is None:
                writer = csv.DictWriter(csv_file, fieldnames=list(row))
                writer.writeheader()
            writer.writerow({k: row.get(k, "") for k in writer.fieldnames})
            csv_file.flush()
            print(f"    step {step:>7d}  normalized score {score:7.2f}  "
                  f"return {stats['return']:10.2f}  ({row['ms_per_step']:.2f} ms/step)")
    csv_file.close()

    agent.save(run_dir / "models" / "final.pt")
    final = float(np.mean(scores[-10:])) if scores else float("nan")
    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump({"task": task, "seed": seed, "final_score_last10": final,
                   "evaluations": len(scores), "train_minutes": t_train / 60.0}, f, indent=2)
    print(f"    final score (mean of the last 10 evaluations): {final:.2f}")
    return final


def main():
    p = argparse.ArgumentParser(description="BDTF: offline training and evaluation")
    p.add_argument("--task", required=True, choices=TASKS)
    p.add_argument("--seed", default="10", help="one seed (10) or several joined by '-' (10-100-1000)")
    p.add_argument("--device", default="auto", help="auto | cuda | cpu")
    p.add_argument("--out", default="results")
    p.add_argument("--quick", action="store_true",
                   help="short functional check (a few thousand steps); not the paper protocol")
    p.add_argument("--set", nargs="*", default=[], dest="overrides", metavar="KEY=VALUE",
                   help="optional overrides, e.g. agent.bias_c=0 (not needed to reproduce the paper)")
    args = p.parse_args()

    overrides = list(args.overrides)
    if args.quick:
        overrides += ["train.steps=2000", "train.eval_every=1000",
                      "train.pretrain_steps=500", "eval.episodes=2"]
    cfg = load_config(args.task, overrides)
    cfg["agent"]["max_steps"] = cfg["train"]["steps"]     # length of the cosine actor-lr schedule
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    seeds = parse_seeds(args.seed)
    print(f"task={args.task}  seeds={seeds}  device={device}")
    print(f"steps={cfg['train']['steps']}  eval_every={cfg['train']['eval_every']}  "
          f"episodes={cfg['eval']['episodes']}  agent={cfg['agent']}")
    finals = [run(args.task, cfg, s, device, args.out) for s in seeds]
    if len(finals) > 1:
        print(f"\n{args.task}: {np.mean(finals):.2f} +/- {np.std(finals):.2f} over {len(finals)} seeds")


if __name__ == "__main__":
    main()
