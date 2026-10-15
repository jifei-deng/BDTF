"""Check the installation before training.

    python check_setup.py            # rolling task + D4RL
    python check_setup.py --rolling  # rolling task only (no MuJoCo needed)
"""

import argparse
import sys

import numpy as np


def check_rolling():
    from bdtf.config import load_config
    from bdtf.envs import make_task
    from bdtf.evaluation import evaluate

    cfg = load_config("rolling")
    env, data = make_task(cfg, seed=0)
    # Reproduces the measurement that defines the zero of the score scale:
    # uniformly random moves, 50 episodes, seed 0 -> score 0.00.
    rng = np.random.RandomState(0)
    stats = evaluate(lambda o: rng.uniform(-1, 1, size=env.action_dim).astype(np.float32), env,
                     episodes=50, horizon=cfg["eval"]["horizon"], seed=0)
    score = env.get_normalized_score(stats["return"])
    status = "ok" if abs(score) < 0.05 else "WARNING: expected 0.00"
    print(f"[{status}] rolling: dataset {len(data['states'])} transitions, observation "
          f"{env.state_dim}-dim, action {env.action_dim}-dim; random actuator moves score "
          f"{score:.2f} (reference value 0.00)")


def check_d4rl():
    from bdtf.config import load_config
    from bdtf.envs import make_task

    for task in ("d4rl-halfcheetah", "d4rl-hopper", "d4rl-walker2d"):
        cfg = load_config(task)
        env, data = make_task(cfg, seed=0)
        env.reset()
        env.step(np.zeros(env.action_dim))
        print(f"[ok] {task}: dataset {len(data['states'])} transitions, "
              f"environment {env.env.spec.id} (state {env.state_dim}, action {env.action_dim})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rolling", action="store_true", help="check the rolling task only")
    args = p.parse_args()

    import torch

    print(f"python {sys.version.split()[0]}, torch {torch.__version__}, numpy {np.__version__}, "
          f"CUDA available: {torch.cuda.is_available()}")
    check_rolling()
    if not args.rolling:
        check_d4rl()
    print("setup complete")


if __name__ == "__main__":
    main()
