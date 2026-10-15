"""Closed-loop evaluation of a policy, plus flatness-control metrics."""

import numpy as np


def control_metrics(flatness, actions):
    """flatness: (T, 4) physical flatness; actions: (T, A) normalised moves."""
    s = np.asarray(flatness, dtype=np.float64)
    a = np.asarray(actions, dtype=np.float64)
    return {
        "RMSE": float(np.sqrt((s ** 2).mean())),
        "max_abs_dev": float(np.abs(s).max()),
        "in_tolerance": float((np.abs(s) <= 1.0).mean()),
        "move_variation": float(np.abs(np.diff(a, axis=0)).sum()) if len(a) > 1 else 0.0,
    }


def rollout(policy_fn, env, horizon):
    obs = env.reset()
    total, flat, acts = 0.0, [], []
    metric_state = getattr(env, "metric_state", None)
    for _ in range(horizon):
        action = np.asarray(policy_fn(obs), dtype=np.float32).reshape(-1)
        obs, reward, done, truncated, _ = env.step(action)
        total += reward
        acts.append(action)
        if metric_state is not None:
            flat.append(metric_state(obs))
        if done or truncated:
            break
    out = {"return": float(total), "length": len(acts)}
    if flat:
        out.update(control_metrics(np.array(flat), np.array(acts)))
    return out


def evaluate(policy_fn, env, episodes, horizon, seed):
    """Episode i is run with env.seed(seed + i), so every evaluation of a run
    uses the same start states and disturbance realisations."""
    runs = []
    for i in range(episodes):
        env.seed(seed + i)
        runs.append(rollout(policy_fn, env, horizon))
    out = {k: float(np.mean([r[k] for r in runs])) for k in runs[0]}
    returns = np.array([r["return"] for r in runs])
    out["return_std"] = float(returns.std())
    out["episode_returns"] = returns.tolist()
    return out
