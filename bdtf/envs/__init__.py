"""Task factory: environment + offline dataset for a task configuration."""


def make_task(cfg, seed):
    task = cfg["task"]
    if task["kind"] == "d4rl":
        from .d4rl import D4RLEnv

        env = D4RLEnv(env=task["env"], level=task["level"], seed=seed + 10000)
    elif task["kind"] == "rolling":
        from .rolling import RollingEnv

        env = RollingEnv(horizon=cfg["eval"]["horizon"], seed=seed + 10000, **task.get("env", {}))
    else:
        raise ValueError(f"unknown task kind: {task['kind']}")
    return env, env.get_dataset()
