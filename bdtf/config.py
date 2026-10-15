"""Configuration: the shared BDTF defaults plus the file of the selected task.

    configs/default.yaml          hyper-parameters shared by all tasks
    configs/<task>.yaml           task definition, training/evaluation protocol
                                  and the BDTF hyper-parameters of that task

Selecting a task with --task therefore selects all of its hyper-parameters;
nothing else has to be set.
"""

from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
TASKS = ("d4rl-halfcheetah", "d4rl-hopper", "d4rl-walker2d", "rolling")


def _merge(base, extra):
    for key, value in (extra or {}).items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = value
    return base


def _read(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(task, overrides=()):
    if task not in TASKS:
        raise ValueError(f"unknown task '{task}'; choose one of {TASKS}")
    cfg = _merge(_read(CONFIG_DIR / "default.yaml"), _read(CONFIG_DIR / f"{task}.yaml"))
    for item in overrides or ():               # optional: section.key=value
        key, _, value = item.partition("=")
        node = cfg
        *parents, leaf = key.split(".")
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = yaml.safe_load(value)
    return cfg
