import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("BDTF_D4RL_DIR", ROOT / "data" / "d4rl"))
DATASET_URL = "http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/{name}.hdf5"

GYM_ID = {"halfcheetah": "HalfCheetah-v3", "hopper": "Hopper-v3", "walker2d": "Walker2d-v3"}
REF_SCORES = {
    "halfcheetah": (-280.178953, 12135.0),
    "hopper": (-20.272305, 3234.3),
    "walker2d": (1.629008, 4592.3),
}


def dataset_path(env, level="medium-replay"):
    name = f"{env}_{level.replace('-', '_')}-v2"
    path = DATA_DIR / f"{name}.hdf5"
    if path.exists() and path.stat().st_size > 0:
        return path
    import urllib.request

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    url = DATASET_URL.format(name=name)
    print(f"downloading {url} -> {path}")
    tmp = path.with_suffix(".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(path)
    return path


def qlearning_dataset(path, max_episode_steps=1000):
    path = Path(path)
    npz_path = path.with_name(path.stem + ".qlearning.npz")
    if npz_path.exists():
        with np.load(npz_path) as f:
            return {k: f[k] for k in f.files}

    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
    import h5py

    with h5py.File(path, "r") as f:
        raw = {k: np.asarray(f[k]) for k in ("observations", "actions", "rewards", "terminals")}
        for optional in ("timeouts", "next_observations"):
            if optional in f:
                raw[optional] = np.asarray(f[optional])

    n = raw["rewards"].shape[0]
    terminals = raw["terminals"][: n - 1].astype(bool)
    if "timeouts" in raw:
        timeouts = raw["timeouts"][: n - 1].astype(bool)
    else:
        timeouts = np.zeros(n - 1, dtype=bool)
        step = 0
        for i in range(n - 1):
            if step == max_episode_steps - 1:
                timeouts[i], step = True, 0
            elif terminals[i]:
                step = 0
            else:
                step += 1

    keep = ~timeouts
    next_obs = (raw["next_observations"][: n - 1] if "next_observations" in raw
                else raw["observations"][1:n])[keep]
    out = {
        "states": np.ascontiguousarray(raw["observations"][: n - 1][keep], dtype=np.float32),
        "actions": np.ascontiguousarray(raw["actions"][: n - 1][keep], dtype=np.float32),
        "next_states": np.ascontiguousarray(next_obs, dtype=np.float32),
        "rewards": np.ascontiguousarray(raw["rewards"][: n - 1][keep], dtype=np.float32),
        "not_dones": 1.0 - terminals[keep].astype(np.float32),
    }
    tmp = npz_path.with_name(f"{path.stem}.qlearning.{os.getpid()}.part.npz")
    np.savez(tmp, **out)
    tmp.replace(npz_path)
    return out


class D4RLEnv:
    def __init__(self, env="halfcheetah", level="medium-replay", seed=0):
        import gym

        if env not in GYM_ID:
            raise ValueError(f"unknown D4RL environment '{env}'; expected one of {sorted(GYM_ID)}")
        self.env_name, self.level = env, level
        try:
            self.env = gym.make(GYM_ID[env], disable_env_checker=True)
        except TypeError:
            self.env = gym.make(GYM_ID[env])
        self.action_low = np.asarray(self.env.action_space.low, dtype=np.float32)
        self.action_high = np.asarray(self.env.action_space.high, dtype=np.float32)
        self.state_dim = int(np.prod(self.env.observation_space.shape))
        self.action_dim = int(np.prod(self.env.action_space.shape))
        self.ref_min, self.ref_max = REF_SCORES[env]
        self._seed = seed
        self.seed(seed)

    def seed(self, seed):
        self._seed = int(seed)
        for target in (self.env, getattr(self.env, "action_space", None)):
            if hasattr(target, "seed"):
                try:
                    target.seed(seed)
                except Exception:
                    pass
        return [seed]

    def get_dataset(self):
        return qlearning_dataset(dataset_path(self.env_name, self.level))

    def get_normalized_score(self, episode_return):
        return 100.0 * (episode_return - self.ref_min) / (self.ref_max - self.ref_min)

    def denormalize_action(self, action):
        a = np.clip(np.asarray(action, dtype=np.float32).reshape(-1), -1.0, 1.0)
        return self.action_low + (a + 1.0) * 0.5 * (self.action_high - self.action_low)

    def reset(self):
        try:
            out = self.env.reset(seed=self._seed)
        except TypeError:
            out = self.env.reset()
        return np.asarray(out[0] if isinstance(out, tuple) else out, dtype=np.float32).reshape(-1)

    def step(self, action):
        out = self.env.step(self.denormalize_action(action))
        if len(out) == 5:
            obs, reward, terminated, truncated, info = out
        else:
            obs, reward, terminated, info = out
            truncated = False
        return (np.asarray(obs, dtype=np.float32).reshape(-1), float(reward),
                bool(terminated), bool(truncated), info)
