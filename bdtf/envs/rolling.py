import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
SIM_PATH = ROOT / "rolling_env" / "flatness_simulator.pt"


class EnsembleLinear(nn.Module):
    def __init__(self, n, in_dim, out_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n, in_dim, out_dim))
        self.bias = nn.Parameter(torch.zeros(n, 1, out_dim))
        nn.init.trunc_normal_(self.weight, std=1.0 / (2.0 * np.sqrt(in_dim)))

    def forward(self, x):
        return torch.baddbmm(self.bias, x, self.weight)


class PNNEnsemble(nn.Module):
    def __init__(self, n_models=5, in_dim=8, out_dim=4, hidden=200, n_layers=3):
        super().__init__()
        self.n_models, self.out_dim = n_models, out_dim
        dims = [in_dim] + [hidden] * n_layers
        self.hidden = nn.ModuleList(EnsembleLinear(n_models, i, o)
                                    for i, o in zip(dims[:-1], dims[1:]))
        self.head = EnsembleLinear(n_models, hidden, 2 * out_dim)
        self.max_logvar = nn.Parameter(torch.full((1, 1, out_dim), 0.5))
        self.min_logvar = nn.Parameter(torch.full((1, 1, out_dim), -10.0))
        self.register_buffer("target_mean", torch.zeros(out_dim))
        self.register_buffer("target_std", torch.ones(out_dim))

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0).expand(self.n_models, -1, -1)
        for layer in self.hidden:
            x = F.silu(layer(x))
        mean, logvar = self.head(x).chunk(2, dim=-1)
        logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)
        return mean, logvar

    @torch.no_grad()
    def predict(self, s_z, a_z):
        mean, logvar = self(torch.cat([a_z, s_z], dim=-1))
        delta = mean * self.target_std + self.target_mean
        std = torch.exp(0.5 * logvar) * self.target_std
        return s_z.unsqueeze(0) + delta, std


def _load_simulator(path=SIM_PATH):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


class RollingEnv:
    def __init__(self, horizon=100, seed=0):
        sim = _load_simulator()
        self.model = PNNEnsemble(**sim["model_config"])
        self.model.load_state_dict(sim["state_dict"])
        self.model.eval()
        self.n_models = self.model.n_models

        def arr(t):
            return t.detach().cpu().numpy().astype(np.float32)

        self.s_mean, self.s_std = arr(sim["s_mean"]), arr(sim["s_std"])
        self.a_low, self.a_high = arr(sim["action_low"]), arr(sim["action_high"])
        self.data_states = arr(sim["start_states"])
        self.data_actions = arr(sim["start_actions"])
        self.rate = arr(sim["rate"])
        self.r_random, self.r_ref = float(sim["r_random"]), float(sim["r_ref"])
        self.horizon = int(horizon)

        self.state_dim, self.action_dim = 8, 4
        self.action_low, self.action_high = -self.rate, self.rate
        self.rng = np.random.RandomState(seed)
        self.s, self.a_prev, self.t = None, None, 0

    def seed(self, seed):
        self.rng = np.random.RandomState(seed)
        return [seed]

    def get_dataset(self):
        import pickle

        with open(ROOT / "data" / "rolling" / "rolling_offline.pickle", "rb") as f:
            return pickle.load(f)

    def to_raw_state(self, s_z):
        return np.asarray(s_z, dtype=np.float32) * self.s_std + self.s_mean

    def reward_batch(self, s_z):
        return -np.abs(self.to_raw_state(s_z)).sum(axis=-1)

    def metric_state(self, obs):
        return self.to_raw_state(np.asarray(obs, dtype=np.float32)[..., :4])

    def _applied_action(self, action):
        a = np.clip(np.asarray(action, dtype=np.float32).reshape(-1), -1.0, 1.0)
        move = (self.action_low + (a + 1.0) * 0.5 * (self.action_high - self.action_low)).astype(np.float32)
        return np.clip(self.a_prev + move, self.a_low, self.a_high).astype(np.float32)

    def _obs(self):
        return np.concatenate([self.s, self.a_prev]).astype(np.float32)

    def reset(self):
        k = self.rng.randint(1, len(self.data_states))
        self.s = self.data_states[k].copy()
        self.a_prev = self.data_actions[k - 1].copy()
        self.t = 0
        return self._obs()

    def step(self, action):
        a_z = self._applied_action(action)
        mean, std = self.model.predict(torch.as_tensor(self.s[None], dtype=torch.float32),
                                       torch.as_tensor(a_z[None], dtype=torch.float32))
        mean, std = mean[:, 0].numpy(), std[:, 0].numpy()
        nxt, sd = mean.mean(0), std.mean(0)
        nxt = (nxt + sd * self.rng.randn(*nxt.shape)).astype(np.float32)
        reward = float(self.reward_batch(nxt))
        self.s, self.a_prev = nxt, a_z
        self.t += 1
        return self._obs(), reward, False, self.t >= self.horizon, {}

    def get_normalized_score(self, episode_return):
        per_step = float(episode_return) / self.horizon
        return 100.0 * (per_step - self.r_random) / (self.r_ref - self.r_random)
