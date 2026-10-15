"""Steel-strip flatness control: simulator of the finishing-mill flatness loop.

Data and units
    The offline dataset (data/rolling/rolling_offline.pickle) consists of 3412
    consecutive transitions logged on the mill under its incremental PI
    controller.  Flatness s (4 channels) and actuator settings a (4 channels)
    are standardised with one scaler each (stored in the simulator checkpoint):
        s_z = (s - s_mean) / s_std,   a_z = (a - a_mean) / a_std.
    Reward: r_t = -sum_i |s_{t+1,i}| in physical flatness units.

MDP (the formulation of the incremental PI loop)
    observation  [s_t, a_{t-1}]                       8-dim, standardised
    action       Δa_t in [-1, 1]^4, scaled per channel by the largest step
                 change of that actuator in the data (actuator rate limit)
    applied      a_t = clip(a_{t-1} + Δa_t, range of the data)
    episode      `horizon` steps (100), no termination; start states are
                 (s_t, a_{t-1}) pairs drawn from the dataset

Dynamics
    An ensemble of probabilistic networks fitted to the mill data predicts the
    next standardised flatness from [a_t, s_t].  The next state is the ensemble
    mean plus `noise_scale` times the predicted (calibrated) aleatoric standard
    deviation; noise_scale = 1 is the setting of the paper.

Normalised score
    100 * (r_bar - r_random) / (r_ref - r_random), with r_bar the average
    per-step reward of a policy; r_random is the per-step reward of uniformly
    random actuator moves and r_ref that of a one-step greedy reference
    controller restricted to moves seen in the data (both measured once on
    this simulator and stored in the checkpoint).  The reference only fixes
    the scale; it does not bound the score from above.
"""

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
        # Overwritten by the checkpoint, but kept: it draws from the global
        # torch RNG exactly as the code that produced the paper's results, so
        # that the agent's networks are initialised identically for a seed.
        nn.init.trunc_normal_(self.weight, std=1.0 / (2.0 * np.sqrt(in_dim)))

    def forward(self, x):                       # (n, B, in) -> (n, B, out)
        return torch.baddbmm(self.bias, x, self.weight)


class PNNEnsemble(nn.Module):
    """Ensemble of Gaussian MLPs: [a_z, s_z] -> mean and log-variance of the
    (normalised) flatness increment."""

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
        """(B, 4), (B, 4) -> next-state mean and aleatoric std, both (n, B, 4)."""
        mean, logvar = self(torch.cat([a_z, s_z], dim=-1))
        delta = mean * self.target_std + self.target_mean
        std = torch.exp(0.5 * logvar) * self.target_std
        return s_z.unsqueeze(0) + delta, std


def _load_simulator(path=SIM_PATH):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:                           # torch < 1.13
        return torch.load(path, map_location="cpu")


class RollingEnv:
    """reset_mode: data (default; dataset start states) | box (state uniform in
                   the [min, max] range of the data, per channel)
       member:     mean (default; ensemble-mean dynamics) | random (one random
                   ensemble member per step) | k (always member k)
       noise_scale: multiple of the predicted aleatoric noise (default 1)."""

    def __init__(self, horizon=100, noise_scale=1.0, member="mean", reset_mode="data", seed=0):
        sim = _load_simulator()
        self.model = PNNEnsemble(**sim["model_config"])
        self.model.load_state_dict(sim["state_dict"])
        self.model.eval()
        self.n_models = self.model.n_models

        def arr(t):
            return t.detach().cpu().numpy().astype(np.float32)

        self.s_mean, self.s_std = arr(sim["s_mean"]), arr(sim["s_std"])
        self.a_low, self.a_high = arr(sim["action_low"]), arr(sim["action_high"])   # actuator range
        self.state_low, self.state_high = arr(sim["state_low"]), arr(sim["state_high"])
        self.data_states = arr(sim["start_states"])
        self.data_actions = arr(sim["start_actions"])
        self.rate = arr(sim["rate"])
        self.r_random, self.r_ref = float(sim["r_random"]), float(sim["r_ref"])
        self.ref_noise_scale = float(sim["ref_noise_scale"])

        if reset_mode not in ("data", "box"):
            raise ValueError(f"unknown reset_mode: {reset_mode}")
        if not (member in ("mean", "random") or (isinstance(member, int) and 0 <= member < self.n_models)):
            raise ValueError(f"member must be 'mean', 'random' or 0..{self.n_models - 1}")
        self.horizon = int(horizon)
        self.noise_scale = float(noise_scale)
        self.member = member
        self.reset_mode = reset_mode

        self.state_dim, self.action_dim = 8, 4
        self.action_low, self.action_high = -self.rate, self.rate   # the agent's Δa box
        self.rng = np.random.RandomState(seed)
        self.s, self.a_prev, self.t = None, None, 0

    # ------------------------------------------------------------------ #
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
        """Physical flatness of an observation (for the control metrics)."""
        return self.to_raw_state(np.asarray(obs, dtype=np.float32)[..., :4])

    def _applied_action(self, action):
        a = np.clip(np.asarray(action, dtype=np.float32).reshape(-1), -1.0, 1.0)
        move = (self.action_low + (a + 1.0) * 0.5 * (self.action_high - self.action_low)).astype(np.float32)
        return np.clip(self.a_prev + move, self.a_low, self.a_high).astype(np.float32)

    def _obs(self):
        return np.concatenate([self.s, self.a_prev]).astype(np.float32)

    def reset(self):
        k = self.rng.randint(1, len(self.data_states))
        if self.reset_mode == "data":
            self.s = self.data_states[k].copy()
        else:
            self.s = self.rng.uniform(self.state_low, self.state_high).astype(np.float32)
        self.a_prev = self.data_actions[k - 1].copy()          # the setting before s_k
        self.t = 0
        return self._obs()

    def step(self, action):
        a_z = self._applied_action(action)
        mean, std = self.model.predict(torch.as_tensor(self.s[None], dtype=torch.float32),
                                       torch.as_tensor(a_z[None], dtype=torch.float32))
        mean, std = mean[:, 0].numpy(), std[:, 0].numpy()      # (n, 4)
        if self.member == "mean":
            nxt, sd = mean.mean(0), std.mean(0)
        else:
            k = self.rng.randint(self.n_models) if self.member == "random" else self.member
            nxt, sd = mean[k], std[k]
        if self.noise_scale > 0:
            nxt = nxt + self.noise_scale * sd * self.rng.randn(*nxt.shape)
        nxt = nxt.astype(np.float32)
        reward = float(self.reward_batch(nxt))
        self.s, self.a_prev = nxt, a_z
        self.t += 1
        return self._obs(), reward, False, self.t >= self.horizon, {}

    def get_normalized_score(self, episode_return):
        per_step = float(episode_return) / self.horizon
        return 100.0 * (per_step - self.r_random) / (self.r_ref - self.r_random)
