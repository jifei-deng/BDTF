"""Offline dataset with per-head bootstrap masks."""

import numpy as np
import torch


class ReplayBuffer:
    """Holds the offline transitions.

    Every transition gets a fixed Bernoulli(bootstrap_p) mask per critic head,
    so that each head is fitted on its own bootstrap subsample of the data.
    """

    FIELDS = ("state", "action", "next_state", "reward", "not_done", "mask")

    def __init__(self, state_dim, action_dim, n_critics=4, bootstrap_p=0.8, device="cpu", seed=0):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_critics = n_critics
        self.bootstrap_p = bootstrap_p
        self.device = device
        self.rng = np.random.RandomState(seed)
        self.size = 0
        self._on_device = False
        self._stats = None

    def load(self, data):
        """`data`: dict with states, actions, next_states, rewards, not_dones."""
        f32 = lambda k: np.asarray(data[k], dtype=np.float32)     # noqa: E731
        self.state = f32("states")
        self.action = f32("actions")
        self.next_state = f32("next_states")
        self.reward = f32("rewards").reshape(-1, 1)
        self.not_done = f32("not_dones").reshape(-1, 1)
        self.size = self.state.shape[0]
        self.mask = (self.rng.rand(self.size, self.n_critics) < self.bootstrap_p).astype(np.float32)
        empty = self.mask.sum(1) == 0                  # every transition is seen by >= 1 head
        if empty.any():
            self.mask[empty, self.rng.randint(0, self.n_critics, empty.sum())] = 1.0
        return self

    def normalize_actions(self, low, high):
        """Map dataset actions from [low, high] into [-1, 1]."""
        low, high = np.asarray(low, dtype=np.float32), np.asarray(high, dtype=np.float32)
        self.action = np.clip(2.0 * (self.action - low) / (high - low + 1e-8) - 1.0, -1.0, 1.0)
        return self

    def finalize(self):
        """Move the whole dataset to the device once (it is small)."""
        self._stats = self.stats()
        if str(self.device).startswith("cuda"):
            for field in self.FIELDS:
                setattr(self, field, torch.as_tensor(getattr(self, field),
                                                     dtype=torch.float32, device=self.device))
            self._on_device = True
        return self

    def sample(self, batch_size):
        idx = self.rng.randint(0, self.size, size=batch_size)
        if self._on_device:
            idx = torch.as_tensor(idx, dtype=torch.long, device=self.device)
            return tuple(getattr(self, f)[idx] for f in self.FIELDS)
        return tuple(torch.as_tensor(getattr(self, f)[idx], dtype=torch.float32, device=self.device)
                     for f in self.FIELDS)

    def stats(self):
        if self._stats is not None:
            return self._stats
        return {"transitions": int(self.size),
                "reward_mean": float(self.reward.mean()),
                "reward_min": float(self.reward.min()),
                "reward_max": float(self.reward.max()),
                "terminal_rate": float(1.0 - self.not_done.mean())}
