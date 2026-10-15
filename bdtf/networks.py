"""Networks used by BDTF.

Shapes: B = batch, N = ensemble size N_Q, S = state dim, A = action dim,
M = action samples per state (support constraint).
All actions handled here are normalised to [-1, 1]^A; the environment
wrappers map them to physical units.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


class EnsembleLinear(nn.Module):
    """N independent linear layers evaluated in one batched matmul."""

    def __init__(self, ensemble_size, in_dim, out_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(ensemble_size, in_dim, out_dim))
        self.bias = nn.Parameter(torch.zeros(ensemble_size, 1, out_dim))
        bound = 1.0 / math.sqrt(in_dim)
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):            # x: (N, B, in_dim)
        return torch.baddbmm(self.bias, x, self.weight)


class EnsembleCritic(nn.Module):
    """N_Q independent Q heads."""

    def __init__(self, state_dim, action_dim, n_critics=4, hidden=256, n_hidden=2):
        super().__init__()
        self.n_critics = n_critics
        dims = [state_dim + action_dim] + [hidden] * n_hidden
        self.layers = nn.ModuleList(
            [EnsembleLinear(n_critics, dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )
        self.out = EnsembleLinear(n_critics, hidden, 1)

    def forward(self, state, action):
        """state: (B, S), action: (B, A)  ->  (B, N)"""
        x = torch.cat([state, action], dim=-1)
        x = x.unsqueeze(0).expand(self.n_critics, *x.shape)
        for layer in self.layers:
            x = F.relu(layer(x))
        return self.out(x).squeeze(-1).transpose(0, 1)


class TanhGaussianActor(nn.Module):
    """Squashed Gaussian policy (stochastic, so that M actions per state can be
    sampled for the state-conditional MMD)."""

    def __init__(self, state_dim, action_dim, hidden=256, n_hidden=2):
        super().__init__()
        layers, d = [], state_dim
        for _ in range(n_hidden):
            layers += [nn.Linear(d, hidden), nn.ReLU()]
            d = hidden
        self.body = nn.Sequential(*layers)
        self.mu = nn.Linear(d, action_dim)
        self.log_std = nn.Linear(d, action_dim)
        self.action_dim = action_dim

    def _dist_params(self, state):
        h = self.body(state)
        return self.mu(h), self.log_std(h).clamp(LOG_STD_MIN, LOG_STD_MAX)

    def forward(self, state, deterministic=False, with_logp=False):
        """-> action (B, A) in [-1, 1], log-prob (B, 1) or None, pre-tanh action (B, A)."""
        mu, log_std = self._dist_params(state)
        pre = mu if deterministic else mu + log_std.exp() * torch.randn_like(mu)
        action = torch.tanh(pre)
        logp = None
        if with_logp:
            std = log_std.exp()
            logp = (-0.5 * ((pre - mu) / std) ** 2 - log_std - 0.5 * math.log(2 * math.pi)).sum(-1, keepdim=True)
            logp = logp - torch.log(1 - action.pow(2) + 1e-6).sum(-1, keepdim=True)
        return action, logp, pre

    def sample_multi(self, state, num_samples):
        """-> action (B, M, A), pre-tanh action (B, M, A)."""
        mu, log_std = self._dist_params(state)
        mu, std = mu.unsqueeze(1), log_std.exp().unsqueeze(1)
        eps = torch.randn(state.shape[0], num_samples, self.action_dim, device=state.device)
        pre = mu + std * eps
        return torch.tanh(pre), pre


class VAEBehavior(nn.Module):
    """Conditional VAE behaviour model."""

    def __init__(self, state_dim, action_dim, latent_dim=None, hidden=256):
        super().__init__()
        latent_dim = latent_dim or 2 * action_dim
        self.latent_dim = latent_dim
        self.action_dim = action_dim

        self.e1 = nn.Linear(state_dim + action_dim, hidden)
        self.e2 = nn.Linear(hidden, hidden)
        self.mean = nn.Linear(hidden, latent_dim)
        self.log_std = nn.Linear(hidden, latent_dim)

        self.d1 = nn.Linear(state_dim + latent_dim, hidden)
        self.d2 = nn.Linear(hidden, hidden)
        self.d3 = nn.Linear(hidden, action_dim)

    def forward(self, state, action):
        z = F.relu(self.e1(torch.cat([state, action], dim=-1)))
        z = F.relu(self.e2(z))
        mean = self.mean(z)
        std = self.log_std(z).clamp(-4, 15).exp()
        latent = mean + std * torch.randn_like(std)
        recon, _ = self.decode(state, latent)
        return recon, mean, std

    def loss(self, state, action):
        recon, mean, std = self.forward(state, action)
        recon_loss = F.mse_loss(recon, action)
        kl = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
        return recon_loss + 0.5 * kl, recon_loss.detach()

    def decode(self, state, latent=None, clip=0.5):
        if latent is None:
            latent = torch.randn(state.shape[0], self.latent_dim, device=state.device).clamp(-clip, clip)
        a = F.relu(self.d1(torch.cat([state, latent], dim=-1)))
        a = F.relu(self.d2(a))
        pre = self.d3(a)
        return torch.tanh(pre), pre

    def decode_multi(self, state, num_samples, clip=0.5):
        """-> action (B, M, A), pre-tanh action (B, M, A)."""
        b = state.shape[0]
        latent = torch.randn(b, num_samples, self.latent_dim, device=state.device).clamp(-clip, clip)
        s_rep = state.unsqueeze(1).expand(b, num_samples, state.shape[-1])
        a = F.relu(self.d1(torch.cat([s_rep, latent], dim=-1)))
        a = F.relu(self.d2(a))
        pre = self.d3(a)
        return torch.tanh(pre), pre
