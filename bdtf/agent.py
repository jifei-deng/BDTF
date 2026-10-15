"""BDTF: bias-aware dual-support target fusion for offline reinforcement learning.

Components (section numbers refer to the Methodology section of the paper):

  * Behaviour model and bootstrap critic ensemble.  A conditional VAE models the
    behaviour policy (pre-trained, then frozen).  The critic has N heads; each
    head is trained on its own fixed Bernoulli subsample of the data (bootstrap
    masks, see buffer.py) and bootstraps from its own target head, and a
    disagreement penalty lambda * Var_k Q_k(s, a) on the dataset actions keeps
    the ensemble bounded.

  * Bias-aware dual-support target fusion (the contribution).  At the next
    state s' two backups are formed for every head k,
        y_phi,k   = r + gamma * Q_k(s', a_phi),     a_phi   ~ target actor
        y_omega,k = r + gamma * Q_k(s', a_omega),   a_omega ~ behaviour model
    and fused as  y_k = (1 - w) y_phi,k + w y_omega,k.  The weight minimises
    the mean-squared error of the fused target,
        w* = (U_phi - C + b_phi d) / (U_phi + U_omega - 2C + d^2),
    where U_phi, U_omega are the head variances of the two backups, C their
    head covariance, d = E[y_phi - y_omega] (head mean of the difference,
    d^2 debiased by its sampling variance), and b_phi = c * sqrt(U_phi) is the
    calibrated overestimation of the policy-based backup (c = bias_c).
    Every term is estimated per sample from the N heads (plug-in estimate);
    the weight is clipped to [0, 1].

  * Support-constrained policy improvement.  The actor maximises the ensemble
    mean value subject to a state-conditional MMD constraint between the actor
    and the behaviour model, MMD^2 <= eps_bar, enforced by dual ascent on a
    Lagrange multiplier delta (capped at delta_max).  The actor's learning
    rate is annealed to zero over training with a cosine schedule.

Ablation switches of the paper (not used by the default configurations):
    agent.w_mode=zero     w = 0, policy-based backup only
    agent.w_mode=one      w = 1, behaviour-based backup only
    agent.bias_c=0        fusion weight without the overestimation term
"""

import copy

import numpy as np
import torch

from .mmd import mmd
from .networks import EnsembleCritic, TanhGaussianActor, VAEBehavior


class BDTF:
    def __init__(
        self,
        state_dim,
        action_dim,
        # --- networks ---
        n_critics=4,
        hidden=256,
        bootstrap_p=0.8,          # used by the replay buffer (bootstrap masks)
        # --- training ---
        discount=0.99,
        tau=0.005,
        policy_freq=2,
        actor_lr=3e-4,
        critic_lr=3e-4,
        vae_lr=1e-3,
        policy_noise=0.2,
        noise_clip=0.5,
        actor_lr_schedule="cosine",   # cosine | constant
        max_steps=0,              # training steps (set by train.py), for the cosine schedule
        # --- bias-aware dual-support target fusion ---
        w_mode="plugin",          # plugin | zero | one
        bias_c=1.0,               # overestimation of the policy-based backup, in head std
        # --- ensemble regularisation ---
        lambda_dis=1.0,
        # --- support constraint ---
        n_mmd_samples=20,
        mmd_kernel="gaussian",
        mmd_sigma=10.0,
        eps_bar=0.05,             # support budget
        delta_init=1.0,           # initial Lagrange multiplier
        delta_lr=1e-3,
        delta_max=100.0,
        device="cpu",
    ):
        self.device = torch.device(device)
        self.action_dim = action_dim

        self.critic = EnsembleCritic(state_dim, action_dim, n_critics, hidden).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.actor = TanhGaussianActor(state_dim, action_dim, hidden).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.behavior = VAEBehavior(state_dim, action_dim, hidden=hidden).to(self.device)
        self.behavior_opt = torch.optim.Adam(self.behavior.parameters(), lr=vae_lr)

        self.log_delta = torch.tensor(np.log(delta_init), dtype=torch.float32,
                                      device=self.device, requires_grad=True)
        self.delta_opt = torch.optim.Adam([self.log_delta], lr=delta_lr)

        if w_mode not in ("plugin", "zero", "one"):
            raise ValueError(f"unknown w_mode: {w_mode}")
        if actor_lr_schedule not in ("cosine", "constant"):
            raise ValueError(f"unknown actor_lr_schedule: {actor_lr_schedule}")
        self.actor_sched = None
        if actor_lr_schedule == "cosine":
            if max_steps <= 0:
                raise ValueError("actor_lr_schedule=cosine needs max_steps (the number of training steps)")
            # one scheduler step per actor update, reaching 0 at the end of training
            self.actor_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.actor_opt, T_max=max(1, max_steps // policy_freq))

        self.n_critics = n_critics
        self.discount = discount
        self.tau = tau
        self.policy_freq = policy_freq
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.w_mode = w_mode
        self.bias_c = bias_c
        self.lambda_dis = lambda_dis
        self.n_mmd_samples = n_mmd_samples
        self.mmd_kernel = mmd_kernel
        self.mmd_sigma = mmd_sigma
        self.eps_bar = eps_bar
        self.delta_max = delta_max
        self.it = 0

    # ------------------------------------------------------------------ #
    def pretrain_behavior(self, buffer, steps, batch_size):
        """Fit the conditional VAE behaviour model, then freeze it."""
        log = []
        for _ in range(steps):
            state, action, *_ = buffer.sample(batch_size)
            loss, recon = self.behavior.loss(state, action)
            self.behavior_opt.zero_grad()
            loss.backward()
            self.behavior_opt.step()
            log.append(float(recon))
        for p in self.behavior.parameters():
            p.requires_grad_(False)
        return float(np.mean(log[-100:])) if log else float("nan")

    def fusion_weight(self, q_phi, q_omega):
        """q_*: (B, N) per-head next-state values of the two backups -> w (B, 1)."""
        if self.w_mode == "zero":
            return torch.zeros_like(q_phi[:, :1])
        if self.w_mode == "one":
            return torch.ones_like(q_phi[:, :1])
        n = q_phi.shape[1]
        diff = q_phi - q_omega
        u_phi = q_phi.var(dim=1, keepdim=True, unbiased=True)
        u_omega = q_omega.var(dim=1, keepdim=True, unbiased=True)
        s = diff.var(dim=1, keepdim=True, unbiased=True)          # U_phi + U_omega - 2C
        cov = 0.5 * (u_phi + u_omega - s)
        d = diff.mean(dim=1, keepdim=True)
        d2 = (d ** 2 - s / n).clamp(min=0.0)                      # debiased squared gap
        d_shr = torch.sign(d) * d2.sqrt()
        b_phi = self.bias_c * u_phi.sqrt()
        num = u_phi - cov + b_phi * d_shr
        # identical backups (s = d = 0) carry no information about w: the small
        # floor sends w to 0 there instead of dividing 0 by 0
        den = s + d2 + 1e-3 * (u_phi + u_omega) + 1e-8
        return (num / den).clamp(0.0, 1.0)

    # ------------------------------------------------------------------ #
    def train_step(self, buffer, batch_size):
        self.it += 1
        state, action, next_state, reward, not_done, mask = buffer.sample(batch_size)

        # ---------------- critic ---------------- #
        with torch.no_grad():
            a_phi, _, _ = self.actor_target(next_state)
            if self.policy_noise > 0:
                noise = (torch.randn_like(a_phi) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
                a_phi = (a_phi + noise).clamp(-1.0, 1.0)
            a_omega, _ = self.behavior.decode(next_state)

            q_phi = self.critic_target(next_state, a_phi)        # (B, N): one backup per head
            q_omega = self.critic_target(next_state, a_omega)
            w = self.fusion_weight(q_phi, q_omega)               # (B, 1)
            y_phi = reward + not_done * self.discount * q_phi
            y_omega = reward + not_done * self.discount * q_omega
            target = (1.0 - w) * y_phi + w * y_omega             # (B, N)

        q = self.critic(state, action)                           # (B, N)
        td = (mask * (target - q) ** 2).sum() / mask.sum().clamp(min=1.0)
        dis = q.var(dim=1, unbiased=True).mean()                 # disagreement penalty
        critic_loss = td + self.lambda_dis * dis

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        info = {"critic_loss": float(critic_loss), "td": float(td), "disagreement": float(dis),
                "w_mean": float(w.mean()), "w_std": float(w.std()), "q_mean": float(q.mean())}

        # ---------------- actor (delayed) ---------------- #
        if self.it % self.policy_freq == 0:
            a_pi, _, _ = self.actor(state)
            q_pi = self.critic(state, a_pi).mean(dim=1)          # ensemble mean, (B,)

            _, pi_pre = self.actor.sample_multi(state, self.n_mmd_samples)
            with torch.no_grad():
                _, vae_pre = self.behavior.decode_multi(state, self.n_mmd_samples)
            mmd2 = mmd(pi_pre, vae_pre, self.mmd_kernel, self.mmd_sigma, squared=True)  # (B,)
            violation = mmd2 - self.eps_bar
            delta = self.log_delta.exp().detach().clamp(max=self.delta_max)

            actor_loss = (-q_pi + delta * violation).mean()
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()
            if self.actor_sched is not None:
                self.actor_sched.step()

            # dual ascent: delta grows while the support constraint is violated
            delta_loss = -(self.log_delta.exp() * violation.detach()).mean()
            self.delta_opt.zero_grad()
            delta_loss.backward()
            self.delta_opt.step()
            with torch.no_grad():
                self.log_delta.clamp_(-8.0, np.log(self.delta_max))

            self._soft_update(self.critic, self.critic_target)
            self._soft_update(self.actor, self.actor_target)
            info.update({"actor_loss": float(actor_loss), "q_pi": float(q_pi.mean()),
                         "mmd2": float(mmd2.mean()), "delta": float(delta),
                         "actor_lr": float(self.actor_opt.param_groups[0]["lr"])})
        return info

    def _soft_update(self, net, target):
        with torch.no_grad():
            for p, tp in zip(net.parameters(), target.parameters()):
                tp.data.mul_(1.0 - self.tau).add_(self.tau * p.data)

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def select_action(self, state, deterministic=True):
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device).reshape(1, -1)
        action, _, _ = self.actor(state, deterministic=deterministic)
        return action.cpu().numpy().flatten()

    def save(self, path):
        torch.save({"critic": self.critic.state_dict(), "actor": self.actor.state_dict(),
                    "behavior": self.behavior.state_dict(), "log_delta": self.log_delta.detach()}, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target = copy.deepcopy(self.critic)
        self.actor.load_state_dict(ckpt["actor"])
        self.actor_target = copy.deepcopy(self.actor)
        self.behavior.load_state_dict(ckpt["behavior"])
        with torch.no_grad():
            self.log_delta.copy_(ckpt["log_delta"].to(self.device))
        return self
