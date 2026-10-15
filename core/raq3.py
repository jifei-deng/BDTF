"""RAQ v3 -- one allocation weight w(s), used by both layers.

Starting point (2026-10-09) = RAQ v2 after the removal of the risk-sensitive
aggregation (beta) and of the state-dependent support budget (eta):

  evaluation layer   y = (1 - w(s')) y_phi + w(s') y_omega        (A2: dual support)
                     w(s') = sigmoid(kappa (U_phi - U_omega) / (U_phi + U_omega))
                                                                  (A1: adaptive allocation)
  ensemble           bootstrap masks + independent targets + lambda_dis * disagreement
  improvement layer  max_pi  mean_k Q_k(s, pi(s))  s.t.  MMD^2(pi, pi_omega) <= eps_bar
                     (dual ascent on delta, clamped at delta_max)

v3 mechanisms (docs/INNOVATION_PLAN.md), each behind a switch:

  A  w_mode = 'plugin'   the MSE-optimal weight of Prop. 4, generalised to
                         correlated branches and computed per sample from the N
                         bootstrap heads instead of the sigmoid surrogate:
                             w* = (U_phi - C + b_phi d) / (U_phi + U_omega - 2C + d^2)
                         C  head covariance of the two branches,
                         d  = E[y_phi - y_omega], estimated by the head mean of the
                            difference, d^2 debiased by its sampling variance,
                         b_phi = bias_c * sqrt(U_phi), the calibration assumption of
                            Cor. 5.1 (bias_c = 0: no overestimation assumed).
                         kappa is not used.
     pooled A            w_mode = 'plugin_pooled': the same model with ONE weight per batch,
                         w = sum(num) / sum(den) over the non-terminal next states -- the
                         minimiser of the batch-average MSE (pooled_weight)
     controls            w_mode = 'plugin_shuffle': A's weights permuted across the batch
                         (same distribution, no state alignment); w_mode = 'fixed' with
                         w_fixed: one constant weight (e.g. A's mean).
  I  w_mode = 'eb'       empirical-Bayes shrinkage of the bootstrapped improvement
                         Q(s', a_phi) - Q(s', a_omega) (eb_weight); eb_z > 0 keeps only
                         the part above its posterior lower bound.
  II prior_scale > 0     randomized prior functions on the critic heads
                         (PriorEnsembleCritic), so the head spread does not vanish
                         off-support.
     w_control           shuffle | mean applied to any weight rule: the same weights
                         randomly re-assigned, or their batch mean (level only).
  G  grad_consensus='and'  improvement layer: the policy follows the ensemble-mean
                         action gradient only where the bootstrap heads agree on its
                         sign (soft AND-mask, Parascandolo et al., ICLR 2021) -- the
                         improvement-side counterpart of A: evaluation allocates by the
                         heads' uncertainty, improvement moves only where they agree.
     actor_lr_schedule='cosine'  control for G: plain annealing of the actor step.
  C  improve = 'wbc'     the support constraint is replaced by the term that bounds
                         the evaluation bias of Thm. 6 per state, w(s) TV(pi, pi_omega):
                             L_pi = -Q(s, pi) / mean|Q| + alpha_bc w(s) ||pi(s) - a||^2
                         with w(s) the evaluation-layer weight at the current state.
                         w_actor = 'const' replaces w(s) by its batch mean (ablation).

With w_mode = 'adaptive' and improve = 'mmd' this class reproduces
raq_v2.core.raq.RAQ (beta = 0, eta = 0) step for step -- same parameters, same
random-number consumption; scripts/check_equivalence.py verifies it.

The networks, the MMD estimator, the replay buffer, the environments and all
baselines are imported from raq_v2 and are not copied.
"""

import copy

import numpy as np
import torch
import torch.nn as nn

from raq_v2.core.mmd import mmd
from raq_v2.core.networks import EnsembleCritic, TanhGaussianActor, VAEBehavior


class PriorEnsembleCritic(nn.Module):
    """Scheme II: randomized prior functions (Osband et al., NeurIPS 2018).

    Q_k(s, a) = f_k(s, a) + scale * p_k(s, a), with p_k a frozen, randomly
    initialised copy of the critic architecture.  Bootstrapping alone cannot
    create disagreement where there is no data; the fixed priors can, because
    the trainable f_k only learns to cancel them where data constrain it.  That
    keeps the head spread -- the input of the allocation weight -- from
    collapsing off-support, which is where the EB weight would otherwise trust
    a consensually overestimated policy branch.

    `scale` is a buffer set by RAQ3.calibrate_prior (saved with the weights)."""

    def __init__(self, base, prior):
        super().__init__()
        self.base = base
        self.prior = prior
        for p in self.prior.parameters():
            p.requires_grad_(False)
        self.register_buffer("scale", torch.zeros(()))
        self.n_critics = base.n_critics

    def forward(self, state, action):
        return self.base(state, action) + self.scale * self.prior(state, action)

    def forward_per_head(self, state, actions):
        """actions (N, B, A) -> (N, B); head k reads actions[k] only."""
        return self.base.forward_per_head(state, actions) + self.scale * self.prior.forward_per_head(state, actions)


class RAQ3:
    def __init__(
        self,
        state_dim,
        action_dim,
        # --- architecture ---
        n_critics=4,
        hidden=256,
        # --- training protocol ---
        discount=0.99,
        tau=0.005,
        policy_freq=2,
        actor_lr=3e-4,
        critic_lr=3e-4,
        vae_lr=1e-3,
        policy_noise=0.2,
        noise_clip=0.5,
        # --- evaluation layer ---
        w_mode="adaptive",        # adaptive | plugin | plugin_pooled | plugin_shuffle | eb | fixed | zero | one
        kappa=5.0,                # adaptive only; 0 recovers the fixed weight
        alpha=0.2,                # only used when w_mode == 'fixed'
        w_fixed=None,             # fixed only: the weight itself (overrides alpha)
        eb_z=0.0,                 # eb only: 0 = posterior mean; z > 0 = lower bound at z sd
        bias_c=1.0,               # plugin only: b_phi = bias_c * sqrt(U_phi)
        w_control="none",         # none | shuffle | mean: controls for any weight rule
        indep_target=True,        # False -> legacy shared min-target
        # --- ensemble regularisation ---
        lambda_dis=1.0,
        prior_scale=0.0,          # scheme II: randomized-prior spread / value scale (0 = off)
        # --- improvement layer ---
        improve="mmd",            # mmd | wbc
        alpha_bc=0.4,             # wbc only; 0.4 = TD3+BC's weight where w(s) = 1
        w_actor="state",          # wbc only: state | const (batch mean, ablation)
        grad_consensus="none",    # mmd only: none | and (ensemble-consensus policy gradient)
        actor_lr_schedule="constant",  # constant | cosine (annealing control for `and`)
        max_steps=0,              # cosine only: training steps (synced by the runner)
        n_mmd_samples=20,
        mmd_kernel="gaussian",
        mmd_sigma=10.0,
        eps_bar=0.05,
        delta_init=1.0,
        delta_lr=1e-3,
        delta_max=100.0,
        device="cpu",
        **_,                      # beta, eta, beta_couple of the v2 configs are ignored
    ):
        self.device = torch.device(device)
        self.action_dim = action_dim

        self.critic = EnsembleCritic(state_dim, action_dim, n_critics, hidden).to(self.device)
        self.prior_scale = prior_scale
        if prior_scale > 0:
            # the prior is drawn from its own random stream, so the rest of the
            # agent (actor, critic, noise) is initialised exactly as without it
            # and the prior is the only difference between the paired runs
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed((torch.initial_seed() + 7919) % (2 ** 31))
                prior = EnsembleCritic(state_dim, action_dim, n_critics, hidden)
            self.critic = PriorEnsembleCritic(self.critic, prior.to(self.device)).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_opt = torch.optim.Adam([p for p in self.critic.parameters() if p.requires_grad],
                                           lr=critic_lr)

        self.actor = TanhGaussianActor(state_dim, action_dim, hidden).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.behavior = VAEBehavior(state_dim, action_dim, hidden=hidden).to(self.device)
        self.behavior_opt = torch.optim.Adam(self.behavior.parameters(), lr=vae_lr)

        self.log_delta = torch.tensor(np.log(delta_init), dtype=torch.float32,
                                      device=self.device, requires_grad=True)
        self.delta_opt = torch.optim.Adam([self.log_delta], lr=delta_lr)

        self.n_critics = n_critics
        self.discount = discount
        self.tau = tau
        self.policy_freq = policy_freq
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip

        self.w_mode = w_mode
        self.kappa = kappa
        self.alpha = alpha
        self.bias_c = bias_c
        self.w_fixed = w_fixed
        self.eb_z = eb_z
        if w_control not in ("none", "shuffle", "mean"):
            raise ValueError(f"unknown w_control: {w_control}")
        self.w_control = w_control
        self.indep_target = indep_target
        self.lambda_dis = lambda_dis

        if improve not in ("mmd", "wbc"):
            raise ValueError(f"unknown improve: {improve}")
        if w_actor not in ("state", "const"):
            raise ValueError(f"unknown w_actor: {w_actor}")
        self.improve = improve
        self.alpha_bc = alpha_bc
        self.w_actor = w_actor
        if grad_consensus not in ("none", "and"):
            raise ValueError(f"unknown grad_consensus: {grad_consensus}")
        self.grad_consensus = grad_consensus
        if actor_lr_schedule not in ("constant", "cosine"):
            raise ValueError(f"unknown actor_lr_schedule: {actor_lr_schedule}")
        self.actor_sched = None
        if actor_lr_schedule == "cosine":
            if max_steps <= 0:
                raise ValueError("actor_lr_schedule=cosine needs max_steps (set agent.max_steps)")
            # one scheduler step per actor update, decaying to 0 at the end of training
            self.actor_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.actor_opt, T_max=max(1, max_steps // policy_freq))
        self.n_mmd_samples = n_mmd_samples
        self.mmd_kernel = mmd_kernel
        self.mmd_sigma = mmd_sigma
        self.eps_bar = eps_bar
        self.delta_max = delta_max

        self.it = 0

    # ------------------------------------------------------------------ #
    # behaviour model pre-training
    # ------------------------------------------------------------------ #
    def pretrain_behavior(self, buffer, steps, batch_size):
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

    # ------------------------------------------------------------------ #
    # evaluation layer
    # ------------------------------------------------------------------ #
    def mix_weight(self, q_phi, q_omega):
        """q_*: (B, N) per-head next-state values of the two branches -> w (B, 1).

        Takes the per-head values rather than the two variances so that weights
        which need more than U_phi and U_omega (covariance, gap) fit the same
        interface."""
        if self.w_mode == "zero":
            return torch.zeros_like(q_phi[:, :1])
        if self.w_mode == "one":
            return torch.ones_like(q_phi[:, :1])
        if self.w_mode in ("plugin", "plugin_shuffle"):
            return self.plugin_weight(q_phi, q_omega)[0]
        if self.w_mode == "plugin_pooled":
            return self.pooled_weight(q_phi, q_omega)[0]
        if self.w_mode == "eb":
            return self.eb_weight(q_phi, q_omega)[0]
        if self.w_mode == "fixed" or self.kappa == 0.0:
            w = self.w_fixed if self.w_fixed is not None else self.alpha / (1.0 + self.alpha)
            return torch.full_like(q_phi[:, :1], w)
        if self.w_mode != "adaptive":
            raise ValueError(f"unknown w_mode: {self.w_mode}")
        u_phi = q_phi.var(dim=1, keepdim=True, unbiased=True)
        u_omega = q_omega.var(dim=1, keepdim=True, unbiased=True)
        rel = (u_phi - u_omega) / (u_phi + u_omega + 1e-8)
        return torch.sigmoid(self.kappa * rel)

    def plugin_weight(self, q_phi, q_omega):
        """Prop. 4 with correlated branches, every term estimated from the heads.

        Model: y_phi = Q + b_phi + e_phi, y_omega = Q + b_omega + e_omega with
        Var e_phi = U_phi, Var e_omega = U_omega, Cov = C.  The MSE of
        (1 - w) y_phi + w y_omega is minimised by
            w* = (U_phi - C + b_phi d) / (U_phi + U_omega - 2C + d^2),  d = b_phi - b_omega.
        U_phi + U_omega - 2C is the variance of the head-wise difference, d its
        mean; d^2 is debiased by the sampling variance of that mean.  Returns
        w (B, 1) and a dict of per-batch diagnostics."""
        num, den, diag = self._plugin_terms(q_phi, q_omega)
        w = (num / den).clamp(0.0, 1.0)
        return w, diag

    def pooled_weight(self, q_phi, q_omega, valid=None):
        """Pooled A: the same model, ONE weight for all next states of the batch.

        MSE_s(w) is quadratic in w with curvature den(s) and minimiser num(s)/den(s)
        (plugin_weight), so the average MSE over states, E_s[MSE_s(w)], is minimised
        by w = E[num] / E[den] -- the MSE-optimal mixing LEVEL.  The per-sample
        estimates of U, C and d enter only through their batch averages, so their
        sampling noise (N heads per state) is averaged out.  Terminal rows (valid = 0)
        have no next-state value and are left out.  Returns w (B, 1) and diagnostics."""
        num, den, diag = self._plugin_terms(q_phi, q_omega)
        m = torch.ones_like(num) if valid is None else valid
        w = ((num * m).sum() / (den * m).sum().clamp(min=1e-8)).clamp(0.0, 1.0)
        return w.expand_as(num), diag

    def _plugin_terms(self, q_phi, q_omega):
        """Numerator, denominator and diagnostics of plugin_weight, per sample (B, 1)."""
        n = q_phi.shape[1]
        diff = q_phi - q_omega
        u_phi = q_phi.var(dim=1, keepdim=True, unbiased=True)
        u_omega = q_omega.var(dim=1, keepdim=True, unbiased=True)
        s = diff.var(dim=1, keepdim=True, unbiased=True)      # U_phi + U_omega - 2C
        cov = 0.5 * (u_phi + u_omega - s)
        d = diff.mean(dim=1, keepdim=True)
        d2 = (d ** 2 - s / n).clamp(min=0.0)
        d_shr = torch.sign(d) * d2.sqrt()
        b_phi = self.bias_c * u_phi.sqrt()
        num = u_phi - cov + b_phi * d_shr
        # identical branches (s = d = 0) carry no information about w: the small
        # floor sends w to 0 there instead of dividing 0 by 0
        den = s + d2 + 1e-3 * (u_phi + u_omega) + 1e-8
        diag = {"rho": cov / (u_phi * u_omega).sqrt().clamp(min=1e-8),
                "gap_sd": d.abs() / u_phi.sqrt().clamp(min=1e-8)}
        return num, den, diag

    @torch.no_grad()
    def calibrate_prior(self, buffer, n=4096):
        """Set the prior's output scale so that its spread across data points is
        prior_scale x the value scale mean|r| / (1 - gamma).

        prior_scale = 0.01 (fixed before any run) puts the prior spread at ~1% of
        the value scale -- the size of the in-distribution head spread measured on
        D4RL and rolling5 (sqrt(U) / |Q| ~ 1-2%), so off-support the priors add
        disagreement of the order the ensemble already shows on data.  Uses evenly
        spaced dataset rows, not buffer.sample, so the batch sequence of the run is
        the same as without the prior."""
        if not isinstance(self.critic, PriorEnsembleCritic):
            return {}
        size = buffer.size
        idx = np.linspace(0, size - 1, min(n, size)).astype(np.int64)

        def rows(x):
            if torch.is_tensor(x):
                return x[torch.as_tensor(idx, device=x.device)].float().to(self.device)
            return torch.as_tensor(np.asarray(x)[idx], dtype=torch.float32, device=self.device)

        state, action = rows(buffer.state), rows(buffer.action)
        r = buffer.reward
        r_abs = float(r.abs().mean()) if torch.is_tensor(r) else float(np.abs(r).mean())
        value_scale = r_abs / (1.0 - self.discount)
        p_std = float(self.critic.prior(state, action).std(dim=0).mean())
        scale = self.prior_scale * value_scale / max(p_std, 1e-8)
        self.critic.scale.fill_(scale)
        self.critic_target = copy.deepcopy(self.critic)          # target starts identical
        self._prior_ready = True
        return {"prior_value_scale": round(value_scale, 3), "prior_raw_std": round(p_std, 5),
                "prior_mult": round(scale, 3)}

    def eb_weight(self, q_phi, q_omega, valid=None):
        """Candidate scheme I (docs/INNOVATION_PLAN.md, section 12): empirical-Bayes
        shrinkage of the bootstrapped policy improvement.

        The dual-support target is rewritten as anchor + improvement,
            y = y_omega + (1 - w) (y_phi - y_omega),
        and the improvement D = Q(s', a_phi) - Q(s', a_omega) is estimated by its
        head mean d with head variance s = U_phi + U_omega - 2C.  With the prior
        D ~ N(0, tau^2) ("no improvement over the behaviour action unless the data
        show one") the posterior mean of D is g d, g = tau^2 / (tau^2 + s), so
        w = 1 - g = s / (tau^2 + s).  tau^2 is estimated from the batch by moments
        (Efron & Morris): tau^2 = mean(d^2) - mean(s), floored at 0.  No tuned
        constant; w is smooth (no clipping), and only the noise of the
        DIFFERENCE enters, which is small when the two branches are correlated.

        valid: (B, 1) not_done mask.  Terminal transitions do not bootstrap, so
        their improvement is irrelevant to the target and is left out of tau^2."""
        diff = q_phi - q_omega
        d = diff.mean(dim=1, keepdim=True)
        s = diff.var(dim=1, keepdim=True, unbiased=True)
        if valid is None:
            valid = torch.ones_like(d)
        n_valid = valid.sum().clamp(min=1.0)
        tau2 = ((d ** 2 * valid).sum() / n_valid - (s * valid).sum() / n_valid).clamp(min=0.0)
        g = tau2 / (tau2 + s + 1e-8)                       # posterior mean factor
        if self.eb_z > 0:
            # high-confidence variant: keep only the part of a positive improvement
            # above its posterior lower bound, g d - z sqrt(g s) (posterior N(g d, g s));
            # an estimated loss (d <= 0) is passed on unchanged
            lcb = (g - self.eb_z * (g * s).sqrt() / d.abs().clamp(min=1e-8)).clamp(min=0.0)
            g = torch.where(d > 0, lcb, g)
        w = 1.0 - g
        diag = {"eb_tau2_over_s": tau2 / s.mean().clamp(min=1e-8),
                "eb_snr": (d ** 2 / s.clamp(min=1e-8)).median()}
        return w, diag

    # ------------------------------------------------------------------ #
    def train_step(self, buffer, batch_size):
        if self.prior_scale > 0 and not getattr(self, "_prior_ready", False):
            print(f"    prior calibrated at the first step: {self.calibrate_prior(buffer)}")
        self.it += 1
        state, action, next_state, reward, not_done, mask = buffer.sample(batch_size)

        # ---------------- critic ---------------- #
        with torch.no_grad():
            a_phi, _, _ = self.actor_target(next_state)
            if self.policy_noise > 0:
                noise = (torch.randn_like(a_phi) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
                a_phi = (a_phi + noise).clamp(-1.0, 1.0)
            a_omega, _ = self.behavior.decode(next_state)

            q_phi = self.critic_target(next_state, a_phi)        # (B, N)
            q_omega = self.critic_target(next_state, a_omega)
            u_phi = q_phi.var(dim=1, keepdim=True, unbiased=True)
            u_omega = q_omega.var(dim=1, keepdim=True, unbiased=True)

            # the weight sees the heads before any min-target collapses them
            if self.w_mode == "plugin":
                w, wdiag = self.plugin_weight(q_phi, q_omega)   # (B, 1)
            elif self.w_mode == "plugin_shuffle":
                # control: the same weights, randomly re-assigned across the batch --
                # keeps their distribution (mean, spread, share at 0/1, noise) and
                # removes only which next state receives which weight
                w, wdiag = self.plugin_weight(q_phi, q_omega)
                w = w[torch.randperm(w.shape[0], device=w.device)]
            elif self.w_mode == "plugin_pooled":
                w, wdiag = self.pooled_weight(q_phi, q_omega, valid=not_done)
            elif self.w_mode == "eb":
                w, wdiag = self.eb_weight(q_phi, q_omega, valid=not_done)
            else:
                w, wdiag = self.mix_weight(q_phi, q_omega), {}
            # controls applicable to any weight rule (does the state assignment matter?):
            #   shuffle  the same weights, randomly re-assigned across the batch
            #   mean     one global weight per step = the batch mean (level only)
            if self.w_control == "shuffle":
                w = w[torch.randperm(w.shape[0], device=w.device)]
            elif self.w_control == "mean":
                w = w.mean().expand_as(w)

            if not self.indep_target:                            # legacy shared min-target
                q_phi = q_phi.min(dim=1, keepdim=True)[0].expand_as(q_phi)
                q_omega = q_omega.min(dim=1, keepdim=True)[0].expand_as(q_omega)

            y_phi = reward + not_done * self.discount * q_phi
            y_omega = reward + not_done * self.discount * q_omega
            target = (1.0 - w) * y_phi + w * y_omega             # (B, N)

        q = self.critic(state, action)                           # (B, N)
        td = (mask * (target - q) ** 2).sum() / mask.sum().clamp(min=1.0)
        dis = q.var(dim=1, unbiased=True).mean()
        critic_loss = td + self.lambda_dis * dis

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        info = {
            "critic_loss": float(critic_loss), "td": float(td), "disagreement": float(dis),
            "w_mean": float(w.mean()), "w_std": float(w.std()),
            "u_phi": float(u_phi.mean()), "u_omega": float(u_omega.mean()),
            "q_mean": float(q.mean()),
        }
        info.update({k: float(v.median()) for k, v in wdiag.items()})

        # ---------------- actor (delayed) ---------------- #
        if self.it % self.policy_freq == 0 and self.improve == "wbc":
            info.update(self._actor_step_wbc(state, action))
        elif self.it % self.policy_freq == 0:
            a_pi, _, _ = self.actor(state)
            if self.grad_consensus == "and":
                q_term, q_pi, gdiag = self._consensus_q_term(state, a_pi)
            else:
                q_pi = self.critic(state, a_pi).mean(dim=1)      # (B,)
                q_term, gdiag = q_pi, {}

            pi_a, pi_pre = self.actor.sample_multi(state, self.n_mmd_samples)
            with torch.no_grad():
                vae_a, vae_pre = self.behavior.decode_multi(state, self.n_mmd_samples)
            mmd2 = mmd(pi_pre, vae_pre, self.mmd_kernel, self.mmd_sigma, squared=True)  # (B,)

            violation = mmd2 - self.eps_bar
            delta = self.log_delta.exp().detach().clamp(max=self.delta_max)

            actor_loss = (-q_term + delta * violation).mean()
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()
            if self.actor_sched is not None:
                self.actor_sched.step()

            delta_loss = -(self.log_delta.exp() * violation.detach()).mean()
            self.delta_opt.zero_grad()
            delta_loss.backward()
            self.delta_opt.step()
            with torch.no_grad():
                self.log_delta.clamp_(-8.0, np.log(self.delta_max))

            self._soft_update(self.critic, self.critic_target)
            self._soft_update(self.actor, self.actor_target)

            info.update({
                "actor_loss": float(actor_loss), "q_pi": float(q_pi.mean()),
                "mmd2": float(mmd2.mean()), "delta": float(delta),
            })
            info.update({k: float(v) for k, v in gdiag.items()})
            if self.actor_sched is not None:
                info["actor_lr"] = float(self.actor_opt.param_groups[0]["lr"])
        return info

    def _consensus_q_term(self, state, a_pi):
        """Ensemble-consensus policy gradient (AND-mask over bootstrap heads).

        The deterministic policy gradient of the ensemble mean is J^T g with
        g = mean_k dQ_k/da.  Each head is a bootstrap replicate of the critic; a
        component of g on which the replicates disagree in sign is not supported
        by the data -- after the critic has converged it is mostly estimation
        noise, and following it makes the policy random-walk along the support
        boundary (the late falls on hopper / walker2d).  Following Parascandolo
        et al. (ICLR 2021), each action component is weighted by the agreement
        of the heads' gradient signs, |mean_k sign(dQ_k/da_j)| in [0, 1] (soft
        AND-mask: 1 = all heads agree, 0 = an even split), so the policy moves
        only where the bootstrap replicates agree.

        Returns the surrogate whose gradient w.r.t. the actor is J^T (m * g),
        the ensemble-mean Q (for logging) and diagnostics."""
        n = self.n_critics
        a_rep = a_pi.detach().unsqueeze(0).expand(n, *a_pi.shape).clone().requires_grad_(True)
        q_heads = self.critic.forward_per_head(state, a_rep)    # (N, B): head k reads a_rep[k]
        g = torch.autograd.grad(q_heads.sum(), a_rep)[0]          # (N, B, A) per-head dQ_k/da
        g_mean = g.mean(dim=0)                                     # (B, A): gradient of mean_k Q_k
        agree = g.sign().mean(dim=0).abs()                         # (B, A) in [0, 1]
        g_used = agree * g_mean
        q_term = (a_pi * g_used.detach()).sum(dim=-1)              # d/dtheta = J^T g_used
        q_pi = q_heads.detach().mean(dim=0)                        # (B,)
        keep = g_used.pow(2).sum() / g_mean.pow(2).sum().clamp(min=1e-12)
        return q_term, q_pi, {"g_agree": agree.mean(), "g_keep": keep.sqrt()}

    def _actor_step_wbc(self, state, action):
        """Improvement layer C: pi may leave the data only where the critic does
        not itself fall back on the data, i.e. the BC weight is the evaluation
        weight w(s) -- the factor in front of TV(pi, pi_omega) in the per-state
        evaluation-bias bound of Thm. 6."""
        a_pi, _, _ = self.actor(state)
        q_pi = self.critic(state, a_pi).mean(dim=1)              # (B,)
        with torch.no_grad():
            a_omega, _ = self.behavior.decode(state)
            w_s = self.mix_weight(self.critic_target(state, a_pi),
                                  self.critic_target(state, a_omega)).squeeze(-1)  # (B,)
            if self.w_actor == "const":
                w_s = w_s.mean().expand_as(w_s)
            q_scale = q_pi.abs().mean().clamp(min=1e-6)          # TD3+BC normalisation
        bc = ((a_pi - action) ** 2).mean(dim=1)                  # (B,)
        actor_loss = (-q_pi / q_scale + self.alpha_bc * w_s * bc).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()
        if self.actor_sched is not None:
            self.actor_sched.step()

        self._soft_update(self.critic, self.critic_target)
        self._soft_update(self.actor, self.actor_target)
        return {"actor_loss": float(actor_loss), "q_pi": float(q_pi.mean()),
                "w_actor": float(w_s.mean()), "w_actor_std": float(w_s.std()),
                "bc": float(bc.mean())}

    def _soft_update(self, net, target):
        with torch.no_grad():
            for p, tp in zip(net.parameters(), target.parameters()):
                tp.data.mul_(1.0 - self.tau).add_(self.tau * p.data)

    # ------------------------------------------------------------------ #
    # interface used by raq_v2.scripts.run (same as raq_v2's RAQAgent)
    # ------------------------------------------------------------------ #
    def pretrain(self, buffer, batch_size, steps):
        out = self.calibrate_prior(buffer)                       # {} without scheme II
        if steps > 0:
            out["vae_recon"] = self.pretrain_behavior(buffer, steps, batch_size)
        return out

    @torch.no_grad()
    def select_action(self, state, deterministic=True):
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device).reshape(1, -1)
        action, _, _ = self.actor(state, deterministic=deterministic)
        return action.cpu().numpy().flatten()

    @torch.no_grad()
    def uncertainty(self, state, action):
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device).reshape(1, -1)
        action = torch.as_tensor(action, dtype=torch.float32, device=self.device).reshape(1, -1)
        return float(self.critic(state, action).var(dim=1, unbiased=True))

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
