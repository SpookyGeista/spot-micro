"""Tanh-normal actions with bounded latent noise and the transformed density.

PPO's stored means/stds are latent Gaussian parameters. Its analytic KL is
valid because the same invertible tanh transform is used for both policies.
Entropy includes the transform Jacobian; latent entropy alone is incorrect.
"""
import math
import torch
from torch import nn
from torch.distributions import Normal
from rsl_rl.modules import ActorCritic


class ActionSquash(nn.Module):
    def __init__(self, mean_bound=4.0, action_bound=.99999):
        super().__init__()
        self.mean_bound = float(mean_bound)
        self.action_bound = float(action_bound)
        if self.mean_bound <= 1 or not 0 < self.action_bound < 1:
            raise ValueError("Require mean_bound > 1 and action_bound in (0,1)")

    def latent_mean(self, raw):
        # Preserve the old clipped actor's executed actions almost exactly.
        # atanh is exact in the interior; extend it C1-linearly before its
        # singularity, then approach the finite latent limit algebraically.
        threshold = .995
        inside = raw.clamp(-threshold, threshold)
        inverse_tanh = .5*(torch.log1p(inside)-torch.log1p(-inside))
        latent = inverse_tanh + (raw-inside)/(1-threshold**2)
        knee = self.mean_bound-1
        distance = (latent.abs()-knee).clamp(min=0)
        capped = latent.sign()*(self.mean_bound-1/(1+distance))
        return torch.where(latent.abs() <= knee, latent, capped)

    def forward(self, raw):
        return self.action_bound * torch.tanh(self.latent_mean(raw))


class SquashedActorCritic(ActorCritic):
    bounded_actions = True

    def __init__(self, *args, min_noise_std=.05, max_noise_std=.5,
                 init_noise_std=.25, latent_mean_bound=4.0, action_bound=.99999, **kwargs):
        if not 0 < min_noise_std < init_noise_std < max_noise_std:
            raise ValueError("Require 0 < minimum < initial < maximum latent noise")
        if kwargs.get("state_dependent_std", False) or kwargs.get("noise_std_type", "log") != "log":
            raise ValueError("Only state-independent log-noise configuration supported")
        kwargs["noise_std_type"] = "log"
        super().__init__(*args, init_noise_std=init_noise_std, **kwargs)
        del self.log_std
        ratio = (init_noise_std - min_noise_std) / (max_noise_std - min_noise_std)
        self.noise_logit = nn.Parameter(torch.full((self.actor[-1].out_features,), math.log(ratio/(1-ratio))))
        self.min_noise_std = float(min_noise_std)
        self.max_noise_std = float(max_noise_std)
        self.action_postprocessor = ActionSquash(latent_mean_bound, action_bound)

    @property
    def latent_std(self):
        return self.min_noise_std + (self.max_noise_std-self.min_noise_std)*self.noise_logit.sigmoid()

    def _update_distribution(self, obs):
        mean = self.action_postprocessor.latent_mean(self.actor(obs))
        self.distribution = Normal(mean, self.latent_std.expand_as(mean))

    def act(self, obs, **kwargs):
        actor_obs = self.actor_obs_normalizer(self.get_actor_obs(obs))
        self._update_distribution(actor_obs)
        return self.action_postprocessor.action_bound * self.distribution.sample().tanh()

    def act_inference(self, obs):
        actor_obs = self.actor_obs_normalizer(self.get_actor_obs(obs))
        return self.action_postprocessor(self.actor(actor_obs))

    @staticmethod
    def log_tanh_jacobian(latent):
        # Stable even for large positive or negative latent samples.
        return 2*(math.log(2)-latent-torch.nn.functional.softplus(-2*latent))

    def get_actions_log_prob(self, actions):
        unit = (actions / self.action_postprocessor.action_bound).clamp(-1+1e-6, 1-1e-6)
        latent = torch.atanh(unit)
        log_det = self.log_tanh_jacobian(latent) + math.log(self.action_postprocessor.action_bound)
        log_prob = self.distribution.log_prob(latent)-log_det
        return torch.where(actions.abs() < self.action_postprocessor.action_bound,
                           log_prob, torch.full_like(log_prob,-torch.inf)).sum(-1)

    @property
    def entropy(self):
        # Reparameterized MC estimate of H(tanh(Z)), retaining its gradient.
        latent = self.distribution.rsample()
        return (self.distribution.entropy() + self.log_tanh_jacobian(latent)
                + math.log(self.action_postprocessor.action_bound)).sum(-1)


class DirectActionSquash(torch.nn.Module):
    """Well-conditioned fresh-policy means; no inverse-tanh legacy mapping."""
    def __init__(self, mean_bound=2.0, action_bound=.99999):
        super().__init__()
        self.mean_bound = float(mean_bound)
        self.action_bound = float(action_bound)

    def latent_mean(self, raw):
        return self.mean_bound * torch.tanh(raw / self.mean_bound)

    def forward(self, raw):
        return self.action_bound * torch.tanh(self.latent_mean(raw))


class DirectSquashedActorCritic(SquashedActorCritic):
    def __init__(self, *args, latent_mean_bound=2.0, action_bound=.99999, **kwargs):
        super().__init__(*args, latent_mean_bound=latent_mean_bound, action_bound=action_bound, **kwargs)
        self.action_postprocessor = DirectActionSquash(latent_mean_bound, action_bound)
        torch.nn.init.orthogonal_(self.actor[-1].weight, gain=.01)
        torch.nn.init.zeros_(self.actor[-1].bias)
