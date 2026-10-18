"""Masked object attention and a Gaussian recurrent state-space model."""

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal, kl_divergence
from torch.nn import functional as F

from .schema import (CONTEXT, CONTINUOUS, LIGHT_CLASSES, OBJECT_FIELDS, OBJECT_OFFSET,
                     SLOTS, WAYPOINTS, continuous_valid)


def mlp(inputs, outputs, hidden):
    return nn.Sequential(nn.Linear(inputs, hidden), nn.ELU(), nn.Linear(hidden, hidden),
                         nn.ELU(), nn.Linear(hidden, outputs))


def tensor_observation(arrays, normalizer, device):
    result = {key: torch.as_tensor(arrays[key], device=device) for key in (
        "present", "classes", "controls", "light")}
    result["continuous"] = torch.as_tensor(normalizer.normalize(
        arrays["continuous"], arrays["present"], arrays["controls"]), device=device)
    result["valid"] = torch.as_tensor(continuous_valid(arrays["present"], arrays["controls"]), device=device)
    return result


class AttentionEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        width = cfg.embedding
        self.ego = nn.Linear(8, width)
        self.route = nn.Linear(6, width)
        self.weather = nn.Linear(4, width)
        self.control = nn.Linear(2 + 2 + len(LIGHT_CLASSES), width)
        self.waypoint = nn.Linear(2, width)
        self.object = nn.Linear(len(OBJECT_FIELDS), width)
        self.object_class = nn.Embedding(3, width)
        self.object_slot = nn.Embedding(SLOTS, width)
        self.token_type = nn.Embedding(6, width)
        self.waypoint_position = nn.Embedding(WAYPOINTS, width)
        self.attention = nn.MultiheadAttention(width, cfg.attention_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(width)
        self.feedforward = nn.Sequential(nn.Linear(width, width * 2), nn.ELU(), nn.Linear(width * 2, width))
        self.norm2 = nn.LayerNorm(width)

    def forward(self, obs, return_attention=False):
        leading = obs["continuous"].shape[:-1]
        obs = {key: value.reshape(-1, *value.shape[len(leading):]) for key, value in obs.items()}
        c = obs["continuous"]
        batch = c.shape[0]
        tokens = [self.ego(c[:, :8]) + self.token_type.weight[0],
                  self.route(c[:, 8:14]) + self.token_type.weight[1],
                  self.weather(c[:, 16:20]) + self.token_type.weight[2],
                  self.control(torch.cat([c[:, 14:16], obs["controls"].float(),
                      F.one_hot(obs["light"].long(), len(LIGHT_CLASSES)).float()], -1)) + self.token_type.weight[3]]
        scene = torch.stack(tokens, 1)
        waypoints = self.waypoint(c[:, CONTEXT:OBJECT_OFFSET].reshape(batch, WAYPOINTS, 2))
        waypoints = waypoints + self.token_type.weight[4] + self.waypoint_position.weight
        objects = self.object(c[:, OBJECT_OFFSET:].reshape(batch, SLOTS, len(OBJECT_FIELDS)))
        objects = (objects + self.token_type.weight[5] + self.object_class(obs["classes"].long()) +
                   self.object_slot.weight)
        tokens = torch.cat([scene, waypoints, objects], 1)
        padding = torch.cat([torch.zeros(batch, 4 + WAYPOINTS, dtype=torch.bool, device=c.device),
                             ~obs["present"].bool()], 1)
        # Slot positions support reconstructing tracked slots; numeric actor IDs are never input.
        attended, weights = self.attention(tokens[:, :1], tokens, tokens, key_padding_mask=padding,
                                          need_weights=return_attention, average_attn_weights=False)
        encoded = self.norm1(tokens[:, :1] + attended)
        encoded = self.norm2(encoded + self.feedforward(encoded)).reshape(*leading, -1)
        if return_attention:
            return encoded, weights.squeeze(-2).reshape(*leading, self.attention.num_heads, -1)
        return encoded


class RSSM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.encoder = AttentionEncoder(cfg)
        self.recurrent = nn.GRUCell(cfg.stochastic + 3 + 4, cfg.deterministic)
        self.prior_net = mlp(cfg.deterministic, cfg.stochastic * 2, cfg.hidden)
        self.posterior_net = mlp(cfg.deterministic + cfg.embedding, cfg.stochastic * 2, cfg.hidden)
        features = cfg.deterministic + cfg.stochastic
        self.continuous_head = mlp(features, CONTINUOUS, cfg.hidden)
        self.presence_head = mlp(features, SLOTS + 2, cfg.hidden)
        self.class_head = mlp(features, SLOTS * 3 + len(LIGHT_CLASSES), cfg.hidden)
        self.reward_head = mlp(features, 1, cfg.hidden)
        self.termination_head = mlp(features, 1, cfg.hidden)
        self.register_buffer("termination_logit_offset", torch.zeros(()))

    @property
    def feature_size(self):
        return self.cfg.deterministic + self.cfg.stochastic + 4 + 3

    def initial(self, batch, device):
        return (torch.zeros(batch, self.cfg.deterministic, device=device),
                torch.zeros(batch, self.cfg.stochastic, device=device))

    def distribution(self, parameters):
        mean, raw_std = parameters.chunk(2, -1)
        return Normal(mean, F.softplus(raw_std.clamp(-10, 10)) + 0.1)

    def posterior(self, h, encoded):
        return self.distribution(self.posterior_net(torch.cat([h, encoded], -1)))

    def observe_initial(self, obs, sample=False, encoded=None):
        h, _ = self.initial(len(obs["continuous"]), obs["continuous"].device)
        posterior = self.posterior(h, self.encoder(obs) if encoded is None else encoded)
        return h, posterior.rsample() if sample else posterior.mean

    def transition(self, state, applied, next_obs=None, sample=True, weather=None, encoded=None):
        h, z = state
        if weather is None:
            if next_obs is None:
                raise ValueError("Prior transitions require the known normalized weather context")
            weather = next_obs["continuous"][:, 16:20]
        h = self.recurrent(torch.cat([z, applied, weather], -1), h)
        prior = self.distribution(self.prior_net(h))
        posterior = self.posterior(h, self.encoder(next_obs) if encoded is None else encoded) if next_obs is not None else None
        distribution = posterior if posterior is not None else prior
        z = distribution.rsample() if sample else distribution.mean
        return (h, z), prior, posterior

    def features(self, state, previous_applied, weather):
        return torch.cat([*state, weather, previous_applied], -1)

    def termination_logits(self, state):
        return self.termination_head(torch.cat(state, -1)).squeeze(-1) + self.termination_logit_offset

    def heads(self, state, prior_mean=None):
        features = torch.cat(state, -1)
        classes = self.class_head(features)
        presence = self.presence_head(features)
        return {"continuous": self.continuous_head(features), "present": presence[..., :SLOTS],
                "controls": presence[..., SLOTS:], "classes": classes[..., :SLOTS * 3].reshape(-1, SLOTS, 3),
                "light": classes[..., SLOTS * 3:], "reward": self.reward_head(features).squeeze(-1),
                "termination": self.termination_logits(state if prior_mean is None else (state[0], prior_mean))}

    def reconstruction_loss(self, prediction, target):
        squared = (prediction["continuous"] - target["continuous"]).square()
        valid = target["valid"].float().clone()
        valid[:, 5:8] = 0
        valid[:, 16:20] = 0
        # Equal group weights prevent padded/object coordinates overwhelming ego and route learning.
        groups = ((0, 8), (8, CONTEXT), (CONTEXT, OBJECT_OFFSET), (OBJECT_OFFSET, CONTINUOUS))
        continuous = sum((squared[:, a:b] * valid[:, a:b]).sum() / valid[:, a:b].sum().clamp_min(1)
                         for a, b in groups) / len(groups)
        presence = F.binary_cross_entropy_with_logits(prediction["present"], target["present"].float())
        presence = presence + F.binary_cross_entropy_with_logits(prediction["controls"], target["controls"].float())
        class_loss = F.cross_entropy(prediction["classes"].reshape(-1, 3), target["classes"].long().reshape(-1),
                                    reduction="none").reshape(-1, SLOTS)
        class_loss = (class_loss * target["present"]).sum() / target["present"].sum().clamp_min(1)
        class_loss = class_loss + F.cross_entropy(prediction["light"], target["light"].long())
        return continuous + presence + class_loss

    def sequence_loss(self, batch, normalizer, device):
        obs = tensor_observation(batch, normalizer, device)
        actions = torch.as_tensor(batch["applied_actions"], device=device)
        rewards = torch.as_tensor(batch["rewards"], device=device)
        ends = torch.as_tensor(batch["terminated"], device=device).float()
        length = actions.shape[1]
        sequence_valid = torch.as_tensor(batch.get("sequence_valid", np.ones(actions.shape[:2], bool)), device=device)
        burn_in = (sequence_valid.sum(1) - 1).clamp(max=self.cfg.burn_in)
        at = lambda index: {key: value[:, index] for key, value in obs.items()}
        encoded = self.encoder(obs)
        state = self.observe_initial(at(0), sample=True, encoded=encoded[:, 0])
        posterior_states, prior_states, priors, posteriors, targets = [], [], [], [], []
        for t in range(length):
            state, prior, posterior = self.transition(state, actions[:, t], at(t + 1),
                weather=obs["continuous"][:, t, 16:20], encoded=encoded[:, t + 1])
            active = sequence_valid[:, t] & (t >= burn_in)
            # Burn-in and padding carry no gradients; short failure episodes still receive supervision.
            state = tuple(torch.where(active[:, None], value, value.detach()) for value in state)
            if not active.any():
                continue
            posterior_states.append(tuple(value[active] for value in state))
            prior_states.append((state[0][active], prior.rsample()[active]))
            priors.append((prior.mean[active], prior.stddev[active]))
            posteriors.append((posterior.mean[active], posterior.stddev[active]))
            targets.append({**{key: value[active] for key, value in at(t + 1).items()},
                            "reward": rewards[active, t], "termination": ends[active, t]})
        concatenate = lambda values: tuple(torch.cat([value[i] for value in values]) for i in (0, 1))
        prior, posterior = Normal(*concatenate(priors)), Normal(*concatenate(posteriors))
        prediction = self.heads(concatenate(posterior_states))
        prior_prediction = self.heads(concatenate(prior_states), prior_mean=prior.mean)
        target = {key: torch.cat([item[key] for item in targets]) for key in targets[0]}
        frozen_q = Normal(posterior.mean.detach(), posterior.stddev.detach())
        frozen_p = Normal(prior.mean.detach(), prior.stddev.detach())
        kl = 0.8 * kl_divergence(frozen_q, prior).sum(-1) + 0.2 * kl_divergence(posterior, frozen_p).sum(-1)
        terms = {"reconstruction": self.reconstruction_loss(prediction, target),
                 "prior_prediction": self.reconstruction_loss(prior_prediction, target),
                 "reward": F.smooth_l1_loss(prior_prediction["reward"], target["reward"]),
                 "termination": F.binary_cross_entropy_with_logits(prior_prediction["termination"], target["termination"]),
                 "kl": kl.clamp_min(self.cfg.free_nats).mean()}
        total = (terms["reconstruction"] + self.cfg.prior_prediction_weight * terms["prior_prediction"] +
                 terms["reward"] + terms["termination"] + self.cfg.kl_weight * terms["kl"])
        return total, {name: float(value.detach()) for name, value in terms.items()}

    @torch.no_grad()
    def encode_episode(self, episode, normalizer, device):
        return self.encode_episodes([episode], normalizer, device)[0]

    @torch.no_grad()
    def encode_episodes(self, episodes, normalizer, device, batch_size=16):
        """Batch independent histories; never reset a retained episode halfway through."""
        result = []
        for start in range(0, len(episodes), batch_size):
            group = episodes[start:start + batch_size]
            length = max(map(len, group))
            from .schema import OBS_KEYS
            arrays = {key: np.stack([np.concatenate([ep.arrays[key],
                      np.repeat(ep.arrays[key][-1:], length - len(ep), axis=0)]) for ep in group])
                      for key in OBS_KEYS}
            obs = tensor_observation(arrays, normalizer, device)
            # Bound attention memory while still batching frames across time and episodes.
            flat = {key: value.reshape(-1, *value.shape[2:]) for key, value in obs.items()}
            encoded = torch.cat([self.encoder({key: value[i:i + 2048] for key, value in flat.items()})
                                 for i in range(0, len(flat["continuous"]), 2048)])
            encoded = encoded.reshape(len(group), length + 1, -1)
            at = lambda t: {key: value[:, t] for key, value in obs.items()}
            state = self.observe_initial(at(0), encoded=encoded[:, 0])
            weather = obs["continuous"][:, 0, 16:20]
            previous = torch.as_tensor(arrays["continuous"][:, 0, 5:8], device=device)
            features = [self.features(state, previous, weather)]
            actions = torch.as_tensor(np.stack([np.pad(ep.arrays["applied_actions"],
                ((0, length - len(ep)), (0, 0))) for ep in group]), device=device)
            for t in range(length):
                state, _, _ = self.transition(state, actions[:, t], at(t + 1), sample=False,
                                              weather=weather, encoded=encoded[:, t + 1])
                features.append(self.features(state, actions[:, t], weather))
            features = torch.stack(features, 1).cpu().numpy()
            result.extend(features[i, :len(ep) + 1].copy() for i, ep in enumerate(group))
        return result
