"""Twin-Q SAC on detached RSSM features, with requested actions in the critics."""

import copy
import math

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as F

from .model import mlp
from .control import apply_action_tensor
from .rewards import learning_rewards


class Actor(nn.Module):
    def __init__(self, features, hidden):
        super().__init__()
        self.net = mlp(features, 6, hidden)
        self.register_buffer("scale", torch.tensor([1.0, 0.5, 0.5]))
        self.register_buffer("bias", torch.tensor([0.0, 0.5, 0.5]))

    def forward(self, features, deterministic=False):
        mean, log_std = self.net(features).chunk(2, -1)
        distribution = Normal(mean, log_std.clamp(-5, 2).exp())
        raw = mean if deterministic else distribution.rsample()
        squashed = raw.tanh()
        action = squashed * self.scale + self.bias
        # Stable log-Jacobian, including the [0,1] throttle/brake scale.
        log_prob = (distribution.log_prob(raw) -
                    2 * (math.log(2) - raw - F.softplus(-2 * raw)) - self.scale.log()).sum(-1)
        return action, log_prob


class SAC:
    def __init__(self, features, cfg, device):
        self.cfg = cfg
        self.device = device
        self.actor = Actor(features, cfg.model.hidden).to(device)
        self.q1 = mlp(features + 3, 1, cfg.model.hidden).to(device)
        self.q2 = mlp(features + 3, 1, cfg.model.hidden).to(device)
        self.target1, self.target2 = copy.deepcopy(self.q1), copy.deepcopy(self.q2)
        self.target1.requires_grad_(False)
        self.target2.requires_grad_(False)
        self.log_alpha = torch.tensor(math.log(cfg.training.initial_alpha), device=device, requires_grad=True)
        lr = cfg.training.learning_rate
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.q_optimizer = torch.optim.Adam([*self.q1.parameters(), *self.q2.parameters()], lr=lr)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)
        self.updates = 0

    @torch.no_grad()
    def act(self, features, deterministic=False):
        return self.actor(torch.as_tensor(features, device=self.device), deterministic)[0].cpu().numpy()

    def clone_expert(self, batch):
        features = torch.as_tensor(batch["features"], device=self.device)
        if self.cfg.training.behavior_cloning_permute_objects:
            from .policy import permute_policy_objects
            features = permute_policy_objects(features, self.cfg)
        actions = torch.as_tensor(batch["actions"], device=self.device)
        chosen, _ = self.actor(features, deterministic=True)
        # A narrow initial exploration distribution follows the demonstrated controls.
        _, log_std = self.actor.net(features).chunk(2, -1)
        loss = self.cloning_loss(chosen, actions, features) + 0.01 * F.mse_loss(log_std, torch.full_like(log_std, -3.0))
        self.actor_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.actor_optimizer.step()
        return float(loss.detach())

    def cloning_loss(self, chosen, labels, features):
        loss = F.mse_loss(chosen, labels)
        if self.cfg.training.behavior_cloning_applied_weight:
            previous = features[:, -3:]
            applied = apply_action_tensor(chosen, previous, self.cfg.environment)
            expected = apply_action_tensor(labels, previous, self.cfg.environment)
            loss = loss + self.cfg.training.behavior_cloning_applied_weight * F.mse_loss(applied, expected)
        return loss

    def update(self, batch, demonstration=None):
        data = {key: torch.as_tensor(value, device=self.device).float() for key, value in batch.items()}
        features, actions, rewards, next_features, terminated = [data[key] for key in (
            "features", "actions", "rewards", "next_features", "terminated")]
        alpha = self.log_alpha.exp().detach()
        with torch.no_grad():
            next_action, next_log_prob = self.actor(next_features)
            next_input = torch.cat([next_features, next_action], -1)
            value = torch.minimum(self.target1(next_input), self.target2(next_input)).squeeze(-1)
            target = rewards + self.cfg.training.gamma * (1 - terminated) * (value - alpha * next_log_prob)
        inputs = torch.cat([features, actions], -1)
        critic_loss = F.mse_loss(self.q1(inputs).squeeze(-1), target) + F.mse_loss(self.q2(inputs).squeeze(-1), target)
        self.q_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_([*self.q1.parameters(), *self.q2.parameters()], 100)
        self.q_optimizer.step()

        self.q1.requires_grad_(False)
        self.q2.requires_grad_(False)
        chosen, log_prob = self.actor(features)
        actor_input = torch.cat([features, chosen], -1)
        q = torch.minimum(self.q1(actor_input), self.q2(actor_input)).squeeze(-1)
        actor_loss = (alpha * log_prob - q).mean()
        cloning_loss = None
        if demonstration is not None and self.cfg.training.behavior_cloning_weight:
            demo_features = torch.as_tensor(demonstration["features"], device=self.device)
            if self.cfg.training.behavior_cloning_permute_objects:
                from .policy import permute_policy_objects
                demo_features = permute_policy_objects(demo_features, self.cfg)
            demo_actions = torch.as_tensor(demonstration["actions"], device=self.device)
            demo_chosen, _ = self.actor(demo_features, deterministic=True)
            cloning_loss = self.cloning_loss(demo_chosen, demo_actions, demo_features)
            # Normalize Q's scale so the explicit demonstration weight stays meaningful.
            actor_loss = (alpha * log_prob - 2.5 * q / q.abs().mean().detach().clamp_min(1e-6)).mean()
            actor_loss = actor_loss + self.cfg.training.behavior_cloning_weight * cloning_loss
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()
        self.q1.requires_grad_(True)
        self.q2.requires_grad_(True)

        # Target entropy is -3 in the actor's physical request coordinates.
        alpha_loss = -(self.log_alpha * (log_prob.detach() - 3)).mean()
        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_optimizer.step()
        with torch.no_grad():
            for target_net, net in ((self.target1, self.q1), (self.target2, self.q2)):
                for target_param, param in zip(target_net.parameters(), net.parameters()):
                    target_param.lerp_(param, self.cfg.training.tau)
        self.updates += 1
        metrics = {"critic_loss": float(critic_loss.detach()), "actor_loss": float(actor_loss.detach()),
                   "alpha": float(self.log_alpha.exp().detach())}
        if cloning_loss is not None:
            metrics["behavior_cloning_loss"] = float(cloning_loss.detach())
        return metrics

    def state_dict(self):
        return {"actor": self.actor.state_dict(), "q1": self.q1.state_dict(), "q2": self.q2.state_dict(),
                "target1": self.target1.state_dict(), "target2": self.target2.state_dict(),
                "log_alpha": self.log_alpha.detach(), "actor_optimizer": self.actor_optimizer.state_dict(),
                "q_optimizer": self.q_optimizer.state_dict(), "alpha_optimizer": self.alpha_optimizer.state_dict(),
                "updates": self.updates}

    def load_state_dict(self, value):
        for name in ("actor", "q1", "q2", "target1", "target2"):
            getattr(self, name).load_state_dict(value[name])
        with torch.no_grad():
            self.log_alpha.copy_(value["log_alpha"])
        for name in ("actor_optimizer", "q_optimizer", "alpha_optimizer"):
            getattr(self, name).load_state_dict(value[name])
        self.updates = value["updates"]


class Replay:
    def __init__(self, arrays, version):
        self.arrays = arrays
        self.version = version

    def __len__(self):
        return len(self.arrays.get("rewards", []))

    def sample(self, count, rng):
        indices = rng.integers(len(self), size=count)
        return {key: value[indices] for key, value in self.arrays.items()}


class Demonstrations(Replay):
    """Expert requests on training episodes, balancing turns/braking against cruising."""

    def __init__(self, episodes, model, normalizer, device, version, cache=None, batch_size=16, policy_skip=False,
                 teacher="recorded", traffic_fraction=0.0):
        if not 0 <= traffic_fraction <= 1:
            raise ValueError("traffic_fraction must be in [0, 1]")
        if teacher not in ("recorded", "geometry_v1", "geometry_v2", "geometry_v3", "geometry_v4"):
            raise ValueError("Unknown demonstration teacher")
        if any(ep.metadata["split"] != "train" or (teacher == "recorded" and
               ep.metadata["collector"] not in ("autopilot", "dagger")) for ep in episodes):
            raise ValueError("Demonstrations must be expert episodes from the training split")
        if not episodes:
            raise ValueError("Expert warm-up requires autopilot training episodes")
        replay = real_replay(episodes, model, normalizer, device, version, cache, batch_size, policy_skip)
        super().__init__({key: replay.arrays[key] for key in ("features", "actions")}, version)
        if teacher != "recorded":
            from .teachers import geometry_actions
            self.arrays["actions"] = np.concatenate([geometry_actions(ep.arrays, calibrated=teacher in ("geometry_v2", "geometry_v3", "geometry_v4"),
                                                                     hold_stops=teacher in ("geometry_v3", "geometry_v4"),
                                                                     early_light_hold=teacher == "geometry_v4")
                                                   for ep in episodes])
        else:
            self.arrays["actions"] = np.concatenate([
                ep.arrays["expert_actions"] if ep.metadata["collector"] == "dagger" else ep.arrays["requested_actions"]
                for ep in episodes])
        actions = self.arrays["actions"]
        self.manoeuvres = np.flatnonzero((np.abs(actions[:, 0]) > .15) | (actions[:, 2] > .1))
        # Most red-light braking states are stationary waits. Sample moving
        # approaches separately so long waits do not drown out braking onset.
        continuous = np.concatenate([ep.arrays["continuous"][:-1] for ep in episodes])
        controls = np.concatenate([ep.arrays["controls"][:-1] for ep in episodes])
        light = np.concatenate([ep.arrays["light"][:-1] for ep in episodes])
        stopping = (controls[:, 0] & np.isin(light, [2, 3]) &
                    (continuous[:, 14] >= 0) & (continuous[:, 14] < 25) & (actions[:, 2] > .1))
        self.traffic_moving = np.flatnonzero(stopping & (continuous[:, 0] > .3))
        self.traffic_waiting = np.flatnonzero(stopping & (continuous[:, 0] <= .3))
        self.traffic_fraction = traffic_fraction
        self.episode_ids = [ep.metadata["episode_id"] for ep in episodes]

    def __len__(self):
        return len(self.arrays["actions"])

    def sample(self, count, rng):
        indices = rng.integers(len(self.arrays["actions"]), size=count)
        traffic_count = int(count * self.traffic_fraction)
        if len(self.manoeuvres):
            manoeuvre_count = (count - traffic_count) // 2
            indices[traffic_count:traffic_count + manoeuvre_count] = rng.choice(
                self.manoeuvres, size=manoeuvre_count)
        if traffic_count:
            moving_count = 3 * traffic_count // 4
            waiting_count = traffic_count - moving_count
            # Preserve ordinary sampling when a stratum has no recorded states.
            if len(self.traffic_moving):
                indices[:moving_count] = rng.choice(self.traffic_moving, size=moving_count)
            if len(self.traffic_waiting):
                indices[moving_count:moving_count + waiting_count] = rng.choice(
                    self.traffic_waiting, size=waiting_count)
        return {key: value[indices] for key, value in self.arrays.items()}


class HistoryCache:
    """Numeric episode IDs and an explicit model version; no content digests."""

    def __init__(self):
        self.version = None
        self.episodes = {}

    def encode(self, episodes, model, normalizer, device, version, batch_size=16):
        if version != self.version:
            self.episodes.clear()
            self.version = version
        missing = [ep for ep in episodes if ep.metadata["episode_id"] not in self.episodes]
        encoded = model.encode_episodes(missing, normalizer, device, batch_size)
        for ep, features in zip(missing, encoded):
            self.episodes[ep.metadata["episode_id"]] = features
        return [self.episodes[ep.metadata["episode_id"]] for ep in episodes]


def real_replay(episodes, model, normalizer, device, version, cache=None, batch_size=16, policy_skip=False, reward_cfg=None):
    chunks = []
    features = (cache or HistoryCache()).encode(episodes, model, normalizer, device, version, batch_size)
    for episode, features in zip(episodes, features):
        a = episode.arrays
        if policy_skip:
            from .model import tensor_observation
            from .policy import augment_policy_features
            features = augment_policy_features(torch.as_tensor(features),
                tensor_observation(a, normalizer, "cpu")).numpy()
        chunks.append({"features": features[:-1], "actions": a["requested_actions"],
                       "rewards": learning_rewards(a, reward_cfg) if reward_cfg is not None else a["rewards"], "next_features": features[1:],
                       "terminated": a["terminated"]})
    if not chunks:
        return Replay({}, version)
    return Replay({key: np.concatenate([chunk[key] for chunk in chunks]) for key in chunks[0]}, version)


def mixed_batch(real, synthetic, count, fraction, rng):
    if synthetic is not None and synthetic.version != real.version:
        raise ValueError("Synthetic and real replay have different representation versions")
    model_count = int(count * fraction) if synthetic is not None and len(synthetic) else 0
    batch = real.sample(count - model_count, rng)
    if model_count:
        generated = synthetic.sample(model_count, rng)
        batch = {key: np.concatenate([batch[key], generated[key]]) for key in batch}
    return batch
