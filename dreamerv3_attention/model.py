"""Upstream DreamerV3 components with one project-specific fusion encoder."""

import torch
from torch import nn
from tensordict import TensorDict
from tensordict.nn import (InteractionType, NormalParamExtractor, ProbabilisticTensorDictModule,
                          ProbabilisticTensorDictSequential, TensorDictModule, TensorDictSequential)
from torchrl.data import Bounded, Composite, Unbounded
from torchrl.envs.model_based import DreamerEnv
from torchrl.modules import (DreamerV3ImageDecoder, DreamerV3ImageEncoder, DreamerV3MLP,
                             RSSMPosteriorV3, RSSMPriorV3, RSSMRolloutV3, RSSMStateEstimatorV3,
                             SymExpTwoHot, TanhNormal, WorldModelWrapper)
from torchrl.objectives import DreamerV3ActorLoss, DreamerV3Loss, DreamerV3ModelLoss, DreamerV3ValueLoss, SoftUpdate
from torchrl.trainers.algorithms import DreamerV3OptimizationStepper, DreamerV3Optimizer


class FusionEncoder(nn.Module):
    """One-query cross attention is linear in token count; RSSM owns time."""

    def __init__(self, cfg, token_count):
        super().__init__()
        self.attention = cfg.attention
        self.image = DreamerV3ImageEncoder(depth=cfg.image_depth, mults=(1, 2, 4))
        self.image_projection = nn.Linear(self.image.output_features((3, cfg.image_size, cfg.image_size)), cfg.hidden)
        self.token_projection = nn.Linear(16, cfg.hidden)
        self.validity_projection = nn.Linear(5, cfg.hidden)
        self.positions = nn.Parameter(torch.zeros(token_count + 2, cfg.hidden))
        self.query = nn.Parameter(torch.zeros(1, 1, cfg.hidden))
        self.pool = nn.MultiheadAttention(cfg.hidden, 1, batch_first=True) if cfg.attention else None
        self.output = DreamerV3MLP(cfg.hidden, cfg.hidden, depth=1, num_cells=cfg.hidden)
        self.last_attention = None

    def forward(self, pixels, tokens, mask, pixels_valid, radar_valid):
        batch = pixels.shape[:-3]
        # Invalid rows never enter a projection, including NaN sensor payloads.
        mask = mask.bool() & torch.isfinite(tokens).all(-1)
        tokens = torch.where(mask[..., None], tokens, 0)
        image = self.image_projection(self.image(pixels))
        features = self.token_projection(tokens)
        null = torch.zeros_like(image).unsqueeze(-2)
        features = torch.cat((image.unsqueeze(-2), features, null), -2) + self.positions
        pixels_valid = pixels_valid.bool().reshape(*batch, 1)
        valid = torch.cat((pixels_valid, mask, torch.ones_like(mask[..., :1])), -1)
        features = features.reshape(-1, features.shape[-2], features.shape[-1])
        valid = valid.reshape(-1, valid.shape[-1])
        if self.pool is not None:
            fused, weights = self.pool(self.query.expand(features.shape[0], -1, -1), features, features,
                                       key_padding_mask=~valid, need_weights=True)
            self.last_attention = weights.detach().reshape(*batch, -1)
            fused = fused[:, 0]
        else:
            fused = (features * valid[..., None]).sum(1) / valid.sum(1, keepdim=True)
            self.last_attention = None
        validity = torch.cat((pixels_valid.float(), radar_valid.float()), -1).reshape(-1, 5)
        return self.output(fused + self.validity_projection(validity)).reshape(*batch, -1)


def structured_target(tokens, mask):
    clean = torch.where(mask[..., None].bool(), torch.nan_to_num(tokens), 0)
    return torch.cat((clean.flatten(-2), mask.float()), -1)


def expert_actor_loss(actor, posterior, sample):
    """Align posterior at t+1 with the expert action chosen at t+1."""
    state = posterior["state"]
    expert = sample.get("expert", None)
    if expert is None or sample.batch_size[-1] < 2:
        return state.new_zeros(())
    mask = expert[..., 1:].bool() & ~sample["next", "done"][..., :-1, 0].bool()
    if not mask.any():
        return state.new_zeros(())
    features = posterior[..., :-1].detach()[mask].clone()
    targets = sample["action"][..., 1:, :][mask].detach().clamp(-0.999, 0.999)
    return -actor.get_dist(features).log_prob(targets).mean()


def corrective_steering_loss(actor, posterior, sample):
    """Teach the actor route feedback on its own replay states, not just demos."""
    tokens = sample["next", "tokens"].detach()
    valid = (sample["next", "mask"][..., 0].bool() & sample["next", "mask"][..., 1].bool() &
             (tokens[..., 1, 15] == -1) & torch.isfinite(tokens[..., 1, :2]).all(-1) &
             (tokens[..., 0, 15] > 0.5) & ~sample["next", "done"][..., 0].bool())
    if not valid.any():
        return posterior["state"].new_zeros(())
    x, y = tokens[..., 1, 0] * 100., tokens[..., 1, 1] * 100.
    # Same pure-pursuit reference as RouteController, before safety limits.
    steer = (torch.atan2(2. * 2.8 * y, (x*x+y*y).clamp_min(1.)) / 0.7).clamp(-0.999, 0.999)
    features = posterior.detach()[valid].clone()
    actor.get_dist(features)
    # Fit the executed mean directly and keep sigma small. A Gaussian NLL of
    # atanh(label) let sigma grow to absorb error (mean 0.34, p90 1.2 on
    # stage7_run_4 replay), which left the head under-steering and noisy.
    # Offline on frozen stage7 latents this reached R2 0.93 vs 0.40 for the NLL.
    # 0.1 (was 0.3): stage7_run_8 under-steered small lateral errors (mean |steer| 0.02-0.03
    # vs 0.05-0.08 for the controller) and drifted 1.5-3 m before colliding.
    error = (torch.tanh(features["loc"][..., 0]) - steer[valid]) / 0.1
    excess_sigma = torch.relu(features["scale"][..., 0] - 0.1)
    return (error ** 2).mean() + 10. * (excess_sigma ** 2).mean()


def corrective_longitudinal_loss(actor, posterior, sample):
    """Imitate safe speed feedback on aligned learner-visited next states."""
    targets = sample.get(("next", "longitudinal_target"), None)
    validity = sample.get(("next", "longitudinal_target_valid"), None)
    if targets is None or validity is None:
        return posterior["state"].new_zeros(())
    targets = targets.detach()[..., 0]
    valid = (validity[..., 0].bool() & ~sample["next", "done"][..., 0].bool() &
             torch.isfinite(targets) & (targets.abs() <= 1.))
    if not valid.any():
        return posterior["state"].new_zeros(())
    features = posterior.detach()[valid].clone()
    actor.get_dist(features)
    normal = torch.distributions.Normal(features["loc"][..., 1], features["scale"][..., 1])
    return -normal.log_prob(torch.atanh(targets[valid].clamp(-0.999, 0.999))).mean()


class ExpertRegularizedLoss(DreamerV3Loss):
    """Add an optional supervised actor term using aligned real expert data."""

    def __init__(self, *args, actor_bc_weight=0., actor_steering_bc_weight=0.,
                 actor_longitudinal_bc_weight=0., **kwargs):
        super().__init__(*args, **kwargs)
        self.actor_bc_weight = actor_bc_weight
        self.actor_steering_bc_weight = actor_steering_bc_weight
        self.actor_longitudinal_bc_weight = actor_longitudinal_bc_weight

    @property
    def in_keys(self):
        keys = super().in_keys
        if getattr(self, "actor_bc_weight", 0.):
            keys = [*keys, "expert"]
        if getattr(self, "actor_steering_bc_weight", 0.):
            for key in (("next", "tokens"), ("next", "mask")):
                if key not in keys:
                    keys.append(key)
        if getattr(self, "actor_longitudinal_bc_weight", 0.):
            keys = [*keys, ("next", "longitudinal_target"), ("next", "longitudinal_target_valid")]
        return keys

    def forward(self, sample):
        result = super().forward(sample)
        if self.actor_bc_weight:
            bc = expert_actor_loss(self.actor_loss.actor_model,
                                   result[self.tensor_keys.replay_context], sample)
            result["loss_actor"] = result["loss_actor"] + self.actor_bc_weight * bc
            result["actor_bc_loss"] = bc.detach()
        if self.actor_steering_bc_weight:
            steering_bc = corrective_steering_loss(self.actor_loss.actor_model,
                                                  result[self.tensor_keys.replay_context], sample)
            result["loss_actor"] = result["loss_actor"] + self.actor_steering_bc_weight * steering_bc
            result["actor_steering_bc_loss"] = steering_bc.detach()
        if self.actor_longitudinal_bc_weight:
            longitudinal_bc = corrective_longitudinal_loss(self.actor_loss.actor_model,
                                                           result[self.tensor_keys.replay_context], sample)
            result["loss_actor"] = result["loss_actor"] + self.actor_longitudinal_bc_weight * longitudinal_bc
            result["actor_longitudinal_bc_loss"] = longitudinal_bc.detach()
        return result


class Agent:
    """Compose public losses and share trained modules with recurrent inference."""

    def __init__(self, cfg, token_count):
        cfg.validate()
        self.cfg, self.token_count = cfg, token_count
        self.device = torch.device(cfg.device)
        self.state_dim = cfg.categoricals * cfg.classes
        latent = self.state_dim + cfg.belief
        self.encoder = FusionEncoder(cfg, token_count)
        self.prior = RSSMPriorV3(action_shape=(2,), action_dim=2, hidden_dim=cfg.hidden,
                                rnn_hidden_dim=cfg.belief, num_categoricals=cfg.categoricals,
                                num_classes=cfg.classes, unimix=0.01)
        self.posterior = RSSMPosteriorV3(hidden_dim=cfg.hidden, rnn_hidden_dim=cfg.belief,
                                        num_categoricals=cfg.categoricals, num_classes=cfg.classes,
                                        obs_embed_dim=cfg.hidden, unimix=0.01, use_rms_norm=True)
        prior = TensorDictModule(self.prior, ["state", "belief", "action"],
                                 [("next", "prior_logits"), ("next", "state"), ("next", "belief")])
        posterior = TensorDictModule(self.posterior, [("next", "belief"), ("next", "encoded_latents")],
                                     [("next", "posterior_logits"), ("next", "state")])
        self.reward = DreamerV3MLP(latent, cfg.bins, depth=2, num_cells=cfg.hidden, outscale=0.0)
        self.continue_net = DreamerV3MLP(latent, 1, depth=2, num_cells=cfg.hidden)
        self.reward_decode = SymExpTwoHot(cfg.bins)
        inputs = [("next", key) for key in ("pixels", "tokens", "mask", "pixels_valid", "radar_valid")]
        self.world = TensorDictSequential(
            TensorDictModule(self.encoder, inputs, [("next", "encoded_latents")]),
            RSSMRolloutV3(prior, posterior),
            TensorDictModule(DreamerV3ImageDecoder(latent, (3, cfg.image_size, cfg.image_size),
                                                  depth=cfg.image_depth, mults=(1, 2, 4), num_blocks=1),
                             [("next", "state"), ("next", "belief")], [("next", "reco_pixels")]),
            TensorDictModule(DreamerV3MLP(latent, token_count * 17, depth=2, num_cells=cfg.hidden),
                             [("next", "state"), ("next", "belief")], [("next", "reco_structured")]),
            TensorDictModule(structured_target, [("next", "tokens"), ("next", "mask")],
                             [("next", "structured")]),
            TensorDictModule(self.reward, [("next", "state"), ("next", "belief")], [("next", "reward_logits")]),
            TensorDictModule(self.reward_decode, [("next", "reward_logits")], [("next", "reward")]),
            TensorDictModule(self.continue_net, [("next", "state"), ("next", "belief")], [("next", "continue_pred")]),
        )
        self.actor = ProbabilisticTensorDictSequential(
            TensorDictModule(DreamerV3MLP(latent, 4, depth=2, num_cells=cfg.hidden),
                             ["state", "belief"], ["params"]),
            TensorDictModule(NormalParamExtractor(), ["params"], ["loc", "scale"]),
            ProbabilisticTensorDictModule(["loc", "scale"], ["action"], distribution_class=TanhNormal,
                                          default_interaction_type=InteractionType.RANDOM, return_log_prob=True),
        )
        self.value = TensorDictSequential(
            TensorDictModule(DreamerV3MLP(latent, cfg.bins, depth=2, num_cells=cfg.hidden, outscale=0.0),
                             ["state", "belief"], ["state_value_logits"]),
            TensorDictModule(SymExpTwoHot(cfg.bins), ["state_value_logits"], ["state_value"]),
        )
        imagination = WorldModelWrapper(
            TensorDictModule(self.prior, ["state", "belief", "action"], ["_", "state", "belief"]),
            TensorDictSequential(
                TensorDictModule(self.reward, ["state", "belief"], ["reward_logits"]),
                TensorDictModule(self.reward_decode, ["reward_logits"], ["reward"])),
        )
        mb_env = DreamerEnv(imagination, prior_shape=(self.state_dim,), belief_shape=(cfg.belief,), device=self.device)
        mb_env.observation_spec = Composite(state=Unbounded(self.state_dim, device=self.device),
                                            belief=Unbounded(cfg.belief, device=self.device), device=self.device)
        mb_env.state_spec = mb_env.observation_spec.clone()
        mb_env.action_spec = Bounded(-1, 1, (2,), device=self.device)
        mb_env.reward_spec = Unbounded((1,), device=self.device)
        continuation = TensorDictSequential(
            TensorDictModule(self.continue_net, ["state", "belief"], ["continue_logits"]),
            TensorDictModule(nn.Sigmoid(), ["continue_logits"], ["continuation"]),
        )
        model_loss = DreamerV3ModelLoss(self.world, kl_mode="separate", lambda_dynamic=1.0,
                                        lambda_representation=0.1, unimix=0.01, free_bits=1.0,
                                        lambda_continue=1.0, num_reward_bins=cfg.bins,
                                        reco_symlog=[False, False], detach_output=False)
        model_loss.set_keys(pixels=["pixels", "structured"], reco_pixels=["reco_pixels", "reco_structured"])
        actor_loss = DreamerV3ActorLoss(self.actor, self.value, mb_env, continuation_model=continuation,
                                       imagination_horizon=cfg.horizon)
        actor_loss.make_value_estimator(gamma=cfg.gamma, lmbda=0.95)
        value_loss = DreamerV3ValueLoss(self.value, value_loss="two_hot", num_value_bins=cfg.bins,
                                       actor_loss=actor_loss, slow_critic_regularization=1.0)
        self.loss = ExpertRegularizedLoss(model_loss, actor_loss, value_loss,
                                         replay_value_loss_weight=0.0,
                                         actor_bc_weight=cfg.actor_bc_weight,
                                         actor_steering_bc_weight=cfg.actor_steering_bc_weight,
                                         actor_longitudinal_bc_weight=cfg.actor_longitudinal_bc_weight).to(self.device)
        updater = SoftUpdate(value_loss, tau=0.02)
        optimizer = DreamerV3Optimizer(self.loss.parameters(), lr=cfg.learning_rate, warmup_steps=cfg.warmup_updates)
        self.stepper = DreamerV3OptimizationStepper(self.loss, optimizer, updater,
                                                    compile_train_step=False, cudagraph=False)
        self.estimator = RSSMStateEstimatorV3(self.prior, self.posterior)

    def initial(self):
        return TensorDict({"state": torch.zeros(self.state_dim, device=self.device),
                           "belief": torch.zeros(self.cfg.belief, device=self.device),
                           "previous_action": torch.zeros(2, device=self.device),
                           "is_init": torch.ones(1, dtype=torch.bool, device=self.device)}, [])

    @torch.no_grad()
    def act(self, observation, carry, deterministic=False):
        from tensordict.nn import set_interaction_type

        td = carry.clone()
        td["encoded_latents"] = self.encoder(*(torch.as_tensor(observation[k], device=self.device)
                                               for k in ("pixels", "tokens", "mask", "pixels_valid", "radar_valid")))
        self.estimator(td)
        with set_interaction_type(InteractionType.DETERMINISTIC if deterministic else InteractionType.RANDOM):
            self.actor(td)
        return td["action"].cpu().numpy(), td.select("state", "belief")

    def update(self, sample, *, world_model_only=False):
        sample = sample.to(self.device)
        sample["next", "structured"] = structured_target(sample["next", "tokens"], sample["next", "mask"])
        if world_model_only:
            # Reuse the upstream model objective and optimizer, leaving actor,
            # critic, return normalization and target-update state untouched.
            optimizer = self.stepper.optimizer
            optimizer.zero_grad(set_to_none=True)
            losses, _ = self.loss.model_loss(sample)
            total = sum(value for key, value in losses.items() if key.startswith("loss_"))
            if not torch.isfinite(total).all():
                raise FloatingPointError("World-model loss is not finite")
            total.backward()
            optimizer.step()
            return losses.detach()
        return self.stepper.step(None, sample)

    @torch.no_grad()
    def validate(self, sample):
        sample = sample.to(self.device).clone()
        sample["next", "structured"] = structured_target(sample["next", "tokens"], sample["next", "mask"])
        losses, _ = self.loss.model_loss(sample)
        return {key: float(value) for key, value in losses.items() if value.numel() == 1}
