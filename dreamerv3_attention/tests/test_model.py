import torch
from tensordict import TensorDict

from dreamerv3_attention.config import Config
from dreamerv3_attention.model import Agent, FusionEncoder


def small_config(**kwargs):
    return Config(image_size=16, hidden=16, belief=16, categoricals=2, classes=4,
                  image_depth=2, bins=15, horizon=3, batch=2, sequence=4,
                  warmup_frames=4, warmup_updates=0, **kwargs)


def synthetic_batch(cfg, tokens=5):
    b, t = cfg.batch, cfg.sequence
    return TensorDict({
        "state": torch.zeros(b, t, cfg.categoricals * cfg.classes),
        "belief": torch.zeros(b, t, cfg.belief),
        "action": torch.rand(b, t, 2) * 2 - 1,
        "next": {"pixels": torch.randint(0, 256, (b, t, 3, cfg.image_size, cfg.image_size), dtype=torch.uint8),
                 "tokens": torch.randn(b, t, tokens, 16), "mask": torch.ones(b, t, tokens, dtype=torch.bool),
                 "pixels_valid": torch.ones(b, t, dtype=torch.bool),
                 "radar_valid": torch.ones(b, t, 4, dtype=torch.bool),
                 "reward": torch.randn(b, t, 1), "done": torch.zeros(b, t, 1, dtype=torch.bool),
                 "terminated": torch.zeros(b, t, 1, dtype=torch.bool)},
    }, [b, t])


def test_upstream_losses_update_model_actor_value_and_attention():
    torch.set_num_threads(1)
    torch.manual_seed(2)
    cfg = small_config()
    agent = Agent(cfg, 5)
    groups = [agent.encoder, agent.prior, agent.actor, agent.value]
    before = [[p.detach().clone() for p in m.parameters()] for m in groups]
    for _ in range(2):
        metrics = agent.update(synthetic_batch(cfg))
        assert all(torch.isfinite(v).all() for v in metrics.values())
    for module, old in zip(groups, before):
        assert any(not torch.equal(p, v) for p, v in zip(module.parameters(), old))
    assert agent.encoder.query.grad is not None
    assert agent.encoder.query.grad.abs().sum() > 0


def test_baseline_and_masked_nan_invariance():
    for enabled in (False, True):
        encoder = FusionEncoder(small_config(attention=enabled), 5)
        pixels = torch.zeros(2, 3, 16, 16, dtype=torch.uint8)
        tokens = torch.randn(2, 5, 16)
        mask = torch.zeros(2, 5, dtype=torch.bool)
        first = encoder(pixels, tokens, mask, torch.ones(2, dtype=torch.bool), torch.ones(2, 4))
        second = encoder(pixels, tokens.fill_(float("nan")), mask, torch.ones(2, dtype=torch.bool), torch.ones(2, 4))
        assert first.shape == (2, 16)
        torch.testing.assert_close(first, second)
        if enabled:
            assert (encoder.last_attention[:, 1:6] == 0).all()


def test_inference_bounds_and_reset_isolation():
    torch.manual_seed(3)
    agent = Agent(small_config(), 5)
    observation = {"pixels": torch.zeros(3, 16, 16, dtype=torch.uint8),
                   "tokens": torch.zeros(5, 16), "mask": torch.ones(5, dtype=torch.bool),
                   "pixels_valid": torch.tensor(True), "radar_valid": torch.ones(4, dtype=torch.bool)}
    first = agent.initial()
    dirty = first.clone()
    dirty["state"].fill_(100)
    dirty["belief"].fill_(100)
    dirty["previous_action"].fill_(1)
    torch.manual_seed(4)
    action, state = agent.act(observation, first, True)
    torch.manual_seed(4)
    action2, state2 = agent.act(observation, dirty, True)
    assert abs(action).max() <= 1
    torch.testing.assert_close(state["belief"], state2["belief"])
    assert (action == action2).all()


def test_model_validation_is_finite_and_does_not_change_parameters():
    torch.set_num_threads(1)
    agent = Agent(small_config(), 5)
    before = [p.detach().clone() for p in agent.loss.parameters()]
    metrics = agent.validate(synthetic_batch(agent.cfg))
    assert "loss_model_reward" in metrics and all(torch.isfinite(torch.tensor(v)) for v in metrics.values())
    for parameter, old in zip(agent.loss.parameters(), before):
        torch.testing.assert_close(parameter, old, rtol=0, atol=0)


def test_world_model_only_updates_encoder_and_dynamics_but_preserves_policy_and_targets():
    torch.set_num_threads(1)
    agent = Agent(small_config(), 5)
    # Include nonzero actor/critic optimizer momentum from a previous joint
    # step, so stale gradients cannot silently move the frozen networks.
    agent.update(synthetic_batch(agent.cfg))
    world_before = {key: value.clone() for key, value in agent.world.state_dict().items()}
    frozen = {key: value.clone() for key, value in agent.loss.state_dict().items()
              if key.startswith(("actor_loss.", "value_loss."))}
    metrics = agent.update(synthetic_batch(agent.cfg), world_model_only=True)
    assert all(torch.isfinite(value).all() for value in metrics.values())
    assert any(not torch.equal(world_before[key], value) for key, value in agent.world.state_dict().items())
    assert "loss_actor" not in metrics and "loss_value" not in metrics
    checked = 0
    for key, value in agent.loss.state_dict().items():
        if key in frozen:
            # Actor loss also owns shared world-model dynamics. Freeze only
            # policy/value/normalization parameters, not those shared dynamics.
            if any(part in key for part in ("actor_model", "value_model", "retnorm")):
                torch.testing.assert_close(value, frozen[key], rtol=0, atol=0)
                checked += 1
    assert checked > 0


def test_pixel_augment_keeps_shape_dtype_and_range_and_is_off_by_default():
    from dreamerv3_attention.model import augment_pixels

    torch.manual_seed(0)
    pixels = torch.randint(0, 256, (3, 4, 3, 16, 16), dtype=torch.uint8)
    assert augment_pixels(pixels, 0.) is pixels
    changed = augment_pixels(pixels, 1.)
    assert changed.shape == pixels.shape and changed.dtype == torch.uint8
    assert (changed != pixels).any()
    # Probability 0.5 leaves some sequences untouched: one draw per batch row.
    mixed = augment_pixels(pixels, .5)
    untouched = [bool((mixed[i] == pixels[i]).all()) for i in range(3)]
    torch.manual_seed(1)
    runs = [[bool((augment_pixels(pixels, .5)[i] == pixels[i]).all()) for i in range(3)] for _ in range(8)]
    assert any(any(r) for r in runs) and any(not all(r) for r in runs), untouched


def test_encoder_corrupts_images_only_during_training_updates():
    torch.manual_seed(0)
    agent = Agent(small_config(pixel_augment=1.), 5)
    batch = synthetic_batch(agent.cfg)
    args = [batch["next", key] for key in ("pixels", "tokens", "mask", "pixels_valid", "radar_valid")]
    clean = agent.encoder(*args)
    assert torch.equal(clean, agent.encoder(*args))  # acting and evaluation never corrupt
    agent.encoder.augment.active = True
    assert not torch.equal(clean, agent.encoder(*args))
    agent.encoder.augment.active = False
    agent.update(batch)
    assert agent.encoder.augment.active is False  # restored after the update


def test_augmentation_and_route_aux_add_no_parameters_so_old_checkpoints_load():
    plain = Agent(small_config(), 5)
    new = Agent(small_config(pixel_augment=.7, route_aux_weight=1.), 5)
    assert plain.loss.state_dict().keys() == new.loss.state_dict().keys()
    new.loss.load_state_dict(plain.loss.state_dict())


def test_route_aux_loss_measures_the_route_point_in_metres_and_ignores_invalid_steps():
    from dreamerv3_attention.model import route_aux_loss, structured_target

    cfg = small_config()
    sample = synthetic_batch(cfg)
    tokens = sample["next", "tokens"]
    tokens[..., 1, 15] = -1.
    tokens[..., 0, 15] = 1.
    tokens[..., 1, 0], tokens[..., 1, 1] = .05, .01  # 5 m ahead, 1 m to the side
    sample["next", "tokens"] = tokens
    reco = structured_target(tokens, sample["next", "mask"])[..., :5 * 16 + 5].clone()
    posterior = TensorDict({"next": {"reco_structured": reco.clone()}}, sample.batch_size)
    assert route_aux_loss(posterior, sample) < 1e-10
    reco[..., 17] += .02  # 2 m lateral error, averaged with the exact x: (0 + 2 ** 2) / 2
    posterior = TensorDict({"next": {"reco_structured": reco}}, sample.batch_size)
    assert abs(route_aux_loss(posterior, sample).item() - 2.) < 1e-3
    tokens[..., 1, 15] = 0.  # no valid route token: nothing to learn from
    sample["next", "tokens"] = tokens
    assert route_aux_loss(posterior, sample) == 0.


def test_route_aux_flows_into_the_world_model_through_the_model_loss():
    torch.manual_seed(3)
    cfg = small_config(route_aux_weight=2.)
    agent = Agent(cfg, 5)
    sample = synthetic_batch(cfg)
    sample["next", "tokens"][..., 1, 15] = -1.
    sample["next", "tokens"][..., 0, 15] = 1.
    result = agent.loss(sample)
    assert result["route_aux_loss"] > 0
    result["loss_model_reco"].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in agent.encoder.parameters())


def test_config_validates_augment_probability_and_aux_weight():
    import pytest

    for bad in (dict(pixel_augment=1.5), dict(pixel_augment=-.1), dict(pixel_augment=float("nan")),
                dict(route_aux_weight=-1.), dict(route_aux_weight=float("inf"))):
        with pytest.raises(ValueError):
            Config(**bad).validate()
    Config(pixel_augment=.7, route_aux_weight=1.).validate()


def test_without_pixels_the_encoder_ignores_the_image_and_checkpoints_still_load():
    torch.manual_seed(0)
    plain = Agent(small_config(), 5)
    blind = Agent(small_config(use_pixels=False), 5)
    assert plain.loss.state_dict().keys() == blind.loss.state_dict().keys()
    blind.loss.load_state_dict(plain.loss.state_dict())
    batch = synthetic_batch(blind.cfg)
    args = [batch["next", key] for key in ("pixels", "tokens", "mask", "pixels_valid", "radar_valid")]
    first = blind.encoder(*args)
    args[0] = torch.randint(0, 256, args[0].shape, dtype=torch.uint8)  # different picture, same tokens
    assert torch.equal(first, blind.encoder(*args))
    assert not torch.equal(first, plain.encoder(*args))  # the default still looks at pixels


def test_training_update_without_pixels_is_finite_and_leaves_the_image_decoder_untrained():
    torch.manual_seed(1)
    cfg = small_config(use_pixels=False)
    agent = Agent(cfg, 5)
    metrics = agent.update(synthetic_batch(cfg))
    assert all(torch.isfinite(v).all() for v in metrics.values())
    decoder = agent.world[2].module
    assert all(p.grad is None or not p.grad.any() for p in decoder.parameters())
