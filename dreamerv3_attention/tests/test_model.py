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
