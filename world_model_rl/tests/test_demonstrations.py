"""Expert initialization trains a fresh actor without fitting held-out episodes."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from world_model_rl.config import Config
from world_model_rl.sac import SAC, Demonstrations


def test_actor_warmup_learns_controls_without_changing_critics_or_sac_count():
    torch.set_num_threads(1)
    torch.manual_seed(7)
    cfg = Config()
    cfg.model.hidden = 16
    sac = SAC(5, cfg, "cpu")
    features = np.random.default_rng(7).normal(size=(64, 5)).astype(np.float32)
    actions = np.tile(np.array([.3, .7, 0], np.float32), (64, 1))
    before_q = {key: value.clone() for key, value in sac.q1.state_dict().items()}
    initial = np.mean((sac.act(features, True) - actions) ** 2)
    for _ in range(250):
        sac.clone_expert({"features": features, "actions": actions})
    assert np.mean((sac.act(features, True) - actions) ** 2) < initial / 3
    assert sac.updates == 0
    assert all(torch.equal(value, before_q[key]) for key, value in sac.q1.state_dict().items())


@pytest.mark.parametrize("split,collector", [("validation", "autopilot"), ("test", "autopilot"), ("train", "sac")])
def test_demonstrations_reject_heldout_or_nonexpert_data_before_encoding(split, collector):
    episode = SimpleNamespace(metadata={"split": split, "collector": collector})
    with pytest.raises(ValueError, match="expert episodes from the training split"):
        Demonstrations([episode], None, None, None, 1)


def test_anchored_sac_updates_both_actor_and_critics():
    cfg = Config()
    cfg.model.hidden = 16
    cfg.training.behavior_cloning_weight = 10
    sac = SAC(5, cfg, "cpu")
    rng = np.random.default_rng(8)
    features = rng.normal(size=(16, 5)).astype(np.float32)
    actions = np.tile(np.array([0., .7, 0.], np.float32), (16, 1))
    before = [value.detach().clone() for value in sac.q1.parameters()]
    metrics = sac.update({"features": features, "actions": actions, "rewards": np.ones(16),
                         "next_features": features, "terminated": np.zeros(16)},
                        {"features": features, "actions": actions})
    assert sac.updates == 1 and np.isfinite(list(metrics.values())).all()
    assert "behavior_cloning_loss" in metrics
    assert any(not torch.equal(a, b) for a, b in zip(before, sac.q1.parameters()))
