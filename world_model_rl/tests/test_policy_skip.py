"""Observed skips match replay; synthetic skips are decoded from causal predictions."""

import numpy as np
import torch

from world_model_rl.model import RSSM, tensor_observation
from world_model_rl.policy import policy_feature_size, augment_policy_features, permute_policy_objects
from world_model_rl.sac import SAC, real_replay
from world_model_rl.schema import Normalizer, CONTINUOUS, OBJECT_OFFSET, SLOTS
from world_model_rl.teachers import geometry_actions
from world_model_rl.training import imagine
from world_model_rl.tests.test_pipeline import cfg, make_episode


def test_replay_and_imagination_observation_skip_contract(cfg, monkeypatch):
    cfg.training.policy_observation_skip = True
    model = RSSM(cfg.model)
    episode = make_episode(cfg)
    norm = Normalizer.fit([episode])
    real = real_replay([episode], model, norm, "cpu", 1, policy_skip=True)
    assert real.arrays["features"].shape[-1] == policy_feature_size(model, cfg)
    start = cfg.model.deterministic + cfg.model.stochastic + 4
    actual = tensor_observation(episode.arrays, norm, "cpu")["continuous"].numpy()
    np.testing.assert_allclose(real.arrays["features"][:, start:start + CONTINUOUS], actual[:-1])
    original_heads = model.heads
    def heads(state, prior_mean=None):
        prediction = original_heads(state, prior_mean)
        prediction["continuous"] = torch.zeros_like(prediction["continuous"])
        prediction["present"] = torch.full_like(prediction["present"], -100)
        prediction["controls"] = torch.full_like(prediction["controls"], -100)
        prediction["termination"] = torch.full_like(prediction["termination"], -100)
        return prediction
    monkeypatch.setattr(model, "heads", heads)
    sac = SAC(policy_feature_size(model, cfg), cfg, "cpu")
    synthetic, _ = imagine(model, sac, real, norm, cfg, "cpu", np.random.default_rng(3))
    assert len(synthetic) > 0
    next_skip = synthetic.arrays["next_features"][:, start:start + CONTINUOUS]
    # The observed speed/route are nonzero. Synthetic skips use the zero prediction.
    np.testing.assert_equal(next_skip[:, :5], 0)
    np.testing.assert_equal(next_skip[:, 8:16], 0)
    np.testing.assert_allclose(next_skip[:, 5:8],
        (synthetic.arrays["next_features"][:, -3:] - norm.mean[5:8]) / norm.scale[5:8])


def test_object_permutation_preserves_scene_labels_history_and_control_context(cfg):
    cfg.training.policy_observation_skip = True
    ep = make_episode(cfg, steps=64)
    a = ep.arrays
    a['continuous'][:, 20:OBJECT_OFFSET] = np.array(
        [[5, 0], [10, 0], [20, 0], [30, 0], [40, 0]]).ravel()
    a['continuous'][:, OBJECT_OFFSET:OBJECT_OFFSET + 7] = [8, 0, -8, 0, 2, 1, 1]
    a['continuous'][:, OBJECT_OFFSET + 6 * 7:OBJECT_OFFSET + 7 * 7] = [20, 3, -2, -1, .3, .3, 1]
    a['present'][:, 6] = True
    a['classes'][:, 6] = 2
    a['controls'][:, 0] = True
    a['light'][:] = 2
    a['continuous'][:, 14] = np.linspace(30, 1, 65)
    norm = Normalizer.fit([ep])
    model = RSSM(cfg.model)
    base = torch.randn(65, model.feature_size)
    features = augment_policy_features(base, tensor_observation(a, norm, 'cpu'))
    original = features.clone()
    changed = permute_policy_objects(features, cfg)
    torch.testing.assert_close(features, original)  # Replay is never modified.
    start = cfg.model.deterministic + cfg.model.stochastic + 4
    torch.testing.assert_close(changed[:, :start + OBJECT_OFFSET], original[:, :start + OBJECT_OFFSET])
    torch.testing.assert_close(changed[:, start + CONTINUOUS + 4 * SLOTS:],
                               original[:, start + CONTINUOUS + 4 * SLOTS:])
    assert not torch.equal(changed, original)
    transformed = dict(a)
    transformed['continuous'] = norm.denormalize(changed[:, start:start + CONTINUOUS].numpy())
    transformed['present'] = changed[:, start + CONTINUOUS:start + CONTINUOUS + SLOTS].numpy().astype(bool)
    transformed['classes'] = changed[:, start + CONTINUOUS + SLOTS:start + CONTINUOUS + 4 * SLOTS].reshape(65, SLOTS, 3).argmax(-1).numpy()
    np.testing.assert_allclose(geometry_actions(transformed, True, True), geometry_actions(a, True, True), atol=1e-6)
