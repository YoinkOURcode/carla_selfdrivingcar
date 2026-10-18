"""Optional observed-state skip for SAC; imagined skips use only model predictions."""

import torch
from torch.nn import functional as F

from .schema import CONTINUOUS, SLOTS, LIGHT_CLASSES


def policy_feature_size(model, cfg):
    extra = CONTINUOUS + SLOTS * 4 + 2 + len(LIGHT_CLASSES)
    return model.feature_size + (extra if cfg.training.policy_observation_skip else 0)


def augment_policy_features(features, obs):
    present = obs["present"].bool()
    classes = F.one_hot(obs["classes"].long(), 3).float() * present[..., None]
    skip = torch.cat([obs["continuous"], present.float(), classes.flatten(-2),
                      obs["controls"].float(), F.one_hot(obs["light"].long(), len(LIGHT_CLASSES)).float()], -1)
    # Keep latent/weather offsets and the trailing applied-action contract unchanged.
    return torch.cat([features[..., :-3], skip, features[..., -3:]], -1)


def predicted_observation(prediction, applied, weather, normalizer):
    continuous = prediction["continuous"].clone()
    continuous[:, 5:8] = (applied - applied.new_tensor(normalizer.mean[5:8])) / applied.new_tensor(normalizer.scale[5:8])
    continuous[:, 16:20] = weather
    present = prediction["present"].sigmoid() >= .5
    controls = prediction["controls"].sigmoid() >= .5
    # Absent dimensions have the same zero-masked representation as real observations.
    continuous[:, 14:16] *= controls
    continuous[:, 30:] *= present.repeat_interleave(7, -1)
    return {"continuous": continuous, "present": present,
            "classes": prediction["classes"].argmax(-1), "controls": controls,
            "light": prediction["light"].argmax(-1)}
