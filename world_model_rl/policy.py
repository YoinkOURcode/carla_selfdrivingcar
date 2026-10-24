"""Optional observed-state skip for SAC; imagined skips use only model predictions."""

import torch
from torch.nn import functional as F

from .schema import CONTINUOUS, CONTROLS, SLOTS, LIGHT_CLASSES, OBJECT_OFFSET, OBJECT_FIELDS, VEHICLE_SLOTS


def policy_feature_size(model, cfg):
    extra = CONTINUOUS + SLOTS * 4 + CONTROLS + len(LIGHT_CLASSES)
    return model.feature_size + (extra if cfg.training.policy_observation_skip else 0)


def augment_policy_features(features, obs):
    present = obs["present"].bool()
    classes = F.one_hot(obs["classes"].long(), 3).float() * present[..., None]
    skip = torch.cat([obs["continuous"], present.float(), classes.flatten(-2),
                      obs["controls"].float(), F.one_hot(obs["light"].long(), len(LIGHT_CLASSES)).float()], -1)
    # Keep latent/weather offsets and the trailing applied-action contract unchanged.
    return torch.cat([features[..., :-3], skip, features[..., -3:]], -1)


def permute_policy_objects(features, cfg):
    """Training-only slot augmentation; keep latent history, context and actions.

    Current object geometry, masks and class indicators travel together. Separate
    vehicle/pedestrian groups preserve the observation's slot contract. Shared
    object normalization makes the permuted skip describe the same physical scene.
    """
    if not cfg.training.policy_observation_skip:
        raise ValueError("Object permutation requires the observed-state policy skip")
    result = features.clone()
    start = cfg.model.deterministic + cfg.model.stochastic + 4
    batch = features.shape[0]
    permutation = torch.cat([
        torch.rand(batch, VEHICLE_SLOTS, device=features.device).argsort(-1),
        VEHICLE_SLOTS + torch.rand(batch, SLOTS - VEHICLE_SLOTS, device=features.device).argsort(-1)], -1)
    fields = len(OBJECT_FIELDS)
    objects = features[:, start + OBJECT_OFFSET:start + CONTINUOUS].reshape(batch, SLOTS, fields)
    result[:, start + OBJECT_OFFSET:start + CONTINUOUS] = objects.gather(
        1, permutation[..., None].expand(-1, -1, fields)).flatten(1)
    present = start + CONTINUOUS
    result[:, present:present + SLOTS] = features[:, present:present + SLOTS].gather(1, permutation)
    classes = present + SLOTS
    indicators = features[:, classes:classes + 3 * SLOTS].reshape(batch, SLOTS, 3)
    result[:, classes:classes + 3 * SLOTS] = indicators.gather(
        1, permutation[..., None].expand(-1, -1, 3)).flatten(1)
    return result


def predicted_observation(prediction, applied, weather, normalizer):
    continuous = prediction["continuous"].clone()
    continuous[:, 5:8] = (applied - applied.new_tensor(normalizer.mean[5:8])) / applied.new_tensor(normalizer.scale[5:8])
    continuous[:, 16:20] = weather
    present = prediction["present"].sigmoid() >= .5
    controls = prediction["controls"].sigmoid() >= .5
    # Absent dimensions have the same zero-masked representation as real observations.
    continuous[:, 14:15] *= controls
    continuous[:, 30:] *= present.repeat_interleave(7, -1)
    return {"continuous": continuous, "present": present,
            "classes": prediction["classes"].argmax(-1), "controls": controls,
            "light": prediction["light"].argmax(-1)}


def physical_observation(obs, normalizer):
    continuous = (obs['continuous'] * obs['continuous'].new_tensor(normalizer.scale) +
                  obs['continuous'].new_tensor(normalizer.mean))
    continuous[..., OBJECT_OFFSET:] = continuous[..., OBJECT_OFFSET:].masked_fill(
        ~obs['present'].bool().repeat_interleave(len(OBJECT_FIELDS), -1), 0)
    continuous[..., 14:15] = continuous[..., 14:15].masked_fill(~obs['controls'].bool(), 0)
    return {**obs, 'continuous': continuous}


def current_policy_observation(features, cfg, normalizer):
    """Decode the current replay/predicted skip; never fetch a future observation."""
    start = cfg.model.deterministic + cfg.model.stochastic + 4
    skip = features[:, start:-3]
    controls = CONTINUOUS + SLOTS * 4
    obs = {'continuous': skip[:, :CONTINUOUS],
        'present': skip[:, CONTINUOUS:CONTINUOUS + SLOTS].bool(),
        'controls': skip[:, controls:controls + CONTROLS].bool(),
        'light': skip[:, controls + CONTROLS:].argmax(-1)}
    return physical_observation(obs, normalizer)
