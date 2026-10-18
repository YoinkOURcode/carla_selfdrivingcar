"""Render throughput can change without accepting different simulation dynamics."""

import pytest

from world_model_rl.config import Config, require_compatible


def test_rendering_option_preserves_legacy_checkpoint_compatibility_and_physics_checks():
    cfg = Config()
    legacy = cfg.to_dict()
    del legacy["environment"]["no_rendering_mode"]
    cfg.environment.no_rendering_mode = True
    require_compatible(legacy, cfg)
    cfg.environment.fixed_delta_seconds = .1
    with pytest.raises(ValueError, match="environment differs"):
        require_compatible(legacy, cfg)


def test_no_rendering_rejects_camera_providers():
    cfg = Config()
    cfg.environment.no_rendering_mode = True
    cfg.environment.observation_provider = "custom_camera:factory"
    with pytest.raises(ValueError, match="ground-truth observation provider"):
        cfg.validate()
