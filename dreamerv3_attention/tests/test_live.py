"""Opt-in integration test; never starts a server or runs during offline tests."""

import os
from dataclasses import replace

import pytest

from dreamerv3_attention.config import Config
from dreamerv3_attention.environment import CarlaEnv
from dreamerv3_attention.training import workload_lock


@pytest.mark.skipif(os.environ.get("DREAMERV3_APPROVED_LIVE_TEST") != "1", reason="Requires approved dedicated CARLA session")
def test_live_sensor_frames_depth_and_cleanup():
    cfg = replace(Config(), max_episode_steps=16)
    with workload_lock(cfg):
        with CarlaEnv(cfg) as env:
            obs = env.reset()
            assert env.last_bundle.rgb.shape == (720, 1280, 3)
            assert env.last_bundle.depth.shape == (720, 1280)
            assert obs["radar_valid"].all()
            assert obs["pixels"].shape == (3, 64, 64)
            frame = env.frame
            _, _, terminated, truncated, info = env.step((0., -1.))
            assert env.frame > frame and info["timestamp"] > 0
            assert info["applied_action"][1] <= 0
            actors = [env.ego, *env.rig.sensors]
        assert all(not actor.is_alive for actor in actors)
