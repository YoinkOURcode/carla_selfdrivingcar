"""Small CPU-only dynamics fixture. This provides no CARLA driving evidence."""

from dataclasses import asdict, replace

import numpy as np

from .driving import MIN_EPISODE_RETURN, WAYPOINT_SPACING_M, DrivingState, RewardConfig, SafetySupervisor, compute_reward
from .observations import pack_observation


class SyntheticEnv:
    def __init__(self, cfg):
        self.cfg = cfg
        self.supervisor = SafetySupervisor()
        self.reward_config = RewardConfig(**cfg.reward)

    def _observation(self):
        pixels = np.zeros((32, 32, 3), np.uint8)
        pixels[:, 12:20, :] = 80
        pixels[:, int(np.clip(16+self.state.lane_error_m, 0, 31)), :] = 255
        context = np.zeros(16, np.float32)
        context[:3] = [self.state.speed_mps/30., 10/30., self.state.lane_error_m/4.]
        context[15] = 1.
        return pack_observation(pixels, context, max_objects=self.cfg.max_objects+3,
                                model_size=(self.cfg.image_size, self.cfg.image_size),
                                radars={side: np.zeros((0, 4), np.float32)
                                        for side in ("front", "left", "right", "rear")})

    def reset(self):
        self.steps = 0
        self.episode_return = 0.0
        self.state = DrivingState(speed_limit_mps=10.)
        self.reward_progress_highwater = 0.0
        self.supervisor.reset()
        self.previous_action = (0., 0.)
        return self._observation()

    def step(self, action):
        before = replace(self.state)
        result = self.supervisor.step(before, action, self.cfg.dt)
        steer, longitudinal = result.action
        speed = max(0., before.speed_mps+longitudinal*3*self.cfg.dt)
        self.state.speed_mps = speed
        self.state.route_progress_m += speed*self.cfg.dt
        self.state.lane_error_m += steer*speed*self.cfg.dt
        self.steps += 1
        self.state.offroad = abs(self.state.lane_error_m) > self.state.lane_width_m / 2.
        self.state.lane_departure = self.state.offroad
        terminated = bool(self.state.collision or self.state.pedestrian_collision or self.state.offroad)
        truncated = self.steps >= self.cfg.max_episode_steps and not terminated
        highwater_before = self.reward_progress_highwater
        highwater_after = max(highwater_before, min(100., self.state.route_progress_m))
        waypoints_reached = (int(highwater_after // WAYPOINT_SPACING_M) -
                             int(highwater_before // WAYPOINT_SPACING_M))
        new_progress_m = highwater_after - highwater_before
        self.reward_progress_highwater = highwater_after
        reward = compute_reward(before, result.action, self.state, self.cfg.dt, terminated=terminated,
                                 truncated=truncated, previous_action=self.previous_action,
                                 waypoints_reached=waypoints_reached, new_progress_m=new_progress_m,
                                 target_speed_mps=result.target_speed_mps,
                                 mandated_stop=result.crossing_phase == "stopped", config=self.reward_config)
        self.episode_return += reward.reward
        reward_limit = self.episode_return < MIN_EPISODE_RETURN and not terminated
        if reward_limit:
            terminated, truncated = True, False
        self.previous_action = result.action
        info = {**asdict(self.state), "timestamp": self.steps*self.cfg.dt, "frame": self.steps,
                "applied_action": result.action, "interventions": result.intervention_reasons,
                "crossing_phase": result.crossing_phase, "target_speed_mps": result.target_speed_mps,
                "reward_components": reward.components, "waypoints_reached": waypoints_reached,
                "new_progress_m": new_progress_m, "route_length_m": 100., "success": False,
                "termination_reason": "reward_limit" if reward_limit else None,
                "episode_return": self.episode_return}
        return self._observation(), reward.reward, terminated, truncated, info

    def close(self):
        pass
