"""Publish real training and evaluation frames for a passive local viewer."""

import fcntl
import json
import time
import traceback
from pathlib import Path

import numpy as np


class LiveTraining:
    def __init__(self, output, cfg, mode):
        self.output, self.cfg = Path(output), cfg
        self.mode = "evaluation" if mode == "evaluation" else "training"
        self.state = {"status": "starting", "message": f"Waiting for the first {self.mode} episode.",
                      "mode": self.mode,
                      "run": str(self.output.resolve()), "training_mode": mode,
                      "total_frames": cfg.steps}
        self.last_publish = 0.
        self.previous = None
        self.distance = 0.
        from .viewer import EpisodeRewards
        self.rewards = EpisodeRewards()
        self.last_learner_mode = None

    def __enter__(self):
        self.lock = (self.output / "live.lock").open("a+")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.write()
        except BaseException:
            self.lock.close()
            raise
        return self

    def write(self):
        self.state["updated_at"] = time.time()
        temporary = self.output / "live.tmp"
        temporary.write_text(json.dumps(self.state, allow_nan=False))
        temporary.replace(self.output / "live.json")

    def publish(self, env, observation, record, *, reset=False):
        from .viewer import snapshot

        env = getattr(env, "env", env)
        position = env.ego.get_location()
        xyz = np.asarray([position.x, position.y, position.z])
        if reset:
            self.distance = 0.
            self.rewards.reset()
        else:
            if self.previous is not None:
                self.distance += float(np.linalg.norm(xyz - self.previous))
            self.rewards.add(record, dt_s=self.cfg.dt)
        if record.get("learner_mode"):
            self.last_learner_mode = record["learner_mode"]
        self.previous = xyz
        now = time.monotonic()
        final = ((self.mode == "training" and record.get("frames", 0) >= self.cfg.steps) or
                 record.get("terminated") or record.get("truncated"))
        if not reset and not final and now-self.last_publish < .2:
            return
        self.last_publish = now
        # The actor attended to the preceding observation. Do not overlay those
        # weights on this post-step image or run extra inference for monitoring.
        frame = snapshot(env, observation, None, self.distance, record, self.mode)
        reward_state = self.rewards.snapshot()
        displayed_record = dict(record)
        if self.last_learner_mode is not None:
            displayed_record.setdefault("learner_mode", self.last_learner_mode)
        scenario = getattr(env, "scenario", {})
        message = (f"Evaluating {record.get('scenario', scenario.get('name', 'scenario'))}."
                   if self.mode == "evaluation" else "Observing real CARLA training episodes.")
        self.state.update(frame, status="evaluating" if self.mode == "evaluation" else "training", message=message,
                          metrics=displayed_record,
                          **reward_state,
                          route_start=scenario.get("route_start", self.cfg.route_start),
                          route_end=scenario.get("route_end", self.cfg.route_end))
        self.write()

    def finish(self, reason):
        self.state.update(status="ended", message=f"{self.mode.capitalize()} ended: {reason}.", stop_reason=reason)
        self.write()

    def __exit__(self, kind, error, tb):
        try:
            if kind is not None:
                (self.output / "error.txt").write_text("".join(traceback.format_exception(kind, error, tb)))
                self.state.update(status="error", message=f"{kind.__name__}: {error}")
                self.write()
        finally:
            self.lock.close()


class TrainingMonitor:
    """Read snapshots only; never connect to, tick, or control CARLA."""

    def __init__(self, run=None, root="runs"):
        self.run = Path(run) if run else None
        self.root = Path(root)

    def read(self):
        paths = [self.run / "live.json"] if self.run else list(self.root.rglob("live.json"))
        states = []
        for path in paths:
            try:
                # A held publisher lock distinguishes a live process from old
                # metadata, including after a crash or a killed process.
                with path.with_name("live.lock").open("r") as lock:
                    try:
                        fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
                        active = False
                    except BlockingIOError:
                        active = True
                    state = json.loads(path.read_text())
                state["active"] = active
                state["age_s"] = max(0., time.time() - state["updated_at"])
                if not active and state["status"] in ("starting", "training", "evaluating"):
                    state.update(status="ended", message="Run ended; showing its last captured frame.")
                states.append(state)
            except (OSError, ValueError, KeyError, TypeError):
                continue
        if not states:
            return {"status": "idle", "active": False,
                    "message": "No live feed found. Waiting for a CARLA training or evaluation run.",
                    "run": str(self.run) if self.run else None}
        return max(states, key=lambda state: (state["active"], state["updated_at"]))
