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

    def write_frame(self, jpeg):
        # Written before live.json so the state never points at a frame that is not on disk yet.
        temporary = self.output / "live.tmp.jpg"
        temporary.write_bytes(jpeg)
        temporary.replace(self.output / "live.jpg")

    def publish(self, env, observation, record, *, reset=False, attention=None, attention_enabled=None):
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
        if not reset and not final and now-self.last_publish < .1:
            return
        self.last_publish = now
        # Callers publish before stepping, so ``attention`` belongs to ``observation``.
        # The final post-step frame has no weights; ``attention_enabled`` keeps its label correct.
        frame = snapshot(env, observation, attention, self.distance, record, self.mode)
        jpeg = frame.pop("image")
        if jpeg:
            self.write_frame(jpeg)
        frame["image"] = f"/api/training/frame.jpg?v={time.time_ns()}" if jpeg else None
        if attention_enabled is not None:
            frame["attention_enabled"] = attention_enabled
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
        self.scanned = 0.  # monotonic time of the last rescan
        self.latest = None

    def frame(self):
        """JPEG beside the live.json most recently returned by read(), or None."""
        try:
            return self.latest.with_name("live.jpg").read_bytes()
        except (OSError, AttributeError):
            return None

    def _states(self, paths):
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
                states.append((path, state))
            except (OSError, ValueError, KeyError, TypeError):
                continue
        return states

    def _rescan(self):
        # Parsing every old live.json is slow, and a live run rewrites its file constantly,
        # so only the most recently modified files can be the best candidates.
        def mtime(path):
            try:
                return path.stat().st_mtime
            except OSError:
                return 0.
        paths = sorted(self.root.rglob("live.json"), key=mtime, reverse=True)[:5]
        self.scanned = time.monotonic()
        return self._states(paths)

    def read(self):
        if self.run:
            states = self._states([self.run / "live.json"])
        else:
            # Polled ~10x/s: re-parse only the shown run, and look for newer runs every few seconds.
            states = (self._states([self.latest])
                      if self.latest is not None and time.monotonic() - self.scanned < 3 else [])
            if not states:
                states = self._rescan()
        if not states:
            return {"status": "idle", "active": False,
                    "message": "No live feed found. Waiting for a CARLA training or evaluation run.",
                    "run": str(self.run) if self.run else None}
        self.latest, best = max(states, key=lambda item: (item[1]["active"], item[1]["updated_at"]))
        return best
