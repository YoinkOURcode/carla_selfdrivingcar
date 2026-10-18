"""A persisted wall-clock deadline shared by offline fitting and SAC commands."""

import hashlib
import fcntl
import json
import math
import shutil
import time
from pathlib import Path


class TrainingClock:
    """Do not reset elapsed time when the next training stage or resume starts.

    Callbacks check this clock between complete optimizer updates. Checkpointing
    never interrupts an optimizer step. Long encoding/inference operations can
    delay observation of the deadline, but cannot authorize another update.
    """

    def __init__(self, path, seconds=18000, checkpoint_seconds=300):
        if not all(math.isfinite(x) and x > 0 for x in (seconds, checkpoint_seconds)):
            raise ValueError("Training and checkpoint intervals must be finite and positive")
        self.path = Path(path)
        self.seconds, self.checkpoint_seconds = seconds, checkpoint_seconds
        self.anchor_wall, self.anchor_monotonic = time.time(), time.monotonic()
        self._lock_file = None
        self.value = json.loads(self.path.read_text()) if self.path.exists() else None
        if self.value is not None:
            if (self.value.get("format") != 1 or self.value.get("seconds") != seconds or
                    self.value.get("checkpoint_seconds") != checkpoint_seconds):
                raise ValueError("Existing training clock settings differ; its deadline cannot be reset")

    def now(self):
        # A backwards wall-clock adjustment must not extend this invocation.
        return max(time.time(), self.anchor_wall + time.monotonic() - self.anchor_monotonic)

    def _write(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(self.value, indent=2) + "\n")
        temporary.replace(self.path)

    def start(self):
        if self._lock_file is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            lock = self.path.with_suffix(self.path.suffix + ".lock").open("a+")
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                lock.close()
                raise RuntimeError(f"Another training command owns this clock: {self.path}") from None
            self._lock_file = lock
            if self.path.exists():
                current = json.loads(self.path.read_text())
                if (current.get("format") != 1 or current.get("seconds") != self.seconds or
                        current.get("checkpoint_seconds") != self.checkpoint_seconds):
                    self.close()
                    raise ValueError("Existing training clock settings differ; its deadline cannot be reset")
                self.value = current
        if self.value is None:
            started = self.now()
            self.value = {"format": 1, "seconds": self.seconds,
                          "checkpoint_seconds": self.checkpoint_seconds,
                          "started_unix": started, "deadline_unix": started + self.seconds,
                          "checkpoints": []}
            self._write()

    def close(self):
        if self._lock_file is not None:
            self._lock_file.close()
            self._lock_file = None

    @property
    def expired(self):
        return self.value is not None and self.now() >= self.value["deadline_unix"]

    @property
    def checkpoint_due(self):
        if self.value is None:
            return False
        records = self.value["checkpoints"]
        return (not records or self.expired or
                self.now() - records[-1]["saved_unix"] >= self.checkpoint_seconds)

    def summary(self):
        if self.value is None:
            return {"started": False, "seconds": self.seconds}
        return {"path": str(self.path), "started": True, "seconds": self.seconds,
                "started_unix": self.value["started_unix"],
                "deadline_unix": self.value["deadline_unix"],
                "elapsed_seconds": max(0, self.now() - self.value["started_unix"]),
                "deadline_reached": self.expired}

    def record_checkpoint(self, path, stage, counters, force=False):
        """Archive an already atomically saved checkpoint; keep all candidates."""
        if not force and not self.checkpoint_due:
            return
        if not stage.replace("_", "").isalnum():
            raise ValueError("Checkpoint stage must contain only letters, numbers and underscores")
        self.start()
        path = Path(path)
        archive = path.parent / "checkpoints" / f"{path.stem}_{len(self.value['checkpoints']):06d}_{stage}.pt"
        if archive.exists():
            raise FileExistsError(f"Refusing to overwrite an archived candidate: {archive}")
        archive.parent.mkdir(parents=True, exist_ok=True)
        temporary = archive.with_suffix(".pt.tmp")
        shutil.copyfile(path, temporary)
        temporary.replace(archive)
        digest = hashlib.sha256()
        with archive.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        self.value["checkpoints"].append({"path": str(archive), "source": str(path),
            "stage": stage, "saved_unix": self.now(), "sha256": digest.hexdigest(),
            "deadline_reached": self.expired, **counters})
        self._write()
