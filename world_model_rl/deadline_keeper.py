"""File-only deadline backup, independent of whether the learning gates pass."""

import argparse
import hashlib
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path


def archive_latest(root, deadline):
    root = Path(root)
    source = root / "agent.pt"
    if not source.exists():
        source = root / "world_model.pt"
    target = root / "deadline_checkpoint.pt"
    if target.exists():
        raise FileExistsError("The deadline backup already exists; it must not be replaced")
    temporary = target.with_suffix(".pt.tmp")
    # One source descriptor pins a complete inode even if the trainer atomically
    # replaces the latest-checkpoint pathname during this copy.
    with source.open("rb") as src, temporary.open("xb") as dst:
        source_mtime = os.fstat(src.fileno()).st_mtime
        shutil.copyfileobj(src, dst)
        dst.flush()
        os.fsync(dst.fileno())
    temporary.replace(target)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    now = time.time()
    report = {"source": str(source), "source_checkpoint_mtime_unix": source_mtime,
              "path": str(target), "sha256": digest, "deadline_unix": deadline,
              "saved_unix": now, "seconds_after_deadline": now - deadline,
              "saved_utc": datetime.fromtimestamp(now, timezone.utc).isoformat(),
              "meaning": "Latest complete checkpoint at deadline; not a claim of gate or driving success"}
    (root / "deadline_checkpoint.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def keep(experiment):
    root = Path(experiment)
    clock = json.loads((root / "training_clock.json").read_text())
    if clock.get("format") != 1 or clock["deadline_unix"] != clock["started_unix"] + clock["seconds"]:
        raise ValueError("Invalid persisted training deadline")
    while True:
        remaining = clock["deadline_unix"] - time.time()
        if remaining <= 0:
            break
        time.sleep(min(10, remaining))
    return archive_latest(root, clock["deadline_unix"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    print(json.dumps(keep(parser.parse_args().experiment), indent=2), flush=True)
