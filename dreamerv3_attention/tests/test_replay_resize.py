"""Capacity changes retain real replay rows and restore writable storage."""

from dataclasses import replace

import pytest
import torch

from dreamerv3_attention.tests.test_model import small_config, synthetic_batch
from dreamerv3_attention.training import Trainer


def replay_rows(cfg, start, count):
    rows = synthetic_batch(cfg).reshape(-1)[:count]
    rows["step_id"] = torch.arange(start, start + count)
    rows["episode"] = rows["step_id"] // cfg.sequence
    return rows


@pytest.mark.parametrize("count", [4, 8, 12])
def test_resume_grows_partial_full_and_wrapped_replay(tmp_path, count):
    torch.set_num_threads(1)
    cfg = small_config(capacity=8)
    original = Trainer(cfg, 5)
    original.agent.update(synthetic_batch(cfg))
    for start in range(0, count, 4):
        original.replay.extend(replay_rows(cfg, start, 4))
    original.frames, original.updates = count, 1
    original.completed_episodes, original.debt = 3, .75
    expected = original.replay[:]
    expected = expected[expected["step_id"].argsort()].exclude("index")
    path = tmp_path / "original.pt"
    original.checkpoint(path)

    resumed = Trainer.restore(path, replace(cfg, capacity=32))
    assert resumed.replay.storage.max_size == 32
    assert resumed.replay.storage.state_dict()["_storage"]["state"].shape[0] == 32
    assert len(resumed.replay) == min(count, 8)
    torch.testing.assert_close(resumed.replay[:].exclude("index"), expected, rtol=0, atol=0)
    torch.testing.assert_close(resumed.agent.loss.state_dict(), original.agent.loss.state_dict(), rtol=0, atol=0)
    torch.testing.assert_close(resumed.agent.stepper.state_dict(), original.agent.stepper.state_dict(), rtol=0, atol=0)
    assert (resumed.frames, resumed.updates, resumed.completed_episodes, resumed.debt) == (count, 1, 3, .75)

    resumed.replay.extend(replay_rows(cfg, count, 4))
    torch.testing.assert_close(resumed.replay[:]["step_id"], torch.arange(max(0, count - 8), count + 4))
    # Cross the old capacity, fill the new capacity, and wrap it too.
    for start in range(count + 4, count + 36, 4):
        resumed.replay.extend(replay_rows(cfg, start, 4))
    assert len(resumed.replay) == 32
    assert resumed.replay[:]["step_id"].sort().values.tolist() == list(range(count + 4, count + 36))
    for _ in range(8):
        sample = resumed.replay.sample().reshape(cfg.batch, cfg.sequence)
        assert (sample["step_id"].diff(dim=-1) == 1).all()
        assert (sample["episode"] == sample["episode"][:, :1]).all()

    destination = tmp_path / "resized.pt"
    resumed.checkpoint(destination)
    restored = Trainer.restore(destination)
    torch.testing.assert_close(restored.replay[:], resumed.replay[:], rtol=0, atol=0)
    assert restored.replay.writer.state_dict() == resumed.replay.writer.state_dict()
    # The source checkpoint retains its original allocation and data.
    unchanged = Trainer.restore(path)
    torch.testing.assert_close(unchanged.replay[:], original.replay[:], rtol=0, atol=0)


def test_resume_recovers_checkpoint_saved_after_out_of_bounds_write(tmp_path):
    torch.set_num_threads(1)
    cfg = small_config(capacity=8)
    original = Trainer(cfg, 5)
    original.replay.extend(replay_rows(cfg, 0, 8))
    path = tmp_path / "failed.pt"
    original.checkpoint(path)
    saved = torch.load(path, weights_only=False)
    saved["config"]["capacity"] = 32
    # Failed writes can inflate length and advance the cursor past allocation.
    saved["replay"]["_storage"]["_len"] = 13
    saved["replay"]["_writer"]["_cursor"] = 9
    torch.save(saved, path)

    resumed = Trainer.restore(path)
    assert len(resumed.replay) == 8
    assert resumed.replay.storage.state_dict()["_storage"]["state"].shape[0] == 32
    assert resumed.replay[:]["step_id"].tolist() == list(range(8))
    resumed.replay.add(replay_rows(cfg, 8, 1)[0])
    assert resumed.replay[:]["step_id"].tolist() == list(range(9))


def test_resume_rejects_capacity_that_would_discard_valid_replay(tmp_path):
    torch.set_num_threads(1)
    cfg = small_config(capacity=8)
    trainer = Trainer(cfg, 5)
    trainer.replay.extend(replay_rows(cfg, 0, 8))
    path = tmp_path / "original.pt"
    trainer.checkpoint(path)
    with pytest.raises(ValueError, match="capacity.*8.*transitions"):
        Trainer.restore(path, replace(cfg, capacity=4))


@pytest.mark.parametrize("capacity", [8, 32])
def test_resume_uses_configured_batch_and_sequence_size(tmp_path, capacity):
    torch.set_num_threads(1)
    cfg = small_config(capacity=8)
    cfg.warmup_frames = 8
    trainer = Trainer(cfg, 5)
    rows = replay_rows(cfg, 0, 8)
    rows["episode"].zero_()
    trainer.replay.extend(rows)
    path = tmp_path / "original.pt"
    trainer.checkpoint(path)

    changed = replace(cfg, capacity=capacity, batch=3, sequence=8)
    resumed = Trainer.restore(path, changed)
    sample = resumed.replay.sample()
    assert sample.numel() == changed.batch * changed.sequence
    sample = sample.reshape(changed.batch, changed.sequence)
    assert (sample["step_id"].diff(dim=-1) == 1).all()
