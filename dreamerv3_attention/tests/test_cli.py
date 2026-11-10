import json

import pytest

from dreamerv3_attention.__main__ import main, prepare_output
from dreamerv3_attention.config import Config


@pytest.mark.parametrize("command", ["train", "collect", "evaluate", "infer"])
@pytest.mark.parametrize("budget", [None, "0", "-1", "86401", "nan", "inf"])
def test_live_commands_require_valid_budget_before_creating_runs(tmp_path, command, budget):
    argv = [command, "--output", str(tmp_path / command)]
    if budget is not None:
        argv += ["--budget-seconds", budget]
    with pytest.raises(ValueError, match="budget-seconds"):
        main(argv)
    assert not (tmp_path / command).exists()


def test_viewer_cli_starts_with_budget(monkeypatch):
    from dreamerv3_attention import viewer

    def serve(cfg, args):
        assert args.budget_seconds == 180
        return "viewer started"

    monkeypatch.setattr(viewer, "serve", serve)
    assert main(["viewer", "--budget-seconds", "180"]) == "viewer started"


def test_shipped_configs_and_matched_ablation():
    from dataclasses import asdict

    base = asdict(Config.load("dreamerv3_attention/configs/baseline.json"))
    attention = asdict(Config.load("dreamerv3_attention/configs/attention.json"))
    assert base.pop("attention") is False and attention.pop("attention") is True
    assert base == attention
    assert Config.load("dreamerv3_attention/configs/evaluation.json").split == "held_out"


def test_cli_smoke_runs_and_verifies_checkpoint(tmp_path):
    result = main(["smoke", "--config", "dreamerv3_attention/configs/smoke.json", "--output", str(tmp_path / "smoke")])
    assert result["frames"] == 16 and result["updates"] == 3 and result["checkpoint_verified"]
    assert json.loads((tmp_path / "smoke" / "result.json").read_text())["synthetic_only"]


def test_output_reuses_empty_failed_attempt_but_preserves_run_files(tmp_path):
    output = tmp_path / "run"
    output.mkdir()
    prepare_output(output)
    checkpoint = output / "latest.pt"
    checkpoint.write_bytes(b"existing checkpoint")
    with pytest.raises(FileExistsError, match="already contains data"):
        prepare_output(output)
    assert checkpoint.read_bytes() == b"existing checkpoint"


@pytest.mark.parametrize("command", ["train", "collect", "evaluate", "infer"])
@pytest.mark.parametrize("budget", ["0.5", "600", "86400"])
def test_live_commands_with_budget_reach_gpu_preflight(monkeypatch, tmp_path, command, budget):
    from contextlib import contextmanager
    from dreamerv3_attention import __main__ as cli

    @contextmanager
    def rejected(cfg, *, launch_server):
        raise RuntimeError("unrelated GPU job")
        yield

    monkeypatch.setattr(cli, "workload_lock", rejected)
    monkeypatch.setattr(cli, "seed_all", lambda seed: None)
    monkeypatch.setattr(cli.torch.cuda, "is_available", lambda: True)
    output = tmp_path / "rejected"
    config = "evaluation" if command == "evaluate" else "attention"
    with pytest.raises(RuntimeError, match="unrelated GPU job"):
        main([command, "--config", f"dreamerv3_attention/configs/{config}.json",
              "--checkpoint", "model.pt", "--budget-seconds", budget, "--output", str(output)])
    assert not output.exists()


@pytest.mark.parametrize("command", ["evaluate", "infer"])
def test_evaluation_cli_publishes_to_viewer_and_closes_feed(monkeypatch, tmp_path, command):
    from contextlib import nullcontext
    from types import SimpleNamespace as NS
    from dreamerv3_attention import __main__ as cli
    from dreamerv3_attention.live import TrainingMonitor

    cfg = Config(device="cpu", split="validation")
    path, output = tmp_path / "config.json", tmp_path / "evaluation"
    cfg.save(path)
    monkeypatch.setattr(cli, "workload_lock", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(cli, "server", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(cli, "register_run", lambda *args: None)
    monkeypatch.setattr(cli.Trainer, "restore", lambda *args, **kwargs: NS(cfg=cfg, agent=NS()))

    def evaluate(run_cfg, agent, factory, **kwargs):
        assert kwargs["observer"].mode == "evaluation"
        assert kwargs["episodes"] == (1 if command == "infer" else None)
        assert TrainingMonitor(output).read()["active"]
        return {"summary": {"episode_count": 0}}

    monkeypatch.setattr(cli, "evaluate", evaluate)
    main([command, "--config", str(path), "--checkpoint", "model.pt",
          "--budget-seconds", "10", "--output", str(output)])
    state = TrainingMonitor(output).read()
    assert state["status"] == "ended" and not state["active"]
    assert state["stop_reason"] == "completed"
    assert (output / "result.json").exists()


def test_controller_evaluation_uses_suite_without_checkpoint(monkeypatch, tmp_path):
    from contextlib import nullcontext
    from dreamerv3_attention import __main__ as cli

    cfg = Config(device="cuda", split="held_out", driving={"default_speed_mps": 6.0})
    path, output = tmp_path / "config.json", tmp_path / "controller"
    cfg.save(path)
    monkeypatch.setattr(cli, "workload_lock", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(cli, "server", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(cli, "register_run", lambda *args: None)
    monkeypatch.setattr(cli, "seed_all", lambda seed: None)

    def evaluate(run_cfg, agent, factory, **kwargs):
        assert run_cfg.device == "cpu"
        assert agent.__class__.__name__ == "RouteController"
        assert kwargs["fixed_scenario"] is False
        assert kwargs["action_source"] == "controller"
        assert agent.target_speed == expected_cap[0]
        assert agent.config.default_speed_mps == 6.0
        return {"episodes": [], "summary": {"episode_count": 0}}

    monkeypatch.setattr(cli, "evaluate", evaluate)
    expected_cap = [None]
    result = main(["evaluate", "--config", str(path), "--controller",
                   "--budget-seconds", "10", "--output", str(output)])
    assert result["policy"] == {"type": "route_controller", "speed_mode": "speed_limit"}
    assert result["checkpoint"] is None
    assert json.loads((output / "result.json").read_text())["policy"] == result["policy"]
    expected_cap[0] = 3.0
    capped_output = tmp_path / "controller-capped"
    result = main(["evaluate", "--config", str(path), "--controller", "--controller-speed", "3",
                   "--budget-seconds", "10", "--output", str(capped_output)])
    assert result["policy"] == {"type": "route_controller", "target_speed_mps": 3.0}


def test_controller_cli_rejects_checkpoint_and_audit_output_overwrite(tmp_path):
    with pytest.raises(ValueError, match="does not use --checkpoint"):
        main(["evaluate", "--controller", "--checkpoint", "x", "--budget-seconds", "10"])
    trace = tmp_path / "trace.jsonl"
    trace.write_text("{}\n")
    with pytest.raises(FileExistsError, match="new path"):
        main(["audit-controls", "--trace", str(trace), "--output", str(trace)])
    existing = tmp_path / "existing.json"
    existing.write_text("preserve")
    with pytest.raises(FileExistsError, match="new path"):
        main(["audit-controls", "--trace", str(trace), "--output", str(existing)])
    assert existing.read_text() == "preserve"


def test_audit_output_is_exclusive_and_report_is_complete(tmp_path, capsys):
    trace, output = tmp_path / "trace.jsonl", tmp_path / "audit.json"
    trace.write_text('{"scenario":"one","speed_mps":0,"target_speed_mps":2}\n')
    main(["audit-controls", "--trace", str(trace), "--output", str(output)])
    capsys.readouterr()
    original = output.read_bytes()
    report = json.loads(original)
    assert report["overall"]["unknown_observation_validity_steps"] == 1
    with pytest.raises(FileExistsError):
        main(["audit-controls", "--trace", str(trace), "--output", str(output)])
    assert output.read_bytes() == original
