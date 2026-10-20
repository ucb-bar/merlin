"""Actual installed continuation with explicit synthetic already-admitted inputs.

Only admission verification and provider execution are substituted. This does not
qualify admission, source proof, brokers, OS isolation or hardware. Real background
cadence, no-submission grading, quota/restore, checkpoint, audit and completion
decisions run. The provider is a local stdout child, never a network client.
This file can be copied beside independently installed wheels outside a checkout.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml


@pytest.mark.parametrize("sandbox", ["bwrap", "none"])
def test_real_options_authoring_startup_uses_frozen_facts_for_grading(tmp_path, monkeypatch, sandbox):
    """Exercise the admitted-run to grading ABI without starting an author or tools."""
    from merlin_experiments.phase1 import authoring, treatments
    from merlin_experiments.phase1.context import InvocationContext
    from merlin_experiments.phase1.options import RunOptions, parse_options
    from merlin_experiments.phase1.session import PreparedRun, RunRequest, WorkspaceTransport

    from merlin.targetgen.sandbox import bwrap

    descriptor = tmp_path / "target_experiment.yaml"
    descriptor.write_text(yaml.safe_dump({"target": "fixture", "capsule_corpus": str(tmp_path / "public")}))
    context = InvocationContext(
        tmp_path, descriptor, tmp_path, "fixture", tmp_path / "runs", tmp_path / "reports", tmp_path / "bundle", ()
    )
    options = parse_options(["--run-id", "startup", "--driver", "codex", "--sandbox", sandbox])
    assert type(options) is RunOptions
    frozen = tmp_path / "snapshot/facts.json"
    monkeypatch.setenv("MERLIN_RTL_FACTS", str(tmp_path / "other-ambient-facts.json"))
    resolved = []

    def frozen_facts(workspace, bundle, *, repo):
        resolved.append((workspace, bundle, repo))
        return frozen

    monkeypatch.setattr(bwrap, "frozen_selected_rtl_facts", frozen_facts)

    def unused(*_args, **_kwargs):
        raise AssertionError("already-admitted startup must not run an admission/provider adapter")

    request = RunRequest(
        context, options, treatments.Treatment(), tmp_path / "bundle/input_bundle_manifest.yaml", (),
        Path(__file__), False, {}, lambda: (),
    )
    prepared = PreparedRun(
        request, tmp_path / "run", tmp_path / "workspace", tmp_path / "bundle", {}, {},
        tmp_path / "public", tmp_path / "policy", None, None, False, None,
        WorkspaceTransport(unused, unused), unused,
    )

    class Configured(Exception):
        pass

    def observe_grading_configuration(callback, stage, on_duration):
        assert stage == "qa"
        assert callback.func is authoring.LG.grade
        inputs = callback.keywords["inputs"]
        assert type(inputs) is authoring.LG.GradingInputs
        assert inputs.rtl_facts == (frozen if sandbox == "bwrap" else None)
        assert inputs.context is context
        assert callback.keywords["qa_runner"] is request.treatment.qa_runner
        raise Configured

    monkeypatch.setattr(authoring.T, "timed", observe_grading_configuration)
    with pytest.raises(Configured):
        authoring.execute(prepared, authoring.AuthoringRuntime("synthetic", tmp_path / "timing.json"))
    assert resolved == ([(prepared.workspace, prepared.bundle, context.repo)] if sandbox == "bwrap" else [])


def _exercise(root: Path, resume: bool) -> int:
    import importlib.abc
    import socket
    from unittest.mock import patch

    class NoNative(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname in {"_common", "run_baseline_qa_loop", "run_agent_experiment", "chia", "ray"}:
                raise AssertionError(f"native or scheduler import: {fullname}")

    sys.meta_path.insert(0, NoNative())
    from merlin_experiments.phase1 import authoring, treatments
    from merlin_experiments.phase1.context import InvocationContext
    from merlin_experiments.phase1.options import parse_options
    from merlin_experiments.phase1.providers import execution
    from merlin_experiments.phase1.session import PreparedRun, RunRequest, WorkspaceTransport

    from merlin.common import arrival_stamp

    def no_network(*args, **kwargs):
        raise AssertionError("continuation fixture attempted network activity")

    def unused(*args, **kwargs):
        raise AssertionError("already-admitted continuation called an admission adapter")

    context = InvocationContext(
        root,
        root / "target_experiment.yaml",
        root,
        "fixture",
        root / "runs",
        root / "reports",
        root / "bundle",
        (),
    )
    options = parse_options(
        [
            "--run-id",
            "synthetic",
            "--arm",
            "raw_baseline",
            "--model",
            "fixture",
            "--driver",
            "codex",
            "--sandbox",
            "none",
            "--allow-unsandboxed",
            "--schedule",
            "rounds",
            "--max-rounds",
            "1",
            *(["--resume"] if resume else []),
        ]
    )
    request = RunRequest(
        context,
        options,
        treatments.Treatment(),
        root / "bundle/input_bundle_manifest.yaml",
        (),
        Path(__file__),
        False,
        {},
        lambda: (),
    )
    prepared = PreparedRun(
        request,
        root / "run",
        root / "workspace",
        root / "bundle",
        {"bundle_id": "synthetic"},
        {},
        None,
        None,
        None,
        None,
        resume,
        None,
        WorkspaceTransport(unused, unused),
        unused,
    )
    checks = []

    def verify(self):
        assert self is prepared
        checks.append("admitted-input seam observed")

    def launch(ws, run, model, effort, sandbox, bundle, rnd, timeout, **kwargs):
        assert ws == prepared.workspace and rnd == 0
        transcript = run / "rounds" / f"round_{rnd:02d}.transcript.jsonl"
        transcript.parent.mkdir(exist_ok=True)
        if not resume:
            events = [
                {
                    "type": "rate_limit_event",
                    "rate_limit_info": {
                        "status": "rejected",
                        "rateLimitType": "seven_day",
                    },
                }
            ]
        else:
            events = [
                {
                    "type": "assistant",
                    "message": {
                        "model": "fixture",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "notes",
                                "name": "Read",
                                "input": {"file_path": "docs/notes.md"},
                            },
                        ],
                    },
                },
                {
                    "type": "user",
                    "message": {
                        "content": [
                            {"type": "tool_result", "tool_use_id": "notes", "content": "No compiler produced."},
                        ]
                    },
                },
                {
                    "type": "result",
                    "num_turns": 1,
                    "duration_ms": 1,
                    "duration_api_ms": 1,
                    "usage": {"input_tokens": 2, "output_tokens": 2},
                },
            ]
        program = "import json\n" + "\n".join(f"print({json.dumps(json.dumps(event))}, flush=True)" for event in events)
        rc = arrival_stamp.stream_stamped(
            [sys.executable, "-c", program],
            cwd=ws,
            transcript=transcript,
            stderr_path=transcript.with_suffix(".stderr.log"),
            timeout=10,
            raw_path=transcript.with_suffix(".raw.jsonl"),
        )
        return rc, transcript

    with (
        patch.object(socket, "socket", no_network),
        patch.object(PreparedRun, "verify_inputs", verify),
        patch.object(execution, "launch", launch),
    ):
        result = authoring.execute(prepared, authoring.AuthoringRuntime("synthetic", root / "timing.json"))
    assert checks, "the engine skipped admitted-input verification"
    (root / ("resumed_checks.json" if resume else "fresh_checks.json")).write_text(json.dumps(checks))
    return result


def test_quota_checkpoint_then_fresh_process_restore_and_honest_incompletion(tmp_path):
    root = tmp_path / "operator"
    for relative in ("run", "workspace", "bundle"):
        (root / relative).mkdir(parents=True)
    (root / "target_experiment.yaml").write_text(
        yaml.safe_dump(
            {
                "target": "fixture",
                "capsule_corpus": str(root / "absent-public"),
            }
        )
    )
    (root / "bundle/input_bundle_manifest.yaml").write_text("bundle_id: synthetic\n")
    env = {**os.environ, "MERLIN_CAPSULE_L3_CHECKPOINT": "0", "MERLIN_AET_SINK": "0"}
    first = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), str(root)],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert first.returncode == 42, first.stdout + first.stderr
    checkpoint = root / "run/qa_loop_state.yaml"
    before = yaml.safe_load(checkpoint.read_text())
    assert before["next_round"] == 0 and not before["converged"]
    second = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), str(root), "resume"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert second.returncode == 1, second.stdout + second.stderr
    after = yaml.safe_load(checkpoint.read_text())
    assert after["next_round"] == 1, second.stdout + second.stderr
    assert after["cumulative"]["started_at"] == before["cumulative"]["started_at"]
    summary = yaml.safe_load((root / "run/qa_loop_summary.yaml").read_text())
    assert summary["n_rounds"] == 1 and summary["formal_complete"] is False
    assert summary["numeric_all_pass"] is False
    assert summary["capsule_timing"]["timing_ready"] is False
    assert json.loads((root / "run/capsule_timing_archive_status.json").read_text())["timing_ready"] is False
    assert "restored" in second.stdout.lower()
    assert json.loads((root / "fresh_checks.json").read_text())
    assert json.loads((root / "resumed_checks.json").read_text())


if __name__ == "__main__":
    raise SystemExit(_exercise(Path(sys.argv[1]), len(sys.argv) > 2))
