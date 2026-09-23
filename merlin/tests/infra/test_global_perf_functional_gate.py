"""A candidate whose executed program fails the model's gate is never the round's best.

The phase-2 loop scored candidates by static counters only and never executed one. With a gate
configured, every iteration builds and runs the objective's emitted program; a ``failed`` verdict
must exclude that revision from ``best_authored_candidate`` (and so from sealing) with the reason
recorded, while ``not_run`` stays visible and never excludes. These tests inject a stub gate runner
so no toolchain is needed.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest
from merlin_experiments.phase2 import contracts as P2_CONTRACTS
from merlin_experiments.phase2 import emission_analysis as EA
from merlin_experiments.phase2 import stage_inputs as INPUTS
from merlin_experiments.phase2.portfolio_analysis import PortfolioAnalysis

from merlin.benchharness import hash_tree
from merlin.common.digest import sha256_bytes
from merlin.common.paths import repo_root
from merlin.perf import functional_gate as FG

SCRIPTS = repo_root() / "merlin/experiments/gemmini_perf_bench/scripts"
sys.path.insert(0, str(SCRIPTS))
G = importlib.import_module("run_global_perf_experiment")
SHA = {name: sha256_bytes(name.encode()) for name in ("target", "graph", "plan")}

# host-lane cost per candidate revision (source.txt content) -> the loop's "best" ordering
COSTS = {"candidate": 1000, "candidate-v2": 500, "candidate-v3": 900}


def _analyzer(calls):
    """Publishes retained artifacts (lowered_text + command_buffer) like the production analyzer."""

    def analyzer(base, current, objective, **kwargs):
        candidate_sha = hash_tree(current)["sha256"]
        revision = Path(current, "source.txt").read_text()
        calls.append(revision)
        lowered = f"llvm.func @main() {{}} // {revision}\n"
        command_buffer = {"commands": [], "tensors": {}, "params": {"revision": revision}}
        command_text = json.dumps(command_buffer, sort_keys=True, separators=(",", ":")) + "\n"
        lowered_sha = sha256_bytes(lowered.encode())
        command_sha = sha256_bytes(command_text.encode())
        source = Path(objective.frozen_source_path) / "capsule.interface.mlir"
        plan = {
            "status": "verified",
            "plan_digest": P2_CONTRACTS.document_sha256({"r": revision}),
            "candidate_sha256": candidate_sha,
            "logical_dispatch_digest": SHA["graph"],
            "source_sha256": P2_CONTRACTS.sha256_file(source),
            "candidate_lowered_sha256": lowered_sha,
            "candidate_command_buffer_sha256": command_sha,
            "emitted_dispatches": 2,
            "host_activity": {
                "status": "derived",
                "load_payload_bytes": COSTS[revision],
                "store_payload_bytes": 0,
                "static_allocation_payload_bytes": 0,
                "static_operations": {"allocation": 1},
                "dynamic_operations": {"llvm.load": COSTS[revision]},
            },
        }
        analysis = {
            "candidate_sha256": candidate_sha,
            "workload": {"capsule_sha256": objective.capsule_sha256},
            "emission": {"candidate_lowered_sha256": lowered_sha, "candidate_command_buffer_sha256": command_sha},
            "diagnostics": {
                "captured_logical_graph": {"status": "verified", "logical_dispatch_digest": SHA["graph"]},
                "verified_global_plan_emission": plan,
                "arms": {
                    "candidate": {
                        "status": "emitted",
                        "macs": 8192,
                        "exact": True,
                        "movement": {"known_bytes": 128, "exact_bytes": True},
                    }
                },
            },
        }
        kwargs["artifact_sink"](
            {
                "lowered_text": lowered,
                "decoded_trace": {"instructions": []},
                "command_buffer": command_buffer,
                "command_buffer_text": command_text,
                "interface": str(source),
                "candidate_sha256": candidate_sha,
                "candidate_lowered_sha256": lowered_sha,
                "candidate_command_buffer_sha256": command_sha,
                "task_instruction_evidence": {"status": "verified", "tasks": []},
            }
        )
        return analysis

    return analyzer


class _StubRunner:
    """Answers each gate invocation from a queue and records exactly what it was handed."""

    def __init__(self, verdicts):
        self.verdicts = list(verdicts)
        self.invocations = []

    def __call__(
        self,
        artifact_text,
        command_buffer,
        *,
        model_payload_dir,
        toolchain,
        gate_spec,
        workdir,
        timeout,
        keep_elf=False,
    ):
        self.invocations.append(
            {
                "artifact_text": artifact_text,
                "command_buffer": command_buffer,
                "workdir": Path(workdir),
                "timeout": timeout,
                "payload": Path(model_payload_dir),
            }
        )
        verdict = self.verdicts.pop(0)
        if isinstance(verdict, Exception):
            raise verdict
        status, reason = verdict
        return FG.FunctionalGateResult(
            status=status,
            reason=reason,
            stage="evaluate" if status != "not_run" else "simulate",
            fields={"bad": "1000", "top1": "556"} if status == "failed" else {"bad": "0", "top1": "258"},
            expected=gate_spec.to_dict(),
            simulation_executed=status != "not_run",
            artifact_sha256=sha256_bytes(artifact_text.encode()),
            workdir=str(workdir),
        )


def _config(tmp_path):
    return FG.FunctionalGateConfig(
        model_payload_dir=tmp_path / "payload_bundle",
        toolchain=FG.FunctionalGateToolchain(
            mlir_translate=tmp_path / "mlir-translate",
            clang=tmp_path / "clang",
            simulator=tmp_path / "sim",
            clang_target="riscv64-unknown-elf",
            march="rv64gc",
            mabi="lp64d",
            simulator_isa="rv64gc_zicntr",
        ),
        gate_spec=FG.FunctionalGateSpec({"bad": 0, "top1": 258}),
        timeout_seconds=90,
        source_path=tmp_path / "gate.json",
        source_sha256="a" * 64,
    )


def _experiment(tmp_path, monkeypatch, **options):
    """Only the production analyzer seam receives ``artifact_sink``, so the fixture sits on it."""
    baseline, candidate, source = (tmp_path / name for name in ("base", "candidate", "model"))
    for path, text in ((baseline, "compiler"), (candidate, "candidate")):
        path.mkdir()
        (path / "source.txt").write_text(text)
    source.mkdir()
    (source / "capsule.yaml").write_text("interface_mlir: capsule.interface.mlir\n")
    (source / "capsule.interface.mlir").write_text("module {}\n")
    sentinel = INPUTS.StageE2ESentinel(
        "real-model", str(source), str(source), P2_CONTRACTS.exact_tree_record(source)["sha256"], ("lane",), ("L2",)
    )
    calls = []
    monkeypatch.setattr(EA, "analyze_whole_model_emission", _analyzer(calls))
    experiment = G.GlobalPerfExperiment(
        baseline=baseline,
        baseline_sha256=hash_tree(baseline)["sha256"],
        sentinel=sentinel,
        target="test-target",
        target_sha256=SHA["target"],
        output=tmp_path / "run",
        analyzer=EA.analyze_whole_model_emission,
        **options,
    )
    return experiment, candidate, calls


def _revise(experiment, candidate, revision, hypothesis):
    (candidate / "source.txt").write_text(revision)
    return experiment.analysis.analyze(candidate, hypothesis=hypothesis)


def test_without_a_gate_every_iteration_says_so_and_nothing_is_excluded(tmp_path, monkeypatch):
    experiment, candidate, _ = _experiment(tmp_path, monkeypatch)
    record = experiment.analysis.analyze(candidate, hypothesis="seed")
    gate = record["functional_gate"]
    assert gate["status"] == "not_run" and gate["configured"] is False
    assert gate["excludes_candidate"] is False and gate["simulation_executed"] is False
    assert "NOT executed" in gate["reason"]
    feedback = record["static_comparison"]["functional_gate"]
    assert feedback["status"] == "not_run" and "NOT executed" in feedback["reading"]
    assert record["readiness"]["status"] == "ready_for_probe_admission"
    best = experiment.analysis.best_authored_candidate()
    assert best["iteration"] == 0 and best["functional_gate"] == "not_run"
    assert best["excluded_functional_gate_failures"] == []


def test_failed_gate_excludes_the_cheapest_revision_and_records_why(tmp_path, monkeypatch):
    runner = _StubRunner([("passed", "ok"), ("failed", "gated field mismatch: bad=1000 (1000 != 0)"), ("passed", "ok")])
    experiment, candidate, _ = _experiment(
        tmp_path, monkeypatch, functional_gate=_config(tmp_path), functional_gate_runner=runner
    )
    seed = experiment.analysis.analyze(candidate, hypothesis="seed")  # cost 1000, passed
    win = _revise(experiment, candidate, "candidate-v2", "cheapest -- but it miscompiles")  # 500, FAILED
    okay = _revise(experiment, candidate, "candidate-v3", "a smaller, valid win")  # 900, passed

    # the runner was handed the objective's retained emission, per iteration, under the run dir
    assert [row["artifact_text"] for row in runner.invocations] == [
        "llvm.func @main() {} // candidate\n",
        "llvm.func @main() {} // candidate-v2\n",
        "llvm.func @main() {} // candidate-v3\n",
    ]
    assert runner.invocations[1]["command_buffer"]["params"] == {"revision": "candidate-v2"}
    assert runner.invocations[1]["workdir"] == tmp_path / "run" / "functional_gate_0001"
    assert runner.invocations[1]["timeout"] == 90
    assert runner.invocations[1]["payload"] == tmp_path / "payload_bundle"

    # the failed revision is marked in every place a reader looks
    gate = win["functional_gate"]
    assert gate["status"] == "failed" and gate["excludes_candidate"] is True
    assert gate["reason"] == "gated field mismatch: bad=1000 (1000 != 0)"
    assert gate["iteration"] == 1 and gate["candidate_sha256"] == win["candidate_sha256"]
    assert gate["candidate_lowered_sha256"] == win["analysis"]["emission"]["candidate_lowered_sha256"]
    assert gate["config_sha256"] == "a" * 64 and gate["configured"] is True
    assert win["readiness"]["status"] == "blocked"
    assert "functional_gate_failed" in win["readiness"]["blockers"]
    assert win["readiness"]["functional_gate"]["reason"] == gate["reason"]
    feedback = win["static_comparison"]["functional_gate"]
    assert feedback["excludes_candidate"] is True and "excluded from selection" in feedback["reading"]
    assert feedback["fields"] == {"bad": "1000", "top1": "556"}
    # the static counters still say it was the biggest win -- which is exactly the trap
    assert PortfolioAnalysis.authored_host_cost(win["analysis"]) == (500, 1)
    # the passing revisions are untouched
    assert seed["readiness"]["status"] == okay["readiness"]["status"] == "ready_for_probe_admission"
    assert seed["functional_gate"]["status"] == okay["functional_gate"]["status"] == "passed"
    assert "reused_from_iteration" not in okay["functional_gate"]

    # selection: the valid 900 wins over the invalid 500, and the exclusion is itemized
    best = experiment.analysis.best_authored_candidate()
    assert best["iteration"] == 2 and best["host_payload_bytes"] == 900
    assert best["functional_gate"] == "passed"
    assert best["excluded_functional_gate_failures"] == [
        {
            "iteration": 1,
            "candidate_sha256": win["candidate_sha256"],
            "reason": "gated field mismatch: bad=1000 (1000 != 0)",
        }
    ]
    # ... and the failed revision cannot be sealed even when asked for by content
    (candidate / "source.txt").write_text("candidate-v2")
    with pytest.raises(ValueError):
        experiment.revision_session.seal(candidate, name="should_not_seal")

    # the verdict is in the persisted iteration record, not only in memory
    persisted = json.loads((tmp_path / "run" / "iteration_0001.json").read_text())
    assert persisted["functional_gate"]["status"] == "failed"
    assert persisted["static_comparison"]["functional_gate"]["excludes_candidate"] is True
    assert "functional_gate_failed" in persisted["readiness"]["blockers"]


def test_exclusion_holds_even_if_readiness_were_left_ready(tmp_path, monkeypatch):
    """The ranking checks the gate itself, not only the blocker derived from it."""
    experiment, candidate, _ = _experiment(tmp_path, monkeypatch)
    experiment.analysis.analyze(candidate, hypothesis="seed")
    _revise(experiment, candidate, "candidate-v2", "cheapest")
    row = experiment.revisions.iterations[1]
    row["functional_gate"] = {"status": "failed", "reason": "bad=1000"}
    row["readiness"]["status"] = "ready_for_probe_admission"
    best = experiment.analysis.best_authored_candidate()
    assert best["iteration"] == 0
    assert best["excluded_functional_gate_failures"][0]["iteration"] == 1


def test_not_run_and_a_broken_runner_stay_visible_but_never_exclude(tmp_path, monkeypatch):
    runner = _StubRunner([("not_run", "simulator exceeded the 90s budget"), RuntimeError("toolchain exploded")])
    experiment, candidate, _ = _experiment(
        tmp_path, monkeypatch, functional_gate=_config(tmp_path), functional_gate_runner=runner
    )
    timed_out = experiment.analysis.analyze(candidate, hypothesis="seed")
    broken = _revise(experiment, candidate, "candidate-v2", "cheapest")
    assert timed_out["functional_gate"]["status"] == "not_run"
    assert timed_out["functional_gate"]["excludes_candidate"] is False
    assert timed_out["readiness"]["status"] == "ready_for_probe_admission"
    assert broken["functional_gate"]["status"] == "not_run"
    assert "runner raised RuntimeError: toolchain exploded" in broken["functional_gate"]["reason"]
    assert broken["readiness"]["status"] == "ready_for_probe_admission"
    for record in (timed_out, broken):
        assert record["static_comparison"]["functional_gate"]["simulation_executed"] is False
        assert "NOT executed" in record["static_comparison"]["functional_gate"]["reading"]
    best = experiment.analysis.best_authored_candidate()
    assert best["iteration"] == 1 and best["host_payload_bytes"] == 500  # eligible, and visible
    assert best["functional_gate"] == "not_run"


def test_exact_reuse_inherits_the_source_verdict_without_re_running(tmp_path, monkeypatch):
    runner = _StubRunner([("passed", "ok"), ("failed", "bad=1000"), ("passed", "ok"), ("failed", "bad=1000 again")])
    experiment, candidate, _ = _experiment(
        tmp_path, monkeypatch, functional_gate=_config(tmp_path), functional_gate_runner=runner
    )
    experiment.analysis.analyze(candidate, hypothesis="seed")
    _revise(experiment, candidate, "candidate-v2", "fails")
    _revise(experiment, candidate, "candidate-v3", "passes")
    experiment.analysis.analyze(candidate, hypothesis="v3 again")  # same bytes: in-place duplicate
    # Revisiting the PASSING bytes after another revision is an exact reuse: the same emission was
    # already executed, so the verdict is inherited rather than re-measured.
    _revise(experiment, candidate, "candidate", "back to the seed bytes")
    revisit = _revise(experiment, candidate, "candidate-v3", "revisit the passing bytes")
    assert revisit["exact_analysis_reused"] is True
    passing_runs = [row for row in runner.invocations if "candidate-v3" in row["artifact_text"]]
    assert len(passing_runs) == 1  # v3 was executed exactly once
    assert revisit["functional_gate"]["status"] == "passed"
    assert revisit["functional_gate"]["reused_from_iteration"] == 2
    assert revisit["functional_gate"]["iteration"] == revisit["iteration"]
    assert revisit["readiness"]["status"] == "ready_for_probe_admission"
    # Revisiting the FAILED bytes is not a reuse (that row is blocked): it is re-analyzed and
    # re-executed, and fails again.
    again = _revise(experiment, candidate, "candidate-v2", "revisit the failing bytes")
    assert "exact_analysis_reused" not in again
    assert again["functional_gate"]["status"] == "failed"
    assert again["functional_gate"]["reason"] == "bad=1000 again"
    assert again["readiness"]["status"] == "blocked"
    best = experiment.analysis.best_authored_candidate()
    assert best["host_payload_bytes"] == 900
    assert [row["iteration"] for row in best["excluded_functional_gate_failures"]] == [1, again["iteration"]]
