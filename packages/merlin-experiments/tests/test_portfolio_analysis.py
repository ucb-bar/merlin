"""Installed analysis lifecycle with synthetic emissions, never a compiler or simulator."""

import importlib.util
import socket
import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin_experiments.phase2 import portfolio_analysis as PA
from merlin_experiments.phase2 import portfolio_checkpoint as PC

from merlin.benchharness import hash_tree
from merlin.common.digest import sha256_bytes


def digest(text):
    return sha256_bytes(text.encode())


@pytest.fixture(autouse=True)
def refuse_processes_and_listeners(monkeypatch):
    def refused(*args, **kwargs):
        pytest.fail("synthetic portfolio analysis must not launch processes or listeners")

    monkeypatch.setattr(subprocess, "Popen", refused)
    monkeypatch.setattr(socket.socket, "bind", refused)


@pytest.fixture
def case(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "portfolio_analysis_fixtures", Path(__file__).with_name("portfolio_analysis_fixtures.py")
    )
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    return helper.build_case(tmp_path, monkeypatch)


def analyze(case, version=None):
    if version is not None:
        (case.candidate / "version.txt").write_text(str(version))
    return case.owner.analyze(case.candidate, hypothesis="synthetic complete-model revision")


def test_fresh_analysis_session_seal_and_installed_consumer(case):
    row = analyze(case)
    assert row["iteration"] == 0
    assert row["readiness"]["status"] == "ready_for_probe_admission"
    assert row["portfolio"]["members_total"] == row["portfolio"]["members_ready"] == 2
    assert row["compiler_dependencies"] == case.inputs.compiler_dependencies(case.candidate)
    assert row["functional_gate"]["status"] == "not_run"
    assert row["fast_evaluation"]["status"] == "exact_only_fallback"
    for index, member in enumerate(case.inputs.portfolio_sentinels):
        context = case.session.current_portfolio_member_context(case.candidate, index=index)
        assert context["member_binding"]["capsule_sha256"] == member.capsule_sha256
        assert context["member_binding"]["lowered_sha256"] == digest(context["artifacts"]["lowered_text"])
    assert len(case.analyzer.calls) == 2
    assert all(not path.stat().st_mode & 0o222 for path in Path(row["submitted_snapshot"]).rglob("*"))
    sealed = case.session.seal(case.candidate)
    document = PC.consume_global_candidate(
        sealed,
        context=PC.CheckpointVerificationContext(case.inputs.host_policy, case.inputs.compiler_shared_source_root),
    )
    assert document["candidate_sha256"] == hash_tree(case.candidate)["sha256"]
    assert document["global_speedup_proven"] is False


def test_duplicate_and_revisit_preserve_chronology_without_reusing_measurements(case):
    first = analyze(case)
    original_bytes = (case.output / "iteration_0000.json").read_bytes()
    duplicate = analyze(case)
    assert duplicate["exact_analysis_reused"] is True
    assert duplicate["iteration"] == 0 and len(case.journal.iterations) == 1
    assert len(case.analyzer.calls) == 2
    case.journal.iterations[0]["probe_receipts"].append({"synthetic": "must-not-reuse"})
    changed = analyze(case, 1)
    assert changed["iteration"] == 1 and len(case.analyzer.calls) == 4
    revisited = analyze(case, 0)
    assert revisited["iteration"] == 2 and len(case.analyzer.calls) == 4
    assert revisited["probe_receipts"] == []
    assert revisited["analysis_reuse"]["source_iteration"] == first["iteration"]
    assert [row["iteration"] for row in case.journal.iterations] == [0, 1, 2]
    assert (case.output / "iteration_0000.json").read_bytes() == original_bytes


@pytest.mark.parametrize("mutation", ["baseline", "dependency", "controller", "capsule"])
def test_frozen_input_drift_refuses_before_analyzer(case, mutation):
    path = {
        "baseline": case.inputs.baseline / "version.txt",
        "dependency": case.inputs.compiler_shared_source_root / "helper.py",
        "controller": case.inputs.controller_source,
        "capsule": Path(case.inputs.sentinel.frozen_source_path) / "capsule.interface.mlir",
    }[mutation]
    path.write_text("CHANGED = True\n")
    with pytest.raises(ValueError):
        analyze(case)
    assert not case.analyzer.calls and not case.journal.iterations


@pytest.mark.parametrize("substitution", ["candidate", "member"])
def test_analyzer_cannot_substitute_content_identity(case, substitution):
    case.analyzer.substitute = substitution
    with pytest.raises(ValueError, match="bound to candidate bytes|substituted its objective"):
        analyze(case)
    assert not case.journal.iterations


def test_member_failure_is_blocked_evidence_not_ready_checkpoint(case):
    case.analyzer.fail_member = case.inputs.portfolio_sentinels[1].capsule
    row = analyze(case)
    assert row["readiness"]["status"] == "blocked"
    assert row["portfolio"]["members_ready"] == 1
    with pytest.raises(ValueError, match="not ready"):
        case.session.seal(case.candidate)
    path = case.session.checkpoint_authoring(case.candidate, name="blocked")
    assert (
        PC.consume_authoring_checkpoint(
            path,
            context=PC.CheckpointVerificationContext(case.inputs.host_policy, case.inputs.compiler_shared_source_root),
        )["readiness"]["status"]
        == "blocked"
    )


def test_member_workers_complete_out_of_order_but_publish_declared_order(case):
    case.owner.portfolio_analysis_workers = 2
    second_finished = threading.Event()
    completed = []

    def hook(candidate, sentinel, kwargs):
        if sentinel == case.inputs.sentinel:
            assert second_finished.wait(5), "second member did not run concurrently"
        else:
            completed.append(sentinel.capsule_sha256)
            second_finished.set()

    case.analyzer.hook = hook
    row = analyze(case)
    assert completed == [case.inputs.portfolio_sentinels[1].capsule_sha256]
    assert [member["identity"]["capsule_sha256"] for member in row["portfolio"]["members"]] == [
        member.capsule_sha256 for member in case.inputs.portfolio_sentinels
    ]
    assert all(0 < timeout <= case.owner.timeout_s for _, _, timeout in case.analyzer.calls)
    assert row["portfolio"]["members_ready"] == 2


def test_lock_wait_budget_refuses_without_analyzer(case):
    assert case.owner._analysis_lock.acquire(timeout=1)
    try:
        with pytest.raises(TimeoutError, match="waiting for the active analysis"):
            case.owner.analyze(case.candidate, hypothesis="bounded lock wait", timeout_s=0.01)
    finally:
        case.owner._analysis_lock.release()
    assert not case.analyzer.calls and not case.journal.iterations


def test_concurrent_duplicate_requests_publish_once(case):
    entered, release = threading.Event(), threading.Event()
    results, errors = [], []

    def hook(candidate, sentinel, kwargs):
        if sentinel == case.inputs.sentinel:
            entered.set()
            assert release.wait(5), "test did not release first analysis"

    case.analyzer.hook = hook

    def request():
        try:
            results.append(analyze(case))
        except BaseException as exc:
            errors.append(exc)

    first, second = [threading.Thread(target=request, daemon=True) for _ in range(2)]
    first.start()
    try:
        assert entered.wait(5)
        second.start()
    finally:
        release.set()
        first.join(5)
        if second.ident is not None:
            second.join(5)
    assert not first.is_alive() and not second.is_alive()
    assert not errors
    assert len(results) == 2 and len(case.journal.iterations) == 1
    assert len(case.analyzer.calls) == 2
    assert sum(row.get("exact_analysis_reused", False) for row in results) == 1


def test_frozen_input_changed_inside_analyzer_refused_before_publication(case):
    def hook(candidate, sentinel, kwargs):
        (case.inputs.baseline / "version.txt").write_text("changed after input admission")

    case.analyzer.hook = hook
    with pytest.raises(ValueError, match="frozen compiler changed"):
        analyze(case)
    assert not case.journal.iterations
    assert not (case.output / "iteration_0000.json").exists()


def test_live_authoring_during_analysis_does_not_change_captured_revision(case):
    before = hash_tree(case.candidate)["sha256"]

    def hook(candidate, sentinel, kwargs):
        assert candidate != case.candidate
        (case.candidate / "version.txt").write_text("99")

    case.analyzer.hook = hook
    row = analyze(case)
    assert row["candidate_sha256"] == before
    assert hash_tree(Path(row["submitted_snapshot"]))["sha256"] == before
    with pytest.raises(ValueError, match="candidate changed"):
        case.session.current(case.candidate)


def test_deadline_after_lock_wait_refuses_and_releases_lock(case, monkeypatch):
    ticks = iter([0.0, 2.0])
    monkeypatch.setattr(PA.time, "monotonic", lambda: next(ticks))
    with pytest.raises(TimeoutError, match="no budget after waiting"):
        case.owner.analyze(case.candidate, hypothesis="expired", timeout_s=1)
    assert not case.owner._analysis_lock.locked()
    assert not case.analyzer.calls


def test_functional_gate_failure_excluded_from_best_despite_lower_host_cost(case):
    case.owner.functional_gate = SimpleNamespace(
        model_payload_dir=case.root,
        toolchain=None,
        gate_spec=None,
        timeout_seconds=1,
        keep_elf=False,
        source_sha256=digest("synthetic gate config"),
        source_path=None,
    )
    calls = []

    def gate(lowered, buffer, **kwargs):
        calls.append(lowered)
        failed = "revision=1 " in lowered
        return {
            "status": "failed" if failed else "passed",
            "reason": "synthetic verdict",
            "simulation_executed": True,
            "excludes_candidate": failed,
        }

    case.owner.functional_gate_runner = gate
    passing = analyze(case)
    failing = analyze(case, 1)
    assert passing["readiness"]["status"] == "ready_for_probe_admission"
    assert failing["readiness"]["status"] == "blocked"
    assert "functional_gate_failed" in failing["readiness"]["blockers"]
    assert failing["static_comparison"]["functional_gate"]["excludes_candidate"] is True
    best = case.owner.best_authored_candidate()
    assert best["iteration"] == 0
    assert best["excluded_functional_gate_failures"][0]["iteration"] == 1
    assert len(calls) == 2
    with pytest.raises(ValueError, match="not ready"):
        case.session.seal(case.candidate)


def test_functional_gate_exception_is_visible_not_run_never_passed(case):
    case.owner.functional_gate = SimpleNamespace(
        model_payload_dir=case.root,
        toolchain=None,
        gate_spec=None,
        timeout_seconds=1,
        keep_elf=False,
        source_sha256=digest("synthetic config"),
        source_path=None,
    )

    def broken(*args, **kwargs):
        raise RuntimeError("synthetic gate unavailable")

    case.owner.functional_gate_runner = broken
    row = analyze(case)
    assert row["functional_gate"]["status"] == "not_run"
    assert row["functional_gate"]["simulation_executed"] is False
    assert "RuntimeError" in row["functional_gate"]["reason"]
    # Preserve historical scientific semantics: not_run is not proof, but is not failed.
    assert row["readiness"]["status"] == "ready_for_probe_admission"


def test_best_selection_skips_drifted_snapshot(case):
    analyze(case)
    second = analyze(case, 1)
    assert case.owner.best_authored_candidate()["iteration"] == 1
    snapshot = Path(second["submitted_snapshot"]) / "version.txt"
    snapshot.chmod(0o644)
    snapshot.write_text("2")
    assert case.owner.best_authored_candidate()["iteration"] == 0


@pytest.mark.parametrize(
    "options", [{"timeout_s": 0}, {"portfolio_analysis_workers": 0}, {"minimum_memory_available_bytes": -1}]
)
def test_invalid_execution_budgets_refused(case, options):
    with pytest.raises(ValueError):
        PA.PortfolioAnalysis(case.session, analyzer=case.analyzer, **options)
    assert not case.analyzer.calls


def test_baseline_cache_binding_scope_and_seed_requires_root(case):
    with pytest.raises(ValueError, match="inside a compiler tree"):
        PA.PortfolioAnalysis(
            case.session, analyzer=case.analyzer, baseline_emission_cache=case.inputs.baseline / "cache"
        )
    with pytest.raises(ValueError, match="require a cache root"):
        PA.PortfolioAnalysis(case.session, analyzer=case.analyzer, baseline_emission_seed_runs=[case.root / "seed"])
    cache = case.root / "baseline-cache"
    owner = PA.PortfolioAnalysis(case.session, analyzer=case.analyzer, baseline_emission_cache=cache)
    assert owner.baseline_emission_cache_binding["root"] == str(cache.resolve())
    assert owner.baseline_emission_cache_binding["compiler_dependencies_sha256"]
    assert owner.baseline_emission_cache_seeds == []
    assert not case.analyzer.calls
