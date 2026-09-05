"""The campaign's two independent phases may overlap without changing the record they produce.

Every launch here is a fake in-process callable; no agent, simulator or subprocess is started.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from merlin.common.paths import merlin_dir

_SCRIPTS = merlin_dir() / "experiments/gemmini_perf_bench/scripts"
sys.path.insert(0, str(_SCRIPTS))
_SOURCE = _SCRIPTS / "run_agentic_perf_experiment.py"
_SPEC = importlib.util.spec_from_file_location("run_agentic_perf_concurrency_under_test", _SOURCE)
assert _SPEC is not None and _SPEC.loader is not None
ORCH = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = ORCH
_SPEC.loader.exec_module(ORCH)

BARRIER_TIMEOUT = 5.0


# --------------------------------------------------------------------------------------------
# The declared width
# --------------------------------------------------------------------------------------------


def test_absent_fanout_declaration_means_the_serial_campaign() -> None:
    assert ORCH.declared_fanout({}) == 1
    assert ORCH.declared_fanout({ORCH.FANOUT_ENVIRONMENT_VARIABLE: ""}) == 1
    assert ORCH.declared_fanout({ORCH.FANOUT_ENVIRONMENT_VARIABLE: "   "}) == 1


def test_declared_fanout_is_read_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    assert ORCH.declared_fanout({ORCH.FANOUT_ENVIRONMENT_VARIABLE: "3"}) == 3
    assert ORCH.declared_fanout({ORCH.FANOUT_ENVIRONMENT_VARIABLE: " 6 "}) == 6
    monkeypatch.setenv(ORCH.FANOUT_ENVIRONMENT_VARIABLE, "4")
    assert ORCH.declared_fanout() == 4
    monkeypatch.delenv(ORCH.FANOUT_ENVIRONMENT_VARIABLE)
    assert ORCH.declared_fanout() == 1


@pytest.mark.parametrize("value", ["0", "-2", "two", "2.5", "1e3", "3 trials", "0x3", "١٢"])
def test_an_unparseable_fanout_is_refused_rather_than_guessed(value: str) -> None:
    with pytest.raises(ORCH.ExperimentError, match="must be a positive integer"):
        ORCH.declared_fanout({ORCH.FANOUT_ENVIRONMENT_VARIABLE: value})


# --------------------------------------------------------------------------------------------
# The fan-out primitive
# --------------------------------------------------------------------------------------------


def test_commits_run_on_the_calling_thread_in_declared_order_while_launches_overlap() -> None:
    """The positive control: three launches must be in flight at once, or the barrier breaks."""
    barrier = threading.Barrier(3, timeout=BARRIER_TIMEOUT)
    launch_threads: list[str] = []
    commit_threads: list[str] = []
    order: list[str] = []
    guard = threading.Lock()

    def launch(name: str) -> None:
        barrier.wait()
        with guard:
            launch_threads.append(threading.current_thread().name)

    def commit(name: str) -> str:
        commit_threads.append(threading.current_thread().name)
        order.append(name)
        return name

    stages = [ORCH.ChildStage(name, lambda name=name: launch(name), lambda name=name: commit(name))
              for name in ("c", "a", "b")]
    assert ORCH.run_child_stages(stages, workers=3) == ["c", "a", "b"]

    assert order == ["c", "a", "b"], "commits must follow the declared order, not completion order"
    assert len(set(launch_threads)) == 3, "the launches did not fan out onto separate threads"
    assert commit_threads == [threading.current_thread().name] * 3


def test_a_single_declared_worker_keeps_every_launch_on_the_calling_thread() -> None:
    seen: list[str] = []
    stages = [ORCH.ChildStage(
        name, lambda name=name: seen.append(f"launch:{name}:{threading.current_thread().name}"),
        lambda name=name: seen.append(f"commit:{name}")) for name in ("a", "b")]
    ORCH.run_child_stages(stages, workers=1)

    main = threading.current_thread().name
    assert seen == [f"launch:a:{main}", f"launch:b:{main}", "commit:a", "commit:b"]


def test_an_adopted_stage_is_committed_without_being_launched() -> None:
    launched: list[str] = []
    stages = [ORCH.ChildStage("adopted", None, lambda: "adopted"),
              ORCH.ChildStage("fresh", lambda: launched.append("fresh"), lambda: "fresh")]
    assert ORCH.run_child_stages(stages, workers=2) == ["adopted", "fresh"]
    assert launched == ["fresh"]


def test_a_concurrent_failure_names_every_failure_and_commits_nothing() -> None:
    committed: list[str] = []

    def boom(name: str) -> None:
        raise ORCH.ExperimentError(f"command failed (2): {name}")

    stages = [
        ORCH.ChildStage("candidate:trial_00", lambda: boom("trial_00"),
                        lambda: committed.append("trial_00")),
        ORCH.ChildStage("candidate:trial_01", lambda: None, lambda: committed.append("trial_01")),
        ORCH.ChildStage("candidate:trial_02", lambda: boom("trial_02"),
                        lambda: committed.append("trial_02")),
    ]
    with pytest.raises(ORCH.ExperimentError) as raised:
        ORCH.run_child_stages(stages, workers=3)

    message = str(raised.value)
    assert "candidate:trial_00" in message and "candidate:trial_02" in message
    assert "2 of 3" in message
    assert committed == [], "a failed phase must not write a partially ordered record"


def test_serial_mode_still_stops_at_the_first_failure() -> None:
    launched: list[str] = []

    def launch(name: str) -> None:
        launched.append(name)
        if name == "a":
            raise ORCH.ExperimentError("command failed (1): a")

    stages = [ORCH.ChildStage(name, lambda name=name: launch(name), lambda: None)
              for name in ("a", "b")]
    with pytest.raises(ORCH.ExperimentError, match="command failed"):
        ORCH.run_child_stages(stages, workers=1)
    assert launched == ["a"], "the serial campaign must not pay for a stage after one failed"


def test_a_non_positive_worker_count_is_refused() -> None:
    with pytest.raises(ORCH.ExperimentError, match="at least one"):
        ORCH.run_child_stages([], workers=0)


# --------------------------------------------------------------------------------------------
# Shared fixtures for the two campaign phases
# --------------------------------------------------------------------------------------------


def _config(root: Path):
    return ORCH.Config(
        experiment_id="exp", root=root / "experiment", functional_run_id="functional",
        functional_submission_sha256="a" * 64, descriptor=root / "target.yaml",
        rtl_facts=root / "rtl.json", perf_profile=root / "perf.yaml",
        gsim_certificate=root / "certificate.json", gsim_certificate_sha256="b" * 64,
        model="gpt-model", effort="high", wall_budget_seconds=60, rounds=2,
        round_timeout_seconds=30, max_tool_calls=5, tool_timeout_seconds=10,
        smoke_replicates=1, holdout_count=4, measurement_timeout=90, gsim_max_cycles=9000,
        functional_gsim_certificate=root / "functional-certificate.json",
        functional_gsim_certificate_sha256="c" * 64,
        telemetry_price_table=root / "prices.yaml", chia_python=root / "chia-python")


def _declaration() -> dict[str, Any]:
    return {"trial_contracts": {trial: {"model": "gpt-model", "trial": trial}
                                for trial in ORCH.TRIALS},
            "agent_telemetry": {"preflight": "pinned"}}


def _certificate(root: Path, name: str, digest: str):
    binary = root / "pinned-gsim"
    if not binary.exists():
        binary.write_bytes(b"exact pinned gsim")
    return SimpleNamespace(
        target="gemmini", path=root / name, sha256=digest,
        pins={"gsim_binary": {"path": str(binary), "sha256": ORCH._sha_file(binary)}})


class _Recorder:
    """A stand-in child launcher that records when, and on which thread, each launch ran."""

    def __init__(self, *, barrier: threading.Barrier | None,
                 write: Any, failures: frozenset[str] = frozenset()):
        self.barrier, self.write, self.failures = barrier, write, failures
        self.launches: list[dict[str, Any]] = []
        self._guard = threading.Lock()

    def __call__(self, argv, cwd, environment):
        run_id = list(argv)[list(argv).index("--run-id") + 1]
        started = time.monotonic()
        if self.barrier is not None:
            self.barrier.wait()
        time.sleep(0.02)
        with self._guard:
            self.launches.append({"run_id": run_id, "thread": threading.current_thread().name,
                                  "started": started, "ended": time.monotonic()})
        if run_id in self.failures:
            return ORCH.CommandResult(3, "", f"child {run_id} refused")
        self.write(run_id)
        return ORCH.CommandResult(0)

    @property
    def order(self) -> list[str]:
        return [row["run_id"] for row in self.launches]

    @property
    def threads(self) -> set[str]:
        return {row["thread"] for row in self.launches}

    def overlapped(self) -> bool:
        return any(one["started"] < other["ended"] and other["started"] < one["ended"]
                   for index, one in enumerate(self.launches)
                   for other in self.launches[index + 1:])


def _normalize(rows, root: Path) -> list[dict[str, Any]]:
    """The record with its absolute run root erased, so two roots can be compared field by field.

    The digests themselves cannot be compared across roots because a checkpoint body carries the
    absolute path of the child evidence it commits; the chain LINKAGE those digests express is
    asserted separately by `_assert_linked`.
    """
    normalized = []
    for row in rows:
        body = {key: value for key, value in row.items()
                if key not in ("path", "sha256", "previous_sha256")}
        text = json.dumps(body, sort_keys=True).replace(str(root.resolve()), "<ROOT>")
        normalized.append(json.loads(text))
    return normalized


def _assert_linked(rows) -> None:
    """The chain is strictly linear: index i links the digest of row i-1, and starts at None."""
    assert [row["index"] for row in rows] == list(range(len(rows)))
    assert [row["previous_sha256"] for row in rows] == \
        [None] + [row["sha256"] for row in rows[:-1]]


# --------------------------------------------------------------------------------------------
# Phase (a): the three authoring trials
# --------------------------------------------------------------------------------------------


def _author(root: Path, monkeypatch: pytest.MonkeyPatch, *, workers: int,
            barrier: threading.Barrier | None):
    root.mkdir(parents=True, exist_ok=True)
    config, declaration = _config(root), _declaration()
    target = SimpleNamespace(target="gemmini")
    stage_root = root / "runs" / "gemmini" / "perf-bench" / "agent_stages"
    telemetry_sha = ORCH._sha_bytes(ORCH._canonical(declaration["agent_telemetry"]))
    commit_threads: list[str] = []

    def write(run_id: str) -> None:
        record = stage_root / run_id / "performance_candidate.json"
        record.parent.mkdir(parents=True, exist_ok=True)
        record.write_text(json.dumps({"run_id": run_id}), encoding="utf-8")

    def handoff(path, _target):
        payload = Path(path).read_bytes()
        trial = json.loads(payload)["run_id"].rsplit("__", 1)[-1]
        commit_threads.append(threading.current_thread().name)
        return SimpleNamespace(
            record_path=Path(path), record_sha256=ORCH._sha_bytes(payload),
            candidate_sha256=ORCH._sha_bytes(payload + b"candidate"),
            agent_contract=dict(declaration["trial_contracts"][trial]),
            telemetry_evidence={"preflight_sha256": telemetry_sha},
            corpus_root=root / "corpus", corpus_manifest_sha256="d" * 64,
            corpus_sha256="e" * 64)

    monkeypatch.setattr(ORCH, "runs_root", lambda target, suite: root / "runs" / target / suite)
    monkeypatch.setattr(ORCH, "_verify_live_agent_treatment", lambda *_args: None)
    monkeypatch.setattr(ORCH, "_handoff", handoff)
    runner = _Recorder(barrier=barrier, write=write)
    state = ORCH.Checkpoints(root / "state", "f" * 64)
    handoffs, evidence = ORCH._author_candidates(
        config, state, target, declaration, environment={"MERLIN_TEST": "1"},
        expected_treatment={"pinned": True}, command_runner=runner, workers=workers)
    return SimpleNamespace(runner=runner, state=state, handoffs=handoffs, evidence=evidence,
                           commit_threads=commit_threads)


def test_concurrent_authoring_writes_the_same_record_as_the_serial_campaign(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    serial_root, concurrent_root = tmp_path / "serial", tmp_path / "concurrent"
    serial = _author(serial_root, monkeypatch, workers=1, barrier=None)
    concurrent = _author(concurrent_root, monkeypatch, workers=3,
                         barrier=threading.Barrier(3, timeout=BARRIER_TIMEOUT))

    assert _normalize(serial.state.load(), serial_root) == \
        _normalize(concurrent.state.load(), concurrent_root)
    _assert_linked(serial.state.load())
    _assert_linked(concurrent.state.load())
    assert [row["stage"] for row in concurrent.state.load()] == \
        [f"candidate:{trial}" for trial in ORCH.TRIALS]
    assert serial.evidence == concurrent.evidence
    assert list(concurrent.handoffs) == list(ORCH.TRIALS)

    # POSITIVE CONTROL: three agent stages were genuinely in flight at once. Without this an
    # implementation that quietly never parallelises would still pass every assertion above.
    assert concurrent.runner.overlapped()
    assert len(concurrent.runner.threads) == 3
    assert serial.runner.threads == {threading.current_thread().name}
    assert serial.runner.order == [f"exp__{trial}" for trial in ORCH.TRIALS]

    # The checkpoint chain is appended only from the thread that owns it.
    assert set(concurrent.commit_threads) == {threading.current_thread().name}


def test_authoring_adopts_a_checkpointed_trial_without_relaunching_it(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    first = _author(tmp_path / "run", monkeypatch, workers=3,
                    barrier=threading.Barrier(3, timeout=BARRIER_TIMEOUT))
    assert len(first.runner.order) == 3

    resumed = _author(tmp_path / "run", monkeypatch, workers=3, barrier=None)
    assert resumed.runner.order == [], "a checkpointed trial must never be paid for twice"
    assert resumed.evidence == first.evidence
    assert [row["stage"] for row in resumed.state.load()] == \
        [f"candidate:{trial}" for trial in ORCH.TRIALS]


def test_a_failed_authoring_trial_reports_every_failure_and_leaves_finished_work_adoptable(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "run"
    root.mkdir()
    config, declaration = _config(root), _declaration()
    target = SimpleNamespace(target="gemmini")
    stage_root = root / "runs" / "gemmini" / "perf-bench" / "agent_stages"

    def write(run_id: str) -> None:
        record = stage_root / run_id / "performance_candidate.json"
        record.parent.mkdir(parents=True, exist_ok=True)
        record.write_text(json.dumps({"run_id": run_id}), encoding="utf-8")

    monkeypatch.setattr(ORCH, "runs_root", lambda target, suite: root / "runs" / target / suite)
    monkeypatch.setattr(ORCH, "_verify_live_agent_treatment", lambda *_args: None)
    runner = _Recorder(barrier=threading.Barrier(3, timeout=BARRIER_TIMEOUT), write=write,
                       failures=frozenset({"exp__trial_00", "exp__trial_02"}))
    state = ORCH.Checkpoints(root / "state", "f" * 64)
    with pytest.raises(ORCH.ExperimentError) as raised:
        ORCH._author_candidates(
            config, state, target, declaration, environment={}, expected_treatment={},
            command_runner=runner, workers=3)

    message = str(raised.value)
    assert "candidate:trial_00" in message and "candidate:trial_02" in message
    assert state.load() == [], "no trial may be checkpointed while a sibling failed"
    # The surviving child's evidence is still on disk, so a resume adopts it instead of paying again.
    assert (stage_root / "exp__trial_01" / "performance_candidate.json").is_file()
    assert ORCH._uncheckpointed_state(
        stage_root / "exp__trial_01",
        stage_root / "exp__trial_01" / "performance_candidate.json",
        label="agent stage trial_01") == "complete"


# --------------------------------------------------------------------------------------------
# Phase (b): the six paired measurement cells
# --------------------------------------------------------------------------------------------


def _measure(root: Path, monkeypatch: pytest.MonkeyPatch, *, workers: int,
             barrier: threading.Barrier | None):
    root.mkdir(parents=True, exist_ok=True)
    config = _config(root)
    runs = root / "perf_runs"
    commit_threads: list[str] = []

    def write(run_id: str) -> None:
        manifest = runs / run_id / "campaign_manifest.json"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps({"run_id": run_id}), encoding="utf-8")

    def verify(path, **_kwargs):
        commit_threads.append(threading.current_thread().name)
        return {"path": str(Path(path).resolve()), "sha256": ORCH._sha_file(Path(path))}

    handoffs = {trial: SimpleNamespace(
        record_path=root / f"{trial}.json", corpus_root=root / "corpus" / trial,
        corpus_manifest_sha256=f"{index}" * 64, corpus_sha256=f"{index + 3}" * 64,
        record_sha256="a" * 64, candidate_sha256="b" * 64)
        for index, trial in enumerate(ORCH.TRIALS)}
    revealed = {"root": str(root / "held_out"), "manifest": str(root / "held_out/manifest.json"),
                "manifest_sha256": "9" * 64, "capsules_sha256": "8" * 64}
    monkeypatch.setattr(ORCH.PB, "RUNS", runs)
    monkeypatch.setattr(ORCH, "_verify_measurement_manifest", verify)
    runner = _Recorder(barrier=barrier, write=write)
    state = ORCH.Checkpoints(root / "state", "f" * 64)
    cells = ORCH._measurement_cells(
        config, handoffs, revealed,
        tuning_certificate=_certificate(root, "certificate.json", "b" * 64),
        heldout_certificate=_certificate(root, "extension.json", "7" * 64))
    manifests = ORCH._measure_cells(cells, config, state, command_runner=runner, workers=workers)
    return SimpleNamespace(runner=runner, state=state, manifests=manifests, cells=cells,
                           commit_threads=commit_threads)


def test_the_paired_matrix_is_every_trial_by_every_phase_in_fixed_order(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    result = _measure(tmp_path / "run", monkeypatch, workers=1, barrier=None)
    assert [(cell.trial, cell.phase) for cell in result.cells] == [
        (trial, phase) for trial in ORCH.TRIALS for phase in ("tuning", "held_out")]


def test_concurrent_measurement_writes_the_same_record_as_the_serial_campaign(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    serial_root, concurrent_root = tmp_path / "serial", tmp_path / "concurrent"
    serial = _measure(serial_root, monkeypatch, workers=1, barrier=None)
    concurrent = _measure(concurrent_root, monkeypatch, workers=6,
                          barrier=threading.Barrier(6, timeout=BARRIER_TIMEOUT))

    assert _normalize(serial.state.load(), serial_root) == \
        _normalize(concurrent.state.load(), concurrent_root)
    _assert_linked(serial.state.load())
    _assert_linked(concurrent.state.load())
    assert [row["stage"] for row in concurrent.state.load()] == [
        f"measurement:{trial}:{phase}"
        for trial in ORCH.TRIALS for phase in ("tuning", "held_out")]
    assert [trial for trial, _path in serial.manifests] == \
        [trial for trial, _path in concurrent.manifests]
    assert [Path(path).relative_to(serial_root) for _trial, path in serial.manifests] == \
        [Path(path).relative_to(concurrent_root) for _trial, path in concurrent.manifests]

    # POSITIVE CONTROL: all six cells were measured at the same time, on six distinct threads.
    assert concurrent.runner.overlapped()
    assert len(concurrent.runner.threads) == 6
    assert serial.runner.threads == {threading.current_thread().name}
    assert serial.runner.order == [f"exp__{trial}__{phase}"
                                   for trial in ORCH.TRIALS
                                   for phase in ("tuning", "held_out")]
    assert set(concurrent.commit_threads) == {threading.current_thread().name}


def test_measurement_adopts_checkpointed_cells_and_relaunches_none(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    first = _measure(tmp_path / "run", monkeypatch, workers=6,
                     barrier=threading.Barrier(6, timeout=BARRIER_TIMEOUT))
    assert len(first.runner.order) == 6

    resumed = _measure(tmp_path / "run", monkeypatch, workers=6, barrier=None)
    assert resumed.runner.order == []
    assert resumed.manifests == first.manifests
