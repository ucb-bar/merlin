"""Operator surgery on a measurement store: every edit states why, is logged, never deletes, never
touches a running job, and never edits what was measured."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import wmm_fixtures as FX
from merlin_experiments.phase2.whole_model_measured import cli
from merlin_experiments.phase2.whole_model_measured import jobs as J
from merlin_experiments.phase2.whole_model_measured import service as S
from merlin_experiments.phase2.whole_model_measured import store_admin as A
from merlin_experiments.phase2.whole_model_measured.identity import read_json, write_json_atomic


def _store(tmp_path: Path, *, refusal: str | None = "build: no such capsule", state: str = J.DONE) -> tuple[Path, str]:
    store = tmp_path / "store"
    job_dir = FX.job_dir_for(
        store, FX.package(tmp_path, "p"), machine=FX.spike_machine(tmp_path), builder=FX.write_builder(tmp_path)
    )
    job = read_json(job_dir / "job.json")
    job["state"] = state
    write_json_atomic(job_dir / "job.json", job)
    status = "REFUSED" if refusal else "MEASURED"
    write_json_atomic(
        job_dir / "result.json",
        {
            **J.result(job, timing_status=status, objective_cycles=None if refusal else 10, refusal=refusal),
            "builder": {"note": "old"},
        },
    )
    (job_dir / "build").mkdir()
    (job_dir / "build" / "x.o").write_text("o")
    return store, job_dir.name


def _log(store: Path) -> list[dict]:
    return [json.loads(line) for line in (store / A.ADMIN_LOG).read_text().splitlines()]


def test_every_edit_states_why_and_is_logged(tmp_path):
    store, key = _store(tmp_path)
    with pytest.raises(A.AdminError, match="why"):
        A.mark_infra(store, [key[:12]], kind="snapshot_contamination", why=" ")
    assert A.mark_infra(store, [key[:12]], kind="snapshot_contamination", why="uncommitted edit in the snapshot") == [
        key
    ]
    # The mark is an annotation BESIDE the result, laid over it on read; the result file is untouched.
    assert A.INFRA_MARK not in read_json(store / key / "result.json")
    assert read_json(store / key / J.ANNOTATIONS_FILE)[A.INFRA_MARK]["kind"] == "snapshot_contamination"
    from merlin_experiments.phase2.whole_model_measured import attempts as AT

    assert AT.effective_result(store / key)[A.INFRA_MARK]["kind"] == "snapshot_contamination"
    assert _log(store)[-1]["operation"] == "mark-infra"


def test_only_a_refusal_can_be_marked_infra_caused(tmp_path):
    store, key = _store(tmp_path, refusal=None)
    with pytest.raises(A.AdminError, match="MEASURED"):
        A.mark_infra(store, [key], kind="x", why="y")


def test_a_marked_infra_refusal_is_remeasured_under_a_new_screen(tmp_path, monkeypatch):
    store, key = _store(tmp_path, refusal="screen: the selfcheck said no")
    monkeypatch.setattr(S, "spawn", lambda argv, **kw: type("P", (), {"pid": 999_999_999})())
    spec, pin = FX.write_builder(tmp_path)
    service = S.MeasurementService(
        store, target="toy", builder=spec, builder_sha256=pin, machine=FX.spike_machine(tmp_path)
    )
    job = read_json(store / key / "job.json")
    job["builder"] = dict(service.builder)
    write_json_atomic(store / key / "job.json", job)
    result = read_json(store / key / "result.json")
    result["builder"] = dict(service.builder)
    write_json_atomic(store / key / "result.json", result)
    service.pre_measure_check = {"argv": ["true"], "label": "fixed harness"}
    assert service._reopen_stale(store / key, read_json(store / key / "job.json"), exempt=False)["state"] == J.DONE
    A.mark_infra(store, [key], kind="harness_import", why="the harness imported the live checkout")
    reopened = service._reopen_stale(store / key, read_json(store / key / "job.json"), exempt=False)
    assert reopened["state"] == J.PENDING and reopened["reopened_infra_build_regression"] is True


def test_requeue_solo_sets_the_attempt_aside_and_never_deletes_it(tmp_path):
    store, key = _store(tmp_path)
    job = A.requeue_solo(store, key, why="control drift before the drift rule existed")
    assert job["state"] == J.PENDING and job["solo"] is True and "requeued by the operator" in job["notice"]
    attempt = store / key / J.ATTEMPTS_DIR / "0"
    assert (attempt / "result.json").is_file() and not (store / key / "result.json").exists()
    assert read_json(attempt / J.ATTEMPT_RECORD)["kind"] == "paused_attempt"


def test_a_running_job_is_never_touched(tmp_path):
    store, key = _store(tmp_path, state=J.RUNNING)
    with pytest.raises(A.AdminError, match="running"):
        A.reopen(store, key, why="x")


def test_a_citation_is_corrected_with_its_old_value_and_a_measurement_never_is(tmp_path):
    store, key = _store(tmp_path, refusal=None)
    with pytest.raises(A.AdminError, match="re-taken"):
        A.correct_citation(store, key, field="objective_cycles", value=1, why="x")
    before = (store / key / "result.json").read_bytes()
    was = A.correct_citation(store, key, field="builder.note", value="the fixed builder", why="stale citation")
    assert was == {"job.json": None, "result.json": "old"}
    from merlin_experiments.phase2.whole_model_measured import attempts as AT

    assert (store / key / "result.json").read_bytes() == before  # a result file is never rewritten
    result = AT.effective_result(store / key)
    assert result["builder"]["note"] == "the fixed builder" and result["citation_corrections"][0]["was"] == "old"
    assert result["objective_cycles"] == 10


def test_the_plateau_operations_run_through_the_command_line(tmp_path, capsys):
    store, _key = _store(tmp_path)
    assert cli.main(["admin", str(store), "reset-plateau", "--at", "20260926T175000Z", "--why", "tooling change"]) == 0
    trace = json.loads((store / "plateau.json").read_text())["trace"]
    assert trace[0]["reset_at"] == "20260926T175000Z" and _log(store)[-1]["operation"] == "reset-plateau"


def test_a_board_reported_back_is_tried_now_and_the_outage_stays_open(tmp_path, capsys):
    """A report is not evidence the board works: the next batch may try it at once, and only a batch
    that runs its workload closes the outage."""
    from merlin_experiments.phase2.whole_model_measured import batch as B

    store, _key = _store(tmp_path)
    with pytest.raises(A.AdminError, match="no open board outage"):
        A.outage_retry_now(store, why="board re-enumerated")
    write_json_atomic(store / B.BOARD_OUTAGE, {"opened_at": "x", "failures": [{}], "retry_after_epoch": 4e9})
    assert cli.main(["admin", str(store), "outage-retry-now", "--why", "U250 re-enumerated, xdma probed"]) == 0
    outage = read_json(store / B.BOARD_OUTAGE)
    assert outage["retry_after_epoch"] < 4e9 and outage["reported_back"][0]["why"] == "U250 re-enumerated, xdma probed"
    assert B.board_outage(store) is not None and _log(store)[-1]["operation"] == "outage-retry-now"
