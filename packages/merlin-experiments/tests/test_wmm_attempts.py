"""A finished result is never overwritten: every retry is a new attempt, and the store reads the history.

The incident this holds against: a crashed loop resubmitted the champion's job, the resubmit's worker
was lost, and the store's ``result.json`` read ``infra_worker_lost`` while the real board reading sat in
an archive nothing read -- the best quietly changed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import wmm_fixtures as FX
from merlin_experiments.phase2.whole_model_measured import attempts as A
from merlin_experiments.phase2.whole_model_measured import jobs as J
from merlin_experiments.phase2.whole_model_measured import service as S
from merlin_experiments.phase2.whole_model_measured import store_admin as ADMIN
from merlin_experiments.phase2.whole_model_measured import worker as W
from merlin_experiments.phase2.whole_model_measured.identity import read_json, write_json_atomic

from merlin.perf import whole_model_verdict as V


class _Popen:
    def __init__(self, argv, **kwargs):
        self.pid = 999_999_999


@pytest.fixture(autouse=True)
def _no_workers(monkeypatch):
    monkeypatch.setattr(S, "spawn", _Popen)
    monkeypatch.setattr(S, "alive", lambda pid, owner: False)


def _service(tmp_path: Path) -> S.MeasurementService:
    spec, pin = FX.write_builder(tmp_path)
    return S.MeasurementService(
        tmp_path / "store", target="toy", builder=spec, builder_sha256=pin, machine=FX.spike_machine(tmp_path)
    )


def _measured(job: dict, cycles: int, **fields) -> dict:
    return J.result(
        job,
        timing_status=V.TIMING_MEASURED,
        objective_cycles=cycles,
        verdict={"timing_status": V.TIMING_MEASURED, "whole_window_cycles": cycles, "objective_cycles": cycles},
        **fields,
    )


def _done(service: S.MeasurementService, name: str, cycles: int) -> tuple[Path, dict]:
    job = service.request(FX.package(service.root.parent, name, doc=name))
    job_dir = service.root / job["package_sha256"]
    J.write_result(job_dir, _measured(job, cycles))
    job.update(state=J.DONE)
    write_json_atomic(job_dir / "job.json", job)
    return job_dir, job


def test_a_result_is_written_once_and_a_second_write_is_refused_with_the_bytes_unchanged(tmp_path):
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    J.write_result(job_dir, {"timing_status": V.TIMING_MEASURED, "objective_cycles": 1})
    before = (job_dir / "result.json").read_bytes()
    with pytest.raises(J.ResultExists):
        J.write_result(job_dir, {"timing_status": V.TIMING_REFUSED, "refusal": "infra_worker_lost: lost"})
    assert (job_dir / "result.json").read_bytes() == before
    assert not [p for p in job_dir.iterdir() if p.name.startswith(".result.json")]  # no temporary left behind


def test_a_lost_resubmit_never_hides_the_board_reading_it_followed(tmp_path):
    """The champion incident, replayed: a measured job is re-queued, its worker is lost twice, and the
    store's current result is ``infra_worker_lost`` -- the measured reading still stands, from its attempt."""
    service = _service(tmp_path)
    job_dir, job = _done(service, "champion", 37_227_228)
    ADMIN.requeue_solo(service.root, job["package_sha256"], why="a solo repeat")
    fresh = read_json(job_dir / "job.json")
    fresh.update(state=J.RUNNING, worker_pid=4)
    write_json_atomic(job_dir / "job.json", fresh)
    for _ in range(S.WORKER_LOSS_REQUEUES + 1):  # lost, re-queued, lost again: FAILED, infra result written
        fresh = read_json(job_dir / "job.json")
        fresh.update(state=J.RUNNING, worker_pid=4)
        write_json_atomic(job_dir / "job.json", fresh)
        service.poll()
    current = read_json(job_dir / "result.json")
    assert current["infra_worker_lost"] is True
    standing = service.result(job["package_sha256"])
    assert standing["objective_cycles"] == 37_227_228 and standing["timing_status"] == V.TIMING_MEASURED
    assert standing["from_attempt"]["attempt"] == 0 and standing["current_attempt"]["timing_status"] == "REFUSED"
    assert service.best()["objective_cycles"] == 37_227_228
    row = next(r for r in service.history() if r["package_sha256"] == job["package_sha256"])
    assert row["from_attempt"] == 0 and row["attempts"] >= 2


def test_reading_only_the_current_attempt_would_lose_the_reading(tmp_path):
    """MUTATION: the same store read the old way (``result.json`` alone) has no best at all -- which is
    exactly what the history read prevents."""
    service = _service(tmp_path)
    job_dir, job = _done(service, "champion", 100)
    ADMIN.reopen(service.root, job["package_sha256"], why="re-measure")
    J.write_result(
        job_dir, J.refused(job, "infra_worker_lost: measurement lost to host pressure", infra_worker_lost=True)
    )
    assert V.objective_cycles((read_json(job_dir / "result.json") or {}).get("verdict")) is None
    assert service.best() is not None and service.best()["objective_cycles"] == 100


def test_a_retracted_verdict_no_longer_stands(tmp_path):
    service = _service(tmp_path)
    job_dir, job = _done(service, "contaminated", 100)
    ADMIN.reopen(service.root, job["package_sha256"], why="the board image was wrong", retract=True)
    assert read_json(job_dir / J.ATTEMPTS_DIR / "0" / J.ATTEMPT_RECORD)["retracted"] is True
    assert service.result(job["package_sha256"]) is None
    J.write_result(job_dir, J.refused(job, "infra_worker_lost: lost", infra_worker_lost=True))
    assert service.result(job["package_sha256"])["infra_worker_lost"] is True and service.best() is None


def test_a_newer_verdict_replaces_an_older_one(tmp_path):
    service = _service(tmp_path)
    job_dir, job = _done(service, "p", 100)
    ADMIN.reopen(service.root, job["package_sha256"], why="the builder changed")
    J.write_result(job_dir, _measured(job, 90))
    assert service.result(job["package_sha256"])["objective_cycles"] == 90
    assert "from_attempt" not in service.result(job["package_sha256"])
    assert len(A.history(job_dir)) == 2


def test_a_worker_finishing_after_its_job_already_ended_keeps_its_result_beside_it(tmp_path, monkeypatch):
    service = _service(tmp_path)
    job_dir, job = _done(service, "p", 100)
    before = (job_dir / "result.json").read_bytes()
    monkeypatch.setattr(W, "work", lambda path: _measured(job, 50))
    W.worker_main(job_dir)
    assert (job_dir / "result.json").read_bytes() == before
    late = job_dir / J.ATTEMPTS_DIR / "0"
    assert read_json(late / "result.json")["objective_cycles"] == 50
    assert read_json(late / J.ATTEMPT_RECORD)["kind"] == "late_result"


def test_a_superseded_job_requested_again_keeps_its_history(tmp_path):
    service = _service(tmp_path)
    job_dir, job = _done(service, "p", 100)
    ADMIN.reopen(service.root, job["package_sha256"], why="again")
    service.supersede(job["package_sha256"], reason="a newer best", stop_running=False)
    service.request(FX.package(tmp_path, "p", doc="p"))
    rows = A.history(job_dir)
    assert [r["result"]["timing_status"] for r in rows] == [V.TIMING_MEASURED, V.TIMING_REFUSED]
    assert service.result(job["package_sha256"])["objective_cycles"] == 100


def test_attempts_archived_by_an_earlier_layout_are_still_history(tmp_path):
    job_dir = tmp_path / "job"
    (job_dir / "lost_attempt_0").mkdir(parents=True)
    write_json_atomic(
        job_dir / "lost_attempt_0" / "result.json",
        {"timing_status": V.TIMING_MEASURED, "objective_cycles": 7, "finished_at": "20261001T000000Z"},
    )
    write_json_atomic(
        job_dir / "result.json",
        {"timing_status": V.TIMING_REFUSED, "refusal": "infra_worker_lost: x", "infra_worker_lost": True},
    )
    standing = A.effective_result(job_dir)
    assert standing["objective_cycles"] == 7 and standing["from_attempt"]["attempt"] == "lost_attempt_0"


def test_an_operator_infra_mark_is_an_annotation_and_does_not_displace_an_earlier_verdict(tmp_path):
    service = _service(tmp_path)
    job_dir, job = _done(service, "p", 100)
    ADMIN.reopen(service.root, job["package_sha256"], why="re-measure")
    J.write_result(job_dir, J.refused(job, "screen: the selfcheck said no"))
    raw = (job_dir / "result.json").read_bytes()
    assert service.result(job["package_sha256"])["timing_status"] == V.TIMING_REFUSED  # a refusal is a verdict...
    ADMIN.mark_infra(service.root, [job["package_sha256"]], kind="harness_import", why="the harness was broken")
    assert (job_dir / "result.json").read_bytes() == raw
    assert service.result(job["package_sha256"])["objective_cycles"] == 100  # ...until it is marked the host's
    assert json.loads((job_dir / J.ANNOTATIONS_FILE).read_text())["infra_marked"]["kind"] == "harness_import"
