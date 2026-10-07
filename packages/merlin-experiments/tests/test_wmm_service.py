"""The measurement store: requests keyed by exact bytes, the screen, reopening stale outcomes, losses."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import wmm_fixtures as FX
from merlin_experiments.phase2.whole_model_measured import jobs as J
from merlin_experiments.phase2.whole_model_measured import service as S
from merlin_experiments.phase2.whole_model_measured.identity import read_json, write_json_atomic


class _Popen:
    started: list[list[str]] = []

    def __init__(self, argv, **kwargs):
        _Popen.started.append(list(argv))
        self.pid = 999_999_999  # never alive: a dispatched worker in these tests does not run


@pytest.fixture(autouse=True)
def _no_workers(monkeypatch):
    """Dispatch is recorded, never executed; a dispatched fake worker counts as alive."""
    _Popen.started = []
    monkeypatch.setattr(S, "spawn", _Popen)
    monkeypatch.setattr(S, "alive", lambda pid, owner: pid == 999_999_999)


def _check(tmp_path: Path, *, passes: bool, tail: str = "") -> dict:
    script = tmp_path / f"check_{int(passes)}.py"
    script.write_text(
        "import json, sys\n"
        f"print({tail!r})\n"
        f"rows = [{{'capsule': 'c', 'pass': {passes}}}]\n"
        f"json.dump({{'all_pass': {passes}, 'per_capsule': rows}}, open(sys.argv[1], 'w'))\n"
    )
    return {"argv": [sys.executable, str(script), "{out}"], "required": True, "label": "screen", "capsules": "c"}


def _service(tmp_path: Path, **kw) -> S.MeasurementService:
    spec, pin = FX.write_builder(tmp_path)
    return S.MeasurementService(
        tmp_path / "store", target="toy", builder=spec, builder_sha256=pin, machine=FX.spike_machine(tmp_path), **kw
    )


def test_the_seed_runs_whatever_its_screen_says_and_a_later_failure_is_not_built(tmp_path):
    service = _service(tmp_path, pre_measure_check=_check(tmp_path, passes=False))
    seed = service.request(FX.package(tmp_path, "seed"))
    assert seed["state"] == J.RUNNING and seed["screen_exempt"] is True
    later = service.request(FX.package(tmp_path, "later", argmax=4))
    assert later["state"] == J.SCREEN_FAILED
    assert service.result(later["package_sha256"])["refusal"].startswith("screen_failed")
    assert len(_Popen.started) == 1


def test_an_infra_screen_failure_under_another_spec_is_rescreened_not_reused(tmp_path):
    broken = _check(tmp_path, passes=False, tail="Traceback (most recent call last): ImportError: harness")
    first = _service(tmp_path, pre_measure_check=broken)
    first.request(FX.package(tmp_path, "seed"))
    candidate = FX.package(tmp_path, "cand", argmax=4)
    assert first.request(candidate)["state"] == J.SCREEN_FAILED
    fixed = _service(tmp_path, pre_measure_check=_check(tmp_path, passes=True))
    reopened = fixed.request(candidate)
    assert reopened["state"] in (J.PENDING, J.RUNNING) and reopened["reopened_infra_snapshot_mismatch"] is True
    attempt = fixed.root / reopened["package_sha256"] / J.ATTEMPTS_DIR / "0"
    assert (attempt / "result.json").is_file() and read_json(attempt / J.ATTEMPT_RECORD)[
        "kind"
    ] == "screen_failed_attempt"


def test_a_package_caused_screen_failure_is_never_reopened_by_a_new_spec(tmp_path):
    first = _service(tmp_path, pre_measure_check=_check(tmp_path, passes=False))
    first.request(FX.package(tmp_path, "seed"))
    candidate = FX.package(tmp_path, "cand", argmax=4)
    assert first.request(candidate)["state"] == J.SCREEN_FAILED
    other = _service(tmp_path, pre_measure_check={**_check(tmp_path, passes=True), "label": "other"})
    assert other.request(candidate)["state"] == J.SCREEN_FAILED


def _done(service: S.MeasurementService, pkg: Path, *, builder_sha: str, refusal: str | None = None) -> Path:
    job = service.request(pkg)
    job_dir = service.root / job["package_sha256"]
    job.update(state=J.DONE)
    write_json_atomic(job_dir / "job.json", job)
    result = J.result(job, timing_status="MEASURED", objective_cycles=10, refusal=refusal)
    result["builder"] = {**job["builder"], "module_identity": {"sha256": builder_sha}}
    write_json_atomic(job_dir / "result.json", result)
    (job_dir / "build").mkdir()
    (job_dir / "build" / "big.o").write_text("x")
    return job_dir


def test_a_done_job_whose_builder_changed_is_remeasured_once_with_the_new_builder_recorded(tmp_path):
    service = _service(tmp_path)
    pkg = FX.package(tmp_path, "p")
    job_dir = _done(service, pkg, builder_sha="an-older-builder")
    reopened = service.request(pkg)
    assert reopened["reopened_builder_changed"] is True and reopened["state"] in (J.PENDING, J.RUNNING)
    assert (job_dir / J.ATTEMPTS_DIR / "0" / "result.json").is_file() and not (job_dir / "result.json").exists()
    assert reopened["builder"]["module_identity"] == service.builder["module_identity"]
    # The refreshed record means the next request finds nothing stale.
    record = read_json(job_dir / "job.json")
    result = J.result(record, timing_status="MEASURED", objective_cycles=10)
    record["state"] = J.DONE
    write_json_atomic(job_dir / "job.json", record)
    write_json_atomic(job_dir / "result.json", result)
    assert service.request(pkg)["state"] == J.DONE


def test_a_done_infra_refusal_is_remeasured_under_a_new_screen_spec(tmp_path):
    service = _service(tmp_path)
    pkg = FX.package(tmp_path, "p")
    current = service.builder["module_identity"]["sha256"]
    _done(service, pkg, builder_sha=current, refusal="build: KeyError: 'g70'")
    assert service.request(pkg)["state"] == J.DONE  # same spec, same builder: kept
    fixed = _service(tmp_path, pre_measure_check=_check(tmp_path, passes=True))
    assert fixed.request(pkg)["reopened_infra_build_regression"] is True


def test_a_documentation_edit_is_aliased_to_the_same_program(tmp_path):
    service = _service(tmp_path)
    first = service.request(FX.package(tmp_path, "a"))
    edited = service.request(FX.package(tmp_path, "b", doc="notes"))
    assert edited["aliased_from"] != first["package_sha256"] and edited["package_sha256"] == first["package_sha256"]
    assert service.measurement_for(edited["aliased_from"])["same_program_as"] == first["package_sha256"]


def test_a_lost_worker_is_requeued_once_under_this_services_screen_then_recorded_as_lost(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "alive", lambda pid, owner: False)
    old = _service(tmp_path, pre_measure_check=_check(tmp_path, passes=True))
    job = old.request(FX.package(tmp_path, "p"))
    assert job["state"] == J.RUNNING
    current = _service(tmp_path, pre_measure_check={**_check(tmp_path, passes=True), "label": "relaunched"})
    current.poll()
    requeued = read_json(current.root / job["package_sha256"] / "job.json")
    # dispatched again (fake worker), carrying THIS service's spec, its loss on record
    assert requeued["worker_losses"] and requeued["pre_measure_check"]["label"] == "relaunched"
    assert (
        read_json(current.root / job["package_sha256"] / J.ATTEMPTS_DIR / "0" / J.ATTEMPT_RECORD)["kind"]
        == "lost_attempt"
    )
    current.poll()
    lost = read_json(current.root / job["package_sha256"] / "job.json")
    assert lost["state"] == J.FAILED and "not a verdict" in lost["failure"]
    assert current.result(job["package_sha256"])["infra_worker_lost"] is True


def test_a_board_job_inside_a_batch_is_never_stopped_by_a_supersede(tmp_path):
    service = _service(tmp_path)
    job = service.request(FX.package(tmp_path, "p"))
    job.update(state=J.RUNNING, batch="b1")
    write_json_atomic(service.root / job["package_sha256"] / "job.json", job)
    assert service.supersede(job["package_sha256"], reason="newer best", stop_running=True) is None
    assert read_json(service.root / job["package_sha256"] / "job.json")["state"] == J.RUNNING


def test_a_full_queue_supersedes_its_oldest_request_and_says_by_whom(tmp_path):
    service = _service(tmp_path, max_pending=1)
    running = service.request(FX.package(tmp_path, "a"))
    older = service.request(FX.package(tmp_path, "b", argmax=10))
    newer = service.request(FX.package(tmp_path, "c", argmax=11))
    assert running["state"] == J.RUNNING and newer["state"] == J.PENDING
    displaced = read_json(service.root / older["package_sha256"] / "job.json")
    assert displaced["state"] == J.SUPERSEDED and displaced["superseded_by"] == newer["package_sha256"]
