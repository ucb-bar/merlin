"""No build starts while the store's or the workers' TMPDIR filesystem is below its floor: the job
waits -- deferred, never refused -- and says why; the store records the hold and closes it when space
returns.  A build that died of ENOSPC halfway would cost its time and read like the candidate's fault."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import wmm_fixtures as FX
from merlin_experiments.phase2.whole_model_measured import config as C
from merlin_experiments.phase2.whole_model_measured import jobs as J
from merlin_experiments.phase2.whole_model_measured import service as S
from merlin_experiments.phase2.whole_model_measured.identity import read_json

GIB = 1024**3


class _Popen:
    started: list[list[str]] = []

    def __init__(self, argv, **kwargs):
        _Popen.started.append(list(argv))
        self.pid = 999_999_999


@pytest.fixture(autouse=True)
def _no_workers(monkeypatch):
    _Popen.started = []
    monkeypatch.setattr(S, "spawn", _Popen)
    monkeypatch.setattr(S, "alive", lambda pid, owner: pid == 999_999_999)


def _service(tmp_path: Path, **kw) -> S.MeasurementService:
    spec, pin = FX.write_builder(tmp_path)
    return S.MeasurementService(
        tmp_path / "store",
        target="toy",
        builder=spec,
        builder_sha256=pin,
        machine=FX.spike_machine(tmp_path),
        environment={"TMPDIR": str(tmp_path / "worker_tmp")},
        **kw,
    )


def test_the_floor_is_read_from_both_filesystems_a_build_writes(tmp_path, monkeypatch):
    seen = []
    (tmp_path / "a").mkdir()
    monkeypatch.setattr(S.shutil, "disk_usage", lambda path: seen.append(str(path)) or type("U", (), {"free": 9})())
    reason = S.build_disk_refusal({"store": tmp_path / "a", "worker TMPDIR": tmp_path / "b"}, minimum=10)
    assert reason.startswith(S.DISK_LOW) and str(tmp_path / "a") in reason and " 9 byte(s)" in reason
    assert S.build_disk_refusal({"store": tmp_path / "a"}, minimum=9) is None
    assert seen[0] == str(tmp_path / "a")


def test_a_low_disk_holds_every_build_and_the_job_says_why(tmp_path, monkeypatch):
    service = _service(tmp_path)
    free = {"store": 50 * GIB, "tmp": 1 * GIB}
    monkeypatch.setattr(S, "build_free_bytes", lambda path: free["tmp"] if "worker_tmp" in str(path) else free["store"])
    job = service.request(FX.package(tmp_path, "seed"))
    assert job["state"] == J.PENDING and _Popen.started == []
    assert job["notice"].startswith(S.DISK_LOW) and "worker TMPDIR" in job["notice"]
    hold = read_json(service.root / S.DISK_HOLD)
    assert hold["minimum_free_bytes"] == S.MIN_BUILD_FREE_BYTES and hold["reason"] == job["notice"]
    assert service.poll() == [] and _Popen.started == []
    # The space returns: the build starts, the hold is closed into the log, the job's notice is cleared.
    free["tmp"] = 50 * GIB
    (key,) = service.poll()
    started = read_json(service.root / key / "job.json")
    assert started["state"] == J.RUNNING and "notice" not in started
    assert not (service.root / S.DISK_HOLD).exists()
    (closed,) = [json.loads(line) for line in (service.root / S.DISK_HOLDS).read_text().splitlines()]
    assert closed["opened_at"] == hold["opened_at"] and closed["closed_at"]


def test_the_floor_is_the_sections_own_when_it_declares_one(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "build_free_bytes", lambda path: 2 * GIB)
    assert _service(tmp_path, min_build_free_bytes=GIB).request(FX.package(tmp_path, "seed"))["state"] == J.RUNNING
    document = {
        "schema": C.CONFIG_SCHEMA,
        "builder": dict(zip(("spec", "sha256"), FX.write_builder(tmp_path), strict=True)),
        "store": str(tmp_path / "store2"),
        "screen": {"machine": FX.spike_machine(tmp_path), "build_options": {}, "min_build_free_bytes": 3 * GIB},
    }
    objective = C.from_config(document, target="toy")
    assert objective.screen.min_build_free_bytes == 3 * GIB


def test_a_tmpdir_not_created_yet_is_measured_where_it_will_be(tmp_path):
    assert S.build_free_bytes(tmp_path / "not" / "yet" / "made") == S.shutil.disk_usage(tmp_path).free
