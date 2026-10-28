"""Pooled and remembered "bytes in, plain data out" jobs return exactly what the serial call returns."""

from __future__ import annotations

import pytest

from merlin.common import parse_pool as PP
from merlin.targetgen.application_graph import application_graph_inventory, application_graph_inventory_from_bytes

GRAPH = "merlin.targetgen.application_graph:application_graph_inventory_from_bytes"
SOURCE = "module {{ func.func @f() {{ %x = arith.constant {value} : i32 func.return }} }}"

_CALLS: list[bytes] = []


def counted(raw: bytes):
    _CALLS.append(raw)
    return {"n": len(raw), "items": [raw[:1]]}


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    PP.clear_memo()
    _CALLS.clear()
    monkeypatch.delenv(PP.WORKERS_ENV, raising=False)
    yield
    PP.clear_memo()


def write(tmp_path, count=4):
    paths = []
    for index in range(count):
        path = tmp_path / f"m{index}.mlir"
        path.write_text(SOURCE.format(value=index + 2))
        paths.append(path)
    return paths


def test_pooled_values_equal_serial_values_in_consumption_order(tmp_path, monkeypatch):
    paths = write(tmp_path)
    serial = [application_graph_inventory(path) for path in paths]
    monkeypatch.setenv(PP.WORKERS_ENV, "2")
    with PP.Prefetch(GRAPH, paths) as jobs:
        assert jobs._pool is not None, "two workers were requested for four files"
        pooled = [jobs.get(path) for path in paths]
    assert pooled == serial
    PP.clear_memo()
    with PP.Prefetch(GRAPH, paths) as jobs:
        assert [jobs.get(path) for path in reversed(paths)] == list(reversed(serial))


def test_a_failing_job_raises_the_serial_exception_where_it_is_consumed(tmp_path, monkeypatch):
    paths = write(tmp_path)
    paths[2].write_text("module { this is not mlir ")
    with pytest.raises(Exception) as serial:
        application_graph_inventory(paths[2])
    monkeypatch.setenv(PP.WORKERS_ENV, "2")
    with PP.Prefetch(GRAPH, paths) as jobs:
        first = jobs.get(paths[0])
        second = jobs.get(paths[1])
        with pytest.raises(type(serial.value)) as pooled:
            jobs.get(paths[2])
        third = jobs.get(paths[3])
    assert str(pooled.value) == str(serial.value)
    assert first == application_graph_inventory(paths[0]) and second == application_graph_inventory(paths[1])
    assert third == application_graph_inventory(paths[3])
    with pytest.raises(FileNotFoundError):
        PP.Prefetch(GRAPH, []).get(tmp_path / "absent.mlir")


def test_bytes_changed_after_the_start_are_never_served_from_the_earlier_read(tmp_path, monkeypatch):
    paths = write(tmp_path)
    monkeypatch.setenv(PP.WORKERS_ENV, "2")
    with PP.Prefetch(GRAPH, paths) as jobs:
        paths[1].write_text(SOURCE.format(value=99))
        got = jobs.get(paths[1])
        assert got == application_graph_inventory_from_bytes(paths[1].read_bytes())
        assert got["capture_sha256"] != application_graph_inventory(paths[0])["capture_sha256"]


def test_memo_is_keyed_by_content_and_hands_out_private_copies(tmp_path):
    path = write(tmp_path, 1)[0]
    first = PP.Prefetch(GRAPH, []).get(path)
    first["capture_graph"]["operations"].clear()
    first["mutated"] = True
    second = PP.Prefetch(GRAPH, []).get(path)
    assert "mutated" not in second and second["capture_graph"]["operations"]
    assert second == application_graph_inventory(path)
    path.write_text(SOURCE.format(value=77))
    third = PP.Prefetch(GRAPH, []).get(path)
    assert third["capture_sha256"] != second["capture_sha256"]
    assert third == application_graph_inventory(path)


def test_cached_runs_each_distinct_content_once_and_the_serial_setting_never_remembers(monkeypatch):
    ref = f"{__name__}:counted"
    assert PP.cached(ref, b"abc") == PP.cached(ref, b"abc") == {"n": 3, "items": [b"a"]}
    PP.cached(ref, b"abd")
    assert len(_CALLS) == 2
    PP.cached(ref, b"abc")["items"].append(b"x")
    assert PP.cached(ref, b"abc") == {"n": 3, "items": [b"a"]}
    monkeypatch.setenv(PP.WORKERS_ENV, "0")
    _CALLS.clear()
    PP.cached(ref, b"abc")
    PP.cached(ref, b"abc")
    assert len(_CALLS) == 2
    jobs = PP.Prefetch(ref, {"a": b"abc", "b": b"abd"}).start()
    assert jobs._pool is None
    jobs.get("a", b"abc")
    jobs.get("a", b"abc")
    assert len(_CALLS) == 4


def test_worker_count_is_bounded_by_the_knob_cpus_and_memory(monkeypatch):
    monkeypatch.setenv(PP.WORKERS_ENV, "0")
    assert PP.worker_count(9) == 0
    monkeypatch.setenv(PP.WORKERS_ENV, "4")
    assert PP.worker_count(9) == 4 and PP.worker_count(3) == 3 and PP.worker_count(1) == 0
    monkeypatch.delenv(PP.WORKERS_ENV)
    monkeypatch.setattr(PP.os, "sched_getaffinity", lambda _: set(range(64)), raising=False)
    monkeypatch.setattr(PP, "_mem_available", lambda: 1 << 40)
    assert PP.worker_count(100) == PP._DEFAULT_CAP and PP.worker_count(5) == 5
    monkeypatch.setattr(PP, "_mem_available", lambda: 5 << 30)
    assert PP.worker_count(100) == 2
    monkeypatch.setattr(PP, "_mem_available", lambda: 1 << 30)
    assert PP.worker_count(100) == 0
    monkeypatch.setattr(PP.os, "sched_getaffinity", lambda _: {0}, raising=False)
    assert PP.worker_count(100) == 0


def test_normalization_job_matches_the_text_function(tmp_path, monkeypatch):
    from merlin.frontends.capture_normalization import normalize_capture_bytes, normalize_capture_mlir

    raw = SOURCE.format(value=5).encode()
    ref = "merlin.frontends.capture_normalization:normalize_capture_bytes"
    monkeypatch.setenv(PP.WORKERS_ENV, "2")
    jobs = PP.Prefetch(ref, {"a": raw, "b": SOURCE.format(value=6).encode()}).start()
    try:
        assert jobs.get("a", raw) == normalize_capture_mlir(raw.decode()) == normalize_capture_bytes(raw)
    finally:
        jobs.close()
