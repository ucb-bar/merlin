"""Content-qualified result ABI reuse cannot backdate another source's signature."""

import os
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from merlin.common import mlir_query as Q
from merlin.frontends import linalg_mlir as L
from merlin.llvmlower import c_runtime as C
from merlin.llvmlower import generic_form as G

GENERIC = """"builtin.module"() ({
 "func.func"() ({
 ^bb0(%arg0: tensor<2xf32>):
  "func.return"(%arg0) : (tensor<2xf32>) -> ()
 }) {function_type = (tensor<2xf32>) -> tensor<2xf32>, sym_name = "forward"} : () -> ()
}) : () -> ()"""


@pytest.fixture(autouse=True)
def fresh_caches(monkeypatch):
    monkeypatch.setattr(Q, "_PARSE_CACHE", OrderedDict())
    monkeypatch.setattr(Q, "_SIGNATURE_CACHE", OrderedDict())


def source(tmp_path, name="model.mlir"):
    path = tmp_path / name
    path.write_text(GENERIC)
    return path


def test_repeated_result_queries_parse_once_and_return_unshared_shapes(tmp_path, monkeypatch):
    path, calls, parser = source(tmp_path), [], Q.parse_mlir_file

    def count(path, **kwargs):
        calls.append(path)
        return parser(path, **kwargs)

    monkeypatch.setattr(Q, "parse_mlir_file", count)
    first = C._out_specs(path)
    first[0][0][0] = 999
    first.append(([5], "i32"))
    for _ in range(66):
        assert C._out_specs(path) == [([2], "f32")]
    assert calls == [path]


def test_same_size_restored_mtime_still_invalidates_both_caches(tmp_path):
    path = source(tmp_path)
    original_stat = path.stat()
    assert Q.forward_signature(Q.parse(path))[1] == [([2], "f32")]
    assert C._out_specs(path) == [([2], "f32")]
    path.write_text(GENERIC.replace("2xf32", "3xf32"))
    os.utime(path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    assert path.stat().st_size == original_stat.st_size
    assert path.stat().st_mtime_ns == original_stat.st_mtime_ns
    assert C._out_specs(path) == [([3], "f32")]
    assert Q.forward_signature(Q.parse(path))[1] == [([3], "f32")]


def test_timestamp_only_change_keeps_valid_content_cache(tmp_path, monkeypatch):
    path = source(tmp_path)
    assert C._out_specs(path) == [([2], "f32")]
    os.utime(path, ns=(1, 1))
    monkeypatch.setattr(Q, "parse_mlir_file", lambda *a, **k: pytest.fail("unchanged bytes were reparsed"))
    assert C._out_specs(path) == [([2], "f32")]


@pytest.mark.parametrize("query", [C._out_specs, Q.parse])
def test_cache_hit_refuses_a_source_change_after_its_initial_read(tmp_path, monkeypatch, query):
    path = source(tmp_path)
    C._out_specs(path)
    original_read, changed = Path.read_bytes, []

    def read_then_change(member):
        payload = original_read(member)
        if member == path and not changed:
            changed.append(True)
            path.write_text(GENERIC.replace("2xf32", "3xf32"))
        return payload

    monkeypatch.setattr(Path, "read_bytes", read_then_change)
    with pytest.raises(ValueError, match="changed during"):
        query(path)


def test_a_to_b_to_a_writer_never_publishes_b_under_a(tmp_path, monkeypatch):
    path = source(tmp_path)
    original_stat, parser = path.stat(), Q.parse_mlir_file

    def transient(path, *, source_text):
        path.write_text(GENERIC.replace("2xf32", "3xf32"))
        parsed = parser(path, source_text=source_text)
        path.write_text(GENERIC)
        os.utime(path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
        return parsed

    monkeypatch.setattr(Q, "parse_mlir_file", transient)
    assert C._out_specs(path) == [([2], "f32")]
    monkeypatch.setattr(Q, "parse_mlir_file", parser)
    assert C._out_specs(path) == [([2], "f32")]
    path.write_text(GENERIC.replace("2xf32", "3xf32"))
    assert C._out_specs(path) == [([3], "f32")]


def test_mutation_left_after_parse_is_refused_not_cached_as_current(tmp_path, monkeypatch):
    path, parser = source(tmp_path), Q.parse_mlir_file

    def changed(path, *, source_text):
        parsed = parser(path, source_text=source_text)
        path.write_text(GENERIC.replace("2xf32", "3xf32"))
        return parsed

    monkeypatch.setattr(Q, "parse_mlir_file", changed)
    with pytest.raises(ValueError, match="changed during signature"):
        C._out_specs(path)
    assert not Q._SIGNATURE_CACHE


def test_independent_parses_do_not_hold_the_signature_cache_lock(tmp_path, monkeypatch):
    first, second = source(tmp_path, "first.mlir"), source(tmp_path, "second.mlir")
    parser, barrier = Q.parse_mlir_file, threading.Barrier(2)

    def parallel(path, **kwargs):
        barrier.wait(timeout=3)
        return parser(path, **kwargs)

    monkeypatch.setattr(Q, "parse_mlir_file", parallel)
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert list(pool.map(C._out_specs, (first, second))) == [[([2], "f32")], [([2], "f32")]]


def test_caches_stay_bounded(tmp_path, monkeypatch):
    monkeypatch.setattr(Q, "_SIGNATURE_CACHE_MAX", 2)
    monkeypatch.setattr(Q, "_PARSE_CACHE_MAX", 2)
    for index in range(5):
        assert C._out_specs(source(tmp_path, f"model-{index}.mlir")) == [([2], "f32")]
    assert len(Q._SIGNATURE_CACHE) == len(Q._PARSE_CACHE) == 2


@pytest.mark.parametrize("snapshot", [False, True])
def test_custom_form_reprint_uses_original_file_or_exact_snapshot(tmp_path, monkeypatch, snapshot):
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    path = tmp_path / "custom.mlir"
    original = "not directly parseable MLIR snapshot A"
    path.write_text("different mutable file B" if snapshot else original)
    seen = []

    def reprint(actual):
        actual = Path(actual)
        seen.append(actual)
        assert actual.read_text() == original
        assert (actual != path) is snapshot
        generic = actual.with_suffix(".generic.mlir")
        generic.write_text(GENERIC)
        return generic

    monkeypatch.setattr(G, "to_generic_form", reprint)
    module = L.parse_mlir_file(path, **({"source_text": original} if snapshot else {}))
    assert Q.forward_signature(module)[1] == [([2], "f32")]
    assert len(seen) == 1
    if snapshot:
        assert not seen[0].exists()
