"""Tiny trusted native tests of host-supplied address maps, never candidate execution."""
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from merlin.perf import native_host_witness_runner as runner


def _request():
    return {"library": "not-loaded-before-validation", "symbol": "witness",
            "limits": {"max_total_storage_bytes": 32768, "max_total_logical_elements": 6},
            "arguments": [
                {"dtype": "i32", "access": "read", "storage_elements": 6,
                 "logical_address_map": {"schema": "logical_address_map_v1",
                                         "offsets_elements": [5, 3, 1]}, "values": [10, 20, 30]},
                {"dtype": "i32", "access": "write", "storage_elements": 5,
                 "logical_address_map": {"schema": "logical_address_map_v1",
                                         "offsets_elements": [2, 0, 4]}}]}


@pytest.fixture
def trusted_library(tmp_path):
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("trusted system C compiler unavailable")
    source = tmp_path / "trusted_witness.c"
    source.write_text("""
        #include <stdint.h>
        void witness(int32_t *x, int32_t *y) {
          static int calls = 0;
          ++calls;
          y[2] = x[5] + 1; y[0] = x[3] + 2; y[4] = x[1] + 3 + calls;
        }
        void legacy(int32_t *x, int32_t *y) {
          for (int i=0; i<3; ++i) y[i] = x[i] + 1;
        }
        void corrupt(int32_t *x, int32_t *y) { ++x[5]; y[2] = 0; }
    """)
    library = tmp_path / "trusted_witness.so"
    subprocess.run([compiler, "-shared", "-fPIC", "-O0", str(source), "-o", str(library)],
                   check=True, capture_output=True, timeout=15)
    return library


@pytest.mark.parametrize("strided", [False, True])
def test_actual_tiny_native_mapped_function(tmp_path, capsys, trusted_library, strided):
    request = _request()
    request["library"] = str(trusted_library)
    if strided:
        request["arguments"][0]["logical_address_map"] = {
            "schema": "logical_address_map_v1", "logical_shape": [3],
            "strides_elements": [-2], "offset_elements": 5}
    path = tmp_path / "request.json"
    path.write_text(json.dumps(request))
    runner.main(path)
    observed = json.loads(capsys.readouterr().out)
    assert observed == {"outputs": [[11, 22, 35]], "warmup_calls": 1,
                        "observed_calls": 1, "timing_measured": False}


def test_legacy_layout_behavior_unchanged(tmp_path, capsys, trusted_library):
    request = {"library": str(trusted_library), "symbol": "legacy", "arguments": [
        {"dtype": "i32", "access": "read", "storage_elements": 3,
         "rows": 1, "cols": 3, "row_stride": 3, "values": [7, 8, 9]},
        {"dtype": "i32", "access": "write", "storage_elements": 3,
         "rows": 1, "cols": 3, "row_stride": 3}]}
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(request))
    runner.main(path)
    assert json.loads(capsys.readouterr().out)["outputs"] == [[8, 9, 10]]


def test_read_only_storage_still_checked(tmp_path, trusted_library):
    request = _request()
    request.update(library=str(trusted_library), symbol="corrupt")
    path = tmp_path / "corrupt.json"
    path.write_text(json.dumps(request))
    with pytest.raises(ValueError, match="read-only"):
        runner.main(path)


@pytest.mark.parametrize("mutation", [
    lambda r: r.pop("limits"),
    lambda r: r["limits"].update(max_total_storage_bytes=1),
    lambda r: r["limits"].update(max_total_logical_elements=5),
    lambda r: r["arguments"][0].update(storage_elements=10**12),
    lambda r: r["arguments"][0].update(values=[1]),
    lambda r: r["arguments"][0].update(values=[1, 2, 2**40]),
    lambda r: r["arguments"][0].update(dtype="unknown"),
    lambda r: r["arguments"][0].update(access="readwrite"),
    lambda r: r["arguments"][0].pop("values"),
    lambda r: r["arguments"][0].update(logical_address_map=None),
    lambda r: r["arguments"][0]["logical_address_map"].update(offsets_elements=[1, 1, 3]),
    lambda r: r["arguments"][0]["logical_address_map"].update(offsets_elements=[1, 3, 6]),
    lambda r: r["arguments"][0]["logical_address_map"].update(offsets_elements=[-1, 3, 5]),
    lambda r: r["arguments"][0]["logical_address_map"].update(offsets_elements=[True, 3, 5]),
    lambda r: r["arguments"][0]["logical_address_map"].update(logical_shape=[3]),
])
def test_bad_maps_refused_before_library_or_buffer_allocation(tmp_path, monkeypatch, mutation):
    request = _request()
    mutation(request)
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(request))
    def forbidden(*args, **kwargs):
        pytest.fail("native library or buffer allocation happened before validation")
    monkeypatch.setattr(runner.ctypes, "CDLL", forbidden)
    monkeypatch.setattr(runner.ctypes, "create_string_buffer", forbidden)
    with pytest.raises(ValueError):
        runner.main(path)


@pytest.mark.parametrize("mapping", [
    {"logical_shape": [2, 2], "strides_elements": [1, 1], "offset_elements": 0},
    {"logical_shape": [3], "strides_elements": [4], "offset_elements": 0},
    {"logical_shape": [10**12], "strides_elements": [1], "offset_elements": 0},
    {"logical_shape": [0], "strides_elements": [1], "offset_elements": 0},
])
def test_strided_invalid_domain_or_alias_refused(mapping):
    request = _request()
    request["arguments"][0]["logical_address_map"] = {"schema": "logical_address_map_v1", **mapping}
    with pytest.raises(ValueError):
        runner._validate_arguments(request, 4096)


def test_rank_zero_logical_map_has_one_value():
    request = _request()
    request["arguments"] = [request["arguments"][0]]
    request["arguments"][0].update(values=[7], logical_address_map={
        "schema": "logical_address_map_v1", "logical_shape": [],
        "strides_elements": [], "offset_elements": 2})
    assert runner._validate_arguments(request, 4096)[0][2] == (2,)
