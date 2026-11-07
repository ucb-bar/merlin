"""The model regression matrix (examples/lib/model_matrix.py): verdicts, hart identity, the table."""

from __future__ import annotations

import importlib.util
import sys

import numpy as np
import pytest

from merlin.common.paths import repo_root
from merlin.runtime.backends import output_identity


@pytest.fixture
def mm():
    lib = repo_root() / "examples" / "lib"
    sys.path.insert(0, str(lib))
    try:
        spec = importlib.util.spec_from_file_location("examples_lib_model_matrix", lib / "model_matrix.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.path.remove(str(lib))


def _ident(values, value):
    return output_identity.output_identity(
        {"prefix": np.asarray(values, dtype=np.float32), "out_hash": {"value": value}}
    )


def _gate(ok=True):
    return {"gate_ok": ok, "w8a8_cos": 0.99999 if ok else 0.9, "w8a8_rel": 0.001, "w8a8_max_rel": 0.01}


def _rvv(harts, ident, ok=True):
    return {
        "model": "m",
        "harts": harts,
        "leg": "rvv",
        "board": "b",
        "compile": {
            "status": "verified" if ok else "run_mismatch",
            "verify": _gate(ok),
            "cycles": 100,
            "output_identity": ident,
        },
    }


def _matrix(harts, twin, shipped, *, on_status="verified", exact=True):
    return {
        "model": "m",
        "harts": harts,
        "leg": "matrix",
        "board": "b",
        "compile": {
            "status": "verified",
            "verify": _gate(),
            "matrix": {
                "routed_contractions": 3,
                "opu_mac_share": 0.9,
                "not_routed_by_reason": {"smaller than one tile": 2},
                "twin_spike_cycles": 500,
                "twin_output_identity": twin,
            },
        },
        "on_unit_model": {
            "status": on_status,
            "verify": _gate(),
            "bit_exact_vs_twin": exact,
            "spike_cycles": 50,
            "output_identity": shipped,
        },
    }


def _provider_matrix():
    """A compiler-owned run needs no runner.spike_extension declaration."""
    identity = _ident([1, 2], 7)
    cell = _matrix(1, identity, identity)
    cell["compile"]["binary"] = "/ship.elf"
    cell["compile"]["matrix"]["shipped_on_simulator"] = {
        "status": "verified",
        "binary": "/provider-simulator",
        "binary_sha256": "0" * 64,
        "provenance": {"pin": "simulator", "ok": True},
        "verify": {**_gate(), "comparison_complete": False, "n_compared": 2, "n_reference": 4},
        "bit_exact_vs_twin": True,
        "spike_cycles": 40,
        "output_identity": identity,
        "accuracy_2percent_certified": False,
        "rtl_certified": False,
    }
    cell["on_unit_model"] = {"status": "skipped", "reason": "no runner.spike_extension"}
    return cell


def test_the_compilers_verified_provider_run_is_the_matrix_verdict(mm):
    import copy
    import json

    cell = _provider_matrix()
    original_compile = copy.deepcopy(cell["compile"])
    original_extra = copy.deepcopy(cell["on_unit_model"])
    checked = mm.summarize(cell)
    assert checked["status"] == "PASS" and checked["cycles"] == 40
    assert checked["shipped"]["identity"] == original_compile["matrix"]["twin_output_identity"]
    assert checked["shipped"]["accuracy_2percent_certified"] is False
    assert checked["shipped"]["rtl_certified"] is False
    assert checked["shipped"]["verify"]["comparison_complete"] is False
    assert checked["shipped"]["verify"]["n_compared"] == 2
    assert checked["shipped"]["verify"]["n_reference"] == 4
    assert cell["compile"] == original_compile and cell["on_unit_model"] == original_extra
    # Re-evaluating a kept cell retains the same actual provider evidence.
    assert mm.summarize(json.loads(json.dumps(checked)))["status"] == "PASS"


@pytest.mark.parametrize(
    "change",
    [
        {"status": "not_run", "reason": "simulator unavailable"},
        {"status": "skipped"},
        {"status": "run_failed"},
        {"status": "ran"},
        {"status": None},
        {"available": False},
        {"verify": {}},
        {"verify": {"gate_ok": False}},
        {"verify": {"gate_ok": "true"}},
        {"bit_exact_vs_twin": False},
        {"bit_exact_vs_twin": None},
        {"output_identity": None},
        {"output_identity": _ident([9], 9)},
    ],
)
def test_an_incomplete_or_failing_provider_run_cannot_pass(mm, change):
    cell = _provider_matrix()
    cell["compile"]["matrix"]["shipped_on_simulator"].update(change)
    # Even a second successful record cannot hide a compiler-owned failure.
    cell["on_unit_model"] = _matrix(1, _ident([1, 2], 7), _ident([1, 2], 7))["on_unit_model"]
    assert mm.summarize(cell)["status"] == "FAIL"


def test_a_verified_flag_without_an_executed_twin_is_not_sufficient(mm):
    cell = _provider_matrix()
    cell["compile"]["matrix"]["twin_output_identity"] = None
    assert mm.summarize(cell)["status"] == "FAIL"
    cell = _provider_matrix()
    cell["compile"]["matrix"]["twin_executed"] = False
    assert mm.summarize(cell)["status"] == "FAIL"


@pytest.mark.parametrize("record", [False, [], "verified"])
def test_a_malformed_compiler_hook_is_not_hidden_by_another_run(mm, record):
    cell = _provider_matrix()
    cell["compile"]["matrix"]["shipped_on_simulator"] = record
    cell["on_unit_model"] = _matrix(1, _ident([1], 1), _ident([1], 1))["on_unit_model"]
    assert mm.summarize(cell)["status"] == "FAIL"


def test_a_compiler_owned_run_is_not_simulated_again(mm, monkeypatch):
    from types import SimpleNamespace

    result = _provider_matrix()["compile"]
    monkeypatch.setattr(mm.chip_flow, "_compile", lambda *a, **kw: result)
    monkeypatch.setattr(mm.chip_flow, "matrix_on_spike", lambda *a, **kw: pytest.fail("reran the shipped image"))
    args = SimpleNamespace(
        board="b", matrix_unit="u", matrix_config="c", matrix_support_target="t", matrix_kernel=None, timeout=1
    )
    assert mm.run_cell(args, "m", 1, "matrix")["status"] == "PASS"


def test_absent_or_undeclared_compiler_hook_keeps_the_existing_extension_route(mm):
    cell = _matrix(1, _ident([1], 1), _ident([1], 1))
    cell["compile"]["matrix"]["shipped_on_simulator"] = {"status": "not_declared"}
    assert mm.summarize(cell)["status"] == "PASS"
    cell.pop("on_unit_model")
    assert mm.summarize(cell)["status"] == "FAIL"


def test_a_cell_passes_only_when_every_gate_it_ran_passed(mm):
    assert mm.summarize(_rvv(1, _ident([1, 2], 7)))["status"] == "PASS"
    failed = mm.summarize(_rvv(1, _ident([1, 2], 7), ok=False))
    assert failed["status"] == "FAIL" and "gate failed" in failed["problems"][0]

    ok = mm.summarize(_matrix(1, _ident([1], 1), _ident([1], 1)))
    assert ok["status"] == "PASS" and ok["cycles"] == 50 and ok["routing"]["opu_mac_share"] == 0.9
    # The twin passing is not enough: the shipped image on the unit's model must agree with it.
    off = mm.summarize(_matrix(1, _ident([1], 1), _ident([2], 2), on_status="run_mismatch", exact=False))
    assert off["status"] == "FAIL" and "bit_exact_vs_twin=False" in off["problems"][0]
    # A shipped-image leg that could not run is a failure to prove, not a pass.
    skipped = mm.summarize(_matrix(1, _ident([1], 1), None, on_status="skipped"))
    assert skipped["status"] == "FAIL"


def test_every_hart_count_must_reproduce_the_smallest_bit_for_bit(mm):
    cells = [mm.summarize(_rvv(1, _ident([1, 2], 7))), mm.summarize(_rvv(2, _ident([1, 2], 7)))]
    mm.harts_identity(cells)
    assert cells[1]["identical_to"] == {"harts": 1, "bit_identical": True} and cells[1]["status"] == "PASS"

    # Same dumped prefix, different whole-tensor hash: a lost work split past the dump cap.
    cells = [mm.summarize(_rvv(1, _ident([1, 2], 7))), mm.summarize(_rvv(2, _ident([1, 2], 8)))]
    mm.harts_identity(cells)
    assert cells[1]["status"] == "FAIL" and "NOT bit-identical" in cells[1]["problems"][-1]

    # Both matrix images are compared: here the twins agree and the shipped images do not.
    cells = [
        mm.summarize(_matrix(1, _ident([1], 1), _ident([1], 1))),
        mm.summarize(_matrix(2, _ident([1], 1), _ident([3], 1))),
    ]
    mm.harts_identity(cells)
    assert cells[1]["identical_to"]["bit_identical"] is False and cells[1]["status"] == "FAIL"


def test_a_kept_cell_reads_back_to_the_same_verdict(mm):
    """Cells are written to disk and summarized again on every later run; summarizing must not overwrite
    the raw record it reads (a first version replaced the shipped image's record with its own summary, so
    every reloaded matrix cell lost its output digest and the hart-identity check went unknown)."""
    import json

    cells = [mm.summarize(_matrix(h, _ident([1], 1), _ident([1], 1))) for h in (1, 2)]
    again = [mm.summarize(json.loads(json.dumps(c))) for c in cells]
    mm.harts_identity(again)
    assert [c["status"] for c in again] == ["PASS", "PASS"]
    assert again[1]["identical_to"]["bit_identical"] is True


def test_identity_that_cannot_be_told_is_not_reported_as_identical(mm):
    cells = [mm.summarize(_rvv(1, {"out_hash": None})), mm.summarize(_rvv(2, {"out_hash": None}))]
    mm.harts_identity(cells)
    assert cells[1]["identical_to"]["bit_identical"] is None and cells[1]["status"] == "FAIL"


def test_a_model_with_nothing_to_route_is_its_own_verdict(mm):
    """Every contraction smaller than one tile: there is no matrix image, and the reason is the result --
    neither a pass nor a crash, and not a hart-identity failure for an image that was never built."""
    reason = "nothing routed to u: 3 contraction(s) declined (3x declined by the selector)"
    cells = [
        mm.summarize(
            {
                "model": "m",
                "harts": h,
                "leg": "matrix",
                "board": "b",
                "compile": {"status": "not_routed", "reason": reason, "matrix": {"not_routed_by_reason": {"x": 3}}},
            }
        )  # fmt: skip
        for h in (1, 2)
    ]
    mm.harts_identity(cells)
    assert [c["status"] for c in cells] == ["NO-ROUTE", "NO-ROUTE"]
    assert cells[1]["problems"] == [f"nothing routed: {reason}"]
    assert "| NO-ROUTE |" in mm.table(cells, "b")


def test_the_table_shows_the_gate_the_share_and_why_the_rest_was_not_routed(mm):
    cells = [mm.summarize(_rvv(1, _ident([1], 1))), mm.summarize(_matrix(1, _ident([1], 1), _ident([1], 1)))]
    text = mm.table(cells, "b")
    row = next(line for line in text.splitlines() if "| matrix |" in line)
    assert "0.9000" in row and "2x smaller than one tile" in row and "| PASS |" in row
    assert "None." in text
