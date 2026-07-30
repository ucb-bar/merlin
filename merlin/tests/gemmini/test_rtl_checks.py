"""Tests for the RTL-derived checks layer (task #131).

Predicate-based, synthetic traces/capsules ONLY — no per-capsule golden fixtures (honours the
abstract-into-compiler-not-overfit principle). We assert the INVARIANT ("illegal funct ⇒ fail",
"over-commit ⇒ tile-coverage fail"), never "capsule X must have N MVOUTs".
"""
from __future__ import annotations

import copy

import pytest

from merlin.targetgen import rtl_check_compiler as CC
from merlin.targetgen import rtl_check_runner as RUN
from merlin.targetgen import rtl_checks as RC
from merlin.targetgen.rtl import circt_introspect as CI
from merlin.targetgen.rtl.facts import load_facts


# --------------------------------------------------------------------------------- synthetic fixtures
def _matmul_capsule(M=16, K=16, N=16, dtype="i32"):
    return {"name": "synthetic_matmul", "operation": {"op": "matmul",
            "attributes": {"output_dtype": dtype}},
            "inputs": [{"role": "weight", "shape": [K, N]}, {"role": "input", "shape": [M, K]}]}


def _trace(classes_functs):
    """classes_functs: list of (class, funct|None) -> a minimal decoded-trace dict."""
    return {"source": "synthetic", "abi": {"custom_opcode": "0x7b", "funct3": "0x3"},
            "instructions": [{"index": i, "class": c, "funct": f, "decoded": {}}
                             for i, (c, f) in enumerate(classes_functs)]}


def _good_single_tile_trace():
    # a valid WS single-tile sequence: configs precede use, PRELOAD precedes COMPUTE, one MVOUT tile.
    return _trace([("FENCE", None), ("CONFIG_EX", 0), ("CONFIG_LD", 0), ("MVIN", 2), ("MVIN", 2),
                   ("CONFIG_ST", 0), ("PRELOAD", 6), ("COMPUTE_PRELOADED", 4), ("MVOUT", 3),
                   ("FENCE", None)])


# Source facts from the regenerating accessor (the CIRCT-generated artifact) rather than a degenerate
# empty-interfaces fallback: the legacy SoC HW-dialect cache is optional, but load_facts always yields a
# real funct_decode_table (mlc extraction, or the header fallback), so the decode-table checks stay
# runnable. The per-target cache path is resolved from the target name (no baked const to import).
_HW_CACHE = CI._soc_hw_path("gemmini")
FACTS = CI.build_facts(target="gemmini") if _HW_CACHE.is_file() else load_facts("gemmini")


# ----------------------------------------------------------------------------- circt_introspect facts
@pytest.mark.skipif(not _HW_CACHE.is_file(), reason="cached HW MLIR not present")
def test_circt_facts_reproduce_contract_and_decode_table():
    import yaml
    from merlin.targetgen import rocc_decode
    rec = CI.build_facts(target="gemmini")
    contract = yaml.safe_load((CI._REPO / "merlin/targets/gemmini/contracts/target_contract.yaml")
                              .read_text())
    res = CI.validate(rec, contract, rocc_decode._FUNCT_CLASS)
    assert not res["diverge"], f"RTL facts diverge from curated sources: {res['diverge']}"
    # accumulator depth/bytes were extracted from the HW dialect (the v1 grep gap)
    acc = next(m for m in rec["facts"]["memories"] if m["name"] == "accumulator")
    assert acc["bytes"] and acc["banks"] >= 1 and acc["addr_width"] > 0
    # legal funct set is the GemminiISA block and is a superset of what rocc_decode classifies
    legal = set(next(i for i in rec["facts"]["interfaces"]
                     if i["name"] == "funct_decode_table")["legal_funct"])
    assert set(rocc_decode._FUNCT_CLASS) <= legal


# ------------------------------------------------------------------------------- Python screen() checks
def test_screen_passes_good_single_tile():
    rep = RC.screen(_good_single_tile_trace(), _matmul_capsule())
    assert rep.verdict == "ok", [c.to_dict() for c in rep.checks if c.status == "fail"]


def test_screen_catches_illegal_funct():
    t = _good_single_tile_trace()
    t["instructions"].append({"index": 99, "class": "UNKNOWN", "funct": 99, "decoded": {}})
    rep = RC.screen(t, _matmul_capsule())
    fails = {c.id for c in rep.checks if c.status == "fail"}
    assert "T0.decode_funct_legal" in fails and rep.verdict == "reject"


def test_screen_catches_over_commit_tiles():
    t = _good_single_tile_trace()
    t["instructions"].append({"index": 50, "class": "MVOUT", "funct": 3, "decoded": {}})  # 2 != 1 tile
    rep = RC.screen(t, _matmul_capsule(16, 16, 16))
    assert "T0.tile_coverage" in {c.id for c in rep.checks if c.status == "fail"}


def test_screen_catches_over_capacity_spad():
    t = _good_single_tile_trace()
    t["instructions"].append({"index": 51, "class": "MVIN", "funct": 2,
                              "decoded": {"spad_addr": 10_000_000}})  # >> scratchpad rows
    rep = RC.screen(t, _matmul_capsule())
    assert "T0.spad_capacity" in {c.id for c in rep.checks if c.status == "fail"}


def test_screen_catches_compute_before_preload():
    t = _trace([("CONFIG_EX", 0), ("CONFIG_LD", 0), ("MVIN", 2), ("CONFIG_ST", 0),
                ("COMPUTE_PRELOADED", 4), ("PRELOAD", 6), ("MVOUT", 3)])  # compute precedes preload
    rep = RC.screen(t, _matmul_capsule())
    assert "T0.preload_before_compute" in {c.id for c in rep.checks if c.status == "fail"}


def test_screen_catches_use_before_config():
    t = _trace([("CONFIG_EX", 0), ("MVIN", 2), ("PRELOAD", 6),
                ("COMPUTE_PRELOADED", 4), ("MVOUT", 3)])  # MVOUT with no preceding CONFIG_ST
    rep = RC.screen(t, _matmul_capsule())
    assert "T0.config_before_use" in {c.id for c in rep.checks if c.status == "fail"}


# --------------------------------------------------------------------------------- FileCheck compiler
@pytest.mark.skipif(RUN.find_filecheck() is None, reason="FileCheck binary not found")
def test_filecheck_trace_passes_good_and_catches_corruptions():
    fc = RUN.find_filecheck()
    cap = _matmul_capsule(16, 16, 16)
    cc = CC.compile_checks(FACTS, cap, "gemmini")

    def trace_ok(t):
        ok, _ = RUN.run_filecheck(fc, cc["trace"], RUN.render_trace(t, FACTS), "TRACE")
        return ok

    good = _good_single_tile_trace()
    assert trace_ok(good)                              # no false reject

    over = copy.deepcopy(good)
    over["instructions"].append({"index": 50, "class": "MVOUT", "funct": 3, "decoded": {}})
    assert not trace_ok(over)                          # over-commit caught (MVOUT_COUNT 2 != 1)

    illegal = copy.deepcopy(good)
    illegal["instructions"].append({"index": 99, "class": "UNKNOWN", "funct": 99, "decoded": {}})
    assert not trace_ok(illegal)                       # illegal funct caught

    nocompute = _trace([("MVIN", 2), ("MVOUT", 3)])
    assert not trace_ok(nocompute)                     # missing COMPUTE caught


def test_filecheck_exact_count_not_substring():
    """Regression: MVOUT_COUNT 1 must NOT substring-match MVOUT_COUNT 16 (needs the {{$}} anchor)."""
    fc = RUN.find_filecheck()
    if fc is None:
        pytest.skip("FileCheck not found")
    cc = CC.compile_checks(FACTS, _matmul_capsule(16, 16, 16), "gemmini")  # expects MVOUT_COUNT 1
    # 16 MVOUTs -> rendered "MVOUT_COUNT 16"; must fail the "MVOUT_COUNT 1" check
    t = _trace([("PRELOAD", 6), ("COMPUTE_PRELOADED", 4)] + [("MVOUT", 3)] * 16)
    ok, _ = RUN.run_filecheck(fc, cc["trace"], RUN.render_trace(t, FACTS), "TRACE")
    assert not ok
