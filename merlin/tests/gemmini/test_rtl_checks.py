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


def _pooled_matmul_capsule():
    c = _matmul_capsule(25, 16, 17, "i8")
    c["inputs"][0]["name"] = "W"
    c["inputs"][1]["name"] = "A0"
    c["numeric_policy"] = {"dtype": "i8", "compare": "exact_int"}
    c["operation"]["attributes"].update({
        "lhs": "A0", "weight": "W", "out": "Y0", "epilogue": ["maxpool"],
        "pool_in_dims": [5, 5], "pool_size": [2, 2], "pool_stride": [2, 2],
        "pool_padding": [0, 0, 0, 0],
    })
    return c


def _pooled_store_trace(*, pocols=2, omit_field=None):
    config = {
        "out_stride_bytes": 32, "pool_stride": 2, "pool_size": 2, "pool_out_dim": 2,
        "porows": 2, "pocols": pocols, "orows": 5, "ocols": 5, "upad": 0, "lpad": 0,
    }
    if omit_field:
        config.pop(omit_field)
    t = _trace([("CONFIG_ST", 0), ("MVOUT", 3), ("MVOUT", 3)])
    t["instructions"][0]["decoded"] = config
    t["instructions"][1]["decoded"] = {
        "dram": {"kind": "argbase", "arg_index": 2, "offset": 0},
        "rows": 0, "cols": 16,
    }
    t["instructions"][2]["decoded"] = {
        "dram": {"kind": "argbase", "arg_index": 2, "offset": 16},
        "rows": 0, "cols": 1,
    }
    return t


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
    from merlin.targetgen.rocc import decode as rocc_decode
    rec = CI.build_facts(target="gemmini")
    contract = yaml.safe_load((CI._REPO / "merlin/targets/gemmini/contracts/target_contract.yaml")
                              .read_text())
    res = CI.validate(rec, contract, rocc_decode.funct_class_for("gemmini"))
    assert not res["diverge"], f"RTL facts diverge from curated sources: {res['diverge']}"
    # accumulator depth/bytes were extracted from the HW dialect (the v1 grep gap)
    acc = next(m for m in rec["facts"]["memories"] if m["name"] == "accumulator")
    assert acc["bytes"] and acc["banks"] >= 1 and acc["addr_width"] > 0
    # legal funct set is the GemminiISA block and is a superset of what rocc_decode classifies
    legal = set(next(i for i in rec["facts"]["interfaces"]
                     if i["name"] == "funct_decode_table")["legal_funct"])
    assert set(rocc_decode.funct_class_for("gemmini")) <= legal


# ------------------------------------------------------------------------------- Python screen() checks
def test_screen_passes_good_single_tile():
    rep = RC.screen(_good_single_tile_trace(), _matmul_capsule(), target="gemmini")
    assert rep.verdict == "ok", [c.to_dict() for c in rep.checks if c.status == "fail"]


def test_screen_catches_illegal_funct():
    t = _good_single_tile_trace()
    t["instructions"].append({"index": 99, "class": "UNKNOWN", "funct": 99, "decoded": {}})
    rep = RC.screen(t, _matmul_capsule(), target="gemmini")
    fails = {c.id for c in rep.checks if c.status == "fail"}
    assert "T0.decode_funct_legal" in fails and rep.verdict == "reject"


def test_screen_catches_over_commit_tiles():
    t = _good_single_tile_trace()
    t["instructions"].append({"index": 50, "class": "MVOUT", "funct": 3, "decoded": {}})  # 2 != 1 tile
    rep = RC.screen(t, _matmul_capsule(16, 16, 16), target="gemmini")
    assert "T0.tile_coverage" in {c.id for c in rep.checks if c.status == "fail"}


def test_screen_catches_over_capacity_spad():
    t = _good_single_tile_trace()
    spad_rows = RC.load_default_facts("gemmini")["scratchpad_rows"]
    t["instructions"].append({"index": 51, "class": "MVIN", "funct": 2,
                              "decoded": {"spad_addr": spad_rows - 8, "rows": 16}})
    rep = RC.screen(t, _matmul_capsule(), target="gemmini")
    assert "T0.spad_capacity" in {c.id for c in rep.checks if c.status == "fail"}


def test_screen_decodes_mvin_accumulator_space_before_capacity_check():
    """MVIN may target the accumulator: its high address bit is a space tag, not a row bit."""
    t = _good_single_tile_trace()
    t["instructions"][3]["decoded"] = {"spad_addr": 0x80000000, "rows": 16}
    t["instructions"][4]["decoded"] = {"spad_addr": 0x80000010, "rows": 16}
    rep = RC.screen(t, _matmul_capsule(), target="gemmini")
    capacity = next(c for c in rep.checks if c.id == "T0.spad_capacity")
    assert capacity.status == "pass", capacity.to_dict()
    assert capacity.evidence["accumulator_max_row"] == 16
    assert capacity.evidence["accumulator_row_mask"] == (
        RC.load_default_facts("gemmini")["accumulator_rows"] - 1
    )


def test_screen_catches_over_capacity_accumulator_mvin_after_decoding_space():
    t = _good_single_tile_trace()
    acc_rows = RC.load_default_facts("gemmini")["accumulator_rows"]
    t["instructions"][3]["decoded"] = {
        "spad_addr": 0x80000000 | (acc_rows - 8), "rows": 16,
    }
    rep = RC.screen(t, _matmul_capsule(), target="gemmini")
    capacity = next(c for c in rep.checks if c.id == "T0.spad_capacity")
    assert capacity.status == "fail"
    assert capacity.evidence["accumulator_max_row"] == acc_rows - 8
    assert capacity.evidence["accumulator_max_row_exclusive"] == acc_rows + 8
    assert capacity.evidence["scratchpad_max_row"] is None


def test_screen_rejects_noncanonical_accumulator_payload_bits():
    """Bits carried by LocalAddr.data but ignored by full_acc_addr silently alias accumulator rows."""
    facts = RC.load_default_facts("gemmini")
    acc_mask = facts["accumulator_row_mask"]
    data_mask = facts["local_address_data_mask"]
    assert data_mask > acc_mask
    t = _good_single_tile_trace()
    t["instructions"][3]["decoded"] = {
        "spad_addr": facts["accumulator_select_bit"] | (acc_mask + 1), "rows": 1,
    }
    rep = RC.screen(t, _matmul_capsule(), target="gemmini")
    capacity = next(c for c in rep.checks if c.id == "T0.spad_capacity")
    assert capacity.status == "fail", capacity.to_dict()
    assert capacity.evidence["noncanonical_accumulator_instruction_indices"] == [3]


def test_capacity_unknown_accumulator_selector_does_not_misclassify_tagged_address_as_spad():
    facts = {
        "mesh": [8, 8], "scratchpad_bytes": 512, "scratchpad_rows": 64,
        "scratchpad_row_mask": 0x3F, "accumulator_rows": 8,
        "accumulator_row_mask": 0x7, "local_address_data_mask": 0x3F,
        "accumulator_select_bit": None,
    }
    trace = _trace([("MVIN", 2)])
    trace["instructions"][0]["decoded"] = {"spad_addr": 0x100, "rows": 1}
    check = RC._check_spad_capacity(trace, facts)
    assert check.status == "skipped", check.to_dict()
    assert check.evidence["unresolved_address_space_instruction_indices"] == [0]


@pytest.mark.parametrize("rows", [0, -1])
def test_capacity_rejects_nonpositive_mvin_row_count(rows):
    facts = RC.load_default_facts("gemmini")
    trace = _trace([("MVIN", 2)])
    trace["instructions"][0]["decoded"] = {"spad_addr": 0, "rows": rows}
    check = RC._check_spad_capacity(trace, facts)
    assert check.status == "fail", check.to_dict()
    assert check.evidence["invalid_row_count_instruction_indices"] == [0]


def test_capacity_skips_undecodable_mvin_row_count():
    facts = RC.load_default_facts("gemmini")
    trace = _trace([("MVIN", 2)])
    trace["instructions"][0]["decoded"] = {"spad_addr": 0}
    check = RC._check_spad_capacity(trace, facts)
    assert check.status == "skipped", check.to_dict()
    assert check.evidence["unresolved_row_count_instruction_indices"] == [0]


def test_capacity_space_decode_uses_derived_selector_not_a_bit31_literal():
    facts = {
        "mesh": [8, 8], "scratchpad_bytes": 512, "scratchpad_rows": 64,
        "accumulator_rows": 8, "accumulator_select_bit": 0x100,
        "accumulator_row_mask": 0x7,
    }
    trace = _trace([("MVIN", 2)])
    # Bits outside the low three row bits model additional LocalAddr metadata; the checker must use
    # the derived row mask, not merely clear the selector/control bits it happens to know about.
    trace["instructions"][0]["decoded"] = {"spad_addr": 0x1C0 | 7, "rows": 1}
    check = RC._check_spad_capacity(trace, facts)
    assert check.status == "pass"
    assert check.evidence["accumulator_max_row"] == 7


def test_screen_applies_operand_load_checks_to_mvin2():
    t = _good_single_tile_trace()
    spad_rows = RC.load_default_facts("gemmini")["scratchpad_rows"]
    t["instructions"][3] = {"index": 3, "class": "MVIN2", "funct": 1,
                             "decoded": {"spad_addr": spad_rows - 8, "rows": 16}}
    rep = RC.screen(t, _matmul_capsule(), target="gemmini")
    assert "T0.spad_capacity" in {c.id for c in rep.checks if c.status == "fail"}


def test_pooled_store_coverage_uses_config_geometry_and_channel_tail():
    capsule = _pooled_matmul_capsule()
    trace = _pooled_store_trace()
    pool_facts = {"mesh": [16, 16], "config_mvout_fields": list(RC._POOL_CONFIG_FIELDS),
                  "max_pool_supported": True}
    pool = RC._check_pool_config(trace, capsule, pool_facts)
    assert pool.status == "pass", pool.to_dict()
    outputs, why = RC.declared_outputs(capsule)
    assert outputs, why
    coverage = RC._store_coverage(trace, outputs, capsule, pool_facts)["Y0"]
    assert coverage["status"] == "covered", coverage
    assert coverage["covered_cells"] == coverage["declared_cells"] == 68
    assert RC.expected_mvout_count(capsule, pool_facts)[0] == 2
    assert RC.expected_mvout_count(_matmul_capsule(25, 16, 17), {"mesh": [16, 16]})[0] == 4


@pytest.mark.parametrize("trace", [
    pytest.param(_pooled_store_trace(pocols=3), id="geometry-mismatch"),
    pytest.param(_pooled_store_trace(omit_field="porows"), id="underived-field"),
])
def test_pooled_store_config_mismatch_never_manufactures_coverage(trace):
    capsule = _pooled_matmul_capsule()
    facts = {"max_pool_supported": True}
    pool = RC._check_pool_config(trace, capsule, facts)
    assert pool.status == "fail", pool.to_dict()
    outputs, _ = RC.declared_outputs(capsule)
    coverage = RC._store_coverage(trace, outputs, capsule, facts)["Y0"]
    assert coverage["status"] == "unknown", coverage
    assert "CONFIG_ST" in coverage["unknown_reason"]


@pytest.mark.parametrize(("supported", "status"), [(False, "fail"), (None, "skipped")])
def test_pool_capability_false_or_unknown_never_manufactures_coverage(supported, status):
    capsule, trace = _pooled_matmul_capsule(), _pooled_store_trace()
    facts = {"mesh": [16, 16], "config_mvout_fields": list(RC._POOL_CONFIG_FIELDS),
             "max_pool_supported": supported}
    pool = RC._check_pool_config(trace, capsule, facts)
    assert pool.status == status, pool.to_dict()
    outputs, _ = RC.declared_outputs(capsule)
    coverage = RC._store_coverage(trace, outputs, capsule, facts)["Y0"]
    assert coverage["status"] == "unknown", coverage
    assert RC.expected_mvout_count(capsule, facts) is None


def test_filecheck_pooled_and_plain_matmul_share_store_count_derivation():
    pooled = CC.compile_trace_checks(FACTS, _pooled_matmul_capsule())
    plain = CC.compile_trace_checks(FACTS, _matmul_capsule(25, 16, 17))
    assert "MVOUT_COUNT 2{{$}}" in pooled
    assert "MVOUT_COUNT 4{{$}}" in plain
    without_layout = copy.deepcopy(FACTS)
    body = without_layout.get("facts", without_layout)
    body["interfaces"] = [i for i in body.get("interfaces", [])
                          if i.get("name") != "register_bundle_layouts"]
    assert "MVOUT_COUNT" not in CC.compile_trace_checks(without_layout, _pooled_matmul_capsule())


def test_filecheck_render_counts_all_mvin_load_states():
    t = _trace([("MVIN", 2), ("MVIN2", 1), ("MVIN3", 14)])
    rendered = RUN.render_trace(t, FACTS)
    assert "MVIN_COUNT 3" in rendered
    assert "MVIN_PRESENT yes" in rendered


def test_screen_catches_compute_before_preload():
    t = _trace([("CONFIG_EX", 0), ("CONFIG_LD", 0), ("MVIN", 2), ("CONFIG_ST", 0),
                ("COMPUTE_PRELOADED", 4), ("PRELOAD", 6), ("MVOUT", 3)])  # compute precedes preload
    rep = RC.screen(t, _matmul_capsule(), target="gemmini")
    assert "T0.preload_before_compute" in {c.id for c in rep.checks if c.status == "fail"}


def test_screen_catches_use_before_config():
    t = _trace([("CONFIG_EX", 0), ("MVIN", 2), ("PRELOAD", 6),
                ("COMPUTE_PRELOADED", 4), ("MVOUT", 3)])  # MVOUT with no preceding CONFIG_ST
    rep = RC.screen(t, _matmul_capsule(), target="gemmini")
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
