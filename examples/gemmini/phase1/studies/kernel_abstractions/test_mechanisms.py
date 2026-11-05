"""Offline study checks; no retained private input or simulator is required."""

import importlib.util
import json
import sys
from dataclasses import replace

import pytest

from merlin.common.paths import repo_root


def module(name):
    path = repo_root() / "examples/gemmini/phase1/studies/kernel_abstractions" / (name + ".py")
    spec = importlib.util.spec_from_file_location("study_" + name, path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def test_declared_pins_fail_on_changed_or_missing_bytes_and_preserve_closure(tmp_path):
    api = module("mechanisms")
    artifact = tmp_path / "artifact"
    artifact.write_text("original")
    document = {"rows": [{"id": "qk", "source_closure": "original source missing",
                          "receipt": api.pin(artifact)}], "observed_pass_compositions": []}
    study = tmp_path / "study.json"
    study.write_text(json.dumps(document))
    assert api.audit_oracle_study(study)["status"] == "pass"
    artifact.write_text("changed")
    audit = api.audit_oracle_study(study)
    assert audit["status"] == "fail" and audit["declared_pin_count"] == 1
    assert api.main(["audit", str(study)]) == 1
    assert audit["source_closure"] == [{"id": "qk", "closure": "original source missing"}]
    artifact.unlink()
    assert api.audit_oracle_study(study)["pins"][0]["actual"] is None


def test_source_use_def_and_epilogue_are_admitted_before_geometry():
    api = module("mechanisms")
    cb = {"tensors": {"a": {"shape": [3, 4], "dtype": "i8"},
                      "w": {"shape": [4, 5], "dtype": "i8"},
                      "y": {"shape": [3, 5], "dtype": "i32"}}, "commands": [
        {"opcode": "RES_PACK", "operands": {"src": "w", "dst": "r"}},
        {"opcode": "MATMUL_RESIDENT", "operands": {"lhs": "a", "rhs": "r", "dst": "acc"}},
        {"opcode": "COMMIT", "operands": {"src": "acc", "dst": "y"},
         "attributes": {"epilogue": [], "output_dtype": "i32"}},
        {"opcode": "EVICT", "operands": {"handle": "r"}}]}
    assert api._geometry(cb) == ((3, 4, 5), ("a", "w", "y"))
    cb["commands"][2]["attributes"]["epilogue"] = ["relu"]
    with pytest.raises(ValueError, match="numerical"):
        api._geometry(cb)
    cb["commands"][2]["attributes"]["epilogue"] = []
    cb["commands"][1]["operands"]["rhs"] = "another_handle"
    with pytest.raises(ValueError, match="use-def"):
        api._geometry(cb)


def test_neighbor_domains_derive_tails_and_the_owned_fit_boundary():
    api = module("mechanisms")
    rows = api.neighbor_shapes((5, 8, 13), tile=3, rhs_rows=42, max_m_blocks=4)
    by_id = {row["id"]: row["shape_mkn"] for row in rows}
    assert by_id["single_row"] == (1, 8, 13)
    assert by_id["row_tail"] == (4, 8, 13)
    assert by_id["operand_upper_tails"][0] == 13
    for key, fits in (("rhs_capacity_fit", True), ("rhs_capacity_miss", False)):
        _m, k, n = by_id[key]
        needed = ((k + 2) // 3) * ((n + 2) // 3) * 3
        assert (needed <= 42) is fits
    with pytest.raises(ValueError, match="two RHS"):
        api.neighbor_shapes((5, 8, 13), tile=3, rhs_rows=3, max_m_blocks=4)


def test_reshaping_keeps_the_complete_source_graph_and_one_output_definition():
    from merlin.targetgen.contract.interface_emit import parse_interface_mlir

    api = module("mechanisms")
    cb = {"target": "selected", "abi_version": "0.1", "tensors": {
        "a": {"shape": [3, 4], "dtype": "i8", "role": "input"},
        "w": {"shape": [4, 5], "dtype": "i8", "role": "weight"},
        "y": {"shape": [3, 5], "dtype": "i32", "role": "output"}}, "commands": [
        {"opcode": "RES_PACK", "operands": {"src": "w", "dst": "r"}},
        {"opcode": "MATMUL_RESIDENT", "operands": {"lhs": "a", "rhs": "r", "dst": "acc"}},
        {"opcode": "COMMIT", "operands": {"src": "acc", "dst": "y"},
         "attributes": {"epilogue": [], "output_dtype": "i32"}},
        {"opcode": "EVICT", "operands": {"handle": "r"}}]}
    text = api.reshape_source(cb, (7, 11, 13))
    actual = parse_interface_mlir(text)
    assert actual["target"] == cb["target"]
    assert actual["tensors"]["a"]["shape"] == [7, 11]
    assert actual["tensors"]["w"]["shape"] == [11, 13]
    assert [row["opcode"] for row in actual["commands"]] == [row["opcode"] for row in cb["commands"]]
    assert text.count("%y =") == 1 and "tensor<7x13xi32>" in text
    assert cb["tensors"]["a"]["shape"] == [3, 4]


def test_llvm_comparison_reports_changed_payload_and_header(tmp_path):
    api = module("mechanisms")
    left, right = tmp_path / "actual.mlir", tmp_path / "expected.mlir"
    original = '''builtin.module { llvm.func @f() {
      %x = llvm.mlir.constant(3 : i64) : i64
      llvm.return
    }}'''
    left.write_text(original)
    right.write_text(original)
    assert api.compare_llvm(left, right)["modules_structurally_equal"]
    right.write_text(original.replace("constant(3", "constant(4"))
    comparison = api.compare_llvm(left, right)
    assert not comparison["modules_structurally_equal"]
    assert not comparison["functions"][0]["body_structurally_equal"]


def test_actual_plan_consumers_emit_and_reject_invalid_contexts(tmp_path):
    result = module("schedule_checks").run_schedule_checks(tmp_path / "schedules")
    assert {row["family"] for row in result["rows"]} == {
        "pointwise", "projection", "shared_rhs_i32", "shared_rhs_i8"}
    for row in result["rows"]:
        assert row["roundtrip_equal"] and row["numeric"] == "pass"
        assert row["llvm"]["bytes"] > 0
        assert all(check["status"] == "rejected" for check in row["counterexamples"].values())
        if isinstance(row["source"], dict):
            assert row["source_identity"] == row["source"]["sha256"]
    assert result["spatial"]["status"] == "representation_gap"


def test_independent_executor_catches_removed_original_liveout():
    api = module("mechanisms")
    sys.path.insert(0, str(repo_root() / "merlin/tests/infra"))
    import test_compact_dense as neutral

    from merlin.sched.ir import Loop

    p, r, s = neutral.fixture(shape=(3, 4, 5), tile=(2, 2, 2))
    result = neutral.schedule(p, r, s)
    def remove(body):
        return tuple(replace(item, body=remove(item.body)) if type(item) is Loop else item
                     for item in body if type(item) is Loop or item.instr != "store")
    with pytest.raises(ValueError, match="live-out"):
        api.execute_dense(replace(result.kernel, body=remove(result.kernel.body)), neutral.ADAPTER,
                          (p.m, p.k, p.n), edge=2, operand_rows=4096, accumulator_rows=4096)
