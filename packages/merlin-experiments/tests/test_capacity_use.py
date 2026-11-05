"""capacity_use: storage footprint against RTL-derived capacity and operand re-reads, on recorded kernels.

The fixture holds two command streams recorded in one fresh-compiler run (a straight-line 12x48x48 case
copied to the author's selfcheck_out, and a loop-emitted 512^3 case from a grade run), their public
declarations and the timing observation's device read counter, plus that run's RTL facts trimmed to
geometry, memories and the RoCC decode tables.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from merlin_experiments.phase1.feedback import capacity_use as U
from merlin_experiments.phase1.feedback import case_declarations as D

FIXTURE = json.loads(Path(__file__).with_name("capacity_use_fresh4.json").read_text(encoding="utf-8"))
TARGET = FIXTURE["cases"]["G_matmul_m12k48n48_raw"]["command_buffer"]["target"]


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    facts = tmp_path / "facts.json"
    facts.write_text(json.dumps(FIXTURE["facts"]))
    monkeypatch.setenv("MERLIN_RTL_FACTS", str(facts))
    monkeypatch.setattr(D, "_ROOTS", [])
    monkeypatch.setattr(D, "_TARGET", [])
    monkeypatch.setattr(D, "_CACHE", {})
    monkeypatch.setattr(U, "_CAPACITY", {})
    corpus = tmp_path / "corpus"
    for name, case in FIXTURE["cases"].items():
        folder = tmp_path / "ws" / "selfcheck_out" / name
        folder.mkdir(parents=True)
        (folder / U.TRACE).write_text(json.dumps(case["instruction_trace"]))
        (folder / U.BUFFER).write_text(json.dumps(case["command_buffer"]))
        (corpus / name).mkdir(parents=True)
        (corpus / name / "capsule.yaml").write_text(yaml.safe_dump(case["declaration"]))
    D.register(corpus, target=TARGET)
    return tmp_path / "ws"


def _row(name):
    reads = FIXTURE["cases"][name]["device_requested_read_bytes"]
    return {"capsule": name, "performance": {"feature_counts": {U.DEVICE_READ: reads}}}


def test_capacity_is_derived_from_rtl_facts(workspace):
    cap = U.capacity(TARGET)
    assert cap["edge"] == 16
    assert cap["scratchpad"]["rows"] == 16384 and cap["scratchpad"]["bytes"] == 262144
    assert cap["accumulator"]["rows"] == 1024 and cap["accumulator"]["bytes"] == 65536


def test_straight_line_kernel_reports_rows_addressed_as_a_lower_bound(workspace):
    block = U.for_case("G_matmul_m12k48n48_raw", _row("G_matmul_m12k48n48_raw"), workspace)
    assert block["scratchpad"]["rows"] == 180 and block["scratchpad"]["pct"] == 1.1
    assert block["scratchpad"]["lower_bound"] is True  # some addresses are run-time values
    assert block["accumulator"]["rows"] is None  # no accumulator address resolved: UNKNOWN, never zero
    assert block["operand_reads"]["compulsory_bytes"] == 12 * 48 + 48 * 48
    assert block["operand_reads"]["ratio"] == 1.0 and block["operand_reads"]["capacity_bound_ratio"] == 1.0
    assert block["reading"].startswith("uses at least 1.1% of scratchpad rows (180 of 16384)")


def test_loop_emitted_kernel_reads_operands_far_above_what_capacity_allows(workspace):
    block = U.for_case("ZA_gemm_512x512x512_native_i8", _row("ZA_gemm_512x512x512_native_i8"), workspace)
    assert block["scratchpad"] is None and block["accumulator"] is None  # run-time addresses only
    assert block["operand_reads"]["compulsory_bytes"] == 2 * 512 * 512
    assert block["operand_reads"]["ratio"] == 8.5
    assert block["operand_reads"]["capacity_bound_ratio"] == 1.5
    assert block["reading"] == "reads 8.50x the compulsory operand bytes; capacity allows ~1.50x"


def test_executed_local_activity_of_a_dynamic_census_is_preferred(workspace):
    from merlin.targetgen import rtl_checks_generic as G

    entry = G.protocol_for(TARGET).local_addresses[0]
    klass = sorted(G.protocol_for(TARGET).classes(str(entry["role"])))[0]
    request = {"class": klass, "address_field": entry["address"], "raw_address": 64,
               "rows_field": entry.get("rows"), "raw_rows": 16, "count": 3}
    row = _row("G_matmul_m12k48n48_raw") | {"dynamic_census": {"observations": {
        "local_address_activity": {"status": "observed", "requests": [request]}}}}
    block = U.for_case("G_matmul_m12k48n48_raw", row, workspace)
    assert block["basis"] == "executed (dynamic census)"
    assert block["scratchpad"]["rows"] == 16 and "lower_bound" not in block["scratchpad"]


def test_reuse_bound_follows_capacity():
    capacity = {"operand_elems": 16384 * 16, "accumulator_elems": 1024 * 16, "edge": 16}
    assert U.reuse_bound(512, 512, 512, **capacity) == 1.5
    assert U.reuse_bound(12, 48, 48, **capacity) == 1.0
    # A smaller store forces more re-reads; nothing feasible is None, never a guess.
    assert U.reuse_bound(512, 512, 512, operand_elems=64 * 512, accumulator_elems=1024, edge=16) > 1.5
    assert U.reuse_bound(64, 64, 64, operand_elems=8, accumulator_elems=8, edge=16) is None


def test_absent_capacity_or_evidence_leaves_the_block_absent(workspace, monkeypatch):
    assert U.for_case("not_a_case", {}, workspace) is None
    monkeypatch.setattr(U, "capacity", lambda target: None)
    assert U.for_case("G_matmul_m12k48n48_raw", _row("G_matmul_m12k48n48_raw"), workspace) is None
