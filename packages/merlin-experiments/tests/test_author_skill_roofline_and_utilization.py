"""The roofline-and-utilization skill: its summary script reads the real feedback formats and agrees
with the harness's own performance writers; the skill stays generic and cites fields that exist."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest

_CHECKS = importlib.util.spec_from_file_location("author_skill_perf_checks",
                                                 Path(__file__).with_name("author_skill_perf_checks.py"))
C = importlib.util.module_from_spec(_CHECKS)
_CHECKS.loader.exec_module(C)

NAME = "roofline-and-utilization"
SCRIPT = C.skill_dir(NAME) / "scripts" / "perf_summary.py"


def _script():
    spec = importlib.util.spec_from_file_location("perf_summary_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _roofline(useful, read_bytes, write_bytes, *, peak=8, rate=4):
    return {"schema": "merlin.perf.partial_roofline.v1", "status": "partial", "useful_macs": useful,
            "structural_peak_macs_per_cycle": peak, "structural_compute_floor_cycles": -(-useful // peak),
            "logical_request_ideal_service": {
                "read": {"requested_bytes": read_bytes, "client_peak_bytes_per_cycle": rate,
                         "ideal_requested_service_cycles": read_bytes / rate},
                "write": {"requested_bytes": write_bytes, "client_peak_bytes_per_cycle": rate,
                          "ideal_requested_service_cycles": write_bytes / rate}},
            "missing_terms": ["array fill/drain and fixed invocation costs"]}


def _performance(cycles, roofline, *, busy=None, issued=None, status="PROVISIONAL"):
    resources = [{"resource_id": i, "name": name, "capacity": 1, "busy_unit_cycles": int(share * cycles),
                  "active_cycles": int(share * cycles), "capacity_normalized_activity": share,
                  "active_fraction": share, "average_modeled_waiting_entries": 0.0}
                 for i, (name, share) in enumerate((busy or {}).items())]
    counts = {"target.commands": 12}
    for side, value in (issued or {}).items():
        counts[f"target.requested_{side}_bytes"] = value
    return {"schema": "merlin.phase1.joint_performance.v1", "status": status,
            "complete_event_accounting": status != "UNKNOWN",
            "estimated_whole_soc_cycles": cycles if status != "UNKNOWN" else None,
            "structural_roofline": roofline, "feature_counts": counts, "timing_mode": "untuned",
            "scoped_resource_feedback": {"status": "modeled", "resources": resources}}


def _write_workspace(root: Path) -> Path:
    """A small anonymized workspace in the formats of a live run's qa/ and .qa_channel/ files."""
    ws = root / "ws"
    big = _roofline(4000, 2000, 400)        # compute 500, read 500, write 100
    small = _roofline(64, 40, 160)          # compute 8, read 10, write 40
    conv = _roofline(1600, 4000, 200)       # compute 200, read 1000, write 50
    C.dumps(ws / "qa" / "latest_completed_grade.json", {
        "graded_at": "2026-01-01T00:00:00+00:00", "snapshot_current": True,
        "latest_completed_grade": {"token": "t1", "graded_at": "2026-01-01T00:00:00+00:00"},
        "joint_test_selection": {"performance_cohorts": {
            "primary": ["case_big", "case_conv"], "correctness_regression": ["case_small", "case_untimed"],
            "calibration_diagnostics": []}},
        "feedback_brief": {"performance_priority": {}},
        "per_capsule": [
            {"capsule": "case_big", "status": "pass", "tiers": {"L0": "pass", "L1": "pass", "L2": "pass"},
             "performance": _performance(1000, big, busy={"unit.storage.acc": 0.7, "cpu.issue": 0.2},
                                         issued={"read": 4000, "write": 400})},
            {"capsule": "case_small", "status": "pass", "tiers": {"L2": "pass"},
             "performance": _performance(50, small)},
            {"capsule": "case_untimed", "status": "pass", "tiers": {"L2": "pass"},
             "performance": _performance(0, big, status="UNKNOWN")},
            # Timed only by a later self-check of the same source (an idle-time or author request).
            {"capsule": "case_conv", "status": "pass", "tiers": {"L2": "pass"},
             "performance": _performance(0, conv, status="UNKNOWN")},
            {"capsule": "case_failing", "status": "fail", "tiers": {"L2": "fail"}},
        ]})
    C.dumps(ws / "qa" / "tier_state.json", {
        "case_small": {"L2": {"status": "pass", "execution_digest": "d_small"},
                       "L3": {"status": "pass", "execution_digest": "d_small"}}})
    channel = ws / ".qa_channel"
    request = C.dumps(channel / "simreq_promo0_aa_case_small.json", {"capsules": "case_small"})
    response = C.dumps(channel / "simresp_promo0_aa_case_small.json", {
        "sim": "gsim", "barrier_tier": "L3", "per_capsule": [{
            "capsule": "case_small", "barrier_tier": "L3", "barrier_status": "pass", "pass": True,
            "numerical_execution_passed": True, "barrier_cycles": 60, "execution_digest": "d_small",
            "performance": _performance(50, small, busy={"memory.l1d.probe": 0.4})}]})
    os.utime(request, (1_000, 1_000))
    os.utime(response, (1_045, 1_045))
    start = 1_700_000_000 * 10**9
    C.dumps(channel / "resp_3_1.json", {
        "schema": "merlin.phase1.public_scalability.v1", "selfcheck_submission": "submission",
        "submission_sha256": "s1", "requested_at_unix_ns": start, "graded_at_unix_ns": start + 90 * 10**9,
        "cases": [{"capsule": "case_conv", "status": "pass", "tiers": {"L2": "pass"},
                   "performance": _performance(2000, conv, busy={"unit.load": 0.5})}]})
    C.dumps(channel / "resp_3_2.json", {  # a variant: never describes submission/
        "schema": "merlin.phase1.public_scalability.v1", "selfcheck_submission": "variants/x",
        "variant": {"name": "x", "official": False}, "submission_sha256": "s2",
        "cases": [{"capsule": "case_conv", "status": "pass", "tiers": {"L2": "pass"},
                   "performance": _performance(1100, conv)}]})
    (ws / "isa" / "case_small").mkdir(parents=True)
    (ws / "isa" / "case_small" / "capsule.yaml").write_text(
        "name: case_small\nkind: isa\ninputs:\n- name: A\n  shape: [4, 4]\noperation:\n  op: matmul\n"
        "  attributes:\n    out: Y\nnumeric_policy:\n  compare: exact_int\n")
    (ws / "semantic_families.py").write_text(
        "def from_op(op):\n    return {'matmul': 'contraction'}.get(op)\n")
    (ws / "public_sources").mkdir()
    (ws / "public_sources" / "case_conv.mlir").write_text(
        'module attributes {iface.version = "0.1"} {\n'
        '  %W = iface.tensor {name = "W"} : tensor<4x4xi8>\n'
        "  %acc = iface.conv2d %A, %W : (tensor<1x4xi8>) -> !iface.acc<i32>\n"
        '  %Y = iface.commit %acc {name = "Y"} : (!iface.acc<i32>) -> tensor<1x4xi32>\n}\n')
    C.dumps(ws / ".model_check_channel" / "resp_3_1.json", {
        "status": "pass", "model": "model_a", "source_sha256": "f" * 64, "report": {"programs": [{
            "program": "main", "census": {"device_mac_fraction": 1.0},
            "kernels": [{"symbol": "k0", "form": "matmul M=1 K=2 N=3 -> i32", "source_region_ids": ["r0", "r1", "r2"],
                         "cycles": {"kind": "functional", "min": 10, "max": 10, "calls": 1}},
                        {"symbol": "k1", "form": "matmul M=4 K=2 N=3 -> i32", "source_region_ids": ["r3"],
                         "cycles": {"kind": "functional", "min": 70, "max": 70, "calls": 1}}]}]}})
    C.dumps(ws / "qa" / "idle_tuning.json", {
        "schema": "merlin.idle_tuning_experiments.v1", "source_sha256": "s1", "experiments": [
            {"test_case": "case_big", "mechanism": "mechanism_a", "measured_cycles_before": 1000,
             "measured_cycles_after": 900, "delta_pct": -10.0,
             "numerics": "unchanged: submission/ and the candidate both pass this case's numerical check",
             "how_to_apply": "copy variants/x into submission/"}]})
    return ws


@pytest.fixture
def workspace(tmp_path):
    return _write_workspace(tmp_path)


def _tree(root: Path) -> dict:
    return {p.relative_to(root).as_posix(): (p.stat().st_size, p.stat().st_mtime_ns)
            for p in root.rglob("*")}


def test_summary_reads_every_feedback_format_and_derives_the_documented_fields(workspace):
    doc = _script().report(workspace, top=5)
    entries = {e["test_case"]: e for e in doc["largest_headroom"]}
    assert doc["counts"] == {"passing": 4, "with_evidence": 3, "measured": 1, "modeled": 2, "without_evidence": 1,
                             "performance_scope_excluded": 0}
    assert doc["without_evidence"] == ["case_untimed"]
    big, small, conv = entries["case_big"], entries["case_small"], entries["case_conv"]
    # bound = max(compute floor, read/write ideal service); util = 100 * useful / (cycles * peak)
    assert (big["roofline_bound_cycles"], big["bound_by"], big["gap_ratio"], big["headroom_cycles"]) == (
        500, "compute", 2.0, 500)
    assert big["array_utilization_pct"] == 50.0 and big["evidence"].startswith("modeled")
    assert big["busiest_modeled_unit"] == {"name": "unit.storage.acc", "modeled_activity": 0.7}
    assert big["traffic"] == {"read": 2.0, "write": 1.0, "commands": 12}
    assert (small["cycles"], small["bound_by"], small["evidence"]) == (60, "write requests", "measured: gSIM")
    assert small["array_utilization_pct"] == round(100 * 64 / (60 * 8), 2)
    assert "measured_on_earlier_executable" not in small
    # The submission self-check is used; the variant's faster estimate is not about submission/.
    assert (conv["cycles"], conv["bound_by"], conv["evidence"]) == (
        2000, "read requests", "modeled: Spike timing model (self-check)")
    assert [e["test_case"] for e in doc["largest_headroom"]] == ["case_conv", "case_big", "case_small"]
    assert doc["lowest_utilization"][0]["test_case"] == "case_conv"
    assert doc["utilization_pct"]["n"] == 3 and doc["utilization_pct"]["median"] == small["array_utilization_pct"]
    assert doc["without_gsim"] == ["case_big", "case_conv"]
    assert {g["group"] for g in doc["by_family"]} == {"contraction", "conv2d", "undeclared"}
    assert {g["group"]: g["cases"] for g in doc["by_cohort"]} == {"primary": 2, "correctness_regression": 1}
    assert {g["group"] for g in doc["by_work_class"]} == {"1e1-1e2 MACs", "1e3-1e4 MACs"}
    calibration = doc["model_vs_measured"]
    assert calibration["pairs"] == 1 and calibration["per_case"][0]["ratio"] == 1.2
    assert calibration["beyond_2x"] == []
    model = doc["connected_models"][0]
    assert model["model"] == "model_a" and model["device_mac_fraction"] == {"main": 1.0}
    assert [(k["symbol"], k["share_pct"]) for k in model["kernels"]] == [("k1", 70.0), ("k0", 30.0)]
    assert doc["idle_tuning"]["experiments"][0]["delta_pct"] == -10.0
    assert doc["latency_s"]["Spike timing (scalability suite) [author]"]["median"] == 90.0
    assert doc["latency_s"]["gSIM (L3) check [author]"]["median"] == 45.0
    units = {u["unit"]: u for u in doc["units"]}
    assert units["unit.storage.acc"]["busiest_in_cases"] == 1


def test_performance_scope_excluded_cases_stay_out_of_every_ranking(workspace):
    grade = workspace / "qa" / "latest_completed_grade.json"
    doc = json.loads(grade.read_text())
    doc["joint_test_selection"]["performance_scope"] = {"excluded": ["case_big"]}
    grade.write_text(json.dumps(doc))
    report = _script().report(workspace, top=5)
    assert "case_big" not in {e["test_case"] for e in report["largest_headroom"]}
    assert report["counts"]["passing"] == 4 and report["counts"]["performance_scope_excluded"] == 1


def test_summary_matches_the_harness_writers_on_the_same_files(workspace):
    from merlin_experiments.phase1.feedback import model_calibration, performance_brief, performance_headroom

    doc = _script().report(workspace, top=5)
    ours = {e["test_case"]: e for e in doc["largest_headroom"]}
    grade = json.loads((workspace / "qa" / "latest_completed_grade.json").read_text())
    harness = performance_headroom.build(workspace, grade)
    assert harness["counts"]["ranked"] == 3
    keys = ("cycles", "roofline_bound_cycles", "bound_by", "gap_ratio", "headroom_cycles", "array_utilization_pct")
    for entry in harness["largest_headroom"]:
        assert {k: entry[k] for k in keys} == {k: ours[entry["test_case"]][k] for k in keys}
        # The grade block takes a measured case's busiest unit from the grade row; the script, like the
        # request-level brief, from the measured row itself (the same executable as the cycles).
        if entry["evidence"].startswith("modeled"):
            assert entry.get("dominant_modeled_resource") == ours[entry["test_case"]]["busiest_modeled_unit"]
    assert harness["without_evidence"] == doc["without_evidence"]
    rows = json.loads((workspace / ".qa_channel" / "simresp_promo0_aa_case_small.json").read_text())["per_capsule"]
    brief = performance_brief.build(rows, workspace=workspace)["largest_headroom"][0]
    assert brief["array_utilization_pct"] == ours["case_small"]["array_utilization_pct"]
    assert brief["dominant_modeled_resource"] == ours["case_small"]["busiest_modeled_unit"]
    pairs = model_calibration.pairs(workspace)
    assert [round(p["ratio"], 3) for p in pairs] == [p["ratio"] for p in doc["model_vs_measured"]["per_case"]]


def test_summary_is_read_only_and_prints_text_and_json(workspace, capsys):
    before = _tree(workspace)
    module = _script()
    assert module.main(["--workspace", str(workspace), "--top", "3"]) == 0
    text = capsys.readouterr().out
    for heading in ("## array utilization %", "## by family", "## largest headroom", "## modeled units",
                    "## modeled vs measured", "## passing cases without gSIM evidence",
                    "## whole-model checks", "## idle tuning", "## request latency"):
        assert heading in text
    assert module.main(["--workspace", str(workspace), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["counts"]["with_evidence"] == 3
    assert _tree(workspace) == before


def test_summary_of_an_empty_workspace_says_there_is_nothing(tmp_path, capsys):
    module = _script()
    assert module.main(["--workspace", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "passing=0 with_evidence=0" in out and "all evidence: none" in out


def test_measured_cycles_of_an_earlier_executable_are_flagged(workspace):
    state = json.loads((workspace / "qa" / "tier_state.json").read_text())
    state["case_small"]["L2"]["execution_digest"] = "d_newer"
    C.dumps(workspace / "qa" / "tier_state.json", state)
    small = next(e for e in _script().report(workspace)["largest_headroom"] if e["test_case"] == "case_small")
    assert small["measured_on_earlier_executable"] is True
    state["case_small"]["<certs>"] = {"L3": {"slot": {"status": "joined", "execution_digest": "d_newer"}}}
    C.dumps(workspace / "qa" / "tier_state.json", state)
    small = next(e for e in _script().report(workspace)["largest_headroom"] if e["test_case"] == "case_small")
    assert "measured_on_earlier_executable" not in small


def test_skill_layout_is_generic_and_cites_existing_fields():
    C.check_layout(NAME)
    C.check_generic(NAME)
    assert C.missing_fields(NAME) == []
    assert {"cycle-model-and-gsim", "ab-library-pass", "pass-library", "read-command-buffer"} <= C.skills_named(NAME)
    _, body = C.split(NAME)
    assert "scripts/perf_summary.py" in body
