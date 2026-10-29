"""Seed preparation: route analysis, variant change inference and promotion trees."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from merlin_experiments.phase1.seed_prep import structure, variants


def _write(path: Path, text: str, mtime: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    os.utime(path, (mtime, mtime))


def test_family_of_reads_words_left_to_right():
    assert structure.family_of("merlin.compile.scheduling.dense_ready_readout") == "dense"
    assert structure.family_of("merlin.compile.scheduling.spatial_readout") == "conv"
    assert structure.family_of("resident_gemv_epochs") == "decode"
    assert structure.family_of("row_softmax") == "attention"
    assert structure.family_of("graph") == structure.SHARED


def test_analyze_tree_finds_library_routes_through_author_imports(tmp_path):
    lib = tmp_path / "compiler_library" / "merlin" / "compile" / "scheduling"
    _write(lib / "dense_panel_epochs.py", "X = 1\n", 0)
    _write(lib / "__init__.py", "", 0)
    _write(tmp_path / "dense.py",
           "from merlin.compile.scheduling.dense_panel_epochs import X\nimport helper\n", 0)
    _write(tmp_path / "helper.py", "import json\n", 0)
    _write(tmp_path / "conv.py", "from dense import lower\n", 0)
    routes = structure.analyze_tree(tmp_path)
    assert routes.library_by_family()["dense"] == {"merlin.compile.scheduling.dense_panel_epochs"}
    assert structure.reachable_library(routes, ["conv"]) == {"merlin.compile.scheduling.dense_panel_epochs"}
    other = structure.TreeRoutes(root="o", library_imports={"conv": {"merlin.x.rolling_spatial"}})
    rows = {r.family: r for r in structure.compare_families(routes, {"other": other})}
    assert rows["conv"].missing_from_seed == {"merlin.x.rolling_spatial": ["other"]}
    assert rows["conv"].seed_author == {"conv": 2}


@pytest.fixture()
def seed_and_variants(tmp_path):
    old = time.time() - 3600
    default = tmp_path / "seed"
    _write(default / "manifest.yaml", "x: 1\n", old)
    _write(default / "dense.py", "a = 1\n", old)
    _write(default / "conv.py", "c = 2\n", old)
    _write(default / "knobs.json", json.dumps({"a": 1, "b": 1}), old)
    root = tmp_path / "variants"
    root.mkdir()
    if variants.birth_time(root) is None:
        pytest.skip("filesystem records no birth time")
    return default, root, old


def test_variant_change_is_files_written_after_the_fork(seed_and_variants):
    default, root, old = seed_and_variants
    later = time.time() + 60
    var = root / "faster"
    _write(var / "manifest.yaml", "x: 1\n", old)
    _write(var / "dense.py", "a = 1\nb = 2\n", later)  # edited after fork
    _write(var / "conv.py", "c = 1\n", old - 10)  # stale copy of an older default
    _write(var / "extra.py", "e = 1\n", later)
    _write(var / "knobs.json", json.dumps({"a": 2, "b": 1}), later)
    # The default moved on after the fork: knob b changed; a key-level merge keeps both.
    _write(default / "knobs.json", json.dumps({"a": 1, "b": 3}), later)
    base = root.parent / "base"
    _write(base / "knobs.json", json.dumps({"a": 1, "b": 1}), old)
    changes = variants.discover(root, default, [base])
    (change, merged), = changes
    kinds = {c.path: c.kind for c in change.changes}
    assert kinds == {"dense.py": "edited", "extra.py": "added", "knobs.json": "merged"}
    assert change.status == "applicable"
    out = root.parent / "promoted"
    applied = variants.materialize(default, out, changes)
    assert applied == ["faster"]
    assert (out / "dense.py").read_text() == "a = 1\nb = 2\n"
    assert (out / "conv.py").read_text() == "c = 2\n"
    assert json.loads((out / "knobs.json").read_text()) == {"a": 2, "b": 3}


def test_conflicting_json_edits_are_refused(seed_and_variants):
    default, root, old = seed_and_variants
    later = time.time() + 60
    var = root / "clash"
    _write(var / "manifest.yaml", "x: 1\n", old)
    _write(var / "knobs.json", json.dumps({"a": 2, "b": 1}), later)
    _write(default / "knobs.json", json.dumps({"a": 5, "b": 1}), later)
    base = root.parent / "base"
    _write(base / "knobs.json", json.dumps({"a": 1, "b": 1}), old)
    (change, _), = variants.discover(root, default, [base])
    assert change.status == "conflict"
    assert variants.materialize(default, root.parent / "p", [(change, {})]) == []


def test_case_family_from_generic_words():
    from merlin_experiments.phase1.seed_prep.baseline import case_family

    assert case_family("ZA_conv_3x3x14x14x256x256x4_native_i8") == "conv"
    assert case_family("JT_tiny_decode_m1k2048n32000_raw") == "decode"
    assert case_family("PD_model_m1k2048n1000_raw") == "decode"
    assert case_family("PD_model_m196k256n1024_raw") == "dense"
    assert case_family("CH_kv_heads_q16kv4s96l96d52") == "attention"
    assert case_family("PQ00_j1_k16") == "other"


def test_summarize_reads_counts_roofline_and_labels_evidence(tmp_path):
    from merlin_experiments.phase1.seed_prep.baseline import summarize

    perf = {
        "status": "PROVISIONAL", "estimated_whole_soc_cycles": 1000, "model_sha256": "m",
        "structural_roofline": {"useful_macs": 25600, "structural_peak_macs_per_cycle": 256,
                                "structural_compute_floor_cycles": 100,
                                "logical_request_ideal_service": {"read": {"ideal_requested_service_cycles": 250}}},
        "feature_counts": {"target.class.load": 10, "target.class.compute_ws": 4, "target.class.compute": 1,
                           "target.class.store": 2, "target.class.store_act1": 2, "target.fences": 8,
                           "target.requested_read_bytes": 640, "target.requested_write_bytes": 256,
                           "target.load_publication.aggregate_fallback": 0},
    }
    doc = {"observed": {"performance": perf, "tiers": {"L2": {"status": "pass"}}}, "grade_dir": str(tmp_path)}
    row = summarize("ZA_gemm_64x64x64", doc)
    assert row["status"] == "timed" and row["model_cycles"] == 1000
    assert row["roofline_cycles"] == 250 and row["bound_by"] == "read requests"
    assert row["headroom_cycles"] == 750 and row["utilization_pct"] == 10.0
    assert (row["loads"], row["computes"], row["stores"]) == (10, 5, 4)
    assert row["fences_per_output_tile"] == 2.0
    assert row["transfer_widths"] == {"read_bytes_per_load": 64.0, "write_bytes_per_store": 64.0}
    assert row["capacity_fallback"] is False
    unknown = summarize("x", {"observed": {"performance": {"status": "UNKNOWN", "reason": "budget"}}})
    assert unknown["status"] == "UNKNOWN" and unknown["model_cycles"] is None


def _workspace(tmp_path):
    from merlin_experiments.phase1.seed_prep import records

    ws = tmp_path / "ws"
    (ws / "qa" / "cases" / "A").mkdir(parents=True)
    ident = {"model_sha256": "m"}
    history = [
        {"kind": "modeled", "test_case": "A", "t": 1, "cycles": 900, "roofline_bound_cycles": 100,
         "graded_source_sha256": "old", "performance_identity": ident},
        {"kind": "modeled", "test_case": "A", "t": 2, "cycles": 800, "roofline_bound_cycles": 100,
         "graded_source_sha256": "newer", "performance_identity": ident},
        {"kind": "modeled", "test_case": "B", "t": 1, "cycles": 50, "roofline_bound_cycles": 10,
         "graded_source_sha256": "seed", "performance_identity": ident},
        {"kind": "status", "test_case": "A", "t": 3},
    ]
    (ws / "qa" / "perf_history.jsonl").write_text("\n".join(json.dumps(r) for r in history) + "\n")
    detail = {"capsule": "A", "tiers": {"L2": "pass"}, "performance": {
        "status": "PROVISIONAL", "estimated_whole_soc_cycles": 800, "model_sha256": "m",
        "structural_roofline": {"structural_compute_floor_cycles": 100},
        "feature_counts": {"target.class.load": 7, "target.class.store": 2, "target.fences": 4}}}
    (ws / "qa" / "cases" / "A" / "d.json").write_text(json.dumps(detail))
    (ws / "qa" / "completed_rows.jsonl").write_text(json.dumps(
        {"capsule": "A", "detail": {"path": "qa/cases/A/d.json"},
         "graded": {"graded_source_sha256": "newer", "graded_at": "2026-10-28T00:00:00+00:00"}}) + "\n")
    tuning = {"experiments": [
        {"candidate": {"kind": "variant", "name": "v"}, "test_case": "A", "measured_cycles_before": 900,
         "measured_cycles_after": 700, "numerics": "unchanged: both pass", "graded_at_unix_ns": 1},
        {"candidate": {"kind": "variant", "name": "w"}, "test_case": "A", "measured_cycles_before": 900,
         "measured_cycles_after": 850, "numerics": "unchanged: both pass", "graded_at_unix_ns": 1}]}
    (ws / "qa" / "idle_tuning.json").write_text(json.dumps(tuning))
    return records, ws


def test_record_baseline_prefers_seed_bytes_then_newest_and_keeps_counts(tmp_path):
    records, ws = _workspace(tmp_path)
    rows = {r["case"]: r for r in records.baseline("seed", ["A", "B", "C"], ws)}
    assert rows["B"]["model_cycles"] == 50 and rows["B"]["source_match"] == "seed"
    assert rows["A"]["model_cycles"] == 800 and rows["A"]["source_match"] == "older source"
    assert rows["A"]["loads"] == 7 and rows["A"]["fences_per_output_tile"] == 2.0
    assert rows["C"]["status"] == "UNKNOWN" and "--measure" in rows["C"]["unknown_reason"]


def test_record_decisions_compare_variant_records_with_the_seed_baseline(tmp_path):
    records, ws = _workspace(tmp_path)
    seed_rows = {r["case"]: r for r in records.baseline("seed", ["A"], ws)}
    recorded = records.variant_records(ws)
    faster = records.decide("applicable", recorded["v"], seed_rows)
    slower = records.decide("applicable", recorded["w"], seed_rows)
    assert faster["promote"] and faster["gain_cycles"] == 100
    assert not slower["promote"]
    assert not records.decide("conflict", recorded["v"], seed_rows)["promote"]
