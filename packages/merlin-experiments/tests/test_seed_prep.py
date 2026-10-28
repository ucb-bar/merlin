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


def _records_workspace(tmp_path, seed_source):
    ws = tmp_path / "ws"
    (ws / "qa").mkdir(parents=True)
    rows = [
        {"kind": "modeled", "test_case": "M_m64k64n64_raw", "cycles": 900, "roofline_bound_cycles": 100,
         "source_sha256": "old", "t": 1, "evidence": "modeled: other"},
        {"kind": "modeled", "test_case": "M_m64k64n64_raw", "cycles": 500, "roofline_bound_cycles": 100,
         "source_sha256": seed_source, "t": 2, "evidence": "modeled: graded",
         "performance_identity": {"model_sha256": "m"}},
        {"kind": "modeled", "test_case": "C_conv_c8_raw", "cycles": 700, "roofline_bound_cycles": 70,
         "source_sha256": "older", "t": 3, "evidence": "modeled: older"},
    ]
    (ws / "qa" / "perf_history.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    grade = {"graded_source_sha256": seed_source, "per_capsule": [
        {"capsule": "M_m64k64n64_raw", "tiers": {"L2": "pass"}, "graded": {"token": "1", "graded_source_sha256": seed_source}},
        {"capsule": "C_conv_c8_raw", "tiers": {"L2": "pass"}, "graded": {"token": "1", "graded_source_sha256": "x"}},
    ]}
    (ws / "qa" / "latest_completed_grade.json").write_text(json.dumps(grade))
    return ws


def test_record_baseline_prefers_the_graded_source_and_labels_everything_else(tmp_path):
    from merlin_experiments.phase1.seed_prep import records

    ws = _records_workspace(tmp_path, "seedsrc")
    rows = {r["case"]: r for r in records.baseline_rows(
        ws, None, ["M_m64k64n64_raw", "C_conv_c8_raw", "N_new_case"], seed_source="seedsrc", facts_sha256=None)}
    assert (rows["M_m64k64n64_raw"]["status"], rows["M_m64k64n64_raw"]["model_cycles"]) == ("timed", 500)
    assert rows["M_m64k64n64_raw"]["headroom_cycles"] == 400
    assert rows["C_conv_c8_raw"]["status"] == "recorded_other_snapshot"
    assert "may differ from the seed" in rows["C_conv_c8_raw"]["evidence"]
    assert rows["N_new_case"]["status"] == "UNKNOWN"
    assert "not graded by the seed run" in rows["N_new_case"]["unknown_reason"]


def test_from_records_never_starts_a_simulation(tmp_path, monkeypatch):
    from merlin_experiments.phase1.seed_prep import __main__ as M
    from merlin_experiments.phase1.seed_prep import baseline

    ws = _records_workspace(tmp_path, "seedsrc")
    seed = tmp_path / "seed"
    _write(seed / "dense.py", "a = 1\n", 0)
    descriptor = tmp_path / "rel/payload/experiment/target_experiment.yaml"
    _write(descriptor, "x: 1\n", 0)
    (tmp_path / "rel/payload/corpus/suite/cases").mkdir(parents=True)
    (tmp_path / "rel/payload/corpus/suite/selection.json").write_text("{}")
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({"members": {"M_m64k64n64_raw": {}}, "facts": {"sha256": "f"}}))

    def refuse(*_a, **_k):
        raise AssertionError("no simulation in --from-records")

    monkeypatch.setattr(baseline, "run_cases", refuse)
    monkeypatch.setattr(baseline, "Grader", refuse)
    out = tmp_path / "out"
    assert M.main(["--seed", str(seed), "--export", str(tmp_path), "--out", str(out), "--descriptor", str(descriptor),
                   "--performance-selection", str(selection), "--from-records", str(ws)]) == 0
    doc = json.loads((out / "SEED_BASELINE.json").read_text())
    assert doc["baseline"]["status_counts"] == {"timed": 1}
    assert (out / "SEED_STRUCTURE.md").is_file()
