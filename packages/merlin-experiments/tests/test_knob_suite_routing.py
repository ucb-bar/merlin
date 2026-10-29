"""Mixed public performance bundles retain native suite custody and matching baselines."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin_experiments.phase1.tools import selfcheck as C


def _workspace(tmp_path):
    root = tmp_path / "workspace"
    (root / "submission").mkdir(parents=True)
    (root / ".qa_channel").mkdir()
    (root / "submission/tuning.json").write_text('{"mode": false}')
    (root / "submission/TUNING_KNOBS.json").write_text(json.dumps({
        "schema": "merlin.tuning_knobs.v1", "settings_file": "tuning.json",
        "knobs": {"mode": {"values": [False, True]}}}))
    (root / "PUBLIC_CAPABILITIES.json").write_text(json.dumps({"members": {
        "large": {"selfcheck_suite": "scalability"},
        "edge": {"selfcheck_suite": "core"},
        "calibration": {"selfcheck_suite": "core"},
        "connected": {"selfcheck_suite": "programs"},
    }}))
    return root


def test_standalone_sweep_partitions_before_publishing_without_extending_deadline(tmp_path, capsys):
    ws = _workspace(tmp_path)
    # Exercise the actual served single-file client, with no import of a host routing helper.
    deployed = ws / "agent_selfcheck.py"
    deployed.write_bytes(Path(C.__file__).read_bytes())
    spec = importlib.util.spec_from_file_location("deployed_knob_client", deployed)
    client = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(client)
    args = SimpleNamespace(sweep="", knob=["mode=true"], capsules="large,edge,calibration,connected",
                           bundle="", timeout=31, workers=2, wait_budget=0.001, out="")
    assert client._run_sweep(args, ws, ws / ".qa_channel") == 3
    capsys.readouterr()
    manifest = json.loads(next((ws / ".qa_channel").glob("sweep_*.json")).read_text())
    requests = [json.loads(p.read_text()) for p in (ws / ".qa_channel").glob("req_*.json")]
    assert len(manifest["baseline_request_ids"]) == 3 and len(requests) == 6
    assert len({r["deadline_unix_ns"] for r in requests}) == 1
    assert all(r["deadline_unix_ns"] == manifest["deadline_unix_ns"] for r in requests)
    by_id = {r["request_id"]: r for r in requests}
    for member in manifest["requests"]:
        body = by_id[member["request_id"]]
        assert body["suite"] == member["suite"]
        assert body["capsules"].split(",") == member["capsules"]
        baseline = by_id[member["baseline_request_id"]]
        assert baseline["suite"] == body["suite"] and baseline["capsules"] == body["capsules"]
        assert baseline["tuning_experiment"]["candidate"]["kind"] == "baseline"
        assert body["workers"] == 2 and body["timeout"] == 31 and body["sim"] == "spike"
        assert body.get("tiers") == ("L0,L1,L2" if body["suite"] == "core" else None)
    candidates = [r for r in requests if r.get("tuning_overrides")]
    assert sorted(n for r in candidates for n in r["capsules"].split(",")) == [
        "calibration", "connected", "edge", "large"]
    assert all(r["tuning_overrides"] == {"mode": True} for r in candidates)
    assert json.loads((ws / "submission/tuning.json").read_text()) == {"mode": False}


@pytest.mark.parametrize("bad", ["hidden", "../private", "all"])
def test_unknown_case_never_becomes_a_request(tmp_path, bad):
    ws = _workspace(tmp_path)
    with pytest.raises(ValueError, match="no declared public"):
        C._case_groups(ws, ["large", bad])
    assert not list((ws / ".qa_channel").glob("req_*.json"))


def test_existing_inventory_cannot_silently_fall_back_when_malformed_or_indirect(tmp_path):
    ws = _workspace(tmp_path)
    path = ws / "PUBLIC_CAPABILITIES.json"
    path.write_text('{"members": {"large": {}}}')
    with pytest.raises(ValueError, match="no declared public"):
        C._case_groups(ws, ["large"])
    path.unlink()
    other = ws / "indirect.json"
    other.write_text('{"members": {"large": {"selfcheck_suite": "scalability"}}}')
    path.symlink_to(other)
    with pytest.raises(ValueError, match="ordinary bounded"):
        C._case_groups(ws, ["large"])


def _row(cycles):
    return {"capsule": "large", "tiers": {"L2": "pass"}, "performance": {
        "estimated_whole_soc_cycles": cycles, "model_sha256": "model", "selection_sha256": "selection",
        "timing_mode": "untuned", "timing_epoch": 0, "scope_sha256": "roi",
        "comparison_domain_sha256": "domain", "elf_sha256": "elf"}}


def test_knob_comparison_rejects_a_different_actual_base_source(tmp_path):
    manifest = {"baseline_request_id": "base", "requests": [
        {"key": "baseline", "request_id": "base"},
        {"key": "knob:mode=true", "request_id": "candidate", "tuning_overrides": {"mode": True}}]}
    baseline = {"submission_sha256": "old", "cases": [_row(100)]}
    candidate = {"submission_sha256": "new", "variant": {"sha256": "private-copy"}, "cases": [_row(50)]}
    rank, compared = C._sweep_rank(tmp_path, manifest, {"base": baseline, "candidate": candidate})
    assert compared[0]["ratio"] is None and compared[0]["delta_pct"] is None
    assert compared[0]["not_compared"] == "knob copy and baseline differ in submission_sha256"
    assert rank[0]["cases_compared"] == 0


def test_partitioned_sweep_joins_each_native_baseline_and_reports_missing_cases(tmp_path):
    def row(name, cycles):
        return {**_row(cycles), "capsule": name}

    manifest = {"baseline_request_id": "base-large", "requests": [
        {"key": "baseline", "request_id": "base-large"},
        {"key": "knob:mode=true", "request_id": "cand-large", "baseline_request_id": "base-large",
         "capsules": ["large"], "tuning_overrides": {"mode": True}},
        {"key": "baseline", "request_id": "base-core"},
        {"key": "knob:mode=true", "request_id": "cand-core", "baseline_request_id": "base-core",
         "capsules": ["edge", "calibration"], "tuning_overrides": {"mode": True}}]}
    responses = {
        "base-large": {"submission_sha256": "base", "cases": [row("large", 100)]},
        "cand-large": {"submission_sha256": "base", "variant": {"sha256": "copy"},
                       "cases": [row("large", 50)]},
        "base-core": {"submission_sha256": "base", "cases": [row("edge", 300), row("calibration", 200)]},
        "cand-core": {"submission_sha256": "base", "variant": {"sha256": "copy"},
                      "cases": [row("edge", 150)]}}
    ranked, compared = C._sweep_rank(tmp_path, manifest, responses)
    assert len(ranked) == 1 and ranked[0]["cases_compared"] == 2
    assert ranked[0]["geomean_ratio"] == 0.5
    assert ranked[0]["numerics_changed_or_unknown"] == 1
    by_name = {row["test_case"]: row for row in compared}
    assert set(by_name) == {"large", "edge", "calibration"}
    assert by_name["edge"]["cycles_before"] == 300 and by_name["edge"]["ratio"] == 0.5
    assert by_name["calibration"]["ratio"] is None


def test_legacy_manifest_and_workspace_keep_their_single_scalability_route(tmp_path):
    assert C._case_groups(tmp_path, ["legacy", "legacy"]) == [("scalability", ["legacy"])]
    manifest = {"baseline_request_id": "base", "requests": [
        {"key": "baseline", "request_id": "base"},
        {"key": "variant:a", "request_id": "candidate"}]}
    baseline = {"submission_sha256": "old", "cases": [_row(100)]}
    candidate = {"submission_sha256": "variant", "cases": [_row(50)]}
    rank, compared = C._sweep_rank(tmp_path, manifest, {"base": baseline, "candidate": candidate})
    assert compared[0]["ratio"] == 0.5 and rank[0]["cases_compared"] == 1
