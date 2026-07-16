"""WS-C: the `merlin-cca-route` CLI — the "which section of the compiler to modify" surface."""
from __future__ import annotations

import json

from merlin.rvvgen import route_report as rr


def test_full_table_runs_and_lists_every_axis(capsys):
    rc = rr.main([])
    assert rc == 0
    out = capsys.readouterr().out
    # a representative sample of routed axes shows up with their ladder
    for axis in ("compute.contraction_form", "compute.accumulator_resident", "vector.sew"):
        assert axis in out


def test_single_axis_ladder(capsys):
    rc = rr.main(["--axis", "compute.accumulator_resident"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "PASS" in out and "CODEGEN" in out          # the two rungs
    assert "NEEDS NEW CODE" in out                      # the CODEGEN rung is a new-pass work-item


def test_json_output_is_machine_readable(capsys):
    rc = rr.main(["--json"])
    assert rc == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["backend"] == "rvv"
    axes = {a["axis"] for a in doc["axes"]}
    assert "compute.epilogue" in axes


def test_regions_mode_lists_all_regions(capsys):
    rc = rr.main(["--regions"])
    assert rc == 0
    out = capsys.readouterr().out
    # phase headers + the fine per-concern regions the user named distinctly
    for phase in ("kernel-codegen", "target-gen", "cross-cutting"):
        assert f"### {phase}" in out
    for region in ("quantization", "data-tiling", "instruction-selection", "target-dialect-gen"):
        assert region in out
    assert "[GAP]" in out                       # honest gaps surfaced, not hidden


def test_regions_mode_json(capsys):
    rc = rr.main(["--regions", "--json"])
    assert rc == 0
    doc = json.loads(capsys.readouterr().out)
    keys = {r["region"] for r in doc["regions"]}
    assert len(keys) >= 20                       # the full fine-grained registry
    assert all(r["phase"] for r in doc["regions"])   # every region tagged with its compilation phase
    assert all(ep["file"] for r in doc["regions"] for ep in r["edit_points"])


def test_routes_a_divergences_file(tmp_path, capsys):
    import yaml
    p = tmp_path / "divergences.yaml"
    p.write_text(yaml.safe_dump({"divergences": [
        {"axis": "compute.contraction_form", "expert": "fused_fma", "ours": "mul_add", "backend": "rvv"},
    ]}))
    rc = rr.main(["--divergences", str(p), "--json"])
    assert rc == 0
    doc = json.loads(capsys.readouterr().out)
    view = doc["axes"][0]
    assert view["routed"] is True
    assert view["action_class"] == "PASS"
    assert "impr_features.py" in view["seam_file"]
