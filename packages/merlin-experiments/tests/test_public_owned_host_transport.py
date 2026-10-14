"""Operator selfcheck children receive exact frozen host inputs before formal grading."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from merlin_experiments.phase1.feedback import public_owned_host_selection as selection


def test_pinned_host_leaf_build_selection_reaches_operator_child(tmp_path, monkeypatch):
    monkeypatch.setenv(selection.ENV, "untrusted inherited value")
    row = {"board": "selected_board", "arena_mb": 32}
    for field in ("board_catalog", "host_dts", "rtl_facts"):
        path = tmp_path / field
        path.write_text(field)
        row[field] = str(path)
        row[field + "_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    declaration = tmp_path / "instruction.json"
    declaration.write_text("{}")
    spec = tmp_path / "spec.yaml"
    spec.write_text(yaml.safe_dump({"target": "selected_target", "models": [row]}))
    record = selection.install_for_session(spec, declaration, target="selected_target", run_dir=tmp_path / "run")
    script = """import os,json,pathlib,hashlib
r=json.loads(os.environ['MERLIN_MODEL_OWNED_HOST_BUILD_SELECTION'])
p=pathlib.Path(r['path']); assert hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256']
s=json.loads(p.read_text()); assert s['arena_mb']==32
for k in ('board_catalog','dts','rtl_facts','instruction_declaration'):
 p=pathlib.Path(s[k]['path']); assert hashlib.sha256(p.read_bytes()).hexdigest()==s[k]['sha256']
print(s['board'])
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", script], env=os.environ.copy(), capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == row["board"]
    assert json.loads(os.environ[selection.ENV]) == record
    (tmp_path / "rtl_facts").write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        selection.install_for_session(spec, declaration, target="selected_target", run_dir=tmp_path / "run")
    assert selection.ENV not in os.environ


def test_missing_launch_inputs_clear_inherited_selection(tmp_path, monkeypatch):
    monkeypatch.setenv(selection.ENV, "untrusted inherited value")
    assert selection.install_for_session(None, None, target="selected_target", run_dir=tmp_path) is None
    assert selection.ENV not in os.environ


@pytest.fixture
def facts_projection(tmp_path):
    row = {"board": "selected_board", "arena_mb": 32}
    original = {
        "schema_version": 4,
        "inputs": {"target": "selected_target", "fir_sha256": "a" * 64},
        "facts": {"arrays": [{"rows": 16, "cols": 16}]},
        "source_consistency": {"status": "verified", "sources": [{"sha256": "b" * 64}]},
    }
    for field in ("board_catalog", "host_dts", "rtl_facts"):
        path = tmp_path / field
        path.write_text(json.dumps(original, separators=(",", ":")) if field == "rtl_facts" else field)
        row[field] = str(path)
        row[field + "_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    declaration = tmp_path / "instruction.json"
    declaration.write_text("{}")
    spec = tmp_path / "spec.yaml"
    spec.write_text(yaml.safe_dump({"target": "selected_target", "models": [row]}))
    public = tmp_path / "admitted-public-facts.json"
    public.write_text(json.dumps(original, indent=2, sort_keys=True) + "\n")
    return spec, declaration, public, original


def test_public_projection_uses_exact_admitted_serialization_without_private_changes(facts_projection, tmp_path):
    spec, declaration, public, _ = facts_projection
    private_before = spec.read_bytes(), (tmp_path / "rtl_facts").read_bytes()
    digest = hashlib.sha256(public.read_bytes()).hexdigest()
    record = selection.install_for_session(
        spec,
        declaration,
        target="selected_target",
        run_dir=tmp_path / "run",
        selected_facts=public,
        selected_facts_sha256=digest,
    )
    projected = json.loads(Path(record["path"]).read_bytes())
    assert projected["rtl_facts"] == {"path": str(public), "sha256": digest}
    assert private_before == (spec.read_bytes(), (tmp_path / "rtl_facts").read_bytes())


@pytest.mark.parametrize(
    "mutation", ["value", "provenance", "extra", "type", "target", "inputs", "duplicate", "nonfinite", "pin"]
)
def test_public_projection_refuses_any_complete_view_or_byte_pin_change(facts_projection, tmp_path, mutation):
    spec, declaration, public, document = facts_projection
    if mutation == "value":
        document["facts"]["arrays"][0]["rows"] += 1
    elif mutation == "provenance":
        document["source_consistency"]["sources"][0]["sha256"] = "c" * 64
    elif mutation == "extra":
        document["unselected_field"] = True
    elif mutation == "type":
        document["facts"]["arrays"][0]["rows"] = 16.0
    elif mutation == "target":
        document["inputs"]["target"] = "another_target"
    elif mutation == "inputs":
        document["inputs"] = []
    public.write_text(json.dumps(document, sort_keys=True))
    if mutation == "duplicate":
        public.write_text('{"inputs":{"target":"selected_target"},"inputs":{"target":"selected_target"}}')
    if mutation == "nonfinite":
        public.write_text('{"inputs":{"target":"selected_target"},"facts":NaN}')
    digest = hashlib.sha256(public.read_bytes()).hexdigest() if mutation != "pin" else "0" * 64
    with pytest.raises(ValueError):
        selection.prepare(
            spec,
            declaration,
            target="selected_target",
            output=tmp_path / "projection.json",
            selected_facts=public,
            selected_facts_sha256=digest,
        )
    assert not (tmp_path / "projection.json").exists()


def test_projection_refuses_partial_selection_and_links(facts_projection, tmp_path):
    spec, declaration, public, _ = facts_projection
    with pytest.raises(ValueError, match="exact admitted byte pin"):
        selection.prepare(
            spec, declaration, target="selected_target", output=tmp_path / "projection.json", selected_facts=public
        )
    link = tmp_path / "link.json"
    link.symlink_to(public)
    with pytest.raises(ValueError, match="ordinary admitted absolute"):
        selection.prepare(
            spec,
            declaration,
            target="selected_target",
            output=tmp_path / "projection.json",
            selected_facts=link,
            selected_facts_sha256=hashlib.sha256(public.read_bytes()).hexdigest(),
        )
