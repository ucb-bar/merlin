"""Preserve protected formal-toolchain selection during prospective source refresh."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import yaml

spec = importlib.util.spec_from_file_location(
    "author_source_refresh", Path(__file__).with_name("prepare_source_release.py")
)
refresh = importlib.util.module_from_spec(spec)
spec.loader.exec_module(refresh)


def selected(tmp_path):
    path = tmp_path / "toolchain.json"
    path.write_text('{"synthetic":true}')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    bundle = {
        "compiler_tool_profile": {"lean_toolchain": {"selection": "explicit_manifest", "manifest_sha256": digest}},
        "host_inputs": [{"closed_toolchain_root": {"manifest": str(path)}}],
    }
    return path, bundle


def test_preserves_one_selected_formal_toolchain(tmp_path):
    path, bundle = selected(tmp_path)
    assert refresh.selected_lean_manifest(bundle) == path


@pytest.mark.parametrize("change", ["missing", "drift", "ambiguous", "unselected"])
def test_refuses_missing_drifted_or_ambiguous_formal_selection(tmp_path, change):
    path, bundle = selected(tmp_path)
    if change == "missing":
        bundle["host_inputs"] = []
    elif change == "drift":
        path.write_text('{"changed":true}')
    elif change == "ambiguous":
        bundle["host_inputs"].append({"closed_toolchain_root": {"manifest": str(tmp_path / "other.json")}})
    else:
        bundle["compiler_tool_profile"]["lean_toolchain"]["selection"] = "not_selected"
    with pytest.raises(ValueError):
        refresh.selected_lean_manifest(bundle)


def carried(tmp_path, monkeypatch, *, gates):
    """A previous release whose overlay holds ``gates`` and a new run whose descriptor moved on."""
    overlay = tmp_path / "overlay.yaml"
    overlay.write_text(
        yaml.safe_dump({"target": "t", "host_lane": {"capability_spec": "old.yaml"}, "phase1_gates": gates})
    )
    previous = tmp_path / "previous"
    (previous / "private").mkdir(parents=True)
    digest = hashlib.sha256(overlay.read_bytes()).hexdigest()
    (previous / "private/preparation.json").write_text(
        json.dumps({"phase1_policy": {"source_path": str(overlay), "sha256": digest}})
    )
    descriptor = tmp_path / "descriptor.yaml"
    descriptor.write_text(
        yaml.safe_dump({"target": "t", "host_lane": {"capability_spec": "new.yaml"}, "phase1_gates": {"own": 1}})
    )
    plan = {"phases": {"0": {"inputs": {"descriptor": str(descriptor)}}}}
    monkeypatch.setattr(refresh.release, "source_run", lambda run: (plan, None, None))
    return previous, descriptor, overlay


def test_carries_previous_phase1_gates_onto_the_new_runs_descriptor(tmp_path, monkeypatch):
    from merlin_experiments.corpus.preparation import stage_phase1_policy_descriptor

    previous, descriptor, _ = carried(tmp_path, monkeypatch, gates={"public_codegen_scalability": {"required": True}})
    output = refresh.carry_phase1_gates(previous, tmp_path, tmp_path / "out/policy.yaml")
    document = yaml.safe_load(output.read_text())
    assert document["phase1_gates"] == {"public_codegen_scalability": {"required": True}}
    assert document["host_lane"] == {"capability_spec": "new.yaml"}
    # The release's own overlay check accepts it against the run's descriptor.
    private = tmp_path / "private"
    private.mkdir()
    _, gates = stage_phase1_policy_descriptor(descriptor, output, private)
    assert gates == {"public_codegen_scalability": {"required": True}}
    with pytest.raises(FileExistsError):
        refresh.carry_phase1_gates(previous, tmp_path, output)


@pytest.mark.parametrize("change", ["drift", "no_gates"])
def test_carrying_refuses_a_changed_or_gateless_previous_overlay(tmp_path, monkeypatch, change):
    previous, _, overlay = carried(tmp_path, monkeypatch, gates={} if change == "no_gates" else {"g": 1})
    if change == "drift":
        overlay.write_text(overlay.read_text() + "# edited\n")
    with pytest.raises(ValueError):
        refresh.carry_phase1_gates(previous, tmp_path, tmp_path / "policy.yaml")
