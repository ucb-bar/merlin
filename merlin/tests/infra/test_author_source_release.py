"""Release refresh retains the exact declared protected formal toolchain."""

import hashlib
import importlib.util

import pytest

from merlin.common.paths import repo_root


@pytest.fixture
def prepare():
    spec = importlib.util.spec_from_file_location(
        "prepare_source_release",
        repo_root() / "examples/gemmini/phase1/launch/prepare_source_release.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bundle(path):
    return {
        "compiler_tool_profile": {"lean_toolchain": {
            "selection": "explicit_manifest",
            "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }},
        "host_inputs": [{"path": str(path), "note": "selected formal verification toolchain manifest"}],
    }


@pytest.mark.parametrize("form", ["ordinary", "grouped", "both"])
def test_manifest_declaration_forms_preserve_selected_identity(prepare, tmp_path, form):
    path = tmp_path / "toolchain.json"
    path.write_text('{}')
    selected = bundle(path)
    if form == "grouped":
        selected["host_inputs"].clear()
    if form != "ordinary":
        selected["host_inputs"].append({"closed_toolchain_root": {"manifest": str(path)}})
    assert prepare.selected_lean_manifest(selected) == path


def test_changed_manifest_refuses_refresh(prepare, tmp_path):
    path = tmp_path / "toolchain.json"
    path.write_text('{}')
    selected = bundle(path)
    path.write_text('{"changed":true}')
    with pytest.raises(ValueError, match="manifest changed"):
        prepare.selected_lean_manifest(selected)


def test_conflicting_declarations_refuse_refresh(prepare, tmp_path):
    path = tmp_path / "toolchain.json"
    path.write_text('{}')
    selected = bundle(path)
    selected["host_inputs"].append({"closed_toolchain_root": {"manifest": str(tmp_path / "other.json")}})
    with pytest.raises(ValueError, match="requires one"):
        prepare.selected_lean_manifest(selected)


def test_undeclared_matching_file_is_not_discovered(prepare, tmp_path):
    path = tmp_path / "toolchain.json"
    path.write_text('{}')
    selected = bundle(path)
    selected["host_inputs"][0]["note"] = "unrelated host input"
    with pytest.raises(ValueError, match="requires one"):
        prepare.selected_lean_manifest(selected)


def test_symlink_manifest_refuses_refresh(prepare, tmp_path):
    path = tmp_path / "toolchain.json"
    path.write_text('{}')
    link = tmp_path / "alias.json"
    link.symlink_to(path)
    with pytest.raises(ValueError, match="manifest changed"):
        prepare.selected_lean_manifest(bundle(link))



def test_reviewed_v16_refresh_forwards_selected_substrate(prepare, tmp_path, monkeypatch):
    import json

    previous = tmp_path / 'previous'
    preparation = previous / 'private/preparation.json'
    preparation.parent.mkdir(parents=True)
    preparation.write_text(json.dumps({
        'assembly': {'mode': 'generated_only', 'baseline': None, 'retirements': {'members': []},
                     'capsule_scope': {'kind': 'all'},
                     'public_supplement': {'selection': {'path': 'supplement.json'}}},
        'scaffolding': {'operand_protocol': {'path': 'operands.json', 'sha256': 'a' * 64},
                        'operand_revision': {'path': 'revision.json', 'sha256': 'b' * 64},
                        'public_reference_sources': {'path': 'references.json'}},
        'source_run': 'source-run', 'phase1_policy': {'source_path': 'policy.yaml'}}))
    scale = previous / 'payload/corpus/_public_scalability/selection.json'
    scale.parent.mkdir(parents=True)
    scale.write_text(json.dumps({'source': {'selection': {'path': 'scale.json'}}}))
    manifest = tmp_path / 'formal.json'
    manifest.write_text('{}')
    selected = (previous / 'payload/experiment/input_bundles/merlin_assisted_rtlchecks_public_v0'
                / 'input_bundle_manifest.yaml')
    selected.parent.mkdir(parents=True)
    selected.write_text(json.dumps(bundle(manifest)))
    calls = []
    monkeypatch.setattr(prepare.release, 'prepare', lambda *args, **kwargs: calls.append(kwargs))
    prepare.prepare(previous, tmp_path / 'output', tmp_path / 'sdk',
                    common_substrate='merlin.common_compiler_substrate.v16')
    assert calls[0]['common_substrate'] == 'merlin.common_compiler_substrate.v16'
    assert calls[0]['generated_only'] is True
    with pytest.raises(ValueError, match='explicitly supported'):
        prepare.prepare(previous, tmp_path / 'other', tmp_path / 'sdk',
                        common_substrate='merlin.common_compiler_substrate.v999')
