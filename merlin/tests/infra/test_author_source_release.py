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
