"""Preserve protected formal-toolchain selection during prospective source refresh."""

import hashlib
import importlib.util
from pathlib import Path

import pytest

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
