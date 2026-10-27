"""Frozen catalog admission happens before any compiler or simulator action."""

import importlib.util
import json
import shutil

import pytest

from merlin.common.paths import repo_root


@pytest.fixture
def builder(tmp_path):
    folder = repo_root() / "examples/gemmini/comparisons/tvm"
    spec = importlib.util.spec_from_file_location("shared_elf_test", folder / "build_shared_chunk_elf.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    catalog = repo_root() / "out/artifacts/handoff/gemmini-shared-model-chunks/v1/latest/catalog.json"
    shutil.copy2(catalog, tmp_path / "catalog.json")
    return module


def test_blocked_diagnostic_cannot_be_compiled(builder, tmp_path):
    with pytest.raises(ValueError, match="Blocked chunk"):
        builder.selected_chunk(tmp_path, "smolvla-action-update")


def test_catalog_cannot_promote_blocked_diagnostic(builder, tmp_path):
    path = tmp_path / "catalog.json"
    catalog = json.loads(path.read_text())
    next(row for row in catalog["chunks"] if row["id"] == "smolvla-action-update")["status"] = "ready_for_tuning"
    path.write_text(json.dumps(catalog))
    with pytest.raises(ValueError, match="frozen"):
        builder.selected_chunk(tmp_path, "smolvla-action-update")


def test_unknown_chunk_does_not_create_build(builder, tmp_path):
    with pytest.raises(ValueError, match="Unknown"):
        builder.selected_chunk(tmp_path, "invented-complete-session")
