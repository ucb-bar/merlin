"""A product dir must be gate-valid the moment it exists, not when its producer finishes.

`check_artifact_layout` scans `out/artifacts/*/v*` on disk regardless of `--staged` -- deliberately,
since that is where reports live. So a product dir without a manifest.yaml fails the layout gate for
EVERY session on this shared tree, for as long as the producer runs. That happened twice in one day:
a verification product left one behind when its session ended, and a multi-model campaign tripped the
gate 39 seconds into a run that would have written its manifest only after the first cell finished.
"""
from __future__ import annotations

import os
import tempfile

import pytest


@pytest.fixture()
def out_root(monkeypatch):
    d = tempfile.mkdtemp(dir="/scratch/agustin/tmp", prefix="test_prod_")
    monkeypatch.setenv("MERLIN_OUT_ROOT", d)
    return d


def test_a_new_product_is_immediately_gate_valid(out_root):
    from merlin.common.artifacts import new_product

    prod = new_product("probe", version=1, notes="created, nothing written yet")
    # the window between mkdir and the producer's own write_manifest() must be zero-width
    assert prod.manifest_path.is_file(), "product dir exists without a manifest"
    assert prod.manifest_path.parent == prod.path


def test_the_placeholder_carries_the_provenance_the_schema_needs(out_root):
    from merlin.common.artifacts import new_product
    from merlin.common.yaml import load_yaml

    prod = new_product("probe", version=1, target=None, notes="n")
    man = load_yaml(prod.manifest_path)
    for field in ("run_id", "timestamp", "git_sha", "version"):
        assert field in man, field
    assert man["run_id"] == prod.path.name
    assert man["artifacts"] == [], "a fresh product claims no artifacts"


def test_the_producers_own_write_still_wins(out_root):
    """The placeholder must not stop a producer recording what it actually made."""
    from merlin.common.artifacts import new_product
    from merlin.common.yaml import load_yaml

    prod = new_product("probe", version=1)
    prod.add_artifact("ledger.jsonl")
    prod.write_manifest()
    assert load_yaml(prod.manifest_path)["artifacts"] == ["ledger.jsonl"]


def test_the_layout_gate_accepts_a_freshly_created_product(out_root):
    """End-to-end against the real checker, not a reimplementation of its rule."""
    from pathlib import Path

    from merlin.common.artifacts import new_product

    prod = new_product("probe", version=1)
    root = Path(out_root).parent if Path(out_root).name == "out" else Path(out_root)
    # the checker walks <root>/out/artifacts/*/v*; point it at our temp root
    import importlib.util

    from merlin.common.paths import repo_root
    spec = importlib.util.spec_from_file_location(
        "_layout", repo_root() / "build_tools" / "scripts" / "check_artifact_layout.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    violations = [v for v in mod.check(root, staged=True) if "manifest.yaml" in v]
    assert not violations, violations
    assert prod.manifest_path.is_file()
