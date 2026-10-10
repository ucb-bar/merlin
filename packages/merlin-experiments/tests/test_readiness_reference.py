"""Operator-selected Chipyard readiness package checks; no simulator is started here."""

from __future__ import annotations

import importlib.util

import pytest

from merlin.common.paths import repo_root

_MODULE = repo_root() / "merlin/experiments/capsule_bench/harness/readiness_reference.py"
_SPEC = importlib.util.spec_from_file_location("readiness_reference", _MODULE)
assert _SPEC is not None and _SPEC.loader is not None
reference = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(reference)


def test_reference_env_python_precedes_chipyard_python_without_losing_tools():
    assert reference.reference_probe_path("/selected/venv/bin/python", "/chipyard/.conda-env", "/usr/bin") == (
        "/selected/venv/bin:/chipyard/.conda-env/bin:/chipyard/.conda-env/riscv-tools/bin:/usr/bin"
    )


def test_reference_backend_selection_is_explicit_and_target_bound(tmp_path):
    with pytest.raises(ValueError, match="requires --reference-backend"):
        reference.select_reference_backend(None, target="gemmini")
    with pytest.raises(ValueError, match="absolute package path"):
        reference.select_reference_backend("relative/package", target="gemmini")

    package = tmp_path / "published"
    package.mkdir()
    manifest = package / "manifest.yaml"
    manifest.write_text(
        "target: gemmini\nartifact_type: mlir_oot_target_backend\n"
        "integrity_exempt: false\nlanguage: python\n"
        "entrypoints: {tool: gemmini-opt}\npackage_id: published_v1\n",
        encoding="utf-8",
    )
    selected, metadata = reference.select_reference_backend(str(package), target="gemmini")
    assert selected == package
    assert metadata["package_id"] == "published_v1"

    with pytest.raises(ValueError, match="expected 'other'"):
        reference.select_reference_backend(str(package), target="other")
    manifest.write_text(manifest.read_text().replace("integrity_exempt: false", "integrity_exempt: true"))
    with pytest.raises(ValueError, match="integrity_exempt: false"):
        reference.select_reference_backend(str(package), target="gemmini")
