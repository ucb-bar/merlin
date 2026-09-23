"""Explicit Muon support imports and ownership only; no compiler or simulator execution."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import PurePosixPath

import pytest

from merlin.common.paths import python_import_roots, repo_root
from merlin.targetgen import target_registry

pytestmark = pytest.mark.target("muon")


@pytest.fixture
def selected_support():
    root = target_registry.explicit_targets().get("muon")
    if root is None:
        pytest.skip("requires Muon support explicitly selected with MERLIN_TARGET_PATH")
    assert (root / "contracts/target_contract.yaml").is_file()
    return root


def _probe(root, tmp_path, body):
    # A fresh interpreter prevents registration state from leaking across selected providers.
    program = (
        """import pathlib, subprocess, sys
def forbidden(*args, **kwargs):
    raise AssertionError("provider metadata/import launched a subprocess")
subprocess.run = forbidden
subprocess.Popen = forbidden
root = pathlib.Path(sys.argv[1]).resolve()
"""
        + body
    )
    environment = {key: value for key, value in os.environ.items() if not key.startswith(("MERLIN_", "PYTHON"))}
    environment.update(
        MERLIN_TARGET_PATH=str(root),
        MERLIN_REPO_ROOT=str(tmp_path),
        MERLIN_TARGETS_DIR=str(tmp_path / "empty-references"),
        MERLIN_OUT_ROOT=str(tmp_path / "out"),
        PYTHONPATH=os.pathsep.join(map(str, python_import_roots())),
    )
    result = subprocess.run(
        [sys.executable, "-c", program, str(root)],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_explicit_backend_origin_and_colocated_carrier(selected_support, tmp_path):
    _probe(
        selected_support,
        tmp_path,
        """
from merlin.runtime.backends import base
from merlin.targetgen.target_registry import explicit_targets
assert explicit_targets() == {"muon": root}
backend = base.get_backend("muon")
assert pathlib.Path(backend.__file__).resolve() == root / "backend/__init__.py"
assert pathlib.Path(backend.muon.__file__).resolve() == root / "backend/muon.py"
carrier = pathlib.Path(backend.muon.__file__).resolve().parent / "soc_carrier/main.c"
assert carrier.is_file() and carrier.stat().st_size > 0
assert base.class_of("muon") == base.TargetClass.GPU
assert not any(name.endswith(".muon_oracles") for name in sys.modules)
""",
    )


def test_sim_oracle_registration_is_metadata_only(selected_support, tmp_path):
    _probe(
        selected_support,
        tmp_path,
        """
from merlin.targetgen import oracle_policy
caps = oracle_policy.sim_oracle_caps("cyclotron")
assert caps is not None and caps.exclusive is True
for callback in (caps.adapters, caps.available, caps.tier_plan, caps.l3_selection):
    assert callable(callback)
    owner = sys.modules[callback.__module__]
    assert pathlib.Path(owner.__file__).resolve() == root / "sim_oracle.py"
# Metadata registration must not construct adapters or load their backend implementation.
assert not any(name.startswith("merlin._oot_backends.muon") for name in sys.modules)
assert not any(name.endswith(".muon_oracles") for name in sys.modules)
""",
    )


def test_selected_support_root_is_private_answer_surface(selected_support, tmp_path):
    _probe(
        selected_support,
        tmp_path,
        """
from types import SimpleNamespace
from merlin.targetgen.sandbox.answer_surfaces import answer_surfaces, audit_tokens
descriptor = SimpleNamespace(
    target="muon", capsule_corpus=None, corpus_siblings=lambda: (),
    hidden_corpus=lambda: None, prior_backends=(), backend_package=None,
)
surfaces = answer_surfaces(descriptor)
support = [surface for surface in surfaces if surface.path.resolve() == root and surface.origin == "backend"]
assert len(support) == 1 and support[0].kind == "dir"
assert "contracts" in support[0].grantable
assert "backend" not in support[0].grantable
assert "sim_oracle.py" not in support[0].grantable
assert any(surface.path.resolve() == root / "backend" and surface.origin == "oracle" for surface in surfaces)
assert any(surface.path.resolve() == root / "sim_oracle.py" and surface.origin == "oracle" for surface in surfaces)
assert any(token in str(root / "sim_oracle.py") for token in audit_tokens(descriptor)["answer"])
assert not any(name.startswith(("merlin._oot_backends.muon", "merlin._oot_sim_oracles.muon")) for name in sys.modules)
""",
    )


def test_migration_receipt_covers_committed_source_bytes(selected_support):
    """Independent Git reads prove all migrated members, not merely receipt self-consistency."""
    receipt = json.loads((selected_support / "provenance.json").read_text())
    assert receipt["schema"] == "merlin.support_migration.v1"
    assert receipt["target"] == "muon"
    assert receipt["qualification"]["hardware_executed"] is False
    assert receipt["qualification"]["candidate_certification"] is False
    revision = receipt["merlin_source_baseline"]
    assert len(revision) == 40 and all(character in "0123456789abcdef" for character in revision)
    records = receipt["files"]
    assert len(records) == 19
    prefix = "merlin/targets/muon/"
    source_paths = subprocess.run(
        [
            "git",
            "ls-tree",
            "-r",
            "--name-only",
            revision,
            "--",
            prefix + "backend",
            prefix + "contracts",
            prefix + "sim_oracle.py",
        ],
        cwd=repo_root(),
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    ).stdout.splitlines()
    assert len(source_paths) == 19
    assert {record["source"] for record in records} == set(source_paths)
    assert len({record["path"] for record in records}) == 19
    provider_backend_files = {
        path.relative_to(selected_support).as_posix()
        for path in (selected_support / "backend").rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    }
    assert provider_backend_files == {record["path"] for record in records if record["path"].startswith("backend/")}
    for record in records:
        relative = PurePosixPath(record["path"])
        assert not relative.is_absolute() and ".." not in relative.parts
        assert record["source"] == prefix + str(relative)
        assert record["source_revision"] == revision
        original = subprocess.run(
            ["git", "show", revision + ":" + record["source"]],
            cwd=repo_root(),
            capture_output=True,
            check=True,
            timeout=10,
        ).stdout
        payload = selected_support / relative
        assert not payload.is_symlink()
        assert payload.read_bytes() == original
        assert hashlib.sha256(original).hexdigest() == record["sha256"]
