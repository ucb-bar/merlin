"""Operator launch and release configs never reach a Phase 1 authoring input or release payload.

``examples/gemmini/phase1/launch/`` (the launch config and its operator errata) and
``examples/gemmini/phase0/release/`` (the release preparation config) are operator inputs. The
author sees the errata text only through the harness's archived, re-verified ``ERRATA.md`` copy,
never the file itself. Like the operator compiler patches, these directories stay invisible only
because nothing names them or a parent directory: no generated or retained bundle grant or frozen
host input reaches them, release preparation copies only the authored task resource (a sibling),
and the replayed author sandbox mount table exposes none of their files. None of the checks is
vacuous: the files must exist, and granting a parent does expose them through the same replay.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import sim_toolchain
import yaml

from merlin.common.paths import repo_root
from merlin.targetgen.generate_bundles import generate_bundles
from merlin.targetgen.sandbox import build_sandbox
from merlin.targetgen.sandbox import bwrap as BW
from merlin.targetgen.target_experiment import load_target_experiment

DESCRIPTOR = "examples/gemmini/target/descriptor.yaml"
OPERATOR_DIRS = ("examples/gemmini/phase1/launch", "examples/gemmini/phase0/release")


@pytest.fixture(autouse=True)
def _sim_family_policy_without_its_checkout(monkeypatch, tmp_path_factory):
    """Mount-table replays only; an absent simulator checkout is stood in by an empty root."""
    sim_toolchain.stand_in_absent_chipyard(monkeypatch, tmp_path_factory.mktemp("absent-chipyard"))


@pytest.fixture(scope="module")
def target():
    return load_target_experiment(repo_root() / DESCRIPTOR)


@pytest.fixture(scope="module")
def operator_files():
    files = sorted(path for name in OPERATOR_DIRS for path in (repo_root() / name).rglob("*") if path.is_file())
    names = {path.name for path in files}
    assert {"el4-fire-aligned.yaml", "fresh-empty.md", "fire-aligned-v10.yaml"} <= names, (
        "operator configs are missing; every check below would pass vacuously"
    )
    return files


def _reaches(raw: str) -> bool:
    """Whether a selected path is an operator directory, one of its parents, or inside one."""
    root = repo_root()
    path = Path(raw)
    path = (path if path.is_absolute() else root / path).absolute()
    for name in OPERATOR_DIRS:
        selected = (root / name).absolute()
        if path == selected or selected.is_relative_to(path) or path.is_relative_to(selected):
            return True
    return False


def test_generated_bundles_never_grant_or_freeze_the_operator_configs(target, operator_files):
    bundles = generate_bundles(target)
    assert bundles
    for bundle_id, manifest in bundles.items():
        for key in ("allowed", "host_inputs"):
            for row in manifest.get(key, []):
                assert not _reaches(row["path"]), f"{bundle_id} {key} reaches operator configs: {row['path']}"


def test_retained_bundle_manifests_never_grant_the_operator_configs(operator_files):
    manifests = sorted((repo_root() / "merlin/experiments").glob("**/input_bundle_manifest.yaml"))
    assert manifests
    for manifest in manifests:
        document = yaml.safe_load(manifest.read_text()) or {}
        for key in ("allowed", "host_inputs"):
            for row in document.get(key) or []:
                assert not _reaches(row["path"]), f"{manifest} {key} reaches operator configs: {row['path']}"


def test_release_payload_task_resource_is_a_sibling_of_the_operator_configs(target, operator_files):
    # Release preparation copies the authored task resource into the payload; nothing else of
    # examples/<target>/phase1 or phase0 is a declared resource.
    task = target.resource_path("task")
    assert task.is_dir()
    assert not _reaches(str(task))
    assert not _reaches(target.experiment_resource("task"))
    assert not any(path.is_relative_to(task) for path in operator_files)


def _present(bundle: dict) -> dict:
    """Keep the grants present on this host; an absent local install mounts nothing."""
    root = repo_root()
    return {
        **bundle,
        **{
            key: [row for row in bundle.get(key, []) if (root / row["path"]).exists()]
            for key in ("allowed", "host_inputs")
        },
    }


def test_author_sandbox_never_mounts_the_operator_configs(target, operator_files, tmp_path):
    for bundle_id, generated in generate_bundles(target).items():
        bundle = _present(generated)
        assert any(row["path"] == target.experiment_resource("task") + "/" for row in bundle["allowed"])
        workspace = tmp_path / bundle_id / "workspace"
        workspace.mkdir(parents=True)
        argv = build_sandbox(target, workspace, bundle, _policy_test_live_inputs=True).argv()
        exposed = [path for path in operator_files if BW.is_exposed(argv, path)]
        assert exposed == [], f"{bundle_id} sandbox exposes operator configs: {exposed}"
    # The replay is not vacuous: a parent grant would expose them, and no mask would hide them.
    workspace = tmp_path / "parent-grant" / "workspace"
    workspace.mkdir(parents=True)
    leaky = {"allowed": [{"path": "examples/gemmini/", "mode": "ro"}]}
    argv = build_sandbox(target, workspace, leaky, _policy_test_live_inputs=True).argv()
    assert all(BW.is_exposed(argv, path) for path in operator_files)
