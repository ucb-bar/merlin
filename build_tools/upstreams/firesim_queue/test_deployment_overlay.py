"""Disposable queue overlay controls; no shared state, FPGA or real lifecycle."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import shutil
import subprocess
import unittest
from pathlib import Path

import pytest

OWNER = Path(__file__).resolve().parent
SOURCE_SHA = "af721197a5f6a5ffd0a2a46da2555c52026c2f2f97eafc84c5dcd976406d256a"
PATCHED_SHA = "2caf88d819c5c58bd84f2c8dcc3cd8d722f1c9bb5b28a60bb7326c469cdc6c42"
spec = importlib.util.spec_from_file_location("_original_committed_queue_controls", OWNER / "test_committed_inputs.py")
controls = importlib.util.module_from_spec(spec)
spec.loader.exec_module(controls)


def apply_source(original, destination):
    if hashlib.sha256(original.read_bytes()).hexdigest() != SOURCE_SHA:
        raise ValueError("selected deployment queue source bytes differ")
    shutil.copyfile(original, destination / "firesim_queue.py")
    subprocess.run(
        ["/usr/bin/patch", "--batch", "--fuzz=0", "-p1", "-i", str(OWNER / "deployment-overlay.patch")],
        cwd=destination,
        env={"PATH": "/usr/bin:/bin", "LANG": "C"},
        capture_output=True,
        check=True,
        timeout=10,
    )
    assert hashlib.sha256((destination / "firesim_queue.py").read_bytes()).hexdigest() == PATCHED_SHA


@pytest.fixture
def queue(tmp_path, monkeypatch):
    selected = os.environ.get("MERLIN_TEST_QUEUE_DEPLOY_SOURCE")
    if not selected:
        pytest.skip("explicit pinned deployment queue source required")
    install = tmp_path / "install"
    install.mkdir()
    apply_source(Path(selected), install)
    shutil.copyfile(OWNER.parents[2] / "src/merlin/common/pinned_files.py", install / "pinned_files.py")
    monkeypatch.setenv("FIRESIM_QUEUE_ROOT", str(tmp_path / "queue"))
    monkeypatch.setenv("FIRESIM_QUEUE_HWDB_SNAPSHOT_ROOT", str(tmp_path / "private"))
    spec = importlib.util.spec_from_file_location("_selected_overlay_queue", install / "firesim_queue.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.POLL_INTERVAL_SECONDS = 0.01
    module.HEARTBEAT_INTERVAL_SECONDS = 0.01
    monkeypatch.setattr(module, "_require_shared_group", lambda: None)
    return module


def manager_source(tmp_path):
    deploy = tmp_path / "source/deploy"
    deploy.mkdir(parents=True)
    (deploy.parent / "platforms").mkdir()
    (deploy.parent / "platforms/public-script.py").write_text("# original selected platform bytes\n")
    (deploy / "firesim").write_text("# selected CLI bytes\n")
    (deploy / "workloads/control").mkdir(parents=True)
    return deploy


def test_changed_source_refuses_before_patch(tmp_path):
    changed = tmp_path / "changed.py"
    changed.write_text("raise RuntimeError('foreign queue')\n")
    with pytest.raises(ValueError, match="source bytes differ"):
        apply_source(changed, tmp_path)
    assert not (tmp_path / "firesim_queue.py").exists()


def test_original_platform_sibling_and_private_writers(queue, tmp_path):
    deploy = manager_source(tmp_path)
    work = tmp_path / "owned-job"
    work.mkdir()
    overlay = queue._deploy_overlay(work, deploy, "control", "control.elf")
    assert (overlay.parent / "platforms").resolve() == (deploy.parent / "platforms").resolve()
    assert (overlay.parent / "platforms/public-script.py").read_bytes() == (
        deploy.parent / "platforms/public-script.py"
    ).read_bytes()
    assert (overlay / "firesim").resolve() == deploy / "firesim"
    for name in ("logs", "results-workload", "workloads/control", "generated-topology-diagrams"):
        assert (overlay / name).is_dir() and not (overlay / name).is_symlink()
    assert queue._deploy_overlay(work, deploy, "control", "control.elf") == overlay


@pytest.mark.parametrize("kind", ["foreign-link", "existing-directory", "missing-source", "source-file"])
def test_unproved_platform_sibling_refuses(queue, tmp_path, kind):
    deploy = manager_source(tmp_path)
    work = tmp_path / "owned-job"
    work.mkdir()
    if kind == "foreign-link":
        foreign = tmp_path / "excluded-owned-platform"
        foreign.mkdir()
        (work / "platforms").symlink_to(foreign, target_is_directory=True)
    elif kind == "existing-directory":
        (work / "platforms").mkdir()
    else:
        shutil.rmtree(deploy.parent / "platforms")
        if kind == "source-file":
            (deploy.parent / "platforms").write_text("wrong source type")
    with pytest.raises((RuntimeError, FileNotFoundError)):
        queue._deploy_overlay(work, deploy, "control", "control.elf")


def test_cross_user_actual_process_uses_overlay_cli_and_keeps_lock(queue, tmp_path, monkeypatch):
    original_submit = queue.cmd_runworkload_full
    original_access = os.access
    selected_deploy = tmp_path / "chipyard/sims/firesim/deploy"

    def prepare(args):
        (selected_deploy.parent / "platforms").mkdir()
        selected = controls.FAKE_FIRESIM.replace(
            "command = sys.argv[-1]",
            "os.chdir(pathlib.Path(__file__).parent)\ncommand = sys.argv[-1]",
        )
        selected = selected.replace(
            '{"command": command, "lock_held": held}',
            '{"command": command, "lock_held": held, "cwd": str(pathlib.Path.cwd()), "cli": __file__}',
        )
        (selected_deploy / "firesim").write_text(selected)
        (selected_deploy / "firesim").chmod(0o755)
        # Bare PATH lookup would invoke a distinct refused launcher.
        (tmp_path / "bin/firesim").write_text("#!/bin/bash\nexit 77\n")
        return original_submit(args)

    monkeypatch.setattr(queue, "cmd_runworkload_full", prepare)
    monkeypatch.setattr(
        os, "access", lambda path, mode: False if Path(path) == selected_deploy else original_access(path, mode)
    )

    def update_fixture(items):
        config = tmp_path / "fake.json"
        values = json.loads(config.read_text())
        values["staged"] = str(queue.JOBS_DIR / "1/deploy_overlay/workloads/control/control.elf")
        config.write_text(json.dumps(values))

    rc, row, events, _ = controls.submit_and_run(queue, tmp_path, "positive", after_submit=update_fixture)
    assert rc == 0 and row["state"] == "DONE"
    assert [event["command"] for event in events] == ["kill", "infrasetup", "runworkload", "kill"]
    assert all(event["lock_held"] for event in events)
    overlay = queue.JOBS_DIR / "1/deploy_overlay"
    assert all(event["cwd"] == str(overlay) and event["cli"] == str(overlay / "firesim") for event in events)


def test_same_user_legacy_path_unchanged(queue, tmp_path):
    rc, row, events, _ = controls.submit_and_run(queue, tmp_path, "legacy", committed=False)
    assert rc == 0 and row["state"] == "DONE"
    assert [event["command"] for event in events] == ["kill", "infrasetup", "runworkload", "kill"]


@pytest.mark.parametrize(
    "name",
    [
        "test_every_lifecycle_command_uses_exact_readonly_snapshot",
        "test_source_tamper_after_submit_fails_before_any_firesim_call",
        "test_untrusted_owner_cannot_authorize_parent_by_mode",
        "test_private_snapshot_tamper_between_commands_stops_lifecycle",
    ],
)
def test_original_neutral_hwdb_lifecycle(queue, name):
    selected = os.environ.get("MERLIN_TEST_QUEUE_ORIGINAL_TESTS")
    if not selected:
        pytest.skip("explicit original neutral lifecycle test selection required")
    path = Path(selected)
    assert (
        hashlib.sha256(path.read_bytes()).hexdigest()
        == "97abd06301b9f4c7fb8e9485e1155e10aa5a5d78991b772687a6b4fc7e10964e"
    )
    spec = importlib.util.spec_from_file_location("_original_overlay_hwdb_controls", path)
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    original.QUEUE_MODULE = Path(queue.__file__)
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream).run(unittest.TestSuite([original.HwdbSnapshotTest(name)]))
    assert result.testsRun == 1 and not result.skipped
    assert result.wasSuccessful(), stream.getvalue()


def test_selected_actual_public_manager_help(queue, tmp_path):
    selected = os.environ.get("MERLIN_TEST_QUEUE_PUBLIC_MANAGER_DEPLOY")
    python = os.environ.get("MERLIN_TEST_FIRESIM_MANAGER_PYTHON")
    if not selected or not python:
        pytest.skip("explicit independently selected public manager and Python required")
    deploy = Path(selected)
    work = tmp_path / "owned-public-cli"
    work.mkdir()
    overlay = queue._deploy_overlay(work, deploy, "owned-public-cli", "control.elf")
    result = subprocess.run(
        [python, str(overlay / "firesim"), "--help"],
        cwd=overlay,
        env={
            "PATH": "/usr/bin:/bin",
            "HOME": str(work),
            "USER": "owned-public-constructor-control",
            "LANG": "C",
            "PYTHONDONTWRITEBYTECODE": "1",
            "AWS_EC2_METADATA_DISABLED": "true",
        },
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr.decode()
    assert b"FireSim Simulation Manager" in result.stdout
    assert (overlay.parent / "platforms").resolve() == (deploy.parent / "platforms").resolve()


def test_selected_actual_constructor_writes_private_topology(queue, tmp_path):
    selected = os.environ.get("MERLIN_TEST_QUEUE_PUBLIC_MANAGER_DEPLOY")
    python = os.environ.get("MERLIN_TEST_FIRESIM_MANAGER_PYTHON")
    if not selected or not python:
        pytest.skip("explicit independently selected public manager and Python required")
    deploy = Path(selected)
    # Private source copy keeps the actual selected code and original config.
    # Supply an existing nonwritable original diagram directory, which the
    # ordinary public constructor must never alias as its writable product.
    source = tmp_path / "public-source/deploy"
    shutil.copytree(deploy, source, symlinks=True)
    (source.parent / "platforms").symlink_to(deploy.parent / "platforms", target_is_directory=True)
    original_diagrams = source / "generated-topology-diagrams"
    original_diagrams.mkdir(exist_ok=True)
    original_rows = {path.name: path.read_bytes() for path in original_diagrams.iterdir() if path.is_file()}
    original_diagrams.chmod(0o550)
    work = tmp_path / "owned-public-constructor"
    work.mkdir()
    overlay = queue._deploy_overlay(work, source, "owned-public-constructor", "control.elf")
    code = """import argparse, json, pathlib, sys
sys.path.insert(0, str(pathlib.Path.cwd()))
from runtools.runtime_config import RuntimeConfig
RuntimeConfig(argparse.Namespace(hwdbconfigfile=sys.argv[2], runtimeconfigfile=sys.argv[1],
    overrideconfigdata='', task='runworkload', buildrecipesconfigfile=sys.argv[3]))
print(json.dumps({'cwd':str(pathlib.Path.cwd())}))
"""
    result = subprocess.run(
        [
            python,
            "-c",
            code,
            str(deploy / "config_runtime.yaml"),
            str(deploy / "config_hwdb.yaml"),
            str(deploy / "config_build_recipes.yaml"),
        ],
        cwd=overlay,
        env={
            "PATH": str(Path(python).parent) + ":/usr/bin:/bin",
            "HOME": str(work),
            "USER": "owned-public-constructor-control",
            "LANG": "C",
            "PYTHONDONTWRITEBYTECODE": "1",
            "AWS_EC2_METADATA_DISABLED": "true",
            "AWS_CONFIG_FILE": str(work / "absent-config"),
            "AWS_SHARED_CREDENTIALS_FILE": str(work / "absent-credentials"),
        },
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr.decode()
    generated = overlay / "generated-topology-diagrams"
    assert generated.is_dir() and not generated.is_symlink()
    assert list(generated.glob("*.gv")) and list(generated.glob("*.pdf"))
    assert {path.name: path.read_bytes() for path in original_diagrams.iterdir() if path.is_file()} == original_rows
