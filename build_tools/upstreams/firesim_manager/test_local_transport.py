"""Pinned opt-in public manager patch and actual unprivileged native controls."""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest

OWNER = Path(__file__).resolve().parent
SELECTION = json.loads((OWNER / "source-pins.json").read_text())


def verify_original(root):
    for item in SELECTION["original_modified_files"]:
        path = root / item["path"]
        if path.is_symlink() or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError("original public manager source pin differs")
    if any((root / name).exists() for name in SELECTION["new_files"]):
        raise ValueError("new transport source already exists")


@pytest.fixture
def public(tmp_path):
    checkout = os.environ.get("MERLIN_TEST_PUBLIC_FIRESIM_CHECKOUT")
    if not checkout:
        pytest.skip("explicit pinned public manager checkout required")
    environment = {"PATH": "/usr/bin:/bin", "LANG": "C"}
    revision = SELECTION["public_commit"]
    members = (
        subprocess.run(
            ["/usr/bin/git", "ls-tree", "-r", "--name-only", revision, "deploy"],
            cwd=checkout,
            env=environment,
            check=True,
            capture_output=True,
            timeout=10,
        )
        .stdout.decode()
        .splitlines()
    )
    selected = [
        name
        for name in members
        if name.endswith(".py") and name.split("/")[1] in {"awstools", "buildtools", "runtools", "util"}
    ]
    selected.extend(("sourceme-manager.sh", "deploy/firesim", "deploy/run-farm-recipes/externally_provisioned.yaml"))
    raw = subprocess.run(
        ["/usr/bin/git", "archive", revision, *selected],
        cwd=checkout,
        env=environment,
        capture_output=True,
        check=True,
        timeout=10,
    ).stdout
    source = tmp_path / "public-manager"
    source.mkdir()
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as archive:
        for member in archive:
            if member.isdir():
                continue
            assert member.isfile() and member.name in selected
            path = source / member.name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(archive.extractfile(member).read())
            path.chmod(member.mode)
    verify_original(source)
    return source


def test_changed_original_source_refuses_before_patch(public):
    path = public / SELECTION["original_modified_files"][0]["path"]
    path.write_bytes(path.read_bytes() + b"\n# independently changed source\n")
    with pytest.raises(ValueError, match="source pin differs"):
        verify_original(public)
    assert not (public / SELECTION["new_files"][0]).exists()


def test_preexisting_transport_refuses(public):
    path = public / SELECTION["new_files"][0]
    path.write_text("raise RuntimeError('unselected transport')\n")
    with pytest.raises(ValueError, match="already exists"):
        verify_original(public)


def test_actual_public_manager_local_controls(public, tmp_path):
    python = os.environ.get("MERLIN_TEST_FIRESIM_MANAGER_PYTHON")
    original_deploy = os.environ.get("MERLIN_TEST_MANAGER_CONFIG_DEPLOY")
    if not python or not original_deploy:
        pytest.skip("explicit native manager Python and owned source-bound config fixture required")
    verify_original(public)
    subprocess.run(
        ["/usr/bin/patch", "--batch", "--fuzz=0", "-p1", "-i", str(OWNER / "local-command-transport.patch")],
        cwd=public,
        env={"PATH": "/usr/bin:/bin", "LANG": "C"},
        capture_output=True,
        check=True,
        timeout=10,
    )
    deploy = public / "deploy"
    original = Path(original_deploy)
    for name in ("config_hwdb.yaml", "config_build_recipes.yaml", "config_runtime.yaml", "selected-local-recipe.yaml"):
        shutil.copyfile(original / name, deploy / name)
    shutil.copytree(original / "workloads", deploy / "workloads")
    root = tmp_path / "controls"
    root.mkdir()
    home = root / "empty-home"
    home.mkdir(mode=0o700)
    output = root / "actual-controls.json"
    fixture = root / "fixture.json"
    fixture.write_text(
        json.dumps(
            {
                "root": str(root),
                "recipe": str(deploy / "selected-local-recipe.yaml"),
                "runtime": str(deploy / "config_runtime.yaml"),
                "hwdb": str(deploy / "config_hwdb.yaml"),
                "build_recipes": str(deploy / "config_build_recipes.yaml"),
                "output": str(output),
            }
        )
    )
    environment = {
        "PATH": "/usr/bin:/bin",
        "LANG": "C",
        "HOME": str(home),
        "USER": "owned-local-manager-control",
        "PYTHONPATH": str(deploy),
        "PYTHONDONTWRITEBYTECODE": "1",
        "AWS_EC2_METADATA_DISABLED": "true",
        "AWS_CONFIG_FILE": str(home / "absent-config"),
        "AWS_SHARED_CREDENTIALS_FILE": str(home / "absent-credentials"),
    }
    result = subprocess.run(
        [python, str(OWNER / "native_controls.py"), str(fixture)],
        cwd=deploy,
        env=environment,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr.decode()
    report = json.loads(output.read_text())
    assert len(report["outcomes"]) == 30
    assert all(row["outcome"] == "passed original assertion" for row in report["outcomes"])
    assert len(report["actual_native_process_events"]) >= 10
    assert "physical timers" in report["unknowns"]
    assert report["complete_original_roster"][0] == "localhost"
