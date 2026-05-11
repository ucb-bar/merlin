from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = REPO_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import merlin  # noqa: E402
from raycp import service as ray_service  # noqa: E402

import ray_cmd  # noqa: E402


def test_merlin_registers_ray_subcommand():
    parser = argparse.ArgumentParser()
    merlin.setup_parser(parser)

    args = parser.parse_args(["ray", "resources", "list"])

    assert args.command == "ray"


def test_ray_jobs_submit_blocks_without_ray_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    parser = argparse.ArgumentParser()
    ray_cmd.setup_parser(parser)
    monkeypatch.setattr(ray_service.shutil, "which", lambda _: None)
    target_dir = tmp_path / "target"
    target_dir.mkdir()
    args = parser.parse_args(
        [
            "--state-root",
            str(tmp_path / "ray-state"),
            "jobs",
            "submit",
            "--target",
            "nvidia_vulkan_ada",
            "--workflow",
            "targetgen_execute",
            "--target-dir",
            str(target_dir),
            "--command",
            "conda run -n merlin-dev uv run tools/merlin.py targetgen execute --engine local --from-dir /tmp/fake",
        ]
    )

    assert ray_cmd.main(args) == 0

    runs_dir = tmp_path / "ray-state" / "runs"
    run_dirs = sorted(runs_dir.glob("*"))
    assert run_dirs
    run_record = json.loads((run_dirs[0] / "run_record.json").read_text(encoding="utf-8"))
    assert run_record["status"] == "blocked"
    assert "ray" in run_record["message"].lower()


def test_resource_lease_lifecycle(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    parser = argparse.ArgumentParser()
    ray_cmd.setup_parser(parser)
    reserve_args = parser.parse_args(
        [
            "--state-root",
            str(tmp_path / "ray-state"),
            "resources",
            "reserve",
            "firesim_u250",
            "--owner",
            "pytest",
            "--metadata",
            "suite=smoke",
        ]
    )
    assert ray_cmd.main(reserve_args) == 0
    reserve_output = capsys.readouterr().out
    assert "Lease ID:" in reserve_output
    lease_id = reserve_output.split("Lease ID:", 1)[1].splitlines()[0].strip()

    list_args = parser.parse_args(["--state-root", str(tmp_path / "ray-state"), "resources", "list"])
    assert ray_cmd.main(list_args) == 0
    list_output = capsys.readouterr().out
    assert "firesim_u250" in list_output
    assert "active" in list_output

    release_args = parser.parse_args(["--state-root", str(tmp_path / "ray-state"), "resources", "release", lease_id])
    assert ray_cmd.main(release_args) == 0
    release_output = capsys.readouterr().out
    assert "released" in release_output


def test_artifacts_cli_lists_indexed_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys):
    monkeypatch.setattr(ray_service.shutil, "which", lambda _: None)
    state_root = tmp_path / "ray-state"
    target_dir = tmp_path / "target"
    target_dir.mkdir()
    (target_dir / "support_plan.json").write_text('{"ok": true}\n', encoding="utf-8")
    run_record = ray_service.submit_job(
        state_root=state_root,
        target="saturn_opu_v128",
        workflow="targetgen_execute",
        source="pytest",
        command=["echo", "hello"],
        target_dir=target_dir,
    )
    parser = argparse.ArgumentParser()
    ray_cmd.setup_parser(parser)
    args = parser.parse_args(["--state-root", str(state_root), "artifacts", "list", run_record.run_id])

    assert ray_cmd.main(args) == 0
    output = capsys.readouterr().out
    assert "support_plan.json" in output
