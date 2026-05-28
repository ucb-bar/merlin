"""Tests for the Pillar 3 LLM harness orchestration.

These tests exercise harness machinery that does NOT require an
``ANTHROPIC_API_KEY``: pinned-model loading, the system-prompt builder,
the worktree contract, and the safety boundary between baseline tools
and MCP-bridged tools.

Markers: ``integration``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PILLAR3 = REPO_ROOT / "eval" / "paper" / "pillar3_endtoend"
if str(PILLAR3) not in sys.path:
    sys.path.insert(0, str(PILLAR3))

from harness.harness import (  # noqa: E402
    _build_system_prompt,
    _load_pinned_model,
    _system_prompt_with_treatment_addendum,
)

pytestmark = [pytest.mark.integration]


def test_pinned_model_yaml_is_real_pin() -> None:
    pinned = _load_pinned_model()
    model = pinned["model"]
    assert model["vendor"] == "anthropic"
    assert model["family"] == "claude"
    assert not str(model["id"]).startswith("REPLACE"), "pinned_model.yaml still has a placeholder model id"
    assert "claude-" in model["id"], f"unexpected model id {model['id']!r} — should be a Claude snapshot"
    assert model["temperature"] == 0.0
    assert int(model["max_output_tokens"]) > 0
    assert int(pinned["per_run_wall_clock_seconds"]) > 0
    assert int(pinned["seeds_per_cell"]) >= 1
    assert int(pinned["max_turns"]) >= 1


def test_pinned_model_targets_have_oracle_yamls() -> None:
    pinned = _load_pinned_model()
    targets_dir = PILLAR3 / "targets"
    for target in pinned["targets"]:
        assert (
            targets_dir / target / "oracle.yaml"
        ).exists(), f"target {target!r} listed in pinned_model.yaml has no oracle.yaml"


def test_oracle_yamls_are_well_formed_yaml() -> None:
    pinned = _load_pinned_model()
    for target in pinned["targets"]:
        path = PILLAR3 / "targets" / target / "oracle.yaml"
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        for required in ("target", "chipyard_generator", "chipyard_config", "simulator", "workload", "run", "success"):
            assert required in payload, f"{target}/oracle.yaml missing {required!r}"


def test_system_prompt_includes_target_arm_and_oracle_path() -> None:
    oracle = PILLAR3 / "targets" / "nvdla" / "oracle.yaml"
    prompt = _build_system_prompt("nvdla", "control", oracle)
    assert "nvdla" in prompt
    assert "control" in prompt
    assert "oracle.yaml" in prompt
    # The agent must be instructed not to read the oracle.
    assert "Do NOT read the oracle YAML" in prompt
    # ./merlin compliance is part of the contract.
    assert "./merlin" in prompt


def test_treatment_addendum_lists_mcp_tools() -> None:
    base = _build_system_prompt("nvdla", "treatment", PILLAR3 / "targets" / "nvdla" / "oracle.yaml")
    mcp_tools = [
        {"name": "targetgen_get_allowed_patch_surfaces"},
        {"name": "targetgen_list_pipeline_stages"},
    ]
    extended = _system_prompt_with_treatment_addendum(base, mcp_tools)
    assert "TREATMENT" in extended
    assert "targetgen_get_allowed_patch_surfaces" in extended
    assert "targetgen_list_pipeline_stages" in extended
    # The addendum must direct the agent to call the patch-surfaces tool
    # before editing.
    assert "BEFORE editing" in extended


def test_treatment_addendum_no_op_when_no_mcp_tools() -> None:
    base = _build_system_prompt("nvdla", "treatment", PILLAR3 / "targets" / "nvdla" / "oracle.yaml")
    assert _system_prompt_with_treatment_addendum(base, []) == base


def test_harness_refuses_to_overwrite_existing_run(tmp_path: Path) -> None:
    """The harness must never silently overwrite a recorded run."""
    runs_root = tmp_path / "runs"
    cell_dir = runs_root / "nvdla" / "control" / "1"
    cell_dir.mkdir(parents=True)
    (cell_dir / "run_summary.json").write_text(json.dumps({"old": "data"}))

    # Spawn the harness CLI; it must error before doing any work.
    env = dict(os.environ)
    env.pop("ANTHROPIC_API_KEY", None)  # absence should not matter — refusal happens first
    env["PYTHONPATH"] = str(REPO_ROOT / "tools") + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [
            sys.executable,
            str(PILLAR3 / "harness" / "harness.py"),
            "--target",
            "nvdla",
            "--arm",
            "control",
            "--seed",
            "1",
            "--runs",
            str(runs_root),
        ],
        capture_output=True,
        text=True,
        env=env,
        cwd=REPO_ROOT,
        check=False,
    )
    assert proc.returncode != 0
    combined = proc.stdout + proc.stderr
    assert "already exists" in combined or "FileExistsError" in combined


def test_harness_validates_unknown_arm(tmp_path: Path) -> None:
    """argparse should reject an arm value outside {control, treatment}."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT / "tools") + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [
            sys.executable,
            str(PILLAR3 / "harness" / "harness.py"),
            "--target",
            "nvdla",
            "--arm",
            "neither",
            "--seed",
            "1",
            "--runs",
            str(tmp_path / "runs"),
        ],
        capture_output=True,
        text=True,
        env=env,
        cwd=REPO_ROOT,
        check=False,
    )
    assert proc.returncode != 0
    assert "neither" in proc.stderr or "invalid choice" in proc.stderr


def test_harness_requires_api_key_when_arm_valid(tmp_path: Path) -> None:
    """When everything else checks out but ANTHROPIC_API_KEY is missing,
    the harness must surface a clear error rather than crashing inside
    the SDK."""
    env = dict(os.environ)
    env.pop("ANTHROPIC_API_KEY", None)
    env["PYTHONPATH"] = str(REPO_ROOT / "tools") + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [
            sys.executable,
            str(PILLAR3 / "harness" / "harness.py"),
            "--target",
            "nvdla",
            "--arm",
            "control",
            "--seed",
            "999",
            "--runs",
            str(tmp_path / "runs"),
        ],
        capture_output=True,
        text=True,
        env=env,
        cwd=REPO_ROOT,
        check=False,
        timeout=60,
    )
    assert proc.returncode != 0
    combined = proc.stdout + proc.stderr
    assert "ANTHROPIC_API_KEY" in combined


def test_harness_imports_anthropic_sdk() -> None:
    """The harness must import the anthropic SDK (not a shim) at module
    import time so a missing dependency is caught early."""
    import anthropic  # noqa: F401
    from harness import harness as harness_mod  # noqa: F401

    assert hasattr(harness_mod, "anthropic")
    assert hasattr(harness_mod.anthropic, "Anthropic")
