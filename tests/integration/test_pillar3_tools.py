"""Tests for the Pillar 3 harness baseline tools and MCP bridge.

Exercises the safety contract (forbidden writes, path-escape rejection,
shell wrapping) and the MCP bridge against the real targetgen MCP
server. No LLM, no full bring-up — just the substrate the LLM uses.

Markers: ``integration``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PILLAR3 = REPO_ROOT / "eval" / "paper" / "pillar3_endtoend"
if str(PILLAR3) not in sys.path:
    sys.path.insert(0, str(PILLAR3))

from harness.mcp_bridge import TargetGenMCP, record_stage_context_if_relevant  # noqa: E402
from harness.tools import (  # noqa: E402
    BASELINE_TOOL_NAMES,
    BASELINE_TOOLS,
    FORBIDDEN_WRITE_PREFIXES,
    execute_baseline_tool,
)

pytestmark = [pytest.mark.integration]


def test_baseline_tools_advertise_required_names() -> None:
    names = {t["name"] for t in BASELINE_TOOLS}
    assert names == BASELINE_TOOL_NAMES
    assert "Done" in names
    assert "Read" in names
    assert "Write" in names
    assert "Edit" in names
    assert "Shell" in names


def test_baseline_tools_have_input_schemas() -> None:
    for tool in BASELINE_TOOLS:
        schema = tool["input_schema"]
        assert schema["type"] == "object"
        assert "properties" in schema


def test_read_tool_reads_a_real_file(tmp_path: Path) -> None:
    (tmp_path / "hello.txt").write_text("hello world", encoding="utf-8")
    transcript: list[dict] = []
    text, is_err = execute_baseline_tool("Read", {"path": "hello.txt"}, worktree=tmp_path, transcript=transcript)
    assert not is_err
    assert text == "hello world"
    assert transcript[-1]["event"] == "tool_call"
    assert transcript[-1]["tool"] == "Read"
    assert transcript[-1]["result"] == "ok"


def test_read_tool_rejects_path_escape(tmp_path: Path) -> None:
    transcript: list[dict] = []
    text, is_err = execute_baseline_tool(
        "Read",
        {"path": "../../../etc/passwd"},
        worktree=tmp_path,
        transcript=transcript,
    )
    assert is_err
    assert "ToolViolation" in text or "outside" in text


def test_write_tool_creates_file(tmp_path: Path) -> None:
    transcript: list[dict] = []
    text, is_err = execute_baseline_tool(
        "Write",
        {"path": "subdir/new.txt", "content": "abc"},
        worktree=tmp_path,
        transcript=transcript,
    )
    assert not is_err, text
    assert (tmp_path / "subdir" / "new.txt").read_text() == "abc"


def test_write_tool_rejects_iree_submodule(tmp_path: Path) -> None:
    transcript: list[dict] = []
    text, is_err = execute_baseline_tool(
        "Write",
        {
            "path": "third_party/iree_bar/runtime/foo.c",
            "content": "// hack",
        },
        worktree=tmp_path,
        transcript=transcript,
    )
    assert is_err
    assert "ToolViolation" in text
    assert not (tmp_path / "third_party" / "iree_bar" / "runtime" / "foo.c").exists()


def test_write_tool_rejects_paper_dir(tmp_path: Path) -> None:
    transcript: list[dict] = []
    text, is_err = execute_baseline_tool(
        "Write",
        {"path": "eval/paper/sneaky.txt", "content": "x"},
        worktree=tmp_path,
        transcript=transcript,
    )
    assert is_err
    assert "ToolViolation" in text


def test_edit_tool_replaces_unique_occurrence(tmp_path: Path) -> None:
    target = tmp_path / "f.txt"
    target.write_text("alpha\nbeta\ngamma\n")
    transcript: list[dict] = []
    text, is_err = execute_baseline_tool(
        "Edit",
        {"path": "f.txt", "old_string": "beta", "new_string": "BETA"},
        worktree=tmp_path,
        transcript=transcript,
    )
    assert not is_err, text
    assert target.read_text() == "alpha\nBETA\ngamma\n"


def test_edit_tool_refuses_non_unique_replacement(tmp_path: Path) -> None:
    target = tmp_path / "f.txt"
    target.write_text("a\na\na\n")
    transcript: list[dict] = []
    text, is_err = execute_baseline_tool(
        "Edit",
        {"path": "f.txt", "old_string": "a", "new_string": "b"},
        worktree=tmp_path,
        transcript=transcript,
    )
    assert is_err
    assert "exactly one occurrence" in text
    assert target.read_text() == "a\na\na\n"  # unchanged


def test_shell_tool_runs_real_subprocess(tmp_path: Path) -> None:
    transcript: list[dict] = []
    text, is_err = execute_baseline_tool(
        "Shell",
        {"cmd": ["echo", "pillar3-tools-smoke"]},
        worktree=tmp_path,
        transcript=transcript,
    )
    assert not is_err, text
    assert "pillar3-tools-smoke" in text
    shell_event = next(e for e in transcript if e.get("event") == "shell")
    assert shell_event["returncode"] == 0
    assert shell_event["cmd"] == ["echo", "pillar3-tools-smoke"]


def test_shell_tool_records_failed_command(tmp_path: Path) -> None:
    transcript: list[dict] = []
    text, is_err = execute_baseline_tool(
        "Shell",
        {"cmd": ["bash", "-c", "exit 7"]},
        worktree=tmp_path,
        transcript=transcript,
    )
    assert is_err
    assert "returncode: 7" in text


def test_done_tool_marks_done(tmp_path: Path) -> None:
    transcript: list[dict] = []
    text, is_err = execute_baseline_tool(
        "Done",
        {"summary": "bring-up complete"},
        worktree=tmp_path,
        transcript=transcript,
    )
    assert not is_err
    assert text.startswith("Done acknowledged")
    assert transcript[-1]["result"] == "done"


def test_unknown_tool_rejected(tmp_path: Path) -> None:
    transcript: list[dict] = []
    text, is_err = execute_baseline_tool("Bogus", {"x": 1}, worktree=tmp_path, transcript=transcript)
    assert is_err
    assert "unknown tool" in text


def test_forbidden_prefixes_include_critical_paths() -> None:
    """Documentation: every prefix listed must be considered a write
    boundary. If a future change adds a new submodule, list it here."""
    assert "third_party/iree_bar/" in FORBIDDEN_WRITE_PREFIXES
    assert "eval/paper/" in FORBIDDEN_WRITE_PREFIXES
    assert ".git/" in FORBIDDEN_WRITE_PREFIXES


# --- MCP bridge tests (use the real targetgen mcp server) ---


def test_mcp_bridge_initialises_and_lists_tools() -> None:
    with TargetGenMCP(REPO_ROOT) as mcp:
        names = mcp.tool_names()
    assert "targetgen_list_pipeline_stages" in names
    assert "targetgen_get_modification_map" in names
    assert "targetgen_get_allowed_patch_surfaces" in names


def test_mcp_bridge_calls_pipeline_stages() -> None:
    with TargetGenMCP(REPO_ROOT) as mcp:
        text, is_err = mcp.call_tool("targetgen_list_pipeline_stages", {})
    assert not is_err, text
    payload = json.loads(text)
    assert len(payload["stages"]) == 9


def test_mcp_bridge_returns_error_for_bad_args() -> None:
    with TargetGenMCP(REPO_ROOT) as mcp:
        text, is_err = mcp.call_tool(
            "targetgen_get_allowed_patch_surfaces",
            {"capability_path": "no-such.yaml", "stage": "hal_driver"},
        )
    # Either is_err is True or the text payload starts with "ToolError" /
    # "Input validation error" — both flag the failure.
    assert is_err or text.startswith(("ToolError", "Input validation error"))


def test_record_stage_context_emits_event_when_relevant() -> None:
    transcript: list[dict] = []
    payload = {
        "target": "radiance_muon",
        "stage": "hal_driver",
        "applies": True,
        "allowed_write_paths": ["runtime/src/iree/hal/drivers/radiance_muon/"],
        "forbidden_unless_approved": ["third_party/iree_bar/"],
    }
    record_stage_context_if_relevant(
        "targetgen_get_allowed_patch_surfaces",
        {"capability_path": "x"},
        json.dumps(payload),
        transcript,
    )
    assert len(transcript) == 1
    ev = transcript[0]
    assert ev["event"] == "stage_context"
    assert ev["stage"] == "hal_driver"
    assert "runtime/src/iree/hal/drivers/radiance_muon/" in ev["allowed_write_paths"]


def test_record_stage_context_ignores_other_tools() -> None:
    transcript: list[dict] = []
    record_stage_context_if_relevant(
        "targetgen_list_pipeline_stages",
        {},
        json.dumps({"stages": ["x"]}),
        transcript,
    )
    assert transcript == []


def test_record_stage_context_handles_non_json_payload() -> None:
    transcript: list[dict] = []
    record_stage_context_if_relevant(
        "targetgen_get_allowed_patch_surfaces",
        {"x": 1},
        "ToolError: bad input",
        transcript,
    )
    assert transcript == []
