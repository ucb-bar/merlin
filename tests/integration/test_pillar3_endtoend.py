"""Real end-to-end Pillar 3 cell test that does NOT call an LLM.

The full ``run_one_cell`` requires an API key. This test exercises the
non-LLM machinery — worktree creation, MCP server bridge spawning,
oracle execution against the worktree state — by patching the LLM call
to a deterministic stub. The harness's safety contract, transcript
capture, and oracle scoring all run for real.

Skipped when the radiance VCS sim isn't built locally (the only sim we
need for the harness contract test).

Markers: ``integration``, ``slow``, ``chipyard``.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PILLAR3 = REPO_ROOT / "eval" / "paper" / "pillar3_endtoend"
if str(PILLAR3) not in sys.path:
    sys.path.insert(0, str(PILLAR3))

from harness.harness import run_one_cell  # noqa: E402

pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.chipyard,
]


class _StubMessage:
    """Minimal Anthropic SDK Message-shaped object."""

    def __init__(self, content_blocks: list[object], stop_reason: str = "end_turn"):
        self.content = content_blocks
        self.stop_reason = stop_reason
        self.usage = type("Usage", (), {"input_tokens": 0, "output_tokens": 0})()


class _StubToolUseBlock:
    """Mimics anthropic.types.ToolUseBlock surface used by the harness."""

    type = "tool_use"

    def __init__(self, name: str, input_args: dict, block_id: str = "stub"):
        self.name = name
        self.input = input_args
        self.id = block_id

    def model_dump(self) -> dict:
        return {"type": "tool_use", "id": self.id, "name": self.name, "input": self.input}


class _StubTextBlock:
    type = "text"

    def __init__(self, text: str):
        self.text = text

    def model_dump(self) -> dict:
        return {"type": "text", "text": self.text}


class _StubAnthropicClient:
    """Plays a fixed three-step transcript so the harness reaches Done."""

    def __init__(self, *args, **kwargs) -> None:
        self._step = 0
        self.messages = self

    def create(self, **_kwargs) -> _StubMessage:  # type: ignore[no-untyped-def]
        self._step += 1
        if self._step == 1:
            return _StubMessage(
                [
                    _StubTextBlock("I will inspect the worktree first."),
                    _StubToolUseBlock("Read", {"path": "CLAUDE.md"}, block_id="b1"),
                ]
            )
        if self._step == 2:
            return _StubMessage(
                [
                    _StubToolUseBlock(
                        "Shell",
                        {"cmd": ["true"], "timeout_seconds": 30},
                        block_id="b2",
                    ),
                ]
            )
        return _StubMessage(
            [
                _StubToolUseBlock(
                    "Done",
                    {"summary": "stub bring-up complete; oracle will fail by design"},
                    block_id="b3",
                ),
            ],
            stop_reason="tool_use",
        )


def _radiance_sim_built(chipyard_root: Path) -> bool:
    return (chipyard_root / "sims" / "vcs" / "simv-chipyard.harness-RadianceMuonConfig").exists()


def test_run_one_cell_drives_lifecycle_without_real_llm(tmp_path: Path, chipyard_root: Path) -> None:
    if not _radiance_sim_built(chipyard_root):
        pytest.skip("radiance VCS simv not built; cannot exercise full oracle path")

    runs_root = tmp_path / "runs"
    pinned = {
        "model": {
            "id": "claude-opus-4-7",
            "snapshot_date": "2026-04-26",
            "temperature": 0.0,
            "max_output_tokens": 4096,
            "vendor": "anthropic",
            "family": "claude",
        },
        "per_run_wall_clock_seconds": 600,
        "max_turns": 6,
    }

    # The harness reads ANTHROPIC_API_KEY but our stub client ignores it.
    os.environ["ANTHROPIC_API_KEY"] = "test-key-not-used"

    with patch("harness.harness.anthropic.Anthropic", _StubAnthropicClient):
        summary = run_one_cell(
            target="radiance_full",
            arm="control",
            seed=999,
            runs_root=runs_root,
            chipyard_root=chipyard_root,
            repo_root=REPO_ROOT,
            pinned=pinned,
        )

    # Lifecycle assertions — harness ran end-to-end:
    assert summary["target"] == "radiance_full"
    assert summary["arm"] == "control"
    assert summary["seed"] == 999
    assert summary["model"] == "claude-opus-4-7"
    # Stub played 3 turns and called Done.
    assert summary["done_signaled"] is True
    assert summary["n_turns"] >= 3
    assert summary["stop_reason"] == "done"
    # Oracle ran for real against the radiance simv.
    assert "oracle" in summary
    assert summary["oracle"]["sim_returncode"] is not None

    cell_dir = runs_root / "radiance_full" / "control" / "999"
    assert (cell_dir / "transcript.jsonl").exists()
    assert (cell_dir / "run_summary.json").exists()

    transcript_lines = (cell_dir / "transcript.jsonl").read_text().splitlines()
    events = [json.loads(line) for line in transcript_lines if line.strip()]
    kinds = [e["event"] for e in events]
    # We saw the lifecycle markers we expect:
    assert "run_started" in kinds
    assert "tool_call" in kinds
    assert "shell" in kinds
    assert any(e.get("tool") == "Done" for e in events)
    assert "run_finished" in kinds


def test_run_one_cell_treatment_arm_records_stage_context(tmp_path: Path, chipyard_root: Path) -> None:
    """Treatment arm spawns the MCP server and exposes its tools.

    A targetgen tool that returns ``allowed_write_paths`` MUST emit a
    ``stage_context`` event so Pillar 4 can score against the latest
    allowed list.
    """
    if not _radiance_sim_built(chipyard_root):
        pytest.skip("radiance VCS simv not built")

    runs_root = tmp_path / "runs"
    pinned = {
        "model": {
            "id": "claude-opus-4-7",
            "snapshot_date": "2026-04-26",
            "temperature": 0.0,
            "max_output_tokens": 4096,
            "vendor": "anthropic",
            "family": "claude",
        },
        "per_run_wall_clock_seconds": 600,
        "max_turns": 6,
    }

    class _TreatmentStub:
        def __init__(self, *_a, **_kw) -> None:
            self._step = 0
            self.messages = self

        def create(self, **_kw):  # type: ignore[no-untyped-def]
            self._step += 1
            if self._step == 1:
                return _StubMessage(
                    [
                        _StubToolUseBlock(
                            "targetgen_get_allowed_patch_surfaces",
                            {
                                "capability_path": str(
                                    REPO_ROOT / "target_specs" / "examples" / "radiance_muon" / "capability.yaml"
                                ),
                                "stage": "hal_driver",
                            },
                            block_id="b1",
                        ),
                    ]
                )
            return _StubMessage(
                [
                    _StubToolUseBlock("Done", {"summary": "stub"}, block_id="b2"),
                ]
            )

    os.environ["ANTHROPIC_API_KEY"] = "test-key-not-used"
    with patch("harness.harness.anthropic.Anthropic", _TreatmentStub):
        summary = run_one_cell(
            target="radiance_full",
            arm="treatment",
            seed=998,
            runs_root=runs_root,
            chipyard_root=chipyard_root,
            repo_root=REPO_ROOT,
            pinned=pinned,
        )

    assert summary["arm"] == "treatment"
    cell_dir = runs_root / "radiance_full" / "treatment" / "998"
    transcript = [json.loads(line) for line in (cell_dir / "transcript.jsonl").read_text().splitlines() if line.strip()]
    stage_events = [e for e in transcript if e.get("event") == "stage_context"]
    assert stage_events, (
        "treatment arm did not emit a stage_context event after calling " "targetgen_get_allowed_patch_surfaces"
    )
    ev = stage_events[0]
    assert ev["stage"] == "hal_driver"
    assert any("runtime/src/iree/hal/drivers/radiance_muon" in p for p in ev["allowed_write_paths"]), ev[
        "allowed_write_paths"
    ]
