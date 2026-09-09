"""The treatment guard binds SERVED BYTES, not the path spelling used to read them.

A pinned-descriptor directory reaches the shared ``_qa_ws`` through a symlink, so a
resumed run legitimately reads ``targets/<pin>/_qa_ws/...`` where setup recorded
``targets/<target>/_qa_ws/...``.  The guard used to compare whole row dicts, so that
cosmetic difference refused an honest resume -- and a refused resume costs the run its
official public+hidden grade.  It must still fail closed on any real change: content,
presence, or the resolved tool set.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from merlin.common.paths import merlin_dir

sys.path.insert(0, str(merlin_dir() / "experiments/capsule_bench/harness"))
import run_baseline_qa_loop as loop  # noqa: E402


TOOLS = ["merlin_infra", "xdsl_kit"]


def _stage(tmp_path: Path) -> tuple[Path, Path, Path]:
    ws = tmp_path / "targets" / "gemmini" / "_qa_ws" / "r0" / "workspace"
    run_dir = tmp_path / "runs" / "r0"
    bundle_dir = tmp_path / "bundles" / "assisted"
    for d in (ws, run_dir, bundle_dir):
        d.mkdir(parents=True)
    (ws / "TASK.md").write_text("the task\n")
    (run_dir / "TASK.md").write_text("the task\n")
    (bundle_dir / "input_bundle_manifest.yaml").write_text("bundle: assisted\n")
    (run_dir / "input_bundle_manifest.yaml").write_text("bundle: assisted\n")
    (bundle_dir / "tools.txt").write_text("merlin_infra\nxdsl_kit\n")
    return ws, run_dir, bundle_dir


def _record(ws, run_dir, bundle_dir, tools=TOOLS):
    return loop._treatment_snapshot_record(ws, run_dir, bundle_dir, tools)


def test_same_bytes_through_a_symlinked_pin_are_not_drift(tmp_path):
    ws, run_dir, bundle_dir = _stage(tmp_path)
    expected = _record(ws, run_dir, bundle_dir)

    # A pinned descriptor dir whose _qa_ws is a symlink to the real target's.
    pin = tmp_path / "targets" / "gemmini_pin97"
    pin.mkdir(parents=True)
    (pin / "_qa_ws").symlink_to(tmp_path / "targets" / "gemmini" / "_qa_ws")
    ws_via_pin = pin / "_qa_ws" / "r0" / "workspace"
    assert str(ws_via_pin) != str(ws)
    assert (ws_via_pin / "TASK.md").read_text() == (ws / "TASK.md").read_text()

    observed = loop._verify_treatment_snapshot(expected, ws_via_pin, run_dir, bundle_dir, TOOLS)
    assert observed["content_sha256"] == expected["content_sha256"]


def test_changed_served_bytes_still_fail_closed(tmp_path):
    ws, run_dir, bundle_dir = _stage(tmp_path)
    expected = _record(ws, run_dir, bundle_dir)
    (ws / "TASK.md").write_text("the task, but edited\n")
    with pytest.raises(RuntimeError) as err:
        loop._verify_treatment_snapshot(expected, ws, run_dir, bundle_dir, TOOLS)
    assert "served/TASK.md" in str(err.value)


def test_same_length_byte_edit_still_fails_closed(tmp_path):
    """A prompt edit that preserves the byte COUNT must still be caught.

    Size alone is a weak witness -- swapping one character in TASK.md changes what the
    agent is told without moving n_bytes, so the digest has to be what decides.
    """
    ws, run_dir, bundle_dir = _stage(tmp_path)
    expected = _record(ws, run_dir, bundle_dir)
    before = (ws / "TASK.md").read_text()
    after = before.replace("the task", "the tesk")
    assert len(after.encode()) == len(before.encode()) and after != before
    (ws / "TASK.md").write_text(after)
    with pytest.raises(RuntimeError) as err:
        loop._verify_treatment_snapshot(expected, ws, run_dir, bundle_dir, TOOLS)
    assert "served/TASK.md" in str(err.value) or "content_sha256" in str(err.value)


def test_appearing_optional_declaration_still_fails_closed(tmp_path):
    ws, run_dir, bundle_dir = _stage(tmp_path)
    (bundle_dir / "tools.txt").unlink()
    expected = _record(ws, run_dir, bundle_dir)
    (bundle_dir / "tools.txt").write_text("merlin_infra\nxdsl_kit\n")
    with pytest.raises(RuntimeError) as err:
        loop._verify_treatment_snapshot(expected, ws, run_dir, bundle_dir, TOOLS)
    assert "source_bundle/tools.txt" in str(err.value)


def test_changed_tool_set_still_fails_closed(tmp_path):
    ws, run_dir, bundle_dir = _stage(tmp_path)
    expected = _record(ws, run_dir, bundle_dir)
    with pytest.raises(RuntimeError) as err:
        loop._verify_treatment_snapshot(
            expected, ws, run_dir, bundle_dir, [*TOOLS, "rtl_facts"])
    assert "resolved_tool_ids" in str(err.value)
