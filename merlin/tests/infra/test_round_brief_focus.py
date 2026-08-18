"""The cross-round brief must point the agent at the FAILING capsules (focus) while locking the passing
ones (don't break) — a prioritization derived from the redacted verdict, never a restriction and never a
golden leak.
"""
from __future__ import annotations

import json
import sys

import pytest

from merlin.common.paths import merlin_dir

_HARNESS = merlin_dir() / "experiments" / "capsule_bench" / "harness"


@pytest.fixture(scope="module")
def rb():
    if str(_HARNESS) not in sys.path:
        sys.path.insert(0, str(_HARNESS))
    import round_brief  # noqa: PLC0415
    return round_brief


def test_focus_lists_failing_and_locks_passing(rb):
    v = {"per_capsule": [
        {"capsule": "R0_gemm_fp32", "status": "pass"},
        {"capsule": "R8_flash_attention_mx", "status": "fail", "failure_plane": "numeric", "mismatch_count": 12},
        {"capsule": "RF0_rmsnorm_qkv_fp16", "status": "incomplete", "plane": "compile"},
    ]}
    sec = "\n".join(rb._focus_section(v))
    assert "Focus THIS round" in sec
    assert "R8_flash_attention_mx" in sec and "mismatch_count 12" in sec   # failing, with detail
    assert "RF0_rmsnorm_qkv_fp16" in sec
    assert "Passing (locked)" in sec and "R0_gemm_fp32" in sec              # passing locked, not reworked
    assert "R0_gemm_fp32" not in sec.split("Passing (locked)")[0]          # R0 not in the failing list


def test_no_focus_when_all_pass(rb):
    v = {"per_capsule": [{"capsule": "R0", "status": "pass"}, {"capsule": "R1", "status": "pass"}]}
    assert rb._focus_section(v) == []                                       # nothing to focus → no noise


def test_no_focus_without_verdict(rb):
    assert rb._focus_section({}) == []


def test_build_includes_focus(rb, tmp_path):
    run_dir = tmp_path / "run"
    (run_dir / "qa_history").mkdir(parents=True)
    (run_dir / "qa_history" / "verdict_round_1.json").write_text(json.dumps({
        "n_passed": 1, "n_capsules": 2, "first_failure_planes": {"numeric": 1},
        "per_capsule": [
            {"capsule": "R0_gemm_fp32", "status": "pass"},
            {"capsule": "R8_flash_attention_mx", "status": "fail", "failure_plane": "numeric",
             "mismatch_count": 7}]}))
    ws = tmp_path / "ws"
    ws.mkdir()
    brief = rb.build(run_dir, ws, 1)
    assert "Focus THIS round" in brief and "R8_flash_attention_mx" in brief
    assert "Progress log" in brief                                          # existing content preserved
