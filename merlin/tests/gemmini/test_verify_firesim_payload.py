from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = (Path(__file__).parents[2] / "experiments/gemmini_perf_bench/scripts"
          / "verify_firesim_payload.py")
SPEC = importlib.util.spec_from_file_location("verify_firesim_payload", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
grade = MODULE.grade


def test_accepts_one_complete_correct_measurement() -> None:
    report = grade("""
MERLIN_METRIC cycles=123
MERLIN_RESULT logits_checked=1000 bad=0 nonfinite=0 top1=258 expected_top1=258
MERLIN_PROFILE measured end rc=0
PASS: exact
""", expected_logits=1000, expected_top1=258)
    assert report["ok"] is True
    assert report["cycles"] == 123


def test_queue_success_cannot_hide_payload_failure() -> None:
    report = grade("""
MERLIN_METRIC cycles=3633774166
MERLIN_RESULT logits_checked=1000 bad=999 nonfinite=0 top1=749 expected_top1=258
MERLIN_PROFILE measured end rc=1
FAIL: output differs
*** FAILED *** (tohost = 1)
""", expected_logits=1000, expected_top1=258)
    assert report["ok"] is False
    assert "explicit payload failure marker" in report["reasons"][-1]
