"""The converse driver's tool-output compaction must NEVER cost the model debug feedback.

`bedrock_agent._extract_output` compacts a long `run_bash` result to save tokens. These tests prove it
preserves every decisive signal — errors, failures, and grade lines — that the model needs to iterate, and
that it strictly beats the old flat head-truncate (which silently dropped the tail AND all stderr on a long
stdout). Short output must pass through verbatim (full fidelity).
"""
from __future__ import annotations

import sys

import pytest

from merlin.common.paths import merlin_dir

_HARNESS = merlin_dir() / "experiments" / "capsule_bench" / "harness"


@pytest.fixture(scope="module")
def be():
    if str(_HARNESS) not in sys.path:
        sys.path.insert(0, str(_HARNESS))
    import bedrock_agent  # noqa: PLC0415 — loaded off the import-isolated harness path
    return bedrock_agent


def _quiet(n):
    # filler lines with NO signal token, so they are the part that may be elided
    return "\n".join(f"quiet row {i} :: routine progress output" for i in range(n))


def test_short_output_is_verbatim(be):
    out = be._extract_output("hello\nworld", "")
    assert out == "hello\nworld"                       # no elision, full fidelity
    out2 = be._extract_output("stdout body", "a stderr note")
    assert "stdout body" in out2 and "a stderr note" in out2


def test_error_at_end_survives(be):
    # the classic flat-[:6000] bug: a long stdout whose error is the LAST line
    stdout = _quiet(400) + "\nERROR: undefined symbol radiance_kernel"
    out = be._extract_output(stdout, "")
    assert "ERROR: undefined symbol radiance_kernel" in out
    assert len(out) < len(stdout)                      # it did compact


def test_error_in_middle_survives(be):
    stdout = _quiet(200) + "\ncompile error: cannot lower attention_mx\n" + _quiet(200)
    out = be._extract_output(stdout, "")
    assert "compile error: cannot lower attention_mx" in out


def test_stderr_preserved_even_with_huge_stdout(be):
    stdout = _quiet(600)                               # big stdout that alone exceeds the cap
    stderr = "fatal: linker error: missing tohost symbol"
    out = be._extract_output(stdout, stderr)
    assert "fatal: linker error: missing tohost symbol" in out   # old code would have dropped this entirely


def test_grade_signal_survives(be):
    stdout = _quiet(200) + "\ncapsule R5_mx_tile_mxfp8 cycles=438326 PASS cos=1.0\n" + _quiet(200)
    out = be._extract_output(stdout, "")
    assert "cycles=438326" in out and "PASS" in out


def test_elision_is_transparent(be):
    out = be._extract_output(_quiet(500), "")
    assert "elided" in out and ("grep" in out or "tail" in out)   # tells the model how to see the rest


def test_all_signal_lines_kept_up_to_cap(be):
    errs = [f"error: capsule C{i} mismatch max_rel=0.5" for i in range(20)]
    stdout = _quiet(150) + "\n" + "\n".join(errs) + "\n" + _quiet(150)
    out = be._extract_output(stdout, "")
    kept = sum(1 for e in errs if e in out)
    assert kept == len(errs)                           # every error line retained (under max_signal)
