"""A run whose DESIGN asserted certifies nothing, whatever the engine did about it afterwards.

The two elaborated-RTL engines disagree about what to do with a failed `assert`, and the disagreement
is in the dangerous direction. Verilator turns it into `$stop` and exits non-zero, which `run_elf`
already catches. The GSIM model prints the same assertion and KEEPS GOING.

Measured on a real submission that issued a zero-byte mvin: Verilator aborted at
`LoadController.sv:236` with rc=255 and no console; GSIM printed three assertion failures
(`LoadController.scala:192`, `DMACommandTracker.scala:88` and `:89`), then finished `done=1
exit_code=0` with a complete OUT/METRIC/DONE console and 359 reported cycles. Nothing downstream reads
a console for assertions -- `run_elf` decided on the exit code alone -- so that capsule would have been
judged purely on its output bytes, and a kernel the hardware refuses PASSES whenever those bytes happen
to match.

An assertion is the design saying the program did something it does not support. Certifying past one is
the wrong-device hazard in miniature: the numbers describe a machine that would not have run this.
"""

from __future__ import annotations

import pytest

from merlin.runtime.backends import base as _backends


@pytest.fixture(scope="module")
def G():
    return _backends.get_backend("gemmini").gemmini


#: The exact text the GSIM model printed on the measured submission.
_GSIM_CONSOLE = """[gsim-emu] reset done, running (max_cycles=200000000)
Assertion failed: A single mvin instruction must load more than 0 bytes
    at LoadController.scala:192 assert(!(cmd_tracker.io.alloc.fire() && ...))
Assertion failed
    at DMACommandTracker.scala:89 assert(cmds(cmd_id).bytes_left >= io.request_returned.bits.bytes_read)
OUT Y0 16 16 0 0 0 0
METRIC cycles 359
DONE
[gsim-emu] FINISHED: cycles=143928 done=1 exit_code=0
"""


def test_an_asserting_run_is_refused_even_though_the_engine_exited_clean(G):
    with pytest.raises(G.GemminiError, match="the DESIGN asserted"):
        G._refuse_on_rtl_assertion("gsim", _GSIM_CONSOLE, "")


def test_the_refusal_quotes_the_assertions_so_the_cause_is_actionable(G):
    with pytest.raises(G.GemminiError) as excinfo:
        G._refuse_on_rtl_assertion("gsim", _GSIM_CONSOLE, "")
    message = str(excinfo.value)
    assert "must load more than 0 bytes" in message, "the reader needs the design's own sentence"
    assert "DMACommandTracker" in message, "every assertion that fired, not only the first"


def test_an_assertion_on_stderr_is_caught_too(G):
    """Which stream carries it is an engine's choice; the verdict must not depend on that."""
    with pytest.raises(G.GemminiError, match="the DESIGN asserted"):
        G._refuse_on_rtl_assertion("verilator", "", _GSIM_CONSOLE)


def test_a_clean_console_passes_through(G):
    G._refuse_on_rtl_assertion("gsim", "OUT Y0 1 1 7\nMETRIC cycles 3\nDONE\n", "")


def test_the_word_assert_alone_is_not_an_assertion_failure(G):
    """A console that merely MENTIONS assertions -- a kernel name, a log line -- must not be refused:
    a false refusal here fails a conformant submission, which is the opposite error and just as bad."""
    G._refuse_on_rtl_assertion("gsim", "note: assertions compiled in\nOUT Y0 1 1 7\nDONE\n", "")


def test_run_elf_applies_it_after_the_exit_code_check(monkeypatch, G, tmp_path):
    """The hole was that the exit code was the ONLY gate, so the order matters: a clean exit must still
    reach the console check."""
    import stat

    emu = tmp_path / "emu"
    emu.write_text("#!/bin/sh\n", encoding="utf-8")
    emu.chmod(emu.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("MERLIN_GEMMINI_GSIM_EMU", str(emu))
    elf = tmp_path / "k.elf"
    elf.write_text("", encoding="utf-8")

    class _Proc:
        returncode = 0  # the engine says everything was fine
        stdout = _GSIM_CONSOLE
        stderr = ""

    monkeypatch.setattr(G.subprocess, "run", lambda *a, **k: _Proc())
    with pytest.raises(G.GemminiError, match="the DESIGN asserted"):
        G.run_elf(elf, simulator="gsim")
