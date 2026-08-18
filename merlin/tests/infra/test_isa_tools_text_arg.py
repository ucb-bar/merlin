"""`isa_tools --file` must accept the ISA text itself, not only a path.

The shim documents `--file` as either a path or the listing, and falls back to treating the argument
as text. But the guard it used, `Path(arg).is_file()`, RAISES instead of returning False once the
argument is longer than NAME_MAX (OSError errno 36) or contains a NUL (ValueError). So the fallback
was unreachable for precisely the inputs it existed to serve: any real kernel listing is kilobytes.

Measured live on an atlas arm-4 run -- the agent passed a ~2 KB `.insn`/`.word` listing and the tool
died in argument handling. In a transcript that reads as the agent failing a tool call, not as the
harness being broken, which is why it survived: arm-4's whole point is that the agent uses this
tooling, so a crash here penalises exactly the runs that do the right thing.
"""
from __future__ import annotations

import sys

import pytest

from merlin.common.paths import repo_root

sys.path.insert(0, str(repo_root() / "merlin/experiments/capsule_bench/harness"))
from isa_tools_shim import _text_of  # noqa: E402


KERNEL = "\n".join([
    "  .word 0x0000035f  # VLI_ALL",
    "  .insn i 0x67, 1, x0, x0, 33  # DELAY",
    "  .word 0x0800c077  # VMATPUSH_ACC_BF16_MXU0",
] * 40)


def test_a_long_inline_listing_is_returned_as_text():
    assert len(KERNEL) > 255, "fixture must exceed NAME_MAX or it does not exercise the bug"
    assert _text_of(KERNEL) == KERNEL


def test_the_old_guard_really_did_raise_on_this_input():
    """Pin the root cause, so a refactor back to a bare is_file() fails here."""
    from pathlib import Path
    with pytest.raises(OSError):
        Path(KERNEL).is_file()


def test_a_real_path_is_still_read(tmp_path):
    f = tmp_path / "kernel.S"
    f.write_text(KERNEL)
    assert _text_of(str(f)) == KERNEL


def test_a_short_non_path_string_is_text_not_an_error():
    assert _text_of("  .word 0x1234") == "  .word 0x1234"


def test_an_embedded_nul_does_not_crash():
    weird = "  .word 0x1\x00234"
    assert _text_of(weird) == weird
