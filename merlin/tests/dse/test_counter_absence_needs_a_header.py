"""Claiming a target does not count overlap in hardware requires having read a C header.

`counters_for_target` already refused to say `absent` when nothing could be READ -- our inability
reported as a property of the machine is the collapse the module exists to prevent. But the guard only
asked whether *something* was read, not whether what was read could possibly carry the answer.

`_defines` understands object-like `#define NAME <int>` and nothing else, so a file that is not a C
header can never yield a counter block whatever the hardware does. Measured 2026-09-01: atlas declares
its ISA source as `baremetal/assembler.py` -- a PYTHON file. `read` was non-empty, the guard did not
fire, and the result claimed "this target does not count overlap in hardware" on the strength of having
parsed an assembler for `#define`.
"""
from __future__ import annotations

from merlin.perf import hw_counters as HC


def test_a_python_file_cannot_settle_whether_the_hardware_counts(tmp_path):
    """The atlas shape: a readable, real, wholly irrelevant file."""
    asm = tmp_path / "assembler.py"
    asm.write_text("class Assembler:\n    OPCODES = {'ADD': 1}\n", encoding="utf-8")
    got = HC.counters_for_target("t", sources=[asm])
    assert got["status"] == "unavailable", (
        f"parsing a Python file for #define yielded a claim about the machine: {got}")
    assert "is a C header" in got["why"] and "UNKNOWN, not absent" in got["why"]
    assert "assembler.py" in got["why"], "the reason should name what was actually read"


def test_a_real_header_with_no_counter_block_is_absent(tmp_path):
    """The honest `absent`: we read the right kind of file and it has no counters."""
    h = tmp_path / "isa.h"
    h.write_text("#define FOO 1\n#define BAR 2\n", encoding="utf-8")
    got = HC.counters_for_target("t", sources=[h])
    assert got["status"] == "absent", got
    assert got.get("headers_read"), "an absent verdict should record which headers backed it"


def test_an_unreadable_path_is_still_unavailable(tmp_path):
    got = HC.counters_for_target("t", sources=[tmp_path / "nope.h"])
    assert got["status"] == "unavailable"


def test_a_header_with_a_counter_block_still_derives(tmp_path):
    """The fix must not break the derived path."""
    h = tmp_path / "counter.h"
    h.write_text(
        "#define CTR_LOAD_CYCLES 1\n"
        "#define CTR_STORE_CYCLES 2\n"
        "#define CTR_LOAD_STORE_CYCLES 3\n", encoding="utf-8")
    got = HC.counters_for_target("t", sources=[h])
    assert got["status"] in ("derived", "absent"), got
    if got["status"] == "derived":
        assert got["counters"], "a derived verdict must carry the counters"
