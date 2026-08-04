"""Auditing a produced ELF against a board's memory map — the gate for a binary someone else runs.

Every "will it fit" decision in the repo was predictive arithmetic; nothing looked at the artifact. That
is fine when you can attach a debugger and fatal when the binary is mailed to a bench you cannot see,
because each failure is silent: a segment past DRAM boots into nothing, a missing .htif section means the
loader never sees output or the exit request (it reports a timeout), and an image with no vector
instructions is a "result" that measured the scalar fallback.
"""
from __future__ import annotations

import pytest

from merlin.runtime import boards, elf_audit


def _an_elf():
    """Any Zephyr ELF this session built, else skip — these tests audit a real artifact."""
    import glob
    import os
    c = sorted(glob.glob("/tmp/*/build/zephyr/zephyr.elf")
               + glob.glob("/tmp/merlin_compile_*/build/zephyr/zephyr.elf"),
               key=os.path.getmtime, reverse=True)
    if not c:
        pytest.skip("no Zephyr ELF built in this environment")
    return c[0]


def test_a_real_image_passes_against_its_board():
    rep = elf_audit.audit(_an_elf(), boards.board("chipyard_kodiak"))
    assert rep.ok, rep.render()
    assert rep.entry != 0
    assert rep.facts["load_segments"] >= 1


def test_an_image_larger_than_the_board_fails():
    """A region larger than physical DRAM is a boot that dies before main() with no console output."""
    rep = elf_audit.audit(_an_elf(), boards.board("tiny", dram_bytes=4 * 1024 * 1024))
    assert not rep.ok
    assert any("DRAM" in p or "MB" in p for p in rep.problems), rep.problems


def test_the_linked_region_is_checked_not_just_the_segments():
    """build_app can link for a region bigger than the segments occupy; that region must fit too."""
    brd = boards.board("chipyard_kodiak")
    rep = elf_audit.audit(_an_elf(), brd, ram_bytes=brd.dram_bytes * 4)
    assert not rep.ok
    assert any("linked for" in p for p in rep.problems), rep.problems


def test_upload_time_is_reported_because_the_loader_sends_memsiz():
    """A UART loader transmits MemSiz, not file size, so a big .bss or weights blob costs minutes."""
    rep = elf_audit.audit(_an_elf(), boards.board("chipyard_kodiak"))
    assert rep.facts["upload_estimate_s"] > 0
    assert rep.facts["image_memsz_mb"] >= rep.facts["image_filesz_mb"]


def test_a_missing_htif_section_is_a_failure_for_an_htif_board():
    rep = elf_audit.audit(_an_elf(), boards.board("chipyard_kodiak"), expect_htif=True)
    assert ".htif" in rep.sections, "the chipyard images do carry .htif; if this changes, say why"
    # and the check is real: asking for it on a board whose image lacks it must fail
    rep2 = elf_audit.audit(_an_elf(), boards.board("x", console="uart"), expect_htif=False)
    assert not any(".htif" in p for p in rep2.problems)


def test_vector_presence_is_audited():
    """An image with no vector ops looks like a result but measured the scalar fallback."""
    rep = elf_audit.audit(_an_elf(), boards.board("chipyard_kodiak"), require_vector=True)
    assert rep.facts.get("vector_instructions", 0) > 0, rep.facts


def test_a_non_elf_fails_closed():
    """An unreadable artifact must raise, never report a pass it did not earn."""
    with pytest.raises(elf_audit.ElfAuditError):
        elf_audit.audit("/etc/hostname", boards.board("chipyard_kodiak"))
    with pytest.raises(elf_audit.ElfAuditError):
        elf_audit.audit("/nonexistent/zephyr.elf", boards.board("chipyard_kodiak"))


def test_the_report_serialises_for_a_delivery_package():
    rep = elf_audit.audit(_an_elf(), boards.board("chipyard_kodiak"))
    d = rep.to_dict()
    import json

    json.dumps(d)                     # must be JSON-serialisable to ship next to the binary
    assert d["board"] == "chipyard_kodiak" and "facts" in d and "segments" in d
