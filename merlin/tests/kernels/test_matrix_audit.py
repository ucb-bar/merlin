"""The instruction audit keyed by MATCH/MASK encodings (targetgen.rtl.matrix_isa), not only by the triple.

A matrix unit can put several instructions in one funct6 slot and tell them apart by a register field;
a triple-keyed table would count all of them as one. These tests feed the audit words that differ only
in such a field and check each is attributed to its own instruction, that the most specific pattern
wins, and that a word no pattern claims is surfaced rather than dropped.
"""

from __future__ import annotations

from merlin.kernels.decode import opu as OA
from merlin.targetgen.rtl.matrix_isa import MaskedEncoding

_SLOT = 0x57 | (0x6 << 12) | (0x10 << 26)
_VM = 1 << 25  # set in every word, decoded by no pattern here
_SLOT_MASK = 0x7F | (0x7 << 12) | (0x3F << 26)
_TABLE = {
    "MOVE": MaskedEncoding("MOVE", _SLOT | (31 << 20), _SLOT_MASK | (0x1F << 20), "t"),
    "ZERO": MaskedEncoding("ZERO", _SLOT | (30 << 20), _SLOT_MASK | (0x1F << 20) | (0x1F << 15), "t"),
    # A whole-word pattern with don't-care rd/rs1 fields, like a host core's BitPat.
    "SETN": MaskedEncoding("SETN", 0x88007057, 0xFFF0707F, "t"),
}


def _dis(*words: int) -> str:
    lines = ["", "Disassembly of section .text:", "", "0000000000000000 <k>:"]
    for i, w in enumerate(words):
        lines.append(f"       {i * 4:x}: {w:08x}\t<unknown>")
    return "\n".join(lines) + "\n"


def test_words_sharing_a_slot_are_told_apart_by_their_register_field():
    move = _VM | _SLOT | (31 << 20) | (16 << 7) | (5 << 15)
    zero = _VM | _SLOT | (30 << 20) | (8 << 7)
    setn = 0x88007057 | (10 << 7) | (11 << 15)
    got = OA.audit_text(_dis(move, zero, zero, setn), _TABLE)
    assert got.counts == {"MOVE": 1, "ZERO": 2, "SETN": 1}
    assert not got.unaccounted


def test_a_word_in_the_slot_that_no_pattern_claims_is_surfaced():
    stray = _VM | _SLOT | (29 << 20)
    got = OA.audit_text(_dis(stray), _TABLE)
    assert got.extension_insns == 0 and len(got.unaccounted) == 1


def test_the_most_specific_pattern_wins():
    loose = MaskedEncoding("ANY", _SLOT, _SLOT_MASK, "t")
    got = OA.audit_text(_dis(_VM | _SLOT | (30 << 20)), {**_TABLE, "ANY": loose})
    assert got.counts["ZERO"] == 1 and got.counts["ANY"] == 0
