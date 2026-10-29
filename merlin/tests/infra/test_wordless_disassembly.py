"""Wordless instruction listings retain exact directives and unknown scalar words."""
import pytest

from merlin.compile.analysis.disassembly import tokenize_text
from merlin.compile.analysis.rocc_commands import decode_stream
from merlin.perf.pc_instruction_listing import control_flow, normalized_text


def test_wordless_add_is_not_a_hex_column_and_directive_roundtrips():
    rows = tokenize_text("100: add a0,a1,a2\n104: .insn 4, 0x0200307b\n108: ret\n")
    assert [(r.mnemonic, r.hexcode) for r in rows] == [
        ("add", ""), (".insn", "0200307b"), ("ret", "")]
    assert tokenize_text(normalized_text(rows)) == rows
    decoded = decode_stream(rows, {"custom_opcode": 123, "legal_funct": [1], "names": {"1": "load"}})
    assert [r.identity for r in decoded] == ["add", "load", "ret"]
    assert decoded[0].fields == {}  # no synthetic scalar word/register values
    assert control_flow(rows)["unresolved"] == ["instruction_encodings_missing"]


@pytest.mark.parametrize("directive", [".insn r 0x7b,3,1,x0,x0,x0", ".insn 4,0x100000000",
                                       ".insn 3,0x20307b", ".insn 4,unknown"])
def test_incomplete_or_unsupported_directive_has_no_guessed_encoding(directive):
    assert tokenize_text("100: " + directive)[0].hexcode == ""


def test_encoded_columns_and_symbol_headers_remain_compatible():
    rows = tokenize_text("00000100 <kernel>:\n100: 0001 nop\n102: 0200307b <unknown>\n")
    assert [(r.hexcode, r.mnemonic) for r in rows] == [("0001", "nop"), ("0200307b", "<unknown>")]
    assert all(r.section == "00000100 <kernel>" for r in rows)
