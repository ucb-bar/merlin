"""An audit of instructions no disassembler can name must not be able to report a clean result wrongly.

The tool this replaces counted `.insn` occurrences in SOURCE and reported "100% coverage" for a kernel
that read past its operand buffer and hung. So the properties under test here are the ones that make
that failure impossible to repeat: counts come from decoded encoding fields rather than from mnemonics,
every declared instruction gets a counter even at zero, a word that is neither nameable nor ours is
surfaced instead of dropped, and the identity digest can distinguish two extension instructions that a
disassembler renders identically.
"""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from merlin.kernels.decode import opu as OA


@dataclass(frozen=True)
class _Enc:
    """Stand-in for a derived encoding — only the three matched fields matter to the audit."""
    opcode: int
    funct3: int
    funct6: int


#: The four-instruction shape this extension has: an accumulate (funct3 2) and three moves (funct3 6),
#: in reserved even funct6 slots.
_TABLE = {"ACC": _Enc(0x57, 2, 40), "MOVEIN": _Enc(0x57, 6, 42),
          "BCAST": _Enc(0x57, 6, 44), "READOUT": _Enc(0x57, 6, 46)}


def _word(opcode: int, funct3: int, funct6: int, rd: int = 1, rs1: int = 5, rs2: int = 4) -> str:
    word = ((funct6 << 26) | (1 << 25) | (rs2 << 20) | (rs1 << 15)
            | (funct3 << 12) | (rd << 7) | opcode)
    return f"{word:08x}"


def _dis(*rows: tuple[str, str, str]) -> str:
    """Objdump-shaped text from (hexcode, mnemonic, operands) rows."""
    lines = ["", "Disassembly of section .text:", "", "0000000000000000 <k>:"]
    for i, (hexcode, mnem, ops) in enumerate(rows):
        lines.append(f"       {i * 4:x}: {hexcode}\t{mnem}\t{ops}".rstrip())
    return "\n".join(lines) + "\n"


_CFG = ("0d077057", "vsetvli", "zero, a4, e32, m1, ta, ma")
_LOAD = ("02058287", "vle8.v", "v5, (a1)")
_ACC = (_word(0x57, 2, 40), "<unknown>", "")
_BCAST = (_word(0x57, 6, 44), "<unknown>", "")
_READOUT = (_word(0x57, 6, 46), "<unknown>", "")


class TestFieldDecode:
    def test_decodes_the_vector_format_fields_of_a_word(self):
        got = OA.fields_of(0xa242a0d7)
        assert got["opcode"] == 0x57 and got["funct3"] == 2 and got["funct6"] == 40
        assert got["vm"] == 1
        assert (got["rd"], got["rs1"], got["rs2"]) == (1, 5, 4)

    def test_a_compressed_16_bit_instruction_is_declined_not_zero_extended(self):
        # `c.jr ra` is 2 bytes; decoding it as a 32-bit word would invent funct6/funct3 values.
        text = _dis(("8082", "c.jr", "ra"))
        a = OA.audit_text(text, _TABLE)
        assert a.extension_insns == 0 and a.unaccounted == ()


class TestCounting:
    def test_counts_by_encoding_not_by_mnemonic(self):
        # Every extension instruction here is spelled `<unknown>`; a mnemonic-based count sees zero.
        a = OA.audit_text(_dis(_CFG, _ACC, _CFG, _ACC, _READOUT), _TABLE)
        assert a.counts["ACC"] == 2
        assert a.counts["READOUT"] == 1

    def test_every_declared_instruction_gets_a_counter_even_at_zero(self):
        # The tool this replaces had NO counter for the row-move, so an accumulate whose operands were
        # never loaded looked identical to a correct kernel.
        a = OA.audit_text(_dis(_CFG, _ACC), _TABLE)
        assert set(a.counts) == set(_TABLE)
        assert a.counts["MOVEIN"] == 0 and a.counts["READOUT"] == 0

    def test_an_emitted_count_is_not_called_coverage(self):
        a = OA.audit_text(_dis(_CFG, _ACC), _TABLE)
        assert a.emitted_extension_ops == 1
        assert not hasattr(a, "coverage"), "a count of instructions is not a share of the work"

    def test_an_empty_stream_says_nothing_was_emitted(self):
        a = OA.audit_text(_dis(_CFG, _LOAD), _TABLE)
        assert a.extension_insns == 0
        assert any("no extension instruction" in n for n in a.notes)


class TestUnaccountedWords:
    def test_a_word_that_is_neither_nameable_nor_ours_is_surfaced(self):
        # A mis-encoded instruction looks exactly like this. Counting it as absent would report a clean
        # audit for a broken kernel.
        bogus = (_word(0x57, 2, 41), "<unknown>", "")     # the ODD neighbour of the accumulate
        a = OA.audit_text(_dis(_CFG, bogus), _TABLE)
        assert len(a.unaccounted) == 1
        assert a.unaccounted[0]["fields"]["funct6"] == 41
        assert any("could not name" in n for n in a.notes)

    def test_a_nameable_instruction_is_not_unaccounted(self):
        a = OA.audit_text(_dis(_CFG, _LOAD), _TABLE)
        assert a.unaccounted == ()


class TestOperandLengthConfiguration:
    def test_counts_configuration_instructions_between_extension_instructions(self):
        # Two configurations before the accumulate is the separate-length-per-operand shape: one for the
        # row lanes, one for the column lanes.
        a = OA.audit_text(_dis(_CFG, _LOAD, _CFG, _LOAD, _ACC), _TABLE)
        assert a.configured_before_each["ACC"] == 2
        assert a.unconfigured == ()

    def test_an_instruction_with_no_configuration_since_the_last_one_is_flagged(self):
        # The readout writes `vl` elements; inheriting a length set for a different operand is the
        # narrow-operand hazard this audit exists to catch.
        a = OA.audit_text(_dis(_CFG, _ACC, _READOUT), _TABLE)
        assert "READOUT" in a.unconfigured
        assert any("inherited" in n for n in a.notes)

    def test_configuration_instructions_are_matched_by_prefix_of_the_base_isa_naming(self):
        a = OA.audit_text(_dis(("0d077057", "vsetivli", "zero, 0x4, e32, m1, ta, ma"), _ACC), _TABLE)
        assert a.vector_config_insns == 1 and a.unconfigured == ()


class TestDigest:
    def test_distinguishes_two_extension_instructions_a_disassembler_renders_identically(self):
        # This is the gap in the mnemonic-stream digest used for inert-lever detection: both of these
        # are `<unknown>` with no operands, so hashing mnemonics makes an accumulate and a readout
        # indistinguishable and marks a real change inert.
        acc = OA.audit_text(_dis(_CFG, _ACC), _TABLE).digest
        out = OA.audit_text(_dis(_CFG, _READOUT), _TABLE).digest
        assert acc != out

    def test_distinguishes_the_same_instruction_used_on_different_registers(self):
        other = (_word(0x57, 2, 40, rd=2, rs1=6, rs2=7), "<unknown>", "")
        assert OA.audit_text(_dis(_ACC), _TABLE).digest != OA.audit_text(_dis(other), _TABLE).digest

    def test_is_stable_across_addresses(self):
        one = _dis(_CFG, _ACC)
        two = one.replace("0000000000000000 <k>:", "0000000000001000 <k>:")
        assert OA.audit_text(one, _TABLE).digest == OA.audit_text(two, _TABLE).digest

    def test_an_unchanged_stream_hashes_identically(self):
        text = _dis(_CFG, _LOAD, _ACC, _READOUT)
        assert OA.audit_text(text, _TABLE).digest == OA.audit_text(text, _TABLE).digest


class TestDecodeStream:
    def test_names_extension_instructions_and_leaves_others_alone(self):
        from merlin.kernels.decode.objdump import tokenize_text
        got = OA.decode_stream(tokenize_text(_dis(_CFG, _ACC)), _TABLE)
        assert [d.identity for d in got] == ["vsetvli", "ACC"]
        assert [d.from_extension for d in got] == [False, True]

    def test_an_empty_table_names_nothing_as_ours(self):
        from merlin.kernels.decode.objdump import tokenize_text
        got = OA.decode_stream(tokenize_text(_dis(_ACC)), {})
        assert got[0].from_extension is False


class TestAgainstRealAssembledCode:
    """The audit against code an assembler actually produced, so the field layout is checked against a
    real encoder rather than against this test's own bit-packing."""

    @pytest.fixture
    def obj(self, tmp_path):
        import subprocess
        from merlin.llvmlower import toolchain
        if not toolchain.available():
            pytest.skip("needs the pinned clang")
        src = tmp_path / "k.c"
        src.write_text(
            "void k(int *c, signed char *a, signed char *b, unsigned long ml, unsigned long nl) {\n"
            '  asm volatile("vsetvli zero, %0, e32, m1, ta, ma" : : "r"(nl));\n'
            '  asm volatile(".insn r 0x57, 0x6, 0x59, x1, x0, x0");\n'
            '  asm volatile("vsetvli zero, %0, e8, m1, tu, ma" : : "r"(ml));\n'
            '  asm volatile("vsetvli zero, %0, e8, m1, ta, ma" : : "r"(nl));\n'
            '  asm volatile(".insn r 0x57, 0x2, 0x51, x1, x5, x4");\n'
            '  asm volatile(".insn r 0x57, 0x6, 0x5d, x0, x0, x1");\n'
            "}\n", encoding="utf-8")
        out = tmp_path / "k.o"
        p = subprocess.run([toolchain.clang(), "--target=riscv64-unknown-elf", "-march=rv64gcv",
                            "-mabi=lp64d", "-O2", "-c", str(src), "-o", str(out)],
                           capture_output=True, text=True)
        if p.returncode != 0:
            pytest.skip(f"cross-compile unavailable: {p.stderr[-300:]}")
        return out

    def test_the_assemblers_own_words_decode_to_the_declared_encodings(self, obj):
        a = OA.audit_object(obj, _TABLE)
        assert a.counts["ACC"] == 1 and a.counts["BCAST"] == 1 and a.counts["READOUT"] == 1
        assert a.unaccounted == (), a.unaccounted

    def test_the_disassembler_really_cannot_name_them(self, obj):
        # If this ever fails, the disassembler learned the extension and a mnemonic-based audit would
        # start working -- worth knowing, because it changes what this module has to do.
        from merlin.kernels.decode.objdump import tokenize
        words = {i.hexcode for i in tokenize(obj)}
        named = [i.mnemonic for i in tokenize(obj) if i.mnemonic != OA.UNKNOWN_MNEMONIC]
        assert words, "expected a non-empty disassembly"
        assert all(not m.startswith("opm") for m in named)
        a = OA.audit_object(obj, _TABLE)
        assert a.extension_insns == 3, "all three must be found by encoding, not by name"

    def test_the_accumulate_sees_two_configurations_from_the_two_operand_loads(self, obj):
        a = OA.audit_object(obj, _TABLE)
        assert a.configured_before_each["ACC"] == 2
