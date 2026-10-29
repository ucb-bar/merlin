"""Actual encoded branches, joins, spills and loop-carried register facts."""

from merlin.compile.analysis.disassembly import RawInsn, tokenize_text
from merlin.compile.analysis.riscv_machine import (
    Bits,
    arithmetic,
    decode_cpu,
    finite_control_counts,
    region_index,
    register_facts,
)


def row(pc, word):
    return RawInsn(pc, "ignored_objdump_alias", [], f"{word:08x}")


def addi(rd, rs, imm):
    return ((imm & 4095) << 20) | (rs << 15) | (rd << 7) | 0x13


def branch(pc, target, rs1, rs2=0, funct=1):
    imm = (target - pc) & 8191
    return (
        (imm >> 12) << 31
        | ((imm >> 5) & 63) << 25
        | rs2 << 20
        | rs1 << 15
        | funct << 12
        | ((imm >> 1) & 15) << 8
        | ((imm >> 11) & 1) << 7
        | 0x63
    )


def test_word_decoding_ignores_wrong_pretty_mnemonics_and_decodes_compressed_stack():
    assert decode_cpu(row(0x100, branch(0x100, 0xF8, 15))).target == 0xF8
    assert decode_cpu(RawInsn(0, "", [], "7139")).imm == -64
    assert decode_cpu(RawInsn(0, "", [], "f822")).rs2 == 8
    assert decode_cpu(RawInsn(0, "", [], "8082")).kind == "return"
    assert decode_cpu(RawInsn(0x8000206A, "", [], "dffd")).target == 0x80002068
    assert decode_cpu(RawInsn(0x80002098, "", [], "c729")).target == 0x800020E2
    assert decode_cpu(RawInsn(0x8000228E, "", [], "fb7d")).target == 0x80002284


def test_llvm_byte_columns_and_gnu_words_produce_identical_cpu_and_custom_words():
    llvm = tokenize_text("100: 39 71 addi sp,sp,-64\n102: 7b 30 f7 0e .insn 4, 0xef7307b\n")
    gnu = tokenize_text("100: 7139 addi sp,sp,-64\n102: 0ef7307b .insn 4, 0xef7307b\n")
    assert llvm == gnu
    assert decode_cpu(llvm[0]).imm == -64


def test_object_relocations_are_annotations_and_refuse_unlinked_control_and_operands():
    from merlin.kernels.instruction_comparison import instruction_summary
    from merlin.perf.pc_instruction_listing import control_flow, normalized_text

    rows = tokenize_text(
        "0: 00000513 addi a0,zero,0\n0: R_RISCV_LO12_I external_symbol\n4: 0000007b <unknown>\n8: 00008067 ret\n"
    )
    assert len(rows) == 3 and rows[0].relocations == ("R_RISCV_LO12_I external_symbol",)
    assert tokenize_text(normalized_text(rows)) == rows
    assert decode_cpu(rows[0]).kind == "opaque"
    states, facts = register_facts(rows)
    assert states == {} and "relocations" in facts["reason"]
    counts = finite_control_counts(rows, custom_opcode=123)
    assert counts["status"] == "unknown" and "commands" not in counts
    assert control_flow(rows)["unresolved"] == ["unresolved_object_relocations"]
    summary = instruction_summary(rows)
    rows[0].relocations = ("R_RISCV_LO12_I another_symbol",)
    assert summary["ordered_structure_sha256"] != instruction_summary(rows)["ordered_structure_sha256"]


def test_join_keeps_only_common_bits_and_never_initializes_arguments_to_zero():
    rows = [
        row(0, branch(0, 12, 10)),
        row(4, addi(11, 0, 16)),
        row(8, 0x0080006F),
        row(12, addi(11, 0, 17)),
        row(16, 0x0000007B),
    ]
    states, report = register_facts(rows)
    assert states[0][10].exact is None
    assert states[16][11].exact is None
    assert states[16][11].mask & 0x10 and states[16][11].value & 0x10
    assert not (states[16][11].mask & 1)
    assert report["reachable_blocks"] == 4


def test_loop_does_not_publish_first_iteration_as_a_constant():
    rows = [
        row(0, addi(11, 0, 0)),
        row(4, addi(12, 0, 4)),
        row(8, 0x0000007B),
        row(12, addi(11, 11, 1)),
        row(16, branch(16, 8, 11, 12)),
        row(20, 0x00008067),
    ]
    states, _ = register_facts(rows)
    assert states[8][11].exact is None
    assert states[8][12].exact == 4
    counted = finite_control_counts(rows, custom_opcode=123)
    assert counted["status"] == "proved" and counted["commands"] == 4
    assert counted["command_executions"] == {8: 4}


def test_data_dependent_control_and_budget_exhaustion_publish_no_partial_counts():
    rows = [row(0, 0x0000007B), row(4, branch(4, 0, 10)), row(8, 0x00008067)]
    counted = finite_control_counts(rows, custom_opcode=123)
    assert counted["status"] == "unknown" and "commands" not in counted
    counted = finite_control_counts([row(0, 0x0000006F)], custom_opcode=123, max_steps=3)
    assert counted["reason"] == "finite_instruction_budget_exhausted" and "commands" not in counted


def test_reserved_instructions_and_system_traps_never_prove_completion():
    for word in (0x00000073, 0x00009067, 0x00007003, 0x00004023, 0x0000403B, 0x00009002):
        result = finite_control_counts([row(0, word), row(4, 0x00008067)], custom_opcode=123)
        assert result["status"] == "unknown" and "commands" not in result


def test_pointer_ordering_needs_an_address_range_even_with_common_symbolic_base():
    rows = [row(0, addi(11, 10, 16)), row(4, branch(4, 12, 10, 11, funct=6)), row(8, 0x0000007B), row(12, 0x00008067)]
    assert finite_control_counts(rows, custom_opcode=123)["status"] == "unknown"


def test_stack_reload_retains_original_pointer_but_overlapping_write_kills_it():
    sd_a0 = 10 << 20 | 2 << 15 | 3 << 12 | 0x23
    ld_a1 = 2 << 15 | 3 << 12 | 11 << 7 | 0x03
    rows = [row(0, sd_a0), row(4, ld_a1), row(8, 0x0000007B)]
    states, _ = register_facts(rows)
    assert states[8][11].base == "arg0"
    rows = [row(0, sd_a0), row(4, 2 << 15 | 0x23), row(8, ld_a1), row(12, 0x0000007B)]
    states, _ = register_facts(rows)
    assert states[12][11].base is None


def test_masked_unknown_low_bits_do_not_destroy_known_transfer_dimensions():
    pointer = Bits()
    low = arithmetic("and", pointer, Bits.const(0xFFFF))
    dimensions = Bits.const((16 << 48) | (64 << 32))
    packed = arithmetic("or", dimensions, low)
    assert (packed.mask >> 32) == 0xFFFFFFFF
    assert packed.value >> 32 == dimensions.value >> 32


def test_region_index_merges_cross_boundary_branches_and_preserves_sites():
    text = """Disassembly of section .text:
00000100 <fromhost>:
100: 0000007b .insn 4, 0x7b
104: 00008067 ret
108: 0000007b .insn 4, 0x7b
10c: fffff06f j 108
"""
    rows = tokenize_text(text)
    regions = region_index(rows, artifact_sha256="a" * 64, custom_opcode=123)
    assert sum(meta["accelerator_sites"] for meta, _ in regions) == 2
    assert len(regions) == 2
    assert all(meta["section"] == ".text" for meta, _ in regions)
    assert regions == region_index(rows, artifact_sha256="a" * 64, custom_opcode=123)
