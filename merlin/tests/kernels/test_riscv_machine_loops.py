"""Natural loops, trip counts, affine sites, call contexts and FP bit copies."""

from merlin.compile.analysis.disassembly import RawInsn
from merlin.compile.analysis.riscv_machine import (
    finite_control_counts,
    register_facts,
    summarize_function,
)

RET = 0x00008067
CUSTOM = 0x7B  # an unassigned custom-0 word stands in for an accelerator site


def row(pc, word):
    return RawInsn(pc, "ignored_objdump_alias", [], f"{word:08x}")


def program(*words, start=0):
    return [row(start + 4 * i, w) for i, w in enumerate(words)]


def addi(rd, rs, imm, op=0x13):
    return ((imm & 4095) << 20) | (rs << 15) | (rd << 7) | op


def slli(rd, rs, shamt):
    return (shamt << 20) | (rs << 15) | (1 << 12) | (rd << 7) | 0x13


def lui(rd, imm20):
    return (imm20 << 12) | (rd << 7) | 0x37


def ld(rd, rs, imm):
    return ((imm & 4095) << 20) | (rs << 15) | (3 << 12) | (rd << 7) | 0x03


def sd(rs2, rs1, imm):
    imm &= 4095
    return ((imm >> 5) << 25) | (rs2 << 20) | (rs1 << 15) | (3 << 12) | ((imm & 31) << 7) | 0x23


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


BEQ, BNE, BLT, BGE, BLTU, BGEU = 0, 1, 4, 5, 6, 7


def jal(pc, target, rd=0):
    imm = (target - pc) & ((1 << 21) - 1)
    return (
        ((imm >> 20) & 1) << 31
        | ((imm >> 1) & 1023) << 21
        | ((imm >> 11) & 1) << 20
        | ((imm >> 12) & 255) << 12
        | rd << 7
        | 0x6F
    )


def custom(rs1, rs2=0):
    return (rs2 << 20) | (rs1 << 15) | CUSTOM


def only_loop(analysis):
    (loop,) = analysis["loops"]
    return loop


def test_bottom_tested_loop_with_constant_bound_proves_header_executions():
    rows = program(addi(10, 0, 0), addi(11, 0, 10), addi(10, 10, 1), branch(12, 8, 10, 11, BLT), RET)
    _, analysis = register_facts(rows)
    loop = only_loop(analysis)
    assert loop["header_pc"] == 8 and loop["latch_pcs"] == [12] and loop["nest_depth"] == 1
    assert loop["parent_header_pc"] is None
    assert {"reg": "x10", "init": 0, "step": 1, "status": "proved"} in loop["induction"]
    assert loop["trip_count"]["status"] == "proved" and loop["trip_count"]["value"] == 10


def test_not_equal_shape_unsigned_and_top_tested_loops():
    rows = program(addi(10, 0, 0), addi(11, 0, 8), addi(10, 10, 2), branch(12, 8, 10, 11, BNE), RET)
    assert only_loop(register_facts(rows)[1])["trip_count"]["value"] == 4
    rows = program(addi(10, 0, 0), addi(11, 0, 5), addi(10, 10, 1), branch(12, 8, 10, 11, BLTU), RET)
    assert only_loop(register_facts(rows)[1])["trip_count"]["value"] == 5
    # while (i < 5) { i++; }: the header test runs once more than the body.
    rows = program(addi(10, 0, 0), addi(11, 0, 5), branch(8, 20, 10, 11, BGE), addi(10, 10, 1), jal(16, 8), RET)
    trip = only_loop(register_facts(rows)[1])["trip_count"]
    assert trip["status"] == "proved" and trip["value"] == 6 and trip["backedges_taken"] == 5
    # Counting down to zero with beq/bne against x0.
    rows = program(addi(10, 0, 12), addi(10, 10, -3), branch(8, 4, 10, 0, BNE), RET)
    assert only_loop(register_facts(rows)[1])["trip_count"]["value"] == 4


def test_invariant_bound_needs_a_seeded_value_and_varying_bound_stays_unknown():
    rows = program(addi(10, 0, 0), addi(10, 10, 1), branch(8, 4, 10, 11, BLT), RET)
    trip = only_loop(register_facts(rows)[1])["trip_count"]
    assert trip["status"] == "unknown" and trip["reason"] == "exit_bound_invariant_but_not_constant"
    _, seeded = register_facts(rows, entry_state={"a1": 6})
    assert only_loop(seeded)["trip_count"]["value"] == 6
    varying = program(addi(10, 0, 0), addi(11, 0, 9), addi(10, 10, 1), addi(11, 11, 1), branch(16, 8, 10, 11, BLT), RET)
    trip = only_loop(register_facts(varying)[1])["trip_count"]
    assert trip["status"] == "unknown" and "reason" in trip
    # An initial value from the caller is unknown unless seeded.
    unknown_start = program(addi(11, 0, 9), addi(10, 10, 1), branch(8, 4, 10, 11, BLT), RET)
    assert only_loop(register_facts(unknown_start)[1])["trip_count"]["status"] == "unknown"


def test_wraparound_is_refused_rather_than_counted():
    # i starts at 10, steps +1 and exits on i == 5: only after a 64-bit wrap.
    rows = program(addi(10, 0, 10), addi(11, 0, 5), addi(10, 10, 1), branch(12, 8, 10, 11, BNE), RET)
    trip = only_loop(register_facts(rows)[1])["trip_count"]
    assert trip["status"] == "unknown" and "wraparound" in trip["reason"]


def test_spilled_counter_in_a_frame_slot_is_an_induction_variable():
    rows = program(
        addi(2, 2, -16),
        sd(0, 2, 0),
        ld(10, 2, 0),
        addi(10, 10, 1),
        sd(10, 2, 0),
        addi(11, 0, 3),
        branch(24, 8, 10, 11, BLT),
        addi(2, 2, 16),
        RET,
    )
    loop = only_loop(register_facts(rows)[1])
    assert {"reg": "frame[-16]:64", "init": 0, "step": 1, "status": "proved"} in loop["induction"]
    assert loop["trip_count"]["value"] == 3


def test_affine_site_address_enumerates_a_value_set_and_nesting():
    rows = program(
        addi(10, 0, 0),
        addi(11, 0, 4),
        slli(12, 10, 4),
        addi(12, 12, 0x100),
        custom(12),
        addi(10, 10, 1),
        branch(24, 8, 10, 11, BLT),
        RET,
    )
    states, _ = register_facts(rows)
    record = states[16][12].record()
    assert record["kind"] == "unknown" and record["value_set"] == [0x100, 0x110, 0x120, 0x130]
    assert record["affine"]["terms"] == [dict(header_pc=8, step=16, count=4)]
    # A non-affine CPU operation is evaluated at every proved iteration point.
    ori = (0xF00 << 20) | (12 << 15) | (6 << 12) | (13 << 7) | 0x13  # ori a3, a2, -256
    rows = rows[:4] + [
        row(16, ori),
        row(20, custom(13)),
        *program(addi(10, 10, 1), branch(28, 8, 10, 11, BLT), RET, start=24),
    ]
    states, _ = register_facts(rows)
    assert states[20][13].record()["value_set"] == [(v | 0xFFFFFFFFFFFFFF00) for v in (0x100, 0x110, 0x120, 0x130)]
    nested = program(
        addi(10, 0, 0),  # 0: i = 0
        addi(13, 0, 0),  # 4: j = 0          (outer header)
        slli(12, 10, 6),  # 8: i * 64        (inner header)
        slli(14, 13, 2),  # 12: j * 4
        0x00E60633,  # 16: add a2, a2, a4
        custom(12),  # 20
        addi(13, 13, 1),  # 24
        addi(15, 0, 3),  # 28
        branch(32, 8, 13, 15, BLT),  # 32: inner latch
        addi(10, 10, 1),  # 36
        addi(15, 0, 2),  # 40
        branch(44, 4, 10, 15, BLT),  # 44: outer latch
        RET,
    )
    states, analysis = register_facts(nested)
    loops = {loop["header_pc"]: loop for loop in analysis["loops"]}
    assert loops[4]["trip_count"]["value"] == 2 and loops[8]["trip_count"]["value"] == 3
    assert loops[8]["parent_header_pc"] == 4 and loops[8]["nest_depth"] == 2
    assert states[20][12].record()["value_set"] == [0, 4, 8, 64, 68, 72]


def test_call_sites_record_arguments_and_summaries_apply_only_under_assumptions():
    callee = program(addi(10, 10, 1), RET, start=100)
    summary = summarize_function(callee, entry_state={"a0": 7})
    assert summary["status"] == "proved" and summary["returns"]["a0"] == 8
    assert summary["terminates"] == "proved" and summary["assumptions"] == [("a0", 7)]
    assert summarize_function(callee)["returns"]["a0"] is None
    caller = program(addi(10, 0, 7), jal(4, 100, 1), addi(11, 10, 0), RET)
    _, analysis = register_facts(caller)
    (site,) = analysis["call_sites"]
    assert site["pc"] == 4 and site["target_pc"] == 100
    assert site["argument_state"]["a0"] == 7 and site["argument_state"]["a1"] is None
    states, _ = register_facts(caller, call_summaries={100: summary})
    assert states[8][10].exact == 8
    states, _ = register_facts(caller)
    assert states[8][10].exact is None
    mismatched = program(addi(10, 0, 6), jal(4, 100, 1), addi(11, 10, 0), RET)
    states, _ = register_facts(mismatched, call_summaries={100: summary})
    assert states[8][10].exact is None
    assert finite_control_counts(caller, custom_opcode=CUSTOM)["status"] == "unknown"
    counted = finite_control_counts(caller, custom_opcode=CUSTOM, call_summaries={100: summary})
    assert counted["status"] == "proved" and counted["dynamic_command_count"]["status"] == "proved"


def test_bump_allocator_return_is_known_only_from_seeded_initialized_memory():
    allocator = program(lui(15, 0x10), ld(10, 15, 0), addi(14, 10, 64), sd(14, 15, 0), RET, start=200)
    assert summarize_function(allocator)["returns"]["a0"] is None
    seeded = summarize_function(allocator, entry_state={("memory", 0x10000, 64): 0x2000})
    assert seeded["returns"]["a0"] == 0x2000 and seeded["writes_memory"]
    assert seeded["memory"] == [(("abs", 0x10000, 64), 0x2040)]
    # An exact address nobody seeded may be a device register: never read back.
    device = program(lui(15, 0x20), addi(14, 0, 5), sd(14, 15, 0), ld(10, 15, 0), RET)
    states, _ = register_facts(device)
    assert states[16][10].exact is None


def test_fp_register_moves_are_exact_bit_copies_and_fp_loads_stay_unknown():
    fmv_w_x, fmv_x_w = (0x78 << 25) | (10 << 15) | (1 << 7) | 0x53, (0x70 << 25) | (1 << 15) | (11 << 7) | 0x53
    fmv_d_x, fmv_x_d = (0x79 << 25) | (10 << 15) | (2 << 7) | 0x53, (0x71 << 25) | (2 << 15) | (12 << 7) | 0x53
    rows = program(lui(10, 0x80000), fmv_w_x, fmv_x_w, fmv_d_x, fmv_x_d, RET)
    states, _ = register_facts(rows)
    assert states[12][11].exact == 0xFFFFFFFF80000000  # low word, sign-extended
    assert states[20][12].exact == 0xFFFFFFFF80000000  # full 64-bit copy
    fld = (10 << 15) | (3 << 12) | (2 << 7) | 0x07
    rows = program(lui(10, 0x1), fmv_d_x, fld, fmv_x_d, RET)
    states, _ = register_facts(rows)
    assert states[16][12].exact is None


def test_existing_callers_keep_their_keys_and_unknown_reasons():
    rows = program(addi(10, 0, 1), RET)
    _, analysis = register_facts(rows)
    assert analysis["status"] == "analyzed" and analysis["loops"] == [] and analysis["call_sites"] == []
    counted = finite_control_counts(rows, custom_opcode=CUSTOM)
    assert counted["status"] == "proved" and counted["dynamic_command_count"]["value"] == 0
