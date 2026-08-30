"""The layer-scale workload generator: the properties that decide whether its number is real.

Every test here runs against a SYNTHETIC machine built in this file -- a small tile edge, a small
register file, a small DRAM window -- so the generator is exercised without naming a target and without
the model venv. That is also the point of the exercise: if the generator needed one specific machine to
be testable, it would be overfit to it.

The four properties under test are the four ways a layer-scale run returns a plausible wrong number:
the loop that never closes, the footprint that silently wraps the simulator's DRAM window, the
reference computed in the wrong format, and the unrolled program that does not fit instruction memory.
"""
from __future__ import annotations

import numpy as np
import pytest

from merlin.perf import workload_gen as WG
from merlin.targetgen.isa_model import IsaModel

# --- a synthetic ISA -------------------------------------------------------------------------------
# RISC-V-shaped field maps, written HERE (in the test) rather than in the library: the generator reads
# every field placement from the model it is handed, which is exactly what makes this fixture possible.
_R = [7, 8, 9, 10, 11]                       # a 5-bit destination field -> a 32-register file
_S1 = [15, 16, 17, 18, 19]
_S2 = [20, 21, 22, 23, 24]
_I12 = list(range(20, 32))
_U20 = list(range(12, 32))
_BIMM = [None, 8, 9, 10, 11, 25, 26, 27, 28, 29, 30, 7, 31]
_V = [7, 8, 9, 10, 11, 12]


def _op(fixed, fields, cls="X", role="scalar"):
    return {"class": cls, "role": role, "fixed_mask": 0x7F, "fixed_value": fixed, "fields": fields}


def synthetic_isa(target: str = "synthetic") -> IsaModel:
    by = {
        "ADD": _op(0x33, {"rd": _R, "rs1": _S1, "rs2": _S2}),
        "ADDI": _op(0x13, {"rd": _R, "rs1": _S1, "imm": _I12}),
        "LUI": _op(0x37, {"rd": _R, "imm": _U20}),
        "BNE": _op(0x1063, {"rs1": _S1, "rs2": _S2, "imm": _BIMM}),
        "DELAY": _op(0x1067, {"rd": _R, "rs1": _S1, "imm": _I12}),
        "HALT": _op(0x73, {}),
        "DLOAD": _op(0x7B, {"rd": _R, "rs1": _S1, "rs2": _S2}),
        "DSTORE": _op(0x0200007B, {"rd": _R, "rs1": _S1, "rs2": _S2}),
        "DWAIT": _op(0x0200007F, {}),
        "TLOAD": _op(0x07, {"vd": _V, "rs1": _S1, "imm": _I12}),
        "TSTORE": _op(0x2007, {"vd": _V, "rs1": _S1, "imm": _I12}),
        "TPOSE": _op(0x6B, {"vd": _V, "vs1": _S1}),
        "WPUSH": _op(0x77, {"vd": _V, "vs1": _S1}, cls="W", role="weight_load"),
        "MUL": _op(0x14000077, {"vd": _V, "vs1": _S1, "vs2": _S2}, cls="M", role="matmul"),
        "MULACC": _op(0x18000077, {"vd": _V, "vs1": _S1, "vs2": _S2}, cls="M", role="matmul"),
        "ACCPOP": _op(0x10000077, {"vd": _V, "vs1": _S1, "vs2": _S2}, cls="P", role="acc_readout"),
    }
    roles: dict[str, list[str]] = {}
    for ent in by.values():
        roles.setdefault(ent["role"], [])
        if ent["class"] not in roles[ent["role"]]:
            roles[ent["role"]].append(ent["class"])
    return IsaModel(target=target, by_mnemonic=by, asm_mnemonics={}, roles=roles,
                    dram_base=0, halt_mnemonics=("HALT",))


OPS = WG.KernelOps(add="ADD", add_imm="ADDI", load_upper="LUI", branch_ne="BNE", stall="DELAY",
                   halt="HALT", dma_load="DLOAD", dma_store="DSTORE", dma_wait="DWAIT",
                   tile_load="TLOAD", tile_store="TSTORE", transpose="TPOSE", weight_push="WPUSH",
                   contract="MUL", contract_accumulate="MULACC", acc_read="ACCPOP")

CF = WG.ControlFlow(2, 1, "fixture")
SETTLE = WG.Settle.uniform(4, "fixture")


def synthetic_facts(*, window: int | None = 1 << 16, edge: int = 4,
                    imem: int | None = 1024) -> WG.MachineFacts:
    return WG.MachineFacts(
        target="synthetic", isa=synthetic_isa(), tile=WG.TileGeometry(edge, edge, "fixture"),
        dram_base=0, word_bytes=4, operand_dtype="fp8_e4m3", accum_dtype="bf16",
        dram_window=window, imem_words=imem, provenance={"fixture": "synthetic"})


def _plan(m, k, n, *, facts=None, cf=CF, A=None, W=None):
    return WG.plan_matmul(facts or synthetic_facts(), OPS, m=m, k=k, n=n, control_flow=cf,
                          settle=SETTLE, A=A, W=W, subnormal_operand_flush=False)


# --- 1. the loop that never closes -----------------------------------------------------------------
def test_branch_immediate_is_the_measured_contract_not_a_byte_offset():
    """The branch immediate is ``scale * (target - branch)`` in INSTRUCTIONS, with the scale measured on
    the machine. Encoding the wrong one produces a kernel that assembles, runs and halts having executed
    its body once -- so this is checked as arithmetic, on the emitted word, per candidate scale."""
    isa = synthetic_isa()
    for scale in (1, 2, 4):
        p = WG._Program(isa, WG.ControlFlow(scale, 1, "fixture"), OPS)
        p.label("top")
        p.emit("ADDI", rd=1, rs1=1, imm=1)
        p.branch("BNE", "top", rs1=1, rs2=2)
        resolved = p.resolved()
        # the branch sits at index 1; the target is index 0
        imm = resolved[1][1]["imm"]
        signed = imm - (1 << 13) if imm & (1 << 12) else imm
        assert signed == scale * (0 - 1), f"scale {scale} encoded {signed}"


def test_the_delay_slot_is_filled_and_is_not_a_branch():
    """The instruction(s) after a branch execute on BOTH paths, and a branch in a delay slot is illegal
    on the machine measured here. The generator fills the slot itself, because leaving it to the caller
    is the failure that looks like a working loop."""
    isa = synthetic_isa()
    for slots in (0, 1, 2):
        p = WG._Program(isa, WG.ControlFlow(2, slots, "fixture"), OPS)
        p.label("top")
        p.branch("BNE", "top", rs1=1, rs2=2)
        resolved = p.resolved()
        assert len(resolved) == 1 + slots
        for mnem, ops in resolved[1:]:
            assert mnem == OPS.add_imm and ops["rd"] == 0     # a write to the fixed-zero register


def test_the_kernel_length_does_not_grow_with_the_shape():
    """The reason the generator loops at all: instruction memory is finite, and an unrolled layer
    overflows it. The emitted program is the same size for a single tile and for a layer 512x larger."""
    small = _plan(4, 4, 4)
    large = _plan(4, 4096, 2048)
    assert large.total_macs() // small.total_macs() > 100_000
    # the emitted program grows only by the extra words a wider constant needs, not with the tile count
    assert len(large.words) < 2 * len(small.words)
    # ... while the unrolled equivalent grows with it, by five orders of magnitude
    assert large.unrolled_word_estimate() > 100 * len(large.words) * 100


def test_the_unrolled_twin_is_what_overflows_instruction_memory():
    """The loops are not tidiness. The emitted program fits a small instruction memory at a shape whose
    unrolled schedule does not, and the capacity is DERIVED -- an underivable one reports unchecked."""
    plan = _plan(4, 512, 512, facts=synthetic_facts(window=1 << 22, imem=1024))
    fit = plan.instruction_memory_fit()
    assert fit["fits"] is True and fit["unrolled_fits"] is False and fit["unrolled_overflow"] > 1
    unchecked = _plan(4, 4, 4, facts=synthetic_facts(imem=None)).instruction_memory_fit()
    assert unchecked["fits"] is None and unchecked["unrolled_fits"] is None


def test_a_shape_that_is_not_a_whole_tile_is_refused():
    with pytest.raises(WG.WorkloadError):
        _plan(4, 6, 4)


# --- 2. the footprint that silently wraps ----------------------------------------------------------
def test_alias_report_flags_a_tensor_that_runs_past_the_window():
    facts = synthetic_facts(window=4096)
    places = (WG.Placement("A", "input", [1], "fp8_e4m3", 5000, 0),)
    rep = WG.alias_report(places, facts.dram_window)
    assert not rep.ok and rep.wrapped == ("A",) and "wrap" in rep.reason


def test_alias_report_flags_two_tensors_that_collide_after_reduction():
    """The addresses differ by a whole window, so they look disjoint and reduce onto each other -- the
    exact failure the runner does not report."""
    places = (WG.Placement("A", "input", [1], "fp8_e4m3", 64, 0),
              WG.Placement("B", "weight", [1], "fp8_e4m3", 64, 4096))
    rep = WG.alias_report(places, 4096)
    assert not rep.ok and ("A", "B") in rep.collisions


def test_an_unknown_window_is_not_a_pass():
    rep = WG.alias_report((WG.Placement("A", "input", [1], "fp8_e4m3", 64, 0),), None)
    assert not rep.ok and "COULD NOT BE CHECKED" in rep.reason


def test_a_clean_footprint_passes_and_a_layer_that_overflows_does_not():
    facts = synthetic_facts(window=1 << 14)
    ok = WG.alias_report(_plan(4, 32, 32, facts=facts).placements, facts.dram_window)
    assert ok.ok
    too_big = WG.alias_report(_plan(64, 256, 256, facts=facts).placements, facts.dram_window)
    assert not too_big.ok


# --- 3. the reference computed in the wrong format --------------------------------------------------
def test_the_reference_rounds_into_the_accumulator_after_every_mac():
    """A narrow-float accumulator rounds every partial sum, so a full-precision reference grades a
    correct device as broken once the values exceed the format's integer range."""
    A = np.full((1, 64), 3.0, dtype=np.float32)
    W = np.full((64, 1), 3.0, dtype=np.float32)
    ref = WG.accumulate_reference(A, W, accum_dtype="bf16")
    assert ref is not None
    assert ref[0, 0] != (A @ W)[0, 0]                 # 576 needs more than bf16's mantissa carries
    small = WG.accumulate_reference(np.ones((1, 4), np.float32), np.ones((4, 1), np.float32),
                                    accum_dtype="bf16")
    assert small[0, 0] == 4.0                        # exact where the format can represent it


def test_an_exact_accumulator_has_no_model_and_says_so():
    assert WG.accumulate_reference(np.ones((1, 2), np.float32), np.ones((2, 1), np.float32),
                                   accum_dtype="int8") is None


def test_the_bank_stream_unpacks_back_to_the_logical_matrix():
    """The accumulator reads out into a register PAIR of column slices, laid down per output tile. The
    reader is the code that chose the order; this checks the two are inverses."""
    facts = synthetic_facts(edge=4)
    plan = _plan(8, 4, 8, facts=facts)
    logical = np.arange(64, dtype=np.float32).reshape(8, 8)
    banks = facts.banks_per_tile
    cpb = facts.tile.cols // banks
    rows = []
    for i in range(2):
        for j in range(2):
            for b in range(banks):
                rows.append(logical[i * 4:(i + 1) * 4, j * 4 + b * cpb: j * 4 + (b + 1) * cpb])
    stream = np.concatenate(rows, axis=0)
    assert np.array_equal(plan.unpack_output(stream), logical)


# --- 4. the facts that must not be assumed ----------------------------------------------------------
def test_an_undefined_instruction_is_an_error_not_a_wrong_word():
    bad = WG.KernelOps(**{**OPS.as_dict(), "contract": "NO_SUCH_OP"})
    with pytest.raises(WG.WorkloadError):
        WG.plan_matmul(synthetic_facts(), bad, m=4, k=4, n=4, control_flow=CF, settle=SETTLE)


def test_the_scalar_register_bound_comes_from_the_encoding_and_fails_closed():
    isa = synthetic_isa()
    bound, why = WG.scalar_register_count("synthetic", isa, OPS)
    assert bound == 32 and "field" in why
    with pytest.raises(WG.WorkloadError) as e:
        WG._ScalarFile("synthetic", isa, OPS, tuple(f"r{i}" for i in range(bound)))
    assert "scalar registers" in str(e.value)


def test_probe_control_flow_keeps_the_candidate_that_actually_loops():
    """The verdict is the measurement, not the candidate order: a runner that only closes the loop for
    one scale must yield that scale, whatever position it holds in the search."""
    facts = synthetic_facts()
    trips, body = 8, 200
    seen = []

    def runner(src, budget):
        # stand-in for a machine whose branch adder quadruples the immediate
        scale = 4 if ".word" in src else 0
        seen.append(scale)
        # decode which candidate this is from the emitted branch immediate
        words = [int(l.split()[1], 16) for l in src.splitlines() if l.strip().startswith(".word")]
        imm = (((words[-3] >> 8) & 0xF) << 1) | (((words[-3] >> 25) & 0x3F) << 5) \
            | (((words[-3] >> 7) & 1) << 11) | (((words[-3] >> 31) & 1) << 12)
        signed = imm - (1 << 13) if imm & (1 << 12) else imm
        return trips * body + 20 if signed == -4 * 2 else body + 8

    cf = WG.probe_control_flow(facts, OPS, runner, trips=trips, body_cycles=body)
    assert cf.branch_imm_scale == 4
    assert "measured on synthetic" in cf.provenance


def test_probe_control_flow_refuses_when_nothing_loops():
    facts = synthetic_facts()
    with pytest.raises(WG.WorkloadError) as e:
        WG.probe_control_flow(facts, OPS, lambda src, budget: 10, trips=8, body_cycles=200)
    assert "no candidate branch encoding closed a loop" in str(e.value)


def test_the_command_buffer_and_the_kernel_use_the_same_addresses():
    """The failure this replaces is a command buffer whose addresses and a kernel whose addresses were
    computed by different code; here they come from one placement."""
    plan = _plan(4, 8, 8)
    cb = plan.command_buffer()
    for p in plan.placements:
        assert cb["tensors"][p.name]["base"] == p.base
    assert len({p.base for p in plan.placements}) == 3
