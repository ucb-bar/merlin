"""Errors shared by workload fact derivation, encoding, and reference numerics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from merlin.analysis.perf.contract.workload_facts import (
    MachineFacts,
    TileGeometry,
    WorkloadError,
    _accumulate_from_rtl_datapaths,
    _ScalarFile,
    _shipped_spec_docs,
    _spec_table_int,
    _spec_table_range,
    datapath_formats,
    dram_window_bytes,
    instruction_memory_words,
    machine_facts,
    scalar_register_count,
    tile_geometry,
)

# ---- workload_errors ---------------------------------------------------------------------------


# ---- workload_numerics -------------------------------------------------------------------------
# Host numerical reference for generated layer-scale workloads.
#
# This module handles accumulator-format rounding independently of instruction
# emission. It does not select a target or infer an undeclared numeric policy.


def encode_operand_bytes(values, dtype: str) -> bytes:
    """Encode values using the registered format, refusing unknown packing."""
    import numpy as np

    from merlin.common import target_data as QF

    a = np.asarray(values, dtype=np.float64)
    if not QF.has(dtype):
        raise WorkloadError(f"unknown operand format {dtype!r}")
    f = QF.get(dtype)
    bits = int(f.element_bits or 0)
    if bits == 0 or bits % 8:
        raise WorkloadError(f"{dtype!r} is a sub-byte format; its packing is not derivable here")
    if f.kind == "int_affine":
        lo, hi = (-(2 ** (bits - 1)), 2 ** (bits - 1) - 1) if f.signed else (0, 2**bits - 1)
        code = "<i" if f.signed else "<u"
        return np.clip(np.rint(a), lo, hi).astype(f"{code}{bits // 8}").tobytes()
    if f.kind == "float_ieee":
        if bits == 32:
            return a.astype("<f4").tobytes()
        if bits == 16 and int(f.exp_bits or 0) == 5:
            return a.astype("<f2").tobytes()
        if bits == 16 and int(f.exp_bits or 0) == 8:
            u = a.astype("<f4").view("<u4").astype(np.uint64)
            return (((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype("<u2")).tobytes()
        raise WorkloadError(f"no byte encoding derivable for {dtype!r}")
    if f.kind == "fp_ocp":
        from merlin.common.fp8_codec import ocp_encode

        eb, mb = int(f.exp_bits or 0), int(f.mant_bits or 0)
        if not eb or bits != 8:
            raise WorkloadError(f"no byte encoding derivable for {dtype!r}")
        return bytes(bytearray(ocp_encode(float(v), eb, mb, signed=bool(f.signed)) for v in a.reshape(-1)))
    raise WorkloadError(f"no byte encoding derivable for {dtype!r}")


def _accum_rounder(accum_dtype: str):
    """Round a running sum into the declared accumulator format, if needed."""
    import numpy as np

    from merlin.common import target_data as QF

    f = QF.get(accum_dtype)
    if f.kind != "float_ieee":
        return None
    mant, exp = int(f.mant_bits or 0), int(f.exp_bits or 0)
    if mant == 10 and exp == 5:
        return lambda x: x.astype("<f2").astype(np.float32)
    if mant == 7 and exp == 8:

        def rnd(x):
            u = np.asarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
            return (((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint32) << 16).astype(np.uint32).view(np.float32)

        return rnd
    return None


def accumulate_reference(
    A, W, *, accum_dtype: str, operand_dtype: str | None = None, subnormal_operand_flush: bool = False
):
    """Host golden rounded after each MAC in the declared accumulator format.

    Operands are flushed only when the selected profile declares that its
    multiplier flushes subnormals. Returns ``None`` for an accumulator that
    does not round partial sums; the plain product is then the reference.
    """
    import numpy as np

    A = np.asarray(A, dtype=np.float32)
    W = np.asarray(W, dtype=np.float32)
    if subnormal_operand_flush and operand_dtype:
        from merlin.common import fp8_formats as FF

        try:
            min_normal, _ = FF.normal_range(operand_dtype)
        except KeyError:
            return None
        A = np.where(np.abs(A) < min_normal, np.float32(0.0), A).astype(np.float32)
        W = np.where(np.abs(W) < min_normal, np.float32(0.0), W).astype(np.float32)
    rnd = _accum_rounder(accum_dtype)
    if rnd is None:
        return None
    acc = np.zeros((A.shape[0], W.shape[1]), dtype=np.float32)
    for i in range(A.shape[1]):
        acc = rnd(acc + np.outer(A[:, i], W[i, :]))
    return acc


# ---- workload_gen ------------------------------------------------------------------------------
# Generate a LAYER-SCALE workload for a self-hosted-ISA target: a command buffer, an emitted
# ``kernel.S``, a host-computed golden, and a proof that the run's DRAM footprint did not wrap.
#
# Why this module exists. Every performance number measured on such a target so far describes a demo
# TILE -- the largest tensor in the graded corpus is a few thousand elements and the kernels halt in
# hundreds of cycles. The programs that exist are static assembly with immediates baked in at authoring
# time, so a new shape means new hand-written assembly, and the "full model shapes" the corpus cites have
# no program, no golden and no run. Performance work is about model LAYERS, so the layer has to be
# generated.
#
# Four properties this generator has that a hand-written kernel does not:
#
# * **It loops.** Instruction memory is finite -- 32,768 words on the target measured here, DERIVED from
#   the instruction-memory region of its own memory map (:func:`instruction_memory_words`) -- and the
#   unrolled schedule for a layer overflows it. The emitted kernel's size is therefore O(1) in the shape:
#   the tile nest is three real backward branches, so the same 96-word program runs a 32x32x32 tile and
#   the 416x832x416 layer whose unrolled twin needs 108,011 words (3.3x the machine's instruction
#   memory). :meth:`MatmulPlan.instruction_memory_fit` reports both, and reports UNKNOWN -- not a pass --
#   when the capacity is not derivable.
# * **Its branches are encoded from the machine's own control-flow contract, MEASURED.** A branch
#   immediate is not portable knowledge. On the target measured here the PC is a WORD index and the
#   target is ``s1_pc + (imm >> 1)``, so a B-type immediate moves ``imm/2`` instructions, and there is an
#   architectural delay slot whose instruction executes on both paths. Encoding a byte offset produces a
#   kernel that assembles, runs, halts, and never closes its loop -- it reads as "this core has no control
#   flow". :func:`probe_control_flow` therefore DERIVES ``(imm_scale, delay_slots)`` by running candidate
#   encodings on the target's own oracle and keeping the one whose observable trip count is right. Nothing
#   about control flow is written down here.
# * **It refuses to produce a cycle count from an aliasing run.** The simulator's DRAM is a finite window
#   and addresses are reduced modulo its size, silently: bytes past the end are dropped or wrap onto other
#   tensors, and nothing in the returned result flags it. One 1024x3072 bf16 tensor is 6 MiB against a
#   1 MiB window -- it would alias six ways and still return a number. :func:`alias_report` computes the
#   masked span of every tensor from the window size DERIVED from the runner and reports the run INVALID
#   if any span wraps or two spans collide. A wrong number is worse than a slow one.
# * **Its golden is computed host-side, in the accumulator's own format.** A narrow-float accumulator
#   rounds every partial sum, so an f32 reference grades a perfectly correct device as broken -- measured
#   here at 796 of 4096 elements on a single layer. The reference models the DECLARED datapath (round
#   after every MAC, operand subnormal flush when the profile declares it) and the gate is BIT-EXACT,
#   which is the right bar precisely because it is reachable: a tolerance would hide the loop-trip and
#   bank-layout errors this generator exists to catch. A mismatch is DESCRIBED
#   (:meth:`MatmulPlan.divergence`) rather than absorbed.
#
# Target-agnostic by construction: the instruction encodings come from
# :func:`~merlin.target.isa.machine_model.isa_model_for_target` (probed from the target's own shipped ISA
# definition), the tile geometry from the RTL-discovered array, the DRAM aperture from the target's memory
# map, the operand/accumulate formats from its capability manifest, the DRAM window from the runner that
# will execute the program, and the control-flow contract from the machine itself. No opcode, funct value,
# field position, mesh dimension, latency or address is a literal in this file.


@dataclass(frozen=True)
class ControlFlow:
    """How a backward branch is encoded on this machine, and how many instructions after it execute
    regardless of the outcome.

    ``branch_imm_scale`` is the factor between the immediate a branch carries and the number of
    INSTRUCTIONS it moves: the immediate written into the encoding is ``scale * (target - branch)``,
    both measured in instruction indices. A machine whose PC counts words and whose branch adder halves
    the immediate has scale 2; one that adds the immediate directly has scale 1. Getting this wrong does
    not raise: the kernel assembles, runs and halts, having executed its loop body once.

    ``delay_slots`` instructions after the branch execute on BOTH paths and must therefore be harmless."""

    branch_imm_scale: int
    delay_slots: int
    provenance: str

    def branch_imm(self, branch_index: int, target_index: int) -> int:
        return self.branch_imm_scale * (target_index - branch_index)


@dataclass(frozen=True)
class Settle:
    """Cycles the scalar stream must idle before a tensor result is architecturally visible to the next
    instruction, per producer class. This machine has no interlock between the scalar stream and the
    tensor pipelines -- reading a register before its producer drains returns stale bytes rather than
    raising, and a vector op issued too soon does not write its destination at all. Both are wrong
    answers with no error attached, so the value is MEASURED on the device (:func:`probe_settle`),
    never picked."""

    tensor: int
    mxu: int
    vpu: int
    provenance: str

    @classmethod
    def uniform(cls, cycles: int, provenance: str) -> "Settle":
        return cls(tensor=int(cycles), mxu=int(cycles), vpu=int(cycles), provenance=provenance)


@dataclass(frozen=True)
class KernelOps:
    """Which of THIS target's instructions play each role in the emitted kernel.

    No mnemonic is written down in this module. A self-hosted ISA's vocabulary is the target's own, and
    the derived role table (:attr:`IsaModel.roles`) narrows the choice but does not settle it: a machine
    with two matrix units and both an overwriting and an accumulating multiply has four instructions
    carrying the single role ``matmul``, and which one to use is a SCHEDULING decision, not a fact about
    the hardware. So the selection is a parameter, made once at the edge that is legitimately about one
    target, and :func:`candidate_ops` reports the derived candidates to make it from. Onboarding a second
    target means writing a different selection, never editing this file.

    Every name is validated against the derived ISA model on construction, so a mnemonic this target does
    not define is an error here rather than a silently wrong word later."""

    add: str  # register + register
    add_imm: str  # register + immediate (also this generator's move and no-op)
    load_upper: str  # upper immediate, for constants wider than the immediate field
    branch_ne: str  # conditional backward branch, taken while two registers differ
    stall: str  # hold issue for an immediate number of cycles
    halt: str  # terminate the program
    dma_load: str  # DRAM -> staging memory
    dma_store: str  # staging memory -> DRAM
    dma_wait: str  # block until the channel is idle
    tile_load: str  # staging memory -> tensor register
    tile_store: str  # tensor register -> staging memory
    transpose: str  # tensor register transpose
    weight_push: str  # tensor register -> the array's weight buffer
    contract: str  # multiply, OVERWRITING the accumulator
    contract_accumulate: str  # multiply, ADDING into the accumulator
    acc_read: str  # accumulator -> tensor register(s)

    def validate(self, isa) -> None:
        missing = [f"{role}={name}" for role, name in self.as_dict().items() if isa.resolve(name) is None]
        if missing:
            raise WorkloadError(
                f"{isa.target!r} defines no instruction for: {', '.join(missing)}. "
                f"Derived candidates per role: {candidate_ops(isa)}"
            )

    def as_dict(self) -> dict[str, str]:
        return {f: getattr(self, f) for f in self.__dataclass_fields__}


def candidate_ops(isa) -> dict[str, list[str]]:
    """The mnemonics this target defines for each DERIVED structural role -- the menu a
    :class:`KernelOps` selection is made from. Reported rather than chosen: where a role resolves to more
    than one instruction, the choice is the caller's."""
    out: dict[str, list[str]] = {}
    for role, classes in (isa.roles or {}).items():
        names = sorted(mn for mn, ent in isa.by_mnemonic.items() if ent.get("class") in classes)
        out[role] = names
    if isa.halt_mnemonics:
        out["halt"] = list(isa.halt_mnemonics)
    return out


class _Program:
    """An instruction list with labels and branch fixups, encoded through
    :mod:`merlin.target.isa.isa_asm` against the derived field maps. Branch immediates are resolved after
    layout using the machine's measured control-flow contract."""

    def __init__(self, isa, cf: ControlFlow, ops: "KernelOps"):
        ops.validate(isa)
        self._isa = isa
        self._cf = cf
        self._ops = ops
        self._items: list[tuple[str, dict]] = []
        self._labels: dict[str, int] = {}
        self._fixups: list[tuple[int, str, str]] = []  # (index, imm operand name, label)

    # -- placement ---------------------------------------------------------------------------------
    def label(self, name: str) -> None:
        if name in self._labels:
            raise WorkloadError(f"duplicate label {name!r}")
        self._labels[name] = len(self._items)

    def emit(self, mnemonic: str, **operands: int) -> None:
        self._items.append((mnemonic, dict(operands)))

    def items(self) -> tuple[tuple[str, dict], ...]:
        """The emitted (mnemonic, operands) sequence, in issue order.

        The builder already holds this, and it is the only lossless form: `words` are encoded and
        recovering operands from them needs a decoder, while `kernel_s` carries the same information as
        comment TEXT. A dependence analysis wants the structure, so it is exposed rather than re-derived.
        """
        return tuple((m, dict(o)) for m, o in self._items)

    def branch(self, mnemonic: str, target: str, *, imm_operand: str = "imm", **operands: int) -> None:
        """A conditional branch to ``target``, followed by the machine's delay slot(s) filled with an
        instruction that is harmless on both paths. The delay slot is filled HERE rather than left to the
        caller because leaving it empty is the failure that looks like a working loop."""
        self._fixups.append((len(self._items), imm_operand, target))
        self._items.append((mnemonic, {**operands, imm_operand: 0}))
        for _ in range(self._cf.delay_slots):
            self.nop()

    def nop(self) -> None:
        """An instruction with no architectural effect: the selected add-immediate writing the register
        file's fixed-zero register (index 0 -- the ABI's convention, not an encoding constant)."""
        self.emit(self._ops.add_imm, rd=0, rs1=0, imm=0)

    def load_imm(self, rd: int, value: int) -> None:
        """Materialise a full-width constant, the way the ISA allows it: an upper immediate plus a
        sign-corrected low add. The low add sign-extends, so a low half with its top bit set must be
        compensated in the upper half or the register lands one page low."""
        width = self._imm_width(self._ops.add_imm, "imm")
        lo_mask = (1 << width) - 1
        sign = 1 << (width - 1)
        lo = value & lo_mask
        hi = (value - (lo - (1 << width) if lo & sign else lo)) >> width
        upper_width = self._imm_width(self._ops.load_upper, "imm")
        if hi:
            if hi >> upper_width:
                raise WorkloadError(f"constant {value} does not fit this ISA's upper immediate")
            self.emit(self._ops.load_upper, rd=rd, imm=hi & ((1 << upper_width) - 1))
            if lo:
                self.emit(self._ops.add_imm, rd=rd, rs1=rd, imm=lo)
        else:
            self.emit(self._ops.add_imm, rd=rd, rs1=0, imm=lo)

    def _imm_width(self, mnemonic: str, operand: str) -> int:
        bits = self._isa.fields_of(mnemonic).get(operand)
        if not bits:
            raise WorkloadError(f"this ISA's {mnemonic} carries no {operand!r} field")
        return len(bits)

    # -- output ------------------------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._items)

    def resolved(self) -> list[tuple[str, dict]]:
        """The instruction list with every branch immediate resolved against the measured contract."""
        out = [(mn, dict(ops)) for mn, ops in self._items]
        for index, operand, target in self._fixups:
            if target not in self._labels:
                raise WorkloadError(f"branch to undefined label {target!r}")
            imm = self._cf.branch_imm(index, self._labels[target])
            width = self._imm_width(out[index][0], operand)
            lo, hi = -(1 << (width - 1)), (1 << (width - 1)) - 1
            if not lo <= imm <= hi:
                raise WorkloadError(
                    f"branch at {index} to {target!r} needs immediate {imm}, outside this ISA's "
                    f"{width}-bit signed branch field; the loop cannot be encoded as a single branch"
                )
            out[index][1][operand] = imm & ((1 << width) - 1)
        return out

    def words(self) -> list[int]:
        from merlin.target.isa import isa_asm

        return [isa_asm.assemble_line(self._isa, mn, ops) for mn, ops in self.resolved()]

    def kernel_s(self, entry: str) -> str:
        """The ``.word`` stream stock LLVM assembles into instruction memory, each line annotated with the
        mnemonic and operands it was encoded from so the kernel stays reviewable."""
        from merlin.target.isa import isa_asm

        lines = [".section .text", f".globl {entry}", f".type {entry},@function", f"{entry}:"]
        at_label = {v: k for k, v in self._labels.items()}
        for i, (mn, ops) in enumerate(self.resolved()):
            if i in at_label:
                lines.append(f"# {at_label[i]}:")
            word = isa_asm.assemble_line(self._isa, mn, ops)
            args = ", ".join(f"{k}={v}" for k, v in ops.items())
            lines.append(f"  .word 0x{word:08x}    # [{i}] {mn}{' ' + args if args else ''}")
        return "\n".join(lines) + "\n"


def control_flow_probe_kernel(
    facts: MachineFacts,
    ops: KernelOps,
    *,
    trips: int,
    body_cycles: int,
    imm_scale: int,
    delay_slots: int,
    entry: str = "_start",
) -> str:
    """A kernel whose RUNTIME reveals whether a candidate branch encoding closes a loop.

    The body is a single stall of ``body_cycles``, so a loop that trips ``trips`` times costs about
    ``trips * body_cycles`` cycles and one that never loops costs about ``body_cycles``. The two are
    orders of magnitude apart, and a candidate that mis-targets the branch either halts early or never
    halts -- three outcomes, all distinguishable, none of them a plausible near-miss."""
    cf = ControlFlow(imm_scale, delay_slots, "candidate under test")
    p = _Program(facts.isa, cf, ops)
    x = _ScalarFile(facts.target, facts.isa, ops, ("counter", "limit"))
    counter, limit = x["counter"], x["limit"]
    p.load_imm(counter, 0)
    p.load_imm(limit, int(trips))
    p.label("loop")
    p.emit(ops.stall, rd=0, rs1=0, imm=int(body_cycles))
    p.emit(ops.add_imm, rd=counter, rs1=counter, imm=1)
    p.branch(ops.branch_ne, "loop", rs1=counter, rs2=limit)
    p.emit(ops.halt)
    return p.kernel_s(entry)


def probe_control_flow(
    facts: MachineFacts,
    ops: KernelOps,
    run_kernel: Callable[[str, int], int | None],
    *,
    trips: int = 8,
    body_cycles: int = 200,
    imm_scales: Sequence[int] = (2, 4, 1),
    delay_slot_candidates: Sequence[int] = (1, 0, 2),
    entry: str = "_start",
) -> ControlFlow:
    """DERIVE the machine's branch contract by executing candidate encodings on it.

    ``run_kernel(kernel_s, max_cycles) -> cycles`` runs one probe and returns the halt cycle, or ``None``
    when the program did not halt. The winning candidate is the one whose measured cycle count matches a
    loop that tripped ``trips`` times; a candidate whose branch lands short halts early, and one that
    lands on itself never halts. Raises when no candidate loops -- an honest "this machine's control flow
    is not encodable by this generator" rather than a kernel that silently runs its body once.

    The candidate ORDER is only a search order; the verdict is the measurement. The candidates are not
    hypothetical: two runners for the SAME machine measured here disagree -- its RTL halves the immediate
    and its functional model reads it as a byte offset -- so this must be run against the tier the
    workload will actually be graded on, and its answer travels with the plan."""
    expected = trips * body_cycles
    budget = int(expected * 4 + 4000)
    tried: list[str] = []
    for scale in imm_scales:
        for slots in delay_slot_candidates:
            src = control_flow_probe_kernel(
                facts, ops, trips=trips, body_cycles=body_cycles, imm_scale=scale, delay_slots=slots, entry=entry
            )
            try:
                cycles = run_kernel(src, budget)
            except Exception as e:  # noqa: BLE001 — a refusing candidate is evidence, not a failure
                tried.append(f"scale={scale} slots={slots}: {type(e).__name__}: {str(e)[-120:]}")
                continue
            if cycles is None:
                tried.append(f"scale={scale} slots={slots}: did not halt within {budget}")
                continue
            # A loop that tripped the full count spends at least (trips-1) further bodies beyond the
            # first; anything less means the branch did not close the loop.
            if cycles >= expected - body_cycles:
                return ControlFlow(
                    scale,
                    slots,
                    f"measured on {facts.target}: a {trips}-trip loop with a "
                    f"{body_cycles}-cycle body halted at {cycles} cycles with "
                    f"imm_scale={scale}, delay_slots={slots}",
                )
            tried.append(f"scale={scale} slots={slots}: halted at {cycles} (loop did not close)")
    raise WorkloadError(
        f"{facts.target!r}: no candidate branch encoding closed a loop on the device. Tried: " + "; ".join(tried)
    )


def probe_settle(
    facts: MachineFacts,
    ops: KernelOps,
    cf: ControlFlow,
    run_matmul: Callable[["MatmulPlan", int], Any],
    *,
    operands,
    ladder: Sequence[int] = (8, 16, 32, 64, 128, 256, 512),
    margin: int = 2,
    entry: str = "_start",
    numeric_policy: dict | None = None,
) -> Settle:
    """DERIVE the settle counts by finding the smallest one that computes the right answer on a single
    tile, then applying ``margin``.

    ``run_matmul(plan, max_cycles)`` runs a plan and returns its device output as a nested list, or
    ``None``. ``operands`` are the single-tile ``(A, W)`` the probe multiplies; the comparison is the
    plan's own bit-exact gate against the accumulator-format golden.

    LIMIT, stated rather than implied: a single tile runs the body ONCE, with no loop and therefore no
    reuse of a tensor register across iterations, so this establishes the settle a producer needs before
    its consumer -- not the settle a loop needs before it overwrites a register the previous iteration is
    still draining into. The margin exists for that gap, and the multi-tile run is what actually
    certifies it: every shape reported from this generator carries its own bit-exact verdict."""
    from merlin.analysis.perf.contract.matmul_plans import plan_matmul

    tile = facts.tile
    for cycles in ladder:
        plan = plan_matmul(
            facts,
            ops,
            m=tile.rows,
            k=tile.cols,
            n=tile.cols,
            control_flow=cf,
            settle=Settle.uniform(cycles, "candidate under test"),
            A=operands[0],
            W=operands[1],
            entry=entry,
            numeric_policy=numeric_policy,
        )
        try:
            got = run_matmul(plan, plan.suggested_max_cycles())
        except Exception:  # noqa: BLE001 — a candidate that faults is simply not the answer
            continue
        if got is not None and plan.matches(got):
            return Settle.uniform(
                cycles * int(margin),
                f"measured on {facts.target}: the smallest uniform settle that made a "
                f"{tile.rows}x{tile.cols}x{tile.cols} tile bit-exact was {cycles} "
                f"cycles; carried with a {margin}x margin",
            )
    raise WorkloadError(
        f"{facts.target!r}: no settle count in {list(ladder)} produced a correct single tile; the "
        f"schedule is wrong for a reason a longer wait does not fix"
    )


@dataclass(frozen=True)
class Placement:
    """One tensor's declared home: its logical shape and dtype, and the ABSOLUTE address the kernel
    stores to and the reader reads back from -- computed once, from one input, so the two cannot drift."""

    name: str
    role: str
    shape: list[int]
    dtype: str
    nbytes: int
    base: int


@dataclass(frozen=True)
class AliasSpan:
    name: str
    base: int
    nbytes: int
    reduced_lo: int
    reduced_hi: int  # exclusive, BEFORE reduction -- so a span past the window end is visible
    wraps: bool


@dataclass(frozen=True)
class AliasReport:
    """Whether this workload's footprint survives the runner's finite DRAM window.

    ``ok`` is True only when the window size was KNOWN and every span both fits inside it and misses
    every other span. ``ok`` False with ``window`` None means the check could not run -- which is not a
    pass. A run whose report is not ``ok`` must not be quoted as a cycle count: its operands aliased, so
    the number describes a computation nobody asked for."""

    window: int | None
    spans: tuple[AliasSpan, ...]
    wrapped: tuple[str, ...]
    collisions: tuple[tuple[str, str], ...]
    footprint_bytes: int
    ok: bool
    reason: str


def alias_report(placements: Sequence[Placement], window: int | None) -> AliasReport:
    """Reduce every tensor's span modulo the runner's window exactly as the runner does, and report any
    span that runs past the window's end (its tail wraps onto low addresses) or overlaps another."""
    spans: list[AliasSpan] = []
    total = 0
    for p in placements:
        lo = p.base & (window - 1) if window else p.base
        hi = lo + p.nbytes
        spans.append(AliasSpan(p.name, p.base, p.nbytes, lo, hi, wraps=bool(window and hi > window)))
        total += p.nbytes
    wrapped = tuple(s.name for s in spans if s.wraps)
    collisions: list[tuple[str, str]] = []
    for i, a in enumerate(spans):
        for b in spans[i + 1 :]:
            if a.reduced_lo < b.reduced_hi and b.reduced_lo < a.reduced_hi:
                collisions.append((a.name, b.name))
    if window is None:
        return AliasReport(
            None,
            tuple(spans),
            wrapped,
            tuple(collisions),
            total,
            False,
            "the runner publishes no DRAM window size, so the footprint COULD NOT BE CHECKED; this is not a pass",
        )
    if wrapped:
        return AliasReport(
            window,
            tuple(spans),
            wrapped,
            tuple(collisions),
            total,
            False,
            f"{len(wrapped)} tensor(s) run past the {window} B window and wrap onto low "
            f"addresses: {', '.join(wrapped)}",
        )
    if collisions:
        pairs = ", ".join(f"{a}~{b}" for a, b in collisions)
        return AliasReport(
            window,
            tuple(spans),
            wrapped,
            tuple(collisions),
            total,
            False,
            f"tensor spans collide after reduction into the {window} B window: {pairs}",
        )
    return AliasReport(
        window,
        tuple(spans),
        (),
        (),
        total,
        True,
        f"all {len(spans)} tensor spans ({total} B) fit disjointly inside the {window} B window; no address wraps",
    )


_SCALAR_ROLES = (
    "m_index",
    "n_index",
    "k_index",  # the three loop counters
    "dma_rd",
    "dma_rs1",
    "dma_len",  # the DMA endpoint registers
    "stage_addr",  # the tensor load/store base
    "a_tile",
    "w_tile",
    "c_tile",  # the three walking operand pointers
    "m_limit",
    "n_limit",
    "k_limit",  # the three loop bounds
    "a_row",
    "w_col",  # per-M and per-N restart points
    "tile_bytes",
    "a_row_stride",
    "w_k_stride",
    "c_stride",  # the four strides
    "w_origin",  # where the weight pointer restarts each M step
)


_M_LHS, _M_WT, _M_ACC = 0, 1, 2


_WEIGHT_SLOT, _ACC_SLOT = 0, 0


@dataclass(frozen=True)
class MatmulPlan:
    """A generated layer: where its tensors live, what its kernel is, and what the right answer is.

    Everything a run needs travels together, because the failure this replaces is a command buffer whose
    addresses and a kernel whose addresses were computed by different code."""

    facts: MachineFacts
    ops: KernelOps
    m: int
    k: int
    n: int
    placements: tuple[Placement, ...]
    kernel_s: str
    words: tuple[int, ...]
    control_flow: ControlFlow
    settle: Settle
    entry: str
    tiles: tuple[int, int, int]  # (m_tiles, k_tiles, n_tiles)
    # measured lengths of the emitted program's parts, so the unrolled-size comparison is not an estimate
    section_words: dict[str, int] = field(default_factory=dict)
    operands: dict[str, Any] = field(default_factory=dict)  # logical float arrays, when supplied
    golden: Any = None  # [m, n] reference, when computable
    #: The emitted (mnemonic, operands) sequence in issue order -- what a dependence
    #: analysis needs. Defaults to empty so an older caller constructing a plan by hand
    #: still type-checks; `plan_matmul` always supplies it.
    instructions: tuple[tuple[str, dict], ...] = ()

    # -- addressing --------------------------------------------------------------------------------
    def placement(self, name: str) -> Placement:
        for p in self.placements:
            if p.name == name:
                return p
        raise KeyError(name)

    def command_buffer(self) -> dict:
        """The declared command buffer: the tensors, at the addresses this plan computed, in the physical
        shape the device actually writes. The output is declared as the BANK STREAM the accumulator
        readout lays down (rows of one register's width), not as the logical matrix: the reader that
        un-permutes it is :meth:`unpack_output`, in this module, which is the same code that decided the
        layout."""
        tensors = {}
        for p in self.placements:
            tensors[p.name] = {"role": p.role, "shape": list(p.shape), "dtype": p.dtype, "base": p.base}
        return {
            "abi_version": "0.1",
            "target": self.facts.target,
            "tensors": tensors,
            "commands": [{"opcode": "MATMUL", "operands": {"lhs": "A", "rhs": "W", "dst": "C"}}],
        }

    def total_macs(self) -> int:
        return self.m * self.k * self.n

    def unrolled_word_estimate(self) -> int:
        """How many instruction words THIS schedule would need with the loops unrolled -- the number
        that decides whether a shape is reachable without control flow at all.

        Counted from the emitted program's own measured parts (one K-step body, one output-tile
        epilogue, the prologue) rather than approximated, so the comparison against the machine's
        instruction memory is the real one. This is the argument for the loops: the emitted program is
        a constant ~90 words at every shape and its unrolled twin is not."""
        mt, kt, nt = self.tiles
        return (
            self.section_words.get("prologue", 0)
            + mt * kt * nt * self.section_words.get("k_step", 0)
            + mt * nt * self.section_words.get("tile_epilogue", 0)
        )

    def moved_bytes(self) -> int:
        """Bytes this kernel's DMA actually moves, counted from the emitted schedule: one operand tile
        pair per K-step and one accumulator tile per output tile. Structural, not algorithmic -- which
        is the point, because a bound built on the bytes an algorithm NEEDS rather than the bytes a
        program MOVES is optimistic by exactly the transfer amplification."""
        mt, kt, nt = self.tiles
        tile = self.facts.tile.register_bytes
        return mt * kt * nt * 2 * tile + mt * nt * self.facts.banks_per_tile * tile

    def transfer_amplification(self) -> float:
        """Moved bytes over the workload's own footprint. At tile scale this ratio is dominated by the
        fixed per-tile transfer cost; at layer scale it is dominated by how often the schedule REFETCHES
        an operand, so the two do not extrapolate to one another and the layer-scale number is the one
        an optimisation argument may use."""
        return self.moved_bytes() / max(1, sum(p.nbytes for p in self.placements))

    def instruction_memory_fit(self) -> dict:
        """Whether the emitted program fits instruction memory, and whether its UNROLLED twin would.

        The second number is the one that matters: this generator loops because the unrolled schedule
        for a layer does not fit, and a program whose tail is never loaded does not fail loudly. When
        the capacity is not derivable both answers are ``None`` -- unchecked, which is not a pass."""
        cap = self.facts.imem_words
        unrolled = self.unrolled_word_estimate()
        return {
            "words": len(self.words),
            "unrolled_words": unrolled,
            "capacity_words": cap,
            "fits": None if cap is None else len(self.words) <= cap,
            "unrolled_fits": None if cap is None else unrolled <= cap,
            "unrolled_overflow": None if cap is None else round(unrolled / cap, 2),
        }

    def suggested_max_cycles(self, *, per_tile_pass: int = 4096, floor: int = 20000) -> int:
        """A HANG backstop sized to this plan's own tile count, not a fixed cap. It is deliberately loose:
        a healthy program halts on its own, and the wall-clock timeout is the outer backstop. A fixed cap
        is the defect that reports a correct long-running program as a missing oracle."""
        mt, kt, nt = self.tiles
        return max(int(floor), int(floor) + per_tile_pass * mt * kt * nt)

    # -- operands and results ----------------------------------------------------------------------
    def preloads(self) -> list[tuple[int, bytes]]:
        """``(address, bytes)`` for every input tensor, in the DEVICE layout this kernel reads."""
        out = []
        for name in ("A", "W"):
            raw = self.operands.get(name + "_bytes")
            if raw is None:
                continue
            out.append((self.placement(name).base, raw))
        return out

    def unpack_output(self, raw) -> Any:
        """The device's bank stream as the logical ``[m, n]`` matrix.

        The accumulator reads out into a register PAIR (or however many registers the accumulate format
        needs), each holding a column slice of the tile, and the kernel lays those banks down back to
        back per output tile in tile order. Un-permuting that here -- in the module that chose the order
        -- keeps the layout knowledge in one place instead of splitting it between a physical-layout
        declaration and a reader."""
        import numpy as np

        t = self.facts.tile
        banks = self.facts.banks_per_tile
        cols_per_bank = t.cols // banks
        mt, _kt, nt = self.tiles
        a = np.asarray(raw, dtype=np.float32).reshape(-1, cols_per_bank)
        out = np.zeros((self.m, self.n), dtype=np.float32)
        for i in range(mt):
            for j in range(nt):
                for b in range(banks):
                    r0 = ((i * nt + j) * banks + b) * t.rows
                    out[
                        i * t.rows : (i + 1) * t.rows,
                        j * t.cols + b * cols_per_bank : j * t.cols + (b + 1) * cols_per_bank,
                    ] = a[r0 : r0 + t.rows, :]
        return out

    def matches(self, device_output) -> bool:
        """True when the device's output equals this plan's golden BIT-EXACTLY. Bit-exact is the right
        bar because the golden models the device's own accumulator; a tolerance here would hide the
        loop-trip and layout errors this generator exists to catch."""
        import numpy as np

        if self.golden is None:
            raise WorkloadError("this plan carries no golden; supply operands to compare against")
        got = self.unpack_output(device_output)
        return bool(got.shape == self.golden.shape and np.array_equal(got, self.golden))

    def divergence(self, device_output) -> dict:
        """A description of HOW the device output differs from the golden, for a run that did not match."""
        import numpy as np

        got = self.unpack_output(device_output)
        ref = np.asarray(self.golden, dtype=np.float32)
        diff = np.abs(got - ref)
        return {
            "shape": list(got.shape),
            "mismatching": int(np.count_nonzero(got != ref)),
            "elements": int(ref.size),
            "max_abs_err": float(diff.max()) if diff.size else 0.0,
            "device_all_zero": bool(not np.any(got)),
        }


__all__ = [
    "AliasReport",
    "AliasSpan",
    "ControlFlow",
    "KernelOps",
    "MachineFacts",
    "MatmulPlan",
    "Placement",
    "Settle",
    "TileGeometry",
    "WorkloadError",
    "accumulate_reference",
    "alias_report",
    "candidate_ops",
    "control_flow_probe_kernel",
    "datapath_formats",
    "dram_window_bytes",
    "encode_operand_bytes",
    "instruction_memory_words",
    "machine_facts",
    "plan_matmul",
    "probe_control_flow",
    "probe_settle",
    "scalar_register_count",
    "tile_geometry",
]
