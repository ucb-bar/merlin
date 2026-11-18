"""Generate a LAYER-SCALE workload for a self-hosted-ISA target: a command buffer, an emitted
``kernel.S``, a host-computed golden, and a proof that the run's DRAM footprint did not wrap.

Why this module exists. Every performance number measured on such a target so far describes a demo
TILE -- the largest tensor in the graded corpus is a few thousand elements and the kernels halt in
hundreds of cycles. The programs that exist are static assembly with immediates baked in at authoring
time, so a new shape means new hand-written assembly, and the "full model shapes" the corpus cites have
no program, no golden and no run. Performance work is about model LAYERS, so the layer has to be
generated.

Four properties this generator has that a hand-written kernel does not:

* **It loops.** Instruction memory is finite -- 32,768 words on the target measured here, DERIVED from
  the instruction-memory region of its own memory map (:func:`instruction_memory_words`) -- and the
  unrolled schedule for a layer overflows it. The emitted kernel's size is therefore O(1) in the shape:
  the tile nest is three real backward branches, so the same 96-word program runs a 32x32x32 tile and
  the 416x832x416 layer whose unrolled twin needs 108,011 words (3.3x the machine's instruction
  memory). :meth:`MatmulPlan.instruction_memory_fit` reports both, and reports UNKNOWN -- not a pass --
  when the capacity is not derivable.
* **Its branches are encoded from the machine's own control-flow contract, MEASURED.** A branch
  immediate is not portable knowledge. On the target measured here the PC is a WORD index and the
  target is ``s1_pc + (imm >> 1)``, so a B-type immediate moves ``imm/2`` instructions, and there is an
  architectural delay slot whose instruction executes on both paths. Encoding a byte offset produces a
  kernel that assembles, runs, halts, and never closes its loop -- it reads as "this core has no control
  flow". :func:`probe_control_flow` therefore DERIVES ``(imm_scale, delay_slots)`` by running candidate
  encodings on the target's own oracle and keeping the one whose observable trip count is right. Nothing
  about control flow is written down here.
* **It refuses to produce a cycle count from an aliasing run.** The simulator's DRAM is a finite window
  and addresses are reduced modulo its size, silently: bytes past the end are dropped or wrap onto other
  tensors, and nothing in the returned result flags it. One 1024x3072 bf16 tensor is 6 MiB against a
  1 MiB window -- it would alias six ways and still return a number. :func:`alias_report` computes the
  masked span of every tensor from the window size DERIVED from the runner and reports the run INVALID
  if any span wraps or two spans collide. A wrong number is worse than a slow one.
* **Its golden is computed host-side, in the accumulator's own format.** A narrow-float accumulator
  rounds every partial sum, so an f32 reference grades a perfectly correct device as broken -- measured
  here at 796 of 4096 elements on a single layer. The reference models the DECLARED datapath (round
  after every MAC, operand subnormal flush when the profile declares it) and the gate is BIT-EXACT,
  which is the right bar precisely because it is reachable: a tolerance would hide the loop-trip and
  bank-layout errors this generator exists to catch. A mismatch is DESCRIBED
  (:meth:`MatmulPlan.divergence`) rather than absorbed.

Target-agnostic by construction: the instruction encodings come from
:func:`~merlin.target.isa.machine_model.isa_model_for_target` (probed from the target's own shipped ISA
definition), the tile geometry from the RTL-discovered array, the DRAM aperture from the target's memory
map, the operand/accumulate formats from its capability manifest, the DRAM window from the runner that
will execute the program, and the control-flow contract from the machine itself. No opcode, funct value,
field position, mesh dimension, latency or address is a literal in this file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from merlin.analysis.perf.contract.workload_facts import WorkloadError
from merlin.analysis.perf.contract.workload_programs import accumulate_reference, encode_operand_bytes

# ---- workload_gen ------------------------------------------------------------------------------


def _tile_counts(facts: MachineFacts, m: int, k: int, n: int) -> tuple[int, int, int]:
    from merlin.analysis.perf.contract.workload_facts import MachineFacts

    t = facts.tile
    for name, extent, edge in (("M", m, t.rows), ("K", k, t.cols), ("N", n, t.cols)):
        if extent <= 0 or extent % edge:
            raise WorkloadError(
                f"{name}={extent} is not a whole number of {edge}-wide tiles; this generator refuses "
                f"rather than contract over a partial tile"
            )
    return m // t.rows, k // t.cols, n // t.cols


def _place(facts: MachineFacts, m: int, k: int, n: int, *, origin: int, align: int) -> tuple[Placement, ...]:
    """Lay the three tensors out from ``origin``, each aligned. Absolute addresses: the kernel stores to
    them and the reader reads them back, and they are computed once here so the two cannot drift."""
    from merlin.analysis.perf.contract.workload_facts import MachineFacts
    from merlin.analysis.perf.contract.workload_programs import Placement

    ob, ab = facts.operand_bytes, facts.accum_bytes
    cur = origin
    out: list[Placement] = []
    t = facts.tile
    cols_per_bank = t.cols // facts.banks_per_tile
    for name, role, shape, dtype, nbytes in (
        ("A", "input", [m, k], facts.operand_dtype, m * k * ob),
        ("W", "weight", [k, n], facts.operand_dtype, k * n * ob),
        # the output's DECLARED shape is the bank stream the readout writes (see unpack_output)
        ("C", "output", [(m * n) // cols_per_bank, cols_per_bank], facts.accum_dtype, m * n * ab),
    ):
        cur = (cur + align - 1) // align * align
        out.append(Placement(name, role, shape, dtype, nbytes, cur))
        cur += nbytes
    return tuple(out)


def _pack_tiled(values, facts: MachineFacts, rows: int, cols: int, tile_rows: int, tile_cols: int) -> bytes:
    """A matrix as the TILE-MAJOR byte stream the kernel loads: each ``tile_rows x tile_cols`` block
    contiguous, blocks in row-major tile order. One transfer per tile instead of one per ROW is the
    difference between a loop body of eight instructions and one of two hundred, and between moving a
    tile's bytes once and moving its neighbours' bytes with it."""
    import numpy as np

    from merlin.analysis.perf.contract.workload_facts import MachineFacts

    a = np.asarray(values, dtype=np.float32).reshape(rows, cols)
    chunks = []
    for i in range(rows // tile_rows):
        for j in range(cols // tile_cols):
            chunks.append(a[i * tile_rows : (i + 1) * tile_rows, j * tile_cols : (j + 1) * tile_cols])
    return encode_operand_bytes(np.concatenate([c.reshape(-1) for c in chunks]), facts.operand_dtype)


def _issue_tile_transfer(
    p: _Program, facts: MachineFacts, ops: KernelOps, x: _ScalarFile, *, vmem_byte: int, dram_reg: int, nbytes: int
) -> None:
    """ISSUE one contiguous DRAM->VMEM transfer, without waiting for it.

    Split out from :func:`_stage_tile` so a second transfer can be issued before the first is awaited --
    the ``PC`` family's whole subject. The DMA argument registers are reused by the next issue, which is
    only sound because the transfer is POSTED: this target's ``explicit_completion`` trait is satisfied
    by its own elaborated FIRRTL (LoadController and StoreController each expose a DECOUPLED completion
    channel), so the controller latched the descriptor when it accepted the command. That is a derived
    fact, not an assumption about DMA engines in general, and on a target where it is UNKNOWN the family
    that uses this skips at its gate rather than emitting a program whose correctness nobody established.
    """
    from merlin.analysis.perf.contract.workload_facts import MachineFacts, _ScalarFile
    from merlin.analysis.perf.contract.workload_programs import KernelOps, _Program

    p.load_imm(x["dma_rd"], vmem_byte // facts.word_bytes)
    p.emit(ops.add_imm, rd=x["dma_rs1"], rs1=dram_reg, imm=0)
    p.load_imm(x["dma_len"], nbytes)
    p.emit(ops.dma_load, rd=x["dma_rd"], rs1=x["dma_rs1"], rs2=x["dma_len"])


def _await_transfer(p: _Program, ops: KernelOps) -> None:
    """Block until the movement channel is idle. Channel-level, not per-descriptor: with two transfers
    outstanding the first wait covers both and the second returns immediately -- which is why hoisting
    preserves the instruction multiset instead of removing a wait."""
    from merlin.analysis.perf.contract.workload_programs import KernelOps, _Program

    p.emit(ops.dma_wait)


def _stage_tile(
    p: _Program, facts: MachineFacts, ops: KernelOps, x: _ScalarFile, *, vmem_byte: int, dram_reg: int, nbytes: int
) -> None:
    """One contiguous transfer from DRAM into a VMEM staging slot, then wait for it. The VMEM side is a
    machine-word index and the DRAM side a byte offset from the aperture floor -- both derived, neither
    written down."""
    from merlin.analysis.perf.contract.workload_facts import MachineFacts, _ScalarFile
    from merlin.analysis.perf.contract.workload_programs import KernelOps, _Program

    _issue_tile_transfer(p, facts, ops, x, vmem_byte=vmem_byte, dram_reg=dram_reg, nbytes=nbytes)
    _await_transfer(p, ops)


def _tensor_op(p: _Program, ops: KernelOps, mnemonic: str, operands: dict, settle: int) -> None:
    """A tensor instruction and the settle its consumer needs. There is no interlock: reading the result
    early returns stale bytes, and a vector op issued too soon does not write its destination at all."""
    from merlin.analysis.perf.contract.workload_programs import KernelOps, _Program

    p.emit(mnemonic, **operands)
    p.emit(ops.stall, rd=0, rs1=0, imm=int(settle))


def _inner_body(
    p: _Program,
    facts: MachineFacts,
    ops: KernelOps,
    x: _ScalarFile,
    settle: Settle,
    *,
    first_k: bool,
    v_lhs: int,
    v_wt: int,
    hoist_transfers: bool = False,
) -> None:
    """One K-step: stage and push the weight tile, stage the activation tile, and multiply into the
    accumulator. The first K-step OVERWRITES the accumulator so this output tile does not inherit the
    previous one's sum; every later step accumulates into it.

    ``hoist_transfers`` issues the ACTIVATION transfer before the WEIGHT wait, so the two are outstanding
    together instead of strictly serialized. It is the ``PC`` family's lever
    (``issue_stage_two_transfer_before_stage_one_wait``), and the pair is a differential: the same
    dependence chain, the same instruction MULTISET, one ordering. Nothing else about the body changes --
    same tiles, same tensor ops, same settles -- because the comparand cancels everything but the order.
    """
    from merlin.analysis.perf.contract.workload_facts import MachineFacts, _ScalarFile
    from merlin.analysis.perf.contract.workload_programs import (
        _ACC_SLOT,
        _M_LHS,
        _M_WT,
        _WEIGHT_SLOT,
        KernelOps,
        Settle,
        _Program,
    )

    tile_bytes = facts.tile.register_bytes
    if hoist_transfers:
        return _inner_body_hoisted(
            p, facts, ops, x, settle, first_k=first_k, v_lhs=v_lhs, v_wt=v_wt, tile_bytes=tile_bytes
        )
    _stage_tile(p, facts, ops, x, vmem_byte=v_wt, dram_reg=x["w_tile"], nbytes=tile_bytes)
    p.load_imm(x["stage_addr"], v_wt // facts.word_bytes)
    _tensor_op(p, ops, ops.tile_load, {"vd": _M_WT, "rs1": x["stage_addr"], "imm": 0}, settle.tensor)
    # The weight buffer is read back with its axes swapped relative to the row-major tile in memory, so
    # the resident pack is a transpose. Pushing the tile as loaded computes A @ W-transpose instead.
    _tensor_op(p, ops, ops.transpose, {"vd": _M_WT, "vs1": _M_WT}, settle.tensor)
    _tensor_op(p, ops, ops.weight_push, {"vd": _WEIGHT_SLOT, "vs1": _M_WT}, settle.tensor)
    _stage_tile(p, facts, ops, x, vmem_byte=v_lhs, dram_reg=x["a_tile"], nbytes=tile_bytes)
    p.load_imm(x["stage_addr"], v_lhs // facts.word_bytes)
    _tensor_op(p, ops, ops.tile_load, {"vd": _M_LHS, "rs1": x["stage_addr"], "imm": 0}, settle.tensor)
    mnem = ops.contract if first_k else ops.contract_accumulate
    _tensor_op(p, ops, mnem, {"vd": _ACC_SLOT, "vs1": _M_LHS, "vs2": _WEIGHT_SLOT}, settle.mxu)
    # advance the two operand pointers to the next K-tile
    p.emit(ops.add, rd=x["a_tile"], rs1=x["a_tile"], rs2=x["tile_bytes"])
    p.emit(ops.add, rd=x["w_tile"], rs1=x["w_tile"], rs2=x["w_k_stride"])


def _inner_body_hoisted(
    p: _Program,
    facts: MachineFacts,
    ops: KernelOps,
    x: _ScalarFile,
    settle: Settle,
    *,
    first_k: bool,
    v_lhs: int,
    v_wt: int,
    tile_bytes: int,
) -> None:
    """The same K-step with both transfers outstanding before either wait.

    Read against :func:`_inner_body`: the only difference is that the activation ISSUE moves ahead of
    the weight WAIT. Every instruction that appears there appears here, once -- two issues, two waits,
    the same three tensor ops with the same settles, the same two pointer bumps -- so a cycle difference
    between the two is attributable to the ordering and nothing else. That equality is not a comment: it
    is what the family's comparand cancels, and a test compares the two mnemonic multisets.
    """
    from merlin.analysis.perf.contract.workload_facts import MachineFacts, _ScalarFile
    from merlin.analysis.perf.contract.workload_programs import (
        _ACC_SLOT,
        _M_LHS,
        _M_WT,
        _WEIGHT_SLOT,
        KernelOps,
        Settle,
        _Program,
    )

    _issue_tile_transfer(p, facts, ops, x, vmem_byte=v_wt, dram_reg=x["w_tile"], nbytes=tile_bytes)
    # THE HOIST: stage two's transfer, issued before stage one's wait.
    _issue_tile_transfer(p, facts, ops, x, vmem_byte=v_lhs, dram_reg=x["a_tile"], nbytes=tile_bytes)
    _await_transfer(p, ops)
    p.load_imm(x["stage_addr"], v_wt // facts.word_bytes)
    _tensor_op(p, ops, ops.tile_load, {"vd": _M_WT, "rs1": x["stage_addr"], "imm": 0}, settle.tensor)
    _tensor_op(p, ops, ops.transpose, {"vd": _M_WT, "vs1": _M_WT}, settle.tensor)
    _tensor_op(p, ops, ops.weight_push, {"vd": _WEIGHT_SLOT, "vs1": _M_WT}, settle.tensor)
    # The second wait: the channel is already idle, so it retires immediately. Kept so the multiset
    # matches the unhoisted body -- dropping it would make the pair differ by an instruction and the
    # cycle delta would no longer isolate the ordering.
    _await_transfer(p, ops)
    p.load_imm(x["stage_addr"], v_lhs // facts.word_bytes)
    _tensor_op(p, ops, ops.tile_load, {"vd": _M_LHS, "rs1": x["stage_addr"], "imm": 0}, settle.tensor)
    mnem = ops.contract if first_k else ops.contract_accumulate
    _tensor_op(p, ops, mnem, {"vd": _ACC_SLOT, "vs1": _M_LHS, "vs2": _WEIGHT_SLOT}, settle.mxu)
    p.emit(ops.add, rd=x["a_tile"], rs1=x["a_tile"], rs2=x["tile_bytes"])
    p.emit(ops.add, rd=x["w_tile"], rs1=x["w_tile"], rs2=x["w_k_stride"])


def plan_matmul(
    facts: MachineFacts,
    ops: KernelOps,
    *,
    m: int,
    k: int,
    n: int,
    control_flow: ControlFlow,
    settle: Settle,
    A=None,
    W=None,
    origin: int | None = None,
    align: int = 64,
    entry: str = "_start",
    subnormal_operand_flush: bool | None = None,
    hoist_transfers: bool = False,
    numeric_policy: dict | None = None,
) -> MatmulPlan:
    """Generate a LOOPED, tiled ``[m, k] x [k, n]`` contraction for ``facts.target``.

    The emitted kernel's length does not grow with the shape: the M, N and K axes are three real backward
    branches over a body that stages one tile of each operand. That is what makes a layer reachable at
    all -- a fully unrolled layer of this size overflows instruction memory several times over.

    ``A``/``W``, when given, are the logical float operands; the plan then carries their device-layout
    bytes and the accumulator-format golden. Without them the plan is still complete as a program (shape,
    addresses, kernel) and simply has no reference.

    **The operand DRAM layout is TILE-MAJOR** -- each tile contiguous, tiles in row-major tile order --
    and :attr:`MatmulPlan.operands` produces exactly those bytes, so the kernel and the preload cannot
    disagree. This is a stated layout DECISION, not a hardware constraint, and it is the one that decides
    the workload's transfer character: a tile-major operand moves in ONE transfer, a row-major one needs
    a gather of ``rows`` descriptors whose bytes are the same but whose descriptor count is ``rows``
    times larger. Report :meth:`MatmulPlan.transfer_amplification` with the layout it was measured under;
    the two are not comparable.
    """
    from merlin.analysis.perf.contract.workload_facts import MachineFacts, _ScalarFile
    from merlin.analysis.perf.contract.workload_programs import (
        _ACC_SLOT,
        _M_ACC,
        _SCALAR_ROLES,
        ControlFlow,
        KernelOps,
        MatmulPlan,
        Settle,
        _Program,
    )

    mt, kt, nt = _tile_counts(facts, m, k, n)
    t = facts.tile
    tile_bytes = t.register_bytes
    banks = facts.banks_per_tile
    origin = facts.dram_base + align if origin is None else origin
    placements = _place(facts, m, k, n, origin=origin, align=align)
    by = {p.name: p for p in placements}
    # The kernel addresses DRAM as an offset from the aperture floor; the command buffer and the reader
    # use the absolute address. One subtraction, in one place.
    a_off = by["A"].base - facts.dram_base
    w_off = by["W"].base - facts.dram_base
    c_off = by["C"].base - facts.dram_base
    # VMEM staging: one slot per operand tile plus the accumulator's banks. Only one of each is live.
    v_lhs, v_wt, v_out = 0, tile_bytes, 2 * tile_bytes

    p = _Program(facts.isa, control_flow, ops)
    x = _ScalarFile(facts.target, facts.isa, ops, _SCALAR_ROLES)
    for role, value in (
        ("m_limit", mt),
        ("n_limit", nt),
        ("k_limit", kt),
        ("tile_bytes", tile_bytes),
        ("a_row_stride", kt * tile_bytes),
        ("w_k_stride", nt * tile_bytes),
        ("c_stride", banks * tile_bytes),
        ("w_origin", w_off),
        ("a_row", a_off),
        ("c_tile", c_off),
    ):
        p.load_imm(x[role], value)
    p.load_imm(x["m_index"], 0)

    p.label("m_loop")
    p.emit(ops.add_imm, rd=x["w_col"], rs1=x["w_origin"], imm=0)
    p.load_imm(x["n_index"], 0)

    p.label("n_loop")
    p.emit(ops.add_imm, rd=x["a_tile"], rs1=x["a_row"], imm=0)
    p.emit(ops.add_imm, rd=x["w_tile"], rs1=x["w_col"], imm=0)
    # The first K-step is PEELED because it is the one that overwrites the accumulator rather than adding
    # to it -- a different instruction, not a different operand, so it cannot be selected inside the loop.
    _prologue_words = len(p)
    _before_body = len(p)
    _inner_body(p, facts, ops, x, settle, first_k=True, v_lhs=v_lhs, v_wt=v_wt, hoist_transfers=hoist_transfers)
    _k_step_words = len(p) - _before_body
    if kt > 1:
        p.load_imm(x["k_index"], 1)
        p.label("k_loop")
        _inner_body(p, facts, ops, x, settle, first_k=False, v_lhs=v_lhs, v_wt=v_wt, hoist_transfers=hoist_transfers)
        p.emit(ops.add_imm, rd=x["k_index"], rs1=x["k_index"], imm=1)
        p.branch(ops.branch_ne, "k_loop", rs1=x["k_index"], rs2=x["k_limit"])

    # this output tile is complete: read the accumulator out, stage it and move it to DRAM
    _before_epilogue = len(p)
    _tensor_op(p, ops, ops.acc_read, {"vd": _M_ACC, "vs1": 0, "vs2": _ACC_SLOT}, settle.vpu)
    for b in range(banks):
        p.load_imm(x["stage_addr"], (v_out + b * tile_bytes) // facts.word_bytes)
        _tensor_op(p, ops, ops.tile_store, {"vd": _M_ACC + b, "rs1": x["stage_addr"], "imm": 0}, settle.tensor)
    p.emit(ops.add_imm, rd=x["dma_rd"], rs1=x["c_tile"], imm=0)
    p.load_imm(x["dma_rs1"], v_out // facts.word_bytes)
    p.load_imm(x["dma_len"], banks * tile_bytes)
    p.emit(ops.dma_store, rd=x["dma_rd"], rs1=x["dma_rs1"], rs2=x["dma_len"])
    p.emit(ops.dma_wait)
    p.emit(ops.add, rd=x["c_tile"], rs1=x["c_tile"], rs2=x["c_stride"])
    _tile_epilogue_words = len(p) - _before_epilogue
    p.emit(ops.add, rd=x["w_col"], rs1=x["w_col"], rs2=x["tile_bytes"])
    if nt > 1:
        p.emit(ops.add_imm, rd=x["n_index"], rs1=x["n_index"], imm=1)
        p.branch(ops.branch_ne, "n_loop", rs1=x["n_index"], rs2=x["n_limit"])
    p.emit(ops.add, rd=x["a_row"], rs1=x["a_row"], rs2=x["a_row_stride"])
    if mt > 1:
        p.emit(ops.add_imm, rd=x["m_index"], rs1=x["m_index"], imm=1)
        p.branch(ops.branch_ne, "m_loop", rs1=x["m_index"], rs2=x["m_limit"])

    p.emit(ops.halt)

    operands: dict[str, Any] = {}
    golden = None
    if A is not None and W is not None:
        import numpy as np

        A = np.asarray(A, dtype=np.float32).reshape(m, k)
        W = np.asarray(W, dtype=np.float32).reshape(k, n)
        operands = {
            "A": A,
            "W": W,
            "A_bytes": _pack_tiled(A, facts, m, k, t.rows, t.cols),
            "W_bytes": _pack_tiled(W, facts, k, n, t.rows, t.cols),
        }
        if subnormal_operand_flush is None:
            from merlin.development.targetgen.corpus import binding as CS

            if numeric_policy is None and not CS.dtype_info(facts.accum_dtype)[3]:
                raise WorkloadError(
                    "floating reference requires an explicit declared numeric_policy or subnormal_operand_flush"
                )
            subnormal_operand_flush = bool((numeric_policy or {}).get("subnormal_operand_flush", False))
        golden = accumulate_reference(
            A,
            W,
            accum_dtype=facts.accum_dtype,
            operand_dtype=facts.operand_dtype,
            subnormal_operand_flush=subnormal_operand_flush,
        )
        if golden is None:
            golden = A @ W

    return MatmulPlan(
        facts=facts,
        ops=ops,
        m=m,
        k=k,
        n=n,
        placements=placements,
        kernel_s=p.kernel_s(entry),
        words=tuple(p.words()),
        control_flow=control_flow,
        settle=settle,
        entry=entry,
        tiles=(mt, kt, nt),
        section_words={
            "prologue": _prologue_words,
            "k_step": _k_step_words,
            "tile_epilogue": _tile_epilogue_words,
            "total": len(p),
        },
        operands=operands,
        golden=golden,
        instructions=p.items(),
    )
