"""Accumulator-resident RVV GEMM micro-kernel codegen (default-off feature).

This is the genuine compiler-emitted answer to the #1 scalable-RVV-GEMM gap documented in
``output/kernels/ceiling/scalable_gap_result.md``: the upstream
``tile -> vectorize -> bufferize`` lowering re-reads/re-writes the MR x NR C-accumulator THROUGH
MEMORY every K-tile (a ``vector.transfer_read``/``transfer_write`` of the accumulator inside the K
loop; after bufferize the K-loop carries the accumulator as a *memref* iter_arg and BOTH
``hoist_redundant_vector_transfers`` and ``loop-invariant-subset-hoisting`` no-op on it). That
operand round-trip — not the vfmacc arithmetic — is the ~19x gap the prior transform-only
``accumulator_resident_microkernel`` could not close (its emitted K-loop still spills the
accumulator via ``vl4re8.v``/``vs4r.v`` per K-tile).

TWO compiler changes (both default-off, both ride the existing seams) make the compiler emit the
SAME register-blocked, accumulator-resident, ``vfmacc.vf`` micro-kernel the hand ceiling reference
has — without a hand kernel:

  1. PRE-bufferize subset hoist (``impr_features._accumulator_resident_v2_pipeline``):
     run ``loop-invariant-subset-hoisting`` on the TENSOR form (before one-shot-bufferize), where the
     K-loop carries the accumulator as a value-semantic ``tensor<MRxNR>`` iter_arg and the
     accumulator transfer pair reads/writes that iter_arg at loop-invariant indices. On THAT form the
     pass fires: it lifts the ``vector.transfer_read`` above the K-loop and the
     ``vector.transfer_write`` below it, threading a pure ``vector<MRxNR>`` through the loop as a
     second iter_arg. After bufferize that lowers to an ``!llvm.array<MR x vector<NRxf32>>``
     loop-carried value the RISC-V backend keeps in vregs across K (register-resident).

  2. A-operand SCALARIZATION (this module, :func:`scalarize_a_reads`): even with the accumulator
     resident, the contraction's A operand was read as a ``vector<MRx1xf32>`` ``vector.transfer_read``
     and each row extracted ``[i,0] : f32``; the RISC-V backend cannot cheaply move a vector LANE into
     the ``.vf`` scalar FP operand, so it reconstructs the broadcast with a ``vmv``/``vslideup`` ladder
     and emits ``vfmacc.vv`` (measured: that ladder, not a spill, dominated the residual instret —
     the v2 feature was still ~19x off). When A is instead read as a SCALAR ``tensor.extract`` /
     ``memref.load`` (``flw`` into an FP register), clang-23 selects the clean ``vfmacc.vf`` directly
     (verified: ``fma(splat(load float), vec, acc) -> vfmacc.vf``). This rewrite matches each
     ``vector.transfer_read`` whose result is ``vector<MRx1xf32>`` and whose only uses are
     ``vector.extract [i,0] : f32``, and replaces it with per-row scalar loads from the same source —
     the SAME ``a[i]`` scalar the hand kernel loads. It is a GENERAL structural rewrite (any MR, any
     contraction whose lhs register tile has a trailing unit dim), not a shape/op-specific kernel, and
     it changes nothing numerically (a scalar load of element ``[i,0]`` == extracting lane ``[i,0]`` of
     the vector load of the same slice) so the result is BIT-EXACT vs the un-rewritten lowering.

It runs as a Python ``ir``-API rewrite spliced into the lowering runner BETWEEN two PassManager
stages (the contraction must already be lowered to ``vector.fma`` with f32 A-extracts, and bufferize
must not have run yet). The pipeline edit inserts a sentinel marker pass name where the split
happens; :func:`run_source` (executed in the model2MLIR venv) parses the pipeline, runs stage 1, does
this rewrite, then runs stage 2. With the feature off this module is never imported and the pipeline
is byte-identical to the baseline.
"""
from __future__ import annotations

from .concat_dps import RUNNER_PRELUDE as _CONCAT_DPS_PRELUDE
from .copy_expand import MID_STAGE_SRC as _MID_STAGE_SRC
from .copy_expand import RUNNER_PRELUDE as _COPY_EXPAND_PRELUDE
from .parallel_grain import LATE_STAGE_SRC as _PARALLEL_GRAIN_LATE_SRC
from .parallel_grain import RUNNER_PRELUDE as _PARALLEL_GRAIN_PRELUDE
from .selfcopy import RUNNER_PRELUDE as _SELFCOPY_PRELUDE
from .transpose_maps import RUNNER_PRELUDE as _TRANSPOSE_MAPS_PRELUDE

# Sentinel pass name spliced into the pipeline string by the feature's edit_pipeline to mark where
# the A-scalarization rewrite runs (after contract->vector.fma lowering, before one-shot-bufferize).
# It is NOT a real MLIR pass; the runner splits the pipeline here and never passes it to mlir-opt.
SCALARIZE_MARKER = "__merlin_scalarize_a__"


# The rewriter source, spliced into the lowering runner (which executes in the m2m venv with the
# upstream MLIR Python bindings). Kept as a source string so the runner stays one self-contained
# script (same mechanism as act_poly.rewrite_source).
_REWRITER_SRC = r'''
def _merlin_walk(op, fn):
    for region in op.regions:
        for block in region.blocks:
            for o in list(block.operations):
                fn(o)
                _merlin_walk(o, fn)


# The widening ops a scalar extract may be sunk BELOW, and the scalar result spellings that mark a
# lane extract (as opposed to a sub-vector slice). Float and integer are the same rewrite: lane i of
# `widen(v)` == `widen(lane i of v)` for an exact widening, which fpext, sext and zext all are.
_WIDEN_OPS = ("arith.extf", "arith.extsi", "arith.extui")
_SCALAR_TYS = ("f16", "bf16", "f32", "f64", "i8", "i16", "i32", "i64")


def scalarize_a_reads(module, ctx):
    """Replace each `vector.transfer_read -> vector<MRx1xT>` (T = f32 / f16 / bf16) whose only uses
    are `vector.extract [i, 0] : T` with per-row scalar `tensor.extract`/`memref.load`, so the A
    operand of the register-blocked vfmacc reaches the backend as a scalar (flw -> vfmacc.vf) instead
    of a reconstructed vector lane (vmv/vslideup -> vfmacc.vv). Returns the count rewritten.

    Numerically identical: element [i,0] of the slice == lane [i,0] of the vector read of the slice.

    ELEMENT TYPES: originally f32-only, which silently excluded the 16-bit-float datapaths --
    an fp16 matmul reached this pass as `vector<MRx1xf16>` + `vector.extract : f16`, failed the
    f32 gate, kept the vector-lane A operand, and so never formed a `.vf` MAC at all (measured:
    ZERO vfmacc/vfwmacc in the emitted fp16 kernel, versus 8 vfmacc.vf for the same shape in
    f32). The rewrite is element-type-agnostic by construction -- it only turns a vector read
    plus scalar extract into a scalar load -- so admitting f16/bf16 puts them on the SAME
    micro-kernel path rather than a parallel one. With a 16-bit A scalar and an f32 accumulator
    the backend forms `vfwmacc.vf` (widening MAC), the fp16 analogue of f32's `vfmacc.vf`.

    WHOLE-MODEL SAFETY: the rewrite must only fire on the matmul register-tile A read it was designed
    for; on a real model the IR carries OTHER `vector<...x1xf32>` transfer_reads consumed only by
    scalar `vector.extract` (e.g. an attention / elementwise read with a permuting or rank-reducing
    permutation_map, or a higher-rank source). For those the naive index reconstruction (one index
    per extract `static_position` entry) emits a `tensor.extract` whose index COUNT does not match the
    source rank -> ``'tensor.extract' op incorrect number of indices`` -> the whole pipeline aborts
    (PipelineError) -> the whole-model build silently falls back to SCALAR. So the matcher is
    restricted to the reads where the per-row scalar load is PROVABLY sound and value-identical:
      * the source is a STATICALLY-ranked tensor/memref;
      * every extract's `static_position` length == the source rank (so reconstruction produces
        exactly ``rank`` indices, one per source dim — the failure mode is a higher-rank source read
        into a lower-rank vector, where poslen < rank and the emitted `tensor.extract` is malformed);
      * the read has an IDENTITY permutation_map whose output rank == the source rank (no transpose /
        broadcast / rank-reduction that would make lane[i,0] of the read != element[pos] of the
        source). A read with no permutation_map attr is the implicit minor-identity; we still require
        the equal-rank identity below via the poslen==rank gate. NOTE: `transfer_read` carries trailing
        non-index operands (the padding scalar, an optional mask), so the index COUNT is NOT
        ``len(operands)-1``; we key the gate on the source rank + extract position length instead.
    Every read that fails these gates is LEFT on the correct (un-rewritten) baseline lowering — it
    still lowers to a valid vfmacc.vv, just not the .vf form. This composes the .vf micro-kernel onto
    the matched matmuls without ever breaking the rest of the model.
    """
    from torch_mlir import ir
    targets = []

    def _src_dims(ty_str):
        # static dim extents of a ranked tensor<...>/memref<...> from its printed type; None if not
        # static. e.g. "tensor<1x16xf32>" -> [1, 16].
        if not (ty_str.startswith("tensor<") or ty_str.startswith("memref<")):
            return None
        inner = ty_str[ty_str.index("<") + 1: ty_str.rindex(">")]
        # drop a memref layout/space suffix after the element type (", strided<...>"/", #space")
        dims = inner.split("x")
        # the last token is the element type (+ optional layout); everything before are dim sizes.
        out = []
        for d in dims[:-1]:
            if d == "?":
                return None  # dynamic dim -> reconstruction not provably 1:1, leave it alone
            try:
                out.append(int(d))
            except ValueError:
                return None
        return out

    def _src_rank(ty_str):
        # rank of a static ranked tensor<...>/memref<...>; -1 if not static.
        d = _src_dims(ty_str)
        return -1 if d is None else len(d)

    def _is_identity_perm(op, rank):
        # accept a missing permutation_map (implicit minor-identity == identity for an equal-rank
        # read), an explicit identity affine map `(d0,...) -> (d0,...)`, OR a rank-reducing MINOR-
        # IDENTITY PROJECTION `(d0,...,dk) -> (dj,...,dk)` whose outputs are the TRAILING inputs in
        # order (the read drops some LEADING source dims but keeps the rest in order — the form the
        # drop-unit-dims patterns emit when MR=1 collapses `vector<1x1>` to `vector<1>`). Soundness of
        # dropping those leading dims is enforced separately in `visit` (each dropped leading dim must
        # have extent 1). A genuine transpose / broadcast / out-of-order projection is still rejected.
        try:
            pm = op.attributes["permutation_map"]
        except KeyError:
            return True
        s = str(pm)               # e.g. `affine_map<(d0, d1) -> (d0, d1)>`
        if "->" not in s:
            return False
        lhs, rhs = s.split("->", 1)
        # take the dim list inside the LAST parens on each side (skips the `affine_map<` prefix).
        def _dims(part):
            if "(" not in part or ")" not in part:
                return None
            inner = part[part.index("(") + 1: part.index(")")]
            return [t.strip() for t in inner.split(",") if t.strip()]
        ins, outs = _dims(lhs), _dims(rhs)
        if ins is None or outs is None or len(ins) != rank:
            return False
        if ins == outs:
            return True                       # full identity (equal-rank read)
        # minor-identity projection: outputs are the trailing `len(outs)` inputs, in order.
        return len(outs) < len(ins) and outs == ins[len(ins) - len(outs):]

    def visit(o):
        if o.operation.name != "vector.transfer_read":
            return
        res = o.results[0]
        ts = str(res.type)
        if not ts.startswith("vector<"):
            return
        # ELEMENT TYPE: f32 is the original fp32 micro-kernel; f16/bf16 are the 16-bit-float
        # datapaths (dtype_strategy fp16_f32acc / bf16_f32acc), whose A operand is a 16-bit
        # scalar feeding an f32 accumulator -> the backend forms `vfwmacc.vf` (widening MAC)
        # exactly as it forms `vfmacc.vf` for f32. The rewrite itself is element-type-agnostic
        # (it turns a vector read + scalar extract into a scalar load), so the ONLY thing that
        # had to change is which element types we admit and what type we build the load with.
        #
        # INTEGER (i8/i16/i32): the same argument, one datapath over. The int8 contract lowers
        # through `outerproduct` to arith.muli/addi with a `vector.extract : i8` A operand, so
        # without an integer admission here every int8 MR>1 tile rebuilt A with a vmv/vslideup
        # lane ladder instead of letting the backend form `vwmacc.vx` from a scalar. That ladder
        # is the mechanism `perop_blocks.DEFAULT_MR` cites for pinning MR=1 -- so excluding the
        # integer types here is what made MR>1 look intrinsically bad on int8. Admitting them
        # changes NO soundness gate below (static rank, identity/minor-identity permutation,
        # poslen<=rank, unit leading dims all still apply); only the spelling set widens.
        elem = None
        for cand in ("f32", "f16", "bf16", "i8", "i16", "i32"):
            if ts.endswith("x" + cand + ">"):
                elem = cand
                break
        if elem is None:
            return
        shape = ts[len("vector<"):ts.rindex("x" + elem + ">")]
        dims = shape.split("x")
        # The register-tile lhs A read, in EITHER of the two equivalent forms the lowering can leave:
        #   (a) rank>=2 with a TRAILING unit dim, `vector<MR x 1>` (the MR>1 register tile); or
        #   (b) the rank-1 single-lane `vector<1>` the drop-unit-dims patterns collapse `vector<1x1>`
        #       to when MR=1 (the whole-model MR_mm=1 M-tail clamp). Both extract a SCALAR A element;
        #       (b) is what made the MR=1 token-decode matmul keep vfmacc.vv (scalarize rewrote 0)
        #       even though the read is just as soundly a scalar A[i] load.
        if dims[-1] != "1":
            return
        if len(dims) < 2 and dims != ["1"]:
            return  # rank-1 only accepted when it is the single-lane `vector<1>` form
        src_ty = str(o.operands[0].type)
        rank = _src_rank(src_ty)
        src_dims = _src_dims(src_ty)
        # source must be statically ranked and identity-mapped (no transpose/broadcast/rank-reduce).
        if rank < 0 or src_dims is None or not _is_identity_perm(o.operation, rank):
            return
        exs = []
        for u in res.uses:
            owner = u.owner
            if owner.name != "vector.extract" or str(owner.results[0].type) != elem:
                return  # a non-scalar use -> leave this read alone (keep it correct/general)
            poslen = len(list(owner.attributes["static_position"]))
            # The extract position addresses the TRAILING `poslen` source dims; the read may have
            # dropped `rank - poslen` LEADING (minor-identity) dims. The per-row scalar load is
            # provably value-identical ONLY when every dropped leading dim has extent 1 (so its sole
            # valid index is the read's base offset). poslen>rank can never be sound; poslen==rank is
            # the original MR>1 case (no dropped dim). Otherwise (poslen<rank) require leading unit.
            if poslen > rank:
                return  # extract position rank > source rank -> reconstruction would be malformed
            if poslen < rank and any(e != 1 for e in src_dims[:rank - poslen]):
                return  # a dropped leading dim with extent>1 -> lane[pos] != element[base..,pos]
            exs.append(owner)
        if exs:
            targets.append((o, exs, elem))

    _merlin_walk(module.operation, visit)
    idxty = ir.IndexType.get(ctx)
    _ELEM_TY = {"f32": ir.F32Type.get(ctx), "f16": ir.F16Type.get(ctx),
                "bf16": ir.BF16Type.get(ctx),
                "i8": ir.IntegerType.get_signless(8, ctx),
                "i16": ir.IntegerType.get_signless(16, ctx),
                "i32": ir.IntegerType.get_signless(32, ctx)}
    n = 0
    for read, exs, elem in targets:
        scalar_ty = _ELEM_TY[elem]
        src_val = read.operands[0]
        base_idx = list(read.operands[1:])
        rank = _src_rank(str(src_val.type))
        src_is_tensor = str(src_val.type).startswith("tensor")
        opname = "tensor.extract" if src_is_tensor else "memref.load"
        for ex in exs:
            pos_attr = ex.operation.attributes["static_position"]
            pos = []
            for a in pos_attr:
                try:
                    pos.append(ir.IntegerAttr(a).value)
                except Exception:  # noqa: BLE001
                    pos.append(int(str(a)))
            # The extract position addresses the TRAILING dims; if the read dropped leading
            # (unit-extent) dims, prepend a zero offset for each so we emit exactly `rank` indices.
            # (gated above: every dropped leading dim has extent 1, so the only index is base+0.)
            pos = [0] * (rank - len(pos)) + pos
            ip = ir.InsertionPoint(ex.operation)
            new_idx = []
            for d, p in enumerate(pos):
                b = base_idx[d] if d < len(base_idx) else None
                if p == 0 and b is not None:
                    new_idx.append(b)
                else:
                    c = ir.Operation.create(
                        "arith.constant", results=[idxty],
                        attributes={"value": ir.IntegerAttr.get(idxty, p)}, ip=ip).results[0]
                    if b is None:
                        new_idx.append(c)
                    else:
                        new_idx.append(ir.Operation.create(
                            "arith.addi", results=[idxty], operands=[b, c], ip=ip).results[0])
            scalar = ir.Operation.create(
                opname, results=[scalar_ty], operands=[src_val, *new_idx], ip=ip).results[0]
            ex.operation.results[0].replace_all_uses_with(scalar)
            ex.operation.erase()
        read.operation.erase()
        n += 1
    return n


def sink_extf_through_extract(module, ctx):
    """Sink a vector float widening below a SCALAR extract:
    ``vector.extract (arith.extf %v : vec<Nxf16> to vec<Nxf32>)[i] : f32``
    -> ``arith.extf (vector.extract %v[i] : f16 from vec<Nxf16>) : f16 to f32``.

    This is what makes the mixed-precision (f16/bf16 operand, f32 accumulator) register-tile A
    operand reach the RISC-V backend as a SCALAR half feeding the multiply, so it forms the WIDENING
    ``vfwmacc.vf`` (e16 operands, e32 accumulator, one rounding) instead of first widening the whole
    A column to f32 with ``vfwcvt.f.f.v`` and issuing an e32 ``vfmacc.vv``. The outerproduct
    lowering emits ``extract(extf(A_col))`` per row; MLIR's own ``sink_ops`` pattern performs exactly
    this sink but REFUSES it when the ``extf`` feeds more than one extract (it will not DUPLICATE the
    extf) -- which is precisely the MR>1 register tile (MR extracts of one A column). Duplicating one
    vector ``extf`` into MR scalar ``extf`` is beneficial here: each scalar ``extf`` is a free
    ``fpext`` of a value already in an FP register and folds into the ``.vf`` scalar operand of
    ``vfwmacc.vf``. Value-identical: ``fpext`` is exact and lane i of ``extf(v)`` == ``extf(lane i of
    v)``. Returns the count of extracts rewritten. Only fires on the widen->scalar-extract shape, so
    the f32 datapath (no operand widening) and every non-widening extract are untouched.

    INTEGER: the identical argument holds for ``arith.extsi`` / ``arith.extui``, so both are accepted.
    The int8 contract widens the A column on the VECTOR (``extsi vector<MRx1xi8> to vector<MRx1xi32>``)
    and extracts an i32 lane, which is the shape that blocked ``scalarize_a_reads`` (its only accepted
    use of the A read is a same-element-type ``vector.extract``, and the interposed vector ``extsi``
    is not one). Measured before this change: int8 MR=4 emitted 8 ``extractelement`` with a
    ``vrgather.vi`` lane ladder + ``vmacc.vv``, while int8 MR=1 emitted ``vwmacc.vx``. Sinking the
    integer widening first exposes the i8 lane extract, after which the scalarization fires and the
    backend can form ``vwmacc.vx`` at MR>1 too. ``sext``/``zext`` are exact, so lane i of ``widen(v)``
    == ``widen(lane i of v)`` just as for ``fpext``.
    """
    from torch_mlir import ir

    def _defop(val):
        try:
            owner = val.owner
        except Exception:  # noqa: BLE001
            return None
        return owner if hasattr(owner, "name") else None   # Operation (OpResult), not a Block

    targets = []

    def visit(o):
        op = o.operation
        if op.name != "vector.extract":
            return
        res = op.results[0]
        # scalar result only (a lane extract, not a sub-vector slice).
        if str(res.type) not in _SCALAR_TYS:
            return
        # all-static position (register-tile indices are constants); a dynamic index would add
        # operands beyond the source -- leave those alone.
        if len(list(op.operands)) != 1:
            return
        defop = _defop(op.operands[0])
        if defop is None or defop.name not in _WIDEN_OPS:
            return
        targets.append(op)

    _merlin_walk(module.operation, visit)
    n = 0
    seen_extf = []
    for ex in targets:
        extf = ex.operands[0].owner              # the widening op (extf / extsi / extui)
        widen_name = extf.name                   # rebuild with the SAME widening, never a fixed one
        narrow_src = extf.operands[0]            # the f16/bf16/i8/i16 vector
        try:
            narrow_elem = ir.VectorType(narrow_src.type).element_type
        except Exception:  # noqa: BLE001
            continue
        pos_attr = ex.attributes["static_position"]
        ip = ir.InsertionPoint(ex)
        lane = ir.Operation.create(
            "vector.extract", results=[narrow_elem], operands=[narrow_src],
            attributes={"static_position": pos_attr}, ip=ip).results[0]
        widened = ir.Operation.create(
            widen_name, results=[ex.results[0].type], operands=[lane], ip=ip).results[0]
        ex.results[0].replace_all_uses_with(widened)
        if extf not in seen_extf:
            seen_extf.append(extf)
        ex.erase()
        n += 1
    # Drop each now-dead vector extf (its lanes are all sunk); leave any with remaining uses.
    for extf in seen_extf:
        if len(list(extf.results[0].uses)) == 0:
            extf.erase()
    return n
'''


def rewrite_source() -> str:
    """Self-contained Python source of the A-scalarization rewriter, prepended to the runner."""
    return _REWRITER_SRC


def run_source() -> str:
    """The lowering-runner body for this feature: split the pipeline at SCALARIZE_MARKER, run stage 1
    (forms the resident accumulator + lowers the contraction to vector.fma with f32 A-extracts), run
    the A-scalarization rewrite, then run stage 2 (bufferize -> LLVM). Mirrors the act_poly runner
    splice; executes in the m2m venv."""
    return (
        "import sys\n"
        "from torch_mlir import ir\n"
        "from torch_mlir.passmanager import PassManager\n"
        "from torch_mlir.dialects import llvm\n"
        + _SELFCOPY_PRELUDE
        + _COPY_EXPAND_PRELUDE
        + _CONCAT_DPS_PRELUDE
        + _TRANSPOSE_MAPS_PRELUDE
        + _PARALLEL_GRAIN_PRELUDE
        + _REWRITER_SRC
        + _MID_STAGE_SRC
        + _PARALLEL_GRAIN_LATE_SRC +
        f"\nMARKER = {SCALARIZE_MARKER!r}\n"
        "src_path, out_path, pipeline = sys.argv[1], sys.argv[2], sys.argv[3]\n"
        "passes = pipeline.split(',')\n"
        "if MARKER in passes:\n"
        "    i = passes.index(MARKER)\n"
        "    stage1 = ','.join(passes[:i])\n"
        "    stage2 = ','.join(passes[i + 1:])\n"
        "else:\n"
        "    stage1, stage2 = pipeline, ''\n"
        "ctx = ir.Context()\n"
        "with open(src_path) as f:\n"
        "    module = ir.Module.parse(f.read(), ctx)\n"
        # Every runner variant runs the SAME pre-pipeline rewrites. A variant that quietly skips one
        # is how erase_self_copy came to read as an inert lever for seven beam rounds.
        "if _FOLD_WEIGHT_TRANSPOSE:\n"
        "    print('OK fold_weight_transpose folded', _fold_weight_transposes(module, ctx)[0])\n"
        "if _CONCAT_DPS:\n"
        "    print('OK concat_dps rewrote', _concat_dps(module, ctx)[0])\n"
        "if stage1:\n"
        "    PassManager.parse('builtin.module(' + stage1 + ')', ctx).run(module.operation)\n"
        "with ctx, ir.Location.unknown():\n"
        # Sink FIRST: a widening interposed between the A read and its lane extract (the int8
        # `extsi vector<MRx1xi8>` shape, and the f16/bf16 `extf` shape) hides the scalar extract
        # that scalarize_a_reads matches on, so sinking is a PRECONDITION of scalarization, not a
        # cleanup after it. Sink again afterwards to catch widenings the scalarization exposes.
        "    _m = sink_extf_through_extract(module, ctx)\n"
        "    _n = scalarize_a_reads(module, ctx)\n"
        "    _m += sink_extf_through_extract(module, ctx)\n"
        "if stage2:\n"
        "    _run_stages(ctx, module, stage2, _ERASE_SELF_COPY, _MID_STAGES, _LATE_STAGES)\n"
        "with open(out_path, 'w') as f:\n"
        "    __MERLIN_EMIT__\n"
        "print('OK scalarize_a rewrote', _n, 'sink_extf', _m)\n"
    )
