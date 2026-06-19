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


def scalarize_a_reads(module, ctx):
    """Replace each `vector.transfer_read -> vector<MRx1xf32>` whose only uses are
    `vector.extract [i, 0] : f32` with per-row scalar `tensor.extract`/`memref.load`, so the A
    operand of the register-blocked vfmacc reaches the backend as a scalar (flw -> vfmacc.vf) instead
    of a reconstructed vector lane (vmv/vslideup -> vfmacc.vv). Returns the count rewritten.

    Numerically identical: element [i,0] of the slice == lane [i,0] of the vector read of the slice.
    """
    from torch_mlir import ir
    targets = []

    def visit(o):
        if o.operation.name != "vector.transfer_read":
            return
        res = o.results[0]
        ts = str(res.type)
        if not ts.startswith("vector<") or "xf32>" not in ts:
            return
        shape = ts[len("vector<"):ts.index("xf32>")]
        dims = shape.split("x")
        # only the register-tile lhs read: rank>=2 with a TRAILING unit dim (vector<MR x 1>).
        if len(dims) < 2 or dims[-1] != "1":
            return
        exs = []
        for u in res.uses:
            owner = u.owner
            if owner.name != "vector.extract" or str(owner.results[0].type) != "f32":
                return  # a non-scalar use -> leave this read alone (keep it correct/general)
            exs.append(owner)
        if exs:
            targets.append((o, exs))

    _merlin_walk(module.operation, visit)
    idxty = ir.IndexType.get(ctx)
    f32 = ir.F32Type.get(ctx)
    n = 0
    for read, exs in targets:
        src_val = read.operands[0]
        base_idx = list(read.operands[1:])
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
                opname, results=[f32], operands=[src_val, *new_idx], ip=ip).results[0]
            ex.operation.results[0].replace_all_uses_with(scalar)
            ex.operation.erase()
        read.operation.erase()
        n += 1
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
        + _REWRITER_SRC +
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
        "if stage1:\n"
        "    PassManager.parse('builtin.module(' + stage1 + ')', ctx).run(module.operation)\n"
        "with ctx, ir.Location.unknown():\n"
        "    _n = scalarize_a_reads(module, ctx)\n"
        "if stage2:\n"
        "    PassManager.parse('builtin.module(' + stage2 + ')', ctx).run(module.operation)\n"
        "with open(out_path, 'w') as f:\n"
        "    f.write(str(llvm.translate_module_to_llvmir(module.operation)))\n"
        "print('OK scalarize_a rewrote', _n)\n"
    )
