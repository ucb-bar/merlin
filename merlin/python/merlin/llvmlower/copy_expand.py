"""Expand `memref.copy` into emitted code instead of a runtime call (default-off feature).

WHAT THE ESCAPE IS
------------------
``finalize-memref-to-llvm`` does not always emit code for a ``memref.copy``. It emits an
``llvm.intr.memcpy`` only when BOTH operands are statically shaped and contiguous row-major;
anything else -- a destination that is a ``memref.subview`` of a larger buffer, i.e. a strided
layout with an offset -- falls through to a call to ``@memrefCopy``, MLIR's rank-generic strided
copy. That helper walks the copy element by element through a dynamic rank/stride descriptor:
MEASURED at ~79 retired instructions per copied ELEMENT (llvmlower/selfcopy.py), versus the 3-5 a
load/store loop costs. It is the compiler declining to emit code and calling a library instead.

WHY THIS IS THE GENERAL FORM OF THE FIX, NOT A SECOND SPECIAL CASE
------------------------------------------------------------------
``erase_self_copy`` removes the copies that are *redundant* (``memref.copy %x, %x``). It cannot
touch a copy that actually moves data, and the whole-model int8 lowering has both kinds. MEASURED
on ``small_llama_int8_consistent`` at the post-bufferization split point:

    memref.copy  self       19   (all inside loops)   608 elements  -> erase_self_copy
    memref.copy  diff-type  24   (prologue)          6144 elements  -> @memrefCopy, ~485K instrs
    memref.copy  same-type  40   (prologue)         21360 elements  -> memcpy

So the runtime-call set stays ``{memrefCopy, memcpy, ...}`` no matter how many self-copies are
erased. This feature closes the axis by construction rather than by luck: every ``memref.copy``
with ranked, statically shaped operands becomes a ``linalg.copy``, which the
``convert-linalg-to-loops`` already in every pipeline turns into an explicit ``scf`` load/store
nest. After it, ``finalize-memref-to-llvm`` has no ``memref.copy`` left to lower, so neither
``@memrefCopy`` nor the copy-derived ``memcpy`` can appear.

It is keyed on STRUCTURE (ranked + static shape), never on an op name, a model, a shape or a
target. A copy it cannot prove static -- dynamic dims, unranked, rank 0 -- is LEFT ALONE and
counted; the count is printed by the runner so a partial expansion is visible rather than silent.

SAFETY
------
``linalg.copy ins(%src) outs(%dst)`` is an elementwise read-then-write over the same index space
``memref.copy`` defines, so it is value-preserving for disjoint buffers (what bufferization
produces) and for the degenerate ``%x -> %x`` self-copy alike. It is NOT equivalent for genuinely
*overlapping but distinct* memrefs -- neither is ``memref.copy``, whose lowering to ``memcpy``
carries the same no-overlap requirement -- so this changes no guarantee that was there before.

Default OFF so the frozen baseline lowering stays byte-identical; the search enables it as the PASS
that closes an ``envelope.runtime_calls`` divergence.
"""
from __future__ import annotations

FEATURE = "expand_memref_copy"

#: Spliced into the lowering runners (they execute in the m2m venv, which owns the MLIR Python
#: bindings). Defines ``_expand_memref_copies(ctx, module) -> int``; ``_run_stages`` calls it at the
#: same post-bufferization split point the self-copy erase uses -- after bufferization has created
#: the copies and before ``finalize-memref-to-llvm`` turns them into calls.
RUNNER_PRELUDE = r'''
def _mc_static_memref(t):
    """The `ir.MemRefType` when `t` is a ranked memref of static shape and rank >= 1, else None.

    Fail-closed: anything this cannot prove static (unranked, a dynamic dim, rank 0) returns None
    and the copy is left for the existing lowering rather than expanded on an assumption."""
    from torch_mlir import ir as _mcir
    try:
        mt = _mcir.MemRefType(t)
    except (ValueError, TypeError):
        return None
    if mt.rank < 1:
        return None
    for d in range(mt.rank):
        if mt.is_dynamic_dim(d):
            return None
    return mt


def _expand_memref_copies(ctx, module):
    """Rewrite every static `memref.copy` to a `linalg.copy` so it lowers to an emitted loop nest.

    Returns the number expanded; prints the number SKIPPED (never zero-filled) so a copy that could
    not be proven static is surfaced instead of silently remaining a runtime call."""
    from torch_mlir import ir as _mcir
    from torch_mlir.dialects import linalg as _mclinalg

    todo = []

    def walk(op):
        for region in op.regions:
            for block in region.blocks:
                for inner in list(block.operations):
                    walk(inner)
                    if inner.operation.name == "memref.copy":
                        todo.append(inner)

    walk(module.operation)
    n = skipped = 0
    with ctx, _mcir.Location.unknown():
        for cp in todo:
            src, dst = cp.operands[0], cp.operands[1]
            s_t, d_t = _mc_static_memref(src.type), _mc_static_memref(dst.type)
            if s_t is None or d_t is None or list(s_t.shape) != list(d_t.shape):
                skipped += 1
                continue
            with _mcir.InsertionPoint(cp):
                op = _mclinalg.CopyOp([], [src], [dst])
                body = op.regions[0].blocks.append(s_t.element_type, d_t.element_type)
                with _mcir.InsertionPoint(body):
                    _mclinalg.YieldOp([body.arguments[0]])
            cp.operation.erase()
            n += 1
    print("OK expand_memref_copy skipped", skipped)
    return n
'''


#: Runner glue: reads the argv gate and builds the ``mid`` list ``_run_stages`` consumes. Lives here
#: (not in pipeline.py) so every runner variant -- plain, act_poly, scalarize -- splices the SAME
#: text and none of them can quietly grow its own version that forgets a rewrite.
MID_STAGE_SRC = r"""
_EXPAND_MEMREF_COPY = len(sys.argv) > 6 and sys.argv[6] == '1'
_MID_STAGES = []
if _EXPAND_MEMREF_COPY:
    _MID_STAGES.append(("expand_memref_copy", _expand_memref_copies))
"""
