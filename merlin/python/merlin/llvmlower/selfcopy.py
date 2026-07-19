"""Erase `memref.copy %x, %x` before it is lowered out of reach.

Bufferizing a tiled reduction emits, per output tile, a subview of the destination, a
``vector.transfer_write`` of the tile result into that subview, and then a SECOND, structurally
identical subview plus a ``memref.copy`` between the two::

    %subview   = memref.subview %arg6[%i, %j] [4, 8] [1, 1]
    vector.transfer_write %acc, %subview[...]          // result is already in place here
    %subview_0 = memref.subview %arg6[%i, %j] [4, 8] [1, 1]
    memref.copy %subview, %subview_0                   // ...and this copies it onto itself

``cse`` collapses the two subviews to one SSA value, leaving a literal ``memref.copy %x, %x`` -- but
nothing upstream folds a self-copy, so it survives ``finalize-memref-to-llvm`` as an opaque
``@memrefCopy`` runtime call. That call is MLIR's rank-generic strided copy, which walks elements
with dynamic rank/stride handling.

MEASURED COST (K1, f32 GEMM 128^3, kernel region): ~79 retired instructions per OUTPUT ELEMENT, i.e.
1.29M of the 1.71M instructions the kernel retired -- 77% of the total, and the entire gap to
XNNPACK. Erasing it: 1,710,650 -> 475,899 instructions and 41,195 -> 21,882 ticks, a 1.88x real
speedup that moves us from 3.57x to 1.90x of XNNPACK, bit-exact.

Erasing is unconditionally safe: identical SSA value means identical base, offsets and region, so
there is nothing to move and no aliasing subtlety. It is nonetheless gated as a default-off compiler
feature (``erase_self_copy``) so the frozen ``hand_v0`` control keeps a byte-identical lowering; the
beam enables it as the PASS that closes an ``envelope.runtime_calls`` divergence.
"""
from __future__ import annotations

FEATURE = "erase_self_copy"

#: Spliced into the lowering runners (they execute in the m2m venv). Defines the erase and a helper
#: that runs a pass pipeline in two halves so the erase lands AFTER bufferization + cse (when the
#: self-copy becomes literal) and BEFORE finalize-memref-to-llvm (after which it is an opaque call).
RUNNER_PRELUDE = r'''
def _erase_self_copies(module):
    """Erase `memref.copy %x, %x` -- copying a buffer onto itself is a no-op. Returns the count."""
    n = 0

    def walk(op):
        nonlocal n
        for region in op.regions:
            for block in region.blocks:
                for inner in list(block.operations):
                    walk(inner)
                    if inner.operation.name == "memref.copy" and inner.operands[0] == inner.operands[1]:
                        inner.operation.erase()
                        n += 1

    walk(module.operation)
    return n


def _run_stages(ctx, module, pipeline, erase):
    """Run `pipeline`; when `erase`, split it after the post-bufferization canonicalize/cse and drop
    self-copies in between. Splitting on buffer-loop-hoisting (not a fixed index) so the hook stays
    put if the pass list moves."""
    from torch_mlir.passmanager import PassManager
    passes = [p for p in pipeline.split(',') if p]
    if not passes:
        return
    k = next((i for i, p in enumerate(passes) if 'buffer-loop-hoisting' in p), -1) if erase else -1
    if k < 0:
        PassManager.parse('builtin.module(' + ','.join(passes) + ')', ctx).run(module.operation)
        return
    head, tail = passes[:k + 3], passes[k + 3:]          # ...hoisting, canonicalize, cse
    PassManager.parse('builtin.module(' + ','.join(head) + ')', ctx).run(module.operation)
    print('OK erase_self_copy', _erase_self_copies(module))
    if tail:
        PassManager.parse('builtin.module(' + ','.join(tail) + ')', ctx).run(module.operation)


_ERASE_SELF_COPY = len(sys.argv) > 4 and sys.argv[4] == '1'
'''


def needs_canonicalize(pipeline: str) -> bool:
    """True when the pass list lacks a canonicalize/cse after bufferization, which the erase needs:
    the two subviews only collapse into one SSA value (making the copy a literal self-copy) once cse
    has run."""
    passes = [p for p in pipeline.split(",") if p]
    try:
        k = next(i for i, p in enumerate(passes) if "buffer-loop-hoisting" in p)
    except StopIteration:
        return False
    return not (len(passes) > k + 2 and "canonicalize" in passes[k + 1] and "cse" in passes[k + 2])


def with_canonicalize(pipeline: str) -> str:
    """Insert the canonicalize+cse the erase depends on, right after buffer-loop-hoisting."""
    if not needs_canonicalize(pipeline):
        return pipeline
    passes = [p for p in pipeline.split(",") if p]
    k = next(i for i, p in enumerate(passes) if "buffer-loop-hoisting" in p)
    return ",".join(passes[:k + 1] + ["canonicalize", "cse"] + passes[k + 1:])
