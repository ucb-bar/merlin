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


def _run_stages(ctx, module, pipeline, erase, mid=(), late=()):
    """Run `pipeline`; when `erase` (or any `mid` rewrite is requested), split it after the
    post-bufferization canonicalize/cse and run the rewrites in between. Splitting on
    buffer-loop-hoisting (not a fixed index) so the hook stays put if the pass list moves.

    `mid` is a sequence of `(label, fn(ctx, module) -> int)` rewrites that need the SAME window as
    the erase: after bufferization has created the buffer ops, before finalize-memref-to-llvm turns
    them into opaque runtime calls. Each reports its count as `OK <label> <n>` so a rewrite that
    matched nothing is visible in the build log instead of passing for applied.

    `late` is the same shape, in a DIFFERENT window: after the forall/linalg -> `scf.parallel`
    conversions and before `convert-scf-to-openmp` turns each `scf.parallel` into a fork. It is a
    separate list rather than more `mid` entries because at the `mid` point no `scf.parallel` exists
    yet -- a grain decision made there would price loops that have not been formed. Empty `late`
    (the default) leaves the pass string split exactly as before, so the lowering is byte-identical.
    """
    from torch_mlir.passmanager import PassManager

    def _run(sub):
        if sub:
            PassManager.parse('builtin.module(' + ','.join(sub) + ')', ctx).run(module.operation)

    def _late_split(sub):
        """Run `sub`, pausing before `convert-scf-to-openmp` to run the `late` rewrites."""
        if not late:
            _run(sub)
            return
        j = next((i for i, p in enumerate(sub) if 'convert-scf-to-openmp' in p), -1)
        if j < 0:
            # No OpenMP conversion in this pass list: run the rewrites at the END, where they still
            # see whatever `scf.parallel` survives, rather than dropping them silently.
            _run(sub)
            for label, fn in late:
                print('OK ' + label, fn(ctx, module))
            return
        _run(sub[:j])
        for label, fn in late:
            print('OK ' + label, fn(ctx, module))
        _run(sub[j:])

    passes = [p for p in pipeline.split(',') if p]
    if not passes:
        return
    want_split = bool(erase) or bool(mid)
    k = next((i for i, p in enumerate(passes) if 'buffer-loop-hoisting' in p), -1) if want_split else -1
    if k < 0:
        _late_split(passes)
        return
    # ...hoisting, canonicalize, cse -- but NEVER past the pass that lowers linalg to loops. A mid
    # rewrite may EMIT linalg (expand_memref_copy rewrites a copy to `linalg.copy` and relies on
    # convert-linalg-to-loops to turn it into an scf nest), and in the scalar pipeline that pass sits
    # at k+1, so a fixed k+3 window put the rewrite AFTER its own lowering: the linalg op survived to
    # LLVM conversion as an unrealized_conversion_cast and the whole build failed. Clamping keeps the
    # RVV window (where the pass is at k+7) exactly where it was.
    end = k + 3
    for i, p in enumerate(passes):
        if 'convert-linalg-to-loops' in p or 'convert-linalg-to-parallel-loops' in p:
            end = min(end, i)
            break
    head, tail = passes[:end], passes[end:]
    _run(head)
    if erase:
        print('OK erase_self_copy', _erase_self_copies(module))
    for label, fn in mid:
        print('OK ' + label, fn(ctx, module))
    _late_split(tail)


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
