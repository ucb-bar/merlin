"""Fold a `linalg.transpose` of a matmul's B operand INTO the matmul's access pattern.

The whole-model profiler measured, on the K1, that ``linalg.transpose`` is the single largest
bucket in openvla -- **393 ms = 57 %** of the model, more than every matmul combined -- and that it
is emitted SCALAR (``convert-linalg-to-loops``; not in the vectorized ``contraction`` family). Every
one of openvla's 26 matmuls is a "transposed-B" GEMM: the frontend tags it ``prov.transposed_b =
"true"`` and emits a *standalone* weight transpose feeding the matmul's B operand::

    %Bt = linalg.transpose ins(%W : tensor<NxKxf32>) outs(... : tensor<KxNxf32>) permutation = [1, 0]
    %C  = linalg.matmul  indexing_maps = [#A=(m,k), #B=(k,n), #C=(m,n)]
                         ins(%A, %Bt : tensor<MxKxf32>, tensor<KxNxf32>) outs(%C0) -> ...

That materializes a full transposed copy of the weight in DRAM every forward and then reads it back
-- pure overhead. BLAS/XNNPACK never do this: they read B transposed via the GEMM's own access
pattern (a "transpose-b" kernel). We are a COMPILER, so we can too.

WHAT THIS DOES. Since LLVM-23 removed the ``linalg.matmul_transpose_b`` named op, transpose-b is
expressed as a plain ``linalg.matmul`` whose B ``indexing_map`` reads the *un-transposed* weight
``(n, k)`` instead of ``(k, n)``. So the fusion is a pure operand+map rewrite, in place:

  1. repoint the matmul's B operand from the transpose RESULT to the transpose's SOURCE (%W);
  2. permute the B ``indexing_map`` results by the transpose permutation
     (``new[j] = old[perm[j]]`` -- for the 2-D ``[1, 0]`` case, a swap ``(k,n) -> (n,k)``);
  3. erase the transpose if it is now dead (single-use is the common case).

The op stays ``linalg.matmul`` with valid contraction ``indexing_maps``, so the FROZEN RVV transform
schedule (which matches ``ops{["linalg.matmul"]}``) still tiles + vectorizes it -- verified with
mlir-opt: the transpose-b matmul lowers through ``tile -> vectorize -> lower_contraction`` to the
same ``vector.fma`` chain. Net effect: the scalar weight transposes DISAPPEAR (no op, no buffer),
and B is read ``(n, k)`` -- contiguous along k in the row-major ``[N, K]`` weight, i.e. a
cache-friendly access, exactly what a transpose-b BLAS kernel does.

CORRECTNESS. The rewrite is value-identical by construction: ``B[k, n]`` on the transposed weight
equals ``W[n, k]`` on the source, and the map change encodes precisely that. It is nonetheless a
default-off compiler feature (``fuse_transpose_b``) so the frozen ``hand_v0`` control keeps a
byte-identical lowering; it is gated on the board with a per-element check, not just cos.
"""
from __future__ import annotations

FEATURE = "fuse_transpose_b"

#: Spliced into the lowering runner (executes in the m2m venv, torch-mlir LLVM-23 bindings). Defines
#: the fold and reads its gate from ``sys.argv[5]`` so the un-instrumented / baseline lowering stays
#: byte-identical (the frozen hand_v0 control passes gate "0" and this never runs).
RUNNER_PRELUDE = r'''
def _fuse_transpose_b(module, ctx):
    """Fold `matmul(A, transpose(B, perm))` into a transpose-b `linalg.matmul` (operand + B
    indexing_map rewrite), erasing the now-dead transpose. Returns the number of matmuls fused."""
    from torch_mlir.ir import AffineMap, AffineMapAttr, ArrayAttr

    def _walk(op, fn):
        for region in op.regions:
            for block in region.blocks:
                for inner in list(block.operations):
                    _walk(inner, fn)
                    fn(inner)

    def _perm(top):
        # linalg.transpose carries `permutation = array<i64: ...>`; parse it structurally.
        try:
            a = top.attributes["permutation"]
        except Exception:
            return None
        s = str(a)
        i = s.find(":")
        if i < 0:
            return None
        try:
            return [int(t.strip()) for t in s[i + 1:s.rfind(">")].split(",")]
        except ValueError:
            return None

    fused = 0
    dead = []
    def _fuse(o):
        nonlocal fused
        op = o.operation
        if op.name != "linalg.matmul" or len(op.operands) < 2:
            return
        b = op.operands[1]
        prod = b.owner
        if not hasattr(prod, "name") or prod.name != "linalg.transpose":
            return
        perm = _perm(prod)
        if perm is None:
            return
        try:
            maps = op.attributes["indexing_maps"]
        except KeyError:
            return                                  # need explicit maps to rewrite B's access
        m_b = maps[1].value                         # AffineMapAttr -> AffineMap
        results = list(m_b.results)
        if len(perm) != len(results):
            return                                  # perm must match B's map arity
        with ctx:
            new_b = AffineMap.get(m_b.n_dims, m_b.n_symbols,
                                  [results[perm[j]] for j in range(len(perm))])
            new_maps = ArrayAttr.get([maps[0], AffineMapAttr.get(new_b), maps[2]])
        op.attributes["indexing_maps"] = new_maps
        op.operands[1] = prod.operands[0]           # read the un-transposed source weight
        dead.append(prod)
        fused += 1

    _walk(module.operation, _fuse)
    for t in dead:
        if len(list(t.results[0].uses)) == 0:
            t.operation.erase()
    return fused


_FUSE_TRANSPOSE_B = len(sys.argv) > 5 and sys.argv[5] == '1'
'''
