"""BOARD (RVV) OURS GEMM kernel backend: route the f32 ``linalg.matmul`` dispatches of a
whole-model lowering to OUR OWN compiler-emitted MR=4 accumulator-resident RVV micro-kernel
(the "v3" kernel) — the ``ours`` analogue of the :mod:`~merlin.runtime.backends.xnnpack_board`
and :mod:`~merlin.runtime.backends.openblas_board` backends (default-off, additive).

Its sole purpose is **attribution measurement**: the XNNPACK/OpenBLAS shims bracket their matmul
loop with ``rdtime`` under ``-DMERLIN_DISPATCH_TIMING`` so the four-way's matmul-vs-dispatch split
is *measured*. The ``ours`` arm previously had no such bracket, so its matmul bucket was *attributed*
(assumed == XNNPACK on small-M). This backend reroutes the same routable matmuls to a shim that
drives our v3 micro-kernel through the **identical** rdtime bracket and **identical** timer symbols
(``merlin_matmul_ticks``), so ``k1_dispatch_breakdown.py`` reads the ours matmul bucket directly —
closing the attribution caveat in ``RUNTIME_INVESTIGATION.md`` §1.

Unlike the expert backends there is **no external kernel repo**: the micro-kernel
(``microkernel_panel``/``gemm_micro`` from ``kernels/ceiling_drivers/ours_intrinsic_gemm_driver.c``)
is inlined in :file:`ours_gemm_rvv_shim.c`, so the build only needs the SpacemiT clang + the RVV
headers. Routable set, rewrite mechanism, and per-signature alias trick are identical to the OpenBLAS
backend, so it routes the same dispatches and links the same way.

Default-off: nothing here runs unless ``build_k1_binary(..., kernel_backend="ours")``.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from .. import _matmul_routing as _mr

_HERE = Path(__file__).resolve().parent
_SHIM_SRC = _HERE / "ours_gemm_rvv_shim.c"


class OursBoardUnavailable(RuntimeError):
    pass


# The routable-matmul matcher and the MLIR matmul->external-call rewrite are SHARED with the other
# board matmul-routing backends (see :mod:`merlin.runtime.backends._matmul_routing`): all three
# route the same set and differ only in the shim symbol they call. Keeping one structural reader
# is the point — three copies of a parser is three chances for the routable sets to drift apart.
_is_routable = _mr.is_routable


def rewrite_matmuls_to_ours(mlir_text: str) -> tuple[str, int]:
    """Replace every routable 2-D f32 ``linalg.matmul`` with ``call @merlin_ours_gemm_f32``.

    Returns ``(rewritten_text, n_routed)``. One numbered decl ``@merlin_ours_gemm_f32_<i>`` per
    distinct (A,B,C,R) type signature, each aliasing the single signature-agnostic shim entry
    ``merlin_ours_gemm_f32`` (reads M/N/K from the memref descriptors) — see
    :func:`build_ours_object`. Default-off: with no routable matmul, returns the input unchanged.
    Raises :class:`~merlin.runtime.backends._matmul_routing.MatmulRoutingError` on a
    ``linalg.matmul`` the structural reader cannot parse, rather than silently routing nothing.
    """
    return _mr.rewrite_matmuls(mlir_text, "merlin_ours_gemm_f32")


def matmul_routing_coverage(mlir_text: str) -> tuple[int, int]:
    """``(n_candidates, n_eligible)`` for the exact rewrite domain — the denominator behind a
    routed-matmul count, so partial coverage cannot read as complete."""
    return _mr.routing_coverage(mlir_text)


def build_ours_object(cc: Path, cflags: list[str], n_sigs: int, work: Path) -> Path:
    """Compile the inline v3 RVV GEMM shim + the per-signature alias wrappers into one ``.o``.

    ``cc`` is the SpacemiT clang, ``cflags`` the K1 RVV flags (rv64gcv / lp64d). Produces an object
    exporting ``merlin_ours_gemm_f32`` and ``merlin_ours_gemm_f32_0..n_sigs-1`` (all the same code,
    each reads its M/N/K from the descriptors), so every monomorphic MLIR decl links.
    """
    if not _SHIM_SRC.is_file():
        raise OursBoardUnavailable(f"ours RVV GEMM shim not found at {_SHIM_SRC}")
    work.mkdir(parents=True, exist_ok=True)

    aliases = []
    for i in range(n_sigs):
        aliases.append(
            f"merlin_memref_2d_f32 merlin_ours_gemm_f32_{i}("
            "float*a0,float*a1,intptr_t a2,intptr_t a3,intptr_t a4,intptr_t a5,intptr_t a6,"
            "float*b0,float*b1,intptr_t b2,intptr_t b3,intptr_t b4,intptr_t b5,intptr_t b6,"
            "float*c0,float*c1,intptr_t c2,intptr_t c3,intptr_t c4,intptr_t c5,intptr_t c6)"
            "{return merlin_ours_gemm_f32(a0,a1,a2,a3,a4,a5,a6,b0,b1,b2,b3,b4,b5,b6,"
            "c0,c1,c2,c3,c4,c5,c6);}")

    obj = work / "ours_gemm_rvv.o"
    combined = work / "ours_gemm_rvv_combined.c"
    combined.write_text(_SHIM_SRC.read_text() + "\n" + "\n".join(aliases) + "\n")
    cmd = [str(cc), *cflags, "-c", str(combined), "-o", str(obj)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0 or not obj.is_file():
        raise OursBoardUnavailable(
            f"ours RVV shim compile failed:\ncmd: {' '.join(cmd)}\n{p.stderr[-1500:]}")
    return obj


def is_available() -> bool:
    try:
        return _SHIM_SRC.is_file()
    except Exception:  # noqa: BLE001
        return False
