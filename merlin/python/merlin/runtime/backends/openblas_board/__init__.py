"""BOARD (RVV) OpenBLAS kernel backend: route the f32 ``linalg.matmul`` dispatches of a
whole-model lowering to OpenBLAS's RVV 8x8 GEMM microkernel — the OpenBLAS analogue of the
:mod:`merlin.runtime.backends.xnnpack_board` backend (default-off, additive). This completes
the four-way whole-model comparison (baseline / ours / XNNPACK / OpenBLAS).

The board runs one compiled C binary (``mining.k1.build_k1_binary``): the whole model lowers
monolithically to ``model.ll`` and is driven by ``_mlir_ciface_forward``. There is no
per-dispatch call boundary, so we create one at the MLIR level exactly like the XNNPACK backend:
:func:`rewrite_matmuls_to_openblas` replaces every routable plain 2-D f32 ``linalg.matmul`` in
the prepared model text with a ``func.call`` to an external symbol ``@merlin_openblas_gemm_f32``
(annotated read/read/write so one-shot-bufferize does NOT defensively copy the weight). That
symbol is implemented by :file:`openblas_gemm_rvv_shim.c`, which drives the SAME RVV 8x8 kernel
(``sgemm_kernel_8x8_zvl128b.c``) that the ceiling driver
``kernels/ceiling_drivers/openblas_sgemm_driver.c`` measures standalone; it is compiled to a
``.o`` and linked into the binary.

Everything else (attention generics, rmsnorm, elementwise) lowers UNCHANGED through the existing
RVV pipeline. The routable set is identical to the XNNPACK backend (a single 2-D f32
``linalg.matmul``), so it routes the same dispatches.

Default-off: nothing here runs unless ``build_k1_binary(..., kernel_backend="openblas")``.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path
from merlin.common.paths import work_dir

_HERE = Path(__file__).resolve().parent
_SHIM_SRC = _HERE / "openblas_gemm_rvv_shim.c"
# The ceiling_drivers common.h shim supplies BLASLONG/FLOAT for the OpenBLAS kernel body.
_CEILING = _HERE.parents[2] / "kernels" / "ceiling_drivers"
_SHIM_INC = _CEILING


def _openblas_repo() -> Path:
    import os

    env = os.environ.get("MERLIN_OPENBLAS_REPO")
    if env:
        return Path(env)
    from merlin.common.paths import repo_root

    return Path(work_dir()) / "tmp" / "kernels" / "OpenBLAS"


def _kernel_src() -> Path:
    """The RVV 8x8 sgemm kernel the isolated openblas ceiling driver #includes."""
    return _openblas_repo() / "kernel" / "riscv64" / "sgemm_kernel_8x8_zvl128b.c"


class OpenblasBoardUnavailable(RuntimeError):
    pass


# A routable matmul line (single textual op) — identical matcher to xnnpack_board.
_MM_RE = re.compile(
    r"(?P<res>%[\w$.]+)\s*=\s*linalg\.matmul\b(?P<attrs>\s*\{[^}]*\})?\s*"
    r"ins\(\s*(?P<a>%[\w$.]+)\s*,\s*(?P<b>%[\w$.]+)\s*:\s*"
    r"tensor<(?P<at>[^>]+)>\s*,\s*tensor<(?P<bt>[^>]+)>\s*\)\s*"
    r"outs\(\s*(?P<c>%[\w$.]+)\s*:\s*tensor<(?P<ct>[^>]+)>\s*\)\s*"
    r"->\s*tensor<(?P<rt>[^>]+)>")


def _is_routable(at: str, bt: str, ct: str, rt: str) -> bool:
    """Plain 2-D f32 matmul (identical to xnnpack_board._is_routable's faithful set)."""
    for t in (at, bt, ct, rt):
        dims = t.split("x")
        if dims[-1] != "f32":
            return False
        if len(dims) != 3:          # 2 shape dims + the element type -> rank 2
            return False
        if any(d.startswith("?") for d in dims[:-1]):   # static only
            return False
    return True


def rewrite_matmuls_to_openblas(mlir_text: str) -> tuple[str, int]:
    """Replace every routable 2-D f32 ``linalg.matmul`` with ``call @merlin_openblas_gemm_f32``.

    Returns ``(rewritten_text, n_routed)``. MLIR func types are monomorphic, so we emit one
    numbered decl ``@merlin_openblas_gemm_f32_<i>`` per distinct (A,B,C,R) type signature. Each
    maps, at link time, to a thin C alias of the single signature-agnostic shim entry
    ``merlin_openblas_gemm_f32`` (which reads M/N/K from the memref descriptors) — see
    :func:`build_openblas_object`. Default-off: with no routable matmul, returns the input
    unchanged.
    """
    sigs: dict[tuple, str] = {}     # (at,bt,ct,rt) -> decl sym name
    n = 0

    def repl(m: re.Match) -> str:
        nonlocal n
        at, bt, ct, rt = m["at"], m["bt"], m["ct"], m["rt"]
        if not _is_routable(at, bt, ct, rt):
            return m.group(0)
        key = (at, bt, ct, rt)
        sym = sigs.get(key)
        if sym is None:
            sym = f"merlin_openblas_gemm_f32_{len(sigs)}"
            sigs[key] = sym
        n += 1
        return (f"{m['res']} = call @{sym}({m['a']}, {m['b']}, {m['c']}) : "
                f"(tensor<{at}>, tensor<{bt}>, tensor<{ct}>) -> tensor<{rt}>")

    body = _MM_RE.sub(repl, mlir_text)
    if n == 0:
        return mlir_text, 0

    decls = []
    for (at, bt, ct, rt), sym in sigs.items():
        decls.append(
            f'func.func private @{sym}('
            f'%a: tensor<{at}> {{bufferization.access = "read"}}, '
            f'%b: tensor<{bt}> {{bufferization.access = "read"}}, '
            f'%c: tensor<{ct}> {{bufferization.access = "write"}}) -> tensor<{rt}>')
    decl_block = "  " + "\n  ".join(decls) + "\n"
    # Insert the decls just before the first top-level func.func (inside the module body — NOT
    # after `module ... {`, which for `builtin.module attributes {...} {` would land in the attr
    # dictionary).
    mi = re.search(r"\n(\s*)func\.func @", body)
    if mi is None:                              # no func to anchor on (unexpected)
        return body, n
    pos = mi.start() + 1                        # after the newline, before the func
    return body[:pos] + decl_block + body[pos:], n


def build_openblas_object(cc: Path, cflags: list[str], n_sigs: int, work: Path) -> Path:
    """Compile the RVV OpenBLAS GEMM shim + the per-signature alias wrappers into one ``.o``.

    ``cc`` is the SpacemiT clang, ``cflags`` the K1 RVV flags (rv64gcv / lp64d). Produces an
    object exporting ``merlin_openblas_gemm_f32`` and ``merlin_openblas_gemm_f32_0..n_sigs-1``,
    all the same code (each reads its M/N/K from the descriptors), so every monomorphic MLIR
    decl links.
    """
    ksrc = _kernel_src()
    if not ksrc.is_file():
        raise OpenblasBoardUnavailable(
            f"OpenBLAS RVV sgemm kernel not found at {ksrc} (set MERLIN_OPENBLAS_REPO)")
    work.mkdir(parents=True, exist_ok=True)

    # Per-signature alias wrappers: each numbered MLIR decl forwards to the single shim entry.
    # They share the descriptor-unpacked ABI, so a thin pass-through is exact. Compiled in the
    # SAME translation unit as the shim (appended) so the SpacemiT clang lays out the struct-return
    # ABI identically to the model.ll caller.
    aliases = []
    for i in range(n_sigs):
        aliases.append(
            f"merlin_memref_2d_f32 merlin_openblas_gemm_f32_{i}("
            "float*a0,float*a1,intptr_t a2,intptr_t a3,intptr_t a4,intptr_t a5,intptr_t a6,"
            "float*b0,float*b1,intptr_t b2,intptr_t b3,intptr_t b4,intptr_t b5,intptr_t b6,"
            "float*c0,float*c1,intptr_t c2,intptr_t c3,intptr_t c4,intptr_t c5,intptr_t c6)"
            "{return merlin_openblas_gemm_f32(a0,a1,a2,a3,a4,a5,a6,b0,b1,b2,b3,b4,b5,b6,"
            "c0,c1,c2,c3,c4,c5,c6);}")

    obj = work / "openblas_gemm_rvv.o"
    # -I the OpenBLAS kernel dir (for the `#include "sgemm_kernel_8x8_zvl128b.c"`) and the
    # ceiling_drivers dir (for the `#include "common.h"` the kernel body pulls in).
    inc = ["-I", str(_SHIM_INC), "-I", str(ksrc.parent)]
    combined = work / "openblas_gemm_rvv_combined.c"
    combined.write_text(_SHIM_SRC.read_text() + "\n" + "\n".join(aliases) + "\n")
    cmd = [str(cc), *cflags, *inc, "-c", str(combined), "-o", str(obj)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0 or not obj.is_file():
        raise OpenblasBoardUnavailable(
            f"OpenBLAS RVV shim compile failed:\ncmd: {' '.join(cmd)}\n{p.stderr[-1500:]}")
    return obj


def is_available() -> bool:
    try:
        return _kernel_src().is_file() and _SHIM_SRC.is_file()
    except Exception:  # noqa: BLE001
        return False
