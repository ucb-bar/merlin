"""BOARD (RVV) XNNPACK kernel backend: route the f32 ``linalg.matmul`` dispatches of a
whole-model lowering to XNNPACK's RVV GEMM microkernel, the K1/RVV analogue of the host
``xnnpack_host`` backend (default-off, additive).

The board runs one compiled C binary (``mining.k1.build_k1_binary``): the whole model lowers
monolithically to ``model.ll`` and is driven by ``_mlir_ciface_forward``. There is no
per-dispatch call boundary like the host Python interpreter's, so we create one at the MLIR
level: :func:`rewrite_matmuls_to_xnn` replaces every routable plain 2-D f32 ``linalg.matmul``
in the prepared model text with a ``func.call`` to an external symbol ``@merlin_xnn_gemm_f32``
(annotated read/read/write so one-shot-bufferize does NOT defensively copy the weight). That
symbol is implemented by :file:`xnn_gemm_rvv_shim.c`, which drives the SAME RVV ukernel
(``xnn_f32_gemm_ukernel_1x4v__rvv``) that ``scripts/k1_cross_framework.py`` already
cross-compiles and runs on the board; it is compiled to a ``.o`` and linked into the binary.

Everything else (attention generics, rmsnorm, elementwise) lowers UNCHANGED through the
existing RVV pipeline — the same hybrid the host prototype uses. The routable set mirrors the
host classifier (``xnnpack_host.classify_matmul_kernel``): a single 2-D f32 ``linalg.matmul``.
On the whole-model graph the matmul is already canonical A·B (transposes are separate ops), so
every 2-D f32 ``linalg.matmul`` is routable, which reproduces the host's per-dispatch count.

Default-off: nothing here runs unless ``build_k1_binary(..., kernel_backend="xnnpack")``.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path
from merlin.common.paths import work_dir

_HERE = Path(__file__).resolve().parent
_SHIM_SRC = _HERE / "xnn_gemm_rvv_shim.c"
# Reuse the ceiling drivers' minimal XNNPACK header shim (it already provides
# xnn_f32_default_params + the RVV-kernel macro surface) instead of a second copy.
_CEILING = _HERE.parents[2] / "kernels" / "ceiling_drivers"
_SHIM_INC = _CEILING            # carries src/xnnpack/*.h


def _xnnpack_repo() -> Path:
    import os

    env = os.environ.get("MERLIN_XNNPACK_REPO")
    if env:
        return Path(env)
    from merlin.common.paths import repo_root

    return Path(work_dir()) / "tmp" / "kernels" / "XNNPACK"


class XnnpackBoardUnavailable(RuntimeError):
    pass


# A routable matmul line (single textual op). Captures the two ins operands + their tensor
# types, the outs init operand + type, and the result type. The prepared model emits each
# linalg.matmul on one line (see output/<model>/model.mlir).
_MM_RE = re.compile(
    r"(?P<res>%[\w$.]+)\s*=\s*linalg\.matmul\b(?P<attrs>\s*\{[^}]*\})?\s*"
    r"ins\(\s*(?P<a>%[\w$.]+)\s*,\s*(?P<b>%[\w$.]+)\s*:\s*"
    r"tensor<(?P<at>[^>]+)>\s*,\s*tensor<(?P<bt>[^>]+)>\s*\)\s*"
    r"outs\(\s*(?P<c>%[\w$.]+)\s*:\s*tensor<(?P<ct>[^>]+)>\s*\)\s*"
    r"->\s*tensor<(?P<rt>[^>]+)>")

def _is_routable(at: str, bt: str, ct: str, rt: str) -> bool:
    """Plain 2-D f32 matmul (mirrors xnnpack_host.classify_matmul_kernel's faithful set)."""
    for t in (at, bt, ct, rt):
        dims = t.split("x")
        if dims[-1] != "f32":
            return False
        if len(dims) != 3:          # 2 shape dims + the element type -> rank 2
            return False
        if any(d.startswith("?") for d in dims[:-1]):   # static only
            return False
    return True


def _rewrite_matmuls(mlir_text: str, sym_base: str) -> tuple[str, int]:
    """Replace every routable 2-D f32 ``linalg.matmul`` with ``call @<sym_base>_<i>``, one numbered
    decl per distinct (A,B,C,R) signature (MLIR func types are monomorphic). Each links to a thin C
    alias of the signature-agnostic shim entry ``<sym_base>``. Shared by the f32 (:func:`rewrite_matmuls_to_xnn`)
    and dynamic-int8 (:func:`rewrite_matmuls_to_qd8`) board arms — they route the SAME f32 matmuls and
    differ only in the shim the symbol links to (plain f32 GEMM vs a dynamically-quantized qd8 GEMM)."""
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
            sym = f"{sym_base}_{len(sigs)}"
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
    # Insert the decls just before the first top-level func.func (the forward), inside the module body.
    mi = re.search(r"\n(\s*)func\.func @", body)
    if mi is None:                              # no func to anchor on (unexpected)
        return body, n
    pos = mi.start() + 1                        # after the newline, before the func
    return body[:pos] + decl_block + body[pos:], n


def rewrite_matmuls_to_xnn(mlir_text: str) -> tuple[str, int]:
    """Route every routable 2-D f32 ``linalg.matmul`` to the plain-f32 RVV GEMM shim
    (``@merlin_xnn_gemm_f32``). Returns ``(rewritten_text, n_routed)``."""
    return _rewrite_matmuls(mlir_text, "merlin_xnn_gemm_f32")


def rewrite_matmuls_to_qd8(mlir_text: str) -> tuple[str, int]:
    """Route every routable 2-D f32 ``linalg.matmul`` to the DYNAMICALLY-QUANTIZED int8 shim
    (``@merlin_xnn_qd8_gemm``), which per-row dynamic-quantizes the activation to int8 + params and
    drives ``xnn_qd8_f32_qc8w_gemm_minmax_ukernel_*__rvv`` with the offline per-channel int8 weight,
    producing f32. Same routable set as the f32 arm; the quantization is inside the shim.

    NOTE: qd8 is LOSSY vs the f32 golden — the correctness gate for this arm must be quantization-aware
    (a wider tolerance / an int8 reference), calibrated from a real K1 run; do NOT gate it at the f32
    ``cos >= 0.9999`` threshold (see the board-validation gate in k1_e2e_xnnpack)."""
    return _rewrite_matmuls(mlir_text, "merlin_xnn_qd8_gemm")


def build_xnn_object(cc: Path, cflags: list[str], n_sigs: int, work: Path) -> Path:
    """Compile the RVV GEMM shim + the per-signature alias wrappers into one ``.o``.

    ``cc`` is the SpacemiT clang, ``cflags`` the K1 RVV flags (rv64gcv / lp64d). Produces an
    object exporting ``merlin_xnn_gemm_f32`` and ``merlin_xnn_gemm_f32_0..n_sigs-1``, all the
    same code (each reads its M/N/K from the descriptors), so every monomorphic MLIR decl links.
    """
    xnn_src = _xnnpack_repo() / "src"
    if not (xnn_src / "f32-gemm" / "gen" / "f32-gemm-1x4v-rvv.c").is_file():
        raise XnnpackBoardUnavailable(
            f"XNNPACK RVV ukernel not found under {xnn_src} (set MERLIN_XNNPACK_REPO)")
    work.mkdir(parents=True, exist_ok=True)

    # Per-signature alias wrappers: each numbered MLIR decl ``@merlin_xnn_gemm_f32_<i>`` forwards
    # to the single shim entry ``merlin_xnn_gemm_f32``. They share the descriptor-unpacked ABI, so
    # a thin pass-through is exact. Compiled in the SAME translation unit as the shim (appended)
    # so the SpacemiT clang lays out the struct-return ABI identically to the model.ll caller.
    aliases = []
    for i in range(n_sigs):
        aliases.append(
            f"merlin_memref_2d_f32 merlin_xnn_gemm_f32_{i}("
            "float*a0,float*a1,intptr_t a2,intptr_t a3,intptr_t a4,intptr_t a5,intptr_t a6,"
            "float*b0,float*b1,intptr_t b2,intptr_t b3,intptr_t b4,intptr_t b5,intptr_t b6,"
            "float*c0,float*c1,intptr_t c2,intptr_t c3,intptr_t c4,intptr_t c5,intptr_t c6)"
            "{return merlin_xnn_gemm_f32(a0,a1,a2,a3,a4,a5,a6,b0,b1,b2,b3,b4,b5,b6,"
            "c0,c1,c2,c3,c4,c5,c6);}")

    obj = work / "xnn_gemm_rvv.o"
    inc = ["-I", str(_SHIM_INC), "-I", str(xnn_src)]
    # Combine shim + aliases into one TU (the aliases reference merlin_memref_2d_f32 + the shim
    # entry, both defined above them in the shim source).
    combined = work / "xnn_gemm_rvv_combined.c"
    combined.write_text(_SHIM_SRC.read_text() + "\n" + "\n".join(aliases) + "\n")
    cmd = [str(cc), *cflags, *inc, "-c", str(combined), "-o", str(obj)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0 or not obj.is_file():
        raise XnnpackBoardUnavailable(
            f"XNNPACK RVV shim compile failed:\ncmd: {' '.join(cmd)}\n{p.stderr[-1500:]}")
    return obj


def is_available() -> bool:
    try:
        xnn_src = _xnnpack_repo() / "src"
        return (xnn_src / "f32-gemm" / "gen" / "f32-gemm-1x4v-rvv.c").is_file() and _SHIM_SRC.is_file()
    except Exception:  # noqa: BLE001
        return False


# The qd8 dynamic-int8 shim (per-row dynamic activation quant + qc8w weight pack + the qd8 RVV ukernel).
# Its NUMERICS are board-validation-gated, so the file is created only once ported from the ceiling-
# driver recipe AND its quantization-aware cos gate is validated on a real K1 — not shipped unverified.
_QD8_SHIM_SRC = _HERE / "xnn_qd8_gemm_rvv_shim.c"


def qd8_is_available() -> bool:
    try:
        xnn_src = _xnnpack_repo() / "src"
        ukernels = list((xnn_src / "qd8-f32-qc8w-gemm" / "gen").glob("*rvv*.c"))
        return bool(ukernels) and _QD8_SHIM_SRC.is_file()
    except Exception:  # noqa: BLE001
        return False


def build_qd8_object(cc: Path, cflags: list[str], n_sigs: int, work: Path) -> Path:
    """Compile the qd8 dynamic-int8 GEMM shim + per-signature alias wrappers into one ``.o``.

    FAIL-CLOSED: qd8 is LOSSY vs the f32 golden, so the shim (dynamic activation quant + qc8w pack + the
    qd8 ukernel) and its quantization-aware correctness gate must be validated on a REAL K1 run before
    use. Until the shim (:data:`_QD8_SHIM_SRC`) is implemented + validated, this raises rather than
    shipping unverified quant numerics. The recipe to port lives in
    ``kernels/ceiling_drivers/xnnpack_qd8_gemm_driver.c``."""
    if not _QD8_SHIM_SRC.is_file():
        raise XnnpackBoardUnavailable(
            "qd8 board arm not yet enabled: the dynamic-int8 shim (xnn_qd8_gemm_rvv_shim.c) is not "
            "implemented + K1-validated. Port the recipe from "
            "kernels/ceiling_drivers/xnnpack_qd8_gemm_driver.c and validate its quantization-aware "
            "cos gate on a real K1 run before enabling this arm (the K1 board is required).")
    xnn_src = _xnnpack_repo() / "src"
    if not list((xnn_src / "qd8-f32-qc8w-gemm" / "gen").glob("*rvv*.c")):
        raise XnnpackBoardUnavailable(
            f"qd8 RVV ukernel not found under {xnn_src} (set MERLIN_XNNPACK_REPO)")
    work.mkdir(parents=True, exist_ok=True)
    aliases = []
    for i in range(n_sigs):
        aliases.append(
            f"merlin_memref_2d_f32 merlin_xnn_qd8_gemm_{i}("
            "float*a0,float*a1,intptr_t a2,intptr_t a3,intptr_t a4,intptr_t a5,intptr_t a6,"
            "float*b0,float*b1,intptr_t b2,intptr_t b3,intptr_t b4,intptr_t b5,intptr_t b6,"
            "float*c0,float*c1,intptr_t c2,intptr_t c3,intptr_t c4,intptr_t c5,intptr_t c6)"
            "{return merlin_xnn_qd8_gemm(a0,a1,a2,a3,a4,a5,a6,b0,b1,b2,b3,b4,b5,b6,"
            "c0,c1,c2,c3,c4,c5,c6);}")
    obj = work / "xnn_qd8_gemm_rvv.o"
    inc = ["-I", str(_SHIM_INC), "-I", str(xnn_src)]
    combined = work / "xnn_qd8_gemm_rvv_combined.c"
    combined.write_text(_QD8_SHIM_SRC.read_text() + "\n" + "\n".join(aliases) + "\n")
    cmd = [str(cc), *cflags, *inc, "-c", str(combined), "-o", str(obj)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0 or not obj.is_file():
        raise XnnpackBoardUnavailable(
            f"qd8 RVV shim compile failed:\ncmd: {' '.join(cmd)}\n{p.stderr[-1500:]}")
    return obj
