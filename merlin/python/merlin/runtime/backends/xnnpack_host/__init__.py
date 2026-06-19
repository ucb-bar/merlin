"""HOST XNNPACK kernel backend for the dispatch runtime (default-off, additive).

This is the third e2e column scaffolding: instead of routing every per-dispatch kernel
through the Merlin-compiled host ``.so`` (``llvmlower.kernel_backend``), the dispatch
runtime can route the dispatches XNNPACK covers -- here, the f32 ``linalg.matmul`` GEMMs --
through XNNPACK's own microkernel (the scalar f32-gemm ukernel, vendored verbatim under
``tmp/kernels/XNNPACK``). Every other dispatch (attention as a batched generic, layernorm /
rmsnorm, elementwise activations not yet wired) falls through to the existing compiled-``.so``
path UNCHANGED. So this is a hybrid kernel-backend swap, and it isolates how much of the
e2e gap is kernel-level vs runtime/glue-level.

HOST ONLY: this uses the portable scalar f32-gemm microkernel so the math is bit-comparable
to the compiled kernel and needs no SIMD/runtime detection. It proves the SEAM (a kernel
dispatch routed through an XNNPACK ukernel, whole-model, gated against the torch golden).
Board (RVV) cross-compile + timing is a separate, later step (it reuses the SAME microkernel
family via the existing K1 ceiling drivers).

Enable by passing ``kernel_backend="xnnpack"`` to ``dispatch_runtime.run_model`` /
``execute`` (or set ``MERLIN_XNNPACK_HOST=1``). Without it the runtime is byte-for-byte the
default compiled path.
"""
from __future__ import annotations

import ctypes
import subprocess
import threading
from pathlib import Path
from typing import Any

import numpy as np

_HERE = Path(__file__).resolve().parent
_SHIM_INC = _HERE / "shim"
_GEMM_SRC = _HERE / "xnn_gemm_shim.c"


def _xnnpack_repo() -> Path:
    """Vendored XNNPACK checkout (matches the ceiling drivers / mining adapter)."""
    import os

    env = os.environ.get("MERLIN_XNNPACK_REPO")
    if env:
        return Path(env)
    from merlin.common.paths import repo_root

    return Path(repo_root()) / "tmp" / "kernels" / "XNNPACK"


class XnnpackUnavailable(RuntimeError):
    pass


_LIB_LOCK = threading.Lock()
_LIB: Any = None
_GEMM_FN: Any = None


def _build_gemm_lib() -> Path:
    """Compile the host XNNPACK GEMM shim ``.so`` once into ``output/.xnnpack_host/``."""
    from merlin.common.paths import repo_root

    out_dir = Path(repo_root()) / "output" / ".xnnpack_host"
    out_dir.mkdir(parents=True, exist_ok=True)
    so = out_dir / "xnn_gemm_host.so"
    xnn_src = _xnnpack_repo() / "src"
    if not (xnn_src / "f32-gemm" / "gen" / "f32-gemm-4x4-minmax-scalar.c").is_file():
        raise XnnpackUnavailable(
            f"XNNPACK source not found under {xnn_src} "
            "(set MERLIN_XNNPACK_REPO to a checkout)")
    # Rebuild if missing or the shim is newer than the artifact.
    if so.is_file() and so.stat().st_mtime >= max(_GEMM_SRC.stat().st_mtime,
                                                   (_SHIM_INC / "src/xnnpack/gemm.h").stat().st_mtime):
        return so
    cmd = ["cc", "-O2", "-fPIC", "-shared",
           "-I", str(_SHIM_INC), "-I", str(xnn_src),
           str(_GEMM_SRC), "-o", str(so)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0 or not so.is_file():
        raise XnnpackUnavailable(f"XNNPACK host GEMM build failed: {p.stderr[-600:]}")
    return so


def _gemm_fn():
    global _LIB, _GEMM_FN
    with _LIB_LOCK:
        if _GEMM_FN is not None:
            return _GEMM_FN
        so = _build_gemm_lib()
        lib = ctypes.CDLL(str(so))
        fn = lib.merlin_xnn_gemm_f32
        fn.restype = ctypes.c_int
        fn.argtypes = [ctypes.c_size_t] * 3 + [ctypes.c_void_p] * 3
        _LIB, _GEMM_FN = lib, fn
        return fn


def is_available() -> bool:
    try:
        _build_gemm_lib()
        return True
    except XnnpackUnavailable:
        return False


def classify_matmul_kernel(kfn) -> dict | None:
    """If kernel func ``kfn`` is a routable plain 2-D f32 ``linalg.matmul``, return
    ``{"a": i, "b": j}`` -- the kernel-block-arg indices of the A and B operands -- else None.

    Routable = the kernel's returned value is produced by a single ``linalg.matmul`` whose
    two ``ins`` are kernel block args (the activation and weight), both 2-D f32, and whose
    result is 2-D f32. The optional ``linalg.fill`` zero-init of the out tensor is ignored
    (XNNPACK accumulates from a zero bias). Batched/transposed-as-generic matmuls and any
    non-f32 matmul are NOT routed -- they fall through to the compiled path.
    """
    block = kfn.body.blocks[0]
    arg_ids = {id(a): i for i, a in enumerate(block.args)}
    ret = next((o for o in block.ops if o.name == "func.return"), None)
    if ret is None or len(ret.operands) != 1:
        return None
    mm = getattr(ret.operands[0], "owner", None)
    if mm is None or getattr(mm, "name", None) != "linalg.matmul":
        return None
    # ins are operands[0], operands[1]; out (init) is the last operand.
    if len(mm.operands) < 3:
        return None
    a_val, b_val = mm.operands[0], mm.operands[1]

    def _f32_2d(v) -> bool:
        from xdsl.dialects.builtin import TensorType

        t = v.type
        if not isinstance(t, TensorType):
            return False
        return len(t.get_shape()) == 2 and str(t.element_type) == "f32"

    if not (_f32_2d(a_val) and _f32_2d(b_val) and _f32_2d(mm.results[0])):
        return None
    if id(a_val) not in arg_ids or id(b_val) not in arg_ids:
        return None  # an operand is computed inside the kernel (e.g. transpose) -> not routed
    return {"a": arg_ids[id(a_val)], "b": arg_ids[id(b_val)]}


def gemm_f32(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """C = A @ B through the XNNPACK scalar f32 microkernel. A: MxK, B: KxN, row-major f32."""
    A = np.ascontiguousarray(A, np.float32)
    B = np.ascontiguousarray(B, np.float32)
    M, K = A.shape
    K2, N = B.shape
    if K != K2:
        raise ValueError(f"gemm shape mismatch {A.shape} @ {B.shape}")
    C = np.zeros((M, N), np.float32)
    rc = _gemm_fn()(M, N, K, A.ctypes.data, B.ctypes.data, C.ctypes.data)
    if rc != 0:
        raise XnnpackUnavailable(f"xnn gemm returned {rc} (alloc failure)")
    return C
