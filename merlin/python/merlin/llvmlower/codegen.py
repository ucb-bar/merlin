"""LLVM IR -> object files (host x86 and bare-metal rv64gcv) and host .so."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from merlin.common.paths import runtime_dir

from ..common.paths import repo_root
from .toolchain import clang

# A bounded per-compile wall clock. A pathological schedule (e.g. an outer-product contraction at a
# large square regime) can make clang -O2 blow up and spin for many minutes on one object file; in a
# serial beam that hangs the whole sweep. Time it out so the fork fails-closed as a build error the
# certify ladder records, instead of stalling. Override with MERLIN_COMPILE_TIMEOUT_S (0 disables).
_COMPILE_TIMEOUT_S = int(os.environ.get("MERLIN_COMPILE_TIMEOUT_S", "300") or "0")

# Host (x86) kernel flags. The compiled kernels are plain clang-vectorized loops (no BLAS),
# so the GEMM-heavy graphs of large models (pi05, openvla) are runtime-bound on them. -O3
# -march=native lets clang use the host's vector units (AVX2/AVX-512) + aggressive unrolling,
# multiplying throughput. IEEE semantics are preserved (NO -ffast-math) so host==torch holds.
X86_FLAGS = ["-O3", "-march=native", "-funroll-loops", "-fPIC"]
RISCV_FLAGS = ["--target=riscv64-unknown-elf", "-march=rv64gcv", "-mabi=lp64d",
               "-mcmodel=medany", "-O2", "-ffreestanding", "-fno-builtin"]


def mlir_runtime_c() -> Path:
    """Merlin's MLIR C-runtime shim (memrefCopy, ...) linked into every target."""
    return runtime_dir() / "abi/mlir_runtime.c"


class CodegenError(RuntimeError):
    pass


def _run(cmd: list[str]) -> None:
    try:
        proc = subprocess.run([str(c) for c in cmd], capture_output=True, text=True,
                              timeout=(_COMPILE_TIMEOUT_S or None))
    except subprocess.TimeoutExpired:
        raise CodegenError(f"clang timed out after {_COMPILE_TIMEOUT_S}s (pathological compile): "
                           f"{' '.join(map(str, cmd))}")
    if proc.returncode != 0:
        raise CodegenError(f"clang failed: {' '.join(map(str, cmd))}\n{proc.stderr}")


def compile_ll(ll_path: str | Path, out_obj: str | Path, target: str = "riscv") -> Path:
    """Compile LLVM IR to an object file for ``riscv`` (rv64gcv) or ``x86``."""
    flags = RISCV_FLAGS if target == "riscv" else X86_FLAGS
    _run([clang(), *flags, "-c", ll_path, "-o", out_obj])
    return Path(out_obj)


def build_host_shared(ll_path: str | Path, out_so: str | Path) -> Path:
    """Host .so (ctypes execution on x86), with the MLIR C-runtime shim.

    clang-23 (the IREE build) has no host C headers, so it only compiles the .ll;
    the runtime C and the final link use the system compiler.
    """
    out_so = Path(out_so)
    model_o = out_so.with_suffix(".o")
    rt_o = out_so.with_name("mlir_runtime_host.o")
    _run([clang(), "-O2", "-fPIC", "-c", ll_path, "-o", model_o])
    _run(["cc", "-O2", "-fPIC", "-c", str(mlir_runtime_c()), "-o", rt_o])
    _run(["cc", "-shared", model_o, rt_o, "-lm", "-o", out_so])
    return out_so
