"""LLVM IR -> object files (host x86 and bare-metal rv64gcv) and host .so."""

from __future__ import annotations

import os
from pathlib import Path

from merlin.common.paths import runtime_dir
from merlin.compiler.build import process as _process
from merlin.compiler.build import shared as _shared

from .optional_passes import child_environment
from .toolchain import clang, host_llc

# A bounded per-compile wall clock. A pathological schedule (e.g. an outer-product contraction at a
# large square regime) can make clang -O2 blow up and spin for many minutes on one object file; in a
# serial beam that hangs the whole sweep. Time it out so the fork fails-closed as a build error the
# certify ladder records, instead of stalling. Override with MERLIN_COMPILE_TIMEOUT_S (0 disables).
# Default unified at 900s across all four compile wrappers (was 300): a per-object 300s ceiling
# tripped legitimate whole-model builds. For a whole-model beam launch set MERLIN_COMPILE_TIMEOUT_S=3600.
_COMPILE_TIMEOUT_S = int(os.environ.get("MERLIN_COMPILE_TIMEOUT_S", "900") or "0")

# Host (x86) kernel flags. The compiled kernels are plain clang-vectorized loops (no BLAS),
# so the GEMM-heavy graphs of large models (pi05, openvla) are runtime-bound on them. -O3
# -march=native lets clang use the host's vector units (AVX2/AVX-512) + aggressive unrolling,
# multiplying throughput. IEEE semantics are preserved (NO -ffast-math) so host==torch holds.
X86_FLAGS = ["-O3", "-march=native", "-funroll-loops", "-fPIC"]
RISCV_FLAGS = [
    "--target=riscv64-unknown-elf",
    "-march=rv64gcv",
    "-mabi=lp64d",
    "-mcmodel=medany",
    "-O2",
    "-ffreestanding",
    "-fno-builtin",
]


def mlir_runtime_c() -> Path:
    """Merlin's MLIR C-runtime shim (memrefCopy, ...) linked into every target."""
    return runtime_dir() / "abi/mlir_runtime.c"


class CodegenError(RuntimeError):
    pass


# Compatibility attributes retain the actual shared objects.
_proc = _process._proc
STAGES = _shared.STAGES


def _run(cmd: list[str], *, timeout_s: float | None = None, inputs=(), outputs=(), stage="object"):
    """Legacy process selection; actual invocation lives in compiler.build.process."""
    return _process.run(
        cmd,
        environment=child_environment(),
        timeout_limit_s=_COMPILE_TIMEOUT_S,
        error=CodegenError,
        dependencies=(Path(__file__), Path(_shared.__file__)),
        timeout_s=timeout_s,
        inputs=inputs,
        outputs=outputs,
        stage=stage,
    )


def compile_ll(
    ll_path: str | Path,
    out_obj: str | Path,
    target: str = "riscv",
    *,
    extra_flags: tuple[str, ...] = (),
    timeout_s: float | None = None,
) -> Path:
    """Compile LLVM IR, optionally under a tighter per-call diagnostic limit."""
    flags = RISCV_FLAGS if target == "riscv" else X86_FLAGS
    from merlin.compiler.build.objects import compile_llvm_object

    return compile_llvm_object(
        ll_path,
        out_obj,
        compiler=clang(),
        flags=(*flags, *extra_flags),
        run=_run,
        timeout_s=timeout_s,
    )


def build_host_shared(ll_path: str | Path, out_so: str | Path, *, source_lowering=None) -> Path:
    """Preserve legacy tool selection; shared construction has one compiler owner."""
    return _shared.build_shared_library(
        ll_path,
        out_so,
        run=_run,
        compiler=clang,
        object_compiler=host_llc,
        runtime_compiler=lambda: os.environ.get("CC", "").strip() or clang(),
        runtime_source=mlir_runtime_c,
        source_lowering=source_lowering,
    )
