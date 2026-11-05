"""LLVM IR -> object files (host x86 and bare-metal rv64gcv) and host .so."""

from __future__ import annotations

import math
import os
import subprocess
from pathlib import Path

from merlin.common import compile_trace
from merlin.common import proc as _proc
from merlin.common.paths import runtime_dir

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


#: What this module produces from LLVM IR, as compile-trace stages: each object it compiles, and each
#: program it links.
STAGES = compile_trace.declare(
    "codegen",
    ("object", "link"),
    entry="merlin.llvmlower.codegen",
    summary="LLVM IR -> object (clang) -> linked program",
)


def _run(cmd: list[str], *, timeout_s: float | None = None, inputs=(), outputs=(), stage="object"):
    if timeout_s is not None and (
        isinstance(timeout_s, bool)
        or not isinstance(timeout_s, (int, float))
        or not math.isfinite(timeout_s)
        or timeout_s <= 0
    ):
        raise ValueError("explicit compile timeout must be positive")
    limit = _COMPILE_TIMEOUT_S or None
    if timeout_s is not None:
        limit = min(limit, timeout_s) if limit is not None else timeout_s
    from merlin.common import invocation_record
    from merlin.common.cpu_pool import shared_compute_pool
    from merlin.runtime.build_runtime import run_in_build_view

    if not outputs:
        with shared_compute_pool(timeout_s=limit):
            return _proc.run_checked(cmd, error=CodegenError, timeout=limit, timeout_hint=" (pathological compile)")
    argv = tuple(str(token) for token in cmd)
    cwd, environment = Path.cwd(), dict(os.environ)
    if "-fstack-usage" in argv:
        outputs = tuple(dict.fromkeys((*map(Path, outputs), Path(outputs[0]).with_suffix(".su"))))
    try:
        with shared_compute_pool(timeout_s=limit):
            selected = run_in_build_view(
                argv,
                inputs=inputs,
                outputs=outputs,
                directory=Path(outputs[0]).parent.resolve(),
                stage=stage,
                cwd=cwd,
                env=environment,
                timeout=limit,
                capture_output=True,
                text=True,
            )
    except subprocess.TimeoutExpired:
        shown = " ".join(argv)[:1000]
        raise CodegenError(f"command timed out after {limit}s (pathological compile): {shown}") from None
    if selected is not None:
        if selected.returncode:
            shown = " ".join(argv)[:1000]
            raise CodegenError(
                f"command failed (rc {selected.returncode}): {shown}\n"
                f"STDOUT:{(selected.stdout or '')[-2000:]}\nSTDERR:{(selected.stderr or '')[-2000:]}"
            )
        return selected
    with (
        shared_compute_pool(timeout_s=limit),
        invocation_record.observe(
            Path(outputs[0]).parent,
            stage=stage,
            argv=argv,
            cwd=cwd,
            env=environment,
            inputs=inputs,
            outputs=outputs,
            dependencies=(Path(__file__),),
        ) as record,
    ):
        result = _proc.run_checked(
            argv,
            error=CodegenError,
            timeout=limit,
            timeout_hint=" (pathological compile)",
            cwd=cwd,
            env=environment,
        )
        record.complete(result)
        return result


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
    _run(
        [clang(), *flags, *extra_flags, "-c", ll_path, "-o", out_obj],
        timeout_s=timeout_s,
        inputs=(ll_path,),
        outputs=(out_obj,),
    )
    compile_trace.artifact("object", [out_obj], pipeline="codegen")
    return Path(out_obj)


def build_host_shared(ll_path: str | Path, out_so: str | Path, *, source_lowering=None) -> Path:
    """Host .so (ctypes execution on x86), with the MLIR C-runtime shim.

    The runtime shim uses ``__bf16``. Use the same selected clang as the model
    object by default, so an older system ``cc`` cannot break host references;
    an explicit ``$CC`` remains an operator selection.
    """
    out_so = Path(out_so)
    model_o = out_so.with_suffix(".o")
    rt_o = out_so.with_name("mlir_runtime_host.o")
    run = _run
    if source_lowering is not None:
        from .native_source_model import NativeSourceModel

        if type(source_lowering) is not NativeSourceModel:
            raise ValueError("native source linking requires its original typed transaction")
        source_lowering.begin_codegen(out_so.parent)

        def run(command, **kwargs):
            return source_lowering.run(command, runner=_run, **kwargs)

    selected_llc = host_llc()
    if selected_llc is None:
        run(
            [clang(), "-O2", "-fPIC", "-c", ll_path, "-o", model_o],
            inputs=(ll_path,),
            outputs=(model_o,),
        )
    else:
        run(
            [selected_llc, "-O2", "-filetype=obj", "-relocation-model=pic", ll_path, "-o", model_o],
            inputs=(ll_path,),
            outputs=(model_o,),
        )
    compile_trace.artifact("object", [model_o], pipeline="codegen")
    host_cc = os.environ.get("CC", "").strip() or clang()
    runtime = mlir_runtime_c()
    run(
        [host_cc, "-O2", "-fPIC", "-c", str(runtime), "-o", rt_o],
        inputs=(runtime,),
        outputs=(rt_o,),
        stage="runtime_object",
    )
    provider_objects = ()
    if source_lowering is not None:
        provider_objects = source_lowering.objects(
            Path(ll_path), model_o, rt_o, runtime_source=runtime, compiler=host_cc, flags=("-O2", "-fPIC"), runner=_run
        )
    from .adjacent_exp import runtime_c_for as exp_runtime_c_for
    from .guarded_zero_contraction import runtime_c_for

    selected_runtime = runtime_c_for(Path(ll_path).read_text()) + exp_runtime_c_for(Path(ll_path).read_text())
    guarded_objects = ()
    if selected_runtime:
        helper_c = out_so.with_name("guarded_zero_contraction_runtime.c")
        helper_o = helper_c.with_suffix(".o")
        helper_c.write_text(selected_runtime)
        run(
            [host_cc, "-O2", "-fPIC", "-c", helper_c, "-o", helper_o],
            inputs=(helper_c,),
            outputs=(helper_o,),
            stage="runtime_object",
        )
        guarded_objects = (helper_o,)
    objects = (model_o, rt_o, *provider_objects, *guarded_objects)
    native_flags = ("-nostartfiles", "-Wl,-z,defs") if source_lowering is not None else ()
    run(
        [host_cc, "-shared", *native_flags, *objects, "-lm", "-o", out_so],
        inputs=objects,
        outputs=(out_so,),
        stage="link",
    )
    compile_trace.artifact("link", [out_so], pipeline="codegen")
    if source_lowering is not None:
        source_lowering.complete(objects, out_so, link_flags=(*native_flags, "-lm"))
    return out_so
