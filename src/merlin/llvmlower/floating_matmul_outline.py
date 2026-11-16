"""Installed host driver and compatibility exports for floating contractions.

Pure outlining is independently exportable from floating_contractions; this
module retains the existing host toolchain entry point for installed callers.
"""

from .floating_contractions import (
    callback_bridge as callback_bridge,
)
from .floating_contractions import (
    contraction_source as contraction_source,
)
from .floating_contractions import (
    outline_floating_matmuls as outline_floating_matmuls,
)
from .floating_contractions import (
    static_tensor_signature as static_tensor_signature,
)


def compile_callback_host(host, records, *, destination, llc, cc):
    """Compile the outlined graph with the existing host pipeline/runtime.

    The caller owns a fresh build directory. Link-time unresolved-symbol
    checking catches missing runtime helpers before an image is published.
    """
    import subprocess
    from pathlib import Path

    from merlin.llvmlower.codegen import mlir_runtime_c
    from merlin.llvmlower.pipeline import lower_to_llvm_ir

    destination = Path(destination)
    llvm = lower_to_llvm_ir(host, workdir=destination / "host-lowering")
    (destination / "host.ll").write_text(llvm)
    (destination / "dispatch.c").write_text(callback_bridge(records))
    runtime = mlir_runtime_c().read_bytes()
    (destination / "mlir_runtime.c").write_bytes(runtime)
    subprocess.run(
        [
            str(llc),
            "-filetype=obj",
            "-relocation-model=pic",
            "-O2",
            str(destination / "host.ll"),
            "-o",
            str(destination / "host.o"),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
    )
    subprocess.run(
        [
            str(cc),
            "-shared",
            "-fPIC",
            "-O2",
            str(destination / "dispatch.c"),
            str(destination / "host.o"),
            str(destination / "mlir_runtime.c"),
            "-Wl,-z,defs",
            "-lm",
            "-o",
            str(destination / "model.so"),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
    )
