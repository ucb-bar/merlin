"""Build one LLVM object using an explicitly selected compiler and target flags.

The caller supplies its existing observed process runner. This owner chooses no
target, compiler defaults, runtime, startup code, or link inputs.
"""

from __future__ import annotations

from pathlib import Path

from merlin.compiler.ir import compile_trace

# ---- objects -----------------------------------------------------------------------------------


def compile_llvm_object(source, destination, *, compiler, flags, run, timeout_s=None):
    """Compile actual LLVM input, retaining the caller's invocation custody."""
    source, destination = Path(source), Path(destination)
    run(
        [compiler, *flags, "-c", source, "-o", destination],
        timeout_s=timeout_s,
        inputs=(source,),
        outputs=(destination,),
    )
    compile_trace.artifact("object", [destination], pipeline="codegen")
    return destination
