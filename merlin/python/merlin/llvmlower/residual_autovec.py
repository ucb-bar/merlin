"""Let clang vectorize loops that the explicit MLIR schedule did not claim.

RVV packages historically carry ``-fno-vectorize`` and ``-fno-slp-vectorize`` so
backend vectorization cannot obscure experiments on explicitly scheduled contraction
IR.  Applying those package flags to the whole-model object also disables every
remaining activation, layout, copy, and quantization loop.  On K1, with the same
LSTMNetVIT W8A8 schedule and runtime, removing only those two global disables changed
the sustained eight-core wall from 284.8 ms to 120.3 ms (2.37x).  The linked model
census changed as well, so this is code generation rather than runtime noise.

The same one-variable test now exists for the panel-packed TinyLlama W8A8 path.  The
prepared LLVM IR and packed weight blob are byte-identical; compiling only the model
object with these disables present versus removed gives median-of-three interleaved
walls of 2080.6 vs 1826.2 ms at one core and 749.4 vs 496.4 ms at eight cores.  Every
arm produces the same complete 256,000-element output SHA.  This matters because it
separates model-object compiler policy from panel layout and OpenMP scheduling: the
LLVM IR, weights, runtime, and link are held fixed in that A/B.

This feature is deliberately default-off and edits only the model-object flags.  The
explicit vector IR produced by the transform schedule remains explicit; clang is
allowed to optimize the scalar residue around it.  Correctness and cross-model
performance still belong to the normal board gates before promotion.
"""
from __future__ import annotations


FEATURE = "vectorize_scalar_residue"
_GLOBAL_DISABLES = frozenset(("-fno-vectorize", "-fno-slp-vectorize"))


def _edit_cflags(flags: list[str]) -> list[str]:
    return [flag for flag in flags if flag not in _GLOBAL_DISABLES]


def ensure_registered() -> str:
    from .impr_features import ImprFeature, known, register

    if FEATURE not in known():
        register(ImprFeature(
            name=FEATURE,
            action_class="HEURISTIC",
            description=(
                "Remove the package-wide -fno-vectorize/-fno-slp-vectorize flags from the "
                "whole-model object so clang may vectorize scalar residue around explicitly "
                "scheduled RVV contractions. Measured on K1: LSTMNetVIT W8A8 284.8 ms to "
                "120.3 ms at eight harts (2.37x), and panel-packed TinyLlama W8A8 median "
                "749.4 ms to 496.4 ms (1.51x) at eight harts with byte-identical LLVM IR, "
                "weights, and complete output; default off and subject to correctness and "
                "cross-model board gates."),
            edit_cflags=_edit_cflags,
        ))
    return FEATURE
