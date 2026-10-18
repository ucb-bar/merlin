---
title: Round-even runtime value contract
kind: reference
status: current
owner: runtime
last_verified: 2026-10-18
related: [runtime, whole_model_op_profile, shared_compiler_passes]
code_refs:
  - merlin/runtime/abi/mlir_runtime.c
  - src/merlin/model_check/full_build.py
  - merlin/tests/runtime/test_roundeven_runtime.py
  - merlin/tests/runtime/test_roundeven_runtime_identity.py
  - merlin/tests/infra/test_full_model_lowering_cache.py
---

# Round-even runtime value contract

MLIR `math.roundeven` and LLVM's round-even intrinsic specify ties-to-even
independently of the incoming floating-point rounding mode. Some native code
generators call `roundevenf` or `roundeven` for this operation. Merlin supplies
these runtime symbols when the selected C library lacks them.

The functions round the IEEE binary32/binary64 representation using integer
operations. They preserve signed zero, return integral large values unchanged,
and distinguish an exact half from values above it. NaNs use `x+x`, inheriting
the selected platform's quieting/canonicalization; infinities remain infinities.
No rounding mode is changed. This is a returned-value contract; floating-point
exception flags and `errno` are not an additional guarantee.

The C build checks byte size, float widths, binary radix, significand widths and
exponent limits. Unsupported C formats fail with a precise diagnostic. These
software-format checks do not establish RTL numerical equivalence or qualify a
new target. Target compiler, runtime and execution qualification remain separate.

The former `rint` delegation followed the incoming mode. The runtime regression
compiles and calls the actual functions under all four C rounding modes, checks
complete binary32/binary64 words against an independent NumPy reference, and
includes the old delegation as a failing mutation. The test restores the complete
incoming floating-point environment. A separate Z3 test checks the word-rounding
identity for every non-NaN input, including infinities and signed zero; it is an
arithmetic theorem, not a proof of C compilation or hardware execution.

Run those tests with the declared `verify` extra installed. The runtime itself
does not depend on Z3. Whole-model image reuse includes the actual MLIR ABI runtime
source bytes, so changing this implementation invalidates the image even if the
file size is unchanged. This pin is not a complete header/toolchain/runtime closure.
