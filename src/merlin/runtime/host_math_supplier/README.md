# Host math

Fixed paths relative to the Merlin checkout:

- `src/merlin/runtime/host_math_supplier/host_math.h`: scalar binary32/binary64 C ABI.
- `src/merlin/runtime/host_math_supplier/__init__.py`: policy admission and evaluator binding.
- `src/merlin/runtime/host_math_supplier/host-unary-math-policy.json`: shared supplier selection.

Generated C includes `host_math.h`, calls `merlin_host_<function>`, and links
`libmerlin-host-math.a`. The seven binary32 functions (expf, sinf, cosf, sqrtf, tanhf, erff, and powf) and binary64 sin, cos, and pow are implemented.
Generated C calls `merlin_host_check_fp_env()` once per calling thread before
math and fails unless it returns 1. It checks x86 RNE with FTZ/DAZ off
(and x87 RNE), or RV64 frm=0. Callers retain this environment.
The caller enables the FPU, selects RNE, preserves subnormals, and compiles
without fast math. Implicit FMA contraction is disabled. Explicit CORE-MATH
FMA is preserved. Native builds require x86-64-v3 for hardware FMA.

Nine functions use unmodified CORE-MATH sources pinned by revision and hash.
`sqrtf` uses IEEE-754 hardware `fsqrt.s` on RV64 and `sqrtss` on x86, with
one correctly rounded binary32 result.

NaN results have canonical bits `0x7fc00000` (binary32) or
`0x7ff8000000000000` (binary64). Signed zeros are preserved.
errno and exception flags are unobserved. `supplier_support.h` supplies local
max, floor, and NaN construction, and omits unobserved exception raising.
It also supplies the admitted RNE query and omits portable flag save/restore
operations, since flags are unobserved. It is force-included without changing
the upstream source files. `round_support.c` supplies exact finite binary32/binary64 ties-away integer rounding
for upstream range reduction. Neither target links libm.

Build both targets with RISC-V tools on PATH (or RISCV_PREFIX/RISCV_CC).
The proof takes SPIKE and PK from the environment or PATH, and inherits
LD_LIBRARY_PATH. No machine paths are embedded in this module.
GNU C11 is needed for upstream inline assembly.

Build both targets:

```sh
python build.py --out /absolute/path/to/out/host-math
python tests/prove_identity.py --build-dir /absolute/path/to/out/host-math
```

Receipts bind policy,
sources, support header, ABI header, wrappers, compiler flags, and libraries.
For unary binary32 functions, the proof covers every BF16 encoding, 65,536 deterministic full-binary32
samples, and 48 explicit edges per binary32 function. Binary64 uses 65,536
exponent/sign/significand combinations (every encoded exponent, both signs,
and 16 selected significands), 65,536 seeded xorshift64 patterns, and
64 explicit edges per function. Both unary binary64 functions also run a shared
57,344-input union: 16,384 seeded random significands over abs(x) in
[2^-30, 2^20] with both signs, 16,384 uniform exp inputs in [-746, 710],
4,096 exp inputs in [-745.2, -708.3], 16,384 random log significands
over floor(log2(x)) from -1074 through 1023, and 4,096 log inputs in
[0.5, 2]. This is 188,480 inputs per unary binary64 function.

Both powers run 65,536 seeded full-word operand pairs, 16,384 pairs with
bases in [0.0625, 16] and exponents in [-32, 32] (every fourth pair has
a negative base and an integral exponent), and 16,384 pairs with bases
in [0.999, 1.001] and exponents in [-2^20, 2^20]. Edge lists are crossed
fully: 48*48 binary32 pairs and 64*64 binary64 pairs. powf additionally
runs every BF16 base with half-integer exponents cycling from -8 to 8.
Both also run 16,384 seeded exact-result candidate pairs with integer bases
of magnitude 2 to 65,536, including squares, fourth powers, and sixteenth
powers, and exponents k/2^j with |k| <= 64 and j from 0 to 6. Exactly 4,096
of these pairs have negative bases and integral exponents. Candidate inputs
are exactly representable in both precisions; outputs need not be exact.
Totals: 182,528 powf pairs and 118,784 pow pairs. Seeds, counts,
and authenticated input words are in the proof receipt. Every input is checked against an independent
corpus recipe, and every output byte is compared. It is finite cross-ISA identity
evidence, not exhaustive binary32/binary64 or independent mathematical certification.

Use `load_policy(required=["expf", ...])` before emitting calls and
`symbol("expf")` for the C symbol. Evaluators read that same policy with
`HostMath(build_dir / "x86")`; `call_bits("expf", input_word)` preserves raw
binary32 transport, or uint64 words for sin, cos, and pow.
Use `call("pow", base, exponent)` or `call_bits("pow", base_word, exponent_word)`
(and likewise powf) for binary operations. `call` accepts
Python floats and selects the ABI by function name. Construction and each
call check the current thread FP environment and raise RuntimeError on failure. The module also loads
by `importlib.util.spec_from_file_location` without importing optional Merlin
framework dependencies. Unknown functions and altered policy or library
hashes refuse; no libm fallback exists.

Additional functions require explicit ABI and policy admission, pinned
upstream sources, both builds, and an extended proof. Emitter integration and
host source-op ordering belong to consuming streams.
