---
title: "Design note: one erf/exp/tanh/log on every target"
kind: design
status: current
owner: compiler
last_verified: 2026-11-05
related: [optional_passes]
code_refs: [merlin/runtime/abi/portable_math.h, merlin/runtime/abi/portable_math_tables.h, merlin/runtime/abi/mlir_runtime.c, src/merlin/llvmlower/portable_math.py, src/merlin/llvmlower/lower.py, src/merlin/llvmlower/optional_passes.py, build_tools/scripts/gen_portable_math_tables.py, build_tools/scripts/make_w8a8_golden.py, merlin/tests/runtime/test_portable_transcendentals.py, merlin/tests/rvv/test_portable_gelu_gate.py]
---

# One erf/exp/tanh/log on every target

**Rule:** a golden and the device it grades must evaluate the same function, bit for bit. The
`portable-transcendentals` numeric policy makes that true for `erf`, `exp`, `tanh` and `log` in
f32, and records which policy produced each golden and each build.

## What broke

`convert-math-to-libm` lowers `math.erf`/`exp`/`tanh`/`log` to calls to the C library: `erff`,
`expf`, `tanhf`, `logf`. The host build that writes `golden_w8a8.npy`
(`make_w8a8_golden.py`, through `dispatch_runtime.run_model`) links glibc. A RISC-V image links
picolibc or newlib. So the "same" model computed a different function on each side.

Spectformer int8 was the first model where this showed. On RISC-V it missed the W8A8 gate's
per-element bound: `w8a8_cos` 0.9999935, `rel` 0.0064, `max_rel` 0.16, same top-1. A host build of
the same model matched the golden bit for bit, and `-ffp-contract=off` changed nothing. The probe
`out/artifacts/probes/erff_glibc_vs_riscv_libm/` found the cause: glibc's and the RISC-V
toolchain's `erff` disagree by one ulp on 76 of 8001 inputs in GELU's range. Dynamic int8
requantization of the GELU output turns an occasional one-ulp difference into a whole int8 step.

No gate was wrong. Its premise, that the two sides run the same program, did not hold.

## The policy

| policy | erf/exp/tanh/log come from | selected by |
|---|---|---|
| `portable` (default) | `merlin_erff`/`merlin_expf`/`merlin_tanhf`/`merlin_logf` in Merlin's MLIR runtime | default; `--pass portable-transcendentals` |
| `libm` | each target's C library (the old behaviour) | `--no-pass portable-transcendentals`, `MERLIN_PASSES=-portable-transcendentals`, or `MERLIN_PORTABLE_TRANSCENDENTALS=0` |

The policy is an entry of the optional-pass registry (`merlin-compile --list-passes`), marked
`numerics-changing`. `merlin.llvmlower.portable_math.bind` applies it on the final LLVM IR in
`lower_model`, the step every host and device lowering goes through. Under `portable` it renames
the bound globals (`@erff` becomes `@merlin_erff`, in the declaration and at every call) with a
tokenizer for LLVM global identifiers, not text matching. Under `libm` the IR is byte-identical to
before. Every lowering records the policy in `stats["transcendental_math"]`: the policy, how many
references were rebound, a digest of the implementation, and `left_on_target_libm`, the
transcendentals the module still takes from a C library.

The definitions live in `merlin/runtime/abi/portable_math.h`. `mlir_runtime.c` includes that file
once, and `mlir_runtime.c` is already linked into the host `.so` (`codegen.build_host_shared`), the
Zephyr image, the bare-metal spike image, the K1 binary and the kernel-on-spike harness. So the
golden and the device link the same source.

Each golden written by `make_w8a8_golden.py` records the policy and the implementation digest in
`golden_w8a8.provenance.json`. A golden with no record counts as `libm`, since that is what every
earlier golden was computed with. When the build's policy differs from the golden's,
`merlin-compile --verify` prints `transcendental_math_warning` and puts it in the result. The gate
is not loosened either way.

### Why `portable` is the default

The W8A8 tier compares a device against the host that wrote its golden. That comparison means
something only if both run the same program. Under `libm` they never did for these functions, and
the difference stayed hidden only until a model's quantization grid happened to land on it. The
portable functions are also more accurate than the C libraries they replace (see below).

### Why a C runtime function and not an IR polynomial

The alternative was to lower `erf` to an explicit polynomial in IR, as the
`approximate_transcendental_activation` feature (`act_poly`) does, and have the golden evaluate the
same polynomial. That ties the bits to how the backend schedules that IR: a vectorized device
build and a scalar host build, each with its own contraction behaviour. A function compiled from
one C source, written so that no compiler can legally change its result, keeps one definition that
both sides link. `act_poly` stays a separate opt-in throughput feature with its own accuracy
contract. It is unaffected by this policy.

## How the bits are pinned

Each function converts its argument to double (exact), evaluates in double, and rounds once to
float. IEEE-754 defines every operation used (`+ - * /`, fused multiply-add, conversion) exactly.
The one freedom a conforming C compiler still has is contraction: fusing a multiply that feeds an
add into one fma. GCC does this across statements by default for GNU C. Clang does it within an
expression, and across statements under `-ffp-contract=fast`, which ignores every pragma. So the
source leaves nothing to fuse. Every multiply-add is an explicit fma, and no other product feeds an
addition. Clang's `__builtin_elementwise_fma` and GCC's `__builtin_fma` emit the fused instruction
(`fmadd.d` on RISC-V, `vfmadd` on x86 with FMA). Without hardware FMA they call the C library's
`fma`, which C99 requires to be correctly rounded. The file refuses `-ffast-math` and
excess-precision evaluation with `#error`. A NaN argument returns the canonical quiet NaN
`0x7fc00000`, because how a NaN payload propagates differs between ISAs.

The algorithms:

- `exp`: Cody-Waite reduction x = k ln2 + r, |r| <= ln2/2, then the Taylor series of degree 13
  (truncation < 5e-18 relative), scaled by an exact 2^k.
- `erf`:
  - |x| < 0.5: the Maclaurin series x * sum c_n x^(2n), 14 terms.
  - 0.5 <= |x| < 4: erf = 1 - exp(-x^2) * g(x), where g(x) = erfc(x) exp(x^2) is a degree-14
    Chebyshev interpolant on each of seven half-unit intervals (fit error < 2e-19 relative).
  - |x| >= 4: 1, because erfc(4) < 2^-25.
- `tanh`: expm1(2a) / (expm1(2a) + 2), with expm1 from its own series near 0. Returns 1 for a >= 10.
- `log`: x = m 2^e with m in [sqrt(1/2), sqrt(2)), then 2 atanh((m-1)/(m+1)) by its series
  (11 terms), plus e ln2 split hi/lo.

All coefficients are mathematical constants computed at 60 digits and rounded once.
`build_tools/scripts/gen_portable_math_tables.py` writes them as exact hexadecimal literals.
`--check` (run by the tests) proves the checked-in header regenerates byte for byte.

## Accuracy (measured 2026-11-05, every float input)

Every float bit pattern except NaNs was evaluated: 4,278,190,082 inputs per function, and
2,139,095,042 positive inputs for `log`. Each result was compared with glibc's double-precision
function rounded to float. glibc's double functions are accurate to better than 2^-52, so
agreement means correct rounding everywhere except possibly where the true value lies near a
rounding midpoint. Every input whose reference lies within 2^-44 (relative) of a midpoint was then
re-decided at 60 digits with mpmath.

| function | inputs ≠ CR | max error (ulp) | near-midpoint inputs re-decided | glibc float: max ulp / inputs ≠ portable |
|---|---|---|---|---|
| `erff` | 0 | 0.5 | 3,884 | 0.968 / 126,805,016 |
| `expf` | 0 | 0.5 | 730 | 0.502 / 170,648 |
| `tanhf` | 0 | 0.5 | 414 | 2.189 / 118,674,314 |
| `logf` | 5 | 0.50000000085 | 2,882 | 0.818 / 416,909 |

`erff`, `expf` and `tanhf` are correctly rounded on every float input. `logf` is correctly
rounded on all but five inputs (`0x3c413d3a`, `0x41178feb`, `0x4c5d65a5`, `0x65d890d3`,
`0x6f31a8ec`). Their true values lie within 1e-9 ulp of a midpoint, which is below what a double
evaluation resolves. Both glibc's double `log` and the portable `logf` round them the same way.

On the probe's GELU grid (8001 points), the portable `erff` differs from glibc's on 104 points and
from the RISC-V toolchain's (newlib) on 112. glibc and newlib differ from each other on 76.

### Cross-compiler identity

`portable_math_sweep.c` is run against the runtime object exactly as each build produces it. It
covers 3.18 million inputs per function: a stride through all 2^32 bit patterns, plus a dense band
below 8 in magnitude. The FNV hash of the result bits must be identical across:

- host `cc -O2 -fPIC` (the golden's build)
- host `-O0`
- host `-O3 -march=native -ffp-contract=fast`
- RISC-V GCC at the device flags plus `-ffp-contract=fast`, run on spike
- RISC-V clang through `spike_model`'s runtime compiler, run on spike

It is identical on all five. The Zephyr SDK's GCC was checked separately and emits the fused
instructions inline, with no call to `fma`.

## What changes for other targets and models

- **Every model that evaluates `erf`, `exp`, `tanh` or `log` in f32 computes slightly different
  outputs than before, on every target, host included.** Each differing value moves by one ulp, in
  the direction of the true value. That is about 3% of `erf`/`tanh` arguments and about 0.004% of
  `exp`/`log` arguments, relative to glibc. Spectformer's regenerated golden moved by at most
  9.5e-7 (rel 2e-7, 380 of 1000 logits). Its cosine to the independent torchao reference is
  unchanged (0.9998784).
- **Goldens written under `libm` are not bit-exact references for `portable` builds.** Regenerate
  `golden_w8a8.npy` with `make_w8a8_golden.py --force` for bundles graded at the W8A8 tier.
  `merlin-compile` warns when the two disagree. The independent torchao golden is not affected:
  it never shared bits with either side.
- `left_on_target_libm` lists any other transcendental still taken from a C library (`sin`,
  `pow`, `expm1`, f64 functions, and LLVM math intrinsics the backend expands into libm calls).
  Those keep the old cross-target divergence until they are covered too.
- Cost: spectformer int8 at h1 on spike executes 1.9% more instructions (4.317e9 to 4.400e9). Spike
  counts instructions, not cycles, so this is not a timing on silicon.
- A Zephyr image's `build_hash` covers the model object, so it changes with the policy, since the
  object names `merlin_erff` or `erff`. It does not cover the runtime sources. An edit to
  `portable_math.h` alone shows up as a different `implementation_sha256` in the lowering stats,
  not as a different `build_hash`. `spike_model`'s hash covers every runtime ABI source.
- `spike_model`'s explicit `host_math_policy="expf_via_double"` wraps `expf` at link time. Under the
  portable policy the model no longer calls `expf`, so that wrap has nothing to act on; the
  portable `expf` already evaluates in double.

## Evidence

- `merlin/tests/runtime/test_portable_transcendentals.py`: the binding, the policy, the tables,
  special values, accuracy within 0.5 ulp on a sample, and the cross-compiler sweep on the host
  and on spike.
- `merlin/tests/rvv/test_portable_gelu_gate.py`: the spectformer failure in miniature. A GELU and
  requantization kernel is built for the host and for spike. Under `libm` it fails the W8A8 gate
  on the per-element bound alone (cos and rel inside their bounds). Under `portable` the two
  builds are bit-identical and the gate passes.
- Spectformer int8, `chipyard_kodiak`, h1, spike, golden regenerated under `portable`:

| image | policy / golden | w8a8_cos | w8a8_rel | w8a8_max_rel | gate |
|---|---|---|---|---|---|
| RVV | libm / glibc golden (before) | 0.9999935 | 0.0064 | 0.161 | fail |
| RVV | portable / portable golden | 1.0 | 0.0 | 0.0 | pass (w8a8 tier) |
| RVV+OPU (twin) | libm / glibc golden (before) | 0.9999935 | 0.0064 | 0.161 | fail (no tier) |
| RVV+OPU (twin) | portable / portable golden | 1.0 | 0.0 | 0.0 | pass (w8a8 tier, per-element guarded) |
