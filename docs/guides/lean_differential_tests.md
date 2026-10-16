---
title: Run Lean semantics differential tests
kind: guide
status: current
owner: verification
last_verified: 2026-10-15
related: [merlin_lean_verification, compiler_verification]
code_refs:
  - merlin/lean/Differential.lean
  - src/merlin/verify/lean/differential.py
  - src/merlin/verify/lean/mlir_execute.py
  - src/merlin/verify/lean/mlir_map_contract.py
  - merlin/tests/ir/test_lean_differential.py
  - merlin/tests/ir/test_lean_map.py
  - merlin/tests/ir/test_lean_typed.py
  - merlin/tests/ir/test_lean_motion.py
  - merlin/lean/MerlinVerify/Rules/Motion.lean
  - src/merlin/verify/lean/check.py
---

# Run Lean semantics differential tests

The fidelity suite compares the actual Lean denotation used by the soundness theorems with
upstream MLIR lowering and native execution. It provides empirical evidence about the model's
agreement with MLIR. Coverage is the explicit corpus in `test_lean_differential.py`; passing it
does not establish fidelity for every MLIR operation, attribute, input, or target.

## Prerequisites and invocation

Use the project Python environment with pytest and the verification dependencies, the pinned
Lean toolchain and cached dependencies, the built `merlin-lean` printer, a compiler Python
environment providing `torch_mlir.execution_engine`, and upstream `libmlir_c_runner_utils.so`.
The runner library comes from an LLVM/MLIR build or installation; Merlin's runtime implementation
is deliberately not used as the execution oracle.

```sh
export PATH="$HOME/.elan/bin:$PATH"
export MERLIN_M2M_PYTHON=/path/to/compiler-environment/bin/python
export MERLIN_MLIR_RUNNER_UTILS=/path/to/llvm-build/lib/libmlir_c_runner_utils.so
MERLIN_LEAN_REQUIRED=1 .venv/bin/python -m pytest \
  merlin/tests/ir/test_lean_differential.py -q -rx
```

Without the interpreter override, tests use `toolchain.compiler_python()`. Without the runtime
override, they look under `toolchain.llvm_install()/lib/`. Missing prerequisites skip ordinary
runs and fail required qualification runs. An installed tool that cannot import, compile, or
execute fails the test. The fixture builds only the stable Lean decoder and evaluator dependencies.
It does not change or rebuild the shared checker driver or proof rules.

For retained diagnostics, give pytest `--basetemp=out/build/lean-differential/pytest` and
`-o tmp_path_retention_policy=all`; the repository normally deletes successful tests' temporary
directories. Completed comparisons record the original MLIR, exported JSON, round-trip text,
runtime inputs, native LLVM-dialect IR, and both evaluators' results. Earlier failures retain the
available inputs and intermediate files. The native result identifies its source hash, MLIR package version,
pass pipeline, and runner-library hash. It also records an executable map capability probe and
hashes of the compiler interpreter, MLIR binding, and aggregate C API library. A package version
is not an exact LLVM revision; a separate source checkout does not identify the installed binary.
The comparison evidence records the Lean adapter, semantics files, specification lock, and NaN
comparison policy. Retain these files when reporting a mismatch.

## What is exercised

- Integer arithmetic, bit operations, signed/unsigned comparisons, select, division, remainder,
  floor/ceiling division, shifts, casts, and unviolated overflow promises. Widths include 1, 4, 8,
  13, 32 and 64 bits; the 1- and 4-bit arithmetic domains are exhaustive. Larger widths combine
  edge values and reproducible random inputs. Scalar index casts/arithmetic run with both 32-
  and 64-bit index lowering.
- Static tensor reshapes, strided and rank-reduced slices, slice insertion, concat, element
  extraction/insertion, splat, from-elements, dense constants, scalar/tensor select, elementwise
  arithmetic, rank-zero tensors, and empty dimensions.
- Fill, mixed-width matmul, batch matmul, quantized matmul, copy, three-dimensional transpose,
  broadcast, reduce, and generic affine indexing with `linalg.index`. Initial accumulators are
  populated with runtime data. Named operations are also generalized by MLIR and checked in
  their generic form.
- Map explicit and shorthand syntax, zero/one/two inputs, captured values, rank-zero and empty
  tensors, integer/f32 elements, reads of initialized destinations, and complete overwrites.
  Both map and MLIR's generalized form run through Lean and native execution.
- Host f32/f64 arithmetic, all float comparison predicates, format and integer conversions,
  bitcasts, sqrt, abs, rounding operations (including ties-to-even), explicit fma, signed zero,
  subnormals, infinities, NaNs, and order-sensitive floating-point reductions.
- Internal calls and a passing `cf.assert`.

Inputs enter the JIT through runtime memrefs. Only upstream passes run on the native side; no
Merlin preprocessing or interpreter computes the expected answer. Elementwise tensor operations
are lowered with upstream `convert-elementwise-to-linalg`; named structured operations are
generalized and converted to loops. Fusion, canonicalization, CSE, fast-math, and JIT optimization
are disabled in this oracle pipeline so reduction order and individual operations remain visible.

Results are observed as integer bit patterns, including tensor shape and element count. Float
arithmetic cases explicitly allow different NaN payloads/signs when both results are NaN; finite
results, infinities and signed zero remain exact. Data movement and bitcast cases compare NaN
payloads bit for bit. A seeded add-to-subtract fault demonstrates that a wrong model result is
detected.

## Map contract and initialization

LLVM [f5e175f](https://github.com/llvm/llvm-project/commit/f5e175f06dacf6751f27f7a4bd9e9a3489e95a5e)
added the output element block argument in October 2025. Lean previously implemented the older
inputs-only convention. The reviewed correction passes inputs followed by the output element,
including when unused; `SPEC_LOCK` records this semantic change. The old strict expected failure
has been replaced by passing cases. The native oracle checks the installed contract before
lowering: explicit forms and upstream generalization must preserve the extra argument and its
reads, legacy explicit arity must be rejected, and shorthand must add an unused destination argument.
An unexpected convention fails; there is no automatic legacy fallback.

Lean now proves map progress and initialized results using the generic-loop proofs. An initialized
destination may be read. A dirty destination is accepted only when the body produces initialized
elements without relying on its old contents, and the iteration box covers the whole destination.
Malformed body signatures and unsupported attributes are refused. A negative differential test
ensures a map returning poison never reaches native execution.

The separate upstream [bufferization issue](https://github.com/llvm/llvm-project/issues/213241)
can omit initialization when directly bufferizing a map that reads its destination. The oracle
and production scalar/parallel defaults generalize before bufferization. A regression checks that
order and that an observable destination read retains its initializer copy in bufferized IR.
This regression does not prove arbitrary downstream pipelines correct.

Map-to-generic **rewrite certification is available on an explicit typed domain** through
`merlin.verify.lean.check.check_typed_stage(stage, m2m_python=compiler, index_bits=64)`.
`Check/TypedRewrite.lean` derives runtime facts from checked source operations and carries them
through intermediate SSA bindings. `Rules/MapGeneralize.lean` uses the justified destination rank
to prove the map expansion. `Check/TypedC1.lean` then checks the actual target with the existing
universal C1 rules. Export retains the original map; normalization is an internal proved step.
The original universal `check_stage` still refuses this type-dependent rewrite.

Typed receipts name `MerlinVerify.Check.TypedModule.check_sound` and record their domain and
index width. `compose_chain` replays typed and universal boundaries together, requires every
boundary to preserve the capture's declared input types/shapes, and checks that typed widths
match the source support/definedness receipts. The resulting typed chain names
`chain_typed_refines` and `chain_typed_defined_refines`. An initialized source still needs its
independent definedness proof; rewrite certification alone permits poison and UB.

Qualification is in `test_lean_typed.py`: actual generalization, rank-zero/empty tensors, captures,
intermediate reshape/splat results, escaping outputs, both index widths, mixed chains, corrupted
rewrites, stale bytes, and tampered domains/theorem labels. The normalizer handles top-level
one-operation expansions that preserve bindings and region closures. Adjacent closed-expression
motion has its own proof in `Rules/Motion.lean`: both expressions must be pure, operand-free,
region-free, single-result operations with distinct source bindings and fresh target bindings.
This permits upstream hoisting a captured constant across static `tensor.empty`. Qualification
includes the typed defined chain, native execution before and after, an independent integer
reference, and rejection of a changed moved constant. `test_lean_motion.py` checks escaping values,
integer widths, floating constants without arithmetic laws, and conservative refusals. Operand-taking
motion, non-adjacent hoisting and nested rewriting still require additional work.

```sh
MERLIN_LEAN_REQUIRED=1 .venv/bin/python -m pytest \
  merlin/tests/ir/test_lean_typed.py merlin/tests/ir/test_lean_motion.py -q
```

The lower-level driver also exposes `merlin-lean check-module-typed before.json after.json 64`.
Differential agreement remains empirical evidence about the semantics; the typed rewrite and
chain theorems supply the formal transformation claims.

## Limits

The corpus chooses defined inputs independently: divisors are nonzero, signed division overflow
is excluded, shift counts are in range, tensor indices are in bounds, and float-to-integer inputs
are representable. A Lean result containing poison, UB or `stuck` cannot count as a successful
comparison and does not reach native execution. Negative qualification tests exercise this rule.
`remsi(INT_MIN, -1)` is excluded because of the documented MLIR/LLVM discrepancy in ledger A2.
Executing undefined native behavior cannot establish what poison or UB means.

The host `FloatModel` lives in the test-only `Differential.lean` adapter, outside the verified
namespace. It adds no axioms or changes to the soundness claims. It is not a proved IEEE instance.
Unsupported operations, non-default rounding, fast-math, f16/bf16, and float-to-integer results
wider than 32 bits are refused. Native memref copies with 32-bit descriptors are also refused
because upstream CRunnerUtils uses 64-bit descriptor indices. Dynamic/encoded tensors, external
calls, failing assertions, exhaustive floating-point testing, and non-host execution remain outside
this corpus.
