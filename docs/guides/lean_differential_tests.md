---
title: Run Lean semantics differential tests
kind: guide
status: current
owner: verification
last_verified: 2026-10-17
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
  - src/merlin/verify/lean/cli.py
  - src/merlin/verify/lean/handoff.py
  - merlin/tests/ir/test_lean_handoff.py
  - merlin/tests/ir/test_lean_cli.py
  - merlin/lean/MerlinVerify/Check/BindingFacts.lean
  - merlin/lean/MerlinVerify/Check/UniformFacts.lean
  - merlin/lean/MerlinVerify/Rules/CopyGeneralize.lean
  - merlin/tests/ir/test_lean_copy_generalize.py
  - merlin/lean/MerlinVerify/Rules/LayoutGeneralize.lean
  - merlin/tests/ir/test_lean_layout_generalize.py
  - merlin/tests/ir/test_lean_integer_fold.py
  - merlin/tests/ir/test_lean_integer_fold_native.py
  - merlin/tests/ir/test_lean_reshape_compose.py
  - merlin/tests/ir/test_lean_reshape_differential.py
  - merlin/tests/ir/test_lean_splat_checker.py
  - merlin/lean/MerlinVerify/Rules/IntegerFold.lean
  - merlin/lean/MerlinVerify/Rules/ReshapeCompose.lean
  - merlin/lean/MerlinVerify/Check/SplatC1.lean
  - merlin/lean/MerlinVerify/Check/DefinedRules.lean
  - merlin/tests/ir/test_lean_float_definedness.py
  - merlin/tests/ir/test_lean_nested_definedness.py
  - merlin/lean/MerlinVerify/Rules/IntegerIdentity.lean
  - merlin/lean/MerlinVerify/Rules/ReshapeMixed.lean
  - merlin/lean/MerlinVerify/Check/SplatNormalizer.lean
  - merlin/lean/MerlinVerify/Check/SplatNestedC1.lean
  - merlin/tests/ir/test_lean_integer_identity.py
  - merlin/tests/ir/test_lean_integer_identity_native.py
  - merlin/tests/ir/test_lean_reshape_mixed.py
  - merlin/tests/ir/test_lean_reshape_mixed_differential.py
  - merlin/tests/ir/test_lean_splat_nested.py
  - merlin/tests/ir/test_lean_splat_nested_differential.py
  - merlin/lean/MerlinVerify/Check/TypedNormalizeC1.lean
  - merlin/lean/MerlinVerify/Check/SharedNormalize.lean
  - merlin/lean/MerlinVerify/Check/DefinedControl.lean
  - merlin/tests/ir/test_lean_control_definedness.py
  - merlin/tests/ir/test_lean_control_differential.py
  - merlin/tests/ir/test_lean_integer_cast.py
  - merlin/tests/ir/test_lean_integer_cast_native.py
  - merlin/tests/ir/test_lean_reshape_typed.py
  - merlin/tests/ir/test_lean_reshape_typed_differential.py
  - merlin/tests/ir/test_lean_integer_body.py
  - merlin/tests/ir/test_lean_integer_body_differential.py
  - merlin/tests/ir/test_lean_shared_normalize.py
  - merlin/tests/ir/test_lean_shared_normalize_differential.py
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

## Check recorded compiler output

`merlin-verify lean` exposes the receipt checkers without a Python harness. Select the same
authoritative compiler interpreter that the build uses. For a recorded stage directory containing
`before.mlir`, `after.mlir`, and `meta.json`:

```sh
merlin-verify lean typed-stage out/artifacts/verification/example/stage \
  --compiler-python "$MERLIN_M2M_PYTHON" --index-bits 64
merlin-verify lean support out/artifacts/verification/example/stage/before.mlir \
  --function forward --work out/artifacts/verification/example/support \
  --compiler-python "$MERLIN_M2M_PYTHON"
merlin-verify lean definedness out/artifacts/verification/example/stage/before.mlir \
  --function forward --work out/artifacts/verification/example/definedness \
  --compiler-python "$MERLIN_M2M_PYTHON"
merlin-verify lean chain out/artifacts/verification/example/stage \
  --function forward --support out/artifacts/verification/example/support/receipt.json \
  --definedness out/artifacts/verification/example/definedness/receipt.json \
  --output out/artifacts/verification/example/chain.json \
  --compiler-python "$MERLIN_M2M_PYTHON"
```

`stage` selects the universal checker; `typed-stage` records a typed domain and index width.
Every command prints JSON, with exit 0 only for an overall verified receipt. Exit 2 means the
check did not establish that claim. Chain replay checks the saved proof premises and current
artifact bytes. A chain with refinement alone, without source definedness, does not exit 0.
Supply multiple stage directories in execution order and `--handoff` when qualifying the exact
recorded OOT handoff. Unsupported transformations remain explicit abstentions.

`merlin-verify lean differential --help` describes a separate native comparison command. It
takes a program, an explicit function, and JSON samples shaped `[sample][argument][element]`;
every element is an integer bit pattern, including floats. Supply the built checker, Lake,
compiler interpreter, upstream runner library, and a new work directory. The required
`--assume-defined-inputs` records the caller's independent defined-execution obligation; the
Lean model alone cannot establish that native execution is safe. Completed comparisons report
`matched_samples`, retain both evaluators' evidence, and never issue a verification receipt.
Failed comparisons and unavailable tools exit 2. Existing comparison directories are refused
so a failed rerun cannot reuse an earlier success.

## Require verification before consuming shared-pass output

The Python lowering API has an explicit mode that stops compilation unless preprocessing has
a replayed, defined Lean chain. Select the same authoritative MLIR interpreter as the compiler
and supply the semantic index width explicitly:

```python
from pathlib import Path
from merlin.llvmlower.lower import lower_model
from merlin.verify.lean.handoff import Options

result = lower_model(
    source_mlir,
    workdir,
    index_bits=64,
    shared_pass_verification=Options("forward", 64, Path(compiler_python)),
)
```

`lower_model_file` accepts the same option. For another consumer, including an OOT driver,
`merlin.verify.lean.handoff.run(source, output_path, options, preprocess)` runs an explicitly
supplied preprocessing function and returns its verified text, statistics and evidence path.
Consume that returned text. The recorder uses a fresh context-local capture; each recorded
boundary, including unchanged ones, is checked. Source support, source definedness, chain replay,
input/output byte identity and capture completeness are all required. A failed check raises
`VerificationError` before downstream lowering starts. Reports are retained beside the handoff
under `lean-shared-passes/<attempt>/`.

This mode currently wraps `lower_model`'s shared preprocessing. It does not include earlier
model capture or preparation, native upstream passes, OOT implementation, LLVM or runtime.
The report states the typed, initialized-input domain and ledger assumptions. Other drivers
must explicitly call the gate at their own tensor handoff; ordinary ungated compilation remains
available. Missing tools, unsupported rewrites and undefined source results stop a gated build.

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

## Closed integer folding and reshape composition

`test_lean_integer_fold.py` certifies actual upstream constant canonicalization through universal
C1 rules. The checker replays closed scalar fixed-width integer definitions using the actual Lean
semantics and compares the final value with the actual target constant. It checks flags, widths,
operand bindings, source/target shadowing and escaping intermediates. The rule excludes floating arithmetic and index-typed definitions. Native tests in the separate
`test_lean_integer_fold_native.py` use independent signed/unsigned/modular references, cast edges
and a verified wrong-constant control. Undefined divisions, signed division overflow and invalid shift counts are excluded
independently before native execution. This is closed-fragment folding, not general constant-DAG
or input-dependent cast simplification.

`Rules/IntegerIdentity.lean` separately certifies input-dependent scalar add/subtract-zero,
multiply-one, bitwise AND-all-ones, and OR/XOR-zero rewrites. The adjacent actual scalar constant
establishes runtime type evidence; poison is preserved. The checker removes both operations or
retains an alpha-equivalent constant when it escapes. Reverse subtraction, flags and tensor
identities remain refused. Actual-pass, mutation, strict-chain and native tests cover these rules;
native inputs are exhaustive at i1/i8 and include edges and seeded samples at i13/i32/i64.

`test_lean_reshape_compose.py` checks two collapses or two expansions against upstream's composed
reshape, including group algebra, exact output shapes/types, escaping inputs, verified mutations
and defined receipt replay. The native module compares both sides with the unchanged row-major
input bits, including floating NaN payloads. These universal sites refuse escaping intermediate
reshapes and bare-input cancellation; the typed provenance route below extends composition coverage. All-unit rank-zero collapse/expand with empty reassociation
is legal upstream but currently stuck in Lean: two dedicated native tests retain that discrepancy,
while binding tests require source support and coverage-chain abstention. A passing conditional
boundary check alone must never be reported as coverage for these unsupported inputs.

`Rules/ReshapeMixed.lean` additionally certifies expansion followed by collapse against one
expansion or collapse. Successful expansion determines the original runtime shape from actual
reassociation products; target legality is recomputed from that shape. This universal mixed rule
refuses collapse-first pairs and escaping intermediates; the typed provenance route below supplies
their runtime premises while retaining original bindings. Bare-input cancellation remains refused.
The mixed suites require strict source-defined chains and compare row-major bits independently,
including empty integer tensors and exact f32/f64 payloads. BF16 has binding coverage only.

Native-only modules are explicitly excluded from binding-only Lean CI. Missing native tools fail
required qualification runs; no interpreter substitutes for native execution. Checker builds and
specification locks must be fixed before receipt tests start. Independent workers use separate
checkouts and binaries, then the coordinator replays integration against one final checker.

## Transpose, broadcast and reduction rules

The typed checker also certifies actual transpose, broadcast and variadic reduction generalization.
`Rules/LayoutGeneralize.lean` derives indexing maps and iterator kinds from the actual source
attributes and checked SSA types, preserves the scalar body and captures, and checks the actual
generic header. Source support and source definedness are independent obligations. Transpose and
broadcast may fully overwrite uninitialized destinations; reductions that read accumulators need
initialized destinations. The integer reduction definedness rule supports multiple inputs/results
and empty iteration spaces. Floating reductions retain their order in the refinement proof.
The definedness leaf rulebook proves totality of the nine default same-format scalar float binary
operations on initialized inputs, under every float interpretation. This permits strict defined
chains for supported floating reductions; it assumes no arithmetic identities or accuracy laws.
Fast-math promises, float-to-integer range checks and arithmetic on poisoned inputs remain refused.
The f32/f64 native reduction cases require source and target definedness and compare with the
independent order-preserving reference. Binding-only checks also cover f16/bf16; those formats have
no new native arithmetic qualification.

Generic source-definedness also composes through nested generic bodies. The body checker is a
proved rulebook parameter; initialized destinations select the accumulator rule, while dirty
destinations select complete-overwrite checking. A budget derived from actual region nesting
ensures finite checking without repeatedly exploring both strategies on a rejected subtree.
Rank-zero `tensor.extract` is total on initialized tensors because it has no index operands.
Positive-rank extraction still needs a separate value-dependent bounds proof and remains refused.
These are model-relative totality proofs, separately from native lowering qualification.

`test_lean_layout_generalize.py` checks both index widths, actual pass output, receipt replay,
stale source rejection, valid MLIR mutations, poisoned inputs, captured values and composition
through intermediate fill/broadcast/transpose/reduce results. The native suite compares both sides
with an independent coordinate-based reference. Data movement includes f32/f64 NaN payloads,
signed zeros, infinities and subnormals. Reduction references use modular integer accumulation or
explicit per-add floating rounding, including cancellation-sensitive inputs. BF16 movement has
binding/proof coverage only; no native BF16 claim is made.

Two boundaries remain explicit: the upstream greedy pass folds some identity operations away,
which requires a separate rewrite rule; the typed checker currently abstains on those outputs.
Rank-zero transpose is not qualified: the current installed binding crashes during verification,
and an assertion-enabled upstream `mlir-opt` reproduces the empty-permutation-map assertion in
`TransposeOp::getIndexingMaps`. Rank-zero broadcasts and scalar reduction results are separate
cases. Generated failure evidence belongs under `out/artifacts/verification/shared-passes/`.

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

The same typed checker certifies `linalg.fill` generalization through
`Rules/FillGeneralize.lean`. It checks the scalar broadcast map, destination identity map,
parallel iterators and operand segments, retaining the actual source body. Fill and map families
compose on one checked SSA domain, including intermediate reshapes. The fill regression suite
checks both index widths, rank-zero/empty tensors, initialized and dirty destinations, integer
and floating element types, defined chains, and independently corrupted inputs, yields and returns.
The native differential corpus executes original and generalized fills on integer, f32 and f64
inputs. This certifies the named-to-generic boundary; it does not prove later bufferization.

`Rules/CopyGeneralize.lean` adds `linalg.copy` on static, unencoded tensor destinations. It
checks both identity maps, parallel iterators, operand segments and supported attributes while
retaining the actual body, casts and SSA bindings. The copy-definedness rule proves the original
source succeeds using a separate success-transfer theorem. A fully overwritten destination may
start uninitialized; a poisoned input still prevents a defined-source claim. The regression suite
checks actual upstream generalization at both index widths, rank-zero/empty tensors, integer and
floating storage types, widening integer copies, intermediate fill/reshape results, escaping inputs,
defined receipt chains and stale output. Numerically wrong mutations first pass MLIR verification
and must then be refused by Lean. Native tests compare before and after against Lean and an
independent input-bit/cast reference, including f32/f64 NaN payloads and signed zero. These tests
cover the named-to-generic boundary; downstream lowering remains outside its formal claim.

`test_lean_splat_inputs.py` executes the default uniform-input compaction pass and compares both
programs through the Lean denotation and native MLIR. It covers multiple compacted inputs, a full
splat result that must remain live, affine indexing, empty/rank-zero tensors and f32/f64 bit copies.
Integer cases also use an independent modular reference; a corrupted scalar payload must produce
a different native result. `Rules/SplatInputs.lean` supplies the general semantic proof under
checked layouts and an actual shared scalar payload. `Check/UniformFacts.lean` proves the actual
splat-producer facts and invalidates them when either dependency is rebound. Source and target
payloads can be coupled only through corresponding actual scalar bindings. `Check/SplatC1.lean`
now checks the actual whole-function boundary by independently proving exact normalization of
both checked programs. A removed producer must be dead in the canonical continuation, including
nested captures. Escaping full tensors remain live. The bare-scalar generic normal form is internal;
exports and the native oracle still use only actual compiler IR. The binding suite checks both index
widths, actual pass output, defined chain replay, stale receipts, captures, repeated generic uses and
MLIR-verified mutations. `Check/SplatNested.lean` lifts exact normalization through supported
single-block generic bodies using checked loop arguments and source safety. Region bindings mask
captured uniform facts before entering the body. `SplatNormalizer` supplies one shared comparison
checker for top-level and nested normalizers; both actual programs are checked independently.
The default nesting budget derives from actual function bodies. Every budget has the same
soundness proof; exhaustion can only cause refusal. Nested tests cover captures, stale SSA facts,
three/four-level bodies, mutations and native results with independently initialized destinations.
Rank-zero extraction permits strict source-defined chains; positive-rank extraction still needs
its separate bounds obligation. Model-scale performance remains unmeasured.

The nested suite also retains S4, a native bufferization discrepancy in the checksum-pinned
torch-mlir 20260531.828 toolchain. A minimal no-splat tensor program reuses a captured initialized
inner destination: Lean and the independent reference return `[2,3]`, but native execution returns
`[2,4]`. Standalone one-shot-bufferization marks that destination in-place and repeatedly writes
the same buffer. A reduction variant also fails; an explicit `scf.for` control inserts a copy and
gets the expected result. This conflicts with the documented
[immutable tensor and destination-copy contract](https://mlir.llvm.org/docs/Bufferization/#destination-passing-style).
Captured-destination examples are outside native-fidelity qualification. Dedicated known-gap
checks retain the mismatch separately from matched cases; identical wrong native results before
and after compaction are not fidelity evidence. Fresh per-iteration destinations supply the
matched nested corpus. No compiler pass, oracle or Lean semantics is changed to hide this finding.

Typed receipts name `MerlinVerify.Check.TypedModule.check_sound` and record their domain and
index width. `compose_chain` replays typed and universal boundaries together, requires every
boundary to preserve the capture's declared input types/shapes, and checks that typed widths
match the source support/definedness receipts. The resulting typed chain names
`chain_typed_refines` and `chain_typed_defined_refines`. An initialized source still needs its
independent definedness proof; rewrite certification alone permits poison and UB.

Qualification is in `test_lean_typed.py`: actual generalization, rank-zero/empty tensors, captures,
intermediate reshape/splat results, escaping outputs, both index widths, mixed chains, corrupted
rewrites, stale bytes, and tampered domains/theorem labels. The original typed expansion route handles
top-level one-operation expansions that preserve bindings and region closures. Adjacent closed-expression
motion has its own proof in `Rules/Motion.lean`: both expressions must be pure, operand-free,
region-free, single-result operations with distinct source bindings and fresh target bindings.
This permits upstream hoisting a captured constant across static `tensor.empty`. Qualification
includes the typed defined chain, native execution before and after, an independent integer
reference, and rejection of a changed moved constant. `test_lean_motion.py` checks escaping values,
integer widths, floating constants without arithmetic laws, and conservative refusals. Operand-taking
motion, non-adjacent hoisting and nested motion still require additional work.

```sh
MERLIN_LEAN_REQUIRED=1 .venv/bin/python -m pytest \
  merlin/tests/ir/test_lean_typed.py merlin/tests/ir/test_lean_motion.py -q
```

The lower-level driver also exposes `merlin-lean check-module-typed before.json after.json 64`.
Differential agreement remains empirical evidence about the semantics; the typed rewrite and
chain theorems supply the formal transformation claims.

`test_lean_control_definedness.py` qualifies integer comparison and selection source domains,
including all ten integer predicates, both index widths, scalar and static tensor branches,
initialized tensor conditions, generic bodies, and refusal of poison or unknown semantic
attributes. Its actual canonicalization chain removes a neutral integer addition while retaining
input-dependent comparison and selection. `test_lean_control_differential.py` compares these
operations with native MLIR and independent signed/unsigned integer and input-bit references.
The native corpus exhausts i1/i4 comparison pairs, samples wider widths deterministically, and
checks bit-preserving f32/f64 selection including NaN payloads. A swapped-branch target is both
rejected by the checker and observably wrong in native execution. These tests qualify this bounded
corpus; they do not prove a general IEEE instance or source capture fidelity.

The input-dependent cast suite (`test_lean_integer_cast.py` and its `_native` companion) checks
four unflagged scalar integer composition families, including computed SSA operands, retained
intermediates and longer compatible chains. Poison and malformed runtime widths are checked in
Lean; native cases use initialized inputs and an independent modulo/sign-extension reference.
Index/tensor casts, extension/truncation alias roundtrips and signed-then-unsigned flattening
remain outside this family. Native widths are at most 64 bits; binding acceptance additionally
exercises wider integer types.

`test_lean_reshape_typed.py` and its `_differential` companion exercise source-provenance mixed
reshape composition, escaping and captured intermediates, zero/unit extents, alternative legal
unit reassociations, and invalid target headers/operands. Native f32/f64 cases preserve input bits,
including signed zero and NaN payloads. Bare-input cancellation, S3 transitions and arbitrary
transitive chains remain outside coverage. The rank-two extraction capture has typed refinement,
support and native evidence, while its separate source-definedness obligation still abstains.

`test_lean_integer_body.py` and its `_differential` companion qualify exact unflagged integer
commutation through generic bodies and nested reductions with fresh destinations. Wrong captured
constants and wrong yields have observable native controls. Numeric SSA ordering is conservative
under independent renaming; floating-point algebra and iteration reordering remain refused.

`test_lean_shared_normalize.py` and its `_differential` companion combine these families in actual
pass output, including Merlin splat compaction followed by upstream canonicalization. Both actual
programs are independently type checked and exactly normalized before the C1 comparison. The
suite checks strict source-defined chains at both index widths, zero extents, escaping tensors,
stale bytes, and valid MLIR corruptions in each family. Native execution is compared with independent
bit arithmetic and flattened tensor references, including the intermediate compacted program.

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
