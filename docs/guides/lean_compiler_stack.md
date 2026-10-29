---
title: Lean verification through the compiler stack
kind: guide
status: current
owner: verification
last_verified: 2026-10-29
related: [lean_pass_gate, lean_loop_transform_gate, lean_buffer_reuse_gate, lean_configuration_elision_gate, lean_configuration_hoist_gate, verified_compiler_tooling, merlin_lean_verification, lean_pass_integration_completion]
code_refs:
  - src/merlin/verify/pass_validation.py
  - src/merlin/verify/compiler_transforms.py
  - src/merlin/verify/compiler_loop_intake.py
  - src/merlin/verify/compiler_macc.py
  - src/merlin/verify/compiler_qdq.py
  - src/merlin/verify/speculative_gate.py
  - src/merlin/llvmlower/lower.py
  - src/merlin/compile/command.py
  - merlin/tests/ir/test_compiler_formal_transforms.py
---

# Lean verification through the compiler stack

Current qualification and remaining limits are in
the [integration handoff](../../out/artifacts/handoff/pass-verif-integration-20261028/README.md).
That handoff distinguishes retained evidence from checks on the current source.
The [completion plan](../design/lean_pass_integration_completion.md) defines the
remaining work to cover Merlin pass proofs and compiler integration through
output IR, leaving only explicit semantic contracts for human review.

Select transformations to change the compiler's output. Select validation to
require a proof of their actual input/output boundaries. Both are opt-in.
`--validate-passes all` selects DCE, CSE, loop and buffer checks. The explicit
`bf16_f32acc_draft_contract`, proposed-model `lower_quant_ext` and ISA-specific
`configuration_write_elision` / `configuration_loop_hoisting` checks stay outside `all`.
Selecting checks does not enable transformations. A selected, changing boundary that cannot be proved
stops compilation. An unchanged boundary needs no proof job.

```mermaid
flowchart TD
  A[Workload or explicit MLIR] --> B[lower_model input]
  B --> N[Optional bf16/f16 MACC under draft f32-accumulation contract]
  N --> E
  B --> C[Optional source SCF transforms: unroll and ordered tile]
  B --> D[Optional private-storage chain: normalize, reuse, ordered tile]
  C --> E[xDSL preprocessing: dead tensor operations]
  D --> E
  B --> E
  E --> F[Native leading CSE: tensor gate when selected]
  F --> T[Tensor route: bufferization and SCF/LLVM lowering]
  F --> M[Explicit memory route: direct SCF/memref lowering]
  T --> G[LLVM/object generation]
  M --> G
  C -. exact captures .-> P[Owned Lean gate workers]
  D -. exact captures .-> P
  E -. exact captures .-> P
  F -. exact captures .-> P
  N -. exact captures .-> P
  P --> H[All requested checks accepted]
  G --> H
  H --> I[Return compiler products]
```

The worker builds/checks proofs, audits their axioms and binds the result to the
actual captured bytes. Compilation may proceed concurrently, but its products
are returned only after the enclosing gate accepts. Failure quarantines emitted
LLVM, object and shared-library products. Each job retains `request.json`,
`verdict.json`, `worker.log` and its proof evidence.

| Registered check | Where it runs | Small example | Established relation |
| --- | --- | --- | --- |
| `prune_dead_pure_tensor_ops` | xDSL preprocessing, before native lowering | Erase an unused `tensor.empty` | Tensor-stage strict refinement on declared input types, with source support checked |
| `cse` | The build runner's leading tensor CSE | Two identical pure expressions share one result | Same tensor-stage strict refinement |
| `ordered_loop_unrolling` | Optional source SCF stage at `lower_model` entry | Two consecutive iterations become two ordered lanes | Same returned values, modeled errors and terminal world in the scalar/SCF interpretation |
| `ordered_loop_segmentation` | Optional source SCF stage; also the final step of the private-buffer chain | Visit `0..10` as ordered groups of three, retaining the short last group | Scalar route: same return/error/world relation; memory route: the complete chain's successful-observation relation |
| `buffer_reuse` | Optional source normalization→reuse chain, with optional initializer segmentation | Two initialized, nonoverlapping private lifetimes share storage | Same possible successful public observations, under the explicitly selected allocator model |
| `bf16_f32acc_draft_contract` (explicit only) | Actual `lower_bf16_matmul_f32acc`, including its optional compiler-entry invocation | Extend half operands, accumulate in f32, truncate once | Conformance to the draft numerical reference; numerical approval remains abstracted |
| `lower_quant_ext` (explicit only) | Actual xDSL Q/DQ pass or complete textual preprocessing | Expand Q/DQ into the specified ordered scalar operations | Strict typed refinement under the proposed Q/DQ source model; optional fresh original-source correspondence |
| `configuration_loop_hoisting` (explicit only) | Actual selected `hoist_loop_configuration` before its ordinary Kernel consumer | Move an invariant total setter before a positive counted loop | Whole partial outcome equality, including failures and terminal state, under indexed instruction/frame premises |
| `configuration_write_elision` (explicit only) | Original selected instruction IR or straight-line Kernel, before its ordinary consumer | Delete a repeated pure setter while retaining work and failure effects | Complete entry error/terminal-state preservation under the original ISA/effect model; no independent implementation approval |

These are separate relations. A buffer success proof does not establish the
loop checker's error/world equality or finite-memory allocator behavior. No row
proves the downstream LLVM/object implementation or hardware execution.

## Tensor DCE and CSE

```sh
merlin-compile ... --validate-passes prune_dead_pure_tensor_ops,cse \
  --lean-sdk-bin /path/to/lean-4.35.0/bin
```

`pass_validation.observe` captures preprocessing pruning; `leading_cse` captures
the actual native runner's CSE boundary. `gate_checks.tensor_refinement` requires
`MerlinVerify.Check.TypedModule.check_sound`, source support and composed strict
refinement. The raw MLIR is exported through `torch_mlir` and round-tripped
through Lean. See [the gate guide](lean_pass_gate.md) for prerequisites.

## Ordered unrolling and tiling

For source SCF loops, select either transformation or both:

```sh
merlin-compile ... --loop-unroll-factor 2 --loop-tile-size 3 \
  --validate-passes ordered_loop_unrolling,ordered_loop_segmentation \
  --lean-sdk-bin /path/to/lean-4.35.0/bin
```

The Python compiler interface accepts the same explicit selection:

```python
from pathlib import Path
from merlin.llvmlower.lower import lower_model
from merlin.verify.compiler_transforms import Options

# Set MERLIN_VALIDATE_PASSES and MERLIN_LEAN_SDK_BIN, or pass an explicit
# PassValidation object. Transformation options alone do not request proofs.
result = lower_model(Path("source.mlir").read_text(), "out/build/loop-example",
                     targets=(), formal_transforms=Options(unroll_factor=2, tile_size=3))
```

For example, a loop that starts at 3, stops before 31 and advances by 2 has
fourteen iterations. Unroll factor two produces seven ordered groups. Tile size
three groups those into three segments of sizes three, three and one. The
carried accumulator flows through every lane and segment in the original order.

A complete source for that example is:

```mlir
builtin.module {
  func.func @forward(%x: i32, %y: i32) -> i32 {
    %lb = arith.constant 3 : index
    %ub = arith.constant 31 : index
    %st = arith.constant 2 : index
    %r = scf.for %i = %lb to %ub step %st iter_args(%c = %x) -> (i32) {
      %next = arith.addi %c, %y : i32
      scf.yield %next : i32
    }
    func.return %r : i32
  }
}
```

For inputs `7, -3`, the original and transformed versions both return `-35`.
Arithmetic wraps at the declared integer width; the Lean relation is quantified
over inputs, while native execution tests provide separate finite witnesses.

`compiler_transforms.apply` calls the existing Python transforms and captures
their complete stages. `compiler_loop_intake.prepare` independently projects
those buffers into the existing scalar/SCF model. `loop_transform_gate.check`
builds a fresh project closure and query, audits all generated theorem roots and
validates the live completion. The all-entry adjacent roots are
`Merlin.GraphGateProfileAdjacent.complete2` for unrolling and `complete3` for
segmentation. Both actual boundaries share one proof job when selected together.
An individual unroll job uses an auxiliary segmented program only to construct
the existing proof scaffold; its adjacent theorem covers the actual unrolled
compiler output, and the auxiliary output is never delivered.

`loop_transform_gate.replay(report, lean=..., destination=..., consumer=...)`
reconstructs these compiler profiles, builds a new private proof closure and
delivers the exact target through the new live completion. It does not require
the historical producer checkout for current compiler captures. A separate
process can replay with no incoming completion authority; saved JSON is input
to a new check. The standalone pinned-producer workflow still takes its explicit
`reference` argument.

The source-SCF route supports multiple changed top-level sibling loops per function,
scalar entry arguments and word carries without separate fixed arity caps
(including zero carries), static nonnegative bounds,
positive stride and index64. Scalar unroll factors and tile sizes are integers
at least two. Unrolling supports empty loops, complete inlining, no-tail bulk
and bulk-plus-remainder modes. Different function definitions can use different
modes; the final common-registry theorem restores every actual defined entry,
including supported unchanged dynamic-loop functions. All outer unroll modes
and standalone tiling preserve supported nested scalar loop bodies, including
dynamic inner controls and modeled external effects. The ordinary compiler
does not yet select inner-depth factors.
Tiling supports short final segments. Complete syntax,
expression/proof expansion, query-size, signed-index and runtime budgets still
apply. External calls retain the modeled state
and failure semantics. Actual prefix/suffix contexts, including partial external
calls, are composed around each selected rewrite. Local wrapper control equations
are derived from the actual prefix; they are absent from the final whole-entry
theorem. Combined unroll→tiling checks join the actual original, intermediate and
final buffers in one canonical job. Selecting inner-depth rewrites remains active
engineering. Untouched context must also fit the scalar/SCF projection.
The gate does not insert a loop-lowering pass to turn
arbitrary tensor programs into supported source SCF.

Multiple changed function definitions are checked together. Exact unchanged
scalar stages are retained in the captures and omitted from positive rewrite
coverage; the proof covers the remaining actual adjacent changing stages.
Private compiled objects are frozen before auditing and checked for byte or
tree-membership changes again before delivery.

## Private buffer reuse

```sh
merlin-compile ... --loop-tile-size 2 \
  --buffer-reuse-allocator total_private_unobservable \
  --validate-passes buffer_reuse,ordered_loop_segmentation \
  --lean-sdk-bin /path/to/lean-4.35.0/bin
```

The equivalent options are
`Options(tile_size=2, buffer_allocator="total_private_unobservable")`.
This selection explicitly states the allocator model. Tiling is optional:
`Options(buffer_allocator="total_private_unobservable")` checks the actual
three-buffer normalization→reuse chain and delivers its untiled output. When
requested, the private-buffer tile-size domain remains 2..16.
It is separate from source unrolling.

These inputs already own their `memref.alloc`/`memref.dealloc` lifetimes. The
adapter selects a direct scalar/SCF/memref-to-LLVM pipeline and disables automatic
tensor feature selection on that route. It does not run tensor ownership
inference on explicit frees. For an independent untransformed native baseline,
`Options(memory_source=True)` selects that same lowering without requesting
storage transformations. Explicit tensor features and vector/parallel schedules
are refused on this route.

The compiler retains three actual modules: original, shape-normalized and reused.
A fourth segmented module is captured only when tiling is requested.
`buffer_reuse_intake.prepare_chain` reconstructs an untrusted memory-plan hint
and independently decodes the endpoints. Untiled checking may construct an
auxiliary epoch hint; the theorem and delivery still bind the actual untiled
output. `buffer_reuse_gate.check`
builds the factored query and audits the complete proof closure.
`GraphMemoryAfterEntries.every_definition` covers the complete supported entry
roster. The compiler adapter requires its fresh live completion before the
speculative gate accepts the chain.

Normalization and reuse may be unchanged. The separate
`GraphLoopsShape.Identity` checker requires a typed complete module, zero shape
rewrite sites and identical complete source/normalized bindings. The original
positive normalization checker is unchanged. A one-to-one epoch/slot plan uses
the existing reuse theorem with every allocation/free binding intact. Such
stages appear in `unchanged`, rather than positive rewrite coverage. Initializer
segmentation is optional; both actual untiled and tiled chains have proof routes.

The current subset uses static rank-two i8/i16/i32/i64 buffers, sequential disjoint
private lifetimes and complete rectangular initialization. For example, two
separate `memref<3x2xi32>` allocations can reuse one slot when each is initialized
before reading and its lifetime ends before the next starts. Caller-visible
observations and supported unchanged functions are retained. Arbitrary escaping
pointers, asynchronous effects, external declarations and callers reaching the
changed kernel need additional proof routes. The allocator premise does not
establish finite capacity, native allocation failure or native allocator fidelity.

Independent qualification covers fresh untiled gates and exact delivery at each
admitted integer width. For i8, i16 and i64, 30 finite native observations include
the true untransformed source, modular overflow and high-bit values. These tests
qualify the existing width domain; they do not extend the rank-two source bridge.

## bf16/f16 MACC draft contract

```sh
merlin-compile ... --macc-f32acc \
  --validate-passes bf16_f32acc_draft_contract \
  --lean-sdk-bin /path/to/lean-4.35.0/bin
```

The Python transform choice is `Options(macc_f32acc=True)`; its environment
equivalent is `MERLIN_BF16_F32ACC=1`. The flag applies the existing
`lower_bf16_matmul_f32acc` at compiler entry. Its recorder also observes existing
call sites when they execute inside a validation scope. Preparation already
performed outside that scope requires its own captured boundary.

For a +0-initialized half-precision matrix product, the transformation is:

```text
bf16/f16 A, B
  → extend each operand to f32
  → accumulate in f32 in the original index/reduction order
  → truncate once to the original result format
```

`compiler_macc.check_boundary` exports and round-trips the complete actual
endpoints, checks equality to the source-derived draft reference, and checks
strict initialized-operand qualification at every recognized site. The positive
`Bf16F32AccBoundary.check_sound` theorem also requires a nonempty site set.
Fresh live completion and replay bind the exact bytes and checker evidence.
Local site wrappers preserve repeated SSA operands as one input and retain
copied initialization when that value is also a contraction input. The packing
lemmas prove unique identities, shared-operand correlation and frozen evaluator
binding; an alpha check binds target captures back to the actual source IDs.

The result is **draft-policy conformance**: `policy_reviewed=false`, numerical
policy verdict `abstracted`, and no original-source equivalence claim. The user
has explicitly retained this draft status pending independent review. `all`
excludes it. Matching bf16 or matching f16 operand/result formats are supported;
mixed source formats, nonzero initialization and incompatible named maps remain
refused. Generic mathematical lemmas do not broaden the source recognizer.

This does not establish whole-capture defined execution, IEEE/native fidelity,
or approved numerical accuracy. Combining options still requires each boundary
to meet its own intake domain; the scalar loop checker does not thereby accept
arbitrary tensor MACC programs.

## Q/DQ expansion under the proposed source model

Select `--validate-passes lower_quant_ext` to require a fresh proof of actual
Q/DQ lowering. It remains outside `all`. For example, an injected quantizer
multiply-to-add bug must stop compilation before products return. This reuses
the existing typed-module, source-support and chain-composition proofs.
`source_spec_approved=false` distinguishes proof acceptance from source-model
approval; quantizer definedness and framework/native fidelity remain separate.

An optional `--qdq-source-capture` binds the actual boundary to a freshly
reconstructed original PT2 graph, pinned producer and Torch payload. It requires
explicit source tools and complete capture artifacts, rather than a saved
`matched` flag. See [the Q/DQ gate guide](lean_qdq_gate.md) for the interfaces,
three distinct guarantees and source-domain limits.

## Selecting and reviewing coverage

Environment equivalents are `MERLIN_LOOP_UNROLL_FACTOR`, `MERLIN_LOOP_TILE_SIZE`
and `MERLIN_BUFFER_REUSE_ALLOCATOR`. Proof selection remains
`MERLIN_VALIDATE_PASSES`; SDK and authoritative exporter selection remain
`MERLIN_LEAN_SDK_BIN` and `MERLIN_COMPILER_PYTHON`.

Use the check names appropriate to the input IR. `all` also demands the tensor
DCE/CSE checks if those boundaries change; scalar/SCF or memory inputs are not
automatically within the tensor checker's supported domain. Unsupported inputs
refuse rather than silently dropping a selected check.

The standalone loop and buffer APIs remain available for developer workflows.
Their raw reports describe those issuers' narrower scopes; the compiler's
`verdict.json` records enforcement at the integrated boundary. Historical
qualification and human approval are distinct from a new live proof.
Approved numerical-policy changes and later LLVM CSE have no registered
compiler gates. The explicit Q/DQ and draft MACC checks retain their qualified
claims. [The verification design](../design/merlin_lean_verification.md)
records the separate Lean-MLIR scalar integration and the policy to reuse suitable
upstream proof components before implementing new ones.

## Other available Lean interfaces

These interfaces retain standalone selections and are not extra names in
`--validate-passes`:

| Interface | Stack location | Example and scope |
| --- | --- | --- |
| `lean_checker.verify_scalar_transform` | Shared scalar transformation broker | Prove wrapping `x + 0` equals `x` under a pinned toolchain; `Foundation.lean` uses Lean-MLIR's dialect interface |
| `loop_transform_gate.transform` | Standalone producer→proof→consumer | Run the pinned unroll/segment producer and deliver only after its complete proof; see [direct loop checking](lean_loop_transform_gate.md) |
| `buffer_reuse_intake.prepare_chain` + `buffer_reuse_gate.check/deliver` | Standalone three- or four-buffer memory boundary | Check initialized disjoint private lifetimes and guard a direct consumer; see [direct buffer checking](lean_buffer_reuse_gate.md) |
| `configuration_elision_intake.prepare` + `configuration_elision_gate.check/deliver` | Selected instruction/configuration IR | Delete a repeated constant setter only under its exact total-overwrite effect contract; preserve modeled errors and terminal state; see [configuration elision](lean_configuration_elision_gate.md) |
| `configuration_hoist_intake.prepare` + `configuration_hoist_gate.check/deliver` | Counted Kernel IR, after instruction/effect selection and before command emission | Move one invariant pure setter before a positive counted loop; preserve modeled success, failure and terminal state; see [scoped configuration hoisting](lean_configuration_hoist_gate.md) |

## Configuration writes and loop hoisting

Configuration APIs use explicit original selected ISA/effect facts. The compiler
selector `configuration_write_elision` guards `ConfigurationIRSelection.apply`
and the configuration stage of `compile_schedule`. Instruction IR may contain
complete function definitions or a closed bare module; Kernel elision currently
requires straight-line calls. An active original validation is joined, or the
owner can pass `pass_validation=validation` explicitly. The result stays held
until the canonical worker completes. Target scheduling and target emission
forward that original owner. Ordinary exported compilation remains independent
of verifier modules when no validation is active.

```python
selection = ConfigurationIRSelection(selected_contracts, "caller-effect-contract-v1")
result = selection.apply(original_module, pass_validation=validation)
consume(result.module)  # the owned gate has joined before apply returns
```

Here `validation.passes` contains `configuration_write_elision`, and
`selected_contracts` comes from the original selected facts/software files.
The caller's effect-contract label is a premise, not independent qualification.
The older elision `prepare`/`check`/`deliver` API and fresh replay remain available.
The explicit `configuration_loop_hoisting` selector guards the real counted-
Kernel producer and joins its original owner before returning the result. It
requires `selected_contracts=...` on `hoist_loop_configuration`; the default
route remains verifier-free. Its current profile has one positive top-level
loop and constant setup/body operands; broader IV-data support is separate.
Its state relation does not approve numerical policy or the selected
instruction implementation.

For example, under an effect contract where `work` preserves the configuration
fields, redundant-write elimination can turn:

```text
set configuration = (3, 5)
work
set configuration = (3, 5)
work
```

into one setter followed by both work calls. The first setter remains because
the initial configuration is unknown. A clobber or unknown/unordered effect
invalidates knowledge; a pulse or enqueue is not a pure setter.

Hoisting moves the setter in each iteration of a positive loop immediately
before that loop. Its new partial-execution theorem includes failing work and
surrounding calls. After any modeled failure, both versions retain the same
error and terminal state. It requires one leading constant pure setter, an
ordered body that preserves its fields, and empty completion/asynchronous
annotations. Zero-trip loops, nesting and overlapping body writes are outside
this initial checker, including identical overlapping writes. A zero-trip loop
cannot be hoisted this way: the original executes no setter.

The bridges bind complete input/output IR, parser/producer origins and the
selected facts. Proof files come from the installed verification package's
canonical source tree. Root names/types, imports, process evidence and fresh
objects are audited before live delivery. Source inventories include the
`owned_children.py` helper used by process cleanup; changing it invalidates a
prepared request. Saved or copied completions grant no authority.

Use an interpreter with the producer's normal dependencies installed; isolated
producer capture intentionally ignores `PYTHONPATH`. The historical standalone elision intake stays
pinned to `4499eebb959c0f92cefd92cc6d80646ba66474ed`; standalone hoisting pins
`e0484f4886ae0780d0325cdfa998842b61bb647c` and its published source manifest.
The current compiler selectors instead bind the original live selected inputs;
they do not require those historical producer checkouts.
Neither qualification labels nor a software file's review status establish
native effects. Both relations remain conditional on the documented common
instruction/state interpretation and nonaliasing field mapping. Human contract
approval, hardware fidelity and composition with adjacent passes are separate.
