# Over-rank matmul repair — review packet

Status: provisional intent, no approval recorded. No new axiom, theorem pin, shared driver
integration, or SPEC_LOCK change. This packet concerns `collapse_overrank_matmul` in
`src/merlin/llvmlower/passes_xdsl.py`; the outdated collapse/matmul/expand plan is not its contract.

## Proposed claim

For an accepted complete boundary, the output program has the same frozen operational
semantics as the source-derived **batched-linear repair reference**, including surrounding SSA,
all function bodies, nested single-block regions, internal calls, and unchanged external-call
interpretations. Every selected reduction has the independent cell meaning below. For the
additional complete-kernel check, execution succeeds and returns nonpoison output elements on
an explicit inhabited domain of initialized, correctly shaped float inputs.

**This is refinement of an intent repair, not literal equivalence to the original operation.**
Upstream MLIR rejects the original over-rank `linalg.matmul`. The default named scalar body
also orders its add as `product + accumulator`; the repair uses `accumulator + product`.
No floating commutativity theorem bridges those expressions. The old destination is discarded.
Those are explicit intent decisions requiring review, not established source equivalences.

## Independent numerical intent

For batch dimensions B, lhs shape B+[K], rhs shape [K,N], output shape B+[N], and output
coordinate (b...,n):

```
a0 = positive_zero(format)
pk = mul_format(lhs[b...,k], rhs[k,n])
a(k+1) = add_format(ak, pk)        for k = 0, ..., K-1
output[b...,n] = aK
```

The RHS is broadcast across every batch dimension. Multiplication and addition are separate
operations in the same input/output format, with fastmath=none, no rounding-mode override,
no widening, reassociation or fusion permission. Output dimensions are parallel in their
original order, followed by the K reduction. No real-number sum replaces this ordered fold.
For K=0 every output element is the literal positive-zero bit pattern. Zero batch/N dimensions
have no observable output elements. Static unencoded f16, bf16, f32 and f64 are supported by
the Lean checker. Native comparisons cover f32/f64.

The frozen FloatModel is a parameter: the same operation keys and operand order appear in
the intent and checked execution, for every model. There is **no numerical equality law to
assume or discharge for this repair**. NaNs, infinities, subnormals and both signed zeros are
permitted bit patterns; NaN is not poison. The exact abstract result follows the selected
model. Host floating exception flags, traps and errno are outside this tensor semantics.

Native NaN payload equality is **not** established. The actual generic and independently
written valid SCF reference can produce different NaN sign/payload bits after lowering.
All initial raw observations and lowered LLVM are retained in
`../counterexamples/initial-nan-payload/`; subsequent comparison files retain every mismatch.
Native results are compared exactly outside NaNs and by NaN classification otherwise.
This is an explicitly weaker empirical observation, not a proof of NaN bit equality or of
FloatModel adequacy for every native lowering choice. Any paper claim must state its floating
abstraction/domain and this limit. No original scalar-square counterexample is changed.

## What acceptance checks

The source census follows lexical SSA bindings and recursively visits nested regions/blocks.
A selected source must have static matching lhs/rhs/result/destination shapes and float formats,
the default invalid named-matmul maps and operand segments, the default scalar body with no
fast-math flags, and only recognized source/provenance attributes. Unsupported selected sites
refuse the entire boundary.

The source-derived repair inserts tensor.empty, the literal positive-zero constant, a fill
whose body yields that constant, then the broadcasted generic with exact maps, iterator order,
operand segments, same-format multiply, accumulator-first add and yield. Complete module
matching checks types, operand/result connections, all operations, attributes, regions,
returns, function inventories and signatures. Fresh target binders and all operand references
are rechecked by the sound alpha checker. No bag-of-sites or sampled-output criterion is used.

Valid source captures use the authoritative MLIR exporter. Invalid source fixtures use xDSL
structural intake and a structural round trip; the valid output always uses MLIR export and an
MLIR round trip. The dedicated intake adapter aligns property placement and the spelling of an
`arith.constant` one-bit true literal (-1 in xDSL, 1 in MLIR). `OverrankIntake.i1_literal` proves
that those spellings decode to exactly the same one-bit scalar. This does not change the shared
exporter, boolean operation semantics, or boolean-repair workstream.

The mathematical tensor proof derives actual mapped reads/writes from the affine bounds
checker, accounts for every lexicographic write, and proves that each output sees exactly the
increasing-K fold above. Allocation/fill/SSA execution is separately proved for a complete valid
reference function. `kernelCheck_sound` transfers both successful execution and ResultsDefined
to an accepted complete actual function. Positive standalone receipts record this additional
check. Nested/contextual boundaries use the complete correspondence theorem, without claiming
that arbitrary unchanged surrounding operations successfully execute.

## Proof map

- `Rules/OverrankIntake.lean`: proved one-bit literal spelling alignment for invalid-source intake.
- `Rules/OverrankIntent.lean`: independent formula, initialized-input domain and inhabitant.
- `Rules/OverrankCoordinates.lean`, `OverrankOrder.lean`: coordinate and exact reduction order.
- `Rules/OverrankBody.lean`, `OverrankTensor.lean`: actual scalar/tensor successful execution.
- `Rules/OverrankDefined.lean`: absence of poison from initialized inputs, not typing alone.
- `Rules/OverrankKernel.lean`: full allocation → constant → fill → generic → return execution.
- `Check/OverrankReference.lean`: source-derived complete repair and coverage/refusal.
- `Check/Overrank.lean`: full-program acceptance theorem and complete-kernel definedness theorem.
- `Driver/Overrank.lean`: isolated executable checker. The audit prints actual theorem statements
  and rejects nonstandard axioms, sorry/admit, native_decide and unsafe implementations.

Stable, already proved tensor trace/fill and alpha/module lemmas are imported read-only.
No W8A8 policy, quantization rule, boolean repair, scalar-square source or frozen file is edited.
Compiled checker execution is evidence under the audited Lean compiler/runtime; the general
soundness proofs and small geometry witnesses are kernel checked. Large model acceptance is
not misreported as a separately reduced closed kernel proof.

## Domain and limits

The complete-kernel domain uses functional tensors with exactly B+[K] and [K,N] shapes and
float elements. Every in-bounds element is Some(bits), never poison; the old destination is
unused (the native ABI still supplies an output-shaped tensor). This domain is inhabited at
every shape by initialized positive-zero operands. A closed Lean witness uses B=[1,2], K=3,
N=2 and proves a successful four-element nonpoison result for every FloatModel. Empty shapes
are additional cases, not the sole nonvacuity witness. The temporary empty tensor's poison is
fully overwritten before any accumulating read.

Typing, successful execution and absence of poison are separate results. General retained
model receipts establish boundary correspondence, not whole-model success or definedness.
External calls retain the same arbitrary external interpretation; no external success or
nonpoison contract is manufactured. Multi-block regions, dynamic/encoded tensors and nonfloat
repair sites are outside the accepted fragment. Integer destinations are not certified: the
production body always constructs floating multiply/add instructions.

## Evidence and reproduction

`../summary.json` binds receipts, model exercise counts, native comparison reports, audit and
preserved checkpoint. `../qualification/` contains final tests and repository gates.
`../boundaries/*/receipt.json` snapshots input/output, exports, audited dependency closure and
SPEC_LOCK, with a re-export/re-check/actual-pass replay alongside each attempt.

Run `build_tools/scripts/verify_overrank.py --compiler-python <model2MLIR-python>` and repeat
`--stage LABEL PATH` for each retained `0003_prepare-collapse-overrank-matmul` directory.
The exact completed command is in summary.json. Then run:

```
MERLIN_M2M_PYTHON=<model2MLIR-python> MERLIN_LEAN_REQUIRED=1 .venv/bin/python -m pytest \
  merlin/tests/ir/test_lean_overrank.py merlin/tests/ir/test_lean_overrank_native.py -m 'not slow'
MERLIN_M2M_PYTHON=<model2MLIR-python> MERLIN_LEAN_REQUIRED=1 .venv/bin/python -m pytest \
  merlin/tests/ir/test_lean_overrank.py -m slow
```

SmolVLA and TinyLlama retained boundaries each exercise **zero** repairs: the current pass
returns zero and complete before/after captures are unchanged. They provide no positive model
coverage. Actual-pass synthetic cases cover output ranks 3–5, singleton dimensions, K=0,
zero batch/N dimensions, all four supported formats, nested regions and an internal call.
Native valid SCF comparisons cover fourteen f32/f64 shape/format cases, six input samples each.

## Human review decisions

1. Approve or reject the independent batched-linear meaning for these invalid captures.
2. Approve or reject discarding the old destination and starting at literal positive zero.
3. Approve or reject the exact same-format, increasing-K, accumulator-first, separate-operation
   arithmetic contract and absence of fast-math permissions.
4. Agree on the paper's float abstraction and exceptional-value scope. In particular, do not
   turn native NaN classification agreement into a bitwise claim.
5. Review the restricted accepted source fragment, complete-output checker, successful/clean
   execution theorem, native evidence, zero retained-model exercise, and trusted exporter/runtime
   boundary. The implementer inspected the pass first as requested; intent remains unreviewed.

Approval must be an explicit human record. Afterwards, coordinate any shared-driver integration
and theorem pins separately with their owners. This work does not change SPEC_LOCK or refresh
other workstreams' receipts. Scalar-square's refuted native power law remains undischarged;
no multiplication-policy approval is inferred from this unrelated task.
