# TinyLlama operational typing review

Scope: Merlin's `prepare-scalar-squares` boundary between model2MLIR and the OOT compiler.
The source is the actual `SY_model_tiny_llama` capture retained by the scalar-square boundary
receipt. This packet supplements the original numerical and composition review packets.
It does not authorize a multiplication policy or make a paper claim of original-source equivalence.

## What is proved

`ScalarSquareTyping.closedCheck_sound` derives
`SourceTyping source (Family M 64) 64` for every floating model `M` from the complete closed-program
checker. There is no operational typing, float equality, external-result, or approval hypothesis
in this closed-capture theorem. `closed_results` additionally establishes that every successful
function execution returns the signature's runtime types, for every call fuel.

The actual source has 10,133 operations including function and terminators, 1,557 nested linalg
bodies, and 45 selected square sites. The checker traverses every operation, including bodies and
operations after the first unsupported operation. Negative tests corrupt late bodies after that
unsupported prefix, so acceptance cannot be explained by stopping at an early failure.
Complete source and target exports are checked; the original boundary receipt still checks the
entire transformation and all selected sites.

Proof layers:

1. `ScalarSquareTypingLinalg`: equality and successful-result typing through loop reads,
   typed body invocations, writes, reductions, and final tensor canonicalization. A body may
   return `stuck` or UB; the theorem preserves those outcomes instead of pretending progress.
2. `ScalarSquareTypingFamilies` / `ScalarSquareTypingRules`: reusable structured, arithmetic,
   math, tensor, constant, index, quantization-refusal, and direct-call rules. Existing shared
   safety rules are reused for regionless operations. Every accepted rule has a Lean proof.
3. `Check/ScalarSquareTyping`: lexical SSA typing, nested region arguments and yields,
   result declarations, and terminators. It checks declared function and call signatures.
4. `ScalarSquareTypingProgram`: function results and internal calls by fuel induction.
   Foreign calls require `ExternalResults`: typed inputs imply typed successful results for
   each declared signature. TinyLlama has no calls or external declarations, so its theorem
   has no foreign contract premise.
5. `ScalarSquareTypingFacts` / `ScalarSquareTypingReady`: exact constant-producer transfer,
   SSA environment extension, and derivation of the composition theorem's `SourceTyping`.

The concrete capture is certified by executing the Lean-sound closed-program checker through
an isolated driver. Its acceptance discharges the finite checker premise of `closedCheck_sound`;
it is not a diagnostic inventory or an asserted operational typing assumption. The general
soundness and composition theorems are kernel checked. We do **not** claim that the entire
10,133-operation Boolean computation was separately reduced to a closed proof term in the kernel.

The evidence trust boundary includes the bound exporter/decoder, Lean compiler/runtime, and
execution harness, as in the existing boundary receipts. Source files, compiled import closure,
Lean executable/toolchain, source and target exports, SPEC_LOCK, and proof-audit statements are
fingerprinted and replayed. No native-evaluation axiom or new numerical/typing axiom is introduced.
Experiments with direct whole-capture kernel reduction were stopped after high runtime/memory
cost; those experiments are not used as evidence of successful certification.

## Composition correction

The former `PrimitiveTyping` quantified over arbitrary source callbacks that agreed with target
callbacks only on typed arguments. This was too strong: an arbitrary body could return an
ill-typed accumulator, causing the next loop iteration to invoke a callback outside that domain.
The corrected obligation fixes the actual source call and region meanings. The new checker proves
their successful-result preservation, and the structured congruence proof uses that fact to keep
subsequent callback arguments typed. This is a change to the owned scalar-square composition
layer, not to the frozen evaluator, typing definitions, AST, numerical law, or SPEC_LOCK.
The earlier composition receipts are archived and regenerated against the corrected theorem.

## Typing, modelledness, success, and poison

| Property | Meaning | Status |
|---|---|---|
| Operational typing | Typed successful inputs/steps/results and valid SSA/region/call signatures | Proved for this capture |
| Modelledness (`NotStuck`) | Evaluation does not return `stuck`; UB is still permitted | Not proved; current quantization contracts are rejected |
| Successful execution | Evaluation equals `.ok results`; excludes both `stuck` and UB | Not proved |
| Poison freedom | Successful results contain no poison | Not proved; `HasTy` explicitly admits poison |
| Original-source preservation | Target refines the original power program on a defined domain | Not claimed; numerical premise remains undischarged and is refuted for the tested native implementation |
| Approved multiplication policy | Target refines a separately approved multiplication specification | Approval absent; policy definedness also pending |

The capture has 399 per-tensor quantization operations whose i64 zero points are incompatible
with the frozen signed-i8/i32/f32 dequantization semantics:

- 155 dequantizations carry `quant_min = -127`, which `metadataOK` refuses.
- 155 carry the supported full range but pass i64 zero-point tensors where runtime semantics
  requires i32. `dequant_i64_stuck` proves that such typed arguments cannot produce a successful
  dequantization result.
- 89 quantizations have no source quantization implementation in the dispatcher and carry
  `output_dtype`, which the interpreted-attribute guard refuses.

The quantization rule checks their full operand and result signatures and proves a `stuck`
outcome. Its successful-result preservation is therefore vacuous **for these operations**.
This is reported explicitly; it is not evidence of executable quantization or whole-program progress.
Other operation families have constructive result-preservation proofs, including all nested bodies.
The operation census is diagnostic; the sound recursive checker and captured proof are the evidence.

## Remaining definedness work

First provide independently reviewed shared semantics for the actual quantization contracts,
including i64 zero points, the narrow range, quantization rounding, exceptional values, and
metadata interpretation. The typing-only refusal theorem must then be replaced by appropriate
operational lemmas for those semantics, and affected receipts regenerated. This packet does not
change that shared workstream.

Next prove `FuncNotStuck` on an explicit, inhabited entry/context/input domain. A typing proof
alone does not discharge operation preconditions, affine/tensor access support, shape conditions,
call-fuel sufficiency, or implementation support. TinyLlama has no foreign calls; library users
must discharge foreign progress contracts separately from `ExternalResults` when proving progress.
For successful execution, additionally exclude UB. For poison freedom, require and propagate
appropriate input definedness and show that initialization/reads cannot expose poison.

None of that repairs the native power-to-multiplication law. The strict source theorem still takes
that law as an explicit argument. A separately approved multiplication policy remains a different
specification, with no implication of original-source equivalence.

## Numerical evidence preserved

All 44 native evidence files and the two original review references remain bound, byte-for-byte,
by `../counterexamples-preserved.json`. In particular, both finite one-ULP counterexamples remain:

- f32 input `0x3f10d742`: power `0x3ea3e5bf`, multiplication `0x3ea3e5be`.
- f64 input `0x3fefb55750829a12`: power `0x3fef6b5cd0aa0e82`, multiplication `0x3fef6b5cd0aa0e83`.

The prior exceptional-value, flags, NaN payload, zero, subnormal, and floating-environment
limitations remain in the original review packet. There is no new axiom or specification pin.

## Replay

From the repository root, with the repository environment:

```sh
.venv/bin/python build_tools/scripts/verify_scalar_square_typing.py \
  --compiler-python /home/jeremy/Projects/model2MLIR/.venv/bin/python --replay
```

Replay re-executes the boundary and complete typing checks and rechecks hashes, the Lean proof
audit, and conservative claim metadata. The slow pytest replay uses the retained receipt.
Test and audit logs are linked by the manifest.
