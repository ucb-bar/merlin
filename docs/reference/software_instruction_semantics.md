---
title: Software instruction semantics declarations
kind: reference
status: current
owner: targetgen
last_verified: 2026-10-23
related: [phase0_specification, independent_instruction_policy, verified_compiler_tooling]
code_refs: [src/merlin/targetgen/instruction_semantics.py, src/merlin/targetgen/corpus_instruction_roles.py, src/merlin/targetgen/isa_spec.py, src/merlin/targetgen/isa_codegen_cli.py]
---

# Software instruction semantics declarations

The minimal software specification declares mathematical behavior and numerical
permissions. Optional instruction/effect declarations bind independent checks to
selected facts; they do not select compiler tiling, layouts or schedules.

## Inspect instruction scaffold readiness

The existing ISA generator can inspect explicit inputs without writing generated
code or invoking a simulator:

```sh
merlin-isa-generate --software-spec software.yaml --facts facts.json \
  --target TARGET --inspect
```

The JSON report pins both input files and distinguishes `missing_declaration`,
`invalid` and `ready_for_generation`; unavailable readiness exits with status 2.
A complete typed `isa` declaration must resolve every encoding field against the
selected facts. This checks generation inputs, not RTL correspondence, compiler
lowering or target execution. Those fields remain `UNKNOWN`, and unknown
instruction effects/semantics remain explicitly listed. The original `--out`
generation route remains unchanged and cannot be combined with `--inspect`.

## Declare effects and operand interpretation

The SW spec states which behavior software may rely on. It does not define an
accelerator instruction set. When a target has an independently reviewed instruction
description, keep it in its target-owned declaration package and select its relative path with
`instruction_semantics` in the target contract. Phase 0 validates its SW-operation
links and binds it to the selected CIRCT facts, then freezes both the exact
authored bytes (`software/instruction-semantics-authored.yaml`) and the normalized
consumer model (`software/instruction-semantics.json`). The normalized model records
the exact authored and selected-input byte hashes when file bytes were supplied;
in-memory selected views carry canonical content hashes instead.
Neither file belongs in Merlin core, and neither is a generated compiler.

Each instruction description needs typed operands/results, a computation pattern
(indexing maps, iterators and scalar SSA body), applicable SW operation, side effects,
and any local-memory constraints. Unknown effects, missing SW links, or incomplete
memory capacity remain explicit `UNKNOWN` entries. CIRCT facts can justify observed
structure, but a decoder field alone cannot supply complete functional semantics.
Do not copy an instruction from a related target or fill missing behavior by name.
If no description is selected, Phase 0 emits an `UNKNOWN` model, not a guessed one.
`described` means the checked declaration is internally complete, not that the
instruction has been proved against RTL or executed on hardware.

This extra input is intentionally separate from the minimal SW spec: an author writes
software-visible behavior once, the OOT owner describes instruction behavior once,
and Phase 0 binds both to exact hardware evidence. The schema is
[`instruction_semantics.schema.yaml`](../../merlin/schemas/instruction_semantics.schema.yaml).

The Atlas and Gemmini examples use directly named operations and transfers; authors
do not need IDs, `signature` wrappers or duplicate copy endpoint constraints.
Expanded v1 list declarations remain readable for compatibility and generated
inspection, but are not the starter format. The examples do not
repeat mesh/memory geometry, opcode maps, calibration coefficients, historical runs or
backend code. Gemmini authors its contraction directly but names fact-derived
`hardware` forms for its fused and standalone elementwise, readout pooling and movement
declarations. Gemmini explicitly constrains the internal i20 partial-sum domain even
though its software readout is i32. Atlas explicitly constrains the finite-normal FP8
and BF16 domain; unknown block-size/scale parameters remain unknown. Host capabilities
are a separate manifest bound to the selected host compiler, not accelerator facts.

Version 1 remains backwards compatible: an old inline `capability_contract` and
`evidence` mapping are optional legacy fields. New minimal specs require a separately
selected same-target contract or explicit backend-config contract. The pure
`capability_contract(spec, base_contract=selected_backend)` projection preserves that
selected policy without filesystem lookup or guessed protocol. Inline legacy declarations
retain their recorded precedence. The exact backend/config bytes are independently
frozen, and its broader capability list cannot bypass typed SW admission.

`MERLIN_TARGET_CONTRACT=/selected/target_contract.yaml` explicitly selects a
declaration independently of executable compiler support. For an admitted
`runner.backend: chipyard_rocc` contract, the generic runtime consumes selected
facts and contract data without a compiler-bearing provider. Provider-bound
workflows separately select their implementation through `MERLIN_TARGET_PATH`;
that implementation is not an input to fresh compiler authoring. Phase 0 records
the effective source selections and refuses a different target identity. Frozen
execution consumes the saved contract bytes, not a live override. Selecting a
declaration does not prove an implementation satisfies it. See
[neutral runtime tooling](neutral_runtime_tooling.md).

Unknowns remain explicit. Review operation legality, exact numeric behavior, quantization
granularity/zero points and ABI ordering before changing a spec's status. A width alone
does not identify a floating-point format. “Not extracted by the current reader” does not
mean “impossible to infer from RTL”: first inspect the available elaboration and extraction
coverage, then document the residual declaration and its independent evidence.

The generated `funct_decode_table` is a historical field name, not a promise of a full
instruction set. Its `scope: observed_decode_field` and `complete_isa: false` mean that
the listed equalities apply only to the field the RTL compared. That field may concatenate
nonadjacent instruction bits. Do not use its width or values to select RoCC versus a
self-hosted endpoint, or to accept/reject complete instruction words. Keep the observation
in `facts.json` for audit; use a separately established executable interface and a
complete-word ISA definition for those decisions. Missing evidence leaves the generated
endpoint unresolved rather than promoting a plausible architecture guess.

TorchAO uses public extension points, not source patches. A selected accelerator format
must also be legal for the operation receiving it. An int8 contraction does not establish
int8 LSTM, general normalization, or support for another hardware configuration.

Do not handwrite a second TorchAO recipe. The evidence export writes
`software/quantization-recipes.json`, indexing generated, content-addressed recipes
under `software/quantization-recipes/`. Each recipe intersects the selected hardware
format with the SW spec's eligible operations and typed constraints. Host-only
operations are excluded; explicit signature refusals remain unquantized. Missing
hardware scale parameters do not acquire a default recipe. These are diagnostic
transformation inputs, not accelerator-admission certificates.
An unknown block size does not obstruct a selected per-tensor or per-channel
recipe, which has no block axis; the generated contract records it as not applicable.

Select a recipe path from that index and use the existing capture worker:

```sh
"$CAPTURE_PYTHON" src/merlin/targetgen/_m2m_capture_worker.py \
  --m2m-dir "$MODEL2MLIR_ROOT" \
  --loader examples/workloads/coverage_mlp/loader.py \
  --dtype int8 --seed 0 --recipe "$GENERATED_RECIPE" \
  --materialize-bundle --out "$QUANTIZED_CAPTURE"
```

Inspect `meta.json`'s `quantization_stats` for the observers, epsilon, actual
calibration source/count and per-operation SW decisions. Observer defaults are
framework transformation policy, not RTL semantics, and need not clutter the
SW spec. The adapter defaults to min/max observation for FP8 because TorchAO PT2E's
histogram observer requires an integer dtype; an explicit FP8 histogram request is
refused. `integerization_receipt` records true integer contractions. When the SW
spec selects `integer_reference`, its `golden_agreement` compares the rewritten
graph exactly against an independent PT2E integer interpreter; the
`integer-reference.json` outputs and reference source identity are byte-bound by
the capture receipt. `portable_agreement` separately compares with TorchAO's
portable Q/DQ graph and may differ because its arithmetic is not integerized.
`recipe_agreement` compares against the original floating model. A single example-input calibration does not
establish model accuracy. Preserve the original FP32 capture as reference lineage,
and derive a fresh corpus from the actual quantized capture before claiming its
precision coverage. Scoped dynamic module transforms are not currently supported
by this signature-screened recipe route.

If the selected host contract admits FP32 but not the source model's BF16,
explicitly select `--stage-fp32` before the recipe transformation. A sealed
capture binds the same choice as `worker_options: {stage_fp32: true}`; it does
not inherit a precision choice from the shell. This requires Model2MLIR's
audited, schema-owned floating-dtype retargeting API. It changes the captured
static program, not dtype-dependent Python branches, and does not add BF16 host
support. The option supports FP32 capture or an explicitly selected int8 recipe,
not additional ad hoc quantization rewrites.

Inspect `meta.json`'s `precision_conversion` and `fp32_staging`: original and
staged graph identities, typed input/output ABIs, source-layer placement plan,
and original-to-FP32 output differences observed **before** quantization.
Loader-owned calibration is collected from the original module before export,
copying each yielded sample before its producer can reuse storage. Collection
is bounded by the quantizer's calibration/agreement budgets, then staged to
FP32. `calibration_input_abis` records original and staged shapes/dtypes;
it is not a numerical-equivalence certificate.
For a semantic session, the audited FP32 stage also freezes its independent
pre-quantization trajectory in `session_quality_fp32.npz`. Floating observation
streams are staged to the actual input ABI; integer/bool streams remain exact.
The integerized trajectory is separately saved in `session_goldens.npz`.
These are stage-local references over declared inputs and recurrence, not a
claim of whole-application FP32 accuracy or BF16 target-host support.
Integer/bool output values and types must remain exact. The recipe agreement
then compares quantized execution against the FP32 stage, while the integer
reference checks the integerized graph independently. None of these selected
input observations proves full-model accuracy; residual unsupported precision
or incomplete precision evidence still blocks admission. Changed producer
bytes or staging options require a fresh capture selection, not edited receipts.

Reviewed operation declarations can carry `numerical_contract` with `status`,
structured `semantics` and review `evidence`, or name a supported contract. The named
`operand_sum_exhaustive_i8_v1` contract admits a composition only with the program's
operand-sum numeric screen: a fact-described software model checks all 65,536 ordered
i8 operand pairs for the selected scale pair against a fact-derived error bound. A missing
screen leaves the operation unknown and an exceeded bound refuses it; the screen is not
RTL execution or target-oracle evidence. An exact-op declaration can also constrain integer
`quantization_parameters` (`zero_point`, `quant_min`, `quant_max`); an unobserved parameter
stays unknown and a different value is refused. The corresponding declaration on
the pinned host capability spec is independent of accelerator arithmetic.
Selected contraction-level `numerical_semantics` apply only to accelerator
contractions, not automatically to host operations or other operation families.
Unknown overflow, rounding or reduction behavior blocks numerical declaration
completeness even when the storage and accumulator widths are known.

Framework realization and integer lowering are also separate. The trace-capable
model2MLIR PT2E integerizer supports symmetric per-tensor operands and per-output-
channel weights for Linear, Conv2d and batched matmul. Reduction-axis scales and
nonzero zero points are refused. Remaining float contractions still block an
integer-accelerator claim; native FP8 tensors projected to FP32 are explicitly
reported as a precision-realization blocker. A framework lowering receipt does
not qualify the target's numerical semantics or compiler implementation.
