# AGENT.md — merlin/python/merlin/frontends

## Purpose

Frontends that ingest external IR into the Merlin pipeline. Today: linalg-on-tensors MLIR as produced by **model2MLIR** (`/path/to/model2MLIR`) — smolVLA and the other VLA workloads.

## What belongs here

- `linalg_mlir.py` — xDSL parsing of linalg-on-tensors artifacts + matmul inventory (shapes, dtypes, weights resolved via the safetensors manifest, `prov.*` provenance).
  Its standard memref and bufferization dialect registrations admit prepared
  fresh borrowed writer wrappers without an unregistered custom-syntax fallback.
  Standard scalar/string `vector.print` also retains complete registered
  fields and effects; its native software implementation is a separate gate.
  Standard materialization accepts compatible strided destinations with the same
  shape/element rules and writable/restrict/result checks as native MLIR.
- `facts.py` — lifting the inventory to contract-level reuse facts, driving the core pipeline with real model shapes, residency-variant DSE measurement.
- `compile_inputs.py` stages compiler-only Torch captures and explicitly derived tensor models.
  `compiler_constants.py` requires complete source-bound lifted constant bindings
  against original prepared placeholders, actual argument types and exact
  safetensors bytes. Retain the public parent ABI for derived models. Legacy
  unbound constants refuse; evaluator archives/digests never supply compiler
  values. The compiler-only projection grants byte custody, not numeric,
  effect, packing, ownership or target permission.
  `derived_compile_inputs.py` keeps parent FX and normalized TTIR correspondence separate,
  checks receipt-bound compiler members, and excludes evaluator values and digests from its staged view.
  Staging grants byte custody only, never source equivalence or numeric admission.
  `kernel_compile_inputs.py` separately stages standalone functional tensor kernels,
  including explicit observable-effect receipts. ABI v6 permits void entries
  only under `effect_only: true`, with readonly inputs and no discarded writes.
  Preserve real result types in initially failing tensor cases. Outcome
  references, native observations and their individual digests stay outside
  compiler staging; source effects grant no implementation/target permission.
  validates declared buffer/result order, projects compiler provenance and records
  the entry-symbol adaptation to `forward` without inventing an FX capture.
  Its ABI v3 profile retains explicitly positioned by-value runtime scalars,
  labels scalar signature entries and excludes invocation values and tensor-only attributes.
  ABI v4 additionally retains representable signed integer range declarations;
  compiler-only staging grants no memory or guard-equivalence proof from those bounds.
  The explicit shared-functional-buffer payload retains the public functional ABI,
  checks byte-bound memory witness/continuation relations and copies their compiler evidence.
  Both standalone and derived staging exclude evaluator values from that view.
  Optional model2MLIR `tensor_words.npz` contains exact BF16 evaluator words.
  Exclude it and its digests with the ordinary inputs/goldens; byte-bound source
  correspondence supplies no permission to read those words during compilation.
  The explicit shared-guarded-buffer profile copies a byte-bound runtime
  admission report instead of a pure tensor witness. Standalone staging retains
  its declared guard-storage cost in the compile signature. It checks declared
  source/access/effect relations and grants no executable memory proof.
  Derived guarded kernels additionally bind per-call scratch costs to their
  typed results and model declaration. Runtime specialization reopens the
  original typed parent call by ordinal, checks scalar attributes, source
  operand/result mappings and representable bounded dimensions, then derives
  each call's concrete cost independently. Zero extents contribute zero; costs
  must fit i64. No evaluator archive/digest or producer-only shape claim supplies
  these facts. The evaluator budgets scratch separately from final outputs.
  V2 memory admission checks the sticky reader/writer protocol and derives
  metadata word count/cost from the maximum logical grid, retaining the explicit
  pre-CSE memory-effects stage. V1 evidence keeps its original scope and cost.
  The source-word buffer profile also retains exact word-stage input/output
  evidence and validates its explicit pass sequence and function access intent.
  V2 word evidence embeds isolated-event input/output, typed source bindings,
  arithmetic-only parent input and both actual native reports. Staging checks
  their byte chain, optimizer identity and declared inventory; producer-side
  event reconstruction and numerical equivalence remain separate obligations.
  Atomic view bindings use the complete source context and actual captured
  storage dtype. One source event may select integer and floating roots;
  a CAS template's logical pointer formal does not replace its storage capture.
  Primitive effect checks and producer reconstruction remain separate gates.
  V3 memory admission additionally retains the explicit private-invocation
  integer atomic contract, or the explicit v4 typed profile with bitwise f32/f64
  exchange and integer RMWs on equal-width floating storage, thirty-bit ownership fragments and source operation/
  scope/order/old-value inventory. Staging checks those declarations and costs;
  CAS additionally retains i16/i32/i64 comparison/old-value facts, its unmasked
  source access, and exact source versus memory opcode correspondence.
  it grants no atomic linearizability, race/initialization or target proof.
  It shares the guarded public ABI without granting source or numeric proof.
  `kernel_shape_abi.py` validates ABI v5's named runtime dimensions and logical
  ceil-divided grids. Standalone compile-signature v2 retains symbolic tensor
  dimensions and typed scalar bindings. Staging checks declared types/relations,
  excludes example sizes and grants no memory, shape-guard or numeric proof.
  Named dimensions may occur on multiple axes with mixed static extents; dense
  stride/address and ownership proofs remain the producer's responsibility.
- `linalg_math_patterns.py` — source-only, closed unary f32 sine/cosine/exponential/roundeven
  scalar-region checks; each operation requires its own explicit declaration. No host admission,
  selected math implementation or numerical-equivalence claim follows. Scalar-literal power refuses.
- `linalg_mixed_patterns.py` preserves exact typed mixed source forms, including signed i32-to-f32
  conversion with its original scalar SSA edge and projected tensor maps. A source pattern does not
  supply a host declaration or certify conversion or execution numerics.
- `linalg_composite_math.py` — separate source-only, closed literal clamp, reciprocal,
  literal-base power, and tanh-GELU scalar regions. It records exact f32 literal bits
  and unresolved math-intrinsic obligations; no host, libm-supplier, or numerical grant.
- `linalg_patterns.py` has separate strict identity and opt-in static singleton-projected
  pointwise schemas. The latter reuses the same closed scalar-region proof, retains each
  positive-static input shape and exact affine map (including rank-zero scalar inputs),
  and grants no host placement, index-width bound, or numerical equivalence by itself.
  Projected i32 addition requires its explicit `integer_dtype: i32` source declaration;
  the closed proof rejects overflow flags and preserves the original integer width.
- `linalg_integer_reductions.py` — closed static i64 sum and masked prefix-sum source checks,
  with an explicit index-width premise; no host admission or linked-build verdict.
- `linalg_f32_maximum_patterns.py` — closed static one-axis f32 `linalg.reduce`
  with an exact negative-infinity seed and unflagged `arith.maximumf`. The
  selected index width is a caller premise; PyTorch reduction equivalence,
  host placement, and linked-build behavior remain separate obligations.
- `linalg_reduction_source_body.py` names the closed integer sum/prefix-sum/paired-minimum
  frontend/root-operation kinds shared by host admission and the independent linked
  witness. Its declaration validator grants no placement or numerical equivalence.
- `prepared_index_source_body.py` screens three exact tensor-index compute roles against
  a caller-reproved, receipt/trace-bound record. It validates parsed types, shapes,
  maps and ordinals but does not produce that record, grant placement, associate
  prepared nodes or establish original/compiled numerical equivalence.
- `bucketize_source.py` — a source-only, trace-bound proof of the closed f32
  bucketize counting reduction. The trace-bound count alone does not grant
  host placement, prove sorted boundaries, or certify generated code and
  numerical equivalence. Its separate opt-in static source-body recognizer
  requires a sorted non-NaN f32 literal
  and checks static index spans; the private issuer still owns right-flag
  ancestry and exact capture-to-build binding.
- `linalg_boolean_patterns.py` also owns a separate, closed dynamic rank-one
  i1-to-i64 unsigned cast source body. It proves input-derived output extent,
  identity maps and scalar extension, but not a runtime bound, host placement,
  external dynamic-output ABI or numerical behavior of a compiled program.

## What does not belong here

- Model capture/conversion (that is model2MLIR's job — don't vendor it).
- Lowering or dialect definitions (`merlin/python/merlin/xdsl_dialects/`).

## Interfaces

- Input: `workloads/<model>/<model>.mlir` + `<model>.safetensors.manifest.json` from model2MLIR.
- Output: `MatmulRecord`/`WeightReuseFact` inventories; `LoweringResult` via the existing pipeline; `dse` IR + `dse_result`-shaped dicts.

## Invariants

- Parse the **full** model artifact, not `sections/*.mlir`: the model2MLIR section splitter currently emits use-before-def SSA references (invalid SSACFG IR; e.g. `%2034` in `sections/smolvla.model.mlir`). Fix upstream before relying on sections.
- `PAREN_RESULTS` exists because xDSL 0.65's linalg parser rejects `} -> (T1, T2)`; remove it once xDSL accepts the parenthesized multi-result form.
- The integer pipeline executes a layer's **i8 deployment GEMM** with the real (M,K,N); the capture dtype is preserved as provenance, never silently claimed as executed.

## Testing expectations

`merlin/python/tests/test_frontend_linalg.py` — synthetic-text tests run everywhere; smolVLA-artifact and spike tests auto-skip when the artifact/toolchain are absent.

## Notes for future agents

smolVLA reuse structure: capture unit = one denoise step; `select_action` runs N steps per tick, so every weight is `reused_across_invocations` — the repeated-RHS pattern with real shapes.

`linalg_mixed_patterns.py` is opt-in typed routing source evidence. It proves closed
bf16/f64/mixed scalar SSA sequences, default tensor layouts and bounded static input
projections; it preserves finite literal bits and refuses init reads/fast-math. Historical
f32 schemas retain their meanings. It grants no framework numerical equivalence.

- `linalg_integer_seams.py` proves exact typed clamp/truncation and low-bit XOR source bodies; it grants no numerical qualification.

- `linalg_quantization_seams.py` proves exact scalar reciprocal and projected FP32 roundeven/clamp/int8-cast source bodies. Native numerical and selected rounding obligations remain separate.

- `connected_region.py` extracts an exact single-block source window with literal/init closure, nested captures and every live output. `verify_window` reconstructs from pinned original bytes before returning operation pairs. Re-run source/trace/index proofs on the projection; do not copy ordinal-bound host admissions. Runtime inputs remain explicit, and no numerical or target qualification follows.

- `bounded_control_source_body.py` admits only exact scalar count/index/guard roles from a caller-reproved whole-source interval context. `connected_index_source_body.py` preserves the closed prepared-index recognizers while requiring a separately reconstructed projection and exact captured typed-root join. Neither supplies linked host qualification, compiled bounds, or numerical credit.
