# AGENT.md — merlin/python/merlin/frontends

## Purpose

Frontends that ingest external IR into the Merlin pipeline. Today: linalg-on-tensors MLIR as produced by **model2MLIR** (`/path/to/model2MLIR`) — smolVLA and the other VLA workloads.

## What belongs here

- `linalg_mlir.py` — xDSL parsing of linalg-on-tensors artifacts + matmul inventory (shapes, dtypes, weights resolved via the safetensors manifest, `prov.*` provenance).
  Its standard memref and bufferization dialect registrations admit prepared
  fresh borrowed writer wrappers without an unregistered custom-syntax fallback.
- `facts.py` — lifting the inventory to contract-level reuse facts, driving the core pipeline with real model shapes, residency-variant DSE measurement.
- `linalg_math_patterns.py` — source-only, closed unary f32 sine/cosine scalar-region checks;
  no host admission or numerical-equivalence claim. Its unary schema refuses scalar-literal power.
- `linalg_composite_math.py` — separate source-only, closed literal clamp, reciprocal,
  literal-base power, sigmoid, and tanh-GELU scalar regions. It records exact f32 literal bits
  and unresolved math-intrinsic obligations; no host, libm-supplier, or numerical grant.
- `linalg_patterns.py` has separate strict identity and opt-in static singleton-projected
  pointwise schemas. The latter reuses the same closed scalar-region proof, retains each
  positive-static input shape and exact affine map (including rank-zero scalar inputs),
  and grants no host placement, index-width bound, or numerical equivalence by itself.
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

- Tensor archives are staged and hashed in chunks and checked against typed headers
  without copying complete payloads. Selective argument reads preserve every
  role in the signature, count repeated stored tensors once, and enforce the
  payload budget before allocation. Extract bytes from the same stream that
  supplies the checked digest; unselected archives still need their identity.
- Compiler-input staging selects only receipt-declared lifted constants and
  manifest-declared buffer initializers from `extra.npz`. Keep the raw archive,
  runtime samples and reference outputs outside the compiler directory. Buffers
  retain their state role. Recover BF16 storage only with exact bits; refuse
  lossy legacy dtype casts and unqualified NaN payloads.
- Parse the **full** model artifact, not `sections/*.mlir`: the model2MLIR section splitter currently emits use-before-def SSA references (invalid SSACFG IR; e.g. `%2034` in `sections/smolvla.model.mlir`). Fix upstream before relying on sections.
- `PAREN_RESULTS` exists because xDSL 0.65's linalg parser rejects `} -> (T1, T2)`; remove it once xDSL accepts the parenthesized multi-result form.
- The integer pipeline executes a layer's **i8 deployment GEMM** with the real (M,K,N); the capture dtype is preserved as provenance, never silently claimed as executed.

## Testing expectations

`merlin/python/tests/test_frontend_linalg.py` — synthetic-text tests run everywhere; smolVLA-artifact and spike tests auto-skip when the artifact/toolchain are absent.

## Notes for future agents

smolVLA reuse structure: capture unit = one denoise step; `select_action` runs N steps per tick, so every weight is `reused_across_invocations` — the repeated-RHS pattern with real shapes.
