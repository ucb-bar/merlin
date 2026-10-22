# AGENT.md — merlin/python/merlin/frontends

## Purpose

Frontends that ingest external IR into the Merlin pipeline. Today: linalg-on-tensors MLIR as produced by **model2MLIR** (`/path/to/model2MLIR`) — smolVLA and the other VLA workloads.

## What belongs here

- `linalg_mlir.py` — xDSL parsing of linalg-on-tensors artifacts + matmul inventory (shapes, dtypes, weights resolved via the safetensors manifest, `prov.*` provenance).
  Its standard memref and bufferization dialect registrations admit prepared
  fresh borrowed writer wrappers without an unregistered custom-syntax fallback.
- `facts.py` — lifting the inventory to contract-level reuse facts, driving the core pipeline with real model shapes, residency-variant DSE measurement.
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
