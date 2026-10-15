# Findings ledger

Defects and discrepancies the verification work surfaced. Each entry states what was observed,
the evidence, whether it is on the paper path, and its status. (Design §8.5.)

## F1 — xDSL and MLIR build different bodies for custom-syntax named linalg ops

- **Observed (2026-10-13, round trip over the repo's 735 `.mlir` files):** for custom-syntax
  named ops (e.g. `linalg.matmul ins(...) outs(...)` with no explicit region), xDSL materializes
  the body as `arith.addf(%mul, %acc)` while MLIR materializes `arith.addf(%acc, %mul)`. 76 files
  differ only in this operand order.
- **Why it matters:** under the parametric float semantics these are different programs. IEEE
  addition is commutative (bit-exact on RISC-V, whose NaN results are canonical), so the compiled
  numbers agree, but a boundary that re-prints such a module through xDSL is unprovable without a
  named `addf` commutativity law, and the verified reading must be MLIR's (design D9).
- **Paper path:** captures and Merlin's stage prints are generic form, which carries the region
  explicitly, so both parsers read the same body. To be confirmed by the measured inventory.
- **Status:** open; exporter will use MLIR's own parser (`torch_mlir.ir`) as the authority.

## F2 — MLIR's Python bindings hide the properties of unregistered ops

- **Observed (2026-10-13):** `torch_mlir.ir` exposes only discardable attributes of unregistered ops
  such as `quant_ext.dequantize_per_tensor`; their properties (`quant_min`, `quant_max`, `axis`) are
  invisible, so a naive exporter silently drops them. The round trip flagged all 8 affected files.
- **Paper path:** yes — `quant_ext` ops are in the captures until `lower_quant_ext` (U15).
- **Status:** fixed in `mlir_export.py` (the property dictionary is recovered from the op's own
  generic print with a balanced, string-aware scan and parsed by MLIR; failure fails the export).

## F3 — xDSL accepts IR that MLIR's verifier rejects

- **Observed:** a fixture with an ill-typed `linalg.transpose` (result 16x16 for an 8x4 permutation)
  parsed and passed through Merlin's xDSL preprocessing; MLIR rejects it
  (`dim(result, 0) = 16 doesn't match dim(input, permutation[0]) = 8`).
- **Why it matters:** Merlin's Python passes can operate on, and emit, IR the compiling MLIR would
  reject or read differently. Exporting every boundary with MLIR's parser (design D9) surfaces it at
  the boundary where it appears.
- **Status:** fixture corrected; the general risk is covered by MLIR-parser export of every stage.

## F4 — Merlin's capture worker is out of step with model2MLIR HEAD

- **Observed (2026-10-13, model2MLIR `905d2c9`):** `_m2m_capture_worker.py --materialize-bundle`
  fails with `reused conversion weights must be the output bundle's exact weights file`, and
  `--diagnostic-model-copy` fails expecting a `weights.safetensors` that is never written; both leave
  an empty `linalg.mlir`. Calling `m2m.convert(..., quantization=QuantizationConfig("int8_dyn_act_int8_weight"))`
  directly succeeds.
- **Paper path:** blocks reproducing captures with the worker at this model2MLIR revision.
- **Status:** open; needs the model2MLIR revision the paper captures were made with.

## F5 — the torch_mlir route prints syntax Merlin's xDSL reader cannot parse

- **Observed:** `m2m.convert(backend="auto")` takes the torch_mlir route and prints custom syntax;
  xDSL fails on `tensor.expand_shape ... output_shape [...]`. A generic-form print by the same MLIR
  is readable. That route also decomposes W8A8 quantization into `arith`/`linalg` (no `quant_ext`
  ops) and emits `cf.assert`, so which Merlin rewrites fire depends on the capture route.
- **Status:** open; the paper's capture route and recipe decide the denominator.

## F6 — `lower_quant_ext` does not preserve `dequantize_per_channel` for large zero points

- **Observed (2026-10-14):** the op means PyTorch's `quantized_decomposed.dequantize_per_channel`,
  which computes `(input - zero_points) * scales`: an INTEGER subtraction (int8 − int32 → int32),
  then one conversion to float32, then the multiply. `lower_quant_ext` emits
  `mulf(subf(sitofp(w), sitofp(zp)), scale)`: two conversions and a FLOAT subtraction.
- **Counterexample (float32, checked against the PyTorch op):** `w = 1`, `zp = 16777217`,
  `scale = 1.0` → PyTorch `-16777216.0`, Merlin's lowering `-16777215.0`.
- **Scope:** the two agree whenever `|zp| < 2^24` and `|w - zp| < 2^24`, which holds for every int8/uint8
  zero point, so compiled numbers on realistic models are unaffected. The rewrite is still not a
  refinement for all inputs, so the boundary cannot be verified as is; under the parametric float
  model it would also need IEEE exactness laws.
- **Paper path:** only on the capture route that keeps `quant_ext` ops (FX-importer captures).
- **Suggested fix (needs sign-off, it changes emitted IR):** emit `extsi` → `subi` → `sitofp` → `mulf`,
  PyTorch's exact order; the rule then needs no float assumption and no input-domain restriction.

## F7 — the torch_mlir route elides the weights, so the capture alone does not define the model

- **Observed (2026-10-14):** `torch_mlir_bridge` prints the module with `large_resource_limit=16`, so
  every weight is a `dense_resource<...>` constant whose blob is dropped (the file ends with an empty
  `{-# #-}` resource section). The exporter emits these as opaque attributes and the semantics is
  `stuck` on them, so `check-support` refuses the coverage_mlp capture at its first weight constant.
- **With the weights bound as function arguments** (the FX-importer route's form, where the
  safetensors manifest maps arguments to weights), the same capture passes `check-support`: all 67
  `linalg.generic`s, their bodies, the reshapes, `linalg.transpose`, `linalg.fill`, `linalg.index`
  and `cf.assert` are inside the proved fragment. Seeded faults (an out-of-bounds map, an integer op on
  floats, a wrong result shape, a dropped iterator, an unknown attribute) are each refused.
- **Paper path:** decides whether the end-to-end claim covers "for all weights" (weights as inputs or
  as context data) or "for these weights" (materialized resource bytes).
- **Status:** open; needs the capture-route decision.

## F8 — re-parsing in one process renames every `dense_resource` key

- **Observed (2026-10-14):** xDSL 0.73's builtin resource storage (`op_asm.declare_resource`)
  outlives a parse and de-duplicates every key it sees against keys an earlier parse registered.
  In one build process, `prepare_for_lowering` parses the capture, and `preprocess_text` parses the
  prepared text again; the second parse renames every weight
  (`torch_tensor_4_torch.float32` → `..._1`, `torch_tensor_8_torch.float32` → `..._3`, ...).
- **Measured on coverage_mlp (torch_mlir route):** of 19 recorded boundaries from the capture to
  the upstream lowering, all are chained (each `before` is the previous `after`); 18 verify,
  including the parse/print round trips and `c-interface` (ABI only, T4); `xdsl-parse` does not,
  because the renamed constants are different opaque attributes.
- **Why it matters:** with the blobs elided (F7) a key is the only link from a constant to its
  weights, so whatever binds weights by key binds them differently after the rename. The new keys
  are fresh (no constant is renamed onto another's key), so nothing is swapped within the module.
- **Status:** open. Fix candidates: a fresh resource storage per parse, or weights bound as
  arguments (the FX route), which also resolves F7.

