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
- **Fix (2026-10-16, separate compiler correction):** emit `extsi` → `subi` → `sitofp` → `mulf`,
  PyTorch's exact order; the rule then needs no float assumption and no input-domain restriction.
  This applies to the signed-i8/i32-zero-point per-channel case. PyTorch v2.10.0's **per-tensor**
  implementation instead converts before subtraction; do not apply this change to every granularity.
- **Status:** fixed for signless i8 storage interpreted as signed, i32 zero points and f32 scale/result
  in both the xDSL and textual per-channel lowerings. Other datatype/precision variants remain
  outside this proof scope.
  The per-tensor arithmetic order is retained. The scalar and tensor proofs cover poison and all
  bit patterns, and the source support and C1 site rules now connect accepted whole functions.
  Tests retain the historical floating-subtraction body as a rejected negative control and retain
  its numeric counterexample alongside the corrected actual pass output. Source-contract review
  and binding to the captured producer remain outstanding for paper qualification.

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
- **Measured on coverage_mlp (torch_mlir route):** the 18 recorded boundaries from the capture to
  the upstream lowering form one chain (each `before` is the previous `after`); 17 verify,
  including the parse/print round trips and `c-interface` (ABI only, T4); `xdsl-parse` does not,
  because the renamed constants are different opaque attributes. The composed chain
  (`compose_chain`) is therefore `abstracted`, for this rename and for F7.
- **Why it matters:** with the blobs elided (F7) a key is the only link from a constant to its
  weights, so whatever binds weights by key binds them differently after the rename. The new keys
  are fresh (no constant is renamed onto another's key), so nothing is swapped within the module.
- **Status:** open. Fix candidates: a fresh resource storage per parse, or weights bound as
  arguments (the FX route), which also resolves F7.

## F9 — W8A8's zero-row guard is not an identity of the abstract float model

- **Observed (2026-10-20):** the declared scale in design §5.5 is
  `max(amax / 127, 1e-5)`. Merlin additionally selects `1e-5` when `amax == 0`.
  TorchAO v0.16.0's `quant_primitives.py` floors the computed scale without that select.
- **Formal evidence:** `Rules/W8A8Scale.lean` proves `guarded_eq_iff`, characterizing the exact
  identity required for agreement, and `guard_counterexample`, an axiom-free countermodel to
  unconditional agreement under the existing unrestricted `FloatModel`. In that deliberately
  non-IEEE model, arithmetic returns zero and comparison returns true: the unguarded scale is
  zero, while the guarded scale is the floor's bit pattern.
- **Meaning:** this is a specification/model obligation, not an observed IEEE numerical defect.
  A checker of the emitted scalar body can prove execution of the guarded formula with no extra
  float law. It cannot silently equate that formula with a different reference expression.
- **Status:** the implemented guarded scalar body and its checked literal captures are proved.
  The user explicitly approved including the zero-row rule in the policy (2026-10-20). The
  guarded recipe therefore needs no extra float law. Full specification review remains separate;
  no assumption, policy lock, frozen semantics, or compiler pass was changed.

## F10 — available W8A8 references and models use different versions and recipes

- **Observed (2026-10-20):** `merlin/tests/data/torchao_affine/PROVENANCE.json` records TorchAO
  0.16.0 and PyTorch 2.10.0+cu128. The local `model2MLIR/.venv` instead reports TorchAO 0.18.0
  and PyTorch 2.14.1. The latter's dynamic W8A8 configuration defaults to version 2.
- **Models:** the checked-in `SY_model_smolvla/capsule.interface.mlir` records
  `int8_dyn_act_int8_weight`; `SY_model_tiny_llama/capsule.interface.mlir` records
  `int8_static_act_int8_weight`. TinyLlama's README describes the dynamic scheme, so its recorded
  IR and YAML must take precedence when inventorying the actual boundary.
- **Meaning/status:** the general proof must state its policy and supported paths explicitly.
  A version or recipe from one fixture cannot establish the meaning of another capture. Model
  sources have been captured under the default paper configuration: TinyLlama's W8A8 stage
  changes and SmolVLA's does not. Their presence does not restrict the universally quantified
  theorem to those models or establish that their complete boundaries verify.

## F11 — W8A8 generic recognition does not establish multiply/add wiring

- **Observed:** `lower_contraction_int8` recognizes a floating generic contraction by the presence
  of `arith.mulf` and `arith.addf`, then emits a signed-byte product of the two input operands.
  It does not check whether the original multiplication actually consumes those two inputs.
- **Valid source witness:** a canonical-map reduction whose body multiplies its first argument
  by itself, then adds the accumulator. Both xDSL and the compiling MLIR verifier accept this
  generic. The real pass rewrites it as an ordinary two-input integer contraction.
- **Meaning:** an operation name list cannot justify source contraction semantics. The reference
  policy/checker must check actual body wiring, types, order, maps and initialization before
  accepting a selected site. Rounding or quantization error cannot justify replacing a square
  with a product using another input.
- **Evidence:** `test_lean_w8a8_module.py`,
  `test_unsupported_source_cannot_choose_its_target_as_reference[wrong_source_body]`, checks the
  real rewrite and authoritatively exports both programs, then requires reference expansion and
  whole-function matching to refuse this source.
- **Status:** the new source-derived draft refuses this case. The production pass is unchanged;
  this is not reported as a compiler fix or as whole-pass verification. The observed TinyLlama
  boundary uses named canonical matmuls and is not a replay of this witness.

## F12 — per-channel i64 zero points still subtract after float conversion

- **Observed:** `lower_quant_ext` uses integer promotion/subtraction for signed-i8, i32-zero-point,
  f32 per-channel dequantization, but the equivalent i64 case falls through to separate f32
  conversions followed by floating subtraction. PyTorch v2.10.0 subtracts the integer tensors
  first, promoting signed bytes to the zero point's width.
- **Executed witness:** signed byte `1`, i64 zero point `16777217`, f32 scale `1`. Pinned CPU
  PyTorch and the additive Lean source-contract candidate yield `-16777216`; actual Merlin
  output, executed through native MLIR, yields `-16777215`. Both results are finite. No overflow,
  NaN, invalid input, or unregistered operation in the executed output is needed.
- **Evidence:** `test_lean_w8a8_dequant_width.py`,
  `test_actual_i64_channel_lowering_still_has_integer_promotion_discrepancy`. The candidate is
  separately compared against pinned PyTorch and standard-MLIR arithmetic for both widths,
  per-tensor/per-channel order, integer endpoints, signed zero, scalar tensors and empty output.
- **Status:** open; the production pass is unchanged. The candidate does not change the frozen
  source dispatcher or qualify complete boundaries. TinyLlama's observed W8A8 reuse has per-tensor
  zero-point-zero dequantization; it is not a replay of this per-channel witness. Source-contract
  review and a separate compiler fix remain distinct obligations.

