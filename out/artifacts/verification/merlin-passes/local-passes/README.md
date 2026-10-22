# Pass-local frontend checkpoint

Work is isolated in `/tmp/merlin-pv-frontend`, branch `pv/frontend`. Production passes,
frozen definitions/specifications, `SPEC_LOCK`, shared Main/CLI and W8A8 files are unchanged.

| Boundary | Formal result | Admission |
| --- | --- | --- |
| `fix_bool_sitofp`, scalar i1 → f32 | Complete before/after program execution matches the declared repair policy, including nested regions, internal calls, poison and ordered suffixes | Proved; contract/instance admission remains provisional |
| `lower_torchao_affine_quant` | Complete target program matches a source-derived materialization; actual scalar quantizer body computes the stated ordered expression | Provisional materialization/component evidence; **captured-call refinement is not proved** |

## Bool-to-float theorem

`BoolToFloatProgram.check_sound_ieee` establishes, for every context, function and input:

```text
Eval(original program with eligible casts interpreted by A2 boolToF32)
  = Eval(actual after program)
```

Both sides use `BoolToFloat.model(base)`: Lean 4.35's existing IEEE `Float32.Model.ofNat`
interprets precisely default `arith.uitofp`, width 1, result f32. All other float operations
retain their arbitrary base interpretation. The two conversions are proved by kernel reduction:
false → bits `0x00000000`, true → bits `0x3f800000`. Poison remains poison. No float law or axiom
is added. This intentionally repairs the original signed-i1 true → −1 behavior.

The checker reads authoritative exported IR. It recomputes source types/eligibility, attributes,
operand identity, the unsigned conversion key, absence of `nonNeg`, result mapping, SSA freshness,
every use, nested region correspondence and function lookup/calls. It accounts for every operation
and preserves module metadata. Source nesting beyond 64, multi-block regions, unsupported source
attributes or unrelated changes refuse. The source policy invokes A2 directly; it does not execute
or consult the target materialization. Its explicitly documented off-domain extension to integers
of other widths supports exact failed-outcome correspondence without a global typing hypothesis.

`LocalPassesChain.bool_sound` composes this repair with every supplied suffix boundary under the
same interpretation. It retains the repair-policy claim; it does not assert equivalence with the
original signed cast. Unchanged computations may remain unsupported by the evaluator. Their exact
correspondence is proved; their initialized execution is not asserted.

### Format and artifact coverage

| Production case | This checkpoint |
| --- | --- |
| Scalar i1 → f32, including scalar bodies of tensor operations | Proved |
| Scalar i1 → f16 / bf16 / f64 | Production fires; authoritative valid examples explicitly refused by Lean |
| Other opaque/non-f32 formats | Unsupported; no format-wide claim |
| Direct tensor cast / already unsigned `arith.uitofp` | Not a production repair site |

The fixed census scanned **596** public capsule interface files; 20 contain relevant casts,
`quant_ext` operations or opaque-name text. `M4_smolvla_denoise_gemmini` has the **two** applicable
f32 sites, both checked in its complete actual before/after program. The older full
`SY_model_smolvla` capsule has **zero** applicable sites and is reported as not applicable.
The census is an inventory, not a proof fact supplied to Lean.

## TorchAO scope and version binding

[Reference intake](reference-intake.json) binds the exact files and recipe hashes from
`feat/el4-experiment` at `0cc289de52a10bed256f32c90cfbb54a21b3c81f`. The selected grouped-16
**static** recipes are distinct from the older **dynamic per-token** affine recipe.
The historical guide identifies Torch 2.10.0, TorchAO 0.15.0 for SmolVLA and 0.16.0 for
ResNet50/TinyLlama, model2MLIR `905d2c918f83`, and its own historical recipe digest.
Those statements do not identify the version of a different capsule or synthetic program.

The selected generated `model.mlir` / `capture_receipt.json` pair was not available in the searched
local artifact stores or committed branch files. The published affine qualification is therefore
explicitly **synthetic**, with the dynamic recipe stated in the intake, rather than falsely bound
to a selected capture. None of the inventoried public capsules contains these opaque TorchAO calls.
Current selected recipe digests are:

| Model | Recipe digest |
| --- | --- |
| ResNet50 | `b840ae04104798799670015e39d2a5be692a70bc2a6ada42eb75d6878dd2ffbd` |
| SmolVLA | `cf31d473d2907e6c7c0a31b0c3101c6b5f1c8005ea6f476eb6ab77b48ac8bfad` |
| TinyLlama | `4300d9d2f46160cab679672d384bdda664fd310862137f354b36c2845f45ed5f` |

The provisional Lean materializer supports positive static unencoded f32 per-token tensors,
rank-reduced f32 scales, i8 quantized results and the two unsuffixed captured callee names. It checks
the complete emitted graph, including declaration removal, constants, maps, iterator kinds,
arithmetic keys, provenance, all uses, surrounding operations, nested regions and calls. Keepdim
layouts, numeric symbol overloads, other output widths/formats and empty extents refuse.

`TorchAOAffineScalar.body_sound` proves the actual scalar evaluator computes reciprocal first,
multiply, half-even round, lower clamp, upper clamp, signed conversion. Conversion failure is poison.
`checkBody_reference` exposes `ZeroPointElisionCorrect` as an explicit hypothesis for the reference's
additional +zero. `LocalPassesChain.torchao_sound` proves execution of the **materialized** contract
through all supplied suffixes. It does not interpret the original opaque calls by the reference.

### Remaining obligations

1. Bind the **actual selected capture** to its capture receipt, Torch/TorchAO/model2MLIR versions,
   recipe bytes/digest and stored scales. The module's scheme string alone is insufficient.
2. Obtain independent review of the proposed contracts and model instances. The earlier f32
   float-to-bool approval does not approve these new claims. This agent has not self-approved them.
3. Finish the captured-call → independent tensor-reference execution bridge, including row
   reduction/layout, poison, parameter addressing and declaration/call correspondence. This is
   remaining engineering work, beyond the completed materialization correspondence.
4. Discharge `AbsReductionCorrect` and `ZeroPointElisionCorrect` for a reviewed concrete model and
   domain, and justify the reference primitive mapping, including clamp, half-even rounding and
   integer conversion. The frozen float model supplies no such laws. `elision_not_parametric` proves a
   countermodel: +zero elimination is not valid for every frozen `FloatModel`. No sample proves
   either universal law, and no assumption has been admitted as an axiom.
5. Resolve the reference-domain issues described in [the review proposal](TORCHAO-REVIEW.md):
   nonfinite conversion and empty tensors. PyTorch 2.10 documents non-fitting float-to-integer
   conversions as undefined/platform-dependent; observed CPU zeros are not a universal contract.
   A finite-domain or explicit undefined-behavior claim must be declared and justified;
   it cannot be inferred from matching tensor types or silently substituted for the requested policy.
6. Extend only the explicitly unsupported production cases after their contracts are justified.
   This checkpoint does not verify the whole TorchAO pass or every `lower_quant_ext` variant.

### Missing `lower_quant_ext` variants actually inventoried

[The byte-bound capsule census](capsule-census.json) records **857** sites in available public
capsules, grouped into 118 exact signatures. None has a bound selected-capture version here.
The existing narrow dequantization proof does not cover these observed variants:

| Observed variant | Sites | Missing obligation |
| --- | ---: | --- |
| f32 quantize_per_tensor, scalar f32 scale, i64 zero point, range −128…127 | 230 | Quantization source contract, scalar/tensor/site proof |
| i8 dequantize_per_tensor → f32, scalar f32 scale, i64 zero point, range −127…127 | 295 | i64 zero point and restricted-range admission |
| Same dequantization, range −128…127 | 332 | i64 zero-point contract and execution proof |

Production also handles per-channel/per-group dequantization and bf16 widening/truncation variants
exercised by the existing tests. Those are distinct from the observed signatures above; absence
from this public-capsule census does not establish absence from the unavailable selected captures.

## Qualification and replay

[qualification.json](qualification.json) records exact test counts, oracle identities, audit and
guard results. Native tests are execution evidence, separate from Lean proofs. Mutated outputs are
parsed and verified by the authoritative compiler before Lean rejects them. Undefined SSA tests
are recorded as compiler refusals, not Lean mutation successes.

[PUBLISHED-EVIDENCE.json](PUBLISHED-EVIDENCE.json) inventories the published evidence by hash.
Large duplicate copies of the older zero-site `SY_model_smolvla` fixture and regenerable compiler
scratch files are omitted. Every immutable stage of both official receipts is included.

Primary numerical oracles use **PyTorch 2.10.0+cpu** and **TorchAO 0.15.0 / 0.16.0** CPU primitives,
with exact package-source hashes retained. Newer PyTorch bool results are supplemental. Cases
include signed zeros, fractions, ordinary values, ties, reciprocal/division-sensitive inputs,
epsilon floors, rank-3 tensors, NaNs and infinities. Nonfinite TorchAO outputs and empty-reference
exceptions are retained as domain evidence; MLIR poison is not treated as a defined native number.

The independent candidate executable and API leave shared Main/CLI unchanged:

```sh
cd merlin/lean
lake -f LocalPasses.lakefile.toml build merlin-lean-local-passes
```

```sh
PYTHONPATH=src python build_tools/scripts/qualify_local_passes.py --receipts \
  --compiler-python /path/to/compiler/python
```

The script regenerates the pinned reference intake and capsule census, applies the real passes,
exports/verifies/round-trips the exact boundaries, audits theorem statements/dependencies, runs
Lean and replays the receipts. [receipts.json](receipts.json) indexes the retained records.
`local_passes.replay` rejects changed source/compiled-object identities, artifacts, proof scope,
results, reference bytes and suffix ordering. On another checkout, regenerate records against its
build instead of relabeling the paths/hashes of an old record.

To retain native/mutation artifacts when rerunning tests, override the repository's default
failed-test-only retention:

```sh
python -m pytest merlin/tests/ir/test_lean_bool_to_float.py \
  merlin/tests/ir/test_lean_torchao_affine.py -o tmp_path_retention_policy=all
```

Set `MERLIN_COMPILER_PYTHON`, `MERLIN_M2M_PYTHON`, `MERLIN_TORCH_PYTHON`,
`MERLIN_TORCHAO_015_PYTHON`, `MERLIN_TORCHAO_016_PYTHON`, the LLVM tools and
`MERLIN_LEAN_REQUIRED=1`. Missing qualification tools fail this required run.

The Lean guard still reports the three pre-existing unadmitted frontend/schema lock differences.
The structure guard still reports the pre-existing `zephyr_model.py` size ceiling. Neither lock
admission nor unrelated cleanup is part of this checkpoint.
