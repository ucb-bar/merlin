# Handoff: bool-to-float and affine checkpoint

Stopped at the user's request after committing the implementation and qualifying the checkpoint.
This file covers only this work, not the earlier float-to-bool/runtime checkpoint.

## Checkout and commits

- Worktree: `/tmp/merlin-pv-frontend`; branch: `pv/frontend`.
- Implementation: `1838049e4c76d3e9013f12f82f45a2ec0b759304`.
- Evidence/report follow in a separate commit on the same branch. No merge into main.
- Main checkout, production passes, frozen definitions/specs, locks, shared Main/CLI and W8A8
  sources were left unchanged.

## Completed work

**`fix_bool_sitofp`, f32 only:** `BoolToFloatProgram.check_sound_ieee` proves acceptance of the
authoritative before/after AST implies complete-program execution of A2's bool-to-f32 intent.
False is positive zero, true positive one, poison stays poison. Eligibility, operands, types,
flags, SSA freshness, replacement uses, nested regions and calls are recomputed in Lean.
The two-case float premise is discharged with Lean's existing IEEE `Float32.Model.ofNat`.
`LocalPassesChain.bool_sound` preserves that policy through checked ordered suffixes.

The actual public `M4_smolvla_denoise_gemmini` capsule has two repaired sites, both accepted in
its complete program. The older full `SY_model_smolvla` has zero sites and is not counted as an
accepted repair. Production also handles f16/bf16/f64; these formats explicitly refuse here.
No unchanged model computation's initialization or numerical correctness is newly claimed.

**`lower_torchao_affine_quant`:** a provisional source-driven materializer checks the complete
actual emitted graph, including declaration removal and surrounding calls/regions. A separate
scalar theorem proves its exact reciprocal/multiply/roundeven/clamp/convert order. The complete
program theorem is against that materialization, **not against execution of the captured opaque
TorchAO calls**. Receipts say `proved_materialization_provisional` and
`source_call_refinement_proved=false`. Do not relabel them as full pass verification.

The Gemmini reference files/recipes are pinned to `feat/el4-experiment` commit
`0cc289de52a10bed256f32c90cfbb54a21b3c81f`. Selected static grouped-16 recipes, historical r53b
captures and the tested dynamic per-token recipe remain distinct. No selected capture receipt was
available; the positive affine boundaries are explicitly synthetic actual-pass executions.

## Qualification

- 33 qualification tests and 39 existing regression tests passed, with no skips.
- One nonfinite test was rerun after clarifying its documentation; it is not an additional distinct test.
- Full Lean audit: 10,502 theorems, zero violations; audited roots use only standard Lean axioms.
- Primary oracles: Torch 2.10.0+cpu; TorchAO 0.15.0 and 0.16.0 CPU primitive implementations.
  Newer Torch bool results are supplemental. Exact package-source and compiler binary hashes are retained.
- Compiler-valid mutations reach and are refused by Lean. Invalid SSA is a compiler refusal.
- The Lean guard retains three pre-existing lock/admission failures; the structure guard retains
  the pre-existing `zephyr_model.py` line-ceiling failure. Neither was hidden or repaired here.

The 596-capsule census found 857 `quant_ext` sites outside the existing narrow proof:
230 per-tensor quantizers, 295 restricted-range per-tensor dequantizers with i64 zero points,
and 332 full-range per-tensor dequantizers with i64 zero points. No inventoried public capsule
contains the opaque TorchAO calls. Census counts are not proof facts supplied to Lean.

## Next work

Read [REMAINING.md](REMAINING.md) and [TORCHAO-REVIEW.md](TORCHAO-REVIEW.md) first.

1. Obtain the actual selected `model.mlir`, `capture_receipt.json`, recipe bytes/hash and scales.
   Do not assign a version from a model name or the installed compiler environment.
2. Resolve/admit the provisional reference contract and its invalid-cast domain. Pinned CPU
   nonfinite casts returned zero, but PyTorch 2.10 documents invalid numeric casts as platform-dependent.
   Those observations are neither a universal zero contract nor a proved compiler counterexample.
3. Implement/prove the captured-call → independent tensor-reference execution bridge, covering
   row reduction/layout, poison, parameter addressing, declarations and calls.
4. Discharge the explicit `AbsReductionCorrect` / `ZeroPointElisionCorrect` obligations and
   primitive clamp/rounding/conversion correspondence for the reviewed concrete model/domain.
   `elision_not_parametric` proves the zero elision cannot hold for every frozen `FloatModel`.
5. Keep unsupported layouts, symbol overloads, output widths and formats explicit. Both pinned
   TorchAO versions raise a TypeError for the tested empty quantization; Lean now refuses empty extents.

There is engineering work beyond human review. No new contract has been self-approved.

## Files and commands

- Lean rules: `merlin/lean/MerlinVerify/Rules/{BoolToFloat*,LocalProgramCorrespondence,LocalPassesChain,TorchAOAffine*}.lean`.
- Separate driver/build: `merlin/lean/LocalPassesCandidate.lean`, `LocalPasses.lakefile.toml`.
- Receipt/replay API: `src/merlin/verify/lean/local_passes.py`.
- Qualification: `merlin/tests/ir/test_lean_bool_to_float.py`, `test_lean_torchao_affine.py`.
- Intake/receipt generator: `build_tools/scripts/qualify_local_passes.py`.
- Evidence collector: `build_tools/scripts/collect_local_passes.py`.

```sh
cd /tmp/merlin-pv-frontend/merlin/lean
PATH=/home/jeremy/.elan/bin:$PATH lake -f LocalPasses.lakefile.toml build merlin-lean-local-passes
```

In the current container, `/tmp/frontend-recovery/env.sh` sets the compiler/LLVM/pinned Torch paths.
The additional oracle paths are read-only inputs:

```sh
export MERLIN_TORCHAO_015_PYTHON=/home/jeremy/Projects/merlin/out/build/lean-w8a8-torchao-reference/torchao-0.15/bin/python
export MERLIN_TORCHAO_016_PYTHON=/home/jeremy/Projects/merlin/out/build/lean-w8a8-torchao-reference/torchao-0.16/bin/python
export PYTHONDONTWRITEBYTECODE=1
```

Run tests with `-o tmp_path_retention_policy=all` to retain native/mutation evidence. The default
project setting deletes successful test directories. `receipts.json` indexes the published
records. Replay recomputes compiler round trips, theorem audits and actual Lean acceptance.
On another checkout, regenerate receipts against its build; do not edit old identity hashes.

[qualification.json](qualification.json) and [README.md](README.md) contain the detailed evidence
and supported scope. `integration.patch` contains the additive implementation commit only.
[PUBLISHED-EVIDENCE.json](PUBLISHED-EVIDENCE.json) lists the committed evidence and omitted
regenerable copies. The large zero-site `SY_model_smolvla` fixture is not bundled; its outcome and
byte-bound census are retained. Both official receipts include all immutable MLIR/export stages.
