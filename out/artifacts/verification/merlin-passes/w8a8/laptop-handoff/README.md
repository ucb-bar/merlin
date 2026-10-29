# W8A8 verification handoff — previous pause checkpoint

Work resumed on 2026-10-21. See the [latest progress](../README.md) for the new source-operation
adapter and numerical-domain checker; the checkpoint below describes the earlier pause.

Previous implementation checkpoint: **`3b6f4a753` on `feat/pass-verification`**, pushed to origin.
Commit `4defd2a57` added this handoff. At that pause there was no unfinished code to recover;
the full verification goal was incomplete. Do not restart old subagents: Jeremy requested one
strong agent for this scope.

## Scope and required review

Verify **Merlin's shared `lower_contraction_int8` W8A8 pass between model2MLIR and the OOT
compiler** using an executable Lean checker and kernel-checked soundness. Upstream LLVM/MLIR
passes are trusted. Do not resume upstream rewrite families, optional gather/conv/nonlinear
quantization, frontend capture correctness or OOT backend verification.

The intended numerical-policy claim is `after ⊑ P_W8A8(before)`, not equality to the original
floating computation. The source-derived reference transformer remains a **draft**. No complete
W8A8 verified receipt or paper correctness claim has been established.

Jeremy approved the explicit zero-row rule: when amax is zero, use f32 scale 1e-5 directly.
That does not approve the whole recipe, numerical domain or source semantics. Jeremy has now
nominated **Agustin** as independent reviewer. Review is pending; nobody has been contacted
through tools. Naming the reviewer is not specification approval.

Read root `AGENT.md`, `CLAUDE.md`, `docs/reference/architecture.md`, local instructions,
`merlin/lean/AGENT.md`, and design §1.2/§4.5/§5.5/§5.10 in
`docs/design/merlin_lean_verification.md`. Also read `merlin/lean/LEDGER.md`,
`merlin/lean/findings.md`, the [review packet](../policy-review/README.md) and the
[progress history](../README.md). Earlier progress entries describe earlier states; this
handoff and the latest entries supersede them.

Frozen interfaces, semantics, `Spec/`, `Claims/`, `Assumptions/`, Lake configuration and
`SPEC_LOCK` must remain unchanged without the required review. No unproved Lean, new float
law or silently narrowed claim is permitted. The implementer has read the pass and cannot
provide the independent specification authorship required by design §1.2. Agustin's review
must preserve that information barrier, using the declared recipe and versioned references.

## Recent committed work

| Commit | Change | Qualification |
|---|---|---|
| `a65b3aefd` | Build all oracle imports with the existing verified-library target, without rebuilding the receipt checker | Included in later numeric runs |
| `8c39043df` | Pure binary32 min/max and byte conversion; clamp safety connected to scalar regions, tensors and the existing checker | 31 tests; 7601-theorem audit; 801 native sample executions |
| `27b1f7d24` | Captured per-tensor source quantization candidate and pinned PyTorch oracle | 50 combined tests; 7646-theorem audit; 64 source samples plus all 89 retained TinyLlama Q parameter sets |
| `3b6f4a753` | Actual source K bound connected to exact accumulation at every mapped prefix; threshold and model reach diagnostics | 22 module tests; 7666-theorem audit; all 155 TinyLlama selected sites |

These runs had zero skipped tests and zero audit violations. Counts overlap; do not sum audits.
No production pass, frozen semantics, Lake configuration, lock or existing statement pin changed.
The receipt checker binary remained SHA-256
`dca757c297b5b7a1e5404495438983cdbf037312cb372034c115276d80cbc7e7` on this machine.

Earlier integrated work is already on this branch:

- `10b0ab7dd`, `079e3d56b`, `79c145fb5`: source-only policy interface, draft W8A8 transformer,
  sound whole-module checker, driver and byte-bound API.
- `fb02c17c3`, `af76e555c`: i32/i64 dequantization candidate; mixed operands, calibrated reuse,
  channel/zero-point refusals and native qualification.
- `f8a67f4bc`: policy/suffix composition, exact joins, typed input domains, receipts and CLI.
- `42d49d15d`, `93871b561`: proved complete AST equality, checker timeouts, actual handoff
  binding and explicitly unmeasured suffix diagnostics after unqualified prefixes.
- `3f6a103e2`, `afaab91fd`: exact conversion-domain obligations and actual-model diagnostics.

## Code entry points

Lean paths below are relative to `merlin/lean/MerlinVerify/`.

| Area | Main files |
|---|---|
| Source-derived draft policy | `Rules/W8A8Reference.lean`, `Rules/W8A8Emission.lean` |
| Whole module and suffix qualification | `Check/PolicyModule.lean`, `Check/PolicyW8A8Module.lean`, `Check/PolicyW8A8Chain.lean` |
| Exact equality and public drivers | `Check/ExactSyntax.lean`, `Driver/W8A8.lean`, `Driver/W8A8Chain.lean` |
| Integer proofs | `Rules/W8A8Accumulation.lean`, `W8A8IntegerBody.lean`, `W8A8IntegerTensor.lean`, `W8A8Matmul.lean`, `W8A8ReferenceBounds.lean` |
| Float/tensor stages and initializers | `Rules/W8A8Scalar.lean`, `W8A8FloatTensor.lean`, `W8A8AbsMax.lean`, `W8A8Fill.lean`; corresponding `Check/PolicyW8A8*.lean` files |
| Concrete byte/clamp model | `Rules/W8A8ByteFloat.lean`, `W8A8ByteModel.lean`, `W8A8ByteTensor.lean`, `Check/PolicyW8A8ByteDefined.lean` |
| Uninstalled source-contract candidates | `Rules/W8A8DequantWidth.lean`, `W8A8SourceByte.lean`, `W8A8SourceQuantize.lean` |

Python integration: `src/merlin/verify/lean/{policy,check,cli}.py`. Test-only execution:
`merlin/lean/Differential.lean`; its host arithmetic is **not** a proved IEEE interpretation.
Explicit modes `w8a8_byte_model`, `w8a8_dequant_width` and `w8a8_source_quantize` qualify
candidates without installing frozen dispatch or issuing verified receipts.

## Exact remaining work

1. Obtain independent policy/source-contract/domain review through Agustin. The concrete
   [packet](../policy-review/README.md) is ready. Do not flip `policy_reviewed`, add assumptions
   or update the lock to bypass review.
2. Integrate the reviewed captured-source contract through the approved semantics and
   support/definedness interfaces. Current frozen dispatch has narrower signed-i8/i32/f32
   dequantization and no quantize semantics. The i64 DQ and Q candidates are additive proof/test
   modules only; their existence does not close whole-program source support.
3. Justify and check the claimed numerical domain. Current `FloatModel` has no laws and its
   `toInt` may return `none` for any bits. Initialization plus named clamps does not prove
   conversion success for every such model. Concrete byte proofs require explicit non-NaN
   intermediates: rounded product for dynamic Q, post-zero-point-addition value for source Q.
   These conditions are not yet established for arbitrary complete-model inputs.
4. Qualify actual-model source support/definedness, then replay the actual suffix through the
   bound OOT compiler handoff. Both retained model suffixes are unmeasured. Do not remove or
   weaken `Qualified` gates to obtain a passing receipt. SmolVLA also has a dynamic source type
   outside the current static domain.

The source Q candidate follows PyTorch 2.10: extract scalar f32 scale and i32/i64 zero point;
compute scalar reciprocal in binary64; convert to f32; multiply; roundeven; add converted zero
point; clamp lower-first to -128/127; cast signed i8. It checks finite positive scalar scales.
Do not replace the reciprocal with f32 division, remove literal-zero addition, change clamp
order or invent NaN-to-zero behavior without a separately justified contract. Its `run_defined`
theorem proves candidate execution returns typed, initialized output under the explicit domain;
it does not establish universal `FuncDefined` for an entire captured model.

The previously suggested numerical-domain task is now implemented in
`Check/PolicyW8A8SourceDomain.lean`, with actual-operation integration in
`Rules/W8A8SourceContract.lean` and `Check/PolicyW8A8Source.lean`. The important remaining
distinction is between these runtime instance checks and all-input program proofs; the existing
whole-policy theorem has not been narrowed or its gates weakened.

## Actual models and discrepancies

TinyLlama's complete W8A8 function conditionally matches the draft, including all 155 changed
contractions. Every source-derived K bound passes: 133 sites use K=2048 and 22 use K=5632.
The conservative full-i8 limit is K < 131072; a larger K is refused, not declared to overflow
for every possible operand value.

TinyLlama contains 310 per-tensor DQ and 89 per-tensor Q operations. All 399 scale/zero-point
parameters are constants through scalar splats: finite positive f32 scales about 0.0004902
through 1.4986, literal i64 zero points. DQ metadata includes -127/127; Q uses -128/127 and
`output_dtype = "int8"`. All 89 Q parameter sets agree across Lean, pinned PyTorch and separate
native MLIR on 1424 synthetic scalar observations. This does not execute the complete model.

SmolVLA's retained captured-scope W8A8 boundary is unchanged with **zero selected W8A8 sites**.
Its first support refusal is dynamic `tensor.empty`, result 1033 of type `tensor<?xi64>`,
dynamic extent operand 1029. An unchanged boundary is not evidence of active policy coverage.

Open findings in `merlin/lean/findings.md`, with production passes unchanged:

- **F11:** generic contraction recognition checks operation presence without proving the
  multiply/add wiring. Valid squaring/wrong-wiring sources can be rewritten; the draft refuses.
- **F12:** i64 per-channel DQ subtracts after float conversion rather than integer promotion
  and subtraction. q=1, zp=16777217, scale=1 produces -16777215 in production versus -16777216
  in pinned PyTorch and candidate semantics.
- **F13:** calibrated per-channel reuse does not check zero-point-zero and the source axis.
  Valid wrong-axis and nonzero-zero-point cases are refused by the reference. TinyLlama's
  observed per-tensor zero-point-zero reuse is not a replay of these witnesses.

Do not fix production passes in passing. Such fixes need separate reviewed changes/regressions.

## Laptop setup and evidence

Proofs, source, tests and Markdown are in Git. **Generated JSON, MLIR, JUnit logs, captured
model boundaries, virtual environments and Lean builds under `out/` are ignored and will not
arrive through `git pull`.** Transfer needed evidence or regenerate it. Do not claim old or
unavailable evidence as a fresh laptop run. Receipts contain absolute paths and tool hashes:
regenerate them on the laptop rather than editing paths or trusting stored verdict strings.

Known tools used here:

- Lean `leanprover/lean4:v4.35.0`; Mathlib revision
  `56a005855ea17e0b0958f73ec801ae140b0a1679`, pinned in the repository.
- Compiler/exporter environment: `torch-mlir 20260531.828`.
- Isolated source oracle: **PyTorch 2.10.0+cpu, NumPy 2.4.6**; scripts enforce both versions.
- Local capture environment: TorchAO 0.18.0/PyTorch 2.14.1. Earlier reference fixture:
  TorchAO 0.16.0/PyTorch 2.10.0+cu128. Neither establishes retained captures' producer version.

Portable outline from a clean checkout already on `feat/pass-verification`; confirm the branch
before pulling and supply actual laptop tool paths:

```sh
git pull --ff-only origin feat/pass-verification
export PATH="$HOME/.elan/bin:$PATH"
# In merlin/lean: lake exe cache get
# In merlin/lean: lake build MerlinVerify merlin-lean merlin-lean-audit
export MERLIN_LEAN_REQUIRED=1
export MERLIN_M2M_PYTHON=/path/to/compiler/environment/bin/python
export MERLIN_TORCH_PYTHON=/path/to/pinned/source-oracle/bin/python
export MERLIN_MLIR_RUNNER_UTILS=/path/to/libmlir_c_runner_utils.so

.venv/bin/python -m pytest -q \
  merlin/tests/ir/test_lean_w8a8_source_quantize.py \
  merlin/tests/ir/test_lean_w8a8_byte_model.py \
  merlin/tests/ir/test_lean_w8a8_dequant_width.py
.venv/bin/python -m pytest -q merlin/tests/ir/test_lean_w8a8_module.py
.venv/bin/python build_tools/scripts/check_lean_guard.py
.venv/bin/python build_tools/scripts/check_artifact_layout.py
.venv/bin/python build_tools/scripts/check_structure.py
```

Use unique retained artifact directories and JUnit output for qualification. The exact 50-test
run additionally selected four `test_lean_differential.py` cases: float arithmetic/rounding,
format/integer conversions, unsupported-model refusal and no-native-poison execution.
`lean_oracle` builds the existing `MerlinVerify` library target. Finalize the receipt checker
before receipt tests; rebuilding during them invalidates evidence. Do not resolve the compiler
virtual-environment Python symlink.

Machine-specific recovery paths under `/home/jeremy/Projects/merlin`:

- Toolchain: `out/build/lean-shared-passes/phase7-worktree/out/build/fresh-environment/toolchain`.
  Here `ELAN_TOOLCHAIN` and `LEAN_SYSROOT` pointed there, with `LEAN_NUM_THREADS=2`.
- Compiler Python: `out/artifacts/verification/lean-review/20261017T204007Z/integration/out/build/lean/mlir-venv/bin/python`.
- Source oracle: same prefix, `torch-venv/bin/python`.
- Runner: `out/build/lean-differential/libmlir_c_runner_utils.so`.
- Actual captures, under this W8A8 artifact root: `model-boundaries/20261020T215617Z/`.
  TinyLlama preparation id: `55b1a0827d7445abb67966a3c8c40195`; SmolVLA:
  `3d2dd40bbd9740f1868470172e4d0aa5`. Read stage before/after/meta and `handoff.json`.

Final generated evidence names under the W8A8 root:

- `byte-model-qualification.json`, `byte-model-qualified-v3.xml`, `byte-model-qualified-v3-guard.log`.
  Earlier v2 failed a temporary Lake configuration experiment; configuration was fully restored.
  Only v3 is final qualification for that batch.
- `source-quantize-qualification.json`, `source-quantize-qualified.xml`,
  `source-quantize-qualified-guard.log`, `source-quantize-captured-parameters/qualification.json`.
- `reference-bounds-qualification.json`, `reference-bounds-qualified-v2.xml`,
  `reference-bounds-qualified-guard.log`, `reference-bounds-{tinyllama,smolvla}.json`.
- `smolvla-dynamic-source-inventory.json`, `tinyllama-quant-{source,parameter}-inventory.json`.
- `model-policy-identity-qualification/` and `policy-identity-qualification.json`: retained
  **abstracted**, handoff-bound model receipts; both actual suffixes remain unmeasured.

Ruff, whitespace, artifact layout and relevant source-layout/access checks pass. Structure's
one pre-existing unrelated failure is `src/merlin/runtime/backends/zephyr_model.py`, 3557 lines
against a 3537 ceiling. Do not change it in this task.

Commits/pushes are authorized on this branch; a PR is not authorized. Fetch/review every
unpublished commit and diff before pushing; confirm the remote is an ancestor; never force-push.
No new assumptions or frozen-file changes are authorized by this handoff.
