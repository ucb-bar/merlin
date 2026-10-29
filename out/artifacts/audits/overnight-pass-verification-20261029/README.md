# Overnight audit and numerical contract strengthening — 2026-10-29

Branch: `feat/pass-verif-integration`. Audit range: `a829c1a66..4d28efe5c`
(30 commits). Subsequent changes are the exact working-tree sources recorded in
`strengthened-source-cut.json` and `audited-contract-sources.json`; a branch name
or an older successful receipt is not their identity.
Implementation commits: `ee99e64fb` (producer/report regression fix) and
`9a92a929f` (additive proofs and kernel regressions). The tested ledger bytes
are included in the accompanying documentation checkpoint.

The selected numerical policy remains ordered f32 accumulation. Specifications,
frozen interfaces, `SPEC_LOCK` and approval flags are unchanged.

## Findings and changes

1. **Confirmed producer/report regression in `3ff55012b`:** all half-result
   generic reductions were classified as MACCs. Valid one-input sums/products
   that the earlier producer left unchanged now raised unsupported-policy errors.
   `half-reductions/` retains the actual IR, old/current outcomes and hashes.
   Ordinary preprocessing did not invoke this optional policy and was unaffected.
   The fix recognizes multiply-add reductions, preserves conservative refusal
   for malformed candidates and leaves ordinary reductions unchanged. Public
   option, whole-output, live-delivery, fresh-replay and unrelated-output mutation
   regressions exercise both sides of the boundary.
2. **Additive Q/DQ definedness:** the new exact conversion domain gives successful,
   nonpoison source execution and transfers through the actual six-operation
   lowering. A legal failing-conversion model remains a checked counterexample
   to unconditional definedness from initialized inputs. Generic quantizer
   receipts correctly remain `abstracted` for unrestricted definedness.
3. **Additive MACC contract:** initialized per-cell f32 accumulators, independent
   operand formats, ordered writes, empty traces and defined output are proved.
   The actual accumulating generic is connected to this contract. The existing
   zero contract is an exact specialization. Broader compiler eligibility is not
   enabled by component lemmas.
4. **Known integration gap retained:** `4d28efe5c` already documents default
   `all` refusing explicit-memory pipelines at later tensor DCE/CSE coverage.
   This is a refusal/coverage gap, not evidence of unsound acceptance. It is
   outside the Q/DQ/MACC changes here.

The [review packet](../../../../docs/design/qdq_macc_numerical_contract_review.md)
contains the theorem/domain/coverage table, every additional assumption/exclusion,
separate IEEE and source-profile proposals, and proposed paper wording.
`commit-roster.json` records each commit's review scope; `commits/` preserves its
full patch. Source review plus focused tests does not qualify every new profile.

## Results and evidence

`results.json` is the final machine-readable test/check summary. Cohorts overlap;
their counts must not be added as a unique-test total.

- `build.log`: original pulled checker/library build, 3,345 jobs.
- `build-strengthened.log`: strengthened build, 3,347 jobs.
- `domains-full-import-audit.json`: **4,272 theorems audited; zero violations**.
  This is the auditor's full imported-theorem mode, separately from the existing
  MACC consumer's exact-root audit. `domains-statements.json` records all eleven
  new elaborated statements. Printed new-root axioms are subsets of
  `{propext, Classical.choice, Quot.sound}`.
- Original-cut cohorts: `host-v2.xml` (80 passed), `macc-v2.xml` (68 passed,
  two retained-artifact tests deselected), `qdq.xml` (28 passed), `kernel.xml`
  (focused unroll/mutation and untiled complete-entry/replay checks).
- Strengthened-source cohorts: `strengthened.xml`, `qdq-strengthened.xml` and
  `final-regressions.xml`: **143 passed / 2 explicit skips**, **71 passed / no
  skips**, and **10 passed / no skips**, respectively. The latter includes the concrete
  mixed-format initialized generic, six quantizer width/shape domains, failed
  conversion refusal, and two mixed-module delivery/replay/mutation cases.
- `receipts-strengthened/{generate,replay}-summary.json`: all six actual
  Q/DQ boundary pairs refine on both routes and replay freshly. Dequantization
  definedness is verified; both quantizer results retain unrestricted
  definedness as `abstracted`. `mutate-summary.json` records a rejected actual
  multiply-to-add corruption and preserves its failed receipt.
- `pinned-statements.log`: **235 pinned statements and all frozen files match
  the unchanged lock**. `pinned-qdq-axioms.json`: **34 existing Q/DQ/typed-chain
  roots** have only the standard axioms. The fresh
  `scalar-scale-counterexample.json` reproduces 50 versus 49 and `ZeroDivisionError`.
- Structure has 16 pre-existing findings after regenerating the docs index;
  core dependency direction has two and no-regex has 44 existing call sites.
  Artifact-layout has 48 existing findings; all its reported files are unchanged.
  `baseline-diagnostic-source-comparison.json` checks that every reported source
  file is byte-identical to `4d28efe5c`. No guard was weakened. Docs, doc paths,
  Ruff, diff whitespace and both frozen-file and pinned-statement checks pass.
  The separate freshness scan has valid schema with 219 drift candidates; it
  does not certify every repository document as current.
  The staged artifact-layout check passes. Provenance still flags two unchanged
  historical reports; its scan is capped at 20,000 entries, so it does not audit
  every local report. Both failure paths and the cap remain explicit in the log.

## Reproduce from these sources

Run at the repository root. The installed SDK is Lean 4.35.0; compiler Python
uses torch-mlir 20260531.828, and the source oracle is PyTorch 2.10.0+cpu. Keep the
venv executable spellings; resolving their symlinks changes the interpreter.

```sh
export MERLIN_LEAN_REQUIRED=1
export MERLIN_LEAN_SDK_BIN="$PWD/out/build/lean/toolchain/lean-4.35.0-linux/bin"
export MERLIN_COMPILER_PYTHON="$PWD/out/build/lean/mlir-venv/bin/python"
export MERLIN_CLANG="$MERLIN_LEAN_SDK_BIN/clang"
export MERLIN_TORCH_PYTHON="$PWD/out/build/lean/torch-venv/bin/python"
export MERLIN_QDQ_CAPTURE_PYTHON="$MERLIN_TORCH_PYTHON"
export MERLIN_QDQ_PRODUCER="$PWD/out/build/lean/qdq-producer-905d2c9"
export MERLIN_QDQ_TORCH_WHEEL="$PWD/out/build/lean/torch-2.10.0+cpu-cp312-cp312-manylinux_2_28_x86_64.whl"
export MERLIN_QDQ_CAPTURE_DEPS="$PWD/out/build/lean/model2mlir-public-deps"
export PATH="$MERLIN_LEAN_SDK_BIN:$PATH"

cd merlin/lean
lake build MerlinVerify merlin-lean merlin-lean-audit
lake env ../../out/build/lean/build/bin/merlin-lean-audit \
  MerlinVerify.Rules.QuantizeDefinedness MerlinVerify.Rules.Bf16F32AccInitialized
cd ../..

.venv/bin/python -B build_tools/scripts/audit_qdq_lean.py \
  --output out/artifacts/verification/qdq-contract-axioms.json
.venv/bin/python -B - <<'PY'
import runpy
guard = runpy.run_path('build_tools/scripts/check_lean_guard.py')
statements = guard['pinned_statements']()
assert statements is not None and guard['check_lock'](False, statements)
PY
"$MERLIN_TORCH_PYTHON" -B build_tools/scripts/probe_qdq_scalar_scale.py

mkdir -p out/build/numerical-contract-replay
.venv/bin/python -m pytest -q -ra --tb=short \
  merlin/tests/ir/test_lean_numerical_domains.py \
  merlin/tests/ir/test_macc_producer_legality.py \
  merlin/tests/ir/test_lean_bf16_policy_gate.py \
  merlin/tests/ir/test_compiler_macc.py \
  merlin/tests/ir/test_macc_ssa_joins.py \
  merlin/tests/ir/test_lean_bf16_f32acc_native.py \
  --basetemp=out/build/numerical-contract-replay/macc

.venv/bin/python -m pytest -q -ra --tb=short \
  merlin/tests/ir/test_lean_quantize_contract.py \
  merlin/tests/ir/test_lean_dequant_contract.py \
  merlin/tests/ir/test_lean_dequant_torch.py \
  merlin/tests/ir/test_compiler_qdq_native_source.py \
  merlin/tests/ir/test_compiler_qdq_prefix.py \
  merlin/tests/ir/test_compiler_qdq_bindings.py \
  --basetemp=out/build/numerical-contract-replay/qdq

.venv/bin/python -B build_tools/scripts/qualify_qdq_lowering.py generate \
  --output out/artifacts/verification/qdq-contract-replay
.venv/bin/python -B build_tools/scripts/qualify_qdq_lowering.py replay \
  --output out/artifacts/verification/qdq-contract-replay
.venv/bin/python -B build_tools/scripts/qualify_qdq_lowering.py mutate \
  --output out/artifacts/verification/qdq-contract-replay
```

Use a new output directory: generation refuses to overwrite retained evidence.
The native-source fixture runs the pinned capture qualifier and source replay,
checks original PT2 execution and exercises both actual compiler routes. MACC
tests perform fresh gate replay; a deserialized success status grants no authority.
The toolchain/capture bootstrap and pins are in the existing
[qualification handoff](../../handoff/quantization-lean-qualification-20261029/README.md).
Only the curated checkpoint, compact hashes/results and original regression
outcomes are tracked. Copy this local artifact tree for raw logs/proofs, or
regenerate it with the commands above. Historical source cuts are not relabeled.

## Remaining blockers and preserved failures

Independent Q/DQ source-specification and MACC numerical-policy approval remain
pending. The current compiler still refuses mixed MACC storage formats, nonzero
initialization and unsupported maps. The new initialized component proof does
not supply source recognition, destination SSA packing, a broader whole-module
reference or actual new-profile compiler acceptance. Universal IEEE/framework
fidelity and arbitrary per-site fusion are not established. W8A8 policy and
upstream transformations remain separate.

The two retained SmolVLA full-shape MACC cases require an absent historical
boundary and are explicitly skipped in the strengthened cohort. No missing
required Lean/compiler/Torch tool is counted as a pass. Standalone `mlir-opt`
and `third_party/llvm-install/lib/libmlir_c_runner_utils.so` are absent; the
broader execution-engine differential suite and upstream transformation checks
were not run. Rank-one native/compiler/replay qualification and the full legacy
repository suite were not independently rerun here.

Initial `macc.log`/`host.log` failed during pytest setup because the basetemp
parent was absent; only their `-v2` reruns count. Prototype domain-test attempts
retain missing-local-build/import errors and test-development failures; the
final tests require the specific conversion failure rather than accepting any
Lean error. Those failures and the original half-reduction counterexamples are
preserved, not relabeled as successful proof evidence.
`initial-strengthened-test-sources/` retains the initial test versions;
`final-regressions.xml` covers the later, stronger generic/mutation cases.
