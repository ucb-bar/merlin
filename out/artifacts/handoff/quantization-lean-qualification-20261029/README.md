# Supported Q/DQ lowering qualification — 2026-10-29

## Verdict and source cut

This work started on `feat/pass-verif-integration` at `4ef31ee8df695a3b8a0846ed6d21f37c9dc4c45e`; `feat/pass-verif-dce-cse` at `3ca03a23f64611dc9cefd91ebc2b5e4deeda844c` is an ancestor/reference. The source-binding implementation is committed at `1cef644ff`; its exact publication source hashes are in `source-bound-v4/source-cut.json`. No specification, `SPEC_LOCK`, software pin, gate registration, or approval flag changed. The built checker SHA-256 is `b590d08ed116f5bba7ccfa24ba5a42caf7c1d6e76147a08ad4a4c2d277c3c4a3`; `SPEC_LOCK` is `6cc97b6a0742795378cca76393642932b62cc9f8998693f3961f22a5bc44b23b`.

**The supported tensor-stage lowering refinement is established relative to Merlin's proposed Q/DQ source model, on declared input types at 64-bit index width. Historical PyTorch capsules remain unbound. The implementation follow-up below supplies fresh, narrow source-correspondence checks; it does not approve the source specification or prove universal IEEE fidelity.** The compiler's `all` selection still contains only DCE, CSE, ordered unrolling, ordered segmentation, and private buffer reuse. `lower_quant_ext` is not an automatic compiler gate on this branch; the unchanged test expecting it to stop compilation remains a recorded failure.

## Implementation follow-up: fresh capture source binding

`src/merlin/targetgen/qdq_capture.py` checks original and prepared graph calls against parsed emitted MLIR. It compares complete ordered arguments, original/prepared tensor identities and types, scalar literals, overloads, bounds, axes and complete returned values. Scalar scales must retain their exact Python value through f32 materialization; quantizer scales must additionally be positive with a finite f32 reciprocal. A lossy scale is refused with its original and materialized hexadecimal values. Dequantization does not acquire the quantizer's reciprocal or positive-scale restriction. This first source-correspondence profile contains one static root function of the three supported Q/DQ operations, splatted literal parameters and return. Other calls, TorchAO sources, other overloads and unbound historical graphs explicitly refuse. Runtime per-channel f32/i32 qparams remain symbolic typed inputs rather than substituted samples.

`qdq_capture_identity.py` checks the imported producer owner, the unchanged software pin, every tracked producer package member and absence of extra Python modules. It compares the installed Torch payload with the actual hash-pinned PyTorch 2.10.0+cpu wheel, including native libraries; a version string or an installation's own RECORD is insufficient. Sources and installed payloads are reopened at completion. The record binds the checker implementation too. These are source observations in the trusted operator process, not adversarial execution attestation or a complete interpreter/system-dependency closure; `source_closure_verified` remains false.

The existing capture worker exposes `--qualify-qdq --qdq-torch-wheel <wheel>` with explicit `--m2m-dir`. Failure writes `qdq-source-check.json`, preserves the actual MLIR where available, writes `meta.json` with `ok=false`, and returns 3. Without this explicit option, historical capture behavior is unchanged. This flag performs source correspondence, not automatic compiler-gate registration or numerical-policy admission.

The reproducible driver is tracked source at `build_tools/scripts/qualify_qdq_capture.py`. It retains original ExportedPrograms, exact inputs/native references, original/prepared traces, producer/package inventories, emitted MLIR and file hashes. Source replay loads and executes the retained original graph, compares its call arguments/types with the recorded graph and recomputes source correspondence. Lowering and replay use those emitted source bytes directly. Every driver phase rechecks its exact source cut, including lowering, exporter, receipt glue and SPEC_LOCK.

```sh
# Prepare an isolated checkout at the unchanged existing pin; do not repoint another worktree.
git -C out/build/lean/model2mlir-public worktree add --detach \
  "$PWD/out/build/lean/qdq-producer-905d2c9" 905d2c918f839596c9d1198cbdeb42d4c12a325b
export MERLIN_QDQ_CAPTURE_PYTHON="$PWD/out/build/lean/torch-venv/bin/python"
export MERLIN_QDQ_PRODUCER="$PWD/out/build/lean/qdq-producer-905d2c9"
export MERLIN_QDQ_TORCH_WHEEL="$PWD/out/build/lean/torch-2.10.0+cpu-cp312-cp312-manylinux_2_28_x86_64.whl"
export MERLIN_QDQ_CAPTURE_DEPS="$PWD/out/build/lean/model2mlir-public-deps"
# Fetch the wheel URL in lean-torch-requirements.txt if absent. The driver verifies its pinned SHA.
# The selected dependency directory contains xDSL 0.68.0, structlog 26.1.0,
# PyYAML 6.0.3 and safetensors 0.8.0, alongside the pinned Torch environment.
export PYTHONPATH="$PWD/src:$MERLIN_QDQ_PRODUCER:$MERLIN_QDQ_CAPTURE_DEPS"
export MERLIN_LEAN_SDK_BIN="$PWD/out/build/lean/elan/bin"
export MERLIN_COMPILER_PYTHON="$PWD/out/build/lean/mlir-venv/bin/python"
export MERLIN_LEAN_REQUIRED=1
qdq_output="$PWD/out/artifacts/verification/qdq-source-binding-new"
"$MERLIN_QDQ_CAPTURE_PYTHON" -B build_tools/scripts/qualify_qdq_capture.py capture \
  --producer "$MERLIN_QDQ_PRODUCER" --torch-wheel "$MERLIN_QDQ_TORCH_WHEEL" --output "$qdq_output"
"$MERLIN_QDQ_CAPTURE_PYTHON" -B build_tools/scripts/qualify_qdq_capture.py replay-source --output "$qdq_output"
# Use Merlin's interpreter for the existing tensor-stage checker/exporter.
PYTHONPATH="$PWD/src" .venv/bin/python -B build_tools/scripts/qualify_qdq_capture.py lower \
  --sdk "$MERLIN_LEAN_SDK_BIN" --compiler-python "$MERLIN_COMPILER_PYTHON" --output "$qdq_output"
PYTHONPATH="$PWD/src" .venv/bin/python -B build_tools/scripts/qualify_qdq_capture.py replay-lowering \
  --sdk "$MERLIN_LEAN_SDK_BIN" --compiler-python "$MERLIN_COMPILER_PYTHON" --output "$qdq_output"
PYTHONPATH="$PWD/src" .venv/bin/python -B -m pytest -q merlin/tests/targetgen/test_qdq_capture.py
```

Fresh publication capture evidence is under `source-bound-v4/`; `source-bound-v1/`, `source-bound-v2/`, `source-bound-v3/` and the failed test attempts remain as earlier evidence. V4 was regenerated after formatting and moving conversion/refusal sidecar writers into their owning helpers to retain the worker's 1500-line ceiling. The v2 lowering attempt explicitly failed because textual materialization used an absent environment-selected compiler despite the supplied `--compiler-python` argument. The driver now binds that argument to the textual helper as well as the receipt exporter, and a native regression runs lowering/replay without inherited compiler-selection variables. Regenerate into a new directory when any bound source changes. Generated payloads and logs remain local under `out/`; tracked build tools and regression sources regenerate them. The original handoff's `qualify.py`, `audit_qdq.py` and `source_probe.py` are retained locally as historical sources; their tracked successors are `qualify_qdq_lowering.py`, `audit_qdq_lean.py` and `probe_qdq_scalar_scale.py` under `build_tools/scripts/`.

Final follow-up qualification:

- `test_qdq_capture.py`: **36 passed, zero skips** in 149.66 s (`pytest-source-binding-v7.log`). Includes real pinned capture/replay, eight lowering/receipt replays without inherited compiler-selection variables, actual worker acceptance/refusal, precision and operand-binding mutations, dirty/untracked/foreign producer sources, and wrong-wheel/changed-installed-payload controls.
- Existing capture pin, failure reporting, quantization manifest, concurrent cache and toy target tests: **48 passed, zero skips** (`pytest-capture-integration-final.log`). These are integration tests, not new numerical coverage.
- `source-bound-v4/capture-summary.json` and `replay-source-summary.json`: four supported source captures matched and replayed, including execution of each retained original ExportedProgram. The retained precision counterexample was explicitly refused both times. `original.pt2`, `inputs.pt` and `reference.pt` are hash-bound; original/prepared graph arguments and result types are rechecked. Only serialization's newly populated **output-node metadata** is ignored; complete output operand identities and all producing-call result types remain mandatory.
- `source-bound-v4/lower-summary.json` and `replay-lowering-summary.json`: **eight actual boundaries**, all strict refinements verified on fresh replay. The four dequantization chains also have verified source definedness. The two quantizer and two Q→DQ composition chains retain overall `abstracted` because quantizer definedness remains unproved. Per-channel textual lowering retains its dedicated rewrite label.
- The tracked audit driver rechecked **34** pinned Q/DQ/typed-chain claims with only approved axioms (`audit-qdq-claims-followup.json`). The tracked scalar probe reproduced **50 versus 49** and `ZeroDivisionError` (`source-probe-followup.json`). The tracked lowering driver replayed all six original v4 receipts (`replay-v4-followup.log`).

The failed new attempts are retained in `pytest-source-binding-v1.log`, `v3.log`, `v4.log`, `v5.log` and the earlier source-bound directories. They exposed unsupported unregistered-op bookkeeping, missing operator registration before PT2 reload, serialization-enriched output metadata, the missing original snapshot for explicitly qualified already-quantized graphs, and the compiler-selection propagation bug. The first standalone source-binding attempt also found missing PyYAML in the selected dependency environment; PyYAML 6.0.3 was installed explicitly before the recorded successful runs. No failure was relabeled as a proof success.

The remaining semantic work is still separate: review the narrowed source predicate and IEEE/exception assumptions; formalize quantizer definedness under an explicit conversion-safe domain; extend precision handling/proofs if arbitrary Python `.default` scales are required; and recapture or recover original artifacts for historical model claims. Refusal of unsupported captures prevents those gaps from being reported as qualified.

## Publication checks

The final formatted sources were retested together: `pytest-publish.log` records **83 passed, one skipped** in 159.39 s. That skip was the existing pinned-producer test looking at an unset `MERLIN_M2M_DIR`, rather than the separately selected Q/DQ producer. Rerunning the entire four-test pin module with `MERLIN_M2M_DIR` set to the pinned checkout gave **four passed, zero skips** (`pytest-publish-capture-pin.log`). All 84 distinct selected cases therefore executed successfully across these publication runs. The fresh v4 source and eight lowering replays bind these final source bytes.

Formatting, artifact layout, no-regex, no-target-name, no-assumed-constants, sign coverage, oracle blind spots, stage parameters, shrinking ledgers, personal-path and documentation checks passed. Generated documentation was regenerated and remained unchanged. `.gitignore` admits only this curated README and review proposal; generated evidence stays local.

The whole repository's pre-commit gate is **not clean**. Structure (16 findings), core dependency direction, wiring, overfit registration and silent-default checks fail with exactly the same diagnostics on a clean detached worktree of the remote baseline `4ef31ee8d`; see `push-baseline-*.log`, `push-baseline-comparison.json` and `push-checks.json`. The provenance gate also flags two older tracked graph/bf16 reports outside this change. These existing failures were preserved; no gate or ledger was weakened to publish this scoped change. No local Git hooks were configured in this checkout; the repository hook and relevant checks were invoked explicitly.

## Reproduce

Run from the repository root. The paths below select installed local environments; their presence is a prerequisite, not an implicit skip. `MERLIN_LEAN_REQUIRED=1` turns missing proof tools into failures. Saved receipt paths are absolute, so regenerate in a new checkout before replay.

The CPU source oracle uses `build_tools/toolchains/lean-torch-requirements.txt`, which pins the PyTorch 2.10.0+cpu wheel SHA-256 `ee40b8a4b4b2cf0670c6fd4f35a7ef23871af956fecb238fbf5da15a72650b1d`. The compiler interpreter reports `torch-mlir 20260531.828`; xDSL is 0.68.0.

```sh
export MERLIN_LEAN_SDK_BIN="$PWD/out/build/lean/elan/bin"
export MERLIN_COMPILER_PYTHON="$PWD/out/build/lean/mlir-venv/bin/python"
export MERLIN_TORCH_PYTHON="$PWD/out/build/lean/torch-venv/bin/python"
export MERLIN_LEAN_REQUIRED=1
(cd merlin/lean && PATH="$MERLIN_LEAN_SDK_BIN:$PATH" lake build merlin-lean merlin-lean-audit)
.venv/bin/python build_tools/scripts/check_lean_guard.py --lock-only
PATH="$MERLIN_LEAN_SDK_BIN:$PATH" .venv/bin/python - <<'PY'
import runpy
guard = runpy.run_path('build_tools/scripts/check_lean_guard.py')
statements = guard['pinned_statements']()
assert statements is not None and guard['check_lock'](False, statements)
PY
MERLIN_LEAN_SDK_BIN="$MERLIN_LEAN_SDK_BIN" .venv/bin/python -B build_tools/scripts/audit_qdq_lean.py
.venv/bin/python -B build_tools/scripts/qualify_qdq_lowering.py generate --output out/artifacts/handoff/quantization-lean-qualification-20261029/receipts-new
.venv/bin/python -B build_tools/scripts/qualify_qdq_lowering.py replay --output out/artifacts/handoff/quantization-lean-qualification-20261029/receipts-new
.venv/bin/python -B build_tools/scripts/qualify_qdq_lowering.py mutate --output out/artifacts/handoff/quantization-lean-qualification-20261029/receipts-new
"$MERLIN_TORCH_PYTHON" -B build_tools/scripts/probe_qdq_scalar_scale.py
.venv/bin/python -B -m pytest -q \
  merlin/tests/ir/test_lean_quantize_contract.py merlin/tests/ir/test_lean_dequant_contract.py \
  merlin/tests/ir/test_lean_dequant_c1.py merlin/tests/ir/test_lean_dequant_scalar.py \
  merlin/tests/ir/test_lean_dequant_torch.py merlin/tests/ir/test_lean_receipts.py \
  merlin/tests/ir/test_lean_receipt_session.py \
  -k 'not test_an_injected_quantizer_bug_stops_compilation'
.venv/bin/python -B -m pytest -q \
  merlin/tests/ir/test_lean_pass_gate_fuzz.py merlin/tests/ir/test_lean_dequant_definedness.py \
  merlin/tests/ir/test_lean_dequant_layout.py
```

The full contract batch is the seven files in `pytest-contract.log` and `pytest-supported.log`: `test_lean_quantize_contract.py`, `test_lean_dequant_contract.py`, `test_lean_dequant_c1.py`, `test_lean_dequant_scalar.py`, `test_lean_dequant_torch.py`, `test_lean_receipts.py`, and `test_lean_receipt_session.py`. Run them with `.venv/bin/python -B -m pytest -q merlin/tests/ir/<file> ...` and the exported variables. The supported rerun excludes only `test_an_injected_quantizer_bug_stops_compilation` with `-k 'not test_an_injected_quantizer_bug_stops_compilation'`; retain its full-run failure separately.

## Results and theorem/domain/coverage map

`lake build merlin-lean merlin-lean-audit` completed successfully (1577 jobs). The lock-only guard passed, and the separate **235 elaborated-statement plus file lock** comparison passed (`guard-statements.log`). The focused Q/DQ module audit inspected **6915 imported theorems, zero violations** (`audit-qdq.json`). All **34** selected pinned Q/DQ and typed-chain claims have only `propext`, `Classical.choice`, and `Quot.sound` (`audit-qdq-claims.json`). A full global `check_lean_guard.py` attempt was stopped during its uncached, broad Mathlib rebuild; it did not produce a full global theorem-audit verdict (`guard-full-attempt.txt`).

| Boundary/source family | Profile and principal theorem | Actual route receipts | Result |
| --- | --- | --- | --- |
| `quant_ext.quantize_per_tensor` | Static f32 input, rank-zero f32 scale, rank-zero i32/i64 zero point, full-range signed i8; `QuantizeRule.check_sound`, `QuantizeCollapse.collapse_sound`, `TypedModule.check_sound` | `receipts-v4/quantize-{xdsl,textual}` | Both typed strict refinements verified; universal conditional boundary verified; overall chain **abstracted** because source definedness is not established. |
| `quant_ext.dequantize_per_tensor` | Signed i8 to f32, scalar f32 scale and i32/i64 zero point; `DequantSite.check_sound`, `DequantPerTensor.generic_result_sound_at`, `TypedModule.check_sound` | `receipts-v4/dequantize_tensor-{xdsl,textual}` | Both typed strict refinements, source definedness, and overall chains verified. |
| `quant_ext.dequantize_per_channel` | Signed i8 to f32, axis-projected f32 scale and i32 zero point; `DequantSite.check_sound`, `DequantTensor.tensor_contract_sound`, `TypedModule.check_sound` | `receipts-v4/dequantize_channel-{xdsl,textual}` | Both typed strict refinements, source definedness, and overall chains verified. The textual route uses the dedicated `preprocess_text_textual.dequant_to_generic` rewrite (`dequantize_lowered=1`, `quant_ext_lowered=0`); the xDSL route uses `lower_quant_ext`. |

Every strict claim uses `supportCheck_sound` to establish source support on the declared typed input domain, then `chain_typed_refines`/`TypedModule.check_sound`; the optional universal `moduleCheck_sound` result does not enlarge that strict domain. The dequantization overall verdict adds `chain_typed_defined_refines` on typed, nonpoison inputs. `compose_chain` rechecks saved proof premises and bytes. `receipts-v4/generate-summary.json` and `replay-summary.json` bind all six before/after SHA-256s. Fresh replay of all six returned `refinement_verdict=verified`; four dequantization overall verdicts were `verified` and two quantization overall verdicts were `abstracted` for the stated definedness reason. The raw before/after MLIR, typed, support, universal, definedness, and chain receipts remain in `receipts-v4/`. The failed `receipts/` and `receipts-v2/` attempts remain untouched. `receipts-v3/` preserves the earlier accepted checks whose textual per-channel boundary was mislabeled as `lower_quant_ext`; the corrected v4 labels are authoritative for this handoff.

`receipts-v4/mutation-quantize-mulf/` retains a valid parsed output mutation (`arith.mulf` to `arith.addf`) against the same quantizer input. Its typed receipt is `abstracted`, and the gate reports `TypedModule.check_sound did not verify the pass`; see `mutation-v4.log` and `mutate-summary.json`.

The retained model capsule corpus has **230** per-tensor quantizers in **35** forms and **627** per-tensor dequantizers in **83** forms. The existing tests exercise each form after the production lowering, Q/DQ composition, nested quantizers, rank-zero and empty dimensions, both zero-point widths, and wrong multiply/rounding/reciprocal/clamp/subtract mutations. Per-channel coverage comes from the separate `small_llama_int8_consistent` fixture plus constructed geometry tests; the ten tracked model capsule interfaces contain no per-channel Q/DQ op. The pinned PyTorch CPU/NumPy/native-MLIR dequant oracle exercises 63 cases, 539 element results, and 49 folded compiler-body comparisons. Those are bounded tests, not a universal IEEE result.

The initial seven-file run: **116 passed, 4 failed** (`pytest-contract.log`). Three stale regressions were corrected: DCE now preserves an unregistered dead quantizer until typed lowering; per-tensor dequantization ignores range metadata. The corrected five targeted cases passed (`pytest-regressions.log`). The final seven-file supported rerun: **120 passed, one explicitly deselected** (`pytest-supported.log`, 623.40 s). The deselected test is the unregistered automatic `lower_quant_ext` gate and its original failure remains in `pytest-contract.log`. No proof-tool skip occurred.

The first additional fuzz/definedness/layout batch had **46 passed, one failed** (`pytest-composition.log`): its old DCE test expected the production pass to erase an unregistered dequantizer. The corrected regression keeps that production refusal and separately proves A9 deletion of the modeled dead source op; its isolated rerun passed (`pytest-fuzz-regression-v2.log`). The failed first correction attempt (`pytest-fuzz-regression.log`) is retained: after lowering, this particular per-channel generic still is not DCE eligible because its `arith.extsi` lacks the pass's speculatability trait. This is a conservative production-pass scope limit, not a failed Q/DQ lowering proof.

The final fuzz/definedness/layout batch: **47 passed** (`pytest-composition-final.log`, 159.64 s). No proof-tool skip occurred.

The new actual compiler-boundary mutation regression was tightened to require `GateRejected` exactly and passed again (`pytest-mutation-final.log`).

## Pinned source audit and capture binding

The pinned PyTorch 2.10.0 [`_decomposed.py`](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/ao/quantization/fx/_decomposed.py) has distinct `.default` scalar and `.tensor` overloads. Quantize `.tensor` calls `.item()` on scale and zero point, then `.default` forms a Python binary64 reciprocal, multiplies f32 input, rounds to even, adds zero point, clamps lower then upper, and converts to the signed storage type. The current Lean quantizer matches that ordered **f32-materialized parameter** reading. Per-tensor dequantization converts input to f32 before subtracting zero point, then multiplies; its range arguments are not read. Per-channel dequantization subtracts in the promoted integer type before multiplication and f32 output conversion, with range validation but no range clamp. The independent bounded oracle confirms selected finite values including wraparound. Poison handling is a Merlin model rule, not a PyTorch exception model.

PyTorch uses `torch.clamp` as one source operation. The Lean contract interprets it as lower `maximumf` followed by upper `minimumf`; equivalence for special values and NaN payloads remains a model-fidelity assumption.

Merlin's `software_pins.yaml` selects model2MLIR `905d2c918f839596c9d1198cbdeb42d4c12a325b` for **fresh** captures. At that commit, `m2m/ir/decompositions.py` maps `quantized_decomposed.quantize_per_tensor.default` and `dequantize_per_tensor.default` through `_pt2e_per_tensor_operands`, which turns scalar FX qparams into rank-zero f32/i64 tensors. It also maps `dequantize_per_channel.default` directly, while a separate TorchAO `dequantize_affine` path can emit the same per-channel IR operation. The local model2MLIR checkout is `8893a37480b4560f701089e5bf408644573a906f`, different from the fresh-capture pin. The retained capsule IR records `prov.aten = "quantized_decomposed.*.default"` for all 230/627 per-tensor operations, but the capsule interfaces contain no per-capture producer commit, PyTorch wheel hash, original scalar FX arguments, or source-closure receipt. The per-channel fixture carries only `prov.op = "dequantize"`. Thus op provenance identifies a probable overload but does not bind the historical capture producer/version or distinguish every per-channel source path.

`probe_qdq_scalar_scale.py` gives a counterexample to treating all `.default` scalar scales as f32-materialized. PyTorch 2.10.0+cpu returns **50** for `x=0.11779459565877914`, original Python scale `0.0023796888063332034`; rounding that scale to f32 first gives **49**. The model2MLIR pin's materialization and current Lean spec use the latter parameter. This does not show that any retained capsule has that scalar; it disproves an unrestricted `.default` source-equivalence claim. PyTorch also raises `ZeroDivisionError` for zero scale, while the parametric Lean value semantics do not represent that Python exception. The reviewed source domain must address both before capture-to-execution wording is used. See `PROPOSAL.md` for the remaining review path; the implementation now refuses these incompatible scalar quantizer domains.

## Assumptions, exclusions and limits

| Boundary | Assumption or exclusion |
| --- | --- |
| Source semantics | The source contracts are proposed, not human-approved. Arbitrary `.default` scalar precision and exception equivalence remain unresolved; the opt-in source check enforces the narrow scalar domain above. No exact historical capture receipt binds producer bytes, framework versions, and original FX scalars. Per-channel TorchAO provenance requires separate review and is refused by the new source check. |
| Quantizer profile | Only static f32 activation tensors, rank-zero f32 scale, rank-zero i32/i64 zero point, full `[-128,127]` signed-i8 result. Excludes half/f64 activations or scale, unsigned/sub-byte storage, other ranges/granularities, dynamic shapes, and any unconditional success or IEEE claim. |
| Dequantizer profile | Per-tensor signed i8/f32 with scalar f32 and i32/i64 zero point; per-channel signed i8/f32 with axis-matched f32 vector and i32 zero point. Excludes per-channel i64 proof, f64 scales/double rounding, unsigned inputs, negative/out-of-range axes, absent zero points, group forms, unsupported attributes, and other result precisions. The compiler may lower more forms than these theorems admit. |
| Tensor model | Floats are parametric, with exact operation keys/order and no floating algebra. Poison propagates according to MerlinSem; `tensor.empty` starts poison (A7). Linalg iteration/order and named-body interpretation use A5/A6 where relevant. Tensor bounds and assertions use A4/A1 in larger compositions. Source support is on typed inputs, not all arbitrary runtime values. |
| Purity/composition | A9 assumes exactly these three region-free Q/DQ operations only compute values, permitting DCE/CSE in the model. Python exceptions and capture-side effects are not proved absent for arbitrary parameters. |
| Trusted boundary | Lean kernel/compiled checker (T1), exporter/round trip/receipt glue (T2), MerlinSem-to-MLIR/PyTorch fidelity (T3), downstream MLIR/LLVM/runtime (T4), target kernels (T5), and concrete float accuracy (T6) remain distinct. No bufferization, LLVM, hardware, W8A8 policy, or upstream MLIR-pass theorem follows. |

No selected test was allowed to skip for missing Lean, compiler, or PyTorch tools. A standalone `mlir-opt` executable and `third_party/llvm-install/lib/libmlir_c_runner_utils.so` are absent in this workspace, so the broader `test_lean_differential.py` execution-engine corpus and upstream MLIR transformation checks were not run. The selected dequant source oracle uses the installed `torch-mlir 20260531.828` Python API and does not need that runner library. The model2MLIR pin mismatch and absent historical capture receipts are reported above rather than treated as skipped checks.

## Paper wording proposed now

> For the declared signed-i8 Q/DQ tensor profiles, a Lean-checked translation validator establishes strict refinement of actual Q/DQ preprocessing output relative to an explicitly ordered, parametric source-operation model on typed inputs. This covers `lower_quant_ext` on xDSL and the textual per-tensor route, plus the dedicated textual per-channel rewrite. Fresh bounded Q/DQ captures bind the pinned producer package, installed PyTorch wheel payload, original graph parameters and emitted IR; source correspondence checks refuse lossy scalar materialization. We replayed byte-bound receipts and tested captured per-tensor forms and selected native references. The result remains conditional on the proposed source model and its IEEE/exception assumptions; historical capture provenance, quantizer definedness and unrestricted PyTorch `.default` scalar fidelity are not established. No downstream MLIR, LLVM, hardware, or W8A8-policy correctness is claimed.

Do not call this a fully qualified PyTorch capture-to-execution theorem until the proposal's source binding and semantic review are resolved.
