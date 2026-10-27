**VERDICT: LITE restored in code; full verified shadow replay is blocked by the supplied capture state, independently of the removed record/campaign gates.**

Branch `lite-restore` starts at C10 (`99d9cd549fec5ce63c1b276afa509c86146e6b18`). No LITE mode switch was added. Ordinary Phase 0 again derives from the selected specification, RTL facts and application captures through Agustin's flow. The 10,353-case record and fusion campaign are not required by derive, synthesis verification, ordinary generation, freezing or release preparation. The diagnostic operation ledger accepts no record.

The shadow's FP32 sealed receipts are rejected by the preserved issuer guard (`pending M2M receipt has an unsupported policy`); C10 produces the same rejection. Seven of thirteen int8 captures are absent. These are replay/input limitations, not record/campaign gates. See the replay results below before treating this as a ready-to-seal release.

Work ran on compute-2 with expensive commands under `nice -n 19`, numerical library threads limited to one, and test/replay processes sharing an affinity of at most six CPUs. The source shadow was mounted read-only at its original absolute path in a temporary bubblewrap namespace; no shadow receipts, inputs or source repositories were edited.

The restoration uses per-file historical checkouts, not a whole-tree reset. `608f3939b` is not an Agustin baseline: as the supplied `/scratch/loren/j/blame/REPORT.md` establishes, it already contains Loren's October 24–26 work. All needed ancestry exists in this clone; no upstream fetch was needed. A pre-gate parent may still include older Loren work. The table identifies its last path edit rather than mislabeling every pre-gate byte as Agustin-authored.

Removed integration and retained fixes:

- `ed9bdc69d`: removed `--spec-validation`, derive-time bind, selected-input revalidation of the record, and descriptor `phase0_spec_validation: required`. Restored the pre-refusal `core_aten_stage`, runner and callers. Deleted `run_spec_validation.py`, `make_spec_validation_selection.py`, `spec_validation.py`, `evidence_attestation.py` and their feature-only tests.
- `563e995e7`, `22d6a2776`: restored ordinary generation, writer, profiles, corpus synthesis and direct CLI before composition insertion/campaign sampling. No automatic composition corpus, bound payload verification, required-regime campaign denominator, fusion placement check, or fusion-generated count is injected into ordinary Phase 0.
- `3c5df3690`, `ada561538`, `22d6a2776`, `0c080c83d`, `b68f01f9f`: restored the original fact-capabilities/resolve-spec flow; removed scheme/composition resolution from that flow and deleted `software_spec_resolution.py`. Restored the bounded runner and Core ATen selector before the shared identity glue. Retained older source-admission code and independent quantizer/issuer guards.
- `2f1596e61`: removed fusion-private-exclusion input freezing and its capsule-shelf alias. Kept `freeze.py` target-data selection (`9ccc851af`), execution tool selection (`8b693dffe`) and frozen probes (`cef761385`), plus their `source_snapshot`/launcher support. Provider-owned semantic probes (`b0322f2ff`) and the cpuinfo guest fixes are also unchanged.
- Restored files retain/reapply rank-zero inventory (`b9771bab1`), conformance output-path fallback (`4d7f7111a`), unresolved performance families (`27334036a`), multi-output goldens (`d71ad55cd`), recipe/issuer nonfinite/complex guards (`6332db937`, `88dba42f2`), quantization prose/review handling (`155965792`, `47fff4a08`, `87b5e2d95`), runtime-weight opt-in (`739a369c3`, `06c2c6506`, `f48209636`), Bool/narrow fills (`9e78dce1a`), W3 complex ABI (`d1be4fda9`), and native/frontend complete-output readers (`0803a551b`). `5ba2f5989`'s zero-versus-None synthesis budget fix was minimally reapplied after restoration.
- All production changes from the 4+3 fusion/generation series (`8b6a0b8d9`, `99d9cd549`) remain in the retained advisory modules/golden readers, including the ledger alias fixes (`4c3caab30`, `91f0abcc1`). C10 itself adds only two test files; its production fixes are already in the preceding commit.

The new regression checks exercise a missing record/campaign with a descriptor that still says `required`, while retaining rejection of changed recipe bytes; they also exercise a capture-only operation ledger after deleting the record file. No historical receipt is promoted to fresh verified capture admission.

Files intentionally retained despite their Loren origin:

`fusion_enumeration.py`, `fusion_cases.py`, `fusion_reference.py`, `fusion_execution.py`, `fusion_validation.py`, `composition_generation.py`, `composition_corpus.py` and `composition_capsules.py` are still imported by the supplied kit, independent runtime-scale diagnostics, compiler helpers or golden readers. Their explicit standalone campaign APIs still validate their own inputs. They have no automatic caller from the restored derive/generate/freeze flow. Keeping them preserves the seven requested fixes and avoids breaking those consumers. Quantization scheme/composition declaration APIs in `quantization_spec.py`, `software_spec.py`, `quant_methodology.py` and `quant_recipe.py` likewise remain for those consumers; they no longer introduce a Phase 0 record/campaign dependency. The optional diagnostic ledger retains its schema/digest reader for an explicitly supplied advisory record and preserves composition aliases; omission does not manufacture passing cases or validation authority.

The upstream input-byte bindings, complete application roster/materialization checks, software review, native numerical checks, capture issuer/recipe guards, instruction/placement checks, budget, hidden disjointness and operator release review remain. Explicitly requesting a standalone validation API still invokes its validation; this is not a Phase 0 prerequisite.

Per-path provenance follows. Full requested `git log --format='%h %an %ad %s' --date=short -- PATH` histories are committed in [PROVENANCE.txt](PROVENANCE.txt). Each table entry gives the exact pre-feature blob and its last edit. Recover the complete historical file with `git show COMMIT:PATH`; “absent” means the file was newly introduced by our series. These are inspection baselines; retained advisory modules were not blindly checked out to a nonexistent file.

Prefixes: `P0/` = `packages/merlin-experiments/src/merlin_experiments/phase0/`; `EX/` = its parent; `TG/` = `src/merlin/targetgen/`; `CC/` = `src/merlin/compile/`.

| Path | Loren feature entry commits | Pre-feature state: blob, lines | Last path edit there |
|---|---|---|---|
| `build_tools/scripts/make_spec_validation_selection.py` | ed9bdc69d | absent at `00c49e99f` | (file absent) |
| `build_tools/scripts/run_bounded_core_aten_suite.py` | ed9bdc69d; b68f01f9f | `91f0abcc1:2d70bf0dd`, 687 | db32d2eac Loren Hung 2026-10-26 fix(runtime): execute only byte-pinned provider code under a guard |
| `build_tools/scripts/run_spec_validation.py` | ed9bdc69d | absent at `00c49e99f` | (file absent) |
| `docs/compiler/phase0_functional_truth.md` | see history | `00c49e99f:6d5a2653f`, 28 | 8968f5be1 Loren Hung 2026-10-23 Merge core-aten into feat/megaphase |
| `docs/guides/generating_capsules.md` | see history | `7d19aef70:a56587bb5`, 374 | 8968f5be1 Loren Hung 2026-10-23 Merge core-aten into feat/megaphase |
| `docs/guides/target_resolution.md` | see history | `7d19aef70:18ecd7d0c`, 250 | 8968f5be1 Loren Hung 2026-10-23 Merge core-aten into feat/megaphase |
| `examples/gemmini/target/descriptor.yaml` | ed9bdc69d | `00c49e99f:c40e4cca5`, 240 | fb322b6e6 Loren Hung 2026-10-24 feat(targetgen): declare constrained schemes and composition truth |
| `packages/merlin-experiments/README.md` | f12c1cd37 | `49cea13b9:1a3f59094`, 183 | 8968f5be1 Loren Hung 2026-10-23 Merge core-aten into feat/megaphase |
| `EX/cli.py` | ed9bdc69d | `00c49e99f:caa7f4bae`, 947 | 8968f5be1 Loren Hung 2026-10-23 Merge core-aten into feat/megaphase |
| `P0/__main__.py` | 22d6a2776 | `7d19aef70:5b486331e`, 162 | 078417c6a Agustin N. Coppari Hollmann 2026-10-11 feat: defer performance preparation for functional phase zero runs |
| `P0/composition_generation.py` | 563e995e7; 22d6a2776 | absent at `5f7753ba6` | (file absent) |
| `P0/core_aten_stage.py` | ed9bdc69d | `00c49e99f:2770eaae5`, 545 | 68f531fd2 Loren Hung 2026-10-23 feat(phase0): freeze Core ATen lane decisions as placement records |
| `P0/evidence.py` | 22d6a2776; 3d277c99c; 0c080c83d; b68f01f9f | `60a18aa46:57028ac4c`, 1469 | 8968f5be1 Loren Hung 2026-10-23 Merge core-aten into feat/megaphase |
| `P0/evidence_attestation.py` | 1489d84be | absent at `0c080c83d` | (file absent) |
| `P0/freeze.py` | 2f1596e61 (fusion inputs); retained 9ccc851af/8b693dffe/cef761385 | `8b693dffe:f00e46cd3`, 582 | 8b693dffe Loren Hung 2026-10-27 fix(experiments): freeze phase zero execution tool selections |
| `P0/fusion_cases.py` | f12c1cd37 | absent at `49cea13b9` | (file absent) |
| `P0/fusion_execution.py` | 7d19aef70 | absent at `f12c1cd37` | (file absent) |
| `P0/fusion_reference.py` | f12c1cd37 | absent at `49cea13b9` | (file absent) |
| `P0/fusion_validation.py` | f12c1cd37 | absent at `49cea13b9` | (file absent) |
| `P0/generation.py` | 563e995e7; ed9bdc69d; 22d6a2776 | `5f7753ba6:c5e20eff4`, 1518 | 377f1b8a4 Loren Hung 2026-10-24 Merge el4-core-aten fixes into megaphase-core-aten |
| `P0/operation_coverage_ledger.py` | ed9bdc69d; 4c3caab30; 91f0abcc1 | absent at `00c49e99f` | (file absent) |
| `P0/profiles.py` | afaeb8176; ed9bdc69d; 22d6a2776 | `5f7753ba6:649dc62cd`, 1253 | 8968f5be1 Loren Hung 2026-10-23 Merge core-aten into feat/megaphase |
| `P0/requirements.py` | ed9bdc69d | `00c49e99f:52b6a6cd5`, 1106 | 377f1b8a4 Loren Hung 2026-10-24 Merge el4-core-aten fixes into megaphase-core-aten |
| `P0/spec_validation.py` | ed9bdc69d | absent at `00c49e99f` | (file absent) |
| `P0/writer.py` | 563e995e7 | `5f7753ba6:7ba2086d3`, 1105 | 073058ca9 Agustin N. Coppari Hollmann 2026-10-21 feat(integration): integrate experiments and tests changes |
| `EX/runner.py` | ed9bdc69d | `00c49e99f:eb770f069`, 1522 | 8968f5be1 Loren Hung 2026-10-23 Merge core-aten into feat/megaphase |
| `EX/source_snapshot.py` | 46e0a7920 | `2f1596e61:b1c8a7b2a`, 817 | 9ccc851af Loren Hung 2026-10-27 fix(experiments): preserve frozen target data selection |
| `CC/fusion_enumeration.py` | 49cea13b9 | absent at `1ccc5a55f` | (file absent) |
| `TG/composition_capsules.py` | 563e995e7 | absent at `5f7753ba6` | (file absent) |
| `TG/composition_corpus.py` | 563e995e7; 22d6a2776 | absent at `5f7753ba6` | (file absent) |
| `TG/core_aten_spec_int8.py` | b68f01f9f | `91f0abcc1:8c19413ef`, 1284 | 14efc9c79 Loren Hung 2026-10-26 feat(lowering): prove and trace runtime f32 readout scales |
| `TG/corpus_synth.py` | 563e995e7; 22d6a2776 | `5f7753ba6:fcebe5d52`, 2618 | 377f1b8a4 Loren Hung 2026-10-24 Merge el4-core-aten fixes into megaphase-core-aten |
| `TG/quant_methodology.py` | 3c5df3690 | `60a18aa46:658ae01fc`, 558 | 00eeb0d0e Agustin N. Coppari Hollmann 2026-10-18 feat(integration): integrate experiments and tests changes |
| `TG/quant_recipe.py` | 3c5df3690 | `60a18aa46:764edb1cd`, 1146 | 9bd82ffb0 Loren Hung 2026-10-24 fix(quantization): default integer sum recipes to exact |
| `TG/quantization_spec.py` | 3c5df3690; afaeb8176; 14efc9c79 | `60a18aa46:6323adb7d`, 613 | 8968f5be1 Loren Hung 2026-10-23 Merge core-aten into feat/megaphase |
| `TG/software_spec.py` | 3c5df3690; 22d6a2776 | `60a18aa46:0bfc26d84`, 774 | 8968f5be1 Loren Hung 2026-10-23 Merge core-aten into feat/megaphase |
| `TG/software_spec_resolution.py` | b68f01f9f | absent at `91f0abcc1` | (file absent) |
| `TG/spec_fact_drift.py` | 3c5df3690; ada561538; 22d6a2776 | `60a18aa46:c029a80c4`, 933 | 8968f5be1 Loren Hung 2026-10-23 Merge core-aten into feat/megaphase |
| `TG/target_experiment.py` | afaeb8176 | `ed174a2be:2a490a439`, 1496 | fb322b6e6 Loren Hung 2026-10-24 feat(targetgen): declare constrained schemes and composition truth |

Validation: **908 passed, 44 skipped, 22 failed, 2 errors** over 976 final cases. C10 controls have the same 22 failing and two erroring shared cases. There are **zero new failures across 967 shared cases**; both new LITE regressions pass. Seven historical retained-stage cases also reappear with the upstream test version. Automatic scheme/composition/record integration tests were restored/deleted with their interfaces (103 C10 case identities are absent), rather than treating those removed features as still mandatory. The zero-budget regression found in the first restoration pass was repaired before this final run. [TEST-RESULTS.md](TEST-RESULTS.md) lists every failing case, the exact file selection, interpreter/source bootstrap, test ceiling and artifact locations. This is not a green-all-tests claim. Independent 4+3 advisory golden/source fixes remain in their production modules.

Shadow replay results, with no record and no campaign supplied:

| Stage | Result |
|---|---|
| Verified sealed FP32 derive | Refused at the preserved capture issuer-policy guard. C10 has the identical `pending M2M receipt has an unsupported policy` failure before either record binding or inventory. |
| Capture-only FP32 diagnostic derive | **Completed** using all nine application and four performance captures. It inventoried **10,031 application operations** and wrote `requirements.yaml`, both demand/plan sidecars, `synthesis.yaml`, the evidence bundle and a **62-candidate plan** (56 `derived_sweep`, six `model_derived`, zero compositions). `requirements.yaml` has no `spec_validation` key. |
| Recipe selection | **Completed** from that newly derived evidence: one `int8__int32` recipe, file SHA-256 `3dc2d9f5b69391f6b5095cdb9c82aa2a1c67eef02e0edcea02fd5edf2e8e479f`; semantic recipe SHA-256 `8712732a0e7ad81dfa11928043f3eea8c0f21866ebb39869ed8f8d44b97c5bab`. Its derived `runtime_weight_contractions` is `true`. No recipe bytes were hand-authored. |
| Subsequent int8 derive | Attempted after recipe selection and refused because `causal_decoder/capture/model.mlir` is absent. Seven of thirteen int8 model files are missing. No record/campaign check was reached or required. |
| C10 capture-only FP32 control | Refused explicitly at `workload_spec.phase0_spec_validation: required, but no spec-validation record was supplied`. |

New replay products are at `/scratch/loren/restore1/replay/captures-fp32/`; the selected recipe pointer and pin are `replay/generated-recipe.path` and `replay/generated-recipe.json`. Logs, replay source and exact invocation are in `/scratch/loren/restore1/artifacts/replay*.{py,log}`. A separate recipe calculation from the shadow's old saved FP32 evidence was also diagnostic; it is not substituted for the newly generated recipe above.

The capture-only result is explicitly `diagnostic` and application coverage remains `unverified`; it does not upgrade rejected sealed receipts or prove generation, numerical admission, prepared counts or sealing. The seven missing int8 labels are `causal_decoder`, `causal_decoder_decode_step`, `core_aten_coverage_reduce`, `core_aten_coverage_movement`, `core_aten_coverage_contraction`, `decoder_scale`, `decoder_step_scale`. The new recipe also differs from the shadow's old recipe (including the requested runtime-weight opt-in), so existing int8 captures cannot be assumed to match it. A complete fresh accepted int8 roster is needed for a verified end-to-end release.

The kit change is orchestration-stage skipping and removal of references to skipped outputs; it needs no replacement derivation algorithm or Merlin bypass switch. The supplied kit is outside this clone and was not edited.

1. Keep setup/backend/facts, the exact workload roster, fresh sealed capture issuance, recipe derivation, W3 checks and ordinary generate/coverage/prepare/hidden/seal/review stages. Point its source revision at this branch's final commit.
2. Skip the fusion campaign/bind/preflight/selection stages and `01-evidence.sh`'s selection, 10k run, regression and bind-record stages. `02-release.sh` should require completed setup, omit `check record`, and remove `--spec-validation "$RECORD"` from both derive commands. Keep `derive-fp32 -> generated recipe -> capture int8 -> derive`.
3. In `kit.py:binding()`, skip only the record-binding assertions and the composition-campaign policy assertions. Keep `verify_selected_synthesis`, selected-byte/roster checks and plan reconciliation. Its restored upstream signature does not take the record-gate-only `rtl_facts`, `capability_contract`, `hardware_spec` verification arguments; omit those three arguments from this call. `seal_descriptor()`/`seal4_fusion.check_composition_policy()` must likewise stop demanding `phase0_spec_validation: required` or a synthesis switch.
4. Keep the operation ledger if desired, calling `prepare` without `spec_validation`/`spec_validation_sha256`; it remains diagnostic. Skip any campaign-regime seal-blocker check on that advisory ledger. Counts must describe the actual generated/prepared corpus; omit the hardcoded `core_aten_evidence_rows=10353` when that stage did not run.
5. Keep `sealcheck()`'s original source-run, lineage, coverage, evidence, frozen-input, synthesis, numerical admission and digest checks. Its calls to `generation()`/`binding()` then have no record/campaign dependency. In `review_commitments()`, omit the skipped RECORD, SELECTION and regression files and `record_file_sha256`; keep all corpus, capture, W3, hidden, input and review-digest commitments. Skip the matching record/regression-only requirements in `review_request()`, `review_transitions()` and `summary()`.

Merely omitting the top-level 10k shell invocation is insufficient: the old kit reads those output files again in its binding, ledger, counts, review and summary helpers. The edits above skip those same advisory stages at their downstream call sites while preserving the upstream release checks. Fresh captures remain necessary where the shadow does not provide an accepted complete roster.

Implementation commit: `3a1fbd875` (`Restore upstream spec-and-capture Phase 0 derivation`), authored by Loren Hung <lorenhung@berkeley.edu>, no trailers. The report/provenance commit follows it on `lite-restore`. The deliverable `../merlin.patch` is the complete diff from C10, including these reports.

Diff size versus C10: **1,052 lines added, 5,618 removed** across all deliverables in the patch; production/runtime/scripts/example configuration only: **109 added, 2,437 removed**. The implementation commit before report/provenance additions is 168 added, 5,618 removed across 41 files. The remaining diff is reporting material. Patch application against an independent C10 export was checked; `git diff --check` passes.
