# Upstream audit and cleanup scope

## Reference points

| Repository/range | Meaning |
| --- | --- |
| TVM `c4dc0c29ff81ddae688da24625a603da3b4a2c0e` → `87f07883b65ee55dbd1b93f665aaab87cd8a1e50` | Complete original fork delta from Apache v0.19.0, including five inherited ONNX fixes; 136 commits |
| TVM `fb78e0e3…` | UCB ONNX-fix base; using it as upstream would omit existing fork changes |
| TVM `6b376cc8895a83387ec38cc06ef8f7b09827b071` | Published original bringup/legacy boundary |
| Merlin `6f1acb6e60dd9ad1fc36e2db629511771c49db66` → `1bd601422fc1defc4343bacbf4227ca17e8aefef` | Original comparison branch: 77 commits, 34 changed tracked files; all in this comparison directory and its tests |
| Merlin `e45205bf90d3fe7652404fd361128a062fad9d31` | Fetched origin/main at audit time |
| Merlin `0cc289de52a10bed256f32c90cfbb54a21b3c81f` | Fetched EL4 integration; chosen delivery base, containing origin/main |

The handoff artifact is the authoritative point-in-time audit. It contains every
changed file (status, line delta, content hash, component/owner), full original
commit lists and patches, dirty-source snapshots, final commit mappings, bundle
verification and test logs. Its separate EL4 inventory covers the inherited
`origin/main..EL4` delta (2,899 rename-aware changes, or 2,906 added/modified/deleted path rows
with rename detection disabled, at the pinned refs). That large existing
integration is **not** attributed to this TVM task or silently rewritten. Its
inventory is exhaustive; its code was reviewed selectively at the model/runtime/
queue/admission boundaries, not certified line by line as a new compiler release.

## TVM findings

The complete architectural and optimization audit is in the companion TVM source,
`apps/gemmini/OWNERSHIP.md`. The fork changes more than accelerator scheduling:

- Standard ONNX import fixes, BF16 storage/casts and recent op semantics; separate
  explicit `merlin.fixed_order`/`merlin.source_preserving` arithmetic contracts.
- Torch exported-program arithmetic/QDQ/integer carriers, views/scatter/creation,
  convolution/pooling/overlap/backward and parameter/output handling.
- Generic Relax/TOPI convolution, pad, pooling and empty-reduction correctness.
- Generic TE/TIR/arith fixes for rounding, reduction initialization, min/extent,
  partitioned buffer identity, schedule copy-on-write, tensorization and legality.
- Stock MetaSchedule trace-legality rejection, with its normal algorithms retained.
- Supplied Gemmini capabilities, intrinsics, semantic recognizers, legal sketch
  families, SRAM/DMA lowering, stock-search adapters and measurement wrappers.
- Custom static graph/session export, primitive runtime, independent verification,
  package/ELF auditing and deployment tools outside stock TVM's graph executor.

The cleanup separates approximately 900 lines of source-specific ONNX arithmetic
into `_source_fidelity.py`, preserving the explicit version/domain/shape checks.
Tuning and compilation now share `program_io.import_program`; tuning accepts real
`.pt2`/Relax JSON and can inventory all tasks without hardware. The inventory
records unsupported CPU tasks instead of silently hiding them. Existing campaign
programs cannot be overwritten by a different semantic workload. Stale current
READMEs/OS-only/old tuning instructions are replaced by the new runbook; original
narratives remain explicitly historical.

## Merlin comparison: every functional area

| Files/area | Responsibility and review finding | Disposition |
| --- | --- | --- |
| `target.yaml`, host config/requirements, `check_target.py`, `build_simulator.py`, `HOST_SETUP.md` | Original pinned compiler/header/runtime selection and host bringup | Retained, explicitly legacy; new TensorIR uses its own capabilities/receipts |
| `byoc/` | Snapshot installer, primitive backend, local validation/coverage | Regression reference; installer writes `gemmini_byoc.py`, not the new `gemmini.py`; not the current tuning implementation |
| `resnet_quantized.py`, `prepare_resnet_inputs.py`, `verify_resnet.py` | Explicit calibration/quantization/BN/residual recipe, split inputs, independent full graph/reference and export | Preserved; coverage now counts contraction commands separately from newly offloaded auxiliary adds/copies; semantic JSON is saved for current compiler intake |
| `tinyllama_integer.py`, `tinyllama_session.py` | Quantized projections, explicit arithmetic/rotary choices, real tokenizer/corpus, prefill/decode routes and state checks | Preserved; short defaults are diagnostic, not EL4 full-session selection |
| `smolvla_session.py`, `prepare_smolvla_observation.py`, `diagnose_smolvla_backends.py` | Real policy/checkpoint/processor and observation provenance, source dispatch/cache diagnostics, flow/session and physical actions | Saved working changes included; dataset/full recurrence required; prior whole-policy numerical failure remains visible |
| `bf16_export.py` | Explicit source-exact and generic fused export contracts, source checks and translations | Preserved, opt-in; cannot be treated as universal BF16 arithmetic equivalence |
| `verify_session.py` | ExternalRuntimeSession checking, all routes/outputs/resets, source identity, stage export and session memory/fixture plans | Saved working changes included; adds opt-in semantic JSON handoff; explicit legacy routing for selected old tile choices |
| `package_session.py`, `build_package_session.py`, `session_memory.py` | Explicit package symbol namespaces, routed candidate state, guarded buffers, reset, all-output validation, near/absolute memory placement, existing Merlin startup/profile/readback and final ELF audits | Added after the v1 audit; small two-stage Spike positive/negative controls pass with near data and 9/13 GiB absolute regions; actual full models and FireSim capacity remain open |
| `import_models.py` | Actual full-model import inventory; native execution supplies stage inputs | Previously untracked source now included; `--save-programs` supplies real semantic input to shared TVM tuning/compile path; still import-only |
| `block_workloads.py`, `smolvla_denoise_block.py`, `verify_blocks.py` | Checkpoint-backed synthetic-input cutouts and independent integer/floating gates | Previously untracked source included, clearly bounded; not full EL4 models |
| `block_tiling.py`, `cpu_row_schedule.py`, `static_buffer_abi.py` | Legacy capacity-derived tile study; opt-in legal row scratch/serial scheduling and compact ABI specialization | Preserved as comparison helpers, no target implementation moved into Merlin core |
| `verify_blocks_rtl.py`, `prepare_firesim.py` | Block ELF/RTL verification and offline FireSim staging | Preserved; staging does not execute FireSim or implement full sessions |
| `merlin/tests/gemmini/test_tvm_*`, TinyLlama/SmolVLA tests | Numerical/refusal/provenance/session/package/coverage regression checks | Copied with their source; legacy tile test now explicitly selects legacy API; source-exact environment gates remain unchanged |
| README / AGENTS / this audit | Current execution/ownership handoff | New canonical guide; historical narrative retained separately |

No new target branches or hardware literals were added to `src/merlin`, `packages`
or shared build tools. Inherited EL4 already owns generic host transforms,
compiler/source admission, functional/timing infrastructure, public corpus/Phase 2
orchestration and model execution; those remain the reusable integration points.
The TVM fork is the trusted comparison compiler, not the generated gemmini-mlir
candidate. Their compiler grants, schedules and protected oracles stay distinct.

## Working changes and history cleanup

All original dirty source files were hashed and snapshotted before copying. The
Merlin comparison changes were ported into a new worktree on pinned EL4. The older
TVM worktree's unfinished custom-tuning work was saved separately and not merged:
its `ObservedXGBModel`, custom records and `case_studies.py` depend on the retired
controller. Standard task extraction/search/JSONDatabase now supply that role.
Useful future ideas may be ported under the current contracts, but the older
source is not discarded or presented as current.

Before cleanup, the reconstructed TVM tree exactly matched the original
`87f0788` Git tree. Final cleanup edits are separately recorded against that tip.
The curated branches group the final code into component commits with imperative
messages and preserve all original refs and measured-source history. They do not
force-push, amend published history or open a PR. Each original commit maps to
its affected final file groups in the artifact; this is a final-tree curation,
not a claim that each historical intermediate revision was correct.

## Remaining qualification and technical debt

The next agent still must close actual full-model numerical/source contracts,
qualify the static-package/session ABI bridge on complete selected models and
verify their complete sizes using its explicit absolute layout, recover the matching
FireSim driver and loader, qualify full memory spans, run complete queued sessions
and report actual E2E measurements. Unsupported CPU-only tasks are inventoried but
not tuned by the current driver. Per-function scheduling does not establish
cross-session residency. Source-specific CPU arithmetic and legacy APIs are
retained compatibility surfaces, not general upstream-ready features. The two source-exact SiLU execution checks
require AVX512 and remain unqualified on this AVX2 host. The five TinyLlama
cache/rotary/fixed-tree checks pass with the selected Transformers 5.4.0 overlay.
The exact environment and validation record is in the handoff artifact.
