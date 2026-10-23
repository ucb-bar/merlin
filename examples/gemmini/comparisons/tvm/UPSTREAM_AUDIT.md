# Upstream audit and cleanup scope

## Reference points

| Repository/range | Meaning |
| --- | --- |
| TVM `c4dc0c29ff81ddae688da24625a603da3b4a2c0e` → `87f07883b65ee55dbd1b93f665aaab87cd8a1e50` | Complete original fork delta from Apache v0.19.0, including five inherited ONNX fixes; 136 commits |
| TVM `fb78e0e3…` | UCB ONNX-fix base; using it as upstream would omit existing fork changes |
| TVM `6b376cc8895a83387ec38cc06ef8f7b09827b071` | Published original bringup/legacy boundary |
| Merlin `6f1acb6e60dd9ad1fc36e2db629511771c49db66` → `1bd601422fc1defc4343bacbf4227ca17e8aefef` | Original comparison branch: 77 commits, 34 changed tracked files; all in this comparison directory and its tests |
| Merlin `e45205bf90d3fe7652404fd361128a062fad9d31` | Fetched origin/main at audit time |
| Merlin `0cc289de52a10bed256f32c90cfbb54a21b3c81f` | EL4 delivery base and current fetched EL4 tip on 2026-10-23; integration code authority for the re-audit |
| Merlin `db2f52415fe9f7d2cd7f028e122f8c9b039a925c` | User's earlier workload reference; selected protocol, three base recipes and committed input definitions are unchanged at current EL4 |
| Merlin handoff `0074fd0af707345c60e59150cf66e67968a2f703`; TVM handoff `5b4c36da713234e48a0deb35365b42a26c196707` | Compiler/runtime sources tested in the re-audit; subsequent handoff edits are documentation only |

The EL4 pull required no merge: the handoff already contains EL4, 13 commits
ahead and zero behind before this documentation correction. The TVM delivery
also matched its fetched published branch. No merge conflict was present.
Fresh updates should follow the README's fetch/review/merge procedure and record
new source hashes; this audit does not silently follow moving branches.
The existing [PR #53](https://github.com/ucb-bar/merlin/pull/53) is a draft from
`reed-nicolas:comparisons/tvm-gemmini` into `main`, not this EL4 handoff branch.
GitHub reports it mergeable, with no reviews; its blocked merge status does not
indicate a source conflict. No open PR targets this handoff branch. Preserve
that older delivery rather than mixing its 75-commit history into this audit.

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

The earlier preliminary delivery added a generic capture option in
`src/merlin/targetgen/_m2m_capture_worker.py`: `--save-exported-program` serializes
the exact retained prepared Torch program and records its hash, including each
session stage. It introduces no target names, numerical policy or TVM dependency
into core. A real two-stage round-trip test checks exact outputs and source-weight
mutation refusal. The comparison factory gained TinyLlama 16+1 support, and the
bounded ResNet input preparer admits the protocol's 260 calibration images.
The current [README](README.md) supersedes that preliminary workload selection:
all three models, TinyLlama 32+32 and SmolVLA's complete 12-call chain, as defined
by the pinned EL4 protocol. It distinguishes committed inputs and specified
recipes from the shared frozen payloads and full-model qualification still needed.
A new numerical release can be prepared without locating historical captures;
it must retain its own identity. Old factory recipes remain explicitly distinct.

The companion TVM update ships nine stock MetaSchedule winner records, matched
byte-for-byte against retained winner traces and successful measurement costs.
`--tuning-preset rtl-v1` verifies capabilities/payloads and reports structural
coverage. The database loader registers all stored workload families, including
those absent from the current model; it does not generate needless sketches for
unmatched input functions. There is no model-name dispatch or new search method.

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
| `read_package_session.py` | Existing Merlin OUT_BIN/UART validation, exact storage decoding and per-invocation NPZ export for the shared offline checker | Added after the v1 audit; requires a matching session policy and complete outputs; does not establish execution provenance, numerical agreement or timing |
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
verify their complete sizes using its explicit absolute layout, bind the actual
FireSim driver and loader, qualify full memory spans, run complete sessions
and report actual E2E measurements. Unsupported CPU-only tasks are inventoried but
not tuned by the current driver. Per-function scheduling does not establish
cross-session residency. Source-specific CPU arithmetic and legacy APIs are
retained compatibility surfaces, not general upstream-ready features. The two source-exact SiLU execution checks
require AVX512 and remain unqualified on this AVX2 host. The five TinyLlama
cache/rotary/fixed-tree checks pass with the selected Transformers 5.4.0 overlay.
The exact environment and validation record is in the handoff artifact.

## Current EL4 integration re-audit

The newer EL4 code already supplies capabilities absent from the earlier pinned
audit. Do not describe these as missing implementations:

- [Native complete-chain images](https://github.com/ucb-bar/merlin/blob/0cc289de52a10bed256f32c90cfbb54a21b3c81f/docs/reference/chain_images.md)
  package retained program objects, preserve computed recurrent state and check
  every output. Compact mode embeds independently admitted references and emits
  comparison statistics after testing every element; it is not checksum-only
  numerical validation. Native per-step counters exclude input transfer and
  descriptor preparation, so they do not alone measure complete E2E latency.
- [Strict manager provenance](https://github.com/ucb-bar/merlin/blob/0cc289de52a10bed256f32c90cfbb54a21b3c81f/examples/gemmini/validation/manager-provenance.md)
  implements versioned ELF/request staging, an unsent locked SSM plan, actual
  runtime boot-file observation, complete compact transcript intake and recorded
  execution/cleanup outcomes. Installation and actual deployment evidence remain
  separate. The supplied publisher requires compact mode even though earlier
  preparation can admit a full-array chain; that combination is not an end-to-end
  supported strict workflow.
- [Lightweight boot-file provenance](https://github.com/ucb-bar/merlin/blob/0cc289de52a10bed256f32c90cfbb54a21b3c81f/examples/gemmini/validation/booted-elf-provenance.md)
  uses the installed direct helper with an actual runtime-host boot-file hash,
  matching simulator argv and full compact transcript. It needs no strict-manager
  upgrade and produces a separately labeled reproducibility observation.

TVM's existing complete-session bridge has a different package ABI, session
manifest and UART protocol. It cannot directly supply the native chain build
evidence or compact transcript expected by these paths. Its current all-array
readback still needs a manager manifest adapter for legacy transport. Add this
adapter in the existing comparison/runtime boundary, with independent reference
and source bindings, rather than manufacturing native receipts. Also add stage
timing to the existing whole-session warm/measured window. The bridge currently
fixes one warm and one measured invocation.

Prepared `.pt2` intake is implemented and tested, but it does not automatically
bind TVM host arithmetic to Merlin's portable reference policies. TinyLlama's
selected calibration/SmoothQuant/outlier preparation, SmolVLA's exact processed
input/environment, the shared frozen payload release, full-model numerical
agreement and actual FireSim driver/memory qualification remain open. The newer
optional speed recipes are prospective and explicitly unselected; they are not
an implicit replacement for the selected base recipes.

Fresh checks: protocol consistency; all 16 committed ResNet/TinyLlama derived
input hashes; three source/calibration bindings; a small EL4 recipe-to-TVM host
comparison; 283 EL4 contract/chain/transport tests; 218 TVM backend/CLI tests; and
two explicit capture round-trip tests pass. The larger comparison run records
256 passes, three failures and five skips. Two failures and three skips need the
pinned AVX512 environment. One failure is the LeRobot 0.5.1/diffusers 0.35.2/
TorchAO 0.16.0 import conflict. Two capture skips pass with their explicit
interpreter in the separate run. The initial transport collection lacked
`jsonschema`; the successful rerun used Merlin's complete dependency environment.
No guards were weakened and no full-model cloud run was performed. See
[EL4_ASSETS.md](EL4_ASSETS.md#audit-validation) for scope and reproduction commands.

The current workhorse serializes complete models through direct SSM; the older
queue remains a separate deployment option. The README distinguishes strict,
lightweight and legacy transport, and the independent numerical/hardware gates
required after collection. Local helper tests do not imply manager installation
or completion of the three-model benchmark.
