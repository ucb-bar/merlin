# Phase 1: functional compiler

Select the reviewed Phase 0 manifest's `phase_corpora.phase1` functional members
and retain its evidence-manifest identity, SW spec and hardware selection.
The retained diagnostic derivation described below is not an admission release.
An admission release comes from a fresh Phase 0 run with verified evidence
(`--phase0-evidence-mode verified`): audited RTL facts, sealed capture
attestations and the reviewed host manifest. Run it, then prepare and seal it.
Do not substitute Phase 2's performance membership for the functional population.

Start from [the experiment definition](../experiment.yaml). Its
[target descriptor](../target/descriptor.yaml) selects the authored prompts in
`task/`, independently of retained harness and bundle resources.

The realistic mode uses `TASK_realistic.md`; the legacy fullsuite and one-shot
launchers use `TASK_full.md` and `TASK.md`. `TASK_pilot.md` is a retained authored
reference. Generated-prompt modes use Merlin's shared renderer. Every run archives
the exact final prompt served, including its selected arm's additions.

Prepare and review a fresh corpus release through the
[reviewed handoff](../../../experiments/README.md#reviewed-phase-0-handoff)
before verified execution. Preparation copies these inputs into the release and
removes the live source pointers. New bundles grant the declared task directory;
old bundles or frozen runs are not rewritten to adopt this layout.
Phase 1 startup rechecks the sealed `phase0_readiness` against the frozen corpus;
the distinct `whole_workload_phase1` verdict remains incomplete until a compiler
produces and executes the required typed routes. A diagnostic capture replay is
not sufficient for the Phase 0 handoff.

## What the functional finish line must prove

Passing generated operation capsules is necessary but not sufficient to claim a
network compiler. For each selected whole-network capture (including ResNet-50
and a specified full SmolVLA graph, not only one denoise step), a functional
Phase 1 result needs a complete
operator inventory, an accounted route for every region (Gemmini or a declared
host lane), a whole-model compile with no unsupported-op escape, executable
artifacts, numerical comparison against an independent framework reference,
and observed dispatch evidence that the admitted accelerator work actually ran.
For fused stages, inspect `planned_outlined_alignment` beside the dynamic
`dispatch_ledger`: a routing plan or group proposal alone does not prove the
runtime emitted one accelerator dispatch, much less executed it. Missing alignment
is incomplete; an eligible group split into host work is a functional placement
failure. The current runtime does not yet execute grouped epilogues on its mesh
path, so do not claim fused-stage coverage from a passing contraction counter.
Keep the capture, quantization scheme, weights/manifest, compiler submission
hash, intermediate MLIR and run receipts together; a storage dtype alone does
not establish the arithmetic or accelerator placement. The
[whole-model example](../whole-model/README.md) explains IR inspection, but its
lowering smoke is not this functional certificate.

Placement is a separate acceptance condition: every source compute region must
appear in the whole-module census, and every region admitted by the independent
Gemmini eligibility contract must execute on Gemmini (possibly as a declared
fused stage). A correct host result does not satisfy that condition. A host
region is acceptable only when the target contract explains why it is not
Gemmini-eligible; an unknown precision, unclassified operation, missing census,
or unobserved dispatch leaves the claim incomplete. Review both region recall
and estimated-work recall, since a single missed contraction can dominate model
time even when the region count looks good. The exact minimum Phase 0 source-op
witness basis is not a substitute for this Phase 1 placement and execution check.

The current example does **not** claim that finish line has been reached.
The descriptor derives mandatory L3 model capstones from the reviewed Phase 0
release and its numerical qualifications; it does not maintain a hand-picked
model list. Retained historical capsules are not automatically part of a new
release. ResNet50, TinyLlama and SmolVLA are held-out generalization claims in
`workload_spec.models`, not Phase 0 derivation sources. Inspect actual
whole-model compilation and numerical receipts before asserting success or
failure on any of them. A resource exclusion is only a cost policy, not a
compiler verdict.

Before attempting that claim, inspect an explicit capture with the
[whole-model preflight](../whole-model/README.md#check-model-readiness). It
compares declared target routes with the operand formats and accelerator groups
in the captured graph. It cannot produce a compiler certificate: a requested
`int8` deployment format does not quantize an FP32 or BF16 model by itself.

### Exact-capture coverage check (diagnostic)

The Phase 0 handoff now checks a closed ledger of normalized MLIR operations and
typed SSA uses. Each independent compute node must have one placement obligation;
structural and nested nodes remain explicitly counted. Every compute-to-compute
SSA use must have a conditional transfer obligation. Missing or duplicate rows,
unknown dynamic extents, absent graph identities, and unverified capture source
closures block the precompiler
coverage commitment. This ledger checks the *captured graph*, not all PyTorch
operators or unseen models; reviewed declarations, lowering and numerical
execution remain separate requirements.

A read-only check of retained captures used the prototype Gemmini capability
view SHA256 `9d93adb81b1a1f9c7866251dae57e9318acbdfc1699a6a32cebcc02e37a36b4b`,
authored software spec SHA256 `d88278e321b0650296f6d34aff35f2b01e7995a5e0ca2ed45cd3a73397f9fc1f`,
and host manifest SHA256 `74a614e86bf27b0e290d06b9b0d7b6c5035b6cb8a1f0a471e5c37eca2275d8c0`.
This combination is a diagnostic screen, not a reviewed frozen Phase 0 selection.

| Exact retained capture | MLIR nodes / SSA uses | Ledger result |
| --- | ---: | --- |
| `coverage-mlp-r5`, `34a57374…` | 134 / 161 | Total |
| `residual_cnn-r5`, `8ee18181…` | 411 / 465 | Total |
| `causal_decoder-r5`, `a2c61d7f…` | 414 / 481 | Total |
| `multimodal_policy-r5`, `ebcd809e…` | 485 / 572 | Total |
| SmolVLA r3 `prefix_encode`, `c9d344cc…` | 13,396 / 15,892 | Dynamic shape unproved |
| SmolVLA r3 `flow_denoise`, `3d672bde…` | 8,724 / 10,156 | Ledger total; source trace incomplete |
| SmolVLA r3 `action_decode`, `cb553c11…` | 3 / 1 | Ledger total; trivial stage only |

The four iteration captures have complete static frontend correspondence, but
their materialized receipts all report `source_closure_verified: false`. Both
large SmolVLA stages report incomplete quantized-to-prepared correspondence;
`prefix_encode` additionally has prepared nodes `n552`–`n557` without final
lowering correspondence. The host manifest that check used (`74a614e8…`) was
`unreviewed`, so it left all host admissions unknown. The current
[`host-capabilities.yaml`](../target/host-capabilities.yaml) is `reviewed` at the
document level and pins the minted scalar package (`89e00e39…`). Of its 63
operation declarations, 18 carry reviewed numerical contracts and 45 remain
`unreviewed`. Host admission is therefore decided per operation. A route whose
matching contract is unreviewed stays unknown. A reviewed route still needs its
own emitted lowering and numerical receipt. Re-run this check against the current
manifest rather than reusing these counts. No target execution or whole-model
numerical match follows from these counts. The selected Gemmini compiler's retained whole-model
`emit_command_buffer` observations explicitly decline routing upstream Linalg
regions, even where parse and native LLVM lowering accepted the model.

The SmolVLA dynamic value is a data-dependent `aten.index.Tensor` mask gather in
vision embeddings (`mask_gather_0`, prepared node `n551`). A 1,024-element mask
is summed to allocate `tensor<?xi64>` for selected positions; a later
`aten.index_put.default` (`mask_scatter_0`, node `n559`) reads those positions.
Its length can vary from 0 to 1,024, so one static-shape guard cannot establish
the needed semantics. A compiler route could carry a fixed 1,024-element scratch
tensor plus `valid_count`, prove the count bound and guarded read invariant, and
test equivalence of the transformed scatter. Alternatively, a reviewed host
island needs a bounded dynamic-buffer ABI, explicit transfer contract and
independent numerical execution receipt. Either route also needs repaired
frontend correspondence before a complete Phase 1 claim.

This is deliberately separate from [Phase 2](../phase2/README.md): Phase 1
establishes functional compiler capability and a frozen submission; Phase 2
holds that functional bar fixed while optimizing cycles, placement, and
whole-model cost under its own measured or model-portfolio evidence.

## Run the installed Phase 1 controller

The catalog ID `gemmini-functional` selects the single definition above. It is
**EL4** (RTL-informed Merlin). The definition's `level: EL4` derives the stable
`arm: merlin_assisted` and `treatment: rtlchecks` execution keys; the
`merlin_assisted_rtlchecks_*` bundle is the selected tool surface, not a separate
experiment. The treatment selects the installed Phase 1 module with explicit bundle
identity, bundle manifest and oracle-timing path. When copying the definition for
a reviewed release, select the release's descriptor, `corpus_seal`, and generated
RTL-checks bundle manifest together. The retained manifest path is a preparation
reference, not proof that its inputs are current or reviewed. Inspect and preflight
the copied definition before running it. Provision its declared toolchains and replace
legacy directory-symlink grants with explicitly owned input trees in a newly generated,
reviewed bundle **inside the same sealed release**; copying an old manifest to a
different path does not qualify it. The frozen-input check deliberately refuses
incomplete closures.
If the installed checkout has no `third_party/llvm-install`, set `MERLIN_CLANG`
to the absolute `bin/clang-23` in the selected LLVM/MLIR install before preparing
the release. New generated bundle manifests then name that exact install, and
the sandbox uses the same compiler and `mlir-opt`; old manifests are unchanged.
When the graded scope contains model capsules, explicitly set `MERLIN_M2M_PYTHON`
to the absolute reference interpreter that imports Torch, NumPy and safetensors.
Startup probes it before authoring and records its path and interpreter bytes;
resume refuses a changed or missing selection. This binding is independent of
`MERLIN_COMPILER_PYTHON` and does not qualify the framework dependency closure or
add BF16 support to the target host. A sibling checkout is not an installed runtime.

The direct installed CLI below selects the same treatment. For an installed
**baseline** catalog route instead, use the shared
[`baseline-functional-template`](../../../experiments/definitions/baseline-functional-template.yaml)
with its required operator inputs; changing treatment changes the experiment.

Phase 1's semantic-search receipt is a host-private diagnostic over the frozen
public capsules. It is not shown to the agent, does not select a treatment, and
does not count as compiler or grading evidence. An agent-visible search helper
would be a separately declared and frozen treatment so its results can be
compared fairly with the current experiment.

For a fresh run, `--public-object-build-budget-s 30` explicitly enables bounded
object-build feedback on the public scalability samples. The default is off;
an enabled run records the selection and cannot change it on resume. Each of
at most four samples receives the selected 1–120 second build budget. The agent
sees emitted size, build status, compile time and object size for its exact
candidate, not private model inputs or host tool paths. Use this advisory to
identify code-size and compilation-cost regressions; it is neither a device
performance measurement nor numerical certification. Selecting it requires a
new frozen run rather than modifying a stored experiment.

For direct invocation, set the variables below to actual operator-selected inputs.
`CORPUS_SEAL` is the release's `private/seal.json`; `DESCRIPTOR` must belong to
that release. `RESOURCE_ROOT` resolves declared resource paths. `BUNDLE_ID` must
match `BUNDLE_MANIFEST`; use reviewed RTL-checks inputs, not an invented bundle.
`ORACLE_TIMING` must name an existing operator-owned timing record. The example
selects `.oracle_timing.gemmini.json` in the target resource directory. The native
readiness check writes that file only after a real L3 pass and binds its target,
declared simulator configuration and simulator SHA256. The installed preflight
rechecks those bytes. Older records under the shared `scripts/` link are diagnostic
only. Provision a genuine record or select an existing measured one; do not
fabricate a placeholder or change it after freezing a run.

```sh
MERLIN_CORPUS_SEAL="${CORPUS_SEAL:?}" python -m merlin_experiments.phase1 \
  --descriptor "${DESCRIPTOR:?}" --repo "${RESOURCE_ROOT:?}" \
  --bundle "${BUNDLE_ID:?}" --bundle-manifest "${BUNDLE_MANIFEST:?}" \
  --oracle-timing "${ORACLE_TIMING:?}" \
  --run-id "${RUN_ID:?}" --level EL4 \
  --driver claudecode --provider subscription --model "${MODEL:?}" --effort high \
  --schedule continuous --max-wall-s 43200 --round-timeout 43200 --grade-interval 900
```

This starts authoring and grading: approve the provider and budget before running
it. The budget matches the example definition; reduce it deliberately if needed.
To qualify one preserved compiler without starting an author, use a **new** run ID
and the same reviewed release, bundle, timing, private full-model spec, and selected
toolchain bindings with `--qualify-submission "${SUBMISSION_DIR:?}"` instead of the
provider, schedule, and round options above. Select the corpus seal explicitly with
`MERLIN_CORPUS_SEAL`. The submission is copied into the new run and graded through
the ordinary public, hidden, private, and RTL gates. Its receipt is qualification
of that selected copy only—not agent convergence, an upgrade of an older run, or
permission to omit the private model gate. A new tool identity needs a new run;
neither `--resume` nor `--seed-submission` can be combined with this mode.
Qualification executes compiler builds and native simulators, but does not
launch a paid author.

```sh
MERLIN_CORPUS_SEAL="${CORPUS_SEAL:?}" python -m merlin_experiments.phase1 \
  --descriptor "${DESCRIPTOR:?}" --repo "${RESOURCE_ROOT:?}" \
  --bundle "${BUNDLE_ID:?}" --bundle-manifest "${BUNDLE_MANIFEST:?}" \
  --oracle-timing "${ORACLE_TIMING:?}" --run-id "${NEW_RUN_ID:?}" \
  --level EL4 --sandbox bwrap --private-full-model-spec "${PRIVATE_FULL_MODEL_SPEC:?}" \
  --qualify-submission "${SUBMISSION_DIR:?}"
```

Provision FileCheck on the selected execution PATH, and supply required toolchain
library paths and credentials explicitly. This installed route does not inherit
the old launcher's LLVM/Chipyard FileCheck candidates or compatibility-library defaults.
Use `python -m merlin_experiments.phase1 --help` for inspection only. The configured
output root owns the run. Resume the same invocation with `--resume` only while
its frozen inputs remain valid; changed inputs require a fresh run.

Installed ownership is not a sandbox or compiler certificate. Verified execution
still requires real bwrap isolation, nonvacuous answer masking, qualified tools and
public/frozen/hidden grading in the prescribed order. Diagnostic unsandboxed runs
cannot become Phase 2 inputs by waiving these integrity gates. Retain the exact
submission hash and qualification evidence for the [Phase 2 handoff](../phase2/README.md).

Compatibility links at the former source paths are for navigation only. Regenerate
and review bundles instead of relying on those links as sandbox grants. Generated
capsules, workspaces, compiler payloads and certification records remain artifacts,
not files in this example. The remaining harness resources still require a checkout.

## Independent runtime support

The copied kernel headers and legacy harness aliases have been removed. Runtime
support is the data-only provider `examples/gemmini/support` (selected by default
when `MERLIN_TARGET_PATH` is unset): the generic `merlin.runtime.backends.chipyard_rocc`
backend reads its contract and ISA-headers spec, builds against the pinned bare-metal
runtime plus a header generated from the RTL facts, and never puts an excluded vendor
header on the include path. Reference compilers and their runtimes stay outside
the repository; they cannot supply authoring or feedback tools.
