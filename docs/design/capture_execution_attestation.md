---
title: Capture execution attestation boundary
kind: design
status: current
owner: targetgen
last_verified: 2026-10-06
related: [phase0_specification, model2mlir, reproducibility]
code_refs:
  - packages/merlin-experiments/src/merlin_experiments/phase0/capture_execution_attestation.py
  - packages/merlin-experiments/src/merlin_experiments/phase0/capture_selection.py
  - packages/merlin-experiments/src/merlin_experiments/phase0/sealed_generation.py
  - packages/merlin-experiments/src/merlin_experiments/capture_execution/sealed_static.py
  - packages/merlin-experiments/src/merlin_experiments/capture_execution/sealed_m2m.py
  - packages/merlin-experiments/src/merlin_experiments/capture_execution/sealed_python.py
  - packages/merlin-experiments/src/merlin_experiments/capture_execution/python_preflight.py
  - src/merlin/targetgen/application_inventory.py
---

# Capture execution attestation boundary

A model2MLIR `m2m.capture-receipt.v1` verifies the materialized bundle members against
recorded byte digests. Its `source_closure_verified: false` is a distinct result: the
receipt does not prove which loader, importer, framework, checkpoint, dependency or
ambient file the capture process read. A later digest of today's checkout cannot
prove what an earlier process executed. Existing captures must retain that status.
When the receipt records a `quantization-manifest.json`, `verify_capture_receipt`
also requires the metadata pointer to name those exact bytes and the model MLIR's
`prov.quantization_manifest_sha256` to bind the manifest; a model that names a
manifest absent from the receipt fails verification.

Merlin reserves `merlin.capture_execution_attestation.v1` for this separate claim.
The diagnostic implementation inventories explicitly selected source bytes and reports
the adjacent materialized receipt. It always writes
`status: diagnostic_only`, `fresh_execution: false`, and
`source_closure_verified: false` to a new evidence path outside the capture and
source trees. This diagnostic does not pass Phase 0 admission. Editing these
fields or copying an old receipt cannot make a capture admissible. The admission
function `require_verified_execution` accepts only the preselected sealed
Model2MLIR CPU runners described below; no diagnostic, historical or static-ELF
receipt passes it.

The separate `merlin.sealed-static-capture.v1` issuer exercises the isolation boundary
for a self-contained static ELF payload. It copies complete selected source and
runtime trees into a private run, rejects links and dynamic executables, and runs
with bubblewrap namespaces and a cleared environment. Only the snapshots are
visible read-only; only a new capture directory is writable. Its replay verifier
checks exact file and directory membership, issuer and bubblewrap bytes, reconstructs
the fixed sandbox policy, and reruns the payload to demand byte-identical output.
Its receipt says `local_sealed_static_execution`; replay returns
`replay_verified_static`. Both keep the generic `source_closure_verified: false`,
because unsigned JSON and replay cannot prove the historical issuing process.
The observed scope is `static_elf_process_only`. This is not a model2MLIR or
PyTorch capture attestation and is not wired into Phase 0 admission.

`sealed_python.py` now tests the analogous *process isolation* seam for a caller-supplied
guest Python root and source tree. It copies and hashes both trees, runs an isolated
Python script without host home, checkout or network access, and independently
replays the saved inputs and output bytes. The guest root is still a caller
selection: this diagnostic cannot prove that its Python packages, native `dlopen`
dependencies, checkpoints and preprocessing data form the complete
Model2MLIR/PyTorch closure. Its receipt explicitly says
`phase0_admissible: false` and `source_closure_verified: false`; Phase 0 rejects it.

A verified issuer must perform a *fresh* capture in a new output directory.
It must privately snapshot the complete loader/importer source, Python runtime and
packages, checkpoint and preprocessing inputs; bind their membership and bytes;
execute only those snapshots with the source and runtime read-only, no network, and
no ambient checkout, home or cache; then verify the source and output bytes again.
The issuer must bind the exact command, environment, isolation controls, fresh run
identity, capture artifact inventory, and materialized receipt into its result.
The admitted issuers are Merlin's sealed Model2MLIR CPU v2 and v3 runners,
under their respective preselection and replay policies described below; these
requirements do not grant admission to another Python/model issuer.

Bubblewrap being installed is insufficient: an ordinary Python virtual environment
may read dependencies and caches outside the declared source selection. Without
a sealed runtime/checkpoint root for the selected model capture, the diagnostic
path must not claim verified source closure. Phase 0's coverage commitment admits
an application only when its `capture_execution_attestation` passes
`require_verified_execution` *and* names that application's selected capture and
materialized-receipt digests; captures without a sealed-M2M attestation remain
blocked on `source_closure_verified: false`, and the diagnostic path cannot
substitute for preselected sealed execution.

For a proposed Python capture, run the separate preflight with explicit paths:

```sh
python -m merlin_experiments.capture_execution.python_preflight \
  --worker /selected/merlin/_m2m_capture_worker.py \
  --loader /selected/model2MLIR/workloads/model/loader.py \
  --m2m-root /selected/model2MLIR \
  --python /selected/model2MLIR/.venv/bin/python \
  --capture-receipt /selected/old-capture/capture_receipt.json \
  --output /generated/private-evidence/model-preflight.json
```

The write-once result inventories the selected interpreter and its symlink target,
venv startup hooks and editable imports, direct source files, loader environment
reads, and directly declared Python/Torch ELF dependencies. Supply exact data paths with
`--data-path` and declared loader settings with `--env NAME=VALUE`; the tool does
not inherit ambient environment variables. It always exits 2 with
`blocked_unsealed_python_capture`, `fresh_execution: false`, and
`source_closure_verified: false`. Missing paths and unselected loader inputs are
remediation data, not a closure proof. A feasible next build on a spacious
filesystem is a private, immutable snapshot of the selected venv, CPython base,
editable sources, OS/CUDA libraries, model inputs/checkpoints and capture worker,
then a fresh empty-root, network-disabled bubblewrap run. None of the current
materialized captures may be upgraded by this preflight.

The static loader scan sees literal environment reads in the loader file, not
reads inside imported helpers. For a known delegated requirement, add
`--require-env NAME` (for example, a token corpus path) so an unselected value
is reported. This is a caller declaration, clearly marked in the result; it
does not establish a complete dynamic environment or file-read inventory.

`--capture-receipt` is optional. It compares that older receipt's loader and named
M2M direct-owner digests against the **current** selected checkout, reporting exact
drift or missing files. A match only means those declared files match now: the
receipt does not enumerate transitive Python imports, runtime libraries, or data
reads, and the comparison does not authenticate the earlier process. Rerun a fresh
capture after any drift; never relabel the older one as source-closed.

The preflight also hashes every regular file and directory under the selected
`m2m` package and reports `.py` members absent from the older receipt's named
direct-owner list. It rejects links and special entries in that tree. This
current-tree inventory exposes a concrete gap such as a loader-imported helper
missing from the receipt, and supplies bytes for planning a private source
snapshot. It is still neither a historical source claim nor a complete Python
runtime/import closure.

When the receipt binds a sibling `meta.json`, the preflight verifies those
metadata bytes and cross-checks its observed M2M import-source hashes against
current files and the receipt's direct-owner list. This can expose an imported
helper omitted from that list. The worker records only modules newly imported
after its observation point, so an empty observed-M2M list does **not** prove
that no M2M modules were executed. Neither metadata nor the receipt authenticates
the historical process. Observed paths with symlinked ancestors or parent
traversal are rejected before their target bytes are read.

The same comparison also lists observed non-package modules under the selected
model2MLIR checkout, such as a workload loader imported by a thin Merlin example
adapter. Those entries appear as `selected_checkout_sources` with observed and
current hashes, separately from `selected_m2m_sources`. They are not silently
absorbed into the receipt's direct-owner list, and a matching hash still does not
prove a complete import or checkpoint-data closure.

The bounded `sealed_m2m` CPU runner has a separate v2 policy for either FP32
without a recipe or static int8 with an explicitly selected, content-validated
`quant_recipe_v1` whose numerical engine is `integer_reference`. Its plan names
the dtype and recipe bytes, selected Model2MLIR revision, workload, Merlin worker
package, its canonical schema tree, and Python runtime. Issuance copies the selected
sources into a private empty-root process. The runtime (venv, base Python and
system libraries) is materialized once per selection identity in a
content-addressed store and hard-linked into each private guest root; every byte
is re-hashed against the plan before execution and again on replay. A namespace
probe runs before anything is snapshotted, and transient `__pycache__` bytecode is
excluded from the selected M2M package. The float dtype token is `fp32` or `f32`;
the only worker option a plan may select is an `agreement_tolerance` pair, which
is bound into the plan and command. Preselected runs also raise the framework's C++
log floor (`TORCH_CPP_LOG_LEVEL=ERROR`, part of the recorded sandbox policy) so
timestamped warnings cannot break byte-identical stderr replay. Replay reconstructs
the selected command and checks bundled schema membership and bytes. It also checks the
recipe against capture metadata and independent integer-reference agreement.
Historical FP32 v1 receipts retain their original replay
policy. Raw replay is not Phase 0 admission; its result explicitly says
`phase0_admission: not_granted`.
Loaders that read ambient environment values or require checkpoints outside the
selected trees remain unsupported by this bounded policy.

Phase 0's experiments-owned `assess_sealed_m2m_capture` accepts a selected
`model.mlir` path and the caller's independently selected SHA-256 digests for
the model, materialized capture receipt and `sealed_m2m_pending.json`.
It requires that path to be the run's `capture/model.mlir`, verifies the adjacent
materialized receipt, checks all three selected byte identities before and
after replay, and invokes the sealed v2 replay verifier. The pending receipt
commits to the issued command, selected input plan, sandbox policy, copied
source/runtime snapshots, process result and output inventory. Selecting its
digest independently prevents a different pending record from silently
satisfying the same assessment. It does not authenticate who issued that
record or which historical process ran. V2 replay compares the copied M2M package,
workload, venv, base Python, and schemas against their selected tree digests,
and checks the selected worker bytes. The Merlin package tree is compared when
the schemas were selected within it; an external schema tree is injected into
the copied package after its original tree digest was recorded, so its complete
copied bytes remain bound by the sealed snapshot and the schema's own digest.
The assessment reports `replay_verified_nonadmissible` and
`phase0_admission: not_granted` on success. It is a diagnostic for a selected
run, not authority to upgrade an old capture. The unsigned M2M receipt and a
copied selected virtual environment remain explicit provenance limits.

Admission instead requires selecting the capture *before* it exists. For the v2
issuer, `phase0.capture_selection.select` writes an owner-only `capture-selection.json`
(`merlin.phase0.capture_selection.v1`) for a fresh run directory: the sealed plan,
an explicit `checkpoint: {kind: none}`, the system-library and bubblewrap bytes,
the issuer source digest and the sandbox policy digest. `issue` reloads it by its
independently supplied SHA-256, re-derives the plan, refuses an existing run
directory and runs the sealed issuer; `verify` checks the pending receipt against
the selection and the copied system libraries, replays the capture, and returns a
`merlin.phase0.preselected_capture_replay.v1` record (`verified_preselected_replay`,
itself still `phase0_admission: not_granted`). `attest_sealed_m2m` turns that
record into a `merlin.capture_execution_attestation.v1` document with
`status: verified_sealed_execution` and `issuer: merlin.sealed_m2m_cpu.v2`.

The reviewed admission policy permits the sealed CPU v2 and v3 issuers under their
respective preselection policies; neither a raw Model2MLIR receipt nor the
diagnostic assessment can acquire this status retroactively.

Admitting these issuers is an explicit operator policy decision, embedded in every
attestation with its accepted residuals: the sealed receipt is unsigned, and the
Python runtime closure is the copied selected venv rather than an independently
pinned dependency set. `require_verified_execution` does not trust the document's
flags: for each issuer it re-reads the selection, the pending sealed receipt, the
model and the materialized receipt, requires the receipt to bind the preselected
plan, policy, bubblewrap and issuer bytes, and compares every digest with the
attestation. Changing any of those bytes revokes admission. In verified Phase 0
generation, PyTorch captures made while writing capsules follow the same
select/issue/verify/attest path (`phase0.sealed_generation`), and a request the
selected policy cannot express (a declared loader environment, a quantization
scheme instead of a recipe, an already materialized model) fails closed.

The v3 policy extends this selection to complete pretrained networks and
multi-program sessions. Its input plan inventories explicitly selected checkpoint
files or trees, their indirect file members, fixed loader-environment values
(including absence), and a bounded execution timeout. The sandbox mounts only
those selected bytes and reconstructs the same command, environment and timeout
for replay. Source-tree bytes, rather than a clean Git label, identify the selected
Model2MLIR implementation. An older capture cannot be adopted into this policy.

Every program in a session must have a materialized receipt and match the root
session's complete stage roster and ABI. Recipe-bearing int8 stages additionally
require preserved recipe identity, independently checked integer-reference
agreement and actual integer contractions. Stages with no recipe-eligible work
retain FP32 semantics; they do not acquire an int8 claim merely by belonging to
the session. The v3 attestation binds all stages, selected inputs and replay,
retaining the same unsigned-receipt and selected-runtime provenance limits.
