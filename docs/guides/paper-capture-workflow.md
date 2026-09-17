---
title: Materialize the frozen paper capture set
kind: guide
status: current
owner: compiler-evaluation
last_verified: 2026-08-31
related:
  - rvv_e2e
  - rvv_kernel_mining_methodology
code_refs:
  - merlin/python/merlin/compare/capture_workflow.py
  - merlin/python/merlin/compare/executorch_packages.py
  - merlin/python/merlin/compare/freeze.py
  - merlin/python/merlin/compare/paper_contract_registry.py
  - merlin/python/merlin/compare/paper_full_model_ablation.py
  - merlin/python/merlin/compare/paper_measurement_controller.py
  - merlin/python/merlin/compare/paper_k1_orchestrator.py
  - merlin/python/merlin/compare/paper_model_object_builder.py
  - merlin/python/merlin/compare/paper_merlin_mlir_producer.py
  - merlin/python/merlin/compare/paper_merlin_producers.py
  - merlin/python/merlin/compare/paper_session_abi.py
  - merlin/benchmarks/rvv_paper/study_v2.yaml
---

# Materialize the frozen paper capture set

`merlin-paper-capture` is the only paper-ready path from the prepared input bundle to Model2MLIR
captures. It treats all five holdouts and both declared precisions as one set. It does not expose a
model loader until the generic CPU-host campaign is `campaign_complete` and its selected policy,
runtime, and compiler digests are sealed.

Run the default preflight first. This writes a timestamped plan under
`out/artifacts/paper-captures/<target>/v1/` but does not import a holdout or invoke a capture:

```bash
merlin-paper-capture \
  --host-experiment merlin/experiments/cpu_host_compiler_v0/experiment.yaml
```

The plan binds the exact paper-input tree digest and every per-model input/checkpoint source, hashes
the actual Model2MLIR source bytes, records its Git revision and dirty paths, resolves each workload's
interpreter, and records the exact command and model environment. Inherited `M2M_*`, `VITFLY_*`,
`HF_*`, and `TRANSFORMERS_*` variables are removed before the recorded paper environment is applied;
this prevents a shell's smoke-layer or synthetic-input setting from leaking into the run.

After the plan says `ready`, explicitly launch all ten heavy captures:

```bash
merlin-paper-capture \
  --host-experiment merlin/experiments/cpu_host_compiler_v0/experiment.yaml \
  --execute
```

Each command has separate stdout/stderr logs and nanosecond elapsed time. A capture is rejected unless
its semantic session is paper-ready, uses the complete checkpoint and exact prepared inputs, contains
the declared continuous stages and observation count, and supplies the full single- or multi-program
bundle. A failed task leaves diagnostic bytes in its timestamped product, but the workflow emits no
registration and no staged study. Existing capture directories are never overwritten.

Only a complete 10/10 set produces `capture-registration.json` and `staged-study.yaml`. The staged
study is not freeze-ready yet: its five FP32 ExecuTorch/XNNPACK package rows remain unresolved.

Build inputs for the four Merlin-side backends are a separate, fail-closed boundary. First issue
the independent linker/compiler authority by naming the executable directly (PATH and repository
`.env` lookup are not used):

```bash
merlin-paper-merlin-packages issue-authority \
  --output /separately-reviewed/k1-paper-toolchain-authority.json \
  --authority-id k1-paper-compiler-2026-08-31 \
  --target k1 \
  --build-tool /absolute/path/to/riscv64-linux-gnu-gcc
```

Review and retain both the authority and its adjacent `.receipt.json`. Backend producers then place
one `producer-input.json` at
`<producer-root>/<backend>/<model>/<precision>/producer-input.json`. The closed receipt binds its
MRLNSES2 compiler-input graph to the exact compiler-source, capture, runtime, compiler-package,
kernel-source, kernel-backend, and (for `merlin_frozen`) promoted compiler submission/source
identities. There are 25 such producer inputs: five W8A8 hand-v0,
ten promoted Merlin (W8A8 and FP32), five FP32 Merlin+XNNPACK, and five FP32 Merlin+OpenBLAS. A
generic Merlin object is never substituted for a missing backend producer.

Audit the concrete backend-producer boundary before attempting registration:

```bash
merlin-paper-merlin-producers \
  --study out/artifacts/paper-captures/<target>/v1/latest/staged-study.yaml \
  --capture-registration out/artifacts/paper-captures/<target>/v1/latest/capture-registration.json \
  --promoted-compiler /absolute/path/to/promoted/compiler-package \
  --runtime-artifact /absolute/path/to/frozen/runtime-artifact \
  --producer-authority /separately-reviewed/multi-toolchain-authority.json
```

This writes a timestamped `producer-plan.json` under
`out/artifacts/paper-merlin-producers/<target>/v1/`. It enumerates every one of the 25 required
backend/model/precision cells, audits each capture's public session ABI, and identifies the exact
missing backend lowering. `--execute` uses the same closed gate: it never turns a prepared paper
input, generic `lower_model` result, or a `mining.k1` benchmark binary into a backend graph by
renaming it. In particular, model weights or immutable context that remain uncovered MLIR
arguments are a blocker, not implicit build input. Today the backend-specific MRLNSES2 adapters
for hand W8A8, the promoted compiler, XNNPACK kernels, and OpenBLAS kernels must exist before this
command can register a graph. A blocked run deliberately returns status 2 and retains only its
audit evidence.

The authoritative frozen study defines `hand_v0_int8` as W8A8. A request for five FP32 hand rows
conflicts with that study and is recorded in the plan; the tool preserves the frozen W8A8 matrix
instead of silently changing the paper comparison.

After an actual backend producer has emitted its MRLNSES2 compiler-input graph, register that graph
without editing JSON by hand:

```bash
merlin-paper-merlin-packages register-producer-input \
  --study out/artifacts/paper-captures/<target>/v1/latest/staged-study.yaml \
  --promoted-compiler /absolute/path/to/promoted/compiler-package \
  --runtime-artifact /absolute/path/to/frozen/runtime-artifact \
  --producer-inputs /absolute/path/to/closed/backend-producer-inputs \
  --backend merlin_frozen --model gemma2_2b --precision fp32 \
  --compiler-input /actual/producer/output/compiler-input.json
```

Registration deep-retains and validates the complete graph and refuses to overwrite an existing
cell. It is not a lowering fallback: the backend producer must already have emitted the graph.

Plan the complete 25-package, 50-template set before executing it:

```bash
merlin-paper-merlin-packages packages \
  --study out/artifacts/paper-captures/<target>/v1/latest/staged-study.yaml \
  --capture-registration out/artifacts/paper-captures/<target>/v1/latest/capture-registration.json \
  --promoted-compiler /absolute/path/to/promoted/compiler-package \
  --producer-inputs /absolute/path/to/closed/backend-producer-inputs \
  --runtime-artifact /absolute/path/to/frozen/runtime-artifact \
  --toolchain-authority /separately-reviewed/k1-paper-toolchain-authority.json \
  --authority-receipt /separately-reviewed/k1-paper-toolchain-authority.json.receipt.json
```

Add `--execute` only after the plan reports all 25 inputs valid. Each MRLNSES2 graph is independently
clean-replayed, linked with the authorized tool, and represented by one template for each declared
core count. The CLI validates all 50 templates before a single directory rename publishes
`package-set/package-ready-study.yaml` and `package-registration.json`. Failed work remains under the
timestamped product's staging name and is never a package registration. This study is still not a
full freeze input until the independent ExecuTorch package workflow has registered its five rows.

Plan the package set next. This hashes all ten registered captures, revalidates each FP32 semantic
session, hashes both complete executed/imported source roots (the ExecuTorch checkout and Merlin's
Python package), the exact Model2MLIR capture/loader sources, and every declared external model-code
closure (including VitFly's `models/` tree for LSTMNetViT). It requires the installed exporter to
report the same full Git commit as the runtime source checkout. It records five exact build commands
and exact sanitized model environments but does not import a loader or build a package. The child
environment inherits only declared host keys; every other value comes from the pinned paper-input
record. `MERLIN_K1_TOOLCHAIN` must be explicit (or supplied with `--k1-toolchain`); repository
`.env` fallback is forbidden. The resolved prefix, compiler binaries, binary digests, and full
version output are bound before the build:

```bash
merlin-paper-executorch-packages \
  --study out/artifacts/paper-captures/<target>/v1/latest/staged-study.yaml \
  --capture-registration out/artifacts/paper-captures/<target>/v1/latest/capture-registration.json \
  --k1-toolchain /absolute/path/to/spacemit-toolchain-prefix
```

After `package-plan.json` says `ready`, explicitly build all five immutable FP32 packages:

```bash
merlin-paper-executorch-packages \
  --study out/artifacts/paper-captures/<target>/v1/latest/staged-study.yaml \
  --capture-registration out/artifacts/paper-captures/<target>/v1/latest/capture-registration.json \
  --execute
```

Each v3 package contains its continuous-session programs, XNNPACK-enabled RISC-V runner, exact
Python package inventory, sanitized invocation environment, and
content/session/framework/Model2MLIR/external-source/compiler identities. Its
`build_environment_sha256` is the canonical digest of that complete embedded record; the distinct
`build_invocation_environment_sha256` binds the exact child process map. A closed
`paper_executorch_session_producer_receipt_v1` classifies and content-addresses every package byte
as public build evidence or private measurement I/O, binds the runner as ELF64, little-endian
`EM_RISCV`, and binds the exact trusted producer source. Its adjacent canonical compiler-input
descriptor refers only to that receipt. Package publication also emits one sealed-executable
package receipt and content-addressed measurement template per requested core count. A failed or
rejected task leaves logs and partial build bytes in its timestamped product, but emits
neither a package registration nor a complete study. Immediately before publication, every
long-lived input/source/identity and all five package digests are observed again. The freeze-ready
study is atomically renamed first; `package-registration.json` is the last completion marker, so a
crash can leave an unregistered study but never a registration naming an absent study. Only that
validated 5/5 pair under
`out/artifacts/paper-executorch-packages/<target>/v1/` is a freeze input.

Freeze that package-complete draft with the selected generic compiler policy and runtime sources:

```bash
merlin-compare \
  --spec out/artifacts/paper-executorch-packages/<target>/v1/latest/freeze-ready-study.yaml \
  --freeze --policy /absolute/path/to/selected-policy \
  --toolchain-authority /separately-reviewed/paper-toolchain-authority.json
```

The freeze operation requires the exact adjacent `package-registration.json`, verifies its study and
five package rows, revalidates every registered content address without rewriting it, and checks each
package-to-capture/session/framework/build-environment identity. Its path and digest are retained in
the frozen study. Never rerun capture or tune a compiler after this freeze; create a new declared
study instead.

The toolchain authority is an independent freeze input, not a file emitted by a backend package.
Create/review it separately with
`paper_toolchain_authority.write_toolchain_authority(...)`, retain its content digest in the frozen
study, and do not regenerate it from a template during a run. It pins the target, compiler role,
exact compiler ELF digest, and a derived identity. Freeze, contract construction, live measurement,
and fresh-process receipt replay all verify that identity. Merlin recipes execute the authorized
compiler. ExecuTorch instead verifies the independently cross-built runner against its closed
producer receipt and never executes that x86-host compiler on K1. Thus a package editor cannot
substitute an ELF that ignores generated source and then refresh the template, package receipt,
object, and result hashes.

Before a study can freeze, every matrix cell must name a content-addressed
`paper_backend_measurement_template_v3` under the backend's `measurement_contracts` mapping
(`model -> precision -> core-count -> {path, sha256}`). The closed production registry accepts only
Merlin `merlin_compile` backends and the ExecuTorch external-runtime backend. `study.run` has no
callable/executor injection seam.

Version scope is deliberately narrow: the legacy v2 package path remains an integrity diagnostic
backed by the non-board affine recipe, while the v3 package path is reserved for the production
Merlin whole-session ABI. The real Merlin MLIR producer emits and independently clean-replays a K1
relocatable object exporting
the closed `MRLNSES2` whole-session ABI. Its canonical compiler-input descriptor binds the public MLIR
closure, multi-toolchain authority, producer receipt, frozen source/capture/artifact identities, and
all generated outputs. Registry, freeze, and controller regeneration require deterministic byte
identity with the cached object, and retention copies that complete graph. For this recipe the
registry selects a receipt-hashed `paper_session_tracer` runner generated with the exact response
capacity implied by the descriptor and compiled output ABI. Freeze encodes the complete capture as
one descriptor-bound request/reference pair, and the controller supplies the request on stdin and
validates the response from stdout before projecting its output frames into the existing trajectory
oracle. ExecuTorch uses its producer-owned continuous-session ABI rather than MRLNSES2: the runner
loads the programs once, restores initial recurrent state for each full-session repeat, advances the
frozen input stream in order, and writes one complete trajectory to the controller-selected path.

The v3 template cannot author source text, build argv, a result-ELF digest, input, reference, or
oracle. It selects exactly five content-addressed resources: `package_receipt`, `compiler_input`,
`model_object`, `build_tool`, and `runtime_artifact`, plus the execution/memory settings. The
package producer emits a closed `paper_backend_package_receipt_v2` for the legacy ABI or v3 for
MRLNSES2; it additionally binds the
registry recipe, frozen compiler/framework input, shipped object-builder source, normalized object
build argv, generated-source digest, derived object, and linked result ELF. Freeze regenerates the
object from `compiler_input` with the registry-owned recipe, requires byte identity with the cached
object, and independently relinks the result before opening private inputs or the eager reference.
For `executorch_aot_model_object_v1`, “model object” is the schema's historical resource name for
the sealed session executable. The recipe validates the canonical compiler-input and producer
receipt, rechecks every public byte and architecture field, and copies that exact executable; it
does not compile or link. Its build receipt records `operation: sealed_executable_verification`, so
the normalized lifecycle's historical `built: true` means “materialized and verified,” not
“compiled on board.”
The MRLNSES2 v3 form also binds its public descriptor and producer-owned runner. The full freeze
performs this package phase before paper-input binding or
`validate_capture_session`, and before loading or content-hashing the v3 full-session package tree;
the live controller likewise completes a pre-private build before retaining input/reference bytes,
and repeats the exact build after retention.

Capture now writes `paper_measurement_sources.json` after independently parsing the semantic input
streams and eager-FP32 trajectory. Freeze ignores any draft-authored `measurement_io` and requires
this capture receipt. The legacy ABI creates at least two framed private-input shards when the
session has multiple observations and emits `paper_measurement_io_generation_receipt_v1`. MRLNSES2
instead emits one canonical whole-session request, one descriptor-bound response, and the closed v2
source/manifest/I/O receipts. Only those generated files populate the canonical
`freeze.measurement_io[backend][model][precision]`. Oracle kind and threshold come from the model's
frozen quality gate. Freeze also binds the exact `baseline_sources.yaml` digest containing the
upstream `hand_v0` and prior tuned-FP32 source pins.

At execution the controller retains the canonical frozen study and template, then either rebuilds a
Merlin ELF or verifies and materializes the sealed ExecuTorch ELF. Public executable provenance is
fully validated before the controller opens private input/reference bytes. It then validates the
complete retained private package graph and invokes the sealed runner directly on K1 with the
package root, controller-selected worker count, and controller-owned observation path. There is no
nested SSH, Python adapter, or editable child argv. The controller owns process affinity and rejects
subprocess descendants. The runner remains resident for all warmups and measured full-session
repeats; its strict, contiguous stage/repeat markers provide internal execution samples while the
controller's single process row records independently observed affinity, CPU work, RSS, and child
count. It would be false to synthesize one OS-process row per internal repeat, so the receipt keeps
those two evidence layers distinct.

The measured child cannot report pass/fail or quality. Those are derived by the controller from the
separately bound reference. OS evidence comes from monotonic timing, `/proc`, each thread's runnable
state, CPU-time delta, running CPU and affinity, and all requested cores' cpufreq state. Production
runs terminate the complete process group on error and require a sampled interval in which every
requested core accrues actual runnable CPU work. A sleeping worker's last CPU does not count.

Retained `paper_controller_measurement_receipt_v6` bundles include the exact build receipt, raw
iteration trace, board receipts, and finalized non-agentic AET lifecycle. The original primary raw
measurement is signed by a unique Ed25519 key. Its private half exists only in a mode-0700 temporary
directory during issuance and is deleted; the detached root retains only the public key, entry, and
signature. Those same-user mutable files are not a trust anchor by themselves. Same-process sealing
uses the live controller issuance registry; a fresh process fails closed unless the caller supplies
the exact `issuance_fingerprint()` value through a separately frozen/notarized channel (the
`trusted_issuance_fingerprints` mapping accepted by report APIs). Replacing the complete keypair,
receipt, entry, and signature then changes that external fingerprint. Fresh-process replay is
labeled semantic reproducibility evidence only: replay timing is never used as a tolerance window
that can authenticate edited primary samples.

## Producer and measurement handoff

`merlin/python/merlin/compare/executorch_packages.py` owns the ExecuTorch handoff. Merlin packages
use the MRLNSES2-aware registry/freeze/controller path. Before paper inputs/reference are made
available:

1. ExecuTorch emits the canonical compiler input and closed producer receipt consumed by
   `executorch_aot_model_object_v1`; the receipt binds the exact source, package, toolchain,
   framework, Model2MLIR, external-model closure, runner bytes, and RISC-V architecture;
2. Merlin package construction retains the generated `MRLNSES2` descriptor and links the verified
   object through the producer-bound stdin/stdout whole-session runner for
   `merlin_paper_session_v1`; it does not use `paper_model_abi_runner.c`, whose
   `merlin_paper_step` ABI is intentionally incompatible;
3. each package must retain the exact compiler ELF and concrete runtime artifact;
4. freeze encodes the captured streams as one descriptor-bound session request and derives the
   quality trajectory from the descriptor-bound response; and
5. each producer must write the closed package receipt and template for every declared core
   count, with identical five resource digests across core-count templates.

The repository implements `unit_test_affine_descriptor_v1` for non-board regression, the trusted
`merlin_mlir_model_object_v1` regeneration and MRLNSES2 measurement path, and the sealed
`executorch_aot_model_object_v1` path. The Merlin path succeeds only
after a complete clean producer replay, exact object-byte comparison, exact public descriptor/runner
binding, canonical request/response validation, and a package-result relink. The v3 package receipt
adds the session protocol and descriptor digest; measurement-source, manifest, and I/O receipts use
their v2 forms for the whole-session path. The ExecuTorch public verification barrier intentionally
does not open or hash private input/reference bytes; after that barrier, retention and execution
require the complete private graph to match the same producer receipt. A cached or manually
supplied model object is never a fallback for either backend.

Then run the existing freeze command above. It is the canonical validator/constructor; there is no
manual `measurement_io` fallback.

## Prepare and run a frozen contract matrix on K1

Materialize every contract without executing a model. The prepare command allocates one native AET
parent, derives child run IDs from it, follows the frozen deterministic execution order, and writes
the transport plan only after every canonical `measurement_contract.yaml` and resource tree has
been validated:

```bash
merlin-paper-k1-matrix prepare \
  --study out/<paper-freeze>/study.frozen.yaml \
  --output-dir out/<prepared-matrix>
```

The resulting plan is `out/<prepared-matrix>/k1-matrix-plan.json`. Preparation never invokes the
measurement controller or contacts K1. A partial failure retains the parent AET lifecycle and
partial contract directories but publishes no complete prepared-matrix receipt.

For an independently materialized contract set, the lower-level plan command remains available. It
freezes the execution roster without opening or changing any contract:

```bash
merlin-paper-k1-matrix plan \
  --contract out/<prepared-matrix>/controller-contracts/000_*/measurement_contract.yaml \
  --contract out/<prepared-matrix>/controller-contracts/001_*/measurement_contract.yaml \
  --output out/<prepared-matrix>/k1-matrix-plan.json
```

Pass every contract explicitly (the shell globs above are illustrative). Planning rejects non-K1
cells, duplicate run/cell identities, mixed study hashes, symlinks, non-canonical contract names,
or a contract/controller source tree that changes after planning. It binds every file's digest,
size, and mode; it introduces no editable command or backend adapter.

Execute the frozen plan from the host controller:

```bash
merlin-paper-k1-matrix run \
  --plan out/<prepared-matrix>/k1-matrix-plan.json \
  --output-dir out/runs/paper-k1/<timestamp>_<matrix-sha-prefix> \
  --host <k1-address> --user root --port <ssh-port> --key <ssh-key>
```

The host creates deterministic controller/contract archives, verifies their content again after
archiving, and stages them under SHA-256-addressed board cache directories. Each cell runs inside a
one-shot `systemd-run --wait --collect` unit on the RISC-V board. A board-global `flock` serializes
paper cells even if two host controllers race; the remote payload independently rejects a non-RISC-V
host. The board writes its terminal marker only after the controller receipt, normalized result,
detached issuance root, and issuance fingerprint exist.

Before inspecting or starting the first cell, the host runs one board-local environment preflight
from the digest-bound controller closure. Its retained `environment-preflight.json` binds the matrix,
controller tree, configured Python bytes, PyYAML and AET source identities, every absolute executable
used by transport/controller, ED25519 support, systemd, procfs task state, mapped libc/loader bytes,
the compiled trusted RVV board probe, and the frequency state of every requested CPU. Resume accepts
that receipt only if its digest and identities still match; terminal cells without the prerequisite
receipt are rejected.

ExecuTorch resolves the K1-without-a-native-compiler constraint by separating trusted host
production from board verification. The package producer cross-builds the RISC-V runner on the
host, records the exact compiler bytes/version and complete source/package identities, and seals the
result bytes. Contract construction reproduces that verification and binds the sealed digest. The
board-local controller verifies the independently authorized compiler identity as provenance but
does not execute it; it revalidates the closed producer graph and copies only the byte-identical
`EM_RISCV` executable. This is explicitly not an on-board rebuild claim. Merlin recipes that still
require a board-local compiler remain subject to their own architecture gate; the exception is
recipe-specific and cannot turn arbitrary source or argv into an executable.

Retrieval is two-phase: the board hashes a tar containing the receipt bundle, result, terminal
marker, and detached issuance root; the host verifies that digest, rejects links, path traversal,
and unexpected entries, validates the issuance fingerprint, then atomically renames the cell into
place. `issuance-notary.partial.yaml` is refreshed after each terminal cell, and the closed
`paper_external_issuance_notary_v1` is published only when every frozen run ID has a fingerprint.

Resume an interrupted host run with the identical plan and output directory:

```bash
merlin-paper-k1-matrix run \
  --plan out/<prepared-matrix>/k1-matrix-plan.json \
  --output-dir out/runs/paper-k1/<existing-run> \
  --resume --host <k1-address> --user root --port <ssh-port> --key <ssh-key>
```

A validated local terminal cell causes no SSH call. If the board completed but retrieval was
interrupted, the host retrieves that terminal without rerunning the measurement. A partial
non-terminal board directory, changed local terminal or plan, failed systemd unit, or corrupt
retrieval fails closed and retains an attempt record for explicit recovery; the orchestrator never
deletes or silently overwrites that evidence.

After the matrix state is complete, ingest it into the canonical results seal:

```bash
merlin-paper-k1-matrix finalize \
  --plan out/<prepared-matrix>/k1-matrix-plan.json \
  --run-dir out/runs/paper-k1/<completed-run> \
  --study out/<paper-freeze>/study.frozen.yaml
```

Finalization accepts exactly the plan-ordered terminal directory roster—no missing, duplicate, or
extra cells. It revalidates the frozen plan and study, prerequisite environment receipt, complete
matrix state, each contract/result identity and retained terminal digest, and the exact external
issuance-notary roster. Compiler results receive only the causal records derived from the already
frozen attribution manifest. On K1 that manifest must be schema v2: its control is a separately
measured `merlin_ablation_control` full-model binary, never the XNNPACK/OpenBLAS/ExecuTorch
comparator. Every AB/BA arm receipt is replayed and the final matrix Merlin binary must equal the
frozen treatment binary; otherwise the comparison remains `advantage_not_claimable`. The existing
results sealer then replays every controller receipt with
the notarized issuance fingerprints; its semantic replay never replaces or authenticates primary
timing through a tolerance window. The public report API must accept the resulting seal before it is
published.

`results-finalization.json` binds every terminal result/state/fingerprint and all prerequisite
digests. `results.yaml` is atomically published last and is never overwritten. A partial, edited, or
already-finalized run therefore cannot become plot input. The report and figure commands consume
`out/runs/paper-k1/<completed-run>/results.yaml` and the same retained
`issuance-notary.yaml`.

The local non-board regression path is:

```bash
PYTHONPATH=merlin/python .venv/bin/pytest -q -p no:cacheprovider \
  merlin/tests/dse/test_paper_study.py::test_freeze_constructor_derives_measurement_io_from_capture_and_package_receipts \
  merlin/tests/dse/test_paper_study.py::test_registered_merlin_builder_produces_independently_replayable_live_result
```
