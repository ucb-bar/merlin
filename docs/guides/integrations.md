---
title: Integrations
kind: guide
status: current
owner: kernels
last_verified: 2026-09-23
related: [kernel_mining, architecture, repo_structure]
code_refs: [src/merlin/kernels/ingest, packages/merlin-experiments/pyproject.toml, packages/merlin-analysis/pyproject.toml, packages/merlin-experiments/src/merlin/benchharness/chia_bridge.py, packages/merlin-experiments/src/merlin/benchharness/chia_tasks.py, packages/merlin-experiments/src/merlin/targetgen/aet_bridge.py, packages/merlin-experiments/src/merlin_experiments/phase2/telemetry.py]
---

# Integrations

Integrations are **adapters**, not copied implementations of their upstream projects.
Adapters live in their owning distribution: shared compiler contracts and runtime integration
under `src/merlin`, and research/evaluation integrations under the relevant `packages/` tree.
Shared upstream import/process adapters live in `src/merlin/integrations`; workflow-specific
adapters remain in their owning distributions, not a separate monolithic integration checkout.
Stable `merlin.*` imports can resolve to an optional distribution; they do not imply core ownership. See the
[architecture reference](../reference/architecture.md) and
[integration rationale](../design/integrations.md).

| Dependency type   | Where                          | When                                          |
| ----------------- | ------------------------------ | --------------------------------------------- |
| Adapter only      | owning core or optional distribution | parse/index/emit-to/call the external tool |
| External checkout | outside the repo, by path/env  | inspect or run the repo; merlin doesn't own it|
| Pinned dependency | `third_party/<name>/`          | merlin cannot build/test without it           |

Choose dependencies for the owning workflow, not for every Merlin install. Required build/test
dependencies may be pinned submodules; optional workflows may use separate Python dependencies
or explicitly selected external checkouts. An existing `third_party/` checkout used only by a
baseline does not make that baseline a core SDK requirement.

AET owns agent accounting and run storage; experiment orchestration adapts it rather than
creating a second accounting system. Framework capture belongs to model2MLIR, RTL extraction
and model execution to ModeLIR, and specification semantics to SpecIR. Merlin's integration
code resolves inputs and translates contracts across those seams. LLVM/CIRCT and PyTorch/torchao
versions retain their own build, regression and numerical qualification requirements; package
relocation does not certify a new upstream version.

## Release compatibility snapshot (2026-09-23)

Read-only upstream branch checks found official [Chia](https://github.com/ucb-bar/chia)
`main` at `16c35e92aaaf9511c6453bf94cd5cf589698f4e3` and official
[AET](https://github.com/ucb-bar/agentic-eval-tool) `main` at
`903d4def8995e4697c6f214cc26c60a5a38d0554`, exactly the experiments
package's selected revisions. Built core and experiments wheels were installed
in an isolated Python 3.13 environment and imported from outside the checkout.
The Chia contract tests passed (5 pass, 1 opt-in skip); bridge/lifecycle/task
tests passed (69 pass, 2 opt-in skips). Four additional managed-Ray tests passed
inside a loopback-only network namespace, including cancellation and worker-loss
cleanup. Those tests use local synthetic work and a managed supervisor, not paid
agents or a remote cluster. The separately published Gemmini backend branch
`stable/gemmini_xdsl_rtl_v0` was at
`390623a67db81bcf595ec3805f0921f1fb69e378`; its independent-clone
interface, command-buffer and target-IR execution is described in the
[Gemmini example](../../examples/gemmini/README.md).

The local model2MLIR checkout's quantization/capture suite passed 21 tests
against PyTorch `2.10.0+cu128`, torchao `0.16.0` and torch-mlir
`20260531.828`. Merlin's integration suite passed 80 tests. Parsed-IR admission
now refuses a dynamic W8A8 request if torchao left the contraction in f32; a
genuine i8×i8→i32 capture passed. ModeLIR's discovery/specmodel suite passed
21 tests (17 optional native/asset skips) and SpecIR's numeric reference suite
passed 61 tests from their separate checkouts.
That checkout has uncommitted changes, so this is an observation of those
working-tree bytes, **not** qualification of its HEAD revision or a new pinned
framework release. LLVM 23 Python lowering emitted LLVM IR; Clang 23 from an
explicitly selected read-only toolchain ran host numerical checks and produced
an RVV object. Clang 18 was incompatible with that LLVM 23 IR syntax. No CIRCT
executable is available. ModeLIR native execution, CIRCT lowering, remote
simulator and hardware performance remain unqualified here. Do not promote
import, quantization, IR or OOT lowering checks into hardware claims.

## ModeLIR import and artifact lifecycles

`merlin.integrations.modelir` owns three distinct contexts. `importable(root)` temporarily
adds an otherwise unavailable checkout to `sys.path`, but retains imported modules for oracle
callers. `discovery_imports(root)` also removes newly loaded `mlc.*` modules on exit, only when
that context inserted the checkout; preexisting modules and installed imports are preserved.
`artifact_context(root)` retains modules and serializes temporary cwd/path changes with one
reentrant lock. Its optional `resolve_root=` callable runs inside the lock, so the RTL bridge's
configured-root lookup preserves its original ordering. The bridge retains its default-resolution
contexts and caller patchpoints; numerical/oracle execution and availability decisions stay there.

These are process-state policies, not isolation between competing ModeLIR checkouts. They neither
undo arbitrary upstream import side effects nor protect unrelated code changing cwd without the
shared context. SpecIR's independent reference-model imports are not routed through this adapter.
Tier-certificate instrument identities pin both the lifecycle adapter and the RTL bridge; changing
either source invalidates reuse in a new qualification. Historical receipts remain untouched.

## SpecIR reference-model imports

`merlin.integrations.specir.importable(root)` scopes checkout paths without changing
SpecIR's independent numerical semantics. Capture retains its explicit root, then
`SPECIR_ROOT` (process or `.env`), then sibling `spec` discovery. Phase-0 numerics keeps
process `SPECIR_ROOT`, then `.env`, then ordinary installed imports; an absent root
never inserts `None` into the Python search path.

Imported modules retain their normal identity. A selected checkout cannot reuse a
different cached SpecIR checkout: start a fresh process to change roots. Missing
explicit checkouts fail instead of falling through to an installed copy. Paths are
restored on success and failure; caller exceptions are not replaced by cleanup errors.
The reentrant lock coordinates these callers, not unrelated imports or arbitrary
upstream side effects.

Phase-0 startup receipts and golden-cache keys include this adapter's source bytes;
unresolvable adapter source bypasses the cache. The shared Phase-1 startup inventory
also carries that source. These commitments do not newly pin all upstream SpecIR
sources or qualify a new upstream version. Historical receipts remain unchanged.

## model2MLIR capture and compiler Python

Capture bundles and the optional typed quant frontend share the configured model2MLIR checkout.
Either process variable takes precedence over either `.env` entry; within each source,
`MERLIN_MODEL2MLIR` precedes `MERLIN_M2M_DIR`. Set these aliases consistently. An invalid preferred
checkout does not silently select a second one.

GGUF framework capture selects `MERLIN_M2M_PYTHON`, then `MERLIN_M2M_VENV/bin/python`, then
the checkout's `.venv/bin/python`. These settings honor `.env`; invalid executables fail explicitly.
`MERLIN_COMPILER_PYTHON` and `MERLIN_COMPILER_VENV` select compiler bindings only, not GGUF capture.
Workload-specific capture continues to use its own `capture.toml` interpreter policy.

The opt-in quant frontend does not retain temporary import-path changes or cache unavailability.
If another model2MLIR checkout is already imported, start a fresh process to select a different
checkout; Merlin does not unload framework modules. Missing optional quant support still falls
back to the existing unregistered parsing path, not invented quantization semantics.

## Chia and AET

Chia owns workflow scheduling, Ray task results, profiling and folding its LLM-call usage.
AET owns run records, accounting and reporting. Merlin owns phase policy, canonical run identity,
source attribution, grading and process supervision. The experiments-owned
`merlin.benchharness.chia_bridge` joins these interfaces; it does not implement another scheduler
or copy Chia's token aggregation. Native-agent transcript accounting remains in
`merlin.targetgen.aet_bridge`, using AET's public APIs.

Install from the Merlin checkout into a separate environment, without changing the driver venv:

```bash
uv venv out/build/chia-venv --python 3.13
uv pip install --python out/build/chia-venv/bin/python -e . -e 'packages/merlin-experiments[chia]'
export MERLIN_CHIA_PYTHON="$PWD/out/build/chia-venv/bin/python"
export MERLIN_EXPERIMENT_PYTHON="$PWD/.venv/bin/python"
"$MERLIN_CHIA_PYTHON" -I -c 'from merlin.benchharness.chia_bridge import require_chia; require_chia()'
```

The distribution is **`chialoops`**, not the unrelated `chia` distribution. Its import name is
`chia`. The experiments extra pins an immutable Chia source archive; its example submodules are
not Python package dependencies. AET retains the experiments package's exact Git revision.
Inspect `packages/merlin-experiments/pyproject.toml` for both pins. Do not install Chia's unpinned
`aet` extra over that environment. Ray/Pydantic requirements belong in the Chia environment,
not in core. Editable installs are for development, not frozen-run qualification.

Interpreter selection has one owner in the bridge. Explicit `--chia-python` (where supported)
precedes `MERLIN_CHIA_PYTHON`, then the configured build root's `chia-venv/bin/python`.
An invalid explicit choice fails; it never falls back silently. Native children use
`MERLIN_EXPERIMENT_PYTHON`, then a physical source checkout's `.venv`, then the current interpreter.
Installed workflows should set both variables. Preserve the venv executable path: resolving its
symlink to a system Python loses the selected environment. These are process-environment settings;
the bridge does not implicitly source `.env`.

`chia_run(...)` creates one canonical AET run and keeps the collector and scalar logger driver-local.
It refuses an existing collector, resets profiler state between runs, and closes resources even
when initialization or workflow execution fails. A borrowed Ray cluster must supply the requested
logical resources and is never shut down by Merlin; a locally created cluster is Merlin's to close.
The import-only capability check does not establish Ray startup, hardware, credentials or paid-agent
readiness. Run one collector-owning workflow at a time in a driver process.

### Task ownership and cancellation limits

The A/B batch, repeatability and agentic-performance Chia launchers use
`merlin.benchharness.chia_tasks` inside `chia_run`. Each group immediately records returned public
Ray `ObjectRef`s, gets normal results through Chia's public result API, and observes outstanding
work before the run's collector closes. On interruption or partial dispatch failure it requests
cooperative cancellation only for its own outstanding references; unrelated borrowed-cluster
peers are neither enumerated nor cancelled. It does not retry submissions or force-kill workers.
Batch and repeatability reports retain completed results alongside task errors. A native child's
nonzero exit remains visible and marks the canonical AET run failed.

Each run writes `chia/tasks.json` with schema `merlin.chia-task-ownership.v1`. A dispatcher that
raises without returning a reference is recorded as `dispatch_unknown`: work may have started
without a handle. Cancellation not acknowledged during observation is recorded as
`cancel_unacknowledged`. Unknown dispatch, observation errors and incomplete cleanup fail the
run; an existing primary exception is preserved with a cleanup note. The default ten-second
shared deadline bounds teardown's public Ray `wait`/`get` observation, **not** normal task
execution, all SDK/network calls or the complete shutdown duration.

`cleanup_complete` concerns the group's known task references and dispatch outcomes only.
Every receipt states `native_descendants: not_verified`. Ray task completion or cancellation
acknowledgement does **not** prove native children or grandchildren stopped, and a released Ray
CPU resource is not that proof either. These launchers do not yet provide qualified native
process-tree cancellation. Do not treat their task receipts as a process-cleanup guarantee.

The agreed native-supervision direction is to adapt Merlin to Chia's documented
setup/cleanup hooks, not modify or fork Chia's lifecycle implementation. Fully
verified cleanup requires managed workers with a supervisor that directly reaps
guardians after worker loss. Neither a running PID1 nor an operator assertion alone
establishes that capability. An unsupported host must refuse that qualification;
it must not silently downgrade to an unverified compatibility mode. Existing v1
receipts retain their historical meaning above.

The managed-process implementation lives in `merlin_experiments.execution`:
`native_supervisor` is an explicitly provisioned foreground service and the direct
parent of each `native_guardian`; `chia_native` supplies the public setup/cleanup
hooks and native command client. The guardian watches kernel process handles for
both the worker and driver, admits one command, and cleans its own descendants.
The supervisor separately waits for the guardian before recording reaping evidence.
Neither import starts a service nor connects to a cluster. Linux pidfd capability,
same-host/PID-namespace identity and a private endpoint are required; unsupported
workers refuse before native admission. Cooperative cancellation periodically returns
from the result socket to Python so Chia/Ray can deliver its cancellation exception;
the driver still requires an independent native lifecycle receipt.

Local real-public-hook qualification passed plain and profiled success, body failure,
cooperative cancellation, worker loss before/after native admission, rejected dispatch,
bypass and local-call refusal on Chia revisions `9bf0bbb98c2a068f4cea79a4a7b968f7db5ce22d`
and `16c35e92aaaf9511c6453bf94cd5cf589698f4e3`. Unrelated work survived every case.
The repeatable cancellation regression is
`packages/merlin-experiments/tests/test_managed_chia_hooks.py`; enable its explicitly
owned local Ray runtimes with `MERLIN_TEST_MANAGED_CHIA=1`. This qualification does
not change the selected dependency pin or establish accounting equivalence between
those revisions.

Frozen provisioning uses `frozen_python.python_command` with the existing verified
snapshot seal. The service captures that guarded startup identity; sessions require
the expected seal, and guardians use `inherited_python_command` and confirm the same
identity before READY. Ambient environment variables or hashing a changed checkout
cannot upgrade an older running service. The shared Phase-1 source inventory includes
the complete execution package. These references attribute service/guardian code,
not an ordinary Chia worker's loaded imports or the candidate command.

The A/B, repeatability and agentic-performance launchers (`chia_ab_batch.py`,
`chia_repeatability.py`, `chia_agentic_perf_experiment.py`) require
`--managed-native-endpoint` for execution; help and dry-run need no service.
Repeatability also needs no service when all selected repeats are already complete.
Its existing import-time Chia dependency remains: service-free help/dry-run is not
a claim that its CLI can run without the Chia environment. None of these launchers starts
a daemon automatically. The performance launcher requires both coordinator resources
(`codex_slots` and `gsim_slots`) on that same managed node.
The endpoint must belong to an explicitly provisioned supervisor on the driver's host.
This first integration supports **one managed Ray node only**: tasks are hard-pinned
to the driver's node, and its per-task CPU/simulator/provider resources are checked
before dispatch. Other nodes' resources cannot satisfy that admission check.

The performance-suite driver also requires `run --managed-native-endpoint PATH`.
This is a deployment input, not a scientific `Config` field: it is forwarded to
each wrapper and retained in its existing launch receipt, without rewriting the
suite definition. The suite checks the service's existing source seal before
provider configuration or its execution preflight; each wrapper checks again.
Archived wrappers predating the managed endpoint interface are refused before
provider/preflight execution. They are not silently upgraded: prepare and qualify
a new suite while leaving historical archives and receipts unchanged. This is not
a claim that every historical suite remains executable through the current driver.
`prepare`, `status`, `preflight`, and help do not require an endpoint (there is no
suite `dry-run` action).

After preparing a suite, an operator can start its supervisor in a separate
foreground terminal using the existing guarded command builder. For example,
from the repository environment (replace both absolute paths):

```python
import os
import sys
from pathlib import Path
from merlin_experiments.frozen_python import python_command

snapshot = Path("/absolute/suite/root/source")
argv = python_command(snapshot, [
    sys.executable, "-m", "merlin_experiments.execution.native_supervisor",
    "--endpoint", "/absolute/private-native/service.sock",
])
os.execv(argv[0], argv)
```

Provision the endpoint directory privately (owner-only permissions) and keep this
foreground service under operator supervision. Launch the suite with that exact
endpoint. A live-checkout supervisor, or one started from another suite's seal,
is rejected; environment variables cannot turn it into the required frozen service.
This command starts only the local AF_UNIX supervisor, not Ray. Ray deployment
still requires the explicit network policy described below. Nothing auto-starts
or stops an operator's service on behalf of a suite.

`chia/native.json` separately maps reservations to task references and records
independently observed native-tree cleanup and guardian reaping. A returned task
requires matching native-start/exit evidence; bypass cannot masquerade as execution.
Dispatch failure still leaves an owned reservation to cancel and observe. Native
teardown precedes task-group/collector shutdown, and incomplete evidence fails the
canonical AET run without replacing an existing primary exception. `chia/tasks.json`
retains its v1 meaning and `native_descendants: not_verified`; read the independent
native receipt rather than interpreting task acknowledgement as process cleanup.

Repeatability integration preserves resume selection, per-repeat scalar/report data
and AET outcome handling. The performance envelope retains its existing plan-digest checks,
native command artifacts, separately bound frozen transport, and content-addressed
launch/completion v1 receipt schemas alongside managed process ownership. The coordinator
still owns campaign ordering and resume. Both launchers' `wall_s` includes managed
command wait and cleanup overhead. Their wiring is covered by synthetic launcher/AET
tests and actual harmless managed AF_UNIX worker-body execution, including the
performance coordinator's original guarded child transport and receipt environment.
This is not a new real-Ray/network-isolated qualification. Neither this
single-node integration nor its tests qualify deployed multinode scheduling, cleanup
after service/host loss, or the complete frozen execution/accounting contract. In
particular, ordinary Ray worker imports are not certified frozen by this adapter.

The Chia extra pins official [ucb-bar/chia main at `16c35e92`](https://github.com/ucb-bar/chia/commit/16c35e92aaaf9511c6453bf94cd5cf589698f4e3),
verified as that branch's tip on 2026-09-23 (commit date 2026-09-04). This replaces the earlier
fork submission revision `9bf0bbb9`; it is not a claim about every development branch.
AET remains pinned independently to official main `903d4def`. Merlin uses the public profiler
module's collector lifecycle, directly constructs its public `MetricsBackend` subclass, and
resolves batches through ordered public scalar `get` calls under one timeout budget.
No private result envelopes, metrics registries or fork-only AET sink are imported.

`CHIA_AET_SINK=1` now opts into **Merlin's** public-trace adapter. It writes
`chia/accounting.json` beside the profile without replacing canonical AET identity/provenance;
inherited `CHIA_AET_RUN_*` cannot redirect it. Identical call-ID usage duplicates count once;
conflicting completion metadata refuses finalization. Only explicit public `add_info` metadata
with `billing_mode` and `cost_source: billed` can carry a classified reported cost into AET's
canonical billing splitter. Per-token spend and subscription dollar-equivalent remain separate.
Official main's built-in providers do not consistently publish those declarations: raw numeric
costs, absent usage and failed calls are **unknown**, never inferred paid or zero. Missing prices
remain unpriced. This adapter does not estimate prices or promise complete failed-attempt coverage.

The three managed native launchers select `accounting="child-ledgers"`: child AET runs retain
exclusive ownership of agent spend, and their Chia scheduling envelopes are never charged again.
Other direct-Chia workflows can use trace accounting, subject to the explicit-metadata limit above.

Scalar measurements remain under `chia/metrics.jsonl`; collected profiles live under `chia/`.
Those diagnostic files are not compiler certificates or official grades. Public collector snapshots
are observed-only, not an exhaustive billing ledger; absence is not evidence of zero consumption.
Frozen experiments retain their own required accounting/receipt gates.

Official-main adapter tests exercise real imports, metrics and AET writers with synthetic collector
and result transports. They do not requalify real-Ray profiled failures/cancellation, native cleanup
or complete frozen worker imports. The earlier real-hook evidence above remains historical.

Dependency updates must run the bridge lifecycle tests in the ordinary development environment,
then `merlin/tests/infra/test_chia_upstream_contract.py` in the isolated Chia environment.
**Network safety:** real-Ray opt-in tests additionally require an operator-provisioned
Linux network namespace/container with only loopback (`--network=none`, loopback up).
The tests refuse any non-loopback interface before starting Ray; setting an opt-in
environment variable does not override this gate. They do not provision namespaces,
containers or firewall rules. `address="local"`, disabling the dashboard, and setting
a node IP to `127.0.0.1` do not establish that every Ray/GCS listener is loopback-bound.
Earlier local test evidence establishes lifecycle behavior, not secure network
isolation. Production remote deployments require an explicit operator network policy;
this adapter does not certify an exposed Ray service as safe.

The latter exercises real Chia/AET accounting with synthetic events and replaced cluster transport;
set `MERLIN_TEST_CHIA_RAY=1` to additionally exercise two sequential workflows on a bounded local
Ray cluster with no LLM calls. Also run `merlin/tests/infra/test_chia_tasks.py`: its ordinary
tests exercise dispatch failures, interruption, unknown outcomes and incomplete cleanup; with
`MERLIN_TEST_CHIA_RAY=1`, its isolated local-Ray test confirms owned-task cancellation leaves a
borrowed peer running and able to finish. This does not qualify native descendants, multi-node
operation or hardware. No private Chia compatibility patches are supported: missing public APIs,
including Ray's `ObjectRef`, `wait`, `get` and `cancel`, produce an actionable environment error.

## Strict Phase 2 Codex accounting

Phase 2 passes the preflight-selected Codex executable and its stage-local
`codex_homes/` root directly to the shared provider. It does not temporarily mutate
`CODEX_BIN` or redirect Merlin's global artifact cache. The provider keeps its
existing per-round home naming, sandbox callback and process supervisor; session
continuation reuses the selected executable and home. Callers omitting these
explicit inputs retain the environment/cache defaults. This removes two shared
mutations, not all provider state or concurrency constraints.

`merlin_experiments.phase2.telemetry` owns preflight, lossless round evidence and final AET
publication for measured-claim authoring. Global/model-portfolio authoring uses its preflight
and round evidence, but does not thereby acquire a finalized tool ledger or cost report.

New preflight/source-policy v4 binds 29 implementation roles, the original explicit price-file
SHA256 and a normalized AET price snapshot. Its package-source role records the complete Phase 2
Python membership and hashes; a separate role pins the shared inventory helper. Added, removed or
changed package members invalidate execution without another role for each extraction. The
original native authoring renderer remains independently pinned. The narrow legacy-input adapter retains omitted
cache-read/write defaults of 0.10×/1.25× the input rate; nonfinite or negative rates refuse.
AET performs cost calculations against this snapshot for the canary, every round and final
trajectory. Ambient `AET_PRICE_TABLE`, cached Merlin rates and AET's bundled OpenAI snapshot
cannot override the declared input. Explicit input omission still uses the existing environment/
checkout configuration lookup at preflight, then pins the resolved file. Processing does not
mutate ambient price settings or the shared price cache.

The normalized snapshot is sealed as `telemetry/price_snapshot.json`, the tenth required final
artifact. Round and final records bind its digest. AET's six-decimal canonical cost is projected
to four-decimal legacy display fields; that rounding policy is recorded in the snapshot.
Subscription-notional amounts remain separate from billed spend. Existing run identity supplies
the target and suite; no target identity is guessed by the telemetry module.

Historical v1 (20-source) and extraction-v2 (23-source) declarations retain their exact source
sets and nine-artifact format; explicit-price v3 retains 27 roles and its ten-artifact format.
Historical v4 closure decoding checks the recorded names and aggregate digest without reading
live source files. Execution separately verifies current membership and bytes.
Readers do not add missing pins, reprice declarations, or rewrite their
reports; changed implementations still refuse live resume across treatment drift. The native
authoring-source role remains the actual prompt renderer, distinct from packaged telemetry.

External repos are passed by env var:

```bash
export MERLIN_XNNPACK_REPO=/path/to/XNNPACK
export MERLIN_AUTOCOMP_REPO=/path/to/autocomp
export MERLIN_EXO_REPO=/path/to/exo
export MERLIN_TRITON_REPO=/path/to/triton
```

Each adapter normalizes its source into `merlin/schemas/` artifacts (e.g. `kernel_record`,
`abstraction_candidate`, `policy_rule`) so all sources are comparable.

Note: xDSL is consumed as an optional library (adapters call its *tooling*); merlin's own prototype
dialects live in `src/merlin/xdsl_dialects/`.
