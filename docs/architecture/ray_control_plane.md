# Ray Control Plane

Merlin's Ray control plane makes Ray the execution and orchestration layer for
TargetGen-driven workflows without replacing TargetGen's capability model or
planner.

The intended split is:

- **TargetGen** owns hardware capability specs, normalization, classification,
  support plans, and execution bundles.
- **Ray** owns cluster lifecycle, job submission, run metadata, artifacts, and
  shared-resource coordination.
- **MCP** is the interoperability layer that future Claude Code, Codex, and
  other agent clients should talk to.

## Control-plane layers

The v1 control plane has four layers.

### 1. Planner layer

`merlin targetgen` remains the single source of truth for:

- capability validation
- deployment overlays
- support-plan generation
- verification ladders
- execution bundles and prompt packets

Nothing in the Ray layer re-derives those concepts.

### 2. Execution layer

`merlin targetgen execute` now supports two engines:

- `--engine local`
- `--engine ray`

`local` runs the current in-process executor and operator-gate state machine.

`ray` persists the same execution bundle and submits a Ray Job whose command is
the local executor:

```text
conda run -n merlin-dev uv run tools/merlin.py targetgen execute --engine local --from-dir <target-dir> --resume
```

This keeps one execution contract while allowing Ray to own scheduling,
submission ids, logs, and future resource placement.

### 3. Run-metadata layer

The Ray control plane stores Merlin-owned metadata under
`build/generated/ray/`.

Current state is file-backed so it remains debuggable even before the full
Serve/MCP layer lands:

- `cluster/bootstrap.json`
- `runs/<run_id>/run_request.json`
- `runs/<run_id>/run_record.json`
- `runs/<run_id>/artifacts.json`
- `resources/leases/<lease_id>.json`

These records are Merlin's durable source of truth. Ray Dashboard or Ray's
State APIs are observability sources, not the canonical ledger for Merlin.

### 4. Shared-resource layer

The first shared-resource mechanism is a lease system exposed by
`merlin ray resources`.

This is intentionally simple:

- leases are explicit
- the resource key is the scheduling key
- only one active lease per resource key is allowed

That is enough to broker scarce assets such as a FireSim U250 slot, a board
host, or a simulator lane while the richer actor-based broker layer is being
built.

## CLI surfaces

### `merlin ray`

The new top-level CLI surface owns Ray-related control-plane operations:

- `cluster start-local|status|stop`
- `jobs submit|status|logs|cancel`
- `resources list|reserve|release`
- `artifacts list|fetch`

The implementation shells out to the `ray` CLI when it is installed. If Ray is
not installed or no cluster is active, Merlin still emits a run record with a
blocked state and an actionable message. This keeps the control plane usable in
developer environments that do not yet have Ray bootstrapped.

### `merlin targetgen execute --engine ray`

This is the main integration point between TargetGen and the Ray control plane.

It does three things:

1. builds or reloads the current execution bundle,
2. persists the bundle to the normal TargetGen output directory,
3. submits the local executor as a Ray Job and records a Merlin run id.

## Resource and artifact model

### Run records

Each Ray-backed run records:

- target name
- workflow name
- source of the request
- exact command submitted
- workdir
- target artifact directory
- Ray submission id
- current run status
- optional status message

### Artifact records

Artifacts are currently indexed by scanning the target artifact directory for
files. This keeps v1 simple and works well for TargetGen-backed runs because
their outputs are already organized under one directory tree.

### Resource leases

Resource keys are intentionally concrete. Examples:

- `firesim_u250`
- `spacemit_x60_board_01`
- `radiance_sim_lane_a`

If you need multi-instance capacity later, model each concrete schedulable unit
as its own key first, then add higher-level resource catalogs after the actor
broker layer lands.

## Next steps

This is the bootstrap slice of the Ray control plane. The next planned layers
are:

- Ray Serve APIs
- an MCP gateway
- actor-based hardware brokers
- direct Claude/Codex runtime adapters
- Ray Tune-backed search workflows
