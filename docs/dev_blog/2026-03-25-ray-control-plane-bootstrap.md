# 2026-03-25: Ray Control Plane Bootstrap

> **Repro pin:** merlin@[`2903e28b`](https://github.com/ucb-bar/merlin/commit/2903e28b0ec45c4e109fa9c98592b7f0353fcf40) · iree_bar@[`ddf4685ae1`](https://github.com/ucb-bar/iree_bar/commit/ddf4685ae1)
> **Status:** Active

This log tracks the first Merlin-owned Ray control-plane slice.

The goal of this workstream is not "use Ray somewhere". The goal is:

- keep TargetGen as the planning source of truth,
- make Ray the run-submission and execution control plane,
- make future MCP and agent interoperability sit on top of that.

## What landed first

The first bootstrap slice adds:

- a new `merlin ray` CLI surface,
- file-backed cluster/run/artifact/resource metadata under
  `build/generated/ray/`,
- `targetgen execute --engine ray`,
- a new architecture note for the Ray control plane.

## Why start this way

Merlin already had a useful local TargetGen executor. Replacing it outright
with a distributed system would have created two moving targets at once.

The bootstrap decision was:

1. keep the local executor,
2. submit that executor as a Ray Job,
3. let Merlin own the run records even before the full Serve/MCP layer exists.

That gives us a clean migration path:

- one planner,
- one execution contract,
- one future distributed control plane.

## Local bootstrap flow

When Ray is installed:

```bash
conda run -n merlin-dev uv run tools/merlin.py ray cluster start-local
conda run -n merlin-dev uv run tools/merlin.py targetgen execute \
  target_specs/examples/nvidia_vulkan_ada/capability.yaml \
  --overlay target_specs/examples/nvidia_vulkan_ada/overlays/desktop_local.yaml \
  --engine ray
```

If Ray is not installed yet, the run is still materialized as a Merlin run
record with a blocked status and an actionable message. That behavior is
intentional because it keeps the interface stable during bring-up.

## Current run metadata

The current run root is:

```text
build/generated/ray/
```

Current records:

- `cluster/bootstrap.json`
- `runs/<run_id>/run_request.json`
- `runs/<run_id>/run_record.json`
- `runs/<run_id>/artifacts.json`
- `resources/leases/<lease_id>.json`

These are meant to be easy to inspect while the Ray Serve API and MCP gateway
are still under construction.

## Test commands

```bash
# Start a local Ray cluster and write the bootstrap record
./merlin ray cluster start-local

# Submit a TargetGen run through the Ray engine
./merlin targetgen execute \
  target_specs/examples/nvidia_vulkan_ada/capability.yaml \
  --overlay target_specs/examples/nvidia_vulkan_ada/overlays/desktop_local.yaml \
  --engine ray

# Verify the run record landed
ls build/generated/ray/runs/
cat build/generated/ray/runs/<run_id>/run_record.json
```

If Ray itself is not installed, the second command still materializes a run
record with status `blocked` and an actionable message — that is intentional
during bring-up.

## Immediate follow-ups

- Add Ray Serve apps for run, artifact, and review APIs.
- Add the MCP gateway on top of those APIs.
- Move board, FireSim, and simulator coordination behind broker actors.
- Add real Ray-backed status and log handling once the runtime is present in the
  default Merlin environment.
