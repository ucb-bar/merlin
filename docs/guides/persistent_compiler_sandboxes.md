---
title: Reuse one compiler sandbox per grading worker
kind: guide
status: current
owner: experiments
last_verified: 2026-11-09
related: [bubblewrap_argument_budget, fresh_campaign_launch]
code_refs:
  - src/merlin/experiments/phase1/feedback/sandbox_pool.py
  - src/merlin/experiments/phase1/feedback/snapshot_execution.py
  - src/merlin/experiments/phase1/options.py
---

# Reuse one compiler sandbox per grading worker

Under `--compiler-execution snapshot_external_v1`, compiler calls can reuse a
sandbox built from the original per-call policy. This avoids repeating namespace
and mount setup for every entrypoint while retaining a required reset boundary.
Actual startup cost depends on the selected host, policy and concurrent work.

## What changed

`--sandbox-reuse persistent|per_call` (default `persistent`) selects the mechanism. The controller
exports it as `MERLIN_SANDBOX_REUSE` to every grading child.

| | sandbox starts |
|---|---|
| before, per compiler call (one case step runs at least one entrypoint) | 1 |
| after, per compiler call | 0 in steady state |
| after, per `SnapshotExecutor` | one per concurrently active worker, plus one after any failed reset |
| package build (`build_package`) | 1 per build, unchanged |
| agent command (Codex) | 1, unchanged (see below) |

A `SnapshotExecutor` keeps a pool of sandboxes. It builds each one from exactly the policy argv a
per-call launch would use, with a private IO directory bound at the executor's template path. A
worker takes an idle sandbox, runs one call through a small server inside it, and returns the
sandbox. Starts and closes happen on one long-lived thread, so `--die-with-parent` does not fire when
a worker thread exits.

## Isolation, per call

- **Same grants.** The persistent sandbox's policy is the per-call policy, so golden masks, answer
  masks, read-only toolchains and the unshared network are unchanged. Expected outputs and grader
  state never enter it, and the grader stays outside.
- **Processes.** After each call the server SIGKILLs every other process in the sandbox's PID
  namespace. Reuse is refused, falling back to one sandbox per call, unless the policy unshares the
  PID namespace and mounts its own `/proc`.
- **Writable state.** tmpfs masks, `/dev/shm`, and capped binds that replace tmpfs under a resource
  boundary are emptied back to the skeleton bwrap created. Mount points are never removed, and
  directory modes are restored. The host stages the call's IO directory in from the call's scratch
  and back out to it, so the directory is empty between calls. Reuse is refused if anything is
  mounted below it.
- **Failure.** If a reset fails (a process survives, the mount table changed, or files remain), that
  sandbox is discarded and the next call starts a fresh one. If the server dies or hangs, the call
  fails like a killed process or a timeout, and the sandbox is discarded.
- **Channel.** The server is non-dumpable, so a call cannot open its request or reply pipes through
  `/proc`. Children get `/dev/null` stdin and memfd stdout/stderr.

**Threat model.** Every call in one executor runs the same candidate compiler, which already
controls its own outputs. Reuse adds no path to goldens. The only new surface would be state carried
from one call into a later one, and the per-call reset removes the carriers the candidate controls:
processes, tmpfs files, `/dev/shm` and IO files. The existing package policy does not unshare the
IPC namespace, so SysV IPC objects were already host-shared across per-call sandboxes. That is
unchanged, and the reset does not touch them because they belong to the host namespace. Kernel
keyrings are likewise unchanged.

## Compatibility

When `MERLIN_SANDBOX_REUSE` is unset, the mechanism is `per_call`, so tests and older launchers
behave as before. `run_config.sandbox_reuse` records the selection. A resumed run adopts its
recorded value, and a record without one means `per_call`, so an older run keeps one sandbox per
call after a restart. Fresh-campaign specs do not pass the flag, so a new launch gets `persistent`.

## Author commands are separate

This pool is used by the candidate compiler executor. The author session and its
tool commands keep their selected provider permission profile and transport;
the compiler reuse option does not alter them. A new provider or sandbox version
still needs its ordinary independent deployment checks.

`sandbox_pool.py` owns policy eligibility, worker reset and disposal.
`snapshot_execution.py` owns the exact per-call staging and invocation records;
`phase1/options.py` owns the fresh-controller selection. This reuse does not
change the author provider transport or its selected permission profile.
