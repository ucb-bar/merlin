# AGENT.md — merlin/experiments/capsule_bench

Status: active

## Purpose
The capsule benchmark: give a coding agent a hardware target it has never seen, let it build a
compiler backend, and grade what it produces against an independent oracle. One harness, six targets,
a fixed four-arm methodology.

A **capsule** is one graded task — an operation, its shapes and dtypes, an MLIR interface the agent
compiles against, and a golden the agent never sees. Capsules live in `merlin/contract/capsules/`,
not here. Shared execution lives in `merlin_experiments.phase1`; this directory
retains native launch adapters and authored experiment resources pending migration.

## Layout
- `packages/merlin-experiments/src/merlin_experiments/phase1/` (from repo root) — canonical
  controller, admission, task/workspace handling, authoring, providers, feedback and audit.
  `run_baseline_qa_loop.py` supplies legacy defaults to this controller; do not add a
  second implementation there. Installed invocation is `python -m merlin_experiments.phase1`.
- `harness/` — the retained native drivers and brokers. Entry points worth knowing: `run_agent_experiment.py`
  (launch), `preflight.py` (the GO/NO-GO gate, and the writer of every `bundle_lock.yaml`),
  `verify_no_cheat.py` (answer-surface audit), `capsule_grade` via `merlin.targetgen`,
  `generalization_difftest.py` (the staircase), `plots/` (report figures).
- `targets/<target>/` — everything specific to one target: `target_experiment.yaml` (the descriptor
  everything else is parameterized by), `task/` (what the agent is asked to do), `contracts/`
  (the ISA/hwbringup material the agent is allowed to read), `input_bundles/` (one per arm),
  `scripts/agent_selfcheck.py`.
- `proxy/` — the model-access shim.
- `full_public_capsules/` (under `harness/`) — the public contract mirror the harness reads.

## How a run is shaped — continuous, per-capsule, and cert-preserving (the DEFAULT)

A run is **one long-lived agent session**, re-graded underneath it. This is the default; there is no
flag to remember. `--schedule rounds` exists only to reproduce a legacy round-relaunch run.

Three properties, each enforced by `merlin/tests/infra/test_continuous_is_the_default.py`:

1. **No round barrier.** A background grader re-grades a *snapshot copy* of the workspace every
   `--grade-interval` seconds (default 900) and refreshes `qa/verdict.json`, so feedback arrives while
   the agent works. Round relaunches discarded the agent's context at every barrier and deferred every
   verdict to the next one. Measured: three round-based grades scored 19 / 18 / 19 of 33 while the
   agent rebuilt context each round — the barrier bought nothing.
2. **Per-capsule tiering, not a batch.** `capsule_runner` walks each capsule's ladder cheapest-first,
   so a capsule that clears the loop tier (L2) continues to the certifying tier (L3) **in the same
   grade**. Independently, `merlin_experiments.phase1.feedback.promotion.promote`
   enqueues a cert job the moment a loop verdict lands,
   so **capsule 2's L3 starts while the agent is still working on capsule 1's L2**.
3. **The certificate is kept.** The same promotion module's `record_cert` resolves the `pending` entry the promotion
   created, against the digest of the bytes that earned it — never re-hashed, so a cert is not
   re-attributed to whatever has been edited since. A result with no pending entry cannot be
   attributed and is therefore not recorded rather than guessed.

**Why this is gated rather than documented and hoped for.** Every one of these failed silently at some
point on `merlincirct_arm4_func_20260901_v4`, and none of them looked like a failure:

| what broke | how it looked |
|---|---|
| promotion enqueued a sim the broker rejects | "nothing needed promoting" |
| the verilator slot dir was owned by another user | "the L3 infrastructure crashed" |
| `--tiers` was forwarded to a child that had no such flag | "no verdict produced" |
| the broker discarded child stdout/stderr | no diagnostic existed at all |
| a completed promotion was never recorded | the capsule is "still pending" |

Promotion is deliberately wrapped in a `try/except` so it can never gate a run — which is exactly why
a broken promotion is indistinguishable from an idle one, and why these are tests and not comments.

## The four arms
Nested, so a delta is attributable to exactly one addition:
`raw_baseline` ⊂ `cpp_merlininfra` ⊂ `merlin_assisted` ⊂ `merlin_assisted_rtlchecks`.
An arm is defined by its **input bundle** — the set of paths it may read — not by its prompt.

## Bundles, locks and the sandbox
`input_bundle_manifest.yaml` declares `allowed` and `denied` paths. Grants are repo-root-relative,
with one shorthand: an `experiments/...` grant resolves under `merlin/`. **Every consumer must
resolve through `merlin.targetgen.sandbox.bwrap.resolve_grant`** — the binder and the lock writer
using different resolution is exactly how 17 grants ended up mounted but unpinned.

`bundle_lock.yaml` is generated by `preflight.py`, never hand-edited. It records the tree hash of
each granted path, plus `unresolvable_grants` for anything that resolves nowhere — a grant the arm
cannot actually read is a defect, not something to omit.

The sandbox is `bwrap`, deny-by-default, with answer surfaces derived rather than listed. Never
weaken a mask to make a run work.

## Hard rules
- **Answer keys are never tracked.** `hidden/`, `golden.yaml`, `expected_instruction_coverage.yaml`,
  `*.hidden.yaml`, `*.safetensors`. This repo is public; committing one defeats the benchmark. The
  gate matches on path, so a holdout name in the *contents* of an ordinary-looking file passes it —
  check contents too.
- **A pass count is not a certification tier.** The default public-corpus workflow preserves the
  highest declared reachable tier using actual constructed adapters, not the cheapest QA-loop tier
  or advertised capability metadata. Missing/disjoint requirements remain explicit; no cheaper tier
  may substitute for one the capsule declared. Report the actual tier evidence beside pass counts.
  `merlin_experiments.corpus.admission` owns selection; core corpus primitives own publication and
  coverage. Keep native promotion and final public/hidden grading authoritative.
- **Core compiler machinery may not depend on this harness.** Core source is `src/merlin`;
  `merlin/python/merlin` is only a compatibility symlink. Shared phase/evaluation tooling belongs to
  `packages/merlin-experiments`, with declared host-only access identities. Structure checks enforce
  the static dependency direction and the existing library-resource rules.
- **Generated output goes to `out/`**, never in-tree. Use `start_run(..., target=...)`.
- The harness is parameterized by `MERLIN_TARGET_EXPERIMENT`; it defaults to one target, which is
  why that target reads the shared un-namespaced corpus roots while the others are namespaced. Do
  not add a second such default.

## Before running anything
`preflight.py` and `readiness_check.py` are the GO/NO-GO gate. A NO_GO that produced results anyway
has happened; treat the verdict as binding.
