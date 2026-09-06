---
name: capsule-run-shape
description: How a capsule-bench run is shaped and why — one continuous agent session, per-capsule L2→L3 promotion, and cert preservation. Read before changing the QA loop, the simjob broker, or tier promotion.
---

# Capsule-bench run shape

## The shape

**`--schedule continuous` with a long `--round-timeout`** (e.g. 43200). Rounds are not a terminator:
the run stops on evidence (converged / plateaued) or a declared budget, and the post-freeze
public+hidden L3 grade still runs, so a formal success is reachable.

**Not `--continuous`.** That is the legacy single-session path: it re-grades under one session but skips
the post-freeze public+hidden grade, returns 1, and hardcodes `formal_complete=False`. Measured
2026-09-01: both gemmini sessions closed after ~1.5 h at 18/33 with `grades=2` when the agent stopped,
well inside a 12 h round timeout.

```
agent works continuously
   ├─ background grader every --grade-interval (900s): grades a SNAPSHOT COPY → qa/verdict.json
   ├─ capsule ladder is per-capsule, cheapest-first: L2 pass → L3 in the SAME grade
   └─ tier_promote: a loop verdict enqueues a cert job at once
        → capsule 2's L3 runs while the agent is still on capsule 1's L2
        → record_cert writes the outcome back against the digest that earned it
```

Grading a snapshot copy is what makes it safe to grade while the agent works: the agent cannot mutate
what is being measured, and the grade cannot be contaminated mid-flight. `_qa_work/cand_NN/` is that
copy; `_qa_work/runs_NN/` holds the per-capsule verdicts for the same grade.

## Why it is gated

Promotion is wrapped in a `try/except` so it can never gate a run. That is correct — but it means a
**broken** promotion is indistinguishable from an **idle** one. Five defects hid in this single path on
`merlincirct_arm4_func_20260901_v4`, and none looked like a failure:

| defect | how it presented |
|---|---|
| enqueued `sim="contract"`, rejected on any bespoke-sim target | "nothing needed promoting" |
| verilator slot dir was a fixed `/tmp` path owned by another **user** → broker crashed | "the L3 infrastructure crashed" |
| forwarded `--tiers` to `agent_selfcheck.py`, which had no such flag | "no verdict produced" |
| broker ran children with `stdout/stderr=DEVNULL` | no diagnostic existed at all |
| a completed promotion's result was never written back | the capsule is "still pending" |

Cost: async L3 certification was **structurally impossible** for three whole grades. The agent made
real fixes it could not verify, its score sat at 19/33, and the scoreboard implied the agent was the
problem. Its own round reports named the cause each time.

**Rule: if you touch this path, add a test.** A comment cannot detect silence.

## A certificate is not re-bought for bytes that have not changed

The cert tier is the whole cost of a grade. Measured on `merlincirct_g4p1_20260905` round 5: the screen
tier cost **2.53 min** of adapter wall over 82 capsules and the cert tier **44.4 min** over 76 — 94.7% of
the total. A converged submission re-paid all of it on every post-turn grade, for capsules whose emitted
program had not changed a byte since a certificate was earned for it, sometimes minutes earlier in the
same turn by the promotion path above.

`merlin.targetgen.tier_cache` is the reader. The ladder asks it before paying for a tier and tells it
after executing one:

- **Key** = the *execution identity* (`tier_promote.execution_digest`'s own — ELF bytes x target x every
  declared hardware pin, deliberately excluding merlin's commit) **x** the *instrument digest* (a
  `source_digest` over the files that decide a verdict, plus the elaborated-RTL engine that would answer
  the tier today). Same program, same device, same judge — or it runs again.
- **Fail closed** on every ambiguity: no ELF, an `UNKNOWN` pin, a missing grading-path file, an
  undecidable engine, an unreadable store, a record whose own copy of the key disagrees. A cache that
  cannot miss is the same defect as a check that cannot fail.
- **Only a `pass` is carried.** A failure is what an agent acts on, so it is re-run for its detail.
- **The first tier the ladder executes is always paid for** — its adapter is what builds the program the
  key is taken from, so nothing can be carried until a real measurement has happened. A stale ELF left in
  a reused run directory is REMOVED before the ladder starts; without that, a lookup would key on a
  program the current compiler no longer emits (the one direction this must never fail in).
- **A carry says so.** The tier record carries `measured_now: false` plus a `carried` block (identity,
  instrument, when, which run, and where the earning run's console lives); the capsule result grows
  `tier_reuse: {executed, carried}`; the grade score and the round verdict carry the same accounting. A
  falling wall clock has two possible causes — a faster grade and a grade that measured less — and only
  the reuse block tells them apart.
- Wall-clock `timing`/`concurrency` are dropped from a carried record (no time was spent now); `cycles`
  are kept (a property of the program and the device, which the key pins).

Knobs: `MERLIN_TIER_CERT_CACHE=0` switches it off, or set it to a directory to relocate the store
(default `out/artifacts/cache/tier-certs/`); `MERLIN_TIER_CERT_LEDGER` names `qa/tier_state.json`
promotion ledgers to consult as a second, strictly weaker source. Tests:
`merlin/tests/targetgen/test_tier_certificate_cache.py` — every direction is a mutation of a state that
hits.

## Invariants to preserve

- `--continuous` stays default-on; `--no-continuous` stays reachable.
- A cert job must name a sim `simjob_broker._allowed_sims()` accepts — derive it, never hardcode a
  sentinel (`tier_promote.cert_sim`).
- Never mark a capsule `pending` for a job that was not actually enqueued; that is how one strands.
- `record_cert` must not recompute the digest. A cert belongs to the bytes that were pending when the
  job was enqueued; re-hashing re-attributes it to whatever has been edited since.
- An unattributable result is not recorded, never guessed.
- Coordination paths (slot semaphores, queues) are **per-user** and honour `TMPDIR`. A fixed `/tmp`
  path is squattable by the first user on a shared host.
- `BMC.run` invokes verilator **directly, outside** the slot arbiter. Anything measuring on RTL should
  acquire a slot rather than queue-jump.

## Known open items

- **`depends_on` is RETIRED** (was: 0/48 gemmini capsules declared it). It is gone from
  `capsule.schema.json` because no capsule can honestly narrow: `capsule_common.run_entrypoints` runs
  every declared command for every capsule, so an edit to any of them can flip any verdict, and a
  narrowed declaration keeps a certificate the current bytes did not earn. What survives an edit is
  decided by the SUBMISSION's decomposition instead — `promote()` gives every capsule the full
  component set, so bytes the derivation proved no command can read (`inert`) stop invalidating
  anything. See the block above `promote()` in `tier_promote.py`.
- `GP0`/`GP1`/`GP2` declare `epilogue: ['maxpool']`, which `command_buffer.schema.json` forbids; the
  interpreter pools via `pool_*` attributes on the store instead. `additionalProperties: True` lets a
  wrong carrier validate silently. Ratcheted in
  `merlin/tests/targetgen/test_capsule_carrier_consistency.py`.
