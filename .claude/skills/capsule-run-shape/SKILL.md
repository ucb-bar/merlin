---
name: capsule-run-shape
description: How a capsule-bench run is shaped and why — one continuous agent session, per-capsule L2→L3 promotion, and cert preservation. Read before changing the QA loop, the simjob broker, or tier promotion.
---

# Capsule-bench run shape

## The shape

**One long-lived agent session, re-graded underneath it.** Default; no flag needed.

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

- **`depends_on` is declared by 0/48 gemmini capsules**, so every agent edit invalidates every
  certificate (`invalidated by <whole-submission>`). The machinery is complete; only adoption is
  missing. Declare it **before** a cohort freeze — never mid-round, which changes what the agent is
  graded against.
- `GP0`/`GP1`/`GP2` declare `epilogue: ['maxpool']`, which `command_buffer.schema.json` forbids; the
  interpreter pools via `pool_*` attributes on the store instead. `additionalProperties: True` lets a
  wrong carrier validate silently. Ratcheted in
  `merlin/tests/targetgen/test_capsule_carrier_consistency.py`.
