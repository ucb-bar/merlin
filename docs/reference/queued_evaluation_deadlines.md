---
title: Queued evaluation deadlines
kind: reference
status: current
owner: merlin-experiments
last_verified: 2026-10-20
related: [phase2_corpus_search, phase2_component_search]
code_refs:
  - packages/merlin-experiments/src/merlin_experiments/phase2/search_coordinator.py
  - packages/merlin-experiments/src/merlin_experiments/phase2/queued_search_actions.py
  - packages/merlin-experiments/src/merlin_experiments/phase2/evaluation_pool.py
---

# Queued evaluation deadlines

An operator may select `SearchSettings(queued_job_timeout_seconds=2400)` in the
ordinary search configuration. This bounds the queued evaluator independently
of its broker submission acknowledgment. The setting accepts an integer from
1 to 2400 seconds that fits within `soft_seconds - final_reserve_seconds`.
Omitting it preserves the original request-derived job timeout.

The broker keeps its original request deadline and acknowledgment checks.
Synchronous actions keep that same deadline. The coordinator checks the full
selected job allowance against the remaining exploration budget and passes an
absolute campaign completion deadline to the original evaluation pool. Normal,
selected, impact-selected and component queue submissions use this route when
their search coordinator is present. Direct pool calls retain their existing
timeout unless their caller explicitly supplies a completion deadline.

Setup and queue waiting cannot renew the campaign boundary. An expired queued
job refuses before launching a worker, including when all worker slots are busy.
At dispatch the evaluator receives the lesser of its selected duration and the
remaining campaign allowance. The existing guardian supervises that deadline;
late results refuse and original descendant cleanup/resource release remain
required. Selected deadlines appear in job status and execution admission.

The setting grants no additional job, CPU, memory, disk, engine, evaluator or
scientific authority. It does not change numerical, source, instruction, timing,
promotion or final qualification gates. Process supervision and cleanup retain
their existing practical limits; this clock selection does not establish a new
OS resource guarantee. Neutral local-worker tests demonstrate transport and
lifecycle behavior only.
