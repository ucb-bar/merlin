# Kernel-versus-compiler implementation status

This register records implementation status for the retained Radiance study tooling. It is separate from the Gemmini paper evaluations. Status labels do not replace numerical checks, certified execution records or measured development costs. The study board reads those quantities from the selected run matrices and baseline records, not from this register.

`DONE` records functionality implemented and verified by a test or recorded run, `PARTIAL` records an incomplete requirement, and `OPEN` records work not completed in this study snapshot. These labels are not a clean-machine reproduction certificate.

## Phase 0 — Measurement validation

| id | task | state |
| --- | --- | --- |
| 0.1 | Whole-model execution: weights-blob fix in `muon_harness` | DONE |
| 0.2 | Kernel provenance audit → contamination fence | DONE |
| 0.2b | Register `radiance-kernels` as an answer surface so `bwrap.coverage_gap` masks it | OPEN |
| 0.3 | Per-run authentication-file mounting | DONE |
| 0.4 | Pick + track the authoritative radiance contract, register pins | PARTIAL |
| 0.5 | Readiness fixes applied in an isolated worktree | DONE |
| 0.6 | Phase-0 exit gate: `readiness_check` GO, `preflight` GO_FOR_PILOT, oracle available, TinyLlama visible | PARTIAL |
## Phase 1 — Library modules (`packages/merlin-experiments/src/merlin/benchharness/`)

| id | task | state |
| --- | --- | --- |
| 1.1 | `evaluation.py` — `EvaluationResult` + whitelist `redact()` | DONE |
| 1.2 | `kernel_library.py` — signature-keyed store + reuse ladder L0–L3 | DONE |
| 1.3 | `task_basis.py` — weighted set cover over a census + eligibility map | DONE |
| 1.4 | `adaptation_ledger.py` — cumulative-cost curves | PARTIAL |
| 1.5 | Experiment scaffolding: `AGENT.md`, `README.md`, `METHODOLOGY.md`, `study.yaml`, `spec/radiance_bench.py`, `spec/reveal_order.yaml`, `spec/ladder.yaml`, `schemas/*.schema.json` | OPEN |
## Phase 2 — The common evaluator

| id | task | state |
| --- | --- | --- |
| 2.1 | One `evaluate()` both arms call | DONE |
| 2.2 | Kernel shim delegating scaffolding to the reference tool | DONE |
| 2.3 | Evaluation policy: `certifying_tiers = {L2}`; execution-only tier cannot certify | DONE |
| 2.4 | Evaluation policy: MX unwinnable = `unsupported`, out of both sides of the ratio | DONE |
| 2.5 | Evaluation policy: unavailable ≠ zero; unpriced turn ≠ $0 | DONE |
| 2.6 | Redaction audited against `answer_surfaces.audit_tokens` | PARTIAL |
| 2.7 | Latency provenance (`cycles_tier`, `cycles_cycle_accurate`) | DONE |
## Phase 3 — Inventory and deterministic task derivation

| id | task | state |
| --- | --- | --- |
| 3.1 | Weight-free census of all five workloads (`work_dir=` mandatory) | DONE |
| 3.2 | Captures for the reveal set | PARTIAL |
| 3.3 | Task basis: signature key, eligibility filter, cost weight, greedy cover ≥95%, family floor | PARTIAL |
| 3.4 | `basis_certificate.json` | PARTIAL |
| 3.5 | Config ladder C0–C4 (~5 per task) | OPEN |
| 3.6 | `holdout_certificate.json` proving shape-tuple disjointness | OPEN |
## Phase 4 — The arms

| id | task | state |
| --- | --- | --- |
| 4.1 | Arm A1 Codex seat, from the specification | DONE |
| 4.2 | Arm A2 Bedrock qwen3-coder-480b | DONE |
| 4.3 | Arm A3 Google gemini-3.5-flash | DONE |
| 4.4 | Mutual blindness enforced, not asserted | DONE |
| 4.5 | Add the blindness + contamination checks to `verify_no_cheat` | OPEN |
| 4.6 | Multi-seed repeats (≥3) | DONE |
| 4.7 | Arm G AutoComp, real framework, the shared evaluator | DONE |
| 4.8 | AutoComp historical re-accounting + contaminated-by-construction label | PARTIAL |
| 4.9 | Arm F kernel-library reuse + generalization matrix | OPEN |
| 4.10 | Arm B `Merlin-Base` — generate the backend, no optimized kernels in scope | OPEN |
| 4.11 | Arm B `Merlin+Seed` — mine TinyLlama trajectories through the promotion ladder | OPEN |
| 4.12 | Anti-specialization audit (5 mechanical tests) | OPEN |
| 4.13 | Freeze + sequential reveal, zero-LLM **enforced** (no creds bound, ledger asserts 0 calls) | OPEN |
| 4.14 | Does the optimization stage pay for itself? | OPEN |
## Phase 5 — Accounting, metrics, plots

| id | task | state |
| --- | --- | --- |
| 5.1 | Token/cost telemetry per round incl. failures | DONE |
| 5.2 | Preserve all intermediates | DONE |
| 5.3 | AET emission (`emit_to_aet`, `MERLIN_AET_SINK=1`) + `agg_by_model` | OPEN |
| 5.4 | Coverage metric: % independently eligible execution cost accelerated | OPEN |
| 5.5 | Performance metric vs reference and best-observed, geomeans, %peak | PARTIAL |
| 5.6 | Break-even per reveal prefix, independently for tokens / time / notional $ | OPEN |
| 5.7 | Statistics: median, geomean, bootstrap CIs; failures stay in the distribution | OPEN |
| 5.8 | Six 2D curves + 3D Pareto, PDF/SVG/PNG + CSV/JSON | OPEN |
| 5.9 | Time-to-first-correct and time-to-within-X%-of-best curves | OPEN |
| 5.10 | Model × family capability matrix | OPEN |
| 5.11 | Cert (GSIM) pass over accepted kernels | PARTIAL |
| 5.12 | Reference baseline every arm is relative to | DONE |
| 5.13 | Failure taxonomy: an unparseable submission is not a `tool_crash` | OPEN |
## Phase 6 — Pilot gate

| id | task | state |
| --- | --- | --- |
| 6.1 | Three structurally different tasks end to end | DONE |
| 6.2 | Full chain incl. AET trajectory, `usage_complete=true`, `freeze.json` | PARTIAL |
| 6.3 | 3×5 reuse matrix at zero LLM cost | OPEN |
| 6.4 | Same pilot on every arm, in parallel and blind | DONE |

## Critical path

Ordered requirements: 5.6, 3.3, 3.5, 3.4, 3.6, 4.9, 1.2, 4.10, 4.13, 5.4, 5.7.

## Check recorded outcomes

Use `merlin experiment study board --register <path-to-this-file> --runs-root <run-directory>` with the selected baseline and `voided_runs.yaml`. A voided run remains in development-cost accounting and receives no performance credit. Missing measurements remain unavailable. Compare cycle counts only at the same execution fidelity and numerical policy.
