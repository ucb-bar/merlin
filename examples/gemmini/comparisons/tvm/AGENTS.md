# Current TVM campaign instructions (2026-10-27)

Read [FIRESIM_HANDOFF.md](FIRESIM_HANDOFF.md) and [TENSORIR_OP_AUDIT.md](TENSORIR_OP_AUDIT.md).
The sole model-running protocol is Merlin `8ec99308cb1182b9b896a6075300312266fadf6f`.
Use the frozen run contract, with the user's approved TinyLlama numerics correction
from 128 to 32 prefill tokens; final chain is 32+32, cache capacity 64. Use only
the pinned protocol's quality policy. Earlier PPL/top-1/cosine thresholds and
checkout-default CURRENT_RUN_POLICY are superseded for this campaign.

All three models are enabled. Tune ResNet for four hours total across its chunks,
TinyLlama and SmolVLA for one hour each, or earlier evidenced convergence. Run
models concurrently subject to actual allocated slots. Final baseline/tuned E2E
runs follow tuning and are outside these allowances. Search existing graph
options and TensorIR schedules. The user approved one slot per model concurrently
and actual tuning-occupancy accounting for now. The new simplified compiler removes automatic prepacking and bias folding.
Begin with fusion=2/CSE off, then search stock fusion/CSE and TensorIR schedules
within the same budgets; old graph-plan identities require regeneration. TinyLlama tuning uses published 16/1 row probes; E2E remains 32/32.
This agent prepares ELFs or build instructions; the remote agent owns deployment.

Keep ordinary mathematical TensorIR and TensorIntrins. Derive device admission
from exact software-spec declarations and their proof obligations; similarity is
not support. No new compiler passes or hidden CPU fallback for eligible device
work. Keep tiling/movement schedule choices visible to stock MetaSchedule and
record unsupported search axes. Device code belongs in the external TVM backend;
this directory owns trusted comparison/preparation orchestration.

Preserve historical source trees, numerical releases, failures and measurements;
do not relabel their inputs, recipes, compiler or hardware identities. Use the
selected capture, independent references, package-session bridge and existing
prepare/collect tools. Validate every declared array including carried state and
padding. Full-model qualification is separate from capsule and small-op tests.
Use actual candidate state between calls; never substitute reference state.

The remote worker binds actual manager, physical slots, image, target clock,
memory map and wall-watchdog/cancellation evidence. Do not ask the user for these
details to finish ELF preparation. No sequential Spike preflight is selected for
remote candidate validation; preserve guest checks and wrong-oracle controls.
Keep baseline and tuned complete chain timing comparable; never sum operator minima into E2E cycles. No cycles
may be supplied from estimates. Final E2E wall timeouts only are 3,600 seconds
for ResNet and 21,600 for TinyLlama/SmolVLA per baseline/tuned run. Tuning budgets
remain 4h/1h/1h; candidate hang watchdogs are separate remote-runner settings.
The user approved median-based candidate wall watchdogs in place of incumbent
cycle cancellation. Retain completed slow cycles; do not require a live observer.
Before tuning, the remote runner must qualify guest correctness, wall cancellation
and safe slot reuse under ADAPTER_CONTRACT.md. Spec-pinned oracle provenance is
still missing; locally repaired diagnostic Spike is distinct evidence.

Read root AGENT.md/CLAUDE.md and the companion TVM apps/gemmini instructions.
Use merlin.common.artifacts.new_product/new_measurement for outputs. Keep large
payloads outside Git. On this shared VM use bounded workers and single-threaded
BLAS; ask before long/heavy preparation. No sudo, user namespaces, destructive
actions, publication or PR without authorization. Preserve other worktrees.

Run relevant tests and python build_tools/scripts/check_structure.py; report
inherited failures separately. No new full-model hardware success is claimed
until source, numerical, placement, exact-kernel, instruction and deployment
checks bind actual complete-chain execution.

At the per-model tuning limit, stop admitting new work and **finish active runs**.
Do not start queued jobs during this completion period. Record the time beyond
budget separately; preserve failure/hang safeguards. This adds
no new tuning allowance and does not renew a continuation's budget.

The user approved retaining only spec-declared, proved composite TensorIntrins
while removing custom graph passes. Composite scope is settled; per-family proof
and execution admission still need evidence. The approved median policy uses three
baseline timings, 3x the latest-nine median with a 3x baseline floor, and an explicit
hard ceiling. Freeze per admitted job; bootstrap/ceiling values come from the remote
pilot. No sequential Spike preflight and no incumbent-cycle filtering.
