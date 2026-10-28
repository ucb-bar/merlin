# Current TVM campaign instructions (2026-10-27)

Read [FIRESIM_HANDOFF.md](FIRESIM_HANDOFF.md) and [TENSORIR_OP_AUDIT.md](TENSORIR_OP_AUDIT.md).
The sole model-running protocol is Merlin `8ec99308cb1182b9b896a6075300312266fadf6f`.
Use the frozen run contract, with the user's approved TinyLlama numerics correction
from 128 to 32 prefill tokens; final chain is 32+32, cache capacity 64. Use only
the pinned protocol's quality policy. Earlier PPL/top-1/cosine thresholds and
checkout-default CURRENT_RUN_POLICY are superseded for this campaign.

The active next campaign is ResNet50 plus the 16 cases in
KERNEL_BENCHMARK_HANDOFF_IREE_TVM_VOYAGER_AGENTIC.md. Full-model TinyLlama and
SmolVLA are backlogged; isolated T01–T07 kernels remain active. ResNet DSE has a
4h allowance and each final E2E run has a 30min upper timeout. The requested
candidate plan is 50 warmup trials plus 5 guided trials per workload, or earlier
verified space exhaustion; see the handoff for pending grouping/budget decisions.
Use XGB warmup=50, batch=1 and report actual learned-ranking availability and errors.
Keep old payloads as historical evidence. Automatic prepacking/bias folding stay
removed; stock fusion/CSE and TensorIR schedules remain. Report unequal model
optimization effort; do not present deferred models as equally tuned.
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
may be supplied from estimates. Final E2E wall timeouts only are 1,800 seconds
for ResNet and 25,200 for TinyLlama/SmolVLA per baseline/tuned run. Tuning budgets
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
