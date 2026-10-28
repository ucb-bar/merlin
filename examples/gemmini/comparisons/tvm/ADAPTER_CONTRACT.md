# FireSim measurement and wall-watchdog contract

The user approved a per-workload median wall watchdog **in place of incumbent
active-cycle cancellation**. A live active-cycle observer is no longer required
for this campaign. Completed correct slow runs remain useful MetaSchedule data.
This policy does not change the measured cycle window or numerical contract.

## Measurement and timeout

`tune_firesim_chunks.py` uses three actual successful baseline jobs per workload.
Their median cycle count is the baseline cost; their median occupied wall duration
is `m0`. Subsequent watchdogs use:

```text
timeout = min(hard_ceiling, max(3*m0, 3*m))
```

`m` is the median of the latest nine successful, uncached candidate wall durations;
until nine exist, use `m0`. Freeze the timeout at physical-slot admission. Histories
are workload-specific and bound to program, fixture, graph plan, compiler,
toolchain and pool identities. A changed context/policy cannot reuse history.
Bootstrap and hard ceiling are explicit remote-pilot settings, never inferred
from final E2E limits. Refuse `hard_ceiling < 3*m0` instead of clipping the baseline.

Wall duration includes startup, loading, warmup, execution, collection and cleanup;
it excludes build/queue time. The pool records occupied duration with a monotonic
clock. Failed/cached jobs are not successful-duration samples. All admitted jobs,
including failures and the three baseline jobs, consume the model's tuning budget.
A high timeout rate requires review; it is not evidence of search convergence.

The cost label remains the complete second, warm `model_run`, including host
work, DMA and final fence. Exclude startup/initialization/warmup from that cycle
window. Store raw cycles, then convert by the documented target counter frequency
for TVM `run_secs`. Never use wall duration, a timeout threshold or failure sentinel
as measured target cycles. Stock TVM's failed-result handling remains explicit;
it is not a censoring-aware cost model.

## Adapter request and completion

The pool sends `gemmini_firesim_probe_request_v4`, an exact ELF identity, bound
hardware identity, slot, fixed-warm/all-output requirements, `wall_seconds` and
an absolute execution deadline. `incumbent_cycle_pruning` is false. No incumbent
or live-limit file is sent. The installed adapter must understand v4; old v3
observer receipts do not qualify it.

For success, return matching ELF/hardware identities, `guest_exit: 0`,
`status: passed`, `wall_limit_enforced: true`, `remote_stopped: true`, and hashed
boot evidence/transcript. The transcript must have one positive
`DSE_CYCLES first=... second=...` pair and `DSE_PASS`, which is emitted only after
all guest checks. Preserve the wrong-oracle ELF for independent runner qualification;
there is no sequential Spike preflight per tuning candidate.

The watchdog must stop the actual remote job. Killing only SSH/local transport is
insufficient. On timeout, the adapter must retain matching cancellation evidence
and `remote_stopped: true`. If termination is unconfirmed, the pool writes
`SLOT_LOCK.quarantine` and refuses reuse. The remote operator may clear quarantine
only after independently confirming that physical-slot execution has stopped.
This is a conservative recovery state, not a mechanism for killing unrelated jobs.

## Required remote qualification

Pool configuration points to `runner_qualification` using path/SHA256. That proof
uses schema `gemmini_firesim_runner_qualification_v1`, `timeout_scope:
wall_execution`, `cycle_scope: fixed_warm_model_run`, matching adapter source
hashes/deployment ID and hashed passing controls named:

- `startup_exclusion`: startup/warmup do not become the warm cost label.
- `same_elf_repeat_agreement`: repeated correct execution uses the same window;
  report actual cycle variability rather than assuming determinism.
- `wrong_oracle_rejection`: the guest rejects deliberately wrong expected output.
- `wall_timeout`: a bounded attempt is stopped by the configured wall deadline.
- `remote_cancellation`: execution has actually stopped, and the slot is safe to reuse.

Local unit tests establish client behavior only. The remote agent binds the actual
image/driver/manager/loader/memory map and produces those controls. Resolve the
selected U250 versus reported F2 identity; this contract does not authorize an
image substitution or require a new live-observer bitstream.

## Budgets and final runs

Only ResNet is active from the model roster (4h DSE); TinyLlama/SmolVLA
are backlogged (retained future allowance 1h each). The 16-case TVM kernel-study safety ceiling is
24 occupied-slot-hours total (planning estimate, not a runtime prediction). Charge time across chunks/graph
plans. The remote aggregate occupancy ledger is authoritative; the local shared
absolute deadline is an additional admission bound, not a substitute for that
ledger. Stop new admissions at budget, finish active jobs under their frozen
watchdogs, and report completion overrun. Do not renew a continuation's allowance.
CPU preparation and final E2E are excluded from tuning occupancy.

Final baseline/tuned E2E wall timeouts are separate: ResNet 1800s and TinyLlama/
SmolVLA 25200s per run. Do not use those values as candidate bootstrap/ceiling values.

## Prior evidence

The October 26 F2 adapter used 200M startup cycles plus four times the incumbent
as a total-cycle cap; all 813 jobs completed. That did not demonstrate in-flight
active-window cancellation. The old delivery recorded `live_observer: NOT_QUALIFIED`;
the local CSR observer was Verilator-only. Preserve those receipts unchanged.
They explain the superseded policy's gap, not a requirement to implement that
observer under the newly approved wall-watchdog policy.
