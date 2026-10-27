# FireSim active-cycle limit contract

Use the pinned hardware's existing capabilities. Unsupported hardware operations
remain unsupported; do not invent support, alter the image or hide an eligible
device contraction in CPU code. A frontend's unsupported Torch spelling is an
import-coverage issue, separate from device support. Record permitted host
operations and bind E2E claims to the exact source and ELF that ran.

`firesim_job_pool.py` sends `gemmini_firesim_probe_request_v3`. The supplied
bitstream already sets memory; DDR3 is expected. The adapter loads the exact
ELF, preserves boot evidence, compares every output and returns the guest's
complete measured second `model_run` interval, including the final fence.

Active mcycle is the latency objective and includes memory stalls, command
backpressure and completion waits. It is not CPU retirement, Gemmini array
activity or a compute/memory bottleneck classification. Additional diagnostic
counters do not replace or automatically qualify the live-cutoff observer.

For a request with `measured_cycle_limit`, the result must declare:

```json
{
  "measured_limit_enforced": true,
  "cycle_limit_scope": "active_measured",
  "moving_limit_enforced": true,
  "limit_refresh_scope": "in_flight"
}
```

The moving fields are required when `live_limit` is supplied. These declarations
must describe a qualified implementation: observe the measured window during
execution, abort at `active_cycles >= incumbent`, and tighten the bound while
the job is running. Exclude loading, initialization and warmup. Preserve positive
and wrong-oracle controls, a same-ELF unpruned comparison and an in-flight
tightening/abort test. Fields alone do not prove hardware capability.

An image without a host-visible active-cycle observer cannot satisfy this by
using `+max-cycles` on total target cycles, refreshing the incumbent only at
launch, or rejecting a completed slow result. Report those modes honestly as
`total_target`, `at_launch` or `completion_only`; the current DSE pool refuses
them for an incumbent-limited request. Do not launch another paid DSE campaign
under the user's live-cutoff policy until that capability is demonstrated.
Standalone baseline execution remains possible with the existing ELF launcher.

The supplied October 26 F2 adapter used a 200M startup allowance plus four times
the incumbent as a total-cycle cap. All 813 task jobs completed; none hit it.
Its old enforcement booleans overstated the implementation. This does not
invalidate the completed guest intervals, but it did not save the compute
promised by the active-cycle cutoff. Keep those historical receipts unchanged.

## Tuning admission and finishing active runs

The current campaign stops new submissions after four tuning hours for ResNet
and one hour each for TinyLlama and SmolVLA. Already active measurements finish;
queued jobs do not start during completion. Record extra completion time separately.
This policy does not remove incumbent cycle pruning, correctness checks, explicit
cancellation or per-job hang safeguards.

With `--finish-active-runs`, the pool rejects new slot admissions at
`admission_deadline_epoch`. An admitted request's `deadline_epoch` and
`wall_seconds` describe its separate per-job watchdog, supplied explicitly through
`--active-run-timeout-seconds` by the remote worker. Do not shorten that watchdog
back to the admission deadline or treat it as new tuning time. The supervisor
waits for active campaigns at the admission cutoff; it can still stop its own
processes on explicit cancellation or critical memory pressure. `occupancy.json`
records each admitted slot lease and its post-budget duration, including failures.
The remote worker retains the aggregate per-model occupancy ledger across chunks,
graph options and restarts; the local elapsed-deadline launcher does not implement
that aggregate meter by itself.
