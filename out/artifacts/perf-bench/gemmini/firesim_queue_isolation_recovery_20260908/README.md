# FireSim queue isolation recovery

This package snapshots the queue implementation used for Universal Gemmini job
533 and its focused private-simulation-directory regression. The live queue is
installed at `/scratch/firesim_queue`; this copy preserves the exact source for
the `feat/target-generalization` experiment record.

Two independent state leaks were diagnosed:

1. Direct jobs inherited Jack's shared `default_simulation_dir`, leaving stale
   simulator state across users. The queue now rewrites that field to
   `/scratch/firesim_queue/jobs/<id>/simulation`; five queue tests pass.
2. A daemon-writable source checkout bypasses the queue's deploy overlay.
   Queue jobs therefore use the repository's tracked read-only view builder and
   cwd-preserving launchers. This retains Jack's exact driver authority but
   makes workloads, logs, topology, driver bundles, and results job-private.

Evidence:

- job 531, shared checkout/shared simulation directory: timed out;
- job 532, shared checkout/private simulation directory: repeated the
  infrasetup host-up stall and was cancelled after 540 seconds;
- job 533, read-only view/private deploy/private simulation directory:
  completed the entire queue lifecycle in 360 seconds and executed the exact
  ELF. Its model payload failed numerically, which is a compiler issue recorded
  separately—not a queue failure.

The queue still reports lifecycle `DONE` when FireSim exits normally even if a
bare-metal program reports `tohost=1`. Always grade the resulting UART with
`merlin/experiments/gemmini_perf_bench/scripts/verify_firesim_payload.py` before
accepting a performance number.
