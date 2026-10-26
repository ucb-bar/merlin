# Next ResNet and TinyLlama FireSim handoff

**Scope instruction: do not implement the proposed optimization knobs.** They
were research ideas only. This handoff uses the existing TVM/Gemmini contraction
schedules and knobs, graph fusion fixed at 2, meaningful chunks, parallel jobs,
incumbent active-cycle limits and connected/E2E confirmation. No new host
scheduling, producer placement or bounded-packing implementation is requested.

The [published delivery README](../../../../out/artifacts/handoff/gemmini-firesim-dse-publication/v1/handoff_gemmini-firesim-dse-publication_v1_20261026T173908Z_238ada8/README.md)
is the fresh-machine entry point on `comparisons/tvm-workload-tuning`. It includes
the baseline ELF and checksum-verified parts of the complete portable archive.
Run its restore command, then read the extracted `README.md`, `study.json`,
`resnet-checks.json`, `task-owners.json` and `tinyllama-coverage.json`.
Sources are already committed in this branch: skip the archived historical
source-patch/overlay application block. No FPGA execution or new performance
result is claimed. The original preparation remains unchanged at
`out/artifacts/handoff/gemmini-firesim-dse-round2/v1/latest/` locally.

ResNet has **18 actual full-shape chunks**: stem, 16 complete residual blocks,
and classifier/pooling. They retain real activations, constants, full reductions
and live residual edges from one prepared image. All 18 host checks have zero
output difference. There are **21 distinct contraction workloads**, verified
by TIR structural equality. Ten chunks own those workloads; other blocks are
confirmation cases rather than duplicate tuning campaigns. This describes task
coverage, not the fraction of legal schedules searched.

The new source adds all 54 biases in INT32. Bound proofs show that this preserves
the former FP32 result for all admitted INT8 operands; weights and calibration
scales are unchanged. This prepares a new numerical identity. Hardware output
checks and fixed-set accuracy remain separate requirements under
[CURRENT_RUN_POLICY.md](../../numerics/CURRENT_RUN_POLICY.md).

TinyLlama's previous F2 coverage was K/V and a down-projection output panel.
Missing coverage includes Q/O, gate/up, full-output down, attention QK/PV and
context/cache effects in both prefill and decode. Connected attention and MLP
chunks must include their host math, conversions and state edges. Current-policy
calibrated prefill/decode exports are missing locally; the bundle records real
checkpoint dimensions and preparation commands, with these campaigns pending.
Do not silently run the older dynamic-row/per-channel probes instead.

The missing TinyLlama inputs are: frozen-policy prefill/decode exports with
calibrated per-tensor activation and grouped-16 weight scales; actual activation
and cache/state fixtures for the selected chunks; and independent expected
outputs for chunk and complete 32+32 sessions. The model revision and dimensions
are already known. A matched FP32 WikiText-2 test PPL baseline/settings is a
separate missing input to the final quality evaluation, not a prerequisite to
every correctness-checked chunk experiment. Preparing these inputs does not
authorize the proposed compiler optimizations.

`run_firesim_study.py` launches independent native TVM campaigns across a shared
physical-slot pool. TVM owns EvolutionarySearch, XGB, scheduling and database
selection. Initial ResNet campaigns reserve 48 proposals; optional continuation
reserves another 20. TinyLlama reserves 24 proposals once actual inputs are
prepared. The study cap is 96, with **one shared deadline of at most four hours**.
Baselines, qualification, confirmations and E2E also consume that wall budget;
proposal count is not a paid-job count or a duration estimate.

The objective is guest active cycles, including memory stalls and completion.
Abort a candidate when `active_cycles >= current_best` for the **same workload**,
and tighten the limit in flight. Exclude boot/loading/init/warmup; preserve
all-output checks. The October 26 adapter did not implement this: total-cycle
caps and completed-result filtering do not qualify. The supervisor requires
source-bound observer tests described in [ADAPTER_CONTRACT.md](ADAPTER_CONTRACT.md)
before launching paid DSE. Memory is already in the bitstream; DDR3 is expected.

This first roster tunes accelerator contraction schedules. It retains and
measures host work in connected confirmation but does not implement the proposed
CPU loop schedules or panelized packing. The historical 70.64% host-function
latency includes memory/instruction stalls; it is not CPU compute utilization.
Contraction-only gains must therefore be assessed against whole-chunk and E2E
cycles. [OPTIMIZATION_KNOBS.md](OPTIMIZATION_KNOBS.md) retains research context
only; its proposed additions must not be implemented in this handoff.
