# TVM Gemmini comparison

This directory holds the Merlin side of the TVM–Gemmini baseline: frozen
inputs, complete-unit ELF builders, the FireSim study runner and the evidence
rules. The compiler itself is the TVM fork
[`grammy117/tvm` `gemmini/direct-intrinsics`](https://github.com/grammy117/tvm/tree/gemmini/direct-intrinsics).

**Status (2026-11-04):** the ResNet50 + 16-kernel campaign ran on AWS F2 FireSim
(60 MHz DDR3+LLC image) on 2026-10-28. Results, setup and the recipe are in
[REPRODUCING_F2_RESULTS.md](REPRODUCING_F2_RESULTS.md). Full TinyLlama and
SmolVLA, and the selected U250 30 MHz target, were not run.

## Start here

| Guide | For |
| --- | --- |
| [USAGE.md](USAGE.md) | Build TVM and the adapter, build untuned/tuned ELFs, check on Spike, tune on FireSim |
| [IMPLEMENTATION.md](IMPLEMENTATION.md) | What the compiler and runner do, in one page |
| [REPRODUCING_F2_RESULTS.md](REPRODUCING_F2_RESULTS.md) | Experimental setup, exact pins, step-by-step recipe, result tables and how to read them |

## Reference

| Document | Purpose |
| --- | --- |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Design record: stock vs custom layers, scheduling, measurement contract, limits on paper claims |
| [TENSORIR_OP_AUDIT.md](TENSORIR_OP_AUDIT.md) | Operation inventory: primitive/composite mappings, spec families and fixture boundaries |
| [OPTIMIZATION_KNOBS.md](OPTIMIZATION_KNOBS.md) | What is searched, fixed or removed |
| [ADAPTER_CONTRACT.md](ADAPTER_CONTRACT.md) | Remote runner contract: watchdog, cycle scope, cancellation qualification |
| [FIRESIM_HANDOFF.md](FIRESIM_HANDOFF.md) | The launch specification the F2 campaign followed (budgets, defaults, build commands) |
| [KERNEL_BENCHMARK_HANDOFF_IREE_TVM_VOYAGER_AGENTIC.md](KERNEL_BENCHMARK_HANDOFF_IREE_TVM_VOYAGER_AGENTIC.md) | Cross-compiler definition of the 16 kernel cases |
| [SPIKE_HANDOFF_EVIDENCE.md](SPIKE_HANDOFF_EVIDENCE.md) | Pre-campaign Spike functional checks (74 operators, 16 kernels, 18 chunks, ResNet parent) |
| [OWNERSHIP.md](OWNERSHIP.md) | Bringup by Nicolas Reed (PR #53, `gemmini/bringup`) and what the curated branches kept |
| [NUMERICAL_DIAGNOSTICS.md](NUMERICAL_DIAGNOSTICS.md) | First-divergence tools |
| [byoc/](byoc/README.md) | The original BYOC bringup path and its validation |

## History

These records are kept unchanged for provenance and are not instructions:
[README_HISTORICAL.md](README_HISTORICAL.md) (the long pre-campaign setup and
run reference), [UPSTREAM_AUDIT.md](UPSTREAM_AUDIT.md),
[PERFORMANCE_AUDIT.md](PERFORMANCE_AUDIT.md), [EL4_INPUT_AUDIT.md](EL4_INPUT_AUDIT.md),
[EL4_ASSETS.md](EL4_ASSETS.md), [FASED_SWEEPS.md](FASED_SWEEPS.md) (deprecated
local Verilator sweeps), [HOST_SETUP.md](HOST_SETUP.md) (bringup pointer, kept
byte-identical) and [FIRESIM_ROUND2_HANDOFF.md](FIRESIM_ROUND2_HANDOFF.md)
(linked from frozen handoff releases).
Superseded launch pages (branch, chunk-tuning, remote-tuning, tuning-plan and
generation guides) were removed on 2026-11-04; their last text is in
[Git history](https://github.com/ucb-bar/merlin/tree/5998ce4ef/examples/gemmini/comparisons/tvm).
Older measurements keep their original source, input and hardware identities.
