# AGENT.md — merlin/python/merlin/perf/layer_bench

## Purpose

Per-layer RTL evidence: the performance loop's acceptance rung. Each measurement builds one small
program for one fused group at its real shape and runs it on a cycle-accurate engine (GSIM first).
The cycles and the output digest come back from the same run. The receipt is keyed so it is reused
only while nothing that could change the number has changed.

## Modules

- `key.py` — `LayerKey`, which carries the design pin, engine sha, group signature, contract digest,
  schedule digest, emitter digest, harness version and protocol, plus its digest.
- `console.py` — structural parsing of the program's `LB_RECORD` lines and of the engine's
  `FINISHED:` line. A malformed record raises; it is never skipped.
- `cache.py` — the content-addressed receipt store. Receipts are self-hashed and verified on read.
- `build.py` — a two-phase reproducible build through the target's own `harness_build_recipe`, plus
  the loaded-bytes guard (sum of PT_LOAD `p_memsz`, i.e. what the loader moves).
- `run.py` — runs on the target's pinned GSIM command (revalidated before and after) under a hard
  wall-clock deadline, and parses the records and the engine completion line.
- `reference.py` — the fill and digest protocol (the LCG and FNV constants a renderer must emit from
  here) and the exact off-device expected output: the same fill stream, exact integer conv/matmul,
  the declared contract's readout, and the same digest. A run whose digest differs computed a
  different function.

## Invariants

- **Image size costs cycles unless the target's load backdoor is used.** Measured on GSIM (E0,
  `out/artifacts/perf-bench/gemmini/layer_bench_e0_20260914/`): the default loader moves the whole
  image, including the `.bss` zero-fill, over TSI at under ~0.47 B/cycle, so a 4 MB static array
  delays `main` by more than 9M cycles. A target that declares `gsim_backdoor_env()` loads by memcpy
  into the DRAM backing store instead. `run_on_gsim` applies it by default and records `load_path`,
  which measured cycle-identical within 0.06% on conv_7. `build_program` keeps its loaded-bytes guard
  unless the caller disables it for a backdoor run.
- **A receipt that fails verification is an error, not a miss.** Re-measuring silently would hide a
  corrupted or mismatched store.
- **Target-neutral.** The build recipe and engine command come from the target's registered backend.
