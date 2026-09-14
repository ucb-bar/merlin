# AGENT.md — merlin/tests/data/voyager_ir

## Purpose

Real Voyager compiler output (the JSON form of its `voyager.Model` protobuf) that pins the behaviour
of `merlin.baselines.voyager_ir.replay`. They are inputs, not answer keys: tests assert structural
facts of the schedule (grid, slots, guarded loads, split-K chains), never a numeric result.

## What is here

- `lin64/` — int8 linear+relu, M=K=N=64: a 4-step output-column grid, double-buffered weight/bias
  slots, the input loaded once before the loop.
- `split_k_256x512x256/` — int8 linear, M=256 K=512 N=256: a `[2,4,2]` grid (N outer, M, K), two K
  splits whose second commit adds into the first in bf16, loads guarded by index change.

Each `manifest.json` records the compiler commit, the accelerator config (16x16 PE, 256 KiB / 4-bank
scratchpad, 16 B bank width), the workload and the quantization flags. Regenerate with
`merlin/experiments/voyager_h2h/scripts/voyager_export.py` in `out/build/voyager-venv`; the JSON is
minified, sorted-key `json_format.MessageToDict` output.

## Invariants

- Output of the pinned `voyager_compiler` revision only (`merlin/contract/hardware_pins.yaml`); a
  regenerated fixture from another revision is a different fixture, so update the manifest with it.
- Public-source provenance only: produced from the public compiler by our own export step.
