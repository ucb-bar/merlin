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
- `conv3x3_s1_28x28x64x64/` — int8 conv+bias+relu, 28x28x64 -> 64, 3x3 stride 1: 7-row output tiles
  with a padded 9x30 halo window, four 16-channel K splits through a one-slot partial buffer, the
  last split into a double-buffered output slot stored after the next tile's first split.
- `conv3x3_s2_28x28x64x128/` — the same at stride 2 into 128 channels: the whole 29x29 padded input
  in one window, innermost OX extent 2.
- `conv1x1_28x28x64x256/` — 1x1 conv, 64 -> 256: no K split, 28x4-pixel output tiles, IC inside the
  pixel loops.

The conv fixtures' outputs end in a host `aten::permute` (NHWC -> NCHW).

Each `manifest.json` records the compiler commit, the accelerator config (16x16 PE, 256 KiB / 4-bank
scratchpad, 16 B bank width), the workload and the quantization flags. Regenerate with
`merlin/experiments/voyager_h2h/scripts/voyager_export.py` in `out/build/voyager-venv`; the JSON is
minified, sorted-key `json_format.MessageToDict` output.

## Invariants

- Output of the pinned `voyager_compiler` revision only (`merlin/contract/hardware_pins.yaml`); a
  regenerated fixture from another revision is a different fixture, so update the manifest with it.
- Public-source provenance only: produced from the public compiler by our own export step.
