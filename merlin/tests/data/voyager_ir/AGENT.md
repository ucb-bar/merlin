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

- `host_ops_golden/` — Voyager's OWN outputs for the ops the bridge leaves on the host (quantize,
  dequantize, max_pool2d on a load-padded tile, adaptive_avg_pool2d, the bf16 classifier linear), on
  seeded inputs chosen to hit rounding ties, saturation and integers past bf16 precision. Recorded in
  Voyager's environment by `merlin/experiments/voyager_h2h/scripts/voyager_host_ops_golden.py`; the
  manifest names the exact call behind every entry. These ARE answer keys for
  `voyager_schedule.host_op_reference`, but of Voyager's public op library, not of any capsule.

Whole-block fixtures (`voyager_export.py` workload kind `resblock`; batch-norm randomized, then
folded as Voyager's harness does). Each adds `scales.json`, the per-tensor scale values the program
references (from Voyager's `dump_tensors` output), so `lower_model` can build the readout:

- `resblock_basic_14x14x32x64/` — BasicBlock 32 -> 64, stride 2: two quantized convs, then the
  downsample conv carrying the residual epilogue into an unquantized output. `data.npz` holds
  Voyager's own int8 input and parameters and the output its bufferized graph computed (bf16 as
  float32): the independent oracle.
- `resblock_splitk_8x8x512x64/` — BasicBlock 512 -> 64, stride 2: conv1 is a K split (one-slot
  partial, final part into the output with relu + quantize).
- `resblock_bottleneck_14x14x1024x256/` — Bottleneck 1024 -> 256, stride 2: the downsample is a K
  split whose last part writes the partial, followed by a STANDALONE `dequantize`-anchored residual
  add (the construct ResNet-50's layer4_0 uses).

Each `manifest.json` records the compiler commit, the accelerator config (16x16 PE, 256 KiB / 4-bank
scratchpad, 16 B bank width), the workload and the quantization flags. Regenerate with
`merlin/experiments/voyager_h2h/scripts/voyager_export.py` in `out/build/voyager-venv`; the JSON is
minified, sorted-key `json_format.MessageToDict` output.

## Invariants

- Output of the pinned `voyager_compiler` revision only (`merlin/contract/hardware_pins.yaml`); a
  regenerated fixture from another revision is a different fixture, so update the manifest with it.
- Public-source provenance only: produced from the public compiler by our own export step.
