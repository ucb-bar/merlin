# AGENT.md — merlin/experiments/voyager_h2h

Status: active — merlin vs the Voyager compiler on three planes; what is verified is in `STATUS.md`.

## Purpose

Head-to-head of merlin against the Voyager compiler (github.com/jeffreyyu0602/voyager-compiler,
DAC 2026 / arXiv 2509.15205). It answers two questions separately and never mixes them:

- **Plane A — same hardware (Gemmini).** Voyager's compiler and merlin, each compiling the same
  PyTorch source for the same Gemmini RTL/bitstream. Voyager's output reaches Gemmini through a
  clean-room bridge that *translates* its tiled IR and never re-optimizes it.
- **Plane B — Voyager's own hardware.** The generated Voyager accelerator RTL
  (code.stanford.edu/voyager/accelerator), where the Voyager compiler is the co-designed expert and
  merlin is a compiler generated for a target it has never seen.
- **Plane C — different hardware, context only.** System-level numbers normalized per MAC and per
  roofline; never presented as a compiler result.

## Hard rules

- **Public sources only.** Voyager evidence comes from the pinned public repositories, the paper, and
  our own runs. Never read or reuse anything under `/scratch/jack`; tell every sub-agent so.
- **Pinned both sides.** Voyager revisions are `voyager_compiler` / `voyager_accelerator` in
  `merlin/contract/hardware_pins.yaml`; every cell records `merlin.common.provenance.record()` for
  both, plus the merlin commit.
- **Stock first.** The stock Voyager arm runs unmodified. A cell that needs a machine-model
  concession is a separate, labelled arm, and the concession is written in the register below with
  the evidence that forced it. A native compile failure is that arm's result, not something to repair.
- **Comparable counters only.** Analytic estimates, Spike "cycles" (instret + 4) and Voyager's SystemC
  model are never ranked against timed RTL/FPGA numbers. Voyager's public RTL regression simulates one
  layer and `min(L2 tiles, MAX_TILES)` tiles at a time and scales the rest; reproduce with all tiles
  before any whole-model comparison.
- **Correctness before speed.** Every cell passes the frozen contract's fp32-logit/top-1 gate, the
  stack's own reference, and (for accuracy claims) a full validation set. No performance credit for an
  incorrect program.

## Concession register (Plane A)

| # | Voyager construct | Gemmini fact | Arm treatment | Evidence |
|---|---|---|---|---|
| C1 | Fused tail `linear -> dequantize(bf16 scale, bf16 qmap) -> relu` writes **bf16** (vector unit) | No bf16 datapath; accumulator readout is int8 (scaled) or int32 | Stock arm: int32 readout + tail on the host. Concession arm: `dequantize -> relu -> next quantize` folded into the accumulator scale where it is exact, otherwise host | Probe compile of int8 linear+relu at the pinned compiler (see `STATUS.md`) |
| C2 | A K split is combined by a second commit `linear -> dequantize -> aten::add` into the first partial, in **bf16** | Partial sums accumulate in the int32 accumulator (ACC_ACCUM readout bit) | Both arms: the bridge keeps the output tile resident in the accumulator across K splits and accumulates in int32 -- exact, so it can only differ from Voyager's bf16 sum in Gemmini's favour numerically; the tile order and every load stay Voyager's | 256x512x256 probe: the only one whose lowered output differs from the quantized reference (max abs 0.0156) |
| C3 | Voyager's model has per-PE L1 input / weight / accumulation buffers (`*_buffer_size` elements per array edge) | No L1 input or weight buffer: A streams from the scratchpad each compute, and the PE array holds exactly one DIM x DIM weight block; the accumulator is the L1 accumulation buffer | Derived, not a choice: `weight_buffer_size = array rows` (one resident block), `input_buffer_size = scratchpad rows`, `accum_buffer_size = accumulator rows` (`merlin.baselines.voyager.accelerator_config_for`). A mapping that needs more resident weights (a 3x3 conv keeps 9 blocks) has no Gemmini equivalent and the stock arm reports Voyager's own refusal | Voyager `hardware_config.py` / `tiler.py` capacity model vs the target's derived address space |

**Packing rule (not a concession, a control).** Voyager specifies no target instructions, so every
packing choice the bridge makes is the bridge's, not Voyager's. The bridge therefore packs each op
exactly as the reference package does (same config words, a `CONFIG_LD` before every `MVIN`, the same
fences), so the two arms differ only in the schedule: which blocks move when, where they sit, and the
order weights become resident and rows stream. A first build that re-issued `CONFIG_LD` only on an
operand change retired 13-19% fewer instructions on multi-tile capsules for reasons that were not
Voyager's; it was discarded before any cycle-accurate run was cited.

Voyager's numerics are bf16-based (the dequantized value is rounded through a 65,536-entry bf16
table before the next quantize), so no Gemmini execution is bit-identical to Voyager's own reference.
Gate the Voyager arm against its reference with the contract's float tolerance, and say so.

## Layout (planned)

- `scripts/` — the Voyager driver run inside `out/build/voyager-venv`, the bridge invocation, and the
  per-plane matrix runners. Reusable pieces (the IR reader, the bridge) live in the library
  (`merlin/python/merlin/baselines/`), not here.
- `STATUS.md` — what is verified now, with artifact paths.
- `LESSONS.md` — what merlin should adopt from Voyager (restated target-agnostically, mapped to the
  plan's workstreams) and what it should not.

## Provenance

Runs under `out/runs/<target>/voyager-h2h/`; tables under `out/artifacts/compare/voyager/` via
`new_product(..., target=...)`. Checkouts: `out/build/external/voyager-{compiler,accelerator}` (env
`MERLIN_EXT_VOYAGER_COMPILER` / `MERLIN_EXT_VOYAGER_ACCELERATOR`). Plan of record:
`merlin/contract/perf_reference_targets.yaml` holds the paper's numbers as `external_claim` entries.
