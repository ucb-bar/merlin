# AGENT.md — merlin/experiments/voyager_h2h

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

Voyager's numerics are bf16-based (the dequantized value is rounded through a 65,536-entry bf16
table before the next quantize), so no Gemmini execution is bit-identical to Voyager's own reference.
Gate the Voyager arm against its reference with the contract's float tolerance, and say so.

## Layout (planned)

- `scripts/` — the Voyager driver run inside `out/build/voyager-venv`, the bridge invocation, and the
  per-plane matrix runners. Reusable pieces (the IR reader, the bridge) live in the library
  (`merlin/python/merlin/baselines/`), not here.
- `STATUS.md` — what is verified now, with artifact paths.

## Provenance

Runs under `out/runs/<target>/voyager-h2h/`; tables under `out/artifacts/compare/voyager/` via
`new_product(..., target=...)`. Checkouts: `out/build/external/voyager-{compiler,accelerator}` (env
`MERLIN_EXT_VOYAGER_COMPILER` / `MERLIN_EXT_VOYAGER_ACCELERATOR`). Plan of record:
`merlin/contract/perf_reference_targets.yaml` holds the paper's numbers as `external_claim` entries.
