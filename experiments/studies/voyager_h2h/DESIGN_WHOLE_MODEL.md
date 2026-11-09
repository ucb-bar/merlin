# Plane A, whole model: running Voyager's compiled program on Gemmini + Rocket

Status: design, 2026-09-14. Written before any whole-model number exists, so the translation rules
below are fixed first and the result is whatever they produce.

## The program to run

Stock Voyager (f9d4c498) with its own `--conv2d_im2col`, torchvision ResNet-50 with conv/BN fused as
Voyager's harness does, the derived Gemmini machine model: `whole_model/resnet50_bnfused_im2col_*`.
Replayed (`baselines/voyager_ir.replay`): 8,132 events, 1,560 fused computes (1,398 `conv2d`, 98
`linear`, 64 dequantize-anchored), 3,400 loads, 1,307 stores, and host-side tensor ops
(`max_pool2d` x28, `quantize` x15, `adaptive_avg_pool2d`, the FC `aten::linear`) that Voyager runs
on its vector unit in bf16.

## Translation, not re-compilation

The IR is a ROLLED program (loops, conditionals, index arithmetic). It is translated to C one
construct at a time -- `for_loop` -> `for`, `cond` -> `if`, scalar ops -> C expressions -- so the
Voyager arm keeps Voyager's control flow and code size. Straight-line unrolling from the replayed
trace would change instruction-cache behaviour and is not used for the timed program; the replay is
the oracle the translation is checked against (same event sequence).

Every tensor operation becomes a call into a Gemmini runtime written against the public Gemmini
programming interface (`gemmini.h` from the pinned `gemmini_isa_headers`), the same interface the other
Gemmini compiler arms use -- no encoding is written by hand:

| Voyager construct | Runtime call | Gemmini realization |
|---|---|---|
| `async_copy` DRAM->scratchpad (2-D / NHWC tile, `pad`) | `vg_load` | `config_ld` + `mvin` per DIM block, at the rows Voyager's slot address selects; halo padding by `mvin` from a zero page |
| `async_copy` scratchpad->DRAM | `vg_store` | `mvout` from the accumulator region that holds the tile |
| `commit` { GEMM } | `vg_gemm` | the schedule `baselines/voyager_schedule` already produces (Voyager's interstellar nest) |
| `commit` { conv2d } | `vg_conv` | implicit GEMM: per (fy, fx, ic, oc) weight block, `preload` once, stream output-row segments of input pixels in Voyager's loop order |
| `async_wait` | none | Gemmini orders by address dependence; the host fences before it READS anything the array wrote |
| host tensor op | `vg_host_*` | plain C on Rocket, bf16 emulated in float with bf16 rounding |

## Translations the target forces (concessions, extend AGENT.md C1-C3)

- **C4: stride-s convolution.** A compute reads DIM consecutive scratchpad rows; a stride-2 tap's
  pixels are every other column. The input tile is loaded *phase-split* (one scratchpad plane per
  column phase, via the `mvin` DRAM stride), so each tap streams contiguous rows. Same bytes, more
  `mvin` instructions than Voyager's single tile copy.
- **C5: fused epilogues on the store path.** `dequantize -> [relu] -> quantize` becomes the
  accumulator store scale and activation (`config_st`); the residual chain
  `dequantize(res), dequantize(acc), add, relu, quantize` becomes a SCALED `mvin` of the int8 residual
  into the accumulator before the store. A chain that ends in bf16 (the stem, the pool inputs) is
  stored as int32 and dequantized on the host. bias -> stride-0 broadcast `mvin` into the
  accumulator. All of this is fp32 math where Voyager's is bf16 (C1): gated by tolerance, never
  bit-exactness against Voyager's own reference.
- **C6: the vector unit.** Voyager's bf16 vector-unit ops run on the Rocket host. That work exists
  only because Gemmini has no vector unit; it is timed, and reported as its own row so the reader can
  see how much of the Voyager arm's time is host emulation of a unit Gemmini lacks. Voyager's own
  `--quantize_fc` (FC as int8 on the array) is a separate, labelled variant.

## Gates

1. Translation fidelity: the C program's event sequence (loads, computes, stores in order, with
   resolved slots) equals `replay()`'s, on a host build with the Gemmini calls logged.
2. Arithmetic: per-layer outputs against the stack's own reference within the contract tolerance; the
   whole model against the frozen contract (top-1 and fp32-logit correlation), with REAL weights
   (IMAGENET1K_V2) and Voyager's own ImageNet calibration (HF token approved).
3. Spike before FireSim; FireSim only through our own queue, only when the FPGA is idle.
