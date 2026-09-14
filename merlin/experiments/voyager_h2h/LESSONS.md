# What merlin should learn from Voyager — and how to take it in merlin's own way

Voyager is one compiler co-designed with one accelerator template. Merlin's thesis is the opposite
(one framework, any target, facts derived from RTL, verified up to silicon). So nothing is copied
as-is: every lesson below is restated as a target-agnostic mechanism that reads the target's derived
facts, and proven by execution or measurement rather than asserted. Evidence is cited from the pinned
compiler (`voyager_compiler`, f9d4c498) and from this experiment's own runs.

Status legend: **measured** (seen in our runs), **read** (in Voyager's code), **gap** (merlin lacks it).

## Lessons worth adopting

### L1. Load a block only when its index changes; rotate N slots behind semaphores — measured

- **Voyager.** `bufferize/pipeline.py` (`_BufferedRef`, `PipelinedKernel`, `AsyncPipelinedKernel`)
  turns a grid plus per-operand block specs into one rolled loop. A copy is guarded so a block moves
  only when its tile index changes, and slots rotate behind counting semaphores. Our replay shows it
  exactly: in 128x256x64, 4 input loads against 16 weight loads.
- **What it bought on Gemmini.** With identical packing, the bridge retires fewer instructions than
  the certified reference package: C0 215 vs 224, GM1 6,290 vs 6,538 (Spike instret). The reason is
  that Voyager loads the resident input once, where the reference reloads it per (k, n) block. The L3
  cycle verdict is pending in STATUS.md.
- **Merlin today.** Each backend hand-writes its loop nest (`_matmul_trace` K→N→M). There is no
  double-buffer transform, and `plan_arena` is analysis-only.
- **Merlin's way.** Add a target-agnostic *block-spec pipeline* pass over the interface IR: grid,
  per-operand index maps, load-on-change, and slot count. Slot count and capacity come from the derived
  address space (`targetgen/address_space.py`), never from a literal. Verify it with an abstract-schedule
  executor (below) before RTL. This is the G2 workstream, and it absorbs the missing double-buffer
  transform.

### L2. Make every lowered stage executable, so it is its own oracle — read; merlin extends it

- **Voyager.** The bufferized graph stays runnable in PyTorch (`__init__.py`).
- **Voyager's weakness.** Its CI checks CNNs, BERT and ViT *before* tiling, and never gates on the
  result (5% tolerance, warn-only).
- **Merlin's way.** Give each new lowering an executable abstract form. `baselines/voyager_schedule.execute()`
  is the first instance: it proves a schedule's integer arithmetic exactly, in milliseconds, before
  Spike. Generalize it into a per-target *schedule executor* whose semantics come from the command-buffer
  ABI. Gate on it bit-exactly for integer paths, so the L0/L1 tiers cover the lowered schedule and not
  only the command buffer.

### L3. Key perf calibration by content, so one measurement transfers across models — read

- **Voyager.** `codegen/reporting/calibration.py` and commit e613cbe key every kernel by a
  content-addressed build key. Measured cycles-per-iteration from a two-layer compile then transfer to
  the full model (28/28 keys matched on AMD-135m).
- **Merlin today.** Merlin measures instead of predicting, and its analytic models have been falsified
  (memories `gemmini-cost-model-falsified`, `cheap-perf-signals-measured-against-l3`).
- **Merlin's way.** Reuse the digest discipline we already have for certificates. Key a GSIM or
  Verilator measurement by the digest of the lowered kernel it timed, and build the calibration corpus
  from those. Every whole-model estimate then states which keys it covers and which it extrapolates.
  Publish the held-out error, which Voyager does not.

### L4. Choose fusion by bytes kept on chip, within a stage budget the hardware defines — read

- **Voyager.** `operator_fusion.py` matches chains to the vector pipeline's stages. It resolves
  conflicts by maximizing the DRAM bytes kept on chip, exhaustively for up to 6 chains and greedily
  above that. A tail-stage budget decides what rides the GEMM drain.
- **Merlin today.** Fusion is upstream `linalg-fuse-elementwise` plus requant and epilogue fusion. The
  root cause of the ResNet-50 cascade is an epilogue the store path cannot hold (memory
  `per-channel-scale-is-the-root-of-the-cascade`).
- **Merlin's way.** Treat fusion as a budgeted cover problem. Derive the stage budget from the target's
  store-path capabilities (e.g. the CONFIG_ST layout fields in `rocc/decode.isa_constants`: activation,
  scale, pool) and score covers by bytes saved. This is G1's device-epilogue work, in general form.

### L5. Express layout changes as ops that keep reference semantics — read

- **Voyager.** `ops/README.md` "twin" ops carry a layout or scratch contract. The graph stays
  executable, and cancelling transposes fold away.
- **Merlin today.** A layout solver exists (`perf/physical_layout.py`) with no caller; the NHWC flip
  lives as an agent patch.
- **Merlin's way.** Add layout-carrying xDSL ops with a reference interpretation, so a layout pass is
  checked by execution the way L2 checks schedules. That makes wiring the solver safe.

### L6. Treat quantization as a compiler layer with one spec grammar and table-driven emulation — read; gap

- **Voyager.** One spec string covers dtype, scheme, block size, axes, scale dtype and outlier rate
  (`quantization/qspec.py`). Qconfig tables apply per module, op or call. Lookup-table rounding
  (`qmap`) makes any format emulatable exactly. On top sit codebooks (LCQ), MX-aware GPTQ, outlier CSR
  and KIVI 2-bit KV.
- **Merlin today.** A format registry exists (`merlin/schemas/quant_formats.registry.yaml`,
  `common/quant_formats.py`); the quantization itself is torchAO upstream.
- **Merlin's way.** Put the spec grammar onto the registry: one token names a registry entry plus its
  scheme. Emulate every format by table, and let the capability manifest decide which formats EXECUTE
  on a target; the rest are accuracy-only. That is the G3 full-parity work, with a dataset-accuracy
  harness (G3b).

### L7. LLM-specific lowering — read; gap

- **Voyager.** A split KV cache with a full-precision residual window (`quant_folding.py`
  `split_kv_cache`), GQA repeat folding, flash attention as explicit hardware passes with online
  softmax (`bufferize/attention*.py`), and prefill/decode/speculative-verify export.
- **Merlin's way.** Adopt them as interface-level rewrites whose legality comes from derived capacity.
  Do causal masking, which Voyager's flash path does not. This is G5.

### L8. Report what the paper shows, but from measurements — read

- **Voyager.** Exact steady-state loop folding keeps estimates for long contexts fast (commit
  90c6a57). It produces a Perfetto trace per unit, a per-module compute/overlap/memory/stall
  breakdown, and DRAM bytes split into weight/activation/KV.
- **Merlin's way.** Emit the same schema from measured counters: FireSim `rdma_bytes_rec` and
  `wdma_bytes_sent`, and `perf/hw_counters.py` occupancy. Keep loop folding for the analytic screen
  only. Add ablation switches per lever (`--no-pipelined`, `--no-fuse` style) so a Table-2-shaped
  ablation is a flag, not a branch. This is G6.

### L9. Take the op-coverage denominator from a public corpus — read

- **Voyager.** `tools/gen_aten_classifier.py` harvests ATen ops from 250 HF/timm models (222
  exported, 144 unique ops).
- **Merlin today.** Merlin measures 51/188 core ops (memory `aten-core-coverage-claim`).
- **Merlin's way.** Use a public harvest as the coverage requirement, the same derived-denominator
  discipline as the capsule corpus.

## Lessons to NOT adopt (their weaknesses are our differentiators)

| Voyager practice | Why merlin keeps its own |
|---|---|
| Hardware constants measured on one SoC, baked in the tiler (`BANK_SWITCH_CYCLES=8`, `SPMM_ROW_CYCLES=8`, "Sphinx") | Merlin derives every value from the target's facts or refuses (`check_no_assumed_constants`) |
| Numerics tolerance of 5%, warn-only, checked before tiling for CNN/BERT/ViT | Bit-exact integer gates at L0-L3 on the lowered program, with mutation controls |
| CI gates on a text diff of its own previous `model.txt` | Independent goldens; a tier that did not run is `incomplete`, never `pass` |
| Whole-model RTL cycles as per-layer, tile-extrapolated sums (`run_regression.py` `MAX_TILES`) | One timed invocation on RTL, FPGA or silicon, with provenance |
| Latest compiler no longer targets the public hardware release (legacy `param.proto` retired) | Pinned revisions per result; a target is a descriptor, not a fork |

## Mapping to the plan

| Lesson | Workstream |
|---|---|
| L1 | G2 (tile search and pipeline pass) |
| L2 | verification layer (extends L0/L1 to schedules) |
| L3 | G2 (calibrated cost model) |
| L4 | G1 (device epilogue, fusion) |
| L5 | G1 (layout solver wiring) |
| L6 | G3 / G3b |
| L7 | G5 |
| L8 | G6 |
| L9 | G4 |

L1 goes first. It is the one lesson already measured on our own hardware, and the reference backend
can adopt it without any new infrastructure.
