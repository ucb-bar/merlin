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
  that Voyager loads the resident input once, where the reference reloads it per (k, n) block. At L3
  the reference backend with only that change (`gemmini_xdsl_rtl_v1_l1`, one guard) takes C0/C1 from
  1,547 to 1,108 cycles (-28%), 5 cycles under Voyager's own schedule (1,113).
- **What not to copy.** The slot ROTATION. On a deep K (GM0/GM1) Voyager's two slots make each load
  wait for the compute still reading the slot it overwrites; merlin's one-region-per-block placement is
  1.38-1.42x faster there. Rotate only when capacity forces it; the derived capacity says when.
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

### L10. Order loads inside a K step so the streamed block arrives first; run one step ahead — measured

- **Voyager.** Its double-buffered pipeline issues the next K step's copies before the current
  step's compute, and within a step it copies the streamed input tile before the weight tile. Replayed
  on Gemmini (fence-corrected bridge) this is its whole short-capsule lead over merlin v1_l1: A3/B0-B2
  by 20 cycles (~5.5%), C2-C4 by 2.
- **What we measured.** Not hoisting: issuing every load first (`v1_hoist`) moved A3 by one cycle and
  made C0 7% slower (1,108 -> 1,186), because the loads crowd the queues ahead of the first compute.
  The order is what matters: input block before weight block, one K step ahead (`v1_la1`) takes A3 to
  359, B0 to 326 and C2 to 459, below Voyager's 365, 330 and 469.
- **Over all 18 capsules** (STATUS.md table): `v1_la1` is 4.4% faster than Voyager's schedule
  (geomean) with 11 wins, but loses the 7 single-block capsules by 1-2 cycles. Placing the weights at a
  bank boundary (`v1_la1b`) ties those and wins 8, but loses C2-C4 by 3 cycles. So operand PLACEMENT is
  a second measured knob next to order: weights sharing the input's bank cost 3-15%, and the position
  inside the other bank moves a capsule by a few cycles.
- **On the FPGA the same knob flips.** With real DRAM (FireSim, same ELFs) C2 reverses: merlin's
  lookahead order wins in Verilator (459 vs 469) and loses on the FPGA (968 vs 861), while grouping
  every input load ahead of the weights -- Voyager's own order for that shape -- matches Voyager
  there (862) yet is the worst arm on C0 (2147 vs 1876). The knob's best value depends on the
  memory system, not only on the array, so it must be measured on the substrate being claimed.
- **Merlin's way.** Make lookahead depth, intra-step operand order and bank placement knobs of the
  L1 block-spec pipeline pass, chosen per target by measurement (cycles keyed by kernel digest, lesson L3), not
  fixed in a backend. The mechanism is target-agnostic; the best order is a measured fact of the
  target.

### L11. Check the memory plan's address invariant directly — measured

- **Voyager.** Its planner checks that no two live buffers overlap (`memory_planning.py:682-744`),
  though it only warns and exempts buffers in the same bank group.
- **What it bought.** In the mutation study it is the one Voyager check that sees an address fault.
  Its eager executor gives every buffer its own tensor, so an overlapping plan changes no output at
  all. It caught the cross-bank-group overlap that merlin's exact check missed: merlin's lowering
  keeps partials and outputs in the accumulator, so that overlap never reaches a planned address it
  executes.
- **Merlin's way.** Put a static address/liveness checker beside the exact executor: scratchpad and
  accumulator row intervals per slot, live ranges from the op stream, and a hard failure (not a
  warning) on any write into a live region. The executor proves the arithmetic; the checker proves
  no live data is overwritten, including hazards an in-order executor cannot see (the study's 8
  concurrency hazards were invisible to both sides).

## Generalization: can Voyager's compiler target merlin's other accelerators?

Checked the only way that does not presume the answer: ask merlin's target-agnostic derivation
(`baselines.voyager.accelerator_config_for`, built on `targetgen.address_space`) to describe each
target in Voyager's machine-model terms, from that target's own RTL facts (2026-09-14).

| Target | What its facts say | Voyager `AcceleratorConfig` |
|---|---|---|
| Gemmini (systolic, RoCC) | 16x16 mesh, 256 KiB 4-bank scratchpad, 64 KiB accumulator | derived. It still took a ~1.2k-line bridge, six concessions (C1-C6), and Voyager's own `--conv2d_im2col` before whole-model ResNet-50 compiled at all |
| Atlas NPU (own ISA, matrix registers) | 32x32 mesh; one 1.5 MB 6-bank vector memory whose role (scratchpad vs register file) is not classified; operands reach the MXU through matrix registers driven by an instruction stream | refused: no store that maps to Voyager's L2 scratchpad. Even with one assigned by hand, Voyager emits parameters for ITS deserializer and has no instruction selection, so Atlas needs a new backend, not a bridge |
| Radiance (SIMT GPU) | 16 lanes/warp, 128 KiB shared memory, no systolic array | refused: no array edge to derive. Voyager's tiler pins an IC x OC spatial partition at level 0 (weight-stationary array); warps, divergence, register blocking and shared-memory banking have no representation |

Even inside the systolic class, the machine model misses what a different array needs. Voyager's
mapper has no notion of a target that streams CONTIGUOUS scratchpad rows into its array: its own
hardware addresses a window per pixel, so it keeps innermost OX extents of 4 (3x3 stride 1) or 2
(stride 2) for ResNet-style convs. Replayed faithfully on Gemmini those are 4- and 2-row computes on a
16-row array (28,224 per layer; `lower_conv`, STATUS.md). Merlin's derivation should carry the
target's stream-contiguity requirement into any tiler it feeds.

So Voyager generalizes across its own template's DESIGN SPACE (array size, buffers, datatypes -- its
DSE), not across ARCHITECTURE CLASSES. That is by design, and it is the axis on which merlin is
distinct: the same derivation that fed Voyager a Gemmini machine also serves merlin's own backends for
all three targets. Co-design also drifts on its own clock: Voyager's latest compiler no longer emits
the IR its own public hardware release consumes (see STATUS.md).

## How Voyager connects the CPU to the accelerator, and what it takes to express one

From the public release (accelerator `e3a725db`, paired compiler `cac504ef`) and the latest compiler
(`f9d4c498`); citations relative to the checkouts under `out/build/external/`.

- **The hardware has no ISA, registers or interrupts. It has per-unit parameter queues.** Each unit
  (matrix, matrix-vector, SpMM, depthwise, vector) has one `params_in` channel of 64-bit words
  (`src/Accelerator.h`). The host bit-packs a `MatrixParams` / `VectorParams` /
  `VectorInstructionConfig` struct in its `Marshall` order, pads it to 64 bits and streams it; the
  unit's `ParamsDeserializer` rebuilds the struct (`src/Params.h`, `src/ParamsDeserializer.h`). The
  structs ARE the instruction set: loop bounds (two levels of six loops), base addresses, dtype
  indices, fusion selectors, and a micro-program of up to 8 vector instructions over four fixed stages.
- **Units are their own bus masters.** Every operand stream has an `(address, burst)` request port;
  there is no separate DMA engine. Completion is a per-unit `start`/`done` sync handshake -- no
  interrupt, no status register, no global done.
- **In the public release the "CPU" is the SystemC testbench.** It parses `model.txt` and
  `tilings.txtpb`, maps each operation to descriptors in hand-written C++
  (`test/toolchain/MapOperation.cc`, `MatrixOps.h`), streams them, waits on `done`, and in `SOC_SIM`
  mode even copies tiles DRAM<->scratchpad itself (`test/common/DataLoader.cc`). The paper's Chipyard
  integration (MMIO control registers, TileLink ports, an interrupt controller) is not in the release.
- **The latest compiler moves sequencing into the IR.** `voyager_ir.proto` adds loops, conditionals,
  `AsyncOp` regions with semaphores, and marks index arithmetic `op: "cpu"`, "run on the control
  processor rather than the accelerator datapath" (`codegen/transform/bufferize/emit.py`). No public
  executor consumes it; the tiler's "Sphinx SoC" calibration constants suggest a private one.
- **To express an accelerator, Voyager needs:** a template instance (compile-time `-D` knobs: datatype
  preset, IC x OC, buffer depths, which units exist), a matching `AcceleratorConfig`, hand-written
  vector-pipeline fusion patterns, and hand-written C++ mappers that know the `Params.h` layout. There
  is no target description, capability declaration or instruction selector; adding an op touches the
  compiler, the mapper, the gold model and possibly the datapath RTL.

What merlin does that this lacks: a declared endpoint kind per target (RoCC `.insn`, MMIO, command
buffer) in its target contract, a documented command-buffer ABI, and facts/capabilities DERIVED from
the RTL instead of `-D` knobs and hand-written mappers.

What merlin should take from it:
- **H1.** An IR that marks, per operation, which work is control-processor and which is datapath, with
  explicit async regions and semaphores. merlin's command buffer is a flat list; the host lane and the
  accelerator lane are separated by placement, not by a first-class async construct.
- **H2.** Descriptor layouts whose bit widths are computed from the struct (`TypeToBits`), never
  written down. merlin already derives register-bundle layouts from Scala; use them to generate
  command encoders and decoders from one source.
- **H3.** A gold model per op living next to the mapper (`test/common/GoldModel.cc`), so every new
  mapping is checked op by op before whole-model runs.

## Lessons to NOT adopt (their weaknesses are our differentiators)

| Voyager practice | Why merlin keeps its own |
|---|---|
| Hardware constants measured on one SoC, baked in the tiler (`BANK_SWITCH_CYCLES=8`, `SPMM_ROW_CYCLES=8`, "Sphinx") | Merlin derives every value from the target's facts or refuses (`check_no_assumed_constants`) |
| Numerics checked with an elementwise `assert_close(rtol=5e-2, atol=1e-4)`, warn-only, and for CNN/BERT/ViT/MobileBERT before `compile()` (tiling, bufferization) | Bit-exact integer gates on the lowered program, with mutation controls. Measured (mutation study, STATUS.md): Voyager's pre-tiling check sees 0/28 seeded schedule faults; the same criterion after bufferization warns on all 28 but also on 3 of 4 CORRECT programs (bf16 reassociation); merlin's exact check flags 24/28, refuses 4, and flags no correct program |
| A local pre-push diff against its own previous `model.txt` | Independent goldens; a tier that did not run is `incomplete`, never `pass`. Measured: the diff fails all 9 legal rewrites and passes all 8 scale faults (`model.txt` names constants, not their values) |
| Whole-model RTL cycles as per-layer, tile-extrapolated sums (`run_regression.py` `MAX_TILES`) | One timed invocation on RTL, FPGA or silicon, with provenance |
| Latest compiler no longer targets the public hardware release (legacy `param.proto` retired) | Pinned revisions per result; a target is a descriptor, not a fork |
| The public release ships a configuration 2x off its own paper: one FIFO is depth 1 for every non-MX datatype, halving matrix throughput (STATUS.md, plane B) | Every claim is regenerated from the artifact under test, never inherited from a paper. Reproduce first, and when the reproduction misses, find the line before comparing anything |

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
| L10 | G2 (lookahead depth and operand order as measured knobs of the pipeline pass) |
| L11 | verification layer (static address/liveness checker beside the exact executor) |

L1 goes first. It is the one lesson already measured on our own hardware, and the reference backend
can adopt it without any new infrastructure.
