# voyager_h2h — status

What is verified right now. Each line names its evidence; nothing here is a performance claim yet.

## 2026-09-14

- **Pins.** `voyager_compiler` = f9d4c498 and `voyager_accelerator` = e3a725db
  (`open-source-release`) in `merlin/contract/hardware_pins.yaml`; `provenance.verify()` is clean for
  both, with checkouts under `out/build/external/`.
- **Voyager environment.** `out/build/voyager-venv` installs the pinned compiler with the versions it
  pins itself (torch 2.12.1, torchao 0.17.0, transformers 5.13.0).
- **Stock compile works for int8 linear+relu** (64x64x64, 16x16 PE, 256 KiB / 4-bank and 2 MiB /
  16-bank scratchpads, accelerator-repo INT8_32 flags). The first attempt failed inside Voyager's
  pipeline export (`all carried_inputs must be tensors for stack_output`) only because parameters
  required grad; Voyager's own harnesses compile under `torch.no_grad()`, and so does
  `scripts/voyager_export.py`.
- **What the IR looks like for one linear.** One rolled `while_loop` over 4 output-column tiles,
  double-buffered scratchpad slots (`bank_count` 2), `async_copy` DRAM<->scratchpad with counting
  semaphores, and a `commit` region holding the fused chain `quantized_ops::linear` (int8 x int8 +
  int32 bias -> int32) -> `dequantize` (bf16 scale, 65,536-entry bf16 qmap) -> `aten::relu`, written to
  a bf16 scratchpad tile and copied out. This is the source of concession C1 in `AGENT.md`.
- **Voyager's RTL cycle counts are extrapolated.** The accelerator repo's `run_regression.py` runs one
  layer at a time and only `min(L2 tiles, MAX_TILES)` tiles (default 1), then weights runtime by
  `full_tiles * count / actual_tiles`. Recorded as a caveat on every paper entry in
  `perf_reference_targets.yaml`.

- **Schedule replay.** `merlin.baselines.voyager_ir.replay` evaluates Voyager's scalar control flow
  and yields the ordered load/store/wait/compute trace with every operand resolved to slot and byte
  address; Voyager's own counting-semaphore discipline balances to zero on all four probe exports
  (64^3; 16x64x64; 128x256x64; 256x512x256). Observed: resident operands are reloaded only when their
  tile index changes (128x256x64: 4 input loads vs 16 weight loads), and a K split is combined by a
  second commit whose tail is `dequantize -> aten::add` in bf16 (256x512x256) -- the one probe whose
  lowered output differs from the quantized reference (max |d| = 0.0156). This is concession C2:
  on Gemmini the partial sums combine in the int32 accumulator instead.

- **First Voyager-scheduled kernel runs on Gemmini (L2).** `scripts/build_bridge_package.py` built a
  package whose `build_trace` replays the pinned compiler's schedule for A2_single_tile_matmul
  (2 block loads, 1 preload, 1 compute, 1 store; lowered arithmetic self-checked with numpy). Through
  `merlin.targetgen.oot_runner --simulator spike`: status pass -- contract, all four entrypoints, the
  command-buffer semantic check, and the spike+libgemmini functional oracle. Control: the certified
  reference package passes the same capsule the same way. Spike's `cycles` (53 vs 52) is instret + 4,
  not timing, and is not a comparison.

- **All 18 lowerable profile GEMM capsules pass L2 on both arms.** `voyager_bridge_v0` lowers 12
  unique Voyager schedules covering 18 of the 21 profile matmul capsules; A7 (20x24x12, partial
  blocks) and GP0/GP1 (maxpool-on-store) are refused with their reasons. `scripts/capsule_h2h.py
  --simulator spike`: reference 18/18 pass, voyager 18/18 pass. Spike counts are instret, not timing.
- **Packing control.** A first build packed loads with fewer `CONFIG_LD`s than the reference package
  (13-19% fewer retired instructions on multi-tile capsules, none of it Voyager's doing); rebuilt to
  pack exactly as the reference does before any cycle-accurate run. What remains is schedule: C0 215 vs
  224 and GM1 6,290 vs 6,538 retired instructions, because Voyager loads the resident input once where
  the reference reloads it per (k, n) block.

- **Plane B feasibility (tools): GO.** Catapult 2022.1_1 / 2023.1_1 are installed with `lmutil`; the
  BWRC license servers are up and issue `CatapultHLS_c`, `CatapultUltra_c` and `msimsystemc` (100
  each, 0 in use at probe time). Synopsys VCS V-2023.12 / W-2024.09 are installed with
  `SNPSLMD_LICENSE_FILE` set. Skew to record, not to hide: upstream tests Catapult 2024.2_2 and VCS
  T-2022.06-SP2; Catapult 2023.1's bundled g++ is 10.3.0 (C++17 as the Makefile needs).
- **Voyager's current compiler does not target its own public hardware release.** The accelerator's
  `open-source-release` consumes the legacy `param.proto` IR through its compiler submodule
  `cac504ef` (2026-01-05). That path was retired at `a884ff3` (2026-07-27); the pinned compiler
  (`f9d4c498`) emits only `voyager_ir.proto` (since `f2cd538`, 2026-07-12). Plane B therefore runs the
  co-designed pair from the release (compiler `cac504ef` + accelerator `e3a725db`), checked out at
  `out/build/external/voyager-compiler-cac504ef`; plane A keeps the latest compiler. Both revisions
  are named in every cell.

- **How plane B must reproduce the paper's numbers (from the release's own flow).** `codegen.mk` runs
  the paired compiler's `test_codegen.py` to `model.txt`; `test/compiler/run_tiler.py` turns that into
  `tilings.txtpb` and **imports interstellar** (the SSH-only submodule). The public copy vendored by the
  latest compiler exposes every name run_tiler uses (`le`, `Layer`, `Resource`, `Schedule`,
  `optimizer`, `cost_model`, `extract_input`, `utils`, `loop_*` mapping fields), so it is the
  substitute -- disclosed in every cell, and checked at run time. `run_regression.py --uniquify_layers`
  keeps one instance of each identical layer and multiplies its runtime by the count; RTL runs only
  `min(L2 tiles, MAX_TILES)` tiles of a layer. A citable reproduction therefore sets `MAX_TILES` to at
  least the largest L2 tile count, and still reports a per-layer sum, not one timed invocation.
  The flow writes `build/`, `test/compiler/networks/` and `regression_results/` into its tree, so it runs
  in a separate work tree (`out/build/external/voyager-accelerator-work`), never the pinned checkout.

- **Plane B environment built.** `out/build/voyager-accel-env` (mamba, Python 3.10, torch 2.6.0,
  libprotobuf/protoc 29.3) with the paired compiler `cac504ef` installed editable. Python protobuf
  resolves to 7.36.1 through pip dependencies; the legacy `param_pb2` imports under it. The
  interstellar substitute resolves through `out/build/external/voyager-interstellar-shim` (only that
  package on the path, so the latest compiler cannot shadow the paired one); `le.IC/OC/OX` resolve.
- **Plane B data dependency.** The release's CNN codegen calibrates on `timm/imagenet-1k-wds`
  (Hugging Face, gated) through `load_dataset(..., streaming=True)`; the accelerator ships no data
  (`data/` holds a README only). Reproducing the paper's INT8 calibration needs a Hugging Face token
  with access to that dataset -- or a disclosed substitute calibration set, which would change the
  scales and must then be reported as a deviation.

- **The release's model assets are LFS objects, and they are anonymously pullable.** `models/**`
  (e.g. `mobilebert-tiny-sst2-bf16/config.json`, a 33 MB `model.safetensors`) are Git LFS pointers in
  a plain clone, so codegen fails with "config file ... is not a valid JSON file" until they are
  fetched. With `git-lfs` 3.8.0 added to the plane B env, `git lfs pull --include` fetched them from
  code.stanford.edu without credentials. MobileBERT-tiny SST-2 (a Table 4 row) is therefore the first
  plane B smoke target: its calibration set (GLUE SST-2) is public, unlike the CNNs' gated ImageNet.

- **Stock Voyager cannot compile whole-model ResNet-50 for Gemmini's 256 KiB scratchpad (valid
  harness).** `whole_model/resnet50_bnfused_stock_20260914T201303Z`: public compiler f9d4c498,
  torchvision ResNet-50 with conv/BN fused exactly as Voyager's own harness does, default derived
  config. The tile search succeeds; Voyager's own memory planner then refuses:
  `scratchpad plan needs 1772624 bytes > scratchpad_size 262144`. The four live buffers are the 7x7
  stem conv, placed on chip UNTILED -- input `1x1x229x229x3` int8 (157,328 B), weight `1x7x7x3x64`
  int8 (9,408 B), bias `1x64` int32 (256 B), output `1x1x112x112x64` bf16 (1,605,632 B). This does
  not depend on our L1 reading: it is the planner's L2 budget, so it holds for any accelerator whose
  scratchpad is under ~1.77 MB. Under the one-block L1 sensitivity reading
  (`whole_model/resnet50_bnfused_peblock_20260914T202255Z`, same fused harness) it fails earlier, in
  the tile search: `layer1_0_conv2: no tiling fits on chip` (a 3x3 conv keeps nine weight blocks
  resident; that reading allows one). With Voyager's own `--conv2d_im2col` it compiles (below).
- **SUPERSEDED -- harness defect, do not cite.** The ResNet-50 runs below were exported WITHOUT the
  conv/batch-norm fusion Voyager's own torchvision harness performs before export
  (`get_conv_bn_layers` + `fuse_modules`), so Voyager compiled a graph its flow would not have given
  it (convs with no bias, BN as separate ops). Found when Voyager's `replace_conv2d_with_im2col` hit a
  `None` bias; `voyager_export.py` now fuses as the harness does, and the runs are being redone as
  `whole_model/resnet50_bnfused_*`. Kept below as the record of what was run.
- ~~Stock Voyager does not compile whole-model ResNet-50 for Gemmini's 256 KiB scratchpad -- under
  either reading of its L1.~~ Public compiler f9d4c498, torchvision ResNet-50 (random init: this is a
  compile-feasibility question), config from `accelerator_config_for`:
  - `weight_residency="pe_block"`: the tile search refuses at `layer1_0_conv2` ("no tiling fits on
    chip" -- a 3x3 conv keeps 9 weight blocks resident, the reading allows 1).
  - default (`scratchpad`): the tile search succeeds, then Voyager's own memory planner refuses: the
    7x7 stem conv is placed on chip UNTILED -- its whole padded 229x229x3 int8 input (157,328 B) and
    whole 112x112x64 bf16 output (1,605,632 B) -- 1,772,624 B against 262,144 B. That fits Voyager's
    2 MiB CI scratchpads and nothing under ~1.77 MB, whatever the accelerator.
  Variants in flight, each a Voyager knob rather than ours: `double_buffered_l2=False`, and Voyager's
  own `--conv2d_im2col` (its CI's remedy for small-channel convs). Runs under
  `out/runs/gemmini/voyager-h2h/whole_model/`.

- **Plane B toolchain substitutions (version skew, disclosed; Voyager's source untouched).** Building
  the release's SystemC harness with Catapult 2023.1's bundled gcc 10.3 needs two include-path fixes,
  both passed through the Makefile's own `BASE_FLAGS`:
  1. `-isystem /usr/include/x86_64-linux-gnu`: the bundled gcc does not search the host's multiarch
     directory, so `<asm/errno.h>` is not found (`fatal error: asm/errno.h`).
  2. ONE missing header, appended after Catapult's own (`-idirafter`): `src/datatypes/StdFloatTypes.h`
     includes `ac_math/ac_gelu_pwl.h`, which Catapult 2023.1 does not ship (upstream tests 2024.2).
     Only that file is taken, from `hlslibs/ac_math` `1fde1dd` (Apache-2.0); Catapult 2023.1's own
     `ac_types` 4.6 stays in use. Putting the whole public `ac_types` HEAD first was tried and fails:
     Voyager's exponent-only scale type `UFloat<8,8>` (`src/datatypes/ScaleTypes.h`) instantiates
     `ac_int<0>` through `ac_std_float::to_float`, which that `ac_types` rejects (`log2_ceil<0>` has no
     `val`). Tagged `ac_types` 4.8.0 / 4.9.0 are staged if a later step needs a newer 4.x.
  3. `-B/usr/lib/x86_64-linux-gnu` in `LDFLAGS`: the bundled gcc links without the host's multiarch
     C runtime start files (`cannot find crt1.o / crti.o`).
  Every plane B result produced this way is labelled with these three deviations.

- **Plane B functional flow runs end to end on this host.** Voyager's own
  `run_regression.py --sims fast-systemc --uniquify_layers` on `mobilebert_encoder`, INT8, 16x16
  (paired compiler `cac504ef`, accelerator `e3a725db`, the three disclosed toolchain deviations, plus
  the variables upstream's `env.sh` sets: `PROJECT_ROOT`, `CODEGEN_DIR`, and a loader path that puts
  the env's libstdc++ before Catapult's): **14/14 non-skipped layers pass against Voyager's gold model
  with error count 0.** The one failure, `slice_tensor` ("Slice indices for the last dimension must be
  multiples of OC_DIMENSION!"), is a layer Voyager's own `ci_skip_rules.json` skips for this model in
  every sim type. SystemC is functional only (upstream says so); cycles come from the RTL flow next.
- **With its own `--conv2d_im2col`, stock Voyager compiles whole-model ResNet-50 for Gemmini's derived
  machine model** (`whole_model/resnet50_bnfused_im2col_20260914T201309Z`, 58 layers, 3.3 MB IR). So
  the Voyager arm has a legitimate whole-model program; lowering it onto Gemmini needs the bridge to
  grow convolution, bias and the vector-unit ops (still open).

- **L3 capsule head-to-head, clean room (Verilator, cycle-accurate).** Products under
  `out/artifacts/compare/gemmini/v1/`; every run passes its RTL oracle.
  - **Bridge correction first.** `voyager_bridge_v0` closed each replayed matmul with a FENCE the
    reference package does not emit: one cycle per capsule of OUR packing, not Voyager's schedule.
    `voyager_bridge_v1` drops it (23a53da1) and is faster on all 16 short capsules
    (`..._20260914T211028Z_2a7a601`). Every v0 Voyager number is superseded; merlin's apparent
    one-cycle wins over v0 were that fence.
  - Cycles, reference `gemmini_xdsl_rtl_v0` / Voyager `voyager_bridge_v1` / merlin `v1_l1` (lesson L1):

    | capsules | reference | Voyager | merlin v1_l1 |
    |---|---|---|---|
    | A0 A2 A5 C5 C6 GS0 | 302 | 302 | 302 |
    | A4 | 269 | 269 | 269 |
    | A3 B1 | 385 | 365 | 385 |
    | B0 B2 | 350 | 330 | 350 |
    | C0 C1 | 1547 | 1113 | 1108 |
    | C2 C3 C4 | 471 | 469 | 471 |
    | GM0 (K=6144) | 19649 | 27971 (v0) | 19649 |
    | GM1 (K=8208) | 27791 | 38277 (v0) | 27791 |

  - **Short capsules: Voyager's schedule leads merlin v1_l1 by 1.4% (geomean of 16).** merlin wins
    C0/C1 by 5 cycles, ties 7, loses A3/B0-B2 by 20 and C2-C4 by 2.
  - **Deep K: merlin is 1.42x (GM0) and 1.38x (GM1) faster** (`..._20260914T211317Z_2a7a601`,
    3600 s oracle wall). Voyager rotates two slots through 384/513 K steps, so each load waits for the
    compute still reading the slot it overwrites; merlin gives every block its own scratchpad rows.
    Spike instret ranked these the other way (Voyager fewer instructions) -- instret is not timing.
    The fence-corrected arm confirms it: GM0 27,236 and GM1 38,223 (full table below).
  - **What Voyager's short-capsule edge is.** Hoisting every load ahead of the computes
    (`gemmini_xdsl_rtl_v1_hoist`) is not it: A3 385 -> 384, and C0 1108 -> 1186 (worse). The order
    inside a K step is: the streamed input block before the weight block. `v1_la1` (loads one K step
    ahead of the computes, input then weight) takes A3 to 359, B0 to 326 and C2 to 459, below
    Voyager's 365, 330 and 469; `v1_grp` (all inputs, then all weights) 363, 328 and 469. The full
    18-capsule result is below.
  - The remaining A0-class cycle: with the same load order, the weight block's row decides 302 vs
    303 (Voyager places it at 4096, the start of bank 1; merlin at 16368, the end of bank 3).

- **Full 18-capsule L3 table: the fence-corrected Voyager arm vs merlin's schedule variants**
  (Verilator, cycle-accurate; every run passes the RTL oracle). All variants are the certified
  reference package with one scheduling change each (packages under
  `out/artifacts/targets/gemmini/gemmini_xdsl_rtl_v1_*`):

  | capsules | reference | Voyager | v1_l1 | v1_la1 | v1_la1b |
  |---|---|---|---|---|---|
  | A0 A2 A5 C5 C6 GS0 | 302 | 302 | 302 | 303 | 302 |
  | A4 | 269 | 269 | 269 | 271 | 269 |
  | A3 B1 | 385 | 365 | 385 | 359 | 356 |
  | B0 B2 | 350 | 330 | 350 | 326 | 323 |
  | C0 C1 | 1547 | 1113 | 1108 | 1099 | 1097 |
  | C2 C3 C4 | 471 | 469 | 471 | 459 | 472 |
  | GM0 | 19649 | 27236 | 19649 | 19541 | 19420 |
  | GM1 | 27791 | 38223 | 27791 | 26860 | 26860 |

  - Geomean vs Voyager over all 18 (below 1 = merlin faster): `v1_l1` 0.977 (4 wins, 7 ties, 7
    losses); `v1_la1` 0.956 (11 wins; 7 losses of 1-2 cycles, all single-block capsules); `v1_la1b`
    0.957 (8 wins, 7 ties; 3 losses, C2-C4 by 3 cycles). The better of la1/la1b per capsule gives
    0.952 with 11 wins, 7 ties, 0 losses, but that choice is tuned on these capsules: cite it as
    tuned, never as a policy.
  - Deep K is merlin's in every variant: 1.40-1.42x fewer cycles than Voyager's schedule.
  - Placement findings: weights sharing the input's bank cost 3-15% (`v1_la1_pafter`: A0 311,
    A3 374, C0 1271, C2 519); banks 1, 2 and 3 are identical (`v1_la1_pbank2/3` = `v1_la1b`); the
    position inside the bank matters (bank end: C2 459, bank start: 472). No fixed rule tried wins
    both: offsetting the weights past the inputs' in-bank span (`v1_la1_pb1off`) matches the bank end
    on C2 (459) but gives A0 303, A4 271 and C0 1,129 (worse than both). Placement is therefore a
    per-shape knob to choose by measurement, which is why the best-of result above stays "tuned".
- **Quantization-accuracy parity (G3b), first milestone** (`merlin/experiments/dataset_accuracy/`,
  commit 1fddadde). Voyager's own quantizer and recipe, BERT-base on the FULL SST-2 validation split
  (872 sentences), against the paper's Table 3 and the accelerator repo's own gold table:
  FP32 93.23 (paper 93.2), BF16 93.23 (93.0), Posit8 92.89 (92.8), MXINT8 93.12 (93.1) -- four cells
  within +-0.3; E4M3 92.66 (93.1) and INT8 91.28 (92.4) are low. Re-running INT8 with Voyager's other
  calibration recipe moves it to 91.86, so calibration count explains about half; the rest is that
  the paper scored the C++ bit-accurate tester over the LOWERED graph under the paired compiler,
  while this arm scores the quantized graph under f9d4c498.
  - **Protocol finding that changes the parity criterion:** Table 3's ImageNet cells come from
    `run_accuracy` over the FIRST 1000 images of the timm validation stream, not 50k. So +-0.3 is
    +-3 images there and only meaningful on those exact images (a different 1000-draw carries ~1.4
    points of standard error). The accelerator repo also ships a second gold table that disagrees
    with the paper by up to 2.0 points (ResNet-18 INT8 69.5 vs 71.5); cite both.
  - **Voyager's own `--evaluate` path does not run at the pinned revision** without four minimal
    fixes (fp32 images into a bf16 model; an fp32 attention mask for a bf16-traced graph; the GLUE
    loader under huggingface_hub 1.x; a forced 32-thread setting), each recorded in the artifacts.
  - ImageNet is blocked on gated access (the user is requesting it); the harness is a drop-in and was
    smoke-tested end to end. MobileBERT-tiny's SST-2 checkpoint is an unfetched LFS pointer here.
- **Proposed replacement for the paper's Voyager row** (`06_evaluation.tex`, marked PENDING there;
  not edited yet). Old, unreproducible: "L3; 18 exact matched capsules; Voyager uses 3.28% fewer
  cycles (0.967)". Clean-room, same 18 capsules, same Verilator binary, fence-corrected bridge:
  - vs the certified agent-built backend (`gemmini_xdsl_rtl_v0`): Voyager/Merlin geomean **0.986**
    (Voyager uses 1.4% fewer cycles; Merlin wins 2, ties 7, loses 9; Merlin 1.38-1.39x faster on
    the two deep-K GEMMs, 0.946 over the 16 short capsules);
  - with one Voyager-inspired change (`v1_l1`, load each input block once): **1.023** (Merlin uses
    2.3% fewer); with the lookahead order (`v1_la1`): **1.046** (4.6% fewer). These are
    hand-applied lessons on the certified package, not the agent's output, and must be labelled so.
  - Suggested cell: "L3, clean-room bridge; 18 exact matched capsules & Voyager uses 1.4% fewer
    cycles than the certified backend (0.986); Merlin 1.38x faster on deep-K GEMMs; applying
    Voyager's load-once lesson reverses it (Merlin 2.3% fewer)". FireSim column pending.
- **Verification mutation study** (`scripts/mutation_study.py`, e21dfb4f; product
  `compare_gemmini_v1_20260914T222217Z_ada9e07`). 64 faults were seeded into Voyager's own compiled
  programs (4 workloads). Each faulty program was judged by Voyager's checks (its own code at
  f9d4c498) and by merlin's replay -> lower -> exact execution:
  - **Schedule faults (28):** Voyager's CNN/BERT/ViT check runs before `compile()` and sees 0. The
    same elementwise `rtol 5e-2 / atol 1e-4` criterion applied after bufferization warns on all 28,
    but also on 3 of 4 correct programs (split reductions reassociate in bf16), and it never gates.
    merlin flags 24 bit-exactly and refuses 4 (pad value 1, a dropped conv partial); it flags no
    correct program.
  - **Memory-plan faults (7):** Voyager's eager run changes no output element (each buffer is its
    own tensor). Its planner's overlap check (warn-only, exempts one bank group) catches 3; merlin
    catches 4, and misses the cross-group case because its outputs live in the accumulator.
  - **`model.txt` diff** (`run_ci.py`, a local pre-push script): fails all 9 legal rewrites, passes
    all 8 scale faults.
  - **merlin's gaps, to state in the paper:** concurrency hazards (8) are invisible to any in-order
    executor, merlin's included; scale faults (8) are outside its integer-accumulator check by
    construction. `voyager_ir.replay` raised KeyError on a `sym_min` (operands `a`/`b`) instead of
    refusing; fixed below.
- **Conv lowering exists and is exact** (`baselines.voyager_schedule.lower_conv`, abcd2847): the
  3x3 s1, 3x3 s2 and 1x1 probes lower bit-exactly against a direct convolution. Faithful to Voyager's
  mapping, the 3x3 layers stream 4-row (s1) and 2-row (s2) computes -- 28,224 per layer, where 16-row
  runs would need a quarter or an eighth as many. Voyager's mapper has no notion of a target that
  streams contiguous rows; its own hardware addresses a window per pixel.
- **Whole-model ResNet-50 lowers, and has an exact executable reference.**
  - `lower_model` (58edbd65) lowers all 59 layers of the stock im2col export into 52 conv
    schedules, 1 GEMM (the stem), 16 host requantizes, 2 host dequantizes and 7 Voyager host ops;
    nothing is refused. The segmentation is one Voyager top-level loop per layer, and matches
    Voyager's own `layers.txt` 59/59. Residual epilogues (476 convs) and the 64 standalone residual
    adds lower through C5; 16 layers whose tile exceeds the accumulator run in passes (C7).
  - `execute_model` (154bb69b) runs it end to end. Every host op reproduces Voyager's own op library
    bit for bit against a recorded golden (quantize, dequantize, max-pool, average pool, bf16 fc),
    each with a failing control. Against Voyager's own lowered output on 4 inputs: cosine
    0.99984-0.99986, max abs diff 1.26-1.94 of 114, top-1 agrees (random weights, so top-1 barely
    discriminates). Layer by layer on Voyager's dumped inputs: host ops exact; all 51 int8 conv
    outputs within one int8 step (95-98% equal), the C1/C5 readout rounding; no layer shows a larger
    local error.
  - What remains for a timed whole-model Voyager arm: the C translation of the schedule (in
    progress), then Spike, then FireSim.

## Open

- Whole-model plane A: the C translation of lowered schedules (in progress), then Spike, then
  FireSim. Lowering and the exact whole-model reference are done (above).
- FireSim: the existing submit tool, approved as-is by the user (2026-09-14), submitting only while
  the FPGA is idle; the capsule batch (`scripts/firesim_h2h.py`) is in flight.
- Plane B: the release's own flow runs every layer about 2x slower than the paper claims
  (MobileBERT 13.63M cycles at 49.7% vs 7.71M at 95.1%; details in `PLANE_B.md`); the root cause is
  under investigation. ResNet-50 waits on gated ImageNet access, which the user is requesting.
