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
  scratchpad is under ~1.77 MB. Voyager's own remedy (`--conv2d_im2col`) and the one-block L1 variant
  are re-running with the fused harness.
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

## Open

- Bridge: Voyager IR (JSON) -> Gemmini command stream, graded at L2/L3 on the capsule corpus.
- Plane B feasibility gate: Catapult 2023.1_1 vs the tested 2024.2; licence; VCS V-2023.12.
- Whether using the shared FireSim queue service is acceptable (ask before the first submission).
