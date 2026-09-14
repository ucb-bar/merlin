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

## Open

- Bridge: Voyager IR (JSON) -> Gemmini command stream, graded at L2/L3 on the capsule corpus.
- Plane B feasibility gate: Catapult 2023.1_1 vs the tested 2024.2; licence; VCS V-2023.12.
- Whether using the shared FireSim queue service is acceptable (ask before the first submission).
