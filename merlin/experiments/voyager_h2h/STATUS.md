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

## Open

- Bridge: Voyager IR (JSON) -> Gemmini command stream, graded at L2/L3 on the capsule corpus.
- Plane B feasibility gate: Catapult 2023.1_1 vs the tested 2024.2; licence; VCS V-2023.12.
- Whether using the shared FireSim queue service is acceptable (ask before the first submission).
