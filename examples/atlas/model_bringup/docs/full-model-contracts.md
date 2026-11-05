# Full-model host contracts

Full ResNet uses `atlas.full_resnet_bundle.v3`; Tiny and Smol use `atlas.full_bundle.v1`. Bundles bind the ELF, build, capture, programs, weights, inputs, and expected outputs by checksum. Preparation receipts are not numerical passes.

The three full-host builders accept `--arena-base`, `--dram-base`, `--dram-bytes`, and `--target firesim`. For VCU118 use `--arena-base 0x90000000 --dram-base 0x80000000 --dram-bytes 0x80000000`; Tiny also needs `--rodata-after-arena`. Fit includes load segments, BSS, stack, and arena. FireSim software packaging can instead admit 16 GiB with `--dram-bytes 0x400000000`; a package does not supply the target configuration or establish hardware execution.

Full hosts report `ATLAS_KERNEL_BEGIN`, `ATLAS_KERNEL_PROGRESS`, and a final `ATLAS_KERNEL_PASS` or `ATLAS_KERNEL_FAIL`, alongside mailbox/tohost. FAIL names the failed check. Native QEMU uses NS16550 at `0x10000000`. An admitted SiFive board profile at `0x10020000` requires explicit `--sifive-uart-bridge`; the bridge checks real guest MMIO and initialization, with no board or baud-timing qualification.

Numerical runs require pinned Model `9e6313bddecb8ba936206eb347f5cd4e5335282c`, backend `curve`, and the recorded curve hash. `run.py` checks actual staged inputs, programs, outputs, guards, DMA, drained completion, and final host state. A changed ELF needs its own execution evidence.

Segment bundles use `atlas.segment_chain.v1`. Each link restores its complete workload-admitted semantic state into a fresh guest/device, executes real host code, and compares every live-out byte to the next captured boundary. `run_tiny_segment.py` and `run_smol_segment.py` produce exact per-kernel ledgers and boundary receipts; `SegmentChain.verify_receipts` requires every ordered link before summing measured cycles. This is segmented qualification, not continuous execution or hardware timing. Native reference controls do not replace device execution.

Tiny full and segment entries share arithmetic but produce different ELFs; identify both explicitly. Smol's default selector executes the full policy, while an admitted phase request selects a restored segment. Reproducing the current Smol host consumes the authenticated archived boundary reference described in [Smol host inputs](smol-full-host.md).

## Result output

Full hosts use `runtime/atlas_result.h` and the generated UART sink. The default board profile is SiFive at `0x10020000`; native QEMU uses `--uart-kind ns16550 --uart-base 0x10000000`. The image must match the target UART. Baud configuration is preserved unless `--uart-divisor` is explicit.

BEGIN, PROGRESS, PASS, and FAIL travel through actual UART MMIO. PROGRESS counts completed and expected launches; FAIL includes the check, observed value, and expected value, and remains sticky. Names and check labels must be nonempty whitespace-free strings. The runner captures `uart.log`; mailbox/tohost completion remains available separately.

## Tiny reference boundaries

`tools/tiny_full/capture_boundaries.py` runs the production C orchestration with checksum-bound captured replies. Supply `--capture PATH --capture-sha256 HASH --gate PATH --gate-sha256 HASH --merlin COMPARISONS_DIR --cc PHYSICAL_CC --out FRESH_DIR`. The input must cover 128-token prefill and two greedy decode steps. It checks all 465 projection predecessors/replies, 132 KV arrays, three last-row logits, and greedy feedback in both full and phased paths.

Preparation writes 70 ordered boundaries with 145 semantic regions, content-addressed payloads, `SHA256SUMS`, and `native-gate.json`. Boundaries have no ELF addresses. The optional `tools/tiny_full/build_host.py --segment-boundaries PREPARED_DIR --segment-sha256 SHA256SUMS_HASH --segment-model-root MODEL` profile derives the writable inventory from a separate segment-capable ELF and binds its entry, stop sites, programs, and native chain.

`tools/host_iss/run_tiny_segment.py --bundle-root BUNDLE --model-root MODEL --qemu QEMU --segment-index INDEX --wall-seconds CAP --out FRESH_DIR` restores semantic bytes into a fresh guest, seeds the global invocation sequence, executes the selected layer/head, and compares every next-boundary region. The index is `23*step+layer`, where layer 22 is the LM head. `--corrupt-region hidden_0|hidden_1 --corrupt-offset BYTE` selects a negative control. `run-state.json` records progress; the final receipt includes peak RSS.

Native preparation checks captured replies with host C. Numerical qualification requires real guest execution and exact drained device receipts for every ordered link; see [host execution](host-iss.md). UART checks establish functional output, while board wiring, baud timing, and platform timing require target evidence.
