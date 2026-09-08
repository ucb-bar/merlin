# Phase-2 native scalar epilogue ResNet-50 hardware candidate

This package is the hardware-promoted first substantial target-neutral Phase-2 win on top of the q530
hardware-qualified schedule. The host lane now emits standard LLVM `roundeven` and `fptosi`
operations instead of rebuilding both operations from IEEE bit fields for every quantized tensor
element. This is a general CPU-lowering repair: it contains no model, layer, shape, or Gemmini
special case. The accelerator command buffer and all DMA/scheduling decisions are unchanged.

An exact one-warm/one-measured local Spike A/B reduced the primary proxy from 976,190,118 to
751,827,143 cycles: 224,362,975 cycles saved, a 22.98% reduction / 1.2984x speedup. All 1,000
ResNet-50 logits remain bit-exact (`bad=0`, `nonfinite=0`, top-1 258, checksum
`c6e777c3fe0aae90`). The measured kernel has no progress instrumentation; eight sparse task
markers exist only in the untimed warm entry, followed by an untimed mutable-arena reset.

FireSim queue job 534 completed successfully. It used only
`/scratch/firesim_queue/bin/firesim-queue`; the daemon owned the exact lifecycle:

1. `firesim kill`
2. `firesim infrasetup`
3. `firesim runworkload`
4. `firesim kill`

The accepted hardware result is **1,537,416,019 measured compute cycles**, all 1,000 logits exact.
This saves 166,648,204 cycles versus q530: a 9.7795% cycle reduction / 1.1084x speedup. The total
4,518,740,312 simulator cycles include warmup and harness work and are not the benchmark metric.
`validation/firesim_queue_job_534_success.json` binds the package, staged and executed ELF,
bitstream, driver, HWDB, lifecycle logs and UART by SHA-256.

Pinned build: `/scratch2/agustin/chipyard` commit
`009e85b05ca817a15fcd6c3fe4eedd86cc2a3617`. ELF SHA-256:
`0b3284a815da3bd6f8b8e0a014f936b0ffc5c5a2c6666916ced45f6c716f62c5`.

Run `./verify_bundle.sh` to verify the sealed result. `./run_firesim.sh` now fails closed because
the unchanged artifact must not be resubmitted. This is a hardware-promoted compiler checkpoint,
but Phase 2 is not complete: native
`LOOP_CONV_WS`, fused narrow epilogues, residual lowering, and dynamic-activation quantized
contractions remain separate, guarded Phase-2 levers.
