# Remote TinyLlama preparation seed

Follow the [current FireSim runner handoff](../../../../../../examples/gemmini/comparisons/tvm/FIRESIM_ROUND2_HANDOFF.md) for exact authority files, restore/setup commands and parallel jobs.

Verify `SHA256SUMS`, extract `tinyllama-seed.tar.gz` to a fresh `unpacked` directory, then verify its own `SHA256SUMS`. All member identities and the exact public checkpoint URL/hash are inside `seed.json`. The checkpoint itself is not included.

This supplies calibration, token inputs, range statistics and pinned preparation source. **Final quantized exports, independent expected outputs and 16/1 chunks remain remote jobs.** Tuning uses 16/1; E2E uses 32/32. **SmolVLA is paused and excluded.** Ready ResNet jobs can run concurrently.
