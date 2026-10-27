# TinyLlama preparation seed

Use the [current FireSim runner handoff](../../../../../../examples/gemmini/comparisons/tvm/FIRESIM_ROUND2_HANDOFF.md) for exact input/output/dimension authority files, setup and parallel jobs.

This seed contains frozen inputs/calibration/range statistics and pinned source, not final quantized programs or golden outputs. Finish those on the remote CPU while ready ResNet jobs run. TinyLlama tuning uses 16/1 tokens; E2E uses 32/32. **SmolVLA is paused and excluded.**

Verify `SHA256SUMS`. `seed.json` binds all inputs, source files, profiles and the checkpoint download. `requirements.txt` records the preparation environment; the source bundle includes the reviewed reference-lifetime fix. No TVM patch, new DSE knob or hardware change is included.
