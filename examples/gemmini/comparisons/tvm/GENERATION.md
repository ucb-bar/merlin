# TVM Gemmini generation

The current top-down implementation is documented in [ARCHITECTURE.md](ARCHITECTURE.md).
Use [TENSORIR_OP_AUDIT.md](TENSORIR_OP_AUDIT.md) for exact operation boundaries,
[OPTIMIZATION_KNOBS.md](OPTIMIZATION_KNOBS.md) for tuning policy/evidence and
[FIRESIM_HANDOFF.md](FIRESIM_HANDOFF.md) for the approved campaign.

The earlier generation account described automatic bias folding, prepacking,
scale/readout preparation, XGB and a 32-plan graph study. Those are historical,
not the current implementation. Its source-bound results and explanation remain
[in the prior revision](https://github.com/ucb-bar/merlin/blob/9ab339ca2/examples/gemmini/comparisons/tvm/GENERATION.md).
Historical speedups must retain those older source, input and memory-model identities.
