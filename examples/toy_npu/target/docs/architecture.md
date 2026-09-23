# ToyNPU Architecture (reference target)

Documentation only.

- A small matmul engine with a resident weight store and an accumulator bank.
- Resident store: fixed byte budget (see `contracts/target_contract.yaml`).
- Accumulators: integer, support a fused epilogue (bias, requant, relu) at commit.

This architecture is chosen to exercise the `resident_packed_tensor` and
`accumulator_commit` abstractions end-to-end through the three workstreams.
