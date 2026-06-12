# Adding a target

Toy/reference targets live in-tree under `merlin/targets/`. Serious targets should become external
repos or MLIR plugins.

## Steps

1. Create `merlin/targets/<name>/` with `docs/`, `contracts/`, `examples/`, `generated/`, `tests/`
   (each with an `AGENT.md`).
2. Write `contracts/target_contract.yaml` (validate against
   `merlin/schemas/target_contract.schema.yaml`).
3. Run TargetGen to produce `contracts/dialect_plan.yaml` and a dialect scaffold
   (`tools/targetgen/`, writing to `output/targetgen/<name>/`).
4. Prototype the dialect in xDSL first (`merlin/python/merlin/xdsl_dialects/` /
   `targetgen/backends/xdsl_backend.py`); promote to MLIR/C++ only when stable.
5. Add conformance tests under `merlin/tests/conformance/` and per-target `tests/`.

## Reference

`merlin/targets/toy_npu/` is the canonical example: `toynpu.{res_pack,matmul,commit,evict}` and
`!toynpu.{resident_tensor,accumulator}`.
