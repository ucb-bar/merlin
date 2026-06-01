# TargetGen (Workstream 1)

Pipeline:

```
ISA/docs/RTL/examples
  -> target_contract.yaml
  -> dialect_plan.yaml
  -> generated dialect scaffold + tests
```

The realistic goal is **structured scaffold generation with validation gates**, not
"RTL to correct dialect."

## Modules

`merlin/python/merlin/targetgen/{ingest,extract,plan,generate,validate,backends}/`. The default
emission backend is `backends/xdsl_backend.py`; `mlir_cpp_backend.py` / `tablegen_backend.py`
come later, once a dialect shape stabilizes.

## Reference target

`merlin/targets/toy_npu/` — eventually exposes `toynpu.{res_pack,matmul,commit,evict}` and
`!toynpu.{resident_tensor,accumulator}`. See `docs/adding_a_target.md`.

## Tool

`tools/merlin-targetgen/` (writes to `output/targetgen/`).

## Must not

Build a generator with no target example; parse arbitrary RTL as if semantics are obvious; modify
kernel-mining or DSE logic except through shared schemas.
