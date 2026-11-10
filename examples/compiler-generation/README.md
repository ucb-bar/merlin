# Generate a target compiler

These directories hold authored target workflows, not generated compiler source.
Use the [compiler-generation guide](../../docs/guides/10-compiler-generation.md)
for the flow from software/hardware inputs through tests, agent authoring and
independent evaluation.

| Target workflow | Catalog ID |
| --- | --- |
| [Gemmini](gemmini/README.md) | `gemmini-functional` |
| [Gemmini Universal](gemmini_universal/README.md) | `gemmini-universal-functional` |
| [Atlas](atlas/README.md) | `atlas-functional` |
| [Radiance](radiance/README.md) | `radiance-functional` |
| [MX Gemmini](mx_gemmini/README.md) | `mx-gemmini-functional` |
| [Saturn OPU](saturn_opu/README.md) | `saturn-opu-functional` |
| [Saturn OPU RVV](saturn_opu_rvv/README.md) | `saturn-opu-rvv-functional` |

Inspect a definition without starting an agent:

```sh
merlin experiment list
```

Each workflow keeps test inputs in `test-generation/` and authoring inputs in
`backend-generation/`. Optional optimization inputs live inside that backend
workflow. Historical engine/configuration keys remain separate compatibility
metadata; they do not define additional public workflows. Generated candidates
and results belong under `out/`.
