# Exported FX graph dumps (P19 forensic audit, Phase 2)

The `<wl>.txt` here are **gitignored** (regenerable, ~3.5 MB total; pi05 alone 1.7 MB). Each is the
re-run `torch.export` graph (ATen op histogram + readable FX) for that workload — the "exported module"
column of the source→export→MLIR→artifact audit. The ExportedProgram is ephemeral (m2m never persists it),
so regenerate with:

```
.venv/bin/python merlin/benchmarks/dse_guidance/dump_exported_fx.py            # all workloads
.venv/bin/python merlin/benchmarks/dse_guidance/dump_exported_fx.py rdt pi05   # specific
```

(small_llama uses the shared model2MLIR/.venv since it has no own venv — see dump_exported_fx.py.)
The key audit signal is the OP HISTOGRAM: attention appears as `aten.scaled_dot_product_attention` in the
exported graph and is only decomposed during torch-mlir lowering.
