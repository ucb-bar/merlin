# Regenerating the multi-level recaptures (capture-level ablation)

The raw `.mlir` here are **gitignored** (regenerable, ~18 MB). Only the committed op-count summary
`../case_study/capture_level_ablation.csv` is tracked. To regenerate, from `model2MLIR/`:

```
for wl in rdt openvla pi05 bitvla; do
  .venv/bin/python workloads/capture.py $wl --formats fp32 --level high-level --no-venv
  cp workloads/$wl/$wl.mlir       <repo>/merlin/benchmarks/dse_guidance/recaptures_levels/$wl/model_highlevel.mlir
  .venv/bin/python workloads/capture.py $wl --formats int8 --no-venv
  cp workloads/$wl/${wl}_int8.mlir <repo>/merlin/benchmarks/dse_guidance/recaptures_levels/$wl/model_qdq.mlir
done
```

Then re-run `merlin-dse-guidance --case-study` to refresh `capture_level_ablation.csv`.

- `--level high-level` emits attention/softmax as named `linalg_ext.*` ops (attention-preserving).
- `--formats int8` triggers `preserve_qdq` -> `quant_ext.dequantize` (quant-metadata-preserving).
- Loop-preserving is **torch.export-blocked** (no flag): torch.export unrolls Python loops.
