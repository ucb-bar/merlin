# Caveat-resolving variant recaptures (P20-S3)

Raw `.mlir` here are **gitignored** (regenerable). Only the committed analysis
`../case_study/manual_validation/capture_shape_sensitivity.{md,csv}` is tracked. Regenerate:

```
.venv/bin/python merlin/benchmarks/dse_guidance/variant_capture.py
```

Produces: `tiny_llama_decode/` (M2M_SEQ=1 → true M=1 decode) and `rdt_depth6/` (M2M_RDT_DEPTH=6). These
bypass workloads/capture.py (whose capture.toml [env] would override the knobs) and call m2m.convert
directly in the model venv with the env override. They resolve two P19 caveats (GEMV-at-decode;
rdt-giant-op-is-depth-2) and do NOT alter the committed corpus.
