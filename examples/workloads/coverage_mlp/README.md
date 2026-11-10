# Inspect frontend coverage with a small mixed-operation model

[`loader.py`](loader.py) provides linear, activation and normalization operations.
It is target-neutral: a selected accelerator may admit some signatures and require
the host to handle others. It does not assert that any target compiles the model.

Capture with a trace-capable model2MLIR installation and Merlin's existing worker:

```sh
/path/to/capture-python src/merlin/targetgen/_m2m_capture_worker.py \
  --m2m-dir /path/to/model2MLIR \
  --loader examples/workloads/coverage_mlp/loader.py \
  --dtype fp32 --out /configured/out/artifacts/captures/coverage-mlp/fp32
```

The output includes `linalg.mlir`, external weights, inputs, the eager reference,
`frontend-trace.json`, `pytorch-opset.json` and metadata binding their identities.
Inspect the trace's original, quantized and prepared graph stages independently.
The original graph is recorded before dtype casting or quantization; static call
counts are not MLIR operation counts or runtime invocation counts.

For target-derived quantization, select a reviewed recipe using `--recipe` rather
than assuming a dtype token expresses the accelerator's format. Missing original
graphs, transformation lineage, host support or precision evidence remain diagnostic.
See the [test generation specification guide](../../../docs/guides/10-compiler-generation.md)
for operation accounting, capsule obligations and verified admission.
