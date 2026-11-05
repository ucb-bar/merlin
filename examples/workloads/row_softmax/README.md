# Row-softmax capture

This operation diagnostic has one 33-by-65 f32 input, one equally shaped result,
and no parameters or mutable state. It exercises normalization across columns
and non-square shapes. It is not a complete attention layer or an iteration model.

Capture it with the selected model2MLIR environment:

```sh
"$CAPTURE_PYTHON" src/merlin/targetgen/_m2m_capture_worker.py \
  --m2m-dir "$MODEL2MLIR_ROOT" --loader examples/workloads/row_softmax/loader.py \
  --dtype fp32 --seed 0 --materialize-bundle \
  --out out/artifacts/recaptures/row-softmax
```

Use a fresh output directory. Stage compiler inputs separately from evaluation
inputs and goldens as described in the [frontend guide](../../../docs/guides/model2mlir.md).
The captured reduction order and math implementation remain source obligations.
A target with different arithmetic needs an explicitly selected precision policy
and separate original-output quality measurements.

Use `batched_loader.py` with the same command to capture a `[2,3,35]` input.
The six independent rows cross a column-tile boundary. A dense reshape may
combine leading dimensions, but the final softmax axis must remain intact.
This example also exercises the collapse/expand reshape chain emitted by the
frontend around each row reduction.
