# Embedding and projection

The input is a dense `[2,3]` signed-i64 token tensor. A `[17,7]` embedding table
feeds a bias-free `[7,5]` projection. The model returns the `[2,3,7]` embeddings,
the `[2,3,5]` projection and the unchanged integer input, in that order.

Capture with the selected model2MLIR environment:

```sh
"$CAPTURE_PYTHON" src/merlin/targetgen/_m2m_capture_worker.py \
  --m2m-dir "$MODEL2MLIR_ROOT" --loader examples/workloads/embedding_projection/loader.py \
  --dtype fp32 --seed 0 --materialize-bundle \
  --out out/artifacts/recaptures/embedding-projection
```

Use a fresh output directory and stage compiler inputs separately from runtime
indices and reference outputs. A target policy must explicitly select the CPU
lookup and any projection precision change. Invalid indices must fail without
publishing any model output. This integration example does not replace the
required full-model workloads.
