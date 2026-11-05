# Attention-core capture

One 33-by-65 query input, static 65-by-65 keys and 65-by-35 values. The program
returns the 33-by-65 probabilities and the 33-by-35 weighted values. Softmax uses
temperature one. This example has no Q/K/V projections, attention mask, MLP or
mutable state; it is not a full transformer layer.

Capture with the selected model2MLIR environment:

```sh
"$CAPTURE_PYTHON" src/merlin/targetgen/_m2m_capture_worker.py \
  --m2m-dir "$MODEL2MLIR_ROOT" --loader examples/workloads/attention_core/loader.py \
  --dtype fp32 --seed 0 --materialize-bundle \
  --out out/artifacts/recaptures/attention-core
```

Use a fresh output directory. Stage declared compiler inputs separately from
runtime tensors and reference outputs. Any target precision change requires an
explicit policy and separate original-output quality measurements.
