# Tensor reduction

The input is `[2,3,35]`. Sum over axes 0 and 2 produces `[3]`, followed by ReLU.
Both tensors are returned. This exercises non-contiguous reduced axes, a
reduction tail, and a result used both internally and as an output.

Capture with the selected model2MLIR environment:

```sh
"$CAPTURE_PYTHON" src/merlin/targetgen/_m2m_capture_worker.py \
  --m2m-dir "$MODEL2MLIR_ROOT" --loader examples/workloads/tensor_reduction/loader.py \
  --dtype fp32 --seed 0 --materialize-bundle \
  --out out/artifacts/recaptures/tensor-reduction
```

Use a fresh output directory. Keep runtime inputs and reference outputs outside
the staged compiler inputs. Target precision changes require explicit selection
and separate checks against the original PyTorch outputs. This integration
example does not replace the required full-model workloads.
