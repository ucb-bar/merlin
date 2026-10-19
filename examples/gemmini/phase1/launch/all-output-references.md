# Supplemental complete-output references

`produce_all_outputs_reference.py` is a private operator. It evaluates every
declared program output at every required step from existing captures and model
inputs. It does not capture models, invoke their FP baselines, change the selected
bundle, execute the candidate compiler, or establish a compiled-model pass.

Select the same source checkout, YAML, preparation directory and run identifier
as the experiment. Use a **new private output directory outside candidate grants**.
Set `PYTHONPATH` to that checkout's `src` and `packages/merlin-experiments/src`.
The selected launcher's configured environment supplies the sealed bundle, math
policy and shared CPU pool. Pass each model explicitly; the example has no default
model roster or input size.

```sh
"$PYTHON" "$REPO/examples/gemmini/phase1/launch/produce_all_outputs_reference.py" \
  --repo "$REPO" --config "$CONFIG" \
  --launcher "$REPO/examples/gemmini/phase1/launch/launch_phase1.py" \
  --task-root "$TASK_DIR" --prep "$PREP" --run-id "$RUN_ID" \
  --out "$PRIVATE_OUT" --deadline "$DEADLINE_WITH_TIMEZONE" \
  --input visible/v00 --model resnet50 --model tiny_llama --model smolvla \
  --timeout 1800 --max-pss-gib 192 --min-headroom-gib 24
```

The local timeout and absolute deadline both apply. Each single-threaded worker
leases a shared auxiliary CPU. The supervisor records process birth identities,
checks memory bounds and reaps all owned children on completion or interruption.
An incomplete worker publishes no complete manifest. The output directory cannot
already exist, so a failed attempt never overwrites earlier evidence.

Each model publishes `reference.json`, `SHA256SUMS` and one `outN` NPZ per program
step using `merlin.operator.full_declared_output_reference.v1`. The manifest binds
the selected bundle/configuration, original inputs/capture, evaluator and selected
math implementation. Every original judged output must bitmatch the sealed
`int8.npz`; every declared output must have finite, stable storage dtype and shape.
Consumed source/input identities are checked again before publication. Reference
adapter build receipts remain in `scratch/host_math`; retain them as provenance.

Before comparison, admit and protect each produced manifest and its payloads and
record its SHA independently. Pass that admitted SHA as
`merlin.model_check.full_output_audit --reference-sha256`; never select a supplement
from an untrusted current file alone. The offline audit compares all arrays with
the unchanged selected numerical policy and original retained Spike readback.
Its pass is an additional gate alongside complete full-model execution, exact
accelerator-kernel checks and native offload/host-admission qualification.

Historical supplements retain their original temporary producer SHA. Promoting
this reusable operator does not relabel or regenerate any prior evidence.
