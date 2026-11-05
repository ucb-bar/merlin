# Atlas: whole-model entrypoints

Whole-model inspection and accelerator deployment are different operations.
Start with an existing [model2MLIR capture](../../../docs/guides/model2mlir.md)
and its real weights/manifest, then follow the shared
[lowering and tensor-inspection guide](../../../docs/guides/model_lowering.md).
`merlin lower` consumes that capture and writes invocation-owned IR stages and
audit records beneath a fresh configured output destination. It invokes native
lowering tools; it does not by itself run or certify an accelerator compiler.

The [accelerator workflow](../../../docs/guides/whole_model_on_accelerator.md)
has separate provider, toolchain, model and execution prerequisites. Select the
exact [target descriptor](../target/descriptor.yaml) and independently qualified
compiler/support inputs; do not substitute another target's historical result.
A Phase 1 capsule certificate is not whole-model numerical or timing evidence.

Keep captures, weights, binary tensor payloads, lowered programs and execution
receipts outside examples. Preserve capture hashes and link each new result to
the exact compiler and target evidence. Compact IR views are inspection artifacts,
not executable replacements. No independently qualified Atlas whole-model
deployment is supplied here.

To inspect the actual generated Atlas package on a held-out capture, install
the experiments package and use its deterministic evaluator:

```sh
python -m merlin_experiments.model_qualification \
  --bundle /absolute/held-out-capture \
  --package /absolute/generated-atlas-compiler \
  --target atlas \
  --evidence-bundle /absolute/saved-phase0-evidence \
  --lower-native \
  --out /configured/out/artifacts/model-qualification/atlas-inspection-001
```

The fresh, read-only output binds exact compiler, capture, weights, session and
target-evidence bytes. Check actual emitted IR and command-buffer contents, not
just a zero exit code: a pass-through graph, empty buffer or explicit decline
does not establish target lowering. `status: completed` only means the diagnostic
observation completed. It never grants a compiler or whole-model certificate.

`--lower-native` audits every complete declared program through the shared LLVM IR
route and binds its weight sidecars. `full_native_lowering_verified` is independent
of Atlas offload or numerical execution: read the OOT compiler decisions separately.
The generated `README.md` links each program's lowering index. Preserve all stages
of TinyLlama prefill/decode and SmolVLA prefix/denoising/action decode; a single
diagnostic stage cannot stand in for the whole workflow.

See the [evaluation walkthrough](../../gemmini/whole-model/README.md#evaluate-a-generated-compiler-without-tuning-on-validation-models)
for receipt fields, resource limits, optional single-forward `--execute`, host
fallbacks and the current multi-program execution limitation. Full TinyLlama,
SmolVLA and ResNet-50 are held-out validation workloads, never capsule derivation
or tuning inputs. Partial checkpoints, synthetic data and one denoising stage
remain explicitly diagnostic. The evaluator uses the selected OOT compiler and
saved facts; it does not download a model or silently select a reference backend.

The model runtime, builders, tests, and runbooks are maintained in [model bringup](../model_bringup/README.md), beside the capture providers in [comparisons](../comparisons/README.md).
