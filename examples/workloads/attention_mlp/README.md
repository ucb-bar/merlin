# Attention followed by an MLP

`loader.py` defines single-head attention and a two-layer MLP with ordinary
dense PyTorch weights. The static example has 32 tokens, width 32 and MLP
hidden width 32. It returns the hidden activation and final MLP output.

Set `ATTENTION_MLP_HIDDEN` before capture to select another positive hidden
width. For example, `ATTENTION_MLP_HIDDEN=288` keeps attention at 32×32 and
gives the last projection nine reduction tiles on a target with K tiles of 32.
The compiler reads the captured shapes; it does not read this environment
variable. The default example is unchanged.

## Capture

Use a provisioned Model2MLIR interpreter and its selected source checkout:

```sh
"$CAPTURE_PYTHON" src/merlin/targetgen/_m2m_capture_worker.py \
  --m2m-dir "$MODEL2MLIR_ROOT" --loader examples/workloads/attention_mlp/loader.py \
  --dtype fp32 --seed 0 --materialize-bundle --out "$CAPTURE"
merlin-targetgen stage-capture --capture "$CAPTURE" --out "$COMPILER_INPUTS"
```

Both output directories must be fresh. Capture retains the framework results
for evaluation. Staging copies only IR, declared weights, signature and source
correspondence; runtime samples and expected results stay outside compilation.
The five linear weights remain separate from the one runtime `tokens` input.

## Compile and export

Application compilation uses the native target binding's optional
`application_adapter` interface. The shared
[application compilation guide](../../../docs/guides/model_memory_and_device_lowering.md#compile-a-staged-application)
shows the CLI and C/H export boundary. A provider must supply every device
region and explicitly supported CPU route.

The full application path requires an Atlas support installation providing the
application adapter, compact tiling, and C/H exporter. An older provider without
that interface fails explicitly. The compiler records CPU layout operations and
Atlas computation separately; it does not silently substitute host arithmetic.

This is a public development example. It does not replace the four iteration
models or establish checkpoint quality. The selected precision transformation,
target-operation domains and actual device execution must be checked separately.

## Attention core and MLP with tails

`tiled_core_loader.py` is a second public input for connected application
compilation. It uses query [33,65], keys [65,65], values [65,35], and MLP
weights [35,67] and [67,29]. It returns probabilities [33,65], the ReLU
activation [33,67], and the final output [33,29]. This exercises multiple
output and reduction tiles, tails, and a shared producer used as both an output
and an input of the next layer.

Capture it with the same command above, replacing `--loader` with
`examples/workloads/attention_mlp/tiled_core_loader.py`. A compatible OOT application provider can accept that capture with an explicit
scaled precision policy and ReLU epilogue selection. It fuses an unobserved contraction into
its terminal ReLU only after the full reduction. The CPU performs boundary
packing/conversion and dispatch; the neural operations emit device code.

This variant has temperature-one attention, immutable keys/values, and a
bias-free MLP. It has no Q/K/V projections, attention scaling/mask, residual,
normalization or mutable state. It complements `loader.py`; it does not
establish full-model coverage. The projected/scaled `loader.py` additionally needs an explicit pointwise
precision policy from its selected provider.
Neither example replaces the required models or establishes model quality.
