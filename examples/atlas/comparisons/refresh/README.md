# Source-bound Atlas workload examples

These examples capture original native operands and replay actual Atlas model outputs through the unchanged native remainder. They are development and evidence tools. Target arithmetic, FP8 packing, layouts, instruction programs, model execution and RTL checks belong to the companion Atlas OOT repository. Performance work compares Atlas variants against their recorded Atlas baselines; native replay is a semantic diagnostic, not an accelerator timing measurement.

## Environments and inputs

An installed Merlin checkout requires Python 3.12 or later. Historical native captures use distinct, manifest-bound Python, Torch and framework environments; replay must use those exact versions and original source files. For an older recorded interpreter, run the example directly with the current checkout's `src` plus the recorded native package locations on `PYTHONPATH`, rather than installing a different framework version. Pure controls need NumPy; native work additionally needs the source-specific packages and external inputs below. Nothing here downloads weights, reconstructs a missing historical artifact or installs an environment.

| Workload | Required external source inputs | Native dependencies |
| --- | --- | --- |
| ResNet | Original checkpoint, image/class stream and authenticated native receipt supplied through `--baseline-root` | Torch, torchvision and the recorded image loader |
| TinyLlama | Original checkpoint/tokenizer, raw-text corpus and pinned token positions | Torch, Transformers, safetensors and tokenizer dependencies |
| SmolVLA | Original checkpoint/config, retained TRAIN prefix/noise, masks/positions, trajectory/cache archives and historical source helpers | Torch, Transformers, LeRobot and safetensors |

Smol projection-chain replay also requires the external `patch` command. Its exact current base replay and bundled patch are hash-pinned, applied with zero fuzz, and checked against the expected generated source hash before import. Use the current `evaluate_smolvla_retained_prefix.py` as `--base-replay` for this release. Earlier source epochs require their own frozen helper bytes and must not be relabeled as the current source.

ResNet uses the local `resnet_native.py` checkpoint/image admission helper without a TVM checkout or runtime. New reports keep the pinned historical `verifier_source` declaration separate from the current helper identity in `native_admission`, bookend current helper bytes, and retain every existing source/preparation binding, including historical helpers declared by older captures.

All inputs use physical paths without symlink components or parent traversal. Archive member metadata is admitted before content hashing or member loading. Every output directory must be fresh and beneath the configured `artifacts_dir()`; inputs and historical results remain untouched. Workload execution needs a separately supervised CPU, RSS and wall-time budget.

## Capture and replay boundaries

| Entry point | Native boundary and extent |
| --- | --- |
| `refresh/capture_resnet.py` | Bias-free stride-one 1x1 convolution before BatchNorm; default `layer1.0.conv1`, first 32 spatial/output positions, complete K64. `--rows 3136 --columns 64` selects the full named convolution. `--site layer1.0.downsample.0 --rows 3136 --columns 256` selects its original shortcut. |
| `refresh/capture_resnet_3x3.py` | Original `layer1.0.conv2`, including native 3x3 padding and complete K576. |
| `refresh/capture_resnet_coupled.py`, `capture_resnet_conv3.py` | A later convolution's native reference on the admitted actual earlier contraction outputs. Source manifests bind the candidate chain. |
| `refresh/evaluate_resnet_substitution.py`, `evaluate_resnet_coupled.py`, `evaluate_resnet_three_convs.py` | One, two, or three main-path substitutions; the last supports the explicit fourth shortcut result. BatchNorm, residual/ReLU and the suffix remain native. |
| `../capture_tinyllama_native.py`, `capture_tinyllama_native_kv.py` | Original layer-zero normalized-input Q or K/V projections before RoPE. Default Q is a 32x32 tile with full K2048; `--full-query` captures all 128 positions and 2048 Q channels. K/V bounds derive from the pinned configuration. |
| `../compare_tinyllama_attention.py`, `compare_tinyllama_qkv_attention.py` | Original native attention/residual with optional admitted Q or Q/K/V replacements. The QKV edge captures conditional O from that actual candidate context. Native masks, positions, RoPE, GQA and cache transforms remain unchanged. |
| `../compare_tinyllama_session.py` | Complete original 22-layer native model with optional layer-zero prefill Q/K/V/O substitutions: 128-token prefill plus 32 teacher-forced inputs, scoring exactly targets 128–159. This is a partial-Atlas hybrid. |
| `../capture_smolvla.py`, `capture_smolvla_full_query.py`, `capture_smolvla_projections.py` | Original first-layer projections. Full export has 192 rows: 177 native positions (141 attention-valid plus 36 masked) and 15 export-only rows of positive-zero BF16 words. Native masks/positions are never extended. K/V width is 320, Q/O width 960. |
| `../evaluate_smolvla_attention.py`, `evaluate_smolvla_retained_prefix.py`, `evaluate_smolvla_projection_chain.py` | Selective native attention or the original closed ten-step policy trajectory, with explicit actual Q or QKV/QKVO replacements. Retained prefix/noise bypass fresh vision. Conditional O must bind the genuine current QKV context; the older Q/native-KV context cannot supply it. |

The ResNet source is torchvision ResNet50 v1.5 `IMAGENET1K_V2`, with resize232/center224 normalization, rather than a reproduction of the original paper's checkpoint or accuracy protocol. Default captures select development20 index 0 in the frozen source order. The capture entry points do not select the held-out cohort. Native pre-BN weights are retained without folding normalization or changing parameters.

## Example commands

Use source-specific interpreter/package locations from the authenticated native manifest. The following shows ResNet setup; replace each placeholder with its explicit physical path. All paths are caller inputs, not checkout-relative guesses.

```sh
merlin=/absolute/path/to/merlin
baseline=/absolute/path/to/recorded-native-inputs
native_python=/absolute/path/to/recorded-native-python
native_packages=/absolute/path/to/recorded-native-packages
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$merlin/src:$native_packages"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$merlin"
artifact_root="$("$native_python" -c 'from merlin.common.paths import artifacts_dir; print(artifacts_dir())')"
capture="$artifact_root/recaptures/resnet50/atlas-sota/unique-run-id"
"$native_python" examples/atlas/comparisons/refresh/capture_resnet.py \
  --baseline-root "$baseline" --output-dir "$capture" --rows 3136 --columns 64
```

Use each entry point's `--help` for its explicit source, manifest, provider and output arguments. For TinyLlama start with `capture_tinyllama_native.py --checkpoint ... --corpus ... --full-query --out ...`; its checkpoint, corpus and token hashes are fixed in source. The session edge takes a hash-pinned `--spec` binding all four native bundles, actual provider receipts, original attention references and selected provider/model roots. For SmolVLA start from the authenticated retained source arguments of `capture_smolvla.py`, then use the full-query/projection edges. These inputs are required external evidence, not artifacts produced by installing Merlin.

## Contracts and claims

`operands.npz` contains contiguous float32 views `A[M,K]`, `B[K,N]`, `native_C[M,N]`. BF16 sources also contain exact uint16 `A_bf16`, `B_bf16`, `native_C_bf16` words. ResNet retains original FP32 values. The manifest binds source/site/ranges, native dtype, raw array identities and NPZ hash; the target provider owns any conversion and reports source error separately.

Native substitution admits an explicitly pinned cold-contraction receipt, all source/helper/model/input bookends, complete disjoint full-K tile coverage and actual physical output bytes. Before importing the OOT validator, the example authenticates its complete three-file producer tuple. The selected validator retains its schema/layout checks while this invocation enforces archive-first hashing and strict physical paths. Corrected post-VPU BF16 words supply replacements; optional raw accumulator words are an independent pre-rescale diagnostic. A whole-program model or RTL receipt is separate evidence and cannot silently replace this cold-contraction interface.

Original, instrumented, repeated candidate and restored native execution remain separate. Source parameters, inputs, cache semantics and RNG restoration retain their original checks. The `rtol=atol=1e-4` native-fidelity diagnostics remain unchanged. A rejected diagnostic is never a source-quality pass; setup/source failures never count as completed evidence. None of these examples establishes a complete Atlas workload, compiler/scheduler qualification, native application quality, fresh Smol vision, physical action units or platform timing. Host framework/packing costs are not target kernel cycles.

## Focused validation

The source-only controls exercise path/archive admission, pre-import binding, source linkage, projection bounds, cache prefixes, RNG restoration and teacher-forced scoring. They do not load checkpoints or execute native models, Atlas models or RTL. From an installed development environment, with bytecode disabled, run the explicit files:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q --noconftest -o addopts='' \
  merlin/tests/infra/test_native_capture_paths.py merlin/tests/infra/test_native_admission.py \
  merlin/tests/infra/test_resnet_native_admission.py \
  merlin/tests/infra/test_native_query_bounds.py merlin/tests/infra/test_native_attention_linkage.py \
  merlin/tests/infra/test_native_kv_capture.py merlin/tests/infra/test_native_qkv_attention.py \
  merlin/tests/infra/test_native_session.py
```

Native source controls and saved result audits need their explicitly supplied external artifacts and budgets. Passing the small source tests does not repeat or replace those qualifications. Historical evidence remains bound to its original frozen sources.
