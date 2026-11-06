# Trained SpectFormer TorchAO W8A8 source parity, 2026-11-06

This is curated diagnostic evidence for the trained SpectFormer-Ti capture. It does **not** certify source parity. The original Merlin host run is finite and has the same top-1 class (885), but its complete 1,000-logit output fails the framework comparison: relative L2 error is `0.007182467872445907`, maximum absolute error is `0.08121049404144287`, and cosine similarity is `0.999977510173146`. No golden or tolerance was changed.

## Bound inputs and replay

The capture used `int8_dyn_act_int8_weight` through TorchAO 0.18.0 with Torch `2.16.0.dev20261104+cpu`. The loader used the trained `spectformer-tiny/checkpoint_best.pth` with all checkpoint keys matched. The capture receipt says `source_closure_verified: false`; its source inventory is complete only for `m2m` Python files, so that separate closure claim remains open.

| Artifact | SHA-256 of file bytes |
| --- | --- |
| `model.mlir` | `9ae3f772ed2d192899cea502e5dec0098d5abf95f56f1707c7ba3cdfcf5481ab` |
| `weights.safetensors` | `77909d9c4f435d9fd438091a84a468a64bbc5023282fb14e28efd237e3f91071` |
| `inputs.npz` | `8c0c5a7e988d6eda1053b9f51a18bd16e8bd7c1f726cc4053a9ec0176c264c1d` |
| `golden.npy` | `a131a6790842bad1e31007382f741536376df1b8d6bfc3252ee82ed09bb572f3` |
| `extra.npz` | `b519a1d3d0300a489cbfb88a8a43701667dff09f6a9a3606588c5df06ee3dc13` |
| `capture_receipt.json` | `bd554a8cba6caf1b9c9c133073b7a839d3e596b70b2304d99c66aa03cd31038d` |
| Model2MLIR `workloads/spectformer/loader.py` | `6b57f9682a89efbc95feddcedbdd45590f6e36986307566193038e2fde1e2aa6` |
| Original `host_source.npy` | `908c7f605b080542e53d0fb4e66eea1d3cc753759b2052a58254c9b7e2e02368` |

The SHA-256 of the **raw float32 tensor bytes** is `68d415e616438deb2a7cce446ebe0415aa22b5fc01af82ea042b1a` for the golden and `514b2778fe222a09d826703d1249c1e6f013b845eb75967eecf50ddac14ce8f8` for the host output. These differ from the `.npy` file hashes because the files include headers. Rebuilding the seeded trained model and applying the capture's TorchAO scheme produced an eager 1,000-logit tensor bitwise identical to the saved golden.

The per-op host tap reused the original run's 1,361 compiled kernel libraries through symlinks in a local cache. It did not rebuild or copy those libraries. The tapped run reproduced `host_source.npy` bitwise and peaked at 598 MiB RSS under a 3 GiB virtual-memory limit. The independent holdout reused those libraries with a newly generated input; no capture artifact was overwritten.

## First divergence and amplification

The first kernel of `conv_0`, `forward$kernel_0__rconv_0`, only rearranges the image into im2col layout. Its `3×16×16×1×14×14` output is bitwise identical to an independent NumPy reshape and transpose of the captured input. The next kernel, `forward$kernel_1__rconv_0`, is a float32 `linalg.matmul`; it is the first arithmetic divergence from Torch's biasless convolution (relative L2 `6.08e-7`, maximum absolute `1.79e-6`).

The complete patch convolution differs from Torch's **fused convolution with bias** by relative L2 `7.08e-7`, maximum absolute `2.96e-5`. Merlin's convolution output is bitwise equal to its im2col matmul output plus bias. Torch's fused convolution differs from Torch's own biasless convolution followed by bias addition by maximum absolute `2.96e-5`; Merlin versus the latter differs by only `1.79e-6`. This directly identifies the source fused-convolution reduction/bias order as a major contributor. Matching a biasless convolution or changing only the patch projection cannot establish whole-model parity.

| Eager boundary versus corresponding host result | Relative L2 | Maximum absolute |
| --- | ---: | ---: |
| Patch + positional add (`add_0`) | `5.81e-7` | `2.96e-5` |
| Block 0 norm1 | `5.77e-7` | `2.86e-5` |
| Block 0 spectral filter | `3.32e-6` | `9.02e-7` |
| Block 0 norm2 | `5.07e-6` | `2.34e-4` |
| Block 0 TorchAO fc1 output (`add_2`) | `1.56e-4` | `0.03696` |
| End of block 0 (`add_5`) | `4.54e-4` | `0.02611` |

The jump at fc1 comes from dynamic activation quantization of slightly different float inputs. Reconstructing the captured per-token qparams from its `amin`, `amax`, divide by `127.5`, reciprocal, round-even and clamp operations matches Merlin's 50,176 block-0 fc1 int8 activations exactly. Applied to the eager norm2 tensor, 27 of those activations differ from Merlin by one int8 unit. For example, token 5/channel 82 is `-14.59184741973877` on the host and `-14.591846466064453` in eager Torch; its quantized value changes from `-128` to `-127` at the half-step. TorchAO fc1 invoked on the **identical host norm2 tensor** produces a bitwise-identical output to Merlin. TorchAO fc2 on the identical host GELU tensor is also bitwise identical; host GELU versus Torch GELU on the same input has maximum absolute error `4.77e-7`. The observed first-block failure is therefore driven by float input drift crossing a legitimate W8A8 quantization threshold, rather than a demonstrated int8 matmul error.

## Independent input and diagnostic substitutions

A holdout image from NumPy `default_rng(20261106).standard_normal((1,3,224,224))`, cast to float32, has raw tensor SHA-256 `454659ed15e54276928ccca0d0e65bd710fe780dae0b6f0b8cf5755d063ca487`. With the same trained weights, its im2col layout is again bitwise exact. The patch convolution has relative L2 `7.16e-7`/maximum absolute `3.05e-5`; block-0 norm2 has `5.09e-6`/`2.88e-4`; fc1 has `1.59e-4`/`0.02804`; the full 1,000-logit output has `0.007264`/`0.07977`. TorchAO fc1 on the holdout host norm2 tensor again matches Merlin bitwise. This validates the mechanism on a separate input but does not qualify model parity.

Diagnostic injections replaced one host intermediate with the exact eager tensor and ran the remaining original host kernels. They are counterfactual probes, not model fixes or golden changes:

| Injected eager boundary | Remaining host relative L2 versus saved golden |
| --- | ---: |
| None (original run) | `0.00718247` |
| Patch + position (`add_0`) | `0.00720419` |
| End of block 0 (`add_5`) | `0.00649796` |
| End of block 3 (`add_20`) | `0.00495616` |
| End of block 11 (`add_100`) | `8.94e-8` |

The patch-only substitution does not repair the full output. Differences continue to accumulate through the spectral and attention blocks; the final norm, mean and head are close once the last block input is made exact. A general source-order treatment needs independent validation for the repeated float operations and an unchanged complete-output gate. No production compiler change is justified by this diagnosis alone.
