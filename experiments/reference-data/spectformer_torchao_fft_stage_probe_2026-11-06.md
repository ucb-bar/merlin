# Trained SpectFormer TorchAO W8A8: first spectral arithmetic divergence

The first same-input spectral-filter difference occurs in the **forward FFT's width DFT**, before the complex gate. The captured complex multiplication is bitwise equal to eager Torch when both receive the same complex FFT tensor. The inverse FFT has a second, smaller same-input difference. This isolates arithmetic stages; it does not qualify the complete model or justify changing a golden, tolerance, or production lowering.

## Reproduction and tensor layout

The exact trained source capture `model.mlir` has SHA-256 `9ae3f772ed2d192899cea502e5dec0098d5abf95f56f1707c7ba3cdfcf5481ab`; its checkpoint has SHA-256 `77cacfdbe53f3615666291d2a9c45aa6dd9132e48e0448bee90c3a3f1a116e3e`. The trained image is the capture image. The independent holdout is NumPy `default_rng(20261106).standard_normal((1,3,224,224)).astype(float32)`, raw tensor SHA-256 `454659ed15e54276928ccca0d0e65bd710fe780dae0b6f0b8cf5755d063ca487`.

Both host runs used the 1,361 existing compiled libraries from the exact capture, with a guard that raised on any compilation request. Both complete replay outputs were bitwise identical to the previously saved host outputs. No JIT, Spike, or FireSim run was made. Eager comparison used the trained checkpoint with zero missing or unexpected keys, Torch `2.16.0.dev20261104+cpu`, TorchAO `0.18.0`, `int8_dyn_act_int8_weight`, and one OpenMP/MKL thread. Each Torch stage received the *saved Merlin host tensor* at the corresponding input boundary, so preceding model drift cannot explain the same-input rows.

The MLIR `fft_0` result has shape `[1,14,8,256,2]`, with the last dimension holding real then imaginary float32 lanes. It is formed by `%399`, the second `tensor.insert_slice` into that five-dimensional tensor. The complex product is `%455`, likewise formed by two lane inserts. The first width DFT is the pair of outlined dispatches `forward$kernel_17__rfft_0` and `forward$kernel_18__rfft_0`, each `[3584,8]`; each is reshaped to `[1,14,256,8]`, transposed to `[1,14,8,256]`, and paired as real and imaginary. The width comparison uses `torch.fft.rfft(norm='backward', dim=2) / 14`, mathematically the same scale as the captured width DFT in the `norm='ortho'` 14×14 transform. That comparison tests the first DFT stage; Torch may place normalization differently internally.

The earlier `fft_0` and `complex_mul_0` *dispatch taps* are internal arrays, not these completed complex tensors. In particular, the first `complex_mul_0` dispatch is only one real lane; interpreting it as the full product yields invalid comparisons. The full `%399` and `%455` values were copied immediately at their insert-slice boundaries.

## Results

All relative L2 errors use the eager Torch stage as denominator and float64 norms of float32 outputs. `max` is maximum absolute difference.

| Input | Comparison | Relative L2 | Max | Elements with absolute error > `1e-6` |
| --- | --- | ---: | ---: | ---: |
| Trained | Width DFT vs Torch width FFT / 14 | `9.573001e-8` | `1.907349e-6` | 2 |
| Holdout | Width DFT vs Torch width FFT / 14 | `9.109810e-8` | `1.430511e-6` | 1 |
| Trained | Complete `rfft2`, same norm1 input | `1.194718e-7` | `7.629395e-6` | 182 |
| Holdout | Complete `rfft2`, same norm1 input | `1.263381e-7` | `7.629395e-6` | 227 |
| Trained | Complex multiply on each path's own FFT | `6.270004e-7` | `1.934823e-7` | 0 |
| Holdout | Complex multiply on each path's own FFT | `6.354410e-7` | `1.797453e-7` | 0 |
| Trained | Complex multiply on **host FFT** in both paths | **0, bitwise** | **0** | 0 |
| Holdout | Complex multiply on **host FFT** in both paths | **0, bitwise** | **0** | 0 |
| Trained | `irfft2` on **host product** in both paths | `1.680815e-7` | `2.607703e-8` | 0 |
| Holdout | `irfft2` on **host product** in both paths | `1.633928e-7` | `1.490116e-8` | 0 |
| Trained | Complete filter, same norm1 input | `6.228843e-7` | `9.313226e-8` | 0 |
| Holdout | Complete filter, same norm1 input | `6.439553e-7` | `1.043081e-7` | 0 |

The complete-filter rows agree with the preceding repeated-float report. The width DFT mismatch is present on both independent inputs. The forward FFT's explicit float32 DFT contractions and Torch's FFT produce close but distinct results. The exact cause within the DFT (coefficient rounding, normalization placement, or reduction order) remains unproven. The product's apparent pathwise difference is entirely inherited from its input FFT in this probe. The inverse FFT contributes its own difference even with identical complex product input.

## Action

No narrow provenance-gated compiler fix is justified by these measurements. A rewrite of FFT arithmetic merely to match one Torch version could alter other captures, and the complete 1,000-logit TorchAO golden still fails source parity. The next focused check is to capture the dynamic activation quantization codes entering the first fc1 for both images, including the codes on host norm2 and on Torch norm2 after the same host filter input. That will test whether the small FFT and norm2 differences actually cross quantization thresholds and explain the larger fc1 divergence. Any FFT candidate fix should then be tested against the unchanged full logits and independent holdout.
