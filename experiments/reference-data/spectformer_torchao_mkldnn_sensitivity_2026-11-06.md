# SpectFormer TorchAO source parity: CPU convolution backend probe

This follow-up to `spectformer_torchao_w8a8_source_parity_2026-11-06.md` isolates the effect of Torch's CPU convolution backend. It does **not** certify complete source parity or a compiler fix. The default TorchAO framework golden and the captured graph are unchanged.

The probe loaded the same trained SpectFormer-Ti checkpoint with all keys matched, used the captured input, Torch `2.16.0.dev20261104+cpu`, TorchAO `0.18.0`, and the saved Merlin host tensors. It compared `torch.nn.functional.conv2d` with bias fused or added separately, with `torch.backends.mkldnn.enabled` true or false. The host graph's im2col output was already bitwise equal to the image reshape/transposition, and the host convolution was bitwise equal to its matmul result followed by bias addition. Errors below are float32 output differences from that unchanged host tensor.

| Torch convolution path | Maximum absolute patch-convolution difference | Relative L2 patch-convolution difference |
| --- | ---: | ---: |
| MKLDNN enabled, fused bias (framework default) | `2.95639038e-5` | `7.07513903e-7` |
| MKLDNN enabled, bias added after convolution | `1.78813934e-6` | `1.06269136e-7` |
| MKLDNN disabled, fused bias | `2.26497650e-6` | `1.08610303e-7` |
| MKLDNN disabled, bias added after convolution | `2.26497650e-6` | `9.64887279e-8` |

The default fused MKLDNN path is a major source of the first discrepancy. Its output cannot be inferred from the source graph's separate matmul and bias operations by merely changing the float matrix reduction order. Disabling MKLDNN also changes the *framework* program's complete output substantially, so it cannot be used to redefine the existing TorchAO golden:

| Input | Torch default versus host, relative L2 | Torch MKLDNN disabled versus host, relative L2 | Torch MKLDNN disabled versus default, relative L2 |
| --- | ---: | ---: | ---: |
| Captured trained input | `0.00718246764` | `0.00557446297` | `0.00603193770` |
| Independent seeded holdout | `0.00726392874` | `0.00643496943` | `0.00579989684` |

The captured-input default Torch output reproduced `golden.npy` **bitwise**. The holdout used the same independent seeded image and host output as the first-divergence report; its Torch default result was the comparison reference for that row. Both runs used one OpenMP/MKL thread, so the MKLDNN toggle was the only changed backend choice in each pair.

The host remains different from Torch with MKLDNN disabled, and the earlier exact patch-output injection did not improve the complete-output gate. The repeated spectral-filter, normalization, and activation paths need their own same-input numerical checks before a portable compiler rewrite can be justified. The signed int8 matrix operations themselves matched eager TorchAO when given identical quantized inputs in the first-divergence probe. A target backend must still pass the unchanged default TorchAO golden before claiming model parity.
