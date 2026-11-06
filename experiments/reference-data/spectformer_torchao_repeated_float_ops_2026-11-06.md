# Trained SpectFormer TorchAO W8A8: repeated float operations, 2026-11-06

The complete trained SpectFormer-Ti capture still fails source parity: the previously measured 1,000-logit Merlin host result has relative L2 error `0.007182467872445907` and maximum absolute error `0.08121049404144287` against the unchanged TorchAO golden. This report isolates further same-input differences and checks them on an independent image. It does not qualify the model or change a golden, threshold, capture, or quantization scheme.

## Provenance and replay

The capture `model.mlir` SHA-256 is `9ae3f772ed2d192899cea502e5dec0098d5abf95f56f1707c7ba3cdfcf5481ab`; the trained `checkpoint_best.pth` SHA-256 is `77cacfdbe53f3615666291d2a9c45aa6dd9132e48e0448bee90c3a3f1a116e3e`. The unchanged `golden.npy` SHA-256 is `a131a6790842bad1e31007382f741536376df1b8d6bfc3252ee82ed09bb572f3`. The independent holdout is NumPy `default_rng(20261106).standard_normal((1,3,224,224)).astype(float32)`, raw tensor SHA-256 `454659ed15e54276928ccca0d0e65bd710fe780dae0b6f0b8cf5755d063ca487`. Torch `2.16.0.dev20261104+cpu`, TorchAO `0.18.0`, one OpenMP/MKL thread, the same checkpoint (zero missing and unexpected keys), and `int8_dyn_act_int8_weight` were used for both images.

The holdout host replay used the original 1,361 compiled `model_host.so` libraries from the exact capture's host workdir, linked into this task's cache by outlined kernel symbol. The current xDSL printer changes MLIR text and therefore cache hashes: 1,171 initial cache lookups missed even though the old binaries existed. No library was rebuilt. A compile guard raised on any remaining miss. All 1,500 outlined dispatches resolved, and the complete replayed holdout output was bitwise identical to the previously saved host output, SHA-256 `08d9cc56f5262ee548db6cbdd922ac414c2abc8b2038804fe77b0b8b740c7a0e`. Replay took 16.99 seconds and peaked at 558,684 KiB RSS. The saved trained and holdout per-op tensors were then compared with eager trained TorchAO operations receiving **identical input tensors**; this separate comparison took 3.02 seconds and peaked at 526,692 KiB RSS. The holdout norm2 tap matched the previously saved holdout norm2 tensor bitwise.

## Same-input numerical results

Each error is measured against the Merlin host tensor for the named operation. The filter check feeds Torch `blocks.0.filter` the saved Merlin `blocks.0.norm1` tensor. The norm2 check feeds Torch `blocks.0.norm2` the saved Merlin filter output. The fc1 check feeds TorchAO fc1 the saved Merlin norm2 output. GELU receives the saved Merlin fc1 output. The last fc1 row instead feeds TorchAO fc1 Torch's norm2 result from the **same saved filter input**, showing propagation of the norm2 difference.

| Image | Operation | Relative L2 | Maximum absolute | Elements with error > `1e-3` |
| --- | --- | ---: | ---: | ---: |
| Trained | spectral filter | `6.228843e-7` | `9.313226e-8` | 0 |
| Holdout | spectral filter | `6.439553e-7` | `1.043081e-7` | 0 |
| Trained | norm2 | `1.571977e-7` | `5.722046e-6` | 0 |
| Holdout | norm2 | `1.611917e-7` | `1.144409e-5` | 0 |
| Trained | fc1 on identical host norm2 | **0, bitwise** | **0** | 0 |
| Holdout | fc1 on identical host norm2 | **0, bitwise** | **0** | 0 |
| Trained | fc1 after Torch norm2 drift | `1.087135e-4` | `0.03696251` | 2,278 |
| Holdout | fc1 after Torch norm2 drift | `1.214196e-4` | `0.04187727` | 2,856 |
| Trained | GELU | `2.097032e-7` | `4.768372e-7` | 0 |
| Holdout | GELU | `2.106901e-7` | `4.768372e-7` | 0 |

These two independent inputs show that the spectral filter, layer norm, and GELU do not reproduce Torch's exact float32 operation order. The spectral filter is the first of these repeated float differences on identical inputs. The float differences are individually small, but the fc1 result changes much more when the input is Torch's slightly different norm2 result. Exact fc1 agreement on the host norm2 input excludes the signed int8 fc1 matrix kernel itself as the cause of this local discrepancy. Crossing dynamic activation quantization thresholds is a plausible mechanism for the amplification; quantized activation codes were not captured in this experiment, so that particular threshold mechanism is not yet proven.

The capture lowers FFT through explicit float32 spectral operations, including `linalg.matmul` stages, while the source invokes `torch.fft.rfft2` and `irfft2` with `norm='ortho'`. The numerical differences are consistent with distinct reduction orders, but this experiment did not isolate one FFT stage or compare individual complex coefficients. No provenance-gated production rewrite follows from the evidence yet. A focused next experiment should capture Torch's complex result immediately after `rfft2` and after complex weight multiplication, compare it with correctly decoded Merlin complex tensors from the same norm1 input, then quantify activation-code changes at the first fc1 boundary for both images. Any candidate rewrite must pass the unchanged complete 1,000-logit golden and an independent holdout before being accepted.
