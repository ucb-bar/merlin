# Initial model inventory

Audited 2026-10-19T19:20:48.606083+00:00 using the coordinator’s rebuilt Lean checker. Initial preparation used source `ea43634ec6005c88610abd754f4e06f04b4793c4`; detailed environment/source records remain in shared `inventory/environment.json`.

Private full-model captures, checkpoints, recipe and model2MLIR/TorchAO environment were unavailable. These are **public stand-ins** from retained model capsules: real preparation with `int8_compute=True`, then xDSL preprocessing. No device build, ELF, upstream tensor coverage or paper qualification is claimed.

**Recommendation remains provisional:** SY_micro_model exercises W8A8 contraction, quant_ext lowering, scalar-square, scalarize-splat and DCE. M3 is smaller but does not fire W8A8. Final model selection requires Gate A review.

| Candidate | Bytes | Preparation | Typed verified | Universal verified | Source support | Source definedness |
|---|---:|---:|---:|---:|---|---|
| M3_host_island_seam_gemmini | 15046 | 0.271 s | 18/18 | 14/18 | verified | abstracted |
| SY_micro_model | 31283 | 0.340 s | 14/18 | 13/18 | abstracted | abstracted |

These are boundary counts, including unchanged boundaries; they are not changed-site coverage. Neither candidate establishes a nonvacuous full-chain claim because source definedness abstains. M3 source support verifies. SY_micro_model source support refuses at `quant_ext.dequantize_per_tensor: no rule accepts this typing`; its source contains eight such operations and two `quant_ext.quantize_per_tensor` operations. Both definedness receipts report “outside the proved definedness fragment or an output may be uninitialized”; this does not establish an actual uninitialized output.

The earlier missing-audit/stale-binary failures are superseded by these audited results. The former raw M3 linalg.reduce refusal is obsolete.

| Candidate | Boundary | Changed | Typed | Universal | Typed time |
|---|---|---|---|---|---:|
| M3_host_island_seam_gemmini | prepare-parse | True | verified | abstracted | 3.937 s |
| M3_host_island_seam_gemmini | prepare-torchao-affine | False | verified | verified | 1.473 s |
| M3_host_island_seam_gemmini | prepare-scalar-squares | False | verified | verified | 1.424 s |
| M3_host_island_seam_gemmini | prepare-collapse-overrank-matmul | False | verified | verified | 1.416 s |
| M3_host_island_seam_gemmini | prepare-propagate-quant-inner | False | verified | verified | 1.297 s |
| M3_host_island_seam_gemmini | prepare-qinner-lift | False | verified | verified | 1.274 s |
| M3_host_island_seam_gemmini | quant-contraction_int8 | False | verified | verified | 1.263 s |
| M3_host_island_seam_gemmini | quant-conv_int8 | False | verified | verified | 1.298 s |
| M3_host_island_seam_gemmini | prepare-lower-quant-ext | False | verified | verified | 1.394 s |
| M3_host_island_seam_gemmini | prepare-bf16-matmul-f32acc | False | verified | verified | 1.317 s |
| M3_host_island_seam_gemmini | prepare-fix-bool-sitofp | False | verified | verified | 1.344 s |
| M3_host_island_seam_gemmini | prepare-fix-bool-fptosi | False | verified | verified | 1.396 s |
| M3_host_island_seam_gemmini | prepare-scalarize-splat-inputs | True | verified | abstracted | 1.455 s |
| M3_host_island_seam_gemmini | prepare-print | True | verified | abstracted | 1.555 s |
| M3_host_island_seam_gemmini | xdsl-parse | True | verified | abstracted | 1.489 s |
| M3_host_island_seam_gemmini | xdsl-pruned | False | verified | verified | 1.457 s |
| M3_host_island_seam_gemmini | xdsl-quant-lowered | False | verified | verified | 1.300 s |
| M3_host_island_seam_gemmini | xdsl-c-interface | True | verified | verified | 1.338 s |
| SY_micro_model | prepare-parse | True | abstracted | abstracted | 1.316 s |
| SY_micro_model | prepare-torchao-affine | False | verified | verified | 1.479 s |
| SY_micro_model | prepare-scalar-squares | True | abstracted | abstracted | 1.488 s |
| SY_micro_model | prepare-collapse-overrank-matmul | False | verified | verified | 1.559 s |
| SY_micro_model | prepare-propagate-quant-inner | False | verified | verified | 1.745 s |
| SY_micro_model | prepare-qinner-lift | False | verified | verified | 1.560 s |
| SY_micro_model | quant-contraction_int8 | True | abstracted | abstracted | 1.612 s |
| SY_micro_model | quant-conv_int8 | False | verified | verified | 1.799 s |
| SY_micro_model | prepare-lower-quant-ext | True | abstracted | abstracted | 1.770 s |
| SY_micro_model | prepare-bf16-matmul-f32acc | False | verified | verified | 1.872 s |
| SY_micro_model | prepare-fix-bool-sitofp | False | verified | verified | 1.575 s |
| SY_micro_model | prepare-fix-bool-fptosi | False | verified | verified | 1.642 s |
| SY_micro_model | prepare-scalarize-splat-inputs | True | verified | abstracted | 1.894 s |
| SY_micro_model | prepare-print | True | verified | verified | 1.642 s |
| SY_micro_model | xdsl-parse | True | verified | verified | 1.546 s |
| SY_micro_model | xdsl-pruned | True | verified | verified | 1.550 s |
| SY_micro_model | xdsl-quant-lowered | False | verified | verified | 1.682 s |
| SY_micro_model | xdsl-c-interface | True | verified | verified | 1.704 s |

Typed abstentions on SY_micro_model are prepare-parse, prepare-scalar-squares, quant-contraction_int8 and prepare-lower-quant-ext; each receipt reports “a function body did not verify.” No unsupported boundary is counted as verified. Universal abstentions additionally include some parse/print and scalarize-splat boundaries, as shown above.

Audited typed totals, including export, roundtrip, receipt binding and theorem audit: M3 27.428 s; SY_micro_model 29.433 s. Universal totals: 26.781 s and 28.843 s. Source support/definedness times: M3 2.392/2.572 s; SY_micro_model 0.913/0.958 s. These compare with preparation only (0.271/0.340 s), not full compile time; per-check subprocess and audit overhead is included.

Exact stage before/after bytes, hashes, before-op censuses, receipts and timings remain under the shared `out/artifacts/verification/merlin-passes/inventory/<candidate>/`. Changed-site counts, full semantic-gap attribution, upstream row-5 coverage and full compile runtime remain pending. They are not inferred from byte changes or op-census totals.

Environment: compiler `/home/jeremymills/projects/merlin/out/build/lean/mlir-venv/bin/python` (Python 3.12.3, torch_mlir 20260531.828); oracle `/home/jeremymills/projects/merlin/out/build/lean/torch-venv/bin/python` (PyTorch 2.10.0+cpu). TorchAO is absent from both; its capture version remains unresolved.
