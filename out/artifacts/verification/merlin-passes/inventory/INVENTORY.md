# Initial model inventory

Measured 2026-10-19T19:17:37.001663+00:00; source `ea43634ec6005c88610abd754f4e06f04b4793c4`.

Private full-model capture and checkpoint inputs were unavailable. These results are **public stand-ins** from retained model capsules, using real preparation with `int8_compute=True` followed by xDSL preprocessing. No device build, ELF, upstream tensor run, or paper qualification is claimed.

Recommendation: **SY_micro_model** for checker development: it fires W8A8 contraction, quant_ext lowering, scalar-square, scalarize-splat and DCE. M3 is smaller but does not fire W8A8. Final paper model selection remains open.

Both programs prepare successfully: M3 in 0.271 s and SY_micro_model in 0.340 s. Exact recorded stages, receipts and raw diagnostics remain under the shared inventory directory.

Audited stage/support/definedness attempts are **unmeasured**: initial attempts lacked lake on PATH; after fixing PATH, the existing audit cache lacks TypedModule.olean. The compiled checker also rejects the current extended-definedness command. Rerun after the coordinator completes lake build. Raw diagnostic support refusals are M3: linalg.reduce; SY_micro_model: quant_ext.dequantize_per_tensor. These diagnostics do not establish audited support verdicts.

| Candidate | Boundary | Changed | Typed/universal |
|---|---|---|---|
| M3_host_island_seam_gemmini | prepare-parse | True | unmeasured |
| M3_host_island_seam_gemmini | prepare-torchao-affine | False | unmeasured |
| M3_host_island_seam_gemmini | prepare-scalar-squares | False | unmeasured |
| M3_host_island_seam_gemmini | prepare-collapse-overrank-matmul | False | unmeasured |
| M3_host_island_seam_gemmini | prepare-propagate-quant-inner | False | unmeasured |
| M3_host_island_seam_gemmini | prepare-qinner-lift | False | unmeasured |
| M3_host_island_seam_gemmini | quant-contraction_int8 | False | unmeasured |
| M3_host_island_seam_gemmini | quant-conv_int8 | False | unmeasured |
| M3_host_island_seam_gemmini | prepare-lower-quant-ext | False | unmeasured |
| M3_host_island_seam_gemmini | prepare-bf16-matmul-f32acc | False | unmeasured |
| M3_host_island_seam_gemmini | prepare-fix-bool-sitofp | False | unmeasured |
| M3_host_island_seam_gemmini | prepare-fix-bool-fptosi | False | unmeasured |
| M3_host_island_seam_gemmini | prepare-scalarize-splat-inputs | True | unmeasured |
| M3_host_island_seam_gemmini | prepare-print | True | unmeasured |
| M3_host_island_seam_gemmini | xdsl-parse | True | unmeasured |
| M3_host_island_seam_gemmini | xdsl-pruned | False | unmeasured |
| M3_host_island_seam_gemmini | xdsl-quant-lowered | False | unmeasured |
| M3_host_island_seam_gemmini | xdsl-c-interface | True | unmeasured |
| SY_micro_model | prepare-parse | True | unmeasured |
| SY_micro_model | prepare-torchao-affine | False | unmeasured |
| SY_micro_model | prepare-scalar-squares | True | unmeasured |
| SY_micro_model | prepare-collapse-overrank-matmul | False | unmeasured |
| SY_micro_model | prepare-propagate-quant-inner | False | unmeasured |
| SY_micro_model | prepare-qinner-lift | False | unmeasured |
| SY_micro_model | quant-contraction_int8 | True | unmeasured |
| SY_micro_model | quant-conv_int8 | False | unmeasured |
| SY_micro_model | prepare-lower-quant-ext | True | unmeasured |
| SY_micro_model | prepare-bf16-matmul-f32acc | False | unmeasured |
| SY_micro_model | prepare-fix-bool-sitofp | False | unmeasured |
| SY_micro_model | prepare-fix-bool-fptosi | False | unmeasured |
| SY_micro_model | prepare-scalarize-splat-inputs | True | unmeasured |
| SY_micro_model | prepare-print | True | unmeasured |
| SY_micro_model | xdsl-parse | True | unmeasured |
| SY_micro_model | xdsl-pruned | True | unmeasured |
| SY_micro_model | xdsl-quant-lowered | False | unmeasured |
| SY_micro_model | xdsl-c-interface | True | unmeasured |

Per-boundary before-op censuses, timing, binding hashes and exact failure reasons are in inventory.json. Changed-site counts, upstream row-5 coverage, complete semantic gaps and checker runtime remain pending; they are not inferred from byte changes or op-census totals.

Environment: Python 3.12.3; compiler torch_mlir 20260531.828; isolated PyTorch 2.10.0+cpu. TorchAO is absent from both discovered environments, so its capture version is unresolved. See environment.json for interpreter paths.
