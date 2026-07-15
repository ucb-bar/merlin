# capsule_bench_v0 — report (Experiment ABI v0.1)

> Generated from recorded `capsule_result.json` + decoded `instruction_trace.json` by `results/gemmini/gen_capsule_bench_report.py` — not from scrollback.

> Artifact under test: `generated_targets/gemmini/agent_spec_v1_mlir_oot/` · runs root: `runs/capsule_bench_v1`

**Public/dev: 20/20 pass** · **Hidden: 5/5 pass**

## Capsule pass table (tiers: L0 golden · L1 ref==sim · L2 spike · L3 verilator-RTL)

| capsule | kind | label | status | L0 | L1 | trace | numeric | L2(cyc) | L3(cyc) | L4 | L5 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A0_config_smoke | isa | public | pass | pass | pass | pass | pass | pass(47) | pass(308) | — | — |
| A1_mvin_mvout | isa | public | pass | pass | pass | pass | pass | pass(27) | pass(143) | — | — |
| A2_single_tile_matmul | isa | public | pass | pass | pass | pass | pass | pass(47) | pass(308) | — | — |
| A3_k_accumulation | isa | public | pass | pass | pass | pass | pass | pass(58) | pass(396) | — | — |
| A4_acc_scale_i8 | isa | public | pass | pass | pass | pass | pass | pass(51) | pass(250) | — | — |
| A5_relu_epilogue | isa | public | pass | pass | pass | pass | pass | pass(47) | pass(308) | — | — |
| A6_resident_reuse | isa | public | pass | pass | pass | pass | pass | pass(56) | pass(428) | — | — |
| A7_edge_padding | isa | public | pass | pass | pass | pass | pass | pass(68) | pass(589) | — | — |
| B0_quantized_linear_i8 | layer | public | pass | pass | pass | pass | pass | pass(62) | pass(358) | — | — |
| B1_linear_relu_i8 | layer | public | pass | pass | pass | pass | pass | pass(58) | pass(396) | — | — |
| B2_linear_acc_scale_relu_i8 | layer | public | pass | pass | pass | pass | pass | pass(62) | pass(358) | — | — |
| B3_conv2d_im2col_i8 | layer | public | pass | pass | pass | pass | pass | pass(94) | pass(1011) | — | — |
| B4_conv2d_relu_i8 | layer | public | pass | pass | pass | pass | pass | pass(94) | pass(1011) | — | — |
| C0_mlp_linear1 | model_slice | public | pass | pass | pass | pass | pass | pass(175) | pass(1888) | — | — |
| C1_mlp_activation_linear2 | model_slice | public | pass | pass | pass | pass | pass | pass(175) | pass(1888) | — | — |
| C2_attention_q_projection | model_slice | public | pass | pass | pass | pass | pass | pass(72) | pass(587) | — | — |
| C3_attention_k_projection | model_slice | public | pass | pass | pass | pass | pass | pass(72) | pass(587) | — | — |
| C4_attention_v_projection | model_slice | public | pass | pass | pass | pass | pass | pass(72) | pass(587) | — | — |
| C5_attention_qk_matmul | model_slice | public | pass | pass | pass | pass | pass | pass(47) | pass(308) | — | — |
| C6_attention_pv_matmul | model_slice | public | pass | pass | pass | pass | pass | pass(47) | pass(308) | — | — |
| H0_matmul_hidden | isa | hidden | pass | pass | pass | pass | pass | pass(47) | pass(308) | — | — |
| H1_acc_scale_hidden | isa | hidden | pass | pass | pass | pass | pass | pass(51) | pass(250) | — | — |
| H2_k_accum_hidden | isa | hidden | pass | pass | pass | pass | pass | pass(58) | pass(396) | — | — |
| H3_movement_hidden | isa | hidden | pass | pass | pass | pass | pass | pass(27) | pass(143) | — | — |
| H4_conv_hidden | layer | hidden | pass | pass | pass | pass | pass | pass(94) | pass(1011) | — | — |

## Oracle-tier table

| tier | oracle | derived_from_rtl | capsules passing |
|---|---|---|---|
| L0 | numpy/Tensor golden | no | 25 |
| L1 | cb reference == simulate | no | 25 |
| L2 | spike (functional) | no | 25 |
| L3 | verilator (RTL, cycle-accurate) | yes | 25 |
| L4 | VCS (RTL) | yes | 0 |
| L5 | FireSim (FPGA) | yes | 0 |

## VCS / FireSim availability (honest)

- **L4 VCS — unavailable (attempted).** The in-environment Gemmini VCS sim (`simv-chipyard.harness-RadianceGemminiOnlyConfig`) **segfaults** on the bare-metal ELF that L2 (spike) and L3 (verilator) validate three-way bit-exact. This is a VCS/config incompatibility in this environment, not a backend defect; recorded `unavailable` (`OracleUnavailable`), never a fabricated pass.
- **L5 FireSim — unavailable.** `FIRESIM_ROOT` + queue daemon are present, but no verified bare-metal Gemmini replay hook exists in this environment; the adapter builds the ELF and records `unavailable` (returning `retry: True` when the shared FPGA queue is busy so the bundle can be re-scheduled rather than blocking).
- coverage counters: VCS-unavailable=0, FireSim-unavailable=0 (L4/L5 are optional tiers for these capsules, so they are not counted as mandatory-incomplete).
- _not-run is not pass: a **mandatory** tier recorded unavailable ⇒ capsule incomplete._

## Cycle table (DIAGNOSTIC ONLY — never gates pass/fail)

| capsule | L2 spike cyc | L3 verilator cyc |
|---|---|---|
| A0_config_smoke | 47 | 308 |
| A1_mvin_mvout | 27 | 143 |
| A2_single_tile_matmul | 47 | 308 |
| A3_k_accumulation | 58 | 396 |
| A4_acc_scale_i8 | 51 | 250 |
| A5_relu_epilogue | 47 | 308 |
| A6_resident_reuse | 56 | 428 |
| A7_edge_padding | 68 | 589 |
| B0_quantized_linear_i8 | 62 | 358 |
| B1_linear_relu_i8 | 58 | 396 |
| B2_linear_acc_scale_relu_i8 | 62 | 358 |
| B3_conv2d_im2col_i8 | 94 | 1011 |
| B4_conv2d_relu_i8 | 94 | 1011 |
| C0_mlp_linear1 | 175 | 1888 |
| C1_mlp_activation_linear2 | 175 | 1888 |
| C2_attention_q_projection | 72 | 587 |
| C3_attention_k_projection | 72 | 587 |
| C4_attention_v_projection | 72 | 587 |
| C5_attention_qk_matmul | 47 | 308 |
| C6_attention_pv_matmul | 47 | 308 |
| H0_matmul_hidden | 47 | 308 |
| H1_acc_scale_hidden | 51 | 250 |
| H2_k_accum_hidden | 58 | 396 |
| H3_movement_hidden | 27 | 143 |
| H4_conv_hidden | 94 | 1011 |

## Failure-plane summary

- none — all recorded capsules pass

## Integrity status

- `integrity_exempt: false`; package invoked only via its 4 CLI entrypoints; integrity scan clean (no harness/reference imports).
- Device code is MLIR-lowered RoCC only; no C compute kernels, no copied/called bareMetalC, no high-level Gemmini kernel calls.
