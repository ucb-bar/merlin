# TorchAO integration plan — numerical-contract candidates

> This is a **plan**, not a sweep. No quantization run is executed here, no speedup is claimed, and no accuracy number is asserted for any format that has not been measured. Only int8 (W8A8) has measured accuracy today (see `accuracy_gate_report.md`); every other format is `unavailable` until a gate is run.

## What this connects
The numerical-contract audit surfaces structural candidates (`resident_packed_lowbit_weights`, `native_lowbit_compute`, `fused_dequant_matmul`, `fused_requant_epilogue`). Each needs a real low-bit format to become concrete. TorchAO is the path to produce those formats and the metadata a compiler needs to keep them.

## Order to try (cheapest-signal first) — accuracy status
| format | TorchAO config | accuracy status | DSE candidate it informs |
|--------|----------------|-----------------|--------------------------|
| int8 W8A8 | `int8_dynamic_activation_int8_weight` | **measured: pass** (5/5, results.md) | `native_lowbit_compute`, `resident_packed_lowbit_weights` |
| fp8 W8A8 | `float8_dynamic_activation_float8_weight` | unavailable (not measured) | `native_lowbit_compute` |
| int4 weight-only | `int4_weight_only` | unavailable (not measured) | `resident_packed_lowbit_weights` |
| int4 weight + fp8 act | (composed) | unavailable (not measured) | `resident_packed_lowbit_weights` |

## What each format needs measured before it is DSE-legal
- **accuracy gate** vs fp32 on the task metric (cos/argmax, and trajectory error for action outputs) — the only measurable-now leg; int8 done, others pending.
- **packed-layout + scale metadata preserved** through capture — today the flat capture dequantizes to f32, so packed layout and scale/zero-point/group-size are erased (`scale_metadata: unavailable` in `numerical_contract.yaml`).
- **low-bit kernel cost** — requires the proposed design (target_measured), not claimed here.

## What compiler facts to preserve (so DSE can target the format)
- packed weight layout as a dispatch-crossing object; per-tensor/-channel/-group scale objects; dequant/requant placement; i32 (or fp32) accumulator width and epilogue commit.

## What is NOT claimed
No speedup, no cycle/area/energy, no accuracy for any `unavailable` format, and no broad dtype sweep is run by this tool. This plan only states which formats to measure, in what order, and which abstraction each would unblock.
