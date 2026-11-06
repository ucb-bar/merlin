# OPU utilization on the one-core FireSim board

The configured unit has a 64 × 64 logical tile and a 256-bit datapath for 8-bit operands. Each
operand takes two beats; a full tile takes four issue cycles. Its **theoretical fully active datapath
peak** is therefore `64 × 64 / 4 = 1,024 MAC per target cycle`. Count a multiply-accumulate as one
MAC. This is a capacity bound, not a measured busy signal.

Three different quantities answer different questions:

| Quantity | Formula | Spectformer result |
|---|---|---:|
| Routed MAC share | valid MACs assigned to OPU / all contraction MACs | 98.9501% |
| Structural tile fill | valid routed MACs / padded OPU tile MAC slots | 75.5733% |
| Whole-inference effective OPU throughput | valid routed MACs / (`METRIC cycles` × 1,024) | 0.0508795% of peak |

The numerator is **valid model work**, `Σ b × m × n × k` over routed contractions, excluding padded
lanes. Spectformer has 57 routed contractions and 1,840,054,272 valid routed MACs; 19,523,584
contraction MACs remain on RVV. With edge tiles rounded to 64, the static padded slot count is
`Σ b × ceil(m/64) × ceil(n/64) × 64² × k = 2,434,793,472`. Thus 594,739,200 slots are padding.
These counts come from the exact OPU image's `opu_signatures.json` and `routing_report.json`.
The two-layer TinyLlama sections also route all 19 contractions in their current h1 builds:
9,865,003,008 valid MACs at 64 tokens and 19,797,114,880 at 128 tokens. Their shapes fill the
64 × 64 OPU tiles completely by this static calculation. The 64-token OPU image completed on the
one-core FireSim bitstream as job 3860 in 15,380,012,801 target `METRIC cycles`, with its full
2,048,000-element output hash matching the exact Spike-gated image. Its whole-inference effective
throughput is `9,865,003,008 / (15,380,012,801 × 1,024) = 0.000626384`, or **0.0626384% of
peak** (0.641417 valid MAC per target cycle). The paired plain-RVV control, job 3861, completed
in 27,955,258,588 target compute cycles on the same bitstream. The hardware W8A8 gate compared
the 4,096-value prefix for each leg; both full-output hashes and prefixes matched their exact
Spike-gated images. The graded OPU speedup is
`27,955,258,588 / 15,380,012,801 = 1.8176×`. The 128-token pair also passed on the same
one-core bitstream: device job 3862 took 16,727,634,195 target compute cycles and RVV control
job 3864 took 41,920,445,165, for **2.5061×** speedup. Both full 4,096,000-value hashes and
prefixes matched their exact Spike images; the hardware W8A8 gate compared each 4,096-value
prefix. Its 19,797,114,880 valid routed MACs give **1.18350 valid MAC per target cycle** and
`19,797,114,880 / (16,727,634,195 × 1,024) = 0.00115576`, or **0.115576%** of whole-inference
peak. The 100% static tile fill does **not** imply 100% active-cycle occupancy.

Static routing also cannot prove that the runtime shim used the OPU. At return from
`merlin_run_multi`, a Spike debugger read the exact shipped s64 ELF's OPU shim counters:
**143 calls, 0 scalar fallbacks**. The same check on the corrected prepacked s64 ELF found
143 calls and 0 fallbacks. The routed sidecars' `Σb` is 143 for both s64 images and 105 for
Spectformer, matching the dynamic call counts; there is no missing batch multiplier in the
valid-MAC sums. These counters establish the executed software path for those exact
images on Spike; they are not FPGA busy-cycle counters. The first prepack trial had an
unaligned embedded weight base and took the shim's scalar fallback even though its ELF still
contained OPU instructions. Merlin `6a6d81d46` aligns that blob in the Zephyr link. The corrected
prepack image matches the complete output hash and prefix digest and measures 598,027,952
Spike compute cycles versus 1,496,997,872 for the baseline. This is **functional-simulator
timing**; its FireSim timing is pending.

The accepted Spectformer h1 OPU ELF likewise recorded **105 calls, 0 scalar fallbacks** at the
same point. `dynamic_opu_counters/receipt.json` records each exact ELF's SHA-256 and the debugger
log's SHA-256, so these counts cannot be transferred silently to a different build.

The corrected profiled Spectformer image was built after the newer routing policy: it routes
90 call sites and 1,853,155,328 valid MACs, versus 57 sites and 1,840,054,272 MACs in the older
accepted unprofiled image. Its Spike profile therefore cannot explain the older FireSim job's
timing. A current unprofiled comparator built with the same settings has a byte-identical
`opu_signatures.json` and pre-instrumentation MLIR, and its exact Spike output hash and prefix
match the profiled image. It takes 1,608,481,484 Spike cycles versus 1,611,163,448 profiled
cycles, an observed 0.167% difference on Spike. The comparator's
`spectformer_current_unprofiled_h1/comparison_receipt.json` records both exact ELF digests.
Hardware profiling and a same-routing FireSim timing comparison are pending. The first full-profile
FireSim job 3865 printed the expected full output hash but was canceled after 176 of 5,172 profile
rows: at that UART rate it could not reach `DONE` before its 7,200-second limit. No hardware
call-window utilization is inferred from that partial dump. A new image prints only the 90 OPU
rows while retaining the full `prof_total_ticks`; its exact Spike gate passed with the same full
output hash and the same 17,063,688 OPU call cycles. The image's generated selector is included in
its build hash, and the report checks its selected-row count against the SHA-pinned profile table
and routing sidecar. Its FireSim run is pending. In the profiled
Spike image alone, 5,172 per-op rows cover 99.66% of compute cycles; OPU call intervals account
for 17,063,688 cycles (1.06% of the profiled total) and `linalg.generic` for 66.76% of
attributed cycles. These are simulator diagnostics, not hardware occupancy.
`python examples/lib/opu_profile_utilization.py
out/artifacts/probes/firesim_kodiak_opu/spectformer_profiled_h1` verifies the exact ELF,
full-output identity, every selected profile row, SHA-pinned table and routing sidecar, and OPU callee
multiset before reporting the two-factor
breakdown. On Spike, its OPU call windows perform 1,853,155,328 valid MACs in 17,063,688
cycles: **10.6057% of peak within those call windows** and 0.112324% across the whole
profiled inference. The call windows include packing, transfers, stalls, and kernel work; the
10.6057% is therefore effective throughput during calls, not array issue duty. For a FireSim
UART, pass `--console <uartlog> --hw-summary <HW_CFG_SUMMARY>` from the same job directory;
the tool requires the pinned bitstream artifact and matching output before producing a hardware
report.

For correctness, the exact shipped OPU Spike ELFs were stopped after computing their whole-output
hashes and their complete output buffers were extracted before the UART prefix dump. Each extracted
FNV hash matched its FireSim plan (and s64 job 3860), then Merlin's W8A8 gate was run with
`min_coverage=1` against the complete host reference. Both sections passed: s64 compared 2,048,000
values with maximum absolute error `1.43051e-6`, maximum significant-element relative error
`2.69542e-7`, and all 64 token argmaxes equal; s128 compared 4,096,000 values with the same
maximum absolute error, maximum significant-element relative error `2.80821e-7`, and all 128
argmaxes equal. The host W8A8 reference uses Merlin's own quantized math, so this checks device
execution agreement; it is not an independent proof of the quantization algorithm.

FireSim job 3858, the Spike-gated spectformer image with build hash `41758fde53331576`, completed
in 3,531,733,777 `METRIC cycles`. That metric is the target `mcycle` delta around the compute region
per inference, excluding FireSim host time; it includes model work, packing, per-inference
preparation, stalls, and other operations. `1,840,054,272 / (3,531,733,777 × 1,024) = 0.000508795`,
or **0.0508795%**. The
64³ GEMM device job 3854 completed 262,144 valid MACs in 792,242 target cycles, or **0.0323134%**
of peak across its full compute region; its tile fill is 100%.
The matched plain-RVV control, job 3859, completed in 6,120,729,118 target cycles on the same
bitstream. Both legs passed W8A8 and matched their own Spike output identities, giving an OPU
speedup of `6,120,729,118 / 3,531,733,777 = 1.7331×`.

These percentages are **end-to-end effective throughput**, not the MAC array's active-cycle
utilization. The current bitstream has no software-visible counter for active cycles or valid lanes.
Actual array occupancy needs an RTL counter or trace measuring busy cycles and useful lanes. Static
opcode counts in an ELF cannot supply it. A timed kernel window would narrow the denominator but
would still include loads, packing, and stalls. FireSim's ~15 MHz simulation rate measures host
emulation speed and does not enter any utilization formula.

For a new instrumented bitstream, accumulate `PopCount(io.op.macc)` continuously. This OPU
has eight rows of 128 int8 MAC cells. FireSim AutoCounter accumulates on every target cycle
but writes CSV samples only while its trigger is active, so the trigger must remain on across
flat, idle sample plateaus before and after compute. Subtract the cumulative row values at
those plateaus; reject missing or non-flat samples, wrap, or OPU work outside the measured
window. With the exact image's target `METRIC cycles` for that same compute region, issued
MAC slots are `128 × row-count delta`, array issue duty is `row-count delta / (8 × compute
cycles)`, and useful slot efficiency is the exact valid MAC count divided by issued slots.
Verify this readout with a known GEMM in metasim before a separate U250 bitstream build; the
currently pinned bitstream has no such counter and cannot yield these values retroactively.

With those counters, report the factors separately over the same inference:
`valid MACs / (1,024 × inference cycles) = (OPU-window cycles / inference cycles)
× (issued slots / (1,024 × OPU-window cycles)) × (valid MACs / issued slots)`.
They distinguish time spent outside OPU calls, array issue duty during those calls, and useful
work per issued slot. The product must reconcile to the existing end-to-end effective peak;
the current bitstream measures only that product. Do not combine these factors from different
images or bitstreams.

The spectformer image passed its W8A8 numerical gate; its complete output hash matched the exact
Spike-gated image. Its hash does not bit-match the older W8A8 golden, whose libm policy differs.
The pinned FireSim bitstream artifact matched the job's tar. The local Saturn source pin still
reports drift in the clock-gate config, so this measurement does not certify all source pins.

Evidence in this task checkout:

- `out/artifacts/probes/firesim_kodiak_opu/spectformer_merlin_compile_h1/firesim/results.json`
- `out/artifacts/probes/firesim_kodiak_opu/spectformer_merlin_compile_h1/routing/opu_signatures.json`
- `out/artifacts/probes/firesim_kodiak_opu/spectformer_merlin_compile_h1/routing/routing_report.json`
- `out/artifacts/probes/firesim_kodiak_opu/tinyllama_l2_s64_firesim_h1/run_1793936108853905576/firesim/device/uartlog`
- `out/artifacts/probes/firesim_kodiak_opu/tinyllama_l2_s64_firesim_h1/run_1793936108853905576/firesim/results.json`
- `out/artifacts/probes/firesim_kodiak_opu/tinyllama_l2_s64_firesim_h1/run_1793936108853905576/firesim/utilization.json`
- `out/artifacts/probes/firesim_kodiak_opu/tinyllama_l2_s64_firesim_h1/run_1793936108853905576/routing/opu_signatures.json`
- `out/artifacts/probes/firesim_kodiak_opu/tinyllama_l2_s128_firesim_h1/run_1793936116770167230/firesim/results.json`
- `out/artifacts/probes/firesim_kodiak_opu/tinyllama_l2_s128_firesim_h1/run_1793936116770167230/firesim/utilization.json`
- `out/artifacts/probes/firesim_kodiak_opu/tinyllama_l2_s128_firesim_h1/run_1793936116770167230/device/routing/opu_signatures.json`
- `out/artifacts/measurements/firesim_alveo_u250_firesim_kodiak_opu_1core/op_gemm_64x64x64/`
- `out/artifacts/probes/firesim_kodiak_opu/dynamic_opu_counters/` (exact-ELF Spike debugger reads)
- `out/artifacts/probes/firesim_kodiak_opu/spectformer_profiled_opu_sparse_h1/spike_gate.json`

For a graded immutable FireSim pair, `python examples/lib/opu_utilization.py <run_dir>` recomputes
the counts and ratios from the exact image's SHA-checked routing sidecar and the two hardware
`METRIC cycles` fields. It accepts only the named one-core bitstream with its pinned
`vLen=512, dLen=256, int8` geometry, and requires both legs verified. For example, passing the
s64 run directory above returns 9,865,003,008 valid MACs, 100% static fill, 0.0626384%
whole-inference effective peak, and 1.8176× control-over-device speedup. It deliberately returns
`null` for array-active occupancy. The older Spectformer run predates the immutable leg receipt,
so its arithmetic is shown explicitly above instead of being silently admitted by this tool.
The FireSim collector saves this calculation automatically as `firesim/utilization.json` and
embeds it in `firesim/results.json` for every fully verified immutable whole-model pair with
an exact OPU signatures sidecar on the pinned one-core bitstream. Older single-operator plans
without a SHA-checked sidecar retain their correctness grade and explicitly report utilization
unavailable. The console summary prints utilization alongside speedup.
