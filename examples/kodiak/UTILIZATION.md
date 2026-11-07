# OPU utilization on the one-core FireSim board

The configured unit has a 64 × 64 logical tile and a 256-bit datapath for 8-bit operands. Each
operand takes two beats; a full tile takes four issue cycles. Its **theoretical fully active datapath
peak** is therefore `64 × 64 / 4 = 1,024 MAC per target cycle`. Count a multiply-accumulate as one
MAC. This is a capacity bound, not a measured busy signal.

Three different quantities answer different questions:

| Quantity | Formula | Original Spectformer image |
|---|---|---:|
| Routed MAC share | valid MACs assigned to OPU / all contraction MACs | 98.9501% |
| Structural tile fill | valid routed MACs / padded OPU tile MAC slots | 75.5733% |
| Conditional one-hit effective OPU throughput | static routed MACs / (`METRIC cycles` × 1,024) | 0.0508795% of peak |

The numerator is **valid model work**, `Σ b × m × n × k` over routed contractions, excluding padded
lanes. Spectformer has 57 routed contractions and 1,840,054,272 valid routed MACs; 19,523,584
contraction MACs remain on RVV. With edge tiles rounded to 64, the static padded slot count is
`Σ b × ceil(m/64) × ceil(n/64) × 64² × k = 2,434,793,472`. Thus 594,739,200 slots are padding.
These counts come from the exact OPU image's `opu_signatures.json` and `routing_report.json`.
The older unprofiled FireSim percentages below divide that **static one-hit work sum** by
measured target cycles. Those ratios are historical one-hit proxies: the sidecar alone does
not prove how often each route executed. The current collector returns `null` for dynamic
valid-MAC rate and whole-inference effective peak until a SHA-pinned per-op table and the
same-image FireSim UART prove one timed inference with exactly one hit at each routed call
site. Static route share, static tile fill, target cycles, and paired speedup remain available.
The separately verified profiled FireSim images below do have that dynamic hit evidence.
For new plans, `python examples/lib/firesim_flow.py legs ... --profile` builds the **shipped**
OPU ELF and its scalar gate twin with the same per-op instrumentation, Spike-gates the exact
shipped ELF, and stages its SHA-pinned `op_profile_table.json` beside the ELF. The collector
uses that table and the same job's FireSim UART to prove dynamic work. Profiling changes the
device ELF and its cycle count, so the report withholds device/control speedup when the RVV
control is unprofiled. The profiled receipt also gives OPU call-window cycles and valid-MAC
effective peak within that window; it still cannot measure PE duty. Measure speedup with a
separate unprofiled pair.
The two-layer TinyLlama sections also route all 19 contractions in their current h1 builds:
9,865,003,008 valid MACs at 64 tokens and 19,797,114,880 at 128 tokens. Their shapes fill the
64 × 64 OPU tiles completely by this static calculation. The 64-token OPU image completed on the
one-core FireSim bitstream as job 3860 in 15,380,012,801 target `METRIC cycles`, with its full
2,048,000-element output hash matching the exact Spike-gated image. Its whole-inference effective
throughput is `9,865,003,008 / (15,380,012,801 × 1,024) = 0.000626384`, or **0.0626384% (conditional one-hit estimate) of
peak** (0.641417 conditional valid MAC per target cycle). The paired plain-RVV control, job 3861, completed
in 27,955,258,588 target compute cycles on the same bitstream. The hardware W8A8 gate compared
the 4,096-value prefix for each leg; both full-output hashes and prefixes matched their exact
Spike-gated images. The graded OPU speedup is
`27,955,258,588 / 15,380,012,801 = 1.8176×`. The 128-token pair also passed on the same
one-core bitstream: device job 3862 took 16,727,634,195 target compute cycles and RVV control
job 3864 took 41,920,445,165, for **2.5061×** speedup. Both full 4,096,000-value hashes and
prefixes matched their exact Spike images; the hardware W8A8 gate compared each 4,096-value
prefix. Its 19,797,114,880 static routed MACs give a conditional **1.18350 valid MAC per target cycle** and
`19,797,114,880 / (16,727,634,195 × 1,024) = 0.00115576`, or **0.115576% (conditional one-hit estimate)** of whole-inference
peak. The 100% static tile fill does **not** imply 100% active-cycle occupancy.

The aligned-prepack 128-token pair was independently graded on the same pinned one-core
bitstream: OPU job 3871 took 2,438,693,540 target compute cycles and its RVV control job
3875 took 27,742,439,123, giving **11.37594×** control/device speedup. Both full hashes and
prefixes matched their exact Spike images, and the hardware W8A8 prefix gate passed. Its
19,797,114,880 valid routed MACs across 19 contractions fill all static tiles and give
`19,797,114,880 / (2,438,693,540 × 1,024) = 0.0079276546`, or **0.792765% (conditional one-hit estimate) of
whole-inference effective peak** (conditional 8.11792 valid MAC per target cycle). The immutable pair's
`firesim/utilization.json` records array occupancy as unmeasured.

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
timing**. The aligned prepack OPU image subsequently completed on FireSim as job 3866 in
1,101,993,218 target cycles, with the same full output hash as the accepted baseline. Its
whole-inference effective throughput is `9,865,003,008 / (1,101,993,218 × 1,024) =
0.00874215`, or **0.874215% (conditional one-hit estimate) of peak**, about 13.96 times the baseline device's conditional rate. The
matched RVV prepack control job 3867 completed in 13,689,726,593 target cycles. Both legs
matched their exact Spike full hashes and prefixes and graded VERIFIED, yielding **12.4227×**
control-over-device speedup. The collector persisted the numerator, denominator, peak and
explicitly unmeasured array occupancy in the pair's `firesim/utilization.json`.

The current generated OPU kernel elides a fresh zero fill only when its direct call overwrites
every output. On the same aligned-prepack s64 model and pinned one-core bitstream, the
fill-elided device job 3874 completed in **1,064,676,843 target cycles**; its matched RVV
control job 3877 took 13,689,726,593. Both full 2,048,000-value hashes and prefixes match
their exact Spike images, and the pair graded VERIFIED at **12.8581×** control/device speedup.
The unchanged 9,865,003,008 valid routed MACs give
`9,865,003,008 / (1,064,676,843 × 1,024) = 0.00904856`, or **0.904856% (conditional one-hit estimate) whole-inference
effective peak**. This is 3.3863% fewer device
cycles than the prepack image before fill elision; physical array occupancy remains unmeasured.
The bitstream artifact SHA matched the pinned U250 image. Source-revision citation remains
limited by the local Saturn checkout's declared pin drift, as recorded in `firesim/results.json`.

An isolated quant-round fusion trial, using the same fill-elided prepack routing, has a matched
one-core FireSim pair: OPU job 3876 ran in 1,022,361,106 target cycles and RVV control 3887 in
13,646,684,998, for **13.3482×** paired speedup. Both full output hashes and prefixes matched
their exact Spike images, and the hardware W8A8 prefix passed. The unchanged 9,865,003,008
valid routed MACs give `9,865,003,008 / (1,022,361,106 × 1,024) = 0.00942308`, or
**0.942308% (conditional one-hit estimate) whole-inference effective peak** (conditional 9.64924 valid MAC/target cycle). This is 3.9745%
fewer OPU target cycles than the verified fill-only device; its matched control improved 0.3144%,
so the paired speedup grew 3.8116%. Array occupancy remains unmeasured. A versioned opt-in
package now carries the same knob and schedule bytes; its rebuilt images have separate identities
and exact Spike gates. The rebuilt Spectformer, s64, and s128 pairs all passed FireSim.

The rebuilt quant-package s64 pair independently graded VERIFIED: device job 3893 took
**1,022,361,106 target compute cycles** and matched plain-RVV control 3894 took
**13,646,684,998**, for **13.3482×** speedup. Both full 2,048,000-value hashes match their
exact Spike images; direct hardware W8A8 comparison covers the 4,096-value UART prefixes.
Its 19 OPU routes have **9,865,003,008 valid MACs** and 100% static tile fill, so
`9,865,003,008 / (1,022,361,106 × 1,024) = 0.0094230815`, or **0.942308% (conditional one-hit estimate)** whole-inference
effective peak (conditional 9.64924 valid MAC per target cycle). Array occupancy remains unmeasured.
Both target cycle counts equal the prior quant-round trial pair 3876/3887, which had the
same knob and schedule bytes but a different package manifest. The
[SHA-bound rebuilt-pair receipt](../../out/artifacts/targets/rvv/kodiak_prepack_quant_w8a8_v1/tinyllama_l2_s64_hardware_evidence.json)
states the exact comparison and reproducibility limits.

The rebuilt quant-package s128 image completed on the pinned one-core FireSim bitstream as
device job 3891 in **2,252,651,428 target compute cycles**. Its matched plain-RVV control,
job 3892, took **27,453,912,924 cycles**, yielding **12.1874×** speedup. Both full
4,096,000-value output hashes and the 4,096-value hardware UART prefixes matched their exact
Spike images; direct hardware W8A8 comparisons cover those prefixes. The exact OPU Spike image
separately passed full W8A8 and all 128 argmax checks. All 19 contractions route to OPU with
19,797,114,880 valid MACs and 100% static tile fill. Thus whole-inference effective peak is
`19,797,114,880 / (2,252,651,428 × 1,024) = 0.008582384`, or **0.858238% (conditional one-hit estimate)**
(conditional 8.78836 valid MAC per target cycle). The bitstream has no measured array occupancy counter.
The [SHA-bound receipt](../../out/artifacts/targets/rvv/kodiak_prepack_quant_w8a8_v1/tinyllama_l2_s128_hardware_evidence.json)
records the exact plans, captures, bitstream and numerical coverage. Relative to the older
s128 pair 3871/3875, device cycles fell 7.6288%; the schedules and source revisions differ,
so this comparison does not isolate quant fusion.

A separate sparse profiled image of that exact prepack lowering passed Spike with the same
full-output hash, prefix, prepared MLIR and routing sidecar. Its 19 selected OPU rows account
for 26,595,355 of 598,466,269 Spike target cycles (4.4439% of the inference). Valid routed
work divided by those OPU call cycles and 1,024 MAC/cycle is **36.2236% effective peak during
OPU calls**; the whole-inference fraction on that profiled Spike image is 1.60975%. These are
simulator rates and must not be combined with FireSim's cycle count to infer hardware issue duty.
The first profile attempt changed the output because fixed memory-diagnostic probes wrote inside
the 147 MiB embedded weights. The corrected image probes only beyond linker `_end` and passed
the exact output gate. The same exact sparse profile image completed on FireSim as job 3873 with
all 19 rows, full hash, and `DONE`. Its **109,580,333 OPU call-window target cycles** are
9.76384% of 1,122,307,701 whole-inference target cycles. The 9,865,003,008 valid routed
MACs therefore give **8.79153% effective peak during hardware OPU call windows** and
0.858391% whole-inference effective peak for this profiled image. The matching unprofiled
prepack image, job 3866, took 20,314,483 fewer cycles (1.8434% profiling overhead). This
profile precedes the separate fill-elision optimization, and its call-window rate must not be
silently assigned to job 3874. Spike's OPU extension does not model timing or stalls, so its
36.2236% diagnostic rate is not a hardware latency forecast. The pinned bitstream and complete
output identity are SHA-checked in `hardware_profile_utilization.json`; actual array issue duty
remains unmeasured.

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
The same-routing unprofiled OPU comparator completed as FireSim job 3870 with the expected full
hash in **3,038,273,721 target compute cycles**. Its SHA-bound receipt reports 1,853,155,328
valid routed MACs (99.6546% of contraction MACs), 73.8223% static tile fill, and **0.0595642% (conditional one-hit estimate)
whole-inference effective peak** (conditional 0.609937 valid MAC per target cycle). This is a 13.9722%
cycle reduction relative to the older accepted OPU Spectformer image, which used a different
routing policy. Array occupancy remains unmeasured.

The same-routing aligned-prepack Spectformer pair then graded VERIFIED on the same bitstream:
OPU job 3886 took **2,733,902,458** target compute cycles and plain-RVV control job 3888 took
**5,840,140,166**, for **2.13619×** paired speedup. Both full 1,000-value output hashes and
prefixes matched their exact Spike images, and the W8A8 gate passed. Its 90 routed contractions
perform 1,853,155,328 valid MACs, so whole-inference effective peak is
`1,853,155,328 / (2,733,902,458 × 1,024) = 0.000661956`, or **0.0661956% (conditional one-hit estimate)**. Static tile fill is
73.8223%, and array occupancy is still unmeasured. The device takes 10.0179% fewer target
cycles than same-routing comparator job 3870. Its SHA-bound receipt is
`out/artifacts/probes/firesim_kodiak_opu/spectformer_firesim_h1/run_1793965368611649945/firesim/pair_receipt.json`.

The opt-in quant fusion Spectformer pair also graded VERIFIED on that one-core bitstream:
OPU job 3889 took **2,578,301,427** target compute cycles and RVV control 3890 took
**5,667,528,436**, giving **2.19816×** paired speedup. Both full 1,000-value output hashes
matched exact Spike and both hardware W8A8 gates covered every value. With the unchanged
1,853,155,328 valid routed MACs, whole-inference effective peak is
`1,853,155,328 / (2,578,301,427 × 1,024) = 0.000701905`, or **0.0701905% (conditional one-hit estimate)**. Static fill remains
73.8223%; physical array occupancy remains unmeasured. OPU cycles are 5.6915% below the
prepack pair, but compiler source revisions differ, so the observed A/B does not isolate the
fusion knob alone. The tracked SHA-bound evidence is
`out/artifacts/targets/rvv/kodiak_prepack_quant_w8a8_v1/spectformer_hardware_evidence.json`.

The first full-profile FireSim job 3865 printed the expected full output hash but was canceled after 176 of 5,172 profile
rows: at that UART rate it could not reach `DONE` before its 7,200-second limit. No hardware
call-window utilization is inferred from that partial dump. A new image prints only the 90 OPU
rows while retaining the full `prof_total_ticks`; its exact Spike gate passed with the same full
output hash and the same 17,063,688 OPU call cycles. The image's generated selector is included in
its build hash, and the report checks its selected-row count against the SHA-pinned profile table
and routing sidecar. The sparse FireSim image completed as job 3872 with all 90 selected rows,
the expected full output hash, and `DONE`. Its SHA-bound hardware report gives **76,045,023 OPU
call-window cycles** out of 3,068,253,185 target compute cycles (2.47845% of inference time).
Those calls execute 1,853,155,328 valid MACs, or **2.37980% effective peak within the hardware
call windows**. The profiled image's whole-inference effective peak is 0.0589822%; the exact
same-routing unprofiled image took 29,979,464 fewer cycles, a 0.98673% instrumentation
overhead. The Kodiak Spike extension explicitly does not model OPU timing or stalls
(`kodiak-mlir/spike/opu_ext.cc`); its 17,063,688 call cycles are functional-simulator diagnostics,
not a hardware latency prediction. The FireSim call windows include packing, transfers, stalls,
and kernel work, and do not measure physical array occupancy.

The call-window collector requires exactly one dynamic hit for each routed
site in its one-inference image. A repeated site would otherwise make its
static routing MAC sum undercount dynamic work; the collector now rejects that
case until per-hit MAC accounting is implemented.

These same-image call windows also bound the benefit of optimizing only the OPU call path.
Even if every recorded OPU call took zero cycles and all other work stayed fixed, the profiled
TinyLlama s64 image could improve by at most `1 / (1 - 0.097638404247) = 1.10820×`, and the
profiled current-routing Spectformer image by at most
`1 / (1 - 0.024784468039) = 1.02541×`. These are Amdahl bounds for the two exact profiled
images, not measured speedups or bounds for the separate fill-elided or prepack delivery
images. They show why reducing work outside the OPU calls is the larger end-to-end opportunity.

In the profiled Spike image alone, 5,172 per-op rows cover 99.66% of compute cycles; OPU call intervals account
for 17,063,688 cycles (1.06% of the profiled total) and `linalg.generic` for 66.76% of
attributed cycles. These are simulator diagnostics, not hardware occupancy.
Twelve GELU rows using scalar `math.erf` account for 414,558,626 of 1,611,163,447 total
profiled Spike compute cycles (25.7304%). An isolated exact-image memoization probe preserved
the full hash but found only 970 cache hits among 2,408,448 erf calls (0.0403%). Its lookup alone
cost eight target cycles per call; even before cache-store cost it needed about 4.82% hits to
break even. Memoization is rejected, and this Spike share is not a FireSim speedup forecast.
This isolated probe is not part of the shipped package evidence.
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
or **0.0508795% (conditional one-hit estimate)**. The
64³ GEMM device job 3854 completed 262,144 valid MACs in 792,242 target cycles, or **0.0323134% (conditional one-hit estimate)**
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

That known-GEMM metasim gate now passes with an instrumented source graph. A hash-only 64³
OPU image preserved the exact model archive, weights, and timed compute code of the accepted
GEMM; it removed only the post-hash 4,096-value UART dump. Pinned OPU Spike returned the same
full hash and compute cycles as the original source image. On bounded Verilator FireSim metasim,
the new image reported full FNV hash `2089162557` over 4,096 values, `DONE`, exit zero, and
**791,141 target compute cycles**. AutoCounter had 1,330 flat pre-event samples, then an
active-row delta of **2,048**, then 651 flat post-event samples; gate-low delta was zero. Thus
`128 × 2,048 = 262,144` issued MAC slots, exactly the 262,144 valid MACs of the 64³ GEMM.
The measured **whole-compute row issue duty** is `2,048 / (8 × 791,141) = 0.0323583%`,
**useful issued-slot efficiency is 100%**, and their product is **0.0323583% useful peak MAC
utilization**. The metasim receipt is not published with this package, so this
result is a method validation rather than portable FPGA evidence. A
counter-equipped U250 bitstream and model run remain pending.

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

For a graded immutable FireSim pair, `python examples/lib/opu_utilization.py <run_dir>` recomputes
static work and fill from the exact image's SHA-checked routing sidecar, and speedup from the
two hardware `METRIC cycles` fields. It accepts only the named one-core bitstream with its
pinned `vLen=512, dLen=256, int8` geometry and a verified pair. On the old s64 run directory,
it returns 9,865,003,008 **static one-hit** routed MACs, 100% static fill, and 1.8176× paired
speedup; dynamic valid-MAC rate, whole-inference effective peak, and array occupancy are
`null`. A same-image per-op FireSim profile with a SHA-pinned table, complete output identity,
one timed inference, and exactly one hit at each routed site enables the dynamic rates. The
older Spectformer run predates the immutable leg receipt, so its historical arithmetic is
shown explicitly above instead of being silently admitted by this tool.
The FireSim collector saves this calculation automatically as `firesim/utilization.json` and
embeds it in `firesim/results.json` for every fully verified immutable whole-model pair with
an exact OPU signatures sidecar on the pinned one-core bitstream. Older single-operator plans
without a SHA-checked sidecar retain their correctness grade and explicitly report utilization
unavailable. The console summary prints utilization alongside speedup.

## Dynamic single-operator calibration on the pinned OPU FireSim image

The profiled 64 × 64 × 64 signed-int8 image in device job 3900 and its independent
RVV control in job 3901 both completed and matched their exact Spike outputs for
all 4,096 values. The SHA-pinned per-op table and same-image FireSim UART prove
**one hit** at the routed site. The numerator is therefore the actual inference's
`64 × 64 × 64 = 262,144` valid MACs, not a guessed static call count. The device
reported 33,087 target cycles inside the OPU call window and 17,687,828 target
cycles for the whole instrumented inference. At the configured capacity bound of
1,024 MACs per cycle:

| Scope | Calculation | Effective fraction of peak |
| --- | --- | ---: |
| Routed call window | `262144 / (33087 × 1024)` | 0.773718% |
| Whole instrumented inference | `262144 / (17687828 × 1024)` | 0.00144732% |

The static logical tile fill and routed MAC share are both 100% for this shape.
Neither value is the physical PE duty cycle. The bitstream lacks valid-lane and
MAC-active counters, so PE occupancy remains **unmeasured**. The profiled device
and unprofiled control have different instrumentation; their cycle ratio is not
a speedup measurement. The [portable SHA-bound summary](evidence/profiled_opu_64cube_firesim_3900_3901.json)
records exact ELFs, bitstream tar, UARTs, result hashes, and the compiler-source
citation caveat. This is a legacy operator calibration; TorchAO-source
SpectFormer and TinyLlama still require separate parity and image gates.

## Current-pin FireSim qualification (2026-11-07)

The [portable receipt](evidence/current_pin_firesim_20261107.json) identifies the
Merlin, RVV, and Kodiak commits, both ELF hashes for each pair, queue jobs, the
one-core U250 bitstream archive SHA, and the HWDB entry SHA. Both legs of the
64³ GEMM and Spectformer pairs completed on FireSim and matched their exact
Spike outputs. The Spectformer images passed the numerical W8A8 gate over all
1,000 outputs. Their full output hash matched Spike's shipped image and twin;
the bitwise host W8A8 golden hash differs under a recorded libm versus portable
math policy mismatch. That policy difference is not independently established
as the sole cause of the hash difference.

| Same-instrumentation pair | OPU target cycles | RVV target cycles | RVV / OPU |
| --- | ---: | ---: | ---: |
| 64³ GEMM, jobs 3904/3905 | 716,259 | 909,119 | 1.26926× |
| Unprofiled Spectformer, jobs 3908/3907 | 3,043,722,303 | 6,117,924,967 | 2.01001× |

The GEMM's 262,144 valid MACs and 100% tile fill are *static one-hit* work and
geometry calculations; that unprofiled image has no same-image dynamic MAC
rate. The separate profiled Spectformer device job 3906 proves exactly one
execution of each of its 90 routed calls. Its 1,853,155,328 valid routed MACs
span 76,211,859 measured OPU call-window target cycles, giving **24.31584 useful
MACs per call-window cycle**, or **2.37459% of the configured 1,024-MAC/cycle
capacity**. Those windows are 2.48865% of its 3,062,371,730-cycle inference.
The share of *static eligible contraction MACs* assigned to OPU is 99.65462%;
the hit-weighted, shape-derived padded tile fill is 73.82233%. Neither number
measures physical PE duty. Job 3906 is profiled and its RVV control 3907 is not,
so their cycle ratio is deliberately excluded from the speedup table.

The [same-image call breakdown](evidence/current_pin_firesim_20261107.json)
shows why tile fill alone does not explain throughput. The 32 spectral calls
perform only 12,845,056 valid MACs (0.69% of routed work) but take 17,170,126
of the 76,211,859 OPU call-window cycles (22.5%, rounded). Across all calls,
geometric padding is 1.355× slots per valid MAC, while measured call-window
time is 42.1× the ideal valid-MAC time at the configured peak. The current
bitstream has no active-cell or valid-lane counters, so packing, transfer, and
hardware stalls cannot be apportioned. Even eliminating all recorded OPU call
cycles would improve this exact profiled inference by at most 1.02552×; most
remaining end-to-end time lies outside those calls.

These are functional results on an identified bitstream archive. The archive
is not declared in Merlin's artifact registry, and the local Saturn source
pin was used under an explicit off-pin allowance. They do not establish
source-to-bitstream reproduction or independent TorchAO-source model parity.

The [final-pin TinyLlama L2 s64 Spike receipt](evidence/current_pin_tinyllama_spike_20261107.json)
covers the shipped OPU ELF, its same-lowering scalar twin, and a plain-RVV
image. The shipped and twin images produced the same FNV-1a hash over all
2,048,000 logits and the same SHA-256 over the first 4,096 values; those first
4,096 values also match the host W8A8 golden bitwise. The full host W8A8 hash
differs, and no full-element comparison of this final ELF to the host golden
was made. All 19 eligible contractions are statically routed to OPU, with
9,865,003,008 static one-hit valid MACs and 100% geometric tile fill. This is
a functional Spike result on `chipyard_kodiak`. The separate true TorchAO
capture remains outside this legacy W8A8 qualification.

The [current-pin TinyLlama FireSim receipt](evidence/current_pin_tinyllama_firesim_20261107.json)
records two independently verified one-core pairs. All four board-specific
ELFs completed on the pinned U250 bitstream, matched their exact Spike full
2,048,000-logit hash and 4,096-value prefix, and passed the hardware W8A8
prefix gate. Both OPU images prove one execution of each of their 19 routed
calls, so the 9,865,003,008 static valid MACs are also the observed
one-inference routed work. The profiled OPU images and unprofiled RVV controls
have different instrumentation; their cycle ratios are **not** speedups.

| Profiled OPU image | OPU target cycles | RVV control cycles | OPU call-window cycles | Useful peak fraction in calls |
| --- | ---: | ---: | ---: | ---: |
| Baseline, jobs 3909/3910 | 15,373,604,548 | 27,951,639,405 | 107,120,367 | 8.9934% |
| Prepacked, jobs 3911/3912 | 1,078,308,198 | 13,689,726,593 | 109,676,962 | 8.7838% |

The prepacked device uses the same profiler and board and produces the same
output. It eliminates 15 runtime i8 weight transposes moving 153,616,384
bytes per inference. Its profiled OPU cycle count is **14.257× lower** than
the baseline's, while OPU call-window time changes little. Cycles outside
those calls fall from 15,266,484,181 to 968,631,236. The prepacked image
also reduces memory need from 353,455,172 to 235,485,760 bytes, fitting the
default 256 MiB region. Whole-inference useful work reaches 0.8934% of
configured peak in that profiled image; this includes all non-OPU work and
must not be read as physical array occupancy. The pinned bitstream has no
active-cell or valid-lane counter. Its SHA is known, but it is not declared
as a reproducible source-built artifact in Merlin's registry.
