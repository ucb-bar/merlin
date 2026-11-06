# Kodiak OPU handoff

Use Merlin branch `feat/OPU-VME`. The Kodiak and Iris package updates are on their respective
`handwritten-implementation` branches. Merlin's setup fetches tested package commits, so the
package branch heads may contain newer documentation than the image-building pins.
For the **torchAO-source W8A8** SpectFormer and TinyLlama captures, use the
[portable capture and relocation instructions](TORCHAO_CAPTURE_DELIVERY.md) and the
[source quantization contract](TORCHAO_W8A8_SOURCE.md). The older archives and
FireSim figures below describe weight-only source captures followed by Merlin's
own W8A8 conversion; they remain exact-image performance diagnostics.
The `./run.sh firesim --models spectformer` command below selects the older
`spectformer_int8_full` capture. It does not build the torchAO-source W8A8
program. A source-authoritative OPU image and its matched Spike and FireSim
checks remain pending.
The [source archive packaging procedure](SOURCE_ARCHIVE_PACKAGING.md) records
the immutable receipt and archive hashes Kevin needs before relocation.
The [captured Torch CPU reduction diagnostic](TORCHAO_CPU_MEAN_FIDELITY.md)
explains an opt-in source-order fix for TinyLlama's first observed divergence;
whole-model framework parity still needs a new gate.

## Compile and check an image

```sh
git clone --branch feat/OPU-VME https://github.com/ucb-bar/merlin.git
cd merlin/examples/kodiak
./run.sh setup --llvm <llvm23-install> --chipyard <chipyard> --zephyr-sdk <zephyr-sdk-0.17.0>
./run.sh preflight --require package
./run.sh op gemm 64 64 64
./run.sh kernel-test opu_tile_loop --quick
```

`setup` prints any missing dependency or pin drift. The required `package` preflight exits
nonzero if its build prerequisites are missing; it must pass before using the output
as a hardware image. `op gemm` compares RVV, a scalar matrix twin, and the shipped OPU image on
its pinned Spike extension. To test a custom C kernel, register it under the Kodiak package's
`kernels/` and pass `--matrix-kernel <name>`; the kernel ABI and test flow are in that package's
README.

On a fresh checkout, the model capture is absent. Transfer the exact capture bundle from this
shared host, or provide the SpectFormer source and trained checkpoint before running
`./run.sh capture spectformer`; the loader otherwise falls back to random weights. The capture
used for the measurements loaded the trained checkpoint with zero missing or unexpected keys.
`./run.sh firesim --models spectformer --gate independent-rvv` then builds both one-core images, audits
OPU instructions, checks the exact images on Spike, and prints an immutable FireSim plan. Add
`--submit` to place the pair in the shared FPGA queue. The control leg is plain RVV on the **same
OPU bitstream**. `./run.sh firesim-collect <plan-dir> --job device=N --job control=M` grades the
returned full-output hashes, W8A8 numerical gate, bitstream identity, and target compute cycles.
For a verified whole-model OPU pair with an exact routing sidecar, it saves
`firesim/utilization.json` with routed MAC share, static tile fill, and paired speedup.
It reports dynamic valid-MAC rate and whole-inference effective peak only when a
same-image per-op FireSim profile proves each routed call executed once; array
occupancy remains explicitly unmeasured on the current bitstream.

The exact capture archives are local delivery artifacts on this shared host, under
`out/artifacts/delivery/` in this Merlin task. Copy each archive to Kevin's machine and check
its SHA-256 before extracting it into `out/artifacts/recaptures/` from the Merlin root:

| Capture archive | SHA-256 | Extracted directory |
| --- | --- | --- |
| `kodiak-spectformer-capture-v1.tar.gz` | `6f59f24bec263a5d0f8f9772005d02a57bf132c794b4771a232d88e00955f166` | `spectformer_int8_full/` |
| `kodiak-tinyllama-l2-s64-capture-v1.tar.gz` | `883ecfaa9295b54a3485e5f507471475f607aa355773927cef076c6f35a891c9` | `tinyllama_l2_s64_int8_full.portable/` |
| `kodiak-tinyllama-l2-s128-capture-v1.tar.gz` | `c3688a964a2eb2bdde3bcb6e71b057055a889221aab8189def28fb203bcc618c` | `tinyllama_l2_s128_int8_full.portable/` |

For Spectformer, run `mkdir -p out/artifacts/recaptures` then
`tar -xzf <copied-spectformer-archive> -C out/artifacts/recaptures`. For TinyLlama, extract
the chosen archive the same way and rename its `.portable` directory to the model name without
that suffix. Both TinyLlama archives were extracted and passed the materialized capture receipt
check after that rename. The relocated Spectformer and s64 TinyLlama captures also passed the
prepack weight rewrite, with the generated bundles pointing at their own weight files. These
receipts do not prove source closure. The archives are ignored local files, not Git downloads.
A capture's `scheme=int8_weight_only` describes its stored torchAO Linear weights; it is not
itself an end-to-end W8A8 verdict. Merlin's selected package uses
`dtype_strategy=int8_w8a8`, dynamically quantizes activations, and grades its images against
Merlin's host W8A8 golden. Spectformer's complete 1,000-value hardware output passes that
gate. The two-layer TinyLlama FireSim legs directly compare the first 4,096 UART values to
that golden; their complete output hashes match the exact Spike images, and the shipped OPU
Spike s64/s128 images pass complete Merlin W8A8 comparisons. No independent torchAO W8A8
parity check for these earlier hardware images was recorded. The old full 22-layer
weight-only-capture path disagrees with an independent torchAO W8A8 reference on its first
token, so that compiled result is not a validated full-model quantized inference.
The source-authoritative next gate is model2MLIR's separate `int8_w8a8` torchAO capture:
it exports dynamic activation quantization and integer `aten._int_mm` contractions. The
current `int8` bundles instead export weight-only float contractions and Merlin adds its own
W8A8 transformation afterward. Keep their performance measurements as measurements of
those exact images; do not present them as parity with the torchAO W8A8 program. A new
trained-checkpoint Spectformer torchAO capture has byte-identical inputs and weights to the
earlier capture, 41 explicit signed i8 × i8 → i32 matrix contractions covering 88.4299% of
its dense MACs, zero opaque ops, and a finite 1,000-value framework golden. The full-model
TinyLlama torchAO capture is described below. Exact compiler, Spike, and FireSim checks for
these source-exported programs remain pending.

### TorchAO quantization contract for OPU and VME

The [hardware quantization audit](TORCHAO_HW_QUANTIZATION.md) records the
receipt-bound activation arithmetic, i32 range bound, and logical tile fill
for both exact source captures.

For a current source host verdict, use the [receipted cache population and strict
replay procedure](../../docs/guides/vision_workloads_rvv_zephyr.md#a-source-quantized-torchao-capture).
The cache key covers Merlin lowering source, active pass settings, compiler and
Python package bytes; each loaded object has a verified construction manifest.
The earlier SpectFormer and TinyLlama caches predate this identity and cannot
serve as current host evidence. A cache-only miss is a failed preflight, not a
model output. Fresh population is deferred until the VME Vivado run releases
scratch and memory.
An [empty-cache probe](../../experiments/reference-data/spectformer_torchao_host_cache_preflight_2026-11-06.md)
against the exact SpectFormer source returned a structured
`HostCacheMiss` for all 1,361 kernels, zero loaded objects, and no compilation.
The task-local `.env` points at removed LLVM paths; export a live LLVM 23
`MERLIN_MLIR_INSTALL` before population and replay, with matching `MERLIN_CLANG` and
`MERLIN_MLIR_TRANSLATE` as shown in the guide.
The source FireSim planner now rechecks the strict host receipt, compiler
identity, each loaded object and its construction manifest before creating
either an acceptance or diagnostic plan. Keep that cache and the same tool
environment available through planning.

The source-authoritative path is torchAO `int8_w8a8`: dynamic signed-int8 activation
quantization, signed-int8 weights derived from the trained checkpoint, and explicit `_int_mm`
contractions with int32 accumulation. Preserve the captured scale, zero-point, rounding,
clamp, and dequantization arithmetic around each contraction. Do not apply Merlin's earlier
W8A8 rewrite to this already-quantized graph. The `int8` storage-format label alone does not
certify the source quantization recipe; the capture scheme and framework golden must travel with the
image. OPU BME and future VME Zvt use different instruction encodings, but both consume
the same integer contraction and must produce the same int32 result before source
dequantization. The current matrix shim passes no integer bias, so any float bias and
scale multiplication stay in the surrounding graph.

For the trained Spectformer source capture, static geometry projects all **41** integer
contractions to the OPU: **1,644,423,168 valid MACs / 2,164,260,864 padded logical tile
slots = 75.9808% projected tile fill**. The compact VME configuration (`VLEN=512`,
`DLEN=128`) projects **40** of them to Zvt; the 1 × 1000 × 256 head stays on the
scalar path under the current cost model. Its projected fill is **1,644,167,168 /
2,147,483,648 = 76.5625%**. These are static estimates, not routed-image evidence or
physical MAC-cell occupancy. Spectformer's integer K dimensions are multiples of 64,
so a separate K33 case must cover the Zvt tail. A signed `-128`/`127` endpoint case is
also required before declaring source quantization compatible with VME RTL. Strict
numerical agreement with the independent torchAO framework golden remains the compiler gate;
the existing Merlin-host discrepancy is under investigation and these projections are not
model correctness claims. The true-source Spectformer host run produced finite 1,000-value
output with the correct argmax, but failed its framework gate (relative L2 `0.00718247`,
cosine `0.99997751`). In a bounded pretrained TinyLlama true-source run, the first three
quantized q/k/v contractions had byte-identical signed operands and int32 accumulators to
independent eager torchAO. The first mismatch was one activation byte at the first attention
`o_proj` (torchAO `-128`, Merlin `-127`), at a negative `-127.5` rounding tie. Both
rounding instructions behave correctly; the prequant float values already differ.
Neither model is source-parity certified yet.
The [trained and holdout activation-code probe](../../experiments/reference-data/spectformer_torchao_activation_code_probe_2026-11-06.md)
checks Spectformer block-0 fc1 at the exact boundary: with identical norm2
input, Merlin and TorchAO have bitwise-equal reciprocal scales, signed i8
activation bytes, i32 matrix output, and f32 fc1 output. Earlier float drift
flips 12 or 16 codes at the source `-127.5` rounding tie. Changing the
quantizer would alter the TorchAO program rather than fix this capture.
The [Torch 2.16 AVX2 LayerNorm reproducer](../../experiments/reference-data/torch216_avx2_ln256_probe_2026-11-06.md)
matches native width-256 means, reciprocal standard deviations, and affine outputs
bitwise on trained, holdout, and independent random inputs. It isolates one
source-arithmetic cause of the drift; the captured LayerNorm is already a
multi-operation region, so the provenance-gated compiler rewrite and unchanged
full-model framework gate are still pending.
The first TinyLlama discrepancy is now localized further: its embedding and first
RMSNorm squared values match eager torchAO exactly, but Merlin's serial float32
mean reduction differs from Torch CPU's reduction tree. This changes the normalization
and ultimately crosses the attention output-projection quantization tie. A reduction
tree fitted only to those eight rows failed on new random rows and was rejected.
The later provenance-gated AVX2 K2048 reduction matched Torch CPU's first
RMSNorm mean bitwise on all eight source rows and 200 independent holdout rows.
The complete TinyLlama host framework gate has not been rerun after that fix.
Keep two verdicts separate: exact target-arithmetic image versus scalar twin, and
framework fidelity versus independent torchAO. The current strict framework gate
still fails. The full 22-layer eight-token source graph has 155 integer Linear roots
and 8,275,361,792 valid integer MACs; its M=8 rows fill only **12.5% of 64-row
logical tiles** before any runtime effects. The OPU and compact VME cost models
project all 155 roots as candidates, but no emitted route or physical-cell duty is
measured for this source image.

A fresh-host check verified the Spectformer capture files and reached compilation, but that host
lacked the Zephyr/Spike toolchain, so full lowering from a transferred capture remains unverified.
On the current shared host, the installed workload is `merlin-kodiak` and the queue client needs
the working-directory launchers. Before `--submit`, set:

```sh
export MERLIN_FIRESIM_WORKLOAD=merlin-kodiak
export MERLIN_FIRESIM_CLIENT_DROP_ENV=HOME,USER,LOGNAME
export MERLIN_FIRESIM_CLIENT_PATH_PREFIX="$(git rev-parse --show-toplevel)/merlin/experiments/gemmini_perf_bench/scripts/firesim_queue_cwd_launcher"
```

On another host, use its installed workload or install the default `merlin-zephyr` manifest as
the printed plan instructs. The plan names the exact workload and boot binary before submission.
The queue can take hours. See [the full example guide](README.md) for capture, build, package,
and board options.

## Verified scope

| Workload | Exact one-core OPU FireSim result | Matched RVV control | Output check |
| --- | ---: | ---: | --- |
| Spectformer int8 | 3,531,733,777 target compute cycles | 6,120,729,118 cycles | W8A8 pass; full hash/prefix match its Spike image |
| TinyLlama two-layer, 64 tokens | 15,380,012,801 cycles | 27,955,258,588 cycles | Full 2,048,000-value hash/prefix match Spike; 4,096-prefix hardware W8A8 pass |
| TinyLlama two-layer, 128 tokens | 16,727,634,195 cycles | 41,920,445,165 cycles | Both full 4,096,000-value hashes/prefixes match Spike; 4,096-prefix hardware W8A8 pass |

The earlier all-NaN Spectformer FireSim job 3856 used a structurally profiled legacy image.
Its exact OPU ELF **and** RVV control ELF both reproduced the same all-NaN hash on Spike;
same-source unprofiled builds were finite. Structural profiling reprinted an f32 `-inf` dense
literal as a finite decimal, changing model semantics. Merlin commit `98f47a3a3` preserves its
raw bits, and the default FireSim flow builds unprofiled images and Spike-gates each exact ELF.
The verified Spectformer pairs above have finite outputs. The old NaN is therefore an image
generation defect, not evidence of an OPU arithmetic failure on the pinned FireSim image.

An aligned-weight prepack s64 OPU image completed as FireSim job 3866 in **1,101,993,218**
target cycles with the same full 2,048,000-value hash as the verified baseline. Its matched RVV
control completed as job 3867 in 13,689,726,593 target cycles. Both images matched their
exact Spike full hashes and prefixes, and the pair graded VERIFIED: **12.4227× OPU speedup**.
The device sidecar records 9,865,003,008 valid routed MACs, yielding a **conditional one-hit
estimate of 0.874215% of the 1,024-MAC/cycle theoretical peak across the inference**;
the earlier OPU image's corresponding estimate was 0.0626384%.

The current-head fill-elided s64 pair also graded VERIFIED: device job 3874 took 1,064,676,843
cycles, RVV control job 3877 took 13,689,726,593, and both full hashes/prefixes matched Spike.
This is **12.8581× speedup** and **0.904856% (conditional one-hit estimate) whole-inference effective peak** for the same
9,865,003,008 valid routed MACs. The array-active fraction is not measurable on this bitstream.
An isolated quant-round fusion trial on top of that schedule also graded VERIFIED on the same
bitstream: OPU job 3876 took **1,022,361,106** target cycles, matched RVV control 3887 took
**13,646,684,998**, and both full 2,048,000-value hashes and prefixes matched their exact
Spike images. Its 4,096-value hardware W8A8 prefix passed. The pair gives **13.3482×** speedup;
9,865,003,008 valid routed MACs yield **0.942308% (conditional one-hit estimate) whole-inference effective peak**. This is
3.9745% fewer device cycles than fill-only job 3874. The
[versioned opt-in quant package](../../out/artifacts/targets/rvv/kodiak_prepack_quant_w8a8_v1/README.md)
adds only `fuse_quantize_round_convert` to the prepack schedule. Its rebuilt s64, s128, and
Spectformer images pass exact OPU and independent-RVV Spike full-hash gates; s128 also passes a
complete 4,096,000-logit W8A8 and 128-token argmax check. The package's rebuilt Spectformer
pair 3889/3890 graded VERIFIED: **2,578,301,427** OPU versus **5,667,528,436** RVV target cycles,
**2.19816×** paired speedup, full 1,000-value hash and W8A8 pass on both legs. Its 1,853,155,328
valid routed MACs give **0.0701905% (conditional one-hit estimate) whole-inference effective peak**; static tile fill is 73.8223%
and array occupancy remains unmeasured. Compared with the prepack pair on the same capture,
routing, and bitstream, OPU cycles fell 5.6915%. Compiler source revisions differ, so this
observed package comparison does not isolate a single knob. The SHA-bound tracked receipt is
`out/artifacts/targets/rvv/kodiak_prepack_quant_w8a8_v1/spectformer_hardware_evidence.json`.
A separate [same-source round-even A/B](SPECTFORMER_ROUNDINTRINSIC_AB.md) on the same
one-core bitstream graded all four exact images VERIFIED. Baseline OPU/control jobs
3895/3896 took 2,733,902,458/5,840,140,166 cycles; intrinsic jobs 3897/3898 took
2,539,617,917/5,660,152,596. The intrinsic saved **7.10649% device cycles** and
raised the conditional one-hit valid-MAC whole-inference estimate from **0.06619556% to 0.07125962%**.
Both images routed the same 1,853,155,328 valid MACs; call-window cycles and physical
array occupancy were not measured. These are older weight-only-source images followed
by Merlin W8A8, not torchAO-source parity evidence.
The rebuilt s128 pair 3891/3892 also graded VERIFIED: **2,252,651,428** OPU versus
**27,453,912,924** RVV target compute cycles, or **12.1874×** speedup. Both full
4,096,000-value hardware hashes match the exact Spike images; direct hardware W8A8 comparison
covers the 4,096-value UART prefix. The separately gated exact OPU Spike image passed all
4,096,000 W8A8 logits and 128 argmaxes. Its 19,797,114,880 valid routed MACs, divided by
1,024 MACs per cycle and the OPU target compute cycles, give **0.858238% (conditional one-hit estimate) whole-inference
effective peak**. Static tile fill is 100%; array occupancy remains unmeasured. The
[SHA-bound evidence](../../out/artifacts/targets/rvv/kodiak_prepack_quant_w8a8_v1/tinyllama_l2_s128_hardware_evidence.json)
records both legs and the comparison scope. The older s64 hardware result is bound to the same
knob/schedule bytes but a different package manifest. The **rebuilt package's own** s64
pair 3893/3894 is now VERIFIED on the pinned one-core bitstream: **1,022,361,106** OPU versus
**13,646,684,998** RVV target compute cycles, or **13.3482×** speedup. Both full
2,048,000-value hardware hashes match exact Spike; direct hardware W8A8 comparison covers
the 4,096-value UART prefix. All 19 contractions and 9,865,003,008 valid MACs route to OPU
with 100% static tile fill. Those valid MACs divided by 1,024 MACs/cycle and measured OPU
cycles give **0.942308% (conditional one-hit estimate) whole-inference effective peak**; array occupancy is unmeasured.
Both cycle counts equal the earlier trial pair 3876/3887, confirming the named package's
observed timing without assuming ELF byte identity. The
[SHA-bound rebuilt-pair evidence](../../out/artifacts/targets/rvv/kodiak_prepack_quant_w8a8_v1/tinyllama_l2_s64_hardware_evidence.json)
records exact plans, captures, bitstream, and numerical scope.
The s128 aligned-prepack OPU/RVV pair also graded VERIFIED on the pinned one-core bitstream:
device job 3871 took 2,438,693,540 target cycles, matched control job 3875 took
27,742,439,123, and both full 4,096,000-value hashes and prefixes matched their exact Spike
images. The hardware W8A8 gate passed on the 4,096-value prefixes. The **11.37594×** paired
speedup accompanies 19,797,114,880 valid routed MACs, 100% static tile fill, and **0.792765% (conditional one-hit estimate)
whole-inference effective peak** for the OPU image. Physical array occupancy remains unmeasured.
The SHA-bound receipt is
`out/artifacts/probes/firesim_kodiak_opu/tinyllama_l2_s128_firesim_h1/run_1793954503610973263/firesim/pair_receipt.json`.

To use the measured aligned-prepack schedule, pass
`--rvv-package rvv/kodiak_prepack_w8a8_v1` to `./run.sh firesim`, `build`, or `package`. The
[versioned package](../../out/artifacts/targets/rvv/kodiak_prepack_w8a8_v1/README.md) records
the exact s64 FireSim pair and preserves the older K1-certified schedule as the default. The
current-head Spectformer h1/h2 delivery has eight images: all four plain images passed W8A8
gates, and all debug twins passed ELF audits, with 90 OPU routes. The two-layer TinyLlama
chipyard delivery labels its unrun OPU ELFs `UNGATED`. Neither package
claims physical Kodiak execution or K1 certification for the new schedule.
On this shared host, the rebuilt Spectformer h1/h2 chipyard delivery is staged at
`out/artifacts/delivery/kodiak-spectformer-prepack-v1-final-r8.zip` (SHA-256
`0e6195e031634eb556ac096c8641b2b0ed1a5e4f08a3f485e664562dec80bcfc`). This revision
refreshes the embedded README's pinned rig guide link and warns that the suggested Arduino `t`
command cannot pulse the current firmware's reset pins, and that the rig wrapper's PASS label
does not check a nonzero loader exit. It preserves r5's corrected source
provenance note: the four ISA/geometry files used for its source digest match committed Saturn
`e16a301d`. All 22 non-README archive members, including every ELF, match r5 byte for byte.
The task-local revision receipt is `.tmp/kodiak-spectformer-prepack-v1-final-r8.receipt.json`. The zip
passed archive integrity and contains eight model ELFs plus a VLEN probe, its manifest, audits,
and grading script.
Its four plain images passed W8A8 gates and h2 matches h1. It is an ignored local artifact,
not a Git-tracked download or a physical-chip execution receipt. Kevin must copy this zip from
the shared host to retain the exact delivered ELF bytes. The manifest observes Saturn
`e16a301d`, which differs from its declared `ea373800` pin; the recorded ISA source digest is
recoverable from [four tracked files at `e16a301d`](SPECTFORMER_R5_SOURCE.md). Kevin can build
a fresh package from the tracked prepack schedule and transferred capture, but must record its
Chipyard, Zephyr, and toolchain pins and rerun Spike, ELF, and hardware gates. A byte-identical
rebuild of these ELFs remains unproven. The exact prepack FireSim device/control jobs
3886/3888 subsequently graded VERIFIED:
**2,733,902,458** versus **5,840,140,166** target compute cycles, **2.13619×** paired speedup,
and identical full 1,000-value hashes to their exact Spike images. Its 90 routed contractions
account for 1,853,155,328 valid MACs, 73.8223% static tile fill, and **0.0661956% (conditional one-hit estimate) whole-inference
effective peak** on the OPU image. Array occupancy remains unmeasured. The SHA-bound pair receipt
is `out/artifacts/probes/firesim_kodiak_opu/spectformer_firesim_h1/run_1793965368611649945/firesim/pair_receipt.json`.

A current-routing unprofiled Spectformer OPU comparator completed as FireSim job 3870 in
**3,038,273,721 target compute cycles** with the same full output hash as its exact Spike image.
Its 90 routed contractions account for 1,853,155,328 valid MACs, yielding **0.0595642% (conditional one-hit estimate)
whole-inference effective peak**; the older accepted OPU image used a different routing policy.
The SHA-bound receipt is in
`out/artifacts/probes/firesim_kodiak_opu/spectformer_current_unprofiled_h1/firesim/`.

The full TinyLlama model's weights alone are about 1.2 GiB and do not fit **physical Kodiak's
512 MiB DRAM**. The pinned one-core OPU FireSim target has 16 GiB of simulated DRAM and a
separate external-weight layout, so its capacity alone does not rule out the full model. A
full 22-layer, 8-token capture with `ids` entry has produced matched one-core shipped OPU and
scalar-twin Zephyr ELFs (~1.30 GB each; 1,336,420,368-byte LOAD memory span). The shipped ELF
completed on the pinned OPU Spike model with `DONE`, 256,000 output elements and whole-output
FNV1a32 `3198730293`, but failed the Merlin host W8A8 gate on its 4,096-value UART prefix
(cosine 0.938137, relative L2 0.551485); its complete hash also differs from the Merlin host
W8A8 and independent torchAO references. This is a numerical failure, not a build or simulator
crash; no full-model FireSim job has been submitted. The capture's source closure remains
unverified and its independent torchAO W8A8 reference also disagrees with the Merlin host
golden on token 0. The matched plain RVV full-model ELF completed on Spike with the **same
full hash and prefix bytes** as the OPU ELF, and failed the same host numerical gate. This
localizes the observed discrepancy to a shared compiled/reference path, not an OPU-only
arithmetic error. Their 612,415,434 versus 4,871,333,825 Spike cycles are diagnostic and
must not be used as a certified speedup. A separate full 22-layer pretrained torchAO
`int8_w8a8` capture now has 155 explicit signed i8 × i8 → i32 Linear contractions, 45 float
attention batch matmuls, zero opaque ops, and a complete ABI. Its framework golden is
byte-identical to an independent eager torchAO run; source closure is still unverified, and
Merlin host execution of this explicit integer graph has a 3–5% per-token relative-L2
disagreement under investigation. The full model remains a required separate compiler gate.
The source-only [eight-request TorchAO experiment](../../experiments/reference-data/tinyllama_torchao_batch_utilization_plan_2026-11-06.md)
ran all 22 layers on one seeded B=8, S=8 input: all 155 eager integer calls
had M=64 and its full 8×8×32,000 logits were bitwise equal to eight separate
B=1 calls. This is a path toward filling the existing 64-row logical tile;
the B8 MLIR capture, route, exact Spike image, FireSim rate, and physical
array duty have not been measured. For a future source TinyLlama image, the
[full-output certificate tool](source_output_certificate.py) can verify the
complete exact Spike output against independent TorchAO `golden.npy` using a
debugger dump from the same run. It has synthetic adversarial tests; no
true-source ELF certificate exists yet, and FireSim acceptance remains gated
separately.
The two-layer sections above are the measured hardware targets. Their 64-token and 128-token
exact OPU Spike images also passed complete W8A8 comparisons and all token argmaxes against
their host reference. The paired FireSim OPU and RVV images are built with one vector hart, matching the
one-core bitstream.

For utilization, use [the formula and evidence](UTILIZATION.md) and
`python examples/lib/opu_utilization.py <verified-plan-dir>`. The current bitstream has 1,024
physical int8 MAC cells but no active-cycle counter. The baseline ~0.05–0.12% figures and the
fill-elided prepacked s64 device's 0.904856% (conditional one-hit estimate) are
**historical static one-hit throughput proxies as a fraction of peak**. They are not
dynamic utilization or array occupancy without same-image call-hit evidence. Earlier
Spectformer and 64-token TinyLlama Spike images executed 105 and 143 OPU shim calls respectively,
with zero scalar fallbacks. The baseline 128-token pair gives **2.5061×** control-over-device
speedup and **0.115576% (conditional one-hit estimate)** whole-inference effective peak for its OPU image; the bitstream still
has no array-active counter.
For a future exact-image dynamic rate, add `--profile` to
`python examples/lib/firesim_flow.py legs ...`. That opt-in route instruments
the shipped OPU ELF and its scalar gate twin, requires the complete Spike
profile, then checks each routed call's hit count and cycle window in the
same-image FireSim UART. It reports valid-MAC throughput over the OPU call
window and full inference. The profiled ELF has extra instructions, so its
result does not inherit an unprofiled pair's speedup; run a separate unprofiled
pair for that comparison. Neither result measures physical PE duty without a
counter-equipped bitstream.

The Kodiak OPU target uses legacy BME instructions. It cannot execute Iris's Zvt/VME encodings;
those need a separate VME-enabled Kodiak RTL configuration and bitstream. On the separate Iris
VME configuration, three direct SerialTL RTL cases (K33, biased edge, batch-3) now pass pinned
Spike with exact digests and zero output and guard errors; evidence and limits are in Iris commit
`e13f7cc` on `handwritten-implementation`. The larger 44-case direct RTL run
timed out after 36 cases passed against pinned Spike; it stopped at `deep_k`,
and guarded split replays remain pending.
Kodiak's `handwritten-implementation` package now declares a separate compact
VME board and `saturn_vme` contract, with exact compiler route identity and
source pins. It shares the signed i8×i8→i32 Zvt emitter with Iris and retains
the OPU route separately. The [TorchAO delivery instructions](TORCHAO_CAPTURE_DELIVERY.md)
give the source-preserving compile-only VME command. A compact signed K33 VME
object compiled with 71 Zvt instructions (20 integer matrix multiplies), zero
in its scalar twin, but linked Spike and model gates remain pending.
The [Kodiak Zvt/VME configuration plan](ZVT_VME_PLAN.md) lists the exact pins and gates needed
before a Kodiak board may advertise Zvt support. A distinct managed Kodiak VME one-core
configuration now passes FIRRTL elaboration, and a fresh VLEN512 biased K-tail ELF passes
the pinned Zvt Spike. The exact ELF also passed direct Kodiak VME Verilator/SerialTL: the
35 × 63 × 33 biased case produced the same digest `0x15d98143c9a17f64`, zero output and guard
errors, and `KT DONE`. The local SHA-bound receipt is in Chipyard managed task
`kodiak-zvt-vme-firechip/.tmp/kodiak_vme_k33_bias_elf/rtl_receipt.json`.
A VME FireSim bitstream and broader Kodiak RTL suite remain pending; these results do not
change the physical OPU ASIC.
The same exact ELF also passed bounded FireSim metasim on the generated VME RTL: zero
output and guard errors, digest `0x15d98143c9a17f64`, `KT DONE`, and exit zero. Its kernel interval
was 2,761 target cycles in this simulator. The SHA-bound receipt is in Chipyard managed task
`kodiak-vme-portability-2/.tmp/vme_k33_metasim/receipt.json`.
The reproducible source is on the [Chipyard fork branch](https://github.com/copparihollmann/chipyard/tree/feat/kodiak-zvt-vme)
at `0c0fcd39` and [Saturn branch](https://github.com/miles-rusch-berkeley/saturn-vectors/tree/feat/kodiak-zvt-vme)
at `4359c36`. Chipyard pins a public Rocket base and carries
the exact CSR guard patch plus application instructions in `patches/kodiak-vme/README.md`;
a fresh managed checkout with that patch passed JDK 17 Firechip compilation and FireSim FIRRTL,
and reproduced the tested device tree. The branch head also fixes an old TraceRV host
`libdwarf` API call; Golden Gate and Verilator metasim build passed without changing generated
RTL. A second SHA-checked script in that patch directory applies the validated FireSim
worktree build fix at the pinned Iris FireSim base. The metasim receipt remains bound to
Chipyard source commit `5311b459`; the newer branch commit adds only build packaging.
These are source and simulator gates, not a VME FPGA image.
The first VLEN512/DLEN256 VME U250 build passed synthesis and `opt_design` but failed placement
DRC `UTLZ-1`: 2,208,803 slice LUTs were required and 1,728,000 are available. The build ended
rc 1 without a bitstream; the bounded log and rc are in Chipyard managed task
`kodiak-vme-portability-2/.tmp/`. A separate VLEN512/DLEN128 compact VME candidate passed
exact-pin direct RTL and FireSim Verilator metasim on its biased K-tail control. Its U250
post-synthesis report uses **1,267,125 / 1,728,000 CLB LUTs (73.33%)** with zero synthesis
errors; the [SHA-bound receipt](https://github.com/copparihollmann/chipyard/blob/1cc1690a608cc995b6c4b896f9e19c8ee4cef556/patches/kodiak-vme/evidence/compact-u250-synthesis.json)
records source pins and the report hash. Placement subsequently completed with
**1,335,383 / 1,728,000 CLB LUTs (77.28%)** and zero errors or critical warnings;
the placed report SHA-256 is
`f13243399ef0a6c725d1de463b4e8c62809e444e9b1cdaa185a891c0f77f32a9`.
The build entered global routing on 2026-11-06, with negative intermediate
timing. Final routing, timing, bitstream generation, and FPGA execution remain
pending. No VME FPGA bitstream or model result is claimed yet.
The current OPU FireSim bitstream is usable for the BME flow above. Its archived artifact identity
was checked during the hardware jobs. Reproducing that archive from a clean source tree remains
unverified because the live Chipyard/Saturn checkout has pin drift. Physical Kodiak rig access
and scheduling are separate from FireSim. For physical operations, follow the newer
[Kodiak RUNNING.md at `f2ebd82`](https://github.com/ucb-bar/kodiak-mlir/blob/f2ebd82302deb6b7a40ba118649df0bfcd393b55/RUNNING.md),
which corrects the historical health check, wrapper behavior, and Arduino reset command. Merlin
setup pins that same package commit; its bundled delivery loader note uses the same proven
OPU-direct control.
Coordinate with the rig owner before any chip load or reset.

On this garden host as of 2026-11-06, `ssh -o BatchMode=yes kodiak-rig hostname` returns
`bwrckodiak` without a laptop relay. Kevin needs owner-provided access on any other host.
In an owner-confirmed exclusive window, a SHA-verified JTAG program restored FPGA
Ethernet and UDP ping. The OPU-direct benchmark at the historically passing path then failed at
the chip's `0x1000` BootAddrReg readback under the historical uncredited `--no-reset` loader path.
An additional bounded retry requested chip 0 after FPGA ping and received a UDP response, but
the current ELF at that historical path still read `None` at `0x1000` before execution. Its
SHA-bound receipt is
`out/artifacts/probes/chipyard_kodiak/physical_probe_20261106/select_chip_diagnostic_20261106/receipt.json`;
a portable receipt is in Kodiak `evidence/bootaddr_select_chip_20261106/` at commit `6b2285f`.
No Merlin image was loaded; existing tmux sessions were preserved. The current rig ELF, loader,
and FPGA bitstream hashes and the August/current section-map comparison are in
`out/artifacts/probes/chipyard_kodiak/physical_probe_20261106/receipt_addendum.json`;
no August ELF hash survives to prove byte identity. An
[offline replay of the SHA-verified loader](https://github.com/ucb-bar/kodiak-mlir/blob/f2ebd82302deb6b7a40ba118649df0bfcd393b55/evidence/bootaddr_host_replay_20261106/receipt.json)
shows that it treats any UDP packet as a write acknowledgment and shortens its nominal two-second
read wait to one second after a read ACK. A valid synthetic response at 1.5 seconds then yields
`None`. The physical logs contain no raw packet timestamps, so this host bug is a plausible
explanation, not proof that the chip replied. Chip power/reset/clock/wiring or FPGA bridge faults
remain possible. A timestamped host packet trace is needed before assigning the failure.
The rig's tracked Arduino firmware accepts only `all`, `fpga`, `0`, and `1`; the suggested
`arduino_all.sh t` is a no-op for its reset pins. Its shell wrapper can return exit zero with no
serial reply. The earlier reset log is empty and does not record its argument, so a confirmed
Arduino chip-0 reset remains untested. [The read-only audit](https://github.com/ucb-bar/kodiak-mlir/blob/f2ebd82302deb6b7a40ba118649df0bfcd393b55/evidence/arduino_reset_command_audit_20261106/receipt.json)
and the pinned rig guide give the exact next exclusive-window sequence: capture the BootAddr
datagrams first, then test a separately acknowledged chip-0 reset if needed. No Arduino or UDP
command was sent in that audit. The rig login cannot currently use `tcpdump` or `dumpcap`, so
raw late-packet evidence requires owner-enabled capture or an instrumented receive path.
The rig's current `execute_program.sh` can print PASS for a nonzero loader exit before its
timeout; [the SHA-bound read-only audit](https://github.com/ucb-bar/kodiak-mlir/blob/f2ebd82302deb6b7a40ba118649df0bfcd393b55/evidence/rig_host_wrapper_audit_20261106/receipt.json)
requires grading the actual console, exit status, and model output.
The pinned [bounded late-reply probe](https://github.com/ucb-bar/kodiak-mlir/blob/f2ebd82302deb6b7a40ba118649df0bfcd393b55/evidence/bootaddr_late_rx_probe_20261106/README.md)
is ready for a separate exclusive-window diagnostic if packet capture remains unavailable;
its offline pass is not a physical-chip verdict.
The [offline host-loader patch candidate](https://github.com/ucb-bar/kodiak-mlir/blob/f2ebd82302deb6b7a40ba118649df0bfcd393b55/evidence/bootaddr_host_patch_candidate_20261106/README.md)
keeps the read deadline at 2 seconds and validates write ACKs. Its SHA-bound
scripted replay passes, but the rig's tracked loader is unchanged and the
physical BootAddrReg failure remains unexplained without packet timing.
The guide now records the two different same-named bitstream files, the exact path used in
both programming logs, and further FPGA-side status probes. Those probes, the physical
chip-select switch, and a serial-confirmed Arduino `0` reset remain untested. The current
rig still has two attached tmux sessions, so another physical test requires a fresh
exclusive-use confirmation; no test or reset was performed during the latest audit.
The Vivado version also differs. Direct Kodiak testing remains gated on a successful historical
control readback; the Spike-gated small GEMM OPU/RVV images are ready for that retry.

A sparse Spectformer OPU profile image passed its exact Spike gate with all 90 routed calls,
matching full output hash, and **10.6057% effective peak during OPU call windows on Spike**.
Its first FireSim submission 3868 failed before boot due to a queue SSH environment error;
the same exact image completed as job 3872 with all 90 selected profile rows, full output hash,
and `DONE`. On FireSim the OPU call windows took **76,045,023 target cycles**, giving **2.37980%
effective peak within calls** and 0.0589822% across this profiled inference. The exact
same-routing unprofiled FireSim image is 0.98673% faster. Call windows include packing and
stalls; they do not measure physical MAC array occupancy. Kodiak's functional Spike extension
does not model OPU timing, so its 10.6057% call-window rate is not a hardware latency forecast.
A separate draft Chipyard counter patch is
available as [PR 2395](https://github.com/ucb-bar/chipyard/pull/2395), but no instrumented
bitstream or FPGA counter measurement exists yet. A managed exact-pin counter source now passes
Firechip compile, target FIRRTL elaboration, Golden Gate, and a Verilator metasim build with
JDK 17 and Espresso. Its bracketed 64³ GEMM metasim run also passed: full expected hash,
`DONE`, exit zero, 791,141 target compute cycles, active-row delta 2,048, gate-low delta zero,
and flat samples before and after the burst. This gives **0.0323583% whole-compute row issue
duty** and **100% useful issued-slot efficiency** for that one fully filled GEMM. The SHA-bound
receipt is in Chipyard managed task
`opu-counter-goldengate-pin/.tmp/metasim_matrix_hashonly/utilization_receipt.json`.
U250 synthesis and actual model/FPGA counter measurements remain pending.

The aligned-prepack TinyLlama s64 image also has an exact Spike-gated sparse profile: 19 OPU
call rows, 26,595,355 call cycles, and **36.2236% effective peak during those calls on Spike**.
Its full hash and routing sidecar match the FireSim device image. The profile image's earlier
debug memory probe had written into embedded weights; Merlin now probes only beyond linker
`_end`, and the corrected image passes the complete output gate. Its exact FireSim retry,
job 3873, measured **109,580,333 OPU call-window cycles** for 9,865,003,008 valid routed
MACs, or **8.79153% effective peak within calls**. Those calls take 9.76384% of the profiled
inference; whole-inference effective peak is 0.858391%. This profile used the aligned prepack
image before fill elision. Its full hash, all 19 rows, and bitstream SHA passed the hardware
verifier; physical array occupancy is still unmeasured.
On those exact profiled images, the OPU call windows occupy 2.47845% of Spectformer and
9.76384% of TinyLlama target cycles. Even zero-cycle OPU calls would limit end-to-end improvement
to 1.02541× and 1.10820× respectively if other work stayed fixed. The next large gains require
reducing the remaining model work as well as improving OPU calls; these bounds do not transfer
to the separate fill-elided image.
