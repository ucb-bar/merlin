# Kodiak OPU handoff

Use Merlin branch `feat/OPU-VME`. The Kodiak and Iris package updates are on their respective
`handwritten-implementation` branches. Merlin's setup fetches tested package commits, so the
package branch heads may contain newer documentation than the image-building pins.

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
`firesim/utilization.json` with routed MAC share, static tile fill, whole-inference effective
peak, and an explicit unmeasured array occupancy field.

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

An aligned-weight prepack s64 OPU image completed as FireSim job 3866 in **1,101,993,218**
target cycles with the same full 2,048,000-value hash as the verified baseline. Its matched RVV
control completed as job 3867 in 13,689,726,593 target cycles. Both images matched their
exact Spike full hashes and prefixes, and the pair graded VERIFIED: **12.4227× OPU speedup**.
The device performs 9,865,003,008 valid routed MACs, or **0.874215% of the 1,024-MAC/cycle
theoretical peak across the inference**; the earlier OPU image measured 0.0626384%.

The current-head fill-elided s64 pair also graded VERIFIED: device job 3874 took 1,064,676,843
cycles, RVV control job 3877 took 13,689,726,593, and both full hashes/prefixes matched Spike.
This is **12.8581× speedup** and **0.904856% whole-inference effective peak** for the same
9,865,003,008 valid routed MACs. The array-active fraction is not measurable on this bitstream.
An isolated quant-round fusion trial on top of that schedule also graded VERIFIED on the same
bitstream: OPU job 3876 took **1,022,361,106** target cycles, matched RVV control 3887 took
**13,646,684,998**, and both full 2,048,000-value hashes and prefixes matched their exact
Spike images. Its 4,096-value hardware W8A8 prefix passed. The pair gives **13.3482×** speedup;
9,865,003,008 valid routed MACs yield **0.942308% whole-inference effective peak**. This is
3.9745% fewer device cycles than fill-only job 3874. The
[versioned opt-in quant package](../../out/artifacts/targets/rvv/kodiak_prepack_quant_w8a8_v1/README.md)
adds only `fuse_quantize_round_convert` to the prepack schedule. Its rebuilt s64, s128, and
Spectformer images pass exact OPU and independent-RVV Spike full-hash gates; s128 also passes a
complete 4,096,000-logit W8A8 and 128-token argmax check. The package's rebuilt Spectformer
pair 3889/3890 graded VERIFIED: **2,578,301,427** OPU versus **5,667,528,436** RVV target cycles,
**2.19816×** paired speedup, full 1,000-value hash and W8A8 pass on both legs. Its 1,853,155,328
valid routed MACs give **0.0701905% whole-inference effective peak**; static tile fill is 73.8223%
and array occupancy remains unmeasured. Compared with the prepack pair on the same capture,
routing, and bitstream, OPU cycles fell 5.6915%. Compiler source revisions differ, so this
observed package comparison does not isolate a single knob. The SHA-bound tracked receipt is
`out/artifacts/targets/rvv/kodiak_prepack_quant_w8a8_v1/spectformer_hardware_evidence.json`.
The s128 pair 3891/3892 is still awaiting final grading. The older
s64 hardware result is bound to the same knob/schedule bytes but a different package manifest;
the rebuilt s64 images have not run on FireSim. Array occupancy is still unmeasured.
The s128 aligned-prepack OPU/RVV pair also graded VERIFIED on the pinned one-core bitstream:
device job 3871 took 2,438,693,540 target cycles, matched control job 3875 took
27,742,439,123, and both full 4,096,000-value hashes and prefixes matched their exact Spike
images. The hardware W8A8 gate passed on the 4,096-value prefixes. The **11.37594×** paired
speedup accompanies 19,797,114,880 valid routed MACs, 100% static tile fill, and **0.792765%
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
`out/artifacts/delivery/kodiak-spectformer-prepack-v1-final-r3.zip` (SHA-256
`1f5a89962e5fd32720b5b62de6d2e57403bb49a8a8632cb9f114d1d5a5009523`). This revision
changes only the embedded README's physical loader instructions to the proven OPU-direct control
and gives standalone readers a pinned link to Kodiak's current rig procedure;
all 22 other archive members, including every ELF, match the previous zip byte for byte. The
task-local revision receipt is `.tmp/kodiak-spectformer-prepack-v1-final-r3.receipt.json`. The zip
passed archive integrity and contains eight ELFs, its manifest, audits, and grading script.
Its four plain images passed W8A8 gates and h2 matches h1. It is an ignored local artifact,
not a Git-tracked download or a physical-chip execution receipt. Kevin can copy that zip from
this shared host or regenerate it after transferring the capture using the tracked prepack
package. The exact prepack FireSim device/control jobs 3886/3888 subsequently graded VERIFIED:
**2,733,902,458** versus **5,840,140,166** target compute cycles, **2.13619×** paired speedup,
and identical full 1,000-value hashes to their exact Spike images. Its 90 routed contractions
account for 1,853,155,328 valid MACs, 73.8223% static tile fill, and **0.0661956% whole-inference
effective peak** on the OPU image. Array occupancy remains unmeasured. The SHA-bound pair receipt
is `out/artifacts/probes/firesim_kodiak_opu/spectformer_firesim_h1/run_1793965368611649945/firesim/pair_receipt.json`.

A current-routing unprofiled Spectformer OPU comparator completed as FireSim job 3870 in
**3,038,273,721 target compute cycles** with the same full output hash as its exact Spike image.
Its 90 routed contractions account for 1,853,155,328 valid MACs, yielding **0.0595642%
whole-inference effective peak**; the older accepted OPU image used a different routing policy.
The SHA-bound receipt is in
`out/artifacts/probes/firesim_kodiak_opu/spectformer_current_unprofiled_h1/firesim/`.

The full TinyLlama model's weights alone are about 1.2 GiB and do not fit this 512 MiB board.
The two-layer sections above are the hardware targets. The 64-token and 128-token exact OPU
Spike images also passed complete W8A8 comparisons and all token argmaxes against their host
reference. The paired FireSim OPU and RVV images are built with one vector hart, matching the
one-core bitstream.

For utilization, use [the formula and evidence](UTILIZATION.md) and
`python examples/lib/opu_utilization.py <verified-plan-dir>`. The current bitstream has 1,024
physical int8 MAC cells but no active-cycle counter. The baseline ~0.05–0.12% figures and the
fill-elided prepacked s64 device's 0.904856% are
**whole-inference effective throughput as a fraction of peak**, not array occupancy. Earlier
Spectformer and 64-token TinyLlama Spike images executed 105 and 143 OPU shim calls respectively,
with zero scalar fallbacks. The baseline 128-token pair gives **2.5061×** control-over-device
speedup and **0.115576%** whole-inference effective peak for its OPU image; the bitstream still
has no array-active counter.

The Kodiak OPU target uses legacy BME instructions. It cannot execute Iris's Zvt/VME encodings;
those need a separate VME-enabled Kodiak RTL configuration and bitstream. On the separate Iris
VME configuration, three direct SerialTL RTL cases (K33, biased edge, batch-3) now pass pinned
Spike with exact digests and zero output and guard errors; evidence and limits are in Iris commit
`e13f7cc` on `handwritten-implementation`. The larger 44-case RTL gate is still unverified.
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
at `0d911e2d` and [Saturn branch](https://github.com/miles-rusch-berkeley/saturn-vectors/tree/feat/kodiak-zvt-vme)
at `4359c36`. Chipyard pins a public Rocket base and carries
the exact CSR guard patch plus application instructions in `patches/kodiak-vme/README.md`;
a fresh managed checkout with that patch passed JDK 17 Firechip compilation and FireSim FIRRTL,
and reproduced the tested device tree. The branch head also fixes an old TraceRV host
`libdwarf` API call; Golden Gate and Verilator metasim build passed without changing generated
RTL. A second SHA-checked script in that patch directory applies the validated FireSim
worktree build fix at the pinned Iris FireSim base. The metasim receipt remains bound to
Chipyard source commit `5311b459`; the newer branch commit adds only build packaging.
These are source and simulator gates, not a VME FPGA image.
The current OPU FireSim bitstream is usable for the BME flow above. Its archived artifact identity
was checked during the hardware jobs. Reproducing that archive from a clean source tree remains
unverified because the live Chipyard/Saturn checkout has pin drift. Physical Kodiak rig access
and scheduling are separate from FireSim. For physical operations, follow the newer
[Kodiak RUNNING.md at `d7c75fc`](https://github.com/ucb-bar/kodiak-mlir/blob/d7c75fc/RUNNING.md),
which corrects the historical health check and wrapper behavior. Merlin setup pins that same
package commit; its bundled delivery loader note uses the same proven OPU-direct control.
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
no August ELF hash survives to prove byte identity. The loader treats any UDP response as a write
acknowledgment and makes one two-second readback attempt without a packet trace, so a host timing
or protocol fault remains possible alongside chip power/reset/clock/wiring or FPGA bridge faults.
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
