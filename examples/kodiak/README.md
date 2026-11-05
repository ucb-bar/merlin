# Kodiak: int8 multicore RVV on a tapeout with a debug host

The Kodiak chip has **three harts, only two of which have a vector unit**, 512 MB of DRAM, and a
host-assisted (HTIF) console served by its own loader. That combination is what this example is for:
building one model three different ways for one chip, and proving each of them before anyone touches the
board.

Kodiak's support (the OPU, its boards, its RTL source) is the **Kodiak chip package**, vendored at
[`support/`](support/) from `ucb-bar/kodiak-mlir`. It is a layer on the RVV base (`examples/rvv/support`,
from `ucb-bar/rvv-mlir`): see [`docs/guides/chip_packages.md`](../../docs/guides/chip_packages.md). The
boards (`chipyard_kodiak`, `kodiak_opu_2core`, `firesim_kodiak_opu`) are data in
[`support/boards.yaml`](support/boards.yaml); the field meanings are on `Board` in
`src/merlin/runtime/boards.py`.

## Quick start on a new machine

```bash
git clone https://github.com/ucb-bar/merlin.git && cd merlin/examples/kodiak
./run.sh setup                    # Merlin venv + the model capture environment + .env;
                                  # prints the exact commands for anything still missing
./run.sh setup --llvm <llvm23> --chipyard <chipyard> --zephyr-sw <ws> --zephyr-sdk <sdk>
./run.sh op gemm                  # one 64x64x64 GEMM: RVV vs RVV+OPU, both gated on spike (~1 min)
./run.sh op gemm 96 80 200        # M N K: partial tiles, odd K
./run.sh op conv2d 16 64 16 3     # CIN COUT HW K (im2col + matmul)
./run.sh op my_op/loader.py       # your own one-operator model (see examples/workloads/gemm/loader.py)
```

`op` prints the status of both images, how much of the work went to the OPU, why anything did not, and
where the ELFs are (`out/artifacts/probes/chipyard_kodiak/<op>/`: `matrix.elf` for the chip,
`matrix_twin.elf` is what spike ran, `rvv.elf` is the baseline). It is the fastest loop for working on the
OPU kernel itself: edit the kernel, rerun `op`.

The stages:

```bash
./run.sh setup                    # see above
./run.sh preflight
./run.sh probe
./run.sh op <gemm|conv2d|encoder_block|loader.py> [dims] [--harts 2]
./run.sh capture <model>          # any model model2MLIR can capture, + its W8A8 golden
./run.sh build [--opu] [--models <model>]   # add --full for the shipped matrix (hours)
./run.sh package [--opu] [--models <model>]
./run.sh grade <console.txt> [model]
./run.sh op gemm --board firesim_kodiak_opu [--submit]     # one operator on FireSim (§6)
./run.sh firesim [--models <model>] [--submit]             # a whole model on FireSim (§6)
./run.sh firesim-collect <job-dir> [--job device=N --job control=N]
```

Running on the chip itself (a shared rig: check it is idle first) is in
[`support/RUNNING.md`](support/RUNNING.md). Running on FireSim, for the OPU's real datapath and cycles,
is [§6](#6-run-on-firesim--the-opus-real-datapath-and-cycles).

## 1. `probe` — ask the chip what it is, before uploading megabytes

`vlen_probe.elf` is a few hundred bytes that reads `mhartid`, `misa`, `mstatus.VS` and `vlenb`, walks the
DRAM region, measures the core clock against the fixed mtime reference, and prints one block per hart.

It exists because **every "what is this chip's VLEN" question here was previously answered by
inference**: Kodiak's board files put no `v` in `riscv,isa` at all, and the only numbers anywhere are
per-sample `CONFIG_RISCV_VECTOR_MAX_LEN` values that size Zephyr's *save area* and so only bound the real
width from above. Guessing costs more than a probe: see the gemmelos example for what a wrong guess did.

The stage also self-checks the probe on spike at a width you choose, which is the point — the probe must
report the width of the machine it runs on, not the width it was compiled for.

## 2. `build` — one model, three ways, each gated

Two tools have to exist before this stage does anything: **`clang-23`**, which compiles every RISC-V
object, and **`mlir-translate`**, which lowers the OpenMP IR — so it is needed for `h2` and `h3_scalar`
but not `h1`. Both come from an LLVM/MLIR 23 build that a fresh clone does **not** have;
[`docs/guides/llvm_toolchain.md`](../../docs/guides/llvm_toolchain.md) is the recipe, including a
two-target build if this example is all you want it for. `./run.sh preflight` names them if they are
missing. The RISC-V linker and `objcopy` come from chipyard's riscv-tools and the Zephyr SDK instead.

The interesting part of Kodiak is that one chip needs three different images:

| image | why |
|---|---|
| `h1` | one worker, RVV. The baseline, and the only one that survives a missing `mstatus.VS` |
| `h2` | RVV across both vector harts via the OpenMP shim |
| `h3_scalar` | no vector instructions at all, so the third core — which has no vector unit — can work |

A scalar image is not "the same thing, slower". Fanning an RVV model onto a hart with no vector unit does
not fail cleanly: that worker takes an illegal instruction, never reaches the barrier its peers wait on,
and the image hangs until whoever is running it gives up. That is why `vector_harts` is a descriptor
field, and why the scalar route exists.

Every image is simulated on spike at the board's real hart count and graded against the W8A8 reference
before it can ship. Two things that gate catches, both of which have shipped broken in the past:

- **A scalar image that computes the wrong answer.** Per-op register blocking was once applied only to
  vector builds; deepjscc's scalar image scored `w8a8_cos 0.9176` and was shipped *unsimulated*. Nothing
  ships now without a gate behind it.
- **`h1` and `hN` disagreeing.** The packager checks the multi-hart output is bit-identical to the
  single-hart one, so a fan-out bug cannot hide behind a cosine that is merely close.

## 2b. `--opu` — route the big int8 matmuls to the outer-product unit

Kodiak's Saturn units carry an int8 **outer-product unit** (OPU, pin `saturn_opu_int8`). With `--opu`,
`build`/`package` also produce images whose int8 contractions with both M and N at least one tile
(64 on Kodiak) call the OPU kernel from the Kodiak package. Everything else stays on RVV.

No simulator here executes the OPU, so each OPU image is gated through its **stand-in twin**, which is
the same lowering (checked by digest) with a scalar tile body. The twin is what spike runs against the
W8A8 golden. The shipped ELF is audited to contain the OPU's instructions and the twin to contain none.
The OPU datapath itself is certified separately on its RTL (`merlin.kernels.opu_cert`).

Every OPU build writes `routing_report.json` next to the image:

* `routed_contractions` and `opu_mac_share`: how much of the model the OPU sees. This bounds the
  speedup, and it is the first thing to read for a new model.
* `not_routed_by_reason`: why the rest did not go. Usually "smaller than one tile", sometimes a
  transposed weight (those are refused, never routed with the wrong layout).

`MERLIN_CHIPYARD` must point at a chipyard whose `generators/saturn` carries the OPU: the encodings
and the tile geometry are derived from it.

## 2c. Running the shipped OPU image on spike

Stock spike has no OPU. The Kodiak package can carry a **functional model** of it as a spike extension
(`spike/` in kodiak-mlir: a C++ `--extlib` plus `build.py`, which generates the instruction words and the
geometry from the same RTL derivation the kernel uses, so nothing in it is typed by hand). It
transliterates the unit's sequencer, array and VRF wiring, so it computes what the RTL computes. It does
not model timing, so its cycle counts mean nothing for the OPU and the sequencer hazard the kernel works
around cannot appear in it. It also does not model the taped-out part's partial-N erratum, which belongs
to a different RTL revision.

```bash
export MERLIN_TARGET_PATH=<kodiak-mlir checkout with spike/>   # until the vendored support/ carries it
./run.sh spike-model                  # -> out/build/spike-ext/kodiak/libsaturn_opu_model.so + manifest.json
export MERLIN_KODIAK_OPU_SPIKE_EXTLIB=$PWD/../../out/build/spike-ext/kodiak/libsaturn_opu_model.so
./run.sh op gemm 96 80 200            # now three legs: rvv, matrix (twin), matrix_on_spike (shipped ELF)
```

The model is loaded only as the package's contract declares it (`runner.spike_extension`: extension
name, `extlib_env`, sha256; see `merlin.targetgen.spike_extension`). A missing file or different bytes
fails the `matrix_on_spike` leg rather than running something else. That includes a rebuild with another
compiler or against another spike, which must be re-declared. A package that declares no model skips
the leg and says so. The leg grades the shipped image against the W8A8 golden and requires it to match
the twin's output bit for bit.

**Bring your own model.** `./run.sh capture <model>` writes `recaptures/<model>_int8_full` and its
W8A8 golden. Use a model2MLIR roster name, or `--loader path/to/loader.py` with a
`get_model_and_inputs()` (see `docs/guides/model2mlir.md`). Then
`./run.sh build --opu --models <model>`. For a single image without packaging:

```bash
merlin-compile --workload <model> --dtype int8 --target rvv --board chipyard_kodiak --harts 2 \
  --matrix-unit saturn_opu --matrix-config KodiakOPUConfig --matrix-support-target kodiak \
  --run spike --verify
```

## 3. What simulation cannot tell you

This is the honest boundary, and it is worth stating precisely because it has cost two delivery rounds.

**Neither spike nor the Saturn RTL enforces `mstatus.VS`.** So a configuration in which no thread carries
vector state runs perfectly in every simulator we have and traps on the first vector instruction on real
silicon. That is exactly what happened: with `CONFIG_RISCV_ISA_EXT_V` unset, a thread's initial `mstatus`
is `MSTATUS_DEF_RESTORE` (MPP|MPIE — no VS), so the OpenMP master lost vector state the moment pool
creation switched it out, and `FPU_SHARING=y` then routed the resulting trap into the FP retry path where
it spun with nothing printed. On the chip: **every single-hart image passed and every multi-hart image
hung.** The fix was to match the settings in that chip's own known-working RVV+SMP sample.

The general lesson, which the `_debug` images exist to serve: when the simulator cannot model the failure,
the binary has to report on itself.

## 4. `package` — what goes in the zip, and why both builds

Every image ships **twice**:

- the plain one, for a number;
- a `_debug` twin — same computation — that prints a `STAGE` line at each boot milestone, an `ALIVE`
  heartbeat naming the op it is inside, `MEM` lines proving the linked DRAM really answers, stack
  high-water marks, and on a fault one greppable line: `FAIL fatal … mcause=… mepc=… build_hash=…`.

Splitting those across two downloads was tried and was the wrong shape: the binary you need when
something goes wrong is the one you do not have. The zip also carries the W8A8 and weight-only
references, an `expected_console.txt` per image, and `grade.py` — a single file needing only numpy, so the
board's owner can score a run without a merlin checkout.

## 5. `grade` — and the one configuration still unexplained

`./run.sh grade <console.txt>` scores a returned log. On a log that stopped early it reports the last
`STAGE`, any `FAIL` line and any failed memory probe, instead of only "this log has no OUT line" — which
is equally true of a hang, a crash, an unfinished upload and a board that never booted.

**Open, and shipped as such:** every `h3` configuration gates `w8a8` on spike at three harts, and every
one of them has failed on the chip. Nothing here reproduces it. The confound is that every `h3` image is
*also* the scalar one, so "h3 fails" has never separated the third core from the scalar multicore path —
which is why the package now contains a **2-hart scalar** image. Running `deepjscc_int8_h2_scalar_debug`
answers it in one log: if it passes and `h3_scalar` fails, the third core is the problem; if both fail,
the scalar route is. Use deepjscc specifically — it is 5.5 G cycles where the scalar spectformer and
whisper images are 42–104 G.

## 6. Run on FireSim — the OPU's real datapath and cycles

Spike cannot execute the OPU, so every `--opu` image above is gated through its stand-in twin. FireSim
runs the RTL itself on an FPGA: the shipped image, the real unit, real cycles. This is how the
single-core Kodiak OPU numbers were measured (SpectFormer int8, see the end of this section).

### The hardware

| | |
|---|---|
| board | `firesim_kodiak_opu` ([`support/boards.yaml`](support/boards.yaml)) |
| design | **one** Shuttle core carrying Kodiak's Saturn vector unit with the OPU: VLEN 512 / DLEN 256 (a 64-lane int8 tile), 16 GiB of FireSim DRAM, HTIF console |
| FireSim config | `FireSimKodiakOPU1CoreConfig`, U250, 15 MHz |
| hwdb key | `alveo_u250_firesim_kodiak_opu_1core` — the board's `bitstream:` field, so the tooling never guesses it |
| pin | `firesim_kodiak_opu_1core` in [`merlin/contract/hardware_pins.yaml`](../../merlin/contract/hardware_pins.yaml): the sha256 of `firesim.tar.gz`, the RTL it was elaborated from, and why |

**One core, not Kodiak's two.** The two-core configuration does not route on a U250. The vector unit,
OPU, caches and bus widths are Kodiak's, so OPU datapath numbers and whole-model grades are statements
about Kodiak's vector unit; nothing about core count or multicore scaling is. Say "single-core Kodiak"
when you cite a number from it.

**Building the bitstream on your own U250.** It is elaborated from saturn `ea373800`
(`miles-rusch-berkeley/saturn-vectors`, branch `opu-int8`, which contains `OuterProductUnit.scala`)
plus the Kodiak OPU configs (`WithKodiakBase(1)` with `WithShuttleVectorUnit(512, 256, opuParams)`) and
the firechip `FireSimKodiakOPU1CoreConfig` target. Two things matter more than they look:

- the clock-gate model must be [`merlin/contract/rtl/eicg_passthrough.v`](../../merlin/contract/rtl/eicg_passthrough.v)
  (selected through `ClockGateModelFile`). With rocket-chip's stock `EICG_wrapper` the gated OPU array
  gets a second global clock net and hold never closes, at any frequency;
- the configs are additions not yet upstream: the two patches and the FPGA clock-gate model are in [`support/firesim/`](support/firesim/README.md)
  rather than re-deriving them.

Register the result in `config_hwdb.yaml` under the **same key** (`alveo_u250_firesim_kodiak_opu_1core`):
the board names that key and every job passes it as `--hw-config`. If your tar's sha256 is not the
pinned one, `firesim-collect` still grades the run but reports the bitstream as UNPINNED; cite it as your
own build.

### Settings

Set these in the repository's `.env` (or the environment). None of them has a default that points at a
particular machine.

| Setting | What it is | Needed for |
|---|---|---|
| `MERLIN_CHIPYARD` | the chipyard whose `sims/firesim` deploys the bitstream (and whose `generators/saturn` the OPU encodings are derived from) | everything |
| `MERLIN_EXT_FIRESIM_QUEUE` | a shared `firesim-queue`'s state directory (`jobs/<id>/` lives there) | `--submit` |
| `MERLIN_FIRESIM_QUEUE_CLIENT` | the raw queue client, if not `$MERLIN_EXT_FIRESIM_QUEUE/bin/firesim-queue` | optional |
| `MERLIN_FIRESIM_WORKLOAD` | the FireSim workload the ELF boots under (default `merlin-zephyr`) | optional |
| `MERLIN_FIRESIM_CLIENT_DROP_ENV` | e.g. `HOME,USER,LOGNAME` when the queue daemon runs as another account | optional |

The workload is a three-line JSON that each job directory carries
(`firesim/workload/<workload>.json`). If your FireSim does not have it yet, the printed plan says the
one `cp` that installs it.

### One operator

```bash
./run.sh op gemm 64 64 64 --board firesim_kodiak_opu            # build + spike-gate + print the jobs
./run.sh op gemm 64 64 64 --board firesim_kodiak_opu --submit   # ...and run them on the FPGA
```

This is the quick-start `op` stage built for the FireSim board: the OPU image and its plain-RVV control, both
gated on spike first (an image that fails its gate is never planned). The images and the plan go to
`out/artifacts/probes/firesim_kodiak_opu/op_gemm_64x64x64_h1/` (`matrix.elf` is the DEVICE leg,
`rvv.elf` the CONTROL leg, `firesim_job.json` the plan). Without `--submit` nothing touches the FPGA:
the exact queue commands are printed, one per leg, each naming the bitstream with `--hw-config`.
`./run.sh firesim <that directory> --submit` submits a printed plan later.

A 64x64x64 GEMM is a few hundred thousand cycles: the run is dominated by FPGA programming and Zephyr
boot, a few minutes per leg.

### A whole model

```bash
./run.sh capture spectformer                      # once: the bundle and its W8A8 golden
./run.sh build --opu --models spectformer         # spike-gate the OPU image through its twin
./run.sh firesim --models spectformer             # build + audit the FireSim legs, print the jobs
./run.sh firesim --models spectformer --submit --timeout 3600
```

`firesim` builds two images with [`build_tools/scripts/zephyr_firesim_leg.py`](../../build_tools/scripts/zephyr_firesim_leg.py):
the DEVICE leg (int8 contractions routed to the OPU) and its CONTROL twin (the same lowering, nothing
routed). Both are debug builds, so a hung run prints an `ALIVE` heartbeat rather than nothing. Each is
disassembled and refused if the device leg carries none of the OPU's instructions or the control leg
carries any, so neither can measure the wrong thing under the right name. They land in
`out/artifacts/probes/firesim_kodiak_opu/<model>_firesim_h1/`. Budget the timeout from the model's
cycles: at 15 MHz, seconds ≈ cycles / 1.5e7, plus a few minutes per job for FPGA programming.

### Without the shared queue

With `MERLIN_EXT_FIRESIM_QUEUE` unset the plan prints the equivalent manual steps for a FireSim you run
yourself: install the workload JSON, set `default_hw_config: alveo_u250_firesim_kodiak_opu_1core` and
`workload_name: <workload>.json` in `config_runtime.yaml`, copy each leg's ELF to
`deploy/workloads/<workload>/<bootbinary>`, then `firesim kill; firesim infrasetup && firesim runworkload;
firesim kill` once per leg. Always run the full four steps, even when the bitstream has not changed: a
skipped `kill`/`infrasetup` leaves a stale simulator attached to the XDMA device.

### Grading a run

```bash
./run.sh firesim-collect <job-dir>                                # after --submit (job ids are recorded)
./run.sh firesim-collect <job-dir> --job device=812 --job control=813
./run.sh firesim-collect <job-dir> --uartlog device=<results-workload/...-q812/<workload>0/uartlog> \
                                   --uartlog control=<...>
```

For each leg it checks, and reports every check it could not make as a problem rather than a pass:

- the bitstream that ran: FireSim's own `HW_CFG_SUMMARY` beside the uartlog must name the board's hwdb
  key. A shared FireSim's default design is global state; a job that omitted `--hw-config` has run on
  someone else's bitstream here, with plausible output;
- the image: the console's `METRIC build_hash` must be the planned image's, and the ELF must not have
  been rebuilt since it was planned;
- the answer: the `OUT` line against the bundle's `golden_w8a8.npy` (Merlin's host W8A8 datapath; the
  device should reproduce it, `w8a8` tier: cos > 0.999, rel < 1e-2) and the whole-tensor `HASH`;
- the cycles: `METRIC cycles` (the compute region), FireSim's wall clock and effective frequency, and
  the device leg's speedup over the control leg.

It writes `firesim/results.json` in the job directory and a measurement under
`out/artifacts/measurements/firesim_alveo_u250_firesim_kodiak_opu_1core/<model>/`, whose provenance
names the pinned bitstream artifact it matched (by the tar FireSim reports, compared by sha256).

### The reference result

SpectFormer int8 (57 contractions routed to the OPU), measured 2026-08-18 on this bitstream:

| leg | cycles | vs W8A8 golden |
|---|---:|---|
| device (OPU) | 3,359,893,080 | cos 1.0, max_rel 0.0 |
| control (RVV only) | 5,322,407,483 | cos 1.0, max_rel 0.0 |

**1.584x** for the OPU leg, 99.6% of the 1.59x Amdahl bound its MAC share allows. Re-grading those two
uartlogs with `firesim-collect` reproduces both cycle counts and matches the bitstream to its pin. Two
caveats travel with the number: those runs did not go through the queue (its daemon was not
dispatching at the time) and were booted by hand, one leg at a time, on an FPGA checked idle first; and
the configs were uncommitted upstream, so the hardware is "saturn `ea373800` plus the Kodiak OPU config
additions". The flow above is that procedure with the queue put back in: same legs, same audit, same
bitstream, named on every job.
