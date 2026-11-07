# Kodiak: int8 multicore RVV on a tapeout with a debug host

The Kodiak chip has **three harts, only two of which have a vector unit**, 512 MB of DRAM, and a
host-assisted (HTIF) console served by its own loader. That combination is what this example is for:
building one model three different ways for one chip, and proving each of them before anyone touches the
board.

Kodiak's support (the OPU, its boards, its RTL source) is the **Kodiak chip package**,
[`ucb-bar/kodiak-mlir`](https://github.com/ucb-bar/kodiak-mlir). Merlin does not copy it: it pins one commit
of it (`build_tools/upstreams/target_support.json`, `pinned_support`), and `./run.sh setup` fetches exactly
that commit into `out/build/generated/pinned/kodiak@<sha12>/`, where every stage finds it
(`merlin-target-fetch kodiak --root` prints the path; below, `<kodiak-mlir>`). It is a layer on the single
RVV base, `ucb-bar/rvv-mlir` at Merlin's base pin (tracked at `examples/rvv/support`): see
[`docs/guides/chip_packages.md`](../../docs/guides/chip_packages.md#what-the-rvv-pin-contains). That pin
contains a capability contract; the executable RVV lowering lives in Merlin. The boards (`chipyard_kodiak`,
`kodiak_opu_2core`, `firesim_kodiak_opu`) are data in `<kodiak-mlir>/boards.yaml`; the field meanings are on
`Board` in `src/merlin/runtime/boards.py`. Not fetched, every stage but `setup` stops and names `setup`.

**Changing the package** (the OPU kernel, the overlay, the boards): commit in kodiak-mlir, publish the
commit, then bump its one sha here (`python build_tools/scripts/vendor_support.py pin kodiak <sha>`) and
re-run `./run.sh setup`. To try an unpublished change first, select your clone explicitly:
`MERLIN_TARGET_PATH=/path/to/kodiak-mlir ./run.sh op gemm` (a clone made with `git clone
--recurse-submodules` carries its rvv base at `base/rvv-mlir`, at the same pinned commit).

## Quick start on a new machine

For measured utilization and its limits, see [UTILIZATION.md](UTILIZATION.md).
For source-quantized model requirements, see the
[TorchAO W8A8 source contract](TORCHAO_W8A8_SOURCE.md).

To see how a compiler-routed convolution selects a handwritten OPU kernel, run
[`examples/conv2d_to_opu.py`](https://github.com/ucb-bar/kodiak-mlir/blob/6710d84bbe0933510a51d0736fb240f89c1d8574/examples/conv2d_to_opu.py)
from the pinned Kodiak package. It checks routing, the selected kernel, and numerical results on the
OPU Spike model. The matching [portable evidence](evidence/compiler_selected_conv_kernel_examples.json)
records the artifact hashes and qualification limits.

```bash
git clone --branch feat/OPU-VME https://github.com/ucb-bar/merlin.git && cd merlin/examples/kodiak
# the three things too big to install for you (setup prints how to get any it cannot find):
./run.sh setup --llvm <llvm23-install> --chipyard <chipyard> --zephyr-sdk <zephyr-sdk-0.17.0>
./run.sh preflight --require package  # fails if the image-build prerequisites are missing
./run.sh op gemm                  # one 64x64x64 GEMM: RVV vs RVV+OPU, gated on spike (~1 min)
./run.sh op gemm 96 80 200        # M N K: partial tiles, odd K
./run.sh op conv2d 16 64 16 3     # CIN COUT HW K (im2col + matmul); CONV_STRIDE=2 for 7x7 output pixels
./run.sh op attention 48 128 4    # SEQ D HEADS: multi-head Q.K^T and P.V (activation x activation)
./run.sh op gemm_nt 64 96 80      # M N K: A.B^T of two activations (a transposed operand)
./run.sh op my_op/loader.py       # your own one-operator model (see examples/workloads/gemm/loader.py)
```

### First hour: checklist

Times measured on a 48-core host from an empty `out/`, no `.venv`, no `.env` (2026-11-05).

1. **Have the big three** (setup finds them, never installs them): LLVM/MLIR 23 at llvm-project
   `a47bddcc` ([`llvm_toolchain.md`](../../docs/guides/llvm_toolchain.md)), chipyard after
   `./build-setup.sh riscv-tools`, the Zephyr SDK 0.17.0. Plus `uv`, `git`, `make`, `g++`, `dtc` on PATH.
2. **`./run.sh setup --llvm <prefix> --chipyard <dir> --zephyr-sdk <dir>`**: about 5 min the first time
   (most of it Zephyr's `west update`), seconds after. Exit 0 means nothing is left; otherwise it
   lists each open item with its commands. Run them, run setup again.
3. **The OPU's saturn.** If setup says `[drift] saturn_opu_int8`, OPU builds refuse that checkout.
   Put it on the pin (the commands below), or, if it is not yours to move, add
   `MERLIN_ALLOW_OFF_PIN=saturn_opu_int8` to `.env` and every build records the drift.
4. **`./run.sh preflight --require package --require-pin saturn_opu_int8`** (1 s): the package
   prerequisites and the OPU source pin must both pass. `--require package` alone checks only
   host prerequisites and can exit zero while the matrix pin is `[stop]`. Other units listed
   in the report, such as the separately pinned VME source, need their own pin gate when selected.
   A `!` in the settings list is a shell export overriding what setup recorded: `unset` it.
5. **`./run.sh op gemm 64 64 64`** (about 1 min): `rvv`, `matrix` and `matrix_on_spike` all `verified`.
6. **A whole model**, with no download (deterministic weights):
   `./run.sh capture encoder_block --loader ../workloads/encoder_block/loader.py` (about 2 min), then
   `./run.sh build --opu --models encoder_block` (about 2 min): each image gated on spike, packaged under
   `out/artifacts/delivery/example-kodiak/`. `build`'s default model, deepjscc, has to be captured
   first (`./run.sh capture deepjscc`), and its model2MLIR loader imports upstream code from a path on
   the machine it was written on: `git clone https://github.com/mingyuyng/DiffJSCC` and
   `export DIFFJSCC_DIR=<that clone>` before capturing it.

`setup` needs `uv` and `git`, and is safe to re-run: each step is skipped when its result is already
there. Exporting the paths instead of passing flags works too (`MERLIN_MLIR_INSTALL` or `MERLIN_CLANG`,
`MERLIN_CHIPYARD`, `ZEPHYR_SDK_INSTALL_DIR`, and `ZEPHYR_BASE` if you already have a Zephyr workspace);
setup records whatever it finds in the repo's `.env`, so later shells need nothing exported. What it does
by itself, and what it only finds and verifies, with sizes and times, is in
[`docs/guides/chip_packages.md`](../../docs/guides/chip_packages.md#setting-up-a-machine). It ends with a
list of whatever is still missing, with the exact commands, and exits non-zero until that list is empty.

Kodiak adds two things to the common setup:

* the **OPU's RTL**: `$MERLIN_CHIPYARD/generators/saturn` must be saturn `ea373800` (pin
  `saturn_opu_int8`, branch `opu-int8` on the fork `miles-rusch-berkeley/saturn-vectors`) with
  `<kodiak-mlir>/firesim/01-saturn-OPUConfigs.patch` applied as shipped, which adds `KodiakOPUConfig`
  (leave its passthrough-model path alone to preserve the pin). For a new FPGA elaboration,
  `$MERLIN_EICG_PASSTHROUGH_V` must reach the **remote build process** and name the passthrough
  Verilog file there; an export only in the local launcher is insufficient. If that cannot be
  arranged, a changed patch and a coordinated source-pin update are required. The
  OPU's encodings and tile geometry are read from there, so setup checks every file that derivation
  reads, by content. Another commit whose read files are byte-identical is a warning (builds record the
  revision). A read file that differs blocks every OPU build (`op`, `build --opu`, `matrix`, `firesim`)
  with these commands, until you either run them or accept the drift with
  `MERLIN_ALLOW_OFF_PIN=saturn_opu_int8` in `.env` (each build then records it):

  ```bash
  PKG="$(<merlin>/.venv/bin/merlin-target-fetch kodiak --root)"    # kodiak-mlir, fetched by setup
  cd $MERLIN_CHIPYARD/generators/saturn
  git fetch https://github.com/miles-rusch-berkeley/saturn-vectors.git ea373800169ce7bbb9ca27ececf74101a7caf9d5
  git checkout -B opu-int8 ea373800169ce7bbb9ca27ececf74101a7caf9d5
  git apply "$PKG/firesim/01-saturn-OPUConfigs.patch"
  ```

* the OPU's **spike model** (§2c), which setup builds into `out/build/spike-ext/kodiak/` and records as
  `MERLIN_KODIAK_OPU_SPIKE_EXTLIB`, so `op` also runs the shipped OPU image on spike.

`op` prints the status of each image (`rvv`, `matrix` = the OPU image's stand-in twin, `matrix_on_spike` =
the shipped OPU image on the spike model), how much of the work went to the OPU, why anything did not, and
where the ELFs are (`out/artifacts/probes/chipyard_kodiak/<op>/`: `matrix.elf` for the chip,
`matrix_twin.elf` is what spike ran, `rvv.elf` is the baseline). It is the fastest loop for working on the
OPU kernel itself: edit the kernel, rerun `op`.

The stages:

```bash
./run.sh setup [--llvm P] [--chipyard D] [--zephyr-sw D] [--zephyr-sdk D] [--m2m D] [--no-fetch]
./run.sh preflight --require package --require-pin saturn_opu_int8
./run.sh probe
./run.sh op <gemm|gemm_nt|conv2d|attention|encoder_block|loader.py> [dims] [--harts 2]
./run.sh capture <model>          # any model model2MLIR can capture, + its W8A8 golden
./run.sh build [--opu] [--models <model>]   # add --full for the shipped matrix (hours)
./run.sh package [--opu] [--models <model>]
./run.sh matrix --models <a,b> [--harts 1,2]   # regression matrix: RVV and RVV+OPU per model x harts
./run.sh grade <console.txt> [model]
./run.sh op gemm --board firesim_kodiak_opu [--submit]     # one operator on FireSim (§6)
./run.sh firesim [--models <model>] [--submit]             # a whole model on FireSim (§6)
./run.sh firesim-collect <job-dir> [--job device=N --job control=N]
```

`build`, `package`, and `firesim` accept `--rvv-package <package>`. The default is the older
K1-certified `rvv/impr_tuned_wholemodel_vf_int8`. For the aligned-weight Kodiak prepack schedule,
select `rvv/kodiak_prepack_w8a8_v1` explicitly. That package has exact Spike and one-core FireSim
evidence for the two-layer TinyLlama 64-token section, recorded in its manifest; it has no K1 or
physical Kodiak certification. For example:

```bash
./run.sh firesim --models tinyllama_l2_s64 --rvv-package rvv/kodiak_prepack_w8a8_v1
./run.sh package --opu --models tinyllama_l2_s64:/path/to/section-bundle --harts 1 \
  --rvv-package rvv/kodiak_prepack_w8a8_v1 --no-spike-backends matrix
```

The last command gates the RVV image on Spike and builds/audits the OPU image. The delivery marks
the OPU image **UNGATED** because its scalar stand-in can take hours. Run the exact OPU image on a
compatible simulator or chip before claiming that particular ELF passed numerically.

Running on the chip itself (a shared rig: check it is idle first) is in the package's `RUNNING.md`
(`<kodiak-mlir>/RUNNING.md`). Running on FireSim, for the OPU's real datapath and cycles,
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

Each OPU image is gated through its **stand-in twin**, which is the same lowering (checked by digest)
with a scalar tile body. The twin is what stock spike runs against the W8A8 golden. When setup built the
OPU's spike model (§2c), the **shipped** image runs on it too, graded against the same golden and
required to match the twin bit for bit (`matrix.shipped_on_simulator` in the manifest); a mismatch fails
the image. The shipped ELF is audited to contain the OPU's instructions and the twin to contain none.
The OPU datapath itself is certified separately on its RTL (`merlin.kernels.opu_cert`).

Every OPU build writes a routing report: `routing_report.json` next to the image for `op` and
`merlin-compile`, and under `binaries[].matrix.routing_report` in the package's `manifest.json` for
`build`/`package`:

* `routed_contractions` and `opu_mac_share`: how much of the model the OPU sees. This bounds the
  speedup, and it is the first thing to read for a new model.
* `not_routed_by_reason`: why the rest did not go -- usually the cost model's verdict that the vector
  unit is cheaper for that shape ("cost model: ..."), with its estimate in `not_routed[].decision`.
* `macs_by_site` and `priced_sub_tile_routes`: where the MACs sit (conv / attention / linear) and every
  route that pads a partial tile, swaps the operands' roles or re-lays-out an operand, with the
  estimate that justified it. `MERLIN_MATRIX_ROUTE_POLICY=tile_filling` restores the old rule (both
  extents at least one tile, canonical layout only).

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

`./run.sh setup` builds it from the fetched package (`<kodiak-mlir>/spike/`) into
`out/build/spike-ext/kodiak/` (`libsaturn_opu_model.so` + `manifest.json`) and records
`MERLIN_KODIAK_OPU_SPIKE_EXTLIB` in `.env`, so `op` runs three legs: `rvv`, `matrix` (the twin) and
`matrix_on_spike` (the shipped ELF), and `build --opu`/`package --opu` and `merlin-compile --matrix-unit`
run the shipped ELF on it as well. It is rebuilt only when the package's declared inputs change. To
build it by hand, or from another kodiak-mlir checkout selected with `MERLIN_TARGET_PATH`:

```bash
./run.sh spike-model                  # -> out/build/spike-ext/kodiak/libsaturn_opu_model.so + manifest.json
```

The model is loaded only as the package's contract declares it (`runner.spike_extension`: extension
name, `extlib_env`, sha256; see `merlin.targetgen.spike_extension`). A missing file or different bytes
fails the `matrix_on_spike` leg rather than running something else. That includes a rebuild with another
compiler or against another spike, which must be re-declared. A package that declares no model skips
the leg and says so. The leg grades the shipped image against the W8A8 golden and requires it to match
the twin's output bit for bit.

**Bring your own model.** `./run.sh capture <model>` writes `recaptures/<model>_int8_full` and its
W8A8 golden. Use a model2MLIR roster name (the directories in `$MERLIN_M2M_DIR/workloads/`, which setup
records in the repo's `.env`), or `--loader path/to/loader.py` with a `get_model_and_inputs()` (see
`docs/guides/model2mlir.md`); `<model>` then only names the bundle. Then
`./run.sh build --opu --models <model>`. For a single image without packaging, from the repository root:

```bash
MERLIN_BOARD_CATALOG=examples/board-catalog.yaml:"$(.venv/bin/merlin-target-fetch kodiak --root)" \
.venv/bin/merlin-compile --workload <model> --dtype int8 --target rvv --board chipyard_kodiak --harts 2 \
  --matrix-unit saturn_opu --matrix-config KodiakOPUConfig --matrix-support-target kodiak \
  --run spike --verify
```

(`run.sh` sets the board catalog for you; a bare `merlin-compile` needs it named.) What a new model
runs into, in the order you will meet it:

* **The loader's own switches.** A roster loader may refuse to run without them. ResNet-50's wants real
  ImageNet inputs or `M2M_RESNET_RANDOM=1` for a smoke capture; its docstring says which. They are
  environment variables, so `M2M_RESNET_RANDOM=1 ./run.sh capture resnet50_v1_5` passes them through.
* **Time.** The capture and the W8A8 golden both run the whole model on the host: about 5 minutes each
  for ResNet-50. Each spike gate is one single-threaded simulation of a whole inference.
* **Kodiak's 512 MiB.** The image holds the weights, the code and a STATIC ACTIVATION ARENA planned at
  build time: each buffer is freed after its last use and all of them are seated in one arena sized to
  the peak of what is live at once, not the sum of everything the model allocates. What the plan cannot
  seat stays on the heap, priced at its sum. Every build prints and records the budget (`memory` in the
  build record, `plan` in the delivery manifest): buffers summed, peak live, arena, heap, weights,
  code, region. ResNet-50 at 224x224 needs 146 MiB (its buffers sum to 375 MiB; the arena is 41 MiB),
  where sizing to the sum asked for 624 MiB. A TinyLlama section (`examples/workloads/causal_lm_section`)
  of 2 layers at 128 tokens fits a 256 MiB region on RVV. Its h1 OPU image relinks for a 448 MiB
  region; the exact OPU and RVV Spike images and a matched one-core FireSim pair have passed
  numerical gates (see [measured scope](UTILIZATION.md)). Eight layers at 128 tokens, or 2 layers at 512 tokens,
  fill physical Kodiak's 512 MiB. The whole model does not fit physical Kodiak: its weights
  alone are about 1.2 GiB. The one-core OPU FireSim target has 16 GiB simulated DRAM and a
  separate external-weight layout; a full-model OPU run there remains unbuilt and unverified. A model
  that does not fit is refused naming every term. `--no-pass static-arena --no-pass sink-deallocs`
  restores the old heap-only layout. The OPU image does not get the plan's reuse yet (see
  `docs/design/static_arena_wiring.md`): ResNet-50's needs 507 MiB at 1 hart and is refused at 2.
* **Weight-only int8.** The int8 capture quantizes weights with torchao's default filter, which matches
  `nn.Linear` only: a CNN's convolution weights stay f32 in the bundle. Merlin's W8A8 build still
  quantizes those convolutions (`conv_int8`) and routes them to the OPU, but the image carries their
  weights as f32: ResNet-50's weights blob is 96 MB, about 4x its int8 size. `--pass prequantize-weights`
  stores them as int8 plus a per-channel scale instead (the same formula, applied at build time; refused
  when a weight cannot be proven to feed only its contraction).
* **What routes is priced.** A contraction routes when the OPU's estimate, derived from its tile edge
  and datapath width, beats the vector unit's, so the deep layers of a small input (7x7 = 49 output
  pixels, padded to a 64-wide tile) route too; `priced_sub_tile_routes` in the routing report lists each
  with its padding and estimated utilisation, and `not_routed[]` names anything left on RVV and why.
* **Say what the model is.** A package's README describes every model in it. For one Merlin does not
  know, write a paragraph (what it is, how it was captured, what to expect of it) to
  `delivery_status.md` in its bundle directory, or `build`/`package` ends with a `NO STATUS` problem
  and exits non-zero.

A loader directory may carry a `capture.toml` with `deps = [...]`, like model2MLIR's roster: `capture`
installs a missing one into the capture venv and refuses one that would replace an installed version.
**A model too big for the board's 512 MB** is captured as a section with the generic
[`causal_lm_section`](../workloads/causal_lm_section/loader.py) loader: the first N decoder layers of a
cached Hugging Face causal LM, real weights, entering at the embedded hidden state so the fp32 embedding
table stays out of the image. The token count is the M of every projection, so it decides what the OPU
sees: below one tile (64 here) nothing is routed and the OPU leg reports that instead of an image.

```bash
HF_HUB_OFFLINE=1 CAUSAL_LM_MODEL=TinyLlama/TinyLlama-1.1B-Chat-v1.0 CAUSAL_LM_LAYERS=2 CAUSAL_LM_ENTRY=embeds \
  M2M_SEQ=64 ./run.sh capture tinyllama_l2_s64 --loader ../workloads/causal_lm_section/loader.py
```

This offline command requires the TinyLlama checkpoint in the local Hugging Face cache first.
The measured s64/s128 captures are ignored local artifacts, not part of a fresh Git checkout;
transfer their exact capture bundles from the shared host to reproduce the published hashes.
A new checkpoint download or recapture has a new capture identity and must pass the full Spike
and hardware gates before its numbers are compared with the results here.

## 2d. `matrix` — a regression matrix over whole models

```bash
./run.sh matrix --models spectformer,tinyllama_l2_s64        # --harts 1,2 by default; --rerun, --jobs N
```

For each captured model and hart count it builds the plain RVV image and the OPU image, and gates them
the way `op` gates one operator: RVV on spike against the W8A8 golden; the OPU image through its stand-in
twin, then the shipped image on the OPU's spike model, which must match the twin bit for bit. It then
checks each model's 2-hart outputs are bit-identical to its 1-hart ones (per leg, both OPU images). Every
image is simulated at the board's VLEN. The table, with each OPU image's routed contractions, MAC share
and not-routed reasons, is `out/artifacts/probes/model_matrix/chipyard_kodiak/MATRIX.md`; each cell's
full record (gate numbers, output digests, routing report, the unit revision it was built from) is a JSON
beside it, kept so a rerun builds only the cells that have none. Whole models are minutes to hours of
spike each; `--jobs` (default 2) bounds how many run at once.

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

Stock Spike cannot execute the OPU; the pinned `saturn_opu` Spike extension used above can run
the shipped OPU image, while its stand-in twin provides an independent numerical gate. FireSim
runs the RTL itself on an FPGA: the shipped image, the real unit, real cycles. This is how the
single-core Kodiak OPU numbers were measured (SpectFormer int8, see the end of this section).

### The hardware

| | |
|---|---|
| board | `firesim_kodiak_opu` (`<kodiak-mlir>/boards.yaml`) |
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
- the configs are additions not yet upstream: the two patches and the FPGA clock-gate model are in `<kodiak-mlir>/firesim/` (its `README.md`)
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
| `MERLIN_FIRESIM_CLIENT_PATH_PREFIX` | directory of client-side `firesim`/`make` launchers when the queue daemon needs a different working directory | optional |

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
./run.sh firesim --models spectformer             # build + spike-gate the exact FireSim ELFs, print the jobs
./run.sh firesim --models spectformer --submit --timeout 7200
```

The SpectFormer capture requires its upstream source tree (`SPECTFORMER_DIR`) and trained
checkpoint (`SPECTFORMER_CKPT`). Without the checkpoint, its loader prints `RANDOM INIT` and
changes the model being measured. On a fresh checkout, transfer the exact capture bundle or
supply both dependencies before the `capture` command, then verify the resulting capture digest.

`firesim` builds both images through `merlin-compile`: the DEVICE leg routes int8 contractions to the
OPU; the CONTROL leg uses plain RVV. Both exact ELFs pass the W8A8 spike gate before submission, and
the shipped DEVICE ELF also runs on the package's OPU spike model and must match its scalar twin bit for
bit. This avoids structural profiling changing the program under test. The images land in
`out/artifacts/probes/firesim_kodiak_opu/<model>_firesim_h1/run_<id>/` so a queued job's ELF cannot be
replaced by a later rebuild. Budget the timeout from the model's
cycles: at 15 MHz, seconds ≈ cycles / 1.5e7, plus a few minutes per job for FPGA programming.
The DEVICE job also archives the exact image's `routing_report.json` and `opu_signatures.json`
with SHA-256 hashes; use those counts with the measured target cycles for utilization.

For the TinyLlama sections, the scalar twin may take hours on Spike. Use `--gate independent-rvv`
to compile the same-lowering pair without executing the twin, run the shipped OPU image on its
declared Spike model, and require its whole-output hash and prefix digest to match a fresh plain-RVV
control built for the **same FireSim board**. Both legs must also pass W8A8. The job plan records
this gate and a digest of the capture bytes explicitly. TinyLlama's W8A8 golden comparison covers
the printed 4,096-element prefix; OPU/RVV whole-output identity uses the images' 32-bit hash and
prefix SHA-256. This does not separately check the scalar twin's packing path. The default remains
the scalar-twin gate.

```bash
./run.sh firesim --models tinyllama_l2_s64 --gate independent-rvv --timeout 7200
./run.sh firesim --models tinyllama_l2_s128 --gate independent-rvv --timeout 14400
```

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
  device should reproduce it, `w8a8` tier: cos > 0.999, rel < 1e-2). If `OUT` is truncated, this
  numerical grade covers only its reported prefix. The FireSim whole-tensor `HASH` must exactly match
  the hash of the same ELF's Spike run; the printed prefix must match that Spike run bit for bit too;
- the cycles: `METRIC cycles` (the compute region), FireSim's wall clock and effective frequency, and
  the device leg's speedup over the control leg.

It writes `firesim/results.json` in the job directory and a measurement under
`out/artifacts/measurements/firesim_alveo_u250_firesim_kodiak_opu_1core/<model>/`, whose provenance
names the pinned bitstream artifact it matched (by the tar FireSim reports, compared by sha256).
See [OPU utilization](UTILIZATION.md) for the target-cycle denominator, routed MAC count, tile fill,
and the limits of the current hardware counters.

### The reference result

The current `merlin-compile` spectformer pair ran through the shared queue on 2026-11-05. Device job
3858: **3,531,733,777** target compute cycles; plain-RVV control job 3859: **6,120,729,118** cycles,
for **1.733×** speedup. Both jobs passed the W8A8 gate and matched their exact Spike-gated output
hashes and prefixes; `firesim/results.json` records the bitstream, image hashes, and full verdict.
These jobs used the uninstrumented images built through the flow above.

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
