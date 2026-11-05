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
```

Running on the chip itself (a shared rig: check it is idle first) is in
[`support/RUNNING.md`](support/RUNNING.md).

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
