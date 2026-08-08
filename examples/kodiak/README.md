# Kodiak: int8 multicore RVV on a tapeout with a debug host

The Kodiak chip has **three harts, only two of which have a vector unit**, 512 MB of DRAM, and a
host-assisted (HTIF) console served by its own loader. That combination is what this example is for:
building one model three different ways for one chip, and proving each of them before anyone touches the
board.

Board facts live in [`merlin/python/merlin/runtime/boards.py`](../../merlin/python/merlin/runtime/boards.py)
as data (`chipyard_kodiak`), not in code — see [`docs/guides/adding_a_target.md`](../../docs/guides/adding_a_target.md).

```bash
./run.sh preflight
./run.sh probe
./run.sh build            # add --full for the shipped matrix (hours)
./run.sh package
./run.sh grade <console.txt>
```

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
