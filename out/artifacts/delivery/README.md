# Delivery packages

Every zip below carries **two builds of every image**: the plain one to run for a number, and a
`_debug_` twin of the same model that explains itself while it runs. Send the whole zip — the point of
the pairing is that nobody has to come back to us for the other half.

## SEND THESE

| zip | for | contents |
|---|---|---|
| `merlin-int8-rvv-kodiak.zip` | **Kevin** (zephyr-chipyard-sw @ `kodiak`) | deepjscc + spectformer + whisper at 1 and 2 harts RVV, plus **2- and 3-hart scalar**, each in plain and `_debug` form, plus `vlen_probe.elf`. HTIF console over TSI/FESVR, `zvl512b`. |
| `merlin-int8-rvv-gemmelos-bearly25-zephyr.zip` | **Nicolas** (gemmelos-bringup) | deepjscc + spectformer + lstmnetvit + whisper at 1 and 2 harts, plain and `_debug`, plus `vlen_probe.elf`. UART0 @115200, 50 MHz reset clock, `zvl256b`. **Run this one first.** |
| `merlin-int8-rvv-gemmelos-bearly25-zephyr-500mhz.zip` | **Nicolas** (gemmelos-bringup) | The same images with the chip's PLL raised to 500 MHz before the console divisor is applied. Identical computation; only the clock differs. Run it *after* the 50 MHz set works. |
| `merlin-int8-rvv-gemmelos-bearly25-baremetal.zip` | **Nicolas** (gemmelos-bringup) | Single hart, no RTOS, built against our own `crt.S`. The closest match to their own SDK. |

### What changed in this round, and why it matters

**gemmelos: we had the chip's vector width wrong, and it corrupted the kernel.** The whisper debug image
came back with `mcause=5, mtval=0` inside `z_check_stack_sentinel`, loading from address 0. The cause was
ours: the descriptor said VLEN 128 while the chip's own probe reports `vlenb 32` (VLEN 256). Zephyr sizes
its per-thread vector save area as a fixed `vreg[32][VLEN/8]` from *our* number, but fills it with a
length read from the hardware — so every context switch wrote 512 bytes past `z_idle_threads[1]` into
`z_main_thread`, zeroing its name and its stack bookkeeping. A thread that has never run a vector
instruction has a zeroed register file, so the overrun wrote zeros, and the next timer tick dereferenced
one. **No simulation can catch this class of bug**, because a simulator is handed the width we declared;
configured and actual agree there by construction. All gemmelos binaries are rebuilt at 256, the emitted
save area is now floored so an under-declaration cannot overrun, and the packager refuses to build when a
returned probe log disagrees with the descriptor. Note the plain images had the same corruption
*silently* — `STACK_SENTINEL` is a debug-only feature, and it is the only reason this was a diagnosable
fault instead of another unexplained hang.

**The probe was unreadable on a multi-hart chip.** `crt.S` runs `main` on every hart by design, but
nothing serialised the console, so the returned log was interleaved characters — which is how a garbled
`vlenb` line hid the problem above. Harts now print one contiguous block each, lowest first, ending in a
single `DONE`.

**Kodiak `h3`: still unexplained, and the package now contains the binary that settles it.** Every
h3 configuration gates `w8a8` on spike at 3 harts, so nothing we can run reproduces it. The confound is
that every h3 image is also the *scalar* one, so "h3 fails" has never distinguished the third hart from
the scalar multicore path. There is now a **2-hart scalar** image. Run `deepjscc_int8_h2_scalar_debug`:
if it passes and `h3_scalar` fails, the third core is the problem; if both fail, the scalar multicore
route is. One log, one answer.

Also unified: `CONFIG_MP_MAX_NUM_CPUS` was computed by two different formulas depending on whether an
image was going to be simulated, so on this 3-hart board an ungated *one*-hart build declared three CPUs
— and a CPU the image does not need is an unbounded spin in `arch_cpu_start`, i.e. a hang with nothing
past the banner. One rule now, clamped to the board's hart count.

Two earlier board-specific bugs, each found in the boards' own repositories rather than guessed:

- **Kodiak: every multi-hart image hung and every single-hart one passed.** The config set no
  `CONFIG_RISCV_ISA_EXT_V`, so no thread's `mstatus` carried VS and vector state was not saved across a
  context switch — the OpenMP master lost it when pool creation switched it out, and `FPU_SHARING=y`
  routed the resulting trap into the FP retry path, where it spun with nothing printed. The settings now
  match that chip's own working RVV+SMP sample (`samples/q8_gemm_minmax`). **No simulator we have
  reproduces this** — neither spike nor the Saturn RTL enforces `mstatus.VS` — which is why it survived
  every simulated run and why the images now report `METRIC hart<N>_mstatus_vs` for themselves.
- **The scalar images computed the wrong answer.** Per-op register blocking was applied only to vector
  builds, and scalar images were shipped without ever being simulated: deepjscc's scored
  `w8a8_cos 0.9176`. Both halves are fixed — the blocking is unconditional, and nothing ships now
  without a gate behind it.

Also corrected: `build_hash` covered only the model object and weights, so a configuration-only change
left it unmoved and a returned log could not say which binary produced it. And upload estimates assumed
`MemSiz` at 921600 baud for every board — `pyuartsi` sends PROGBITS sections and the Kodiak loader
command uses 57600, so the figures were wrong in both directions. **Raising Kodiak's `--baudrate` to
921600 is the single biggest win available there**: it turns spectformer from a 35-minute upload into a
2-minute one.

## Reading a returned log

Each zip's `grade.py` scores a console log with only numpy. On a log that stopped early it now reports
the last `STAGE`, any `FAIL fatal` line and any failed memory probe, instead of only saying the run did
not complete. `<model>_h<N>.op_table.json` decodes the `PROF <id>` trace and the `op=<id>` field of
`ALIVE`, so a stalled run names the operator it stopped in.

## DO NOT SEND

`superseded-*/` hold earlier packages, kept so the before/after is checkable rather than only described.
They are **not** deliverables. Most recent is
`superseded-20260806-pre-vector-state-fix/`, which holds the packages whose multi-hart images hung on
Kodiak and whose scalar images computed the wrong answer — worth keeping because a log someone mails
back may have come from one of them, and `METRIC build_hash` is how you tell.

It also holds the short-lived separate `merlin-debug-*` zips. Those are retired not because they were
wrong but because splitting plain and diagnostic builds across two downloads was the wrong shape: the
binary you need when something goes wrong is the one you do not have. Both builds now live in every zip.

To tell any two packages apart from the binaries themselves:

```bash
riscv64-unknown-elf-nm <elf> | grep -c tohost        # gemmelos: 0 = send, 1 = superseded
riscv64-unknown-elf-nm <elf> | grep -c uart_sifive   # gemmelos: non-zero = the chip's own UART
```

**Not** `readelf -S | grep .htif`. Zephyr allocates an empty `NOBITS` `.htif` section for its reboot
path regardless of which console is selected, so the current gemmelos images carry one too — an earlier
version of this file said `0 = send, 1 = superseded` and would have pointed at the wrong package. The
`tohost` symbol is the honest discriminator: an HTIF image spins on that word, a UART image has no
reference to it.

On Kodiak a `.htif` section is **expected and required** — that board's DTS selects `&htif`, its
defconfig disables the SiFive UART port, and `pyuartsi --fesvr` serves `tohost`/`fromhost` for the whole
run. Do not "fix" the Kodiak console.

## The per-board directories

`chipyard_kodiak/`, `gemmelos_bearly25/`, `gemmelos_bearly25_zephyr/` are the timestamped
`new_product()` outputs from individual packager runs (build trees and manifests). The zips above are
what was assembled for sending.
