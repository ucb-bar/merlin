# Delivery packages

Every zip below carries **two builds of every image**: the plain one to run for a number, and a
`_debug_` twin of the same model that explains itself while it runs. Send the whole zip — the point of
the pairing is that nobody has to come back to us for the other half.

## SEND THESE

| zip | for | contents |
|---|---|---|
| `merlin-int8-rvv-kodiak.zip` | **Kevin** (zephyr-chipyard-sw @ `kodiak`) | deepjscc + spectformer at 1 and 2 harts RVV, plus 3-hart scalar, each in plain and `_debug` form, plus `vlen_probe.elf`. HTIF console over TSI/FESVR, `zvl512b`. |
| `merlin-int8-rvv-gemmelos-bearly25-zephyr.zip` | **Nicolas** (gemmelos-bringup) | deepjscc + spectformer + lstmnetvit at 1 and 2 harts, plain and `_debug`, plus `vlen_probe.elf`. UART0 @115200, 50 MHz reset clock. **Run this one first.** |
| `merlin-int8-rvv-gemmelos-bearly25-zephyr-500mhz.zip` | **Nicolas** (gemmelos-bringup) | The same images with the chip's PLL raised to 500 MHz before the console divisor is applied. Identical computation; only the clock differs. Run it *after* the 50 MHz set works. |
| `merlin-int8-rvv-gemmelos-bearly25-baremetal.zip` | **Nicolas** (gemmelos-bringup) | Single hart, no RTOS, built against our own `crt.S`. The closest match to their own SDK. |

### What changed in this round, and why it matters

Two board-specific bugs, each found in the boards' own repositories rather than guessed:

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
riscv64-unknown-elf-nm <elf> | grep -c uart_sifive_data_0   # 1 = the chip's own UART (gemmelos: send)
riscv64-unknown-elf-readelf -S <elf> | grep -c '\.htif'     # gemmelos: 0 = send, 1 = superseded
```

On Kodiak a `.htif` section is **expected and required** — that board's DTS selects `&htif`, its
defconfig disables the SiFive UART port, and `pyuartsi --fesvr` serves `tohost`/`fromhost` for the whole
run. Do not "fix" the Kodiak console.

## The per-board directories

`chipyard_kodiak/`, `gemmelos_bearly25/`, `gemmelos_bearly25_zephyr/` are the timestamped
`new_product()` outputs from individual packager runs (build trees and manifests). The zips above are
what was assembled for sending.
