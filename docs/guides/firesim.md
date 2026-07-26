---
title: FireSim — whole-model cycle truth on the FPGA
kind: guide
status: current
owner: runtime
last_verified: 2026-07-26
related: [zephyr, tinyllama_int8_rvv_zephyr, getting_started, reproducibility]
code_refs: [merlin/python/merlin/runtime/backends/zephyr_model.py, build_tools/chipyard/setup_multicore_saturn.py, build_tools/chipyard/MerlinSaturnConfigs.scala, build_tools/scripts/firesim_sweep.py, build_tools/scripts/fsq.py, build_tools/firesim/README.md, build_tools/firesim/preflight.py]
---

# FireSim — whole-model cycle truth on the FPGA

## What FireSim is for here

Merlin has several rungs of execution truth, and each one runs out of road somewhere:

| Rung | Gives you | Runs out at |
|---|---|---|
| host / numpy | functional truth, instantly | no cycles at all |
| spike | functional truth on the real ISA | no timing (functional model) |
| Verilator | cycle-accurate RTL | **~10⁴ cycles/s** — a whole-model inference is ~10¹⁰ cycles, i.e. weeks |
| **FireSim** | cycle-accurate RTL, whole model | hours per model; one FPGA, shared |
| SpacemiT K1 board | wall-clock on real silicon | not *our* SoC — fixed vendor design |

FireSim is the only rung where **our own RTL** executes a **whole model**. Verilator certifies
the mechanism (a kernel, a dispatch, a multicore hand-off); the K1 answers "is it fast on
silicon someone else built". Only FireSim answers "how many cycles does the SoC we designed
take on the model we care about".

The rate is the whole point. Measured on this host on 2026-07-26, with the dual-Saturn
bitstream: **25.08 MHz effective** (3.67 G target cycles in 148 s, read off `heartbeat.csv`).
That is roughly 2500× Verilator. A 10¹⁰-cycle inference is ~7 minutes; a 10¹¹-cycle one is
~1.1 hours. Both are impossible in RTL simulation and routine here.

Do not carry that number forward as a constant. The effective clock is a property of the
design + bitstream + host, and it is *measured per run* (see [Reading a run](#reading-a-run)).
An earlier single-vector-unit bitstream ran considerably slower.

## What is verified, and what is not

Be exact about this before planning work on it.

| Claim | Status |
|---|---|
| The dual-Saturn bitstream exists, is registered in hwdb, and is the current `default_hw_config` | **verified** (config files read 2026-07-26) |
| It closes timing at 25 MHz and the FPGA sustains ~25 MHz effective | **verified** — recipe `fpga_frequency: 25`, heartbeat slope 25.08 MHz |
| 28.65% LUT utilisation (495k/1.73M), FF 6.7%, BRAM 8.9%, DSP 1.7%, WNS +0.034 ns | reported from the build; **not re-derived here** |
| Whole models run end-to-end on FireSim through the **Zephyr** path, accuracy-gated | **verified**, RVV backend, `small_llama` int8: 1 hart `cos=0.9999999`, 362,862,321 cycles |
| The **RVV** backend on a real Saturn tile under Zephyr | **verified** (2026-07-26) — see below; this was previously recorded as unsolved |
| **Multicore** RVV (OpenMP over 2 pinned harts, both Saturn tiles) | **verified** — `cos=0.9999999`, 230,548,059 cycles, **bit-identical** to the 1-hart run, **1.574×** |
| The **bare-metal** ELF path on FireSim (`build_tools/firesim/firesim_baremetal.ld`) | **not run on the FPGA** — derived from the validated spike layout |
| Gemmini / capsule-bench L5 FireSim oracle | **not wired** — `heavy_oracles.firesim_adapter` builds the ELF and then reports `OracleUnavailable` rather than fabricate a result |
| Multi-GB (>2 GB) images booting and loading on the FPGA | links, but **HW-unconfirmed** |

## Prerequisites

Complete the base install in [Getting started](getting_started.md), then set these in `.env`
(never export-only — merlin resolves them through `merlin.common.paths.env`, which reads `.env`):

| Setting | What it points at | Consequence if wrong |
|---|---|---|
| `MERLIN_CHIPYARD` / `MERLIN_EXT_CHIPYARD` | the chipyard checkout | no FireSim manager, no configs |
| `MERLIN_EXT_FIRESIM_QUEUE` | the shared job queue directory | runs bypass the queue and collide on the single FPGA |
| `MERLIN_MODELBLASTER` | the ModelBlaster checkout, whose `validation/firesim_runner.py` merlin reuses | `ModuleNotFoundError: No module named 'modelblaster'` — an error naming neither the setting nor the path |
| `ZEPHYR_BASE`, `MERLIN_ZEPHYR_SW`, `ZEPHYR_SDK_INSTALL_DIR` | the Zephyr workspace and SDK | cannot build the ELF to run |

Then check the host, read-only — this submits nothing and does not touch the FPGA:

```bash
.venv/bin/python build_tools/firesim/preflight.py
```

It verifies the chipyard checkout, the ModelBlaster runner, the queue **daemon's liveness**,
the XDMA devices, the `default_hw_config` → hwdb → bitstream-tar chain, and prints the
effective clock of the most recent run. `check_repro_env.py` also reports a `firesim`
capability, but its liveness test only checks that `daemon.pid` *exists* — which stays true
for weeks after the daemon dies. Prefer the preflight.

## Installing the Saturn configs

No stock chipyard config gives every tile a vector unit: the Saturn configs are single-core and
the one multi-tile SoC builds its vector unit on tile 1 only. Merlin's configs live in-repo and
are installed into the out-of-repo chipyard checkout:

```bash
.venv/bin/python build_tools/chipyard/setup_multicore_saturn.py --check   # report only
.venv/bin/python build_tools/chipyard/setup_multicore_saturn.py          # install
```

Idempotent — re-running detects what is already present and changes nothing. It installs
`chipyard.DualSaturnV256D128ShuttleConfig` / `MultiSaturnV256D128ShuttleConfig` (2 and 4
Shuttle tiles, each with its own Saturn unit at vLen=256/dLen=128 — matching the K1, so a
schedule tuned on the board transfers without a re-tune), the matching `FireSim*` target
configs appended to firechip's `TargetConfigs.scala`, and the Alveo U250 build recipes appended
to `sims/firesim/deploy/config_build_recipes.yaml`.

## Building a bitstream

Hours of Vivado, unattended. Add the recipe name to `builds_to_run` in
`$MERLIN_CHIPYARD/sims/firesim/deploy/config_build.yaml`, then:

```bash
cd $MERLIN_CHIPYARD/sims/firesim && source sourceme-manager.sh --skip-ssh-setup
cd deploy && ./firesim buildbitstream
```

The recipe fields matter more than they look. A recipe **must** carry
`metasim_customruntimeconfig` and `bit_builder_recipe` or the build dies immediately with
`KeyError: 'bit_builder_recipe'` — before any synthesis, but also before any useful message.
The installed recipes already have them; copy one rather than writing a new one from the
FireSim docs.

`fpga_frequency` is deliberately conservative (25 MHz for the dual-Saturn recipe, against the
30 MHz an earlier single-vector-unit bitstream used). Several vLen=256 vector units are a large
timing step up, and a frequency that will not close wastes the entire run — which you discover
hours in.

## Registering the bitstream

When the build finishes it prints an hwdb entry. Paste it into
`sims/firesim/deploy/config_hwdb.yaml` (name → `bitstream_tar: file:///…/firesim.tar.gz`), then
point `config_runtime.yaml`'s `default_hw_config` at it.

> `config_runtime.yaml` is **shared state** on this machine — one file selects the bitstream for
> everyone. Back it up before changing it and restore it afterwards. Someone else's run failing
> mysteriously an hour later is the usual symptom of forgetting.

`preflight.py` reports the current selection and whether its tar still exists.

## Running — always through the queue

There is **one** physical FPGA. Every run goes through the job queue; a direct `firesim
runworkload` collides with whatever is already on the board.

```bash
Q=$MERLIN_EXT_FIRESIM_QUEUE           # /scratch2/agustin/firesim_queue on this host
$Q/bin/firesim-queue status           # active jobs
$Q/bin/firesim-queue status --all     # including terminal
$Q/bin/firesim-queue tail <job_id> -f
$Q/bin/firesim-queue cancel <job_id>
$Q/bin/firesim-queue daemon           # start the daemon; leave it running
$Q/bin/firesim-queue stop-daemon
```

The daemon holds an `flock` on `fpga.lock` for the whole duration of a job and dispatches by
priority, then round-robin across users. **It must be alive** — nothing moves otherwise, and the
queue simply fills up silently. It has been found down for two weeks with jobs waiting behind
it. Before starting it, run `status` and look for work that has been PENDING since before you
arrived; that is someone else's job about to start the moment you bring the daemon up.

From merlin, the queue is the default and needs no argument:

```python
from merlin.runtime.backends import zephyr_model as zm
b = zm.build_app(model_dir, work, board="chipyard_riscv64", backend="scalar", cpus=2)
r = zm.run_on_firesim(b["elf"], reference=golden, timeout=5400)   # queue=True by default
```

`run_on_firesim` sets `FIRESIM_QUEUE=1` and `FIRESIM_QUEUE_TIMEOUT`, tags jobs with
`FIRESIM_PROJECT=merlin-oscar` so they are distinguishable from other workflows on the shared
queue, runs under the `merlin-oscar` workload definition (`deploy/workloads/merlin-oscar.json`,
`common_bootbinary: zephyr0-zephyr.elf`), and repairs `SSH_AUTH_SOCK` when the submitting
session's agent is dead — the queue records the *submitter's* environment and the daemon later
runs `firesim kill`, which SSHes to localhost.

Mind the **timeout**. `run_on_firesim` defaults to 900 s, which is a small model. Whole models
are hours: at 25 MHz, cycles/2.5×10⁷ = seconds. `firesim_sweep.py` defaults to 5400 s and takes
`--timeout`.

For a batch:

```bash
.venv/bin/python build_tools/scripts/firesim_sweep.py BUNDLE [BUNDLE ...] --timeout 25200
.venv/bin/python build_tools/scripts/fsq.py            # queue, with model names resolved
```

`firesim_sweep.py` builds each `chipyard_riscv64` image locally, submits the run to the queue,
gates `cos` against `golden.npy`, and appends to a JSONL ledger so a re-run skips what already
passed. `fsq.py` exists because the native `status` labels every merlin job by its staged
bootbinary (`zephyr0-zephyr.elf`) — indistinguishable across model × dtype — so it recovers the
bundle name from each job's `stage_from` path.

## Reading a run

FireSim writes into the run farm's `default_simulation_dir` (read it from
`config_runtime.yaml`; `/scratch2/agustin/FIRESIM_RUNS_DIR` here), under `sim_slot_0/`:

**`uartlog`** — the full console capture, from PCIe/XDMA discovery through the target's own
output. Merlin's Zephyr app prints markers the tooling parses:

```
*** Booting Zephyr OS build 852bb170cc56 ***
=== merlin_zephyr hart=0 ===
METRIC iter_cycles 0 22377479
OUT 2048 3202454993 3195417912 ...
=== MODELBLASTER_WALL_CYCLES === <n>
DONE
```

`MODELBLASTER_WALL_CYCLES` is the terminal marker the runner waits for; `OUT`/`METRIC`/`DONE`
are what merlin parses and gates on. A run that reaches `=== merlin_zephyr hart=… ===` and then
goes quiet has *started* correctly and is either slow or wedged — which the heartbeat tells you.

**`heartbeat.csv`** — two columns, `target cycles` and `seconds since start`:

```
Target Cycle (fastest), Seconds Since Start
158965925, 8
...
3030811510, 123
```

Divide a span to get the effective clock — `(3030811510 − 158965925)/(123 − 8) = 25.0 MHz`
here. Two things follow, and they are the whole diagnostic value of the file:

- **Slow vs hung.** If cycles are still advancing, the run is alive; estimate the remaining
  wall clock from the model's known cycle count and stop guessing. If the last sample is old
  and the count is frozen, it is hung.
- **A sanity check on the design.** An effective clock far below the recipe's `fpga_frequency`
  means host-side stalls, not a slow target.

`preflight.py` prints the slope of the most recent run for you.

## Link-time artifacts

See [`build_tools/firesim/README.md`](../../build_tools/firesim/README.md). The one thing to
carry in your head: the **Zephyr** path takes **no linker script** from that directory —
`zephyr.lds` owns the link, and a large weights blob is placed by a devicetree memory region
that `zephyr_model.py` emits. The linker script and specs there are for the **bare-metal** path
only, which has not run on the FPGA yet.

The shared constraint behind both is `-mcmodel=medany`: every reference is PC-relative within
±2 GB, so a multi-GB weights blob linked next to the code pushes ordinary symbols out of reach
and the failure is *silent* — correct arithmetic on the wrong bytes. Both paths answer it the
same way: keep code and data compact, put the blob somewhere absolute, and reach it by a
literal rather than a relocation.

## Troubleshooting

Ordered roughly by how much time each one costs before you work out what it was.

**`ModuleNotFoundError: No module named 'modelblaster'`.** `MERLIN_MODELBLASTER` is unset or
wrong. The message names neither the setting nor the path it wanted. `run_on_firesim` resolves
it through `.env`, so it does not need exporting — but it does need to be *there*.

**`insmod: ERROR: could not load module poll_mode=1` at INFRASETUP.** The XDMA kernel module is
not loaded. FireSim's helper searches for a literal `xdma.ko`; a modern kernel ships
`xdma.ko.zst`, so the search comes back empty and `poll_mode=1` — the *argument* — is mistaken
for the module path. Check with `lsmod | grep xdma` or `ls /dev/xdma0_*`; loading the module is
a host-admin action, not something merlin does.

**Jobs go QUEUED → INFRASETUP → FAILED at ~32 s, with "Needed to prompt for a connection or sudo
password (host: localhost)".** FireSim's fabric manager does not read `~/.ssh/config` unless
`env.use_ssh_config` is set, so a `Host localhost` key block is inert and authentication falls
through to a password prompt it cannot answer. The fix is `env.use_ssh_config = True` in
`deploy/firesim`'s `main()` — already applied in this checkout (line ~465). If `ssh localhost`
works but fabric does not, this is why.

**Nothing runs and nothing fails.** The queue daemon is dead. `daemon.pid` outliving the process
is normal, so a pid file is not evidence — `preflight.py` checks the process itself.

**A run times out with an empty `uartlog`.** The image never booted. The usual cause is an
over-large `ram0` devicetree overlay: at the stock 256 MB a model boots, and the same model with
`ram0` grown to a few hundred MB wedges before any output. `zephyr_model.py` only emits the
`&ram0` override when the model genuinely needs more than the default. Note also
`zero_out_dram: false` — never rely on DRAM being zero.

**A run hangs with no fault printed.** Several distinct causes, all now identified — none of them
is "RVV does not work on FireSim", which is what the symptom used to be recorded as.

- `CONFIG_FPU_SHARING=y` mis-routes a V-illegal-instruction trap to the FP path, which retries
  forever, silently. merlin's generated `prj.conf` sets `=n`.
- `CONFIG_UART_HTIF_BUFFERED_OUTPUT=y` + `CONFIG_UART_HTIF_SYSCALL_PRINT=y`: direct-putchar HTIF
  races under SMP and wedges.
- **Multicore only:** the OpenMP shim used to hang on *nested* parallel regions, and then to
  corrupt the master's stack once that was fixed. Both are fixed (`bf5052db`); the signature to
  recognise is a 22.4M-cycle model running past 11.4 **billion** cycles with nothing after the
  pool banner. If you see that again, build with `MERLIN_OMP_DEBUG_SPLIT=1` — the shim now reports
  which worker it is waiting on and what that worker last published, and gives up on a wall-clock
  deadline instead of spinning forever.

The RVV backend is the working FireSim path as of 2026-07-26, single- and multi-hart.

**`KeyError: 'bit_builder_recipe'` at buildbitstream.** The recipe is missing
`bit_builder_recipe` and/or `metasim_customruntimeconfig`. Copy a working recipe.

**Someone else's results appear, or yours vanish.** `config_runtime.yaml` is shared, and so is
the queue. Check `default_hw_config` is still what you set, and use `fsq.py` to see whose jobs
are actually on the board.
