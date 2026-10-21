---
title: Run whole programs on FireSim
kind: guide
status: current
owner: runtime
last_verified: 2026-09-29
related: [zephyr, tinyllama_int8_rvv_zephyr, getting_started, reproducibility, whole_model_on_accelerator]
code_refs: [src/merlin/runtime/backends/zephyr_model.py, src/merlin/runtime/backends/firesim_runner.py, src/merlin/perf/firesim_receipt.py, src/merlin/perf/firesim_checkpoint.py, build_tools/chipyard/setup_multicore_saturn.py, build_tools/chipyard/MerlinSaturnConfigs.scala, build_tools/scripts/firesim_sweep.py, build_tools/scripts/fsq.py, build_tools/firesim/README.md, build_tools/firesim/preflight.py, examples/firesim_modelblaster/README.md]
---

# Run whole programs on FireSim

For a workhorse that drives a remote manager through a locked SSM document, use
the [post-freeze workhorse workflow](../../examples/gemmini/validation/README.md).
That deployment offers direct `~/bin/fpga-run-elf ELF NAME` / `FsimRunElf`
transport and the legacy `~/bin/fpga-run WORKLOAD` interface. It has no local
queue; the local queue/manager commands below do not apply to it. Each direct
ELF contains the complete model chain; upload removes per-model registration,
not frozen-input, uploaded-byte, manager-result or numerical proof requirements. The workflow
prepares source-bound requests after either phase's compiler freeze and imports
observations without replacing the model checker's numerical or placement gates.

## Execution and timing scope

FireSim executes a selected RTL design on an FPGA. Use it for frozen complete
program checks when the required runtime and design are qualified. Functional
simulators such as Spike provide ISA/numerical evidence; CPU RTL engines provide
selected RTL execution and counter evidence. Their wall costs depend on the
actual design, program and worker, so measure tool cost before choosing a tier.

Bind every result to its exact design, configuration, bitstream, executable,
inputs, output checks and timer boundary. Cycles from different host cores or
memory configurations are not interchangeable. Measure effective simulation
rate per run from the heartbeat; a historical rate is not a new worker's limit.

Build or runner-selection checks do not qualify hardware execution. Keep model
results and deployment qualification records in generated run artifacts. A
missing bitstream, engine, output check or required oracle remains unavailable.

## Prerequisites

Complete the base install in [Getting started](getting_started.md), then set these in `.env`
(never export-only — merlin resolves them through `merlin.common.paths.env`, which reads `.env`):

| Setting | What it points at | Consequence if wrong |
|---|---|---|
| `MERLIN_CHIPYARD` / `MERLIN_EXT_CHIPYARD` | the chipyard checkout | no FireSim manager, no configs |
| `MERLIN_EXT_FIRESIM_QUEUE` | the shared job queue directory | runs bypass the queue and collide on the single FPGA |
| `MERLIN_FIRESIM_RUNNER` | explicitly selected installed runner name, e.g. `modelblaster` | Merlin refuses to submit without a selected provider |
| `MERLIN_MODELBLASTER` | ModelBlaster checkout, only for the optional example adapter | the adapter preflight names the missing checkout or runner source |
| `ZEPHYR_BASE`, `MERLIN_ZEPHYR_SW`, `ZEPHYR_SDK_INSTALL_DIR` | the Zephyr workspace and SDK | cannot build the ELF to run |

Then check the host, read-only — this submits nothing and does not touch the FPGA:

```bash
.venv/bin/python build_tools/firesim/preflight.py
```

It verifies the chipyard checkout, the selected runner (including its optional read-only preflight), the queue **daemon's liveness**,
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

On a shared FPGA deployment, submit every run through its job queue. Direct
`firesim runworkload` can collide with another user's active job.

```bash
Q=$MERLIN_EXT_FIRESIM_QUEUE           # set per host; see .env.example
$Q/bin/firesim-queue status           # active jobs
$Q/bin/firesim-queue status --all     # including terminal
$Q/bin/firesim-queue tail <job_id> -f
$Q/bin/firesim-queue cancel <job_id>
$Q/bin/firesim-queue daemon           # start the daemon; leave it running
$Q/bin/firesim-queue stop-daemon
```

The daemon holds an `flock` on `fpga.lock` for the whole duration of a job and dispatches by
priority, then round-robin across users. **It must be alive** — nothing moves otherwise, and the
queue can retain pending jobs without execution. Before starting it, inspect
`status`; existing pending work may start as soon as the daemon is available.

Install the optional [ModelBlaster runner example](../../examples/firesim_modelblaster/README.md)
or a separately maintained runner distribution, and select its entry-point name
with `MERLIN_FIRESIM_RUNNER` in `.env`. Merlin's core wheel contains no runner.
This is a breaking migration from older checkouts: `run_on_firesim` no longer
imports ModelBlaster implicitly, and generic `build_app` no longer embeds its
terminal marker. Existing FireSim callers must install/select a runner and
pass its `completion_metric_prefix` when building the ELF.
From Merlin, the queue is the default and needs no argument:

```python
from merlin.runtime.backends import zephyr_model as zm
from merlin.runtime.backends.firesim_runner import select_runner
runner = select_runner()  # MERLIN_FIRESIM_RUNNER from .env or process environment
b = zm.build_app(model_dir, work, board="chipyard_riscv64", backend="scalar", cpus=2,
                 completion_metric_prefix=getattr(runner, "completion_metric_prefix", None))
r = zm.run_on_firesim(b["elf"], reference=golden, timeout=5400, runner=runner)
```

The image carries the selected runner's terminal marker. The example runner refuses
an image without it before submitting work. The example runner sets `FIRESIM_QUEUE=1`
and `FIRESIM_QUEUE_TIMEOUT`, tags jobs with
`FIRESIM_PROJECT=merlin-oscar` so they are distinguishable from other workflows on the shared
queue, runs under the `merlin-oscar` workload definition (`deploy/workloads/merlin-oscar.json`,
`common_bootbinary: zephyr0-zephyr.elf`), and repairs `SSH_AUTH_SOCK` when the submitting
session's agent is dead — the queue records the *submitter's* environment and the daemon later
runs `firesim kill`, which SSHes to localhost.

These settings belong to the optional ModelBlaster adapter, not the Merlin core.
`run_on_firesim` resolves the explicitly selected installed entry point; missing or duplicate
providers fail before submission. A caller can also pass `runner_name=` or `runner=` to
`run_on_firesim`. The callable receives the ELF
path and keyword arguments `firesim_root`, `firesim_env`, `timeout`, and `queue`, and returns raw
UART text. The out-of-tree runner owns staging, workload/project identity, queue submission, and
simulator imports; Merlin alone parses `OUT`/`METRIC`/`DONE` and applies the reference gate. A
runner must honor `queue=True` on shared hardware. Passing one does not by itself certify that it
used the queue or that the expected bitstream executed.

Mind the **timeout**. `run_on_firesim` defaults to 900 s, which is a small model. Whole models
are hours: at 25 MHz, cycles/2.5×10⁷ = seconds. `firesim_sweep.py` defaults to 5400 s and takes
`--timeout`.

For a batch:

```bash
.venv/bin/python build_tools/scripts/firesim_sweep.py BUNDLE [BUNDLE ...] --runner modelblaster --timeout 25200
.venv/bin/python build_tools/scripts/fsq.py            # queue, with model names resolved
```

`firesim_sweep.py` builds each `chipyard_riscv64` image with the selected runner's
terminal marker, submits the run to the queue,
gates `cos` against `golden.npy`, and appends to a JSONL ledger so a re-run skips what already
passed. `fsq.py` exists because the native `status` labels every merlin job by its staged
bootbinary (`zephyr0-zephyr.elf`) — indistinguishable across model × dtype — so it recovers the
bundle name from each job's `stage_from` path.

### The four-step sequence is non-negotiable

Every FireSim run is **`kill` -> `infrasetup` -> `runworkload` -> `kill`**, in that order, always —
including when the bitstream has not changed. Skipping the leading `kill` or the `infrasetup` is what
leaves a stale simulator attached to `/dev/xdma0_*`, and the next `runworkload` then takes the host
down; the reset looks like a build or bitstream fault rather than a dirty device.

You do not run those steps yourself. `firesim-queue runworkload-full` **is** the sequence — it is the
only client entry point, and the tool deliberately exposes no bare `infrasetup`/`kill` subcommands so
it cannot be partially executed. Reference implementation:
`_run_one_job_runworkload_full()` in the queue's `firesim_queue.py` —

| phase | what the daemon runs | guarantee |
|---|---|---|
| `STAGING` | copy `--stage-from` to `workloads/<w>/<boot>` | happens **under the lock**, so nothing swaps the ELF mid-run |
| `INFRASETUP` | `firesim kill` (best-effort), then `firesim -c <per-job yaml> infrasetup` | unconditional; a non-zero rc aborts before the FPGA is touched |
| `RUNNING` | `firesim runworkload` | reached only if infrasetup returned 0 |
| `TEARDOWN` | `firesim kill` | in a `finally:` — runs on **every** path: success, failure, cancel, timeout |

All four sit inside one continuously-held `fcntl.flock(fpga.lock, LOCK_EX)`. That trailing kill is why
the next run's leading kill is normally a no-op. `--timeout` bounds only the `RUNNING` phase, so a long
`infrasetup` is never cut off part-way.

### `--hw-config` — which bitstream the job flashes

**Pass it whenever more than one design is registered.** Without it, the design comes from the shared
`deploy/config_runtime.yaml::target_config.default_hw_config` — global mutable state. Two users wanting
different designs race, and the loser *silently runs the other's hardware*: a gemmini job executing on
whatever the template last pointed at, with no error and plausible-looking output.

```bash
$Q/bin/firesim-queue runworkload-full \
    --chipyard "$MERLIN_EXT_CHIPYARD" --workload merlin-perfbench \
    --bootbinary merlin-perfbench.elf --stage-from /path/to/your.elf \
    --hw-config alveo_u250_firesim_shuttle_gemmini_opu \
    --priority 5 --project my-project --timeout 900
```

The final promoted ResNet-50 measurement pins the raw client path to
`$MERLIN_EXT_FIRESIM_QUEUE/bin/firesim-queue`. Submit exactly one `runworkload-full` job. The
ELF performs its unmeasured warm invocation and its one measured invocation internally; submitting
separate warm and measured queue jobs would discard the warmed process state. The final receipt must
record the queue job ID and content hashes for the client log, daemon job log, and UART log. It is
invalid if it came from a direct `firesim` invocation, if the daemon evidence does not show the exact
`kill -> infrasetup -> runworkload -> kill` lifecycle, or if the UART exposes any profile metric
other than the single measured compute-cycle count.

Build that ELF through the explicit production profile option; an environment-only cache-state knob
is not the final-measurement contract:

```python
from merlin.perf.execution_policy import WarmProfileContract
from merlin.targetgen.contract.compile import compile_lowered_to_elf

elf = compile_lowered_to_elf(
    command_buffer, lowered_mlir, output_dir, target=target,
    warm_profile=WarmProfileContract(),
)
```

The opt-in accepts only one warm run, one measured run and `total_compute_cycles`. It bypasses the
legacy ELF cache because the harness protocol is not part of that cache's historical key. The target's
`HarnessAbi` supplies launch and completion; result `OUT` lines and `DONE` follow the one
`METRIC cycles N`, outside the measured window. The same option composes with explicit host prepacking
and the verified compact-pointer caller.

### Sealing a completed queue run

`merlin-firesim-receipt` is the fail-closed, post-run boundary. It runs no queue or simulator
command. Give it absolute paths to the captured queue-client output, the job's own daemon
`stdout.log`, and the resulting UART log. Also give it the raw queue executable and the exact argv
that was submitted as a JSON string array. For example, `submission.json` contains:

```json
[
  "/absolute/queue/bin/firesim-queue",
  "runworkload-full",
  "--chipyard", "/absolute/chipyard",
  "--workload", "model-evaluation",
  "--stage-from", "/absolute/model.elf"
]
```

Correctness is not hardcoded in the parser. The target/workload adapter emits a policy with exact
successful-validation lines. A digest is a useful marker because it binds the observed output rather
than merely saying that execution ended:

```json
{
  "schema": "merlin_firesim_uart_validation_policy_v1",
  "policy_id": "target-contract/exact-output-digest-v1",
  "workload": "model-evaluation",
  "success_markers": [
    "VALIDATION output_digest=0123456789abcdef",
    "VALIDATION status=PASS"
  ]
}
```

Then seal the run:

```bash
merlin-firesim-receipt \
  --queue-client-log /absolute/evidence/queue-client.log \
  --queue-daemon-log /absolute/queue/jobs/123/stdout.log \
  --uart-log /absolute/evidence/uartlog \
  --queue-executable /absolute/queue/bin/firesim-queue \
  --submission-json /absolute/evidence/submission.json \
  --job-id 123 \
  --workload model-evaluation \
  --validation-policy /absolute/evidence/validation-policy.json \
  --output /absolute/evidence/firesim-receipt.json
```

The command refuses unless the client terminates that exact job as `DONE` with exit code zero; the
daemon log proves exactly `kill -> infrasetup -> runworkload -> kill` in its ordered phases; and the
UART contains one successful warm window before one successful measured window. There must be one
positive `METRIC cycles N` line inside the measured protocol window and no other metrics. Every
policy marker must appear exactly once after `measured begin` and before the metric is published.

The JSON is deterministic and replaced atomically. It content-hashes the three logs, the raw queue
executable, the submission file, its canonical argv, and the validation policy. Serialization
rechecks every file, so evidence modified after parsing is refused. This proves agreement among the
captured artifacts; it cannot reconstruct an argv that the installed queue client did not echo, so
capture `submission.json` before invocation and preserve it beside the client log.

The daemon rewrites that one key in the per-job `config_runtime.yaml` and leaves every other setting —
and the shared template — untouched. Omit the flag to inherit the template, which is the older
behaviour. Registered designs come from `deploy/config_hwdb.yaml`; **check which one the template
defaults to before assuming**, because it is frequently not the one you want.

### A checkpoint: many programs, one sealed table

`merlin-firesim-checkpoint` is the half before the receipt, and the loop around it. Given a manifest
of ELFs it builds the one permitted submission for each (always naming `--hw-config`), refuses if the
installed queue does not accept a flag it is about to pass, refuses while the shared queue is busy,
runs the jobs one after another, copies each job's client log, daemon log and UART out of the
queue's own tree, and seals each with the receipt above. The output is a `checkpoint.json` whose
rows are sealed receipts or `not_sealed` with the reason; a row without a receipt carries no cycle
count. `--previous` compares against an earlier checkpoint and refuses when the two were not
measured on the same design.

```yaml
schema: merlin_firesim_checkpoint_manifest_v1
checkpoint: <name>
host:
  queue_executable: <raw client path, not a PATH symlink>
  queue_state_root: <where the daemon writes jobs/<id>/>
  chipyard: ext:chipyard            # ext:<key> reads .env; anything else is a literal path
design:
  pin: <hardware_pins.yaml entry>
  hw_config: <hwdb key>
  hwdb_config_artifact: <one-entry hwdb file>   # evidence jobs should pass it
  substrate: firesim_<bitstream>
workload: {name: <workload>, bootbinary: <elf basename>}
entries:
  - {label: ..., model: ..., experiment: ..., elf: ..., validation_policy: ..., timeout_s: 1800}
client_env:                          # host policy; omit when you run the daemon yourself
  path_prefix: [<dir holding the queue cwd launchers>]
  drop: [HOME, USER, LOGNAME]
```

`client_env` exists for a daemon run by another account. That daemon cannot write into your
checkout, so the queue gives each job a private deploy overlay; the FireSim manager then resolves
its deploy root from the path it was found on and walks straight back out of it. The first symptom
is `PermissionError` opening `deploy/logs/...` during the leading `kill`, before the FPGA is
touched. The launchers under `gemmini_perf_bench/scripts/firesim_queue_cwd_launcher/` preserve the
daemon's cwd, and dropping the identity variables keeps the daemon on a home it can read.

Validate without touching the FPGA with `--dry-run`. Each job pays its own bitstream flash: the queue
flashes once per job and a job boots one binary. Several programs per flash therefore means several
programs linked into ONE boot binary. A queue job that itself calls `firesim infrasetup` or
`firesim runworkload` in a loop is the pattern the queue exists to prevent.

### Before you submit

Check the queue and executable before submission:

1. **`firesim-queue status` must say `daemon: ALIVE`.** A dead daemon strands every submission as
   `QUEUED` until a daemon is available. Pending state alone does not indicate
   that simulation has started.
2. **Look for an orphaned `RUNNING` row.** A daemon that dies mid-job leaves one behind permanently.
   Check `cancel_requested` and `pid` in `queue.db`: a row with `pid: None` and a cancel already
   requested is a zombie its owner asked to end, and clearing it (`cancel <id> --force`) completes
   their request rather than killing live work. Never assume a `RUNNING` row means a running job.
3. **Pre-flight the ELF**, seconds well spent against an hour of simulation: `readelf -lW` for LOAD
   segments inside the design's DRAM, and `objdump -d | grep -c` for instructions the core does not
   implement. See [whole-program build and execution](whole_model_on_accelerator.md).

A daemon this host runs cannot write another user's job dir; it logs `not runnable by this daemon …
skipping it and continuing` and moves on. Their queued jobs neither run nor block yours.

## Reading a run

FireSim writes into the run farm's `default_simulation_dir` (read it from
`config_runtime.yaml`; `$FIRESIM_RUNS_DIR` here), under `sim_slot_0/`:

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

`MODELBLASTER_WALL_CYCLES` is the optional example runner's terminal marker; generic
Merlin images do not embed it. `OUT`/`METRIC`/`DONE` are what Merlin parses and gates on.
A run that reaches `=== merlin_zephyr hart=… ===` and then
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
only. Each selected path needs its own deployment qualification.

The shared constraint behind both is `-mcmodel=medany`: every reference is PC-relative within
±2 GB, so a multi-GB weights blob linked next to the code pushes ordinary symbols out of reach
and the failure is *silent* — correct arithmetic on the wrong bytes. Both paths answer it the
same way: keep code and data compact, put the blob somewhere absolute, and reach it by a
literal rather than a relocation.

## Troubleshooting

Ordered roughly by how much time each one costs before you work out what it was.

**No selected FireSim runner.** Install a runner package and set `MERLIN_FIRESIM_RUNNER`
or pass `runner_name=`. For the example runner, set `MERLIN_MODELBLASTER` to a checkout
containing `validation/firesim_runner.py`; preflight reports the exact missing path.

**`insmod: ERROR: could not load module poll_mode=1` at INFRASETUP.** The XDMA kernel module is
not loaded. FireSim's helper searches for a literal `xdma.ko`; a modern kernel ships
`xdma.ko.zst`, so the search comes back empty and `poll_mode=1` — the *argument* — is mistaken
for the module path. Check with `lsmod | grep xdma` or `ls /dev/xdma0_*`; loading the module is
a host-admin action, not something merlin does.

**Jobs go QUEUED → INFRASETUP → FAILED at ~32 s, with "Needed to prompt for a connection or sudo
password (host: localhost)".** FireSim's fabric manager does not read `~/.ssh/config` unless
`env.use_ssh_config` is set, so a `Host localhost` key block is inert and authentication falls
through to a password prompt it cannot answer. The fix is `env.use_ssh_config = True` in
`deploy/firesim`'s manager configuration. Verify the selected manager revision; if `ssh localhost`
works but fabric does not, this is why.

**Nothing runs and nothing fails.** The queue daemon is dead. `daemon.pid` outliving the process
is normal, so a pid file is not evidence — `preflight.py` checks the process itself.

**A run times out with an empty `uartlog`.** The image never booted. The usual cause is an
over-large `ram0` devicetree overlay: at the stock 256 MB a model boots, and the same model with
`ram0` grown to a few hundred MB wedges before any output. `zephyr_model.py` only emits the
`&ram0` override when the model genuinely needs more than the default. Note also
`zero_out_dram: false` — never rely on DRAM being zero.

**A run hangs with no fault printed.** Inspect the selected runtime, trap handling,
console transport and worker publication state. A progressing cycle counter does
not prove program progress.

- Check that floating-point and vector trap handling matches the selected ISA.
- Under SMP, verify that the selected console implementation supports concurrent
  output without racing its transport.
- For the supported OpenMP shim, `MERLIN_OMP_DEBUG_SPLIT=1` records worker waits
  and publication state. Keep the original deadline and numerical/output gates.

**`KeyError: 'bit_builder_recipe'` at buildbitstream.** The recipe is missing
`bit_builder_recipe` and/or `metasim_customruntimeconfig`. Copy a working recipe.

**Someone else's results appear, or yours vanish.** `config_runtime.yaml` is shared, and so is
the queue. Check `default_hw_config` is still what you set, and use `fsq.py` to see whose jobs
are actually on the board.

**The live `uartlog` goes to zero bytes mid-run.** Do not start a second
`run_on_firesim(...)` while one is still running, even though the queue will happily accept the
job. The queue serializes *execution*, but each caller's marker-waiter clears the shared
`sim_slot_0/uartlog` when it starts watching — so submitting a second run truncates the console
of the run currently on the FPGA, and worse, the second waiter is then watching a log that the
*first* run is about to write its results into. That is a silent mis-attribution: run B reports
run A's numbers.

Submit one at a time and wait for each to return. A queued job is staged at dispatch, not at
submit (`stage_from` in the job JSON is only copied when the daemon picks it up), so cancelling
a still-QUEUED job with `firesim-queue cancel <id>` is safe and does not disturb the running one.
The results themselves are per-job — `results-workload/<timestamp>-<workload>-q<id>/` — so a run
that completes is still attributed correctly; it is the *live* log and the waiters that collide.
