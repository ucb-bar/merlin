# 2026-05-13: Zephyr SMP on a 2-Tile Heterogeneous Shuttle FireSim Build

> **Status:** Landed — Zephyr SMP boots on `FireSimGemminiAndOPUShuttleConfig`
> (2 Shuttle tiles, Gemmini RoCC on hart 0, Saturn OPU vector unit on hart 1)
> at 30 MHz on Xilinx Alveo U250.  Three escalating tests pass end-to-end:
> (1) per-hart smoke (Gemmini flush + int8 `vmul.vv`);
> (2) real accelerator instructions — Gemmini `mvin/mvout` 16×16 int8 scratchpad
> roundtrip + Saturn OPU `OPMVINBCAST`+`VOPACC` int8→int32 outer product;
> (3) proven concurrency — 500-iteration rendezvous loop with both accelerators
> running, parallelism factor 1.93/2.00; and a symmetric scalar SMP companion
> (same `scalar_matmul_slab` kernel pinned to each hart) reaches 1.99× speedup
> over the single-hart baseline.

Related entries:

- [2026-04-29 Zephyr × Merlin × FireSim integration — initial landing](2026-04-29-zephyr-firesim-integration-status.md) — first Zephyr-on-FireSim bring-up on `FireSimQuadRocketConfig` (homogeneous Rocket). This entry is the heterogeneous-Shuttle follow-up.
- [2026-03-18 Chipyard bare-metal integration](2026-03-18-chipyard-bare-metal-integration.md) — the bare-metal HTIF toolchain we fell back to once before getting Zephyr working.

## Context and Goal

Validate a freshly built FireSim bitstream for a 2-tile heterogeneous SoC:

- **Tile 0:** Shuttle + Gemmini (RoCC, default int8 systolic config)
- **Tile 1:** Shuttle + Saturn OPU (vector unit, `vLen=128`, `dLen=64`)

The user goal was: boot Zephyr SMP, run something on each accelerator
("test all cores, test gemmini, test opu, int8 stuff, Zephyr"). The
prior entry validated Zephyr on a homogeneous 4-Rocket SoC; this entry
ports the same flow to the heterogeneous Shuttle build, which surfaced
four distinct issues that the homogeneous-Rocket path had hidden.

## Repro environment

- Chipyard at `/scratch2/agustin/chipyard`, with submodules:
  - `generators/gemmini` at `9c94a39` (gemmini-mx branch HEAD).
    Driver elaboration for our config currently requires
    `git checkout 8c3f9923` (vanilla master) before running
    `firesim infrasetup`, then a restore afterwards. This is a
    workaround for a `chisel3.IO` `None.get` in elaboration on the
    `gemmini-mx` tip and is not in scope here.
  - `generators/radiance/.git` temporarily renamed to
    `.git.may12_runtime` during driver elaboration so the SBT graph
    drops radiance (avoids a downstream `FireSimRadianceClusterSynConfig`
    reference that is currently commented out in `TargetConfigs.scala`).
- `zephyr-chipyard-sw` at `/scratch2/agustin/zephyr-chipyard-sw` on the
  `dev` branch (commit `bde9dab`), with `zephyr_ws/zephyr` initialized
  to `ucb-bar/zephyr@5a06eb0d`.
- Toolchain: reuse dima's pre-installed Zephyr SDK + conda env at
  `/scratch2/dima/zephyr-chipyard-sw-fresh/{tools-manual/zephyr-sdk-1.0.0-beta1,tools/miniforge3/envs/zephyr}`
  (read-only).
- FireSim: `firesim infrasetup` flashes the bitstream via
  `firesim-fpga-util.py` (Vivado HW manager + hw_server). The Ubuntu
  `libxrt-utils` `xbutil` wrapper on this machine is broken (expects
  `/usr/bin/unwrapped/xbutil2` which Ubuntu's repackaged xrt does not
  ship). FireSim's flashing path does not need `xbutil` — that whole
  thread is a red herring.

## What was on the bitstream

`FireSimGemminiAndOPUShuttleConfig` at 30 MHz, timing closed cleanly
(WNS = +0.076 ns, WHS = +0.009 ns, all 56k hold endpoints fixed by
`route_design -directive Explore -tns_cleanup`). Bitstream artifact at:

```
sims/firesim/deploy/results-build/2026-05-06--22-08-53-alveo_u250_firesim_shuttle_gemmini_opu/.../firesim.tar.gz
```

`config_hwdb.yaml` entry: `alveo_u250_firesim_shuttle_gemmini_opu`.
`config_runtime.yaml` `default_hw_config` set to that entry.

## Implementation Changes

All edits live in the Zephyr workspace and the FireSim deploy
config — none of the submodule SHAs were bumped.

### 1. Custom out-of-tree sample

`samples/gemmini_opu_test/` under `zephyr-chipyard-sw`:

- `src/main.c` — SMP test: hart 0 thread issues `gemmini_flush` x3
  via a `.insn r 0x7b, 0x3, 7, x0, %0, %1` inline form (CUSTOM-3,
  `funct3=3`, `funct7=k_FLUSH=7`); hart 1 thread runs a 16-lane
  int8 `vmul.vv` via `.option arch, +v` inline asm. Each thread
  pins itself with `k_thread_cpu_pin(tid, N)` and asserts
  `arch_curr_cpu()->id == N`. Hart 0 prints the per-tile summary
  and exits via HTIF; hart 1 reports pass/fail through shared
  flags (only hart 0 writes the console — see lesson 3 below).
- `prj.conf` (minimum viable for this SoC, see lessons 2–3):

  ```
  CONFIG_MP_MAX_NUM_CPUS=2
  CONFIG_SMP=y
  CONFIG_SCHED_CPU_MASK=y
  CONFIG_SCHED_CPU_MASK_PIN_ONLY=y
  CONFIG_SCHED_DUMB=y
  CONFIG_UART_HTIF_BUFFERED_OUTPUT=y
  CONFIG_UART_HTIF_BUFFERED_OUTPUT_SIZE=256
  CONFIG_UART_HTIF_SYSCALL_PRINT=y
  # NOTE: do NOT set CONFIG_RISCV_ISA_EXT_V=y; see lesson 2.
  ```

- `boards/chipyard_riscv64.overlay` — disables `cpu@2..cpu@7`
  (see lesson 1).
- `CMakeLists.txt`, `sample.yaml` — standard Zephyr boilerplate.

### 2. FireSim workload wiring

- `sims/firesim/deploy/workloads/zephyr_gemmini_opu_test/zephyr_gemmini_opu_test.elf`
  — the built Zephyr ELF (~137 KB).
- `sims/firesim/deploy/workloads/zephyr_gemmini_opu_test.json`:

  ```
  {
    "benchmark_name": "zephyr_gemmini_opu_test",
    "common_simulation_outputs": ["uartlog"],
    "common_bootbinary": "zephyr_gemmini_opu_test.elf",
    "common_rootfs": "../../../../../software/firemarshal/boards/default/installers/firesim/dummy.rootfs"
  }
  ```

  The `dummy.rootfs` is required because the FireSim Alveo driver
  always passes `+blkdev0=...` to the binary regardless of whether
  the SoC actually instantiates a block device. Same pattern dima
  used for `zephyr.json` on the Rocket build.
- `config_runtime.yaml`: `default_hw_config: alveo_u250_firesim_shuttle_gemmini_opu`
  and `workload_name: zephyr_gemmini_opu_test.json`.

### 3. Bare-metal fallback (kept for reference)

While debugging the Zephyr boot, an equivalent multi-hart bare-metal
C test was written and passed on the same bitstream:

- `/scratch2/agustin/baremetal-tests/gemmini_opu_test/{start.S,main.c,link.ld,Makefile}`
- Built with `riscv64-unknown-elf-gcc` (chipyard's conda toolchain).
- Same `gemmini_flush` on hart 0 + `vmul.vv` int8 on hart 1, but with
  a hand-rolled crt0 (`start.S` reads `mhartid`, dispatches to `main0`
  or `main1`, sets `mstatus.{FS,XS,VS}` so vector and accelerator
  state are usable on the harts that have them).
- Deployed as `sims/firesim/deploy/workloads/baremetal_gemmini_opu/`.
- Passed on the FPGA in 17.4 s wallclock, FMR 1.00 (real-time).

This was useful as the "is the bitstream itself good?" probe while we
were debugging the Zephyr boot — bare-metal works the moment the SoC
boots from DRAM, with no DT or SBI assumptions.

## What Worked

End-to-end, with the four fixes below applied, `firesim runworkload`
on this SoC produces:

```
*** Booting Zephyr OS build 5a06eb0d14f9 ***
=== hetero accel smoke test on chipyard_riscv64/rocketchip_virt_riscv64 ===
zephyr cpus = 2
[tile0] hello from hart 0 (expecting 0)
[tile1] hello from hart 1 (expecting 1)
[tile0] gemmini_flush x2 completed on hart 0  -> OK
[tile1] vmul.vv int8 on hart 1, errors=0 -> OK
  out[ 0] =    2 ... out[ 7] =   16
  out[ 8] =   -3 ... out[15] =  -24
=== summary ===
tile0 (gemmini): PASS
tile1 (opu)    : PASS
>>> ALL TESTS PASSED <<<
```

Both Shuttle harts boot under Zephyr SMP, Gemmini RoCC dispatch on
hart 0 succeeds, OPU vector `vmul.vv` on hart 1 produces the
expected `int8` outputs (the negative-lane products wrap correctly:
`-1 * 3 = -3`, ..., `-8 * 3 = -24`).

## What Did Not Work — and the four fixes

These are the lessons we want to carry forward. Three of them are
not Shuttle-specific; one is. None of them were obvious from the
existing Zephyr-on-Rocket entry because the homogeneous Rocket build
masked them.

### Lesson 1 — `chipyard_riscv64` declares 8 CPUs; you must disable the extras

`zephyr_ws/zephyr/dts/riscv/chipyard/chipyard-riscv64.dtsi` declares
`cpu@0`..`cpu@7`, all `status = "okay"`, modeled on the qemu virt
RISC-V dtsi. Even with `CONFIG_MP_MAX_NUM_CPUS=2`, Zephyr's SMP
boot path iterates DT CPUs and tries to wake every `okay` hart via
CLINT IPI. On a 2-hart SoC, the IPIs to harts 2–7 go nowhere and
Zephyr stalls waiting for ACKs — silently. The symptom is "sim
runs at full speed, no UART output ever", and you only realize it
because heartbeat cycles advance but the log stays at
`Commencing simulation.` past hundreds of millions of target cycles.

**Fix:** a DT overlay in the sample (no fork of the board needed):

```
samples/gemmini_opu_test/boards/chipyard_riscv64.overlay:

/ {
    cpus {
        cpu@2 { status = "disabled"; };
        cpu@3 { status = "disabled"; };
        cpu@4 { status = "disabled"; };
        cpu@5 { status = "disabled"; };
        cpu@6 { status = "disabled"; };
        cpu@7 { status = "disabled"; };
    };
};
```

Sanity-check this landed by grepping the merged DTS at
`<build-dir>/zephyr/zephyr.dts` for the six `status = "disabled"`
lines pointing back at the overlay file.

### Lesson 2 — Don't set `CONFIG_RISCV_ISA_EXT_V=y` globally on a hetero hart layout

This is the Shuttle-specific one. Our config puts the Saturn OPU
vector unit on **hart 1 only**:

```scala
new saturn.shuttle.WithShuttleVectorUnit(
    vLen = 128, dLen = 64,
    params = saturn.common.VectorParams.opuParams,
    cores = Some(Seq(1)))
```

so `misa.V` is set on hart 1 and clear on hart 0. Zephyr's RISC-V
arch code interprets `CONFIG_RISCV_ISA_EXT_V=y` as "every hart has
V" — it unconditionally manages V context during thread switches
on all harts. The first context save on hart 0 then traps as an
illegal instruction inside the kernel, before `main()` ever runs.

The symptom *with* `CONFIG_RISCV_ISA_EXT_V=y` is the same silent
hang as lesson 1 — heartbeat advances, no UART output. The symptom
*without* the first fix (overlay) plus this option set is the same
silent hang for two reasons stacked.

**Fix:** leave `CONFIG_RISCV_ISA_EXT_V=n` at the global Zephyr
level. Per-hart V usage is then opt-in via inline asm gated on
`.option arch, +v` (see lesson 4 for the mstatus follow-up).

### Lesson 3 — Zephyr's `uart_htif` direct-putchar path has an SMP race

With lesson 1 + 2 applied, Zephyr boots far enough to write the
first `*` of the `*** Booting Zephyr OS ***` banner, then fesvr
aborts:

```
* terminate called after throwing an instance of 'std::runtime_error'
  what():  bad syscall #0
```

fesvr reports `bad syscall #0` when it sees a `tohost` write whose
top byte (device id) is 0 — i.e. the magic-mem-pointer encoding —
and dereferences `buf[0]` as the syscall number, finding 0. The
direct-putchar code path in `drivers/serial/uart_htif.c` writes
`(1<<56) | (1<<48) | ch` (device 1, command 1, char), guarded by
`k_mutex_lock(&htif_lock, K_FOREVER)`. Under SMP, the mutex acquire
seems to interact poorly with very early boot: we observed the
correct putchar value being shadowed by what fesvr decodes as a
magic-mem pointer with `buf[0]=0`. We did not chase the exact race
to ground — the workaround was sufficient.

**Fix:** use the syscall-print path (the same one dima's working
config uses on the homogeneous Rocket build):

```
CONFIG_UART_HTIF_BUFFERED_OUTPUT=y
CONFIG_UART_HTIF_BUFFERED_OUTPUT_SIZE=256
CONFIG_UART_HTIF_SYSCALL_PRINT=y
```

This batches characters until a `\n` or buffer-full, then issues
one `SYS_write(fd=stdout, buf, len)` magic-mem syscall under the
htif mutex. The single-syscall-per-flush pattern apparently
avoids whatever the direct-putchar race window was.

(The driver still has a small bug to mind separately: in
`uart_htif_buffer_flush()` the syscall path calls `_write(0, ...)`
where `fd=0` is stdin — should be `1`. fesvr happens to accept fd
0 here because it treats any positive write as stdout, but it is
worth fixing upstream.)

### Lesson 4 — If global V is off, you must enable `mstatus.VS` per-hart

With lessons 1–3 applied, Zephyr boots, both harts are alive, both
threads land on their target hart, hart 0's Gemmini flush completes.
Then hart 1 traps:

```
mcause: 2, Illegal instruction  mtval: 0
mepc:   00000000800004ca
mstatus: 0000000a00001880
```

`mcause=2` is illegal instruction; `mstatus & 0x600 == 0` means
`VS = Off`. The hardware has the V extension on hart 1, but
Zephyr left `mstatus.VS=0` because lesson 2 told it not to manage
V state. The `vsetivli` at `mepc` then traps before retiring.

**Fix:** at the top of the hart-1 thread (or anywhere that runs
strictly on hart 1 before the first vector op), set `mstatus.VS`
explicitly:

```c
unsigned long vs_bits = (3UL << 9);  /* VS = Dirty */
__asm__ __volatile__("csrs mstatus, %0" :: "r"(vs_bits));
```

Zephyr threads on RISC-V run in M-mode by default so plain
`csrs mstatus` works. We chose `VS = Dirty` (3) rather than
`Initial` (1) so any vector state we leave behind in the registers
is preserved across the rest of the thread's run; for a smoke test
either value works.

A cleaner long-term option is to fork the chipyard Zephyr board and
either declare a per-CPU `riscv,isa` that includes V on hart 1 only
(modeling the actual asymmetric SoC) or to add a hart-specific
`PRE_KERNEL_2` hook that sets `mstatus.VS`. We did neither here —
the in-thread CSR write is sufficient for the smoke test and keeps
the board files unmodified.

## Debugging Notes

A few stage-by-stage observations that would have shortened the loop:

- `firesim runworkload` produces eight `tsibridge_t::tick skipping
  tick` lines *before* the program ever prints anything; that is
  fesvr completing TSI init, not a stall. Don't take eight as
  cause for alarm.
- `heartbeat.csv` updates on a 5-second cadence — if it is
  advancing while `uartlog` looks frozen, the CPU is executing
  but no `tohost` writes are reaching fesvr (most often a boot
  hang, not a console issue).
- Comparing the *generated* SoC DTS at
  `sims/firesim-staging/generated-src/<config>/<config>.dts` to
  the Zephyr board DTS gave the address-map confirmation we needed
  (CLINT @ `0x2000000`, PLIC @ `0xc000000`, UART @ `0x10020000`,
  DRAM @ `0x80000000` — all match). When the board DTS *almost*
  matches the SoC, the remaining mismatch is usually the hart
  count (lesson 1) or the per-CPU ISA strings (lesson 2/4).
- `bad syscall #N` from fesvr is the host-side abort; the FireSim
  driver dies with `terminate called after throwing an instance of
  'std::runtime_error'`. Treat that as "the guest just wrote
  something invalid to `tohost`" and disassemble around the most
  recent `printk`/HTIF call.
- The chipyard FireSim `+blkdev0=...rootfs` plusarg is always
  passed, even when the SoC has no block device. Use the
  `firemarshal/boards/default/installers/firesim/dummy.rootfs`
  placeholder.

## Test Coverage and Commands

### Build the Zephyr ELF

```bash
cd /scratch2/agustin/zephyr-chipyard-sw
export ZEPHYR_BASE=$PWD/zephyr_ws/zephyr
export ZEPHYR_SDK_INSTALL_DIR=/scratch2/dima/zephyr-chipyard-sw-fresh/tools-manual/zephyr-sdk-1.0.0-beta1
export ZEPHYR_TOOLCHAIN_VARIANT=zephyr
export PATH=/scratch2/dima/zephyr-chipyard-sw-fresh/tools/miniforge3/envs/zephyr/bin:$ZEPHYR_SDK_INSTALL_DIR/gnu/riscv64-zephyr-elf/bin:$PATH

west build -p -b chipyard_riscv64/rocketchip_virt_riscv64 \
    samples/gemmini_opu_test/ \
    --build-dir /scratch2/agustin/zephyr-builds/gemmini_opu_test
```

### Stage and run on FireSim

Workload ELF goes to `sims/firesim/deploy/workloads/zephyr_gemmini_opu_test/zephyr_gemmini_opu_test.elf`.

```bash
cd /scratch2/agustin/chipyard/sims/firesim
source /scratch2/agustin/miniforge3/etc/profile.d/conda.sh
conda activate /scratch2/agustin/chipyard/.conda-env
source ./sourceme-manager.sh --skip-ssh-setup

firesim kill          # defensive
firesim infrasetup    # flashes our bitstream, builds driver
firesim runworkload   # boots Zephyr ELF, captures uartlog
firesim kill          # release the U250
```

If `firesim infrasetup` fails the driver elaboration with a
`chisel3.IO` `None.get` on the gemmini-mx tip, revert gemmini to
vanilla master temporarily (`git -C generators/gemmini checkout 8c3f9923`),
move `generators/radiance/.git` aside, wipe the staging cache
(`rm -rf sims/firesim-staging/generated-src/firechip.chip.FireSim.FireSimGemminiAndOPUShuttleConfig`),
re-run infrasetup, then restore submodule state when you're done.

### Bare-metal equivalent (no Zephyr)

```bash
cd /scratch2/agustin/baremetal-tests/gemmini_opu_test
make                           # produces gemmini_opu_test.elf (~7.7 KB)
# Stage as workloads/baremetal_gemmini_opu/baremetal_gemmini_opu (no .elf)
# JSON has "common_rootfs": null
```

The bare-metal ELF is useful as a fast "is the bitstream itself
healthy?" probe (~17 s end-to-end on the FPGA at 30 MHz) before
touching Zephyr.

## Update — Real Instructions, Parallelism Proof, Symmetric SMP

The smoke test above (Gemmini `flush` on hart 0 + plain `vmul.vv` on
hart 1) was enough to prove the four Zephyr lessons but only lightly
touched the accelerators.  Three follow-up tests inside the same
Zephyr sample now exercise much more of the design and quantify the
parallelism on a real workload.  The sample at
`/scratch2/agustin/zephyr-chipyard-sw/samples/gemmini_opu_test/` is
reused; only `src/main.c` is swapped between modes.

### Real instructions: Gemmini mvin/mvout + Saturn OPMVINBCAST/VOPACC

References used to write the inlined ops:

- `/scratch2/agustin/chipyard/generators/gemmini/software/gemmini-rocc-tests/bareMetalC/mvin_mvout.c`
  and the surrounding `include/gemmini.h` / `include/gemmini_params.h` for
  CUSTOM-3 funct codes (`k_FLUSH=7`, `k_CONFIG=0`, `k_MVIN=2`, `k_MVOUT=3`,
  `CONFIG_LD=1`, `CONFIG_ST=2`).
- `/scratch2/agustin/chipyard/generators/saturn/benchmarks/opu-gemm/{kernel.h,main.c}`
  and `generators/saturn/.../bme.h` for the OPU encodings:
  `OPMVINBCAST = .insn r 0x57, 0x6, 0x59, md, x0, vs2`,
  `VOPACC      = .insn r 0x57, 0x2, 0x51, md, vs1, vs2`,
  `VMV_VR      = .insn r 0x57, 0x6, 0x5d, vd, rs1, ms2`.

Hart 0 per iteration:

```c
gemmini_flush();
gemmini_config_ld(/*stride=*/16);
gemmini_config_st(/*stride=*/16);
gemmini_mvin (&gem_in [0][0], /*spad_addr=*/0);  /* 16x16 int8 -> spad */
gemmini_mvout(&gem_out[0][0], /*spad_addr=*/0);  /* spad -> 16x16 int8 */
gemmini_fence();
```

Hart 1 per iteration:

```c
/* setup: vsetvli e32 m4 + vle32.v v0 zero bias */
OPMVINBCAST_HELPER(1, 0);           /* m1[*][j] = v0[j]                 */
/* vsetvli e8 m1 + vle8.v v5 = a + vle8.v v4 = b */
VOPACC_HELPER(1, 4, 5);             /* m1[i][j] += v5[i] * v4[j]        */
/* vsetvli e32 m4 + VL × (VMV_VR v0, r, m1 ; vse32.v v0, &C[r][0])     */
```

Vector mnemonics are wrapped with `.option push / .option arch, +v /
.option pop` because the Zephyr build's global `-march` is
`rv64imafdc_zicsr_zifencei` (no V); gas would otherwise refuse to
assemble them.

First single-shot run (one iteration, hand-checked output):

```
*** Booting Zephyr OS build 5a06eb0d14f9 ***
=== hetero accel real-instruction test ===
[tile0] gemmini_flush
[tile1] OPU outer-product (vl=4): errors=0 -> OK
[tile1] a=[1 2 3 4]  b=[10 20 30 40]
[tile1]   C[0][.] = [  10   20   30   40]   (expected [  10   20   30   40])
[tile1]   C[1][.] = [  20   40   60   80]   (expected [  20   40   60   80])
[tile1]   C[2][.] = [  30   60   90  120]   (expected [  30   60   90  120])
[tile1]   C[3][.] = [  40   80  120  160]   (expected [  40   80  120  160])
[tile0] mvin/mvout roundtrip: 256/256 cells, errors=0 -> OK
>>> ALL TESTS PASSED <<<
```

This proves the actual DMA + scratchpad path on Gemmini (256 int8
cells moved through `spad@0` and verified bit-exact) and the OPU's
matrix-datapath outer product (`VOPACC` accumulating into `m1` after
`OPMVINBCAST`-ing a zero bias, then read back through `VMV_VR`).
Plain `vmul.vv` from the original smoke test runs on the vector unit
too but only exercises the generic RVV pipeline, not the OPU's
outer-product fast path.

### Asymmetric parallelism proof — 500-iteration rendezvous

To prove the two accelerators are running *at the same time* (not
just that both eventually run), the same per-hart work is wrapped in
a 500-iteration loop with a 2-flag rendezvous barrier per iter:

```c
hart0_iter = it + 1;  __sync_synchronize();
uint64_t wait_start = rdmcycle();
while (hart1_iter < it + 1) {
    if (rdmcycle() - wait_start > DEADLOCK_GUARD_CYCLES) {
        hart0_deadlock = 1; goto done;
    }
    __sync_synchronize();
}
```

(Symmetric on hart 1.) If Zephyr were time-multiplexing both threads
onto a single hart the barrier would deadlock on iteration 0 — the
spinning hart never yields.  Completing 500 rounds is itself the
proof of concurrent execution; the cycle counters quantify how busy
each hart was.

Per-iter work is heavier than the smoke test:
- Hart 0: full `mvin`+`mvout` of a 16×16 int8 matrix through the
  Gemmini scratchpad (DMA in + DMA out + `fence`).
- Hart 1: `vsetvli e32 → vle32.v` bias → `OPMVINBCAST` → `vsetvli e8
  → vle8.v ×2` → `VOPACC` → `vsetvli e32 → 4× VMV_VR + vse32.v`.

Result on the FPGA (30 MHz, FMR ≈ 1.0):

```
hart0 alive=1, gemmini errors=0, deadlocked=0
hart1 alive=1, opu errors=0,     deadlocked=0
hart0 reached iter 500 / 500
hart1 reached iter 500 / 500
hart0 busy cycles: 743694
hart1 busy cycles: 695770
wallclock cycles : 751960
parallelism factor x100 (sum_busy / max_busy): 193
>>> ALL TESTS PASSED -- PARALLEL EXECUTION PROVEN (factor x100 >= 150) <<<
```

`sum_busy / max_busy = (743,694 + 695,770) / 751,960 = 1.93` out of a
theoretical max of `2.00`: both Shuttle harts were busy ~96% of the
~25 ms wallclock, with their respective accelerators in flight.  No
deadlock, 0 errors on either accelerator across 500 iters.

The `parallel_x100 ≥ 150` heuristic in the test is intentionally
loose; the actual measured ratio sits comfortably above it (1.93×).

### Symmetric scalar SMP — same kernel on both harts

The asymmetric proof above runs *different* code on each hart.  The
companion test answers a different question: can both harts cooperate
on the *same* workload, identical code, divided by row?  This is the
textbook SMP-of-scalar pattern, and on our current bitstream it works
without any hardware change because both Shuttles have full RV64GC
(only V is asymmetric — see the next subsection).

The same `scalar_matmul_slab` function is pinned to each hart with a
disjoint row range:

```c
static __attribute__((noinline))
void scalar_matmul_slab(int row_lo, int row_hi,
                        const int8_t (*restrict a)[K],
                        const int8_t (*restrict b)[K],
                        int32_t (*restrict c)[N])
{
    for (int i = row_lo; i < row_hi; i++) {
        for (int j = 0; j < N; j++) {
            int32_t acc = 0;
            const int8_t *arow = a[i];
            const int8_t *brow = b[j];
            for (int k = 0; k < K; k++) {
                acc += (int32_t)arow[k] * (int32_t)brow[k];
            }
            c[i][j] = acc;
        }
    }
}

/* main thread (hart 0) first runs scalar_matmul_slab(0, M, ..., Cref)
 * as a single-hart serial baseline, then spawns:
 *     k_thread_create(hart_worker, 0, 0, M/2 ...);   pinned to hart 0
 *     k_thread_create(hart_worker, 1, M/2, M ...);   pinned to hart 1
 * Each worker times itself with rdmcycle and writes hartN_busy.
 */
```

Workload: int8 `M=N=K=64` GEMM (`C = A * B^T`), int32 accumulator.
Result on the FPGA:

```
[serial]   one hart, full 64x64 matmul: 2,149,014 cycles
[parallel] wallclock (incl. spawn/join):  1,083,444 cycles
           hart0 busy: 1,072,725
           hart1 busy: 1,076,374
           hart parallelism x100: 199        (~ideal 2-way SMP)
           speedup vs serial x100: 199
           output verify: 4096 / 4096 cells correct, errors=0
>>> SMP SCALAR-PARALLEL TEST PASSED <<<
```

Numbers worth remembering:

- **1.99× speedup** vs the single-hart serial baseline computed in
  the same run (so the host-side wall-clock measurement noise cancels
  out).
- **199/200 parallelism** (`sum_busy / max_busy`) — both harts busy
  99.5% of the parallel-section wallclock; the remaining 0.5% is
  thread spawn + join overhead.
- **4096/4096 cells exact** vs the serial reference, confirming the
  partition is correct and there's no shared-cache aliasing
  pathology between the two halves of `C`.

This is what "real multiprocessor" looks like on this SoC for any
embarrassingly-parallel scalar workload.

### What about RVV on both harts?

The natural next question is whether the same row-partition pattern
works for an *RVV* kernel.  On this bitstream it does not — Tile 0
has no V extension by construction:

```scala
new saturn.shuttle.WithShuttleVectorUnit(
    vLen = 128, dLen = 64,
    params = saturn.common.VectorParams.opuParams,
    cores = Some(Seq(1)))   // only tile 1 gets V
```

Any RVV instruction issued on hart 0 traps as illegal-instruction —
that's also why lesson 4 (manual `mstatus.VS=Dirty` on hart 1 only)
exists.  Two paths to fix it:

1. **No rebuild — reuse an existing dual-V bitstream.** dima has
   built `alveo_u250_firesim-dual-rocket-saturn-gemmini-q31-no-nic-l2-llc4mb-ddr3`
   (target `FireSimREFV256D128DualRocketGemminiQ31Config`) and
   `alveo_u250_firesim-quad-rocket-saturn-no-nic-l2-llc4mb-ddr3`
   (`FireSimREFV256D128QuadRocketConfig`).  Saturn's Rocket variant
   (`saturn.rocket.WithRocketVectorUnit(...)`) does not take a
   `cores=` filter and is applied to every tile, so on those
   bitstreams every hart can issue RVV.  Driver elaboration would
   need the gemmini-q31 submodule state dima used, so a small
   submodule juggle (similar to the gemmini@`8c3f9923` revert we
   already do for our config) may be required.
2. **Rebuild our hetero config with OPU on both Shuttle tiles.** Flip
   one argument in `HeteroConfigs.scala`:

   ```scala
   new saturn.shuttle.WithShuttleVectorUnit(
       vLen = 128, dLen = 64,
       params = saturn.common.VectorParams.opuParams,
       cores = None)        // None = every Shuttle tile gets OPU
   ```

   then `firesim buildbitstream` (~3 h Vivado, same shape as the
   May 6 build).  This is the cleaner long-term option because it
   keeps Gemmini on tile 0 and gives both tiles the full OPU
   outer-product datapath, not just stock Saturn.

In either case the kernel itself is the same shape as
`scalar_matmul_slab` above: each hart owns `M/HARTS` rows of `C`, the
inner triple loop swaps `acc += a[k] * b[k]` for a `vsetvli e8 →
vle8.v → vmul.vv → vredsum` (or `OPMVINBCAST`+`VOPACC` if going OPU),
and the outer pinning + barrier are unchanged.  No further Zephyr
plumbing is needed — only `mstatus.VS=Dirty` becomes a per-hart
preamble (or, equivalently, set globally in a `PRE_KERNEL_2` hook
once both harts have V hardware).

## Follow-Up Tasks

Status legend: ✅ done in this entry · 🟡 partially done · ⬜ still open.

- ✅ **Saturn OPU outer-product op (`VOPACC`) instead of `vmul.vv`.**
  Done — hart-1 path now goes through `OPMVINBCAST` + `VOPACC` +
  `VMV_VR` against an `m1` matrix register; 0 errors across 500
  iters (see *Update — Real Instructions, Parallelism Proof,
  Symmetric SMP*).  The OPU's specialized outer-product datapath
  is now demonstrably alive.
- 🟡 **Multi-hart Gemmini matmul, not just `gemmini_flush`.** The
  hart-0 path now does `mvin` + `mvout` of a 16×16 int8 matrix
  through the scratchpad every iteration (full DMA + spad path).
  Still open: wire up an actual `tiled_matmul_ws` int8 systolic
  compute from `generators/gemmini/software/gemmini-rocc-tests`,
  not just data movement.
- ⬜ **RVV-on-both-harts.** This bitstream cannot do it (tile 0
  has no V).  Two clear paths exist (see *What about RVV on both
  harts?* above): reuse one of dima's dual/quad-Rocket+Saturn
  bitstreams without a Vivado rebuild, or change
  `HeteroConfigs.scala`'s `cores = Some(Seq(1))` to
  `cores = None` and rebuild (~3 h).
- ⬜ **Upstream the per-CPU V hint** rather than `csrs mstatus` in
  thread code. A `samples/gemmini_opu_test/boards/chipyard_riscv64.overlay`
  with a per-cpu `riscv,isa = "rv64gcv"` override on `cpu@1`, plus
  a small `PRE_KERNEL_2` hook in a Zephyr `soc/` shim that sets
  `mstatus.VS` based on the matching DT property, would be the
  cleaner version of lesson 4.  Becomes more important once both
  tiles have V (it would replace the explicit `csrs` in our
  workers).
- ⬜ **Investigate the direct-putchar HTIF SMP race** that lesson 3
  worked around. The buffered/syscall path works but is heavier
  than needed for boot-banner-grade output. Adding a `smp_lock`
  around the `tohost` access in `uart_htif_poll_out`'s direct
  path is the likely fix.
- ⬜ **Fix the `_write(0, ...)` fd-stdin bug** in
  `drivers/serial/uart_htif.c::uart_htif_buffer_flush` upstream
  (should be `_write(1, ...)`).
