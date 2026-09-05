# GSIM as a fast register-observation engine for ModeLIR M2/M3

Goal: run our real accelerators (gemmini-mx, atlas) under **GSIM** (OpenXiangShan's
FIRRTL→C++ RTL simulator, ~20× Verilator) so we can read RTL registers **every
cycle** — the fast substrate for the M2 cycle-accounting waterfall.

**Bottom line (four configs):**

| Config | Design | GSIM parse→C++ | C++ compiles | Runs + per-cycle regs | Status |
|---|---|---|---|---|---|
| **atlas M2** | `SystolicArray` (MXU, accel-only) | ✅ | ✅ (after vector-IO fix) | ✅ **wavefront observed** | **WORKING** |
| **gemmini M2** | `Gemmini` (accel, incl. controllers) | ✅ | ✅ (after vector-IO + codegen shim) | ✅ **compute waterfall** (RoCC cmd stream driven, §4a) | **WORKING** |
| **gemmini M3** | `ChipTop` (full SoC DUT, harness pruned) | ✅ | ✅ (after 4 blackbox stubs + temp-leak shim) | ✅ **Rocket EXECUTES BootROM-baked matmul; real busy waterfall (compute 418/451, 15 RoCC dispatches)** | **WORKING** (§5.5) |
| **atlas M3** | `ChipTop` (full Atlas SoC DUT, harness pruned) | ✅ (after 2 gsim patches) | ✅ (48 files + SoC blackbox stubs) | ✅ **Rocket bootrom executing; PC + Atlas fetch-PC read/cycle** | **WORKING** (re-rooted below `TestHarness`) |

The headline finding: the old "chipyard `firrtl2` `.fir` is a dialect gsim rejects
(`unexpected 'is'`)" blocker from `runs/gsim/RUN.md` is **stale**. GSIM's parser in
this tree was already patched for FIRRTL 6/3.3.0 (`layerblock`, multi-operand `cat`,
`invalidate`). The `unexpected 'is'` reject is specific to the **Scala FIRRTL
Compiler (SFC)** emission (`*.sfc.fir`, uses old-style `X is invalid`). Our
**CIRCT/firtool** emissions use new-style `invalidate` and parse fine. The real
remaining walls are GSIM **C++ codegen** bugs on wide-IO accelerators and SoC-level
blackboxes, documented per-config below.

---

## 0. Environment (all commands assume this)

```bash
export GSIM=/scratch/agustin/projects/gsim
export LD_LIBRARY_PATH=/scratch2/agustin/miniforge3/envs/merlin-dev/lib
export CPLUS_INCLUDE_PATH=$GSIM/.flexinc
export LIBRARY_PATH=/usr/lib/x86_64-linux-gnu
export CXX=/scratch2/agustin/miniforge3/envs/merlin-dev/bin/clang++   # clang 21
export CC=/scratch2/agustin/miniforge3/envs/merlin-dev/bin/clang
```

- GSIM binary: `$GSIM/build/gsim/gsim` (rebuild with `make build-gsim` under the env above).
- GSIM compile-to-C++:  `$GSIM/build/gsim/gsim --dir=<outdir> <input.fir>` → emits `<outdir>/<Top>.h` + `<outdir>/<Top>0.cpp`.
- Helper scripts (this dir): `gsim_fix_vector_io.py`, `gsim_fix_gemmini_codegen.py`.
- Example harnesses (this dir): `gsim_atlas_mxu_harness.cpp`, `gsim_gemmini_harness.cpp`.

---

## 1. Baseline — Rocket + CoreMark (reproduces the speed anchor)

```bash
cd $GSIM
tar xjf /scratch/agustin/projects/gsim-ready-to-run/TestHarness-rocket.tar.bz2 -C ready-to-run 2>/dev/null || true
ln -sf /scratch/agustin/projects/gsim-ready-to-run/bin/coremark-rocket.bin ready-to-run/bin/coremark-rocket.bin
/usr/bin/time -v make run dutName=rocket mainargs=ready-to-run/bin/coremark-rocket.bin
```

Measured: **4.2M cycles in ~11.4 s wall, ~560k cycles/s** (matches `runs/gsim/rocket_coremark_speed.json`).
`TestHarness-rocket.fir` is FIRRTL 3.3.0 chirrtl, 224,900 lines — GSIM accepts it end to end.

---

## 2. The shared crux: GSIM-compatible FIRRTL emission

**Which `.fir` to feed.** Use CIRCT/firtool emissions (new-style `invalidate`), **not**
SFC (`*.sfc.fir`, old-style `is invalid` → `unexpected 'is'`). We already have
committed CIRCT emissions:

| Config | File |
|---|---|
| atlas M2 | `artifacts/atlas/firrtl/SystolicArray.fir` (FIRRTL 6.0.0, 264k lines) |
| gemmini M2 | `runs/essent/gemmini/inputs/Gemmini.fir` (FIRRTL 3.3.0, 199k lines) |
| gemmini M3 | `artifacts/gemmini/firrtl/chipyard.harness.TestHarness.GemminiRocketConfig.fir` (3.3.0, 660k lines) |

firtool versions on this box (all CIRCT): chipyard `.conda-env` = **firtool-1.75.0**;
`/scratch/agustin/cache/llvm-firtool/1.128.0`; `third_party/circt/build` (dev).
Any of the 1.75+ CIRCT firtools emit the `invalidate` dialect GSIM accepts — **no
downgrade to an older firtool is needed.** Re-emit with e.g.
`firtool --format=fir --emit-chirrtl` or the existing chipyard/atlas make targets.

**Annotations.** SoC `.fir` files carry a large inline annotation blob
(`circuit TestHarness :%[[ …JSON… ]]`). GSIM's `anno` lexer state handles the empty
form (`%[[]]`) but the multi-kline SoC blob is not the crash cause (see §5). Stripping
it is simulation-semantics-preserving (annotations only drive dedup/transforms):

```bash
F=artifacts/gemmini/firrtl/chipyard.harness.TestHarness.GemminiRocketConfig.fir
END=$(grep -nE '^]]' "$F" | head -1 | cut -d: -f1)          # first '  module' is END+1
{ echo "FIRRTL version 3.3.0"; echo "circuit TestHarness :"; tail -n +$((END+1)) "$F"; } > soc_stripped.fir
```

**The vector-IO codegen wall (applies to every accel-only module).** GSIM emits
`set_<port>(<T> val)` / `<T> get_<port>()` for each top port. That is correct for
the RISC-V cores it ships (scalar top IO). Accelerators expose **wide vector top
ports** — e.g. `io.computeReq.bits.act` is 32-wide, `io.weights0` is 32×32. GSIM
declares the member correctly as an array but still emits a **scalar** accessor whose
body does `if (member != val) { member = val; }` → does not compile
(`comparison between pointer and integer` / `array type … is not assignable`).

Fix: `gsim_fix_vector_io.py <Module> <outdir>` rewrites **only** those accessor stubs
(setter → `memcpy(member, val, sizeof(member))` element-copy keeping the identical
`activeFlags[...]` node-activation lines; getter → gains `int i`, returns `member[i]`).
Boundary-only; the datapath/register model is untouched. No regex.

---

## 3. atlas M2 — `SystolicArray` (MXU) — **WORKING**

```bash
SP=/scratch/agustin/projects/gsim-work/atlas   # any scratch dir
mkdir -p $SP && cd $SP
$GSIM/build/gsim/gsim --dir=$SP artifacts/…/SystolicArray.fir      # ~16 s → SystolicArray.h + SystolicArray0.cpp
python3 <char>/gsim_fix_vector_io.py SystolicArray $SP             # repairs 5 vector accessors
cp <char>/gsim_atlas_mxu_harness.cpp $SP/harness.cpp
$CXX -O1 -std=c++2b -I. SystolicArray0.cpp harness.cpp -o atlas_mxu_sim   # ~30 s
./atlas_mxu_sim
```

The model has **10,055 registers**, all public members of `class SSystolicArray`
(read them as `dut->weightSelSkewed_r_80`, `dut->peMesh_io_addendVec_8_r`, …).
The harness pulses a compute request and prints per cycle:

```
cycle,outValid_r,outValid_r8,addend8,addend9,io_outValid,io_outBits0
0,0,0,0,0,0,0
1,1,0,0,0,0,0     <- input-valid pipeline reg asserts while req is driven (cyc 1-5)
...
5,1,0,0,0,0,0
6,0,0,0,0,0,0
9,0,1,0,0,0,0     <- SAME token re-appears in the deeper stage (outValid_r_8)
...              exactly 8 cycles later — the systolic wavefront marching through
13,0,1,0,0,0,0    the array. Per-cycle register observation confirmed.
```

This is the M2 primitive: a driven valid token is watched propagating stage-by-stage
through the pipeline registers.

---

## 4. gemmini M2 — `Gemmini` accelerator (incl. controllers) — **WORKING**

```bash
SP=/scratch/agustin/projects/gsim-work/gemmini; mkdir -p $SP; cd $SP
$GSIM/build/gsim/gsim --dir=$SP runs/essent/gemmini/inputs/Gemmini.fir   # ~11 s
python3 <char>/gsim_fix_vector_io.py Gemmini $SP           # 12 vector setters + 1 getter (PTW/CSR ports)
python3 <char>/gsim_fix_gemmini_codegen.py Gemmini $SP     # see below
cp <char>/gsim_gemmini_harness.cpp $SP/harness.cpp
$CXX -O1 -std=c++2b -DNDEBUG -I. Gemmini0.cpp harness.cpp -o gemmini_sim   # ~22 s
./gemmini_sim 2>/dev/null
```

`Gemmini.fir` parses and emits C++ cleanly, but the generated `.cpp` hit **two GSIM
codegen defects** (distinct from the vector-IO issue; both are gsim backend bugs that
leave the `.cpp` referencing undeclared identifiers):

1. **Orphan output-alias setters** — `set_io$$csrs$$ren/wen/wdata/value` (and a couple
   `auto$$…$$data`, `io$$fpu_resp$$…$$data`) whose backing member was never declared
   (the real ports are the `io$$ptw$$customCSRs$$csrs$$*[4]` arrays). Duplicate
   accessor stubs for outputs. Fix: neutralise their bodies to a no-op (you never
   *drive* an accelerator output).
2. **Cross-partition temp leak** — the reset-init block references
   `im2col$_modulo_block_done_T` / `_modulo_block_save_T`, which GSIM only declared as
   *locals inside a different partition function*. Fix: declare them locally `=0` at
   the point of use (reset value; im2col is independent of the controller FSMs we
   observe). **This is a documented value judgement.**

Both are applied by `gsim_fix_gemmini_codegen.py` (no regex). `-DNDEBUG` disables the
design's FIRRTL `assert`s: GSIM does not reset-gate them, so a TLB assertion
(`tlb$tlbs_0$tlb$_T_30`) fires spuriously at idle. Assertions are verification
constructs, not datapath — safe to disable for a cycle-count/observation run.

The model has **17,199 registers** (all public). The harness reads the three
controller FSM-state registers each cycle:
`dut->ex_controller$control_state`, `dut->load_controller$control_state`,
`dut->store_controller$control_state`, plus `dut->get_io$$busy()` and
`dut->reservation_station$_io_busy_T_4`.

**Honest caveat (superseded below):** with no RoCC command driven the controllers sit
**idle (state 0)**, so the bare `gsim_gemmini_harness.cpp` proves per-cycle register
*observation* works on the real gemmini accelerator but does not itself show a busy
waterfall. The next subsection drives a real command stream and gets the waterfall.

### 4a. Driving a valid RoCC command stream — controllers leave idle, real waterfall

`gsim_gemmini_cmd_harness.cpp` + `gsim_gemmini_cmd_encode.py` drive a **DRAM-free**
gemmini command stream on `io.cmd`, so the controllers leave state 0 and produce a
real M2 occupancy waterfall — **no TLB assertion**, because no `mvin/mvout` (no DMA)
is issued. Build exactly as §4 but with `harness.cpp = gsim_gemmini_cmd_harness.cpp`
after `python3 gsim_gemmini_cmd_encode.py $SP` (emits `gemmini_cmd_stream.h`).

**io.cmd setter names GSIM emitted** (RoCC `DecoupledIO` command bundle, from `Gemmini.h`):

```
set_io$$cmd$$valid(uint8)
set_io$$cmd$$bits$$inst$$funct(uint8)     set_io$$cmd$$bits$$inst$$opcode(uint8)
set_io$$cmd$$bits$$inst$$xd/xs1/xs2/rd/rs1/rs2(uint8)   // rs1/rs2 here = 5-bit reg indices
set_io$$cmd$$bits$$rs1(uint64)            set_io$$cmd$$bits$$rs2(uint64)   // the operand words
get_io$$cmd$$ready()  -> uint8            // handshake: fire == (valid && ready) in a cycle
```

**Command encoding** (decoded from `GemminiISA.scala` funct codes + `gemmini.h`
rs1/rs2 packing by `gsim_gemmini_cmd_encode.py`; all operands in scratchpad / accumulator):

| funct | rs1 | rs2 | meaning |
|---|---|---|---|
| 0 (`CONFIG_CMD`, rs1[1:0]=`CONFIG_EX`=0) | `0x3f80000000010004` | `0x0001000000000000` | config_ex WS, act=0, shift=0, acc_scale=1.0f |
| 6 (`PRELOAD_CMD`) | `0x0010001000000010` | `0x0010001080000000` | preload BD=spad row16 → C=acc row0 (16×16) |
| 4 (`COMPUTE_AND_FLIP_CMD`) | `0x0010001000000000` | `0x0010001000000010` | compute_preloaded A=spad0, BD=spad16 |
| 5 (`COMPUTE_AND_STAY_CMD`) | `0x0010001000000000` | `0x0010001000000010` | compute_accumulated A=spad0, BD=spad16 |

opcode = `0x7B` (custom-3); gemmini does not decode it internally. `config_ex` is
processed **inline in the ex-controller's `waiting_for_cmd` state** (ExecuteController.scala
line 541) so it does not by itself move `control_state`; it is the `compute` commands
(the `DoComputes(0)` single-mul branch, line 613) that drive `control_state`
`waiting_for_cmd(0) → compute(1)`.

**Observed controller trace (idle → busy).** CSV columns
`phase,cycle,ex_state,load_state,store_state,io_busy,rs_busy_T4`; compacted to changes:

```
idle_pre,0,   0,0,0, 0,0     <- baseline: all controllers idle, not busy
gap,4,        0,0,0, 1,0     <- io.busy asserts once config_ex is enqueued
gap,8,        0,0,0, 1,1     <- reservation_station busy (commands in flight)
drain,15,     1,0,0, 1,1     <- ex_controller$control_state = 1 (compute): LEFT IDLE
... ex_state holds 1 for the compute run (toggles 1↔0 as the mesh flushes) ...
drain,62,     0,0,0, 1,1
drain,63,     1,0,0, 1,1
```

`ex_controller$control_state` first goes non-zero at **cycle 15** and is busy for
**54 of 71** observed cycles. `load`/`store` stay 0 (DRAM-free: no DMA). The handshake
fires (`io.cmd.ready`=1) for every command; `io.busy` asserts from the first enqueue.

**Occupancy waterfall** (feed the CSV through `engines.gsim_gemmini_busy_trace` →
`engines.occupancy_from_busy_trace`; note the engine wants the tuple order
`(load_state, store_state, ex_state)`):

```
{'buckets': {'compute': 54, 'dma': 0, 'overlap': 0, 'idle': 17}, 'total': 71, 'active': 54, 'closes': True}
```

This is the gemmini M2 waterfall with **real non-idle buckets** — an EX-dominated
compute run (76% occupancy) with zero DMA, exactly as expected for an on-scratchpad
matmul. Reaching a `mvin`-fed matmul (load/store buckets) is the next step and needs
the TLB satisfied; the config+preload+compute path above deliberately avoids it.

Files added this session: `gsim_gemmini_cmd_encode.py` (no-regex bit encoder),
`gsim_gemmini_cmd_harness.cpp`. The **atlas M2** in §3 already demonstrated the
waterfall mechanism; this closes it for gemmini specifically.

### 4b. Shape/size sweep — compute cycles scale linearly with matmul count

`gsim_gemmini_sweep_harness.cpp` takes **K = argv[1]** (number of back-to-back
compute matmuls) so the SAME binary is reused across the sweep — only one compile,
one run per K. It drives `config_ex + preload + K × compute_and_stay` (all DRAM-free
on scratchpad), drains to quiescence (`io.busy`=0 for 24 consecutive cycles), and
`gsim_gemmini_sweep.py` computes each K's waterfall through the repo engines and
least-squares-fits `compute_cycles` vs `K`.

GSIM-measured sweep (`gsim_gemmini_m2_sweep.json` / `.csv`, this dir):

| num_computes | total_cycles | compute_cycles | idle_cycles | dma |
|---|---|---|---|---|
| 1  | 119  | 46   | 73 | 0 |
| 2  | 119  | 77   | 42 | 0 |
| 4  | 181  | 139  | 42 | 0 |
| 8  | 309  | 263  | 46 | 0 |
| 16 | 565  | 511  | 54 | 0 |
| 32 | 1077 | 1007 | 70 | 0 |

**Fit: `compute_cycles = 15 + 31·num_computes`, R² = 1.000000 (every point exactly
on the line).** Slope 31 cyc/matmul (one 16×16 WS accumulate pass through the mesh),
intercept 15 cyc (fixed pipeline fill/flush). `dma`=0 throughout (on-scratchpad).
This is the gemmini analogue of the atlas `num_tiles` sweep (`steps = 106 +
slope·num_tiles`, R²=1.0) — a real GSIM-measured M2 cost curve for gemmini compute.

Files added: `gsim_gemmini_sweep_harness.cpp`, `gsim_gemmini_sweep.py`,
`gsim_gemmini_m2_sweep.{json,csv}`. Re-fit only: `gsim_gemmini_sweep.py --analyze`.

---

## 5. gemmini M3 — full SoC DUT `ChipTop` — **WORKING (re-rooted below `TestHarness`)**

**Result:** the full gemmini SoC compiles AND runs under GSIM. The Rocket core boots
from the on-chip BootROM (reset vector `0x10000`) and executes real instructions;
its pipeline **PC registers and the Gemmini controller FSM state are read every
cycle** off the public members of `class SChipTop`. This is the M3 register-observation
substrate on the whole CPU + TileLink interconnect + Gemmini SoC. **§5.5 bakes a real
gemmini matmul kernel into the BootROM so the Rocket + Gemmini actually EXECUTE it — a
real end-to-end M3 busy waterfall (CPU commits + Gemmini compute per cycle), not idle
boot.**

### 5.1 The old blocker and how it was removed

The 660k-line SoC `TestHarness.fir` (annotation-stripped, §2) advanced past graph
construction after two prior GSIM **engine** patches (in `$GSIM/src`, uncommitted):

- `AST2Graph.cpp:1653/1657` — guard `if (it->second) …->constructSuperNode()/Connect()`.
- `Node.cpp:100` — `NODE_EXT_OUT` guarded against a **null `parent`** (orphan extmodule
  output): `if (parent) parent->constructSuperNode(); else super = new SuperNode(this);`.

…but then **aborted** at `splitArray.cpp:115` on chipyard **test-harness-only** nodes:
the TSI↔TileLink serial bridge (`ram$tsi2tl$…`, `SimTSI`) and the harness
success/assertion nodes (`_success_T`, `WHEN_COND_*`). Those nodes exist **only because
`TestHarness` instantiates them.**

### 5.2 The unblock — re-root the circuit at `ChipTop` (path (a))

Module hierarchy in the `.fir`: `DigitalTop` (Rocket + L1/L2 + SystemBus/PeripheryBus
TileLink + Gemmini RoCC) → **`ChipTop`** (adds IOCells; exposes `axi4_mem_0` AXI-to-DRAM,
`serial_tl_0` TSI bringup, `uart_0`, `jtag`) → `TestHarness` (adds `SimDRAM`, `SerialRAM`,
`SimTSI`, `TSIToTileLink`, `UARTAdapter` — the offending plumbing).

`gsim_prune_to_dut.py` rewrites the `.fir` to re-root the circuit at `ChipTop`: it strips
the annotation blob (§2), sets `circuit ChipTop :`, and keeps **only the modules reachable
by `inst … of` from `ChipTop`** (structural BFS, no regex). This drops 453 harness-only
modules — `TestHarness`, `SerialRAM`, `SimTSI`, `TSIToTileLink`, `SimDRAM`, `UARTAdapter`,
all `*_SerialRAM` serdes — so the `ram$tsi2tl` / `_success_T` nodes never exist and
`splitArray` no longer aborts.

```bash
F=artifacts/gemmini/firrtl/chipyard.harness.TestHarness.GemminiRocketConfig.fir
SP=/scratch/agustin/projects/gsim-work/gemmini-soc; mkdir -p $SP
# 1. re-root TestHarness.fir at the ChipTop DUT (3099 modules -> 2646 reachable)
python3 <char>/gsim_prune_to_dut.py $F ChipTop $SP/ChipTop.fir
# 2. GSIM parse -> C++ : ~34 s, EXIT 0, emits ChipTop.h (34160 nodes) + ChipTop0.cpp (130 MB)
$GSIM/build/gsim/gsim --dir=$SP $SP/ChipTop.fir
# 3. one GSIM codegen defect recurs (im2col cross-partition temp leak, now hierarchy-
#    prefixed system$…$gemmini$im2col$_modulo_block_*_T); the prefix-agnostic
#    gsim_fix_gemmini_codegen.py handles it. (0 orphan setters here — that was accel-only.)
python3 <char>/gsim_fix_gemmini_codegen.py ChipTop $SP
# 4. compile with the four blackbox stubs + the harness
cp <char>/gsim_soc_blackboxes.cpp $SP/blackboxes.cpp
cp <char>/gsim_gemmini_soc_harness.cpp $SP/harness.cpp
$CXX -O0 -std=c++2b -DNDEBUG -I$SP $SP/ChipTop0.cpp $SP/harness.cpp $SP/blackboxes.cpp -o $SP/gemmini_soc_sim
./gemmini_soc_sim
```

### 5.3 The four blackboxes (`gsim_soc_blackboxes.cpp`)

GSIM emits *calls* to every FIRRTL `extmodule` but leaves the bodies to the user (its
shipped Rocket testbench links its own). The pruned `ChipTop.fir` reaches exactly four;
semantics are fixed by their FIRRTL port lists (no datapath reinterpretation):

| extmodule | ports (from `.fir`) | stub |
|---|---|---|
| `GenericDigitalInIOCell`  | `pad→i` (ie ignored)     | `i = pad & 1` |
| `GenericDigitalOutIOCell` | `o→pad` (oe ignored)     | `pad = o & 1` |
| `plusarg_reader`          | `(DEFAULT,FORMAT,WIDTH)→out` | `out = DEFAULT` (no runtime +args) |
| `EICG_wrapper`            | clock gate; GSIM passes only `test_en,en` (Clock in/out abstracted) | no-op |

### 5.4 Observed — per-cycle SoC register readout

`gsim_gemmini_soc_harness.cpp` holds `reset_io` for 10 cycles, deasserts, ties off the
AXI/serial slave readies, and reads per cycle: three Rocket pipeline PCs
(`…$rockettile$core$wb_reg_pc / ex_reg_pc / mem_reg_pc`), the three Gemmini controller
states (`…$gemmini$ex_controller$control_state`, `load_controller`, `store_controller`),
`reservation_station$_io_busy_T_4`, and `axi4_mem_0` `ar/aw` valids. GSIM's own commit
trace (`C0: 20 [1] pc=[10004] …`) agrees with the register readout:

```
phase,cycle,wb_reg_pc,ex_reg_pc,mem_reg_pc,gem_ex_state,…,axi_ar_valid,axi_aw_valid
run,19, 0x0,     0x10000, 0x0,     0,0,0,0, 0,0     <- boot vector enters the pipe
run,20, 0x0,     0x10004, 0x10000, 0,0,0,0, 0,0
run,21, 0x10000, 0x10008, 0x10004, 0,0,0,0, 0,0     <- PCs march stage-by-stage each cycle
run,39, 0x10020, 0x10028, 0x10024, 0,0,0,0, 0,0
```

`$` is part of the C identifier here (a clang extension), so the hierarchical member
names cannot be macro-abbreviated — the harness spells them in full.

**Old wall (now removed — see §5.5).** The core boots BootROM but then **spins at
pc≈`0x10034`**: the stock chipyard bootrom `wfi`-spins waiting for a **TSI serial
bringup** IPI to load the actual program — and TSI backing memory (`SerialRAM`/`SimTSI`)
is exactly what we pruned. `axi_ar_valid_cycles=0` (never reaches DRAM), Gemmini
controllers idle (`state 0`), no RoCC kernel. So bare-boot M3 proves whole-SoC per-cycle
**register observation**; a busy CPU/gemmini waterfall needs a program driven in, via
either (i) feeding the pruned `serial_tl_0` port a TSI bringup stream, or (ii) baking the
workload into the BootROM. §5.5 does (ii). Speed at `-O0`: **200k cycles in ~10 s (~20k
cyc/s)**; an `-O1` build is far faster to run but its 130 MB `.cpp` exceeds a 2-min
compile budget on this box.

### 5.5 Driving a real workload — BootROM-baked gemmini matmul → **real M3 busy waterfall**

We **bake a bare-metal gemmini matmul kernel into the BootROM** so the Rocket boots
straight into a RoCC command stream (no TSI, no DRAM). The stock bootrom disassembles to
a `wfi`/`j` spin at `0x10034` waiting for a TSI IPI; we replace it. ChipTop's BootROM is a
combinational `wire rom : UInt<64>[512]` mapped at reset vector **`0x10000`** (read path
`index = address[11:3]`, `TLROM`/`BootROM.scala:60`), lowered by GSIM to the public array
`system$bootrom_domain$bootrom$rom[512]`. GSIM **re-drives** those rom constants every
`step()` (event-gated init block in `subStep0`), so a per-cycle harness overwrite is
clobbered before the fetch — the kernel must be **baked into the generated
`ChipTop0.cpp`** source, not written from the harness.

The kernel (`gsim_gemmini_bootrom.S`, RV64, assembled with the extracted riscv-gcc14
`as`/`ld`/`objcopy` under `/scratch/agustin/projects/riscv-gcc14`) issues the SAME
DRAM-free RoCC stream as the M2 harness §4a — `config_ex` + `preload` +
`compute_preloaded` (funct 4) + **12× `compute_and_stay`** (funct 5) — then spins at
`0x1006e`. RoCC custom-3 encoding `opcode 0x7b, funct3=0b011 {xd=0,xs1=1,xs2=1}, funct7 =
gemmini funct`; operands are `li`-loaded into `a0`/`a1` (scratchpad/accumulator addrs,
no `mvin/mvout` ⇒ no DMA ⇒ no TLB).

```bash
SP=/scratch/agustin/projects/gsim-work/gemmini-soc
char=experiments/characterization
# 1. assemble kernel -> rom word image (gemmini_bootrom.h)
python3 $char/gsim_gemmini_bootrom_bake.py $SP
# 2. bake rom[0..13] into the generated ChipTop0.cpp (idempotent, no regex)
python3 $char/gsim_gemmini_bootrom_patch_cpp.py $SP/ChipTop0.cpp $SP/gemmini_bootrom.h
# 3. recompile the object once (~20s @ -O0), then relink is ~1s
$CXX -O0 -std=c++2b -DNDEBUG -I$SP -c $SP/ChipTop0.cpp -o $SP/ChipTop0.o
cp $char/gsim_gemmini_soc_matmul_harness.cpp $SP/matmul_harness.cpp
$CXX -O0 -std=c++2b -DNDEBUG -I$SP $SP/ChipTop0.o $SP/matmul_harness.cpp $SP/blackboxes.cpp -o $SP/gemmini_soc_matmul_sim
$SP/gemmini_soc_matmul_sim 800 > $SP/matmul_trace_raw.csv
# 4. waterfall via the repo engines
python3 $char/gsim_gemmini_soc_waterfall.py $SP/matmul_trace_raw.csv $SP/gemmini_m3_matmul_trace
```

The harness reads a REAL end-to-end trace per cycle off `SChipTop`: Rocket **commit**
stage (`core$wb_reg_valid / wb_reg_pc / wb_reg_inst`; a committed `inst&0x7f == 0x7b` is a
gemmini RoCC dispatch), the three Gemmini controller FSMs, reservation-station busy, and
`axi4_mem_0` ar/aw valid.

**Observed — CPU + accelerator BOTH live.** The Rocket executes the kernel (commit PC
marches `0x10000→0x1006e`, real instructions, not the wfi spin); `ex_controller$
control_state` **leaves idle** at cycle 53 and toggles `1↔0` with a **~32-cycle period**
— one 16×16 WS accumulate pass, matching the M2 slope of **31 cyc/matmul** (§4b):

```
cyc=28  commit inst=0x00b5307b  is_rocc=1   <- config_ex dispatched
cyc=36  commit inst=0x0cb5307b  is_rocc=1   <- preload
cyc=44  commit inst=0x08b5307b  is_rocc=1   <- compute_flip; rs_busy=1
cyc=53  gem_ex_state=1                       <- Gemmini LEAVES IDLE (compute)
cyc=65..76 commit inst=0x0ab5307b is_rocc=1  <- 12x compute_and_stay
... ex_state toggles 1<->0 every ~32 cyc through cycle 484 (13 matmul passes) ...
cyc=485 gem_ex_state=0, rs_busy=0            <- drains to quiescence
```

**Waterfall** (`gsim_gemmini_busy_trace` → `occupancy_from_busy_trace`,
`gemmini_m3_matmul_trace_summary.json`):

```
CPU: 398 commits, 15 gemmini RoCC dispatches   (config+preload+compute_flip+12x stay)
whole trace   (800 cyc):   {compute:418, dma:0, overlap:0, idle:382}  closes=true
active window [34..484]:    {compute:418, dma:0, overlap:0, idle:33}   closes=true  (93% compute)
```

This is a **real M3 busy waterfall**: the Rocket core and the Gemmini FSM are both active
per cycle (not idle boot). `dma`/`axi` stay 0 **by design** (DRAM-free kernel). The one
remaining wall is a `mvin/mvout`-fed matmul (nonzero load/store DMA buckets + AXI
activity), which needs the page-table walk / TLB satisfied — the same TLB wall documented
for gemmini M2 §4; the config+preload+compute path deliberately avoids it. Artifacts:
`gemmini_m3_matmul_trace.csv`, `..._summary.json`, `gemmini_m3_matmul_SUMMARY.md`,
`gemmini_bootrom.{S,h,elf,bin}` (all under `$SP`).

---

## 6. atlas M3 — full Atlas SoC `ChipTop` — **WORKING (re-rooted below `TestHarness`)**

The full Atlas SoC runs under GSIM with per-cycle register readout, and the Rocket
core **actually executes the chipyard bootrom** (not just idle observation). The DUT
is `ChipTop` (one level below `TestHarness`), which contains the Rocket scalar core +
the **Atlas NPU tile** (`atlasTile`, custom scalar core + VPU) + the TileLink
interconnect + internal bootrom — but **not** the TSI/serial harness plumbing that
blocks the naive `TestHarness` top (same lesson as gemmini §5).

### 6.1 The `.fir` was already on disk — no re-elaboration needed

Contrary to the earlier note in this section, the **full Atlas SoC FIRRTL already
exists**, produced by the chipyard-atlas verilator build (Jul 2026):

```
CHIPYARD_ATLAS=/scratch/agustin/projects/chipyard-atlas        # mlc/paths.py CHIPYARD_ATLAS_ROOT
FIR=$CHIPYARD_ATLAS/sims/verilator/generated-src/\
chipyard.harness.TestHarness.AtlasRocketConfig/\
chipyard.harness.TestHarness.AtlasRocketConfig.fir             # 132 MB, 1,005,921 lines
```

It is Chisel-emitted **FIRRTL 3.3.0, CIRCT/new-style** (`invalidate`, 10,419 of them;
**zero** `is invalid`), with the usual leading annotation blob (`circuit TestHarness
:%[[ … ]]`, ends line 39999). GSIM parses it fine (the "unexpected 'is'" reject is an
SFC-emission artifact, §2). To regenerate from scratch if ever deleted:
`cd $CHIPYARD_ATLAS && make CONFIG=AtlasRocketConfig` (chipyard's Chisel elaboration
emits this `.fir` before firtool lowers it to `.sv`). **This session did NOT invoke
mill/`Elaborate`/firtool at all** — it reused the on-disk `.fir`. (The tracked edits
under `third_party/atlas-npu/src/main/scala/atlas/*.scala` are dated 2026-07-05, five
weeks pre-session prior-M2 WIP; the untracked `atlas/Elaborate.scala` is likewise not
from this session — left untouched.)

### 6.2 Re-root `TestHarness` → `ChipTop` (drop the harness plumbing)

`TestHarness` instantiates `chiptop0 of ChipTop` **plus** `SerialRAM`(`tsi2tl`),
`SimTSI`, `SimDRAM`, `SimJTAG`, `UARTAdapter` — the exact §5 blocker class. `ChipTop`
instantiates only `system of DigitalTop` + pure `GenericDigitalIn/OutIOCell` +
`EICG_wrapper` (no harness blackboxes). So we extract the `ChipTop` module subtree
into a fresh circuit (**no regex**, structured line scan + instance-reachability BFS):

```bash
SP=/scratch/agustin/projects/gsim-work/atlas-soc; mkdir -p $SP
python3 experiments/characterization/gsim_extract_subtree.py "$FIR" ChipTop $SP/ChipTop.fir
#  root=ChipTop  modules 7475 -> 7022 reachable  (232 extmodule: IOCells+plusarg+EICG)
#  drops SerialRAM/tsi2tl/SimTSI/SimDRAM/SimJTAG/UARTAdapter/TestHarness (453 modules)
```

(`gsim_extract_subtree.py` is a generalization of the gemmini `gsim_prune_to_dut.py`;
either works. Rooting at `DigitalTop` — one level deeper — is a **dead end**: its
reset/clock inputs become undriven top ports and GSIM aborts at
`resetAnalysis.cpp:11 inferReset(): assignTree.size() > 0`. `ChipTop` is the sweet
spot: harness plumbing gone, reset tree still driven.)

### 6.3 Two GSIM engine patches (both confined to the debug/JTAG clock+reset domain)

`$GSIM/build/gsim/gsim --dir=$SP $SP/ChipTop.fir` hit **two new engine walls**,
both traced (by an added diagnostic) to the **same 3 nodes** — the chip-level debug
(JTAG) clock-gate + reset synchronizer that `ChipTop` wraps around the debug module:
`gated_clock_debug_clock_gate$en` / `$EICG_wrapper`, and
`debug_reset_syncd_debug_reset_sync$output_chain$reset`. Both patches are in
`$GSIM/src` (uncommitted), additive, and fire **only** on this degenerate debug
domain (a normal, non-JTAG run never exercises it); datapath regs are untouched:

1. **`graphPartition.cpp:45` `resort()` — `Assertion sortedSuper.size()==prevSize`
   (`invalid size 392331 392334`).** GSIM's Kahn topo-sort over the
   `depPrev`/`depNext` relation (register *activation ordering* — "clear activeFlags
   before activating reg", `Node::updateDep`, async-reset only) leaves **exactly 3
   supernodes** unvisited — a cycle in that relation, even though GSIM's `prev/next`
   loop detector correctly prints "NO Loop!" (different relation). It is **not**
   merge-induced (`--when-size=100000000` to disable `mergeWhenNodes` reproduces it).
   *Fix:* append the unvisited nodes in their prior `topoSort` order (valid w.r.t. the
   **hard** `prev/next` dependency); this relaxes only the **soft** activation
   ordering for the debug clock-gate/reset-sync cycle.
2. **`cppEmitter.cpp:690` `genResetAll()` — `Assertion mpz_sgn(consVal)==0`
   ("reset … is always true").** The debug reset-sync's async reset constant-folds to
   **1**. GSIM asserts a constant reset must be 0. *Fix:* treat always-true exactly
   like the already-handled always-false case — skip emitting a dynamic reset for it
   (the reg keeps its normal update; harmless for the inactive debug domain).

Rebuild after patching: `cd $GSIM && make build-gsim` (~env §0, clang 21).

### 6.4 Emit → compile → run

```bash
source $SP/env.sh          # = env §0
# 1) GSIM emit (split into ~4 MB files so clang stays sane; 1 file = 226 MB .cpp)
$GSIM/build/gsim/gsim --cpp-max-size-KB=4096 --dir=$SP $SP/ChipTop.fir   # ~60 s
#    -> ChipTop.h (3.7 MB) + ChipTop0..47.cpp (226 MB total, 48 files)
#    prints the 2 [atlas-M3 …] relaxation lines above, then "[cppEmitter] finish"

# 2) vector-IO fix: NO-OP here (ChipTop top IO is all scalar; bundles flatten to scalars)
python3 experiments/characterization/gsim_fix_vector_io.py ChipTop $SP   # "repaired 0"

# 3) SoC blackbox bodies — the 4 extmodules GSIM leaves to the user. The gemmini
#    peer's file has the exact matching signatures (IOCell passthrough, plusarg
#    default, EICG no-op — Clock ports abstracted by GSIM):
cp experiments/characterization/gsim_soc_blackboxes.cpp $SP/blackboxes.cpp
cp experiments/characterization/gsim_atlas_soc_harness.cpp $SP/harness.cpp

# 4) compile all 48 + harness + blackboxes to .o in parallel (-O0), then link
cd $SP
printf '%s\n' ChipTop*.cpp harness.cpp blackboxes.cpp | \
  xargs -P16 -I{} bash -c '"$CXX" -O0 -std=c++2b -I. -c "$1" -o "$1.o"' _ {}   # ~8 s wall
"$CXX" -O0 -std=c++2b ChipTop*.cpp.o harness.cpp.o blackboxes.cpp.o -o atlas_soc_sim
./atlas_soc_sim                                    # instant
```

### 6.5 Result — Rocket boots, per-cycle PC readout on the full SoC

The `ChipTop` model has **35,772 registers/signals**, all public members of
`class SChipTop` (`dut->system$tile_prci_domain$element_reset_domain$rockettile$core$wb_reg_pc`,
`dut->system$domain$atlasTile$core$scalar$pc_ctrl$fetch_pc_reg`, …). The harness
(`gsim_atlas_soc_harness.cpp`) drives `reset_io` then reads the Rocket writeback/mem
PC + wb-valid and the Atlas-tile scalar-core fetch PC every cycle:

```
cycle,rocket_wb_pc,rocket_mem_pc,rocket_wb_valid,atlas_fetch_pc
...
20,0x0,    0x10000,0,0x0     <- core released from reset, fetches reset vector 0x10000
21,0x10000,0x10004,1,0x0     <- first bootrom instr retires (wb_valid=1)
22,0x10004,0x10008,1,0x0
23,0x10008,0x10008,1,0x0
...
31,0x10010,0x10014,1,0x0     <- PC marching through the internal bootrom
```

GSIM also emits a **built-in commit log** to stderr — real RISC-V commits:
```
C0: 19 [1] pc=[10000] W[r10=10000][1] R[r0=0] R[r0=0] inst=[517]      DASM(517)      # auipc
C0: 20 [1] pc=[10004] W[r10=10040][1] R[r10=10000] R[r0=0] inst=[4050513]            # addi
C0: 21 [1] pc=[10008] W[r0=0][1] R[r10=10040] inst=[30551073]                        # csrw mtvec
C0: 26 [1] pc=[1000c] W[r5=800000000094112d][1] inst=[301022f3]                      # csrr mtvec-ish
C0: 29 [1] pc=[10010] W[r5=ffffe00000000025][1] inst=[4122d293]                      # srai
```

The core fetches from the SoC's **internal TileLink bootrom** (reset vector `0x10000`)
— no external DRAM is attached (`axi4_mem_0` / `serial_tl_0` are the ports we
intentionally left open by pruning the harness), so it runs the bootrom prologue and
then stalls awaiting DRAM. The Atlas tile's `fetch_pc_reg` is read every cycle (idle
at `0x0` until the boot code hands it work). **This is the M3 substrate: per-cycle RTL
register observation across the whole Atlas SoC — CPU + interconnect + NPU tile — at
GSIM speed, with a real instruction stream already flowing.** To drive the Atlas NPU
itself, load a kernel into DRAM via an AXI4/serial-TL harness model (next step; the
port is exposed on `SChipTop`).

### 6.6 Timings (this box)

| Step | atlas M3 (`ChipTop`) |
|---|---|
| extract `ChipTop.fir` from `TestHarness.fir` | ~15 s (948,911 lines out) |
| GSIM parse→C++ (split, 48 files) | ~60 s (peak RSS ~8 GB) |
| clang -O0 compile 48+2 files (`-P16`) | ~8 s wall |
| link + run (tens of cycles) | instant |

### 6.7 Files (atlas M3 deliverables)

- `gsim_extract_subtree.py` — re-root a `TestHarness.fir` at any submodule via
  instance-reachability BFS (no regex). Used: `… ChipTop ChipTop.fir`.
- `gsim_atlas_soc_harness.cpp` — reads Rocket PC (wb/mem) + wb-valid + Atlas fetch-PC
  per cycle.
- `gsim_soc_blackboxes.cpp` — (shared with gemmini M3) the 4 SoC extmodule bodies.
- GSIM engine patches in `$GSIM/src/{graphPartition.cpp,cppEmitter.cpp}` (uncommitted;
  each gated to the debug clock/reset cycle, with an `[atlas-M3 …]` stderr breadcrumb).
- Scratch outputs (heavy, not in repo): `/scratch/agustin/projects/gsim-work/atlas-soc/`
  (`ChipTop.fir`, `ChipTop*.cpp/.h`, `atlas_soc_sim`, `atlas_soc_run.csv`,
  `atlas_soc_commitlog.txt`, `env.sh`).

---

## 7. How register readout works (general)

GSIM emits every signal — **including all registers** — as a **public data member**
of `class S<Top>`. Read any register directly: `dut->my_reg`, `dut->sub$child_reg`
(FIRRTL hierarchy `.` becomes `$`, bundles `$$`). Top-level IO also has
`get_<port>()` / `set_<port>(val)` (repair vector ports with §2's script). One cycle =
`dut->step()`; reset = `dut->set_reset(1); …step…; dut->set_reset(0)`. There is **no
clock toggling** — `step()` is one full cycle. Register list + widths are in the
generated `<Top>.h` (grep `// width = … lineno = …`). This is exactly the per-cycle
observation the M2 waterfall needs, at ~GSIM speed.

---

## 8. Files (this deliverable)

- `gsim_fix_vector_io.py` — repair GSIM's vector top-IO accessor stubs (all accel modules).
- `gsim_fix_gemmini_codegen.py` — work around 2 GSIM codegen defects in the gemmini emission.
- `gsim_atlas_mxu_harness.cpp`, `gsim_gemmini_harness.cpp` — minimal standalone harnesses.
- `gsim_gemmini_cmd_encode.py` — no-regex RoCC command-stream encoder (emits `gemmini_cmd_stream.h`).
- `gsim_gemmini_cmd_harness.cpp` — command-driving gemmini harness (§4a: idle→compute waterfall).
- `gsim_prune_to_dut.py` — re-root a chipyard `TestHarness.fir` at a DUT submodule
  (`ChipTop`/`DigitalTop`) by structural instance-reachability BFS; drops the harness
  plumbing that trips `splitArray` (§5.2). No regex.
- `gsim_soc_blackboxes.cpp` — the four extmodule blackbox bodies the SoC DUT needs (§5.3).
- `gsim_gemmini_soc_harness.cpp` — full-SoC (ChipTop) M3 harness: per-cycle Rocket PC +
  Gemmini FSM readout (§5.4).
- `gsim_gemmini_bootrom.S` — bare-metal gemmini matmul kernel baked into the BootROM (§5.5).
- `gsim_gemmini_bootrom_bake.py` — assemble the kernel → BootROM word image `gemmini_bootrom.h`.
- `gsim_gemmini_bootrom_patch_cpp.py` — bake rom[0..N] into the generated `ChipTop0.cpp` (idempotent, no regex).
- `gsim_gemmini_soc_matmul_harness.cpp` — M3 harness that EXECUTES the baked kernel and reads
  per-cycle Rocket commit (pc/inst/valid) + Gemmini FSM + AXI valids (§5.5).
- `gsim_gemmini_soc_waterfall.py` — CSV → occupancy waterfall via the repo engines (§5.5).
- GSIM engine patches live in `$GSIM/src/{AST2Graph.cpp,Node.cpp,clockOptimize.cpp}`
  (uncommitted); required only for the SoC/M3 path.

## 9. Timings (this box)

| Step | atlas M2 | gemmini M2 |
|---|---|---|
| GSIM parse→C++ | 16.4 s | 10.6 s |
| clang -O1 compile | ~30 s | ~22 s |
| run (tens of cycles) | instant | instant |

Baseline Rocket+CoreMark: 4.2M cycles / 11.4 s ≈ 560k cyc/s.
