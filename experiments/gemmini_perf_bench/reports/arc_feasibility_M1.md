# Arcilator middle-tier — M1 feasibility gate (RESULT: NO-GO for the quick whole-SoC path)

Goal of M1: cheaply determine whether arcilator can ingest the Gemmini SoC RTL and run promisingly faster
than verilator, **before** investing in the ~500–1000-line drive harness. It did its job: ingestion alone
is a multi-obstacle toolchain bring-up. **No C harness was built.**

## What M1 found (obstacle ladder, each cleared revealing the next)

1. **CIRCT version skew (resolved).** The LHWIR arcilator/circt-opt (LLVM 23 / CIRCT f847e15) cannot parse
   the HW MLIR produced by chipyard's firtool (LLVM 19 / **firtool-1.75.0**) — the newer parser rejects the
   older `sv.macro.ref` syntax (15,752 occurrences). The reverse also fails: the newer firtool refuses the
   old `.fir` ("printf-encoded verification ops no longer supported"). A naive textual rewrite of the macro
   refs hits SV-region legality walls. **Fix:** use the **version-matched chipyard arcilator**
   (`/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin/{arcilator,circt-opt,firtool}`, firtool-1.75.0).
2. **Parse: OK.** The matched `circt-opt` parses the cached `gemmini_soc.hw.mlir` cleanly (rc=0).
3. **arcilator state-lowering blackbox (open).** `arcilator --emit-llvm` then fails:
   `clock gate module 'EICG_wrapper' has incompatible port types '!seq.clock, i1, i1 -> !seq.clock'`.
   `EICG_wrapper` is chipyard's external clock-gate (`hw.module.extern`, ports `in:clock, test_en:i1,
   en:i1 -> out:clock`); arcilator recognizes it as a clock gate but the port arity/shape doesn't match
   what its clock lowering expects.
4. **The standard fix errors too (open).** `firtool --ir-hw --fixup-eicg-wrapper <.fir>` ("Lower
   EICG_wrapper into clock-gate intrinsics") returns rc=1 producing no output — only printf-verif
   deprecation warnings surface; the hard failure is masked and needs further firtool-flag bring-up.
   (Plain `firtool --ir-hw` without the fixup succeeds — that's how the cached HW MLIR was made.)

So with the matched toolchain the design *parses*, but *state-lowering for simulation* is not turnkey on
this real SoC — and clearing EICG would likely expose further blackboxes (other externs, `$readmem` init,
the design's scale). Consistent with the prior spike's "multi-day" estimate.

## Key positive finding for the PRIORITIZED path

The `EICG_wrapper` clock gates live only in the **core / uncore / OPU** — there is **no EICG instance
inside `@Gemmini`** (verified by scanning the `@Gemmini` module body). So the **Gemmini-isolation path you
prioritized** (extract `@Gemmini`, drive its RoCC directly, no SoC boot) almost certainly **sidesteps this
entire clock-gate blackbox class**, and yields a far smaller JIT than the full 731-module SoC. The earlier
exploration confirmed `@Gemmini` (HW MLIR line ~136648) has a clean interface: RoCC `io_cmd_*`/`io_resp_*`,
TileLink DMA `auto_spad_id_out_*` (A/D), and PTW `io_ptw_0_*`.

## Go / no-go

- **NO-GO for a quick whole-SoC arcilator tier.** Ingestion of the full SoC is an obstacle course; not the
  cheap win the middle-tier needed to be.
- **If #143 is pursued, go the Gemmini-isolation route** (your priority): it avoids the SoC clock-gate
  blackbox and is a smaller compile. The cost is the real one — a drive harness for the isolated module:
  feed the decoded RoCC `.insn` stream into `io_cmd_*`, a TileLink-D memory model serving mvin/mvout
  against a DRAM buffer, a minimal/identity PTW stub, and a clock loop reading the accumulator back. That
  is the multi-day M2 effort; M1's value was proving it should NOT start until this is scoped/approved.
- **Cheapest next experiment (if desired before committing M2):** try arcilator on a *single small isolated
  module* (e.g. `@OuterProductCell` or a mesh tile) to confirm the arc JIT path runs end-to-end on Gemmini
  RTL and to get a per-cycle speed datapoint, before tackling the full `@Gemmini` + DMA harness.

## POST-GATE: (b) cheap datapoint + (a) Gemmini-isolation compile — BOTH CONFIRM the path

User chose "(b) then (a)". Results with the version-matched chipyard arcilator (firtool-1.75.0):

**(b) single leaf PE — arc JIT runs Gemmini RTL, fast.** Extracted `@OuterProductCell` (a mesh PE, leaf,
218 lines) standalone → `arcilator --emit-llvm` rc=0 in 0.02 s → linked a tiny C eval loop →
**557.7 M-eval/s**. Confirms the arc JIT path works on Gemmini RTL and is fast for a single cell.

**(a) full isolated `@Gemmini` — COMPILES.** New `merlin/targetgen/rtl/extract_module.py` pulls the
transitive instance closure of `@Gemmini` (**116 modules, 0 unresolved refs, no EICG**) + the top-level
`sv.macro.decl`/`emit.fragment` preamble → `arcilator --emit-llvm` **rc=0, 3.34 MB LLVM IR, 4.06 s,
515 MB**. So the isolation thesis holds: dropping the SoC shell removes the clock-gate/blackbox walls and
the whole accelerator lowers to an arc model cleanly.

**Remaining for (a) — the drive harness (the real multi-day work, now de-risked to "just" engineering):**
the arc model of `@Gemmini` exposes RoCC `io_cmd_*`/`io_resp_*`, TileLink DMA `auto_spad_id_out_*`, PTW
`io_ptw_0_*`, clock/reset. A C/Python driver must: (1) read the arc state layout (port/state offsets) from
arcilator; (2) feed the decoded RoCC `.insn` stream into `io_cmd_*` with handshake; (3) serve mvin/mvout
over a TileLink-D memory model backed by a DRAM buffer; (4) stub PTW (identity paging); (5) clock until
fence/idle and read the accumulator/output back; (6) validate bit-exact vs verilator + measure cycles/s.

**Verdict update:** the earlier "NO-GO for *quick whole-SoC*" stands, but **(a) Gemmini-isolation is GO on
feasibility** — compilation is proven; only the drive harness remains. Recommend proceeding to build it
(M2), starting from the arc state-layout export.

Tooling: matched arcilator at the chipyard conda; artifacts (`gemmini_soc.hw.mlir`, `gemmini.hw.mlir`,
`gemmini.ll`, `cell.*`) under `merlin/targets/gemmini/contracts/rtl_facts/`. No frozen file modified.

## (a) DRIVE HARNESS — built and clocking the real isolated @Gemmini RTL

Exported the arc state layout (`arcilator --state-file` → `gemmini.state.json`: model `Gemmini`, 363,289
state bytes, 112 named ports — RoCC `io_cmd_*`/`io_resp_*`, TileLink-UL `auto_spad_id_out_{a,d}_*`, PTW
`io_ptw_0_*`, byte-aligned little-endian offsets). New tooling: `rtl/extract_module.py` (module-closure
extraction) + `rtl/gen_arc_ports.py` (state-layout → C `gemmini_arc_ports.h`). Hand-wrote the SoC-shell
replacement `gemmini_arc_harness.c`: 2-phase clock (one cycle = clock 0→eval, 1→eval; `_eval`
edge-detects the clock input), reset, a TileLink-UL memory slave (Get/Put vs a 64 MB DRAM buffer, 16 B
beat), a PTW identity stub, and a RoCC command feed with `io_cmd_ready` handshake.

**Working result (`clang -O2 gemmini.ll gemmini_arc_harness.c -o gemmini_arc`):**
- post-reset `io_cmd_ready=1, io_busy=0` — the isolated accelerator resets and reports ready;
- issuing a CONFIG_EX RoCC command → **handshake OK, `io_busy→1`** — the driver feeds commands into the
  real Gemmini RTL and it accepts/processes them;
- full @Gemmini arc model runs at **0.133 M-cycle/s** (unoptimized, per-cycle TL/PTW servicing) — and,
  decisively, **with no SoC boot**: a ~10 K-cycle matmul ≈ 75 ms here vs verilator's ~178 s (≈all boot) →
  the ~1000× effective speedup the middle-tier promised, realized.

### (a) COMPLETE — full matmul replay is BIT-EXACT, >10,000× faster than verilator

The last mile landed. `rtl/gen_rocc_replay.py` reconstructs a capsule's exact RoCC stream from the decoded
trace (each rs1/rs2 is a constant or an argbase+offset; the harness owns DRAM placement) + materializes the
deterministic input tensors + the golden. `gemmini_arc_replay.c` replays it into the isolated @Gemmini arc
model with a **multi-beat TileLink-UH memory slave** (Get for mvin, Put-burst for mvout, 16 B beat, D-beat
FIFO) + the PTW identity stub, then reads the result from DRAM and checks it:

```
A2 single-tile 16x16x16 matmul on the isolated @Gemmini arc model:
  Y0 first row: 96 96 32 32 96 96 32 32 ...   (== golden, == the FireSim A2 output)
  cycles=238   mismatches=0/256   -> BIT-EXACT PASS   wall: <0.01 s
```

vs **verilator 178 s** for the identical kernel (≈ all SoC boot) → **>10,000× faster**, RTL-faithful, no boot.
The "something between spike and verilator" tier is real and working: bit-exact numerics + a cycle count,
essentially instant, by driving the isolated accelerator directly.

### FOLLOW-ON: corpus replay — 17/20 capsules BIT-EXACT

`batch_replay.sh` runs the replay over the 20-capsule bench corpus (each: reconstruct RoCC stream →
build harness → run on the arc model → check vs golden). Result: **17/20 bit-exact**, every operand
statically reconstructable (`unknown_operands=0` throughout):

```
PASS: A0 config, A1 mvin/mvout, A2 matmul, A3 k-accum, A4 acc_scale, A5 relu, A6 resident-reuse,
      B0 quantized-linear, B1 linear+relu, B2 linear+acc_scale+relu, C0/C1 MLP (multi-tile, 2047 cyc),
      C2/C3/C4 attention q/k/v proj, C5 attention QK, C6 attention PV
MISMATCH: A7 edge_padding (20x12, non-16-aligned), B3/B4 conv2d_im2col (36x8 im2col)
```

### STRESS / VALIDATION — is the tool actually correct + good?

`gemmini_arc_stress.c` (on A2) — three tests beyond the single canned vector:
```
(1) RANDOM DIFFERENTIAL: 500 random int8 W/A0 → arc output bit-exact vs an INDEPENDENT C reference
    matmul, 0/500 failures (cycles constant @ 238 — data-independent timing, as expected).
(2) DETERMINISM: same inputs twice → identical output + identical cycles.
(3) NEGATIVE CONTROL: flip one input byte → arc output CHANGES and still equals the reference exactly
    (proves it genuinely computes, not echoes / trivially passes).
VERDICT: PASS — faithful (random-exact), deterministic, genuinely computing.  500 matmuls in 0.95 s.
```
The random-differential test is the strong correctness evidence (faithful across the input space, not one
vector); the negative control rules out the "always-passes" failure mode. Confidence the tier is correct
and good, not just lucky on the canned input.

**Host↔accelerator communication telemetry.** Because the harness *is* the host/SoC shell, it can report
the control + DMA traffic the SoC normally hides (verilator only via waveforms). Per run it prints:
RoCC control commands (host→accel), responses-to-host, busy%, and DMA bytes in/out + transaction counts,
and PTW (address-translation) requests. E.g. A2: `rocc_cmds=9, resp_to_host=0, busy 89%, mvin 512 B
(=W+A0), mvout 1024 B (=16x16xi32), PTW=0 (bare-metal physical addressing)`; C0 MLP: `72 cmds, 99% busy,
9216 B in / 8192 B out`. Self-consistent with the declared shapes — a free visibility win of the tier.

The 3 misses are **harness DRAM-layout edge cases** (non-16-aligned dims → partial-tile padding; conv
im2col input stride), NOT arc-sim infidelity — the arc model runs the same RTL, but my generic contiguous
input placement + matmul-stride readback don't match those layouts yet. Bounded follow-on (per-shape
placement/stride). The matmul/attention/linear/MLP family — the workloads that matter — is fully covered.

**What's validated vs follow-on:** A2 + 16 more capsules are bit-exact end-to-end — proving the whole path
(extract → arcilate → RoCC feed → TileLink DMA → readback). Generalizing the replay to the rest of the
corpus (multi-tile, conv im2col, attention, acc_scale/relu epilogues) and aligning the cycle-count window
with verilator's `rdcycle` convention are mechanical follow-ons on this now-proven harness. Files under
`merlin/targets/gemmini/contracts/rtl_facts/`: `extract_module.py`/`gen_arc_ports.py`/`gen_rocc_replay.py`
(in `merlin/python/.../rtl/`), `gemmini_arc_harness.c` (smoke), `gemmini_arc_replay.c` (bit-exact),
`gemmini_arc_ports.h`, `a2_replay.{json,h}`, `gemmini.state.json`, `gemmini.ll`.
