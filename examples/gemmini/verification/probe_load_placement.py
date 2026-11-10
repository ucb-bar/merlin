#!/usr/bin/env python3
"""What load states, move-in placement, readout placement and weight reuse are worth to an open-coded nest.

This is a target-local measurement probe, not a compiler test. It builds ONE bare-metal binary carrying the
same contraction under several scheduling policies, runs it on a selected RTL engine and reads the unit's own
performance counters. The findings it produced, and the hardware revision they belong to, are recorded in
`docs/guides/12-experiments.md`.

An open-coded (sequencer-free) contraction has to configure the load unit before every move-in whose
DRAM row pitch differs from the last one configured. A backend that uses ONE configuration state must
therefore re-issue that configuration every time it switches operand -- the stationary operand, the
moving operand and the bias seed have three different pitches -- and, more consequentially, it can
only ever emit the move-ins of one operand in an uninterrupted burst, because interleaving them would
cost a configuration instruction between every pair.

The unit has more than one such state. How many is a property of the RTL, read here from the target's
own ISA header (the move-in opcodes it defines) rather than assumed, and the configuration instruction
names which state it writes. Giving each operand its own state makes the per-block reconfiguration
disappear AND lifts the ordering constraint, so the two effects are measured apart:

    ld1_hoist   one state, per-block reconfiguration, operands moved in in bursts   (the baseline)
    ld3_hoist   one state per operand, configured once, same burst order            (- the config cost)
    ld3_jit     one state per operand, each tile moved in just before its first use  (+ the reordering)
    ld1_jit     one state, just-in-time order, paying a reconfiguration per move-in  (the control that
                shows the states are what makes the reordering affordable)

`loop_ws` -- the hardware loop sequencer -- is measured alongside as the correctness oracle every arm
is compared against element for element. It is NOT the thing being optimised: a result that routes
through the sequencer measures the hardware's tiling choice, not a compiler's schedule.

Arms live in ONE bare-metal binary and are compared only against each other inside it, because an
arm's measured cycles move with how many output buffers its neighbours in the same binary declared.

    probe_load_placement.py --target gemmini --shape 3136x64x64 --emulator <engine> --out <dir>
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from merlin.common import provenance
from merlin.runtime.backends import base

#: The hardware revision a cycle number here belongs to.
RTL_PIN = "gemmini_rtl"

CFLAGS = (
    "-DPREALLOCATE=1", "-DMULTITHREAD=1", "-mcmodel=medany", "-std=gnu99", "-O2", "-ffast-math",
    "-fno-common", "-fno-builtin-printf", "-fno-tree-loop-distribute-patterns", "-march=rv64gc",
    "-Wa,-march=rv64gc", "-lm", "-lgcc", "-DID_STRING=", "-Wno-incompatible-pointer-types",
    "-nostdlib", "-nostartfiles", "-static", "-DBAREMETAL=1",
)  # fmt: skip

BANKS: dict[str, tuple[str, ...]] = {
    "occupancy": (
        "MAIN_LD_CYCLES",
        "MAIN_ST_CYCLES",
        "MAIN_EX_CYCLES",
        "MAIN_LD_ST_CYCLES",
        "MAIN_LD_EX_CYCLES",
        "MAIN_ST_EX_CYCLES",
        "MAIN_LD_ST_EX_CYCLES",
        "RESERVATION_STATION_ACTIVE_CYCLES",
    ),  # fmt: skip
    "stall": (
        "EXE_ACTIVE_CYCLE",
        "EXE_FLUSH_CYCLE",
        "EXE_CONTROL_Q_BLOCK_CYCLE",
        "EXE_PRELOAD_HAZ_CYCLE",
        "EXE_OVERLAP_HAZ_CYCLE",
        "SCRATCHPAD_A_WAIT_CYCLE",
        "SCRATCHPAD_B_WAIT_CYCLE",
        "RESERVATION_STATION_FULL_CYCLES",
    ),  # fmt: skip
    "dma": (
        "LOAD_ACTIVE_CYCLE",
        "LOAD_DMA_WAIT_CYCLE",
        "LOAD_SCRATCHPAD_WAIT_CYCLE",
        "STORE_ACTIVE_CYCLE",
        "STORE_DMA_WAIT_CYCLE",
        "STORE_SCRATCHPAD_WAIT_CYCLE",
        "RDMA_ACTIVE_CYCLE",
        "WDMA_ACTIVE_CYCLE",
    ),  # fmt: skip
}


def _defines(text: str) -> dict[str, str]:
    """``#define NAME VALUE`` out of a C header, read structurally (no pattern matching).

    Only the un-parameterised defines are returned: a name followed by ``(`` is a macro, whose value
    is not a constant of this build.
    """
    out: dict[str, str] = {}
    for line in text.splitlines():
        head, marker, rest = line.partition("#define ")
        if head.strip() or not marker:
            continue
        name, _, value = rest.strip().partition(" ")
        if name and "(" not in name:
            out[name] = value.strip()
    return out


class Facts:
    """The geometry and capacity this probe must respect, read from the target harness's own headers.

    Nothing here is a literal: a target whose array edge, accumulator depth or operand-store depth
    differs produces a different blocking, and a target whose ISA header defines fewer move-in
    opcodes yields fewer load states -- which the probe reports rather than assuming three.
    """

    def __init__(self, harness: Path) -> None:
        params = _defines((harness / "include" / "gemmini_params.h").read_text(encoding="utf-8"))
        isa = _defines((harness / "include" / "gemmini.h").read_text(encoding="utf-8"))
        self.dim = int(params["DIM"])
        self.addr_len = int(params["ADDR_LEN"])
        self.acc_rows = int(params["ACC_ROWS"])
        self.spad_rows = int(params["BANK_NUM"]) * int(params["BANK_ROWS"])
        self.max_bytes = int(params["MAX_BYTES"])
        # The move-in opcodes the ISA defines ARE the load states: `k_MVIN`, `k_MVIN2`, `k_MVIN3` name
        # one configuration state each, and the configuration instruction's id field selects among
        # exactly those. Counting them is how many states this build has.
        self.mvin_functs = [n for n in ("k_MVIN", "k_MVIN2", "k_MVIN3") if n in isa]
        self.load_states = len(self.mvin_functs)
        # The local-address flags, transcribed from the header's own spellings rather than invented:
        # `1 << (ADDR_LEN-1)` is "this is an accumulator address", `1 << (ADDR_LEN-2)` "accumulate
        # into it rather than overwrite".
        self.acc_bit = 1 << (self.addr_len - 1)
        self.accumulate_bit = 1 << (self.addr_len - 2)


def counter_codes(harness: Path) -> dict[str, int]:
    codes: dict[str, int] = {}
    incremental = 0
    for line in (harness / "include" / "gemmini_counter.h").read_text(encoding="utf-8").splitlines():
        head, marker, rest = line.partition("#define ")
        if head.strip() or not marker or not rest.strip():
            continue
        name, _, value = rest.strip().partition(" ")
        value = value.strip()
        if name == "INCREMENTAL_COUNTERS":
            incremental = int(value)
        elif value.isdigit():
            codes[name] = int(value)
        elif value.startswith("(INCREMENTAL_COUNTERS + ") and value.endswith(")"):
            codes[name] = incremental + int(value[len("(INCREMENTAL_COUNTERS + ") : -1])
    return codes


def block_shape(f: Facts, mt: int, nt: int, kt: int) -> tuple[int, int, int]:
    """The blocking every arm shares: whole K where it fits, then the least operand re-reading.

    The accumulator holds ``bm*bn`` output tiles and the operand store ``(bm+bn)*bk``. K is taken as
    deep as both budgets allow so the accumulator is read out once per output block -- the
    accumulator-resident-K shape the open-coded path already has. Among the (bm, bn) that then fit,
    the one chosen is the one whose operand traffic is smallest: the moving operand is re-read once
    per column block and the stationary one once per row block, so a lopsided block pays for itself
    in DMA. The choice is HELD FIXED across arms -- it is not what is being measured.
    """
    best, best_cost = (1, 1, 1), None
    for bk in range(min(kt, f.spad_rows // f.dim), 0, -1):
        for bn in range(min(nt, f.acc_rows // f.dim), 0, -1):
            for bm in range(min(mt, f.acc_rows // (f.dim * bn)), 0, -1):
                if (bm + bn) * bk * f.dim > f.spad_rows:
                    continue
                cost = mt * kt * (-(-nt // bn)) + kt * nt * (-(-mt // bm))
                if best_cost is None or cost < best_cost:
                    best, best_cost = (bm, bn, bk), cost
                break
        if best_cost is not None:
            return best
    return best


# --------------------------------------------------------------------------------------------------
# The arms.  Every arm runs the SAME loop order (k, j, i -- the weight-stationary order the vendor's
# own open-coded nest uses) over the SAME blocking and issues the SAME move-ins to the same addresses.
# What differs between them is (a) which load configuration state each move-in reads and (b) where in
# the nest the move-in sits.  Nothing else.
# --------------------------------------------------------------------------------------------------

_SEED = """
        /* seed this output block's accumulator region with the bias row */
        {cfg_bias}
        for (int a = 0; a < bmc; a++)
          for (int d = 0; d < bnc; d++)
            {mvin_bias}((const elem_t *)(D + (n0 + d) * DIM),
                        ACC_NEW + (a * BN + d) * DIM, DIM, TR);
"""

_BURST = """
            {cfg_a}
            for (int a = 0; a < bmc; a++)
              for (int b = 0; b < bkc; b++)
                {mvin_a}(A + ((m0 + a) * TR) * K + (k0 + b) * DIM,
                         (a * BK + b) * DIM, DIM, TR);
            {cfg_b}
            for (int b = 0; b < bkc; b++)
              for (int d = 0; d < bnc; d++)
                {mvin_b}(B + ((k0 + b) * DIM) * N + (n0 + d) * DIM,
                         (BM * BK + b * BN + d) * DIM, DIM, DIM);
"""

_CONTRACT_HOISTED = """
            for (int b = 0; b < bkc; b++)
              for (int d = 0; d < bnc; d++)
                for (int a = 0; a < bmc; a++) {{
{stationary}
{fused_store}
                }}
"""

_CONTRACT_JIT = """
            for (int b = 0; b < bkc; b++)
              for (int d = 0; d < bnc; d++)
                for (int a = 0; a < bmc; a++) {{
                  /* each tile moved in immediately before the first compute that reads it */
                  if (d == 0) {{
                    {cfg_a}
                    {mvin_a}(A + ((m0 + a) * TR) * K + (k0 + b) * DIM,
                             (a * BK + b) * DIM, DIM, TR);
                  }}
                  if (a == 0) {{
                    {cfg_b}
                    {mvin_b}(B + ((k0 + b) * DIM) * N + (n0 + d) * DIM,
                             (BM * BK + b * BN + d) * DIM, DIM, DIM);
                  }}
{stationary}
{fused_store}
                }}
"""

#: The stationary operand shifted into the array ONCE for the run of rows that read it: the tile is
#: named on the first of them and the rest say `garbage`, which is how the vendor's own open-coded
#: nest spells weight reuse. The mesh pays a full array-edge of shift-in cycles per NAMED tile.
_STATIONARY_REUSE = """                  gemmini_extended_preload(
                      a == 0 ? (BM * BK + b * BN + d) * DIM : GARBAGE_ADDR,
                      ACC_ACC + (a * BN + d) * DIM, DIM, DIM, DIM, TR);
                  if (a == 0) {
                    gemmini_extended_compute_preloaded((a * BK + b) * DIM, GARBAGE_ADDR,
                                                       DIM, TR, DIM, DIM);
                  } else {
                    gemmini_extended_compute_accumulated((a * BK + b) * DIM, GARBAGE_ADDR,
                                                         DIM, TR, DIM, DIM);
                  }"""

#: The same contraction with the stationary operand NAMED on every compute -- the array reloads the
#: identical weights before each row of the moving operand. Not a straw man: this is what an emitter
#: writes when the loop that reuses the stationary tile is not the one the preload is keyed on.
_STATIONARY_RELOAD = """                  gemmini_extended_preload(
                      (BM * BK + b * BN + d) * DIM,
                      ACC_ACC + (a * BN + d) * DIM, DIM, DIM, DIM, TR);
                  gemmini_extended_compute_preloaded((a * BK + b) * DIM, GARBAGE_ADDR,
                                                     DIM, TR, DIM, DIM);"""

#: The readout of one output tile, issued the moment its last contribution retires rather than after
#: the whole block has finished. An output tile is final as soon as the last k-tile's compute for it
#: has issued, so no second accumulator region is needed -- only a different place for the same
#: instruction. The store unit then drains one tile while the mesh is still working on the next.
_FUSED_STORE = """                  if (k0 + BK >= KT && b == bkc - 1)
                    gemmini_extended_mvout(C + ((m0 + a) * TR) * N + (n0 + d) * DIM,
                                           ACC_ACC + (a * BN + d) * DIM, DIM, TR);
"""

_BLOCK_READOUT = """      /* the whole block read out after its contraction has finished */
      for (int a = 0; a < bmc; a++)
        for (int d = 0; d < bnc; d++)
          gemmini_extended_mvout(C + ((m0 + a) * TR) * N + (n0 + d) * DIM,
                                 ACC_ACC + (a * BN + d) * DIM, DIM, TR);
"""

_ARM = """
static void {name}_fn(const elem_t *A, const elem_t *B, const acc_t *D, elem_t *C) {{
  gemmini_extended_config_ex(WEIGHT_STATIONARY, ACT & 3, 0, 1, 0, 0);
  gemmini_extended_config_st(N * sizeof(elem_t), ACT & 3, SCALE);
{prologue}
  for (int m0 = 0; m0 < MT; m0 += BM) {{
    const int bmc = (m0 + BM <= MT) ? BM : MT - m0;
    for (int n0 = 0; n0 < NT; n0 += BN) {{
      const int bnc = (n0 + BN <= NT) ? BN : NT - n0;
{seed}
      for (int k0 = 0; k0 < KT; k0 += BK) {{
        const int bkc = (k0 + BK <= KT) ? BK : KT - k0;
{body}
      }}
{readout}    }}
  }}
  gemmini_fence();
}}
"""

_LOOPWS = """
static void {name}_fn(const elem_t *A, const elem_t *B, const acc_t *D, elem_t *C) {{
  /* repeating_bias: the bias is ONE row broadcast down the output, which is what every arm here
     seeds the accumulator with. Passing it as a full M-row matrix would make the oracle read
     N*(M-1) elements past the end of the bias buffer -- and it would still be called a comparison. */
  tiled_matmul_auto(M, N, K, A, B, (const void *)D, (void *)C,
                    K, N, N, N, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
                    ACT, SCALE, 0, true, false, false, false, false, 0, WS);
}}
"""


def arm_source(
    name: str, states: int, placement: str, store: str, stationary: str, f: Facts, a_pitch: int, b_pitch: int
) -> str:
    """One arm's C: the load states, where the move-ins sit, where the readout sits, weight reuse."""
    # A configuration state per operand where the build has them; otherwise everything shares state 0
    # and every switch of pitch costs a reconfiguration where the move-in is.
    ids = (0, 1, 2) if states >= 3 else (0, 0, 0)
    mv = [f"gemmini_extended_mvin{'' if i == 0 else i + 1}" for i in ids]
    shared = states < 3

    def cfg(stride: str, which: int) -> str:
        line = f"gemmini_extended3_config_ld({stride}, MVIN_SCALE_IDENTITY, false, {ids[which]});"
        if not shared:
            return line
        # One state means a single memo slot, and a backend that owns one re-issues the
        # configuration exactly when the pitch it must carry is not the pitch already live. The
        # guard below IS that memo, evaluated where the move-in is; it costs the host a compare and
        # the accelerator nothing, so the arm is charged for the configurations a single-slot
        # backend would really issue and not for the ones it would elide.
        return f"if (live_ld != (int32_t)({stride})) {{ {line} live_ld = (int32_t)({stride}); }}"

    cfg_a, cfg_b, cfg_d = cfg("K * sizeof(elem_t)", 0), cfg("N * sizeof(elem_t)", 1), cfg("0", 2)
    prologue = "  int32_t live_ld = -1; (void)live_ld;" if shared else "  " + cfg_a + "\n  " + cfg_b + "\n  " + cfg_d
    seed = _SEED.format(cfg_bias=cfg_d if shared else "", mvin_bias=mv[2])
    fused = _FUSED_STORE if store == "fused" else ""
    stat = _STATIONARY_RELOAD if stationary == "reload" else _STATIONARY_REUSE
    if placement == "burst":
        body = _BURST.format(
            cfg_a=cfg_a if shared else "",
            mvin_a=mv[0],
            cfg_b=cfg_b if shared else "",
            mvin_b=mv[1],
        ) + _CONTRACT_HOISTED.format(fused_store=fused, stationary=stat)
    else:
        body = _CONTRACT_JIT.format(
            cfg_a=cfg_a if shared else "",
            mvin_a=mv[0],
            cfg_b=cfg_b if shared else "",
            mvin_b=mv[1],
            fused_store=fused,
            stationary=stat,
        )
    return _ARM.format(
        name=name, prologue=prologue, seed=seed, body=body, readout="" if store == "fused" else _BLOCK_READOUT
    )


#: name -> (load states, where the move-ins sit, where the readout sits, stationary-operand policy)
ARMS: dict[str, tuple[int, str, str, str]] = {
    "ld1_hoist": (1, "burst", "block", "reuse"),
    "ld3_hoist": (3, "burst", "block", "reuse"),
    "ld3_jit": (3, "jit", "block", "reuse"),
    "ld1_jit": (1, "jit", "block", "reuse"),
    "ld3_jit_store": (3, "jit", "fused", "reuse"),
    "ld1_hoist_store": (1, "burst", "fused", "reuse"),
    # The stationary operand named on every compute -- what the convolution nest does today.
    "ld3_hoist_reload": (3, "burst", "block", "reload"),
    "ld3_jit_store_reload": (3, "jit", "fused", "reload"),
}

PROGRAM = """/* GENERATED by probe_load_placement.py -- one contraction, several scheduling policies, counted. */
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include "include/gemmini_testutils.h"

#define M {m}
#define N {n}
#define K {k}
#define MT {mt}
#define NT {nt}
#define KT {kt}
#define BM {bm}
#define BN {bn}
#define BK {bk}
#define TR {tile_rows}
#define SCALE {scale}f
#define ACT {relu}
#define WEIGHT_STATIONARY WS
#define ACC_NEW ((uint32_t){acc_bit}u)
#define ACC_ACC ((uint32_t){acc_acc}u)

#define BLOB(sym, file, type) \\
    __asm__(".section .rodata\\n.balign 64\\n.global " #sym "\\n" #sym ":\\n.incbin \\"" file "\\"\\n.previous\\n"); \\
    extern const type sym[];
{blobs}

{buffers}

{definitions}

static const char *const COUNTER_NAMES[8] = {{{counter_names}}};
static const int COUNTER_CODES[8] = {{{counter_codes}}};

static void arm_counters_configure(void) {{
    counter_reset();
    for (int i = 0; i < 8; i++) counter_configure(i, COUNTER_CODES[i]);
}}

static void arm_counters_print(const char *arm) {{
    for (int i = 0; i < 8; i++)
        printf("AB_COUNTER %s %s=%u\\n", arm, COUNTER_NAMES[i], (unsigned)counter_read(i));
}}

int main(void) {{
    gemmini_flush(0);
    uint64_t t0, cyc;
    printf("MERLIN_PROFILE warmup begin\\n");
{warmups}
    printf("MERLIN_PROFILE warmup end rc=0\\n");
    printf("MERLIN_PROFILE measured begin\\n");
{measures}
    printf("MERLIN_PROFILE measured end rc=0\\n");
    return 0;
}}
"""

MEASURE = """    arm_counters_configure();
    t0 = read_cycles(); {arm}_fn((const elem_t *)IN, (const elem_t *)WT, (const acc_t *)BS, OUT_{arm}); cyc = read_cycles() - t0;
    printf("AB_CYCLES {arm}=%llu\\n", (unsigned long long)cyc);
    arm_counters_print("{arm}");
    {{ long long diffs = 0, first = -1, sum = 0;
      for (size_t i = 0; i < {elements}; i++) {{
        sum += (long long)OUT_{arm}[i];
        if (OUT_{arm}[i] != OUT_{ref}[i]) {{ if (first < 0) first = (long long)i; diffs++; }} }}
      printf("AB_CHECK {arm} diffs=%lld first=%lld sum=%lld\\n", diffs, first, sum); }}
"""


def _rng(shape, seed, dtype=np.int8):
    g = np.random.default_rng(seed)
    if dtype == np.int8:
        return g.integers(-127, 128, size=shape, dtype=np.int16).astype(np.int8)
    return g.integers(-(1 << 14), 1 << 14, size=shape, dtype=np.int32)


def build(names, sources, arrays, elements, ref, bank, codes, geom, out, harness, compiler) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    for symbol, array in arrays.items():
        (out / f"{symbol}.bin").write_bytes(np.ascontiguousarray(array).tobytes())
    blobs = "\n".join(f'BLOB({s}, "{s}.bin", {"acc_t" if s == "BS" else "elem_t"})' for s in arrays)
    source = out / "ldprobe.c"
    source.write_text(
        PROGRAM.format(
            blobs=blobs,
            buffers="\n".join(f"static elem_t OUT_{a}[{elements}] row_align(1);" for a in names),
            definitions="\n".join(sources),
            warmups="\n".join(
                f"    {a}_fn((const elem_t *)IN, (const elem_t *)WT, (const acc_t *)BS, OUT_{a});" for a in names
            ),
            measures="".join(MEASURE.format(arm=a, elements=elements, ref=ref) for a in names),
            counter_names=", ".join(f'"{n}"' for n in bank),
            counter_codes=", ".join(str(codes[n]) for n in bank),
            **geom,
        ),
        encoding="utf-8",
    )
    common = harness / "riscv-tests" / "benchmarks" / "common"
    includes = [f"-I{harness / 'riscv-tests'}", f"-I{harness / 'riscv-tests' / 'env'}", f"-I{harness}", f"-I{common}"]
    objects, log = [], []
    for src in (source, common / "syscalls.c", common / "crt.S"):
        unit = out / f"{src.stem}.o"
        done = subprocess.run(
            [str(compiler), *CFLAGS, *includes, f"-Wa,-I{out}", "-c", str(src), "-o", str(unit)],
            capture_output=True, text=True, cwd=out,
        )  # fmt: skip
        log.append(done.stdout + done.stderr)
        if done.returncode:
            (out / "build.log").write_text("".join(log), encoding="utf-8")
            raise SystemExit(f"compile of {src.name} failed:\n{done.stderr[-4000:]}")
        objects.append(str(unit))
    elf = out / "ldprobe.elf"
    done = subprocess.run(
        [str(compiler), *CFLAGS, "-T", str(common / "test.ld"), *objects, "-o", str(elf)],
        capture_output=True, text=True, cwd=out,
    )  # fmt: skip
    (out / "build.log").write_text("".join(log) + done.stdout + done.stderr, encoding="utf-8")
    if done.returncode:
        raise SystemExit(f"link failed:\n{done.stderr[-4000:]}")
    return elf


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", required=True)
    parser.add_argument("--shape", default="3136x64x64", help="m x n x k")
    parser.add_argument("--arms", default="loop_ws," + ",".join(ARMS))
    parser.add_argument("--bank", default="occupancy", choices=sorted(BANKS))
    parser.add_argument("--tiles", default="", help="bm,bn,bk override (in array-edge tiles)")
    parser.add_argument("--tile-rows", type=int, default=0,
                        help="rows of the moving operand per tile (default: the array edge). A "
                             "convolution's output-row tile is its output-column count, which on the "
                             "deep stages is well under the edge.")  # fmt: skip
    parser.add_argument("--scale", default="0.0277")
    parser.add_argument("--no-relu", action="store_true")
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--emulator", type=Path)
    parser.add_argument("--max-cycles", default="400000000")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument(
        "--harness",
        type=Path,
        help="bare-metal harness tree (include/, riscv-tests/); default: the selected backend's own",
    )
    parser.add_argument("--cc", type=Path, help="RISC-V C compiler; default: the selected backend's own")
    args = parser.parse_args(argv)

    # The harness and compiler come from the selected target's backend unless named explicitly; the
    # geometry is then read from that harness's own headers either way.
    backend = None if args.harness and args.cc else base.get_backend(args.target)
    harness = args.harness or Path(backend.rocc_tests_dir())
    compiler = args.cc or Path(backend.gcc_path())
    facts = Facts(harness)
    codes = counter_codes(harness)
    bank = BANKS[args.bank]
    missing = [name for name in bank if name not in codes]
    if missing:
        raise SystemExit(f"this harness's counter header defines no {missing}")

    m, n, k = (int(v) for v in args.shape.split("x"))
    # The moving operand's tile is TR rows deep. It is DIM by default, but a convolution's output-row
    # tile is as many rows as the layer has output columns, which on the deep stages is far fewer --
    # and how many rows a compute streams is the thing a weight reload is amortised over, so it has to
    # be a knob rather than the array edge. Addresses still step by DIM so a shorter tile only leaves
    # rows unused; only the row COUNTS of the transfers and computes change.
    tile_rows = args.tile_rows or facts.dim
    if not 0 < tile_rows <= facts.dim:
        raise SystemExit(f"a tile of {tile_rows} rows does not fit the {facts.dim}-row array edge")
    if m % tile_rows or n % facts.dim or k % facts.dim:
        raise SystemExit(
            f"this probe measures whole tiles only; {args.shape} is not a multiple of "
            f"({tile_rows}, {facts.dim}, {facts.dim})"
        )
    mt, nt, kt = m // tile_rows, n // facts.dim, k // facts.dim
    bm, bn, bk = tuple(int(v) for v in args.tiles.split(",")) if args.tiles else block_shape(facts, mt, nt, kt)

    names = args.arms.split(",")
    sources = []
    for name in names:
        if name == "loop_ws":
            sources.append(_LOOPWS.format(name=name))
        elif name in ARMS:
            states, placement, store, stationary = ARMS[name]
            sources.append(
                arm_source(
                    name, min(states, facts.load_states), placement, store, stationary, facts, a_pitch=k, b_pitch=n
                )
            )
        else:
            raise SystemExit(f"unknown arm {name!r}; known: loop_ws, {sorted(ARMS)}")

    arrays = {"IN": _rng((m, k), args.seed), "WT": _rng((k, n), args.seed + 1),
              "BS": _rng((n,), args.seed + 2, np.int32)}  # fmt: skip
    geom = dict(
        m=m, n=n, k=k, mt=mt, nt=nt, kt=kt, bm=bm, bn=bn, bk=bk, tile_rows=tile_rows,
        scale=args.scale, relu="NO_ACTIVATION" if args.no_relu else "RELU",
        acc_bit=facts.acc_bit, acc_acc=facts.acc_bit | facts.accumulate_bit,
    )  # fmt: skip
    elf = build(names, sources, arrays, m * n, names[0], bank, codes, geom, args.out, harness, compiler)
    try:
        pins, note = {RTL_PIN: provenance.verify(RTL_PIN)}, None
    except Exception as why:  # noqa: BLE001 -- record WHY rather than dropping the revision
        pins, note = {}, f"UNKNOWN({type(why).__name__}: {why})"
    record: dict[str, Any] = {
        "schema": "merlin.gemmini-load-placement-probe.v1",
        "shape": args.shape,
        "target": args.target,
        "bank": args.bank,
        "counters": list(bank),
        "blocking": {"bm": bm, "bn": bn, "bk": bk, "mt": mt, "nt": nt, "kt": kt, "tile_rows": tile_rows},  # fmt: skip
        "load_states_in_this_build": facts.load_states,
        "mvin_opcodes": facts.mvin_functs,
        "arms": names,
        "cycle_scope": "one call of one contraction, CPU cycle counter across the call including the "
        "closing fence; comparable only against the other arms in this same binary",
        "elf": str(elf),
        "provenance": provenance.record(
            pins=pins,
            sources=[
                str(harness / "include" / "gemmini.h"),
                str(harness / "include" / "gemmini_params.h"),
                str(harness / "include" / "gemmini_counter.h"),
            ],  # fmt: skip
            artifacts={"emulator": str(args.emulator)} if args.emulator else None,
            extra={"pin_note": note} if note else None,
        ),
    }
    (args.out / "arms.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"# blocking bm={bm} bn={bn} bk={bk} over tiles ({mt},{nt},{kt}); load states {facts.load_states}")
    if args.build_only:
        return 0
    if args.emulator is None:
        raise SystemExit("--emulator is required unless --build-only")
    began = time.time()
    done = subprocess.run(
        [str(args.emulator), str(elf), f"+max-cycles={args.max_cycles}", f"+loadmem={elf}"],
        capture_output=True, text=True, cwd=args.out,
    )  # fmt: skip
    (args.out / "run.log").write_text(done.stdout + done.stderr, encoding="utf-8")
    measured: dict[str, dict[str, Any]] = {}
    for line in (done.stdout + done.stderr).splitlines():
        if not line.startswith("AB_"):
            continue
        print(line)
        kind, _, rest = line.partition(" ")
        if kind == "AB_CYCLES":
            arm, _, value = rest.partition("=")
            measured.setdefault(arm, {})["cycles"] = int(value)
        elif kind == "AB_COUNTER":
            arm, _, pair = rest.partition(" ")
            counter, _, value = pair.partition("=")
            measured.setdefault(arm, {}).setdefault("counters", {})[counter] = int(value)
        elif kind == "AB_CHECK":
            arm, _, fields = rest.partition(" ")
            measured.setdefault(arm, {})["bit_exact_vs_" + names[0]] = fields.split()[0] == "diffs=0"
    record["measured"] = measured
    record["engine_returncode"] = done.returncode
    (args.out / "arms.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"# engine wall {time.time() - began:.0f}s rc={done.returncode}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
