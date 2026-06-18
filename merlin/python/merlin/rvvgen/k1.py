"""SpacemiT K1 board adapter — real RVV silicon (VLEN=256, Bianbu Linux/glibc).

K1 is the fast real-hardware cycle target that pairs with spike (spike = correctness +
instruction evidence; K1 = cycle truth). It is Linux-hosted (no HTIF/bare-metal), so it needs a
Linux userspace runtime cross-compiled with the SpacemiT toolchain, scp'd to the board, run, and
parsed for the same OUT/METRIC/DONE markers via zephyr_model._parse_console.

This module is intentionally fail-closed: when the toolchain or the board is unavailable,
:func:`available` returns False and the runner records the K1 rung as ``not_run`` (NEVER a false
pass). The cross-compile + deploy path (S2.4-S2.6) reuses the SAME portable artifacts as the spike
build — the LLVM IR ``model.ll`` (``llvmlower.lower``) and the data-driven C runtime
(``llvmlower.c_runtime`` + ``merlin/runtime/c/merlin_model.c``); only the harness differs:

  * spike/Zephyr — bare-metal: ``-mcmodel=medany -ffreestanding -nostdlib``, HTIF console, mcycle.
  * K1 — glibc Linux userspace: ``-march=rv64gcv -mabi=lp64d -O2`` (NO medany/freestanding),
    a ``main()`` that prints OUT/METRIC/DONE to stdout, cycles via the ``rdcycle`` CSR.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

# Board access (memory: spacemit-k1-board-access). IP is a DHCP lease — override via env.
K1_SSH_KEY = os.environ.get("MERLIN_K1_SSH_KEY", "/scratch2/agustin/DIMA_SLICE")
K1_HOST = os.environ.get("MERLIN_K1_HOST", "")  # e.g. root@10.44.97.186; empty => unset/unreachable
# SpacemiT cross-toolchain. The repo keeps only setup_toolchain.sh as reference under
# build_tools/SpacemiT/ (the toolchain itself is huge); locate the real install via env, default
# to the known /scratch2 path. ``toolchain_cc()` tolerates either the bin/ layout or the
# extracted ``spacemit-toolchain-*`` subdir layout.
_REPO = Path(__file__).resolve().parents[4]
_DEFAULT_TOOLCHAIN = "/scratch2/agustin/merlin/build_tools/riscv-tools-spacemit"
K1_TOOLCHAIN = Path(os.environ.get("MERLIN_K1_TOOLCHAIN", _DEFAULT_TOOLCHAIN))

# K1 X60 target: rv64gcv glibc Linux. NOT medany/freestanding — it is hosted userspace.
K1_MARCH = "rv64gcv"
K1_MABI = "lp64d"
VLEN = 256  # K1 X60 vector length, bits; the runtime reads vlenb at run time and records it.
# This Bianbu kernel does NOT delegate the userspace `cycle` CSR — `rdcycle` traps as an illegal
# instruction. The `time` CSR IS delegated, so the harness times with `rdtime` (a fixed-frequency
# platform counter, NOT core cycles) + wall-clock, and derives an estimated core-cycle count from
# the timebase->CPU frequency ratio. So K1 numbers are REAL-SILICON wall/timebase measurements,
# reported as cycle_accurate=False (spike/FireSim remain the cycle-accurate authorities).
K1_TIMEBASE_HZ = 24_000_000   # /proc/device-tree/cpus/timebase-frequency (rdtime tick rate)
K1_CPU_HZ = 1_600_000_000     # X60 scaling_cur_freq (for the rdtime-ticks -> core-cycle estimate)


def _toolchain_root() -> Path | None:
    """Resolve the directory that actually contains ``bin/clang``.

    Accepts either ``$MERLIN_K1_TOOLCHAIN`` pointing directly at a toolchain prefix
    (``<prefix>/bin/clang``) OR at a parent holding the extracted
    ``spacemit-toolchain-linux-glibc-x86_64-*`` subdir.
    """
    if (K1_TOOLCHAIN / "bin" / "clang").is_file():
        return K1_TOOLCHAIN
    if K1_TOOLCHAIN.is_dir():
        for sub in sorted(K1_TOOLCHAIN.glob("spacemit-toolchain-*")):
            if (sub / "bin" / "clang").is_file():
                return sub
    return None


def toolchain_cc() -> Path | None:
    """Path to the SpacemiT linux-gnu clang/gcc, or None if the toolchain isn't installed."""
    root = _toolchain_root()
    if root is None:
        return None
    for rel in ("bin/clang", "bin/riscv64-unknown-linux-gnu-gcc"):
        c = root / rel
        if c.is_file():
            return c
    return None


def available() -> bool:
    """True only if BOTH the cross-toolchain is installed AND the board is reachable over SSH.

    Fail-closed: any uncertainty -> False -> runner records ``not_run`` (never a false pass).
    """
    if toolchain_cc() is None or not K1_HOST or not Path(K1_SSH_KEY).is_file():
        return False
    if shutil.which("ssh") is None:
        return False
    try:
        r = subprocess.run(
            ["ssh", "-i", K1_SSH_KEY, "-o", "BatchMode=yes",
             "-o", "ConnectTimeout=5", "-o", "StrictHostKeyChecking=no", K1_HOST, "true"],
            capture_output=True, timeout=15)
        return r.returncode == 0
    except Exception:  # noqa: BLE001
        return False


# ---- generated Linux harness --------------------------------------------------------

def main_linux_c(dump_cap: int = 4096) -> str:
    """Generate the K1 Linux ``main()``: build the memref descriptors from the generated arg
    table, call ``merlin_run`` once, time it with the ``rdcycle`` CSR, and print the SAME
    OUT/ARGMAX/SUM/METRIC/DONE markers the spike harness uses so ``zephyr_model._parse_console``
    consumes the console unchanged. Glibc-hosted: stdio, malloc, no HTIF/medany."""
    return f"""/* Generated by merlin.rvvgen.k1 — K1 (SpacemiT) Linux RVV runtime harness. Do not edit. */
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "merlin_model.h"
#include "model_gen.h"
#include "model_io.h"

/* weights.bin linked in as a binary blob (ld -r -b binary): glibc Linux is small-code-model
 * and fully relocatable, so we address the blob through its symbol (no medany window). */
extern const unsigned char _binary_weights_bin_start[];
#define MERLIN_WEIGHTS_BASE ((const void *)_binary_weights_bin_start)

#define MERLIN_DUMP_CAP {dump_cap}
/* Timebase->CPU frequency ratio for the estimated core-cycle count (see module doc): this
 * kernel traps userspace `rdcycle`, so we time with the delegated `rdtime` counter. */
#define MERLIN_TIMEBASE_HZ {K1_TIMEBASE_HZ}ULL
#define MERLIN_CPU_HZ {K1_CPU_HZ}ULL

static float OUT[MERLIN_OUT_ELEMS];
static merlin_descriptor_t DESCS[MERLIN_N_ARGS];

static inline uint64_t rd_time(void) {{
  uint64_t t;
  __asm__ volatile("rdtime %0" : "=r"(t));
  return t;
}}

static inline uint64_t rd_vlenb(void) {{
  uint64_t v;
  __asm__ volatile("csrr %0, vlenb" : "=r"(v));
  return v;
}}

static uint64_t wall_ns(void) {{
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}}

int main(void) {{
  printf("=== merlin_k1 vlenb=%llu ===\\n", (unsigned long long)rd_vlenb());
  uint64_t w0 = wall_ns();
  uint64_t t0 = rd_time();
  merlin_run(MERLIN_ARGS, MERLIN_N_ARGS, MERLIN_WEIGHTS_BASE,
             MERLIN_INPUT_PTR, OUT, DESCS);
  uint64_t t1 = rd_time();
  uint64_t w1 = wall_ns();

  int k = MERLIN_OUT_ELEMS < MERLIN_DUMP_CAP ? MERLIN_OUT_ELEMS : MERLIN_DUMP_CAP;
  printf("OUT %d", k);
  for (int i = 0; i < k; i++) {{
    uint32_t bits;
    memcpy(&bits, &OUT[i], 4);
    printf(" %u", (unsigned)bits);
  }}
  printf("\\n");

  if (MERLIN_OUT_ELEMS > MERLIN_DUMP_CAP) {{
    int rows = MERLIN_OUT_ELEMS / MERLIN_OUT_LASTDIM;
    printf("ARGMAX %d", rows);
    for (int r = 0; r < rows; r++) {{
      const float *row = &OUT[(long)r * MERLIN_OUT_LASTDIM];
      int best = 0; float bv = row[0];
      for (int j = 1; j < MERLIN_OUT_LASTDIM; j++)
        if (row[j] > bv) {{ bv = row[j]; best = j; }}
      printf(" %d", best);
    }}
    printf("\\n");
    float s = 0.0f;
    for (int i = 0; i < MERLIN_OUT_ELEMS; i++) s += OUT[i];
    uint32_t sb; memcpy(&sb, &s, 4);
    printf("SUM %u\\n", (unsigned)sb);
  }}
  uint64_t ticks = t1 - t0;
  /* estimated core cycles = timebase ticks * (cpu_hz / timebase_hz). The METRIC cycles line is
   * what _parse_console reads; the raw timebase ticks + wall ns are emitted alongside as the
   * ground truth (the cycles value is an estimate — see K1_CPU_HZ/K1_TIMEBASE_HZ). */
  uint64_t est_cycles = ticks * (MERLIN_CPU_HZ / MERLIN_TIMEBASE_HZ);
  printf("METRIC cycles %llu\\n", (unsigned long long)est_cycles);
  printf("METRIC time_ticks %llu\\n", (unsigned long long)ticks);
  printf("METRIC wall_ns %llu\\n", (unsigned long long)(w1 - w0));
  printf("DONE\\n");
  fflush(stdout);
  return 0;
}}
"""


# ---- build / deploy / run -----------------------------------------------------------

def _run(cmd: list, **kw) -> subprocess.CompletedProcess:
    proc = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, **kw)
    if proc.returncode != 0:
        raise K1Error(
            f"command failed: {' '.join(map(str, cmd))}\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")
    return proc


class K1Error(RuntimeError):
    pass


def build_k1_binary(model_dir: str | Path, work: str | Path, pkg,
                    inputs_npz: str | Path | None = None) -> Path:
    """Cross-compile a self-contained K1 Linux RVV binary from the workload + RVV package.

    Reuses the EXACT spike/Zephyr lowering (``zephyr_model._prepare_model_mlir`` ->
    ``lower_model_file(vectorize=True, transform_schedule=pkg.schedule_text)``) so the K1 binary
    executes the same emitted RVV as the spike correctness run, and the same data-driven C runtime
    artifacts (``c_runtime.generate`` -> model_gen.h / model_call.c / weights.bin / model_io.h).
    Only the harness (``main_linux.c``) and the compile flags differ (glibc Linux, not medany).
    Returns the path to the built ELF.
    """
    model_dir, work = Path(model_dir).resolve(), Path(work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    inputs_npz = inputs_npz or (model_dir / "inputs.npz")
    cc = toolchain_cc()
    if cc is None:
        raise K1Error("SpacemiT toolchain not found (set MERLIN_K1_TOOLCHAIN)")
    root = _toolchain_root()
    # binutils ld for the binary-blob object (use the toolchain's GNU ld, not lld).
    ld = root / "bin" / "riscv64-unknown-linux-gnu-ld"
    if not ld.is_file():
        ld = Path(shutil.which("ld") or "ld")

    from ..llvmlower import c_runtime, toolchain
    from ..llvmlower.lower import lower_model_file
    from ..runtime.backends import zephyr_model as zm

    # 1. model.mlir -> normalize (same prep passes as the package/spike build) -> model.ll.
    prepared = zm._prepare_model_mlir(model_dir / "model.mlir", work, int8_compute=pkg.is_int8)
    res = lower_model_file(prepared, work / "lower", targets=(), textual=True,
                           vectorize=True, transform_schedule=pkg.schedule_text)

    # 2. compile model.ll -> K1 Linux object. The IR is emitted by the repo's clang-23 toolchain
    #    and carries LLVM-23 attribute syntax (e.g. `captures(none)`) the SpacemiT clang-19 can't
    #    parse, so the .ll -> object step uses THE SAME clang-23 (re-targeted to linux-gnu); the
    #    SpacemiT clang is used for the C sources + final glibc link below. NO medany/freestanding
    #    — glibc hosted, small code model, fully relocatable.
    clang23 = toolchain.clang()
    model_o = work / "model.o"
    _run([clang23, "--target=riscv64-unknown-linux-gnu", f"-march={K1_MARCH}", f"-mabi={K1_MABI}",
          "-O2", "-Wno-override-module", "-c", res.ll_path, "-o", model_o])

    # 3. data-driven runtime artifacts (arg table, ciface, weights.bin, embedded io).
    cgen = work / "cgen"
    c_runtime.generate(model_dir, cgen, inputs_npz)

    # 4. weights.bin -> binary blob object.
    weights_o = work / "weights_blob.o"
    _run([ld, "-r", "-b", "binary", "-o", weights_o, "weights.bin"], cwd=cgen)

    # 5. the Linux harness.
    main_c = work / "main_linux.c"
    main_c.write_text(main_linux_c())

    # 6. link the final binary. Reuse the repo's portable C runtime + generated ciface. Prefer a
    #    static binary (no glibc-version coupling to the board); fall back to dynamic, which is
    #    fine since it runs on the board's own glibc.
    rt = _REPO / "merlin/runtime/c"
    abi = _REPO / "merlin/runtime/abi"
    binary = work / "merlin_k1"
    srcs = [main_c, cgen / "model_call.c", rt / "merlin_model.c", abi / "mlir_runtime.c"]
    base = [cc, f"--target=riscv64-unknown-linux-gnu", f"-march={K1_MARCH}", f"-mabi={K1_MABI}",
            "-O2", f"-I{rt}", f"-I{cgen}", *srcs, model_o, weights_o, "-lm", "-o", binary]
    try:
        _run([*base, "-static"])
    except K1Error:
        _run(base)
    if not binary.is_file():
        raise K1Error(f"K1 cross-compile produced no binary at {binary}")
    return binary


def _ssh(*args: str, timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["ssh", "-i", K1_SSH_KEY, "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
         "-o", "StrictHostKeyChecking=no", K1_HOST, *args],
        capture_output=True, text=True, timeout=timeout)


def board_vlenb() -> int | None:
    """Read VLENB (bytes) from the board, or None if unreachable. VLEN_bits = vlenb*8."""
    try:
        r = _ssh("cat /proc/cpuinfo | grep -m1 -i vlenb || true", timeout=20)
        for line in r.stdout.splitlines():
            if "vlenb" in line.lower():
                return int(line.split(":")[-1].strip())
    except Exception:  # noqa: BLE001
        pass
    return None


def run_on_k1(model_dir: str | Path, work: str | Path, pkg, *, timeout: int = 600) -> dict[str, Any]:
    """Cross-compile the workload for K1, deploy over scp, run, and parse OUT/METRIC/DONE.

    Returns the ``zephyr_model._parse_console`` result dict augmented with ``vlen`` (the bit-width
    the board reports — vlenb*8, falling back to the :data:`VLEN` constant). The K1 ``cycles`` from
    ``rdcycle`` are real silicon cycles (``cycle_accurate`` in the runner's measurement record).
    """
    from ..runtime.backends import zephyr_model as zm

    if not K1_HOST:
        raise K1Error("MERLIN_K1_HOST unset — board unreachable")
    binary = build_k1_binary(model_dir, work, pkg)

    remote = f"/tmp/{Path(model_dir).name}_{pkg.run_id}_merlin_k1"
    _run(["scp", "-i", K1_SSH_KEY, "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no",
          str(binary), f"{K1_HOST}:{remote}"])
    try:
        _ssh(f"chmod +x {remote}", timeout=30)
        proc = _ssh(remote, timeout=timeout)
        console = proc.stdout + proc.stderr
        res = zm._parse_console(console, proc.returncode)
    finally:
        try:
            _ssh(f"rm -f {remote}", timeout=30)
        except Exception:  # noqa: BLE001
            pass

    # Prefer the vlenb the harness printed (=== merlin_k1 vlenb=N ===); else probe; else constant.
    vlen_bits = VLEN
    for line in res["console"].splitlines():
        if "vlenb=" in line:
            try:
                vlen_bits = int(line.split("vlenb=")[1].split()[0]) * 8
            except (IndexError, ValueError):
                pass
            break
    res["vlen"] = vlen_bits
    return res
