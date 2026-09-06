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

import fcntl
import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
import time
from contextvars import ContextVar
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from merlin.common.paths import env, repo_root, runtime_dir
from typing import Any

# Board access — set both via env (no personal defaults committed). The board IP is a DHCP lease.
# Read through paths.env (os.environ -> .env -> default) so a repo-local .env configures the board +
# toolchain without exporting into the shell (the same .env-everywhere contract as spike/zephyr).
K1_SSH_KEY = env("MERLIN_K1_SSH_KEY", "")  # path to the SSH private key; empty => unset
K1_HOST = env("MERLIN_K1_HOST", "")  # e.g. root@<board-ip>; empty => unset/unreachable
# Optional non-standard SSH port. The board sits on the Berkeley-IoT WiFi (/22); the campus path from
# the wired net filters inbound :22 to that segment (ICMP + high ports pass, port 22 is dropped), so
# the board also runs sshd on a high port and this points ssh/scp at it. Empty => default port 22.
K1_SSH_PORT = env("MERLIN_K1_SSH_PORT", "")
_SSH_PORT_OPTS = ["-p", K1_SSH_PORT] if K1_SSH_PORT else []   # ssh uses -p
_SCP_PORT_OPTS = ["-P", K1_SSH_PORT] if K1_SSH_PORT else []   # scp uses -P (capital)
# SpacemiT cross-toolchain. The repo keeps only setup_toolchain.sh as reference under
# build_tools/SpacemiT/ (the toolchain itself is huge); locate the real install via env, default
# to the known /scratch2 path. ``toolchain_cc()` tolerates either the bin/ layout or the
# extracted ``spacemit-toolchain-*`` subdir layout.
_REPO = repo_root()
# Locate the (huge) SpacemiT cross-toolchain via env; no personal path committed as a default.
K1_TOOLCHAIN = Path(env("MERLIN_K1_TOOLCHAIN", str(_REPO / "build_tools" / "SpacemiT" / "riscv-tools-spacemit")))

# K1 X60 target: glibc Linux userspace (NOT medany/freestanding). The board's ISA carries the
# half-precision extensions zfh/zfhmin (scalar) + zvfh/zvfhmin (VECTOR fp16) — see /proc/cpuinfo — so
# the march MUST include them or clang targets generic rv64gcv and mis/soft-compiles fp16 (and the
# bf16<->f32 conversions), which the K1 has in hardware. This matches the march llama.cpp's own
# SpacemiT toolchain file selects for the K1 (rv64gcv_zfh_zvfh...). Overridable via MERLIN_K1_MARCH.
K1_MARCH = env("MERLIN_K1_MARCH", "rv64gcv_zfh_zvfh")
K1_MABI = "lp64d"
#: Declared vector length of THIS target, in bits. Overridable so the value is a per-target
#: DECLARATION rather than a compiled-in assumption -- pinning a VLEN the hardware does not have is a
#: miscompile, not a missed optimization (see codegen_march / verify_vlen).
#:
#: NOTE ON GENERALITY: pinning a known VLEN is a STOPGAP that trades portability for lanes. The
#: target-agnostic fix is VL-agnostic codegen -- `MicrokernelSpec.vl_strategy = VL_DYNAMIC`, a
#: vsetvli loop that sizes to whatever VLEN the hardware reports at run time, which is exactly what
#: the expert kernels do (XNNPACK calls __riscv_vsetvl_e32m4). Until that axis is emitted, this pin
#: recovers the same lanes for targets whose VLEN we can state and verify.
VLEN = int(env("MERLIN_K1_VLEN", "256"))  # K1 X60; the runtime reads vlenb and records it.


class VlenMismatch(RuntimeError):
    """The board reported a vector length different from the one codegen pinned."""


def verify_vlen(reported_vlenb: int, vlen: int | None = None) -> None:
    """Fail closed when the board's actual ``vlenb`` contradicts the VLEN we compiled against.

    ``vlenb`` is VLEN in BYTES and the harness banner already prints it, so the true value is
    observable on every run -- there is no reason to keep trusting a constant. Pinning a VLEN larger
    than the hardware's is not a lost optimization but a MISCOMPILE (the backend sizes register
    groups for lanes that do not exist), so a mismatch must stop the run rather than be measured."""
    want = int(vlen or VLEN)
    got = int(reported_vlenb) * 8
    if got != want:
        raise VlenMismatch(
            f"board reports VLEN={got} bits (vlenb={reported_vlenb}) but codegen pinned {want}. "
            f"Set MERLIN_K1_VLEN={got} for this board, or drop the pin. Pinning a VLEN the hardware "
            f"lacks miscompiles; pinning a smaller one silently idles the datapath.")


def codegen_march(march: str | None = None, vlen: int | None = None) -> str:
    """``march`` with the board's REAL vector length pinned (``_zvl<VLEN>b``).

    Why this is not cosmetic. ``-march=rv64gcv`` only promises the RVV *minimum* VLEN of 128 bits
    (``zvl128b``), so when our codegen emits a FIXED-width vector the LLVM backend must size the
    register group for that worst case: a ``vector<16xf32>`` (512 bits) becomes ``e32,m4`` and a
    ``vector<32xf32>`` becomes ``e32,m8`` — exactly DOUBLE the LMUL the K1 needs. Two costs follow,
    and together they were the whole "vector width vs IPC" tradeoff (measured, f32 GEMM 128^3):

      * REGISTER PRESSURE doubles. An ``m8`` value occupies 8 of the 32 architectural vector
        registers, so an MR=4/NR=32 register block wants 32 registers for accumulators alone and the
        allocator spills inside the K loop (objdump: 25 inner-loop instructions with 4 ``vl8r.v``/
        ``vs8r.v``, vs 12 and none once VLEN is pinned). MR=8/NR=32 went 92 -> 46.
      * HALF THE DATAPATH IS IDLE. ``vsetivli zero, 16, e32, m4`` on a VLEN=256 core sets ``vl=16``
        against a ``VLMAX`` of 32 — the instruction reserves a 4-register group and uses half of it.

    An expert kernel never hits this because it sizes to the vector length it *queries* at run time
    (XNNPACK's ``__riscv_vsetvl_e32m4``), which is why its advantage looked like "lanes per issue".
    Pinning the known target VLEN recovers the same lanes-per-issue by CODE GENERATION.

    Kept separate from :data:`K1_MARCH` (which the cross-framework baseline arms compile against)
    so pinning VLEN for our own codegen never silently re-flags a baseline mid-campaign.
    """
    m = march or K1_MARCH
    v = int(vlen or VLEN)
    return m if f"zvl{v}b" in m else f"{m}_zvl{v}b"


# This Bianbu kernel does NOT delegate the userspace `cycle` CSR — `rdcycle` traps as an illegal
# instruction. The `time` CSR IS delegated, so the harness times with `rdtime` (a fixed-frequency
# platform counter, NOT core cycles) + wall-clock, and derives an estimated core-cycle count from
# the timebase->CPU frequency ratio. So K1 numbers are REAL-SILICON wall/timebase measurements,
# reported as cycle_accurate=False (spike/FireSim remain the cycle-accurate authorities).
K1_TIMEBASE_HZ = 24_000_000   # /proc/device-tree/cpus/timebase-frequency (rdtime tick rate)
K1_CPU_HZ = 1_600_000_000     # X60 scaling_cur_freq (for the rdtime-ticks -> core-cycle estimate)
# The lowered model alloca's large intermediate buffers; the default 8MB Linux stack overflows
# (SIGSEGV store-fault) for VLAs. We run merlin_run on a pthread with an explicitly-mmap'd large
# stack (one contiguous region — more robust than main-thread RLIMIT growth, which collides with
# lower mmaps for the bigger models). Sized to leave headroom on the ~3.5G board.
K1_STACK_BYTES = 2 * 1024 * 1024 * 1024

# Cross-built static OpenMP runtime (riscv64, rv64gcv) — provides the __kmpc_* symbols the
# multicore (parallel) lowering emits. Built once into build_tools/k1_openmp (see the
# OpenMP path in llvmlower.pipeline._parallel_pipeline). Linking it fans the model's parallel
# loops across the board's 8 cores (set OMP_NUM_THREADS at run time). Optional: only the
# `parallel=True` build path references it.
K1_OPENMP_DIR = _REPO / "build_tools" / "k1_openmp"
K1_OMP_THREADS = int(env("MERLIN_K1_OMP_THREADS", "8"))  # board has 8 cores (2x4)
# Big mmap'd weight blobs MUST live on real storage, NOT /tmp: the board's /tmp is tmpfs
# (RAM-backed, only 1.9G) so a multi-GB weights file there both fails to fit and consumes the
# RAM we are trying to save. The rootfs (/dev/mmcblk2p6, ~12G free) is real flash — mmap from
# there demand-pages off disk. Binary (small) stays in /tmp; weights go here.
K1_REMOTE_DIR = env("MERLIN_K1_REMOTE_DIR", "/root/merlin_k1")


def _board_lock_path() -> Path:
    """A host-wide lockfile identifying THIS board. Keyed by K1_HOST so two different boards do not
    block each other, but every process on this machine (concurrent beams, other Claude sessions)
    contends on the same file when hitting the same board."""
    tag = hashlib.sha1((K1_HOST or "unset").encode()).hexdigest()[:12]
    return Path(tempfile.gettempdir()) / f"merlin_k1_board_{tag}.lock"


@contextmanager
def board_lock(timeout: int = 1800):
    """Serialize physical-board access across ALL processes on this host (the single K1 has one set
    of /tmp deploy paths + 8 cores; concurrent forks would corrupt each other's scp'd binaries and
    contend for cores, poisoning the cycle measurement). A file ``flock`` is the cross-process lock
    that a ThreadPool max_workers=1 (single-beam only) and a chia ``resources={"k1":1}`` gate (single
    Ray cluster only) both miss. ``MERLIN_K1_NO_BOARD_LOCK=1`` disables it (single-user runs)."""
    if os.environ.get("MERLIN_K1_NO_BOARD_LOCK") == "1":
        yield
        return
    lockf = _board_lock_path()
    fh = open(lockf, "w")
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)  # blocks until the board is free
        yield
    finally:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        finally:
            fh.close()


def _toolchain_root() -> Path | None:
    """Resolve the directory that actually contains ``bin/clang``.

    Accepts either ``$MERLIN_K1_TOOLCHAIN`` pointing directly at a toolchain prefix
    (``<prefix>/bin/clang``) OR at a parent holding the extracted
    ``spacemit-toolchain-linux-glibc-x86_64-*`` subdir.
    """
    if (K1_TOOLCHAIN / "bin" / "clang").is_file():
        return K1_TOOLCHAIN
    if K1_TOOLCHAIN.is_dir():
        candidates = [*K1_TOOLCHAIN.glob("spacemit-toolchain-*"),
                      *K1_TOOLCHAIN.glob("*/spacemit-toolchain-*")]
        for sub in sorted(candidates):
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


def available(*, deadline_ns: int | None = None) -> bool:
    """True only if BOTH the cross-toolchain is installed AND the board is reachable over SSH.

    Fail-closed: any uncertainty -> False -> runner records ``not_run`` (never a false pass).
    """
    if toolchain_cc() is None or not K1_HOST or not Path(K1_SSH_KEY).is_file():
        return False
    if shutil.which("ssh") is None:
        return False
    try:
        timeout = 15
        if deadline_ns is not None:
            remaining_ns = int(deadline_ns) - time.monotonic_ns()
            if remaining_ns <= 0:
                return False
            timeout = min(timeout, max(1, math.ceil(remaining_ns / 1_000_000_000)))
        r = subprocess.run(
            ["ssh", "-i", K1_SSH_KEY, *_SSH_PORT_OPTS, "-o", "BatchMode=yes",
             "-o", "ConnectTimeout=5", "-o", "StrictHostKeyChecking=no", K1_HOST, "true"],
            capture_output=True, timeout=timeout)
        return r.returncode == 0
    except Exception:  # noqa: BLE001
        return False


# ---- generated Linux harness --------------------------------------------------------

def main_linux_c(dump_cap: int | None = 4096, mmap_weights: bool = False,
                 dispatch_timing: bool = False, op_profile: bool = False) -> str:
    """Generate the K1 Linux ``main()``: build the memref descriptors from the generated arg
    table, call ``merlin_run`` once, time it with the ``rdcycle`` CSR, and print the SAME
    OUT/ARGMAX/SUM/METRIC/DONE markers the spike harness uses so ``zephyr_model._parse_console``
    consumes the console unchanged. Glibc-hosted: stdio, malloc, no HTIF/medany.

    ``mmap_weights=True``: instead of embedding the weight blob in the binary (``ld -r -b
    binary`` -> ``_binary_weights_bin_start``), ``mmap`` the blob read-only from a file path
    (``MERLIN_WEIGHTS`` env or argv[1]). The OS then demand-pages the weights, so resident RAM
    is the working set, not the whole blob — required for the big VLAs (groot 1.8G, pi05 ~5G)
    whose embedded binary won't fit /tmp and whose all-resident weights exceed the 3.4G board.

    ``dump_cap`` bounds how many output elements the console prints. The host gate compares the
    printed PREFIX against the leading elements of the reference, so this number is the ceiling on
    how much of the answer any accuracy verdict can actually cover: at the historical 4096 a
    256,000-element LM head was graded on 1.6% of its output, and the resulting cos read exactly
    like a whole-tensor score. ``dump_cap=None`` removes the ceiling by deriving it from the
    model's OWN generated ``MERLIN_OUT_ELEMS`` rather than a host-side literal, so the console
    carries the whole tensor and ``_gate`` reports ``comparison_complete``. The default stays 4096
    so an unchanged caller emits byte-identical C."""
    weights_decl = ("/* weights mmap'd from a file at run time (see main). */\n"
                    "static const void *MERLIN_WEIGHTS_PTR;\n"
                    if mmap_weights else
                    "/* weights.bin linked in as a binary blob (ld -r -b binary): glibc Linux is\n"
                    " * small-code-model and fully relocatable, so address the blob via its symbol. */\n"
                    "extern const unsigned char _binary_weights_bin_start[];\n"
                    "static const void *MERLIN_WEIGHTS_PTR;\n")
    mmap_includes = ("#include <fcntl.h>\n#include <sys/mman.h>\n#include <sys/stat.h>\n"
                     "#include <unistd.h>\n#include <stdlib.h>\n" if mmap_weights else "")
    if mmap_weights:
        weights_init = """  /* mmap the weight blob read-only: the loader demand-pages it, so resident RAM is the
   * working set, not the whole blob (the big-VLA fit lever). */
  const char *wpath = getenv("MERLIN_WEIGHTS");
  if (!wpath && argc > 1) wpath = argv[1];
  if (!wpath) {{ fprintf(stderr, "FAIL no MERLIN_WEIGHTS path\\n"); return 2; }}
  int wfd = open(wpath, O_RDONLY);
  if (wfd < 0) {{ fprintf(stderr, "FAIL open weights %s\\n", wpath); return 2; }}
  struct stat wst; fstat(wfd, &wst);
  void *wmap = mmap(NULL, (size_t)wst.st_size, PROT_READ, MAP_PRIVATE, wfd, 0);
  if (wmap == MAP_FAILED) {{ fprintf(stderr, "FAIL mmap weights\\n"); return 2; }}
  MERLIN_WEIGHTS_PTR = wmap;"""
    else:
        weights_init = "  MERLIN_WEIGHTS_PTR = (const void *)_binary_weights_bin_start;"
    # Per-dispatch matmul-bucket timing (default-OFF). Only emitted when the routed-matmul shim
    # was built with -DMERLIN_DISPATCH_TIMING; then the harness reads the accumulated GEMM-ukernel
    # ticks/calls and prints them as extra METRIC lines. With dispatch_timing=False the harness is
    # byte-identical to the un-instrumented path.
    dtiming_decl = (
        "extern unsigned long long merlin_matmul_ticks(void);\n"
        "extern unsigned long long merlin_matmul_calls(void);\n"
        if dispatch_timing else "")
    dtiming_print = (
        '  printf("METRIC matmul_ticks %llu\\n", (unsigned long long)merlin_matmul_ticks());\n'
        '  printf("METRIC matmul_calls %llu\\n", (unsigned long long)merlin_matmul_calls());\n'
        if dispatch_timing else "")
    # Per-op whole-model profile (default-OFF). Emitted only when the IR was instrumented with
    # `@merlin_prof_mark` (llvmlower.op_profile) and the shim object is linked; with
    # op_profile=False the harness is byte-identical to the un-instrumented path.
    oprof_decl = "extern void merlin_prof_dump(void);\n" if op_profile else ""
    oprof_print = "  merlin_prof_dump();\n" if op_profile else ""
    # The cap is a TOKEN, not a number: `None` renders the model's own generated element count, so
    # "print everything" is derived from the target's header instead of a host-side guess that a
    # bigger model would silently outgrow. A non-positive cap would print an empty OUT line that
    # parses fine and grades as zero coverage -- refuse it rather than emit it.
    if dump_cap is None:
        dump_cap_token = "MERLIN_OUT_ELEMS"
    else:
        if int(dump_cap) < 1:
            raise ValueError("dump_cap must be positive, or None for the model's full output")
        dump_cap_token = str(int(dump_cap))
    return f"""/* Generated by merlin.mining.k1 — K1 (SpacemiT) Linux RVV runtime harness. Do not edit. */
#define _GNU_SOURCE
#include <stdint.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/resource.h>
#include <pthread.h>
{mmap_includes}
#include "merlin_model.h"
#include "model_gen.h"
#include "model_io.h"

{weights_decl}
{dtiming_decl}{oprof_decl}#define MERLIN_DUMP_CAP {dump_cap_token}
/* Timebase->CPU frequency ratio for the estimated core-cycle count (see module doc): this
 * kernel traps userspace `rdcycle`, so we time with the delegated `rdtime` counter. */
#define MERLIN_TIMEBASE_HZ {K1_TIMEBASE_HZ}ULL
#define MERLIN_CPU_HZ {K1_CPU_HZ}ULL

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

static int merlin_one_step(long step, int validate_session) {{
  merlin_prepare_step(step);
  merlin_run_multi(MERLIN_ARGS, MERLIN_N_ARGS, MERLIN_WEIGHTS_PTR,
                   MERLIN_INPUT_PTR, MERLIN_OUTPUT_PTR, DESCS);
#if MERLIN_N_STATE_PAIRS > 0
  if (merlin_commit_state(MERLIN_ARGS, MERLIN_N_ARGS, MERLIN_INPUT_PTR,
                          MERLIN_OUTPUT_PTR, MERLIN_N_STATE_PAIRS,
                          MERLIN_STATE_INPUT_ARGS, MERLIN_STATE_OUTPUT_INDICES) != 0)
    return -1;
#endif
  if (validate_session) merlin_validate_step(step);
  return 0;
}}

static int merlin_one_session(int validate_session) {{
  merlin_reset_session();
  for (long step = 0; step < MERLIN_SESSION_STEPS; step++)
    if (merlin_one_step(step, validate_session) != 0) return -1;
  return 0;
}}

static void *worker(void *arg) {{
  (void)arg;
  printf("=== merlin_k1 vlenb=%llu ===\\n", (unsigned long long)rd_vlenb());
  cpu_set_t allowed;
  CPU_ZERO(&allowed);
  if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) {{
    fprintf(stderr, "FAIL sched_getaffinity\\n"); return NULL;
  }}
  printf("METRIC affinity_cpus %d\\n", CPU_COUNT(&allowed));

  /* Diagnostic sustained mode uses MERLIN_WARMUP + MERLIN_ITERS for individual transitions.
   * Paper mode uses MERLIN_SESSION_WARMUPS + MERLIN_SESSION_REPEATS: every timing sample then
   * contains every semantic observation from reset state. Observation count and statistical
   * repeat count are deliberately distinct.
   *
   * Transition passes use the SAME buffers -- which is the point: a per-iteration cost that creeps
   * (arena growth,
   * allocator churn) is invisible in a single run, and a single run's cold caches make the one
   * number you quote either optimistic or pessimistic depending on which you pick. Defaults
   * (1 timed, 0 warmup) reproduce the historical single-shot console byte-for-byte. Each timed
   * pass emits `METRIC iter_cycles <i> <cycles>`, the same protocol the Zephyr harness uses, so
   * _parse_console reads both without a special case. */
  long n_iters = 1, n_warmup = 0, n_session_repeats = 0, n_session_warmups = 0;
  int validate_session = 0;
  {{
    const char *e = getenv("MERLIN_ITERS");
    if (e && *e) {{ n_iters = strtol(e, NULL, 10); if (n_iters < 1) n_iters = 1; }}
    e = getenv("MERLIN_WARMUP");
    if (e && *e) {{ n_warmup = strtol(e, NULL, 10); if (n_warmup < 0) n_warmup = 0; }}
    e = getenv("MERLIN_SESSION_REPEATS");
    if (e && *e) {{
      n_session_repeats = strtol(e, NULL, 10);
      if (n_session_repeats < 1) n_session_repeats = 1;
    }}
    e = getenv("MERLIN_SESSION_WARMUPS");
    if (e && *e) {{
      n_session_warmups = strtol(e, NULL, 10);
      if (n_session_warmups < 0) n_session_warmups = 0;
    }}
    e = getenv("MERLIN_VALIDATE_SESSION");
    if (e && *e && strcmp(e, "0") != 0) validate_session = 1;
  }}
  uint64_t w0, w1, t0, t1;
  if (n_session_repeats > 0) {{
    for (long repeat = 0; repeat < n_session_warmups; repeat++)
      if (merlin_one_session(0) != 0) return NULL;
    w0 = wall_ns(); t0 = rd_time();
    for (long repeat = 0; repeat < n_session_repeats; repeat++) {{
      uint64_t iw0 = wall_ns(), it0 = rd_time();
      if (merlin_one_session(validate_session && repeat == 0) != 0) return NULL;
      uint64_t it1 = rd_time(), iw1 = wall_ns();
      printf("METRIC iter_cycles %ld %llu\\n", repeat,
             (unsigned long long)((it1 - it0) * (MERLIN_CPU_HZ / MERLIN_TIMEBASE_HZ)));
      printf("METRIC iter_wall_ns %ld %llu\\n", repeat, (unsigned long long)(iw1 - iw0));
    }}
    t1 = rd_time(); w1 = wall_ns();
    t1 = t0 + (t1 - t0) / (uint64_t)n_session_repeats;
    w1 = w0 + (w1 - w0) / (uint64_t)n_session_repeats;
    goto session_complete;
  }}
  merlin_reset_session();
  for (long i = 0; i < n_warmup; i++) {{
    merlin_prepare_step(i);
    merlin_run_multi(MERLIN_ARGS, MERLIN_N_ARGS, MERLIN_WEIGHTS_PTR,
                     MERLIN_INPUT_PTR, MERLIN_OUTPUT_PTR, DESCS);
#if MERLIN_N_STATE_PAIRS > 0
    if (merlin_commit_state(MERLIN_ARGS, MERLIN_N_ARGS, MERLIN_INPUT_PTR,
                            MERLIN_OUTPUT_PTR, MERLIN_N_STATE_PAIRS,
                            MERLIN_STATE_INPUT_ARGS, MERLIN_STATE_OUTPUT_INDICES) != 0) {{
      fprintf(stderr, "FAIL state ABI mismatch during warmup\n"); return NULL;
    }}
#endif
  }}

  /* Warmup may advance KV/recurrent state. Reset before the measured trajectory so cache warming
   * never changes the semantic input to the reported session. */
  merlin_reset_session();
  w0 = wall_ns();
  t0 = rd_time();
  for (long i = 0; i < n_iters; i++) {{
    uint64_t iw0 = wall_ns();
    uint64_t it0 = rd_time();
    merlin_prepare_step(i);
    merlin_run_multi(MERLIN_ARGS, MERLIN_N_ARGS, MERLIN_WEIGHTS_PTR,
                     MERLIN_INPUT_PTR, MERLIN_OUTPUT_PTR, DESCS);
#if MERLIN_N_STATE_PAIRS > 0
    if (merlin_commit_state(MERLIN_ARGS, MERLIN_N_ARGS, MERLIN_INPUT_PTR,
                            MERLIN_OUTPUT_PTR, MERLIN_N_STATE_PAIRS,
                            MERLIN_STATE_INPUT_ARGS, MERLIN_STATE_OUTPUT_INDICES) != 0) {{
      fprintf(stderr, "FAIL state ABI mismatch during timed session\n"); return NULL;
    }}
#endif
    if (validate_session) merlin_validate_step(i);
    uint64_t it1 = rd_time();
    uint64_t iw1 = wall_ns();
    if (n_iters > 1) {{
      printf("METRIC iter_cycles %ld %llu\\n", i,
             (unsigned long long)((it1 - it0) * (MERLIN_CPU_HZ / MERLIN_TIMEBASE_HZ)));
      printf("METRIC iter_wall_ns %ld %llu\\n", i, (unsigned long long)(iw1 - iw0));
    }}
  }}
  t1 = rd_time();
  w1 = wall_ns();
  if (n_iters > 1) {{                 /* report the PER-ITERATION cost, not the batch total */
    t1 = t0 + (t1 - t0) / (uint64_t)n_iters;
    w1 = w0 + (w1 - w0) / (uint64_t)n_iters;
  }}

session_complete: ;
  int k = MERLIN_OUT_ELEMS < MERLIN_DUMP_CAP ? MERLIN_OUT_ELEMS : MERLIN_DUMP_CAP;
  printf("OUT %d", k);
  for (int i = 0; i < k; i++) {{
    uint32_t bits;
    memcpy(&bits, &((float*)MERLIN_OUTPUT_PTR[0])[i], 4);
    printf(" %u", (unsigned)bits);
  }}
  printf("\\n");

  if (MERLIN_OUT_ELEMS > MERLIN_DUMP_CAP) {{
    int rows = MERLIN_OUT_ELEMS / MERLIN_OUT_LASTDIM;
    printf("ARGMAX %d", rows);
    for (int r = 0; r < rows; r++) {{
      const float *row = &((float*)MERLIN_OUTPUT_PTR[0])[(long)r * MERLIN_OUT_LASTDIM];
      int best = 0; float bv = row[0];
      for (int j = 1; j < MERLIN_OUT_LASTDIM; j++)
        if (row[j] > bv) {{ bv = row[j]; best = j; }}
      printf(" %d", best);
    }}
    printf("\\n");
    float s = 0.0f;
    for (int i = 0; i < MERLIN_OUT_ELEMS; i++) s += ((float*)MERLIN_OUTPUT_PTR[0])[i];
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
  struct rusage usage;
  if (getrusage(RUSAGE_SELF, &usage) == 0)
  printf("METRIC peak_rss_kb %llu\\n", (unsigned long long)usage.ru_maxrss);
#if MERLIN_HAS_SESSION_CORRECTNESS
  if (validate_session) {{
    printf("METRIC correctness_trajectory_steps %ld\\n", merlin_correctness_steps());
    printf("METRIC correctness_trajectory_min_cos_ppm %ld\\n", merlin_correctness_min_cos_ppm());
    printf("METRIC correctness_trajectory_max_rel_ppm %ld\\n", merlin_correctness_max_rel_ppm());
    printf("METRIC correctness_trajectory_top1_matches %ld\\n", merlin_correctness_top1());
  }}
#endif
#if MERLIN_HAS_SESSION_QUALITY
  if (validate_session) {{
    printf("METRIC trajectory_steps %ld\\n", merlin_quality_steps());
    printf("METRIC trajectory_min_cos_ppm %ld\\n", merlin_quality_min_cos_ppm());
    printf("METRIC trajectory_max_rel_ppm %ld\\n", merlin_quality_max_rel_ppm());
    printf("METRIC trajectory_top1_matches %ld\\n", merlin_quality_top1());
  }}
#endif
{dtiming_print}{oprof_print}  printf("DONE\\n");
  fflush(stdout);
  return NULL;
}}

int main(int argc, char **argv) {{
  (void)argc; (void)argv;
{weights_init}
  /* The lowered model alloca's large intermediate buffers; the default 8MB stack overflows
   * (SIGSEGV) for VLAs. Run merlin_run on a pthread with an explicitly-mmap'd {K1_STACK_BYTES}-byte
   * stack (one contiguous region — robust vs main-thread growth, which collides with lower mmaps).
   * Also raise RLIMIT_STACK as a fallback if pthread creation fails. */
  struct rlimit rl = {{ {K1_STACK_BYTES}ULL, {K1_STACK_BYTES}ULL }};
  setrlimit(RLIMIT_STACK, &rl);
  pthread_attr_t attr;
  pthread_t th;
  if (pthread_attr_init(&attr) == 0 &&
      pthread_attr_setstacksize(&attr, (size_t){K1_STACK_BYTES}ULL) == 0 &&
      pthread_create(&th, &attr, worker, NULL) == 0) {{
    pthread_join(th, NULL);
  }} else {{
    worker(NULL);   /* fallback: run inline (relies on the raised RLIMIT) */
  }}
  return 0;
}}
"""


def main_linux_session_c(dump_cap: int | None = 4096) -> str:
    """K1 harness for a generated multi-program session scheduler.

    Every ``iter_wall_ns`` sample covers the full ordered stage graph from reset state. Per-stage
    totals are emitted as diagnostic attribution and are never substituted for that headline total.

    ``dump_cap`` bounds the printed quality output exactly as in :func:`main_linux_c`; ``None``
    prints all of it. The session's element count is a RUN-TIME value here (the scheduler reports
    it), not a compile-time macro, so the uncapped form drops the ceiling from the comparison
    rather than defining a macro over a local variable.
    """
    if dump_cap is None:
        cap_define, k_expr = "", "(int)elems"
    else:
        if int(dump_cap) < 1:
            raise ValueError("dump_cap must be positive, or None for the full session output")
        cap_define = f"#define MERLIN_DUMP_CAP {int(dump_cap)}\n"
        k_expr = "elems < MERLIN_DUMP_CAP ? (int)elems : MERLIN_DUMP_CAP"
    return f'''/* Generated K1 multi-program continuous-session harness. */
#define _GNU_SOURCE
#include <fcntl.h>
#include <pthread.h>
#include <stdint.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include "merlin_session.h"
{cap_define}#define MERLIN_TIMEBASE_HZ {K1_TIMEBASE_HZ}ULL
#define MERLIN_CPU_HZ {K1_CPU_HZ}ULL
static const void *WEIGHTS[MERLIN_SESSION_N_PROGRAMS];
static inline uint64_t rd_time(void) {{ uint64_t t; __asm__ volatile("rdtime %0":"=r"(t)); return t; }}
static inline uint64_t rd_vlenb(void) {{ uint64_t v; __asm__ volatile("csrr %0, vlenb":"=r"(v)); return v; }}
static uint64_t wall_ns(void) {{
  struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}}
static int run_session(int validate, uint64_t *stage_ns) {{
  merlin_session_reset();
  for (int program = 0; program < MERLIN_SESSION_N_PROGRAMS; program++) {{
    if (merlin_session_prepare_program(program) != 0) return -1;
    uint64_t start = wall_ns();
    long steps = merlin_session_program_steps(program);
    for (long step = 0; step < steps; step++)
      if (merlin_session_run_step(program, WEIGHTS[program], step, validate) != 0) return -1;
    if (stage_ns) stage_ns[program] = wall_ns() - start;
  }}
  return 0;
}}
static void *worker(void *unused) {{
  (void)unused;
  printf("=== merlin_k1_session vlenb=%llu ===\\n", (unsigned long long)rd_vlenb());
  cpu_set_t allowed;
  CPU_ZERO(&allowed);
  if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) {{
    fprintf(stderr, "FAIL sched_getaffinity\\n"); return 0;
  }}
  printf("METRIC affinity_cpus %d\\n", CPU_COUNT(&allowed));
  long repeats = 1, warmups = 0; int validate = 0;
  const char *e = getenv("MERLIN_SESSION_REPEATS");
  if (e && *e) {{ repeats = strtol(e, 0, 10); if (repeats < 1) repeats = 1; }}
  e = getenv("MERLIN_SESSION_WARMUPS");
  if (e && *e) {{ warmups = strtol(e, 0, 10); if (warmups < 0) warmups = 0; }}
  e = getenv("MERLIN_VALIDATE_SESSION");
  if (e && *e && strcmp(e, "0") != 0) validate = 1;
  for (long repeat = 0; repeat < warmups; repeat++) if (run_session(0, 0) != 0) return 0;
  uint64_t all_w0 = wall_ns(), all_t0 = rd_time();
  for (long repeat = 0; repeat < repeats; repeat++) {{
    uint64_t stage_ns[MERLIN_SESSION_N_PROGRAMS];
    uint64_t w0 = wall_ns(), t0 = rd_time();
    if (run_session(validate && repeat == 0, stage_ns) != 0) return 0;
    uint64_t t1 = rd_time(), w1 = wall_ns();
    printf("METRIC iter_cycles %ld %llu\\n", repeat,
           (unsigned long long)((t1 - t0) * (MERLIN_CPU_HZ / MERLIN_TIMEBASE_HZ)));
    printf("METRIC iter_wall_ns %ld %llu\\n", repeat, (unsigned long long)(w1 - w0));
    for (int program = 0; program < MERLIN_SESSION_N_PROGRAMS; program++)
      printf("STAGE %ld %s %llu\\n", repeat, merlin_session_program_name(program),
             (unsigned long long)stage_ns[program]);
  }}
  uint64_t all_t1 = rd_time(), all_w1 = wall_ns();
  uint64_t ticks = (all_t1 - all_t0) / (uint64_t)repeats;
  uint64_t elapsed = (all_w1 - all_w0) / (uint64_t)repeats;
  size_t elems = merlin_session_quality_output_elems();
  const float *output = (const float *)merlin_session_quality_output();
  int k = {k_expr};
  printf("OUT %d", k);
  for (int i = 0; i < k; i++) {{ uint32_t bits; memcpy(&bits, &output[i], 4); printf(" %u", bits); }}
  printf("\\n");
  printf("METRIC cycles %llu\\n", (unsigned long long)(ticks * (MERLIN_CPU_HZ / MERLIN_TIMEBASE_HZ)));
  printf("METRIC time_ticks %llu\\n", (unsigned long long)ticks);
  printf("METRIC wall_ns %llu\\n", (unsigned long long)elapsed);
  struct rusage usage;
  if (getrusage(RUSAGE_SELF, &usage) == 0)
    printf("METRIC peak_rss_kb %llu\\n", (unsigned long long)usage.ru_maxrss);
  if (validate) {{
    printf("METRIC correctness_trajectory_steps %ld\\n", merlin_session_correctness_steps());
    printf("METRIC correctness_trajectory_min_cos_ppm %ld\\n",
           merlin_session_correctness_min_cos_ppm());
    printf("METRIC correctness_trajectory_max_rel_ppm %ld\\n",
           merlin_session_correctness_max_rel_ppm());
    printf("METRIC correctness_trajectory_top1_matches %ld\\n",
           merlin_session_correctness_top1());
    printf("METRIC trajectory_steps %ld\\n", merlin_session_quality_steps());
    printf("METRIC trajectory_min_cos_ppm %ld\\n", merlin_session_quality_min_cos_ppm());
    printf("METRIC trajectory_max_rel_ppm %ld\\n", merlin_session_quality_max_rel_ppm());
    printf("METRIC trajectory_top1_matches %ld\\n", merlin_session_quality_top1());
  }}
  printf("DONE\\n"); fflush(stdout); return 0;
}}
int main(void) {{
  for (int program = 0; program < MERLIN_SESSION_N_PROGRAMS; program++) {{
    char key[64]; snprintf(key, sizeof(key), "MERLIN_WEIGHTS_%d", program);
    const char *path = getenv(key); if (!path) {{ fprintf(stderr, "FAIL %s unset\\n", key); return 2; }}
    int fd = open(path, O_RDONLY); if (fd < 0) return 2;
    struct stat st; if (fstat(fd, &st) != 0) return 2;
    if (st.st_size == 0) {{ close(fd); WEIGHTS[program] = (const void *)1; continue; }}
    void *mapped = mmap(0, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0); close(fd);
    if (mapped == MAP_FAILED) return 2; WEIGHTS[program] = mapped;
  }}
  struct rlimit rl = {{ {K1_STACK_BYTES}ULL, {K1_STACK_BYTES}ULL }}; setrlimit(RLIMIT_STACK, &rl);
  pthread_attr_t attr; pthread_t thread;
  if (pthread_attr_init(&attr) == 0 &&
      pthread_attr_setstacksize(&attr, (size_t){K1_STACK_BYTES}ULL) == 0 &&
      pthread_create(&thread, &attr, worker, 0) == 0) pthread_join(thread, 0);
  else worker(0);
  return 0;
}}
'''


# ---- build / deploy / run -----------------------------------------------------------

# A bounded wall clock for the K1 cross-compile/link/scp steps. A pathological schedule (e.g. the
# outer-product contraction feature at a large square regime) makes clang-23 spin for many minutes on
# one object file, hanging a serial beam; time it out so the fork fails-closed as a K1 build error the
# certify ladder records. Board-run (_ssh) already carries its own timeout. Override/disable (0) with
# MERLIN_COMPILE_TIMEOUT_S — the same knob the host/spike codegen path honors. Default unified at
# 900s across all four compile wrappers (was 300): a per-object 300s ceiling tripped legitimate
# whole-model builds (bitvla int8 fails even the baseline at 600s). For a whole-model beam launch
# set MERLIN_COMPILE_TIMEOUT_S=3600; 0 (or empty) disables the ceiling.
_K1_CMD_TIMEOUT_S = int(os.environ.get("MERLIN_COMPILE_TIMEOUT_S", "900") or "0")
_ACTIVE_CELL_DEADLINE_NS: ContextVar[int | None] = ContextVar(
    "merlin_k1_cell_deadline_ns", default=None)


def _bounded_timeout(requested: int | float | None) -> int | float | None:
    deadline = _ACTIVE_CELL_DEADLINE_NS.get()
    if deadline is None:
        return requested
    remaining_ns = deadline - time.monotonic_ns()
    if remaining_ns <= 0:
        raise K1Error("paper cell exhausted its whole-cell deadline")
    remaining = max(1, math.ceil(remaining_ns / 1_000_000_000))
    return remaining if requested is None else min(requested, remaining)


def _whole_cell_deadline(function):
    """Bind one optional monotonic deadline across build, deploy, perf, and validation calls."""
    @wraps(function)
    def wrapped(*args, deadline_ns: int | None = None, **kwargs):
        token = _ACTIVE_CELL_DEADLINE_NS.set(
            int(deadline_ns) if deadline_ns is not None else None)
        try:
            return function(*args, **kwargs)
        finally:
            _ACTIVE_CELL_DEADLINE_NS.reset(token)
    return wrapped


def _run(cmd: list, **kw) -> subprocess.CompletedProcess:
    kw.setdefault("timeout", _K1_CMD_TIMEOUT_S or None)
    kw["timeout"] = _bounded_timeout(kw["timeout"])
    try:
        proc = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, **kw)
    except subprocess.TimeoutExpired:
        raise K1Error(f"command timed out within the bounded compile/cell deadline: "
                      f"{' '.join(map(str, cmd))}")
    if proc.returncode != 0:
        raise K1Error(
            f"command failed: {' '.join(map(str, cmd))}\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")
    return proc


class K1Error(RuntimeError):
    pass


# Weight blobs at/above this size are mmap'd (file, demand-paged) instead of embedded in the
# binary: a ~5G embedded blob makes a binary that won't fit the board /tmp (1.9G) and forces all
# weights resident; mmap keeps resident RAM to the working set. Small models stay embedded
# (self-contained, proven path).
_MMAP_WEIGHTS_THRESHOLD = 1_500_000_000


def build_k1_binary(model_dir: str | Path, work: str | Path, pkg,
                    inputs_npz: str | Path | None = None,
                    force_scalar: bool | None = None,
                    parallel: bool = False,
                    parallel_harts: int | None = None,
                    fallback_policy: str = "allow",
                    mmap_weights: bool | None = None,
                    kernel_backend: str | None = None,
                    dispatch_timing: bool = False, op_profile: bool = False,
                    dump_cap: int | None = 4096,
                    max_session_steps: int | None = None,
                    ours_mr: int = 4, ours_pack_b: bool = False) -> Path:
    """Cross-compile a K1 Linux RVV binary from the workload + RVV package.

    Reuses the EXACT spike/Zephyr lowering (``zephyr_model.prepare_for_lowering`` ->
    ``lower_model_file(vectorize=True, transform_schedule=pkg.schedule_text)``) so the K1 binary
    executes the same emitted RVV as the spike correctness run, and the same data-driven C runtime
    artifacts (``c_runtime.generate`` -> model_gen.h / model_call.c / weights.bin / model_io.h).
    Only the harness (``main_linux.c``) and the compile flags differ (glibc Linux, not medany).

    ``mmap_weights``: True -> the weight blob is NOT embedded; the binary ``mmap``s it from a file
    at run time (deployed alongside, ``MERLIN_WEIGHTS`` env). None (default) -> auto by blob size
    (>= :data:`_MMAP_WEIGHTS_THRESHOLD` -> mmap). When mmap is used a marker file
    ``<work>/USE_MMAP_WEIGHTS`` records the weights.bin path so :func:`run_on_k1` deploys it.
    Returns the path to the built ELF.

    ``kernel_backend="xnnpack"`` (default-off, additive): route the routable plain 2-D f32
    ``linalg.matmul`` dispatches to XNNPACK's RVV GEMM ukernel instead of the Merlin-emitted RVV.
    The prepared MLIR's matmuls are rewritten to ``call @merlin_xnn_gemm_f32`` (see
    ``runtime.backends.xnnpack_board``); the rest of the model lowers UNCHANGED through the same
    RVV pipeline, and the XNNPACK ukernel shim ``.o`` is linked in. With ``None`` (default) the
    binary is byte-for-byte the existing path."""
    if fallback_policy not in ("allow", "forbid"):
        raise ValueError("fallback_policy must be 'allow' or 'forbid'")
    # Per-dispatch matmul-bucket timing (default-OFF) is only meaningful with a routed
    # kernel_backend (the timed region lives in the GEMM shim). Guard FIRST so it fails loud
    # before any toolchain/model work, and never silently no-op's into an always-zero bucket.
    if dispatch_timing and kernel_backend not in ("xnnpack", "openblas", "ours"):
        raise K1Error("dispatch_timing=True requires kernel_backend in {xnnpack, openblas, ours} "
                      "(the matmul-bucket timer lives in the routed GEMM shim)")
    model_dir, work = Path(model_dir).resolve(), Path(work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    # PREPACKED WEIGHT LAYOUT (`prepack_weight_layout`, default-off). Apply the model's weight
    # re-layout ONCE, here, instead of on every inference: materialize a rewritten bundle (the
    # argument transposes erased and the arguments retyped, the safetensors bytes physically
    # transposed) and use it as `model_dir` from this point on. It has to reach BOTH consumers --
    # `prepare_for_lowering` below reads `model_dir/model.mlir`, while `c_runtime.generate` (step 3)
    # independently re-parses the SAME bundle for the ABI table and copies its weight payload
    # verbatim into `weights.bin`. A rewrite that reached only one of them would leave the compiled
    # object indexing a transposed weight that the blob stores untransposed. MEASURED on the live K1,
    # interleaved same-session: 1.70x on small_llama int8. The rewrite is cached per (bundle,
    # rewrite) and NEVER mutates the source recapture, which every other session reads.
    from ..llvmlower import weight_prepack as _wp
    _wp.ensure_registered()   # unregistered => `_composes` swallows the KeyError and never proposes it
    if _wp.FEATURE in (getattr(pkg, "compiler_features", None) or ()):
        try:
            model_dir, _prepack = _wp.prepacked_bundle(model_dir)
        except _wp.PrepackRefused as exc:
            # Fail CLOSED. Falling back to the stock bundle would build the BASELINE while reporting
            # the lever as applied -- the inert-lever failure, but silent and un-auditable.
            raise K1Error(f"{_wp.FEATURE}: {exc}") from exc
        (work / "PREPACKED_BUNDLE").write_text(str(model_dir))
        print(f"[prepack] {model_dir.name}: "
              f"{_prepack.get('weights_pre_transposed')} weights pre-transposed, "
              f"{_prepack.get('transposes_removed')} transposes removed, "
              f"{_prepack.get('mib_moved_per_inference_before')} MiB/inference no longer moved "
              f"(cached={_prepack.get('cached')})")
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
    #
    # `prepare_for_lowering`, NOT the lower-level `_prepare_model_mlir` this used to call. That was a
    # real gap, not a style point: three feature families live in the preparation layer, not in the
    # pass pipeline -- per-op register blocking (the `perop_register_block` sentinel is derived from the
    # PREPARED IR, tagged onto it, and swapped for a concrete feature), the non-contraction vec-rank
    # tagging, and matrix-unit routing. Calling the low-level helper meant a package naming any of them
    # either silently lowered as the BASELINE (nothing tags the IR, so no schedule arm matches and every
    # contraction falls to convert-linalg-to-loops) or died in `normalize` on the unresolved sentinel.
    # Since the K1 is this repo's performance oracle, that made those levers unmeasurable on silicon --
    # while this function's own docstring claimed it "reuses the EXACT spike/Zephyr lowering". Both
    # `zephyr_model.build_app` and `spike_model.build` go through this same entry point, so now all
    # three agree by construction.
    feats_in = frozenset(getattr(pkg, "compiler_features", []) or []) or None
    prepared, feats = zm.prepare_for_lowering(
        model_dir / "model.mlir", work, int8_compute=pkg.is_int8, features=feats_in,
        harts=(int(parallel_harts) if parallel_harts else 1), vlen=VLEN)
    feats = feats or None
    # XNNPACK kernel-backend (default-off, additive): rewrite the routable f32 linalg.matmul ops
    # in the PREPARED MLIR to external calls (@merlin_xnn_gemm_f32). Everything else lowers
    # unchanged. n_xnn_routed records the count; the shim .o is built + linked below.
    _dt = ["-DMERLIN_DISPATCH_TIMING"] if dispatch_timing else []
    xnn_obj = None
    n_xnn_routed = 0
    n_xnn_candidates = 0
    n_xnn_eligible = 0
    if kernel_backend == "xnnpack":
        from ..runtime.backends import xnnpack_board as xb

        if not xb.is_available():
            raise K1Error("kernel_backend='xnnpack' but the XNNPACK RVV ukernel/shim is unavailable "
                          "(see MERLIN_XNNPACK_REPO / tmp/kernels/XNNPACK)")
        if pkg.is_int8:
            raise K1Error("kernel_backend='xnnpack' is an f32 GEMM path; int8 datapath not supported")
        prepared_text = prepared.read_text()
        n_xnn_candidates, n_xnn_eligible = xb.matmul_routing_coverage(prepared_text)
        rewritten, n_xnn_routed = xb.rewrite_matmuls_to_xnn(prepared_text)
        if n_xnn_routed != n_xnn_eligible:
            raise K1Error(
                "XNNPACK rewrite did not route the complete declared eligible GEMM set: "
                f"routed={n_xnn_routed} eligible={n_xnn_eligible}")
        prepared = work / "model.prepared.xnn.mlir"
        prepared.write_text(rewritten)
        # one numbered alias per distinct matmul signature (private decl per signature).
        n_sigs = rewritten.count("func.func private @merlin_xnn_gemm_f32_")
        xnn_obj = xb.build_xnn_object(cc, ["--target=riscv64-unknown-linux-gnu",
                                           f"-march={K1_MARCH}", f"-mabi={K1_MABI}", "-O3",
                                           "-ffast-math", "-DNDEBUG", *_dt],
                                      n_sigs, work / "xnn")
    # OpenBLAS kernel-backend (default-off, additive): the OpenBLAS analogue of the XNNPACK path
    # above. Routes the SAME routable f32 linalg.matmul ops to OpenBLAS's RVV 8x8 GEMM ukernel
    # (@merlin_openblas_gemm_f32); everything else lowers unchanged. n_openblas_routed records the
    # count; the shim .o is built + linked below.
    openblas_obj = None
    n_openblas_routed = 0
    n_openblas_candidates = 0
    n_openblas_eligible = 0
    if kernel_backend == "openblas":
        from ..runtime.backends import openblas_board as ob

        if not ob.is_available():
            raise K1Error("kernel_backend='openblas' but the OpenBLAS RVV ukernel/shim is "
                          "unavailable (see MERLIN_OPENBLAS_REPO / tmp/kernels/OpenBLAS)")
        if pkg.is_int8:
            raise K1Error("kernel_backend='openblas' is an f32 GEMM path; int8 datapath not supported")
        prepared_text = prepared.read_text()
        n_openblas_candidates, n_openblas_eligible = ob.matmul_routing_coverage(prepared_text)
        rewritten, n_openblas_routed = ob.rewrite_matmuls_to_openblas(prepared_text)
        if n_openblas_routed != n_openblas_eligible:
            raise K1Error(
                "OpenBLAS rewrite did not route the complete declared eligible GEMM set: "
                f"routed={n_openblas_routed} eligible={n_openblas_eligible}")
        prepared = work / "model.prepared.openblas.mlir"
        prepared.write_text(rewritten)
        n_sigs = rewritten.count("func.func private @merlin_openblas_gemm_f32_")
        openblas_obj = ob.build_openblas_object(cc, ["--target=riscv64-unknown-linux-gnu",
                                                     f"-march={K1_MARCH}", f"-mabi={K1_MABI}", "-O3",
                                                     "-ffast-math", "-DNDEBUG", *_dt],
                                                n_sigs, work / "openblas")
    # OURS kernel-backend (default-off, additive): the attribution-measurement analogue of the
    # XNNPACK/OpenBLAS paths above. Routes the SAME routable f32 linalg.matmul ops to OUR OWN MR=4
    # accumulator-resident RVV v3 ukernel (@merlin_ours_gemm_f32), through the IDENTICAL rdtime
    # bracket — so the ours matmul bucket is MEASURED, not attributed. n_ours_routed records the
    # count; the shim .o is built + linked below.
    ours_obj = None
    n_ours_routed = 0
    if kernel_backend == "ours":
        from ..runtime.backends import ours_board as our

        if not our.is_available():
            raise K1Error("kernel_backend='ours' but the ours RVV GEMM shim is unavailable "
                          "(runtime/backends/ours_board/ours_gemm_rvv_shim.c)")
        if pkg.is_int8:
            raise K1Error("kernel_backend='ours' is an f32 GEMM path; int8 datapath not supported")
        rewritten, n_ours_routed = our.rewrite_matmuls_to_ours(prepared.read_text())
        prepared = work / "model.prepared.ours.mlir"
        prepared.write_text(rewritten)
        n_sigs = rewritten.count("func.func private @merlin_ours_gemm_f32_")
        _ourdt = ["-DOURS_PACK_B"] if ours_pack_b else []
        ours_obj = our.build_ours_object(cc, ["--target=riscv64-unknown-linux-gnu",
                                              f"-march={K1_MARCH}", f"-mabi={K1_MABI}", "-O3",
                                              "-ffast-math", "-DNDEBUG", f"-DOURS_MR={int(ours_mr)}",
                                              *_ourdt, *_dt],
                                         n_sigs, work / "ours")
    # qd8 kernel-backend (default-off, additive): the DYNAMIC-INT8 analogue of the XNNPACK f32 path.
    # Routes the SAME routable f32 linalg.matmul ops to XNNPACK's qd8-f32-qc8w RVV GEMM ukernel via a
    # shim that per-row dynamic-quantizes the activation to int8 (+ the offline per-channel int8 weight),
    # producing f32. FAIL-CLOSED: build_qd8_object raises until the shim is implemented + K1-validated
    # (qd8 is lossy vs the f32 golden; its cos gate must be quantization-aware, calibrated on the board).
    qd8_obj = None
    n_qd8_routed = 0
    if kernel_backend == "qd8":
        from ..runtime.backends import xnnpack_board as xb

        rewritten, n_qd8_routed = xb.rewrite_matmuls_to_qd8(prepared.read_text())
        prepared = work / "model.prepared.qd8.mlir"
        prepared.write_text(rewritten)
        n_sigs = rewritten.count("func.func private @merlin_xnn_qd8_gemm_")
        qd8_obj = xb.build_qd8_object(cc, ["--target=riscv64-unknown-linux-gnu",
                                           f"-march={K1_MARCH}", f"-mabi={K1_MABI}", "-O3",
                                           "-ffast-math", "-DNDEBUG", *_dt],
                                      n_sigs, work / "qd8")
    # PER-OP WHOLE-MODEL PROFILE (default-off, additive). Interleave `@merlin_prof_mark(i32)`
    # calls between the top-level ops of @forward so the board can attribute wall time to the ops
    # the model actually executes — the 94-97% of model time the matmul-bucket timer above cannot
    # see. Runs LAST (after any kernel-backend rewrite) so a routed matmul is profiled as the call
    # it became. The id->op table is written next to the build for the driver to join against.
    if op_profile:
        from ..llvmlower import op_profile as opf

        instrumented, prof_table = opf.instrument(prepared.read_text())
        prepared = work / "model.prepared.opprof.mlir"
        prepared.write_text(instrumented)
        opf.write_table(prof_table, work / "opprof_table.json")

    # hoist_static_allocs=False: keep big intermediate buffers on the HEAP (board RAM) instead of
    # promoting them to stack alloca — large models otherwise overflow even a multi-GB stack.
    from ..llvmlower.pipeline import PipelineError
    # impr-fork compiler features (PASS/HEURISTIC/PATTERN). Without threading these the K1 build
    # silently used the BASELINE codegen for any feature-bearing fork — invalidating its
    # benchmark. Pass through so the K1 binary matches what spike certified.
    # `feats` was resolved above by prepare_for_lowering (which may have swapped a sentinel for a
    # concrete, table-specific feature) — do NOT re-read the package's raw list here, or the schedule
    # would be built from a name the IR was never tagged for.
    if force_scalar is None:
        force_scalar = bool(os.environ.get("MERLIN_K1_FORCE_SCALAR"))
    if parallel_harts:
        # VECTOR + MULTICORE: the package's RVV schedule with an outer OpenMP-parallel loop
        # layered under it, so the board runs real RVV on several cores at once. This is the
        # same lowering the multicore Zephyr image uses, which makes the K1 the honest
        # PERFORMANCE oracle for it: spike cannot answer that question (it simulates a spinning
        # hart at full speed), and RTL sim cannot run a whole model. Thread count is chosen at
        # RUN time via OMP_NUM_THREADS, so one binary yields a whole scaling curve.
        res = lower_model_file(prepared, work / "lower_vecomp", targets=(), textual=True,
                               vectorize=True, transform_schedule=pkg.schedule_text,
                               hoist_static_allocs=False, features=feats,
                               parallel_harts=parallel_harts)
    elif parallel:
        # Multicore path: scalar int8 datapath + OpenMP parallel loops (no fixed-width
        # vectorize). Used for the big models (rdt/smolvla) that crash the vectorized lowering
        # AND are too slow single-core — the parallel loops fan across the board's 8 cores.
        res = lower_model_file(prepared, work / "lower_omp", targets=(), textual=True,
                               vectorize=False, hoist_static_allocs=False, parallel=True)
    else:
        try:
            if force_scalar:
                raise PipelineError("forced scalar (MERLIN_K1_FORCE_SCALAR)")
            res = lower_model_file(prepared, work / "lower", targets=(), textual=True,
                                   vectorize=True, transform_schedule=pkg.schedule_text,
                                   hoist_static_allocs=False, features=feats)
        except PipelineError:
            # Some models (e.g. xr0's rank-4 two-batch attention) hit a vectorize-path
            # specialization (linalg-specialize-generic-ops) that emits an invalid rank-4
            # linalg.batch_matmul. Fall back to the SCALAR lowering (no specialize pass): the int8
            # datapath is intact, only the contraction stays a scalar loop (correct, unvectorized).
            if fallback_policy == "forbid":
                raise
            res = lower_model_file(prepared, work / "lower_scalar", targets=(), textual=True,
                                   vectorize=False, hoist_static_allocs=False)

    # 2. compile model.ll -> K1 Linux object. The IR is emitted by the repo's clang-23 toolchain
    #    and carries LLVM-23 attribute syntax (e.g. `captures(none)`) the SpacemiT clang-19 can't
    #    parse, so the .ll -> object step uses THE SAME clang-23 (re-targeted to linux-gnu); the
    #    SpacemiT clang is used for the C sources + final glibc link below. NO medany/freestanding
    #    — glibc hosted, small code model, fully relocatable.
    clang23 = toolchain.clang()
    model_o = work / "model.o"
    # codegen_march(), not K1_MARCH: this is OUR emitted model.ll becoming the model object, so it
    # pays the doubled-LMUL penalty when the march string leaves VLEN at the RVV minimum. The C glue
    # and the baseline shim arms below deliberately keep K1_MARCH so their flags are not re-written
    # mid-campaign. Whether this transfers to whole models is exactly what is being measured.
    # MERLIN_K1_MODEL_OPT (default -O2 -> byte-identical) lowers the OPT LEVEL of the model object
    # only. It exists for one job: when the same LLVM IR is correct on x86 and wrong on every riscv64
    # target, the next cut is whether the RISC-V backend's OPTIMIZED codegen is what breaks it, and
    # that needs -O0/-O1 builds of the model while everything else (harness, runtime, flags) is held
    # fixed. Diagnostic only -- never set it for a measurement, since it changes the emitted kernel.
    model_opt = os.environ.get("MERLIN_K1_MODEL_OPT", "-O2").split()
    # CFLAGS-CLASS FEATURES apply to the MODEL OBJECT ONLY. Some levers are not expressible in the
    # IR or the transform schedule at all: the vector register-group width comes from clang's own
    # RISC-V vectorizer (`-riscv-v-register-bit-width-lmul`, default 2), so a feature that resolves
    # and is never handed to the compiler emits byte-identical code while reporting as applied --
    # the inert-lever failure this file's own comments warn about. Only the model object gets these;
    # the harness and runtime stay on fixed flags so a measurement changes one thing.
    from ..llvmlower.impr_features import apply_cflags as _apply_cflags
    _model_flags = _apply_cflags(
        [f"-march={codegen_march()}", f"-mabi={K1_MABI}", *model_opt, "-Wno-override-module"],
        feats or frozenset())
    _run([clang23, "--target=riscv64-unknown-linux-gnu", *_model_flags,
          "-c", res.ll_path, "-o", model_o])
    # 2b. POST-CODEGEN CENSUS: is the model still IN the object? A backend that deletes reachable
    # code still links, still produces a binary, and then reports a spectacular speedup for
    # computing nothing -- smolvla linked a 512 MB ELF whose `forward` was 3,654 bytes with a
    # complete call set of malloc/memset/roundevenf. Nothing between here and the board could see
    # it. The bound is derived from the prepared IR (one instruction per structured op that reaches
    # an output) and carries no constant; it raises, so an erased model cannot be timed.
    from ..llvmlower.codegen_census import require_commensurate as _census_require
    _census = _census_require(prepared, model_o, "forward")
    print(f"[census] {_census.as_dict()}", flush=True)

    # 3. data-driven runtime artifacts (arg table, ciface, weights.bin, embedded io).
    cgen = work / "cgen"
    # `max_session_steps` caps how much of a session corpus is EMBEDDED AS C LITERALS. It is not a
    # micro-optimization for big corpora: resnet50's 256-step, 154 MB session_inputs.npz becomes a
    # 770 MB model_io.h that costs ~7 GB of RSS to compile, and a correctness or per-step latency
    # run needs a handful of steps. None (the default) embeds the whole corpus, unchanged.
    cinfo = c_runtime.generate(model_dir, cgen, inputs_npz,
                               max_session_steps=max_session_steps)
    if cinfo.get("has_session_quality"):
        (work / "HAS_SESSION_QUALITY").write_text("1")

    # Decide embed-vs-mmap by the weight-blob size (auto unless caller forced it).
    weights_bin = cgen / "weights.bin"
    if mmap_weights is None:
        mmap_weights = (weights_bin.stat().st_size >= _MMAP_WEIGHTS_THRESHOLD
                        if weights_bin.is_file() else False)

    # 4. weights.bin -> binary blob object (embedded path only; mmap reads the file at run time).
    weights_o = work / "weights_blob.o"
    if not mmap_weights:
        _run([ld, "-r", "-b", "binary", "-o", weights_o, "weights.bin"], cwd=cgen)
    else:
        (work / "USE_MMAP_WEIGHTS").write_text(str(weights_bin))

    # 5. the Linux harness.
    main_c = work / "main_linux.c"
    main_c.write_text(main_linux_c(dump_cap=dump_cap, mmap_weights=mmap_weights,
                                   dispatch_timing=dispatch_timing, op_profile=op_profile))

    # 6. link the final binary. Reuse the repo's portable C runtime + generated ciface. Prefer a
    #    static binary (no glibc-version coupling to the board); fall back to dynamic, which is
    #    fine since it runs on the board's own glibc.
    rt = runtime_dir() / "c"
    abi = runtime_dir() / "abi"
    binary = work / "merlin_k1"
    import os as _os
    k1_opt = _os.environ.get("MERLIN_K1_OPT", "-O2")

    # The whole-model ciface `model_call.c` is one huge function (200+ weight args) that crashes the
    # SpacemiT clang-19 (v1.1.2) register scavenger ("Incomplete scavenging after 2nd pass") for large
    # models — precision- and opt-level-independent (both -O2 and -O1 crash; small models are fine).
    # Fix: compile JUST model_call.c with the repo's clang-23 (which handles it; it needs no stdlib
    # headers), and compile the other small C sources + link with the SpacemiT clang-19 — which has its
    # libc/crt + builtin headers (the clang-23 install lacks stddef.h). clang-19 links the clang-23
    # object fine (ELF/ABI compatible; model.o above is likewise clang-23-produced).
    model_call_c = cgen / "model_call.c"
    sysroot = root / "sysroot"
    other_c = [main_c, rt / "merlin_model.c", abi / "mlir_runtime.c"]
    if op_profile:                      # per-op tick accumulator the instrumented IR calls
        other_c = other_c + [rt / "merlin_op_prof.c"]
    # MEASUREMENT PROXY for the lean runtime's static arena (env MERLIN_BUMP_MALLOC, default OFF):
    # link a bump allocator that overrides glibc malloc/free, isolating how much of the dispatch
    # overhead is the ~4391 per-op malloc/free the lean arena eliminates. Baseline byte-identical when off.
    if _os.environ.get("MERLIN_BUMP_MALLOC"):
        other_c = other_c + [rt / "merlin_bump_linux.c"]

    if Path(clang23).exists() and sysroot.is_dir():
        model_call_o = work / "model_call.o"
        _run([str(clang23), "--target=riscv64-unknown-linux-gnu", f"-march={K1_MARCH}",
              f"-mabi={K1_MABI}", k1_opt, "-Wno-override-module", f"--sysroot={sysroot}",
              f"--gcc-toolchain={root}", f"-I{rt}", f"-I{cgen}", "-c", str(model_call_c),
              "-o", str(model_call_o)])
        c_inputs = [*other_c, model_call_o]
    else:
        c_inputs = [*other_c, model_call_c]   # small models: clang-19 compiles model_call.c fine
    base = [cc, "--target=riscv64-unknown-linux-gnu", f"-march={K1_MARCH}", f"-mabi={K1_MABI}",
            k1_opt, "-Wno-override-module", f"-I{rt}", f"-I{cgen}", *c_inputs, model_o]
    if xnn_obj is not None:                       # XNNPACK RVV GEMM ukernel shim
        base += [str(xnn_obj)]
    if openblas_obj is not None:                  # OpenBLAS RVV GEMM ukernel shim
        base += [str(openblas_obj)]
    if ours_obj is not None:                       # ours v3 RVV GEMM ukernel shim (attribution)
        base += [str(ours_obj)]
    if qd8_obj is not None:                        # qd8 dynamic-int8 RVV GEMM ukernel shim
        base += [str(qd8_obj)]
    if not mmap_weights:
        base += [str(weights_o)]
    if parallel or parallel_harts:
        # Resolve the cross-built __kmpc_* symbols against the static libomp; its LLVM runtime
        # needs the C++/dl deps (see the libomp build note). Order matters: libomp before its deps.
        libomp = K1_OPENMP_DIR / "libomp.a"
        if not libomp.is_file():
            raise K1Error(f"parallel build needs {libomp} (cross-build libomp first)")
        base += [str(libomp), "-lstdc++", "-ldl"]
    base += ["-lm", "-lpthread", "-o", binary]
    try:
        _run([*base, "-static"])
    except K1Error:
        _run(base)
    if not binary.is_file():
        raise K1Error(f"K1 cross-compile produced no binary at {binary}")
    if kernel_backend == "xnnpack":
        (work / "N_XNN_ROUTED").write_text(str(n_xnn_routed))
        (work / "N_XNN_ELIGIBLE").write_text(str(n_xnn_eligible))
        (work / "N_XNN_CANDIDATES").write_text(str(n_xnn_candidates))
    if kernel_backend == "openblas":
        (work / "N_OPENBLAS_ROUTED").write_text(str(n_openblas_routed))
        (work / "N_OPENBLAS_ELIGIBLE").write_text(str(n_openblas_eligible))
        (work / "N_OPENBLAS_CANDIDATES").write_text(str(n_openblas_candidates))
    if kernel_backend == "ours":
        (work / "N_OURS_ROUTED").write_text(str(n_ours_routed))
    return binary


def build_k1_session_binary(model_dir: str | Path, work: str | Path, pkg, *,
                            force_scalar: bool = False, parallel_harts: int | None = None,
                            fallback_policy: str = "forbid",
                            kernel_backend: str | None = None,
                            dump_cap: int | None = 4096) -> Path:
    """Cross-compile all programs in a version-2 session and link one K1 process.

    XNNPACK/OpenBLAS swaps use one process-wide shim object. Stage-local monomorphic aliases are
    renumbered into a global symbol range before lowering, avoiding both duplicate definitions and
    cross-stage ABI aliasing while preserving one memory/timing domain for the complete session.
    """
    if fallback_policy not in {"allow", "forbid"}:
        raise ValueError("fallback_policy must be 'allow' or 'forbid'")
    if kernel_backend not in {None, "xnnpack", "openblas"}:
        raise K1Error(f"multi-program kernel backend {kernel_backend!r} is not supported")
    if kernel_backend is not None and pkg.is_int8:
        raise K1Error(f"kernel_backend={kernel_backend!r} is an f32 GEMM path")
    model_dir, work = Path(model_dir).resolve(), Path(work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    cc = toolchain_cc()
    if cc is None:
        raise K1Error("SpacemiT toolchain not found (set MERLIN_K1_TOOLCHAIN)")
    root = _toolchain_root()
    from ..llvmlower import session_bundle, toolchain
    from ..llvmlower.lower import lower_model_file
    from ..llvmlower.pipeline import PipelineError
    from ..runtime.backends import zephyr_model as zm

    generated = work / "session_gen"
    manifest = session_bundle.generate(model_dir, generated)
    clang23 = toolchain.clang()
    sysroot = root / "sysroot"
    rt = runtime_dir() / "c"
    abi = runtime_dir() / "abi"
    model_opt = os.environ.get("MERLIN_K1_MODEL_OPT", "-O2").split()
    k1_opt = os.environ.get("MERLIN_K1_OPT", "-O2")
    features = frozenset(getattr(pkg, "compiler_features", []) or []) or None
    objects: list[Path] = []
    weights: list[dict[str, str]] = []
    routed_total = 0
    eligible_total = 0
    candidate_total = 0
    signature_total = 0
    backend_module = None
    backend_symbol = ""
    if kernel_backend == "xnnpack":
        from ..runtime.backends import xnnpack_board as backend_module
        backend_symbol = "merlin_xnn_gemm_f32"
        if not backend_module.is_available():
            raise K1Error("multi-program XNNPACK RVV shim is unavailable")
    elif kernel_backend == "openblas":
        from ..runtime.backends import openblas_board as backend_module
        backend_symbol = "merlin_openblas_gemm_f32"
        if not backend_module.is_available():
            raise K1Error("multi-program OpenBLAS RVV shim is unavailable")

    for record in manifest["programs"]:
        index, name = int(record["index"]), str(record["name"])
        bundle_dir = Path(record["bundle"])
        stage_work = work / f"stage_{index}_{name}"
        stage_work.mkdir(parents=True, exist_ok=True)
        prepared = zm._prepare_model_mlir(
            bundle_dir / "model.mlir", stage_work, int8_compute=pkg.is_int8)
        renamed = session_bundle.rename_forward(
            prepared.read_text(encoding="utf-8"), str(record["entrypoint"]))
        if backend_module is not None:
            candidates, eligible = backend_module.matmul_routing_coverage(renamed)
            if kernel_backend == "xnnpack":
                renamed, routed = backend_module.rewrite_matmuls_to_xnn(renamed)
            else:
                renamed, routed = backend_module.rewrite_matmuls_to_openblas(renamed)
            if routed != eligible:
                raise K1Error(
                    f"kernel_backend={kernel_backend!r} did not route the complete declared "
                    f"eligible GEMM set for stage {name!r}: routed={routed} eligible={eligible}")
            renamed, local_signatures = _renumber_backend_aliases(
                renamed, backend_symbol, signature_total)
            signature_total += local_signatures
            routed_total += routed
            eligible_total += eligible
            candidate_total += candidates
        prepared = stage_work / "model.prepared.renamed.mlir"
        prepared.write_text(renamed, encoding="utf-8")
        try:
            if force_scalar:
                raise PipelineError("forced scalar")
            lowered = lower_model_file(
                prepared, stage_work / "lower", targets=(), textual=True, vectorize=True,
                transform_schedule=pkg.schedule_text, hoist_static_allocs=False,
                features=features, parallel_harts=parallel_harts)
        except PipelineError:
            if fallback_policy == "forbid":
                raise
            lowered = lower_model_file(
                prepared, stage_work / "lower_scalar", targets=(), textual=True,
                vectorize=False, hoist_static_allocs=False)
        model_object = stage_work / "model.o"
        # Same cflags-class features as the primary site; a staged sibling that skipped them would
        # measure a different compiler than the one the feature set names.
        from ..llvmlower.impr_features import apply_cflags as _apply_cflags_stage
        _stage_flags = _apply_cflags_stage(
            [f"-march={codegen_march()}", f"-mabi={K1_MABI}", *model_opt, "-Wno-override-module"],
            features or frozenset())
        _run([clang23, "--target=riscv64-unknown-linux-gnu", *_stage_flags,
              "-c", lowered.ll_path, "-o", model_object])
        # Same post-codegen census as the primary site (see there): a staged sibling that skipped it
        # would be the one path on which an erased model still reaches a timing run.
        from ..llvmlower.codegen_census import require_commensurate as _census_require_stage
        _stage_census = _census_require_stage(prepared, model_object,
                                              str(record["entrypoint"]))
        print(f"[census] stage {name!r} {_stage_census.as_dict()}", flush=True)
        model_object = _namespace_stage_object(
            model_object, stage_work / "model.namespaced.o", index=index,
            entrypoint=str(record["entrypoint"]))
        objects.append(model_object)

        cgen = Path(record["generated"])
        call_object = stage_work / "model_call.o"
        call_command = [str(clang23), "--target=riscv64-unknown-linux-gnu", f"-march={K1_MARCH}",
                        f"-mabi={K1_MABI}", k1_opt, "-Wno-override-module"]
        if sysroot.is_dir():
            call_command += [f"--sysroot={sysroot}", f"--gcc-toolchain={root}"]
        _run([*call_command, f"-I{rt}", f"-I{cgen}", "-c", cgen / "model_call.c",
              "-o", call_object])
        adapter_object = stage_work / "stage_adapter.o"
        _run([cc, "--target=riscv64-unknown-linux-gnu", f"-march={K1_MARCH}",
              f"-mabi={K1_MABI}", k1_opt, f"-I{rt}", f"-I{cgen}", "-c",
              cgen / "stage_adapter.c", "-o", adapter_object])
        objects += [call_object, adapter_object]
        weights.append({"program": str(index), "name": name,
                        "path": str(cgen / "weights.bin"),
                        "sha256": str(record["weights_sha256"])})

    if backend_module is not None:
        if routed_total < 1 or signature_total < 1:
            raise K1Error(
                f"kernel_backend={kernel_backend!r} routed no eligible operations in the session")
        backend_work = work / str(kernel_backend)
        flags = ["--target=riscv64-unknown-linux-gnu", f"-march={K1_MARCH}",
                 f"-mabi={K1_MABI}", "-O3", "-ffast-math", "-DNDEBUG"]
        if kernel_backend == "xnnpack":
            objects.append(backend_module.build_xnn_object(
                cc, flags, signature_total, backend_work))
            (work / "N_XNN_ROUTED").write_text(str(routed_total), encoding="utf-8")
            (work / "N_XNN_ELIGIBLE").write_text(str(eligible_total), encoding="utf-8")
            (work / "N_XNN_CANDIDATES").write_text(str(candidate_total), encoding="utf-8")
        else:
            objects.append(backend_module.build_openblas_object(
                cc, flags, signature_total, backend_work))
            (work / "N_OPENBLAS_ROUTED").write_text(str(routed_total), encoding="utf-8")
            (work / "N_OPENBLAS_ELIGIBLE").write_text(str(eligible_total), encoding="utf-8")
            (work / "N_OPENBLAS_CANDIDATES").write_text(str(candidate_total), encoding="utf-8")

    marker = work / "USE_MMAP_WEIGHTS.json"
    marker.write_text(json.dumps(weights, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    main_c = work / "main_session.c"
    main_c.write_text(main_linux_session_c(dump_cap), encoding="utf-8")
    binary = work / "merlin_k1_session"
    base = [cc, "--target=riscv64-unknown-linux-gnu", f"-march={K1_MARCH}",
            f"-mabi={K1_MABI}", k1_opt, f"-I{rt}", f"-I{generated}", main_c,
            generated / "merlin_session.c", rt / "merlin_model.c", abi / "mlir_runtime.c",
            *objects]
    if parallel_harts:
        libomp = K1_OPENMP_DIR / "libomp.a"
        if not libomp.is_file():
            raise K1Error(f"parallel session build needs {libomp}")
        base += [str(libomp), "-lstdc++", "-ldl"]
    base += ["-lm", "-lpthread", "-o", binary]
    try:
        _run([*base, "-static"])
    except K1Error:
        _run(base)
    if not binary.is_file():
        raise K1Error(f"K1 session cross-compile produced no binary at {binary}")
    (work / "HAS_SESSION_QUALITY").write_text("1", encoding="utf-8")
    return binary


def _renumber_backend_aliases(text: str, symbol_base: str, offset: int) -> tuple[str, int]:
    """Move one stage's zero-based backend aliases into a process-wide symbol range."""
    if offset < 0:
        raise ValueError("backend alias offset must be non-negative")
    local_signatures = text.count(f"func.func private @{symbol_base}_")
    for local in range(local_signatures - 1, -1, -1):
        # Include `(` so `_1` cannot match `_10`; both MLIR calls and declarations use this token.
        text = text.replace(
            f"@{symbol_base}_{local}(", f"@{symbol_base}_{offset + local}(")
    return text, local_signatures


def _namespace_stage_object(source: Path, destination: Path, *, index: int,
                            entrypoint: str) -> Path:
    """Namespace compiler-emitted public helpers while preserving the stage ABI entry points.

    MLIR's LLVM lowering emits helpers such as ``dealloc_helper`` with external linkage. Linking
    several independently lowered modules therefore collides even though their public forwards are
    unique. Rename every other defined external symbol in the ELF object; llvm-objcopy also updates
    its internal relocations, while undefined libc/backend references remain untouched.
    """
    llvm_bin = repo_root() / "third_party" / "llvm-install" / "bin"
    nm, objcopy = llvm_bin / "llvm-nm", llvm_bin / "llvm-objcopy"
    if not nm.is_file() or not objcopy.is_file():
        nm_fallback, objcopy_fallback = Path("/usr/lib/llvm-18/bin/llvm-nm"), Path(
            "/usr/lib/llvm-18/bin/llvm-objcopy")
        nm = nm if nm.is_file() else nm_fallback
        objcopy = objcopy if objcopy.is_file() else objcopy_fallback
    if not nm.is_file() or not objcopy.is_file():
        raise K1Error("multi-program linking requires llvm-nm and llvm-objcopy")
    listed = _run([nm, "--defined-only", "--extern-only", "--format=posix", source])
    preserve = {entrypoint, f"_mlir_ciface_{entrypoint}"}
    symbols = []
    for line in listed.stdout.splitlines():
        parts = line.split()
        if parts and parts[0] not in preserve:
            symbols.append(parts[0])
    if not symbols:
        shutil.copy2(source, destination)
        return destination
    command: list[object] = [objcopy]
    for symbol in symbols:
        command.append(f"--redefine-sym={symbol}=merlin_stage_{index}__{symbol}")
    command += [source, destination]
    _run(command)
    if not destination.is_file():
        raise K1Error(f"stage symbol namespacing produced no object at {destination}")
    return destination


def _ssh(*args: str, timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["ssh", "-i", K1_SSH_KEY, *_SSH_PORT_OPTS, "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
         "-o", "StrictHostKeyChecking=no", K1_HOST, *args],
        capture_output=True, text=True, timeout=_bounded_timeout(timeout))


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


def board_conditions() -> dict[str, Any]:
    """Read the K1 conditions that can otherwise confound cross-framework latency.

    This is deliberately a separate, untimed SSH probe. Paper adapters call it immediately before
    and after a headline session and preserve both observations; they never infer frequency from
    the requested governor or from Merlin's cycle-conversion constant.
    """
    command = (
        "p=/sys/devices/system/cpu/cpufreq/policy0; "
        "printf 'governor=%s\\n' \"$(cat $p/scaling_governor 2>/dev/null)\"; "
        "printf 'current_khz=%s\\n' \"$(cat $p/scaling_cur_freq 2>/dev/null)\"; "
        "printf 'max_khz=%s\\n' \"$(cat $p/cpuinfo_max_freq 2>/dev/null)\"; "
        "printf 'max_thermal_millic=%s\\n' \"$(cat /sys/class/thermal/thermal_zone*/temp "
        "2>/dev/null | sort -nr | head -1)\""
    )
    proc = _ssh(command, timeout=30)
    if proc.returncode:
        raise K1Error(f"K1 board-condition probe failed: {proc.stderr.strip()}")
    values: dict[str, Any] = {}
    for line in proc.stdout.splitlines():
        key, sep, value = line.partition("=")
        if not sep or not key or not value:
            continue
        try:
            values[key] = int(value)
        except ValueError:
            values[key] = value
    required = {"governor", "current_khz", "max_khz", "max_thermal_millic"}
    if set(values) != required:
        raise K1Error(f"incomplete K1 board-condition observation: {values}")
    return values


def run_arch_probe(source: str | Path, *, timeout: int = 60) -> dict[str, Any]:
    """Cross-compile and run a small architecture-only probe on the K1.

    The source is part of the target contract, not a benchmark candidate: it may read architectural
    CSRs and operating-system facts but must not time a kernel.  The returned source digest and raw
    console make the preflight evidence reproducible.  Deployment is serialized with every other K1
    user and the temporary remote binary is removed on exit.
    """
    source = Path(source)
    if not source.is_file():
        raise K1Error(f"K1 architecture probe does not exist: {source}")
    cc = toolchain_cc()
    if cc is None:
        raise K1Error(f"K1 cross-toolchain is unavailable under {K1_TOOLCHAIN}")
    if not K1_HOST or not Path(K1_SSH_KEY).is_file():
        raise K1Error("K1 SSH endpoint or key is unavailable")
    source_bytes = source.read_bytes()
    digest = hashlib.sha256(source_bytes).hexdigest()
    remote = f"/tmp/merlin_k1_arch_probe_{digest[:12]}"
    with tempfile.TemporaryDirectory(prefix="merlin-k1-probe-") as tmp:
        binary = Path(tmp) / "probe"
        _run([
            str(cc), "--target=riscv64-unknown-linux-gnu", f"-march={K1_MARCH}",
            f"-mabi={K1_MABI}", "-O2", str(source), "-static", "-o", str(binary),
        ])
        with board_lock():
            _run([
                "scp", "-i", K1_SSH_KEY, *_SCP_PORT_OPTS, "-o", "BatchMode=yes",
                "-o", "StrictHostKeyChecking=no", str(binary), f"{K1_HOST}:{remote}",
            ])
            try:
                _ssh(f"chmod +x {remote}", timeout=30)
                proc = _ssh(remote, timeout=timeout)
            finally:
                try:
                    _ssh(f"rm -f {remote}", timeout=30)
                except Exception:  # noqa: BLE001 -- cleanup must not hide the probe verdict
                    pass
    if proc.returncode:
        raise K1Error(f"K1 architecture probe failed ({proc.returncode}): {proc.stderr.strip()}")
    values: dict[str, Any] = {}
    for line in proc.stdout.splitlines():
        key, sep, value = line.partition("=")
        if not sep or not key:
            continue
        value = value.strip()
        try:
            values[key.strip()] = int(value)
        except ValueError:
            values[key.strip()] = value
    return {"source": str(source), "source_sha256": digest, "values": values,
            "stdout": proc.stdout, "stderr": proc.stderr, "returncode": proc.returncode}


#: Above this, the deployed binary goes to the rootfs (flash) instead of /tmp. The board's /tmp
#: is a 1.9 GB tmpfs -- i.e. RAM -- so a weights-embedded binary lands in the same memory the
#: model needs to run in, and two of them do not fit at all. This is the SAME reason the mmap
#: path already stages weights.bin under K1_REMOTE_DIR; a binary with the weights linked in IS
#: the weights. Measured: a 1.3 GB TinyLlama int8 binary fills /tmp and scp fails with a bare
#: "write remote ... Failure" that looks like a network error rather than ENOSPC.
_TMPFS_BINARY_LIMIT = 256 * 1024 * 1024


def _remote_binary_path(binary: str | Path, name: str) -> str:
    """Where to stage the binary on the board: /tmp for small ones, flash for large."""
    try:
        big = Path(binary).stat().st_size > _TMPFS_BINARY_LIMIT
    except OSError:
        big = False
    return f"{K1_REMOTE_DIR}/{name}" if big else f"/tmp/{name}"


def run_binary_on_k1(model_dir: str | Path, bwork: str | Path, pkg, binary: str | Path, *,
                     env: dict[str, str] | None = None, timeout: int = 600) -> dict[str, Any]:
    """Deploy and run an ALREADY-BUILT K1 binary, with an explicit environment.

    Split out of :func:`run_on_k1`'s build+run ladder so a measurement can run the SAME binary
    repeatedly under different settings — the multicore scaling curve varies only
    ``OMP_NUM_THREADS``, and rebuilding per point would put a different object under each
    measurement and make the curve unattributable.

    Takes the board lock around deploy+run only (the cross-compile is CPU-local), matching
    ``run_on_k1``: concurrent agents can build in parallel while the physical board serializes.
    """
    from ..runtime.backends import zephyr_model as zm

    if not K1_HOST:
        raise K1Error("MERLIN_K1_HOST unset — board unreachable")
    bwork = Path(bwork)
    remote = _remote_binary_path(binary, f"{Path(model_dir).name}_{pkg.run_id}_envrun_merlin_k1")
    with board_lock():
        if remote.startswith(K1_REMOTE_DIR):
            _ssh(f"mkdir -p {K1_REMOTE_DIR}", timeout=30)
        _run(["scp", "-i", K1_SSH_KEY, *_SCP_PORT_OPTS, "-o", "BatchMode=yes",
              "-o", "StrictHostKeyChecking=no", str(binary), f"{K1_HOST}:{remote}"])
        marker = bwork / "USE_MMAP_WEIGHTS"
        wenv, remote_w = "", None
        if marker.is_file():
            _ssh(f"mkdir -p {K1_REMOTE_DIR}", timeout=30)
            remote_w = f"{K1_REMOTE_DIR}/{Path(remote).name}.weights.bin"
            _run(["scp", "-i", K1_SSH_KEY, *_SCP_PORT_OPTS, "-o", "BatchMode=yes",
                  "-o", "StrictHostKeyChecking=no", marker.read_text().strip(),
                  f"{K1_HOST}:{remote_w}"])
            wenv = f"MERLIN_WEIGHTS={remote_w} "
        envs = "".join(f"{k}={v} " for k, v in (env or {}).items())
        try:
            _ssh(f"chmod +x {remote}", timeout=30)
            proc = _ssh(f"{wenv}{envs}{remote}", timeout=timeout)
            return zm._parse_console(proc.stdout + proc.stderr, proc.returncode)
        finally:
            try:
                _ssh(f"rm -f {remote}" + (f" {remote_w}" if remote_w else ""), timeout=30)
            except Exception:  # noqa: BLE001
                pass


def _is_multi_program(model_dir: str | Path) -> bool:
    """True when this bundle is a VERSION-2 session, i.e. several programs in one image.

    Version 1 declares a step count and stays a single program (the harness loops its steps
    internally), so it takes the ordinary single-program build path. Only version 2 routes to the
    multi-program builder. Kept as its own predicate so a caller can ask the question without
    duplicating the version check -- and so a capability that path does not have can be refused
    before anything is built.
    """
    path = Path(model_dir) / "session_contract.yaml"
    if not path.is_file():
        return False
    from ..common.yaml import load_yaml
    value = load_yaml(path)
    return isinstance(value, dict) and int(value.get("version", 0) or 0) == 2


@_whole_cell_deadline
def run_on_k1(model_dir: str | Path, work: str | Path, pkg, *, timeout: int = 600,
              kernel_backend: str | None = None, dispatch_timing: bool = False,
              op_profile: bool = False, force_scalar: bool = False,
              parallel_harts: int | None = None, fallback_policy: str = "allow",
              require_csr_vlen: bool = False,
              iters: int = 1, warmup: int = 0,
              session_repeats: int | None = None,
              dump_cap: int | None = 4096,
              max_session_steps: int | None = None,
              ours_mr: int = 4, ours_pack_b: bool = False) -> dict[str, Any]:
    """Cross-compile the workload for K1, deploy over scp, run, and parse OUT/METRIC/DONE.

    Returns the ``zephyr_model._parse_console`` result dict augmented with ``vlen`` (the bit-width
    the board reports — vlenb*8, falling back to the :data:`VLEN` constant). The K1 ``cycles`` from
    ``rdcycle`` are real silicon cycles (``cycle_accurate`` in the runner's measurement record).

    ``kernel_backend="xnnpack"`` / ``"openblas"`` (default-off): route the f32 matmul dispatches
    to XNNPACK's / OpenBLAS's RVV GEMM ukernel. Surfaced as ``n_xnn_routed`` / ``n_openblas_routed``
    in the result. These are the vectorized path only — they do NOT fall back to scalar/omp (those
    would drop the routing and silently lie about what ran), so a build/run failure surfaces as an
    honest exception, not a fallback.

    ``dump_cap`` is the number of output elements the board prints, and therefore the CEILING on
    how much of the answer an accuracy verdict can cover -- the host gate scores the printed prefix
    against the leading elements of the reference, so a short console produces a real-looking
    cos/rel over a slice nobody asked about. ``dump_cap=None`` prints the model's whole output
    (derived from its own ``MERLIN_OUT_ELEMS``). The default is unchanged at 4096, so an existing
    caller gets a byte-identical binary; callers that grade accuracy should pass ``None`` and check
    ``comparison_complete`` on the gate. The value used is echoed back as ``dump_cap``.
    """
    from ..runtime.backends import zephyr_model as zm

    if fallback_policy not in ("allow", "forbid"):
        raise ValueError("fallback_policy must be 'allow' or 'forbid'")
    if parallel_harts is not None and int(parallel_harts) < 1:
        raise ValueError("parallel_harts must be positive when provided")

    if not K1_HOST:
        raise K1Error("MERLIN_K1_HOST unset — board unreachable")

    if op_profile and _is_multi_program(model_dir):
        # FAIL CLOSED. The multi-program session build path does not take an ``op_profile``
        # parameter, so asking for a per-op profile of a version-2 session builds the
        # UN-instrumented binary, writes no ``opprof_table.json``, and returns a result with no
        # ``op_profile`` key. Downstream that reads as an empty op table, and the profiler reports a
        # breakdown of a model with no ops -- "nothing was measured" wearing the clothes of "there
        # was nothing to measure". It also costs a full build and a board slot to produce that.
        raise K1Error(
            "op_profile=True is not supported for a version-2 multi-program session: the session "
            "build path does not instrument the IR, so the run would produce no PROF lines. "
            "Profile each program's bundle (the `stages/<name>` directories) on its own instead.")

    requested_mode = ("scalar" if force_scalar else
                      "rvv_openmp" if parallel_harts else "rvv")
    session_path = Path(model_dir) / "session_contract.yaml"
    multi_program = False
    if session_path.is_file():
        from ..common.yaml import load_yaml
        session_value = load_yaml(session_path)
        multi_program = isinstance(session_value, dict) and int(session_value.get("version", 0)) == 2

    def _build_deploy_run(mode: str, tag: str) -> dict:
        # mode: "rvv" vectorized (fixed-width RVV); "omp" scalar int8 + OpenMP across 8 cores;
        # "scalar" single-core scalar int8 (last-resort correctness).
        bwork = Path(work) / tag
        # the cross-compile is CPU-local (no board) — do it OUTSIDE the board lock so concurrent
        # forks still build in parallel; only the deploy+run below serializes on the physical board.
        if multi_program:
            if mode == "omp":
                raise K1Error("multi-program scalar OpenMP fallback is not implemented")
            binary = build_k1_session_binary(
                model_dir, bwork, pkg, force_scalar=(mode == "scalar"),
                parallel_harts=(parallel_harts if mode == "rvv_openmp" else None),
                fallback_policy=fallback_policy, kernel_backend=kernel_backend,
                dump_cap=dump_cap)
        else:
            binary = build_k1_binary(
                model_dir, bwork, pkg, force_scalar=(mode == "scalar"), parallel=(mode == "omp"),
                parallel_harts=(parallel_harts if mode == "rvv_openmp" else None),
                fallback_policy=fallback_policy, kernel_backend=kernel_backend,
                dispatch_timing=dispatch_timing, op_profile=op_profile, dump_cap=dump_cap,
                max_session_steps=max_session_steps,
                ours_mr=ours_mr, ours_pack_b=ours_pack_b)
        with board_lock():
            result = _deploy_run(mode, tag, bwork, binary)
        result["local_binary"] = str(binary)
        # RECORD the ceiling the console was produced under. `_gate` already reports
        # n_compared/n_reference/compared_fraction/comparison_complete from the arrays, but those
        # say only THAT a run was short, never why. Without this a truncated verdict and a complete
        # one are indistinguishable in the artifact after the fact.
        result["dump_cap"] = None if dump_cap is None else int(dump_cap)
        result["output_complete"] = dump_cap is None or len(
            result.get("outputs", ())) < int(dump_cap)
        result["memory_policy"] = (
            "mmap" if ((bwork / "USE_MMAP_WEIGHTS").is_file()
                        or (bwork / "USE_MMAP_WEIGHTS.json").is_file()) else "resident")
        result["execution_mode"] = mode
        result["requested_execution_mode"] = requested_mode
        result["fallback_used"] = mode != requested_mode
        requested_cores = (int(parallel_harts) if mode == "rvv_openmp" else
                           K1_OMP_THREADS if mode == "omp" else 1)
        # Do not infer the paper core count from OMP_NUM_THREADS. The generated harness reads the
        # process affinity mask on silicon, and the deploy command pins that mask explicitly.
        # A missing/incorrect observation therefore fails the paper cell instead of echoing intent.
        result["requested_core_count"] = requested_cores
        result["core_count"] = int(result.get("metrics", {}).get("affinity_cpus") or 0)
        result["affinity_source"] = "sched_getaffinity"
        return result

    def _deploy_run(mode: str, tag: str, bwork: Path, binary) -> dict:
        remote = _remote_binary_path(binary,
                                     f"{Path(model_dir).name}_{pkg.run_id}_{tag}_merlin_k1")
        if remote.startswith(K1_REMOTE_DIR):
            _ssh(f"mkdir -p {K1_REMOTE_DIR}", timeout=30)
        _run(["scp", "-i", K1_SSH_KEY, *_SCP_PORT_OPTS, "-o", "BatchMode=yes",
              "-o", "StrictHostKeyChecking=no", str(binary), f"{K1_HOST}:{remote}"])
        # Big-model mmap path: build_k1_binary left a marker with the weights.bin to deploy
        # alongside; the binary mmaps it (resident RAM = working set, not the whole blob).
        marker = bwork / "USE_MMAP_WEIGHTS"
        wenv, remote_w = "", None
        remote_weights: list[str] = []
        multi_marker = bwork / "USE_MMAP_WEIGHTS.json"
        if multi_marker.is_file():
            _ssh(f"mkdir -p {K1_REMOTE_DIR}", timeout=30)
            by_digest: dict[str, str] = {}
            for item in json.loads(multi_marker.read_text(encoding="utf-8")):
                digest, program = str(item["sha256"]), int(item["program"])
                remote_stage = by_digest.get(digest)
                if remote_stage is None:
                    remote_stage = f"{K1_REMOTE_DIR}/{Path(remote).name}.weights.{digest[:16]}.bin"
                    _run(["scp", "-i", K1_SSH_KEY, *_SCP_PORT_OPTS, "-o", "BatchMode=yes",
                          "-o", "StrictHostKeyChecking=no", str(item["path"]),
                          f"{K1_HOST}:{remote_stage}"])
                    by_digest[digest] = remote_stage
                    remote_weights.append(remote_stage)
                wenv += f"MERLIN_WEIGHTS_{program}={remote_stage} "
        elif marker.is_file():
            # weights to the rootfs (real flash), NOT /tmp (tmpfs/RAM) — see K1_REMOTE_DIR.
            _ssh(f"mkdir -p {K1_REMOTE_DIR}", timeout=30)
            remote_w = f"{K1_REMOTE_DIR}/{Path(remote).name}.weights.bin"
            _run(["scp", "-i", K1_SSH_KEY, *_SCP_PORT_OPTS, "-o", "BatchMode=yes",
                  "-o", "StrictHostKeyChecking=no", marker.read_text().strip(),
                  f"{K1_HOST}:{remote_w}"])
            wenv = f"MERLIN_WEIGHTS={remote_w} "
        # Fan the OpenMP loops across the board's cores (spread over the two clusters).
        thread_count = int(parallel_harts) if mode == "rvv_openmp" else K1_OMP_THREADS
        env = (f"OMP_NUM_THREADS={thread_count} OMP_PROC_BIND=spread "
               if mode in ("omp", "rvv_openmp") else "")
        affinity_count = thread_count if mode in ("omp", "rvv_openmp") else 1
        # PIN ONLY WHEN PINNING IS THE POINT. `taskset` here exists to bound a MULTICORE run to a
        # known core count. A single-threaded run does not need it, and pinning one to cpu0
        # specifically is actively wrong for a comparison: cpu0 carries this board's interrupt
        # load, and pinning there measured lstmnetvit at 452 ms against 89.5 ms unpinned -- a 5.07x
        # penalty from the mask alone, on a BYTE-IDENTICAL binary (model.ll, model.o and the linked
        # ELF all unchanged; only the launch command differed).
        #
        # It is also an ASYMMETRY, which is the part that makes a ratio wrong rather than merely
        # slow: the ExecuTorch arm takes `--cpu_threads=1` and is never tasksetted, so its one
        # thread migrates freely over all 8 cores. Pinning ours and not theirs compares a pinned
        # arm against a floating one. Single-threaded therefore gets the same treatment they get --
        # one thread, unpinned -- and the mask is recorded either way so the choice is auditable
        # from the artifact instead of having to be re-derived from a 5x anomaly.
        affinity_list = None if affinity_count == 1 else f"0-{affinity_count - 1}"
        taskset = "" if affinity_list is None else f"taskset -c {affinity_list} "
        # Sustained mode: warmup + N timed passes against the same buffers. Only emitted when
        # asked for, so the default console stays byte-identical to the single-shot path.
        if session_repeats is not None:
            if int(session_repeats) < 1:
                raise ValueError("session_repeats must be positive when provided")
            env += (f"MERLIN_SESSION_REPEATS={int(session_repeats)} "
                    f"MERLIN_SESSION_WARMUPS={int(warmup)} ")
        elif int(iters) > 1 or int(warmup) > 0:
            env += f"MERLIN_ITERS={int(iters)} MERLIN_WARMUP={int(warmup)} "
        try:
            _ssh(f"chmod +x {remote}", timeout=30)
            conditions_before = board_conditions()
            proc = _ssh(f"{wenv}{env}{taskset}{remote}", timeout=timeout)
            r = zm._parse_console(proc.stdout + proc.stderr, proc.returncode)
            conditions_after = board_conditions()
            r["board_conditions"] = {"before": conditions_before, "after": conditions_after}
            # The mask this wall was produced under. Recorded even when absent (None = unpinned),
            # because "no taskset" and "pinned to cpu0" differ by 5x here and a bare number cannot
            # tell them apart after the fact.
            r["affinity_mask"] = affinity_list
            if (bwork / "HAS_SESSION_QUALITY").is_file():
                quality_env = ("MERLIN_VALIDATE_SESSION=1 MERLIN_SESSION_REPEATS=1 "
                               "MERLIN_SESSION_WARMUPS=0 ")
                qproc = _ssh(
                    f"{wenv}{env}{quality_env}{taskset}{remote}",
                    timeout=timeout)
                qres = zm._parse_console(qproc.stdout + qproc.stderr, qproc.returncode)
                qm = qres.get("metrics", {})
                steps = int(qm.get("trajectory_steps") or 0)
                r["trajectory_quality"] = {
                    "scope": "trajectory",
                    "steps": steps,
                    "min_cosine": ((qm.get("trajectory_min_cos_ppm") or 0) / 1_000_000.0),
                    "max_relative_error": ((qm.get("trajectory_max_rel_ppm") or 0) / 1_000_000.0),
                    "top1_matches": int(qm.get("trajectory_top1_matches") or 0),
                    "top1_agreement": ((int(qm.get("trajectory_top1_matches") or 0) / steps)
                                       if steps else None),
                    "validation_console": qres.get("console", ""),
                }
                correctness_steps = int(qm.get("correctness_trajectory_steps") or 0)
                r["trajectory_correctness"] = {
                    "scope": "trajectory",
                    "steps": correctness_steps,
                    "min_cosine": (
                        (qm.get("correctness_trajectory_min_cos_ppm") or 0) / 1_000_000.0),
                    "max_relative_error": (
                        (qm.get("correctness_trajectory_max_rel_ppm") or 0) / 1_000_000.0),
                    "top1_matches": int(
                        qm.get("correctness_trajectory_top1_matches") or 0),
                    "top1_agreement": (
                        int(qm.get("correctness_trajectory_top1_matches") or 0)
                        / correctness_steps if correctness_steps else None),
                    "validation_console": qres.get("console", ""),
                }
            nx = bwork / "N_XNN_ROUTED"
            if nx.is_file():
                r["n_xnn_routed"] = int(nx.read_text().strip())
                r["n_xnn_eligible"] = int((bwork / "N_XNN_ELIGIBLE").read_text().strip())
                r["n_xnn_candidates"] = int((bwork / "N_XNN_CANDIDATES").read_text().strip())
            no = bwork / "N_OPENBLAS_ROUTED"
            if no.is_file():
                r["n_openblas_routed"] = int(no.read_text().strip())
                r["n_openblas_eligible"] = int(
                    (bwork / "N_OPENBLAS_ELIGIBLE").read_text().strip())
                r["n_openblas_candidates"] = int(
                    (bwork / "N_OPENBLAS_CANDIDATES").read_text().strip())
            nu = bwork / "N_OURS_ROUTED"
            if nu.is_file():
                r["n_ours_routed"] = int(nu.read_text().strip())
            # Per-op profile: join the board's `PROF <id> <ticks> <hits>` lines against the
            # id->op table the build emitted. Absent (or empty) unless op_profile=True.
            tbl = bwork / "opprof_table.json"
            if op_profile and tbl.is_file():
                import json as _json

                from ..llvmlower.op_profile import parse_prof_lines
                table = _json.loads(tbl.read_text())
                ticks = parse_prof_lines(r["console"])
                for rec in table:
                    t, h = ticks.get(rec["id"], (0, 0))
                    rec["ticks"], rec["hits"] = t, h
                r["op_profile"] = table
            return r
        finally:
            try:
                cleanup = ([remote_w] if remote_w else []) + remote_weights
                _ssh(f"rm -f {remote}" + (" " + " ".join(cleanup) if cleanup else ""), timeout=30)
            except Exception:  # noqa: BLE001
                pass

    if force_scalar:
        # Single-core scalar int8: the SAME lowering minus vectorization. Its only use is as a
        # differential oracle — when a vectorized board run disagrees with the host, running the
        # scalar build on the same silicon says whether the fault is in the RVV codegen or in
        # everything else (harness, toolchain, arg table, board). Slow; never a perf path.
        res = _build_deploy_run("scalar", "scalar")
    elif parallel_harts:
        # Exact vector+multicore mode. A paper run must never silently turn this into scalar OpenMP
        # or single-core scalar; the caller selects fallback_policy='forbid' for that contract.
        res = _build_deploy_run("rvv_openmp", "rvv_openmp")
    elif kernel_backend in ("xnnpack", "openblas", "ours"):
        # XNNPACK/OpenBLAS/ours are the vectorized f32 routed paths; NO scalar/omp fallback (those
        # drop the routing and would silently misreport what ran). Any failure is an honest exception.
        res = _build_deploy_run("rvv", "rvv")
    else:
        try:
            res = _build_deploy_run("rvv", "rvv")
        except zm.ZephyrModelError:
            if fallback_policy == "forbid":
                raise
            # The vectorized int8 lowering OOB-stores / crashes for some shapes (e.g. rdt's
            # diffusion transformer) while smaller models vectorize fine. Retry with the MULTICORE
            # lowering (scalar int8 datapath + OpenMP parallel loops) — correct AND fast enough to
            # finish on the 8-core board, where single-core scalar times out. Final fall-back:
            # single-core scalar.
            try:
                res = _build_deploy_run("omp", "omp")
            except (zm.ZephyrModelError, K1Error):
                res = _build_deploy_run("scalar", "scalar")

    # Prefer the vlenb the harness printed (=== merlin_k1 vlenb=N ===); else probe; else constant.
    vlen_bits = VLEN
    vlen_source = "declared"
    for line in res["console"].splitlines():
        if "vlenb=" in line:
            try:
                vlen_bits = int(line.split("vlenb=")[1].split()[0]) * 8
                vlen_source = "csr"
            except (IndexError, ValueError):
                pass
            break
    if vlen_source == "declared":
        probed = board_vlenb()
        if probed is not None:
            vlen_bits = int(probed) * 8
            vlen_source = "procfs"
    if require_csr_vlen and vlen_source != "csr":
        raise K1Error("paper run requires vlenb read from the harness CSR; no CSR value was parsed")
    if vlen_source in ("csr", "procfs"):
        verify_vlen(vlen_bits // 8)
    res["vlen"] = vlen_bits
    res["vlen_source"] = vlen_source
    return res
