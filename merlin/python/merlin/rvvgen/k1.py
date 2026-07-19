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
import os
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path
from merlin.common.paths import env, repo_root, runtime_dir
from typing import Any

# Board access — set both via env (no personal defaults committed). The board IP is a DHCP lease.
# Read through paths.env (os.environ -> .env -> default) so a repo-local .env configures the board +
# toolchain without exporting into the shell (the same .env-everywhere contract as spike/zephyr).
K1_SSH_KEY = env("MERLIN_K1_SSH_KEY", "")  # path to the SSH private key; empty => unset
K1_HOST = env("MERLIN_K1_HOST", "")  # e.g. root@<board-ip>; empty => unset/unreachable
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
VLEN = 256  # K1 X60 vector length, bits; the runtime reads vlenb at run time and records it.


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

def main_linux_c(dump_cap: int = 4096, mmap_weights: bool = False,
                 dispatch_timing: bool = False) -> str:
    """Generate the K1 Linux ``main()``: build the memref descriptors from the generated arg
    table, call ``merlin_run`` once, time it with the ``rdcycle`` CSR, and print the SAME
    OUT/ARGMAX/SUM/METRIC/DONE markers the spike harness uses so ``zephyr_model._parse_console``
    consumes the console unchanged. Glibc-hosted: stdio, malloc, no HTIF/medany.

    ``mmap_weights=True``: instead of embedding the weight blob in the binary (``ld -r -b
    binary`` -> ``_binary_weights_bin_start``), ``mmap`` the blob read-only from a file path
    (``MERLIN_WEIGHTS`` env or argv[1]). The OS then demand-pages the weights, so resident RAM
    is the working set, not the whole blob — required for the big VLAs (groot 1.8G, pi05 ~5G)
    whose embedded binary won't fit /tmp and whose all-resident weights exceed the 3.4G board."""
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
    return f"""/* Generated by merlin.rvvgen.k1 — K1 (SpacemiT) Linux RVV runtime harness. Do not edit. */
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <sys/resource.h>
#include <pthread.h>
{mmap_includes}
#include "merlin_model.h"
#include "model_gen.h"
#include "model_io.h"

{weights_decl}
{dtiming_decl}#define MERLIN_DUMP_CAP {dump_cap}
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

static void *worker(void *arg) {{
  (void)arg;
  printf("=== merlin_k1 vlenb=%llu ===\\n", (unsigned long long)rd_vlenb());
  uint64_t w0 = wall_ns();
  uint64_t t0 = rd_time();
  merlin_run(MERLIN_ARGS, MERLIN_N_ARGS, MERLIN_WEIGHTS_PTR,
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
{dtiming_print}  printf("DONE\\n");
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


# ---- build / deploy / run -----------------------------------------------------------

# A bounded wall clock for the K1 cross-compile/link/scp steps. A pathological schedule (e.g. the
# outer-product contraction feature at a large square regime) makes clang-23 spin for many minutes on
# one object file, hanging a serial beam; time it out so the fork fails-closed as a K1 build error the
# certify ladder records. Board-run (_ssh) already carries its own timeout. Override/disable (0) with
# MERLIN_COMPILE_TIMEOUT_S — the same knob the host/spike codegen path honors.
_K1_CMD_TIMEOUT_S = int(os.environ.get("MERLIN_COMPILE_TIMEOUT_S", "300") or "0")


def _run(cmd: list, **kw) -> subprocess.CompletedProcess:
    kw.setdefault("timeout", _K1_CMD_TIMEOUT_S or None)
    try:
        proc = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, **kw)
    except subprocess.TimeoutExpired:
        raise K1Error(f"command timed out after {_K1_CMD_TIMEOUT_S}s (pathological compile): "
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
                    mmap_weights: bool | None = None,
                    kernel_backend: str | None = None,
                    dispatch_timing: bool = False,
                    ours_mr: int = 4, ours_pack_b: bool = False) -> Path:
    """Cross-compile a K1 Linux RVV binary from the workload + RVV package.

    Reuses the EXACT spike/Zephyr lowering (``zephyr_model._prepare_model_mlir`` ->
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
    # Per-dispatch matmul-bucket timing (default-OFF) is only meaningful with a routed
    # kernel_backend (the timed region lives in the GEMM shim). Guard FIRST so it fails loud
    # before any toolchain/model work, and never silently no-op's into an always-zero bucket.
    if dispatch_timing and kernel_backend not in ("xnnpack", "openblas", "ours"):
        raise K1Error("dispatch_timing=True requires kernel_backend in {xnnpack, openblas, ours} "
                      "(the matmul-bucket timer lives in the routed GEMM shim)")
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
    # XNNPACK kernel-backend (default-off, additive): rewrite the routable f32 linalg.matmul ops
    # in the PREPARED MLIR to external calls (@merlin_xnn_gemm_f32). Everything else lowers
    # unchanged. n_xnn_routed records the count; the shim .o is built + linked below.
    _dt = ["-DMERLIN_DISPATCH_TIMING"] if dispatch_timing else []
    xnn_obj = None
    n_xnn_routed = 0
    if kernel_backend == "xnnpack":
        from ..runtime.backends import xnnpack_board as xb

        if not xb.is_available():
            raise K1Error("kernel_backend='xnnpack' but the XNNPACK RVV ukernel/shim is unavailable "
                          "(see MERLIN_XNNPACK_REPO / tmp/kernels/XNNPACK)")
        if pkg.is_int8:
            raise K1Error("kernel_backend='xnnpack' is an f32 GEMM path; int8 datapath not supported")
        rewritten, n_xnn_routed = xb.rewrite_matmuls_to_xnn(prepared.read_text())
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
    if kernel_backend == "openblas":
        from ..runtime.backends import openblas_board as ob

        if not ob.is_available():
            raise K1Error("kernel_backend='openblas' but the OpenBLAS RVV ukernel/shim is "
                          "unavailable (see MERLIN_OPENBLAS_REPO / tmp/kernels/OpenBLAS)")
        if pkg.is_int8:
            raise K1Error("kernel_backend='openblas' is an f32 GEMM path; int8 datapath not supported")
        rewritten, n_openblas_routed = ob.rewrite_matmuls_to_openblas(prepared.read_text())
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
    # hoist_static_allocs=False: keep big intermediate buffers on the HEAP (board RAM) instead of
    # promoting them to stack alloca — large models otherwise overflow even a multi-GB stack.
    from ..llvmlower.pipeline import PipelineError
    # impr-fork compiler features (PASS/HEURISTIC/PATTERN). Without threading these the K1 build
    # silently used the BASELINE codegen for any feature-bearing fork — invalidating its
    # benchmark. Pass through so the K1 binary matches what spike certified.
    feats = frozenset(getattr(pkg, "compiler_features", []) or []) or None
    if force_scalar is None:
        force_scalar = bool(os.environ.get("MERLIN_K1_FORCE_SCALAR"))
    if parallel:
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
            res = lower_model_file(prepared, work / "lower_scalar", targets=(), textual=True,
                                   vectorize=False, hoist_static_allocs=False)

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
    main_c.write_text(main_linux_c(mmap_weights=mmap_weights, dispatch_timing=dispatch_timing))

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
    if not mmap_weights:
        base += [str(weights_o)]
    if parallel:
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
    if kernel_backend == "openblas":
        (work / "N_OPENBLAS_ROUTED").write_text(str(n_openblas_routed))
    if kernel_backend == "ours":
        (work / "N_OURS_ROUTED").write_text(str(n_ours_routed))
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


def run_on_k1(model_dir: str | Path, work: str | Path, pkg, *, timeout: int = 600,
              kernel_backend: str | None = None, dispatch_timing: bool = False,
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
    """
    from ..runtime.backends import zephyr_model as zm

    if not K1_HOST:
        raise K1Error("MERLIN_K1_HOST unset — board unreachable")

    def _build_deploy_run(mode: str, tag: str) -> dict:
        # mode: "v" vectorized (fixed-width RVV); "omp" scalar int8 + OpenMP across 8 cores;
        # "scalar" single-core scalar int8 (last-resort correctness).
        bwork = Path(work) / tag
        # the cross-compile is CPU-local (no board) — do it OUTSIDE the board lock so concurrent
        # forks still build in parallel; only the deploy+run below serializes on the physical board.
        binary = build_k1_binary(model_dir, bwork, pkg,
                                 force_scalar=(mode == "scalar"), parallel=(mode == "omp"),
                                 kernel_backend=kernel_backend, dispatch_timing=dispatch_timing,
                                 ours_mr=ours_mr)
        with board_lock():
            return _deploy_run(mode, tag, bwork, binary)

    def _deploy_run(mode: str, tag: str, bwork: Path, binary) -> dict:
        remote = f"/tmp/{Path(model_dir).name}_{pkg.run_id}_{tag}_merlin_k1"
        _run(["scp", "-i", K1_SSH_KEY, "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no",
              str(binary), f"{K1_HOST}:{remote}"])
        # Big-model mmap path: build_k1_binary left a marker with the weights.bin to deploy
        # alongside; the binary mmaps it (resident RAM = working set, not the whole blob).
        marker = bwork / "USE_MMAP_WEIGHTS"
        wenv, remote_w = "", None
        if marker.is_file():
            # weights to the rootfs (real flash), NOT /tmp (tmpfs/RAM) — see K1_REMOTE_DIR.
            _ssh(f"mkdir -p {K1_REMOTE_DIR}", timeout=30)
            remote_w = f"{K1_REMOTE_DIR}/{Path(remote).name}.weights.bin"
            _run(["scp", "-i", K1_SSH_KEY, "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no",
                  marker.read_text().strip(), f"{K1_HOST}:{remote_w}"])
            wenv = f"MERLIN_WEIGHTS={remote_w} "
        # Fan the OpenMP loops across the board's cores (spread over the two clusters).
        env = (f"OMP_NUM_THREADS={K1_OMP_THREADS} OMP_PROC_BIND=spread "
               if mode == "omp" else "")
        try:
            _ssh(f"chmod +x {remote}", timeout=30)
            proc = _ssh(f"{wenv}{env}{remote}", timeout=timeout)
            r = zm._parse_console(proc.stdout + proc.stderr, proc.returncode)
            nx = bwork / "N_XNN_ROUTED"
            if nx.is_file():
                r["n_xnn_routed"] = int(nx.read_text().strip())
            no = bwork / "N_OPENBLAS_ROUTED"
            if no.is_file():
                r["n_openblas_routed"] = int(no.read_text().strip())
            nu = bwork / "N_OURS_ROUTED"
            if nu.is_file():
                r["n_ours_routed"] = int(nu.read_text().strip())
            return r
        finally:
            try:
                _ssh(f"rm -f {remote}" + (f" {remote_w}" if remote_w else ""), timeout=30)
            except Exception:  # noqa: BLE001
                pass

    if kernel_backend in ("xnnpack", "openblas", "ours"):
        # XNNPACK/OpenBLAS/ours are the vectorized f32 routed paths; NO scalar/omp fallback (those
        # drop the routing and would silently misreport what ran). Any failure is an honest exception.
        res = _build_deploy_run("v", "v")
    else:
        try:
            res = _build_deploy_run("v", "v")
        except zm.ZephyrModelError:
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
    for line in res["console"].splitlines():
        if "vlenb=" in line:
            try:
                vlen_bits = int(line.split("vlenb=")[1].split()[0]) * 8
            except (IndexError, ValueError):
                pass
            break
    res["vlen"] = vlen_bits
    return res
