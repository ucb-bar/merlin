"""Remote Linux board adapter: cross-compile for a RISC-V Linux board, deploy over SSH, run, pull outputs.

A board whose catalog entry says ``flow: linux`` is real RVV silicon running a glibc userspace. It has
no HTIF or bare-metal console, so it needs a cross-compiled Linux runtime that reports the same
OUT/METRIC/DONE markers ``zephyr_model._parse_console`` reads. Every fact about the board -- its
``-march``/``-mabi``/target triple, vector length, timebase and core clock, model-thread stack, remote
staging directories and their limits, OpenMP binding -- is a field of its catalog entry
(:class:`merlin.runtime.boards.Board`). Host-specific access -- SSH key, host, port and the
cross-toolchain install -- comes from per-board environment variables ``MERLIN_<BOARD>_<WHAT>``
(:func:`env_name`), read through ``paths.env`` so a repo-local ``.env`` configures them.

This module is intentionally fail-closed: when the toolchain or the board is unavailable,
:func:`available` returns False and the runner records the board rung as ``not_run`` (NEVER a false
pass). The cross-compile + deploy path reuses the SAME portable artifacts as the spike build -- the
LLVM IR ``model.ll`` (``llvmlower.lower``) and the data-driven C runtime (``llvmlower.c_runtime`` +
``src/merlin/runtime/native/c/merlin_model.c``); only the harness differs:

  * spike/Zephyr -- bare-metal: ``-mcmodel=medany -ffreestanding -nostdlib``, HTIF console, mcycle.
  * a Linux board -- glibc userspace: the board's ``-march``/``-mabi`` at ``-O2`` (NO
    medany/freestanding), a ``main()`` that prints OUT/METRIC/DONE, timed with the user-readable
    ``time`` CSR at the board's declared ``timebase_hz``.

Every public entry point takes the board explicitly; nothing here selects a board by default.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import shlex
import shutil
import subprocess
import tempfile
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any

from merlin.common import proc as _proc
from merlin.common import toolchain_locator as _toolchains
from merlin.common.paths import env as _env
from merlin.common.paths import repo_root, runtime_dir, target_env_name, tracked_build_dir

if TYPE_CHECKING:
    from merlin.runtime.boards import Board


class RemoteLinuxError(RuntimeError):
    """A build, deploy or run step for a remote Linux board failed."""


class VlenMismatch(RuntimeError):
    """The board reported a vector length different from the one codegen pinned."""


# ---- the board profile ---------------------------------------------------------------


def env_name(board: Board, what: str) -> str:
    """The per-board environment variable ``MERLIN_<BOARD>_<WHAT>`` (``paths.target_env_name``)."""
    return target_env_name(board.name, what)


def binary_name(board: Board) -> str:
    """The deployed binary's name, also the harness banner and the lock/probe file prefix."""
    return f"merlin_{board.name}"


def substrate(board: Board) -> str:
    """The measurement-substrate label this board's numbers are recorded under.

    Target contracts name the same label as a measurement authority (``measurement.wall_from``) and
    :func:`merlin.analysis.kernels.measurement.pick` matches measurements on it, so callers read it from the
    board instead of restating it. Defaults to the board name.
    """
    return _require_linux(board).substrate or board.name


def _require_linux(board: Board) -> Board:
    from merlin.runtime.boards import FLOW_LINUX

    if getattr(board, "flow", None) != FLOW_LINUX:
        raise RemoteLinuxError(f"board {getattr(board, 'name', board)!r} is not a `flow: {FLOW_LINUX}` board")
    return board


def resolve_board(name: str, catalog: str | Path | None = None) -> Board:
    """The ``flow: linux`` board ``name`` from ``catalog``, else from the selected board catalog."""
    from merlin.runtime import boards

    if catalog is None:
        selected = boards.board(name)
    else:
        found = boards.load_boards(catalog)
        if name not in found:
            raise boards.BoardRegistryError(f"board {name!r} is absent from {catalog}")
        selected = found[name]
    return _require_linux(selected)


@dataclass(frozen=True)
class Profile:
    """One ``flow: linux`` board with this host's settings for it resolved.

    ``march``, ``vlen``, ``omp_threads`` and ``remote_dir`` are catalog facts a host may override
    (``MERLIN_<BOARD>_MARCH`` / ``_VLEN`` / ``_OMP_THREADS`` / ``_REMOTE_DIR``); the access fields
    exist only on the host. Empty access fields read as "unset" and fail closed.
    """

    board: Board
    march: str
    vlen: int
    omp_threads: int
    remote_dir: str
    host: str
    ssh_key: str
    ssh_port: str
    toolchain_install: Path | None

    @property
    def name(self) -> str:
        return self.board.name

    @property
    def binary(self) -> str:
        return binary_name(self.board)

    def env(self, what: str) -> str:
        return env_name(self.board, what)

    @property
    def triple(self) -> str:
        return str(self.board.triple)

    @property
    def target_flag(self) -> str:
        return f"--target={self.triple}"

    @property
    def ssh_port_opts(self) -> list[str]:
        return ["-p", self.ssh_port] if self.ssh_port else []  # ssh uses -p

    @property
    def scp_port_opts(self) -> list[str]:
        return ["-P", self.ssh_port] if self.ssh_port else []  # scp uses -P (capital)

    def scratch_path(self, name: str) -> str:
        return f"{str(self.board.scratch_dir).rstrip('/')}/{name}"


def profile(board: Board | Profile) -> Profile:
    """Resolve ``board``'s profile against this host's environment (process env, then ``.env``)."""
    if isinstance(board, Profile):
        return board
    _require_linux(board)
    toolchain_var = env_name(board, "TOOLCHAIN")
    # The toolchain's layout -- the vendor's extracted-release directory naming, and the install used
    # when the variable is unset -- is declared by the owning target contract under
    # ``runtime.toolchain`` (keyed by the variable) and read through merlin.common.toolchain_locator.
    install = _env(toolchain_var, "") or _toolchains.default_install(toolchain_var)
    return Profile(
        board=board,
        march=_env(env_name(board, "MARCH"), board.march),
        vlen=int(_env(env_name(board, "VLEN"), str(board.vlen))),
        omp_threads=int(_env(env_name(board, "OMP_THREADS"), str(board.harts))),
        remote_dir=_env(env_name(board, "REMOTE_DIR"), board.remote_dir),
        host=_env(env_name(board, "HOST"), "") or "",  # e.g. root@<board-ip>; empty => unreachable
        ssh_key=_env(env_name(board, "SSH_KEY"), "") or "",  # path to the SSH private key
        # Optional non-standard SSH port, for a network that filters inbound :22 to the board's
        # segment while passing high ports. Empty => the default port 22.
        ssh_port=_env(env_name(board, "SSH_PORT"), "") or "",
        toolchain_install=Path(install) if install else None,
    )


def verify_vlen(board: Board | Profile, reported_vlenb: int, vlen: int | None = None) -> None:
    """Fail closed when the board's actual ``vlenb`` contradicts the VLEN we compiled against.

    ``vlenb`` is VLEN in BYTES and the harness banner already prints it, so the true value is
    observable on every run -- there is no reason to keep trusting a constant. Pinning a VLEN larger
    than the hardware's is not a lost optimization but a MISCOMPILE (the backend sizes register
    groups for lanes that do not exist), so a mismatch must stop the run rather than be measured."""
    p = profile(board)
    want = int(vlen or p.vlen)
    got = int(reported_vlenb) * 8
    if got != want:
        raise VlenMismatch(
            f"board reports VLEN={got} bits (vlenb={reported_vlenb}) but codegen pinned {want}. "
            f"Set {p.env('VLEN')}={got} for this board, or drop the pin. Pinning a VLEN the hardware "
            f"lacks miscompiles; pinning a smaller one silently idles the datapath."
        )


def codegen_march(board: Board | Profile, march: str | None = None, vlen: int | None = None) -> str:
    """``march`` with the board's REAL vector length pinned (``_zvl<VLEN>b``).

    Why this is not cosmetic. ``-march=rv64gcv`` only promises the RVV *minimum* VLEN of 128 bits
    (``zvl128b``), so when our codegen emits a FIXED-width vector the LLVM backend must size the
    register group for that worst case: on a VLEN=256 core a ``vector<16xf32>`` (512 bits) becomes
    ``e32,m4`` and a ``vector<32xf32>`` becomes ``e32,m8`` — exactly DOUBLE the LMUL the hardware
    needs. Two costs follow, and together they were the whole "vector width vs IPC" tradeoff
    (measured, f32 GEMM 128^3):

      * REGISTER PRESSURE doubles. An ``m8`` value occupies 8 of the 32 architectural vector
        registers, so an MR=4/NR=32 register block wants 32 registers for accumulators alone and the
        allocator spills inside the K loop (objdump: 25 inner-loop instructions with 4 ``vl8r.v``/
        ``vs8r.v``, vs 12 and none once VLEN is pinned). MR=8/NR=32 went 92 -> 46.
      * HALF THE DATAPATH IS IDLE. ``vsetivli zero, 16, e32, m4`` on a VLEN=256 core sets ``vl=16``
        against a ``VLMAX`` of 32 — the instruction reserves a 4-register group and uses half of it.

    An expert kernel never hits this because it sizes to the vector length it *queries* at run time
    (XNNPACK's ``__riscv_vsetvl_e32m4``), which is why its advantage looked like "lanes per issue".
    Pinning the known target VLEN recovers the same lanes-per-issue by CODE GENERATION.

    NOTE ON GENERALITY: pinning a known VLEN is a STOPGAP that trades portability for lanes. The
    target-agnostic fix is VL-agnostic codegen -- `MicrokernelSpec.vl_strategy = VL_DYNAMIC`, a
    vsetvli loop that sizes to whatever VLEN the hardware reports at run time. Until that axis is
    emitted, this pin recovers the same lanes for boards whose VLEN we can state and verify
    (:func:`verify_vlen`).

    Kept separate from the board's plain ``march`` (which the C glue and the cross-framework baseline
    arms compile against) so pinning VLEN for our own codegen never silently re-flags a baseline
    mid-campaign.
    """
    p = profile(board)
    m = march or p.march
    v = int(vlen or p.vlen)
    return m if f"zvl{v}b" in m else f"{m}_zvl{v}b"


def _board_lock_path(p: Profile) -> Path:
    """A host-wide lockfile identifying THIS board. Keyed by the board's host so two different boards
    do not block each other, but every process on this machine (concurrent beams, other sessions)
    contends on the same file when hitting the same board."""
    tag = hashlib.sha1((p.host or "unset").encode()).hexdigest()[:12]
    return Path(tempfile.gettempdir()) / f"{p.binary}_board_{tag}.lock"


@contextmanager
def board_lock(board: Board | Profile, timeout: int = 1800):
    """Serialize physical-board access across ALL processes on this host (a board has one set of
    deploy paths and one set of cores; concurrent forks would corrupt each other's scp'd binaries and
    contend for cores, poisoning the measurement). A file ``flock`` is the cross-process lock that a
    ThreadPool max_workers=1 (single-beam only) and a Ray resource gate (single cluster only) both
    miss. ``MERLIN_<BOARD>_NO_BOARD_LOCK=1`` disables it (single-user runs)."""
    p = profile(board)
    if os.environ.get(p.env("NO_BOARD_LOCK")) == "1":
        yield
        return
    lockf = _board_lock_path(p)
    fh = open(lockf, "w")
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)  # blocks until the board is free
        yield
    finally:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        finally:
            fh.close()


def toolchain_root(board: Board | Profile) -> Path | None:
    """Resolve the directory that actually contains ``bin/clang``.

    Accepts either ``$MERLIN_<BOARD>_TOOLCHAIN`` pointing directly at a toolchain prefix
    (``<prefix>/bin/clang``) OR at a parent holding an extracted vendor release, whose directory
    naming the owning target contract declares (see :mod:`merlin.common.toolchain_locator`).
    """
    p = profile(board)
    if p.toolchain_install is None:
        return None
    return _toolchains.find_prefix(p.toolchain_install, p.env("TOOLCHAIN"), tools=("clang",))


def toolchain_cc(board: Board | Profile) -> Path | None:
    """Path to the board's linux-gnu cross clang/gcc, or None if the toolchain isn't installed."""
    p = profile(board)
    root = toolchain_root(p)
    if root is None:
        return None
    for rel in ("bin/clang", f"bin/{p.triple}-gcc"):
        c = root / rel
        if c.is_file():
            return c
    return None


def available(board: Board | Profile, *, deadline_ns: int | None = None) -> bool:
    """True only if BOTH the cross-toolchain is installed AND the board is reachable over SSH.

    Fail-closed: any uncertainty -> False -> runner records ``not_run`` (never a false pass).
    """
    p = profile(board)
    if toolchain_cc(p) is None or not p.host or not Path(p.ssh_key).is_file():
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
            [
                "ssh",
                "-i",
                p.ssh_key,
                *p.ssh_port_opts,
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=5",
                "-o",
                "StrictHostKeyChecking=no",
                p.host,
                "true",
            ],
            capture_output=True,
            timeout=timeout,
        )
        return r.returncode == 0
    except Exception:  # noqa: BLE001
        return False


# ---- generated Linux harness --------------------------------------------------------


def main_linux_c(
    board: Board | Profile,
    dump_cap: int | None = 4096,
    mmap_weights: bool = False,
    dispatch_timing: bool = False,
    op_profile: bool = False,
    persistent_openmp_pool: int | None = None,
) -> str:
    """Generate the board's Linux ``main()``: build the memref descriptors from the generated arg
    table, call ``merlin_run`` once, time it with the ``time`` CSR, and print the SAME
    OUT/ARGMAX/SUM/METRIC/DONE markers the spike harness uses so ``zephyr_model._parse_console``
    consumes the console unchanged. Glibc-hosted: stdio, malloc, no HTIF/medany.

    ``mmap_weights=True``: instead of embedding the weight blob in the binary (``ld -r -b
    binary`` -> ``_binary_weights_bin_start``), ``mmap`` the blob read-only from a file path
    (``MERLIN_WEIGHTS`` env or argv[1]). The OS then demand-pages the weights, so resident RAM
    is the working set, not the whole blob — required for the big VLAs (groot 1.8G, pi05 ~5G)
    whose embedded binary won't fit a RAM-backed scratch directory and whose all-resident weights
    can exceed the board's memory.

    ``dump_cap`` bounds how many output elements the console prints. The host gate compares the
    printed PREFIX against the leading elements of the reference, so this number is the ceiling on
    how much of the answer any accuracy verdict can actually cover: at the historical 4096 a
    256,000-element LM head was graded on 1.6% of its output, and the resulting cos read exactly
    like a whole-tensor score. ``dump_cap=None`` removes the ceiling by deriving it from the
    model's OWN generated ``MERLIN_OUT_ELEMS`` rather than a host-side literal, so the console
    carries the whole tensor and ``_gate`` reports ``comparison_complete``. The default stays 4096
    so an unchanged caller emits byte-identical C."""
    brd = profile(board).board
    timebase_hz, cpu_hz, stack_bytes, banner = brd.timebase_hz, brd.chip_freq_hz, brd.stack_bytes, binary_name(brd)
    title = brd.name.upper()
    weights_decl = (
        "/* weights mmap'd from a file at run time (see main). */\nstatic const void *MERLIN_WEIGHTS_PTR;\n"
        if mmap_weights
        else "/* weights.bin linked in as a binary blob (ld -r -b binary): glibc Linux is\n"
        " * small-code-model and fully relocatable, so address the blob via its symbol. */\n"
        "extern const unsigned char _binary_weights_bin_start[];\n"
        "static const void *MERLIN_WEIGHTS_PTR;\n"
    )
    mmap_includes = (
        "#include <fcntl.h>\n#include <sys/mman.h>\n#include <sys/stat.h>\n#include <unistd.h>\n#include <stdlib.h>\n"
        if mmap_weights
        else ""
    )
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
        "extern unsigned long long merlin_matmul_ticks(void);\nextern unsigned long long merlin_matmul_calls(void);\n"
        if dispatch_timing
        else ""
    )
    dtiming_print = (
        '  printf("METRIC matmul_ticks %llu\\n", (unsigned long long)merlin_matmul_ticks());\n'
        '  printf("METRIC matmul_calls %llu\\n", (unsigned long long)merlin_matmul_calls());\n'
        if dispatch_timing
        else ""
    )
    # Per-op whole-model profile (default-OFF). Emitted only when the IR was instrumented with
    # `@merlin_prof_mark` (llvmlower.op_profile) and the shim object is linked; with
    # op_profile=False the harness is byte-identical to the un-instrumented path.
    oprof_decl = "extern void merlin_prof_dump(void);\n" if op_profile else ""
    oprof_print = "  merlin_prof_dump();\n" if op_profile else ""
    if persistent_openmp_pool is not None and int(persistent_openmp_pool) < 1:
        raise ValueError("persistent_openmp_pool must be positive when provided")
    openmp_include = '#include "libomp_pthread.h"\n' if persistent_openmp_pool is not None else ""
    openmp_census = (
        f"""  int omp_threads = merlin_omp_init_from_env({int(persistent_openmp_pool)});
  printf("METRIC omp_threads %d\\n", omp_threads);
  printf("METRIC omp_worker_cpus");
  for (int tid = 0; tid < omp_threads; ++tid)
    printf(" %d", merlin_omp_worker_cpu(tid));
  printf("\\n");
"""
        if persistent_openmp_pool is not None
        else ""
    )
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
    return f"""/* Generated by merlin.runtime.remote_linux — {title} Linux RVV runtime harness. Do not edit. */
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
{openmp_include}

{weights_decl}
{dtiming_decl}{oprof_decl}#define MERLIN_DUMP_CAP {dump_cap_token}
/* Timebase->CPU frequency ratio for the estimated core-cycle count (see module doc): this
 * kernel traps userspace `rdcycle`, so we time with the delegated `rdtime` counter. */
#define MERLIN_TIMEBASE_HZ {timebase_hz}ULL
#define MERLIN_CPU_HZ {cpu_hz}ULL

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
  printf("=== {banner} vlenb=%llu ===\\n", (unsigned long long)rd_vlenb());
  cpu_set_t allowed;
  CPU_ZERO(&allowed);
  if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) {{
    fprintf(stderr, "FAIL sched_getaffinity\\n"); return NULL;
  }}
  printf("METRIC affinity_cpus %d\\n", CPU_COUNT(&allowed));
{openmp_census}

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
  const char *output_path = getenv("MERLIN_OUTPUT_FILE");
  if (output_path && *output_path) {{
    FILE *output_file = fopen(output_path, "wb");
    if (!output_file) {{ fprintf(stderr, "FAIL open output file %s\\n", output_path); return NULL; }}
    size_t written = fwrite(MERLIN_OUTPUT_PTR[0], sizeof(float), MERLIN_OUT_ELEMS, output_file);
    if (fclose(output_file) != 0 || written != MERLIN_OUT_ELEMS) {{
      fprintf(stderr, "FAIL write output file %s\\n", output_path); return NULL;
    }}
    /* Preserve the parser framing without streaming a multi-megabyte line through SSH. */
    printf("OUT 0\\n");
    printf("METRIC output_file_elems %d\\n", MERLIN_OUT_ELEMS);
  }} else {{
    printf("OUT %d", k);
    for (int i = 0; i < k; i++) {{
      uint32_t bits;
      memcpy(&bits, &((float*)MERLIN_OUTPUT_PTR[0])[i], 4);
      printf(" %u", (unsigned)bits);
    }}
    printf("\\n");
  }}

  if ((!output_path || !*output_path) && MERLIN_OUT_ELEMS > MERLIN_DUMP_CAP) {{
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
   * ground truth (the cycles value is an estimate — see MERLIN_CPU_HZ/MERLIN_TIMEBASE_HZ). */
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
   * (SIGSEGV) for VLAs. Run merlin_run on a pthread with an explicitly-mmap'd {stack_bytes}-byte
   * stack (one contiguous region — robust vs main-thread growth, which collides with lower mmaps).
   * Also raise RLIMIT_STACK as a fallback if pthread creation fails. */
  struct rlimit rl = {{ {stack_bytes}ULL, {stack_bytes}ULL }};
  setrlimit(RLIMIT_STACK, &rl);
  pthread_attr_t attr;
  pthread_t th;
  if (pthread_attr_init(&attr) == 0 &&
      pthread_attr_setstacksize(&attr, (size_t){stack_bytes}ULL) == 0 &&
      pthread_create(&th, &attr, worker, NULL) == 0) {{
    pthread_join(th, NULL);
  }} else {{
    worker(NULL);   /* fallback: run inline (relies on the raised RLIMIT) */
  }}
  return 0;
}}
"""


def main_linux_session_c(
    board: Board | Profile, dump_cap: int | None = 4096, persistent_openmp_pool: int | None = None
) -> str:
    """Board harness for a generated multi-program session scheduler.

    Every ``iter_wall_ns`` sample covers the full ordered stage graph from reset state. Per-stage
    totals are emitted as diagnostic attribution and are never substituted for that headline total.

    ``dump_cap`` bounds the printed quality output exactly as in :func:`main_linux_c`; ``None``
    prints all of it. The session's element count is a RUN-TIME value here (the scheduler reports
    it), not a compile-time macro, so the uncapped form drops the ceiling from the comparison
    rather than defining a macro over a local variable.
    """
    brd = profile(board).board
    timebase_hz, cpu_hz, stack_bytes, banner = brd.timebase_hz, brd.chip_freq_hz, brd.stack_bytes, binary_name(brd)
    title = brd.name.upper()
    if dump_cap is None:
        cap_define, k_expr = "", "(int)elems"
    else:
        if int(dump_cap) < 1:
            raise ValueError("dump_cap must be positive, or None for the full session output")
        cap_define = f"#define MERLIN_DUMP_CAP {int(dump_cap)}\n"
        k_expr = "elems < MERLIN_DUMP_CAP ? (int)elems : MERLIN_DUMP_CAP"
    if persistent_openmp_pool is not None and int(persistent_openmp_pool) < 1:
        raise ValueError("persistent_openmp_pool must be positive when provided")
    openmp_include = '#include "libomp_pthread.h"\n' if persistent_openmp_pool is not None else ""
    openmp_census = (
        f"""  int omp_threads = merlin_omp_init_from_env({int(persistent_openmp_pool)});
  printf("METRIC omp_threads %d\\n", omp_threads);
  printf("METRIC omp_worker_cpus");
  for (int tid = 0; tid < omp_threads; ++tid)
    printf(" %d", merlin_omp_worker_cpu(tid));
  printf("\\n");
"""
        if persistent_openmp_pool is not None
        else ""
    )
    return f"""/* Generated {title} multi-program continuous-session harness. */
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
{openmp_include}
{cap_define}#define MERLIN_TIMEBASE_HZ {timebase_hz}ULL
#define MERLIN_CPU_HZ {cpu_hz}ULL
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
  printf("=== {banner}_session vlenb=%llu ===\\n", (unsigned long long)rd_vlenb());
  cpu_set_t allowed;
  CPU_ZERO(&allowed);
  if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) {{
    fprintf(stderr, "FAIL sched_getaffinity\\n"); return 0;
  }}
  printf("METRIC affinity_cpus %d\\n", CPU_COUNT(&allowed));
{openmp_census}
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
  struct rlimit rl = {{ {stack_bytes}ULL, {stack_bytes}ULL }}; setrlimit(RLIMIT_STACK, &rl);
  pthread_attr_t attr; pthread_t thread;
  if (pthread_attr_init(&attr) == 0 &&
      pthread_attr_setstacksize(&attr, (size_t){stack_bytes}ULL) == 0 &&
      pthread_create(&thread, &attr, worker, 0) == 0) pthread_join(thread, 0);
  else worker(0);
  return 0;
}}
"""


# ---- build / deploy / run -----------------------------------------------------------

# A bounded wall clock for the cross-compile/link/scp steps. A pathological schedule (e.g. the
# outer-product contraction feature at a large square regime) makes clang spin for many minutes on
# one object file, hanging a serial beam; time it out so the fork fails-closed as a board build error
# the certify ladder records. Board-run (_ssh) already carries its own timeout. Override/disable (0)
# with MERLIN_COMPILE_TIMEOUT_S — the same knob the host/spike codegen path honors. Default unified at
# 900s across all four compile wrappers (was 300): a per-object 300s ceiling tripped legitimate
# whole-model builds (bitvla int8 fails even the baseline at 600s). For a whole-model beam launch
# set MERLIN_COMPILE_TIMEOUT_S=3600; 0 (or empty) disables the ceiling.
_CMD_TIMEOUT_S = int(os.environ.get("MERLIN_COMPILE_TIMEOUT_S", "900") or "0")
_ACTIVE_CELL_DEADLINE_NS: ContextVar[int | None] = ContextVar("merlin_remote_linux_cell_deadline_ns", default=None)


def _bounded_timeout(requested: int | float | None) -> int | float | None:
    deadline = _ACTIVE_CELL_DEADLINE_NS.get()
    if deadline is None:
        return requested
    remaining_ns = deadline - time.monotonic_ns()
    if remaining_ns <= 0:
        raise RemoteLinuxError("paper cell exhausted its whole-cell deadline")
    remaining = max(1, math.ceil(remaining_ns / 1_000_000_000))
    return remaining if requested is None else min(requested, remaining)


def _whole_cell_deadline(function):
    """Bind one optional monotonic deadline across build, deploy, perf, and validation calls."""

    @wraps(function)
    def wrapped(*args, deadline_ns: int | None = None, **kwargs):
        token = _ACTIVE_CELL_DEADLINE_NS.set(int(deadline_ns) if deadline_ns is not None else None)
        try:
            return function(*args, **kwargs)
        finally:
            _ACTIVE_CELL_DEADLINE_NS.reset(token)

    return wrapped


def _run(cmd: list, **kw) -> subprocess.CompletedProcess:
    timeout = _bounded_timeout(kw.pop("timeout", _CMD_TIMEOUT_S or None))
    return _proc.run_checked(
        cmd, error=RemoteLinuxError, timeout=timeout, timeout_hint=" within the bounded compile/cell deadline", **kw
    )


def _model_compile_flags(
    p: Profile, pkg, features, model_opt: list[str], *, honor_pkg_cflags: bool = True
) -> list[str]:
    """Compose package tuning flags under the board adapter's non-negotiable target identity.

    Packages commonly carry a generic ``-march=rv64gcv`` for portable runners. The board's model
    object needs the adapter's verified VLEN and the board's full extension set, so package march/ABI
    declarations are superseded while every actual optimization flag is preserved. This makes
    flags such as ``-fno-vectorize`` reach the compiler without allowing a package to erase the
    target contract.
    """
    base = [f"-march={codegen_march(p)}", f"-mabi={p.board.mabi}", *model_opt, "-Wno-override-module"]
    if honor_pkg_cflags:
        base = merge_package_cflags(base, getattr(pkg, "cflags", ()))
    from merlin.compiler.optimizations.registry import apply_cflags

    return apply_cflags(base, features or frozenset())


@dataclass(frozen=True)
class _OpenMPSelection:
    parallel_harts: int | None
    provider: str | None
    worker_pool_size: int | None


def _resolve_openmp_selection(*, parallel: bool, parallel_harts: int | None, host_effects=None) -> _OpenMPSelection:
    """Resolve the lowering width and the ABI provider materialized by this adapter."""
    if parallel_harts is not None and int(parallel_harts) < 1:
        raise ValueError("parallel_harts must be positive when provided")
    # A one-core budget is a serial lowering, even when a measurement harness
    # spells it explicitly as ``parallel_harts=1``.  Selecting OpenMP here used
    # to reach a transform schedule that correctly rejects widths below two.
    # Keep the legacy ``parallel=True`` switch strict because it explicitly asks
    # for a parallel lowering.
    if parallel_harts is not None and int(parallel_harts) == 1 and not parallel:
        parallel_harts = None
    if host_effects is None:
        active = bool(parallel or parallel_harts)
        return _OpenMPSelection(
            int(parallel_harts) if parallel_harts is not None else None,
            "llvm" if active else None,
            None,
        )

    from merlin.compiler.lower.model.host_policy import HostLoweringEffects

    if not isinstance(host_effects, HostLoweringEffects):
        raise TypeError("host_effects must be HostLoweringEffects or None")
    policy_harts = host_effects.parallel_harts
    if parallel_harts is not None and policy_harts != int(parallel_harts):
        raise RemoteLinuxError(
            f"explicit parallel_harts={parallel_harts} conflicts with host-policy parallel_harts={policy_harts}"
        )
    if parallel and policy_harts is not None:
        raise RemoteLinuxError("legacy scalar parallel=True conflicts with a host-policy RVV parallel lowering")
    harts = int(parallel_harts) if parallel_harts is not None else policy_harts
    provider = host_effects.openmp_runtime
    if harts is not None and provider is None:
        provider = "llvm"
    return _OpenMPSelection(harts, provider, host_effects.worker_pool_size)


def _require_worker_census(metrics: dict[str, Any], *, requested: int) -> tuple[int, ...]:
    """Fail closed unless the persistent pool observed the exact requested team."""
    observed_threads = int(metrics.get("omp_threads") or 0)
    if observed_threads != int(requested):
        raise RemoteLinuxError(f"persistent OpenMP census observed {observed_threads} threads; requested {requested}")
    raw = metrics.get("omp_worker_cpus")
    if isinstance(raw, int):
        cpus = (raw,)
    elif isinstance(raw, str):
        try:
            cpus = tuple(int(value) for value in raw.split())
        except ValueError as exc:
            raise RemoteLinuxError(f"invalid persistent OpenMP worker CPU census: {raw!r}") from exc
    else:
        cpus = ()
    if len(cpus) != observed_threads:
        raise RemoteLinuxError(
            f"persistent OpenMP worker CPU census has {len(cpus)} entries; expected {observed_threads}"
        )
    if any(cpu < 0 for cpu in cpus):
        raise RemoteLinuxError(f"persistent OpenMP worker CPU census contains invalid CPU: {cpus}")
    if len(set(cpus)) != len(cpus):
        raise RemoteLinuxError(f"persistent OpenMP worker CPU census contains duplicate CPUs: {cpus}")
    return cpus


def _openmp_library(p: Profile) -> Path:
    """The board's cross-built static OpenMP runtime (``libomp.a``), which provides the ``__kmpc_*``
    symbols the multicore (parallel) lowering emits; the thread count is chosen at run time with
    OMP_NUM_THREADS. Built under the repository build root (see the OpenMP path in
    compiler.lower.native.policy) -- a machine asset, so it is not redirected with the output root."""
    if not p.board.openmp_runtime_dir:
        raise RemoteLinuxError(f"board {p.name!r} declares no openmp_runtime_dir; it has no multicore runtime")
    return tracked_build_dir() / p.board.openmp_runtime_dir / "libomp.a"


#: Flag classes the board's model-object build DERIVES for itself and a package may therefore not
#: override. ``-march``/``-mabi`` are TARGET facts (the board's real VLEN, its extensions, its ABI)
#: resolved by :func:`codegen_march` and the board's ``mabi``; a package that also spells an
#: ``-march`` is spelling a weaker one (``rv64gcv`` promises only the RVV MINIMUM VLEN of 128, which
#: doubles every register group on a board with a wider VLEN -- see :func:`codegen_march`). Letting
#: the package win there would silently un-pin the vector length and drop extensions while appearing
#: to be a change of vectorizer flags, so those two are dropped from the package list and the derived
#: values are kept.
_PKG_CFLAGS_DERIVED_HERE = ("-march=", "-mabi=")


def merge_package_cflags(base: list[str], pkg_cflags) -> list[str]:
    """``base`` (the derived model-object flags) extended with the PACKAGE's declared cflags.

    A package's ``knobs.yaml`` may declare cflags (``rvv/hand_v0_int8`` declares
    ``-fno-vectorize -fno-slp-vectorize``). Until this existed the board build never read them, so
    clang's own loop and SLP vectorizers ran on our emitted ``model.ll`` and an unknown share of the
    vector code in the binary was theirs rather than the transform schedule's -- the same
    inert-declaration failure this module's comments warn about, in the other direction: there a flag
    was never handed to the compiler; here a flag was never READ from the thing that declared it.

    FAIL CLOSED. A package cflag that is neither derived here nor understood is an ERROR rather than
    a silent drop: silently dropping is precisely the defect being removed. Flags whose class this
    build derives for itself (:data:`_PKG_CFLAGS_DERIVED_HERE`) are dropped DELIBERATELY and the
    derived value kept, because they are facts about the board rather than package preferences.
    """
    out = list(base)
    for flag in pkg_cflags or ():
        if flag.startswith(_PKG_CFLAGS_DERIVED_HERE):
            continue  # derived from the target; the package's weaker spelling loses
        if not flag.startswith("-"):
            raise RemoteLinuxError(f"package cflag {flag!r} is not a flag; refusing to guess its meaning")
        if flag not in out:
            out.append(flag)
    return out


def _require_toolchain(p: Profile) -> tuple[Path, Path]:
    """The board's cross compiler driver and toolchain prefix, or a fail-closed error naming the variable."""
    cc = toolchain_cc(p)
    root = toolchain_root(p)
    if cc is None or root is None:
        where = (
            p.toolchain_install
            if p.toolchain_install is not None
            else f"${p.env('TOOLCHAIN')} (unset, and no contract declares a default install)"
        )
        raise RemoteLinuxError(f"{p.name} cross-toolchain not found under {where} (set {p.env('TOOLCHAIN')})")
    return cc, root


def _sysroot(p: Profile, root: Path) -> Path | None:
    return root / p.board.toolchain_sysroot if p.board.toolchain_sysroot else None


def build_binary(
    board: Board | Profile,
    model_dir: str | Path,
    work: str | Path,
    pkg,
    inputs_npz: str | Path | None = None,
    force_scalar: bool | None = None,
    parallel: bool = False,
    parallel_harts: int | None = None,
    host_effects=None,
    fallback_policy: str = "allow",
    mmap_weights: bool | None = None,
    kernel_backend: str | None = None,
    dispatch_timing: bool = False,
    op_profile: bool = False,
    dump_cap: int | None = 4096,
    max_session_steps: int | None = None,
    honor_pkg_cflags: bool = False,
) -> Path:
    """Cross-compile a Linux RVV binary for ``board`` from the workload + RVV package.

    Reuses the EXACT spike/Zephyr lowering (``zephyr_model.prepare_for_lowering`` ->
    ``lower_model_file(vectorize=True, transform_schedule=pkg.schedule_text)``) so the board binary
    executes the same emitted RVV as the spike correctness run, and the same data-driven C runtime
    artifacts (``c_runtime.generate`` -> model_gen.h / model_call.c / weights.bin / model_io.h).
    Only the harness (``main_linux.c``) and the compile flags differ (glibc Linux, not medany).

    ``mmap_weights``: True -> the weight blob is NOT embedded; the binary ``mmap``s it from a file
    at run time (deployed alongside, ``MERLIN_WEIGHTS`` env). None (default) -> auto by blob size
    (>= the board's ``mmap_weights_threshold_bytes`` -> mmap). When mmap is used a marker file
    ``<work>/USE_MMAP_WEIGHTS`` records the weights.bin path so :func:`run` deploys it.
    Returns the path to the built ELF.

    ``kernel_backend="outlined_int8"`` links the reusable outlined W8A8 RVV GEMM body; ``None``
    (default) lowers every operation through the ordinary pipeline."""
    p = profile(board)
    mabi = p.board.mabi
    if fallback_policy not in ("allow", "forbid"):
        raise ValueError("fallback_policy must be 'allow' or 'forbid'")
    if kernel_backend not in {None, "outlined_int8"}:
        raise RemoteLinuxError(f"unknown kernel backend {kernel_backend!r}")
    openmp = _resolve_openmp_selection(parallel=parallel, parallel_harts=parallel_harts, host_effects=host_effects)
    parallel_harts = openmp.parallel_harts
    # Per-dispatch matmul-bucket timing needs a routed GEMM shim that carries the timer.
    if dispatch_timing:
        raise RemoteLinuxError("dispatch_timing=True requires a routed GEMM shim with a matmul-bucket timer")
    model_dir, work = Path(model_dir).resolve(), Path(work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    # PREPACKED WEIGHT LAYOUT (`prepack_weight_layout`, default-off). Apply the model's weight
    # re-layout ONCE, here, instead of on every inference: materialize a rewritten bundle (the
    # argument transposes erased and the arguments retyped, the safetensors bytes physically
    # transposed) and use it as `model_dir` from this point on. It has to reach BOTH consumers --
    # `prepare_for_lowering` below reads `model_dir/model.mlir`, while `c_runtime.generate` (step 3)
    # independently re-parses the SAME bundle for the ABI table and copies its weight payload
    # verbatim into `weights.bin`. A rewrite that reached only one of them would leave the compiled
    # object indexing a transposed weight that the blob stores untransposed. Measured on silicon,
    # interleaved same-session: 1.70x on small_llama int8. The rewrite is cached per (bundle,
    # rewrite) and NEVER mutates the source recapture, which every other session reads.
    from merlin.compiler.lower.model import weight_prepack as _wp

    _wp.ensure_registered()  # unregistered => `_composes` swallows the KeyError and never proposes it
    if _wp.FEATURE in (getattr(pkg, "compiler_features", None) or ()):
        try:
            model_dir, _prepack = _wp.prepacked_bundle(model_dir)
        except _wp.PrepackRefused as exc:
            # Fail CLOSED. Falling back to the stock bundle would build the BASELINE while reporting
            # the lever as applied -- the inert-lever failure, but silent and un-auditable.
            raise RemoteLinuxError(f"{_wp.FEATURE}: {exc}") from exc
        (work / "PREPACKED_BUNDLE").write_text(str(model_dir))
        print(
            f"[prepack] {model_dir.name}: "
            f"{_prepack.get('weights_pre_transposed')} weights pre-transposed, "
            f"{_prepack.get('transposes_removed')} transposes removed, "
            f"{_prepack.get('mib_moved_per_inference_before')} MiB/inference no longer moved "
            f"(cached={_prepack.get('cached')})"
        )
    # AOT CONSTANT-WEIGHT QUANTIZATION (`prequantize_constant_weights`, default-off). This is the
    # same kind of ABI-bearing bundle rewrite as prepacking above: the prepared compiler input and
    # c_runtime's argument table must both see the rewritten signature + payload. Apply it after
    # layout prepacking so two requested bundle transforms form one explicit provenance chain.
    from merlin.compiler.lower.model import weight_prequant as _wq

    _wq.ensure_registered()
    if _wq.FEATURE in (getattr(pkg, "compiler_features", None) or ()):
        if not pkg.is_int8:
            raise RemoteLinuxError(f"{_wq.FEATURE} requires an INT8 package")
        try:
            model_dir, _prequant = _wq.prequantized_bundle(model_dir)
        except _wq.PrequantizeRefused as exc:
            raise RemoteLinuxError(f"{_wq.FEATURE}: {exc}") from exc
        (work / "PREQUANTIZED_BUNDLE").write_text(str(model_dir))
        print(
            f"[prequant] {model_dir.name}: "
            f"{_prequant.get('weights_prequantized')} constant matrices stored as i8+scale "
            f"(cached={_prequant.get('cached')})"
        )
    inputs_npz = inputs_npz or (model_dir / "inputs.npz")
    cc, root = _require_toolchain(p)
    # binutils ld for the binary-blob object (use the toolchain's GNU ld, not lld).
    ld = root / "bin" / f"{p.triple}-ld"
    if not ld.is_file():
        ld = Path(shutil.which("ld") or "ld")

    from merlin.compiler.lower.model import c_runtime, toolchain
    from merlin.compiler.lower.model.lower import lower_model_file

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
    # Since a silicon board is this repo's performance oracle, that made those levers unmeasurable on
    # silicon -- while this function's own docstring claimed it "reuses the EXACT spike/Zephyr
    # lowering". Both `zephyr_model.build_app` and `spike_model.build` go through this same entry
    # point, so now all three agree by construction.
    feats_in = frozenset(getattr(pkg, "compiler_features", []) or []) or None
    prepared, feats = zm.prepare_for_lowering(
        model_dir / "model.mlir",
        work,
        int8_compute=pkg.is_int8,
        features=feats_in,
        harts=(int(parallel_harts) if parallel_harts else 1),
        vlen=p.vlen,
        outline_int8=kernel_backend == "outlined_int8",
    )
    feats = feats or None
    # OUTLINED W8A8 backend (default-off): the preparation seam above replaces each legal rank-2
    # i8xi8->i32 contraction with a call before panel/block expansion. One small MR4/NR32 RVV body
    # is compiled here and linked below; batched attention contractions continue through Merlin.
    outlined_int8_obj = None
    n_outlined_int8_routed = 0
    if kernel_backend == "outlined_int8":
        from ..runtime.backends import outlined_int8_board as oi

        if not pkg.is_int8:
            raise RemoteLinuxError("kernel_backend='outlined_int8' requires an INT8 package")
        if not oi.is_available():
            raise RemoteLinuxError("kernel_backend='outlined_int8' but its RVV shim is unavailable")
        signatures = oi.load_signatures(work)
        if not signatures:
            raise RemoteLinuxError("outlined INT8 preparation routed no signatures")
        n_outlined_int8_routed = sum(1 for record in json.loads((work / oi.SIDECAR_NAME).read_text()).get("routed", ()))
        outlined_int8_obj = oi.build_object(
            cc,
            [p.target_flag, f"-march={p.march}", f"-mabi={mabi}", "-O3", "-DNDEBUG"],
            signatures,
            work / "outlined_int8",
            parallel=bool(parallel_harts),
        )
    # PER-OP WHOLE-MODEL PROFILE (default-off, additive). Interleave `@merlin_prof_mark(i32)`
    # calls between the top-level ops of @forward so the board can attribute wall time to the ops
    # the model actually executes — the 94-97% of model time the matmul-bucket timer above cannot
    # see. Runs LAST (after any kernel-backend rewrite) so a routed matmul is profiled as the call
    # it became. The id->op table is written next to the build for the driver to join against.
    if op_profile:
        from merlin.compiler.lower.model import op_profile as opf

        instrumented, prof_table = opf.instrument(prepared.read_text())
        prepared = work / "model.prepared.opprof.mlir"
        prepared.write_text(instrumented)
        opf.write_table(prof_table, work / "opprof_table.json")

    # hoist_static_allocs=False: keep big intermediate buffers on the HEAP (board RAM) instead of
    # promoting them to stack alloca — large models otherwise overflow even a multi-GB stack.
    from merlin.compiler.lower.native.pipeline import PipelineError

    # impr-fork compiler features (PASS/HEURISTIC/PATTERN). Without threading these the board build
    # silently used the BASELINE codegen for any feature-bearing fork — invalidating its
    # benchmark. Pass through so the board binary matches what spike certified.
    # `feats` was resolved above by prepare_for_lowering (which may have swapped a sentinel for a
    # concrete, table-specific feature) — do NOT re-read the package's raw list here, or the schedule
    # would be built from a name the IR was never tagged for.
    if force_scalar is None:
        force_scalar = bool(os.environ.get(p.env("FORCE_SCALAR")))
    if parallel_harts:
        # VECTOR + MULTICORE: the package's RVV schedule with an outer OpenMP-parallel loop
        # layered under it, so the board runs real RVV on several cores at once. This is the
        # same lowering the multicore Zephyr image uses, which makes a silicon board the honest
        # PERFORMANCE oracle for it: spike cannot answer that question (it simulates a spinning
        # hart at full speed), and RTL sim cannot run a whole model. Thread count is chosen at
        # RUN time via OMP_NUM_THREADS, so one binary yields a whole scaling curve.
        # The split `prepare_for_lowering` derived and TAGGED the IR with. Reading it back (rather
        # than re-deriving it here) is what keeps the tags and the schedule arms one decision: a
        # second derivation could disagree with the tags and silently leave every contraction
        # unsplit, which looks exactly like a model that does not parallelize.
        res = lower_model_file(
            prepared,
            work / "lower_vecomp",
            targets=(),
            textual=True,
            vectorize=True,
            transform_schedule=pkg.schedule_text,
            hoist_static_allocs=False,
            features=feats,
            parallel_harts=parallel_harts,
            parallel_chunks=zm.parallel_arms(work),
        )
    elif parallel:
        # Multicore path: scalar int8 datapath + OpenMP parallel loops (no fixed-width
        # vectorize). Used for the big models (rdt/smolvla) that crash the vectorized lowering
        # AND are too slow single-core — the parallel loops fan across the board's cores.
        res = lower_model_file(
            prepared,
            work / "lower_omp",
            targets=(),
            textual=True,
            vectorize=False,
            hoist_static_allocs=False,
            parallel=True,
        )
    else:
        try:
            if force_scalar:
                raise PipelineError(f"forced scalar ({p.env('FORCE_SCALAR')})")
            res = lower_model_file(
                prepared,
                work / "lower",
                targets=(),
                textual=True,
                vectorize=True,
                transform_schedule=pkg.schedule_text,
                hoist_static_allocs=False,
                features=feats,
            )
        except PipelineError:
            # Some models (e.g. xr0's rank-4 two-batch attention) hit a vectorize-path
            # specialization (linalg-specialize-generic-ops) that emits an invalid rank-4
            # linalg.batch_matmul. Fall back to the SCALAR lowering (no specialize pass): the int8
            # datapath is intact, only the contraction stays a scalar loop (correct, unvectorized).
            if fallback_policy == "forbid":
                raise
            res = lower_model_file(
                prepared, work / "lower_scalar", targets=(), textual=True, vectorize=False, hoist_static_allocs=False
            )

    # 2. compile model.ll -> the board's Linux object. The IR is emitted by the repo's LLVM and
    #    carries its attribute syntax (e.g. `captures(none)`), which an older vendor cross-clang
    #    cannot parse, so the .ll -> object step uses THE SAME repo clang re-targeted to the board's
    #    triple; the board toolchain's own compiler builds the C sources + final glibc link below.
    #    NO medany/freestanding — glibc hosted, small code model, fully relocatable.
    clang23 = toolchain.clang()
    model_o = work / "model.o"
    # codegen_march(), not the board's plain march: this is OUR emitted model.ll becoming the model
    # object, so it pays the doubled-LMUL penalty when the march string leaves VLEN at the RVV
    # minimum. The C glue and the baseline shim arms below deliberately keep the plain march so their
    # flags are not re-written mid-campaign. Whether this transfers to whole models is exactly what
    # is being measured.
    # MERLIN_<BOARD>_MODEL_OPT (default -O2 -> byte-identical) lowers the OPT LEVEL of the model
    # object only. It exists for one job: when the same LLVM IR is correct on x86 and wrong on every
    # riscv64 target, the next cut is whether the RISC-V backend's OPTIMIZED codegen is what breaks
    # it, and that needs -O0/-O1 builds of the model while everything else (harness, runtime, flags)
    # is held fixed. Diagnostic only -- never set it for a measurement, since it changes the kernel.
    model_opt = os.environ.get(p.env("MODEL_OPT"), "-O2").split()
    # CFLAGS-CLASS FEATURES apply to the MODEL OBJECT ONLY. Some levers are not expressible in the
    # IR or the transform schedule at all: the vector register-group width comes from clang's own
    # RISC-V vectorizer (`-riscv-v-register-bit-width-lmul`, default 2), so a feature that resolves
    # and is never handed to the compiler emits byte-identical code while reporting as applied --
    # the inert-lever failure this file's own comments warn about. Only the model object gets these;
    # the harness and runtime stay on fixed flags so a measurement changes one thing.
    # ``honor_pkg_cflags=False`` is retained for the explicit A/B diagnostic. Production builds
    # honor package optimization flags while the helper keeps target-owned march/ABI facts pinned.
    _model_flags = _model_compile_flags(p, pkg, feats, model_opt, honor_pkg_cflags=honor_pkg_cflags)
    _run([clang23, p.target_flag, *_model_flags, "-c", res.ll_path, "-o", model_o])
    from ..common import compile_trace as _trace

    _trace.artifact("object", [model_o], pipeline="codegen")
    # 2b. POST-CODEGEN CENSUS: is the model still IN the object? A backend that deletes reachable
    # code still links, still produces a binary, and then reports a spectacular speedup for
    # computing nothing -- smolvla linked a 512 MB ELF whose `forward` was 3,654 bytes with a
    # complete call set of malloc/memset/roundevenf. Nothing between here and the board could see
    # it. The bound is derived from the prepared IR (one instruction per structured op that reaches
    # an output) and carries no constant; it raises, so an erased model cannot be timed.
    from merlin.compiler.lower.model.codegen_census import require_commensurate as _census_require

    _object_census = _census_require(prepared, model_o, "forward")
    print(f"[census gate object] {_object_census.as_dict()}", flush=True)

    # 3. data-driven runtime artifacts (arg table, ciface, weights.bin, embedded io).
    cgen = work / "cgen"
    # `max_session_steps` caps how much of a session corpus is EMBEDDED AS C LITERALS. It is not a
    # micro-optimization for big corpora: resnet50's 256-step, 154 MB session_inputs.npz becomes a
    # 770 MB model_io.h that costs ~7 GB of RSS to compile, and a correctness or per-step latency
    # run needs a handful of steps. None (the default) embeds the whole corpus, unchanged.
    # PANEL-PACKED WEIGHTS (`prepack_weight_panels`, default-off). `prepare_for_lowering` retyped the
    # packed `@forward` arguments in the module that became `model.ll`; this materializes the bundle
    # whose weights.safetensors and manifest were packed to match and checks the two signatures agree
    # argument by argument before the ABI table is built from it. Returns `model_dir` unchanged when
    # the build packed nothing, so a run naming no feature is byte-identical. The packer belongs to the
    # selected vector support provider; with none selected nothing was packed.
    from merlin.compiler.backend.provided import abi_bundle

    model_dir, _panel = abi_bundle(model_dir, work, prepared)
    if _panel is not None:
        (work / "PANEL_PACKED_BUNDLE").write_text(str(model_dir))
        print(
            f"[weight_panel] {Path(model_dir).name}: {_panel['weights_packed']} weights packed "
            f"into panels of {_panel['panel_widths']} columns, {_panel['args_retyped']} arguments "
            f"retyped, {_panel['abi_args_checked']} ABI arguments checked "
            f"(cached={_panel.get('cached')})"
        )
    cinfo = c_runtime.generate(model_dir, cgen, inputs_npz, max_session_steps=max_session_steps, prepared_dir=work)
    if cinfo.get("has_session_quality"):
        (work / "HAS_SESSION_QUALITY").write_text("1")

    # Decide embed-vs-mmap by the weight-blob size (auto unless caller forced it). A blob at or
    # above the board's threshold would make a binary too large for its scratch directory and force
    # every weight resident; mmap keeps resident RAM to the working set. Small models stay embedded
    # (self-contained, proven path).
    weights_bin = cgen / "weights.bin"
    if mmap_weights is None:
        threshold = int(p.board.mmap_weights_threshold_bytes)
        mmap_weights = weights_bin.stat().st_size >= threshold if weights_bin.is_file() else False

    # 4. weights.bin -> binary blob object (embedded path only; mmap reads the file at run time).
    weights_o = work / "weights_blob.o"
    if not mmap_weights:
        _run([ld, "-r", "-b", "binary", "-o", weights_o, "weights.bin"], cwd=cgen)
    else:
        (work / "USE_MMAP_WEIGHTS").write_text(str(weights_bin))

    # 5. the Linux harness.
    main_c = work / "main_linux.c"
    main_c.write_text(
        main_linux_c(
            p,
            dump_cap=dump_cap,
            mmap_weights=mmap_weights,
            dispatch_timing=dispatch_timing,
            op_profile=op_profile,
            persistent_openmp_pool=openmp.worker_pool_size,
        )
    )

    # 6. link the final binary. Reuse the repo's portable C runtime + generated ciface. Prefer a
    #    static binary (no glibc-version coupling to the board); fall back to dynamic, which is
    #    fine since it runs on the board's own glibc.
    rt = runtime_dir() / "c"
    abi = runtime_dir() / "abi"
    binary = work / p.binary
    opt = os.environ.get(p.env("OPT"), "-O2")

    # The whole-model ciface `model_call.c` is one huge function (200+ weight args). A board whose
    # vendor compiler cannot handle it (a register-scavenger crash on large models, independent of
    # precision and opt level) declares `model_call_compiler: llvm`: just model_call.c is compiled
    # with the repo's clang (it needs no stdlib headers) against the toolchain's sysroot, and the
    # other small C sources + the link stay on the board toolchain, which has the libc/crt and
    # builtin headers. The toolchain links that object fine (ELF/ABI compatible; model.o above is
    # likewise produced by the repo's clang).
    from merlin.runtime.boards import MODEL_CALL_LLVM

    model_call_c = cgen / "model_call.c"
    sysroot = _sysroot(p, root)
    other_c = [main_c, rt / "merlin_model.c", abi / "mlir_runtime.c"]
    if openmp.provider == "merlin_pthread_pool":
        other_c = other_c + [rt / "libomp_pthread.c"]
    if op_profile:  # per-op tick accumulator the instrumented IR calls
        other_c = other_c + [rt / "merlin_op_prof.c"]
    # MEASUREMENT PROXY for the lean runtime's static arena (env MERLIN_BUMP_MALLOC, default OFF):
    # link a bump allocator that overrides glibc malloc/free, isolating how much of the dispatch
    # overhead is the ~4391 per-op malloc/free the lean arena eliminates. Baseline byte-identical when off.
    if os.environ.get("MERLIN_BUMP_MALLOC"):
        other_c = other_c + [rt / "merlin_bump_linux.c"]

    if (
        p.board.model_call_compiler == MODEL_CALL_LLVM
        and sysroot is not None
        and Path(clang23).exists()
        and sysroot.is_dir()
    ):
        model_call_o = work / "model_call.o"
        _run(
            [
                str(clang23),
                p.target_flag,
                f"-march={p.march}",
                f"-mabi={mabi}",
                opt,
                "-Wno-override-module",
                f"--sysroot={sysroot}",
                f"--gcc-toolchain={root}",
                f"-I{rt}",
                f"-I{cgen}",
                "-c",
                str(model_call_c),
                "-o",
                str(model_call_o),
            ]
        )
        c_inputs = [*other_c, model_call_o]
    else:
        c_inputs = [*other_c, model_call_c]  # the board toolchain compiles model_call.c itself
    base = [
        cc,
        p.target_flag,
        f"-march={p.march}",
        f"-mabi={mabi}",
        opt,
        "-Wno-override-module",
        f"-I{rt}",
        f"-I{cgen}",
        *c_inputs,
        model_o,
    ]
    if outlined_int8_obj is not None:  # reusable W8A8 RVV GEMM body
        base += [str(outlined_int8_obj)]
    if not mmap_weights:
        base += [str(weights_o)]
    if (parallel or parallel_harts) and openmp.provider == "llvm":
        # Resolve the cross-built __kmpc_* symbols against the static libomp; its LLVM runtime
        # needs the C++/dl deps (see the libomp build note). Order matters: libomp before its deps.
        libomp = _openmp_library(p)
        if not libomp.is_file():
            raise RemoteLinuxError(f"parallel build needs {libomp} (cross-build libomp first)")
        base += [str(libomp), "-lstdc++", "-ldl"]
    base += ["-lm", "-lpthread", "-o", binary]
    try:
        _run([*base, "-static"])
    except RemoteLinuxError:
        _run(base)
    if not binary.is_file():
        raise RemoteLinuxError(f"{p.name} cross-compile produced no binary at {binary}")
    _trace.artifact("link", [binary], pipeline="codegen")
    # The object census above is the strong erasure gate because its whole-object count cannot be
    # inflated by libc/runtime code.  It is NOT the delivered-code measurement: unresolved local
    # calls are only attributed to ``forward`` after relocation (122x difference on tiny_llama).
    # Record both roles explicitly and make the linked ELF the only headline census we print.
    _linked_census = _census_require(prepared, binary, "forward")
    _census_report = {
        "gate_object": _object_census.as_dict(),
        "linked_elf": _linked_census.as_dict(),
    }
    (work / "codegen_census.json").write_text(
        json.dumps(_census_report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"[census linked ELF] {_linked_census.as_dict()}", flush=True)
    if kernel_backend == "outlined_int8":
        (work / "N_OUTLINED_INT8_ROUTED").write_text(str(n_outlined_int8_routed))
    return binary


def build_session_binary(
    board: Board | Profile,
    model_dir: str | Path,
    work: str | Path,
    pkg,
    *,
    force_scalar: bool = False,
    parallel_harts: int | None = None,
    host_effects=None,
    fallback_policy: str = "forbid",
    kernel_backend: str | None = None,
    dump_cap: int | None = 4096,
) -> Path:
    """Cross-compile all programs in a version-2 session and link one board process.

    All stages share one memory and timing domain in a single process.
    """
    p = profile(board)
    mabi = p.board.mabi
    if fallback_policy not in {"allow", "forbid"}:
        raise ValueError("fallback_policy must be 'allow' or 'forbid'")
    openmp = _resolve_openmp_selection(parallel=False, parallel_harts=parallel_harts, host_effects=host_effects)
    parallel_harts = openmp.parallel_harts
    if kernel_backend is not None:
        raise RemoteLinuxError(f"multi-program kernel backend {kernel_backend!r} is not supported")
    model_dir, work = Path(model_dir).resolve(), Path(work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    cc, root = _require_toolchain(p)
    from merlin.compiler.lower.model import session_bundle, toolchain
    from merlin.compiler.lower.model.lower import lower_model_file
    from merlin.compiler.lower.native.pipeline import PipelineError

    from ..runtime.backends import zephyr_model as zm

    generated = work / "session_gen"
    manifest = session_bundle.generate(model_dir, generated)
    clang23 = toolchain.clang()
    sysroot = _sysroot(p, root)
    rt = runtime_dir() / "c"
    abi = runtime_dir() / "abi"
    model_opt = os.environ.get(p.env("MODEL_OPT"), "-O2").split()
    opt = os.environ.get(p.env("OPT"), "-O2")
    features = frozenset(getattr(pkg, "compiler_features", []) or []) or None
    objects: list[Path] = []
    weights: list[dict[str, str]] = []
    stage_censuses: list[tuple[Path, str, dict[str, Any]]] = []

    for record in manifest["programs"]:
        index, name = int(record["index"]), str(record["name"])
        bundle_dir = Path(record["bundle"])
        stage_work = work / f"stage_{index}_{name}"
        stage_work.mkdir(parents=True, exist_ok=True)
        # A session stage is a whole model with a different entrypoint.  Route it through the same
        # preparation seam as the single-program board/Spike/Zephyr builders: per-operation blocks,
        # parallel splits, vector-rank tags, and concrete feature names are derived from THIS
        # stage's prepared IR.  Calling `_prepare_model_mlir` here used to omit those tables and
        # handed the lowering the package's unresolved, session-global request instead.
        prepared, stage_features = zm.prepare_for_lowering(
            bundle_dir / "model.mlir",
            stage_work,
            int8_compute=pkg.is_int8,
            features=features,
            harts=(int(parallel_harts) if parallel_harts else 1),
            vlen=p.vlen,
        )
        stage_features = stage_features or None
        renamed = session_bundle.rename_forward(prepared.read_text(encoding="utf-8"), str(record["entrypoint"]))
        prepared = stage_work / "model.prepared.renamed.mlir"
        prepared.write_text(renamed, encoding="utf-8")
        try:
            if force_scalar:
                raise PipelineError("forced scalar")
            lowered = lower_model_file(
                prepared,
                stage_work / "lower",
                targets=(),
                textual=True,
                vectorize=True,
                transform_schedule=pkg.schedule_text,
                hoist_static_allocs=False,
                features=stage_features,
                parallel_harts=parallel_harts,
                parallel_chunks=zm.parallel_arms(stage_work),
            )
        except PipelineError:
            if fallback_policy == "forbid":
                raise
            lowered = lower_model_file(
                prepared,
                stage_work / "lower_scalar",
                targets=(),
                textual=True,
                vectorize=False,
                hoist_static_allocs=False,
            )
        model_object = stage_work / "model.o"
        # Same cflags-class features as the primary site; a staged sibling that skipped them would
        # measure a different compiler than the one the feature set names.
        _stage_flags = _model_compile_flags(p, pkg, stage_features, model_opt)
        _run([clang23, p.target_flag, *_stage_flags, "-c", lowered.ll_path, "-o", model_object])
        # Same post-codegen census as the primary site (see there): a staged sibling that skipped it
        # would be the one path on which an erased model still reaches a timing run.
        from merlin.compiler.lower.model.codegen_census import require_commensurate as _census_require_stage

        _stage_census = _census_require_stage(prepared, model_object, str(record["entrypoint"]))
        print(f"[census gate object] stage {name!r} {_stage_census.as_dict()}", flush=True)
        stage_censuses.append((prepared, str(record["entrypoint"]), _stage_census.as_dict()))
        model_object = _namespace_stage_object(
            model_object, stage_work / "model.namespaced.o", index=index, entrypoint=str(record["entrypoint"])
        )
        objects.append(model_object)

        cgen = Path(record["generated"])
        call_object = stage_work / "model_call.o"
        call_command = [
            str(clang23),
            p.target_flag,
            f"-march={p.march}",
            f"-mabi={mabi}",
            opt,
            "-Wno-override-module",
        ]
        if sysroot is not None and sysroot.is_dir():
            call_command += [f"--sysroot={sysroot}", f"--gcc-toolchain={root}"]
        _run([*call_command, f"-I{rt}", f"-I{cgen}", "-c", cgen / "model_call.c", "-o", call_object])
        adapter_object = stage_work / "stage_adapter.o"
        _run(
            [
                cc,
                p.target_flag,
                f"-march={p.march}",
                f"-mabi={mabi}",
                opt,
                f"-I{rt}",
                f"-I{cgen}",
                "-c",
                cgen / "stage_adapter.c",
                "-o",
                adapter_object,
            ]
        )
        objects += [call_object, adapter_object]
        weights.append(
            {
                "program": str(index),
                "name": name,
                "path": str(cgen / "weights.bin"),
                "sha256": str(record["weights_sha256"]),
            }
        )

    marker = work / "USE_MMAP_WEIGHTS.json"
    marker.write_text(json.dumps(weights, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    main_c = work / "main_session.c"
    main_c.write_text(
        main_linux_session_c(p, dump_cap, persistent_openmp_pool=openmp.worker_pool_size), encoding="utf-8"
    )
    binary = work / f"{p.binary}_session"
    base = [
        cc,
        p.target_flag,
        f"-march={p.march}",
        f"-mabi={mabi}",
        opt,
        f"-I{rt}",
        f"-I{generated}",
        main_c,
        generated / "merlin_session.c",
        rt / "merlin_model.c",
        abi / "mlir_runtime.c",
        *objects,
    ]
    if openmp.provider == "merlin_pthread_pool":
        base += [rt / "libomp_pthread.c"]
    if parallel_harts and openmp.provider == "llvm":
        libomp = _openmp_library(p)
        if not libomp.is_file():
            raise RemoteLinuxError(f"parallel session build needs {libomp}")
        base += [str(libomp), "-lstdc++", "-ldl"]
    base += ["-lm", "-lpthread", "-o", binary]
    try:
        _run([*base, "-static"])
    except RemoteLinuxError:
        _run(base)
    if not binary.is_file():
        raise RemoteLinuxError(f"{p.name} session cross-compile produced no binary at {binary}")
    # As in the single-program path, retain the object census as the erasure gate and report each
    # preserved stage entry from the linked artifact that actually runs.
    from merlin.compiler.lower.model.codegen_census import require_commensurate as _census_require_linked

    linked_stages = []
    for prepared, entrypoint, object_census in stage_censuses:
        linked = _census_require_linked(prepared, binary, entrypoint)
        linked_stages.append({"entrypoint": entrypoint, "gate_object": object_census, "linked_elf": linked.as_dict()})
        print(f"[census linked ELF] stage {entrypoint!r} {linked.as_dict()}", flush=True)
    (work / "codegen_census.json").write_text(
        json.dumps({"stages": linked_stages}, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (work / "HAS_SESSION_QUALITY").write_text("1", encoding="utf-8")
    return binary


def _namespace_stage_object(source: Path, destination: Path, *, index: int, entrypoint: str) -> Path:
    """Namespace compiler-emitted public helpers while preserving the stage ABI entry points.

    MLIR's LLVM lowering emits helpers such as ``dealloc_helper`` with external linkage. Linking
    several independently lowered modules therefore collides even though their public forwards are
    unique. Rename every other defined external symbol in the ELF object; llvm-objcopy also updates
    its internal relocations, while undefined libc/backend references remain untouched.
    """
    llvm_bin = repo_root() / "third_party" / "llvm-install" / "bin"
    nm, objcopy = llvm_bin / "llvm-nm", llvm_bin / "llvm-objcopy"
    if not nm.is_file() or not objcopy.is_file():
        nm_fallback, objcopy_fallback = Path("/usr/lib/llvm-18/bin/llvm-nm"), Path("/usr/lib/llvm-18/bin/llvm-objcopy")
        nm = nm if nm.is_file() else nm_fallback
        objcopy = objcopy if objcopy.is_file() else objcopy_fallback
    if not nm.is_file() or not objcopy.is_file():
        raise RemoteLinuxError("multi-program linking requires llvm-nm and llvm-objcopy")
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
        raise RemoteLinuxError(f"stage symbol namespacing produced no object at {destination}")
    return destination


def _ssh(p: Profile, *args: str, timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            "ssh",
            "-i",
            p.ssh_key,
            *p.ssh_port_opts,
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=10",
            "-o",
            "StrictHostKeyChecking=no",
            p.host,
            *args,
        ],
        capture_output=True,
        text=True,
        timeout=_bounded_timeout(timeout),
    )


def _scp(p: Profile, source, destination) -> list:
    """The ``scp`` command copying ``source`` to ``destination`` (either may be ``<host>:<path>``)."""
    return [
        "scp",
        "-i",
        p.ssh_key,
        *p.scp_port_opts,
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=no",
        source,
        destination,
    ]


def _remote_bounded(command: str, timeout: int) -> str:
    """Make the board own the deadline so a lost SSH client cannot orphan an inference.

    A host-side ``subprocess.run(timeout=...)`` kills only the local ssh process.  The remote child
    keeps running after that connection disappears (observed: runs consumed board CPU for more than
    a day after their cells timed out).  The board's userland must provide GNU timeout; put the
    measured command under it and quote the complete shell fragment as one argument.
    """
    seconds = int(timeout)
    if seconds < 1:
        raise ValueError("remote timeout must be positive")
    return f"timeout --signal=TERM --kill-after=5s {seconds}s sh -c {shlex.quote(command)}"


def board_vlenb(board: Board | Profile) -> int | None:
    """Read VLENB (bytes) from the board, or None if unreachable. VLEN_bits = vlenb*8."""
    p = profile(board)
    try:
        r = _ssh(p, "cat /proc/cpuinfo | grep -m1 -i vlenb || true", timeout=20)
        for line in r.stdout.splitlines():
            if "vlenb" in line.lower():
                return int(line.split(":")[-1].strip())
    except Exception:  # noqa: BLE001
        pass
    return None


def board_conditions(board: Board | Profile) -> dict[str, Any]:
    """Read the board conditions that can otherwise confound cross-framework latency.

    This is deliberately a separate, untimed SSH probe. Paper adapters call it immediately before
    and after a headline session and preserve both observations; they never infer frequency from
    the requested governor or from Merlin's cycle-conversion constant.
    """
    p = profile(board)
    command = (
        "p=/sys/devices/system/cpu/cpufreq/policy0; "
        "printf 'governor=%s\\n' \"$(cat $p/scaling_governor 2>/dev/null)\"; "
        "printf 'current_khz=%s\\n' \"$(cat $p/scaling_cur_freq 2>/dev/null)\"; "
        "printf 'max_khz=%s\\n' \"$(cat $p/cpuinfo_max_freq 2>/dev/null)\"; "
        "printf 'max_thermal_millic=%s\\n' \"$(cat /sys/class/thermal/thermal_zone*/temp "
        '2>/dev/null | sort -nr | head -1)"'
    )
    proc = _ssh(p, command, timeout=30)
    if proc.returncode:
        raise RemoteLinuxError(f"{p.name} board-condition probe failed: {proc.stderr.strip()}")
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
        raise RemoteLinuxError(f"incomplete {p.name} board-condition observation: {values}")
    return values


def run_arch_probe(board: Board | Profile, source: str | Path, *, timeout: int = 60) -> dict[str, Any]:
    """Cross-compile and run a small architecture-only probe on the board.

    The source is part of the target contract, not a benchmark candidate: it may read architectural
    CSRs and operating-system facts but must not time a kernel.  The returned source digest and raw
    console make the preflight evidence reproducible.  Deployment is serialized with every other user
    of the board and the temporary remote binary is removed on exit.
    """
    p = profile(board)
    source = Path(source)
    if not source.is_file():
        raise RemoteLinuxError(f"{p.name} architecture probe does not exist: {source}")
    cc, _root = _require_toolchain(p)
    if not p.host or not Path(p.ssh_key).is_file():
        raise RemoteLinuxError(f"{p.name} SSH endpoint or key is unavailable ({p.env('HOST')}, {p.env('SSH_KEY')})")
    source_bytes = source.read_bytes()
    digest = hashlib.sha256(source_bytes).hexdigest()
    remote = p.scratch_path(f"{p.binary}_arch_probe_{digest[:12]}")
    with tempfile.TemporaryDirectory(prefix=f"merlin-{p.name}-probe-") as tmp:
        binary = Path(tmp) / "probe"
        _run(
            [
                str(cc),
                p.target_flag,
                f"-march={p.march}",
                f"-mabi={p.board.mabi}",
                "-O2",
                str(source),
                "-static",
                "-o",
                str(binary),
            ]
        )
        with board_lock(p):
            _run(_scp(p, str(binary), f"{p.host}:{remote}"))
            try:
                _ssh(p, f"chmod +x {remote}", timeout=30)
                proc = _ssh(p, remote, timeout=timeout)
            finally:
                try:
                    _ssh(p, f"rm -f {remote}", timeout=30)
                except Exception:  # noqa: BLE001 -- cleanup must not hide the probe verdict
                    pass
    if proc.returncode:
        raise RemoteLinuxError(f"{p.name} architecture probe failed ({proc.returncode}): {proc.stderr.strip()}")
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
    return {
        "source": str(source),
        "source_sha256": digest,
        "values": values,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "returncode": proc.returncode,
    }


def _remote_binary_path(p: Profile, binary: str | Path, name: str) -> str:
    """Where to stage the binary on the board: the scratch directory for small ones, ``remote_dir``
    for large.

    Above the board's ``scratch_limit_bytes`` the binary goes to ``remote_dir`` on real storage. A
    scratch directory that is a tmpfs is RAM, so a weights-embedded binary there lands in the same
    memory the model needs to run in, and two of them may not fit at all. This is the SAME reason the
    mmap path stages weights.bin under ``remote_dir``; a binary with the weights linked in IS the
    weights. A full tmpfs makes scp fail with a bare "write remote ... Failure" that looks like a
    network error rather than ENOSPC.
    """
    try:
        big = Path(binary).stat().st_size > int(p.board.scratch_limit_bytes)
    except OSError:
        big = False
    return f"{p.remote_dir}/{name}" if big else p.scratch_path(name)


def _pull_full_output(p: Profile, remote_out: str, bwork: Path, result: dict[str, Any]) -> None:
    """Replace the empty console output with the complete binary tensor from the board.

    The generated harness deliberately leaves an ``OUT 0`` framing record in stdout so the
    existing parser still proves that the process reached ``DONE``.  The element count printed
    beside the file is the independent completeness witness: a short or missing transfer fails
    before any accuracy gate can mistake it for the model's answer.
    """
    local_out = bwork / "board_output.bin"
    command = _scp(p, f"{p.host}:{remote_out}", local_out)
    # The inference and its output file already exist on the board.  A transient network/SSH close
    # while pulling that file must retry BEFORE the caller's finally block deletes it; otherwise
    # a multi-minute, correct run is irretrievably discarded for a transport failure unrelated to
    # the model.  Three immediate attempts are bounded and preserve the original exception if all
    # fail.  The element-count check below still rejects partial transfers.
    for attempt in range(3):
        try:
            _run(command)
            break
        except RemoteLinuxError:
            local_out.unlink(missing_ok=True)
            if attempt == 2:
                raise
            # A dropped SSH transport can remain unusable for the next immediate
            # connection attempt while sshd tears the old session down.  Back off
            # briefly; the remote result is immutable and still present until the
            # caller's cleanup, so retrying the pull is safe and idempotent.
            time.sleep(1 << attempt)
    import numpy as _np

    output = _np.fromfile(local_out, dtype=_np.float32)
    expected = int(result.get("metrics", {}).get("output_file_elems") or 0)
    if not expected or len(output) != expected:
        raise RemoteLinuxError(f"board output file has {len(output)} elements; expected {expected}")
    result["outputs"] = output
    result["prefix"] = output
    result["output_complete"] = True


def _deploy_weights(p: Profile, bwork: Path, remote: str) -> tuple[str, str | None, list[str]]:
    """Stage the mmap'd weight blob(s) a build left markers for; return ``(env prefix, single blob,
    per-program blobs)``. Weights go to ``remote_dir`` on real storage, never the scratch directory."""
    marker = bwork / "USE_MMAP_WEIGHTS"
    multi_marker = bwork / "USE_MMAP_WEIGHTS.json"
    wenv, remote_w = "", None
    remote_weights: list[str] = []
    if multi_marker.is_file():
        _ssh(p, f"mkdir -p {p.remote_dir}", timeout=30)
        by_digest: dict[str, str] = {}
        for item in json.loads(multi_marker.read_text(encoding="utf-8")):
            digest, program = str(item["sha256"]), int(item["program"])
            remote_stage = by_digest.get(digest)
            if remote_stage is None:
                remote_stage = f"{p.remote_dir}/{Path(remote).name}.weights.{digest[:16]}.bin"
                _run(_scp(p, str(item["path"]), f"{p.host}:{remote_stage}"))
                by_digest[digest] = remote_stage
                remote_weights.append(remote_stage)
            wenv += f"MERLIN_WEIGHTS_{program}={remote_stage} "
    elif marker.is_file():
        _ssh(p, f"mkdir -p {p.remote_dir}", timeout=30)
        remote_w = f"{p.remote_dir}/{Path(remote).name}.weights.bin"
        _run(_scp(p, marker.read_text().strip(), f"{p.host}:{remote_w}"))
        wenv = f"MERLIN_WEIGHTS={remote_w} "
    return wenv, remote_w, remote_weights


def _omp_env(p: Profile, threads: int) -> str:
    bind = p.board.omp_proc_bind
    return f"OMP_NUM_THREADS={threads} " + (f"OMP_PROC_BIND={bind} " if bind else "")


def run_binary(
    board: Board | Profile,
    model_dir: str | Path,
    bwork: str | Path,
    pkg,
    binary: str | Path,
    *,
    env: dict[str, str] | None = None,
    timeout: int = 600,
    capture_full_output: bool = False,
    parallel_harts: int | None = None,
) -> dict[str, Any]:
    """Deploy and run an ALREADY-BUILT board binary, with an explicit environment.

    Split out of :func:`run`'s build+run ladder so a measurement can run the SAME binary
    repeatedly under different settings — the multicore scaling curve varies only
    ``OMP_NUM_THREADS``, and rebuilding per point would put a different object under each
    measurement and make the curve unattributable.

    Takes the board lock around deploy+run only (the cross-compile is CPU-local), matching
    :func:`run`: concurrent agents can build in parallel while the physical board serializes.
    """
    from ..runtime.backends import zephyr_model as zm

    p = profile(board)
    if not p.host:
        raise RemoteLinuxError(f"{p.env('HOST')} unset — board unreachable")
    if parallel_harts is not None and int(parallel_harts) < 1:
        raise ValueError("parallel_harts must be positive when provided")
    bwork = Path(bwork)
    remote = _remote_binary_path(p, binary, f"{Path(model_dir).name}_{pkg.run_id}_envrun_{p.binary}")
    remote_out = f"{p.remote_dir}/{Path(remote).name}.output.bin" if capture_full_output else None
    with board_lock(p):
        if remote.startswith(p.remote_dir) or remote_out:
            _ssh(p, f"mkdir -p {p.remote_dir}", timeout=30)
        _run(_scp(p, str(binary), f"{p.host}:{remote}"))
        wenv, remote_w, remote_weights = _deploy_weights(p, bwork, remote)
        output_env = f"MERLIN_OUTPUT_FILE={remote_out} " if remote_out else ""
        run_env = dict(env or {})
        if parallel_harts is not None:
            run_env["OMP_NUM_THREADS"] = str(int(parallel_harts))
            if p.board.omp_proc_bind:
                run_env["OMP_PROC_BIND"] = p.board.omp_proc_bind
        envs = "".join(f"{k}={v} " for k, v in run_env.items())
        affinity_list = (
            f"0-{int(parallel_harts) - 1}" if parallel_harts is not None and int(parallel_harts) > 1 else None
        )
        taskset = f"taskset -c {affinity_list} " if affinity_list else ""
        try:
            _ssh(p, f"chmod +x {remote}", timeout=30)
            conditions_before = board_conditions(p)
            command = f"{wenv}{output_env}{envs}{taskset}{remote}"
            proc = _ssh(p, _remote_bounded(command, timeout), timeout=timeout + 15)
            result = zm._parse_console(proc.stdout + proc.stderr, proc.returncode)
            if remote_out:
                _pull_full_output(p, remote_out, bwork, result)
            result["board_conditions"] = {"before": conditions_before, "after": board_conditions(p)}
            result["affinity_mask"] = affinity_list
            result["requested_core_count"] = int(parallel_harts or 1)
            result["core_count"] = int(parallel_harts or 1)
            result["affinity_available_cpus"] = int(result.get("metrics", {}).get("affinity_cpus") or 0)
            return result
        finally:
            try:
                cleanup = [remote]
                if remote_w:
                    cleanup.append(remote_w)
                cleanup.extend(remote_weights)
                if remote_out:
                    cleanup.append(remote_out)
                _ssh(p, "rm -f " + " ".join(cleanup), timeout=30)
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
def run(
    board: Board | Profile,
    model_dir: str | Path,
    work: str | Path,
    pkg,
    *,
    timeout: int = 600,
    kernel_backend: str | None = None,
    dispatch_timing: bool = False,
    op_profile: bool = False,
    force_scalar: bool = False,
    parallel_harts: int | None = None,
    fallback_policy: str = "allow",
    host_effects=None,
    require_csr_vlen: bool = False,
    iters: int = 1,
    warmup: int = 0,
    session_repeats: int | None = None,
    dump_cap: int | None = 4096,
    max_session_steps: int | None = None,
) -> dict[str, Any]:
    """Cross-compile the workload for ``board``, deploy over scp, run, and parse OUT/METRIC/DONE.

    Returns the ``zephyr_model._parse_console`` result dict augmented with ``vlen`` (the bit-width
    the board reports — vlenb*8, falling back to the declared VLEN). The ``cycles`` metric is an
    ESTIMATE from the board's ``time`` CSR scaled by ``chip_freq_hz / timebase_hz`` (not
    cycle-accurate); ``wall_ns`` is the silicon authority.

    ``dump_cap`` is the number of output elements the board prints, and therefore the CEILING on
    how much of the answer an accuracy verdict can cover -- the host gate scores the printed prefix
    against the leading elements of the reference, so a short console produces a real-looking
    cos/rel over a slice nobody asked about. ``dump_cap=None`` prints the model's whole output
    (derived from its own ``MERLIN_OUT_ELEMS``). The default is unchanged at 4096, so an existing
    caller gets a byte-identical binary; callers that grade accuracy should pass ``None`` and check
    ``comparison_complete`` on the gate. The value used is echoed back as ``dump_cap``.
    """
    from ..runtime.backends import zephyr_model as zm

    p = profile(board)
    if fallback_policy not in ("allow", "forbid"):
        raise ValueError("fallback_policy must be 'allow' or 'forbid'")
    openmp = _resolve_openmp_selection(parallel=False, parallel_harts=parallel_harts, host_effects=host_effects)
    parallel_harts = openmp.parallel_harts
    if force_scalar and host_effects is not None:
        raise RemoteLinuxError("force_scalar=True conflicts with host-policy parallel lowering effects")

    if not p.host:
        raise RemoteLinuxError(f"{p.env('HOST')} unset — board unreachable")

    if op_profile and _is_multi_program(model_dir):
        # FAIL CLOSED. The multi-program session build path does not take an ``op_profile``
        # parameter, so asking for a per-op profile of a version-2 session builds the
        # UN-instrumented binary, writes no ``opprof_table.json``, and returns a result with no
        # ``op_profile`` key. Downstream that reads as an empty op table, and the profiler reports a
        # breakdown of a model with no ops -- "nothing was measured" wearing the clothes of "there
        # was nothing to measure". It also costs a full build and a board slot to produce that.
        raise RemoteLinuxError(
            "op_profile=True is not supported for a version-2 multi-program session: the session "
            "build path does not instrument the IR, so the run would produce no PROF lines. "
            "Profile each program's bundle (the `stages/<name>` directories) on its own instead."
        )

    requested_mode = "scalar" if force_scalar else "rvv_openmp" if parallel_harts else "rvv"
    multi_program = _is_multi_program(model_dir)

    def _build_deploy_run(mode: str, tag: str) -> dict:
        # mode: "rvv" vectorized (fixed-width RVV); "omp" scalar int8 + OpenMP across the board's
        # cores; "scalar" single-core scalar int8 (last-resort correctness).
        bwork = Path(work) / tag
        # the cross-compile is CPU-local (no board) — do it OUTSIDE the board lock so concurrent
        # forks still build in parallel; only the deploy+run below serializes on the physical board.
        if multi_program:
            if mode == "omp":
                raise RemoteLinuxError("multi-program scalar OpenMP fallback is not implemented")
            binary = build_session_binary(
                p,
                model_dir,
                bwork,
                pkg,
                force_scalar=(mode == "scalar"),
                parallel_harts=(parallel_harts if mode == "rvv_openmp" else None),
                host_effects=(host_effects if mode == "rvv_openmp" else None),
                fallback_policy=fallback_policy,
                kernel_backend=kernel_backend,
                dump_cap=dump_cap,
            )
        else:
            binary = build_binary(
                p,
                model_dir,
                bwork,
                pkg,
                force_scalar=(mode == "scalar"),
                parallel=(mode == "omp"),
                parallel_harts=(parallel_harts if mode == "rvv_openmp" else None),
                host_effects=(host_effects if mode == "rvv_openmp" else None),
                fallback_policy=fallback_policy,
                kernel_backend=kernel_backend,
                dispatch_timing=dispatch_timing,
                op_profile=op_profile,
                dump_cap=dump_cap,
                max_session_steps=max_session_steps,
            )
        with board_lock(p):
            result = _deploy_run(mode, tag, bwork, binary)
        result["local_binary"] = str(binary)
        # RECORD the ceiling the console was produced under. `_gate` already reports
        # n_compared/n_reference/compared_fraction/comparison_complete from the arrays, but those
        # say only THAT a run was short, never why. Without this a truncated verdict and a complete
        # one are indistinguishable in the artifact after the fact.
        result["dump_cap"] = None if dump_cap is None else int(dump_cap)
        result["output_complete"] = dump_cap is None or len(result.get("outputs", ())) < int(dump_cap)
        result["memory_policy"] = (
            "mmap"
            if ((bwork / "USE_MMAP_WEIGHTS").is_file() or (bwork / "USE_MMAP_WEIGHTS.json").is_file())
            else "resident"
        )
        result["execution_mode"] = mode
        result["requested_execution_mode"] = requested_mode
        result["fallback_used"] = mode != requested_mode
        requested_cores = int(parallel_harts) if mode == "rvv_openmp" else p.omp_threads if mode == "omp" else 1
        # A serial image has exactly one calling thread but deliberately remains unpinned, so its
        # affinity mask may contain every CPU. A parallel image is taskset-bounded and must observe
        # exactly the requested mask; fail closed instead of publishing a mislabeled paper cell.
        result["requested_core_count"] = requested_cores
        affinity_cpus = int(result.get("metrics", {}).get("affinity_cpus") or 0)
        result["affinity_available_cpus"] = affinity_cpus
        if mode in ("omp", "rvv_openmp"):
            if affinity_cpus != requested_cores:
                raise RemoteLinuxError(
                    f"requested {requested_cores} cores but sched_getaffinity exposed {affinity_cpus}"
                )
            result["core_count"] = affinity_cpus
        else:
            result["core_count"] = 1
        result["affinity_source"] = "sched_getaffinity"
        if mode == "rvv_openmp" and openmp.provider == "merlin_pthread_pool":
            worker_cpus = _require_worker_census(result.get("metrics", {}), requested=int(parallel_harts))
            result["openmp_runtime"] = openmp.provider
            result["worker_cpus"] = list(worker_cpus)
            result["worker_affinity_ok"] = True
        return result

    def _deploy_run(mode: str, tag: str, bwork: Path, binary) -> dict:
        remote = _remote_binary_path(p, binary, f"{Path(model_dir).name}_{pkg.run_id}_{tag}_{p.binary}")
        if remote.startswith(p.remote_dir):
            _ssh(p, f"mkdir -p {p.remote_dir}", timeout=30)
        _run(_scp(p, str(binary), f"{p.host}:{remote}"))
        # Big-model mmap path: build_binary left a marker with the weights.bin to deploy
        # alongside; the binary mmaps it (resident RAM = working set, not the whole blob).
        wenv, remote_w, remote_weights = _deploy_weights(p, bwork, remote)
        # Fan the OpenMP loops across the board's cores (bound per the board's omp_proc_bind).
        thread_count = int(parallel_harts) if mode == "rvv_openmp" else p.omp_threads
        env = _omp_env(p, thread_count) if mode in ("omp", "rvv_openmp") else ""
        affinity_count = thread_count if mode in ("omp", "rvv_openmp") else 1
        # PIN ONLY WHEN PINNING IS THE POINT. `taskset` here exists to bound a MULTICORE run to a
        # known core count. A single-threaded run does not need it, and pinning one to cpu0
        # specifically is actively wrong for a comparison: cpu0 commonly carries the board's
        # interrupt load, and pinning a single-threaded run there has measured several times slower
        # than unpinned on a BYTE-IDENTICAL binary (model.ll, model.o and the linked ELF all
        # unchanged; only the launch command differed).
        #
        # It is also an ASYMMETRY, which is the part that makes a ratio wrong rather than merely
        # slow: the ExecuTorch arm takes `--cpu_threads=1` and is never tasksetted, so its one
        # thread migrates freely over every core. Pinning ours and not theirs compares a pinned
        # arm against a floating one. Single-threaded therefore gets the same treatment they get --
        # one thread, unpinned -- and the mask is recorded either way so the choice is auditable
        # from the artifact instead of having to be re-derived from an anomaly.
        affinity_list = None if affinity_count == 1 else f"0-{affinity_count - 1}"
        taskset = "" if affinity_list is None else f"taskset -c {affinity_list} "
        # Sustained mode: warmup + N timed passes against the same buffers. Only emitted when
        # asked for, so the default console stays byte-identical to the single-shot path.
        if session_repeats is not None:
            if int(session_repeats) < 1:
                raise ValueError("session_repeats must be positive when provided")
            env += f"MERLIN_SESSION_REPEATS={int(session_repeats)} MERLIN_SESSION_WARMUPS={int(warmup)} "
        elif int(iters) > 1 or int(warmup) > 0:
            env += f"MERLIN_ITERS={int(iters)} MERLIN_WARMUP={int(warmup)} "
        remote_out = f"{p.remote_dir}/{Path(remote).name}.output.bin" if dump_cap is None else None
        output_env = f"MERLIN_OUTPUT_FILE={remote_out} " if remote_out else ""
        if remote_out:
            _ssh(p, f"mkdir -p {p.remote_dir}", timeout=30)
        try:
            _ssh(p, f"chmod +x {remote}", timeout=30)
            conditions_before = board_conditions(p)
            command = f"{wenv}{output_env}{env}{taskset}{remote}"
            proc = _ssh(p, _remote_bounded(command, timeout), timeout=timeout + 15)
            r = zm._parse_console(proc.stdout + proc.stderr, proc.returncode)
            if remote_out:
                _pull_full_output(p, remote_out, bwork, r)
            conditions_after = board_conditions(p)
            r["board_conditions"] = {"before": conditions_before, "after": conditions_after}
            # The mask this wall was produced under. Recorded even when absent (None = unpinned),
            # because "no taskset" and "pinned to cpu0" can differ several-fold and a bare number
            # cannot tell them apart after the fact.
            r["affinity_mask"] = affinity_list
            if (bwork / "HAS_SESSION_QUALITY").is_file():
                quality_env = "MERLIN_VALIDATE_SESSION=1 MERLIN_SESSION_REPEATS=1 MERLIN_SESSION_WARMUPS=0 "
                quality_command = f"{wenv}{env}{quality_env}{taskset}{remote}"
                qproc = _ssh(p, _remote_bounded(quality_command, timeout), timeout=timeout + 15)
                qres = zm._parse_console(qproc.stdout + qproc.stderr, qproc.returncode)
                qm = qres.get("metrics", {})
                steps = int(qm.get("trajectory_steps") or 0)
                r["trajectory_quality"] = {
                    "scope": "trajectory",
                    "steps": steps,
                    "min_cosine": ((qm.get("trajectory_min_cos_ppm") or 0) / 1_000_000.0),
                    "max_relative_error": ((qm.get("trajectory_max_rel_ppm") or 0) / 1_000_000.0),
                    "top1_matches": int(qm.get("trajectory_top1_matches") or 0),
                    "top1_agreement": ((int(qm.get("trajectory_top1_matches") or 0) / steps) if steps else None),
                    "validation_console": qres.get("console", ""),
                }
                correctness_steps = int(qm.get("correctness_trajectory_steps") or 0)
                r["trajectory_correctness"] = {
                    "scope": "trajectory",
                    "steps": correctness_steps,
                    "min_cosine": ((qm.get("correctness_trajectory_min_cos_ppm") or 0) / 1_000_000.0),
                    "max_relative_error": ((qm.get("correctness_trajectory_max_rel_ppm") or 0) / 1_000_000.0),
                    "top1_matches": int(qm.get("correctness_trajectory_top1_matches") or 0),
                    "top1_agreement": (
                        int(qm.get("correctness_trajectory_top1_matches") or 0) / correctness_steps
                        if correctness_steps
                        else None
                    ),
                    "validation_console": qres.get("console", ""),
                }
            ni = bwork / "N_OUTLINED_INT8_ROUTED"
            if ni.is_file():
                r["n_outlined_int8_routed"] = int(ni.read_text().strip())
            # Per-op profile: join the board's `PROF <id> <ticks> <hits>` lines against the
            # id->op table the build emitted. Absent (or empty) unless op_profile=True.
            tbl = bwork / "opprof_table.json"
            if op_profile and tbl.is_file():
                from merlin.compiler.lower.model.op_profile import parse_prof_lines

                table = json.loads(tbl.read_text())
                ticks = parse_prof_lines(r["console"])
                for rec in table:
                    t, h = ticks.get(rec["id"], (0, 0))
                    rec["ticks"], rec["hits"] = t, h
                r["op_profile"] = table
            return r
        finally:
            try:
                cleanup = ([remote_w] if remote_w else []) + remote_weights
                if remote_out:
                    cleanup.append(remote_out)
                _ssh(p, f"rm -f {remote}" + (" " + " ".join(cleanup) if cleanup else ""), timeout=30)
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
    else:
        try:
            res = _build_deploy_run("rvv", "rvv")
        except zm.ZephyrModelError:
            if fallback_policy == "forbid":
                raise
            # The vectorized int8 lowering OOB-stores / crashes for some shapes (e.g. rdt's
            # diffusion transformer) while smaller models vectorize fine. Retry with the MULTICORE
            # lowering (scalar int8 datapath + OpenMP parallel loops) — correct AND fast enough to
            # finish on a multicore board, where single-core scalar times out. Final fall-back:
            # single-core scalar.
            try:
                res = _build_deploy_run("omp", "omp")
            except (zm.ZephyrModelError, RemoteLinuxError):
                res = _build_deploy_run("scalar", "scalar")

    # Prefer the vlenb the harness printed (=== <binary> vlenb=N ===); else probe; else the declaration.
    vlen_bits = p.vlen
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
        probed = board_vlenb(p)
        if probed is not None:
            vlen_bits = int(probed) * 8
            vlen_source = "procfs"
    if require_csr_vlen and vlen_source != "csr":
        raise RemoteLinuxError("paper run requires vlenb read from the harness CSR; no CSR value was parsed")
    if vlen_source in ("csr", "procfs"):
        verify_vlen(p, vlen_bits // 8)
    res["vlen"] = vlen_bits
    res["vlen_source"] = vlen_source
    return res
