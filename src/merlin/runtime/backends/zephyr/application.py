"""Whole-model execution through a selected Zephyr RISC-V board port.

The model is lowered once to an object and linked with Merlin's data-driven C
runtime into a Zephyr application. A worker is pinned to a declared compatible
hart; on heterogeneous boards this prevents vector code from running on a scalar
hart. The selected out-of-tree board catalog supplies the physical memory,
Zephyr port, console and hart facts. The same image can be checked with Spike
when its board protocol is simulator-compatible.

Toolchains are selected with ``ZEPHYR_BASE``, ``ZEPHYR_SDK_INSTALL_DIR`` and
``MERLIN_CHIPYARD`` (for optional host tools). No target checkout or board map is
assumed by importing this module.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from merlin.common.process import invocation_records as _proc
from merlin.compiler.build.zephyr import TreeConfiguration, project_config, vector_max_len_bits
from merlin.compiler.build.zephyr import ZephyrBuildError as ZephyrModelError
from merlin.compiler.lower.model import toolchain
from merlin.runtime.backends.baremetal import spike_launcher as _spike

# ---- zephyr_model ------------------------------------------------------------------------------

# clang flags for the model object. medany keeps it position-tolerant;
# -ffreestanding/-fno-builtin so it needs only the symbols mlir_runtime.c + libc(picolibc)
# provide (cosf/expf/.../memrefCopy/rsqrtf/malloc). Two backends:
#   rvv    — vector-capable harts (rv64gcv), selected by the board descriptor.
#   scalar — no vector (rv64gc); may run on any declared hart.
_CFLAGS_COMMON = ["-mabi=lp64d", "-mcmodel=medany", "-O2", "-ffreestanding", "-fno-builtin"]
# RVV: the ONLY vector ops are the controlled fixed-width ones baked into the IR by the
# transform schedule (linalg.matmul/batch_matmul -> vector<MxNxf32/i32> at e32,m1/m2). clang's
# auto-vectorizer is DISABLED (-fno-vectorize -fno-slp-vectorize): left on, it emits
# fractional-LMUL (mf2/mf4/mf8) and other configs unsupported by some vector
# ports. With autovec off, non-contraction generics fall through to scalar loops;
# only the transform path's fixed-width contraction vectors reach vector lanes.
RVV_CFLAGS = ["-march=rv64gcv", "-fno-vectorize", "-fno-slp-vectorize", *_CFLAGS_COMMON]
SCALAR_CFLAGS = ["-march=rv64gc", *_CFLAGS_COMMON]


def march_with_vlen(cflags: list[str], vlen: int | None) -> list[str]:
    """``cflags`` with the ``-march=`` entry pinned to ``vlen`` via ``zvl<N>b``.

    The COMPILER's assumed vector length has to match the one the code will run on, or a fixed-width
    schedule silently lands at a different LMUL: plain ``-march=rv64gcv`` means the V minimum of 128,
    so on a 256-bit unit every vector group doubles (measured on the K1: doubled LMUL -> spills). Pair
    this with :func:`spike_isa` so the build and the simulation state the same number. ``None`` leaves
    the flags untouched, so the default path is byte-identical.
    """
    if vlen is None:
        return list(cflags)
    out = []
    for f in cflags:
        if f.startswith("-march=") and "zvl" not in f:
            out.append(f"{f}_zvl{int(vlen)}b")
        else:
            out.append(f)
    return out


def _cflags(backend: str) -> list[str]:
    if backend == "scalar":
        return SCALAR_CFLAGS
    if backend == "rvv":
        return RVV_CFLAGS
    raise ZephyrModelError(f"unknown backend {backend!r} (want 'scalar' | 'rvv')")


# ---- environment / toolchain resolution -------------------------------------------


def _pick(env_var: str) -> Path | None:
    """Resolve an explicitly configured toolchain path from the shell or ``.env``."""
    from merlin.common.paths import _dotenv

    v = os.environ.get(env_var)
    if not v:
        v = _dotenv().get(env_var)
    return Path(v).expanduser() if v else None


def _zephyr_base() -> Path:
    selected = _pick("ZEPHYR_BASE")
    if selected is None:
        raise ZephyrModelError("ZEPHYR_BASE is not configured")
    return selected


def _sdk_dir() -> Path:
    selected = _pick("ZEPHYR_SDK_INSTALL_DIR")
    if selected is None:
        raise ZephyrModelError("ZEPHYR_SDK_INSTALL_DIR is not configured")
    return selected


def _conda_bin() -> Path | None:
    from merlin.common.paths import env as _env

    selected = _env("MERLIN_CHIPYARD")
    return Path(selected) / ".conda-env" / "bin" if selected else None


def build_tool(name: str) -> Path | None:
    """Locate a host build tool (``cmake``, ``ninja``), or None.

    Prefer the chipyard conda env, because that is the version this flow is exercised against and
    pinning it keeps a machine with three cmakes reproducible. But fall back to PATH: these are ordinary
    host tools, not part of the RISC-V toolchain, and requiring them to live inside a *chipyard* checkout
    made the Zephyr path unbuildable on any machine that has cmake and ninja the normal way -- with no
    env var to say so, and an `available()` that reported the Zephyr tree as the problem.
    """
    conda_bin = _conda_bin()
    pinned = conda_bin / name if conda_bin is not None else None
    if pinned is not None and pinned.is_file():
        return pinned
    from shutil import which

    found = which(name)
    return Path(found) if found else None


def _tool_env() -> dict:
    env = dict(os.environ)
    env["ZEPHYR_BASE"] = str(_zephyr_base())
    env["ZEPHYR_TOOLCHAIN_VARIANT"] = "zephyr"
    env["ZEPHYR_SDK_INSTALL_DIR"] = str(_sdk_dir())
    conda_bin = _conda_bin()
    if conda_bin is not None and conda_bin.is_dir():
        env["PATH"] = f"{conda_bin}:{env.get('PATH', '')}"
    return env


def build_available() -> bool:
    """Check Zephyr build prerequisites independently of any simulator."""
    try:
        gcc = _spike.gcc_path()
        return (
            _zephyr_base().is_dir()
            and _sdk_dir().is_dir()
            and build_tool("cmake") is not None
            and build_tool("ninja") is not None
            and toolchain.available()
            and gcc.is_file()
            and all(gcc.with_name(f"riscv64-unknown-elf-{name}").is_file() for name in ("ld", "ar", "objcopy"))
        )
    except Exception:  # noqa: BLE001
        return False


def available() -> bool:
    """True when Zephyr build tools and the Spike execution path are present."""
    return build_available() and _spike.available()


# Bounded wall clock for the build's clang/ld steps (this is the seam apply_rvv_package's spike/K1
# build routes through). A pathological schedule (e.g. an outer-product contraction at a large square
# regime) makes clang -O2 spin for many minutes on one object; in a serial beam that hangs the whole
# sweep. Time it out so the fork fails-closed as a build error the certify ladder records. Same
# MERLIN_COMPILE_TIMEOUT_S knob as the host/spike/K1 paths. run_on_spike/FireSim carry their own timeout.
# Default unified at 900s across all four compile wrappers (was 600). For a whole-model beam launch set
# MERLIN_COMPILE_TIMEOUT_S=3600; 0 (or empty) disables the ceiling.
_BUILD_CMD_TIMEOUT_S = int(os.environ.get("MERLIN_COMPILE_TIMEOUT_S", "900") or "0")


def _run(cmd: list, **kw) -> subprocess.CompletedProcess:
    timeout = kw.pop("timeout", _BUILD_CMD_TIMEOUT_S or None)
    return _proc.run_checked(cmd, error=ZephyrModelError, timeout=timeout, timeout_hint=" (pathological compile)", **kw)


# ---- generated Zephyr-app sources --------------------------------------------------

#: Seconds between `ALIVE` lines in a debug image. Slow enough to be free on a 115200-baud console at
#: 50 MHz, fast enough that nobody has to decide whether to keep waiting.
DEBUG_HEARTBEAT_S = 5


class _DebugHarness:
    """The C fragments a debug image adds to the generated harness, or empty strings for a delivery
    image. Grouped in one object so `_main_c`'s f-string stays readable and so the OFF case is
    provably inert (every field is "")."""

    __slots__ = ("decls", "boot", "post_init", "pre_run", "post_run", "post_dump", "stacks")

    def __init__(self, decls="", boot="", post_init="", pre_run="", post_run="", post_dump="", stacks=""):
        self.decls, self.boot, self.post_init = decls, boot, post_init
        self.pre_run, self.post_run, self.stacks = pre_run, post_run, stacks
        #: diagnostics emitted AFTER the answer -- see `_main_c`'s ordering note.
        self.post_dump = post_dump


def _debug_harness(debug: bool, dram_base: int, region_bytes: int, n_harts: int) -> _DebugHarness:
    """Instrumentation for an image that will be run by someone we cannot talk to mid-run.

    Four things, each earned by a specific failure we could not diagnose from what came back:

    * ``STAGE`` lines — the gemmelos report was "banner, build_hash, nothing". Between `build_hash` and
      the first result there is a vector-state write, a pool spin-up and an unbounded stretch of
      compute, and nothing distinguished them. Now each announces itself with a timestamp.
    * an ``ALIVE`` heartbeat naming the current op — separates "slow" from "hung", which on a 50 MHz
      core running a model whose vendor equivalent takes 15 s is the entire question. It reads the op
      index the profiler publishes, so it also says WHERE.
    * a ``MEM`` probe — we build gemmelos images for 1 GB on the owner's word while their own linker
      script declares 256 MB. Writing and reading back a pattern across the linked region answers that
      from inside the run that depends on it, instead of by correspondence.
    * ``STACK`` high-water marks — `MERLIN_WORKER_STACK` is 8 MB and each pool hart adds another, which
      for a small model is ~96% of everything a `uart_tsi` loader transmits. It was chosen deliberately
      and must not be shrunk on a hunch; this measures it so it can be shrunk on evidence.

    The fatal-error hook is separate (`_fatal_c`): it has to be a distinct translation unit because it
    overrides a weak kernel symbol.
    """
    if not debug:
        return _DebugHarness()
    if dram_base <= 0 or region_bytes <= 0:
        raise ZephyrModelError("debug memory probe requires the selected board's DRAM base and region size")
    end = dram_base + region_bytes
    return _DebugHarness(
        decls=f"""
/* ---- debug instrumentation (MERLIN_DEBUG) ------------------------------------------ */
#define MERLIN_REGION_BASE  {hex(dram_base)}ULL
#define MERLIN_REGION_END   {hex(end)}ULL

/* Published by the op profiler when it is linked in; a weak definition keeps this compiling when it
 * is not, in which case the heartbeat still reports elapsed time and just has no op to name. */
__attribute__((weak)) volatile int32_t merlin_prof_last_id = -1;

/* Weak no-op so a single-hart image, which links no OpenMP shim at all, still compiles. The shim's
 * strong definition wins whenever there is a pool to report on. */
__attribute__((weak)) void merlin_omp_report_stacks(void) {{ }}
/* Same reason: the per-op table only exists when the IR was instrumented. */
__attribute__((weak)) void merlin_prof_dump(void) {{ }}

static volatile int merlin_dbg_running;

static void merlin_stage(const char *name)
{{
  printk("STAGE %s hart=%d t=%lld\\n", name, arch_curr_cpu()->id,
         (long long)k_uptime_get());
}}

/* Does the linked region actually EXIST? A region larger than the chip's DRAM boots into nothing, and
 * the symptom is silence -- the same silence as a hang. Probe a handful of points rather than the whole
 * span: a walk of hundreds of MB would itself take minutes on a slow core. Each write is read back
 * through a volatile pointer so the compiler cannot fold it away. */
static void merlin_probe_memory(void)
{{
  const unsigned long long span = MERLIN_REGION_END - MERLIN_REGION_BASE;
  /* Skip the first slice: that is where this very code and its data live. */
  for (int i = 1; i <= 4; i++) {{
    unsigned long long off = (span / 4ULL) * (unsigned long long)i;
    if (off >= span) {{
      off = span - 4096ULL;
    }}
    volatile uint64_t *p = (volatile uint64_t *)(uintptr_t)(MERLIN_REGION_BASE + off);
    uint64_t want = 0xA5A5A5A500000000ULL | (uint64_t)i;
    uint64_t got;
    *p = want;
    __asm__ volatile("fence rw, rw" ::: "memory");
    got = *p;
    printk("MEM %llx %s\\n", (unsigned long long)(uintptr_t)p, got == want ? "ok" : "FAIL");
  }}
}}

/* The ALIVE heartbeat is NOT here. It lives in merlin_op_prof.c, printed from the model's own thread
 * between two top-level ops, because the two obvious places both fail on this hardware: a k_timer
 * expiry prints from interrupt context, which corrupts the non-reentrant HTIF console (observed as
 * `bad syscall #1243416269594910946`), and a low-priority thread is never scheduled because the model
 * runs on a pinned COOPERATIVE worker that does not yield. See the comment there. */

/* How much of each stack was actually touched. MERLIN_WORKER_STACK is 8 MB and every pool hart adds
 * another; for a small model that is the overwhelming majority of everything a MemSiz-transmitting
 * loader puts on the wire. The size was chosen deliberately and must not be cut on a hunch -- so
 * measure it, on the real model, and cut it on the number. */
static void merlin_report_stacks(void)
{{
  size_t unused = 0;

  if (k_thread_stack_space_get(&merlin_worker_thread, &unused) == 0) {{
    printk("STACK worker size=%u unused=%u used=%u\\n",
           (unsigned)MERLIN_WORKER_STACK, (unsigned)unused,
           (unsigned)(MERLIN_WORKER_STACK - unused));
  }}
  merlin_omp_report_stacks();
}}
""",
        stacks="",
        boot="""  merlin_stage("boot");
""",
        post_init="""  merlin_stage("init_done");
  merlin_probe_memory();
  merlin_stage("mem_probed");
  merlin_dbg_running = 1;
""",
        pre_run="""  merlin_stage("warmup_done");
""",
        post_run="""  merlin_dbg_running = 0;
  merlin_stage("compute_done");
""",
        post_dump="""  merlin_report_stacks();
  merlin_prof_dump();
""",
    )


def _completion_metric_line(prefix: str | None) -> str:
    if prefix is None:
        return ""
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 _=<>-:."
    if not isinstance(prefix, str) or not prefix or any(char not in allowed for char in prefix):
        raise ZephyrModelError("completion_metric_prefix contains unsupported C-string characters")
    return f'  printk("{prefix} %llu\\n", (unsigned long long)(c1 - c0));\n'


def _main_c(
    rvv_hart: int,
    dump_cap: int = 4096,
    weights_base: int | None = None,
    n_harts: int = 1,
    iters: int = 1,
    warmup: int = 0,
    omp_threads: int | None = None,
    build_hash: str = "",
    console: str = "htif",
    debug: bool = False,
    dram_base: int = 0,
    region_bytes: int = 0,
    completion_metric_prefix: str | None = None,
) -> str:
    """Generate the Zephyr worker main: one COOP thread pinned to ``rvv_hart`` calls
    ``merlin_run`` and dumps the output with the same OUT/ARGMAX/METRIC/DONE protocol the
    bare-metal harness uses (so the host parser is shared).

    ``n_harts > 1`` additionally spins up the OpenMP pool (``merlin_omp_init``) that the
    multicore-lowered model object's ``__kmpc_*`` calls fan out over. The pool is started
    BEFORE the timed region so thread spin-up is never charged to inference.

    ``iters``/``warmup`` drive SUSTAINED inference: the model is invoked ``warmup + iters``
    times against the same arena, one ``METRIC iter_cycles`` line per timed iteration, so the
    host can report steady-state min/median/p95 instead of a single cold sample. The final
    iteration's output is the one dumped, so every existing correctness gate is unchanged.

    An optional runner may need a terminal cycle marker before ``DONE``. Its
    prefix is supplied by that runner's build selection, never hardcoded here.
    """
    marker_line = _completion_metric_line(completion_metric_prefix)
    # omp_threads decouples "threads the pool fans out to" from "harts the image was lowered
    # for". Same binary, fewer threads: the A/B that separates a threading bug from a codegen
    # bug (at 1 thread the parallel regions run serially on the master through the very same
    # outlined functions).
    nth = n_harts if omp_threads is None else omp_threads
    # How the image STOPS. On a host-assisted console `sys_reboot` is the exit protocol: the SoC turns
    # it into an HTIF `tohost=1` and the loader reports the run finished. With the HTIF console
    # configured out that SoC's `sys_arch_reboot` is an empty function, while `sys_reboot` is declared
    # noreturn and ends in `CODE_UNREACHABLE` -- so calling it would fall through into undefined
    # behavior right after the results were printed, which on a board reads as a crash at the end of a
    # good run. Idle instead: the output is already out, and the operator resets the board anyway.
    halt = "sys_reboot(SYS_REBOOT_COLD)" if console == "htif" else "for (;;) k_cpu_idle()"
    omp_include = '#include "libomp_zephyr.h"\n' if n_harts > 1 else ""
    omp_init = (
        f'  int nth = merlin_omp_init({nth});\n  printk("METRIC omp_threads %d\\n", nth);\n' if n_harts > 1 else ""
    )
    # DEBUG BUILD. Everything below is off by default and costs nothing in a delivery image; it exists
    # because the two boards we ship to are ones we cannot attach to, and both of the last two rounds
    # came back as "it printed three lines and stopped". Three lines is not a diagnosis. With these on,
    # a single returned console log names the stage it reached, the hart it was on, the op index it was
    # inside, and the fault that stopped it -- so one round trip replaces a guess.
    dbg = _debug_harness(debug, dram_base, region_bytes, n_harts)
    return f"""/* Generated by merlin.runtime.backends.zephyr.application — do not edit. */
#include <stdint.h>
#include <string.h>
#include <zephyr/kernel.h>
#include <zephyr/sys/printk.h>
#include <zephyr/sys/reboot.h>
#include <zephyr/arch/cpu.h>

#include "merlin_model.h"
#include "model_gen.h"
#include "model_io.h"
{omp_include}

/* weights.bin linked as a binary blob (ld -r -b binary). In external-weights mode the
 * blob lives in its own high DT memory-region addressed by a compile-time literal
 * (>2GB from .text -> a symbol ref would overflow medany PC-rel); otherwise it is linked
 * into the image and addressed via its symbol. */
extern const unsigned char _binary_weights_bin_start[];
#define MERLIN_WEIGHTS_BASE {("(const void *)" + hex(weights_base) + "ULL") if weights_base is not None else "(const void *)_binary_weights_bin_start"}

#define MERLIN_RVV_HART {rvv_hart}
#define MERLIN_DUMP_CAP {dump_cap}
#define MERLIN_WORKER_STACK (8 * 1024 * 1024)
#define MERLIN_ITERS {iters}
#define MERLIN_WARMUP {warmup}
#define MERLIN_BUILD_HASH "{build_hash}"

#define OUT ((float *)MERLIN_OUTPUT_PTR[0])
static merlin_descriptor_t DESCS[MERLIN_N_ARGS];

K_THREAD_STACK_DEFINE(merlin_worker_stack, MERLIN_WORKER_STACK);
static struct k_thread merlin_worker_thread;
static struct k_sem merlin_done;
{dbg.stacks}
static inline uint64_t rd_mcycle(void) {{
  uint64_t c;
  __asm__ volatile("csrr %0, mcycle" : "=r"(c));
  return c;
}}
{dbg.decls}
static void merlin_worker(void *a, void *b, void *c) {{
  (void)a; (void)b; (void)c;
  printk("=== merlin_zephyr hart=%d ===\\n", arch_curr_cpu()->id);
  /* BUILD IDENTITY. A console log returned by someone running this image on their own board is
   * otherwise unattributable -- there is nothing in the protocol tying it to a specific binary, so a
   * stale log and a fresh one are indistinguishable. This is the sha256 prefix of the lowered model
   * object plus the weights blob, i.e. of exactly what computes the answer. */
  printk("METRIC build_hash %s\\n", MERLIN_BUILD_HASH);
{dbg.boot}
  /* ENABLE VECTOR STATE FOR THIS THREAD, before any vector instruction runs.
   *
   * A freshly created Zephyr thread starts with mstatus.VS = Off: the RISC-V port builds a thread's
   * initial mstatus from MSTATUS_DEF_RESTORE (MPP | MPIE), and the VS bit is only OR'd in under
   * CONFIG_RISCV_ISA_EXT_V, which these images do not set (it puts `v` in the GLOBAL -march, and no
   * matching libgcc multilib exists). reset.S does enable VS, but only for the BOOT context -- a
   * context switch into this thread restores mstatus from the thread's own frame and VS goes back to
   * Off. With VS = Off every vector instruction AND every vector CSR read traps.
   *
   * That trap is not hypothetical: the lowered model's entry function begins with `csrr a0, vlenb`
   * (LLVM sizing a VLEN-scaled stack frame for scalable vectors), so the image dies in the prologue
   * of forward() with mcause=2 before computing anything. It was reported from a tapeout that
   * enforces VS; spike and the vector RTL we simulate do not, which is why every simulated run passed.
   *
   * `libomp_zephyr.c::omp_enable_vector()` already does this for each OpenMP worker, for exactly this
   * reason -- but a single-hart image has no pool, so nothing enabled it. Setting FS too keeps the FP
   * state live. Harmless when the state is already dirty. */
  {{
    unsigned long ms;
    __asm__ volatile("csrr %0, mstatus" : "=r"(ms));
    /* Report VS AS FOUND, before we touch it. A 0 here says the RTOS is not managing vector state
     * for this thread, so it is off again after every context switch and only the write below is
     * holding the image up -- the precise signature of the multi-hart hang, visible in a log from a
     * board we cannot attach to. The multicore path prints the same fact per hart, under its own key:
     * two different values must never share one METRIC name, or whichever is parsed last wins. */
    printk("METRIC worker_hart %d\\n", arch_curr_cpu()->id);
    printk("METRIC worker_mstatus_vs %u\\n", (unsigned)((ms >> 9) & 3));
    ms |= 0x00000600UL | 0x00006000UL;      /* mstatus.VS | mstatus.FS */
    __asm__ volatile("csrw mstatus, %0" ::"r"(ms));
  }}
{omp_init}{dbg.post_init}
  /* SUSTAINED INFERENCE: warmup runs settle caches/branch predictors and (on multicore)
   * the pool; only the MERLIN_ITERS runs after them are reported. Every iteration reuses
   * the same arena and output buffer, so a steady-state drift in iter_cycles is real
   * (allocator churn / leak), not measurement setup. */
  merlin_reset_session();
  for (int w = 0; w < MERLIN_WARMUP; w++) {{
    merlin_prepare_step(w);
    merlin_run_multi(MERLIN_ARGS, MERLIN_N_ARGS, MERLIN_WEIGHTS_BASE,
                     MERLIN_INPUT_PTR, MERLIN_OUTPUT_PTR, DESCS);
#if MERLIN_N_STATE_PAIRS > 0
    if (merlin_commit_state(MERLIN_ARGS, MERLIN_N_ARGS, MERLIN_INPUT_PTR,
                            MERLIN_OUTPUT_PTR, MERLIN_N_STATE_PAIRS,
                            MERLIN_STATE_INPUT_ARGS, MERLIN_STATE_OUTPUT_INDICES) != 0) {{
      printk("FAIL state ABI mismatch during warmup\n"); k_sem_give(&merlin_done); return;
    }}
#endif
  }}
  merlin_reset_session();
{dbg.pre_run}  uint64_t c0 = rd_mcycle();
  for (int it = 0; it < MERLIN_ITERS; it++) {{
    uint64_t i0 = rd_mcycle();
    merlin_prepare_step(it);
    merlin_run_multi(MERLIN_ARGS, MERLIN_N_ARGS, MERLIN_WEIGHTS_BASE,
                     MERLIN_INPUT_PTR, MERLIN_OUTPUT_PTR, DESCS);
#if MERLIN_N_STATE_PAIRS > 0
    if (merlin_commit_state(MERLIN_ARGS, MERLIN_N_ARGS, MERLIN_INPUT_PTR,
                            MERLIN_OUTPUT_PTR, MERLIN_N_STATE_PAIRS,
                            MERLIN_STATE_INPUT_ARGS, MERLIN_STATE_OUTPUT_INDICES) != 0) {{
      printk("FAIL state ABI mismatch during timed session\n"); k_sem_give(&merlin_done); return;
    }}
#endif
    uint64_t i1 = rd_mcycle();
    printk("METRIC iter_cycles %d %llu\\n", it, (unsigned long long)(i1 - i0));
  }}
  uint64_t c1 = rd_mcycle();
{dbg.post_run}
  /* CHEAPEST DECISIVE EVIDENCE FIRST — the console is the scarce resource on a long run.
   *
   * A whole-model FireSim console moves single-digit BYTES PER SECOND, so what is printed and in
   * what order decides whether a finished computation is gradeable. Measured on the Gemma 2 2B run:
   * 7.6 B/s, which makes the 4096-number OUT line worth about ninety minutes of wall clock on its
   * own. Everything below is therefore ordered by evidence-per-byte, so a run truncated at any
   * point has produced the most it could: the metrics and the whole-tensor checks cost a few dozen
   * bytes and come first, the OUT sample costs tens of kilobytes and comes after them, and the
   * per-op profile costs tens of kilobytes more and comes last.
   *
   * `cycles` stays PER-INFERENCE whatever MERLIN_ITERS is, so every existing consumer of this
   * metric keeps comparing like with like (at the default iters=1 it is bit-identical to the
   * single-shot number this harness always reported). The full series is in the METRIC
   * iter_cycles lines above; `total_cycles` is the sustained wall for all of them. */
  printk("METRIC cycles %llu\\n", (unsigned long long)((c1 - c0) / MERLIN_ITERS));
  printk("METRIC total_cycles %llu\\n", (unsigned long long)(c1 - c0));
  printk("METRIC iters %d\\n", MERLIN_ITERS);

  if (MERLIN_OUT_ELEMS > MERLIN_DUMP_CAP) {{
    int rows = MERLIN_OUT_ELEMS / MERLIN_OUT_LASTDIM;
    printk("ARGMAX %d", rows);
    for (int r = 0; r < rows; r++) {{
      const float *row = &OUT[(long)r * MERLIN_OUT_LASTDIM];
      int best = 0; float bv = row[0];
      for (int j = 1; j < MERLIN_OUT_LASTDIM; j++)
        if (row[j] > bv) {{ bv = row[j]; best = j; }}
      printk(" %d", best);
    }}
    printk("\\n");
    float s = 0.0f;
    for (int i = 0; i < MERLIN_OUT_ELEMS; i++) s += OUT[i];
    uint32_t sb; memcpy(&sb, &s, 4);
    printk("SUM %u\\n", (unsigned)sb);
  }}

  /* WHOLE-TENSOR BIT-EXACTNESS IN ONE LINE. OUT below is capped at MERLIN_DUMP_CAP elements — on
   * this model that is 4096 of 2,048,000, i.e. 0.2% of the answer — so on its own it cannot say the
   * other 99.8% is right. FNV-1a over the output bytes checks all of it for ~20 bytes of console
   * instead of ~45 KB. It is a bit-exactness check and nothing else: it says identical or not, and
   * says nothing about HOW far off a mismatch is, which is exactly what the OUT sample is for.
   * Bytes, not floats, so it is insensitive to how the host chooses to read the tensor. */
  {{
    const unsigned char *ob = (const unsigned char *)OUT;
    uint32_t h = 2166136261u;
    for (long i = 0; i < (long)MERLIN_OUT_ELEMS * 4; i++) {{
      h ^= ob[i];
      h *= 16777619u;
    }}
    printk("HASH fnv1a32 %d %u\\n", (int)MERLIN_OUT_ELEMS, (unsigned)h);
  }}

  int k = MERLIN_OUT_ELEMS < MERLIN_DUMP_CAP ? MERLIN_OUT_ELEMS : MERLIN_DUMP_CAP;
  printk("OUT %d", k);
  for (int i = 0; i < k; i++) {{
    uint32_t bits;
    memcpy(&bits, &OUT[i], 4);
    printk(" %u", (unsigned)bits);
  }}
  printk("\\n");
  /* DIAGNOSTICS COME AFTER THE ANSWER, and this ordering is load-bearing on a slow console.
   * The per-op profile is ~2.5k lines; over a simulated UART that is over an hour of wall clock.
   * Emitted before the result it starves it: a whole-model Gemma 2 2B FireSim run reached
   * `STAGE compute_done`, computed every one of its 11,160 ops, and was then killed by the run
   * cap 2,469 PROF lines into the dump -- eight hours spent, no logits printed, nothing gradeable.
   * The answer is the product; the profile is commentary. Print the product first. */
{dbg.post_dump}{marker_line}
  printk("DONE\\n");
  k_sem_give(&merlin_done);
}}

int main(void) {{
  k_sem_init(&merlin_done, 0, 1);
  k_tid_t t = k_thread_create(&merlin_worker_thread, merlin_worker_stack,
                              MERLIN_WORKER_STACK, merlin_worker, NULL, NULL, NULL,
                              K_PRIO_COOP(0), 0, K_FOREVER);
  int rc = k_thread_cpu_pin(t, MERLIN_RVV_HART);
  if (rc != 0) {{
    printk("FAIL k_thread_cpu_pin hart %d rc=%d\\n", MERLIN_RVV_HART, rc);
    {halt};
    return 1;
  }}
  k_thread_start(t);
  k_sem_take(&merlin_done, K_FOREVER);
  {halt};
  return 0;
}}
"""


def _selected_zephyr_tree() -> TreeConfiguration:
    try:
        base = _zephyr_base()
    except ZephyrModelError:
        base = None
    return TreeConfiguration(base)


def _vector_max_len_bits(brd) -> int:
    return vector_max_len_bits(brd, _selected_zephyr_tree())


def _prj_conf(cpus: int, backend: str, brd, console_facts=None, debug: bool = False) -> str:
    return project_config(cpus, backend, brd, console_facts, debug, tree=_selected_zephyr_tree())


def _cmakelists(
    model_archive: Path,
    rt: Path,
    abi: Path,
    cgen: Path,
    weights_section_ld: Path | None = None,
    omp: bool = False,
    n_harts: int = 1,
    hart_ids: tuple[int, ...] | None = None,
    backend: str = "rvv",
    debug: bool = False,
    build_hash: str = "",
    console_facts=None,
    chip_freq_hz: int | None = None,
    op_profile: bool = False,
) -> str:
    # In external-weights mode, a linker snippet places the renamed .merlin_weights
    # section into the WEIGHTS DT memory-region (its own high address). main.c addresses
    # the blob by literal (not its symbol), so --gc-sections would drop the whole
    # weights_blob.o; -u forces the blob symbol as a GC root (no PC-relative reloc) to
    # keep .merlin_weights in the image.
    ext = (
        f'zephyr_linker_sources(SECTIONS "{weights_section_ld}")\n'
        "zephyr_link_libraries(-Wl,-u,_binary_weights_bin_start)\n"
        if weights_section_ld is not None
        else ""
    )
    # Opt-in OpenMP diagnostics, off unless MERLIN_OMP_DEBUG_SPLIT=1 in the environment. The
    # shim's worksharing split is the one place where a wrong answer looks like a crash (a bad
    # loop bound becomes an out-of-range vector load), so being able to dump every
    # (gtid, nthreads, in-range, out-range) without editing C is worth the two lines.
    omp_debug = (
        "target_compile_definitions(app PRIVATE MERLIN_OMP_DEBUG_SPLIT=1)\n"
        if omp and os.environ.get("MERLIN_OMP_DEBUG_SPLIT") == "1"
        else ""
    )
    # Size the worker-stack ARRAY to the harts actually requested. Each stack must be as large
    # as the master's model stack (a worker runs the same outlined code), so leaving the pool at
    # its 8-slot maximum would reserve 64 MB of BSS to use two of them.
    if omp:
        omp_debug += f"target_compile_definitions(app PRIVATE MERLIN_OMP_MAX_THREADS={max(int(n_harts), 1)})\n"
        # Whether the pool must stay on vector-capable harts. For a VECTOR image the shim DISCOVERS
        # which those are at startup (each hart probes its own mstatus.VS writability, which is
        # hardwired to zero without a vector unit) -- so a heterogeneous SoC needs nobody to tell us
        # which harts they are, and the image cannot deadlock by fanning out onto a scalar hart. A
        # SCALAR image sets 0: it may use every hart, which is the point of building one.
        omp_debug += f"target_compile_definitions(app PRIVATE MERLIN_OMP_VECTOR_POOL={1 if backend == 'rvv' else 0})\n"
        # A descriptor may still PIN the set (`vector_hart_ids`), which the runtime probe overrides
        # when it is on. Kept for a chip whose probe cannot be trusted, and as documentation.
        if hart_ids is not None:
            ids = ", ".join(str(int(h)) for h in hart_ids)
            omp_debug += f'target_compile_definitions(app PRIVATE "MERLIN_OMP_HART_IDS={{{ids}}}")\n'
    # A DEBUG image also carries the fatal-error hook. It has to be its own translation unit (it
    # overrides a weak kernel symbol) and it needs the build identity as a macro, so that a fault line
    # names the binary without depending on a banner thousands of lines earlier in the capture.
    dbg_src = f"  {rt}/merlin_fatal_zephyr.c\n" if debug else ""
    if debug and op_profile:
        dbg_src += f"  {rt}/merlin_op_prof.c\n"
        omp_debug += (
            "target_compile_definitions(app PRIVATE MERLIN_PROF_ZEPHYR=1)\n"
            f"target_compile_definitions(app PRIVATE "
            f"MERLIN_PROF_HEARTBEAT_MS={DEBUG_HEARTBEAT_S * 1000})\n"
        )
    # The build identity is a macro in EVERY image, not just debug ones: the allocation guard below
    # is always linked, and a `FAIL alloc` line has to name the binary it came from.
    omp_debug += f'target_compile_definitions(app PRIVATE "MERLIN_BUILD_HASH=\\"{build_hash}\\"")\n'
    # PLL BRING-UP, only when the descriptor asks for a clock the chip is not already on. Without it
    # the image runs at the reset clock, which on the SoC this was written for is a tenth of what the
    # vendor's own demos use. The macros are the SAME derived set the bare-metal console uses -- one
    # derivation, two consumers, so the two paths cannot disagree about where this chip's PLL is.
    if chip_freq_hz and console_facts is not None:
        dbg_src += f"  {rt}/merlin_socinit_zephyr.c\n"
        for flag in console_facts.macros(chip_freq_hz=chip_freq_hz):
            omp_debug += f"target_compile_definitions(app PRIVATE {flag[2:] if flag.startswith('-D') else flag})\n"
    return f"""# Generated by merlin.runtime.backends.zephyr.application.
cmake_minimum_required(VERSION 3.20.0)
find_package(Zephyr REQUIRED HINTS $ENV{{ZEPHYR_BASE}})
project(merlin_zephyr_model)

target_sources(app PRIVATE
  src/main.c
  {cgen}/model_call.c
  {rt}/merlin_model.c
  {abi}/mlir_runtime.c
  {rt}/merlin_alloc_guard_zephyr.c
{dbg_src}{"  " + str(rt) + "/libomp_zephyr.c" + chr(10) if omp else ""})
target_include_directories(app PRIVATE {rt} {cgen})
# Route the lowered model's allocations through the guard, so a heap that runs out reports the size
# it could not satisfy instead of storing through a null pointer. See merlin_alloc_guard_zephyr.c.
zephyr_link_libraries(-Wl,--wrap=malloc)
{omp_debug}
{ext}
# The lowered model object + weights blob, pre-built (clang rv64gcv / ld -r -b binary),
# wrapped in a static archive. --whole-archive so the binary-blob symbol survives gc.
zephyr_link_libraries(-Wl,--whole-archive {model_archive} -Wl,--no-whole-archive)
"""


# ---- _model_statistics -------------------------------------------------------------------------
# Output hashes and steady-state summaries for model execution records.


def out_hash_fnv1a32(a: np.ndarray) -> int:
    """The digest the generated harness prints, recomputed host-side over the same bytes.

    Kept beside the C that emits it so the two cannot drift apart: the image hashes
    ``MERLIN_OUT_ELEMS * 4`` bytes of its little-endian f32 output buffer in memory order, and so
    does this. It answers one question -- is the WHOLE tensor bit-identical -- for about twenty
    bytes of console, where dumping the tensor itself costs tens of kilobytes and a run cap.
    """
    b = np.ascontiguousarray(a, dtype="<f4").tobytes()
    h = 2166136261
    for byte in b:
        h = ((h ^ byte) * 16777619) & 0xFFFFFFFF
    return h


def _sustained_stats(cycles: list[int]) -> dict[str, Any]:
    """Steady-state summary of a sustained run.

    Reports the MEDIAN as the headline: the K1 measurement noise floor is >= 1.9%, so a
    single sample (or a mean skewed by one outlier) is not a result. ``drift`` compares the
    median of the last third against the first third — a growing per-iteration cost is the
    signature of arena growth / allocator churn across invocations, which a single-shot run
    cannot see at all.
    """
    xs = sorted(cycles)
    n = len(xs)

    def _pct(p: float) -> int:
        if n == 1:
            return xs[0]
        idx = min(n - 1, max(0, int(round(p * (n - 1)))))
        return xs[idx]

    med = xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) // 2
    third = max(1, n // 3)
    early = sorted(cycles[:third])[third // 2]
    late = sorted(cycles[-third:])[third // 2]
    return {
        "n": n,
        "min": xs[0],
        "median": med,
        "p95": _pct(0.95),
        "max": xs[-1],
        "early_median": early,
        "late_median": late,
        "drift": (late - early) / early if early else 0.0,
    }
