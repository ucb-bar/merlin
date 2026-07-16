"""Whole-model execution on **Zephyr** (SMP) — spike today, FireSim on the 2-tile board.

This is the bring-up bridge from merlin's bare-metal spike path
(:mod:`spike_model`) to a real RTOS image that runs on the FireSim
``GemminiAndOPUShuttleConfig`` SoC (tile 0 = scalar/Gemmini, tile 1 = Saturn-OPU
``rv64gcv`` vLen=128). It reuses the *entire* data-driven C runtime
(:mod:`merlin.llvmlower.c_runtime` + ``merlin/runtime/c/merlin_model.c`` +
``merlin/runtime/abi/mlir_runtime.c``) and the single ``model.o`` lowered from
``model.mlir``; only the harness changes: instead of ``crt.S``/``htif.c`` +
absolute-addressed arena, it emits a **Zephyr application** with one worker thread
**pinned to the RVV tile** (``k_thread_cpu_pin``) that calls ``merlin_run`` and dumps
the output over the console (HTIF on spike, UART on FireSim).

Why a pinned worker and not ``main()``: on the FireSim board only tile 1 has the
Saturn vector unit, so the ``rv64gcv`` ``model.o`` must execute on hart 1 — running
it on the boot hart (tile 0, scalar) would trap an illegal vector instruction. On
spike (``-p2``) every hart has V, so the same image runs there too.

The Zephyr V recipe mirrors the verified ``samples/test_mt_rvv`` sample:
eager per-thread V save/restore (``RISCV_ISA_EXT_V_LAZY=n`` + ``V_KERNEL_ONLY=y``),
``SMP`` + ``SCHED_CPU_MASK_PIN_ONLY``. The model ``.o`` is built once with clang
(``rv64gcv``) and linked as a static archive into the Zephyr image.

Toolchain/env (see memory ``zephyr-multicore-rvv``): ``ZEPHYR_BASE``,
``ZEPHYR_SDK_INSTALL_DIR`` (0.17.0), and the chipyard conda ``bin`` (west/cmake/ninja)
on PATH. Resolved with sensible defaults, overridable via env.
"""
from __future__ import annotations

import os
import struct
import subprocess
from pathlib import Path
from merlin.common.paths import runtime_dir
from typing import Any

import numpy as np

from ...common.paths import repo_root
from ...llvmlower import c_runtime, toolchain
from ...llvmlower.lower import lower_model_file
from . import spike as _spike

# clang flags for the model object. medany keeps it position-tolerant;
# -ffreestanding/-fno-builtin so it needs only the symbols mlir_runtime.c + libc(picolibc)
# provide (cosf/expf/.../memrefCopy/rsqrtf/malloc). Two backends:
#   rvv    — vector tile (rv64gcv); runs on the Saturn tile (FireSim hart 1).
#   scalar — no vector (rv64gc); runs on the scalar tile (FireSim hart 0). The portable
#            FireSim-safe path: no V means no Saturn-V trap (the FPU_SHARING silent-retry
#            hang the vector path is still being brought up against).
_CFLAGS_COMMON = ["-mabi=lp64d", "-mcmodel=medany", "-O2", "-ffreestanding", "-fno-builtin"]
# RVV: the ONLY vector ops are the controlled fixed-width ones baked into the IR by the
# transform schedule (linalg.matmul/batch_matmul -> vector<MxNxf32/i32> at e32,m1/m2). clang's
# auto-vectorizer is DISABLED (-fno-vectorize -fno-slp-vectorize): left on, it emits
# fractional-LMUL (mf2/mf4/mf8) and other configs the Saturn-OPU (vLen=128, tuned for LMUL=1)
# wedges on — the documented RVV-on-FASED hang. With autovec off, the non-contraction generics
# fall through convert-linalg-to-loops to scalar code (Saturn-safe), and only the transform
# path's fixed-width contraction vectors reach the Saturn vector lanes.
RVV_CFLAGS = ["-march=rv64gcv", "-fno-vectorize", "-fno-slp-vectorize", *_CFLAGS_COMMON]
SCALAR_CFLAGS = ["-march=rv64gc", *_CFLAGS_COMMON]


def _cflags(backend: str) -> list[str]:
    if backend == "scalar":
        return SCALAR_CFLAGS
    if backend == "rvv":
        return RVV_CFLAGS
    raise ZephyrModelError(f"unknown backend {backend!r} (want 'scalar' | 'rvv')")


class ZephyrModelError(RuntimeError):
    pass


# ---- environment / toolchain resolution -------------------------------------------

def _pick(env_var: str, default: str) -> Path:
    """Prefer the env value, but only if it actually exists — the shell may carry a
    stale ``ZEPHYR_BASE`` (e.g. a moved ``backup/`` path); fall back to ``.env`` (a real config
    source, so a checkout configured there resolves without hand-exporting), then to the known-good
    default, rather than silently failing ``available()``."""
    from ...common.paths import _dotenv
    v = os.environ.get(env_var)
    if v and Path(v).exists():
        return Path(v)
    dv = _dotenv().get(env_var)
    if dv and Path(dv).exists():
        return Path(dv)
    return Path(default)


def _zephyr_base() -> Path:
    return _pick("ZEPHYR_BASE", "/path/to/zephyr-chipyard-sw/zephyr_ws/zephyr")


def _zephyr_sw_root() -> Path:
    # the samples/ tree (for the chipyard board overlay we clone).
    return _pick("MERLIN_ZEPHYR_SW", "/path/to/zephyr-chipyard-sw")


def _sdk_dir() -> Path:
    return _pick("ZEPHYR_SDK_INSTALL_DIR", "/home/rose/zephyr-sdk-0.17.0")


def _conda_bin() -> Path:
    from ...common.paths import env as _env
    return Path(_env("MERLIN_CHIPYARD", "/path/to/chipyard")) / ".conda-env" / "bin"


def _tool_env() -> dict:
    env = dict(os.environ)
    env["ZEPHYR_BASE"] = str(_zephyr_base())
    env["ZEPHYR_TOOLCHAIN_VARIANT"] = "zephyr"
    env["ZEPHYR_SDK_INSTALL_DIR"] = str(_sdk_dir())
    env["PATH"] = f"{_conda_bin()}:{env.get('PATH','')}"
    return env


def available() -> bool:
    """True when the Zephyr build + spike toolchain are present."""
    cmake = _conda_bin() / "cmake"
    ninja = _conda_bin() / "ninja"
    try:
        return (_zephyr_base().is_dir() and _sdk_dir().is_dir() and cmake.is_file()
                and ninja.is_file() and _spike.available())
    except Exception:  # noqa: BLE001
        return False


def _run(cmd: list, **kw) -> subprocess.CompletedProcess:
    proc = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, **kw)
    if proc.returncode != 0:
        raise ZephyrModelError(
            f"command failed: {' '.join(map(str, cmd))}\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")
    return proc


DEFAULT_RAM_BYTES = 256 * 1024 * 1024   # spike/chipyard `ram0` default (0x10000000)

# Above this, a weights blob linked into the image's .data overflows the medany ±2GB
# PC-relative window (Zephyr's own .text<->.bss refs break). Past it we switch to
# "external weights": the blob lives in its own DT memory-region at a fixed high absolute
# address, referenced by integer-constant base (no far symbol), keeping ram0 compact.
LINK_LIMIT = 1900 * 1024 * 1024
EXT_RAM0_BYTES = 0x40000000             # 1 GB ram0 (code + activation arena) in ext mode
EXT_WEIGHTS_BASE = 0xC0000000           # weights region origin (3 GB), right after ram0
DRAM_END = 0x80000000 + 16 * 1024**3    # FireSim WithExtMemSize = 16 GB at 0x80000000
# The WEIGHTS region uses a 2-cell DT container (#address/size-cells=2) so its reg can
# express a 64-bit base+size — handling blobs > 4 GB (tiny 4.1 G, pi05 fp32 13 G), not just
# the 1-cell <4 GB case. The only hard cap is physical DRAM: weights at EXT_WEIGHTS_BASE
# must end before the 16 GB DRAM end (leave ram0 + margin).
EXT_MAX_WEIGHTS = DRAM_END - EXT_WEIGHTS_BASE - (256 * 1024 * 1024)   # ~14.75 GB


def _ram_for_weights(weights_bytes: int) -> int:
    """RAM-region size to hold the weights blob (linked into .data) plus an activation
    arena (the leftover, claimed by ARENA_SIZE=-1). Headroom scales with the model
    (30% of weights + 128 MB) rather than a fat fixed floor, so small models stay at the
    256 MB default — important on FireSim, where the whole-model image only boots reliably
    at the stock `ram0` size and an over-large region wedges the boot (no uartlog). Models
    that genuinely need more grow the region; rounded up to 16 MB."""
    total = weights_bytes + (weights_bytes * 3) // 10 + 128 * 1024 * 1024
    align = 16 * 1024 * 1024
    total = ((total + align - 1) // align) * align
    return max(DEFAULT_RAM_BYTES, total)


def _prepare_model_mlir(mlir_path: Path, work: Path, *, int8_compute: bool = False) -> Path:
    """Apply the dispatch_runtime normalization passes to ``model.mlir`` and write the
    prepared module to ``work/model.prepared.mlir``. These make quantized / bf16 /
    over-rank-matmul / bool-cast models lowerable to a single object — the same fixes the
    numpy interpreter applies in ``dispatch_runtime.run_model`` (collapse over-rank
    `aten.linear` matmuls; dequantize int8 weights; f32-accumulate bf16 matmuls; fix the
    sitofp(i1) bool-cast sign). Without them only clean LLMs (small_llama) lower."""
    from ...frontends.linalg_mlir import parse_mlir_file
    from ...xdsl_dialects._common import text as to_text
    from ...llvmlower.passes_xdsl import (collapse_overrank_matmul, fix_bool_sitofp,
                                          lower_bf16_matmul_f32acc, lower_quant_ext)
    from ..dispatch_runtime import _propagate_quant_inner

    module = parse_mlir_file(mlir_path)
    collapse_overrank_matmul(module)
    _propagate_quant_inner(module)
    if int8_compute:
        # Real W8A8 integer datapath (matmul/conv/attention -> i8xi8->i32 + requant; the
        # transcendentals -> integer/RVV). lower_quant_ext stays AFTER as the f32 fallback
        # for any dequant the int8 passes did not convert (nonzero-zp, embeddings).
        from ...llvmlower.passes_quant_int import (lower_contraction_int8, lower_conv_int8,
                                                   lower_softmax_int, lower_gelu_int,
                                                   lower_silu_int, lower_rsqrt_int)
        lower_contraction_int8(module)
        lower_conv_int8(module)
        lower_softmax_int(module)
        lower_gelu_int(module)
        lower_silu_int(module)
        lower_rsqrt_int(module)
    lower_quant_ext(module)
    lower_bf16_matmul_f32acc(module)
    fix_bool_sitofp(module)
    # PER-RANK VECTORIZE TAGGING (env MERLIN_VEC_RANK, default OFF -> baseline byte-identical): tag each
    # all-parallel (non-reduction) linalg.generic with `merlin.vec_r{rank}` so the transform schedule can
    # BOUNDED-vectorize the scalar non-matmul ops by rank (the win lever for openvla — ~900ms of scalar
    # parallel generics -> ~110ms vectorized). Reductions (softmax/norm) skipped (need stable form).
    import os as _os
    if _os.environ.get("MERLIN_VEC_RANK"):
        from xdsl.dialects.builtin import UnitAttr
        n_tag = 0
        for op in module.walk():
            if op.name != "linalg.generic":
                continue
            its = str(op.properties.get("iterator_types", ""))
            if "reduction" in its or "parallel" not in its:
                continue
            rank = its.count("iterator_type")
            if 1 <= rank <= 4:
                op.attributes[f"merlin.vec_r{rank}"] = UnitAttr()
                n_tag += 1
        print(f"[MERLIN_VEC_RANK] tagged {n_tag} all-parallel generics for bounded vectorize")
    out = work / "model.prepared.mlir"
    out.write_text(to_text(module))
    return out


# ---- generated Zephyr-app sources --------------------------------------------------

def _main_c(rvv_hart: int, dump_cap: int = 4096, weights_base: int | None = None) -> str:
    """Generate the Zephyr worker main: one COOP thread pinned to ``rvv_hart`` calls
    ``merlin_run`` and dumps the output with the same OUT/ARGMAX/METRIC/DONE protocol the
    bare-metal harness uses (so the host parser is shared)."""
    return f"""/* Generated by merlin.runtime.backends.zephyr_model — do not edit. */
#include <stdint.h>
#include <string.h>
#include <zephyr/kernel.h>
#include <zephyr/sys/printk.h>
#include <zephyr/sys/reboot.h>
#include <zephyr/arch/cpu.h>

#include "merlin_model.h"
#include "model_gen.h"
#include "model_io.h"

/* weights.bin linked as a binary blob (ld -r -b binary). In external-weights mode the
 * blob lives in its own high DT memory-region addressed by a compile-time literal
 * (>2GB from .text -> a symbol ref would overflow medany PC-rel); otherwise it is linked
 * into the image and addressed via its symbol. */
extern const unsigned char _binary_weights_bin_start[];
#define MERLIN_WEIGHTS_BASE {("(const void *)" + hex(weights_base) + "ULL") if weights_base is not None else "(const void *)_binary_weights_bin_start"}

#define MERLIN_RVV_HART {rvv_hart}
#define MERLIN_DUMP_CAP {dump_cap}
#define MERLIN_WORKER_STACK (8 * 1024 * 1024)

static float OUT[MERLIN_OUT_ELEMS];
static merlin_descriptor_t DESCS[MERLIN_N_ARGS];

K_THREAD_STACK_DEFINE(merlin_worker_stack, MERLIN_WORKER_STACK);
static struct k_thread merlin_worker_thread;
static struct k_sem merlin_done;

static inline uint64_t rd_mcycle(void) {{
  uint64_t c;
  __asm__ volatile("csrr %0, mcycle" : "=r"(c));
  return c;
}}

static void merlin_worker(void *a, void *b, void *c) {{
  (void)a; (void)b; (void)c;
  printk("=== merlin_zephyr hart=%d ===\\n", arch_curr_cpu()->id);
  uint64_t c0 = rd_mcycle();
  merlin_run(MERLIN_ARGS, MERLIN_N_ARGS, MERLIN_WEIGHTS_BASE,
             MERLIN_INPUT_PTR, OUT, DESCS);
  uint64_t c1 = rd_mcycle();

  int k = MERLIN_OUT_ELEMS < MERLIN_DUMP_CAP ? MERLIN_OUT_ELEMS : MERLIN_DUMP_CAP;
  printk("OUT %d", k);
  for (int i = 0; i < k; i++) {{
    uint32_t bits;
    memcpy(&bits, &OUT[i], 4);
    printk(" %u", (unsigned)bits);
  }}
  printk("\\n");

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
  printk("METRIC cycles %llu\\n", (unsigned long long)(c1 - c0));
  /* Terminal sentinel reused from the ModelBlaster FireSim runner: its
   * run_firesim() waits for this marker to know the block is complete. */
  printk("=== MODELBLASTER_WALL_CYCLES === %llu\\n", (unsigned long long)(c1 - c0));
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
    sys_reboot(SYS_REBOOT_COLD);
    return 1;
  }}
  k_thread_start(t);
  k_sem_take(&merlin_done, K_FOREVER);
  sys_reboot(SYS_REBOOT_COLD);
  return 0;
}}
"""


def _prj_conf(arena_mb: int, cpus: int, backend: str) -> str:
    arena = arena_mb * 1024 * 1024
    common = f"""# Generated by merlin.runtime.backends.zephyr_model (backend={backend}).
CONFIG_PRINTK=y
CONFIG_STDOUT_CONSOLE=y
CONFIG_THREAD_STACK_INFO=y

CONFIG_MP_MAX_NUM_CPUS={cpus}
CONFIG_SMP=y
CONFIG_SCHED_CPU_MASK=y
CONFIG_SCHED_CPU_MASK_PIN_ONLY=y
CONFIG_SCHED_DUMB=y

# SMP HTIF console: direct-putchar HTIF races under SMP (a worker on hart!=0
# printing) and silently wedges — use the buffered + syscall path. (Lesson 3,
# merlin_hetero_runner.)
CONFIG_UART_HTIF_BUFFERED_OUTPUT=y
CONFIG_UART_HTIF_BUFFERED_OUTPUT_SIZE=256
CONFIG_UART_HTIF_SYSCALL_PRINT=y

CONFIG_FPU=y
CONFIG_FLOAT_HARD=y
# FPU_SHARING=y mis-routes V-illegal-instruction traps to the FP path, which
# silently retries forever (a hang with NO fault printed) — the FireSim
# Saturn-tile hang. Keep it off so traps surface, and so single-COOP-worker FP
# context stays eager. (merlin_hetero_runner.)
CONFIG_FPU_SHARING=n

# malloc for the lowered model's memref.alloc buffers (the whole-model activation
# arena). ARENA_SIZE=-1 = "use all leftover RAM" (a linker-defined region at the end of
# RAM), NOT a fixed static BSS array — a large fixed array overflows R_RISCV_PCREL_HI20
# relocations once it exceeds the model-link window. So the arena scales to the board's
# RAM (spike -m / FireSim DRAM) for free, and big models just work.
CONFIG_COMMON_LIBC_MALLOC=y
CONFIG_COMMON_LIBC_MALLOC_ARENA_SIZE=-1

CONFIG_MAIN_STACK_SIZE=32768
CONFIG_HEAP_MEM_POOL_SIZE=65536
"""
    if backend == "scalar":
        return common  # no vector config at all — the model object is rv64gc.
    # rvv: eager per-thread V save/restore + keep global -march scalar so the
    # non-V picolibc multilib links (the rv64gcv model object re-adds V itself).
    return common + """
CONFIG_RISCV_ISA_EXT_V=y
CONFIG_RISCV_VECTOR_MAX_LEN=256
CONFIG_RISCV_V_KERNEL_ONLY=y
CONFIG_RISCV_ISA_EXT_V_LAZY=n
"""


def _cmakelists(model_archive: Path, rt: Path, abi: Path, cgen: Path,
                weights_section_ld: Path | None = None) -> str:
    # In external-weights mode, a linker snippet places the renamed .merlin_weights
    # section into the WEIGHTS DT memory-region (its own high address). main.c addresses
    # the blob by literal (not its symbol), so --gc-sections would drop the whole
    # weights_blob.o; -u forces the blob symbol as a GC root (no PC-relative reloc) to
    # keep .merlin_weights in the image.
    ext = (f'zephyr_linker_sources(SECTIONS "{weights_section_ld}")\n'
           "zephyr_link_libraries(-Wl,-u,_binary_weights_bin_start)\n"
           if weights_section_ld is not None else "")
    return f"""# Generated by merlin.runtime.backends.zephyr_model.
cmake_minimum_required(VERSION 3.20.0)
find_package(Zephyr REQUIRED HINTS $ENV{{ZEPHYR_BASE}})
project(merlin_zephyr_model)

target_sources(app PRIVATE
  src/main.c
  {cgen}/model_call.c
  {rt}/merlin_model.c
  {abi}/mlir_runtime.c
)
target_include_directories(app PRIVATE {rt} {cgen})
{ext}
# The lowered model object + weights blob, pre-built (clang rv64gcv / ld -r -b binary),
# wrapped in a static archive. --whole-archive so the binary-blob symbol survives gc.
zephyr_link_libraries(-Wl,--whole-archive {model_archive} -Wl,--no-whole-archive)
"""


# ---- build / run -------------------------------------------------------------------

def build_app(model_dir: str | Path, work: str | Path, *, board: str = "spike_riscv64",
              backend: str = "rvv", rvv_hart: int = 0, arena_mb: int = 64, cpus: int = 2,
              inputs_npz: str | Path | None = None, ram_bytes_override: int | None = None,
              int8_compute: bool = False, rvv_schedule: str | None = None,
              cflags_override: list[str] | None = None,
              features: "frozenset[str] | None" = None) -> dict:
    """Lower the model, generate the Zephyr app, and build ``zephyr.elf``.

    ``backend``: ``"rvv"`` (vector tile / Saturn) or ``"scalar"`` (scalar tile). The
    scalar build is the portable FireSim-safe path; the vector build targets the Saturn
    tile (worker on ``rvv_hart``). Returns ``{elf, app_dir, backend, **c_runtime_info}``.

    ``rvv_schedule`` overrides the default ``RVV_TRANSFORM_SCHEDULE`` (the seam through which
    an isolated RVV target-package supplies its own transform schedule); ``cflags_override``
    replaces ``_cflags(backend)``. Both default ``None`` -> the build is byte-identical to the
    shipping codegen, so the global flow is never perturbed by the package machinery.
    """
    model_dir, work = Path(model_dir).resolve(), Path(work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    inputs_npz = inputs_npz or (model_dir / "inputs.npz")
    if not available():
        raise ZephyrModelError("Zephyr/spike toolchain unavailable (see env in module doc)")
    cflags = cflags_override or _cflags(backend)

    gcc = _spike.gcc_path()
    ld = gcc.with_name("riscv64-unknown-elf-ld")
    ar = gcc.with_name("riscv64-unknown-elf-ar")
    objcopy = gcc.with_name("riscv64-unknown-elf-objcopy")
    clang = toolchain.clang()

    # 1. model.mlir -> normalize (SAME prep passes the dispatch_runtime applies, so
    #    quantized / bf16 / over-rank / bool-cast models lower correctly — without these
    #    the whole-model path only handles already-clean LLMs) -> LLVM IR -> object.
    prepared = _prepare_model_mlir(model_dir / "model.mlir", work, int8_compute=int8_compute)
    # For the rvv backend, bake native RVV (fixed-width vector ops on the matmuls) into the
    # IR rather than leaving it to clang's auto-vectorizer — see llvmlower.pipeline.
    res = lower_model_file(prepared, work / "lower", targets=(), textual=True,
                           vectorize=(backend == "rvv"), transform_schedule=rvv_schedule,
                           features=features)
    _run([clang, "--target=riscv64-unknown-elf", *cflags, "-c", res.ll_path,
          "-o", work / "model.o"])

    # 2. data-driven runtime artifacts (arg table, ciface, weights.bin, embedded io).
    cgen = work / "cgen"
    info = c_runtime.generate(model_dir, cgen, inputs_npz)

    # 3. weights.bin -> binary blob object; archive it with model.o.
    _run([ld, "-r", "-b", "binary", "-o", work / "weights_blob.o", "weights.bin"], cwd=cgen)
    weights_size = (cgen / "weights.bin").stat().st_size

    # External-weights mode for blobs that would overflow medany linked into .data: rename
    # the blob's section so Zephyr's default linker won't pull it into the image .data; a
    # snippet diverts it to the WEIGHTS region at a fixed high address, and main.c addresses
    # it by literal. Keeps ram0 compact (code + arena), so big fp32 models link.
    external = ram_bytes_override is None and weights_size > LINK_LIMIT
    weights_base = None
    if external:
        if weights_size > EXT_MAX_WEIGHTS:
            raise ZephyrModelError(
                f"weights blob {weights_size / 2**30:.1f} GB does not fit the 16 GB DRAM "
                f"after ram0 (max ~{EXT_MAX_WEIGHTS / 2**30:.1f} GB) — out of envelope.")
        _run([objcopy, "--rename-section", ".data=.merlin_weights", work / "weights_blob.o"])
        weights_base = EXT_WEIGHTS_BASE
        ram_bytes = (EXT_WEIGHTS_BASE - 0x80000000) + weights_size   # spike -m span
    else:
        # Size ram0 to the in-image weights blob + activation-arena headroom (default 256 MB
        # is too small for multi-hundred-MB int8/fp8 blobs). ARENA_SIZE=-1 claims the
        # leftover. Spike gets a matching -m; FireSim DRAM is fixed by the bitstream.
        ram_bytes = (ram_bytes_override if ram_bytes_override is not None
                     else _ram_for_weights(weights_size))

    archive = work / "libmerlinmodel.a"
    archive.unlink(missing_ok=True)
    _run([ar, "rcs", archive, work / "model.o", work / "weights_blob.o"])

    # 4. emit the Zephyr application tree.
    app = work / "app"
    (app / "src").mkdir(parents=True, exist_ok=True)
    (app / "boards").mkdir(parents=True, exist_ok=True)
    (app / "src" / "main.c").write_text(_main_c(rvv_hart, weights_base=weights_base))
    (app / "prj.conf").write_text(_prj_conf(arena_mb, cpus, backend))
    rt = runtime_dir() / "c"
    abi = runtime_dir() / "abi"
    weights_section_ld = None
    if external:
        # snippet: divert the renamed blob section into the WEIGHTS DT memory-region.
        weights_section_ld = app / "weights_sections.ld"
        weights_section_ld.write_text(
            ".merlin_weights : {\n"
            "\tKEEP(*(.merlin_weights))\n"
            "\tKEEP(*(.merlin_weights.*))\n"
            "} GROUP_LINK_IN(WEIGHTS)\n")
    (app / "CMakeLists.txt").write_text(
        _cmakelists(archive, rt, abi, cgen, weights_section_ld))
    # Board overlay. External mode: ram0 = 1 GB (code + arena) and a separate WEIGHTS
    # memory-region holding the blob at EXT_WEIGHTS_BASE. Otherwise: grow ram0 only when
    # the model needs > the stock 256 MB (small models keep the default that boots reliably
    # on FireSim). Plus, for chipyard, the disable-cpu@2..7 overlay.
    overlay = ""
    if external:
        wsz = (weights_size + 0xFFF) & ~0xFFF                       # 4 KB align
        # The WEIGHTS region lives under a 2-cell (#address/size-cells=2) container so its
        # reg can express a 64-bit base+size — letting the blob exceed 4 GB (tiny, pi05).
        # The chipyard root is 1-cell, so we add a child bus with 2/2 cells. base & size are
        # emitted as <hi lo> pairs.
        b_hi, b_lo = (EXT_WEIGHTS_BASE >> 32) & 0xFFFFFFFF, EXT_WEIGHTS_BASE & 0xFFFFFFFF
        s_hi, s_lo = (wsz >> 32) & 0xFFFFFFFF, wsz & 0xFFFFFFFF
        overlay += (f"&ram0 {{\n\treg = <0x80000000 {hex(EXT_RAM0_BYTES)}>;\n}};\n\n"
                    f"/ {{\n\tweights_bus {{\n"
                    f"\t\t#address-cells = <2>;\n\t\t#size-cells = <2>;\n\t\tranges;\n"
                    f"\t\tweights0: memory@{EXT_WEIGHTS_BASE:x} {{\n"
                    f"\t\t\tcompatible = \"zephyr,memory-region\", \"mmio-sram\";\n"
                    f"\t\t\treg = <{hex(b_hi)} {hex(b_lo)} {hex(s_hi)} {hex(s_lo)}>;\n"
                    f"\t\t\tzephyr,memory-region = \"WEIGHTS\";\n\t\t}};\n\t}};\n}};\n")
    elif ram_bytes > DEFAULT_RAM_BYTES:
        overlay += f"&ram0 {{\n\treg = <0x80000000 {hex(ram_bytes)}>;\n}};\n"
    if board.startswith("chipyard"):
        src_overlay = (_zephyr_sw_root() / "samples" / "merlin_hetero_runner"
                       / "boards" / "chipyard_riscv64.overlay")
        if src_overlay.is_file():
            overlay += ("\n" if overlay else "") + src_overlay.read_text()
    if overlay:
        (app / "boards" / f"{board}.overlay").write_text(overlay)

    # 5. configure + build.
    build_dir = work / "build"
    env = _tool_env()
    _run([_conda_bin() / "cmake", "-B", build_dir, "-G", "Ninja",
          f"-DBOARD={board}", "-S", app], env=env)
    _run([_conda_bin() / "ninja", "-C", build_dir], env=env)
    elf = build_dir / "zephyr" / "zephyr.elf"
    if not elf.is_file():
        raise ZephyrModelError(f"build produced no elf at {elf}")
    return {"elf": elf, "app_dir": app, "build_dir": build_dir, "backend": backend,
            "ram_bytes": ram_bytes, **info}


def run_on_spike(elf: str | Path, *, harts: int = 2, isa: str = "rv64gcv_zfh_zvfh",
                 mem_bytes: int = 1 << 31, timeout: int = 3600) -> dict[str, Any]:
    """Run the Zephyr ELF on spike ``-pN``; parse the OUT/ARGMAX/METRIC/DONE markers."""
    cmd = [_spike.spike_path(), f"--isa={isa}", f"-p{harts}",
           f"-m{hex(0x80000000)}:{hex(mem_bytes)}", str(elf)]
    proc = subprocess.run([str(c) for c in cmd], capture_output=True, text=True,
                          timeout=timeout)
    console = proc.stdout + proc.stderr
    return _parse_console(console, proc.returncode)


def _parse_console(console: str, rc: int) -> dict[str, Any]:
    out_line = next((l for l in console.splitlines() if l.startswith("OUT ")), None)
    if out_line is None or "DONE" not in console:
        raise ZephyrModelError(
            f"run did not produce OUT/DONE (rc={rc}):\n{console[-2000:]}")
    parts = out_line.split()
    n = int(parts[1])
    bits = [int(x) for x in parts[2:2 + n]]
    flat = np.array([struct.unpack("<f", struct.pack("<I", b & 0xFFFFFFFF))[0]
                     for b in bits], dtype=np.float32)
    metrics, argmax, sumval = {}, None, None
    for l in console.splitlines():
        if l.startswith("METRIC "):
            _, kk, vv = l.split()
            metrics[kk] = int(vv)
        elif l.startswith("ARGMAX "):
            p = l.split()
            argmax = np.array([int(x) for x in p[2:2 + int(p[1])]], dtype=np.int64)
        elif l.startswith("SUM "):
            sumval = struct.unpack("<f", struct.pack("<I", int(l.split()[1]) & 0xFFFFFFFF))[0]
    return {"outputs": flat, "prefix": flat, "argmax": argmax, "sum": sumval,
            "metrics": metrics, "console": console}


def _gate(prefix: np.ndarray, references) -> dict:
    """Multi-tier accuracy gate for an int8 (W8A8) run. ``references`` is either a single
    fp32 reference array (legacy: one ``cos``/``rel``/``ok`` at the strict fp32 threshold) or
    a ``{tier: array}`` dict. Tiers (literature-backed for W8A8 transformers — I-BERT /
    SmoothQuant put W8A8 at cos 0.99–0.999 vs fp32):
      * ``w8a8`` — vs a torch/host W8A8 reference: T1 cos > 0.999 AND rel < 1e-2 (the int8
        datapath is faithful to the W8A8 math it intends to run);
      * ``fp32`` — vs the fp32 golden: T2 cos > 0.99 AND top-1 argmax matches (the quantization
        degradation is within the accepted band).
    ``ok = T1 or T2``. Emits per-tier ``<tier>_cos``/``<tier>_rel``/``<tier>_argmax`` keys."""
    pref = np.asarray(prefix, dtype=np.float32).ravel()
    if not isinstance(references, dict):
        references = {"fp32": references}
    out: dict[str, Any] = {}
    k = len(pref)
    for tier, ref in references.items():
        if ref is None:
            continue
        r = np.asarray(ref, dtype=np.float32).ravel()[:k]
        rel = float(np.abs(pref - r).max()) / max(1e-9, float(np.abs(r).max()))
        cos = float((pref @ r) / (np.linalg.norm(pref) * np.linalg.norm(r) + 1e-12))
        out[f"{tier}_cos"] = cos
        out[f"{tier}_rel"] = rel
        out[f"{tier}_argmax"] = bool(int(np.argmax(pref)) == int(np.argmax(r)))
    t1 = out.get("w8a8_cos", 0.0) > 0.999 and out.get("w8a8_rel", 1.0) < 1e-2
    t2 = out.get("fp32_cos", 0.0) > 0.99 and out.get("fp32_argmax", False)
    # legacy single-reference callers gate at the strict fp32 threshold
    legacy = "w8a8" not in out and out.get("fp32_cos", 0.0) > 0.9999 and out.get("fp32_rel", 1.0) < 1e-3
    out["cos"] = out.get("w8a8_cos", out.get("fp32_cos"))
    out["rel"] = out.get("w8a8_rel", out.get("fp32_rel"))
    out["ok"] = bool(t1 or t2 or legacy)
    return out


def run_on_firesim(elf: str | Path, *, reference: np.ndarray | None = None,
                   references: dict | None = None,
                   timeout: int = 900, queue: bool = True,
                   firesim_root: str | None = None, firesim_env: str | None = None
                   ) -> dict[str, Any]:
    """Run a ``chipyard_riscv64`` ELF on FireSim by reusing ModelBlaster's queue-safe
    ``validation.firesim_runner.run_firesim`` (single physical FPGA → always go through
    the queue). Parses our OUT/METRIC/DONE markers from the captured uartlog and (when a
    reference is given) gates ``cos``/``rel``. Requires the firesim env activated and the
    queue daemon up (see module doc / the FireSim section of the plan)."""
    import sys
    mb = os.environ.get("MERLIN_MODELBLASTER", "/path/to/ModelBlaster")
    for p in (f"{mb}/src", mb):
        if p not in sys.path:
            sys.path.insert(0, p)
    env = os.environ
    fr = firesim_root or env.get("FIRESIM_ROOT", "/path/to/chipyard/sims/firesim")
    fe = firesim_env or env.get("FIRESIM_ENV", "/path/to/chipyard/env.sh")
    # Run under OUR FireSim workload name, not ModelBlaster's. firesim_runner reads
    # FIRESIM_WORKLOAD_NAME into a module constant AT IMPORT, so this must be set before
    # the (lazy) import below. The workload def lives at deploy/workloads/merlin-oscar.json.
    os.environ.setdefault("FIRESIM_WORKLOAD_NAME", "merlin-oscar")
    # The firesim-queue records the SUBMITTER's env, and the daemon runs the job (incl.
    # `firesim kill`, which SSHes to localhost) with it. If we submit from a session whose
    # SSH_AUTH_SOCK points at a dead/empty agent (e.g. a VS Code agent), fabric can't
    # authenticate and every run aborts at the kill step. Point at the FireSim agent (the
    # one the daemon uses, holding the localhost key) when ours is missing/dead.
    _fs_sock = os.environ.get("FIRESIM_SSH_AUTH_SOCK", "/tmp/firesim_ssh_agent.sock")
    _cur = os.environ.get("SSH_AUTH_SOCK", "")
    if os.path.exists(_fs_sock) and (not _cur or not os.path.exists(_cur)):
        os.environ["SSH_AUTH_SOCK"] = _fs_sock
    # Tag our jobs with a distinct project so they are unmistakably separable from other
    # workflows on the shared queue (e.g. ModelBlaster's xpurt_demo runs) — and never
    # confused for cancellation. firesim_runner reads FIRESIM_PROJECT (default modelblaster).
    os.environ.setdefault("FIRESIM_PROJECT", "merlin-oscar")
    if queue:
        os.environ["FIRESIM_QUEUE"] = "1"
        os.environ.setdefault("FIRESIM_QUEUE_TIMEOUT", str(timeout))
    try:
        from modelblaster.validation.firesim_runner import run_firesim  # type: ignore
    except ModuleNotFoundError:
        from validation.firesim_runner import run_firesim  # type: ignore

    uart = run_firesim(str(elf), models=None, firesim_root=fr, firesim_env=fe,
                       timeout=float(timeout))
    res = _parse_console(uart, 0)
    refs = references if references is not None else reference
    if refs is not None:
        res.update(_gate(res["prefix"], refs))
    return res


def build_and_run(model_dir: str | Path, work: str | Path, *, board: str = "spike_riscv64",
                  backend: str = "rvv", rvv_hart: int = 0, harts: int = 2, arena_mb: int = 64,
                  reference: np.ndarray | None = None, references: dict | None = None,
                  timeout: int = 3600, int8_compute: bool = False) -> dict[str, Any]:
    """Build the Zephyr image and (for spike) run + gate. ``references`` (a ``{tier: array}``
    dict, e.g. ``{"w8a8": ..., "fp32": ...}``) drives the multi-tier int8 gate; a single
    ``reference`` array keeps the legacy strict fp32 gate."""
    b = build_app(model_dir, work, board=board, backend=backend, rvv_hart=rvv_hart,
                  arena_mb=arena_mb, cpus=max(harts, rvv_hart + 1), int8_compute=int8_compute)
    result: dict[str, Any] = {"elf": str(b["elf"]), "app_dir": str(b["app_dir"]),
                              "ram_bytes": b["ram_bytes"]}
    if board != "spike_riscv64":
        return result  # FireSim path runs the elf separately (firesim_runner / queue)
    run = run_on_spike(b["elf"], harts=harts, mem_bytes=b["ram_bytes"], timeout=timeout)
    result.update(run)
    refs = references if references is not None else reference
    if refs is not None:
        result.update(_gate(run["prefix"], refs))
    return result
