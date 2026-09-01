"""Whole-model execution on **Zephyr** (SMP) — spike today, FireSim on the 2-tile board.

This is the bring-up bridge from merlin's bare-metal spike path
(:mod:`spike_model`) to a real RTOS image that runs on the FireSim
``GemminiAndOPUShuttleConfig`` SoC (tile 0 = scalar/Gemmini, tile 1 = Saturn-OPU
``rv64gcv`` vLen=128). NOTE: this whole-model Zephyr/FireSim SMP path deliberately
uses the 2-tile gemmini+OPU SoC (it needs the Saturn vector tile); it is a distinct
path from the Gemmini C0 RTL-certification oracle, which now runs the pure
``GemminiRocketConfig`` (single-tile Rocket host, same 16x16 int8 Gemmini core). It reuses the *entire* data-driven C runtime
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
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
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


def build_tool(name: str) -> Path | None:
    """Locate a host build tool (``cmake``, ``ninja``), or None.

    Prefer the chipyard conda env, because that is the version this flow is exercised against and
    pinning it keeps a machine with three cmakes reproducible. But fall back to PATH: these are ordinary
    host tools, not part of the RISC-V toolchain, and requiring them to live inside a *chipyard* checkout
    made the Zephyr path unbuildable on any machine that has cmake and ninja the normal way -- with no
    env var to say so, and an `available()` that reported the Zephyr tree as the problem.
    """
    pinned = _conda_bin() / name
    if pinned.is_file():
        return pinned
    from shutil import which
    found = which(name)
    return Path(found) if found else None


def _tool_env() -> dict:
    env = dict(os.environ)
    env["ZEPHYR_BASE"] = str(_zephyr_base())
    env["ZEPHYR_TOOLCHAIN_VARIANT"] = "zephyr"
    env["ZEPHYR_SDK_INSTALL_DIR"] = str(_sdk_dir())
    env["PATH"] = f"{_conda_bin()}:{env.get('PATH','')}"
    return env


def available() -> bool:
    """True when the Zephyr build + spike toolchain are present."""
    try:
        return (_zephyr_base().is_dir() and _sdk_dir().is_dir()
                and build_tool("cmake") is not None and build_tool("ninja") is not None
                and _spike.available())
    except Exception:  # noqa: BLE001
        return False


# Bounded wall clock for the build's clang/ld steps (this is the seam apply_rvv_package's spike/K1
# build routes through). A pathological schedule (e.g. an outer-product contraction at a large square
# regime) makes clang -O2 spin for many minutes on one object; in a serial beam that hangs the whole
# sweep. Time it out so the fork fails-closed as a build error the certify ladder records. Same
# MERLIN_COMPILE_TIMEOUT_S knob as the host/spike/K1 paths. run_on_spike/FireSim carry their own timeout.
# Default unified at 900s across all four compile wrappers (was 600). For a whole-model beam launch set
# MERLIN_COMPILE_TIMEOUT_S=3600; 0 (or empty) disables the ceiling.
_BUILD_CMD_TIMEOUT_S = int(os.environ.get("MERLIN_COMPILE_TIMEOUT_S", "900") or "0")


def _run(cmd: list, **kw) -> subprocess.CompletedProcess:
    kw.setdefault("timeout", _BUILD_CMD_TIMEOUT_S or None)
    try:
        proc = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, **kw)
    except subprocess.TimeoutExpired:
        raise ZephyrModelError(f"command timed out after {_BUILD_CMD_TIMEOUT_S}s "
                               f"(pathological compile): {' '.join(map(str, cmd))}")
    if proc.returncode != 0:
        raise ZephyrModelError(
            f"command failed: {' '.join(map(str, cmd))}\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")
    return proc


#: Innermost lane count the per-rank bounded vectorize uses; the tagging predicate and the
#: schedule arms in ``impr_features._VEC_RANK_ARMS`` must agree on it.
_VEC_RANK_LANES = 8

#: Caps for per-op register blocking. NR is the widest N tile a block may use (the champion's
#: cap); KC is carried through to the v3 recipe, which tiles K by 1 and does not use it. MR is
#: NOT capped here -- perop_blocks.DEFAULT_MR pins it at 1 for a measured instruction-selection
#: reason (the vfmacc.vf scalar-A form).
#:
#: 16 is the champion's value, measured on a VLEN=256 board. It is an ELEMENT COUNT, so it does not
#: scale with the vector length -- and a fixed element count on a wider unit is spent as a smaller
#: LMUL, not as more work. Measured on the same model built two ways:
#:
#:   VLEN=128:  e16,m2  / e8,m1  / e16,m1     -- 16 elements across one or two whole registers
#:   VLEN=512:  e16,mf2 / e8,mf4 / e16,mf4    -- the same 16 elements in HALF or a QUARTER of one
#:
#: i.e. the 512-bit machine issued the same number of vector instructions doing the same 16 elements
#: each as the 128-bit one. Three quarters of the datapath went unused.
_PEROP_NR_CAP = 16

#: The VLEN the champion cap above was measured at. Used only as the ratio's denominator, so this
#: board's tuned value is reproduced exactly rather than re-derived.
_PEROP_NR_CAP_REF_VLEN = 256


def perop_nr_cap(vlen: int | None) -> int:
    """The N-tile cap for a board with this vector length.

    Scales the measured cap with the vector length so a wider unit gets a wider tile (the same LMUL)
    instead of the same tile at a smaller LMUL. Deliberately **scales up only**: at or below the
    reference width the champion's value is returned unchanged, so no already-validated configuration
    moves as a side effect of this function existing.

    What is verified is the EMITTED CODE -- that the wider cap turns fractional-LMUL ops into whole-
    register ones. Whether that is faster on a given chip is a cycle question we cannot answer without
    that chip: a wider tile also raises register pressure, and past some width it spills. So this
    changes what a wide-VLEN board is offered, and the cycle claim belongs to whoever runs it.
    """
    if not vlen or vlen <= _PEROP_NR_CAP_REF_VLEN:
        return _PEROP_NR_CAP
    return _PEROP_NR_CAP * (int(vlen) // _PEROP_NR_CAP_REF_VLEN)


#: M-tile cap for a per-op register block. NOT the same kind of number as the NR cap above: NR is a
#: LANE count (a vector-length question), MR is an ACCUMULATOR-ROW count (a register-file question) --
#: MR rows of NR lanes must sit in registers across the whole K loop, so MR is bounded by pressure, not
#: by VLEN. 4 is the point this repo has measured end to end, so it is the point offered here.
#:
#: ``perop_blocks.DEFAULT_MR`` is 1 and stays 1 -- it is the module default for callers that pass no
#: cap. The reason IT gives for pinning 1 (at MR>1 the A column is rebuilt with a vmv/vslideup lane
#: ladder) was TRUE and is now FIXED: ``accum_microkernel`` admits the integer element types and sinks
#: integer widenings below the lane extract, so int8 MR>1 reaches ``vwmacc.vx`` from a scalar (measured:
#: ``vrgather.vi`` 4 -> 0, ``vmacc.vv`` -> ``vwmacc.vx``). A SECOND defect then hid the win -- the
#: per-tile ``@memrefCopy`` self-copy, worth 187,520 instructions at 64^3 in BOTH dtypes -- and that is
#: now implied by every MR>1 recipe (``impr_features._tile_epilogue_hygiene``).
#:
#: With both fixed, MEASURED on the live K1 (VLEN=256, 128^3, interleaved same-session arms, n=3 x 5
#: iterations, min-of-n, cos-gated) -- per-op MR=1 vs MR=4 on the same lowering:
#:
#:   f32   3,783,252 -> 1,181,928 cyc   2,388,555 -> 746,588 ns   3.20x   cos 1.000000
#:   int8  1,930,566 -> 1,223,376 cyc   1,219,091 -> 772,504 ns   1.58x   cos 0.999959 (both arms)
#:
#: Both are far outside the board's 2.6% noise band. Raising the cap does not FORCE MR=4 anywhere:
#: ``from_strategy._rvv_best_block`` treats it as an upper bound and returns only a divisor of the
#: observed gcd(M) that its lowering predicate accepts, so a shape with no clean M-tile still gets
#: MR=1. That is why this is a cap and not a pin.
_PEROP_MR_CAP = 4


def perop_mr_cap() -> int:
    """The M-tile cap offered to the per-op block policy.

    Deliberately takes no ``vlen``: MR is a register-file bound, not a vector-length one, so unlike
    :func:`perop_nr_cap` there is nothing about the board's width to scale by. See the measurements on
    :data:`_PEROP_MR_CAP`.
    """
    return _PEROP_MR_CAP


_PEROP_KC = 16

DEFAULT_RAM_BYTES = 256 * 1024 * 1024   # spike/chipyard `ram0` default (0x10000000)

# Above this, a weights blob linked into the image's .data overflows the medany ±2GB
# PC-relative window (Zephyr's own .text<->.bss refs break). Past it we switch to
# "external weights": the blob lives in its own DT memory-region at a fixed high absolute
# address, referenced by integer-constant base (no far symbol), keeping ram0 compact.
LINK_LIMIT = 1900 * 1024 * 1024
EXT_RAM0_BYTES = 0x40000000             # 1 GB ram0 (code + activation arena) in ext mode
EXT_WEIGHTS_BASE = 0xC0000000           # weights region origin (3 GB), right after ram0  # derived-ok: this backend's ext-mode layout origin, chosen together with DRAM_END below
DRAM_END = 0x80000000 + 16 * 1024**3    # FireSim WithExtMemSize = 16 GB at 0x80000000
# The WEIGHTS region uses a 2-cell DT container (#address/size-cells=2) so its reg can
# express a 64-bit base+size — handling blobs > 4 GB (tiny 4.1 G, pi05 fp32 13 G), not just
# the 1-cell <4 GB case. The only hard cap is physical DRAM: weights at EXT_WEIGHTS_BASE
# must end before the 16 GB DRAM end (leave ram0 + margin).
EXT_MAX_WEIGHTS = DRAM_END - EXT_WEIGHTS_BASE - (256 * 1024 * 1024)   # ~14.75 GB


#: What one heap allocation looks like in the emitted LLVM IR. The lowered model reaches the C library
#: through this symbol, so a call to it is an allocation the arena has to hold.
_ALLOC_SYMBOL = "@malloc("


def allocation_bytes(ll_path: str | Path) -> tuple[int, int]:
    """``(total_constant_bytes, dynamic_calls)`` requested by the lowered model's allocations.

    WHY THE SUM AND NOT THE PEAK. :func:`mlir_query.activation_peak_bytes` reports LIVE bytes and says
    itself that it is a lower bound. Whether the live figure is the one that has to fit depends on the
    allocator underneath, and one of ours does not reclaim at all (the bare-metal arena's ``free`` is a
    no-op), so there the arena must hold every byte ever requested. Sizing to the sum is the bound that
    holds under both, and it is derived from the IR that will actually run rather than assumed.

    Parsed STRUCTURALLY -- ``partition`` on the call, then on the argument's type -- not by pattern. The
    IR spells the same call several ways (``call``, ``tail call``, with and without ``noalias``, with a
    trailing attribute group), and a pattern narrow enough to be readable silently misses one spelling,
    which here would under-report the requirement and produce an image that dies in its tail.

    ``dynamic_calls`` counts allocations whose size is a value rather than a literal. They are REPORTED,
    never assumed to be zero: their bytes are genuinely unknown at build time, and a caller that needs a
    guarantee has to know the answer is incomplete.
    """
    total = dynamic = 0
    for line in Path(ll_path).read_text(errors="replace").splitlines():
        _, sym, rest = line.partition(_ALLOC_SYMBOL)
        if not sym:
            continue
        args, close, _ = rest.partition(")")
        if not close:
            continue
        # `i64 1024`, and also `i64 noundef 1024` -- the parameter attributes clang inserts sit BETWEEN
        # the type and the value, so the size is the last token, not the second one. Reading the second
        # token classified an attributed call as dynamic, which under-reports the requirement.
        tokens = args.split()
        if len(tokens) < 2 or not tokens[0].startswith("i") or not tokens[-1].isdigit():
            dynamic += 1
            continue
        total += int(tokens[-1])
    return total, dynamic


def _ram_for_weights(weights_bytes: int, activation_bytes: int | None = None,
                     allocation_bytes_total: int | None = None) -> int:
    """RAM-region size to hold the weights blob (linked into .data) plus an activation
    arena (the leftover, claimed by ARENA_SIZE=-1). Headroom scales with the model
    (30% of weights + 128 MB) rather than a fat fixed floor, so small models stay at the
    256 MB default — important on FireSim, where the whole-model image only boots reliably
    at the stock `ram0` size and an over-large region wedges the boot (no uartlog). Models
    that genuinely need more grow the region; rounded up to 16 MB.

    TWO ESTIMATES OF THE SAME QUANTITY, and the measured one wins.

    ``activation_bytes`` (from :func:`mlir_query.activation_peak_bytes`) is LIVE bytes inferred from the
    MLIR. It exists because a flat 128 MB of headroom does not always cover a model's working set:
    whisper_tiny's encoder attention peaks at 210 MB live where the weights-scaled formula left it a
    163 MB arena, i.e. a provisioning failure on a board with enough physical DRAM.

    ``allocation_bytes_total`` (from :func:`allocation_bytes`) is what the EMITTED IR actually asks the
    heap for, summed. It is the better number wherever it exists, in both directions:

    * It is an UPPER bound on live bytes -- every live byte was allocated -- so it cannot under-provision
      the way a liveness estimate can when bufferization inserts copies the estimate never saw.
    * It prices only what comes out of the arena. The MLIR estimate prices tensors that never become heap
      allocations at all, and it can therefore be enormously PESSIMISTIC: measured on a 12-layer Gemma
      section, the estimate said 2827 MB while the emitted IR's allocations total 1533 MB -- an estimate
      larger than the sum of every allocation the program makes, which is impossible for real arena
      demand. Sizing from it asked for a 4416 MB region on a 4096 MB design and refused to build a model
      that fits in 3136 MB.

    So when the measured total is available it REPLACES the estimate rather than joining it in the max.
    That is deliberately not monotone with the older formula: a model whose estimate was pessimistic gets
    a smaller region, which is the whole point. The 128 MB slack stays on top of whichever number is used,
    because neither covers allocator fragmentation and because an allocation whose size is computed rather
    than literal is not in the measured total (:func:`allocation_bytes` reports how many).

    A shortfall is not silent, which is what makes this safe to tighten: the image links a malloc guard
    that prints ``FAIL alloc bytes=<n> op=<id>`` and terminates in the protocol, so a returned log names
    the allocation that did not fit."""
    headroom = (weights_bytes * 3) // 10 + 128 * 1024 * 1024
    demand = int(allocation_bytes_total or 0) or int(activation_bytes or 0)
    if demand:
        headroom = max(headroom, demand + 128 * 1024 * 1024)
    total = weights_bytes + headroom
    align = 16 * 1024 * 1024
    total = ((total + align - 1) // align) * align
    return max(DEFAULT_RAM_BYTES, total)


def _prepare_model_mlir(mlir_path: Path, work: Path, *, int8_compute: bool = False,
                        tag_vec_ranks: bool = False,
                        vec_lanes: int = _VEC_RANK_LANES) -> Path:
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
        # transcendentals -> integer/RVV), via the quant-pass registry (byte-identical default set).
        # lower_quant_ext stays AFTER as the f32 fallback for any dequant the int8 passes did not
        # convert (nonzero-zp, embeddings).
        from ...llvmlower.quant_passes import apply_quant
        apply_quant(module)
    lower_quant_ext(module)
    lower_bf16_matmul_f32acc(module)
    fix_bool_sitofp(module)
    # PER-RANK VECTORIZE TAGGING (default OFF -> baseline byte-identical): tag each all-parallel
    # (non-reduction) linalg.generic with `merlin.vec_r{rank}` so the transform schedule can
    # BOUNDED-vectorize the scalar non-matmul ops by rank (the win lever for openvla — ~900ms of scalar
    # parallel generics -> ~110ms vectorized). Reductions (softmax/norm) skipped (need stable form).
    #
    # Driven by ``tag_vec_ranks`` — which build_app sets from the FEATURE
    # ``impr_features.VEC_NONCONTRACTION_NAME`` — so the lever is selectable by the tuning loop instead
    # of only by an env var it cannot set. MERLIN_VEC_RANK still forces it on for a manual A/B.
    import os as _os
    if tag_vec_ranks or _os.environ.get("MERLIN_VEC_RANK"):
        from xdsl.dialects.builtin import UnitAttr
        n_tag = skip_gather = skip_extent = skip_math = 0
        for op in module.walk():
            if op.name != "linalg.generic":
                continue
            its = str(op.properties.get("iterator_types", ""))
            if "reduction" in its or "parallel" not in its:
                continue
            rank = its.count("iterator_type")
            if not 2 <= rank <= 4:
                continue
            # A DATA-DEPENDENT GATHER (a tensor.extract / memref.load in the body — the im2col and
            # pad-index generics) has no affine access to vectorize: `structured.vectorize` fails the
            # whole pipeline with "Attempted to vectorize, but failed" rather than declining the op.
            # Measured on deepjscc: tagging these is what made the lever unusable.
            body = op.regions[0].blocks[0] if op.regions and op.regions[0].blocks else None
            if body is not None and any(inner.name in ("tensor.extract", "memref.load")
                                        for inner in body.ops):
                skip_gather += 1
                continue
            # A TRANSCENDENTAL in the body (math.exp/erf/tanh, i.e. a sigmoid/GELU/SiLU activation) has
            # no vector form here: convert-math-to-libm scalarizes it back into per-lane extracts + libm
            # calls, and that pass runs AFTER vector->LLVM, so those extracts reach translation with
            # nothing left to lower them ("missing LLVMTranslationDialectInterface ... vector.extract",
            # measured on deepjscc's sigmoid). Those ops are the OTHER lever's job --
            # `vectorized_transcendental_activation` rewrites them to arith polynomials BEFORE
            # vectorization -- so the two features compose instead of fighting.
            if body is not None and any(inner.name.startswith("math.") for inner in body.ops):
                skip_math += 1
                continue
            # The innermost extent must be a whole multiple of the vector width. A partial tail means a
            # MASKED parallel dim, which does not lower at all on the integer path (see
            # mining.from_strategy._rvv_blocking_lowers) -- the same predicate the register block obeys.
            try:
                shape = list(op.results[0].type.get_shape())
            except Exception:                                      # noqa: BLE001
                shape = []
            if not shape or shape[-1] % vec_lanes:
                skip_extent += 1
                continue
            op.attributes[f"merlin.vec_r{rank}"] = UnitAttr()
            n_tag += 1
        print(f"[vec_rank] tagged {n_tag} all-parallel generics for bounded vectorize "
              f"(skipped {skip_gather} gathers, {skip_math} with a transcendental body, "
              f"{skip_extent} on a non-multiple innermost extent)")
    out = work / "model.prepared.mlir"
    out.write_text(to_text(module))
    return out


@dataclass(frozen=True)
class MatrixRouting:
    """Which matrix extension a build routes contractions to, and in which configuration.

    Carried as one object so the tile edge has exactly ONE source. The alternative -- deriving the edge
    here from the caller's ``vlen`` and again at object-build time from the config's Scala -- is two
    statements of the same fact that can disagree, and the disagreement would be silent: the selector
    would choose contractions for a geometry the compiled kernel does not have.

    ``select`` overrides the default tile-filling decision, which is the seam the cost model and the
    e-graph plug into. Nothing here decides profitability on its own.
    """

    unit: str                                       # a block in llvmlower/opu_shim.CONTRACT_PATH
    config: str                                     # the elaborated hardware configuration
    select: "Callable[[Any], bool] | None" = None

    def tile_edge(self) -> int:
        """The edge, read from this configuration's own declaration.

        Through the contract's own accessor rather than re-resolving the paths here: this used to rebuild
        them, which meant two places had to agree about where a unit's configs live -- and when the contract
        grew a second declaration site they stopped agreeing.
        """
        from ...llvmlower import opu_shim
        return opu_shim.load_contract(self.unit).geometry(self.config)[0]

    def selector(self) -> "Callable[[Any], bool]":
        from ...llvmlower.passes_opu import tile_filling_selector
        return self.select if self.select is not None else tile_filling_selector(self.tile_edge())


def prepare_for_lowering(mlir_path: Path, work: Path, *, int8_compute: bool = False,
                         features: "frozenset[str] | None" = None,
                         blocking: bool = True, harts: int = 1,
                         vlen: int | None = None,
                         matrix: "MatrixRouting | None" = None,
                         device: "Any | None" = None) -> tuple[Path, frozenset[str]]:
    """``(prepared_mlir, concrete_features)`` — everything that must happen to a captured module
    before ``lower_model_file``, shared by every whole-model backend.

    This is deliberately NOT Zephyr-specific: a bare-metal image (``spike_model.build``) has to apply
    the identical preparation or it computes different numbers from the same bundle. That is not
    hypothetical — the bare-metal path, lowering ``model.mlir`` raw, scored ``cos 0.925`` on a bundle
    the prepared path gets bit-exact, because without ``int8_compute`` it runs a dequantize-to-f32
    datapath and without the per-op tags every contraction falls to scalar loops.

    ``blocking`` gates the per-op register-blocking step (opt-in via the ``PEROP_BLOCK_NAME`` sentinel
    in ``features``): each contraction's block is derived from the PREPARED IR — the geometry the
    pipeline will actually see — then the ops are specialized + tagged and the sentinel is swapped for
    the concrete, table-specific feature. Doing both here is what keeps the schedule's arms and the
    IR's tags in sync: they are generated from ONE table. Skipping the tagging leaves nothing for any
    arm to match and every contraction silently falls to convert-linalg-to-loops (measured: deepjscc
    484M -> 1242M cycles at bit-identical output — a 2.56x regression that looks like a bad block but
    is an untagged build).
    """
    from ...llvmlower.impr_features import vec_noncontraction_lanes as _vec_lanes
    features = frozenset(features or frozenset())
    _lanes = _vec_lanes(features)
    prepared = _prepare_model_mlir(mlir_path, work, int8_compute=int8_compute,
                                   tag_vec_ranks=_lanes is not None,
                                   vec_lanes=_lanes or _VEC_RANK_LANES)
    # MATRIX-UNIT ROUTING, before the register blocking below: a contraction that has become a call is no
    # longer on the vector path, so the block table must be derived from the IR that remains. Doing it the
    # other way round would tag ops that no longer exist and leave the routed ones double-claimed.
    from ...llvmlower.impr_features import OPU_MATMUL_NAME
    if OPU_MATMUL_NAME in features:
        if matrix is None:
            # Silently not routing would be indistinguishable from a feature that did nothing, and the
            # model would grade correctly while reporting a capability it never used.
            raise ValueError(
                f"feature {OPU_MATMUL_NAME!r} is enabled but no `matrix=` routing was supplied, so the "
                "unit and configuration to route to are unknown; pass MatrixRouting(unit=..., config=...)")
        from ...llvmlower.passes_opu import rewrite_prepared_file
        routed = rewrite_prepared_file(prepared, work, select=matrix.selector(),
                                       tile_edge=matrix.tile_edge())
        print(f"[matrix] routed {routed.count} contraction(s) to {matrix.unit} "
              f"({matrix.config}) across {len(routed.signatures)} signature(s)"
              + (f"; declined: {[why for _w, why in routed.skipped]}" if routed.skipped else ""))
        # A CUSTOM selector may route contractions the default tile-filling rule refuses. That is
        # allowed -- it is the seam the cost model plugs into -- but it must never be SILENT.
        # A whole-model Gemma image was built this way: `select=` dropped the M requirement, 183
        # contractions ran at M=8 on a 64-lane unit, and neither the build log nor the sidecar nor
        # the measurement said so. The run cost 11.6 hours before the output could be graded.
        sub = routed.sub_tile()
        if sub:
            worst = min(sub, key=lambda r: min(r.m, r.n))
            print(f"[matrix] WARNING: {len(sub)} of {routed.count} routed contraction(s) do NOT fill "
                  f"a {matrix.tile_edge()}-lane tile in both parallel dimensions -- the default "
                  f"tile_filling_selector would refuse them. Smallest: {worst.symbol} "
                  f"m={worst.m} n={worst.n} k={worst.k} ({worst.fqn or 'unnamed'}). "
                  f"This build supplied select= explicitly; the sidecar records it under "
                  f"`routing_rule`.")

    # DEVICE OFFLOAD, for the same reason and in the same place as the matrix routing above: a
    # contraction that has become a call is no longer on the vector path, so the register-block table
    # below must be derived from the IR that REMAINS. Inert unless a routing was supplied, and inert
    # again unless that routing carries a selector -- the placement decision is made elsewhere
    # (merlin.system.place) and passed in, never taken here.
    if device is not None and getattr(device, "select", None) is not None:
        from ...llvmlower.device_offload import rewrite_prepared_file as _dev_rewrite
        moved = _dev_rewrite(prepared, work, device.device, select=device.select)
        print(f"[device] routed {moved.moved} contraction(s) to {device.device} across "
              f"{len(moved.signatures)} signature(s)"
              + (f"; declined: {[why for _s, why in moved.skipped]}" if moved.skipped else ""))

    if not blocking:
        return prepared, features
    from ...llvmlower import perop_blocks as _pb
    from ...llvmlower.impr_features import (PEROP_BLOCK_NAME, PEROP_NR_FILL_NAME,
                                            ensure_perop_block)
    # The N-fill request is a SEARCH KNOB, off by default, because its sign is model-dependent: on the
    # K1 it measured 1.159x FASTER on spectformer int8 and 1.196x SLOWER on small_llama int8, both far
    # outside the 2.6% band. The accumulator is i32, not i8, so at VLEN=256 an NR=16 tile is already
    # LMUL m4 and widening pushes it to m8 and spills (decoded at 128^3: 0 -> 6 accumulator spill ops).
    # Passing vlen here is the ONLY thing that turns it on, so the default path is byte-identical.
    nr_fill_vlen = vlen if PEROP_NR_FILL_NAME in features else None
    features = features - {PEROP_NR_FILL_NAME}
    if PEROP_BLOCK_NAME in features:
        from ...kernels.shapes import contraction_shapes as _cshapes
        # The N-tile cap follows the board's vector length: a fixed element count on a wider unit
        # is spent as a smaller LMUL rather than as more work per instruction (measured: the same
        # model emitted e8,m1 at VLEN=128 and e8,mf4 -- a quarter of a register -- at VLEN=512).
        # `vlen=nr_fill_vlen` is None unless the N-fill knob asked for it (see above). When it is set,
        # each contraction's N cap is widened for ITS OWN narrowest element width; measured effect on
        # the block tables of the models on disk is MAC-weighted NR 16.00 -> 32.00 on every int8 model
        # and UNCHANGED on fp32 -- see perop_blocks.nr_cap_for_dtypes.
        table = _pb.block_table(_cshapes(prepared), mr_cap=perop_mr_cap(),
                                nr_cap=perop_nr_cap(vlen), harts=harts, vlen=nr_fill_vlen)
        if table:
            prepared = _pb.tag_prepared_mlir(prepared, table, work=work)
            features = (features - {PEROP_BLOCK_NAME}) | {ensure_perop_block(table, _PEROP_KC)}
        else:
            features = features - {PEROP_BLOCK_NAME}
    return prepared, features


# ---- generated Zephyr-app sources --------------------------------------------------

#: Seconds between `ALIVE` lines in a debug image. Slow enough to be free on a 115200-baud console at
#: 50 MHz, fast enough that nobody has to decide whether to keep waiting.
DEBUG_HEARTBEAT_S = 5


class _DebugHarness:
    """The C fragments a debug image adds to the generated harness, or empty strings for a delivery
    image. Grouped in one object so `_main_c`'s f-string stays readable and so the OFF case is
    provably inert (every field is "")."""

    __slots__ = ("decls", "boot", "post_init", "pre_run", "post_run", "post_dump", "stacks")

    def __init__(self, decls="", boot="", post_init="", pre_run="", post_run="",
                 post_dump="", stacks=""):
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
""")


def _main_c(rvv_hart: int, dump_cap: int = 4096, weights_base: int | None = None,
            n_harts: int = 1, iters: int = 1, warmup: int = 0,
            omp_threads: int | None = None, build_hash: str = "",
            console: str = "htif", debug: bool = False,
            dram_base: int = 0x80000000, region_bytes: int = 0) -> str:
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
    """
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
    halt = ("sys_reboot(SYS_REBOOT_COLD)" if console == "htif"
            else "for (;;) k_cpu_idle()")
    omp_include = '#include "libomp_zephyr.h"\n' if n_harts > 1 else ""
    omp_init = (f'  int nth = merlin_omp_init({nth});\n'
                f'  printk("METRIC omp_threads %d\\n", nth);\n'
                if n_harts > 1 else "")
    # DEBUG BUILD. Everything below is off by default and costs nothing in a delivery image; it exists
    # because the two boards we ship to are ones we cannot attach to, and both of the last two rounds
    # came back as "it printed three lines and stopped". Three lines is not a diagnosis. With these on,
    # a single returned console log names the stage it reached, the hart it was on, the op index it was
    # inside, and the fault that stopped it -- so one round trip replaces a guess.
    dbg = _debug_harness(debug, dram_base, region_bytes, n_harts)
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
   * enforces VS; spike and the Saturn RTL do not enforce it, which is why every simulated run passed.
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
{dbg.post_dump}  /* Terminal sentinel reused from the ModelBlaster FireSim runner: its
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
    {halt};
    return 1;
  }}
  k_thread_start(t);
  k_sem_take(&merlin_done, K_FOREVER);
  {halt};
  return 0;
}}
"""


def _chipyard_cpu_overlay(n_harts: int, max_dt_cpus: int = 8) -> str:
    """Disable the DT CPUs the SoC does not have.

    ``chipyard-riscv64.dtsi`` declares 8 CPUs, but a given bitstream/sim has however many
    tiles its config built; Zephyr's SMP boot hangs trying to wake harts that do not exist.
    This is GENERATED from ``n_harts`` rather than copied from
    ``samples/merlin_hetero_runner/boards/chipyard_riscv64.overlay``, which hard-disables
    cpu@2..7 for the 2-tile FireSim SoC and silently caps every image at 2 harts — invisible
    but fatal for a 4-tile multicore Saturn run.
    """
    if n_harts >= max_dt_cpus:
        return ""
    disabled = "".join(f"\t\tcpu@{i} {{ status = \"disabled\"; }};\n"
                       for i in range(n_harts, max_dt_cpus))
    return ("/*\n * Generated by merlin.runtime.backends.zephyr_model.\n"
            f" * SoC has {n_harts} tile(s); disable the other DT CPUs so SMP boot does not\n"
            " * try to wake harts that do not exist.\n */\n\n"
            "/ {\n\tcpus {\n" + disabled + "\t};\n};\n")


def _kconfig_has(symbol: str) -> bool:
    """Does THIS Zephyr tree define ``CONFIG_<symbol>``?

    Zephyr versions differ in their vector Kconfig, and setting a symbol a tree does not have is a hard
    build failure ("attempt to assign the value 'y' to the undefined symbol ..." -> "Aborting due to
    Kconfig warnings"), not a warning. Measured: `RISCV_V_KERNEL_ONLY` exists on the zephyr-chipyard-sw
    `dev` Zephyr and is ABSENT in the one the `kodiak` branch pins, so a config written for one tree
    cannot build for the other. Probing the tree is the difference between targeting a board and
    assuming its Zephyr matches ours.
    """
    return symbol in _kconfig_symbols()


@lru_cache(maxsize=1)
def _kconfig_symbols() -> frozenset[str]:
    """Every ``config <NAME>`` this tree declares, across the subsystems we configure from.

    Scoped to specific roots rather than the whole tree: a full rglob walks tens of thousands of files
    for an answer we need on every build. The roots are the ones merlin actually sets symbols in --
    `arch/` for the ISA and vector options, `kernel/` and `subsys/debug/` for the diagnostic ones a
    debug image turns on. A symbol outside them reads as absent, which is the safe direction: we
    comment it out rather than emit it, and an unknown symbol is a HARD build failure ("attempt to
    assign the value 'y' to the undefined symbol"), never a warning.
    """
    base = _zephyr_base()
    roots = [base / "arch", base / "kernel", base / "subsys" / "debug"]
    found: set[str] = set()
    try:
        for root in roots:
            if not root.is_dir():
                continue
            for f in root.rglob("Kconfig*"):
                if not f.is_file():
                    continue
                for line in f.read_text(errors="replace").splitlines():
                    s = line.strip()
                    # `config NAME` and `menuconfig NAME` both declare a settable symbol.
                    head, _, name = s.partition(" ")
                    if head in ("config", "menuconfig") and name and " " not in name:
                        found.add(name)
    except Exception:                                            # noqa: BLE001
        return frozenset(found)
    return frozenset(found)


@lru_cache(maxsize=None)
def _kconfig_default_int(symbol: str) -> int | None:
    """The tree's own ``default <int>`` for an ``int``-typed Kconfig symbol, or None.

    Read from the tree rather than written here because it is the tree that decides what is safe: see
    `_vector_max_len_bits` for why the default is a FLOOR and not just a fallback.
    """
    base = _zephyr_base()
    roots = [base / "arch", base / "kernel", base / "subsys" / "debug"]
    for root in roots:
        if not root.is_dir():
            continue
        for f in root.rglob("Kconfig*"):
            if not f.is_file():
                continue
            in_sym = False
            for line in f.read_text(errors="replace").splitlines():
                s = line.strip()
                head, _, name = s.partition(" ")
                if head in ("config", "menuconfig"):
                    in_sym = name == symbol
                    continue
                if not in_sym:
                    continue
                key, _, val = s.partition(" ")
                if key == "default":
                    val = val.strip()
                    if val.isdigit():
                        return int(val)
                    return None                  # a conditional/symbolic default: no single number
    return None


def _vector_max_len_bits(brd) -> int:
    """Bits to declare for the per-thread vector save area, floored at the tree's own default.

    `CONFIG_RISCV_VECTOR_MAX_LEN` sizes a FIXED buffer -- `vreg[32][MAX_LEN/8]` in
    `arch/riscv/thread.h` -- but the code that fills it takes its length from the HARDWARE:
    `z_riscv_vstate_save` does `vsetvli %0, x0, e8, m8` and then four `vse8.v`, i.e. it stores
    32 x vlenb bytes with no clamp and no check against the buffer it was given. So the two numbers are
    not symmetric. Declaring MORE than the silicon has costs RAM per thread and nothing else.
    Declaring LESS is an unchecked buffer overrun on every context switch, of
    `32 * (vlenb_hw - MAX_LEN/8)` bytes, into whatever the linker put after that thread struct.

    That is not hypothetical: a descriptor saying `vlen=128` against silicon whose own probe reports
    `vlenb 32` (VLEN 256) overran `z_idle_threads[1]` by 512 bytes into the adjacent `z_main_thread`,
    zeroing its name and `stack_info.start` -- a thread that has never executed a vector instruction
    has a zeroed register file, so the overrun writes zeros. The first tick after `z_smp_init()` then
    loaded from address 0. No simulator gate can see it, because a spike run is given the VLEN we
    declared: configured and actual agree by construction and the overrun cannot happen there.

    So: take the board's declared VLEN, but never emit below the tree's own default for the symbol.
    A wrong-but-larger declaration wastes memory; a wrong-but-smaller one corrupts the kernel.
    """
    want = int(brd.vector_max_len)
    floor = _kconfig_default_int("RISCV_VECTOR_MAX_LEN")
    return max(want, floor) if floor else want


def _debug_conf() -> str:
    """Kconfig a DEBUG image adds. Every symbol here exists in the pinned tree and was checked before
    being emitted -- an unknown symbol aborts the build, and a debug image that will not build is worse
    than no debug image.

    `EXCEPTION_DEBUG` is already `y` by default (it only needs PRINTK), so the register dump is not
    what these buy. What they buy is: a stack overflow reported as a stack overflow instead of as
    corruption somewhere else entirely (`STACK_SENTINEL`), the high-water measurement that tells us
    whether the 8 MB worker stack is 8 MB of evidence or 8 MB of caution (`INIT_STACKS` +
    `THREAD_STACK_INFO`), assertions that fire instead of being compiled away (`ASSERT`), and a thread
    NAME in the fault line so "which of the pool workers" is answerable (`THREAD_NAME`).
    """
    want = [("EXCEPTION_DEBUG", "y"), ("ASSERT", "y"), ("THREAD_NAME", "y"),
            ("THREAD_STACK_INFO", "y"), ("INIT_STACKS", "y"), ("THREAD_MONITOR", "y"),
            ("STACK_SENTINEL", "y")]
    out = ["\n# DEBUG image: diagnosis over size. Not shipped in a delivery build."]
    for sym, val in want:
        if _kconfig_has(sym):
            out.append(f"CONFIG_{sym}={val}")
        else:
            out.append(f"# CONFIG_{sym} not defined by this Zephyr tree")
    return "\n".join(out) + "\n"


def image_cpus(brd, harts: int, rvv_hart: int = 0) -> int:
    """``CONFIG_MP_MAX_NUM_CPUS`` for an image that fans out over ``harts``.

    One rule, in one place, because there were two. The packager computed this differently on its gated
    path (``max(2, harts, 2)``) and on its build-only path (``max(harts, brd.harts)``), so the same model
    at the same hart count got a different CPU count depending on whether it was going to be simulated --
    and on a 3-hart board the ungated build of a ONE-hart image declared three CPUs.

    That is not a cosmetic difference. ``z_smp_init()`` starts CPUs 1..MP_MAX_NUM_CPUS-1, and
    ``arch_cpu_start`` spins on ``riscv_cpu_boot_flag`` with **no timeout**, so a CPU the image does not
    need is a hang with nothing printed past the boot banner if that hart does not answer.

    Hence: enough CPUs for every hart the image fans out to plus the master's, never more than the board
    has. The floor of 2 is kept so an image that is gated today is byte-identical tomorrow; the clamp is
    the new part, and it is the half that prevents the hang.
    """
    need = max(2, int(harts), int(rvv_hart) + 1)
    return min(need, int(brd.harts)) if getattr(brd, "harts", None) else need


def _prj_conf(cpus: int, backend: str, brd=None, console_facts=None, debug: bool = False) -> str:
    """Generated app config. ``brd`` is a :class:`runtime.boards.Board`; None keeps the
    historical chipyard/HTIF defaults so existing callers are byte-identical.

    ``console_facts`` is a :class:`runtime.sdk_facts.UartConsoleFacts` and is REQUIRED when the board
    declares a UART console: the baud divisor the RTOS driver computes depends on two clock numbers
    that belong to the chip, and defaulting either of them produces a console that emits garbage.
    """
    from ..boards import CONSOLE_HTIF, CONSOLE_UART, board as _board
    brd = brd if brd is not None else _board("spike_riscv64")
    # HTIF: the direct-putchar path races under SMP (a worker on hart != 0 printing) and silently
    # wedges, so use the buffered + syscall path. It is also the fix for an apparent hang: unbuffered
    # HTIF emits ONE CHARACTER per host round-trip, which on a ~20 MHz core looks like the model never
    # finishing. Both options default to `n` in Zephyr, so this must be stated. (Lesson 3,
    # merlin_hetero_runner.) A non-HTIF board configures its own console via its defconfig; we say
    # nothing rather than setting options its driver does not have.
    # A tick every few cycles is a real pathology, not a tuning preference: measured on the Kodiak
    # config, four spike images each burned 58 minutes of CPU without finishing a single inference,
    # because the board pairs a 40 kHz timer with a 10 kHz tick rate and its required FPU_SHARING=y
    # makes every tick save 32 vector registers. Our image has one pinned COOP worker per hart, no
    # preemption and no timeouts, so it needs almost no ticks. Emitted only when the descriptor asks.
    tick_conf = ("" if getattr(brd, "tick_hz", None) is None else f"""
# This image is a single-shot inference on pinned cooperative workers: no preemption, no timeouts to
# resolve, so it does not need the board's default tick rate (and paying it costs a vector-state
# save/restore per tick under FPU_SHARING).
CONFIG_SYS_CLOCK_TICKS_PER_SEC={brd.tick_hz}
""")
    if brd.console == CONSOLE_HTIF:
        console_conf = """# SMP-safe HTIF console (buffered + syscall; both default to n upstream).
CONFIG_UART_HTIF_BUFFERED_OUTPUT=y
CONFIG_UART_HTIF_BUFFERED_OUTPUT_SIZE=256
CONFIG_UART_HTIF_SYSCALL_PRINT=y
"""
    elif brd.console == CONSOLE_UART:
        if console_facts is None:
            raise RuntimeError(
                f"board {brd.name} declares a UART console but no SDK facts were derived; the baud "
                "divisor depends on the chip's own clock rates and must not be defaulted")
        # Saying nothing here was a SILENT failure: the chipyard board's own defconfig sets
        # CONFIG_UART_HTIF=y, so "let the board configure its console" left the image on a
        # host-assisted channel that hangs on silicon. Turn it off explicitly.
        #
        # The SiFive-style driver computes
        #   div = (SYS_CLOCK_HW_CYCLES_PER_SEC * RTC_CLOCK_DIVIDER_VALUE) / baud - 1
        # so both terms must describe THIS chip. The board defaults (1 MHz mtime x 1000) imply a 1 GHz
        # peripheral clock; against real hardware that programs a wildly wrong divisor and the console
        # emits GARBAGE rather than nothing -- which reads as a corrupt program, not a misconfigured
        # UART. Both numbers come from the SDK's chip_config.h.
        divider = console_facts.sys_clk_hz // console_facts.mtime_hz
        if divider * console_facts.mtime_hz != console_facts.sys_clk_hz:
            raise RuntimeError(
                f"core clock {console_facts.sys_clk_hz} is not an integer multiple of the mtime rate "
                f"{console_facts.mtime_hz}; the RTOS peripheral-clock model cannot express it")
        console_conf = f"""# Console: the chip's own UART. Facts derived from its SDK headers, not assumed.
CONFIG_UART_HTIF=n
CONFIG_CONSOLE=y
CONFIG_SERIAL=y
CONFIG_UART_CONSOLE=y
CONFIG_UART_SIFIVE=y
CONFIG_UART_SIFIVE_PORT_0=y
# Peripheral clock = {console_facts.sys_clk_hz} Hz, expressed as the mtime rate times the
# core/mtime divider, which is how this SoC's headers model it.
CONFIG_SYS_CLOCK_HW_CYCLES_PER_SEC={console_facts.mtime_hz}
CONFIG_RTC_CLOCK_DIVIDER_VALUE={divider}
"""
    else:
        raise RuntimeError(f"unknown console kind {brd.console!r} for board {brd.name}")
    # FPU_SHARING=y mis-routes V-illegal-instruction traps into the FP path, which retries forever --
    # a hang with NO fault printed (the FireSim Saturn-tile hang). Off unless a board needs otherwise;
    # note this also compiles out fpu.c/fpu.S, i.e. no FP/vector state is saved across a context
    # switch, which is safe for the single pinned cooperative worker this image runs.
    fpu_sharing = "y" if brd.fpu_sharing else "n"
    vector_max_len = _vector_max_len_bits(brd)
    common = f"""# Generated by merlin.runtime.backends.zephyr_model (backend={backend}, board={brd.name}).
CONFIG_PRINTK=y
CONFIG_STDOUT_CONSOLE=y
CONFIG_THREAD_STACK_INFO=y

CONFIG_MP_MAX_NUM_CPUS={cpus}
CONFIG_SMP=y
CONFIG_SCHED_CPU_MASK=y
CONFIG_SCHED_CPU_MASK_PIN_ONLY=y
CONFIG_SCHED_DUMB=y

{console_conf}
CONFIG_FPU=y
CONFIG_FLOAT_HARD=y
# FPU_SHARING=y mis-routes V-illegal-instruction traps to the FP path, which
# silently retries forever (a hang with NO fault printed) — the FireSim
# Saturn-tile hang. Keep it off so traps surface, and so single-COOP-worker FP
# context stays eager. (merlin_hetero_runner.)
CONFIG_FPU_SHARING={fpu_sharing}

# malloc for the lowered model's memref.alloc buffers (the whole-model activation
# arena). ARENA_SIZE=-1 = "use all leftover RAM" (a linker-defined region at the end of
# RAM), NOT a fixed static BSS array — a large fixed array overflows R_RISCV_PCREL_HI20
# relocations once it exceeds the model-link window. So the arena scales to the board's
# RAM (spike -m / FireSim DRAM) for free, and big models just work.
CONFIG_COMMON_LIBC_MALLOC=y
CONFIG_COMMON_LIBC_MALLOC_ARENA_SIZE=-1

CONFIG_MAIN_STACK_SIZE=32768
CONFIG_HEAP_MEM_POOL_SIZE=65536
{tick_conf}{_debug_conf() if debug else ""}"""
    if backend == "scalar":
        return common  # no vector config at all — the model object is rv64gc.
    # rvv: eager per-thread V save/restore + keep global -march scalar so the
    # non-V picolibc multilib links (the rv64gcv model object re-adds V itself).
    #
    # VECTOR_MAX_LEN sizes `vreg[32][N/8]` per thread, so it must cover the board's real VLEN and
    # costs memory if it over-covers. LAZY=n is deliberate: the other chipyard silicon boards disable
    # lazy vector switching explicitly, and Kodiak's defconfig is the one that leaves it at its
    # default y — we do not inherit that.
    # Whether Zephyr's own kernel can be built with V is a property of the TREE, not of the board:
    # without RISCV_V_KERNEL_ONLY, setting CONFIG_RISCV_ISA_EXT_V puts `v` in the global -march and no
    # matching libgcc multilib exists (the link falls back to a 32-bit one). So require both the board's
    # permission and the tree's capability -- otherwise pointing ZEPHYR_BASE at a tree that lacks the
    # symbol breaks every board, including spike_riscv64.
    tree_can = _kconfig_has("RISCV_V_KERNEL_ONLY")
    if tree_can and not brd.zephyr_vector_ext:
        # The only reason to deny Zephyr vector state is that the TREE cannot express it. When the tree
        # can and a descriptor still says no, the descriptor is stale -- and the failure it produces is
        # the worst kind: threads whose mstatus never carries VS, so the image works on one hart and
        # hangs (silently, with FPU_SHARING=y) the moment a context switch costs the master its vector
        # state. That shipped once on Kodiak. Fail closed rather than rebuild it.
        raise RuntimeError(
            f"board {brd.name} sets zephyr_vector_ext=False, but this Zephyr tree "
            f"({_zephyr_base()}) defines RISCV_V_KERNEL_ONLY -- V can be enabled without putting `v` "
            "in the global -march. The flag exists for trees that CANNOT do this; on a tree that can, "
            "leaving it off denies every thread its vector state and hangs any multi-hart image on "
            "silicon that enforces mstatus.VS. Set zephyr_vector_ext=True.")
    if not (brd.zephyr_vector_ext and tree_can):
        # Zephyr's kernel stays scalar (see Board.zephyr_vector_ext); the model object still carries V
        # from its own -march, and mstatus.VS is enabled at boot by reset.S under CONFIG_FPU.
        return common + ("\n# CONFIG_RISCV_ISA_EXT_V intentionally NOT set: this Zephyr has no\n"
                         "# RISCV_V_KERNEL_ONLY, so setting it would put `v` in the global -march and\n"
                         "# no matching libgcc multilib exists. mstatus.VS comes from CONFIG_FPU.\n")
    vec = [f"\nCONFIG_RISCV_ISA_EXT_V=y",
           f"CONFIG_RISCV_VECTOR_MAX_LEN={vector_max_len}"]
    # Emit only what this Zephyr tree actually defines -- an unknown symbol aborts the build.
    if _kconfig_has("RISCV_V_KERNEL_ONLY"):
        vec.append("CONFIG_RISCV_V_KERNEL_ONLY=y")
    if _kconfig_has("RISCV_ISA_EXT_V_LAZY"):
        vec.append("CONFIG_RISCV_ISA_EXT_V_LAZY=n")
    return common + "\n".join(vec) + "\n"


def _cmakelists(model_archive: Path, rt: Path, abi: Path, cgen: Path,
                weights_section_ld: Path | None = None, omp: bool = False,
                n_harts: int = 1, hart_ids: "tuple[int, ...] | None" = None,
                backend: str = "rvv", debug: bool = False,
                build_hash: str = "", console_facts=None,
                chip_freq_hz: int | None = None, op_profile: bool = False) -> str:
    # In external-weights mode, a linker snippet places the renamed .merlin_weights
    # section into the WEIGHTS DT memory-region (its own high address). main.c addresses
    # the blob by literal (not its symbol), so --gc-sections would drop the whole
    # weights_blob.o; -u forces the blob symbol as a GC root (no PC-relative reloc) to
    # keep .merlin_weights in the image.
    ext = (f'zephyr_linker_sources(SECTIONS "{weights_section_ld}")\n'
           "zephyr_link_libraries(-Wl,-u,_binary_weights_bin_start)\n"
           if weights_section_ld is not None else "")
    # Opt-in OpenMP diagnostics, off unless MERLIN_OMP_DEBUG_SPLIT=1 in the environment. The
    # shim's worksharing split is the one place where a wrong answer looks like a crash (a bad
    # loop bound becomes an out-of-range vector load), so being able to dump every
    # (gtid, nthreads, in-range, out-range) without editing C is worth the two lines.
    omp_debug = ("target_compile_definitions(app PRIVATE MERLIN_OMP_DEBUG_SPLIT=1)\n"
                 if omp and os.environ.get("MERLIN_OMP_DEBUG_SPLIT") == "1" else "")
    # Size the worker-stack ARRAY to the harts actually requested. Each stack must be as large
    # as the master's model stack (a worker runs the same outlined code), so leaving the pool at
    # its 8-slot maximum would reserve 64 MB of BSS to use two of them.
    if omp:
        omp_debug += (f"target_compile_definitions(app PRIVATE "
                      f"MERLIN_OMP_MAX_THREADS={max(int(n_harts), 1)})\n")
        # Whether the pool must stay on vector-capable harts. For a VECTOR image the shim DISCOVERS
        # which those are at startup (each hart probes its own mstatus.VS writability, which is
        # hardwired to zero without a vector unit) -- so a heterogeneous SoC needs nobody to tell us
        # which harts they are, and the image cannot deadlock by fanning out onto a scalar hart. A
        # SCALAR image sets 0: it may use every hart, which is the point of building one.
        omp_debug += (f"target_compile_definitions(app PRIVATE "
                      f"MERLIN_OMP_VECTOR_POOL={1 if backend == 'rvv' else 0})\n")
        # A descriptor may still PIN the set (`vector_hart_ids`), which the runtime probe overrides
        # when it is on. Kept for a chip whose probe cannot be trusted, and as documentation.
        if hart_ids is not None:
            ids = ", ".join(str(int(h)) for h in hart_ids)
            omp_debug += (f"target_compile_definitions(app PRIVATE "
                          f'"MERLIN_OMP_HART_IDS={{{ids}}}")\n')
    # A DEBUG image also carries the fatal-error hook. It has to be its own translation unit (it
    # overrides a weak kernel symbol) and it needs the build identity as a macro, so that a fault line
    # names the binary without depending on a banner thousands of lines earlier in the capture.
    dbg_src = f"  {rt}/merlin_fatal_zephyr.c\n" if debug else ""
    if debug and op_profile:
        dbg_src += f"  {rt}/merlin_op_prof.c\n"
        omp_debug += ("target_compile_definitions(app PRIVATE MERLIN_PROF_ZEPHYR=1)\n"
                      f"target_compile_definitions(app PRIVATE "
                      f"MERLIN_PROF_HEARTBEAT_MS={DEBUG_HEARTBEAT_S * 1000})\n")
    # The build identity is a macro in EVERY image, not just debug ones: the allocation guard below
    # is always linked, and a `FAIL alloc` line has to name the binary it came from.
    omp_debug += (f'target_compile_definitions(app PRIVATE '
                  f'"MERLIN_BUILD_HASH=\\"{build_hash}\\"")\n')
    # PLL BRING-UP, only when the descriptor asks for a clock the chip is not already on. Without it
    # the image runs at the reset clock, which on the SoC this was written for is a tenth of what the
    # vendor's own demos use. The macros are the SAME derived set the bare-metal console uses -- one
    # derivation, two consumers, so the two paths cannot disagree about where this chip's PLL is.
    if chip_freq_hz and console_facts is not None:
        dbg_src += f"  {rt}/merlin_socinit_zephyr.c\n"
        for flag in console_facts.macros(chip_freq_hz=chip_freq_hz):
            omp_debug += (f"target_compile_definitions(app PRIVATE "
                          f"{flag[2:] if flag.startswith('-D') else flag})\n")
    return f"""# Generated by merlin.runtime.backends.zephyr_model.
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


# ---- build / run -------------------------------------------------------------------

def build_app(model_dir: str | Path, work: str | Path, *, board: str = "spike_riscv64",
              backend: str = "rvv", rvv_hart: int = 0, arena_mb: int = 64, cpus: int = 2,
              # NOTE: `arena_mb` is ACCEPTED AND IGNORED. The activation arena is
              # CONFIG_COMMON_LIBC_MALLOC_ARENA_SIZE=-1, i.e. "all leftover RAM", because a fixed
              # array overflows R_RISCV_PCREL_HI20 once it exceeds the model-link window. The
              # parameter is kept so existing callers (zephyr_matrix.py passes 192) still work; the
              # arena is sized by `_ram_for_weights` instead. Do not add code that reads it.
              inputs_npz: str | Path | None = None, ram_bytes_override: int | None = None,
              int8_compute: bool = False, rvv_schedule: str | None = None,
              cflags_override: list[str] | None = None,
              features: "frozenset[str] | None" = None,
              n_harts: int = 1, iters: int = 1, warmup: int = 0,
              omp_threads: int | None = None, vlen: int | None = None,
              sdk_dir: str | Path | None = None, debug: bool = False,
              matrix: "MatrixRouting | None" = None,
              matrix_scalar_tile: bool = False) -> dict:
    """Lower the model, generate the Zephyr app, and build ``zephyr.elf``.

    ``backend``: ``"rvv"`` (vector tile / Saturn) or ``"scalar"`` (scalar tile). The
    scalar build is the portable FireSim-safe path; the vector build targets the Saturn
    tile (worker on ``rvv_hart``). Returns ``{elf, app_dir, backend, **c_runtime_info}``.

    ``rvv_schedule`` overrides the default ``RVV_TRANSFORM_SCHEDULE`` (the seam through which
    an isolated RVV target-package supplies its own transform schedule); ``cflags_override``
    replaces ``_cflags(backend)``. Both default ``None`` -> the build is byte-identical to the
    shipping codegen, so the global flow is never perturbed by the package machinery.

    ``n_harts > 1`` builds the MULTICORE image: the model is lowered with an outer
    OpenMP-parallel loop under the RVV schedule (llvmlower ``parallel_harts``) and linked
    against the Zephyr OpenMP shim (``merlin/runtime/c/libomp_zephyr.c``), whose pool pins
    one COOP worker per hart. ``n_harts=1`` (default) is byte-identical to the shipping
    single-worker image.

    ``iters``/``warmup`` build the SUSTAINED-inference image (per-iteration cycle metrics
    over a reused arena); the defaults are the single-shot behavior.

    ``sdk_dir`` is the target's own SDK checkout, REQUIRED when the board declares a UART console:
    the UART address and the two clock rates its baud divisor depends on are derived from that SDK's
    headers (``runtime.sdk_facts``) rather than written down here.
    """
    model_dir, work = Path(model_dir).resolve(), Path(work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    inputs_npz = inputs_npz or (model_dir / "inputs.npz")
    if not available():
        raise ZephyrModelError("Zephyr/spike toolchain unavailable (see env in module doc)")
    # NOTE scalar multicore IS supported (see the lowering call below): the scalar path cannot use the
    # forall-under-the-RVV-schedule route, so it parallelizes at the linalg loop level instead. This is
    # the ONLY way to use a hart that has no vector unit -- a heterogeneous SoC may bring up more cores
    # than it attaches vector units to, and those extra cores are otherwise unreachable.
    if iters < 1:
        raise ZephyrModelError(f"iters must be >= 1, got {iters}")
    # A VLEN the build assumes must match the one it will run on -- see march_with_vlen.
    cflags = march_with_vlen(cflags_override or _cflags(backend), vlen)

    gcc = _spike.gcc_path()
    ld = gcc.with_name("riscv64-unknown-elf-ld")
    ar = gcc.with_name("riscv64-unknown-elf-ar")
    objcopy = gcc.with_name("riscv64-unknown-elf-objcopy")
    clang = toolchain.clang()

    # 1. model.mlir -> normalize (SAME prep passes the dispatch_runtime applies, so
    #    quantized / bf16 / over-rank / bool-cast models lower correctly — without these
    #    the whole-model path only handles already-clean LLMs) -> LLVM IR -> object.
    # Board facts as DATA (runtime.boards): the console options, the vector-state width, the DT RAM
    # label and the DRAM ceiling all come from the descriptor instead of being assumed. `vlen` given
    # explicitly wins over the board's, so a caller can sweep it.
    from ..boards import board as _board_desc
    brd = _board_desc(board, **({"vlen": vlen} if vlen is not None else {}))
    if vlen is None and brd.vlen is not None:
        vlen = brd.vlen                       # build for the board's real vector length by default
    # A vector model may only fan out over harts that HAVE a vector unit. A heterogeneous SoC is normal
    # -- three cores brought up, a vector unit on two of them -- and the difference is invisible in the
    # device tree, which lists identical cpu@N nodes. Fanning out too far does not fail cleanly: the
    # worker on the scalar hart takes an illegal instruction, never reaches the barrier its peers wait
    # on, and the image hangs until the operator times out. Measured on a 3-core tapeout with 2
    # vector cores: the 1-hart images passed and every 3-hart image timed out at 10 minutes.
    # The harts this image may actually run on. A vector image is confined to the vector-capable ones;
    # a scalar image may use every hart, which is the only way to reach a core with no vector unit.
    # Emitted into the OpenMP shim only when it differs from the 0..n-1 default, so an ordinary
    # homogeneous board's image is byte-identical to before.
    _usable = brd.hart_ids_for(backend)
    hart_ids = _usable if _usable != tuple(range(n_harts)) else None
    if backend == "rvv" and n_harts > brd.n_vector_harts:
        raise ZephyrModelError(
            f"{brd.name}: refusing an RVV image over {n_harts} harts -- only {brd.n_vector_harts} of "
            f"its {brd.harts} harts can execute vector code. The extra worker would trap on its first "
            f"vector instruction and deadlock the barrier (a timeout, with no fault printed). Build "
            f"{brd.n_vector_harts} harts or fewer, or use backend='scalar' to use every hart.")
    # Parse + lower under IR_LOCK: xDSL's parser is not thread-safe, and a delivery builds several
    # images in one process. See common.ir_lock -- the symptom is a bogus ParseError on valid IR.
    from ...common.ir_lock import IR_LOCK
    prof_table: list[dict] | None = None
    with IR_LOCK:
        prepared, features = prepare_for_lowering(model_dir / "model.mlir", work,
                                                 int8_compute=int8_compute, features=features,
                                                 blocking=True, harts=n_harts,
                                                 vlen=vlen, matrix=matrix)
        # A DEBUG image interleaves a mark between the top-level ops of @forward. Two things come out
        # of it: a per-op cost table at the end of a successful run, and -- the reason it is here at
        # all -- a continuously published "which op are we in", which the heartbeat prints while the
        # model is still running. Without it a stalled image can only say that it stalled; with it, it
        # says where. The table is written next to the image so a returned PROF line resolves to a name.
        if debug:
            from ...llvmlower import op_profile as _op_profile
            try:
                text, prof_table = _op_profile.instrument(Path(prepared).read_text())
                prepared = Path(work) / "model_prof.mlir"
                Path(prepared).write_text(text)
                _op_profile.write_table(prof_table, Path(work) / "op_profile_table.json")
            except _op_profile.OpProfileError as exc:
                # Losing per-op detail must not cost us the rest of the diagnostics.
                print(f"[zephyr_model] op profiling unavailable ({exc}); "
                      "debug image keeps STAGE/ALIVE/MEM but ALIVE cannot name an op")
                prof_table = None
    # For the rvv backend, bake native RVV (fixed-width vector ops on the matmuls) into the
    # IR rather than leaving it to clang's auto-vectorizer — see llvmlower.pipeline.
        # Multicore reaches the two backends by DIFFERENT routes, because they parallelize at
        # different levels. The rvv path layers an explicit outer `scf.forall` UNDER the package's
        # transform schedule (`parallel_harts`), so the inner tiling and the emitted vfmacc/vwmacc are
        # untouched. The scalar path has no such schedule to sit under; it parallelizes at the linalg
        # loop level instead (`parallel=True` -> convert-linalg-to-parallel-loops + convert-scf-to-
        # openmp), which is the same pipeline the K1 big models use. Both emit `__kmpc_*` satisfied by
        # the Zephyr OpenMP shim, so the worker side is identical.
        res = lower_model_file(prepared, work / "lower", targets=(), textual=True,
                               vectorize=(backend == "rvv"), transform_schedule=rvv_schedule,
                               features=features,
                               parallel=(backend != "rvv" and n_harts > 1),
                               parallel_harts=(n_harts if n_harts > 1
                                               and backend == "rvv" else None))
    # What this lowering will ask the heap for, read off the IR that is about to be compiled. Measured
    # here rather than estimated later: the file exists for exactly this build, and the number decides the
    # region size below.
    alloc_total, alloc_dynamic = allocation_bytes(res.ll_path)

    _run([clang, "--target=riscv64-unknown-elf", *cflags, "-c", res.ll_path,
          "-o", work / "model.o"])

    # 2. data-driven runtime artifacts (arg table, ciface, weights.bin, embedded io).
    cgen = work / "cgen"
    info = c_runtime.generate(model_dir, cgen, inputs_npz)

    # 3. weights.bin -> binary blob object; archive it with model.o.
    _run([ld, "-r", "-b", "binary", "-o", work / "weights_blob.o", "weights.bin"], cwd=cgen)
    weights_size = (cgen / "weights.bin").stat().st_size

    # Build identity, part one: the lowered model object plus the weights blob -- what computes the
    # answer. The app configuration is folded in further down, once it exists, because the compute is
    # only half of what makes two images behave differently. Hashing the model alone was a real defect:
    # the Kodiak vector-state fix changed the Kconfig and the DT overlay and nothing else, so the fixed
    # and the hanging image reported the SAME build_hash, and the one thing the protocol exists to
    # answer -- "which binary produced this log?" -- could not be answered.
    import hashlib as _hashlib
    _h = _hashlib.sha256()
    _h.update((work / "model.o").read_bytes())
    _h.update((cgen / "weights.bin").read_bytes())

    # External-weights mode for blobs that would overflow medany linked into .data: rename
    # the blob's section so Zephyr's default linker won't pull it into the image .data; a
    # snippet diverts it to the WEIGHTS region at a fixed high address, and main.c addresses
    # it by literal. Keeps ram0 compact (code + arena), so big fp32 models link.
    # DECIDE ON THE LINKED FOOTPRINT, NOT THE BLOB SIZE. medany constrains the whole ram0 span
    # (code + weights + activation arena), so a blob that fits LINK_LIMIT on its own can still push
    # the region past the window once the arena is added -- and then linked mode is chosen for an
    # image that cannot work. MEASURED on gemma2_2b_int8_section12: 1462 MiB of weights sat under the
    # 1900 MiB threshold, linked mode was selected, the arena took ram0 to ~3.7 GiB, and the image
    # never reached its first op in 30 min (silent -- no fault, which is what made it expensive to
    # find). The SAME bundle forced external, ram0 a compact 1 GiB and the blob at EXT_WEIGHTS_BASE,
    # reaches op 0 in 8 minutes. Comparing the computed region is strictly safer than comparing the
    # blob: it can only move builds from linked to external, and external is the mode that works at
    # every size tested (1034 MiB and 2485 MiB regions both execute).
    from ...common.ir_lock import IR_LOCK
    from ...common.mlir_query import activation_peak_bytes
    peak = None                       # measured only on the path that sizes the region from it
    linked_region = None
    if ram_bytes_override is not None:
        external = False
    else:
        # Same lock rationale as the sizing parse below: three concurrent build_app calls produced a
        # bogus ParseError on valid IR, which read as a broken build rather than a race.
        with IR_LOCK:
            peak = int(activation_peak_bytes(model_dir / "model.mlir") or 0) or None
        linked_region = _ram_for_weights(weights_size, peak, alloc_total)
        external = linked_region > LINK_LIMIT
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
        # `linked_region` was already computed above to MAKE the linked/external decision; reusing it
        # is what keeps the decision and the sizing from disagreeing (recomputing invites a future
        # edit to change one and not the other).
        ram_bytes = ram_bytes_override if ram_bytes_override is not None else linked_region

    archive = work / "libmerlinmodel.a"
    archive.unlink(missing_ok=True)
    # The matrix-unit shim, when the rewrite routed anything to one, goes into the SAME archive as the
    # model object -- so it is covered by the --whole-archive the generated CMakeLists already applies and
    # no CMake change is needed. Built from the SIDECAR the rewrite wrote: the symbols the module actually
    # calls are the ones that must be defined, and a set reconstructed here could drift into a link error.
    matrix_objs: list[Path] = []
    matrix_build = None
    from ...llvmlower.passes_opu import load_sidecar as _load_matrix_sidecar
    matrix_sigs = _load_matrix_sidecar(work)
    if matrix_sigs:
        if matrix is None:
            raise ZephyrModelError(
                f"{len(matrix_sigs)} matrix-unit signature(s) were routed but no `matrix=` routing is "
                "available to build them against; the image would not link")
        from ...llvmlower import opu_shim
        # A routed contraction is an opaque `call` by the time the parallel transform schedule runs, so
        # that schedule -- which matches `linalg.matmul`/`linalg.batch_matmul` -- cannot split it. On a
        # chip with a matrix unit per core, the ONLY place the other units can be reached from is inside
        # the kernel's own tile loop, so a multi-hart image compiles that loop with OpenMP. A serial
        # object on such a chip is not an error and not a wrong answer: it uses one unit, silently.
        matrix_build = opu_shim.build_object(
            matrix_sigs, work / "matrix", unit=matrix.unit, config=matrix.config,
            cc=clang, cflags=["--target=riscv64-unknown-elf", *cflags],
            scalar_tile=matrix_scalar_tile, parallel_tiles=n_harts > 1)
        matrix_objs = [matrix_build.object_path]
        print(f"[matrix] linked {len(matrix_sigs)} entry point(s) for {matrix.unit} "
              f"({matrix.config}, tile edge {matrix_build.tile_edge}, "
              f"{'SCALAR STAND-IN' if matrix_build.scalar_tile else 'device instructions'}, "
              f"{matrix_build.scratch_bytes} B pack scratch, "
              f"tile loop {'parallel over ' + str(n_harts) + ' hart(s)' if matrix_build.parallel_tiles
                           else 'serial — ONE unit will be used'})")
    _run([ar, "rcs", archive, work / "model.o", work / "weights_blob.o", *matrix_objs])

    # 4. emit the Zephyr application tree.
    app = work / "app"
    (app / "src").mkdir(parents=True, exist_ok=True)
    (app / "boards").mkdir(parents=True, exist_ok=True)
    # A multicore image needs every hart it fans out to, plus the master's.
    cpus = max(cpus, n_harts, rvv_hart + 1)
    # Console facts, when the board has a console of its own to bring up. Derived here (not in
    # _prj_conf) because the DT overlay below needs the same values, and deriving twice invites the
    # config and the device tree to disagree about which UART this is.
    from ..boards import CONSOLE_UART
    from ..sdk_facts import DEFAULT_BAUD as _DEFAULT_BAUD
    console_facts = None
    if brd.console == CONSOLE_UART:
        from ..sdk_facts import derive_uart_console
        if not sdk_dir or not brd.sdk_chip:
            raise ZephyrModelError(
                f"{brd.name} declares a UART console, so it needs sdk_dir + the descriptor's "
                "sdk_chip: the UART address and its clock rates are derived from the target SDK's "
                "own headers, never hardcoded or defaulted")
        console_facts = derive_uart_console(sdk_dir, brd.sdk_chip)
    prj_conf_text = _prj_conf(cpus, backend, brd, console_facts, debug=debug)
    (app / "prj.conf").write_text(prj_conf_text)
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
    def _cmake(bh: str) -> str:
        return _cmakelists(archive, rt, abi, cgen, weights_section_ld, omp=(n_harts > 1),
                           n_harts=n_harts, hart_ids=hart_ids, backend=backend,
                           debug=debug, build_hash=bh, console_facts=console_facts,
                           chip_freq_hz=brd.chip_freq_hz, op_profile=prof_table is not None)

    # Written below, once the identity it embeds is known.
    cmakelists_text = _cmake("")
    # Board overlay. External mode: ram0 = 1 GB (code + arena) and a separate WEIGHTS
    # memory-region holding the blob at EXT_WEIGHTS_BASE. Otherwise: grow ram0 only when
    # the model needs > the stock 256 MB (small models keep the default that boots reliably
    # on FireSim). Plus, for chipyard, the disable-cpu@2..7 overlay.
    overlay = ""
    if console_facts is not None:
        # Point `chosen` at the chip's UART and state its address from the DERIVED fact rather than
        # trusting the board DT to already agree: the config above selects the driver, but `chosen`
        # is what the console actually binds to, and a board whose DT says HTIF would keep using it.
        overlay += (f"/ {{\n\tchosen {{\n"
                    f"\t\tzephyr,console = &{brd.uart_label};\n"
                    f"\t\tzephyr,shell-uart = &{brd.uart_label};\n\t}};\n}};\n\n"
                    f"&{brd.uart_label} {{\n\tstatus = \"okay\";\n"
                    f"\treg = <{hex(console_facts.uart_base)} 0x1000>;\n"
                    f"\tcurrent-speed = <{_DEFAULT_BAUD}>;\n}};\n\n")
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
    def _region_overlay(size: int) -> str:
        """The ram0 override for a region of ``size``, or "" at the DTS default.

        A function rather than a literal because the region may have to be CORRECTED after the link: the
        arena is the leftover after the image, and the image's size is not known until it exists.
        """
        if size <= DEFAULT_RAM_BYTES:
            return ""
        # Never past what the chip HAS: a region larger than physical DRAM is a boot that dies before
        # main() with no console output at all.
        if size > brd.dram_bytes:
            raise ZephyrModelError(
                f"{brd.name}: this model needs a {size / 2**20:.0f} MB region but the board has "
                f"{brd.dram_bytes / 2**20:.0f} MB of DRAM. Shrink the model (a smaller capture), or "
                f"correct the board descriptor if the real DRAM is larger.")
        return (f"&{brd.ram_label} {{\n\treg = <{hex(brd.dram_base)} "
                f"{hex(size)}>;\n}};\n")

    # Keyed on the board we actually BUILD (`build_board`), not on this descriptor's name: a chip with
    # no Zephyr port of its own is built against a generic port, and testing the descriptor name here
    # silently skipped this overlay for exactly those boards.
    cpu_overlay = (_chipyard_cpu_overlay(cpus) if brd.build_board.startswith("chipyard") else "")
    overlay_base = overlay

    def _overlay_for(size: int) -> str:
        """The whole overlay for a region of ``size``. Assembled from parts so the region can change."""
        text = overlay_base + ("" if external else _region_overlay(size))
        return text + (("\n" if text else "") + cpu_overlay if cpu_overlay else "")

    overlay = _overlay_for(ram_bytes)

    # Build identity, part two: the app configuration. Two images with identical compute but a
    # different Kconfig, device tree or link are different images -- they take different code paths on
    # the silicon, and one can hang where the other passes. Folding these in is what makes
    # `METRIC build_hash` answer "which binary produced this log?" rather than "which model did it
    # compute?". `brd.build_board` is in there too, since the same app text against a different board
    # port is a different image again.
    def _harness(bh: str) -> str:
        return _main_c(rvv_hart, weights_base=weights_base, n_harts=n_harts,
                       iters=iters, warmup=warmup, omp_threads=omp_threads,
                       build_hash=bh, console=brd.console, debug=debug,
                       dram_base=brd.dram_base, region_bytes=ram_bytes)

    _h.update(prj_conf_text.encode())
    _h.update(cmakelists_text.encode())
    _h.update(brd.build_board.encode())
    # The OVERLAY and the harness are folded in per build pass rather than here, because a corrected
    # region changes both -- and an image linked for a different region is a different image, which is
    # exactly what this hash exists to distinguish.
    # 5. configure + build. Possibly TWICE, because the region and the image are circular: the arena is
    #    whatever is left of the region after the image, and the image's size is not known until it has
    #    been linked. The first pass sizes the region from the weights plus the measured allocation
    #    demand; if the linked image turns out to be larger than the weights by more than the slack --
    #    on a 1.5 GB model the code, the .bss and the per-hart stacks came to 150 MB, and the arena
    #    landed 17 MB short of what the model asks for -- the region is corrected to the MEASURED image
    #    plus that demand and the app is rebuilt. Bounded at one correction: the image does not change
    #    size when the region does (only the linker script's region length does), so one pass settles it.
    build_dir = work / "build"
    env = _tool_env()
    elf = build_dir / "zephyr" / "zephyr.elf"

    def _emit_and_build() -> str:
        h = _hashlib.sha256(_h.digest())
        h.update(overlay.encode())
        # The harness with an EMPTY identity, so the hash is not an input to itself. This is what makes
        # iters/warmup/rvv_hart/console and the linked region part of the identity too.
        h.update(_harness("").encode())
        bh = h.hexdigest()[:16]
        (app / "prj.conf").write_text(prj_conf_text)
        (app / "src" / "main.c").write_text(_harness(bh))
        (app / "CMakeLists.txt").write_text(_cmake(bh))
        # Pass the overlay EXPLICITLY. Zephyr's automatic discovery matches `boards/<BOARD>.overlay`
        # against the board being built, so writing it under this descriptor's name meant that for any
        # board built through a generic port (build_board != name) the overlay was silently IGNORED --
        # taking the DRAM-size override and the CPU-disable overlay with it, with no warning and a
        # successful build. Naming one file and pointing cmake at it cannot miss that way.
        extra = []
        if overlay:
            overlay_file = app / "merlin.overlay"
            overlay_file.write_text(overlay)
            extra.append(f"-DEXTRA_DTC_OVERLAY_FILE={overlay_file}")
        _run([build_tool("cmake") or "cmake", "-B", build_dir, "-G", "Ninja",
              f"-DBOARD={brd.build_board}", *extra, "-S", app], env=env)
        _run([build_tool("ninja") or "ninja", "-C", build_dir], env=env)
        if not elf.is_file():
            raise ZephyrModelError(f"build produced no elf at {elf}")
        return bh

    build_hash = _emit_and_build()
    demand = alloc_total + 128 * 1024 * 1024
    if not external and alloc_total:
        from ..elf_audit import read_elf as _read_elf
        image = sum(sg.memsz for sg in _read_elf(elf)[1])
        if ram_bytes - image < demand:
            align = 16 * 1024 * 1024
            grown = ((image + demand + align - 1) // align) * align
            print(f"[zephyr_model] region {ram_bytes / 2**20:.0f} MB leaves "
                  f"{(ram_bytes - image) / 2**20:.0f} MB of arena for a model that asks for "
                  f"{alloc_total / 2**20:.0f} MB; the linked image measures {image / 2**20:.0f} MB. "
                  f"Relinking for {grown / 2**20:.0f} MB.", flush=True)
            ram_bytes = grown
            overlay = _overlay_for(ram_bytes)
            build_hash = _emit_and_build()
    out = {"elf": elf, "app_dir": app, "build_dir": build_dir, "backend": backend,
           "ram_bytes": ram_bytes, "build_hash": build_hash, **info}
    # What the matrix-unit shim in this image actually is, for a caller that has to state it: the tile
    # edge and alignment it was built for, whether the tile loop can reach more than one unit, and WHICH
    # REVISION of the unit's sources the encodings were derived from. The last one is the whole point --
    # an image carrying instructions derived from the wrong revision links and runs.
    if matrix_build is not None:
        out["matrix_build"] = matrix_build.to_dict()
    # The memory arithmetic, so a caller can verify the region it got rather than trusting the formula
    # that produced it. `allocation_bytes_total` is the requirement under an allocator that never
    # reclaims; `allocation_dynamic_calls` says how much of the answer is unknown at build time.
    out["allocation_bytes_total"] = alloc_total
    out["allocation_dynamic_calls"] = alloc_dynamic
    out["activation_peak_bytes"] = peak
    # The per-op table has to travel WITH the image. A debug run emits a thousand `PROF <id> ...` lines
    # and an `ALIVE ... op=<id>` naming where it stopped, and every one of those ids is meaningless
    # without the mapping from id to op name, family and shape. Shipping the trace without the table is
    # shipping a log nobody can read.
    if prof_table is not None:
        out["op_profile_table"] = Path(work) / "op_profile_table.json"
        out["op_profile_ops"] = len(prof_table)
    return out


#: Base ISA string for a whole-model RVV spike run. VLEN is NOT part of it — see `spike_isa`.
DEFAULT_SPIKE_ISA = "rv64gcv_zfh_zvfh"


def spike_isa(vlen: int | None = None, base: str = DEFAULT_SPIKE_ISA) -> str:
    """``base`` ISA string with an explicit VLEN, expressed as the ``zvl<N>b`` extension.

    spike takes its vector length from the ISA string, and with no ``zvl`` extension it uses the V
    minimum of 128. That is a silent trap for a board whose vector unit is wider: the image is
    validated at a vector length the hardware does not have, and a fixed-width schedule compiled for
    one VLEN maps to a different LMUL on the other (the measured K1 case -- ``-march=rv64gcv`` assumes
    128 while the board is 256, which doubles LMUL and spills). Pinning it here lets a run state the
    vector length it actually exercised. ``None`` keeps spike's default.
    """
    if vlen is None:
        return base
    if vlen < 128 or vlen & (vlen - 1):
        raise ZephyrModelError(f"VLEN must be a power of two >= 128, got {vlen}")
    return f"{base}_zvl{int(vlen)}b"


def run_on_spike(elf: str | Path, *, harts: int = 2, isa: str = DEFAULT_SPIKE_ISA,
                 vlen: int | None = None,
                 mem_bytes: int = 1 << 31, timeout: int = 3600) -> dict[str, Any]:
    """Run the Zephyr ELF on spike ``-pN``; parse the OUT/ARGMAX/METRIC/DONE markers.

    ``vlen`` pins the simulated vector length (via :func:`spike_isa`); None = spike's default 128.
    """
    isa = spike_isa(vlen, isa) if vlen is not None else isa
    cmd = [_spike.spike_path(), f"--isa={isa}", f"-p{harts}",
           f"-m{hex(0x80000000)}:{hex(mem_bytes)}", str(elf)]
    proc = subprocess.run([str(c) for c in cmd], capture_output=True, text=True,
                          timeout=timeout)
    console = proc.stdout + proc.stderr
    return _parse_console(console, proc.returncode)


# The chipyard Verilator sim for a multicore Saturn SoC. Built from the config added in
# generators/chipyard/src/main/scala/config/MerlinSaturnConfigs.scala; every stock Saturn
# config is single-core and the only 2-tile SoC puts a vector unit on one tile only, so
# neither can host a multi-hart RVV image.
DEFAULT_SATURN_SIM_CONFIG = "MultiSaturnV256D128ShuttleConfig"


def verilator_sim(config: str = DEFAULT_SATURN_SIM_CONFIG) -> Path | None:
    """Path to the chipyard Verilator sim for ``config``, or None when it is not built.

    Resolution order: ``MERLIN_SATURN_VERILATOR`` (an explicit binary), then the standard
    chipyard build location under ``$MERLIN_CHIPYARD/sims/verilator``. Returns None rather
    than raising so callers record an honest ``not_run``.
    """
    from ...common.paths import env as _env
    explicit = _env("MERLIN_SATURN_VERILATOR")
    if explicit and Path(explicit).is_file():
        return Path(explicit)
    cy = _env("MERLIN_CHIPYARD")
    if not cy:
        return None
    sim = Path(cy) / "sims" / "verilator" / f"simulator-chipyard.harness-{config}"
    return sim if sim.is_file() else None


def run_on_verilator(elf: str | Path, *, config: str = DEFAULT_SATURN_SIM_CONFIG,
                     timeout: int = 7200, references: dict | None = None,
                     reference: np.ndarray | None = None) -> dict[str, Any]:
    """Run a ``chipyard_riscv64`` Zephyr ELF on the chipyard Verilator sim.

    The Zephyr chipyard board's console is HTIF (``zephyr,console = &htif``), which is what
    the Rocket-chip emulator harness speaks, so the ELF runs unmodified and the SAME
    OUT/ARGMAX/METRIC/DONE parser is reused.

    SCOPE — this is cycle-accurate RTL, i.e. roughly 10^4 simulated cycles/second. A whole
    22-layer TinyLlama inference is ~10^10 cycles and is NOT feasible here; that is a
    property of RTL simulation, not a configuration to tune. Use this to certify the
    multicore-RVV MECHANISM (a GEMM micro-benchmark, or a 1-2 layer slice) and take
    whole-model functional truth from spike and whole-model cycle truth from the K1 board.
    """
    sim = verilator_sim(config)
    if sim is None:
        raise ZephyrModelError(
            f"no Verilator sim for {config!r} (set MERLIN_SATURN_VERILATOR, or build it: "
            f"make -C $MERLIN_CHIPYARD/sims/verilator CONFIG={config})")
    proc = subprocess.run([str(sim), str(elf)], capture_output=True, text=True,
                          timeout=timeout)
    res = _parse_console(proc.stdout + proc.stderr, proc.returncode)
    res["sim"] = str(sim)
    res["sim_config"] = config
    refs = references if references is not None else reference
    if refs is not None:
        res.update(_gate(res["prefix"], refs))
    return res


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
    out_hash: dict[str, Any] | None = None
    iter_cycles: list[int] = []
    iter_wall_ns: list[int] = []
    stage_wall_ns: dict[str, list[int]] = {}
    for l in console.splitlines():
        if l.startswith("METRIC "):
            parts = l.split()
            if parts[1] == "iter_cycles":       # METRIC iter_cycles <i> <cycles>
                iter_cycles.append(int(parts[3]))
            elif parts[1] == "iter_wall_ns":    # K1: METRIC iter_wall_ns <i> <ns>
                iter_wall_ns.append(int(parts[3]))
            else:
                # Most metrics are counters, but not all: `build_hash` is a hex identity string. A
                # parser that assumes int() crashes on the first non-numeric metric and takes the whole
                # run with it, so keep the raw string when it is not a number.
                #
                # And some carry a LIST: `METRIC vector_harts 0 1 2` names every vector-capable hart.
                # Reading only the first token reported it as `0`, which says "zero vector harts" --
                # precisely the opposite of what a chip with three of them was reporting.
                if len(parts) > 3:
                    metrics[parts[1]] = " ".join(parts[2:])
                else:
                    try:
                        metrics[parts[1]] = int(parts[2])
                    except ValueError:
                        metrics[parts[1]] = parts[2]
        elif l.startswith("ARGMAX "):
            p = l.split()
            argmax = np.array([int(x) for x in p[2:2 + int(p[1])]], dtype=np.int64)
        elif l.startswith("SUM "):
            sumval = struct.unpack("<f", struct.pack("<I", int(l.split()[1]) & 0xFFFFFFFF))[0]
        elif l.startswith("HASH "):
            # `HASH <algo> <elems> <value>` -- a whole-tensor digest. `outputs` above is only the
            # first MERLIN_DUMP_CAP elements (0.2% of a 2,048,000-element logit tensor), so this is
            # the only thing in the protocol that speaks for the rest of the answer. The algorithm
            # is named rather than assumed, so a future image can change it without silently
            # comparing two different digests.
            p_ = l.split()
            if len(p_) >= 4:
                out_hash = {"algo": p_[1], "elems": int(p_[2]), "value": int(p_[3]) & 0xFFFFFFFF}
        elif l.startswith("STAGE "):
            # K1 multi-program protocol: STAGE <session-repeat> <program> <wall-ns>.
            # Keep the repeat index structural: missing/duplicate rows are evidence of an invalid
            # diagnostic series, not values that should be silently packed into a shorter list.
            p_ = l.split()
            if len(p_) == 4:
                repeat, name, value = int(p_[1]), p_[2], int(p_[3])
                values = stage_wall_ns.setdefault(name, [])
                if repeat != len(values):
                    raise ZephyrModelError(
                        f"non-contiguous STAGE series for {name}: expected {len(values)}, got {repeat}")
                values.append(value)
    res = {"outputs": flat, "prefix": flat, "argmax": argmax, "sum": sumval,
           "metrics": metrics, "console": console}
    if out_hash is not None:
        res["out_hash"] = out_hash
    # `PROF <id> <ticks> <hits>` from an op_profile build. The K1 path already reads these; this one
    # dropped them on the floor, so a Zephyr/FireSim console carried per-op cycles that no caller could
    # reach -- which is why the vector unit had no measured rate while the matrix unit had one.
    from ...llvmlower import op_profile as _op_profile          # `op_profile` is a parameter name here
    prof = _op_profile.parse_prof_lines(console)
    if prof:
        res["op_prof"] = prof
    if iter_cycles:
        res["iter_cycles"] = iter_cycles
        res["sustained"] = _sustained_stats(iter_cycles)
    if iter_wall_ns:
        res["iter_wall_ns"] = iter_wall_ns
        res["sustained_wall_ns"] = _sustained_stats(iter_wall_ns)
    if stage_wall_ns:
        res["stage_wall_ns"] = stage_wall_ns
    return res


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

    med = (xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) // 2)
    third = max(1, n // 3)
    early = sorted(cycles[:third])[third // 2]
    late = sorted(cycles[-third:])[third // 2]
    return {"n": n, "min": xs[0], "median": med, "p95": _pct(0.95), "max": xs[-1],
            "early_median": early, "late_median": late,
            "drift": (late - early) / early if early else 0.0}


# Per-element relative-error ceiling for the strict tiers. The aggregate cos and global-max-
# normalized ``rel`` terms can BOTH stay near-perfect while a single output element is
# catastrophically wrong: a measured fp16-accumulate GEMM passed the old gate at
# cos=0.9999986 while being 1209% wrong (per-element rel ≈ 12.09) on one element, because the
# blow-up was localized and drowned by the aggregates. Folding a per-element ceiling into each
# conjunction closes that hole. Tunable via MERLIN_GATE_MAX_REL (default 0.05 rejects the
# 1209% case while clearing real recaptures, whose genuine per-element spread runs ~3e-3 —
# e.g. bitvla's fp32 baseline). 0 (or empty) DISABLES the per-element term entirely.
_GATE_MAX_REL = float(os.environ.get("MERLIN_GATE_MAX_REL", "0.05") or "0")

# Localized-error veto for the COSINE-ONLY tier (t3), expressed against the run's own quantization
# floor rather than a constant. Measured across the tracked recaptures, the deviation of a CORRECT
# host int8 reference from the fp32 golden spans 0.027 (small_llama) to 1.88 (gemma2_2b) of the
# output RMS, and 1.3 to 99.0 in per-element relative terms. So NO fixed bound -- relative or
# absolute -- is satisfiable by correct implementations across models: the 0.05 per-element ceiling
# above rejects every one of them, which is why t3 carries no per-element term at all today.
# What IS comparable is the run against the SAME MODEL's own quantization noise: golden_w8a8 is the
# host's int8 result, so max|golden_w8a8 - golden_fp32| is the deviation that correct W8A8 arithmetic
# already costs on this exact output. A conformant accelerator computes the same quantized math in a
# different (equally valid) rounding/accumulation order, so it should land within a small multiple of
# that floor; a localized blow-up lands far outside it. The bound is therefore DERIVED per run and
# needs no per-model tuning. Requires BOTH references (the floor is unmeasurable otherwise) and a
# non-degenerate floor; 0 (or empty) disables it.
_GATE_QUANT_EXCESS = float(os.environ.get("MERLIN_GATE_QUANT_EXCESS", "4.0") or "0")


def _gate(prefix: np.ndarray, references, *, max_rel: float | None = None) -> dict:
    """Multi-tier accuracy gate for an int8 (W8A8) run. ``references`` is either a single
    fp32 reference array (legacy: one ``cos``/``rel``/``ok`` at the strict fp32 threshold) or
    a ``{tier: array}`` dict. Tiers (literature-backed for W8A8 transformers — I-BERT /
    SmoothQuant put W8A8 at cos 0.99–0.999 vs fp32):
      * ``w8a8`` — vs a torch/host W8A8 reference: T1 cos > 0.999 AND rel < 1e-2 AND per-element
        max-rel < ``max_rel`` (the int8 datapath is faithful to the W8A8 math AND no single
        element blew up);
      * ``fp32`` — vs the fp32 golden: T2 cos > 0.99 AND top-1 argmax matches AND per-element
        max-rel < ``max_rel`` (quant degradation within band, no localized blow-up).
    ``ok = T1 or T2``. The per-element term is ``perel = |pref - r| / (|r| + floor)`` with the
    denominator floored at 0.1% of ``|r|.max()`` so genuine near-zero reference elements don't
    manufacture spurious blow-ups, while a real localized error (large element, or a near-zero
    element that jumped) still trips it. ``max_rel`` defaults to the ``MERLIN_GATE_MAX_REL``
    module knob; a value <= 0 disables the per-element term (recovering the old aggregate-only
    behavior). Emits per-tier ``<tier>_cos``/``<tier>_rel``/``<tier>_argmax``/``<tier>_max_rel``."""
    pref = np.asarray(prefix, dtype=np.float32).ravel()
    if not isinstance(references, dict):
        references = {"fp32": references}
    thresh = _GATE_MAX_REL if max_rel is None else float(max_rel)
    out: dict[str, Any] = {}
    _arr: dict[str, np.ndarray] = {}
    k = len(pref)
    for tier, ref in references.items():
        if ref is None:
            continue
        r = np.asarray(ref, dtype=np.float32).ravel()[:k]
        _arr[tier] = r
        rmax = max(1e-9, float(np.abs(r).max()))
        rel = float(np.abs(pref - r).max()) / rmax
        cos = float((pref @ r) / (np.linalg.norm(pref) * np.linalg.norm(r) + 1e-12))
        # per-element relative error over SIGNIFICANT reference elements (|r_i| >= 1% of the RMS
        # scale). Masking by the TYPICAL scale (RMS), not the max, is what makes this robust on both
        # spiky classification logits AND whole-model regression outputs: a near-zero output element
        # with a large relative but negligible ABSOLUTE error (invisible to cos — measured: bitvla
        # raw max-rel 1.05 at cos 0.9999945) is below the RMS floor and does not trip the gate, while
        # a real error on a meaningful element (the fp16-accumulate 1209% disaster; the blow-up
        # constructions) sits above it and is caught. rmax-based flooring failed here because a spiky
        # output makes 1e-3*rmax too small to suppress the near-zero element.
        rms = max(1e-9, float(np.sqrt(np.mean(r.astype(np.float64) ** 2))))
        _sig = np.abs(r) >= 1e-2 * rms
        perel = float((np.abs(pref[_sig] - r[_sig]) / np.abs(r[_sig])).max()) if _sig.any() else 0.0
        out[f"{tier}_cos"] = cos
        out[f"{tier}_rel"] = rel
        out[f"{tier}_argmax"] = bool(int(np.argmax(pref)) == int(np.argmax(r)))
        out[f"{tier}_max_rel"] = perel

    # ---- derived localized-error veto (see _GATE_QUANT_EXCESS): judge the run against the
    # quantization noise THIS model's own references already exhibit, not against a constant.
    _excess_ok, _excess_applied = True, False
    _f, _q = _arr.get("fp32"), _arr.get("w8a8")
    if _GATE_QUANT_EXCESS > 0 and _f is not None and _q is not None:
        _floor = float(np.abs(_q.astype(np.float64) - _f.astype(np.float64)).max())
        _scale = max(1e-12, float(np.abs(_f.astype(np.float64)).max()))
        if _floor > 1e-9 * _scale:  # degenerate (identical references) => unmeasurable, veto stays off
            _run = float(np.abs(pref.astype(np.float64)[:len(_f)] - _f.astype(np.float64)).max())
            out["quant_floor_abs"] = _floor
            out["run_dev_abs"] = _run
            out["quant_excess"] = _run / _floor
            _excess_applied = True
            _excess_ok = out["quant_excess"] <= _GATE_QUANT_EXCESS

    def _perel_ok(tier: str, bound: float | None = None) -> bool:
        # bound <= 0 disables the term; missing tier => vacuously ok (that tier isn't in play).
        b = thresh if bound is None else bound
        return b <= 0 or out.get(f"{tier}_max_rel", 0.0) < b

    t1 = (out.get("w8a8_cos", 0.0) > 0.999 and out.get("w8a8_rel", 1.0) < 1e-2
          and _perel_ok("w8a8"))
    t2 = (out.get("fp32_cos", 0.0) > 0.99 and out.get("fp32_argmax", False)
          and _perel_ok("fp32"))
    # legacy single-reference callers gate at the strict fp32 threshold (+ per-element ceiling)
    legacy = ("w8a8" not in out and out.get("fp32_cos", 0.0) > 0.9999
              and out.get("fp32_rel", 1.0) < 1e-3 and _perel_ok("fp32"))
    # T3 fp32 REGRESSION tier: whole-model regression outputs (VLA policies) are cosine-tight but have
    # NO meaningful top-1 argmax and a global rel ~3e-3 (> the legacy 1e-3 bit-close bar), so t2/legacy
    # reject them despite cos ~0.9999945. This tier is EXACTLY the k1_e2e_xnnpack four-way's accepted
    # whole-model gate: cos >= 0.9999, COSINE-ONLY. Evidence forces this: a whole-model regression
    # output legitimately carries high PER-ELEMENT relative error on its many small (low-absolute)
    # elements -- bitvla's real seed measured per-element max-rel 1.1 (even RMS-masked) at cos
    # 0.9999945 / global rel 0.0031, i.e. numerically correct but per-element-noisy by nature. No
    # per-element RELATIVE bound can pass such an output, which is why the four-way authority (that has
    # always accepted these results) gates on cosine alone. A per-element ABSOLUTE bound CAN gate one,
    # and t3 now carries it whenever it is measurable -- see _GATE_QUANT_EXCESS. That veto is what
    # closes the localized-blow-up hole at this tier; without it a 1209%-style single-element error
    # reaching t3 passed untouched, since cosine is exactly what such an error hides from.
    # The per-element RELATIVE veto is retained where outputs are well-scaled and it is meaningful: the int8 (t1/w8a8) and classification
    # (t2/argmax) and bit-close (legacy) tiers, plus the fp16 driver's own gate. Whole-model fp16/int8
    # forks are still per-element-guarded via t1 (w8a8 ref) and the dtype drivers.
    t3 = (out.get("fp32_cos", 0.0) > 0.9999 and "fp32_cos" in out and _excess_ok)
    out["cos"] = out.get("w8a8_cos", out.get("fp32_cos"))
    out["rel"] = out.get("w8a8_rel", out.get("fp32_rel"))
    out["max_rel"] = out.get("w8a8_max_rel", out.get("fp32_max_rel"))
    out["ok"] = bool(t1 or t2 or legacy or t3)
    # Which tiers were actually in play, and which one carried the verdict. Without this the
    # fallback is INVISIBLE: grading a W8A8 run with no `golden_w8a8.npy` silently drops to the
    # fp32 tier, and the ordinary weight-only-vs-W8A8 gap then reads as a codegen defect. That
    # exact confusion cost a multi-hour hunt for a TinyLlama int8 "board bug" that did not exist
    # (the board matched the host W8A8 reference at rel 0.0). Callers that know the run is int8
    # should treat a missing `w8a8` tier as an incomplete measurement, not a failure.
    out["tiers"] = sorted(t for t in ("w8a8", "fp32") if f"{t}_cos" in out)
    out["tier_ok"] = ("w8a8" if t1 else "fp32" if t2 else
                      "fp32_legacy" if legacy else "fp32_cos_only" if t3 else None)
    # DID A PER-ELEMENT VETO ACTUALLY APPLY? t3 is cosine-only only WHEN THE DERIVED VETO CANNOT BE
    # MEASURED (one reference, or two identical ones) -- a whole-model regression
    # output legitimately carries high per-element relative error on its many small elements, and no
    # per-element bound can pass one. But t3 fires for ANY output whose cosine clears 0.9999, including
    # one where the veto WOULD have been meaningful and WOULD have failed: a measured whole model passed
    # here at cos 0.99993 with per-element max-rel 2.82 and global rel 0.0148, having failed both t1 and
    # t2 on exactly those numbers. That is a real pass and also a much weaker one, and a caller quoting
    # `ok` alone cannot tell the two apart. Stated as a field so a certification can carry its own
    # strength instead of the reader having to reverse-engineer it from a tier name.
    # When both references are present the run IS per-element guarded -- by the quantization-excess
    # bound, the veto actually applicable to this output class (the relative one rejects correct
    # reference implementations, so its absence here was never evidence of a defect).
    out["per_element_guarded"] = bool(t1 or t2 or legacy or (t3 and _excess_applied))
    out["per_element_basis"] = ("relative" if (t1 or t2 or legacy)
                                else "quantization_excess" if (t3 and _excess_applied) else None)
    return out


def _check_firesim_workload(firesim_root: str, workload: str) -> None:
    """Fail closed when the shared ``config_runtime.yaml`` names a different workload than the one we
    stage into.

    ``FIRESIM_WORKLOAD_NAME`` only tells the RUNNER where to put the ELF. What FireSim boots comes from
    ``workload.workload_name`` in ``config_runtime.yaml``, and when the two disagree the simulator loads
    whatever binary the other workload last left behind -- with no error anywhere, because from
    FireSim's side nothing is wrong. That cost a 63-minute FPGA run whose uartlog turned out to be a
    months-old image for a different accelerator, trapping on its first custom instruction.

    The queue path is immune (the daemon is passed ``--workload`` and writes a per-job config), so this
    is checked only where the shared file is actually consulted.
    """
    import yaml
    cfg = Path(firesim_root) / "deploy" / "config_runtime.yaml"
    if not cfg.is_file():
        return
    declared = ((yaml.safe_load(cfg.read_text()) or {}).get("workload", {}) or {}).get("workload_name")
    want = f"{workload}.json"
    if declared != want:
        raise RuntimeError(
            f"{cfg} declares workload_name={declared!r} but this run stages into {workload!r}. FireSim "
            f"would boot the binary belonging to {declared!r}, not ours. Set workload_name to {want!r} "
            f"(and restore it afterwards), or run through the queue, which manages its own config.")


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
    # Resolve through paths.env (os.environ -> .env -> default), NOT os.environ alone: every
    # other external dependency in the repo is configurable from the gitignored .env without
    # exporting into the shell, and reading os.environ directly silently ignored a configured
    # MERLIN_MODELBLASTER and failed with a bare ModuleNotFoundError for 'modelblaster'.
    from ...common.paths import env as _env
    mb = _env("MERLIN_MODELBLASTER", "/path/to/ModelBlaster")
    for p in (f"{mb}/src", mb):
        if p not in sys.path:
            sys.path.insert(0, p)
    # FIRESIM_ROOT/FIRESIM_ENV default to the configured chipyard checkout rather than a
    # placeholder, so a repo with MERLIN_CHIPYARD set needs no extra FireSim-specific config.
    _cy = _env("MERLIN_CHIPYARD", "/path/to/chipyard")
    fr = firesim_root or _env("FIRESIM_ROOT", f"{_cy}/sims/firesim")
    fe = firesim_env or _env("FIRESIM_ENV", f"{_cy}/env.sh")
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
    else:
        # Without the queue the shared config_runtime.yaml is what FireSim reads, so it has to agree
        # with where we stage. See _check_firesim_workload.
        _check_firesim_workload(fr, os.environ["FIRESIM_WORKLOAD_NAME"])
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
                  timeout: int = 3600, int8_compute: bool = False,
                  n_harts: int = 1, iters: int = 1, warmup: int = 0,
                  omp_threads: int | None = None,
                  rvv_schedule: str | None = None,
                  cflags_override: list[str] | None = None,
                  features: "frozenset[str] | None" = None,
                  vlen: int | None = None) -> dict[str, Any]:
    """Build the Zephyr image and (for spike) run + gate. ``references`` (a ``{tier: array}``
    dict, e.g. ``{"w8a8": ..., "fp32": ...}``) drives the multi-tier int8 gate; a single
    ``reference`` array keeps the legacy strict fp32 gate.

    ``n_harts``/``iters``/``warmup`` select the multicore and sustained-inference images;
    ``rvv_schedule``/``cflags_override``/``features`` pass an RVV package's codegen through
    (the same seam :mod:`merlin.mining.apply` uses) so a package can be measured here.
    """
    # spike needs at least as many harts as the image will use, or the pinned workers have
    # nowhere to run and the pool silently clamps to fewer threads.
    harts = max(harts, n_harts, rvv_hart + 1)
    b = build_app(model_dir, work, board=board, backend=backend, rvv_hart=rvv_hart,
                  arena_mb=arena_mb, cpus=max(harts, rvv_hart + 1), int8_compute=int8_compute,
                  n_harts=n_harts, iters=iters, warmup=warmup, omp_threads=omp_threads,
                  rvv_schedule=rvv_schedule,
                  cflags_override=cflags_override, features=features, vlen=vlen)
    result: dict[str, Any] = {"elf": str(b["elf"]), "app_dir": str(b["app_dir"]),
                              "ram_bytes": b["ram_bytes"], "n_harts": n_harts, "iters": iters,
                              # WHICH image produced this run. A caller that gates here and ships a
                              # differently-configured build (a board whose console spike cannot
                              # service) needs to be able to say so, rather than leave the recipient
                              # to discover that the console beside their ELF names another binary.
                              "build_hash": b.get("build_hash", ""),
                              # the vector length the build AND the run agreed on (None = spike's
                              # default 128); recorded so a result cannot be read as another VLEN's
                              "vlen": vlen}
    if board != "spike_riscv64":
        return result  # FireSim path runs the elf separately (firesim_runner / queue)
    run = run_on_spike(b["elf"], harts=harts, mem_bytes=b["ram_bytes"], timeout=timeout,
                       vlen=vlen)
    result.update(run)
    refs = references if references is not None else reference
    if refs is not None:
        result.update(_gate(run["prefix"], refs))
    return result
