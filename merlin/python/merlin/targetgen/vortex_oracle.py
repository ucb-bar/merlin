"""Vortex oracle adapters — run a package's emitted LLVM-dialect module on simx / rtlsim.

The Vortex analog of the chipyard spike/verilator adapters in :mod:`capsule_runner`, and the reason
``target_experiment.yaml`` declares ``toolchain.sim_via: vortex``. Tiers:

    L2  simx    functionally complete, cycle-APPROXIMATE. The default numeric oracle: fast enough to
                grade every capsule (a 64-point vecadd runs in seconds).
    L3  rtlsim  Verilator on the real RTL, cycle-exact. Reserved for the capsules that declare it —
                measured throughput is only ~5-30 kHz simulated cycles.

Both run the SAME image, so a capsule that passes L2 and L3 differs only in cycle fidelity.

Pipeline (all runner-owned; the package only supplies the LLVM-dialect text):

    llvm-dialect MLIR --mlir-translate--> LLVM IR --stock clang--> rv64 object
      --link with the curated harness--> ELF --vxbin.py--> .vxbin --host driver--> OUT/METRIC/DONE

Two properties this preserves, both established during bring-up and both load-bearing:

* **Stock LLVM.** The package's object is built with an unmodified clang. The one piece that needs the
  Vortex toolchain — the ``annotate("vortex.kernel")`` KMU entry stub — is prebuilt into the harness
  (``vx_entry.o``), so it never reaches the package's compiler.
* **Bit-exact inputs.** The host driver fills operands from a fixed LCG that :func:`fill_f32` /
  :func:`fill_i8` / :func:`fill_i32` reproduce here, so goldens can be computed offline without
  running the device. Verified element-for-element.

Unavailability is honest: if the harness is not staged or the Vortex build tree is absent, the
adapters raise ``OracleUnavailable`` rather than silently passing (``not_run_is_not_pass``).
"""
from __future__ import annotations

import json
import math
import os
import shutil
import struct
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..common.paths import build_dir, env

# The console protocol the host driver emits (see contracts/harness_curated/.../merlin_vx_host.cpp).
# Parsed structurally (no regex): one helper per line kind, each with exactly one meaning.

def _iter_out_lines(console: str):
    """Yield (name, dtype, nbytes, hex_payload) for each `OUT <name> <dtype> <bytes> <hex>` line."""
    for ln in console.splitlines():
        if not ln.startswith("OUT "):
            continue
        toks = ln.split()
        if len(toks) < 4 or not toks[3].isdigit():
            continue
        yield toks[1], toks[2], toks[3], (toks[4] if len(toks) > 4 else "")


def _console_done(console: str) -> bool:
    """True if the driver printed its terminal `DONE` line."""
    return any(ln.strip() == "DONE" for ln in console.splitlines())


def _parse_perf(console: str) -> tuple[int | None, int | None]:
    """(instrs, cycles) from the runtime `PERF: instrs=<n>,cycles=<n>,IPC=<..>` line, else (None, None)."""
    for ln in console.splitlines():
        if "instrs=" not in ln or "cycles=" not in ln:
            continue
        fields: dict[str, str] = {}
        for tok in ln.replace("PERF:", " ").split(","):
            k, _, v = tok.strip().partition("=")
            fields[k.strip()] = v.strip()
        try:
            return int(fields["instrs"]), int(fields["cycles"])
        except (KeyError, ValueError):
            continue
    return None, None

DRIVERS = {"L2": "simx", "L3": "rtlsim"}

# The frozen machine from target_experiment.yaml, in the DEVICE's units.
#
# Careful: `cores` here is TOTAL cores as `VX_CAPS_NUM_CORES` reports them, whereas the descriptor's
# `NUM_CORES` (and the build macro) is cores PER CLUSTER. The frozen 2 clusters x 2 cores-per-cluster
# therefore shows up as cores=4. Conflating the two makes this guard either never fire or always fire.
#
# Why the guard exists: simx and rtlsim are separately-built simulators with no structural link, so
# they drift — and the failure is silent. An L2-less 2x2 rtlsim left over from an unrelated sweep
# returned 0xBAADF00D poison rather than an error, which grades as a wrong answer from the backend
# instead of a broken rig. Verified per run against the host driver's `METRIC geometry` line.
FROZEN_GEOMETRY = {"clusters": 2, "cores": 4, "warps": 8, "threads": 8}

# The one true build string; every consumer (simx, rtlsim, the HW-dialect import, any arc model) must
# be built from exactly this or its results are not comparable.
FROZEN_BUILD_MACRO = ("-DVX_CFG_NUM_CLUSTERS=2 -DVX_CFG_NUM_CORES=2 -DVX_CFG_SOCKET_SIZE=2 "
                      "-DVX_CFG_NUM_WARPS=8 -DVX_CFG_NUM_THREADS=8 -DVX_CFG_L2_ENABLE")
def geometry_from_console(console: str) -> dict[str, int] | None:
    """The geometry the run actually executed on, or None if the driver did not report it."""
    for ln in console.splitlines():
        if ln.startswith("METRIC geometry "):
            rest = ln[len("METRIC geometry "):].strip()
            try:
                return {k: int(v) for k, v in (kv.split("=") for kv in rest.split())}
            except ValueError:
                return None
    return None


class VortexUnavailable(RuntimeError):
    """The Vortex oracle cannot run here (missing harness, build tree, or toolchain)."""


# --------------------------------------------------------------------------------------- inputs
# EXACT mirror of fill() in merlin_vx_host.cpp. Changing either without the other silently
# invalidates every golden, so they are pinned by tests/targetgen/test_vortex_oracle.py.

def _lcg(seed: int, n: int) -> list[int]:
    s, out = seed & 0xFFFFFFFF, []
    for _ in range(n):
        s = (s * 1664525 + 1013904223) & 0xFFFFFFFF
        out.append(s)
    return out


def fill_f32(seed: int, count: int) -> list[float]:
    """`count` float32 values in [-1, 1), as the host driver writes them."""
    return [float((v >> 8) - (1 << 23)) / float(1 << 23) for v in _lcg(seed, count)]


def fill_i8(seed: int, count: int) -> list[int]:
    """`count` int8 values, as the host driver writes them."""
    return [((v >> 24) ^ 0x80) - 0x80 for v in _lcg(seed, count)]


def fill_i32(seed: int, count: int) -> list[int]:
    """`count` int32 values, as the host driver writes them."""
    return [((v >> 16) & 0xFFFF) - (1 << 15) for v in _lcg(seed, count)]


_UNPACK = {"f32": "<%df", "i32": "<%di", "i8": "<%db"}
_WIDTH = {"f32": 4, "i32": 4, "i8": 1}
FILLS = {"f32": fill_f32, "i32": fill_i32, "i8": fill_i8}


# ------------------------------------------------------------------------------- the launch plan

def operand_seed(capsule_name: str, operand_name: str) -> int:
    """The deterministic RNG seed for one capsule operand (FNV-1a over "<capsule>/<operand>").

    Keyed by NAME, not position, so adding or reordering an operand does not silently change the data
    every other operand sees — which would invalidate goldens that were not regenerated.
    """
    h = 0x811C9DC5
    for b in f"{capsule_name}/{operand_name}".encode():
        h = ((h ^ b) * 0x01000193) & 0xFFFFFFFF
    return h or 1                     # the host treats 0 as "unseeded"


def resample_seed(capsule_name: str, operand_name: str, salt: str) -> int:
    """A per-run HELD-OUT seed for one operand — the public :func:`operand_seed` reseeded by ``salt``.

    The public seed is FNV over "<capsule>/<operand>", so the inputs a capsule is graded on are
    derivable by anyone who knows its name — which means a kernel that ignores its inputs and writes a
    baked-in constant equal to the (also-derivable) golden passes. Grading a frozen package on inputs
    drawn from a per-run SECRET ``salt`` instead closes that: the salt is chosen at grade time OUTSIDE
    the agent sandbox and never enters the tree the agent compiled against, so a constant-folding kernel
    cannot have baked in the right answer. Same fill distribution as the public draw (this is A.1 —
    DIFFERENT values, not edge values; edge regimes are the A.2 numeric-stress capsules); only the seed
    differs, so no host-driver change is needed (the driver fills from whatever seed the plan carries).
    """
    return operand_seed(f"{salt}\x1f{capsule_name}", operand_name)


def _elems(shape) -> int:
    n = 1
    for d in shape:
        n *= int(d)
    return n


def plan_from_capsule(capsule: dict[str, Any], *, grid: int, salt: str | None = None) -> dict[str, Any]:
    """Build the host driver's launch plan from a capsule.

    Vortex capsules list **every** operand in `inputs[]`, outputs included (`role: output` is already
    in the schema enum) — unlike the gemmini capsules, which name the output only in
    `operation.attributes.out`. A programmable core needs the output's shape and dtype up front to
    size the buffer, and there is no command buffer to infer it from.

    `grid` is NOT derived here. How many coordinates to launch is a *mapping* decision, and mapping is
    the compiler work under test — so it comes from the package's own module annotation
    (:func:`grid_from_module`), not from the capsule.

    ``salt`` (A.1 held-out grading) reseeds every input operand via :func:`resample_seed`, so a frozen
    package is graded on a fresh input draw it could not have baked in. ``None`` (the default) keeps the
    public :func:`operand_seed` draw, so the plan is byte-identical to before.
    """
    name = capsule["name"]
    args = []
    for spec in capsule.get("inputs", []):
        dtype = spec["dtype"]
        if dtype not in _WIDTH:
            raise ValueError(f"{name}: operand {spec['name']} has dtype {dtype!r}, "
                             f"which the Vortex host driver cannot fill (known: {sorted(_WIDTH)})")
        if spec["role"] == "output":
            seed = 0
        elif salt is not None:
            seed = resample_seed(name, spec["name"], salt)
        else:
            seed = operand_seed(name, spec["name"])
        args.append({"name": spec["name"], "role": spec["role"], "dtype": dtype,
                     "bytes": _elems(spec["shape"]) * _WIDTH[dtype], "seed": seed})
    if not any(a["role"] == "output" for a in args):
        raise ValueError(f"{name}: no operand with role 'output'; the host has nothing to read back")
    return {"grid": int(grid), "args": args}


def grid_from_module(llvm_text: str) -> int:
    """The launch grid the package's module declares (`merlin.grid = <n>` module attribute).

    Absence is a package error, not something to paper over with a default: silently launching one
    coordinate per output element would hand every backend the same mapping and stop the benchmark
    measuring the thing it exists to measure. Parsed structurally: the first `merlin.grid` token
    followed by `= <digits>` wins.
    """
    key = "merlin.grid"
    start = 0
    while True:
        idx = llvm_text.find(key, start)
        if idx == -1:
            break
        start = idx + len(key)
        rest = llvm_text[start:].lstrip()
        if rest.startswith("="):
            digits = rest[1:].lstrip()
            n = ""
            for ch in digits:
                if not ch.isdigit():
                    break
                n += ch
            if n:
                return int(n)
    raise ValueError("module declares no `merlin.grid` attribute; the backend must state the "
                     "launch grid it was compiled for")


# ------------------------------------------------------------------------------ float tolerance
# f32 unit roundoff. IEEE-754 binary32 has a 24-bit significand, so u = 2^-24.
F32_UNIT_ROUNDOFF = 2.0 ** -24


def dot_error_bound(term_magnitudes: list[float], *, safety: float = 2.0) -> float:
    """Worst-case f32 error for a length-K dot product, from the classical gamma_K bound.

    For a summation of K products, |fl(sum) - sum| <= gamma_K * sum|x_i*y_i| with
    gamma_K = K*u / (1 - K*u). Two properties make this the right basis for a capsule's `atol`:

    * It is the bound for **sequential** summation, which is the loosest legal order. Tree/pairwise
      reduction has a strictly tighter bound (gamma_log2(K)), so a tolerance derived here admits any
      reassociation the compiler may legally choose — exactly what must not be penalised.
    * It scales with the actual operand magnitudes, so a capsule's tolerance follows its data instead
      of being a round number picked by hand.

    `safety` covers what the bound does not model (the compiler may also contract to FMA, reorder
    across tiles, or accumulate in a wider register and round once). Keep it small: the point is a
    tolerance with a stated basis, not a tolerance wide enough to hide bugs.
    """
    k = len(term_magnitudes)
    if k == 0:
        return 0.0
    ku = k * F32_UNIT_ROUNDOFF
    if ku >= 1.0:                    # bound degenerates; K is far beyond anything a capsule uses
        raise ValueError(f"reduction length {k} is too large for the gamma_K bound")
    return (ku / (1.0 - ku)) * math.fsum(term_magnitudes) * safety


def derive_matmul_atol(lhs: list[float], rhs: list[float], m: int, k: int, n: int,
                       *, safety: float = 2.0, rhs_transposed: bool = False) -> float:
    """`atol` for an (m,k)x(k,n) f32 matmul: the worst per-element dot-product bound.

    Operands are row-major flat lists (as :func:`reference_inputs` returns them). Computed in Python
    floats (binary64) so the bound itself does not suffer the error it is bounding.

    `rhs_transposed` selects the B[j,k] indexing of an `A @ Bt` capsule. The bound is the same
    quantity either way — only which products are summed changes — so the two share this code rather
    than drifting as separate copies.
    """
    worst = 0.0
    for i in range(m):
        for j in range(n):
            mags = [abs(lhs[i * k + p] * (rhs[j * k + p] if rhs_transposed else rhs[p * n + j]))
                    for p in range(k)]
            worst = max(worst, dot_error_bound(mags, safety=safety))
    return worst


def derive_reduction_atol(values: list[float], rows: int, cols: int,
                          *, safety: float = 2.0) -> float:
    """`atol` for a row-wise f32 sum of a (rows, cols) tensor.

    A reduction is a dot product against 1, so this is :func:`dot_error_bound` applied per row and
    maxed — the same gamma_K reasoning, and likewise loose enough for any legal reassociation.
    """
    worst = 0.0
    for i in range(rows):
        worst = max(worst, dot_error_bound([abs(v) for v in values[i * cols:(i + 1) * cols]],
                                           safety=safety))
    return worst


# Relative accuracy we allow the backend's `exp`. This one number is NOT derived from a bound — it is
# a stated assumption, and the only such number in the corpus. Rationale: `math.exp` has no
# hardware instruction on Vortex, so the compiler supplies a polynomial/table approximation whose
# error is an implementation choice, not a property of the arithmetic. 2^-20 (~16 ulps) admits any
# competently-implemented approximation while still being ~250x tighter than a wrong reduction order
# or a mis-indexed exponent would produce. If a softmax capsule ever fails only just, suspect this
# constant before suspecting the bound around it.
F32_EXP_REL_ERR = 2.0 ** -20


def derive_softmax_atol(rows: int, cols: int, *, safety: float = 2.0) -> float:
    """`atol` for a row-wise f32 softmax over `cols` elements.

    The chain is max (exact) -> exp (approximate) -> sum (gamma_K) -> divide (one rounding), and the
    output is in [0, 1], so bounding the largest possible output at 1 gives a data-independent bound:

        atol <= 1 * (exp_rel_err + gamma_cols + u) * safety

    Unlike the other derivations here this is not purely a rounding bound — see
    :data:`F32_EXP_REL_ERR` for the one assumption it rests on.
    """
    ku = cols * F32_UNIT_ROUNDOFF
    if ku >= 1.0:
        raise ValueError(f"softmax width {cols} is too large for the gamma_K bound")
    gamma = ku / (1.0 - ku)
    return (F32_EXP_REL_ERR + gamma + F32_UNIT_ROUNDOFF) * safety


def derive_elementwise_atol(values: list[float], *, safety: float = 2.0) -> float:
    """`atol` for an f32 elementwise op: a single rounding of the largest result magnitude."""
    if not values:
        return 0.0
    return F32_UNIT_ROUNDOFF * max(abs(v) for v in values) * safety


# Relative accuracy we allow the backend's `tanh`. The SECOND stated assumption in the corpus, and it
# exists for the same reason as `F32_EXP_REL_ERR`: `math.tanh` has no RISC-V instruction, so whatever
# polynomial or rational approximation the backend supplies is an implementation choice rather than a
# property of the arithmetic. Set to the same 2^-20 (~16 ulps) — tanh is bounded in [-1, 1] and easier
# to approximate well than exp, so a budget that is adequate for exp is not generous here.
#
# NOTE `math.sqrt` deliberately gets NO such constant. IEEE-754 requires sqrt to be correctly rounded
# and Vortex implements `fsqrt` in hardware (hw/rtl/fpu/VX_fsqrt_unit.sv), so it is worth exactly one
# unit roundoff like any other arithmetic op — inventing a slack constant for it would widen a
# tolerance that the hardware does not need widened.
F32_TANH_REL_ERR = 2.0 ** -20


def derive_rmsnorm_atol(values: list[float], rows: int, cols: int, *, eps: float = 1e-5,
                        safety: float = 2.0) -> float:
    """`atol` for a row-wise f32 RMS normalisation, y = x / sqrt(mean(x^2) + eps).

    `values` are the OUTPUTS (the normalised tensor), because the bound is relative to the result.

    The chain and where each term comes from:

    * ``sum(x^2)`` — a length-`cols` reduction, so ``gamma_cols`` and, as everywhere else here, the
      SEQUENTIAL bound, which admits any reassociation the compiler may legally pick;
    * ``/ cols`` and ``+ eps`` — one rounding each;
    * ``sqrt`` — correctly rounded (see :data:`F32_TANH_REL_ERR`'s note), and a square root HALVES the
      relative error of its argument, hence the ``/2``;
    * ``x / r`` — one final rounding.

    Unlike softmax this is a pure rounding bound with no approximation constant in it, which is the
    point of choosing sqrt over rsqrt for the capsule: the tolerance stays tight enough to catch a
    wrong reduction or a mis-broadcast row.
    """
    ku = cols * F32_UNIT_ROUNDOFF
    if ku >= 1.0:
        raise ValueError(f"rmsnorm width {cols} is too large for the gamma_K bound")
    gamma = ku / (1.0 - ku)
    rel_r = (gamma + 2.0 * F32_UNIT_ROUNDOFF) / 2.0 + F32_UNIT_ROUNDOFF
    peak = max((abs(v) for v in values), default=0.0)
    return peak * (rel_r + F32_UNIT_ROUNDOFF) * safety


def derive_gelu_atol(values: list[float], *, safety: float = 2.0) -> float:
    """`atol` for the tanh-approximation GELU, y = 0.5*x*(1 + tanh(c1*(x + c2*x^3))).

    Purely elementwise — no reduction, so no gamma term. The error is the backend's `tanh` accuracy
    (:data:`F32_TANH_REL_ERR`) plus the handful of roundings around it: three multiplies and an add
    building the argument, then the add/multiply/multiply forming the result. Counted as 6u rather
    than bounded op-by-op, which is loose in the right direction and still orders of magnitude below
    what a wrong constant or a missing cubic term would produce.

    That this capsule needs NO reduction term is exactly why it is worth having next to softmax: if a
    backend fails softmax but passes this, its transcendental is fine and its two-pass reduction is
    not, and the corpus can now tell those apart.
    """
    if not values:
        return 0.0
    peak = max(abs(v) for v in values)
    return peak * (F32_TANH_REL_ERR + 6.0 * F32_UNIT_ROUNDOFF) * safety


def derive_attention_atol(q: list[float], k: list[float], v: list[float], s: int, d: int,
                          *, safety: float = 2.0) -> float:
    """`atol` for fused single-head attention ``out = softmax(Q @ K^T) @ V`` (f32).

    A composition of the three primitives' bounds, propagated in order — the same shape of reasoning as
    :func:`derive_gelu_atol`'s note on a matmul feeding a transcendental, one stage longer:

    * ``scores = Q @ K^T`` — a length-``d`` dot per element, so its worst absolute error is a
      :func:`dot_error_bound` (``e_scores``).
    * ``P = softmax(scores)`` — softmax is 1-Lipschitz-ish from logits to probabilities in the sense
      ``||dP||_1 <= 2 * ||d(logits)||_inf``, so the score error contributes ``2 * e_scores`` to the
      row's L1 probability error; the softmax's OWN rounding adds ``exp_rel + gamma_s + u`` per element
      (bounded, since each probability is in [0, 1]), i.e. ``s *`` that across the row.
    * ``out = P @ V`` — each output element is ``sum_j P[i,j] V[j,d]`` with ``sum_j P = 1``, so the final
      matmul's own rounding is ``gamma_s * max|V|`` and the propagated probability error is
      ``||dP||_1 * max|V|``.

    Operands are row-major flat lists (as :func:`reference_inputs` returns them); computed in binary64.
    """
    u = F32_UNIT_ROUNDOFF
    e_scores = 0.0
    for i in range(s):
        for j in range(s):
            mags = [abs(q[i * d + p] * k[j * d + p]) for p in range(d)]
            e_scores = max(e_scores, dot_error_bound(mags, safety=1.0))
    ku = s * u
    if ku >= 1.0:
        raise ValueError(f"attention seq length {s} is too large for the gamma_K bound")
    gamma_s = ku / (1.0 - ku)
    l1_dp = 2.0 * e_scores + s * (F32_EXP_REL_ERR + gamma_s + u)
    max_v = max((abs(x) for x in v), default=0.0)
    return (gamma_s * max_v + l1_dp * max_v) * safety


def reference_inputs(capsule: dict[str, Any], *, salt: str | None = None) -> dict[str, list]:
    """The exact operand values the device will be given — for computing goldens offline.

    ``salt`` (A.1 held-out grading) draws each operand from :func:`resample_seed` instead of the public
    :func:`operand_seed`, so a golden computed here matches a device run launched from the same salt.
    ``None`` (the default) is the public draw, byte-identical to before.
    """
    out = {}
    for spec in capsule.get("inputs", []):
        if spec["role"] == "output":
            continue
        dtype = spec["dtype"]
        seed = (resample_seed(capsule["name"], spec["name"], salt) if salt is not None
                else operand_seed(capsule["name"], spec["name"]))
        out[spec["name"]] = FILLS[dtype](seed, _elems(spec["shape"]))
    return out


def decode_out(console: str) -> dict[str, list]:
    """Parse the driver's `OUT <name> <dtype> <bytes> <hex>` lines into value lists."""
    outputs: dict[str, list] = {}
    for name, dtype, nbytes, payload in _iter_out_lines(console):
        raw = bytes.fromhex(payload)
        if len(raw) != int(nbytes):
            raise ValueError(f"output {name}: declared {nbytes}B, got {len(raw)}B")
        width = 1 if dtype == "i8" else 4
        outputs[name] = list(struct.unpack(_UNPACK[dtype] % (len(raw) // width), raw))
    return outputs


# ------------------------------------------------------------------------------------- toolchain

def _vortex_home() -> Path:
    home = env("MERLIN_EXT_VORTEX") or os.environ.get("VORTEX_HOME")
    if not home:
        raise VortexUnavailable("set MERLIN_EXT_VORTEX (or VORTEX_HOME) to the Vortex checkout")
    return Path(home)


def _harness() -> Path:
    """The staged curated harness (contracts/harness_curated/vortex-baremetal/scripts/build_harness.sh)."""
    staged = Path(env("MERLIN_VORTEX_HARNESS") or (build_dir() / "vortex-harness"))
    if not (staged / "lib" / "libvortex_curated.a").is_file():
        raise VortexUnavailable(f"curated harness not staged at {staged}; run build_harness.sh")
    return staged


def available(tier: str = "L2") -> bool:
    """True if this tier can actually run here (never raises)."""
    try:
        _harness()
        home = _vortex_home()
        return (home / "build" / "sw" / "runtime").is_dir() and shutil.which("clang") is not None
    except Exception:  # noqa: BLE001 - availability probe
        return False


def _sh(cmd: list[str], timeout: int, cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=cwd)


# ----------------------------------------------------------------------------------------- build

def _llvm_tool(name: str) -> Path:
    """A STOCK LLVM binutil (`llvm-objcopy` / `llvm-objdump`), never the Vortex fork's copy.

    These two are plain ELF utilities with no Vortex-specific behaviour, so the fork's copies bought
    nothing — and cost a great deal: `tools/llvm-vortex` is DENIED to every arm (its clang implements
    `+xvortex`, which inserts the split/join reconvergence V10/H9 exist to measure), so reaching them
    meant binding a denied tree into the agent sandbox. Resolving stock copies instead lets the sandbox
    bind only the sysroot and the driver libs.

    Order: an explicit `MERLIN_LLVM_<TOOL>`, then the stock clang's own bin dir (they ship together),
    then PATH. Raises VortexUnavailable rather than silently falling back to the fork.
    """
    override = env(f"MERLIN_LLVM_{name.replace('-', '_').upper()}")
    if override and Path(override).is_file():
        return Path(override)
    clang = shutil.which("clang") or shutil.which("clang-23")
    if clang:
        sibling = Path(clang).resolve().parent / name
        if sibling.is_file():
            return sibling
    found = shutil.which(name)
    if found:
        return Path(found)
    raise VortexUnavailable(
        f"no stock {name} (looked at MERLIN_LLVM_{name.replace('-', '_').upper()}, the stock clang's "
        f"bin dir, and PATH). The Vortex fork's copy is deliberately NOT used — it lives under the "
        f"denied tools/llvm-vortex tree.")


def _stock_clangxx() -> Path:
    """The stock ``clang++`` used to drive the harness link (never the fork's)."""
    override = env("MERLIN_CLANGXX")
    if override and Path(override).is_file():
        return Path(override)
    found = shutil.which("clang++") or shutil.which("clang++-23")
    if not found:
        raise VortexUnavailable("no stock clang++ on PATH (set MERLIN_CLANGXX)")
    return Path(found)


def _clang_flags(tools: Path) -> list[str]:
    """The frozen-ABI clang flags, shared by the compile and the link.

    One list because the two steps MUST agree: `compile_object` produces the object the coverage gate
    reads and the link consumes, and a divergence in `-march`/`-mabi` between them would surface as a
    link error rather than as the ABI mismatch it is. The width follows `target_experiment.yaml`
    (`toolchain.compiler`), which `test_vortex_oracle` pins against this function.
    """
    sysroot = tools / "riscv64-gnu-toolchain" / "riscv64-unknown-elf"
    return ["--target=riscv64-unknown-elf", f"--sysroot={sysroot}",
            f"--gcc-toolchain={tools / 'riscv64-gnu-toolchain'}",
            "-march=rv64imafd", "-mabi=lp64d", "-O3", "-mcmodel=medany"]


def compile_object(llvm_text: str, workdir: Path, *, timeout: int = 600) -> Path:
    """LLVM-dialect MLIR -> the agent's `kernel.o`, via mlir-translate + STOCK clang.

    Split out of :func:`build_image` because the SIMT coverage gate needs exactly this artifact and
    nothing after it. The gate must read the AGENT's object rather than the linked image: the harness's
    startup contains SIMT ops of its own, so scanning the ELF would credit them to the agent (see
    `targetgen.vortex_coverage`). Keeping one compile path means the graded object and the gated object
    cannot drift apart.
    """
    home = _vortex_home()
    tools = home / "tools"
    workdir.mkdir(parents=True, exist_ok=True)
    mlir, ll, obj = workdir / "kernel.mlir", workdir / "kernel.ll", workdir / "kernel.o"
    mlir.write_text(llvm_text)

    translate = env("MERLIN_MLIR_TRANSLATE") or shutil.which("mlir-translate")
    if not translate:
        raise VortexUnavailable("no mlir-translate (set MERLIN_MLIR_TRANSLATE)")
    r = _sh([str(translate), "--mlir-to-llvmir", str(mlir), "-o", str(ll)], timeout)
    if r.returncode != 0:
        raise ValueError(f"mlir-translate failed: {r.stderr[:400]}")

    # STOCK clang: the package's code never touches the Vortex fork.
    r = _sh([shutil.which("clang"), *_clang_flags(tools), "-c", str(ll), "-o", str(obj)], timeout)
    if r.returncode != 0:
        raise ValueError(f"clang failed on the emitted IR: {r.stderr[:400]}")
    return obj


def build_image(llvm_text: str, workdir: Path, *, timeout: int = 600) -> Path:
    """LLVM-dialect MLIR -> a loadable `.vxbin`, via stock clang + the curated harness."""
    home, harness = _vortex_home(), _harness()
    tools = home / "tools"
    obj = compile_object(llvm_text, workdir, timeout=timeout)
    elf, vxbin = workdir / "kernel.elf", workdir / "kernel.vxbin"
    common = _clang_flags(tools)

    # The MINIMAL startup goes ahead of the archive so the archive's fully-featured vx_start is never
    # pulled in: its __init_tls / __libc_init_array run on every hart at CTA entry, which corrupts
    # state across the 4 cores' non-coherent L1s (silent on simx and on a 1-core RTL build).
    # STOCK clang++ drives the link too. This step is a pure linker invocation — a linker script, the
    # prebuilt harness objects, and static archives; nothing is compiled, so the fork's `+xvortex`
    # codegen never applies. Using the fork here was the LAST thing forcing `tools/llvm-vortex` (denied
    # to every arm) into the sandbox. Verified end-to-end: a stock-linked image builds AND runs on simx.
    vx_clang = _stock_clangxx()
    r = _sh([str(vx_clang), *common, "-nostartfiles", "-nostdlib",
             str(harness / "lib" / "vx_start_min.o"),
             str(harness / "lib" / "vx_entry.o"), str(obj),
             "-Wl,-Bstatic,--gc-sections,-T," + str(harness / "link" / "link64.ld")
             + ",--defsym=STARTUP_ADDR=0x180000000",
             str(harness / "lib" / "libvortex_curated.a"),
             f"-L{tools / 'libc64' / 'lib'}", "-lm", "-lc",
             str(tools / "libcrt64" / "lib" / "baremetal" / "libclang_rt.builtins-riscv64.a"),
             "-o", str(elf)], timeout)
    if r.returncode != 0:
        raise ValueError(f"link against the curated harness failed: {r.stderr[:400]}")

    # The staged startup provides KMU dispatch only. If the linked image actually needs gp / TLS /
    # init_array, that startup is wrong for it — fail loudly rather than launch a kernel whose
    # prologue silently does not run (the failure mode is unwritten output buffers, not a crash).
    detect = harness / "kernel_startup.sh"
    if detect.is_file():
        d = _sh(["bash", str(detect), str(_llvm_tool("llvm-objdump")), str(elf)], timeout)
        if d.stdout.strip():
            raise ValueError(
                f"emitted kernel needs startup features the minimal startup does not provide "
                f"({d.stdout.strip()}); compiler output should not require libc init or TLS")

    env2 = {**os.environ, "OBJCOPY": str(_llvm_tool("llvm-objcopy"))}
    r = subprocess.run(["python3", str(harness / "vxbin.py"), str(elf), str(vxbin)],
                       capture_output=True, text=True, timeout=timeout, env=env2)
    if r.returncode != 0 or not vxbin.is_file():
        raise ValueError(f"vxbin conversion failed: {r.stderr[:400]}")
    return vxbin


def run_image(vxbin: Path, plan: dict[str, Any], workdir: Path, *,
              driver: str = "simx", timeout: int = 1800) -> dict[str, Any]:
    """Run a `.vxbin` under `driver` and parse the console protocol."""
    home, harness = _vortex_home(), _harness()
    host = harness / "host" / "merlin_vx_host"
    if not host.is_file():
        raise VortexUnavailable(f"host driver not built at {host}; run build_harness.sh")
    plan_path = workdir / "plan.json"
    plan_path.write_text(json.dumps(plan))

    r = subprocess.run(
        [str(host), str(vxbin), str(plan_path)],
        capture_output=True, text=True, timeout=timeout,
        env={**os.environ,
             "LD_LIBRARY_PATH": f"{home / 'build' / 'sw' / 'runtime'}:{os.environ.get('LD_LIBRARY_PATH', '')}",
             "VORTEX_DRIVER": driver})
    console = r.stdout + r.stderr
    if r.returncode != 0 or not _console_done(console):
        raise ValueError(f"{driver} run did not complete: {console[-400:]}")

    geom = geometry_from_console(console)
    if geom is not None and geom != FROZEN_GEOMETRY:
        raise VortexUnavailable(
            f"{driver} is built for {geom}, not the frozen {FROZEN_GEOMETRY}. Results from this "
            f"tier are not comparable with the others; rebuild it with "
            f'CONFIGS="{FROZEN_BUILD_MACRO}"')

    instrs, cycles = _parse_perf(console)
    return {"outputs": decode_out(console),
            "geometry": geom,
            "cycles": cycles,
            "instrs": instrs,
            "oracle": {"kind": f"vortex_{driver}", "derived_from_rtl": driver == "rtlsim"},
            "console": console}


# --------------------------------------------------------------------------------------- adapters

def vortex_adapter(tier: str) -> Callable:
    """An oracle adapter for `tier` ("L2" -> simx, "L3" -> rtlsim).

    Matches the capsule_runner adapter signature ``(cb, llvm_text, workdir, timeout)``. For Vortex the
    first slot carries the **capsule dict** rather than a command buffer: a programmable core has no
    command stream, so what the host driver needs is the operand table, which the capsule already
    declares. The launch plan is derived here — operands from the capsule, grid from the *package's*
    module annotation, keeping the mapping decision with the compiler under test.

    A plan dict may also be passed directly (it is used verbatim if it already has ``grid``/``args``),
    which is what the corpus generator does when it computes goldens.
    """
    driver = DRIVERS[tier]

    def run(cb, llvm_text, workdir, timeout):
        if not available(tier):
            raise VortexUnavailable(f"vortex {driver} oracle unavailable")
        wd = Path(workdir)
        plan = cb if (isinstance(cb, dict) and "args" in cb and "grid" in cb) \
            else plan_from_capsule(cb, grid=grid_from_module(llvm_text))
        vxbin = build_image(llvm_text, wd, timeout=min(timeout, 600))
        # SIMT coverage gate — a capsule that ships an expected_simt_coverage.yaml must actually drive the
        # target ISA (GMEM_LD/GMEM_ST/CTA_CSR): a scalar-collapsed / host-computed kernel is rejected here
        # (a fail, not a pass) rather than having correct output read as a pass. Runs ONCE, on the
        # functional tier, over the kernel.o build_image just emitted (no recompile), and self-gates —
        # skipped for a raw launch plan (the generator path, which carries "args") or a capsule that
        # declares no coverage document. The RTL oracle is the spelling-independent backstop; this is the
        # cheap, explicit pre-check on the AGENT's own object (the harness startup is excluded by design).
        if tier == "L2" and isinstance(cb, dict) and "args" not in cb:
            from . import vortex_coverage as _VXC
            expected = _VXC.expected_for(cb)
            if expected is not None:
                cov = _VXC.check_object(expected, wd / "kernel.o")
                if cov["status"] != "pass":
                    raise ValueError(
                        f"SIMT coverage gate failed for {cb.get('name')}: required instruction classes "
                        f"not emitted ({cov['violations']}) — the kernel must drive the target ISA on the "
                        f"device, not compute on the host")
        return run_image(vxbin, plan, wd, driver=driver, timeout=timeout)

    return run


def adapters() -> dict[str, Callable]:
    """The Vortex tier ladder: simx everywhere, rtlsim for capsules that declare L3."""
    return {tier: vortex_adapter(tier) for tier in DRIVERS}
