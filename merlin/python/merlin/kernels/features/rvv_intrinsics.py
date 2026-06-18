"""RVV intrinsic *decisions* — the concrete codegen choices expert RVV kernels make, which the
compiler's RVV schedule must reproduce. This is the feature layer the kernel miner needs that
``extract_vector`` (scalable-vs-fixed only) does not provide.

Every RVV decision is recoverable from the intrinsic spelling:
  * LMUL grouping            -> the ``_e<sew>m<lmul>`` suffix (e32m4 = aggressive grouping)
  * scalar-broadcast vs vv   -> ``vfmacc_vf`` (the register-blocked GEMM idiom) vs ``vfmacc_vv``
  * int8 widening MAC        -> ``vwmacc`` (i8xi8->i32 datapath)
  * reduction tree           -> ``vfredusum`` / ``vredsum`` / ``vredmax``
  * VL-polymorphic tail      -> ``vsetvl`` re-queried in the loop vs a single ``vsetvlmax``
  * requant narrowing store  -> ``vfncvt`` / ``vnclip`` / ``vse8``

OpenBLAS hides LMUL + fma form behind ``#define VSETVL __riscv_vsetvl_e32m2`` / ``#define
VFMACCVF __riscv_vfmacc_vf_f32m2`` macros, so we resolve ``#define`` lines too. Values are
DECISIONS/enums + structural counts (LMUL bucket, vf/vv, mr/nr) — never free-floating constants;
the concrete knob value (e.g. LMUL=4 -> vector width) is derived from the enum by the
motif->knob mapping (S5), not stored as a tuned constant here.
"""
from __future__ import annotations

import re
from collections import Counter

from merlin.kernels.markers import target_family
from merlin.kernels.types import NormalizedKernel

# `_e<sew>m<lmul>` from a raw intrinsic OR a `#define` alias (OpenBLAS). lmul may be fractional (mf2).
_E_LMUL = re.compile(r"_e(\d+)m(f?\d+)\b")
_FMA = re.compile(r"__riscv_vf?macc_(vf|vv)")
_WIDENING = re.compile(r"__riscv_vw(?:macc|maccu|maccsu|maccus|add|sub|mul)\w*")
_REDUCE = re.compile(r"__riscv_v(?:f)?red(usum|osum|sum|max|min|maxu|minu)\w*")
_VSETVL_LOOP = re.compile(r"__riscv_vsetvl_e\d+mf?\d+")      # non-max: VL re-queried per iteration
_VSETVLMAX = re.compile(r"__riscv_vsetvlmax_e\d+mf?\d+")
_REQUANT = re.compile(r"__riscv_v(?:f)?ncvt\w*|__riscv_vnclip\w*|__riscv_vse8\b")
_VACC = re.compile(r"\bvacc(\d+)\w*\b")


def _lmul_class(text: str) -> str:
    """Dominant LMUL group across compute intrinsics + macro aliases, as an enum (m1/m2/m4/m8,
    or mf2/mf4/mf8 fractional). Returns 'na' when no RVV vector op is present."""
    toks = _E_LMUL.findall(text)  # [(sew, lmul), ...] from intrinsics AND #define lines
    if not toks:
        return "na"
    # Prefer the compute SEW (the widest non-mask element width drives the GEMM body).
    sews = [int(s) for s, _ in toks]
    main_sew = max(sews) if sews else 0
    lmuls = [m for s, m in toks if int(s) == main_sew] or [m for _, m in toks]
    dom = Counter(lmuls).most_common(1)[0][0]
    return f"m{dom}" if not dom.startswith("f") else f"m{dom}"  # 'f2' -> 'mf2'


def _fma_form(text: str) -> str | None:
    m = _FMA.findall(text)
    if not m:
        return None
    return Counter(m).most_common(1)[0][0]  # 'vf' (scalar broadcast) | 'vv'


def _reduction_form(text: str) -> str:
    m = _REDUCE.search(text)
    if not m:
        return "none"
    full = m.group(0)
    for tag in ("redusum", "redosum", "redsum", "redmaxu", "redmax", "redminu", "redmin"):
        if tag in full:
            return ("vfred" if "vfred" in full else "vred") + tag[3:]
    return "other"


def _vl_strategy(text: str) -> str:
    if _VSETVL_LOOP.search(text):
        return "vsetvl_loop"          # VL-polymorphic tail (the portable RVV idiom)
    if _VSETVLMAX.search(text):
        return "vsetvlmax_fixed"      # single max-VL, fixed body
    return "na"


def _accumulator_dtype(text: str, nk_dtype: str) -> str:
    if _WIDENING.search(text):
        return "i32"                  # i8xi8 -> i32 widening accumulate
    # Prefer the kernel's canonical dtype to disambiguate sources that compile to multiple
    # widths (OpenBLAS #ifdef DOUBLE emits both e32 and e64 macros from one file).
    if nk_dtype in ("f32", "bf16", "f16"):
        return "f32"                  # low-precision inputs accumulate in f32
    if nk_dtype == "f64":
        return "f64"
    if _FMA.search(text):
        return "f64" if re.search(r"vfmacc_v[fv]_f64", text) else "f32"
    return nk_dtype or "unknown"


def _register_block(text: str) -> dict:
    """Structural register blocking: mr = distinct accumulator registers fed by one loaded
    RHS vector (the reuse factor); nr_v = vector-register width of the N tile if discoverable."""
    mr = len(set(_VACC.findall(text)))
    nr_v = None
    m = re.search(r"\b(\d+)x(\d+)v\b", text)  # XNNPACK '1x4v' tiling token, if echoed in the body
    if m:
        nr_v = int(m.group(2))
    return {"mr": mr, "nr_v": nr_v}


def extract_rvv_intrinsics(nk: NormalizedKernel, fired: dict[str, list[str]]) -> dict:
    """RVV decision sub-dict, or {} for non-RVV kernels (no-op for gemmini/avx/neon)."""
    if target_family(nk.target) != "rvv":
        return {}
    text = nk.raw_text or ""
    if "__riscv_v" not in text:
        return {}
    return {"rvv": {
        "lmul_class": _lmul_class(text),
        "fma_form": _fma_form(text),
        "int_widening": bool(_WIDENING.search(text)),
        "accumulator_dtype": _accumulator_dtype(text, nk.dtype),
        "reduction_form": _reduction_form(text),
        "vl_strategy": _vl_strategy(text),
        "requant_epilogue": bool(_REQUANT.search(text)),
        "register_block": _register_block(text),
    }}
