"""Generated-vs-curated structural comparison — quantifies the gap between what OUR compiler
emits for an op-shape and what the expert kernel emits, so the gap-router knows which knob/lever
is wrong (and the beam-search has a structural ranking signal alongside cycles).

Both sides are reduced to the SAME `RvvFingerprint`:
  * `decisions` — the RVV decision vector (lmul_class, fma_form, int_widening, reduction_form,
    vl_strategy, requant_epilogue), derived from C intrinsics (curated) or objdump asm (generated)
    via one shared canonicalization, so they are directly comparable.
  * `histogram` — canonical-op counts (vfmacc/vfmul/vle32/vsetvl/...) for a cosine similarity.

`compare_fingerprints` returns per-decision match flags + a `divergences` list of human strings
(e.g. "fma_form: expert vf, we emit none (vfmul+vfadd, no fusion)") + a scalar `structural_match`.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from .features.rvv_intrinsics import (_E_LMUL, _FMA, _REDUCE, _REQUANT, _VSETVL_LOOP,
                                      _VSETVLMAX, _WIDENING, _accumulator_dtype, _fma_form,
                                      _lmul_class, _reduction_form, _vl_strategy)

# Decision keys compared on BOTH sides (register_block omitted: not recoverable from asm).
DECISION_KEYS = ("lmul_class", "fma_form", "int_widening", "reduction_form",
                 "vl_strategy", "requant_epilogue")

# objdump asm mnemonic, e.g. "vfmacc.vv" / "vsetivli" / "vle32.v".
_ASM_MNEMONIC = re.compile(r"^v[a-z0-9]+(?:\.[a-z0-9]+)*$")
# asm vset has element/lmul inline: "vsetivli a0,8,e32,m1,ta,ma".
_ASM_VSET = re.compile(r"vset[i]?vli?\b[^\n]*?e(\d+),\s*m(f?\d+)")


def _canon_op(tok: str) -> str:
    """Canonical op token shared by C intrinsics and asm: '__riscv_vfmacc_vf_f32m4' -> 'vfmacc',
    'vfmacc.vv' -> 'vfmacc', 'vle32.v' -> 'vle32', 'vsetivli'/'vsetvli'/'vsetvlmax' -> 'vsetvl'."""
    t = tok.replace("__riscv_", "")
    t = t.split(".")[0].split("_")[0]            # drop .vv/.vf or _vf_f32m4 suffixes
    if t.startswith("vset"):
        return "vsetvl"                          # vsetvli/vsetivli/vsetvlmax all unify
    return t


def _asm_histogram(objdump: str) -> dict[str, int]:
    h: Counter[str] = Counter()
    for line in objdump.splitlines():
        f = line.split("\t")
        if len(f) < 3 or not f[2].strip():
            continue
        m = f[2].strip().split()[0]
        if _ASM_MNEMONIC.match(m):
            h[_canon_op(m)] += 1
    return dict(h)


def _c_histogram(text: str) -> dict[str, int]:
    h: Counter[str] = Counter()
    for m in re.findall(r"__riscv_v[a-z0-9_]+", text):
        h[_canon_op(m)] += 1
    return dict(h)


def _decisions_from_asm(objdump: str) -> dict[str, Any]:
    """Derive the RVV decision vector from objdump asm (the generated side)."""
    lmuls = _ASM_VSET.findall(objdump)           # [(sew, lmul), ...]
    if lmuls:
        sews = [int(s) for s, _ in lmuls]
        main = max(sews)
        dom = Counter(m for s, m in lmuls if int(s) == main).most_common(1)[0][0]
        lmul = f"m{dom}"
    else:
        lmul = "na"
    has_fma = bool(re.search(r"\bvf?macc\.(vf|vv)", objdump))
    fma = (re.search(r"\bvf?macc\.(vf|vv)", objdump).group(1) if has_fma else None)
    # vsetvli (register VL) => polymorphic loop; vsetivli (immediate) => fixed; vsetvlmax => fixed.
    if re.search(r"\bvsetvli\b", objdump):
        vl = "vsetvl_loop"
    elif re.search(r"\bvsetivli\b", objdump):
        vl = "vsetivli_fixed"
    elif re.search(r"\bvsetvlmax", objdump):
        vl = "vsetvlmax_fixed"
    else:
        vl = "na"
    red = re.search(r"\bv(?:f)?red(usum|osum|sum|max|min|maxu|minu)\w*", objdump)
    return {
        "lmul_class": lmul,
        "fma_form": fma,
        "int_widening": bool(re.search(r"\bvwmacc", objdump)),
        "reduction_form": (("vfred" if red.group(0).startswith("vfred") else "vred")
                           + red.group(1)) if red else "none",
        "vl_strategy": vl,
        "requant_epilogue": bool(re.search(r"\bv(?:f)?ncvt|\bvnclip|\bvse8\b", objdump)),
    }


def _decisions_from_c(text: str, dtype: str = "unknown") -> dict[str, Any]:
    return {
        "lmul_class": _lmul_class(text),
        "fma_form": _fma_form(text),
        "int_widening": bool(_WIDENING.search(text)),
        "reduction_form": _reduction_form(text),
        "vl_strategy": _vl_strategy(text),
        "requant_epilogue": bool(_REQUANT.search(text)),
    }


@dataclass
class RvvFingerprint:
    key: dict[str, str]                          # {op, dtype, shape_regime}
    decisions: dict[str, Any]
    histogram: dict[str, int]
    source: str                                  # "curated:<src>" | "generated:<run_id>"

    @classmethod
    def from_curated(cls, raw_text: str, key: dict, src: str) -> "RvvFingerprint":
        return cls(key, _decisions_from_c(raw_text, key.get("dtype", "unknown")),
                   _c_histogram(raw_text), f"curated:{src}")

    @classmethod
    def from_objdump(cls, objdump: str, key: dict, run_id: str) -> "RvvFingerprint":
        return cls(key, _decisions_from_asm(objdump), _asm_histogram(objdump),
                   f"generated:{run_id}")


def _cosine(a: dict[str, int], b: dict[str, int]) -> float:
    keys = set(a) | set(b)
    if not keys:
        return 1.0
    va = [a.get(k, 0) for k in keys]
    vb = [b.get(k, 0) for k in keys]
    na = math.sqrt(sum(x * x for x in va))
    nb = math.sqrt(sum(x * x for x in vb))
    if na == 0 or nb == 0:
        return 0.0
    return sum(x * y for x, y in zip(va, vb)) / (na * nb)


def compare_fingerprints(curated: RvvFingerprint, generated: RvvFingerprint) -> dict[str, Any]:
    """Structural diff curated (expert) vs generated (our compiler). `structural_match in [0,1]`
    = mean(decision-match flags) weighted toward histogram cosine."""
    flags: dict[str, bool] = {}
    divergences: list[str] = []
    for k in DECISION_KEYS:
        cv, gv = curated.decisions.get(k), generated.decisions.get(k)
        match = (cv == gv)
        flags[k] = match
        if not match:
            divergences.append(f"{k}: expert={cv!r} vs ours={gv!r}")
    cos = _cosine(curated.histogram, generated.histogram)
    decision_score = sum(flags.values()) / len(DECISION_KEYS)
    structural_match = round(0.6 * decision_score + 0.4 * cos, 4)
    return {
        "key": curated.key,
        "decision_match": flags,
        "divergences": divergences,
        "histogram_cosine": round(cos, 4),
        "structural_match": structural_match,
        "curated_decisions": curated.decisions,
        "generated_decisions": generated.decisions,
        "curated_source": curated.source,
        "generated_source": generated.source,
    }
