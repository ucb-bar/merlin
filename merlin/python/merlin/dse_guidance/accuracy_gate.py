"""Quantization accuracy gate — the measurable-now leg of the numerical contract.

Accuracy depends only on the numerics, not on the future accelerator, so it is the right real
measurement to decide whether a low-bit candidate is even *legal*. This module records the MEASURED
W8A8-vs-fp32 accuracy (from ``measured_accuracy.yaml`` / ``accuracy_gate.yaml``, sourced from
``docs/results.md``) and exposes a per-(model, dtype) ``accuracy_status`` of ``pass`` / ``fail`` /
``unavailable``. Only int8/W8A8 is measured; fp8/int4/fp4/fp6 are ``unavailable`` (not assumed).

It claims no speedup and no performance number — only whether precision preserves output quality.
"""
from __future__ import annotations

from dataclasses import dataclass

from merlin.common import paths
from merlin.common.yaml import load_yaml


@dataclass
class AccuracyPoint:
    model: str
    dtype: str
    cos_vs_fp32: float
    rel: float
    status: str            # pass | fail | unavailable
    source: str


def load(path=None) -> list[AccuracyPoint]:
    p = path or (paths.bench_dir() / "dse_guidance" / "accuracy_gate.yaml")
    doc = load_yaml(p)
    src = doc.get("source", "?")
    return [AccuracyPoint(model=r["model"], dtype=r["dtype"],
                          cos_vs_fp32=float(r["cos_vs_fp32"]), rel=float(r.get("rel", 0.0)),
                          status=str(r.get("status", "unavailable")), source=src)
            for r in doc.get("points", [])]


def status_for(model: str, dtype: str, points: list[AccuracyPoint] | None = None) -> str:
    """accuracy_status for a (model, dtype): pass/fail from measured data, else unavailable.

    ``dtype`` may be a candidate format label (e.g. ``int8_w8a8``, ``int4_weight_only``); we match
    on the bit-family (int8/fp8/int4/...). Only what is measured returns pass/fail.
    """
    fam = _family(dtype)
    for p in (points or load()):
        if p.model == model and _family(p.dtype) == fam:
            return p.status
    return "unavailable"


def _family(dtype: str) -> str:
    d = dtype.lower()
    # fp8 must be checked before the generic "w8a8" (fp8_w8a8 is fp8, not int8) so a measured int8
    # result is never falsely inherited by an unmeasured fp8 format.
    if "fp8" in d or "float8" in d:
        return "fp8"
    if "int8" in d or d == "i8" or "w8a8" in d:
        return "int8"
    if "int4" in d or "i4" in d:
        return "int4"
    if "fp4" in d:
        return "fp4"
    if "fp6" in d:
        return "fp6"
    return d


def report_md(points: list[AccuracyPoint] | None = None) -> str:
    ms = points or load()
    L = ["# Quantization accuracy gate (measurable-now)\n"]
    L.append("> Accuracy depends on the numerics, not the future hardware, so it is measured now to "
             "decide whether a low-bit candidate is legal. W8A8 (int8) vs fp32 golden, host "
             "interpreter (`docs/results.md`). Multi-tier gate: T1 cos>0.999 vs W8A8 ref, T2 "
             "cos>0.99 vs fp32 + top-1 argmax. No speedup is claimed.\n")
    L.append("| model | dtype | cos vs fp32 | rel | status |")
    L.append("|-------|-------|-------------|-----|--------|")
    for m in ms:
        L.append(f"| {m.model} | {m.dtype} | {m.cos_vs_fp32:.5f} | {m.rel:.3f} | {m.status} |")
    L.append("")
    passed = [m for m in ms if m.status == "pass"]
    L.append(f"**Finding:** {len(passed)}/{len(ms)} measured int8 variants pass the W8A8 accuracy "
             "band — so the int8 low-bit residency/compute candidates are accuracy-legal. "
             "fp8/int4/fp4/fp6 are **unavailable** (not yet measured) and stay gated, not assumed.\n")
    return "\n".join(L)


def to_csv(points: list[AccuracyPoint] | None = None) -> str:
    import csv
    import io
    cols = ["model", "dtype", "cos_vs_fp32", "rel", "status", "source"]
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols)
    w.writeheader()
    for m in (points or load()):
        w.writerow({"model": m.model, "dtype": m.dtype, "cos_vs_fp32": m.cos_vs_fp32,
                    "rel": m.rel, "status": m.status, "source": m.source})
    return buf.getvalue()
