"""Registry over the integer (W8A8) quantization passes — the ``quantization`` region's registrable
edit-point.

Mirrors ``impr_features``: the six ``lower_*_int`` passes in ``passes_quant_int`` become NAMED, ordered,
toggleable entries instead of a hardcoded call sequence, so "change how we apply quantization / real-quant
vs QDQ-fake" is an obvious edit-point (register a pass, toggle/reorder the set) rather than editing a
runtime call site. ``apply_quant()`` with the default set runs EXACTLY the historical sequence in the same
order, so the int8 datapath stays byte-identical — the registry is a seam, not a behavior change.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

# Canonical order (the historical sequence in dispatch_runtime / zephyr_model). Static — no import.
_ORDER = ("contraction_int8", "conv_int8", "softmax_int", "gelu_int", "silu_int", "rsqrt_int")


@dataclass(frozen=True)
class QuantPass:
    name: str            # short registry name
    op_class: str        # contraction | conv | softmax | gelu | silu | rsqrt
    fn: Callable         # (module) -> int  (count of ops lowered; mutates module in place)
    description: str


def registry() -> dict[str, QuantPass]:
    """The registered int8 quant passes wired to the real ``passes_quant_int`` functions (lazy import —
    that module needs xDSL). Rebuilt per call so a test can monkeypatch the underlying functions."""
    from .passes_quant_int import (lower_contraction_int8, lower_conv_int8, lower_gelu_int,
                                   lower_rsqrt_int, lower_silu_int, lower_softmax_int)
    passes = [
        QuantPass("contraction_int8", "contraction", lower_contraction_int8,
                  "matmul/attention -> i8xi8->i32 + requant"),
        QuantPass("conv_int8", "conv", lower_conv_int8,
                  "conv2d -> i8xi8->i32 (per-tensor act-quant) + requant"),
        QuantPass("softmax_int", "softmax", lower_softmax_int,
                  "softmax exp -> integer I-BERT i-exp (no math.exp)"),
        QuantPass("gelu_int", "gelu", lower_gelu_int, "GELU erf -> integer I-BERT i-GELU (no math.erf)"),
        QuantPass("silu_int", "silu", lower_silu_int, "SiLU sigmoid -> integer i-sigmoid"),
        QuantPass("rsqrt_int", "rsqrt", lower_rsqrt_int, "rsqrt -> integer fast-rsqrt"),
    ]
    return {p.name: p for p in passes}


def known() -> tuple[str, ...]:
    """The registered pass names in canonical order (import-free)."""
    return _ORDER


def apply_quant(module: Any, passes: "list[str] | None" = None, *,
                named_contraction: bool = False,
                prequant_gather: bool = False,
                report_out: "dict[str, dict] | None" = None,
                select: "Callable[[Any], bool] | None" = None) -> dict[str, int]:
    """Run the selected int8 quant passes IN CANONICAL ORDER (mutating ``module``). ``passes=None`` runs
    all six = the historical sequence (byte-identical datapath). Returns per-pass lowered-op counts.

    ``select`` is an ``(op) -> bool`` predicate restricting WHICH ops the selected passes may
    rewrite (default None = every op the pass recognizes, i.e. the shipped datapath). It makes the
    pass set's REACH measurable independently of its arithmetic: running only the contractions that
    descend from one framework construct answers "how much of the gap against a reference that
    quantizes fewer operations is policy rather than error". The predicate is the caller's policy —
    this module never assumes one.

    ``named_contraction`` asks the contraction pass to emit a mixed-type ``linalg.matmul`` for the
    canonical 2-D case instead of a ``linalg.generic``. Default False keeps the datapath
    byte-identical; True is what makes the 39 named-op transform-schedule levers reachable on int8
    at all (they match on the op NAME, and this pass otherwise leaves zero of them in the module).

    ``prequant_gather`` asks the contraction pass to quantize an activation BEFORE the pure gather
    that expands it (the ``quantize_before_gather`` feature). Default False keeps the datapath
    byte-identical; True is a genuine NUMERIC change (per-tensor instead of per-parallel-row
    activation scale) and is gated on the accuracy check like any other, never assumed bit-exact.

    ``report_out``, when given, collects each pass's own counters under its registry name — how many
    ops it rewrote, and for the two passes that can find nothing to do, WHY. A pass reporting only
    "0" cannot be told apart from a pass that cannot fire at all.
    """
    reg = registry()
    want = set(_ORDER) if passes is None else set(passes)
    out: dict[str, int] = {}
    for n in _ORDER:
        if n not in want:
            continue
        fn = reg[n].fn
        kw: dict[str, Any] = {}
        if select is not None:
            kw["select"] = select          # every pass honors the reach restriction
        # Only the contraction pass takes the flag; passing it to the others would couple every
        # quant pass to a decision that is not theirs to make.
        if named_contraction and n == "contraction_int8":
            kw["named_contraction"] = True
        if prequant_gather and n == "contraction_int8":
            kw["prequant_gather"] = True
        if report_out is not None and n in ("contraction_int8", "conv_int8"):
            kw["report_out"] = report_out.setdefault(n, {})
        out[n] = fn(module, **kw)
    return out
