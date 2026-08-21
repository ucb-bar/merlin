"""Model-level accelerator coverage: of a REAL model's regions, how many can a target actually take?

Capsule pass-rate answers "does the corpus pass". It cannot answer "does a model compile": a corpus is a
closed set of hand-picked families, while a captured model is dominated by unnamed ``linalg.generic``
regions whose meaning lives in their bodies. A backend can pass every capsule and still lower almost
nothing of a real model -- and nothing in the capsule verdict would say so.

This module measures the second question, target-agnostically: walk a captured model's linalg, describe
each region structurally, and ask the target's OWN capability contract
(:mod:`merlin.targetgen.eligibility`) whether that region is acceleratable. Nothing here is target-specific
-- the target is a parameter and every capability fact comes from its contract.

Four counts per (model, target), and the last is the one that keeps the rest honest:

``routed``
    regions the contract says this target can accelerate.
``fallback``
    regions it cannot -- which a general compiler must hand to the scalar/vector path. Not a failure: a
    real compiler's job is to cover these, and a coverage number that hides them describes a kernel
    library rather than a compiler.
``unclassified``
    regions whose family could NOT be determined at all.

``unclassified`` is reported separately and never folded into either bucket. A region we cannot name is
evidence neither of coverage nor of a gap, and folding it into one is exactly how a coverage number
becomes a lie. Provenance tags are used when present but never trusted to be present -- real captures
leave a large fraction of regions untagged, and tags disagree with the IR often enough that they are a
hint, not an authority (structural evidence wins where both exist).
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from merlin.targetgen import semantic_families as sf
from merlin.targetgen.eligibility import RegionDescriptor, capability_map_for_target, is_eligible

#: MLIR element-type spelling -> quant-format registry name. Types with no registry entry map to None so
#: the descriptor carries an honest "unknown dtype" rather than a guessed width (a wrong dtype silently
#: changes an eligibility verdict, so guessing here would fabricate coverage).
_ELEM_DTYPE: dict[str, str] = {
    "f32": "fp32", "f16": "fp16", "bf16": "bf16", "i8": "int8", "i4": "int4",
}


@dataclass
class CoverageReport:
    """Per (model, target) region accounting. ``routed + fallback + unclassified == n_regions``."""

    model: str
    target: str
    n_regions: int = 0
    routed: int = 0
    fallback: int = 0
    unclassified: int = 0
    by_family: Counter = field(default_factory=Counter)
    by_op: Counter = field(default_factory=Counter)
    unclassified_ops: Counter = field(default_factory=Counter)
    fallback_families: Counter = field(default_factory=Counter)

    @property
    def routed_fraction(self) -> float:
        """Routed as a share of CLASSIFIED regions — the honest denominator. Reporting over all regions
        would let a model full of unnameable regions look well covered."""
        classified = self.routed + self.fallback
        return (self.routed / classified) if classified else 0.0

    @property
    def classified_fraction(self) -> float:
        return ((self.n_regions - self.unclassified) / self.n_regions) if self.n_regions else 0.0

    def to_dict(self) -> dict:
        return {
            "model": self.model, "target": self.target, "n_regions": self.n_regions,
            "routed": self.routed, "fallback": self.fallback, "unclassified": self.unclassified,
            "routed_fraction_of_classified": round(self.routed_fraction, 4),
            "classified_fraction": round(self.classified_fraction, 4),
            "by_family": dict(self.by_family.most_common()),
            "by_op": dict(self.by_op.most_common()),
            "unclassified_ops": dict(self.unclassified_ops.most_common()),
            "fallback_families": dict(self.fallback_families.most_common()),
        }


def _short_op(op_name: str) -> str:
    """``linalg.matmul`` -> ``matmul``. Structural split on the dialect separator, no pattern matching."""
    return op_name.rpartition(".")[2] or op_name


def _attr_str(op, key: str) -> str | None:
    """A string attribute's value, or None. Tolerates attributes that are not string-typed."""
    attr = op.attributes.get(key)
    data = getattr(attr, "data", None)
    return data if isinstance(data, str) else None


def _elem_dtype(op) -> str | None:
    """Registry dtype name for the region's first tensor operand element type, or None when the type is
    absent or has no registry entry. Never guesses."""
    for operand in getattr(op, "operands", ()):  # first ranked operand wins (the activation/lhs)
        elem = getattr(getattr(operand, "type", None), "element_type", None)
        if elem is None:
            continue
        return _ELEM_DTYPE.get(str(elem))
    return None


def _is_region_op(op) -> bool:
    """Structure-carrying ops only. A terminator or a pure-init op is not a unit of computation to route,
    and counting them inflates every denominator."""
    name = op.name
    if not name.startswith("linalg."):
        return False
    return _short_op(name) not in ("yield", "index", "init_tensor")


def regions_from_module(module) -> tuple[RegionDescriptor, ...]:
    """Describe every computation-carrying linalg region in a parsed model module.

    Family resolution order: the op's own NAME first (structural — ``linalg.matmul`` is a contraction
    whatever a tag claims), then provenance tags as a fallback for the unnamed ``linalg.generic`` case.
    Unresolved stays None, so the caller counts it as unclassified instead of assuming a family.
    """
    out: list[RegionDescriptor] = []
    for op in module.walk():
        if not _is_region_op(op):
            continue
        short = _short_op(op.name)
        family = sf.from_op(short)
        if family is None:  # unnamed region: fall back to whatever provenance the capture stamped
            family = sf.from_prov(_attr_str(op, "prov.family"), _attr_str(op, "prov.op"))
        out.append(RegionDescriptor(source=short, op=short, family=family,
                                    in_dtype=_elem_dtype(op)))
    return tuple(out)


def coverage_for(regions: tuple[RegionDescriptor, ...], target: str, *,
                 model: str = "") -> CoverageReport:
    """Ask ``target``'s capability contract about each region. Pure accounting — no lowering is attempted,
    so this is the CEILING a submission for this target could reach, not what any submission does reach."""
    cap_map = capability_map_for_target(target)
    rep = CoverageReport(model=model, target=target, n_regions=len(regions))
    for region in regions:
        rep.by_op[region.op or "?"] += 1
        family = region.resolved_family()
        if family is None:
            rep.unclassified += 1
            rep.unclassified_ops[region.op or "?"] += 1
            continue
        rep.by_family[family] += 1
        if is_eligible(region, cap_map).eligible:
            rep.routed += 1
        else:
            rep.fallback += 1
            rep.fallback_families[family] += 1
    return rep


def load_module(path: str | Path):
    """Parse a captured model's MLIR. Imported lazily so this module stays importable without xDSL."""
    from xdsl.context import Context
    from xdsl.parser import Parser
    from xdsl.universe import Universe

    # Register the FULL upstream dialect set. ``allow_unregistered`` alone is not enough: an unregistered
    # op can only be parsed in generic form, and a capture is written in custom assembly
    # (``tensor.empty() : tensor<...>``), so every real model fails to parse without this.
    ctx = Context(allow_unregistered=True)
    for name, factory in Universe.get_multiverse().all_dialects.items():
        ctx.register_dialect(name, factory)
    return Parser(ctx, Path(path).read_text(encoding="utf-8"), str(path)).parse_module()
