"""Compute-unit capability model — the datatype -> unit -> op mapping a target declares.

A hardware target is a set of **compute units** (a systolic mesh, SIMT lanes, a vector unit, a scalar
core, ...), each of which supports a set of numeric **formats** (named against the target-agnostic
:mod:`merlin.common.quant_formats` registry) on a set of **ops**, with an allowed
``(input, weight) -> accumulator`` matrix and an optional **scaling** format. This is what a target
contract's ``compute_units`` field expresses; :mod:`merlin.targetgen.routing` reads it to bind each op
in a model to a legal ``(unit, dtype)`` — or report an honest gap.

Two deliberate boundaries keep this generic and un-overfit:
- **Formats are referenced by name, never redefined here.** "This unit runs mxfp6" is a reference to
  the registry entry, not a copy of its encoding.
- **Requant/quantize semantics are NOT modelled here.** A unit carries only an opaque ``requant``
  reference (``{ref: <lowering-id>}``) pointing at the target's own out-of-tree lowering. The exact
  arithmetic (Gemmini's rounding-shift, gemmini-mx's E8M0 block-exponent add + LUT, radiance's
  none) is mutually incompatible across targets and lives with the target, not in shared code.

Compute units may **compose** (`contains`): a unit can embed others (e.g. a gemmini-mx systolic unit
inside a radiance cluster), and its *effective* capability is the union of itself and what it
contains — so gemmini-mx works standalone or as a sub-unit.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from merlin.common import quant_formats as qf

#: Coarse compute-unit kinds (aligned with runtime.backends.base.TargetClass at the silicon level).
KINDS: frozenset[str] = frozenset({"systolic", "simt", "vector", "scalar"})


@dataclass(frozen=True)
class AccumRule:
    """One allowed ``(input, weight) -> accumulator`` triple for a unit.

    ``inp``/``weight`` are quant-format names; ``acc`` is a raw MLIR accumulator type token
    (e.g. ``i32``/``f32``) — accumulators are not storage formats, so they are not registry entries.
    """

    inp: str
    weight: str
    acc: str


@dataclass(frozen=True)
class ComputeUnit:
    name: str
    kind: str
    dtypes: tuple[str, ...] = ()          # quant_format names this unit computes on
    ops: tuple[str, ...] = ()
    accumulate: tuple[AccumRule, ...] = ()
    scaling: str | None = None            # a quant_formats SCALE_KIND (per_channel/block_e8m0/none/...)
    requant: dict[str, Any] | None = None  # opaque {ref: <out-of-tree lowering id>}; not interpreted
    contains: tuple[str, ...] = ()

    def supports_dtype(self, fmt_name: str) -> bool:
        return fmt_name in self.dtypes

    def supports_op(self, op: str) -> bool:
        return not self.ops or op in self.ops


def _accum(raw: Any) -> AccumRule:
    return AccumRule(inp=raw.get("in", ""), weight=raw.get("weight", raw.get("in", "")), acc=raw.get("acc", ""))


def _unit(raw: dict[str, Any]) -> ComputeUnit:
    kind = raw.get("kind")
    if kind not in KINDS:
        raise ValueError(f"compute unit {raw.get('name')!r}: kind {kind!r} not in {sorted(KINDS)}")
    dtypes = tuple(raw.get("dtypes", ()) or ())
    unknown = [d for d in dtypes if not qf.has(d)]
    if unknown:
        raise ValueError(f"compute unit {raw.get('name')!r}: unknown quant formats {unknown} "
                         f"(known: {qf.names()})")
    scaling = raw.get("scaling")
    if scaling is not None and scaling not in qf.SCALE_KINDS:
        raise ValueError(f"compute unit {raw.get('name')!r}: scaling {scaling!r} not in "
                         f"{sorted(qf.SCALE_KINDS)}")
    return ComputeUnit(
        name=raw["name"],
        kind=kind,
        dtypes=dtypes,
        ops=tuple(raw.get("ops", ()) or ()),
        accumulate=tuple(_accum(a) for a in raw.get("accumulate", ()) or ()),
        scaling=scaling,
        requant=raw.get("requant"),
        contains=tuple(raw.get("contains", ()) or ()),
    )


def compute_units(contract: dict[str, Any]) -> list[ComputeUnit]:
    """Parse + validate a contract's ``compute_units`` (empty list if the field is absent)."""
    raw = contract.get("compute_units")
    if not raw:
        return []
    if not isinstance(raw, list):
        raise ValueError("compute_units must be a list")
    return [_unit(u) for u in raw]


def effective(unit: ComputeUnit, all_units: list[ComputeUnit]) -> ComputeUnit:
    """Return ``unit`` with contained units' dtypes/ops/accumulate folded in (composition union).

    A composed unit (``contains: [...]``) has the combined capability of itself plus everything it
    embeds — so a gemmini-mx unit contributes its fp4/fp6/fp8 support when embedded in a larger target.
    """
    if not unit.contains:
        return unit
    by_name = {u.name: u for u in all_units}
    dtypes = list(unit.dtypes)
    ops = list(unit.ops)
    accum = list(unit.accumulate)
    for child_name in unit.contains:
        child = by_name.get(child_name)
        if child is None:
            raise ValueError(f"compute unit {unit.name!r} contains unknown unit {child_name!r}")
        child = effective(child, all_units)
        dtypes += [d for d in child.dtypes if d not in dtypes]
        ops += [o for o in child.ops if o not in ops]
        accum += [a for a in child.accumulate if a not in accum]
    return ComputeUnit(
        name=unit.name, kind=unit.kind, dtypes=tuple(dtypes), ops=tuple(ops),
        accumulate=tuple(accum), scaling=unit.scaling, requant=unit.requant, contains=unit.contains,
    )


def datatype_tokens(unit: ComputeUnit) -> set[str]:
    """MLIR-ish element tokens a unit's storage dtypes present as (for RTL-fact cross-checks).

    Resolves each format name to its canonical name + aliases, so a datapath token like ``i8``
    matches the ``int8`` format (whose aliases include ``i8``).
    """
    tokens: set[str] = set()
    for name in unit.dtypes:
        fmt = qf.get(name)
        tokens.add(fmt.name)
        tokens.update(fmt.aliases)
    return tokens
