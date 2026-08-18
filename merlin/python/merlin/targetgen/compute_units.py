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
from merlin.targetgen import semantic_families as _sf

#: Coarse compute-unit kinds (aligned with runtime.backends.base.TargetClass at the silicon level).
#: ``spatial`` is the non-systolic spatial tensor tile (an OuterProductUnit-style grid of accumulator
#: cells that reduces via rank-1 outer-product accumulate, not a stationary-weight systolic wavefront) —
#: NPU-class silicon like ``systolic`` but a distinct datapath, so it carries its own fact family.
KINDS: frozenset[str] = frozenset({"systolic", "simt", "vector", "scalar", "spatial"})


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
class SemanticCapability:
    """What a unit can compute at the **semantic-family** level — the HARDWARE-truth declaration the
    eligibility oracle (:mod:`merlin.targetgen.eligibility`) reads as the ARR *denominator*.

    Deliberately independent of ``ComputeUnit.ops`` (which is what the generated compiler's *routing*
    binds against, the ARR *numerator*): this says "the silicon can run this family over these formats
    and shapes", not "the compiler currently has a lowering for it". Authored per target in the
    residual, from hardware facts — the gap between this and what routing achieves IS the compiler
    deficiency ARR is meant to surface. ``family`` is a :mod:`merlin.targetgen.semantic_families` name;
    ``dtypes`` reference the quant-format registry.
    """

    family: str
    dtypes: tuple[str, ...] = ()
    ranks: tuple[int, ...] = ()           # legal tensor ranks (e.g. (2, 3) for 2D + batched); () = any
    transpose: bool = True                # transposed-operand variants legal where applicable
    arbitrary_mnk: bool = True            # M/N/K need not be tile multiples (tails handled)
    batch: bool = True                    # batch dimensions supported
    layouts: tuple[str, ...] = ()         # legal layout tags (coarse); () = unconstrained
    notes: str = ""


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
    semantic_capabilities: tuple[SemanticCapability, ...] = ()
    #: How SOFTWARE drives this unit (a ``families.ENDPOINT_KINDS`` token), independent of ``kind``.
    #: None means "not declared" and defers to the target, then to the family default — see
    #: :func:`resolve_exposure`. This is a PER-UNIT axis on purpose: a hybrid target has two units with
    #: two different exposures at once, and a single target-wide endpoint cannot express that.
    exposure: str | None = None

    def supports_dtype(self, fmt_name: str) -> bool:
        return fmt_name in self.dtypes

    def supports_op(self, op: str) -> bool:
        return not self.ops or op in self.ops


def _accum(raw: Any) -> AccumRule:
    return AccumRule(inp=raw.get("in", ""), weight=raw.get("weight", raw.get("in", "")), acc=raw.get("acc", ""))


def _sem_cap(raw: dict[str, Any], unit_name: str) -> SemanticCapability:
    family = raw.get("family")
    if not _sf.is_family(family):
        raise ValueError(f"compute unit {unit_name!r}: semantic_capabilities family {family!r} not in "
                         f"{sorted(_sf.FAMILIES)}")
    dtypes = tuple(raw.get("dtypes", ()) or ())
    unknown = [d for d in dtypes if not qf.has(d)]
    if unknown:
        raise ValueError(f"compute unit {unit_name!r}: semantic_capability {family!r} unknown quant "
                         f"formats {unknown} (known: {qf.names()})")
    return SemanticCapability(
        family=family,
        dtypes=dtypes,
        ranks=tuple(raw.get("ranks", ()) or ()),
        transpose=bool(raw.get("transpose", True)),
        arbitrary_mnk=bool(raw.get("arbitrary_mnk", True)),
        batch=bool(raw.get("batch", True)),
        layouts=tuple(raw.get("layouts", ()) or ()),
        notes=raw.get("notes", "") or "",
    )


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
    name = raw["name"]
    exposure = raw.get("exposure")
    if exposure is not None:
        from . import families                  # lazy: families is a sibling registry, not a dependency
        if exposure not in families.ENDPOINT_KINDS:
            raise ValueError(f"compute unit {raw.get('name')!r}: exposure {exposure!r} not in "
                             f"{list(families.ENDPOINT_KINDS)}")
    return ComputeUnit(
        name=name,
        kind=kind,
        dtypes=dtypes,
        ops=tuple(raw.get("ops", ()) or ()),
        accumulate=tuple(_accum(a) for a in raw.get("accumulate", ()) or ()),
        scaling=scaling,
        requant=raw.get("requant"),
        contains=tuple(raw.get("contains", ()) or ()),
        semantic_capabilities=tuple(_sem_cap(s, name) for s in raw.get("semantic_capabilities", ()) or ()),
        exposure=exposure,
    )


def resolve_exposure(unit: ComputeUnit, *, target_endpoint_kind: str | None = None) -> str:
    """How software drives ``unit``: the unit's own declaration, else the target's, else its family's.

    The precedence is the same FACTS > residual > family-default shape the target-wide ``endpoint_kind``
    already uses, which is what makes this change inert for existing targets: a single-unit target that
    declares no per-unit exposure resolves to exactly what it resolved to before.

    The reason this is per-unit at all is that a datapath class does not imply a software exposure. A
    spatial tensor tile is driven by one-hot command ports *inside* a vector unit, but software issues
    vector instructions to reach it — so its family default (``command_buffer``) describes the datapath
    correctly and the exposure wrongly. Letting the class imply the exposure makes a hybrid target
    inexpressible, which is a taxonomy failure rather than a missing special case.
    """
    if unit.exposure is not None:
        return unit.exposure
    if target_endpoint_kind is not None:
        return target_endpoint_kind
    from . import families
    return families.family_profile(unit.kind).endpoint_kind_default


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
    sem = list(unit.semantic_capabilities)
    for child_name in unit.contains:
        child = by_name.get(child_name)
        if child is None:
            raise ValueError(f"compute unit {unit.name!r} contains unknown unit {child_name!r}")
        child = effective(child, all_units)
        dtypes += [d for d in child.dtypes if d not in dtypes]
        ops += [o for o in child.ops if o not in ops]
        accum += [a for a in child.accumulate if a not in accum]
        sem += [s for s in child.semantic_capabilities if s not in sem]
    return ComputeUnit(
        name=unit.name, kind=unit.kind, dtypes=tuple(dtypes), ops=tuple(ops),
        accumulate=tuple(accum), scaling=unit.scaling, requant=unit.requant, contains=unit.contains,
        semantic_capabilities=tuple(sem),
        # Composition unions CAPABILITY (what can be computed), not exposure: how software drives this
        # unit is a property of this unit, and inheriting a child's would silently retarget the parent.
        exposure=unit.exposure,
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


def _merge_caps(a: SemanticCapability, b: SemanticCapability) -> SemanticCapability:
    """Union two same-family capabilities into the most-permissive combined capability (a target is
    capable of a family if ANY of its units is): union dtypes/layouts/ranks, OR the boolean flags."""
    def _u(x: tuple, y: tuple) -> tuple:
        out = list(x)
        out += [v for v in y if v not in out]
        return tuple(out)

    notes = "; ".join(n for n in (a.notes, b.notes) if n)
    return SemanticCapability(
        family=a.family,
        dtypes=_u(a.dtypes, b.dtypes),
        ranks=_u(a.ranks, b.ranks),
        transpose=a.transpose or b.transpose,
        arbitrary_mnk=a.arbitrary_mnk or b.arbitrary_mnk,
        batch=a.batch or b.batch,
        layouts=_u(a.layouts, b.layouts),
        notes=notes,
    )


def semantic_capability_map(units: list[ComputeUnit]) -> dict[str, SemanticCapability]:
    """The target's HARDWARE semantic capability, folded across all units (composition resolved): a
    ``family -> merged SemanticCapability`` map. This is the independent ARR denominator source the
    eligibility oracle reads — derived only from the declared contract, never from routing/lowering.
    """
    merged: dict[str, SemanticCapability] = {}
    for u in units:
        for cap in effective(u, units).semantic_capabilities:
            merged[cap.family] = _merge_caps(merged[cap.family], cap) if cap.family in merged else cap
    return merged
