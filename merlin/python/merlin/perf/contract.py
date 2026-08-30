"""The performance contract: what each of a target's resources costs, and what is not known.

A capability manifest says what a target is ALLOWED to do. This is the other half: for each resource
the target has, the terms that price it -- a peak rate, a pipeline fill, a capacity, a latency --
each one a :class:`~merlin.perf.term.PerformanceTerm` carrying its provenance and the regime it was
established over.

The contract is built from the :class:`~merlin.perf.profile.TargetProfile`, so it inherits the same
discipline: the resources come from what the target's own sources evidence, and a quantity nothing
grounds is recorded UNKNOWN with a reason rather than filled in.

WHY MOST OF IT IS UNKNOWN, AND WHY THAT IS THE PRODUCT
------------------------------------------------------
Per-resource peaks are **not generally structurally derivable**, and UNKNOWN is the common case, not
the edge. The RTL timing walk resolves feed-forward depth; the units that dominate a runtime are
sequenced (their latency is a function of state and operands), so they refuse -- correctly. On one
machine the mesh container itself refuses, because weight-stationary accumulation routes back
through the array and no finite wiring depth *is* its latency. On another the data-movement engine
refuses while data movement is most of every cycle.

So a contract that reported a number for every resource would be reporting fabrications, and it
would be wrong in the flattering direction -- an optimistic peak makes a target look further from
its ceiling than it is, which is exactly the error that gets cited. The UNKNOWN terms here are the
contract's most useful output: each one names what would settle it, which is the measurement backlog
for the target, computed rather than chosen.

THREE HAZARDS THIS MODULE IS BUILT AROUND
-----------------------------------------
* **A resolved depth of 0 is a real answer.** Combinational, no registers on any output path. It is
  preserved as ``0`` and never collapsed into UNKNOWN. ``if depth is None:`` is the check;
  ``if not depth:`` is the bug, and it is the UNKNOWN-reads-as-0.0 bug one level up.
* **Sources that disagree do not get a winner.** Where two sources state a capacity and the values
  differ, the term is UNKNOWN and names both. Silently preferring one is how a manifest ends up
  asserting a capacity nobody can reproduce.
* **A rate alone cannot price a tiled unit.** Every peak-rate term is emitted with a validity domain
  saying it holds only at full occupancy, and with its fixed term (the fill intercept) as a separate,
  first-class term -- because a rate-only model mispredicts every small workload, and it is the
  small workloads a corpus is mostly made of.

Two UNKNOWN singletons exist in this package (:mod:`merlin.perf.term` and
:mod:`merlin.perf.decompose` each define one, and they are not equal to each other). Everything that
becomes a term's VALUE uses :data:`merlin.perf.term.UNKNOWN`; :class:`~merlin.perf.decompose.
Unavailable` is used only for whole analyses that could not run.
"""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .decompose import ResourceKind, Unavailable
from .profile import Elaboration, TargetProfile, derive_profile
from .term import UNKNOWN, Bounds, PerformanceTerm, Provenance, Validity

__all__ = [
    "PerformanceContract",
    "ResourceTerms",
    "capacity_terms",
    "derive_contract",
]

#: Evidence kinds used here. A structural value read out of the RTL facts is a ``structural_bound``;
#: a value a human wrote into the residual is ``assumed``; a conclusion drawn from the facts without
#: a measurement (e.g. "this engine's cost is not derivable") is ``analytical``.
_KIND_STRUCTURAL = "structural_bound"
_KIND_DECLARED = "assumed"
_KIND_ANALYTICAL = "analytical"


def _validity(elaboration: Elaboration, regime: str, *, expected_error: str = "",
              weak_regime: str = "", escalate_when: str = "") -> Validity:
    """A validity domain that NAMES the elaboration the value belongs to.

    A structural number is a property of the design that was read. Two elaborations of the same
    target -- a config variant, a different commit, a tapeout that dropped a unit -- give different
    numbers, so a term whose domain does not name its elaboration is claimed everywhere and
    reproducible nowhere.
    """
    return Validity(validated_regime=f"{regime}; {elaboration.describe()}",
                    expected_error=expected_error, weak_regime=weak_regime,
                    escalate_when=escalate_when or ("a different RTL elaboration of this target "
                                                    "(re-derive; do not carry this value across)"))


@dataclass(frozen=True)
class ResourceTerms:
    """One resource and the terms that price it.

    ``kind`` is the shared :class:`~merlin.perf.decompose.ResourceKind`, so a contract's resources
    and an activity source's resources are the same vocabulary and a decomposition can be matched to
    a contract without a name table.
    """

    name: str
    kind: ResourceKind
    terms: dict[str, PerformanceTerm] = field(default_factory=dict)
    evidence: str = ""

    def term(self, quantity: str) -> PerformanceTerm:
        try:
            return self.terms[quantity]
        except KeyError:
            raise KeyError(f"resource {self.name!r} has no term {quantity!r}; it carries "
                           f"{sorted(self.terms)}") from None

    def unknown(self) -> tuple[str, ...]:
        return tuple(sorted(q for q, t in self.terms.items() if t.is_unknown))

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "kind": self.kind.value, "evidence": self.evidence,
                "terms": {q: t.to_dict() for q, t in sorted(self.terms.items())}}


@dataclass(frozen=True)
class PerformanceContract:
    """A target's resources, their terms, and the analyses its evidence cannot support."""

    target: str
    profile: TargetProfile
    resources: tuple[ResourceTerms, ...]
    gaps: tuple[Unavailable, ...] = ()

    def resource(self, name: str) -> ResourceTerms:
        for r in self.resources:
            if r.name == name:
                return r
        raise KeyError(f"{self.target}: no resource {name!r}; the contract carries "
                       f"{[r.name for r in self.resources]}")

    def resources_of(self, kind: ResourceKind) -> tuple[ResourceTerms, ...]:
        return tuple(r for r in self.resources if r.kind is kind)

    def terms(self) -> dict[str, PerformanceTerm]:
        """Every term, keyed ``<resource>.<quantity>`` so the set can be merged into a record."""
        return {f"{r.name}.{q}": t for r in self.resources for q, t in sorted(r.terms.items())}

    def unknown_terms(self) -> dict[str, str]:
        """``{qualified term name: why it is not known}`` -- the measurement backlog, computed."""
        return {k: t.unknown_reason for k, t in self.terms().items() if t.is_unknown}

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "profile": self.profile.to_dict(),
            "resources": [r.to_dict() for r in self.resources],
            "gaps": [{"what": g.what, "missing": list(g.missing), "detail": g.detail}
                     for g in self.gaps],
            "unknown_terms": self.unknown_terms(),
        }

    def write(self, path: "str | Path") -> Path:
        """Serialize to ``path`` (caller-chosen, so generated output lands under the ``out/`` root
        the caller already resolved with :mod:`merlin.common.paths`)."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=False) + "\n", encoding="utf-8")
        return p


# --------------------------------------------------------------------------------------------
# Compute resources
# --------------------------------------------------------------------------------------------


def _bind_arrays(profile: TargetProfile) -> list[tuple[dict[str, Any], dict[str, Any] | None]]:
    """Pair each declared compute unit with a discovered array, positionally.

    The same correspondence ``derive_manifest`` uses to ground per-unit datapath dtypes: the primary
    unit takes the primary array. It is structural, not a name table -- and a unit with no array
    simply gets ``None``, which becomes UNKNOWN geometry rather than a borrowed one.
    """
    units = profile.sources.units()
    arrays = profile.sources.arrays()
    if not units:
        # No declared unit but a discovered array is still a compute resource: the array is the
        # evidence, and refusing to price it because nobody wrote it down would discard the fact.
        return [({"name": a.get("name") or "array", "kind": None}, a) for a in arrays]
    return [(u, arrays[i] if i < len(arrays) else None) for i, u in enumerate(units)]


def _peak_term(unit: dict[str, Any], array: dict[str, Any] | None,
               elaboration: Elaboration) -> PerformanceTerm:
    """Peak MACs per cycle for one compute unit, from the discovered array's own geometry.

    This is a STRUCTURAL BOUND at full occupancy and nothing more. Its validity says so, and its
    lower bound is 0 rather than the value, because no workload is obliged to reach it. A single
    rate cannot price a tiled unit -- the fill intercept is a separate term and is where the small
    shapes actually live.
    """
    name = "peak_macs_per_cycle"
    if array is None or array.get("rows") is None or array.get("cols") is None:
        return PerformanceTerm.unknown(
            name, "mac/cycle",
            Provenance(_KIND_STRUCTURAL, (f"facts.arrays (no array bound to unit "
                                          f"{unit.get('name')!r})",)),
            _validity(elaboration, "no geometry was discovered for this unit"),
            "no discovered array grounds this unit's geometry, and a peak invented from a declared "
            "op list would be a fabricated hardware claim")
    rows, cols = int(array["rows"]), int(array["cols"])
    # An element is not obliged to hold exactly one multiplier. Where the extractor discovered the
    # element's MAC idiom, use its multiplier count; where it did not, this counts ELEMENTS and the
    # evidence says so rather than assuming one multiplier each.
    idiom = array.get("mac_idiom") or {}
    muls = idiom.get("muls") if isinstance(idiom, Mapping) else None
    per_element = int(muls) if isinstance(muls, int) and not isinstance(muls, bool) else 1
    idiom_note = (f"x{per_element} multiplier(s) per element (facts mac_idiom)" if muls is not None
                  else "counting ELEMENTS: no mac_idiom states the multipliers per element")
    peak = rows * cols * per_element
    return PerformanceTerm(
        name=name, value=peak, unit="mac/cycle",
        provenance=Provenance(_KIND_STRUCTURAL,
                              (f"facts.arrays[{array.get('name')}]: {rows}x{cols} "
                               f"{array.get('element')} elements in {array.get('container')}, "
                               f"{idiom_note}",)),
        validity=_validity(elaboration,
                           f"full occupancy: all {rows * cols} elements retiring {per_element} "
                           f"MAC(s) each, every cycle",
                           expected_error="an upper bound, not an expectation",
                           weak_regime="any tile smaller than the array, where the fill intercept "
                                       "dominates and this rate mispredicts"),
        bounds=Bounds(lower=0, upper=peak))


def _depth_term(name: str, module: str | None, profile: TargetProfile, elaboration: Elaboration,
                *, regime: str, unknown_hint: str) -> PerformanceTerm:
    """A pipeline-depth term for one RTL module. ``0`` is preserved as a real value.

    The walk's refusal is carried through verbatim as the unknown reason: it already says WHY (the
    outputs are reached through feedback, so no finite wiring depth is the unit's latency), which is
    what tells a reader this needs a measurement rather than a better walk.
    """
    depth, evidence = profile.timing.depth(module)
    prov = Provenance(_KIND_STRUCTURAL,
                      (f"facts.timing[{module}].pipeline_depth" if module
                       else "facts.timing (no module named)",))
    if depth is None:
        return PerformanceTerm.unknown(
            name, "cycles", prov,
            _validity(elaboration, f"the RTL timing walk over module {module!r}"),
            f"{evidence}. {unknown_hint}")
    return PerformanceTerm(
        name=name, value=depth, unit="cycles", provenance=prov,
        validity=_validity(elaboration, f"{regime}; {evidence}",
                           expected_error="exact for this elaboration: a register on the path IS a "
                                          "pipeline stage",
                           weak_regime="any cost that is a function of state or operands rather "
                                       "than of wiring depth"),
        bounds=Bounds(lower=0))


def _fill_term(container: str | None, profile: TargetProfile,
               elaboration: Elaboration) -> PerformanceTerm:
    """The FILL: cycles from the first operand entering the datapath to the first result leaving.

    This is deliberately NOT the container module's depth, even when that depth resolved. The array
    fact names the module that holds the elements; what an operand actually traverses is that module
    plus whatever wraps it -- input skew, output drain, a result bus. On one measured elaboration
    those differ by a factor of two (the element grid resolves ``rows-1`` while the enclosing
    datapath is ``2*DIM-2``), and reporting the inner number as the fill would understate every
    small tile by the difference -- in the flattering direction, which is the direction that gets
    cited.

    So the fill is UNKNOWN, and the container's resolved depth becomes its LOWER BOUND: a real,
    usable constraint that cannot be mistaken for the answer. An unbounded end and an unknown end
    are different claims, which is why the bound is set only when the container actually resolved.
    """
    depth, evidence = profile.timing.depth(container)
    lower = depth if depth is not None else UNKNOWN
    if depth is None:
        reason = (f"{evidence}. Nothing establishes the depth of the datapath the operands "
                  "traverse, so the fill is not derivable and must be measured (>=2 tile sizes "
                  "separate the fill intercept from the per-tile rate)")
    else:
        reason = (f"the array's container {container!r} resolves to {depth} cycles, but nothing "
                  "establishes that the container IS the whole datapath: a module that skews the "
                  "inputs and drains the outputs adds depth on top of the element grid. "
                  f"{depth} is therefore a LOWER BOUND on the fill (recorded as such), not the "
                  "fill. Measuring one small tile settles it")
    return PerformanceTerm.unknown(
        "pipeline_fill_cycles", "cycles",
        Provenance(_KIND_STRUCTURAL, (f"facts.timing[{container}].pipeline_depth" if container
                                      else "facts.timing (no container named)",)),
        _validity(elaboration,
                  "the cycles between the first operand entering the datapath and the first "
                  "result leaving it"),
        reason, bounds=Bounds(lower=lower))


def _compute_resources(profile: TargetProfile) -> list[ResourceTerms]:
    elaboration = profile.elaboration
    out: list[ResourceTerms] = []
    for unit, array in _bind_arrays(profile):
        uname = str(unit.get("name") or "compute")
        container = (array or {}).get("container")
        element = (array or {}).get("element")
        terms = {
            "peak_macs_per_cycle": _peak_term(unit, array, elaboration),
            "container_depth_cycles": _depth_term(
                "container_depth_cycles", container, profile, elaboration,
                regime=f"the feed-forward depth of {container!r}, the module the array fact names "
                       "as its container",
                unknown_hint="This module's depth is NOT structurally derivable and must come from "
                             "the sequencer's own limits or from a measurement; a depth of 0 "
                             "assumed here would understate every small tile."),
            "pipeline_fill_cycles": _fill_term(container, profile, elaboration),
            "element_latency_cycles": _depth_term(
                "element_latency_cycles", element, profile, elaboration,
                regime="one array element's feed-forward depth (0 means combinational: a real "
                       "answer, not a missing one)",
                unknown_hint="The element's latency is sequenced, so it is not a wiring depth."),
            "initiation_interval_cycles": PerformanceTerm.unknown(
                "initiation_interval_cycles", "cycles",
                Provenance(_KIND_ANALYTICAL, ("facts.timing (feed-forward depth walk)",)),
                _validity(elaboration, "no source in the contract states an initiation interval"),
                "the structural walk derives COMPLETION latency (registers on a path), which is a "
                "different number from the interval between successive issues; conflating them is "
                "how a vendor latency table makes a correctly-scheduled program look under-delayed. "
                "Needs a two-point measurement (>=2 points per fitted parameter)"),
        }
        out.append(ResourceTerms(
            name=uname, kind=ResourceKind.COMPUTE, terms=terms,
            evidence=(f"declared compute unit {uname!r} (kind {unit.get('kind')!r})"
                      + (f" bound to discovered array {array.get('name')!r} "
                         f"[{array.get('container')}/{array.get('element')}]" if array else
                         "; no discovered array"))))
    return out


# --------------------------------------------------------------------------------------------
# Memory resources -- where two sources disagree, nobody wins
# --------------------------------------------------------------------------------------------


def _declared_capacities(profile: TargetProfile) -> dict[str, tuple[int, str]]:
    """``{memory name: (bytes, where it was declared)}`` from the residual.

    Reads the two schema-blessed places a capacity is declared: any ``<name>_bytes`` key under
    ``memory_model``, and ``capabilities.resident_storage_bytes``. Both are DECLARATIONS.
    """
    out: dict[str, tuple[int, str]] = {}
    suffix = "_bytes"
    mm = profile.sources.residual.get("memory_model") or {}
    for key, value in (mm.items() if isinstance(mm, Mapping) else ()):
        if str(key).endswith(suffix) and isinstance(value, int) and not isinstance(value, bool):
            out[str(key)[: -len(suffix)]] = (int(value), f"residual.memory_model.{key}")
    caps = profile.sources.residual.get("capabilities") or {}
    rs = caps.get("resident_storage_bytes") if isinstance(caps, Mapping) else None
    if isinstance(rs, int) and not isinstance(rs, bool):
        out.setdefault("resident_storage", (int(rs), "residual.capabilities.resident_storage_bytes"))
    return out


def capacity_terms(profile: TargetProfile) -> list[ResourceTerms]:
    """One resource per memory, with ``capacity_bytes`` refusing to pick a side on a disagreement.

    A capacity that two sources state differently is not a rounding question: one of them describes
    a different elaboration, or a shipped model config that was never the RTL. Choosing silently
    produces a number that reproduces nowhere, so the term stays UNKNOWN and names both values --
    which is a finding, and is actionable, in a way that a quietly-preferred value is not.
    """
    elaboration = profile.elaboration
    grounded = {str(m["name"]): (int(m["bytes"]), m) for m in profile.sources.memories()
                if m.get("name") and isinstance(m.get("bytes"), int)}
    declared = _declared_capacities(profile)
    out: list[ResourceTerms] = []
    for name in sorted(set(grounded) | set(declared)):
        g = grounded.get(name)
        d = declared.get(name)
        if g is not None and d is not None and g[0] != d[0]:
            term = PerformanceTerm.unknown(
                "capacity_bytes", "bytes",
                Provenance(_KIND_STRUCTURAL, (f"facts.memories[{name}].bytes", d[1])),
                _validity(elaboration, f"two sources state {name}'s capacity"),
                f"the sources DISAGREE: the RTL facts say {g[0]} bytes and {d[1]} says {d[0]} "
                f"bytes. Neither is preferred here -- a capacity chosen between disagreeing "
                f"sources is unreproducible, and the disagreement itself is the finding")
            ev = f"memory {name!r}: facts {g[0]} B vs declaration {d[0]} B -- unresolved"
        elif g is not None:
            mem = g[1]
            term = PerformanceTerm(
                name="capacity_bytes", value=g[0], unit="bytes",
                provenance=Provenance(_KIND_STRUCTURAL,
                                      (f"facts.memories[{name}]: {g[0]} bytes, depth "
                                       f"{mem.get('depth')}, via {mem.get('source')}",)),
                validity=_validity(elaboration, f"the {name} discovered in this elaboration"),
                bounds=Bounds(lower=0))
            ev = f"memory {name!r} discovered in the RTL facts"
        elif d is not None:
            term = PerformanceTerm(
                name="capacity_bytes", value=d[0], unit="bytes",
                provenance=Provenance(_KIND_DECLARED, (d[1],)),
                validity=_validity(elaboration,
                                   f"{name}'s capacity as DECLARED by {d[1]}; no RTL fact "
                                   "corroborates it",
                                   weak_regime="a declaration is intent, not evidence: it may "
                                               "describe a different elaboration than the one this "
                                               "contract's structural terms came from"),
                bounds=Bounds(lower=0))
            ev = f"memory {name!r} declared in the residual, not grounded"
        else:                                       # unreachable: name came from one of the two
            continue
        out.append(ResourceTerms(name=name, kind=ResourceKind.OTHER,
                                 terms={"capacity_bytes": term}, evidence=ev))
    if not out:
        # No memory is named by either source. The trait still says whether the machine HAS a managed
        # store; the capacity is what is missing, and it is recorded as such rather than omitted.
        trait = profile.trait("managed_scratchpad")
        out.append(ResourceTerms(
            name="operand_store", kind=ResourceKind.OTHER,
            terms={"capacity_bytes": PerformanceTerm.unknown(
                "capacity_bytes", "bytes",
                Provenance(_KIND_ANALYTICAL, ("facts.memories (empty)", "residual.memory_model")),
                _validity(elaboration, "no source states an operand-store capacity"),
                f"no memory capacity is grounded or declared for this target "
                f"(managed_scratchpad={trait.satisfied!r}: {trait.evidence}). A capacity guessed "
                f"from a shipped model config would be residual-tier at most and is never a fact")},
            evidence="no memory is named by either source"))
    return out


# --------------------------------------------------------------------------------------------
# Movement + fixed
# --------------------------------------------------------------------------------------------


def _movement_resource(profile: TargetProfile) -> ResourceTerms | None:
    """The data-movement resource, when the target's own facts evidence one.

    Both terms are UNKNOWN by construction today: the engines that move data are sequenced, so the
    structural walk refuses them, and their cost is a function of descriptor bytes rather than of
    wiring depth. That is worth emitting rather than omitting -- on a machine where movement is most
    of every cycle, an absent resource reads as a machine with no movement cost.
    """
    trait = profile.trait("explicit_dma")
    if trait.satisfied is not True:
        return None
    elaboration = profile.elaboration
    return ResourceTerms(
        name="data_movement", kind=ResourceKind.MOVEMENT,
        terms={
            "peak_bytes_per_cycle": PerformanceTerm.unknown(
                "peak_bytes_per_cycle", "bytes/cycle",
                Provenance(_KIND_STRUCTURAL, ("facts.interfaces (movement engine)",)),
                _validity(elaboration, "the movement engine named by the facts interfaces"),
                "no fact states this engine's beat width or its issue rate; the structural walk "
                "derives feed-forward depth, and a sequenced engine has none. Needs a measurement "
                "at >=2 transfer sizes to separate the rate from the fixed per-transfer cost"),
            "base_latency_cycles": PerformanceTerm.unknown(
                "base_latency_cycles", "cycles",
                Provenance(_KIND_STRUCTURAL, ("facts.timing (sequenced units refuse)",)),
                _validity(elaboration, "the movement engine named by the facts interfaces"),
                "the movement engine is sequenced: its latency is a function of state and byte "
                "count, so no wiring depth is its latency. This is the term to buy first where "
                "movement dominates the cycle count"),
        },
        evidence=f"movement engine evidenced by {trait.evidence}")


def _fixed_resource(profile: TargetProfile) -> ResourceTerms:
    """The intercept: what a run costs before any rate applies.

    Fixed terms are first-class. A rate-only model mispredicts every small workload, and a corpus of
    small tiles is mostly intercept.
    """
    elaboration = profile.elaboration
    return ResourceTerms(
        name="fixed", kind=ResourceKind.FIXED,
        terms={"startup_cycles": PerformanceTerm.unknown(
            "startup_cycles", "cycles",
            Provenance(_KIND_ANALYTICAL, ("no static source states a startup cost",)),
            _validity(elaboration, "the run's fixed cost before any rate applies"),
            "reset, program load and the first issue are not properties of the wiring, so no static "
            "fact carries them. They are separable by an isolation experiment (a program whose "
            "first instruction is the halt) and never by fitting a rate")},
        evidence="the intercept every rate-only model omits")


# --------------------------------------------------------------------------------------------
# The contract
# --------------------------------------------------------------------------------------------


def derive_contract(target: str, *, profile: TargetProfile | None = None,
                    facts: Mapping[str, Any] | None = None,
                    residual: Mapping[str, Any] | None = None,
                    allow_extraction: bool = False) -> PerformanceContract:
    """Derive ``target``'s performance contract from its profile.

    One code path for every target. Which resources appear, and which of their terms carry a value,
    is decided entirely by what that target's own sources evidence.
    """
    prof = profile if profile is not None else derive_profile(
        target, facts=facts, residual=residual, allow_extraction=allow_extraction)

    resources: list[ResourceTerms] = list(_compute_resources(prof))
    movement = _movement_resource(prof)
    if movement is not None:
        resources.append(movement)
    resources.extend(capacity_terms(prof))
    resources.append(_fixed_resource(prof))

    gaps: list[Unavailable] = []
    if movement is None:
        dma = prof.trait("explicit_dma")
        gaps.append(Unavailable(
            "a data-movement resource", tuple(dma.missing),
            detail=("no movement engine is evidenced, so this contract prices none. That is NOT a "
                    "claim that the target does not move data -- where it does, the missing "
                    "resource is likely the dominant one, and its absence here is a hole in the "
                    "evidence rather than in the machine")))
    completion = prof.trait("explicit_completion")
    ports = prof.trait("independent_engine_ports")
    if completion.satisfied is not True or ports.satisfied is not True:
        gaps.append(Unavailable(
            "the composition operator (how these resources' times compose into a runtime)",
            tuple(dict.fromkeys((*completion.missing, *ports.missing))),
            detail=("it is NEVER defaulted to max: textbook roofline assumes perfect overlap, and a "
                    "target that takes turns sums instead, so deriving max where the truth is sum "
                    "understates runtime in the flattering direction. "
                    "merlin.perf.headroom.composition_operator answers it from an activity source "
                    "with an independent overlap observation")))
    if not prof.elaboration.evidenced:
        gaps.append(Unavailable(
            "an evidenced elaboration for every term",
            ("a digest for the dialect the extractor actually read",),
            detail=prof.elaboration.note or "the facts artifact records no digest for its input"))
    if prof.timing.status != "present":
        gaps.append(Unavailable(
            "structural pipeline depths",
            ("a facts artifact carrying a timing block",),
            detail=(f"the timing fact class is {prof.timing.status} for this target. UNCACHED is "
                    "not the same as absent: the fact class exists and a re-extraction answers it")))
    return PerformanceContract(target=target, profile=prof, resources=tuple(resources),
                               gaps=tuple(gaps))


def contract_table(contracts: Sequence[PerformanceContract]) -> str:
    """Side-by-side term values for several targets -- the anti-overfit result, readable."""
    keys: list[str] = []
    for c in contracts:
        for k in c.terms():
            if k not in keys:
                keys.append(k)
    width = max([len(k) for k in keys] + [len("term")])
    head = "term".ljust(width) + "".join(f"  {c.target:>14}" for c in contracts)
    rows = [head, "-" * len(head)]
    for k in keys:
        cells = []
        for c in contracts:
            t = c.terms().get(k)
            cells.append(f"  {'--' if t is None else ('UNKNOWN' if t.is_unknown else t.value):>14}")
        rows.append(k.ljust(width) + "".join(str(x) for x in cells))
    return "\n".join(rows)
