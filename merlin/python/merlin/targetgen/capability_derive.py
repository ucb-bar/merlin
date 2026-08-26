"""Derive a target's SEMANTIC CAPABILITIES from its own evidence, and audit what it declares.

The ARR denominator is a claim about hardware: "this target can execute these families over these
formats". A claim nothing checks is an assertion, and an assertion is exactly what a reviewer will
attack — the more so because both directions of error move the score. Over-declare and the compiler
carries permanent ``false_fallback`` it can never clear; under-declare and recall flatters itself by
shrinking its own denominator.

So the facts are produced **during the run, from the target's own sources**, and compared against what
the contract declares. Nothing here knows a target name, an opcode or a mnemonic.

**Evidence ladder**, strongest first — each finding records which rung produced it and the literal thing
observed, so a declaration can always be traced back to hardware:

``isa_role``
    The structural role census (:mod:`merlin.targetgen.isa_taxonomy`), whose roles come from each
    instruction's own typed operands — a class whose destination is the accumulator and whose sources
    include a weight is a contraction whatever it is called. This is the only rung that works on a
    self-hosted ISA, and it is deliberately name-blind.
``isa_class``
    The SHARED, closed ``encoding.semantic_class`` vocabulary a contract declares, via
    :func:`merlin.targetgen.semantic_families.from_isa_class`. Not mnemonics — the human-owned class
    names the compiler and the trace decoder both speak.
``rtl_facts``
    Extracted hardware: a MAC array licenses contraction; an input datapath grounds its dtypes; a DMA
    interface licenses movement.
``unit_intent``
    The residual's own ``ops`` / ``scaling`` / ``requant`` — weakest, and the only rung that can produce
    an epilogue-only capability.

**Three states, never two.** ``supported`` / ``unsupported`` / ``unknown``. ``unknown`` means no rung
capable of deciding the family was available — which is not the same as "the hardware cannot". It is
reported and excluded from both sides of the ratio (see
:func:`merlin.targetgen.eligibility.is_eligible`), because scoring an undecidable family either way
would move ARR for a reason about our evidence rather than about the compiler.

**Standing limit: the ladder DECIDES families; the shape axes it only AUDITS.** Every rung below
answers "can this target contract at all", never "over which ranks, dtypes or layouts". That gap
once cost a real number: gemmini declared ``contraction ranks: [2]`` while its own funct table
carries ``LOOP_CONV_WS`` and its config companions, so every shipped rank-4 conv2d capsule scored
ineligible and quietly left the ARR denominator -- flattering recall by exactly the regions the mesh
handles best. A human review caught that one; nothing would have caught the next.

:func:`reconcile` now closes the detection half. :func:`_axis_findings` compares each declared axis
against whatever evidence the rungs did produce and reports ``missing_axis`` (a rung evidenced a
value the contract omits -- under-declared, and it RAISES recall) or ``unaudited_axis`` (the
contract declares values no rung could confirm). So a narrow axis is still a review obligation, but
it is now a PRINTED one rather than one that depends on somebody noticing.

Deciding the axes is blocked on evidence, not on this module. Ranks would need the loop-instruction
census, and the facts bundle's funct decode table carries instruction NAMES only -- matching them is
exactly the hardcoding the repo forbids, and the conv and GEMM loop nests each carry six config
companions, so their structure does not separate them either. Every target's ``compute_units[].ops``
is matmul-only, so an op-derived rung could only ever answer 2. Promoting the audit to a decision
needs a new extractor (loop-bound register counting out of the RTL decoder), not a new rung here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from merlin.targetgen import semantic_families as _sf

#: Structural ISA roles -> the family that role licenses. Roles are derived from an instruction's typed
#: operands, so this table reads hardware structure, not names. ``acc_readout_scaled`` is the epilogue:
#: a SCALED accumulator pop is a requant on the readout path, which is an elementwise map available only
#: fused with the contraction that filled the accumulator -- never standalone.
_ROLE_FAMILY: dict[str, tuple[str, tuple[str, ...]]] = {
    "matmul": ("contraction", ()),
    "memory": ("movement", ()),
    "tensor_compute_unary": ("elementwise_map", ()),
    "tensor_compute_binary": ("elementwise_map", ()),
    "acc_readout_scaled": ("elementwise_map", ("contraction",)),
    # weight_load / acc_seed / acc_readout are contraction PLUMBING: they feed or drain the mesh and
    # license nothing on their own. scalar is host-side. Absent on purpose.
}

#: Roles whose presence proves the census actually ran and covered compute, so a family it did NOT find
#: can be reported ``unsupported`` rather than ``unknown``. Without one of these the census is silent,
#: not negative.
_CENSUS_IS_CONCLUSIVE_FOR = ("contraction",)

#: The same structural roles -> the ENGINE (a CCA facet name) that role evidences.
#:
#: Read this next to ``_ROLE_FAMILY`` above, because the two disagree on purpose and the disagreement
#: is the point. That table deliberately omits ``weight_load`` / ``acc_seed`` / ``acc_readout`` --
#: "contraction PLUMBING ... license nothing on their own" -- which is right about FAMILIES and exactly
#: wrong about ENGINES: an instruction whose job is to push weights into a grid or drain an accumulator
#: is the strongest possible evidence that a grid EXISTS. So the roles one table throws away are the
#: ones the other is built on, and atlas's census reports precisely those three as ``unmapped`` today.
#:
#: Values are FACET names, not compute-unit kinds, because that is the granularity this evidence
#: actually reaches. A role census cannot tell a stationary-weight wavefront from a rank-1
#: outer-product tile -- both push weights and drain accumulators -- so claiming ``systolic`` over
#: ``spatial`` would assert something no rung observed. The facet is the honest unit of the claim.
_ROLE_ENGINE: dict[str, str] = {
    "matmul": "spatial",
    "weight_load": "spatial",
    "acc_seed": "spatial",
    "acc_readout": "spatial",
    "acc_readout_scaled": "spatial",
    "tensor_compute_unary": "vector",
    "tensor_compute_binary": "vector",
    # `memory` and `scalar` evidence NO engine: every target has data movement and a scalar core, so
    # their presence distinguishes nothing. Absent on purpose, as above.
}


@dataclass(frozen=True)
class FamilyEvidence:
    """One rung's verdict on one family, with the literal observation that produced it."""

    family: str
    status: str                              # supported | unsupported | unknown
    source: str                              # isa_role | isa_class | rtl_facts | unit_intent
    evidence: str
    dtypes: tuple[str, ...] = ()
    ranks: tuple[int, ...] = ()
    composed_with: tuple[str, ...] = ()
    unit: str = ""


@dataclass
class DerivedCapabilities:
    """What the evidence says, keyed by family, plus everything it could not decide."""

    supported: dict[str, FamilyEvidence] = field(default_factory=dict)
    unknown: dict[str, FamilyEvidence] = field(default_factory=dict)
    unmapped: list[str] = field(default_factory=list)   # observed classes/roles no rung could place

    def families(self) -> list[str]:
        return sorted(self.supported)

    def to_dict(self) -> dict[str, Any]:
        return {
            "semantic_capabilities_derived": [
                {"family": e.family, "dtypes": list(e.dtypes), "ranks": list(e.ranks),
                 "composed_with": list(e.composed_with), "source": e.source,
                 "evidence": e.evidence, "unit": e.unit}
                for e in (self.supported[f] for f in sorted(self.supported))
            ],
            "semantic_capabilities_unknown": [
                {"family": e.family, "reason": e.evidence, "sources_tried": e.source}
                for e in (self.unknown[f] for f in sorted(self.unknown))
            ],
            "unmapped_observations": sorted(self.unmapped),
        }


def _record(out: DerivedCapabilities, ev: FamilyEvidence) -> None:
    """Keep the STRONGEST verdict per family. A later, weaker rung may add dtypes but must not
    downgrade a family an earlier rung established, and must never overwrite its evidence string."""
    if ev.status != "supported":
        if ev.family not in out.supported:
            out.unknown.setdefault(ev.family, ev)
        return
    out.unknown.pop(ev.family, None)
    prev = out.supported.get(ev.family)
    if prev is None:
        out.supported[ev.family] = ev
        return
    merged_dtypes = tuple(dict.fromkeys((*prev.dtypes, *ev.dtypes)))
    # composed_with INTERSECTS: if any rung saw the family standalone, it is standalone.
    merged_comp = tuple(c for c in prev.composed_with if c in ev.composed_with)
    out.supported[ev.family] = FamilyEvidence(
        family=prev.family, status="supported", source=prev.source,
        evidence=prev.evidence, dtypes=merged_dtypes,
        # ranks UNION like dtypes. `prev.ranks or ev.ranks` kept only the first rung's ranks, so a
        # later rung that evidenced a DIFFERENT rank had it silently dropped -- an axis lost inside the
        # deriver, in the same direction (narrower than the evidence) as the gap this module documents.
        ranks=tuple(sorted({*prev.ranks, *ev.ranks})),
        composed_with=merged_comp, unit=prev.unit or ev.unit)


def _unit_dtypes(unit: dict) -> tuple[str, ...]:
    """Operand formats a unit accepts: its declared ``dtypes``, plus every ``in``/``weight`` named in an
    accumulate rule (a rule naming a format is proof the unit takes it)."""
    seen = list(unit.get("dtypes") or ())
    for rule in unit.get("accumulate") or ():
        if isinstance(rule, dict):
            for key in ("in", "weight"):
                v = rule.get(key)
                if v and v not in seen:
                    seen.append(v)
    return tuple(seen)


def _from_isa_roles(taxonomy: dict, out: DerivedCapabilities, dtypes: tuple[str, ...]) -> bool:
    """Rung 1 — the structural role census. Returns whether it ran conclusively."""
    from merlin.targetgen import isa_taxonomy as _it

    by_role = _it._classes_by_role(taxonomy or {})
    if not by_role:
        return False
    for role, classes in sorted(by_role.items()):
        mapped = _ROLE_FAMILY.get(role)
        if mapped is None:
            if role != "scalar" and classes:
                out.unmapped.append(f"isa_role:{role}({len(classes)})")
            continue
        fam, comp = mapped
        _record(out, FamilyEvidence(family=fam, status="supported", source="isa_role",
                                    evidence=f"ISA role {role!r} -> {classes[:3]}",
                                    dtypes=dtypes, composed_with=comp))
    return any(_ROLE_FAMILY.get(r, ("", ()))[0] in _CENSUS_IS_CONCLUSIVE_FOR for r in by_role)


# ---------------------------------------------------------------------------------------------
# Engines — which COMPUTE DATAPATHS the evidence says a target has
# ---------------------------------------------------------------------------------------------
#
# Same discipline as the family ladder above and for the same reason: a target's declared
# ``compute_units`` is a claim about hardware, and a claim nothing checks is an assertion. The
# measured motivation is atlas -- its contract declares ONE systolic unit whose ops are ``('matmul',)``
# while 50 of its 137 expert kernels drive a vector engine EXCLUSIVELY -- so a target described by its
# declaration alone can be most of a machine short.
#
# Nothing here writes ``compute_units``. The findings are recorded BESIDE the declaration, exactly as
# ``derive`` records semantic capabilities beside it, because a derivation bug must never be able to
# silently redescribe the silicon.


@dataclass(frozen=True)
class EngineEvidence:
    """One rung's verdict that a target has a given engine, with the literal observation."""

    engine: str                     # a CCA facet name (see merlin.kernels.engines.ENGINE_FACET)
    source: str                     # which rung: isa_role | rtl_facts
    evidence: str                   # the literal thing observed


#: What each rung is CAPABLE of observing, independent of what it found on any particular target.
#:
#: This exists because "a rung ran and did not find X" and "no rung can see X" are different facts that
#: want opposite responses, and the first reads as an over-declaration while the second is a gap in our
#: own instruments. Without this table the report tells a target it over-declares its SIMT cluster when
#: the truth is that neither rung has ever been able to see one.
#:
#: ⚠️ Note what is missing: NO rung observes ``simt``. A structural role census reads an instruction's
#: typed operands, and the thing that makes a SIMT engine SIMT -- many threads of control over one
#: instruction stream -- is not a property of any single instruction's operands. Nothing here can fix
#: that by trying harder; it needs a rung that reads warp/barrier structure. Until one exists, every
#: SIMT declaration is UNCHECKED, and this table is what makes the report say so instead of guessing.
_RUNG_CAN_SEE: dict[str, frozenset[str]] = {
    "isa_role": frozenset({"spatial", "vector"}),
    "rtl_facts": frozenset({"spatial"}),
}


@dataclass
class DerivedEngines:
    """Engines the evidence reaches, and which rungs were in a position to decide."""

    evidenced: dict[str, EngineEvidence] = field(default_factory=dict)
    #: Rungs that actually ran on this target -- not merely "were importable". A census that saw only
    #: scalar classes has not covered compute, and does not count.
    rungs: tuple[str, ...] = ()
    #: Observations a rung PRODUCED but could not corroborate, engine -> what was seen. These are NOT
    #: evidence and never enter `evidenced`.
    #:
    #: The fourth state, and the one the first three could not express. `undeclared` / `unevidenced` /
    #: `unchecked` all locate a disagreement in the CONTRACT -- it over-declares, it under-declares, or
    #: we could not look. None of them can say the instrument is at fault, so a bad reading gets
    #: reported as a hardware finding with full confidence. Measured: a SIMT target was told it had an
    #: undeclared array engine on the strength of an RTL "17x17 mesh" that was 289 D-flip-flops inside a
    #: float divide/sqrt unit, 289 being a perfect square. Nothing in the report could have said so.
    suspect: dict[str, str] = field(default_factory=dict)

    def engines(self) -> list[str]:
        return sorted(self.evidenced)

    def observable(self) -> frozenset[str]:
        """Engines SOME rung that ran could have evidenced. Anything outside this is unchecked."""
        out: set[str] = set()
        for r in self.rungs:
            out |= _RUNG_CAN_SEE.get(r, frozenset())
        return frozenset(out)

    def to_dict(self) -> dict[str, Any]:
        return {
            "engines_derived": [
                {"engine": e.engine, "source": e.source, "evidence": e.evidence}
                for e in sorted(self.evidenced.values(), key=lambda x: x.engine)
            ],
            "engines_rungs_ran": list(self.rungs),
            "engines_observable": sorted(self.observable()),
            "engines_suspect": [{"engine": e, "observation": o}
                                for e, o in sorted(self.suspect.items())],
        }


def _array_observation(a: dict) -> str:
    """The literal thing observed in the RTL, not the dimension concluded from it.

    ``16x16`` is the CONCLUSION. ``256 instances of 'Tile' in 'Mesh'`` is the observation, and only the
    second can be read and refuted by someone who knows the hardware."""
    dims = f"{a.get('rows')}x{a.get('cols')}"
    what = f"array {a.get('name')!r} {dims}"
    element, instances, container = a.get("element"), a.get("instances"), a.get("container")
    if element and instances:
        what += f" <- {instances} instances of {element!r}"
        if container:
            what += f" in {container!r}"
        variants = a.get("element_variants") or []
        if len(variants) > 1:
            what += f" (across {len(variants)} structural variants)"
    mac = a.get("mac_idiom") or {}
    if mac:
        what += f", mac idiom {dict(sorted(mac.items()))}"
    elif not a.get("corroborated"):
        what += ", NO multiply-accumulate idiom confirmed"
    return f"{what} ({a.get('source', 'rtl')})"


def _engine(out: DerivedEngines, ev: EngineEvidence) -> None:
    """Record an engine, keeping the FIRST (strongest-rung) observation for it."""
    out.evidenced.setdefault(ev.engine, ev)


def derive_engines(target: str, contract: dict, facts: dict | None = None, *,
                   taxonomy: dict | None = None) -> DerivedEngines:
    """Which compute engines this target's OWN evidence reaches. Never raises on missing evidence.

    Two rungs, strongest first, mirroring the family ladder:

    ``isa_role``   the structural role census -- name-blind, and the only rung that works on a
                   self-hosted ISA.
    ``rtl_facts``  an extracted MAC array. A grid of multiply-accumulate cells IS an array engine;
                   nothing weaker is claimed from it.

    ``unit_intent`` is deliberately NOT a rung here. The contract's own ``compute_units`` is the thing
    being audited, so reading it back would make every target trivially agree with itself.
    """
    out = DerivedEngines()
    rungs: list[str] = []

    if taxonomy is None:
        try:
            from merlin.targetgen import isa_taxonomy as _it
            taxonomy = _it.taxonomy_for_target(target)
        except Exception:  # noqa: BLE001 — no self-hosted ISA (a RoCC target); the other rung still runs
            taxonomy = None
    if taxonomy:
        from merlin.targetgen import isa_taxonomy as _it
        by_role = _it._classes_by_role(taxonomy or {})
        # A census that found ONLY scalar classes has read the ISA without covering compute, so it is
        # silent about engines rather than negative about them -- the same distinction
        # `_CENSUS_IS_CONCLUSIVE_FOR` draws for families. Radiance's taxonomy is exactly this shape
        # (6 scalar classes, nothing else), and counting it as a compute census made the report accuse
        # its contract of over-declaring an engine the census never looked for.
        if any(role != "scalar" and classes for role, classes in by_role.items()):
            rungs.append("isa_role")
            for role, classes in sorted(by_role.items()):
                eng = _ROLE_ENGINE.get(role)
                if eng and classes:
                    _engine(out, EngineEvidence(engine=eng, source="isa_role",
                                                evidence=f"ISA role {role!r} -> {list(classes)[:3]}"))

    body = (facts or {}).get("facts") or {}
    arrays = [a for a in (body.get("arrays") or ()) if isinstance(a, dict)]
    # An array fact evidences an engine only when it carries the corroboration that makes it one: WHICH
    # element was found replicated, how many times, and the multiply-accumulate idiom confirming the
    # element computes. A bare rows x cols is a ranking, not an observation -- the extractor that
    # produced it could not fail, so it reported a grid for every design it was given.
    good = [a for a in arrays if a.get("corroborated")]
    if good:
        rungs.append("rtl_facts")
        for a in good:
            _engine(out, EngineEvidence(engine="spatial", source="rtl_facts",
                                        evidence=_array_observation(a)))
    for a in arrays:
        if not a.get("corroborated"):
            # Recorded, never promoted, and pointedly NOT counted as a rung that ran: an uncorroborated
            # reading must not make a declared engine look over-declared ("a capable rung missed it")
            # when the truth is that the rung returned something we cannot trust.
            out.suspect.setdefault("spatial", _array_observation(a))

    out.rungs = tuple(rungs)
    return out


def reconcile_engines(declared: "set[str] | frozenset[str]", derived: DerivedEngines) -> list[str]:
    """Compare the DECLARED engine facets against the evidenced ones. Returns drift, newest concern first.

    FOUR outcomes, never fewer, for the same reason the family ladder insists on three: "the evidence
    contradicts this", "no rung could look", and "the rung answered but we cannot trust it" want
    different responses, and collapsing them is how an evidence gap gets actioned as a hardware fact.
    The fourth (``suspect_evidence``) is the one that has to point at OUR instrument rather than at the
    contract -- see :attr:`DerivedEngines.suspect`.
    """
    drift: list[str] = []
    ev = set(derived.evidenced)
    observable = derived.observable()
    for engine in sorted(set(derived.suspect) - ev):
        drift.append(f"suspect_evidence {engine}: a rung produced {derived.suspect[engine]}, which "
                     f"carries no corroboration that it is a compute grid -- this is a finding about "
                     f"OUR extractor, not about the contract, and neither confirms nor refutes the "
                     f"declaration")
    for engine in sorted(ev - set(declared)):
        e = derived.evidenced[engine]
        drift.append(f"undeclared_engine {engine}: evidenced by {e.source} ({e.evidence}) but the "
                     f"contract's compute_units declare no unit of that kind")
    for engine in sorted(set(declared) - ev):
        if engine in observable:
            drift.append(f"unevidenced_engine {engine}: declared, and a rung that CAN see this engine "
                         f"class ran ({', '.join(derived.rungs)}) without finding it -- the "
                         f"declaration over-reaches, or the evidence for it is missing")
        else:
            why = (f"the rung(s) that ran ({', '.join(derived.rungs)}) cannot observe this engine class"
                   if derived.rungs else "no rung was available to check it")
            drift.append(f"unchecked_engine {engine}: declared, and {why} -- this is a gap in OUR "
                         f"instruments, not a finding about the hardware, and must not be actioned "
                         f"as one")
    return drift


def _from_isa_classes(contract: dict, out: DerivedCapabilities, dtypes: tuple[str, ...]) -> None:
    """Rung 2 — the shared ``encoding.semantic_class`` vocabulary."""
    classes = (contract.get("encoding") or {}).get("semantic_class") or {}
    for name in (classes.values() if isinstance(classes, dict) else classes):
        fam = _sf.from_isa_class(str(name))
        if fam is None:
            out.unmapped.append(f"isa_class:{name}")
            continue
        _record(out, FamilyEvidence(family=fam, status="supported", source="isa_class",
                                    evidence=f"declared semantic_class {name!r}", dtypes=dtypes))


def _from_rtl_facts(facts: dict, out: DerivedCapabilities) -> None:
    """Rung 3 — extracted hardware. A MAC array is a contraction engine; an input datapath grounds the
    formats; a DMA interface moves data without arithmetic, which is the movement family by definition."""
    body = (facts or {}).get("facts") or {}
    dtypes: tuple[str, ...] = ()
    for dp in body.get("datapaths") or ():
        if isinstance(dp, dict) and dp.get("name") == "input" and dp.get("dtype"):
            dtypes = (str(dp["dtype"]),)
            break
    for arr in body.get("arrays") or ():
        if isinstance(arr, dict) and arr.get("rows") and arr.get("cols"):
            _record(out, FamilyEvidence(
                family="contraction", status="supported", source="rtl_facts",
                evidence=f"RTL array {arr.get('name')!r} {arr['rows']}x{arr['cols']}",
                dtypes=dtypes, ranks=(2,)))
            break
    for iface in body.get("interfaces") or ():
        nm = str((iface or {}).get("name", "")) if isinstance(iface, dict) else ""
        if nm and "dma" in nm.split("_"):
            _record(out, FamilyEvidence(family="movement", status="supported", source="rtl_facts",
                                        evidence=f"RTL interface {nm!r}", dtypes=dtypes))


def _from_unit_intent(contract: dict, out: DerivedCapabilities) -> None:
    """Rung 4 — the residual's own declaration of what each unit is for. Weakest rung, and the only one
    that can yield an EPILOGUE-only capability: a unit whose elementwise hardware is a readout requant
    can fuse it onto a contraction and cannot run it standalone."""
    for unit in contract.get("compute_units") or ():
        if not isinstance(unit, dict):
            continue
        name = str(unit.get("name", ""))
        dtypes = _unit_dtypes(unit)
        for op in unit.get("ops") or ():
            # A unit's ``ops`` list mixes op names ("matmul") with the coarse family words the capture
            # tags use ("elementwise"). Both tables are closed, target-agnostic and already reviewed, so
            # consult both rather than record a real capability as unmapped -- radiance's SIMT cluster
            # declares ``elementwise`` and would otherwise look epilogue-only, which it is not.
            fam = _sf.from_op(str(op)) or _sf.from_prov(str(op))
            if fam is None:
                out.unmapped.append(f"unit_op:{op}")
                continue
            _record(out, FamilyEvidence(family=fam, status="supported", source="unit_intent",
                                        evidence=f"unit {name!r} declares op {op!r}",
                                        dtypes=dtypes, unit=name))
        requant = (unit.get("requant") or {}).get("ref") if isinstance(unit.get("requant"), dict) else None
        scaling = unit.get("scaling")
        if (requant and str(requant).lower() != "none") or (scaling and str(scaling).lower() != "none"):
            _record(out, FamilyEvidence(
                family="elementwise_map", status="supported", source="unit_intent",
                evidence=f"unit {name!r} declares a readout epilogue (scaling={scaling!r}, "
                         f"requant={requant!r}) -- fused only",
                dtypes=dtypes, composed_with=("contraction",), unit=name))


def derive(target: str, contract: dict, facts: dict | None = None, *,
           taxonomy: dict | None = None) -> DerivedCapabilities:
    """Run the evidence ladder for one target. Never raises on missing evidence — a rung that cannot
    run simply contributes nothing, and every family no rung decided is reported ``unknown``."""
    out = DerivedCapabilities()
    units = [u for u in (contract.get("compute_units") or ()) if isinstance(u, dict)]
    dtypes = tuple(dict.fromkeys(d for u in units for d in _unit_dtypes(u)))

    conclusive = False
    if taxonomy is None:
        try:
            from merlin.targetgen import isa_taxonomy as _it
            taxonomy = _it.taxonomy_for_target(target)
        except Exception:  # noqa: BLE001 — no self-hosted ISA (a RoCC target); other rungs still run
            taxonomy = None
    if taxonomy:
        conclusive = _from_isa_roles(taxonomy, out, dtypes)

    _from_isa_classes(contract, out, dtypes)
    _from_rtl_facts(facts or {}, out)
    _from_unit_intent(contract, out)

    # Anything still unplaced is UNKNOWN unless a conclusive compute census ran and did not find it.
    for fam in _sf.PRIMITIVES:
        if fam in out.supported:
            continue
        if conclusive and fam in ("contraction",):
            out.unknown.pop(fam, None)
            continue
        out.unknown.setdefault(fam, FamilyEvidence(
            family=fam, status="unknown", source="isa_role,isa_class,rtl_facts,unit_intent",
            evidence="no evidence source could decide this family for this target"))
    return out


# --- audit: what the contract DECLARES vs what the evidence SHOWS ---------------------------------

#: Drift kinds, worst first. ``missing_declaration`` flatters ARR (hardware hidden from the
#: denominator); ``overbroad_declaration`` deflates it (work demanded that the hardware cannot do
#: standalone). Both are errors because both make the number mean something other than it says.
DRIFT_KINDS = ("missing_declaration", "overbroad_declaration", "unsupported_declaration",
               "undetermined_declaration")


#: The shape AXES a capability declares, and which direction an error in each one moves ARR. Within a
#: NON-EMPTY declaration both axes narrow: a value the hardware has but the contract omits makes every
#: region of that shape score ineligible, which removes it from the ARR DENOMINATOR and therefore
#: RAISES recall. That is the direction that flatters us, so it is the one worth naming.
#:
#: An EMPTY declaration is where the two axes part company, and assuming otherwise was a real defect:
#: `is_eligible` guards the rank check with `if c.ranks` (empty admits every rank) while `_dtype_ok`
#: returns False on an empty allowed-set (empty admits nothing). The audit reads which is which from
#: `eligibility.empty_declaration_is_narrowing` rather than restating it, so the auditor and the
#: grader cannot drift on what an omitted axis means.
_SHAPE_AXES = ("ranks", "dtypes")


def _axis_findings(fam: str, dec, ev: FamilyEvidence) -> list[dict]:
    """Audit one family's declared shape axes against the evidence.

    The ladder decides FAMILIES; no rung can currently decide an axis on its own (the funct table
    carries instruction NAMES and matching them is forbidden, the mlc encoding fact is absent for the
    reference target, and the declared unit ``ops`` are matmul-only). Silence about that is what let a
    narrow axis sit unnoticed: gemmini declared ``contraction ranks: [2]`` while its funct table
    carried a conv loop nest, so every rank-4 conv2d capsule scored ineligible and quietly left the
    denominator -- caught by a human, invisible to every check. This does not invent the missing rung.
    It makes the gap SAY SO, in the two directions that matter:

    * ``missing_axis`` -- a rung evidenced a value the contract does not declare. Under-declared, and
      it shrinks the denominator.
    * ``unaudited_axis`` -- the contract declares values no rung could confirm. Not an error: it is the
      REVIEW OBLIGATION the module docstring names, now written down where a report can print it
      instead of living in one reviewer's memory.
    """
    from merlin.targetgen import eligibility as _el   # lazy: eligibility must stay importable alone

    out: list[dict] = []
    for axis in _SHAPE_AXES:
        evidenced = tuple(getattr(ev, axis, ()) or ())
        declared_vals = tuple(getattr(dec, axis, ()) or ())
        if not declared_vals and not _el.empty_declaration_is_narrowing(axis):
            # The contract constrains NOTHING on this axis, and for this axis an empty declaration
            # admits every value (`is_eligible` guards the rank check with `if c.ranks`). So no region
            # is excluded, nothing leaves the denominator, and there is no claim to audit -- the two
            # findings below are both about CLAIMS. Reporting `missing_axis` here would be false in its
            # stated direction and, worse, actively harmful: its remedy is to declare the evidenced
            # value, which turns an axis that excluded nothing into one that excludes everything else.
            # Over-declaration is the opposite error and is caught downstream, by
            # `must_accelerate_violations` and `acceleration_precision`, not here.
            continue
        if not declared_vals and not evidenced:
            continue                       # nothing claimed and nothing seen: the axis is unconstrained
        if axis == "dtypes":
            # Compare through the FORMAT REGISTRY, not as strings. The deriver reads the RTL datapath's
            # spelling (`i8`) and the contract carries the capability vocabulary's (`int8`); the same
            # format under two names is not a missing axis, and reporting it as one would bury the real
            # findings in noise. `_dtype_ok` is the alias-aware predicate eligibility already grades with,
            # so the audit and the oracle agree on what "the same dtype" means.
            missing = {d for d in evidenced if not _el._dtype_ok(d, declared_vals)}
            unconfirmed = {d for d in declared_vals if not _el._dtype_ok(d, evidenced)}
        else:
            missing = set(evidenced) - set(declared_vals)
            unconfirmed = set(declared_vals) - set(evidenced)
        if missing:
            out.append({"kind": "missing_axis", "family": fam, "axis": axis, "source": ev.source,
                        "evidence": ev.evidence,
                        "detail": f"evidence shows {axis} {sorted(missing)} that the contract does not "
                                  f"declare (declared: {sorted(declared_vals)}); every region of that "
                                  f"shape scores ineligible and leaves the ARR denominator, which "
                                  f"RAISES recall"})
        if unconfirmed:
            out.append({"kind": "unaudited_axis", "family": fam, "axis": axis,
                        "source": ev.source if evidenced else "none",
                        "evidence": ev.evidence if evidenced else "no rung reported this axis",
                        "detail": f"contract declares {axis} {sorted(unconfirmed)} that no rung could "
                                  f"confirm; the ladder decides families, not shape axes, so this "
                                  f"stands on human review rather than on evidence"})
    return out


def reconcile(declared: dict, derived: DerivedCapabilities) -> list[dict]:
    """Compare a declared ``family -> SemanticCapability`` map against derived evidence.

    Deliberately does NOT rewrite the declaration. The denominator has to stay something a human
    reviewed: if a derivation bug could silently move it, ARR would drift with nobody noticing, and a
    target whose evidence is thin (empty RTL facts) would have its reviewed families deleted outright.
    The machine's job here is to be the auditor, not the author.
    """
    out: list[dict] = []
    for fam, ev in sorted(derived.supported.items()):
        dec = declared.get(fam)
        if dec is None:
            out.append({"kind": "missing_declaration", "family": fam, "source": ev.source,
                        "evidence": ev.evidence,
                        "detail": "evidence shows this family but the contract does not declare it; "
                                  "it is excluded from the ARR denominator"})
            continue
        if ev.composed_with and not getattr(dec, "composed_with", ()):
            out.append({"kind": "overbroad_declaration", "family": fam, "source": ev.source,
                        "evidence": ev.evidence,
                        "detail": f"evidence shows this family only fused with "
                                  f"{list(ev.composed_with)}, but it is declared standalone; every "
                                  f"standalone region becomes an unclearable false_fallback"})
        out.extend(_axis_findings(fam, dec, ev))
    for fam in sorted(declared):
        if fam in derived.supported:
            continue
        ev = derived.unknown.get(fam)
        kind = "undetermined_declaration" if ev else "unsupported_declaration"
        out.append({"kind": kind, "family": fam, "source": (ev.source if ev else "none"),
                    "evidence": (ev.evidence if ev else "no rung reported this family"),
                    "detail": "declared but not evidenced" + (
                        "; no source could decide it, so it stands unaudited" if ev else "")})
    return out
