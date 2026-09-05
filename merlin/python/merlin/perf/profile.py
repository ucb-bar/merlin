"""What KIND of machine this is, and which performance questions it can be asked.

A performance analysis needs two different things from a target. It needs a **prior** -- roughly
what shape of machine is this, so which questions are even worth asking -- and it needs, for each of
those questions, **evidence that the machine actually has the property**. This module derives both,
and keeps them strictly apart:

* an :class:`Archetype` is ONLY a prior. It decides *which questions to ask*. It never gates an
  analysis, it is never a target name, and being "the same archetype" as another target licenses no
  conclusion about either.
* a :class:`~merlin.perf.decompose.Trait` is the answer, derived from this target's own sources, and
  it is what the analyses gate on. Traits are tri-state: ``True`` / ``False`` / ``None``, and
  ``None`` ("could not be established") is not ``False``. "This machine has no DMA engine" and
  "nothing we read describes its data movement" demand different follow-up, and only the second one
  is a hole in our instruments.

THE THREE SOURCES, and their standing
-------------------------------------
Exactly the three that :func:`merlin.targetgen.capability_manifests.derive_manifest` uses, with the
same precedence, because a profile that disagreed with the capability manifest about what a target IS
would be a second, competing description of the same machine:

1. **RTL facts** (``merlin.targetgen.rtl.facts.load_facts``) -- geometry, memories, datapaths, the
   decode table, the per-module ``timing`` walk. GROUNDED: extracted from the target's own RTL.
2. **Family defaults** (``merlin.targetgen.families.family_profile``, keyed by compute-unit KIND) --
   what a machine of this kind does by default when nothing else says.
3. **The residual** (``<target_base>/contracts/residual.yaml``) -- the human intent and ABI prose the
   RTL cannot ground. DECLARED, never grounded: a residual saying a target has a scratchpad is a
   claim by its author, and the profile records it as such (see :attr:`TargetProfile.trait_tier`).

Facts win over the residual wherever they overlap, exactly as in ``derive_manifest``.

One thing, and only one, is also read from the target's **contract**: its ``compute_units`` block, as
the BASE declaration the residual overrides (:meth:`Sources.units`). The contract is not a fourth
trait source and gates nothing; it is consulted because a target that ships no residual file was
otherwise credited with no compute units at all, which lost its datapath kind and degraded its
archetype to ``unknown-datapath`` -- an absent OVERRIDE read as an absent DECLARATION. A kind that
came from the contract is tiered :data:`TIER_CONTRACT`, never :data:`TIER_FACTS`: it is declared, and
the RTL has not been asked.

WHAT THIS MODULE REFUSES TO DO
------------------------------
* It never infers a trait from a module's or a target's NAME. The interface tokens it reads
  (:data:`_IFACE_DMA` and friends) are the *extractor's own* interface-class vocabulary, emitted for
  any target that has such an interface -- the same tokens :mod:`merlin.system.derive` keys on.
* It never defaults an unestablished trait to a plausible value. Every ``None`` carries what was
  looked at and what is missing, so the gap is a work item rather than a silence.
* It never reads the ``timing`` walk's resolved-module COUNT as coverage. The walk resolves
  combinational leaves and refuses sequenced units, so its resolved fraction is biased *away* from
  where the time goes -- on one archetype the mesh itself refuses (accumulation feeds back), on
  another the DMA engine does while DMA is most of every cycle. The count is reported as a module
  count and labelled as one.
* ``pipeline_depth == 0`` is a REAL answer (combinational, no registers) and is preserved as ``0``.
  The consumer-side rule is ``if depth is None:``; ``if not depth:`` is the bug, and it is the same
  bug as reading UNKNOWN as ``0.0`` one level up.
"""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .decompose import Trait

__all__ = [
    "Archetype",
    "Elaboration",
    "Sources",
    "TRAITS",
    "TargetProfile",
    "TimingWalk",
    "derive_profile",
    "load_sources",
]

# --------------------------------------------------------------------------------------------
# The extractor's interface-CLASS vocabulary.
#
# These are NOT target names and not module names: they are the class tokens merlin's own RTL
# extractors emit for whichever target has such an interface (``circt_introspect`` /
# ``rtl.introspect`` / ``mlc_bridge``). Keying on them is how :mod:`merlin.system.derive` already
# derives operand placement and address translation; this module reads the same facts for the
# performance questions rather than growing a second vocabulary.
# --------------------------------------------------------------------------------------------
_IFACE_DECODE = "funct_decode_table"      # an instruction/command decode table was discovered
_IFACE_HOST_QUEUE = "rocc_cmd"            # a host co-processor command queue (decoupled dispatch)
_IFACE_DMA = "dma_tlb"                    # a data-movement engine with its own address translation
_IFACE_SELF_HOSTED = "self_hosted_isa"    # the device carries its own instruction encoding

#: Every trait this module derives. The order is the order they are reported in.
TRAITS: tuple[str, ...] = (
    "self_hosted_program",
    "host_dispatched_queue",
    "explicit_dma",
    "managed_scratchpad",
    "banked_memory",
    "persistent_configuration_state",
    "multiple_engine_groups",
    "independent_engine_ports",
    "explicit_completion",
    "structural_pipeline_depth",
    # The gate an L6_global (encoding/packing/layout) family needs. On a single-format machine the
    # question is absent rather than hard, so this derives a REFUTATION -- an evidence-bearing skip
    # saying "this target has one format" rather than an unestablished "we could not tell".
    "multiple_operand_encodings",
    "feedback_sequenced_units",
    # The gate an `L6_global` (encoding / packing / layout) family needs: a lever over operand
    # encoding only exists where the target admits more than one. `evaluate_gate` refuses any trait
    # outside this tuple, so declaring such a family without adding it here is not a YAML edit away --
    # it is unrepresentable.
    "multiple_operand_encodings",
)

#: Evidence tier tokens for :attr:`TargetProfile.trait_tier`. What a term built on this trait may
#: claim as its provenance kind depends on which of these settled it.
TIER_FACTS = "rtl_facts"                  # extracted from the target's own RTL
TIER_RESIDUAL = "residual_declared"       # declared by a human in the residual; intent, not evidence
TIER_CONTRACT = "contract_declared"       # declared in the target's own contract; the BASE declaration
TIER_FAMILY = "family_default"            # the compute-unit kind's default, nothing target-specific
TIER_NONE = "not_established"             # nothing settled it


# --------------------------------------------------------------------------------------------
# Sources
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Elaboration:
    """WHICH elaboration of the RTL a derived value belongs to.

    A structural number is a property of the design that was read, not of the target's name, so
    every term built from one has to be able to say which elaboration it holds for. When the facts
    artifact does not record a digest for the dialect it actually read, ``evidenced`` is False and
    the validity domain says the elaboration is *asserted* rather than evidenced -- which is a much
    weaker claim than a term with a digest, and must not read the same.
    """

    dialect: str | None = None
    digest: str | None = None
    extractor: str | None = None
    #: A path a fact block reported having read, when the ``inputs`` block names a different file.
    read_path: str | None = None
    evidenced: bool = False
    note: str = ""

    def describe(self) -> str:
        """One line naming the elaboration, for a term's ``validated_regime``."""
        if self.dialect is None and self.read_path is None:
            return f"no RTL elaboration is recorded ({self.note or 'no facts artifact'})"
        # With a digest, the recorded input IS the evidence. Without one, a path a fact block
        # reports having read is the better witness than an input nobody can show was opened.
        what = self.dialect if self.evidenced else (self.read_path or self.dialect)
        digest = self.digest if self.evidenced else "digest NOT recorded"
        return (f"the elaboration {what} ({digest}) as extracted by "
                f"{self.extractor or 'an unrecorded extractor'}")

    def to_dict(self) -> dict[str, Any]:
        return {"dialect": self.dialect, "digest": self.digest, "extractor": self.extractor,
                "read_path": self.read_path, "evidenced": self.evidenced, "note": self.note}


@dataclass(frozen=True)
class TimingWalk:
    """The state of the per-module RTL timing walk, with ABSENT and EMPTY kept apart.

    ``status`` is one of:

    ``present``
        The walk ran and reported modules.
    ``empty``
        The walk ran and reported no modules -- a design nobody's walk found anything in.
    ``uncached``
        The facts artifact carries no ``timing`` block at all. This means the artifact predates the
        timing extractor or was served from a stale cache; it does NOT mean the fact class does not
        exist and it must never be designed around. Re-extract to answer.
    ``no_facts``
        There is no facts artifact for this target on this host.

    ``resolved`` / ``refused`` are MODULE COUNTS, never coverage: the walk resolves the cheap
    combinational leaves and refuses the expensive sequenced units, so a high resolved fraction can
    coexist with every dominant resource being UNKNOWN.
    """

    status: str
    modules: dict[str, dict[str, Any]] = field(default_factory=dict)
    resolved: int = 0
    refused: int = 0

    @property
    def available(self) -> bool:
        return self.status == "present"

    def depth(self, module: str | None) -> tuple[int | None, str]:
        """``(pipeline_depth, evidence)`` for one module. ``None`` is UNKNOWN, ``0`` is REAL.

        The returned depth is passed through verbatim -- a resolved ``0`` (combinational, no
        registers on any output path) is a measured answer and stays ``0``. Callers must test
        ``is None``; ``if not depth:`` collapses the two and repeats, one level up, the bug of
        reading UNKNOWN as ``0.0``.
        """
        if module is None:
            return None, "no module was named to look up"
        if not self.available:
            return None, f"the RTL timing walk is {self.status} for this target"
        rec = self.modules.get(module)
        if rec is None:
            return None, f"module {module!r} is not among the {len(self.modules)} modules walked"
        depth = rec.get("pipeline_depth")
        evidence = str(rec.get("evidence") or "")
        if depth is None:
            return None, evidence or f"module {module!r} recorded no pipeline_depth"
        return int(depth), evidence

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status, "modules_walked": len(self.modules),
                "resolved_modules": self.resolved, "refused_modules": self.refused,
                "note": ("resolved/refused are MODULE COUNTS, not coverage: the walk resolves "
                         "combinational leaves and refuses sequenced units, so it is biased away "
                         "from where the cycles are spent")}


@dataclass(frozen=True)
class Sources:
    """The three sources, as read, plus what was missing."""

    target: str
    facts: dict[str, Any] = field(default_factory=dict)
    residual: dict[str, Any] = field(default_factory=dict)
    #: ``facts["facts"]`` -- the fact body, or ``{}``.
    body: dict[str, Any] = field(default_factory=dict)
    present: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()
    #: The target's own contract, as read. NOT a fourth source of TRAITS -- it is consulted for one
    #: thing only (see :meth:`units`), and it never overrides a residual or a fact.
    contract: dict[str, Any] = field(default_factory=dict)

    def interfaces(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for itf in self.body.get("interfaces") or []:
            if isinstance(itf, Mapping) and itf.get("name"):
                out[str(itf["name"])] = dict(itf)
        return out

    def arrays(self) -> list[dict[str, Any]]:
        return [dict(a) for a in (self.body.get("arrays") or []) if isinstance(a, Mapping)]

    def memories(self) -> list[dict[str, Any]]:
        return [dict(m) for m in (self.body.get("memories") or []) if isinstance(m, Mapping)]

    def units(self) -> list[dict[str, Any]]:
        """The target's compute units -- from the RESIDUAL if it declares any, else its CONTRACT.

        THE PRECEDENCE IS DELIBERATE AND IT IS NOT ARBITRARY. A contract's ``compute_units`` block is
        the target's own BASE declaration of what computes: every target that ships a contract has one,
        it is schema-validated, and :mod:`merlin.targetgen.compute_units` already parses it as the
        authority for routing. The residual is an OVERRIDE -- the side-input a human writes when the
        base declaration is wrong or incomplete for this analysis -- so where both speak the residual
        wins, exactly as it does for every other field this module reads.

        Reading only the residual (which is what this did) meant a target that ships no residual file
        had NO compute units at all, so its datapath kind was lost and its archetype degraded to
        ``unknown-datapath`` -- costing it the questions its class exists to ask. That is not a fact
        about the machine: the contract declares its kinds plainly. An absent OVERRIDE was being read
        as an absent DECLARATION, which is the same collapse as reading UNKNOWN as False one level up.
        """
        declared = [dict(u) for u in (self.residual.get("compute_units") or [])
                    if isinstance(u, Mapping)]
        if declared:
            return declared
        return [dict(u) for u in (self.contract.get("compute_units") or [])
                if isinstance(u, Mapping)]

    def units_source(self) -> str:
        """Which source supplied :meth:`units` -- ``residual`` / ``contract`` / ``none``."""
        if [u for u in (self.residual.get("compute_units") or []) if isinstance(u, Mapping)]:
            return "residual"
        if [u for u in (self.contract.get("compute_units") or []) if isinstance(u, Mapping)]:
            return "contract"
        return "none"

    def units_tier(self) -> str:
        """The evidence tier a claim resting on :meth:`units` may cite.

        Both sources are DECLARED, and neither is RTL: a kind read from a contract is not the same
        evidence as one extracted from the design, and it must not be reported as if it were. They are
        kept apart from each other too, because "a human wrote this into the residual for this
        analysis" and "the target ships this in its contract" are different claims, and a reader who
        wants to check one goes to a different file than a reader checking the other.
        """
        return {"residual": TIER_RESIDUAL, "contract": TIER_CONTRACT}.get(self.units_source(),
                                                                          TIER_NONE)

    def unit_kinds(self) -> tuple[str, ...]:
        seen: list[str] = []
        for u in self.units():
            k = u.get("kind")
            if k and k not in seen:
                seen.append(str(k))
        return tuple(seen)


def _residual_path(target: str) -> Path:
    """The residual side-input for ``target``.

    Resolved exactly where ``capability_manifests`` puts it -- ``<target_base>/contracts/
    residual.yaml`` -- via the shared :func:`~merlin.targetgen.rtl.facts.target_base`, so a target
    that moves brings its residual with it and nothing here holds a per-target path.
    """
    from merlin.targetgen.rtl.facts import target_base
    return target_base(target) / "contracts" / "residual.yaml"


def _read_residual(target: str) -> dict[str, Any]:
    import yaml
    p = _residual_path(target)
    if not p.is_file():
        return {}
    doc = yaml.safe_load(p.read_text(encoding="utf-8"))
    return dict(doc) if isinstance(doc, dict) else {}


def _read_contract(target: str) -> dict[str, Any]:
    """The target's own contract, or ``{}`` when it has none on this host.

    Resolved through the target registry rather than by path, so a target that lives out-of-tree (an
    injected reference package, ``MERLIN_TARGET_PATH``) brings its contract with it and nothing here
    holds a per-target location. An unresolvable target is an answer -- no contract, hence no base
    declaration -- never a raise: this is a side-input read on a profile call, and a profile call has
    no business failing because a registry lookup did.
    """
    try:
        from merlin.targetgen.target_registry import load_contract
        doc = load_contract(target)
    except Exception:                       # noqa: BLE001 -- no contract == no base declaration
        return {}
    return dict(doc) if isinstance(doc, Mapping) else {}


def _read_facts(target: str, *, allow_extraction: bool) -> dict[str, Any]:
    """The facts artifact for ``target``, or ``{}`` when none exists on this host.

    Deliberately does NOT trigger an extraction by default. ``load_facts`` regenerates a cold cache
    by running CIRCT over the design (~70 s and an external checkout), which a profile call has no
    business doing behind the caller's back; and an absent artifact is a real, reportable state
    ("uncached"), not a failure. Pass ``allow_extraction=True`` to accept the cost.
    """
    from merlin.targetgen.rtl import facts as _facts
    if allow_extraction:
        try:
            return dict(_facts.load_facts(target) or {})
        except Exception:                       # noqa: BLE001 -- absent facts are an answer
            return {}
    for path in (_facts.rtl_facts_path(target), _facts.target_base(target) / "contracts"
                 / "rtl_facts" / "facts.json"):
        try:
            if Path(path).is_file():
                return dict(json.loads(Path(path).read_text(encoding="utf-8")))
        except Exception:                       # noqa: BLE001 -- unreadable == absent, and said so
            continue
    return {}


def load_sources(target: str, *, facts: Mapping[str, Any] | None = None,
                 residual: Mapping[str, Any] | None = None,
                 contract: Mapping[str, Any] | None = None,
                 allow_extraction: bool = False) -> Sources:
    """Read the three sources for ``target``.

    ``facts`` / ``residual`` override the on-disk reads, which is how a caller supplies a target
    that is not on this host -- and how a test can DELETE one fact and check that the trait it
    grounded goes UNKNOWN rather than falling back to a default.

    ``residual`` also accepts a fully derived capability manifest: a manifest is a superset of the
    residual's fields (it is the residual with the facts layered on), so passing one lets the profile
    see units the residual never declared but the evidence reached.

    ``contract`` overrides the on-disk contract read the same way, and passing ``{}`` suppresses it
    entirely. That matters for a test that DELETES a declaration to check the trait resting on it goes
    UNKNOWN: without the override the base declaration on disk would quietly restore what the test
    removed, and the test would pass while proving nothing.
    """
    doc = dict(facts) if facts is not None else _read_facts(target, allow_extraction=allow_extraction)
    res = dict(residual) if residual is not None else _read_residual(target)
    con = dict(contract) if contract is not None else _read_contract(target)
    body = doc.get("facts") if isinstance(doc.get("facts"), Mapping) else {}
    present, missing = [], []
    (present if body else missing).append("rtl_facts")
    (present if res else missing).append("residual")
    (present if con else missing).append("contract")
    return Sources(target=target, facts=doc, residual=res, body=dict(body or {}),
                   present=tuple(present), missing=tuple(missing), contract=con)


def elaboration_of(sources: Sources) -> Elaboration:
    """Which elaboration these facts were extracted from.

    Prefers ``inputs.core_hw_mlir`` / ``core_hw_sha`` (the dialect the extractor actually READ) over
    the legacy ``hw_mlir`` / ``hw_sha`` pair, which on some artifacts names a path the extractor
    never opened and records ``"missing"`` for its digest. When only the legacy pair is present and
    its digest is absent or ``"missing"``, the elaboration is reported as NOT evidenced: a term may
    still be derived, but its validity says the elaboration is asserted rather than shown.
    """
    inputs = sources.facts.get("inputs") if isinstance(sources.facts.get("inputs"), Mapping) else {}
    inputs = dict(inputs or {})
    gen = sources.facts.get("generator") if isinstance(sources.facts.get("generator"), Mapping) else {}
    extractor = None
    if gen:
        extractor = f"{gen.get('name')} {gen.get('version')}".strip()
    if not inputs and not sources.body:
        return Elaboration(extractor=extractor, note="no facts artifact for this target on this host")

    dialect = inputs.get("core_hw_mlir") or inputs.get("hw_mlir")
    digest = inputs.get("core_hw_sha") or inputs.get("hw_sha")
    evidenced = bool(digest) and str(digest) not in ("missing", "n/a", "unknown")
    # A fact block may record the file it read even when ``inputs`` does not; that path is weaker
    # than a digest but it is the difference between naming the wrong input and naming none.
    read_path = None
    for itf in sources.interfaces().values():
        if itf.get("hw_source"):
            read_path = str(itf["hw_source"])
            break
    note = "" if evidenced else (
        f"inputs record dialect {dialect!r} with digest {digest!r}: the elaboration is ASSERTED, "
        "not evidenced -- a term derived from it cannot name the bytes it was read from")
    return Elaboration(dialect=(str(dialect) if dialect else None),
                       digest=(str(digest) if digest else None), extractor=extractor,
                       read_path=read_path, evidenced=evidenced, note=note)


def timing_walk(sources: Sources) -> TimingWalk:
    """The timing fact class for these sources, with absent / empty / present kept distinct."""
    if not sources.body:
        return TimingWalk(status="no_facts")
    if "timing" not in sources.body:
        # ABSENT, not empty: this artifact was produced by (or served from a cache written by) an
        # extraction that carried no timing block. The fact class exists; nobody looked for THIS
        # target. Never design around its absence and never synthesize it.
        return TimingWalk(status="uncached")
    rows = sources.body.get("timing") or []
    mods: dict[str, dict[str, Any]] = {}
    for rec in rows:
        if isinstance(rec, Mapping) and rec.get("module"):
            mods[str(rec["module"])] = dict(rec)
    if not mods:
        return TimingWalk(status="empty")
    resolved = sum(1 for r in mods.values() if r.get("pipeline_depth") is not None)
    return TimingWalk(status="present", modules=mods, resolved=resolved,
                      refused=len(mods) - resolved)


# --------------------------------------------------------------------------------------------
# Archetype -- the PRIOR
# --------------------------------------------------------------------------------------------

#: Which traits each dispatch mode makes worth asking about. A prior, nothing more: every trait is
#: derived for every target regardless, and this only marks the ones whose answer this shape of
#: machine turns on. Keyed on the DERIVED transport (:mod:`merlin.system.derive`), never on a name.
_QUESTIONS_FOR_DISPATCH: dict[str, tuple[str, ...]] = {
    # The host issues an instruction the device decodes: the device runs behind a queue, holds
    # configuration across host instructions, and moves its own operands.
    "host_instruction": ("host_dispatched_queue", "persistent_configuration_state", "explicit_dma",
                         "managed_scratchpad"),
    # The device fetches and decodes its own stream: the program itself carries the schedule, so
    # what matters is whether it can see its engines finish and whether they are separate at all.
    "device_native": ("self_hosted_program", "explicit_completion", "multiple_engine_groups",
                      "independent_engine_ports", "explicit_dma", "managed_scratchpad"),
    # No command ISA at all; a buffer is handed over.
    "command_buffer": ("explicit_dma", "managed_scratchpad", "persistent_configuration_state",
                       "explicit_completion"),
}

#: Which traits a compute-unit KIND makes worth asking about, on top of the dispatch questions.
_QUESTIONS_FOR_KIND: dict[str, tuple[str, ...]] = {
    "systolic": ("structural_pipeline_depth", "feedback_sequenced_units", "banked_memory"),
    "spatial": ("structural_pipeline_depth", "feedback_sequenced_units", "banked_memory"),
    "simt": ("multiple_engine_groups", "independent_engine_ports", "banked_memory"),
    "vector": ("banked_memory", "structural_pipeline_depth"),
    "scalar": ("structural_pipeline_depth",),
}


@dataclass(frozen=True)
class Archetype:
    """A PRIOR about the shape of a machine: how work reaches it, and what computes it.

    Both axes are derived -- the dispatch from the decode facts (an opcode set too wide for the
    host's co-processor funct field is a device that decodes its own stream), the datapath from the
    primary compute unit's KIND. ``label`` composes the two and is deliberately a description, not a
    name: two targets with the same label are the same *shape*, which licenses asking them the same
    questions and nothing else.
    """

    dispatch: str | None
    datapath_kind: str | None
    endpoint_kind: str | None
    questions: tuple[str, ...]
    evidence: dict[str, str] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return f"{self.dispatch or 'unknown-dispatch'}/{self.datapath_kind or 'unknown-datapath'}"

    def to_dict(self) -> dict[str, Any]:
        return {"label": self.label, "dispatch": self.dispatch,
                "datapath_kind": self.datapath_kind, "endpoint_kind": self.endpoint_kind,
                "questions": list(self.questions), "evidence": dict(self.evidence)}


def _endpoint_kind(sources: Sources) -> tuple[str | None, str, str]:
    """``(endpoint_kind, tier, evidence)`` -- how the target's work is delivered.

    The facts-derived classification is the one in
    :func:`merlin.targetgen.capability_manifests._endpoint_from_facts`: a decode table whose legal
    opcodes all fit the host co-processor's funct field is a host-decoded co-processor; any wider
    opcode means the device decodes its own stream. That derivation is reused rather than restated
    so the funct-width constant lives in exactly one place.
    """
    from merlin.targetgen import capability_manifests as _cm
    derived = _cm._endpoint_from_facts(sources.body)
    if derived:
        itf = sources.interfaces().get(_IFACE_DECODE, {})
        legal = list(itf.get("legal_funct") or [])
        detail = (f"{len(legal)} legal opcodes, widest 0x{max(legal):x}" if legal
                  else f"a {_IFACE_SELF_HOSTED} interface declaring its own encoding")
        return derived, TIER_FACTS, f"facts.interfaces: {detail} -> endpoint {derived}"
    declared = sources.residual.get("endpoint_kind")
    if declared:
        return str(declared), TIER_RESIDUAL, f"residual declares endpoint_kind={declared!r}"
    kinds = sources.unit_kinds()
    if kinds:
        try:
            from merlin.targetgen import families as _families
            prof = _families.family_profile(kinds[0])
        except Exception:                       # noqa: BLE001 -- unknown kind: no family default
            return None, TIER_NONE, f"no decode facts, no declared endpoint, unknown kind {kinds[0]!r}"
        return (prof.endpoint_kind_default, TIER_FAMILY,
                f"family default for compute-unit kind {kinds[0]!r}")
    return None, TIER_NONE, "no decode facts, no declared endpoint, no declared compute unit"


def archetype_of(sources: Sources) -> Archetype:
    """Derive the prior. Never a target name; never a gate."""
    # One map, not a second copy: the endpoint -> transport correspondence is already derived once
    # for the system model, and two answers to "how is this device reached" is how they drift.
    from merlin.system.derive import _TRANSPORT_FOR_ENDPOINT as _TRANSPORT

    endpoint, tier, evidence = _endpoint_kind(sources)
    dispatch = _TRANSPORT.get(endpoint or "")
    kinds = sources.unit_kinds()
    kind = kinds[0] if kinds else None
    questions: list[str] = []
    for name in (*_QUESTIONS_FOR_DISPATCH.get(dispatch or "", ()),
                 *_QUESTIONS_FOR_KIND.get(kind or "", ())):
        if name not in questions:
            questions.append(name)
    ev = {
        "endpoint_kind": f"{evidence} [{tier}]",
        "dispatch": (f"endpoint_kind={endpoint!r} -> transport {dispatch!r}" if dispatch
                     else f"endpoint_kind={endpoint!r} implies no distinct transport"),
        "datapath_kind": (f"the {sources.units_source()}'s compute_units declare kinds "
                          f"{list(kinds)}; the primary is {kind!r} [{sources.units_tier()}]"
                          if kinds else "no compute unit is declared in the residual or the contract"),
    }
    return Archetype(dispatch=dispatch, datapath_kind=kind, endpoint_kind=endpoint,
                     questions=tuple(questions), evidence=ev)


# --------------------------------------------------------------------------------------------
# Traits -- the ANSWERS
# --------------------------------------------------------------------------------------------


def _t_self_hosted_program(sources: Sources) -> tuple[Trait, str]:
    endpoint, tier, evidence = _endpoint_kind(sources)
    if endpoint is None:
        return Trait("self_hosted_program", None, evidence=evidence,
                     missing=("a decode table in facts.interfaces, or a declared endpoint_kind",)), TIER_NONE
    # ``external_backend`` is precisely "the device fetches and decodes its own instruction stream".
    satisfied = endpoint == "external_backend"
    if tier != TIER_FACTS:
        # A declaration can say which endpoint is emitted; it cannot establish the decode width that
        # makes the machine self-hosted. Report the weaker tier rather than borrowing the strong one.
        return Trait("self_hosted_program", None, evidence=evidence,
                     missing=("RTL decode facts (facts.interfaces.funct_decode_table)",)), tier
    return Trait("self_hosted_program", satisfied, evidence=evidence), TIER_FACTS


def _t_host_dispatched_queue(sources: Sources) -> tuple[Trait, str]:
    ifaces = sources.interfaces()
    if _IFACE_HOST_QUEUE in ifaces:
        ev = ifaces[_IFACE_HOST_QUEUE].get("evidence") or _IFACE_HOST_QUEUE
        return Trait("host_dispatched_queue", True,
                     evidence=f"facts.interfaces[{_IFACE_HOST_QUEUE}]: {ev}"), TIER_FACTS
    return Trait("host_dispatched_queue", None,
                 evidence=f"facts declare interfaces {sorted(ifaces) or 'none'}",
                 missing=("a host command-queue interface in facts.interfaces",)), TIER_NONE


def _t_explicit_dma(sources: Sources) -> tuple[Trait, str]:
    ifaces = sources.interfaces()
    if _IFACE_DMA in ifaces:
        ev = ifaces[_IFACE_DMA].get("evidence") or _IFACE_DMA
        return Trait("explicit_dma", True,
                     evidence=f"facts.interfaces[{_IFACE_DMA}]: {ev}"), TIER_FACTS
    return Trait("explicit_dma", None,
                 evidence=f"facts declare interfaces {sorted(ifaces) or 'none'}",
                 missing=("a data-movement interface in facts.interfaces",
                          "or DMA-roled mnemonics in the ISA model: a role census that cannot "
                          "separate an asynchronous channel move from a local operand load answers "
                          "this question about an engine it has never seen")), TIER_NONE


def _t_managed_scratchpad(sources: Sources) -> tuple[Trait, str]:
    mems = [m for m in sources.memories() if m.get("name")]
    if mems:
        named = ", ".join(f"{m['name']}" for m in mems)
        return Trait("managed_scratchpad", True,
                     evidence=f"facts.memories discovered {len(mems)} explicitly sized on-chip "
                              f"memories ({named})"), TIER_FACTS
    mm = sources.residual.get("memory_model") or {}
    declared = [k for k, v in mm.items() if v is True]
    if declared:
        return Trait("managed_scratchpad", True,
                     evidence=f"residual memory_model DECLARES {sorted(declared)} (intent, not "
                              "RTL-grounded: no memory is discovered in these facts)"), TIER_RESIDUAL
    return Trait("managed_scratchpad", None,
                 evidence="facts discovered no memories and the residual declares no memory model",
                 missing=("a memory in facts.memories, or a residual memory_model",)), TIER_NONE


def _t_banked_memory(sources: Sources) -> tuple[Trait, str]:
    mems = sources.memories()
    if not mems:
        return Trait("banked_memory", None,
                     evidence="no memory is discovered in these facts",
                     missing=("a memory fact carrying a bank count",)), TIER_NONE
    # THE BANK COUNT IS DERIVABLE, and this trait used to say it was not.
    #
    # The old reasoning -- "bytes and depth give a row WIDTH, which is not a bank count" -- was sound
    # while a row width was all anyone computed. It is not sound now. `depth` in these facts is the
    # PER-BANK row count, so once the row WIDTH is known (the compute array's column count times the
    # datapath element width) the total row count follows from the capacity, and total_rows / depth is
    # the number of banks. Reading `depth` as the whole address space instead under-addresses by exactly
    # that factor.
    #
    # Measured on the interlocked target here: scratchpad 262144 B at a 16 B row = 16384 rows over a
    # declared depth of 4096 -> 4 banks; accumulator 65536 B at a 64 B row = 1024 rows over 512 -> 2.
    # Both corroborate against the sibling SRAM groups mlc discovers in the RTL, so this is two
    # independent instruments agreeing rather than one arithmetic guess.
    #
    # Three states are kept apart, because the whole value of this trait is that a perf family gets
    # gated on it. The division must be EXACT -- a residue means the geometry is not what we think and
    # the honest answer is "we cannot tell", not a rounded bank count. And a single bank is FALSE, not
    # None: a memory genuinely has no banking to exploit, which is a fact about the target rather than
    # a hole in our evidence.
    have = sorted({k for m in mems for k in m})
    try:
        from merlin.targetgen.address_space import derive_address_space
        space = derive_address_space(sources.target)
    except Exception as e:                                     # noqa: BLE001 — unresolvable target
        return Trait("banked_memory", None,
                     evidence=f"facts.memories record {have}, but the address space could not be "
                              f"derived to turn a capacity into a bank count: "
                              f"{type(e).__name__}: {str(e)[:120]}",
                     missing=("a derivable row width (an array column count and a datapath element "
                              "width) so capacity/depth resolves to banks",)), TIER_NONE

    counted = {s.name: int(s.banks) for s in (getattr(space, "stores", ()) or ())
               if getattr(s, "banks", None) and not getattr(s, "bank_residue_rows", 0)}
    inexact = sorted(s.name for s in (getattr(space, "stores", ()) or ())
                     if getattr(s, "bank_residue_rows", 0))
    if not counted:
        undecidable = sorted(s.name for s in (getattr(space, "stores", ()) or ()))
        return Trait("banked_memory", None,
                     evidence=f"facts.memories record {have}; no store's bank count divides exactly "
                              f"(stores {undecidable or 'none'}"
                              + (f", inexact {inexact}" if inexact else "") + ")",
                     missing=("a store whose capacity divides exactly into rows of the derived width, "
                              "and those rows into its declared per-bank depth",)), TIER_NONE
    banked = {n: b for n, b in counted.items() if b > 1}
    detail = ", ".join(f"{n}={b}" for n, b in sorted(counted.items()))
    if banked:
        return Trait("banked_memory", True,
                     evidence=f"derived from facts.memories capacity and declared per-bank depth "
                              f"against the row width the compute array and datapath element imply: "
                              f"{detail} bank(s)"
                              + (f"; {inexact} did not divide exactly and are excluded" if inexact
                                 else "")), TIER_FACTS
    return Trait("banked_memory", False,
                 evidence=f"every derived store resolves to a single bank ({detail}), so there is no "
                          f"banking to exploit -- a fact about this target, not missing evidence"), \
        TIER_FACTS


def _t_persistent_configuration_state(sources: Sources) -> tuple[Trait, str]:
    enc = sources.residual.get("encoding") or {}
    subtypes = enc.get("config_subtype") or {}
    if subtypes:
        return Trait("persistent_configuration_state", True,
                     evidence=f"residual encoding DECLARES a configuration command with "
                              f"{len(subtypes)} sub-types {sorted(subtypes.values())}: state set by "
                              "one command and read by later ones"), TIER_RESIDUAL
    return Trait("persistent_configuration_state", None,
                 evidence=f"the residual's encoding block declares {sorted(enc) or 'nothing'}",
                 missing=("a configuration op class in the encoding ABI (config_subtype), or a "
                          "roled configuration mnemonic in the ISA model",)), TIER_NONE


def _engine_groups(sources: Sources) -> tuple[list[str], Trait, str]:
    """The engine groups this target's own sources evidence, and the movement trait behind them."""
    groups = list(sources.unit_kinds())
    dma, dma_tier = _t_explicit_dma(sources)
    return groups, dma, dma_tier


def _t_multiple_engine_groups(sources: Sources) -> tuple[Trait, str]:
    groups, dma, dma_tier = _engine_groups(sources)
    if dma.satisfied is True:
        return Trait("multiple_engine_groups", True,
                     evidence=f"{len(groups)} declared compute-unit kind(s) {groups} plus a "
                              f"data-movement engine grounded by {dma.evidence}"), TIER_FACTS
    if len(groups) >= 2:
        return Trait("multiple_engine_groups", True,
                     evidence=f"the {sources.units_source()} DECLARES {len(groups)} compute-unit "
                              f"kinds {groups}"), sources.units_tier()
    return Trait("multiple_engine_groups", None,
                 evidence=f"{len(groups)} declared compute-unit kind(s) {groups}; no second engine "
                          f"group is evidenced ({dma.evidence})",
                 missing=("a second engine group: a second declared compute unit, or a movement "
                          "engine in facts.interfaces",)), TIER_NONE


def _t_independent_engine_ports(sources: Sources) -> tuple[Trait, str]:
    groups, dma, _ = _engine_groups(sources)
    if groups and dma.satisfied is True:
        return Trait("independent_engine_ports", True,
                     evidence=f"a compute datapath ({groups[0]}) and a data-movement engine "
                              f"({dma.evidence}) are different kinds of unit reached through "
                              "different interfaces, so they cannot be one issue port"), TIER_FACTS
    if len(groups) >= 2:
        return Trait("independent_engine_ports", None,
                     evidence=f"the {sources.units_source()} declares {len(groups)} unit kinds "
                              f"{groups}, but declaring two units is not observing two ports",
                     missing=("evidence that two engines issue independently (a movement interface, "
                              "or a workload showing both carrying work in one run)",)), TIER_NONE
    return Trait("independent_engine_ports", None,
                 evidence=f"declared kinds {groups}; {dma.evidence}",
                 missing=("a second engine group at all",)), TIER_NONE


def _t_explicit_completion(sources: Sources) -> tuple[Trait, str]:
    """Whether a unit SIGNALS that it finished, per engine.

    Still not established for any target here, and the reason is worth stating precisely rather than
    as "no completion signal": on at least one target the signal EXISTS IN THE RTL and the fact
    extractor does not look for it. Its three command controllers each carry
    ``val completed = Decoupled(UInt(log2Up(reservation_station_entries).W))`` driven from a command
    tracker and tagged with the reservation-station id -- which is exactly a per-engine completion
    port, and exactly what a scheduling family needs in order to price an issue against a wait.

    So the honest state is "our extractor cannot see it", not "the hardware does not have it", and the
    two must not read alike. ``facts.interfaces`` records a NAME and an evidence string per interface
    and no port list at all, so nothing in the bundle could carry this even when the RTL does. Naming
    the shape of the missing extraction is the difference between a blocker somebody can close and one
    that reads as a property of the machine.
    """
    ifaces = sorted(sources.interfaces())
    # READ THE PORTS. `facts.interfaces` carries a name and an evidence string and no port list, so
    # this trait used to conclude from their absence that the target does not signal completion -- a
    # statement about our extractor delivered as a statement about the hardware. The elaborated FIRRTL
    # the facts THEMSELVES name is the authority for what this configuration emitted, and a decoupled
    # `completed` channel is what a scheduling family needs in order to price an issue against a wait.
    #
    # Decoupled is the bar, deliberately. A bare `completed` wire says a unit finished SOMETHING; a
    # ready/valid handshake carries a tag and lets a consumer attribute the completion to a command,
    # which is the difference between knowing a unit is done and knowing WHICH work it finished.
    try:
        from merlin.targetgen.rtl.ports import port_facts
        pf = port_facts(sources.target, fields=("completed",))
    except Exception as e:                                     # noqa: BLE001
        pf = {"status": "unavailable", "why": f"{type(e).__name__}: {str(e)[:120]}", "fields": {}}
    if pf.get("status") == "derived":
        got = (pf.get("fields") or {}).get("completed") or {}
        handshaken = list(got.get("decoupled") or ())
        if handshaken:
            return Trait("explicit_completion", True,
                         evidence=f"{len(handshaken)} module(s) in this target's own elaborated FIRRTL "
                                  f"expose a DECOUPLED completion channel ({', '.join(handshaken)}); "
                                  f"read from the .fir the facts name, so it describes what this "
                                  f"configuration emitted rather than what the generator can emit"), \
                TIER_FACTS
        named = list(got.get("modules") or ())
        # ⚠️ THE DECIDING RUNG RAN, SO SILENCE IS ABSENT -- not UNKNOWN. This returned None either way,
        # which is the collapse `capability_discovery._FAMILY_DECIDED_BY` exists to prevent in the
        # other direction: a trait nobody could measure and a trait measured absent read alike, and
        # only the second is a result a claim can be refuted on.
        #
        # Measured on atlas once its CIRCT hw-dialect elaboration became readable: 84 modules, zero
        # exposing a completion field. Its engines expose per-engine BUSY ports (vecBusy, vloadBusy,
        # xluBusy, ...) and its decoupled channels are TileLink a/d, so the machine says a unit is
        # busy and never which command finished -- which is exactly the bar below, not met.
        n_read = int(pf.get("n_modules") or 0)
        dialect = pf.get("dialect") or "fir"
        if not named:
            return Trait("explicit_completion", False,
                         evidence=f"{n_read} module(s) of this target's own elaboration were read "
                                  f"({dialect} dialect) and NONE exposes a completion field, so no "
                                  f"engine signals that its work finished; the elaboration is the "
                                  f"authority for what this configuration emitted, and it was read "
                                  f"rather than missing"), TIER_FACTS
        return Trait("explicit_completion", False,
                     evidence=f"{len(named)} module(s) of {n_read} read ({dialect} dialect) expose a "
                              f"completion field {named}, none of them a ready/valid handshake, so a "
                              f"completion cannot be attributed to a particular command -- the bar "
                              f"this trait sets, measured and not met",
                     missing=("a decoupled (ready/valid) completion channel, or an activity source "
                              "declaring completion_observable "
                              "(merlin.perf.decompose.ActivitySource)",)), TIER_FACTS
    return Trait("explicit_completion", None,
                 evidence=f"the facts declare interfaces {ifaces or 'none'} and carry no port list, and "
                          f"the elaborated FIRRTL could not be read to recover one "
                          f"({pf.get('why', 'unavailable')}). UNKNOWN, not absent",
                 missing=("the elaborated .fir this target's facts name, so module ports can be read; "
                          "or an activity source declaring completion_observable "
                          "(merlin.perf.decompose.ActivitySource) -- headroom.concurrency_traits "
                          "never defaults this",)), TIER_NONE


def _t_multiple_operand_encodings(sources: Sources) -> tuple[Trait, str]:
    """Can this target compute the SAME contraction in more than one operand encoding?

    The gate an ``L6_global`` family needs. A packing/encoding/layout lever only exists where there is
    a choice to make: on a single-format machine the question is not hard, it is absent, and the honest
    answer is a REFUTATION rather than an unestablished trait -- a refuted gate leaves an
    evidence-bearing skip that says "this target has one format", which is informative, where an
    unestablished one says only "we could not tell".

    Derived from the contraction family's declared dtypes, which is where the choice actually lives.
    """
    try:
        from merlin.targetgen.eligibility import capability_map_for_target
        cap = (capability_map_for_target(sources.target) or {}).get("contraction")
    except Exception as exc:                       # noqa: BLE001 -- an unresolvable map is not a format
        return Trait("multiple_operand_encodings", None,
                     evidence=f"no capability map resolved ({type(exc).__name__}: {exc})",
                     missing=("a capability manifest declaring the contraction family's dtypes",),
                     ), TIER_NONE
    if cap is None:
        return Trait("multiple_operand_encodings", None,
                     evidence="the capability map declares no contraction family, so there is no "
                              "operand encoding to choose between",
                     missing=("a declared contraction family",)), TIER_NONE
    dtypes = tuple(getattr(cap, "dtypes", ()) or ())
    if len(dtypes) >= 2:
        return Trait("multiple_operand_encodings", True,
                     evidence=f"the contraction family declares {len(dtypes)} operand encoding(s) "
                              f"{sorted(dtypes)}, so an encoding choice exists"), TIER_CONTRACT
    return Trait("multiple_operand_encodings", False,
                 evidence=f"the contraction family declares {sorted(dtypes) or 'no'} operand "
                          f"encoding(s); with fewer than two there is no encoding to choose, so an "
                          f"encoding lever cannot be exercised here"), TIER_CONTRACT


def _t_structural_pipeline_depth(sources: Sources) -> tuple[Trait, str]:
    walk = timing_walk(sources)
    if walk.status == "present":
        satisfied = walk.resolved > 0
        return Trait("structural_pipeline_depth", satisfied,
                     evidence=f"the RTL timing walk resolved a finite pipeline_depth for "
                              f"{walk.resolved} of {len(walk.modules)} walked hw.modules and "
                              f"refused {walk.refused} (a MODULE COUNT, not coverage: the walk "
                              "resolves combinational leaves and refuses sequenced units, so it is "
                              "biased away from where the cycles are)"), TIER_FACTS
    if walk.status == "empty":
        return Trait("structural_pipeline_depth", False,
                     evidence="the RTL timing walk ran and reported no modules"), TIER_FACTS
    return Trait("structural_pipeline_depth", None,
                 evidence=f"the timing fact class is {walk.status} for this target",
                 missing=("a facts artifact carrying a timing block -- UNCACHED is not the same as "
                          "absent, and this fact class exists: re-extract to answer",)), TIER_NONE


def _t_feedback_sequenced_units(sources: Sources) -> tuple[Trait, str]:
    walk = timing_walk(sources)
    if walk.status != "present":
        return Trait("feedback_sequenced_units", None,
                     evidence=f"the timing fact class is {walk.status} for this target",
                     missing=("a facts artifact carrying a timing block",)), TIER_NONE
    if walk.refused:
        sample = sorted(m for m, r in walk.modules.items() if r.get("pipeline_depth") is None)[:5]
        return Trait("feedback_sequenced_units", True,
                     evidence=f"{walk.refused} of {len(walk.modules)} modules reach an output "
                              f"through feedback (e.g. {sample}): their latency is a function of "
                              "state and operands, so it must come from the sequencer's own limits "
                              "or from measurement, never from a wiring depth"), TIER_FACTS
    return Trait("feedback_sequenced_units", False,
                 evidence=f"all {len(walk.modules)} walked modules resolved a finite depth"), TIER_FACTS


def _t_multiple_operand_encodings(sources: Sources) -> tuple[Trait, str]:
    """Can this target compute the SAME contraction in more than one operand encoding?

    The gate an ``L6_global`` family needs. A packing/encoding/layout lever only exists where there is
    a choice to make: on a single-format machine the question is not hard, it is absent, and the honest
    answer is a REFUTATION rather than an unestablished trait -- a refuted gate leaves an
    evidence-bearing skip that says "this target has one format", which is informative, where an
    unestablished one says only "we could not tell".

    Derived from the contraction family's declared dtypes, which is where the choice actually lives.
    """
    try:
        from merlin.targetgen.eligibility import capability_map_for_target
        cap = (capability_map_for_target(sources.target) or {}).get("contraction")
    except Exception as exc:                       # noqa: BLE001 -- an unresolvable map is not a format
        return Trait("multiple_operand_encodings", None,
                     evidence=f"no capability map resolved ({type(exc).__name__}: {exc})",
                     missing=("a capability manifest declaring the contraction family's dtypes",),
                     ), TIER_NONE
    if cap is None:
        return Trait("multiple_operand_encodings", None,
                     evidence="the capability map declares no contraction family, so there is no "
                              "operand encoding to choose between",
                     missing=("a declared contraction family",)), TIER_NONE
    dtypes = tuple(getattr(cap, "dtypes", ()) or ())
    if len(dtypes) >= 2:
        return Trait("multiple_operand_encodings", True,
                     evidence=f"the contraction family declares {len(dtypes)} operand encoding(s) "
                              f"{sorted(dtypes)}, so an encoding choice exists"), TIER_CONTRACT
    return Trait("multiple_operand_encodings", False,
                 evidence=f"the contraction family declares {sorted(dtypes) or 'no'} operand "
                          f"encoding(s); with fewer than two there is no encoding to choose, so an "
                          f"encoding lever cannot be exercised here"), TIER_CONTRACT


_DERIVERS = {
    "multiple_operand_encodings": _t_multiple_operand_encodings,
    "self_hosted_program": _t_self_hosted_program,
    "host_dispatched_queue": _t_host_dispatched_queue,
    "explicit_dma": _t_explicit_dma,
    "managed_scratchpad": _t_managed_scratchpad,
    "banked_memory": _t_banked_memory,
    "persistent_configuration_state": _t_persistent_configuration_state,
    "multiple_engine_groups": _t_multiple_engine_groups,
    "independent_engine_ports": _t_independent_engine_ports,
    "explicit_completion": _t_explicit_completion,
    "structural_pipeline_depth": _t_structural_pipeline_depth,
    "multiple_operand_encodings": _t_multiple_operand_encodings,
    "feedback_sequenced_units": _t_feedback_sequenced_units,
}


# --------------------------------------------------------------------------------------------
# The profile
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class TargetProfile:
    """One target's archetype (the prior), its traits (the answers), and where they came from."""

    target: str
    archetype: Archetype
    traits: dict[str, Trait]
    trait_tier: dict[str, str]
    elaboration: Elaboration
    timing: TimingWalk
    sources: Sources

    def trait(self, name: str) -> Trait:
        try:
            return self.traits[name]
        except KeyError:
            raise KeyError(f"{self.target}: no trait {name!r}; derived traits are "
                           f"{sorted(self.traits)}") from None

    def has(self, name: str) -> bool | None:
        """Tri-state. ``None`` means not established and must not be read as ``False``."""
        return self.trait(name).satisfied

    def satisfied(self) -> tuple[str, ...]:
        return tuple(n for n in TRAITS if self.traits[n].satisfied is True)

    def refuted(self) -> tuple[str, ...]:
        return tuple(n for n in TRAITS if self.traits[n].satisfied is False)

    def unestablished(self) -> tuple[str, ...]:
        return tuple(n for n in TRAITS if self.traits[n].satisfied is None)

    def worklist(self) -> tuple[tuple[str, tuple[str, ...]], ...]:
        """The traits the ARCHETYPE says matter and the evidence has not settled, with what is
        missing. This is the measurement backlog for this target, computed rather than chosen."""
        return tuple((n, self.traits[n].missing) for n in self.archetype.questions
                     if n in self.traits and self.traits[n].satisfied is None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "archetype": self.archetype.to_dict(),
            "traits": {n: {"satisfied": self.traits[n].satisfied,
                           "tier": self.trait_tier.get(n, TIER_NONE),
                           "evidence": self.traits[n].evidence,
                           "missing": list(self.traits[n].missing)} for n in TRAITS},
            "elaboration": self.elaboration.to_dict(),
            "timing": self.timing.to_dict(),
            "sources_present": list(self.sources.present),
            "sources_missing": list(self.sources.missing),
            "worklist": [{"trait": n, "missing": list(m)} for n, m in self.worklist()],
        }


def derive_profile(target: str, *, facts: Mapping[str, Any] | None = None,
                   residual: Mapping[str, Any] | None = None,
                   contract: Mapping[str, Any] | None = None,
                   allow_extraction: bool = False) -> TargetProfile:
    """Derive ``target``'s profile from RTL facts + family defaults + its residual.

    The same code path for every target. What differs between two targets is what their own sources
    say, which is the whole point: a profile that had to be edited to onboard a second machine would
    be a hand-written description wearing a deriver's clothes.
    """
    src = load_sources(target, facts=facts, residual=residual, contract=contract,
                       allow_extraction=allow_extraction)
    traits: dict[str, Trait] = {}
    tiers: dict[str, str] = {}
    for name in TRAITS:
        trait, tier = _DERIVERS[name](src)
        traits[name], tiers[name] = trait, tier
    return TargetProfile(target=target, archetype=archetype_of(src), traits=traits,
                         trait_tier=tiers, elaboration=elaboration_of(src),
                         timing=timing_walk(src), sources=src)


def profile_table(profiles: Sequence[TargetProfile]) -> str:
    """A side-by-side trait table for several targets -- the anti-overfit result, readable.

    ``+`` satisfied, ``-`` refuted, ``?`` not established. The tier letter says what settled it
    (``f`` RTL facts, ``r`` residual declaration, ``c`` contract declaration, ``F`` family default),
    so a trait that is True because somebody wrote it down never reads like one the RTL grounded.
    """
    tier_mark = {TIER_FACTS: "f", TIER_RESIDUAL: "r", TIER_CONTRACT: "c", TIER_FAMILY: "F",
                 TIER_NONE: " "}
    mark = {True: "+", False: "-", None: "?"}
    width = max([len(t) for t in TRAITS] + [len("trait")])
    head = "trait".ljust(width) + "".join(f"  {p.target:>18}" for p in profiles)
    rows = [head, "-" * len(head)]
    for name in TRAITS:
        cells = "".join(f"  {mark[p.traits[name].satisfied] + tier_mark[p.trait_tier.get(name, TIER_NONE)]:>18}"
                        for p in profiles)
        rows.append(name.ljust(width) + cells)
    rows.append("")
    rows.append("archetype".ljust(width) + "".join(f"  {p.archetype.label:>18}" for p in profiles))
    return "\n".join(rows)
