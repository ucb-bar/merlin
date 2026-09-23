"""The independent eligibility oracle — the ARR *denominator*.

Given a region of computation (its semantic family, operand dtypes, shape, layout) and a target's
**declared hardware capability** (the ``semantic_capabilities`` in its contract, folded by
:func:`merlin.targetgen.compute_units.semantic_capability_map`), decide whether the *hardware is
capable of executing that region* — independent of whether the generated compiler currently has a
lowering for it.

This is deliberately a **pure predicate over the capability map**: it never imports
:mod:`merlin.targetgen.routing` and never consults a lowering. Acceleratable Region Recall compares
what the compiler actually accelerated (routing / the coverage certificate = the *numerator*) against
what this oracle says the hardware *could* accelerate (the *denominator*). If this module and routing
shared a code path, the ratio would be trivially 1.0 and would measure nothing — so the separation is
the whole point (see :mod:`merlin.targetgen.compute_units.SemanticCapability`).

Fail-closed: a region whose family cannot be recognized is reported **ineligible with a reason**, never
silently assumed eligible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from merlin.common import quant_formats as qf
from merlin.targetgen import semantic_families as sf
from merlin.targetgen.compute_units import SemanticCapability


@dataclass(frozen=True)
class RegionDescriptor:
    """A single region of computation, described structurally (no lowering assumed).

    ``family`` may be omitted and is then resolved from ``op`` via
    :mod:`merlin.targetgen.semantic_families`. ``in_dtype``/``weight_dtype`` are quant-format names.
    """

    source: str = ""
    op: str | None = None
    family: str | None = None
    in_dtype: str | None = None
    weight_dtype: str | None = None
    m: int | None = None
    k: int | None = None
    n: int | None = None
    rank: int | None = None
    batch: int = 1
    layout: str | None = None
    #: Require the region to run on a specific compute-unit KIND (a ``compute_units.KINDS`` token).
    #: None == "any engine will do", which is what almost every region means; a caller sets this only
    #: when it is asking a narrower question ("can the ARRAY run this?", distinct from "can the target
    #: run this?"). A hybrid is the only place the two answers differ, which is why the axis exists.
    engine: str | None = None
    #: The op's OWN declared attributes, verbatim and unfiltered. Eligibility does not read this -- a
    #: convolution's padding does not decide whether the hardware can run it -- but the REQUIREMENT does,
    #: and the requirement is derived from these descriptors. Without it a conv's padding, stride and
    #: dilation never reach the corpus, which is how every convolution capsule came to declare the same
    #: geometry while the builder had accepted all three parameters all along. Kept unfiltered on
    #: purpose: an attribute this module cannot interpret is still evidence about the capture, and a
    #: known-key filter is a per-target fact in disguise.
    config: dict | None = None
    #: How the region's scale varies over its result, in the readout facet's vocabulary
    #: (``readout_facet.GRANULARITIES``); ``None`` when the region applies no scale or the capture
    #: does not say. A requantization is one family at every granularity, and whether a unit can
    #: absorb it is decided by exactly this: a store path holding one scale per command takes a
    #: per-tensor requantization and cannot take a per-channel one.
    scale_granularity: str | None = None

    def resolved_family(self) -> str | None:
        """The CANONICAL family, resolving a capture's own coarse tag rather than trusting it.

        ``family`` arrives from two kinds of caller and they do not speak the same vocabulary: a
        synthesizer passes a canonical family (``elementwise_map``), a capture-driven census passes
        ``prov.family`` verbatim (``quantize``, ``minmax``, ``pool``). This used to return whichever
        string it was handed, so a capture tag reached the capability check unchanged and was refused
        as ``undeclared_family`` -- a refusal that reads as "the hardware does not have this" when
        what happened is that nobody translated the word. Measured on a captured ResNet-50: 115 of
        116 host regions were refused this way, under three tags the taxonomy could translate.

        Canonical in, canonical out -- an already-canonical family is returned untouched, so no
        existing caller changes. A tag neither table recognises is PRESERVED rather than dropped, so
        the refusal still names the string the capture actually used.
        """
        raw = (self.family or "").strip()
        if not raw:
            return sf.from_op(self.op)
        if sf.is_family(raw):
            return raw
        return sf.from_prov(raw, self.op) or raw


@dataclass(frozen=True)
class EligibilityVerdict:
    eligible: bool
    family: str | None
    reason: str
    #: The evidence could not DECIDE this family — distinct from "the hardware cannot do it". An
    #: undetermined region belongs in neither the ARR numerator nor its denominator: counting it
    #: ineligible would flatter recall by shrinking the denominator, and counting it eligible would
    #: deflate recall by demanding work the hardware may not support. It is reported as unmeasured.
    undetermined: bool = False
    #: Which compute-unit KINDS can execute this region, and which declared units they are.
    #:
    #: A verdict that says only "eligible" leaves the caller to guess where the work lands, and the
    #: folded capability map used to make that guess impossible: it recorded that the target supports
    #: elementwise_map, not that the VPU does and the MXU does not. Empty on an ineligible verdict and
    #: on a contract that predates the attribution.
    engines: tuple[str, ...] = ()
    units: tuple[str, ...] = ()
    #: WHICH check refused, as one of :data:`REFUSALS`; ``None`` on an eligible verdict. ``reason``
    #: is for a reader. This is for a consumer that must act on the cause (a placement census
    #: attributing a host placement to its owner) without parsing a sentence to find it.
    refusal: str | None = None


#: Every way :func:`is_eligible` refuses, in the order it checks. Closed: a new check adds a name.
REFUSALS: tuple[str, ...] = (
    "unrecognized_family",
    "undetermined_family",
    "undeclared_family",
    "input_dtype",
    "weight_dtype",
    "rank",
    "batch",
    "layout",
    "engine",
    "fused_only",
    "scale_granularity",
    "scale_granularity_unknown",
)


#: Whether an EMPTY declaration of a shape axis narrows eligibility, per axis. The two axes this
#: module checks do not agree, and the disagreement is in the code right below: ``_dtype_ok`` returns
#: False for an empty ``allowed`` (so a capability declaring no dtypes admits nothing), while the rank
#: check is guarded by ``if c.ranks`` (so a capability declaring no ranks admits every rank).
#:
#: This is stated HERE, beside the checks that implement it, because the capability AUDITOR has to
#: reason about the same question and must not restate it: an auditor that assumes "omitted == the
#: hardware has it but the contract hides it" reports an under-declaration for an axis that in fact
#: excludes nothing, and the remedy it implies -- declare the evidenced value -- would NARROW a
#: previously unconstrained axis and cause the very denominator loss the audit exists to catch.
#: Consumed by :func:`merlin.targetgen.capability_derive._axis_findings`; agreement between this
#: table and :func:`is_eligible` is asserted by test, not assumed.
#: ``engines`` is False for the same reason ``ranks`` is: the check below is guarded by ``if
#: c.engines``, so a capability naming no engine admits every engine. Declaring one on a contract that
#: names none would NARROW an axis that currently excludes nothing -- the exact denominator loss that
#: the mx_gemmini rank bug was. The attribution is derived from compute_units rather than authored, so
#: in practice this is empty only for a target whose capability predates the attribution.
_EMPTY_IS_NARROWING = {"dtypes": True, "ranks": False, "layouts": False, "engines": False}


def empty_declaration_is_narrowing(axis: str) -> bool:
    """Does declaring NOTHING on ``axis`` exclude regions (True) or constrain nothing (False)?

    Raises for an axis this module does not check, so a new shape axis cannot be audited against a
    semantics nobody wrote down.
    """
    try:
        return _EMPTY_IS_NARROWING[axis]
    except KeyError:
        raise KeyError(
            f"no declared empty-set semantics for shape axis {axis!r}; add it beside the "
            f"check in is_eligible that implements it"
        ) from None


def _dtype_ok(want: str | None, allowed: tuple[str, ...]) -> bool:
    """Is format ``want`` covered by ``allowed`` (registry-name/alias aware)? ``None`` want == n/a."""
    if want is None:
        return True
    if not allowed:
        return False
    if want in allowed:
        return True
    if qf.has(want):
        wn = qf.get(want).name
        for a in allowed:
            if qf.has(a) and qf.get(a).name == wn:
                return True
    return False


def _family_support(family: str, cap_map: dict[str, SemanticCapability]):
    """Return the caps to check for ``family``: the direct capability if declared, else the composite's
    primitive capabilities if ALL of them are declared, else ``(None, None)`` (unsupported).

    Returns ``(caps, how)`` where ``how`` ∈ {"direct", "primitives", None}.
    """
    if family in cap_map:
        return [cap_map[family]], "direct"
    prims = sf.primitives_of(family)
    if prims and family not in sf.PRIMITIVES and all(p in cap_map for p in prims):
        return [cap_map[p] for p in prims], "primitives"
    return None, None


def is_eligible(
    region: RegionDescriptor,
    cap_map: dict[str, SemanticCapability],
    *,
    undetermined: "frozenset[str] | tuple[str, ...] | None" = None,
    providers: "dict[str, tuple[tuple[str, str], ...]] | None" = None,
    fused_with: "frozenset[str] | tuple[str, ...] | None" = None,
    readout: "Any | None" = None,
) -> EligibilityVerdict:
    """Can the hardware described by ``cap_map`` execute ``region``? Pure declarative check.

    ``providers`` is :func:`compute_units.semantic_engine_map` for the same units. It supplies the
    declared UNIT NAMES behind each family; the engine KINDS ride on the capability itself and need no
    such argument. Optional because the verdict is well defined without it -- naming the engine is the
    load-bearing part, naming the unit is a convenience for a report.

    ``fused_with`` names the families that PRODUCE this region's operands in its graph context, when
    the caller knows them. A capability declared ``composed_with`` exists only attached to one of
    those producers, so it admits a region only when one is actually there. ``None`` is "asked
    standalone", and standalone such a capability admits nothing -- whether the region named the
    family directly or reached it as a primitive of a composite. The second case used to slip
    through: a normalization is a reduction plus an elementwise map, both declared fused-only
    behind a contraction, and with no contraction anywhere in it the composite was still scored
    eligible while the router, asked the same question, refused it. One capsule was unwinnable for
    every arm on that disagreement, and every model's eligible set was inflated by its norms and
    softmaxes.

    ``readout`` is the unit's derived readout facet (:mod:`merlin.targetgen.readout_facet`). A
    region that carries a ``scale_granularity`` is admitted only at a granularity the readout was
    derived to hold. A granularity the facet could not derive is not a refusal by the hardware: the
    verdict is flagged ``undetermined`` so it is reported as unmeasured rather than scored.

    ``undetermined`` names families for which no evidence source could reach a verdict (see
    :mod:`merlin.targetgen.capability_derive`). Such a family is NOT silently treated as unsupported:
    that would shrink the ARR denominator and flatter recall. It returns a verdict flagged
    ``undetermined`` so the caller can report it as unmeasured instead of scoring it either way.
    """
    family = region.resolved_family()
    if family is None:
        return EligibilityVerdict(
            False, None, "unrecognized semantic family (fail-closed)", refusal="unrecognized_family"
        )
    caps, how = _family_support(family, cap_map)
    if how is None:
        if undetermined and family in undetermined:
            return EligibilityVerdict(
                False,
                family,
                f"UNDETERMINED: no evidence source could decide family {family!r} for this target",
                undetermined=True,
                refusal="undetermined_family",
            )
        return EligibilityVerdict(
            False, family, f"target declares no capability for family {family!r}", refusal="undeclared_family"
        )
    for c in caps:
        if not _dtype_ok(region.in_dtype, c.dtypes):
            return EligibilityVerdict(
                False,
                family,
                f"input dtype {region.in_dtype!r} not in {c.family} formats {list(c.dtypes)}",
                refusal="input_dtype",
            )
        if region.weight_dtype is not None and not _dtype_ok(region.weight_dtype, c.dtypes):
            return EligibilityVerdict(
                False,
                family,
                f"weight dtype {region.weight_dtype!r} not supported by {c.family}",
                refusal="weight_dtype",
            )
        if c.ranks and region.rank is not None and region.rank not in c.ranks:
            return EligibilityVerdict(
                False, family, f"rank {region.rank} not in {c.family} legal ranks {list(c.ranks)}", refusal="rank"
            )
        if region.batch > 1 and not c.batch:
            return EligibilityVerdict(
                False,
                family,
                f"batched region (batch={region.batch}) but {c.family} declares batch=false",
                refusal="batch",
            )
        if c.layouts and region.layout is not None and region.layout not in c.layouts:
            return EligibilityVerdict(
                False,
                family,
                f"layout {region.layout!r} not in {c.family} legal layouts {list(c.layouts)}",
                refusal="layout",
            )
        if c.engines and region.engine is not None and region.engine not in c.engines:
            return EligibilityVerdict(
                False,
                family,
                f"engine {region.engine!r} does not provide {c.family} on this target; declared on {list(c.engines)}",
                refusal="engine",
            )
        # A capability available only FUSED admits a region only when one of the producers it
        # attaches to is present. Being a primitive of a composite is not such a producer: the
        # composite names what the region computes, not what feeds it.
        if c.composed_with and not any(p in (fused_with or ()) for p in c.composed_with):
            if how == "direct":
                why = f"{c.family} is available only fused with {list(c.composed_with)} on this target, not standalone"
            else:
                why = (
                    f"{family} needs {c.family}, which this target provides only fused with "
                    f"{list(c.composed_with)}; no such producer feeds this region"
                )
            return EligibilityVerdict(False, family, why, refusal="fused_only")
    if region.scale_granularity is not None and readout is not None:
        admitted = readout.admits_granularity(region.scale_granularity)
        if admitted is None:
            return EligibilityVerdict(
                False,
                family,
                f"UNDETERMINED: the readout's scale granularity is not derived "
                f"({readout.unknown.get('scale_granularities') or readout.unknown.get('scale_granularities_finer') or 'no reason recorded'})",
                undetermined=True,
                refusal="scale_granularity_unknown",
            )
        if not admitted:
            return EligibilityVerdict(
                False,
                family,
                f"the region scales per {region.scale_granularity}; this readout holds scales per "
                f"{list(readout.scale_granularities)}",
                refusal="scale_granularity",
            )
    engines = tuple(dict.fromkeys(e for c in caps for e in c.engines))
    units = tuple(dict.fromkeys(n for c in caps for n, _ in (providers or {}).get(c.family, ())))
    attached = sorted({p for c in caps for p in c.composed_with if p in (fused_with or ())})
    how_text = f"{how}, attached to {attached}" if attached else how
    return EligibilityVerdict(True, family, f"eligible ({how_text})", engines=engines, units=units)


# --- convenience: build the capability map from a contract / named target ---------------------------


def capability_map_from_contract(contract: dict) -> dict[str, SemanticCapability]:
    """Fold a contract's ``compute_units`` into the ``family -> SemanticCapability`` denominator map."""
    from merlin.targetgen import compute_units as cu

    return cu.semantic_capability_map(cu.compute_units(contract))


def providers_from_contract(contract: dict) -> dict[str, tuple[tuple[str, str], ...]]:
    """``family -> ((unit_name, kind), ...)`` for a contract — the ``providers`` argument above."""
    from merlin.targetgen import compute_units as cu

    return cu.semantic_engine_map(cu.compute_units(contract))


def providers_for_target(target_name: str) -> dict[str, tuple[tuple[str, str], ...]]:
    from merlin.targetgen import target_registry as tr

    return providers_from_contract(tr.load_contract(target_name))


def capability_map_for_target(target_name: str) -> dict[str, SemanticCapability]:
    """The declared hardware semantic-capability map for a named target (its contract's
    ``semantic_capabilities``). Loads only the DECLARED contract — never routing/lowering."""
    from merlin.targetgen import target_registry as tr

    return capability_map_from_contract(tr.load_contract(target_name))


def undetermined_families_from_contract(contract: dict) -> frozenset[str]:
    """Families the contract records as UNDECIDABLE from the evidence available when it was derived
    (``semantic_capabilities_unknown``), written by :mod:`merlin.targetgen.capability_derive`.

    Absent key == nothing undetermined, which is the honest reading for a contract that predates the
    deriver: it made no claim either way, so nothing is excused from the denominator."""
    out = set()
    for entry in contract.get("semantic_capabilities_unknown") or []:
        fam = entry.get("family") if isinstance(entry, dict) else entry
        if fam:
            out.add(str(fam))
    return frozenset(out)


def undetermined_families_for_target(target_name: str) -> frozenset[str]:
    """:func:`undetermined_families_from_contract` for a named target."""
    from merlin.targetgen import target_registry as tr

    try:
        return undetermined_families_from_contract(tr.load_contract(target_name))
    except Exception:  # noqa: BLE001 — no resolvable contract: nothing is excused
        return frozenset()
