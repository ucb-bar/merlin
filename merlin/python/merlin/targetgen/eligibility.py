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

    def resolved_family(self) -> str | None:
        return self.family or sf.from_op(self.op)


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


def is_eligible(region: RegionDescriptor, cap_map: dict[str, SemanticCapability],
                *, undetermined: "frozenset[str] | tuple[str, ...] | None" = None) -> EligibilityVerdict:
    """Can the hardware described by ``cap_map`` execute ``region``? Pure declarative check.

    ``undetermined`` names families for which no evidence source could reach a verdict (see
    :mod:`merlin.targetgen.capability_derive`). Such a family is NOT silently treated as unsupported:
    that would shrink the ARR denominator and flatter recall. It returns a verdict flagged
    ``undetermined`` so the caller can report it as unmeasured instead of scoring it either way.
    """
    family = region.resolved_family()
    if family is None:
        return EligibilityVerdict(False, None, "unrecognized semantic family (fail-closed)")
    caps, how = _family_support(family, cap_map)
    if how is None:
        if undetermined and family in undetermined:
            return EligibilityVerdict(False, family,
                                      f"UNDETERMINED: no evidence source could decide family "
                                      f"{family!r} for this target", undetermined=True)
        return EligibilityVerdict(False, family,
                                  f"target declares no capability for family {family!r}")
    for c in caps:
        if not _dtype_ok(region.in_dtype, c.dtypes):
            return EligibilityVerdict(False, family,
                                      f"input dtype {region.in_dtype!r} not in {c.family} formats "
                                      f"{list(c.dtypes)}")
        if region.weight_dtype is not None and not _dtype_ok(region.weight_dtype, c.dtypes):
            return EligibilityVerdict(False, family,
                                      f"weight dtype {region.weight_dtype!r} not supported by "
                                      f"{c.family}")
        if c.ranks and region.rank is not None and region.rank not in c.ranks:
            return EligibilityVerdict(False, family,
                                      f"rank {region.rank} not in {c.family} legal ranks "
                                      f"{list(c.ranks)}")
        if region.batch > 1 and not c.batch:
            return EligibilityVerdict(False, family,
                                      f"batched region (batch={region.batch}) but {c.family} declares "
                                      f"batch=false")
        if c.layouts and region.layout is not None and region.layout not in c.layouts:
            return EligibilityVerdict(False, family,
                                      f"layout {region.layout!r} not in {c.family} legal layouts "
                                      f"{list(c.layouts)}")
        # A capability available only FUSED cannot execute the region standalone. ``how == "direct"``
        # means the region asked for this family on its own; reached via ``primitives`` it is already
        # part of the composite that licenses it, which is exactly the fused form.
        if c.composed_with and how == "direct" and family == c.family:
            return EligibilityVerdict(False, family,
                                      f"{c.family} is available only fused with "
                                      f"{list(c.composed_with)} on this target, not standalone")
    return EligibilityVerdict(True, family, f"eligible ({how})")


# --- convenience: build the capability map from a contract / named target ---------------------------

def capability_map_from_contract(contract: dict) -> dict[str, SemanticCapability]:
    """Fold a contract's ``compute_units`` into the ``family -> SemanticCapability`` denominator map."""
    from merlin.targetgen import compute_units as cu

    return cu.semantic_capability_map(cu.compute_units(contract))


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
