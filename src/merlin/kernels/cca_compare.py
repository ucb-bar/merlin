"""Multi-level CCA comparator — structured, typed divergences (target-agnostic).

Given an expert CCA and ours (both lifted to the Common Compute Abstraction, ideally at the asm
level — the authoritative substrate), emit ``Divergence`` records: one per populated facet field
that differs. These feed the action catalog (``action_catalog.py``), which maps each to a typed
compiler change. Comparison is per-facet, so a target only diffs the facets it has (vector for
RVV, spatial for a systolic array, …) — nothing RVV-specific here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass, replace
from typing import Any, get_args, get_type_hints

from .cca import CCA


@dataclass
class Divergence:
    axis: str  # "compute.contraction_form", "vector.lmul", ...
    expert: Any
    ours: Any
    backend: str
    evidence: list[str] = field(default_factory=list)  # kernel ids justifying the expert value


def _populated_pairs(fa, fb) -> dict[str, tuple]:
    if fa is None or fb is None:
        return {}
    out = {}
    for k, va in asdict(fa).items():
        vb = asdict(fb).get(k)
        if va is not None and vb is not None and va != vb:
            out[k] = (va, vb)
    return out


def uncomparable_axes(expert: CCA, ours: CCA) -> list[tuple[str, str]]:
    """Axes one side populated and the other did NOT -- returned as ``(axis, which_side_is_missing)``.

    ``compare`` deliberately emits no divergence when either side is None: it cannot claim a gap it
    cannot see, and inventing one would be worse. But SILENCE is not the same as "no gap", and the
    difference matters because the loop can only discover what it is shown. This module already knew
    that hazard and patched it for ONE axis -- see the MR-aware special case in ``compare``, added
    because "an UNBLOCKED kernel lifts to register_block=None, so 'expert blocks to MR=k, ours
    doesn't' never surfaced". Every other axis kept the blindness.

    OBSERVED: selecting a dtype-matched int8 expert (correctly, to stop a cross-dtype comparison
    manufacturing fake divergences) took the divergence count 8 -> 5. Two of the three that vanished
    did so because the qd8 fixture lifts ``compute.register_block=None`` and no ``memory`` facet at
    all -- so real gaps on register blocking and access pattern became invisible rather than resolved.
    Trading visible noise for invisible blindness is the wrong trade; this makes the blindness visible.

    Report this next to ``compare``'s output, the way ``build_catalog`` reports unrouted divergences.
    """
    out: list[tuple[str, str]] = []
    for facet in _facet_names():
        fe, fo = getattr(expert, facet, None), getattr(ours, facet, None)
        if fe is None and fo is None:
            continue
        if fe is None or fo is None:
            # the whole facet is missing on one side; name it once rather than per field
            out.append((facet, "expert" if fe is None else "ours"))
            continue
        de, do = asdict(fe), asdict(fo)
        for k in sorted(set(de) | set(do)):
            ve, vo = de.get(k), do.get(k)
            if (ve is None) != (vo is None):
                out.append((f"{facet}.{k}", "expert" if ve is None else "ours"))
    return out


def _facet_names() -> tuple[str, ...]:
    """Facet fields of :class:`CCA`, reflected rather than hardcoded.

    This list used to be a literal, which meant a newly added facet was silently never compared --
    the lift would populate it, the contract would demand a route for it, and `compare` would still
    report no divergence, so the beam would never see it. Reflecting keeps a new facet wired by
    construction. (Found the hard way: `region` was invisible here even after being lifted.)

    A facet is identified STRUCTURALLY -- a field whose declared type is one of the ``*Facet``
    dataclasses -- rather than by subtracting a hardcoded list of non-facets. That subtraction had
    the mirror image of the same bug: adding a plain scalar field to ``CCA`` (``scope``) put it in
    this list, and ``compare`` then tried to read facet fields off a string. A list of what to
    EXCLUDE must be updated by whoever adds a non-facet; deriving what to INCLUDE need not be.
    """
    from dataclasses import fields as _fields
    from dataclasses import is_dataclass

    from merlin.kernels import cca as _cca

    facet_types = {n for n, o in vars(_cca).items() if is_dataclass(o) and n.endswith("Facet")}
    return tuple(f.name for f in _fields(CCA) if any(t in str(f.type) for t in facet_types))


def compare(expert: CCA, ours: CCA, *, evidence: list[str] | None = None) -> list[Divergence]:
    """expert-vs-ours CCA -> typed Divergences (the authoritative gap when both are asm-lifted).

    REFUSES a cross-scope comparison. The same axis name asks a different question at each scope --
    ``dispatch.n_dispatches`` means "launches in this region" at kernel scope and "launches in the
    model" at program scope -- so comparing across scopes yields divergences that are real numbers
    answering different questions, which is worse than no comparison because they look routable.
    """
    es, os_ = getattr(expert, "scope", "kernel"), getattr(ours, "scope", "kernel")
    if es != os_:
        raise ValueError(
            f"refusing to compare a {es!r}-scoped CCA against a {os_!r}-scoped one: the same axis "
            f"means a different question at each scope, so the differences would not be divergences"
        )
    ev = evidence or [expert.provenance.get("source", "expert")]
    backend = (expert.backend or ours.backend or ["?"])[0]
    out: list[Divergence] = []
    for facet in _facet_names():
        for k, (ve, vo) in _populated_pairs(getattr(expert, facet), getattr(ours, facet)).items():
            out.append(Divergence(axis=f"{facet}.{k}", expert=ve, ours=vo, backend=backend, evidence=list(ev)))

    # register_block (MR) is the #1 GEMM data-movement decision and the one we were structurally
    # blind to: _populated_pairs only fires when BOTH sides report it, but an UNBLOCKED kernel lifts
    # to register_block=None (== MR 1, no row reuse), so "expert blocks to MR=k, ours doesn't" never
    # surfaced. Compare MR-aware (None -> MR 1) so the gap becomes a real, routable divergence.
    def _mr(cca):
        rb = getattr(cca.compute, "register_block", None) if cca and cca.compute else None
        if isinstance(rb, (tuple, list)) and rb and isinstance(rb[0], int):
            return rb[0]
        return 1

    emr, omr = _mr(expert), _mr(ours)
    if emr > omr and not any(d.axis == "compute.register_block" for d in out):
        out.append(
            Divergence(
                axis="compute.register_block",
                expert=expert.compute.register_block,
                ours=(ours.compute.register_block if (ours and ours.compute) else None),
                backend=backend,
                evidence=list(ev),
            )
        )
    return out


@dataclass(frozen=True)
class EvidenceComparison:
    """Supported structural comparisons, never an agreement or qualification verdict."""

    status: str
    divergences: tuple[Divergence, ...]
    compared_axes: tuple[str, ...]
    unknowns: tuple[dict, ...]
    advisory_scope: str = "Structural evidence only; semantic, numerical and performance equality are not established."

    @property
    def denominator(self) -> int:
        return len(self.compared_axes)

    def to_dict(self) -> dict:
        return dict(asdict(self), denominator=self.denominator)


_RESIDENCY_AXES = frozenset((
    "memory.onchip_resident", "memory.panel_reuse", "compute.accumulator_resident",
    "spatial.accumulator_resident", "simt.smem_resident", "communication.resident_across_calls",
))
_COUNT_AXES = frozenset(("dispatch.n_dispatches", "dispatch.n_fences", "dispatch.config_fraction",
                         "dispatch.dma_issue_to_wait", "compute.reduction_partition"))


def _axis_evidence(cca, axis):
    evidence = cca.provenance.get("axis_evidence", {})
    row = evidence.get(axis, {}) if isinstance(evidence, dict) else {}
    return row if isinstance(row, dict) else {}


def _supported(row):
    witnesses = row.get("evidence")
    return (row.get("status") in ("observed", "inferred")
            and isinstance(witnesses, (tuple, list)) and bool(witnesses)
            and all(type(witness) is str and bool(witness.strip()) for witness in witnesses))


def _known_scope(value):
    if isinstance(value, dict):
        return bool(value) and all(type(key) is str and key and _known_scope(item) for key, item in value.items())
    if isinstance(value, (tuple, list)):
        return bool(value) and all(_known_scope(item) for item in value)
    return (type(value) in (int, bool)
            or type(value) is str and bool(value.strip()) and value.strip().lower() not in ("unknown", "?"))


def compare_evidence(expert: CCA, ours: CCA, *, evidence: list[str] | None = None) -> EvidenceComparison:
    """Compare only populated axes with explicit status, witnesses and compatible scopes.

    Both absent values remain unknown. Residency and dispatch metrics additionally
    require matching per-axis ``axis_evidence[axis]['scope']`` records; a global
    residency label alone cannot prove matching lifetimes or count boundaries.
    Callers retain responsibility for source fusion/numerics, target identity and
    trusted artifact binding. ``compared`` means a nonzero denominator, not equality.
    """
    if type(expert) is not CCA or type(ours) is not CCA:
        raise ValueError("evidence comparison requires existing typed CCAs")
    unknowns, identity_mismatch = [], False
    for key in ("op", "scope"):
        values = (getattr(expert, key), getattr(ours, key))
        missing = tuple(side for side, value in zip(("expert", "ours"), values)
                        if type(value) is not str or value.strip().lower() in ("", "unknown", "?"))
        if missing or values[0] != values[1]:
            identity_mismatch |= not missing
            unknowns.append(dict(axis=key, reason="identity_unknown" if missing else "identity_mismatch",
                                 missing_sides=missing, expert=values[0], ours=values[1]))
    problems = (expert.scope_problems(), ours.scope_problems())
    if any(problems):
        unknowns.append(dict(axis="scope", reason="invalid_scope", expert=problems[0], ours=problems[1]))
    blocked = bool(unknowns)
    eligible, witnesses = [], {}
    annotations = get_type_hints(CCA)
    for facet in _facet_names():
        candidates = get_args(annotations[facet]) or (annotations[facet],)
        schema = next(candidate for candidate in candidates if is_dataclass(candidate))
        for item in fields(schema):
            axis = facet + "." + item.name
            values = tuple(getattr(getattr(cca, facet), item.name, None) for cca in (expert, ours))
            rows = tuple(_axis_evidence(cca, axis) for cca in (expert, ours))
            missing = tuple(side for side, value in zip(("expert", "ours"), values) if value is None)
            reason = "value_unavailable" if missing else "comparison_identity_unresolved" if blocked else None
            if reason is None:
                missing = tuple(side for side, row in zip(("expert", "ours"), rows) if not _supported(row))
                reason = "axis_evidence_unavailable" if missing else None
            scopes = tuple(row.get("scope") for row in rows)
            if reason is None:
                required = axis in _RESIDENCY_AXES | _COUNT_AXES or any(scope is not None for scope in scopes)
                missing = tuple(side for side, scope in zip(("expert", "ours"), scopes)
                                if not isinstance(scope, (str, dict)) or not _known_scope(scope))
                if required and missing:
                    reason = "axis_scope_unavailable"
                elif required and scopes[0] != scopes[1]:
                    reason = "axis_scope_mismatch"
                else:
                    legacy = tuple(cca.provenance.get("residency_scope") for cca in (expert, ours))
                    if axis in _RESIDENCY_AXES and all(legacy) and legacy[0] != legacy[1]:
                        reason = "residency_scope_mismatch"
            if reason:
                unknowns.append(dict(axis=axis, reason=reason, missing_sides=missing,
                                     expert_scope=scopes[0], ours_scope=scopes[1]))
            else:
                eligible.append(axis)
                witnesses[axis] = tuple(dict.fromkeys((*rows[0]["evidence"], *rows[1]["evidence"])))
    divergences = tuple(replace(row, evidence=list(witnesses[row.axis]))
                        for row in (() if blocked else compare(expert, ours, evidence=evidence))
                        if row.axis in witnesses)
    status = "mismatch" if identity_mismatch else "compared" if eligible else "unknown"
    return EvidenceComparison(status, divergences, tuple(sorted(eligible)), tuple(unknowns))
