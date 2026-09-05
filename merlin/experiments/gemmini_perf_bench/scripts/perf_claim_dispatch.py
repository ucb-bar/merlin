"""Route a family's frozen acceptance contract to the analyzer it names.

Three claim analyzers exist and each is reachable only by name from its own family's contract, so
adding a family meant adding a caller -- and nobody added one.  The consequence was silent: `PM`
and `PV` shipped declaring `PREDICTS` with a frozen contract that no code path ever evaluated, so
they produced measurements and no verdict, while reading in every artifact exactly like `PK`, which
is evaluated.  A claim nothing can decide is not a claim.

Dispatch is on the contract's own `analyzer` identity, never on the family name: the family is a
label, the analyzer identity is the thing the contract froze, and it already carries a version so a
retuned analyzer cannot silently score an old declaration.  An identity this registry does not know
is REFUSED and named -- never routed to a "closest" analyzer, and never quietly skipped, which is
the failure this module exists to end.

Two entry points, one rule.  :func:`resolve` is what the AUTHORING stage runs before a run: it turns
a cohort's declaration into the module, its single `preflight_*` precondition check and its decision
function, and raises with the reason when it cannot -- so an undecidable family is refused at launch
instead of after every L3 cell has been paid for.  :func:`analyze` is what a REPORT runs after, over
already-measured rows.  Both read the same frozen field, so the procedure that admitted a run is the
procedure that decides it.
"""
from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Callable

REFUSED = "REFUSED"

#: The prefix an analyzer's single launch-time precondition check is published under. The reporting
#: gate resolves it the same way -- by prefix, requiring exactly one -- so a module publishing none
#: or several is a wiring defect that must be named here rather than at report time, when every
#: measurement has already been paid for.
PREFLIGHT_PREFIX = "preflight_"


class DispatchError(RuntimeError):
    """A cohort whose declared decision procedure cannot be resolved, with the reason why."""


@dataclass(frozen=True)
class ResolvedAnalyzer:
    """Everything a caller needs to run one family's declared procedure."""

    identity: Any
    module: Any
    preflight: Callable[..., dict]
    analyze: Callable[..., dict]


def resolve(descriptors: Sequence[Mapping[str, Any]]) -> ResolvedAnalyzer:
    """Resolve the ONE procedure a cohort's own contracts name, or raise with the reason.

    This is the resolution the authoring stage runs BEFORE a run, where :func:`analyze` is the one a
    report runs after. Both read the declaration and neither consults a family name: a family the
    code has never heard of is dispatched by what it froze, and one that declares nothing is refused
    by name instead of falling through to whichever analyzer happened to be imported.
    """
    from merlin.perf import claim_reach

    if not isinstance(descriptors, Sequence) or isinstance(descriptors, str) or not descriptors:
        raise DispatchError("no capsule descriptors were supplied")
    identities: dict[str, Any] = {}
    for descriptor in descriptors:
        performance = (descriptor.get("performance") if isinstance(descriptor, Mapping) else None)
        name = (descriptor.get("name") if isinstance(descriptor, Mapping) else None) or "<unnamed>"
        try:
            identity = claim_reach.analyzer_identity(
                performance if isinstance(performance, Mapping) else {})
        except ValueError as exc:
            raise DispatchError(
                f"frozen capsule {str(name)!r} names an unusable claim analyzer: {exc}") from exc
        if identity is None:
            raise DispatchError(
                f"frozen capsule {str(name)!r} declares no acceptance.analyzer, so no procedure "
                "decides its family's claim")
        identities[identity.declared] = identity
    if len(identities) != 1:
        raise DispatchError(
            f"the cohort declares {len(identities)} claim analyzers {sorted(identities)}; one "
            "campaign seals one claim, so a mixed cohort is refused rather than split")
    identity = next(iter(identities.values()))
    try:
        module = importlib.import_module(identity.module)
    except Exception as exc:                                        # noqa: BLE001
        raise DispatchError(
            f"the declared claim analyzer {identity.declared!r} is unavailable: {exc}") from exc
    entries = sorted(name for name in dir(module)
                     if name.startswith(PREFLIGHT_PREFIX) and callable(getattr(module, name, None)))
    if len(entries) != 1:
        raise DispatchError(
            f"analyzer module {identity.module!r} publishes {len(entries)} preflight entry points; "
            "exactly one is required")
    decide = getattr(module, identity.function, None)
    if not callable(decide):
        raise DispatchError(
            f"the declared claim analyzer {identity.declared!r} is unavailable: module "
            f"{identity.module!r} publishes no {identity.function!r}")
    return ResolvedAnalyzer(identity=identity, module=module,
                            preflight=getattr(module, entries[0]), analyze=decide)


def _registry() -> dict[str, Callable[..., dict]]:
    """Imported lazily so one broken analyzer cannot make the whole dispatcher unimportable."""
    table: dict[str, Callable[..., dict]] = {}
    try:
        import perf_pk_claim as PK
        table[PK._ACCEPTANCE_BASE["analyzer"]] = PK.analyze_pk_claim
    except Exception:  # noqa: BLE001 - an absent analyzer is reported at dispatch, not at import
        pass
    try:
        import perf_affine_claim as AF
        table[AF.ANALYZER] = AF.analyze_affine_claim
    except Exception:  # noqa: BLE001
        pass
    try:
        import perf_paired_claim as PD
        table[PD.ANALYZER] = PD.analyze_paired_claim
    except Exception:  # noqa: BLE001
        pass
    try:
        # This analyzer keeps its identity inside its acceptance template rather than as a module
        # constant, so it is read from there -- never re-spelled here, which would let the registry
        # and the contract drift apart silently.
        import perf_pr_claim as PR
        for attr in ("_ACCEPTANCE_BASE", "ACCEPTANCE_BASE", "_ACCEPTANCE",
                     "_PROPOSED_ACCEPTANCE"):
            base = getattr(PR, attr, None)
            if isinstance(base, dict) and isinstance(base.get("analyzer"), str):
                table[base["analyzer"]] = PR.analyze_pr_claim
                break
    except Exception:  # noqa: BLE001
        pass
    return table


def declared_analyzer(descriptors: Sequence[Mapping[str, Any]]) -> str | None:
    """The single analyzer identity the cohort's contracts agree on, or None if they do not."""
    seen = set()
    for d in descriptors:
        if not isinstance(d, Mapping):
            return None
        acceptance = ((d.get("performance") or {}) if isinstance(d.get("performance"), Mapping)
                      else {}).get("acceptance")
        if not isinstance(acceptance, Mapping):
            return None
        seen.add(str(acceptance.get("analyzer")))
    return seen.pop() if len(seen) == 1 else None


def analyze(descriptors: object, results: object) -> dict[str, Any]:
    """Decide one family by handing it to the analyzer its own contract names."""
    if not isinstance(descriptors, Sequence) or not descriptors:
        return {"verdict": REFUSED, "reason": "no capsule descriptors were supplied"}
    identity = declared_analyzer(descriptors)
    if identity is None:
        return {"verdict": REFUSED,
                "reason": ("the cohort does not agree on one frozen analyzer identity, so there is "
                           "no single procedure that decides it")}
    table = _registry()
    if identity not in table:
        return {"verdict": REFUSED, "declared_analyzer": identity,
                "reason": (f"no analyzer registered for {identity!r}; the contract names a "
                           f"procedure this build cannot run, so the claim is undecided rather "
                           f"than assumed"),
                "registered": sorted(table)}
    verdict = table[identity](descriptors, results)
    if isinstance(verdict, Mapping):
        return {**verdict, "declared_analyzer": identity}
    return {"verdict": REFUSED, "declared_analyzer": identity,
            "reason": "the analyzer returned something other than a verdict mapping"}
