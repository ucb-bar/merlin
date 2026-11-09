"""Selected direct-certification minimum for newly derived and staged corpora.

This is an input check, not a tier promotion. A cost-capped member remains
cost-capped until a new producer honestly derives a directly certified member.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence


def _tier_index(value: object) -> int | None:
    if not isinstance(value, str) or not value.startswith("L") or not value[1:].isdigit():
        return None
    return int(value[1:])


def selected_floor(workload_spec: Mapping | None, declared_tiers: Sequence[str]) -> str | None:
    """Return the explicitly selected floor, refusing absent or invalid tier vocabulary."""
    raw = (workload_spec or {}).get("certification_floor")
    if raw is None:
        return None
    if _tier_index(raw) is None or raw not in declared_tiers:
        raise ValueError("certification_floor must name a tier in the selected recipe")
    return raw


def require_direct_tier(
    capsules: Sequence[Mapping], *, floor: str | None, declared_tiers: Sequence[str] | None = None
) -> dict:
    """Check every member against a selected floor without changing membership or tiers.

    ``declared_tiers`` supplies the recipe default before materialization.
    Materialized capsules must state their own required tiers instead.
    """
    if floor is None:
        return {"floor": None, "checked": len(capsules), "below_floor": []}
    floor_index = _tier_index(floor)
    if floor_index is None:
        raise ValueError("certification_floor is not a fidelity tier")
    refused: list[str] = []
    for capsule in capsules:
        name = capsule.get("name") if isinstance(capsule, Mapping) else None
        if not isinstance(name, str) or not name:
            raise ValueError("certification-floor preflight requires named capsules")
        tiers = capsule.get("required_oracle_tiers")
        if tiers is None:
            tiers = declared_tiers
        if not isinstance(tiers, (list, tuple)) or floor not in tiers:
            refused.append(name)
            continue
        cap = capsule.get("max_oracle_tier")
        if cap is not None and (_tier_index(cap) is None or _tier_index(cap) < floor_index):
            refused.append(name)
    if refused:
        raise ValueError(
            f"certification_floor {floor} requires direct certification for every capsule; "
            f"{len(refused)} of {len(capsules)} fall below it: {', '.join(sorted(refused))}"
        )
    return {"floor": floor, "checked": len(capsules), "below_floor": []}
