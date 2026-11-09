"""How one self-check request relates to the official grade's own oracle selection.

The grade (``qa.run``) grades with ``capsule_runner.qa_checkpoint_adapters``. A self-check that uses
the same adapters reaches the grade's verdict on the same bytes and shares its exact-byte simulator
certificate instrument. These helpers describe that selection; they select nothing on their own and
never relax a tier: certification still requires the capsule's own ``pass`` status.
"""

from __future__ import annotations


def tier_engines(adapters: dict) -> dict[str, str]:
    """The engine each selected tier runs on, read from the adapter's own recorded identity."""
    return {
        tier: str(getattr(adapter, "_merlin_simulator_engine", "") or "contract-resolved")
        for tier, adapter in sorted(adapters.items())
    }


def grade_l3_selection(target: str, engine: str) -> dict | None:
    """The official grade's L3 selection record, when the grade selects the ``engine`` named here.

    The grade builds its elaborated-RTL adapter with this record, and an exact-byte simulator
    certificate's instrument digest binds the adapter's record. A ``--sim gsim`` adapter built without
    it earned certificates the grade could never carry, and could carry none of the grade's: measured
    on one run, every official L3 certificate used one instrument and every self-check or promotion
    certificate another. Same engine, same record; a different or unselectable engine keeps none.
    """
    from merlin.targetgen import oracle_policy

    try:
        selection = oracle_policy.chipyard_tier_plan(target).selection
    except Exception:  # noqa: BLE001 -- no selectable engine: the named engine keeps no grade record
        return None
    return dict(selection) if isinstance(selection, dict) and selection.get("engine") == engine else None


def capsule_barrier(tier_results: dict, adapters: dict, declared: str) -> str:
    """The tier one capsule is judged at under the grade's own selection.

    The official grade passes a capsule when every tier IT declares mandatory passed, so the barrier is
    the deepest of those tiers this selection runs. A capsule recording no mandatory selected tier keeps
    the corpus-level ``declared`` barrier, whose absence the caller already handles.
    """
    mandatory = sorted(
        tier for tier, record in (tier_results or {}).items()
        if tier in adapters and isinstance(record, dict) and record.get("mandatory")
    )
    return mandatory[-1] if mandatory else declared


def selection_note(*, combined: bool, required: bool, sim: str, adapters: dict, barrier_tier: str) -> str:
    """The sentence naming what this request ran, for the agent-facing note."""
    if combined:
        return (
            "Self-check on every tier the official grade selects ("
            + ", ".join(f"{tier} {engine}" for tier, engine in tier_engines(adapters).items())
            + "): a capsule passes only when every tier it declares mandatory passed, the same rule and "
            "adapters the grade applies to these bytes. "
        )
    if required:
        return (
            f"Self-check on the grade's own adapters narrowed by --tiers to {sorted(adapters)}; a capsule "
            "with a mandatory tier outside them is not certified by this request. "
        )
    return f"Self-check on {sim} ({barrier_tier}). "


def select_tiers(full: dict, default: dict, requested: str) -> tuple[dict, str | None]:
    """Select explicit tiers from the full reachable map, or retain the default.

    The default loop ladder may omit expensive certificate tiers. Explicit
    requests are checked against the full endpoint inventory so those tiers
    remain reachable. Return (adapters, error); never silently drop an
    unavailable requested tier."""
    want = {t.strip().upper() for t in requested.split(",") if t.strip()}
    if not want:
        return default, None
    missing = sorted(want - set(full))
    if missing:
        return {}, (f"--tiers names {missing}, which this endpoint does not reach; reachable: {sorted(full)}")
    return {t: ad for t, ad in full.items() if t in want}, None
