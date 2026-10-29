"""Human-readable Phase 1 experiment levels over stable, frozen arm identities.

Levels name comparisons, not compiler capability or an ordering of quality.  In
particular, the optional EGraph arm branches from EL3 and the verification arm
branches from EL4.  Existing bundle ids, treatment keys and frozen runs retain
their original machine-readable names.
"""

from __future__ import annotations

from collections.abc import Mapping

LEVELS = (
    {
        "id": "EL1",
        "name": "Raw baseline",
        "bundle_arm": "raw_baseline",
        "runner_arm": "raw_baseline",
        "treatment": "baseline",
        "parent": None,
    },
    {
        "id": "EL2",
        "name": "C++ infrastructure",
        "bundle_arm": "cpp_merlininfra",
        "runner_arm": "cpp_merlininfra",
        "treatment": "baseline",
        "parent": "EL1",
    },
    {
        "id": "EL3",
        "name": "Merlin-assisted",
        "bundle_arm": "merlin_assisted",
        "runner_arm": "merlin_assisted",
        "treatment": "baseline",
        "parent": "EL2",
    },
    {
        "id": "EL4",
        "name": "RTL-informed Merlin",
        "bundle_arm": "merlin_rtlchecks",
        "runner_arm": "merlin_assisted",
        "treatment": "rtlchecks",
        "parent": "EL3",
    },
    {
        "id": "EL3-E",
        "name": "EGraph variant",
        "bundle_arm": "merlin_eqsat",
        "runner_arm": "merlin_assisted",
        "treatment": "baseline",
        "parent": "EL3",
    },
    {
        "id": "EL4-V",
        "name": "RTL + verification tools (baseline feedback)",
        "bundle_arm": "merlin_verify",
        "runner_arm": "merlin_assisted",
        "treatment": "baseline",
        "parent": "EL4",
    },
    {
        "id": "EL4-P",
        "name": "RTL + static pass catalog",
        "bundle_arm": "merlin_pass_catalog",
        "runner_arm": "merlin_assisted",
        "treatment": "rtlchecks",
        "parent": "EL4",
    },
    {
        "id": "EL5",
        "name": "RTL + ISA-derived instruction dialect",
        "bundle_arm": "merlin_isa",
        "runner_arm": "merlin_assisted",
        "treatment": "rtlchecks",
        "parent": "EL4",
    },
    {
        "id": "EL6",
        "name": "ISA generation + interactive Lean proof tooling",
        "bundle_arm": "merlin_lean",
        "runner_arm": "merlin_assisted",
        "treatment": "rtlchecks",
        "parent": "EL5",
    },
)

_BY_ARM = {level["bundle_arm"]: level["id"] for level in LEVELS}
_BY_ID = {level["id"]: level for level in LEVELS}


def materialize_level(config: dict[str, object]) -> None:
    """Expand an authored level into the historical execution keys in place.

    The saved plan continues to carry the real runner arm and treatment. This is
    deliberate: changing either key in an existing run would change its inputs.
    """
    selected = config.get("level")
    if selected is None:
        return
    if not isinstance(selected, str) or selected not in _BY_ID:
        raise ValueError(f"unknown Phase 1 experiment level {selected!r}")
    entry = _BY_ID[selected]
    for key, expected in (("arm", entry["runner_arm"]), ("treatment", entry["treatment"])):
        actual = config.setdefault(key, expected)
        if actual != expected:
            raise ValueError(f"{selected} requires {key}: {expected}, got {actual!r}")
    bundle = config.get("bundle")
    if isinstance(bundle, str):
        from merlin.targetgen.generate_bundles import _ALL_ARMS

        matches = [name for name, stem in _ALL_ARMS.items() if bundle.startswith(stem + "_")]
        if matches:
            bundle_arm = max(matches, key=lambda name: len(_ALL_ARMS[name]))
            if bundle_arm != entry["bundle_arm"]:
                raise ValueError(f"{selected} contradicts the selected bundle for {_BY_ARM[bundle_arm]}")


def level_for_phase1(config: Mapping[str, object]) -> str | None:
    """Label a declared selection without changing its execution identity.

    The runner's ``arm`` is historically ``merlin_assisted`` for both EL3 and
    EL4.  A concrete bundle stem, when available, distinguishes their tool
    grants; the RTL-checks treatment distinguishes EL4 in authored definitions.
    Unknown or contradictory selections stay unlabeled instead of acquiring a
    plausible but incorrect level.
    """
    declared = config.get("level")
    arm = config.get("arm")
    treatment = config.get("treatment", "baseline")
    bundle = config.get("bundle")
    if not isinstance(arm, str) or not isinstance(treatment, str):
        return None
    if isinstance(bundle, str):
        from merlin.targetgen.generate_bundles import _ALL_ARMS

        matches = [name for name, stem in _ALL_ARMS.items() if bundle.startswith(stem + "_")]
        if matches:
            selected = max(matches, key=lambda name: len(_ALL_ARMS[name]))
            if selected in {"merlin_rtlchecks", "merlin_pass_catalog", "merlin_isa", "merlin_lean"}:
                inferred = _BY_ARM[selected] if arm == "merlin_assisted" and treatment == "rtlchecks" else None
                return inferred if declared in (None, inferred) else None
            if selected in {"merlin_assisted", "merlin_eqsat", "merlin_verify"}:
                inferred = _BY_ARM[selected] if arm == "merlin_assisted" and treatment == "baseline" else None
                return inferred if declared in (None, inferred) else None
            inferred = _BY_ARM[selected] if arm == selected and treatment == "baseline" else None
            return inferred if declared in (None, inferred) else None
    if arm == "merlin_assisted":
        inferred = "EL4" if treatment == "rtlchecks" else "EL3" if treatment == "baseline" else None
        if declared == "EL4-P" and treatment == "rtlchecks":
            inferred = declared
        if declared in ("EL3-E", "EL4-V") and treatment == "baseline":
            inferred = declared
        if declared in ("EL5", "EL6") and treatment == "rtlchecks":
            inferred = declared
    else:
        inferred = _BY_ARM.get(arm) if treatment == "baseline" else None
    return inferred if declared in (None, inferred) else None
