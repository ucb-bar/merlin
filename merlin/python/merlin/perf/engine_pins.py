"""The simulator artifacts a cross-validation capture is ABOUT, resolved from the pin registry.

A capture asserts that two engines agree on one ELF. It is therefore only valid for those two engines,
and the four artifacts that identify them -- each engine's binary and the elaboration it was built from,
plus the generated-model manifest -- have to travel with it. Until now the only place that set existed
was inside a certificate produced by a command line that named all five by hand, so nothing else could
say what a capture was about: not the grade that produced the ELF, not a cache deciding whether an
earlier capture answers today's question.

RESOLVED BY DECLARED ROLE, never by a naming convention. Each registry entry states its ``target`` and
its ``role``; one target's elaboration is named after its own configuration, which is not a convention
any other target shares, so a consumer that pattern-matched names would work for exactly one target.

VERIFIED, NOT JUST LOCATED. Every artifact is checked against its declared digest before the set is
returned. A path that resolves to different bytes is the wrong-device hazard in its quiet form -- the
registry exists to make that loud -- so a mismatch refuses the whole set rather than returning four good
pins and one lie.

PARTIAL IS REFUSED. `capture_store.capture_key` returns None on an incomplete pin set, so a half
resolved set silently files nothing. Refusing here, by name, turns that silence into a sentence.
"""
from __future__ import annotations

from typing import Any

__all__ = ["REQUIRED_ROLES", "EnginePinsUnavailable", "engine_pins"]

#: The roles a capture is keyed on. The same five `perf_gsim_gate.REQUIRED_PINS` names, because this
#: resolves exactly that set and a second spelling of it would drift.
REQUIRED_ROLES: tuple[str, ...] = (
    "gsim_binary", "gsim_firrtl", "gsim_model", "verilator_binary", "verilator_firrtl")


class EnginePinsUnavailable(RuntimeError):
    """The engine set for a target could not be established, and why."""


def engine_pins(target: str, *, roles: "tuple[str, ...]" = REQUIRED_ROLES,
                registry: Any = None) -> dict[str, dict[str, str]]:
    """``{role: {"path": str, "sha256": str}}`` for ``target``, every entry verified against disk.

    Raises :class:`EnginePinsUnavailable` naming the roles that could not be established. Callers that
    would rather degrade than fail catch it; nothing here decides that for them.
    """
    from merlin.common import provenance as _prov

    artifacts = _prov.load_artifacts(registry)
    by_role: dict[str, list[str]] = {}
    for name, artifact in artifacts.items():
        if str(getattr(artifact, "target", "")) == str(target) and getattr(artifact, "role", ""):
            by_role.setdefault(str(artifact.role), []).append(name)

    resolved: dict[str, dict[str, str]] = {}
    problems: list[str] = []
    for role in roles:
        names = sorted(by_role.get(role, ()))
        if not names:
            problems.append(f"{role}: no artifact declares target={target!r} role={role!r}")
            continue
        if len(names) > 1:
            # Two artifacts claiming one role is a registry error, not something to break ties on:
            # picking one would make the capture describe an engine chosen by sort order.
            problems.append(f"{role}: {len(names)} artifacts claim it ({', '.join(names)})")
            continue
        check = _prov.verify_artifact(names[0], path=registry)
        if not check.present:
            problems.append(f"{role}: {names[0]} is not present ({'; '.join(check.gaps) or check.path})")
            continue
        if check.matches is not True:
            problems.append(
                f"{role}: {names[0]} is present but does not match its declared digest"
                if check.matches is False else
                f"{role}: {names[0]} declares no digest, so it would certify itself")
            continue
        resolved[role] = {"path": str(check.path), "sha256": str(check.digest)}

    if problems:
        raise EnginePinsUnavailable(
            f"engine pins for {target!r} are incomplete: " + "; ".join(problems))
    return resolved
