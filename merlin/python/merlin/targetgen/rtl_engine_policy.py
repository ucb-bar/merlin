"""Which elaborated-RTL simulator certifies a capsule — and why that one.

The cert tier is a FIDELITY, not a simulator. VCS, GSIM and Verilator all run the elaborated design and
all produce an ``elaborated_rtl`` verdict; which one answers is an availability and cost decision, not a
statement about how trustworthy the result is. Binding a tier index to one binary (``L3 = verilator``)
made that decision invisible and unchangeable, and it put two different fidelities on the same rung
across targets.

The order is COST, since the fidelity is equal:

* ``vcs`` — the reference commercial simulator; used when a license and the resources are actually free.
* ``gsim`` — the fast FIRRTL simulator, and the default working choice. Measured at corpus scale on the
  SIMT target: 25 capsules certified in 48 min, mean 115 s/capsule, against ~45 min/capsule on Verilator
  — ~23x, i.e. the same sweep is ~19 h on Verilator. That is the difference between a cert tier that
  runs per-capsule and one affordable only once per run.
* ``verilator`` — last resort, for a target with no GSIM adapter yet. Not a fidelity compromise; it is
  simply the slow one.

Selection is by AVAILABILITY in that order, and every engine passed over is recorded with the reason it
was passed over. A tier that cannot run must come back as unavailable with that record — never silently
downgraded to a model tier, which is how a functional result gets read as an RTL certification.
"""
from __future__ import annotations

from typing import Any, Callable

# Declared once, with the rationale above. Cost order among engines of EQUAL fidelity.
ENGINE_PRIORITY: tuple[str, ...] = ("vcs", "gsim", "verilator")

# Every engine here answers at this fidelity; the tier records it rather than inferring from the name.
ELABORATED_RTL = "elaborated_rtl"


class UnrecordedSelection(RuntimeError):
    """An engine reported itself available without saying why. Refused: a tier that resolved to an engine
    for no recorded reason cannot be audited afterwards, and reads as if it were the declared one."""

    def __init__(self, target: str, engine: str):
        self.target, self.engine = target, engine
        super().__init__(f"{target}: engine {engine!r} reported available with no reason recorded; a "
                         f"selection that cannot be explained afterwards is refused, not defaulted")


class NoEngineAvailable(RuntimeError):
    """No elaborated-RTL engine can run for this target. Carries the per-engine reasons."""

    def __init__(self, target: str, considered: list[dict[str, Any]]):
        self.target, self.considered = target, considered
        detail = "; ".join(f"{c['engine']}: {c['reason']}" for c in considered) or "none registered"
        super().__init__(f"{target}: no elaborated-RTL engine available ({detail})")


def _ordered(engines: dict[str, Any]) -> list[str]:
    """Registered engines in priority order; anything unknown to the policy sorts last, alphabetically,
    so a newly added engine is USED rather than silently dropped before anyone declares its priority."""
    known = [e for e in ENGINE_PRIORITY if e in engines]
    return known + sorted(e for e in engines if e not in ENGINE_PRIORITY)


def select(target: str, engines: dict[str, Callable[[], tuple[bool, str]]]) -> dict[str, Any]:
    """Choose the elaborated-RTL engine for ``target``.

    ``engines`` maps an engine name to a probe returning ``(available, reason)``. Probes are called in
    priority order and STOP at the first available one, so an expensive probe for a lower-priority
    engine is never paid. Returns the selection record; raises :class:`NoEngineAvailable` when none can
    run (fail closed — the caller reports the tier unavailable, it does not substitute a lesser tier).
    """
    considered: list[dict[str, Any]] = []
    for name in _ordered(engines):
        try:
            ok, reason = engines[name]()
        except Exception as exc:                      # noqa: BLE001 - a broken probe is not availability
            ok, reason = False, f"probe raised {type(exc).__name__}: {exc}"
        considered.append({"engine": name, "available": bool(ok), "reason": reason})
        if ok and not str(reason or "").strip():
            # An engine that resolved for NO RECORDED REASON is the silent-degradation shape: the tier
            # answers with a different engine than the capsule asked for, the numbers look right, and the
            # result gets cited. Refuse it rather than defaulting.
            raise UnrecordedSelection(target, name)
        if ok:
            return {"engine": name, "fidelity": ELABORATED_RTL, "reason": reason,
                    "considered": considered,
                    "passed_over": [c["engine"] for c in considered[:-1]]}
    raise NoEngineAvailable(target, considered)


def describe(selection: dict[str, Any]) -> str:
    """One line for a report: what ran, and what it was chosen over."""
    over = selection.get("passed_over") or []
    tail = f" (over {', '.join(over)})" if over else ""
    return f"{selection['engine']} [{selection['fidelity']}]{tail}"
