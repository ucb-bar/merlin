"""How a new gate ships: report, then incomplete where it cannot decide, then fail.

A gate that flips to ``fail`` on the day it lands is indistinguishable from a regression. Everyone
downstream sees a run that used to pass and now does not, and nobody can tell whether the code got
worse or the rule got stricter -- so the first thing they do is route around it. Every gate in this
package therefore carries an explicit phase, and the phase is a PARAMETER rather than a comment,
because "we will turn it on later" written in prose is not a state anything can read.

Three states, and the middle one is the one that gets skipped:

``report``      the verdict is computed in full and recorded beside the grade; nothing is blocked.
                A live experiment's arms stay comparable, because the rule that scores them did not
                change.
``incomplete``  the gate RAN and could not decide. Never a pass, at any phase. This repo's recurring
                failure is the check that could not run and reported success (`codegen_smoke
                n/a -> true` burned 101 minutes), and a two-valued gate has nowhere to put that.
``fail``        the gate decides and blocks. Only a DECIDED failure blocks; ``incomplete`` still
                does not, because blocking on "we could not tell" is a different rule that needs its
                own decision.

The vocabulary lives here, in one module, so two gates cannot spell the same three states
differently and drift.

WHICH phase a given gate is in is not decided here either: :func:`configured_phase` reads it from
``merlin/contract/gate_phases.yaml``. Making the phase a parameter was only half the idea -- every
caller still passed ``PHASE_REPORT`` as a literal, so a rollout meant editing Python and left no
reviewable record that holding a gate back had been a decision rather than an oversight. A gate the
declaration does not name RAISES; there is no default, because a default phase is a phase nobody
chose and it fails in the quietest possible way.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

__all__ = [
    "PHASE_REPORT",
    "PHASE_FAIL",
    "PHASES",
    "STATUS_INCOMPLETE",
    "GatePhaseError",
    "blocks",
    "configured_phase",
    "declared_gates",
]

#: Computes the verdict and records it; blocks nothing. Where a new gate lands.
PHASE_REPORT = "report"
#: Blocks on a DECIDED failure.
PHASE_FAIL = "fail"
PHASES = (PHASE_REPORT, PHASE_FAIL)

#: The gate ran and could not decide. Never a pass, at either phase.
STATUS_INCOMPLETE = "incomplete"


def blocks(phase: str, status: str, *, failing: tuple[str, ...]) -> bool:
    """Does ``status`` block at ``phase``?

    ``failing`` names the statuses that are DECIDED failures. Passing it rather than assuming
    "anything that is not a pass" is the whole point: an undecided verdict must not block, and a
    gate that treats them the same has quietly turned "we could not measure this" into a failure a
    submission cannot fix.
    """
    if phase not in PHASES:
        raise ValueError(f"phase must be one of {PHASES}, got {phase!r}")
    return phase == PHASE_FAIL and status in failing


class GatePhaseError(ValueError):
    """The declared phase for a gate could not be resolved, so no phase is assumed."""


#: The tracked declaration. One line per gate; see the file's own header for why there is no default.
_DECLARATION = ("contract", "gate_phases.yaml")


@lru_cache(maxsize=1)
def _declared() -> dict[str, str]:
    from ..common.paths import merlin_dir

    return _read_declared(merlin_dir().joinpath(*_DECLARATION))


def _read_declared(path: Path) -> dict[str, str]:
    """Read one selected declaration; explicit callers do not share the default cache."""
    import yaml

    if not path.is_file():
        raise GatePhaseError(f"no gate-phase declaration at {path}; a phase is declared, never assumed")
    body = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    gates = body.get("gates")
    if not isinstance(gates, dict) or not gates:
        raise GatePhaseError(f"{path} declares no gates; an empty declaration is not 'everything reports'")
    out: dict[str, str] = {}
    for gate, phase in gates.items():
        # REFUSED BY NAME, not coerced. `incomplete` is the value someone reaches for when they mean
        # "ran but could not decide" -- which is a STATUS this module already has, orthogonal to
        # phase. Silently reading it as `report` would make the file say something it does not.
        if phase not in PHASES:
            raise GatePhaseError(f"{path}: gate {gate!r} declares phase {phase!r}; must be one of {PHASES}")
        out[str(gate)] = str(phase)
    return out


def configured_phase(gate: str, *, declaration: str | Path | None = None) -> str:
    """The declared phase for ``gate``, or raise.

    There is deliberately no default. A gate this file does not name is a gate whose rollout nobody
    decided, and resolving that to ``report`` would hide it forever: the gate would run, block
    nothing, and look exactly like one that had been considered and held back on purpose.

    Explicit declarations are read on every call and never fall back to the checkout.
    """
    declared = _declared() if declaration is None else _read_declared(Path(declaration))
    if gate not in declared:
        known = ", ".join(sorted(declared)) or "(none)"
        raise GatePhaseError(
            f"gate {gate!r} is not declared in {declaration or '/'.join(_DECLARATION)}; declared gates are: {known}. "
            "Add it with the phase it should ship in -- there is no default phase."
        )
    return declared[gate]


def declared_gates(*, declaration: str | Path | None = None) -> tuple[str, ...]:
    """Every gate the declaration names, sorted. For tests and for reporting what exists."""
    declared = _declared() if declaration is None else _read_declared(Path(declaration))
    return tuple(sorted(declared))
