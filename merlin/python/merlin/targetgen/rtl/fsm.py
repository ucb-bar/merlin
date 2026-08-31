"""Which registers of a design are control FSMs — derived from its RTL, not written down.

An occupancy vector needs to know which signals are engines. Naming them in a script is the failure
this repo's cardinal rule exists to stop: it makes onboarding a target an editing job, and it cannot
distinguish "this target has no such unit" from "nobody listed it". The set is derivable, and the
derivation already exists — a synthesis pass detects state machines structurally and exports one
transition table per FSM register, named ``<Module>.<register>``.

DETECTION, NOT EXPORT -- THE SYNTHESIS TOOL ANSWERS A DIFFERENT QUESTION
------------------------------------------------------------------------
The obvious source is the exported transition tables, and it is the wrong one. A synthesis flow
exports only the state machines whose RE-ENCODING would pay off, and drops the rest with a note that
"users of the state reg look like FSM recoding might result in larger circuit". That is a question
about optimisation, not about observability -- and it discards precisely the controllers whose state
is compared against constants elsewhere, which is most of the interesting ones. Measured on one
target: **15 state machines detected, 3 exported**, and the two controllers whose concurrency was the
entire point of the measurement were among the 12 dropped.

So the inventory comes from what the tool DETECTED, and the exported tables only enrich it where they
exist. Using the export as the inventory would have silently answered "this design has three
controllers" about a design with fifteen.

WHAT THIS GIVES, AND WHAT IT DELIBERATELY DOES NOT
---------------------------------------------------
It gives the FSM **inventory**: which registers are control state, how many states each has, and
which state the design resets into. That is the hard half, and it is the half that finds units with
no top-level busy port — the class of unit whose invisibility inflated an idle figure by thirty
points on one target before anyone noticed it existed.

It does NOT give the idle *value*. A transition table names states symbolically (``s0``, ``s1``)
while a trace carries the register's numeric encoding, and inventing a mapping between them would be
exactly the baked-encoding guess the gates forbid. The numeric idle value is instead **observed**:
reset the model and read the register (:func:`reset_values`). That is a measurement of the design,
needs no encoding table, and is why nothing here has to declare that zero means idle.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

__all__ = ["FsmRegister", "fsm_inventory", "kiss2_dir_for", "reset_values"]

#: What a synthesis FSM export names its files. ``<Module>.<register>.kiss2``.
_SUFFIX = ".kiss2"


@dataclass(frozen=True)
class FsmRegister:
    """One control-state register the RTL declares."""

    module: str                 # the RTL module that owns it, e.g. "LoadController"
    register: str               # the state register, e.g. "control_state"
    #: How many states the transition table declares -- None when the FSM was detected but its table
    #: was not exported. None means UNKNOWN, never zero: the machine exists either way.
    states: int | None = None
    #: The symbolic state the design resets into, when a table was exported. Not the numeric idle
    #: encoding -- see :func:`reset_values`, which observes that instead of mapping to it.
    reset_state: str | None = None
    exported: bool = False      # a transition table exists for it
    source: str = "yosys-fsm"

    @property
    def qualified(self) -> str:
        return f"{self.module}.{self.register}"

    def matches_signal(self, path: str) -> bool:
        """Whether a traced signal path is plausibly THIS register.

        A synthesis export names the module CLASS; a simulator's state manifest names the INSTANCE
        path. The two meet at the register leaf, so the match is on the leaf plus a containing
        instance -- structural, and it never invents a path that the manifest does not already have.
        """
        parts = path.replace("/", ".").split(".")
        return len(parts) >= 2 and parts[-1] == self.register


def kiss2_dir_for(target: str) -> Path | None:
    """Where this target's exported transition tables live, or None when none were exported."""
    root = os.environ.get("MERLIN_FSM_DIR")
    if root:
        d = Path(root) / target
        return d if d.is_dir() else None
    mlc = os.environ.get("MERLIN_MLC_DIR")
    if not mlc:
        return None
    d = Path(mlc) / "runs" / "yosys-fsm" / target
    return d if d.is_dir() else None


def _parse_header(path: Path) -> tuple[int, str] | None:
    """``(state_count, reset_state)`` from a transition table's header, or None if it declares none.

    Structural line dispatch on the leading key -- no pattern matching, and a table missing either
    key yields None rather than a defaulted count.
    """
    states = reset = None
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line.startswith("."):
                break                       # the header is over; transitions follow
            key, _, rest = line.partition(" ")
            if key == ".s":
                try:
                    states = int(rest.strip())
                except ValueError:
                    return None
            elif key == ".r":
                reset = rest.strip()
    if states is None or not reset:
        return None
    return states, reset


#: How the synthesis flow mangles an exported table's filename: ``<Module>-$fsm$\\<register>$<id>``.
#: A curated export renames these to ``<Module>.<register>``, so both spellings appear in practice and
#: the reader must accept either or an exported table silently fails to join its detected entry.
_MANGLE = "-$fsm$"


def _table_name(stem: str) -> tuple[str, str]:
    """``(module, register)`` from an exported table's filename, either spelling."""
    if _MANGLE in stem:
        module, _, rest = stem.partition(_MANGLE)
        register = rest.lstrip("\\").rpartition("$")[0] or rest.lstrip("\\")
        return module, register
    module, _, register = stem.rpartition(".")
    return (module, register) if module else ("", stem)


#: What the extraction reports for each state machine it found, in the tool's own words.
_DETECTED = "Extracting FSM"
_FROM = "from module"


def detected_registers(log_text: str) -> list[tuple[str, str]]:
    """``(module, register)`` for every state machine the extraction reports finding.

    Parsed structurally by line dispatch and ``partition`` -- the tool states one per line, and this
    reads that statement rather than pattern-matching around it.
    """
    out: list[tuple[str, str]] = []
    for line in log_text.splitlines():
        line = line.strip()
        if not line.startswith(_DETECTED) or _FROM not in line:
            continue
        reg_part, _, mod_part = line.partition(_FROM)
        reg = reg_part[len(_DETECTED):].strip().strip("`'\\ ")
        mod = mod_part.strip().rstrip(".").strip("`'\\ ")
        if reg and mod:
            out.append((mod, reg))
    return out


def fsm_inventory(target: str, kiss2_dir: Path | str | None = None) -> list[FsmRegister]:
    """Every control-state register the extraction found for ``target``, sorted.

    An empty list means no extraction was found -- a statement about the extraction, NOT about the
    design. A target with no inventory has UNKNOWN engines, not zero, and callers must keep those
    apart or a design nobody analysed reads as a design with nothing in it.
    """
    d = Path(kiss2_dir) if kiss2_dir else kiss2_dir_for(target)
    if d is None or not d.is_dir():
        return []

    tables: dict[tuple[str, str], tuple[int, str]] = {}
    for p in sorted(d.glob(f"*{_SUFFIX}")):
        parsed = _parse_header(p)
        if parsed is not None:
            tables[_table_name(p.name[: -len(_SUFFIX)])] = parsed

    found: dict[tuple[str, str], None] = {}
    for log in sorted(d.glob("*.log")):
        for pair in detected_registers(log.read_text(errors="replace")):
            found[pair] = None
    for pair in tables:                     # a table with no log line still counts as found
        found.setdefault(pair, None)

    out: list[FsmRegister] = []
    for module, register in sorted(found):
        t = tables.get((module, register))
        out.append(FsmRegister(module=module, register=register,
                               states=(t[0] if t else None), reset_state=(t[1] if t else None),
                               exported=t is not None))
    return out


def reset_values(peek, signals) -> dict[str, str]:
    """The value each signal holds immediately after reset -- the design's own idle encoding.

    Read rather than declared. A transition table names its reset state symbolically and a trace
    carries a numeric encoding; rather than map between them, ask the model what the register
    actually holds once reset. ``peek`` is any callable taking a signal path and returning its value.
    """
    return {s: str(peek(s)) for s in signals}
