"""Which of the tools an arm was GRANTED the agent actually reached for.

The performance lane records its tool surface directly (``STAGE_CONTEXT.broker_actions``). The
functional lane does not: an arm is defined by a bundle of paths it may read, and whether the agent
ever opened them is a separate question the run does not answer anywhere. It is worth answering,
because granting a tool and the tool being used are different claims and only the second one is
evidence that the tool mattered.

The invocation count is derived from the command text of the run's own tool spans, matched against
the filenames the harness stages for each granted tool. That is a lower bound by construction: a
tool imported inside a script the agent wrote is used without ever appearing in a command line, and
this reader cannot see that. It is reported as a lower bound and never as "the agent did not use it".
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

from merlin.agentreport.availability import Availability, derived, unavailable
from merlin.agentreport.spans import SpanSet

#: The substrate every arm gets, whatever its bundle: a synchronous self-check and an asynchronous
#: oracle. Named here because they are staged by the run loop rather than by the tool registry.
SUBSTRATE = {"agent_selfcheck.py": "synchronous redacted self-check",
             "simjob.py": "asynchronous oracle (submit / poll / wait)"}


@dataclass
class ToolUse:
    name: str                 # the granted tool's registry name, or the substrate filename
    blurb: str = ""
    invocations: int = 0
    granted: bool = True
    invocable: bool = False   # does this grant expose a filename a command line could name?


@dataclass
class Phase1Tools:
    arm: str = ""
    tools: list[ToolUse] = field(default_factory=list)
    availability: Availability = field(default_factory=Availability)


def _stage_names(spec) -> list[str]:
    """Filenames the agent can actually type for this grant. Empty for a read-only path grant."""
    broker = getattr(spec, "broker", None)
    if broker is None:
        return []
    return [staged for _, staged in getattr(broker, "shims", ()) or ()]


def read_phase1_tools(spanset: SpanSet, arm_name: str, registry: Mapping,
                      arm_tools: Mapping[str, Sequence[str]]) -> Phase1Tools:
    """Granted tools for ``arm_name`` and how often each was named on a command line.

    ``registry`` and ``arm_tools`` are passed in rather than imported so this stays a pure function
    over the vocabulary its caller uses."""
    out = Phase1Tools(arm=arm_name)
    granted = arm_tools.get(arm_name)
    if granted is None:
        out.availability.set("phase1_tools", unavailable(
            f"arm {arm_name!r} is not in the tool registry, so what it was granted cannot be stated"))
        return out

    commands = [(sp.detail or "") for sp in spanset.spans]
    blob = "\n".join(commands)

    for filename, blurb in SUBSTRATE.items():
        out.tools.append(ToolUse(name=filename, blurb=blurb, invocations=blob.count(filename),
                                 granted=True, invocable=True))
    for name in granted:
        spec = registry.get(name)
        blurb = (getattr(spec, "blurb", "") or "").strip()
        names = _stage_names(spec)
        count = sum(blob.count(n) for n in names)
        out.tools.append(ToolUse(name=name, blurb=blurb, invocations=count, granted=True,
                                 invocable=bool(names)))

    if not spanset.spans:
        out.availability.set("phase1_tools", unavailable(
            spanset.availability.get("spans").reason
            or "this run has no tool spans, so nothing can be said about what it invoked"))
    else:
        out.availability.set("phase1_tools", derived(
            "counted by matching staged tool filenames in the command text of this run's own spans; "
            "a LOWER BOUND, because a tool imported inside a script the agent wrote never appears on "
            "a command line", source=spanset.source))
    return out
