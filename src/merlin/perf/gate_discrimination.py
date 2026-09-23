"""Whether a functional gate CAN fail, asked of the gate's own specification before it is used.

A gate is evidence only in the direction it can refuse. The defect this looks for has a plain
shape: a gate that checks ``bad=0`` and never checks how many values were compared is passed by a
program that compared none. Nothing about the candidate has to be wrong for that gate to be
useless, and nothing in a passing result shows it.

So the specification is run against transcripts it must REFUSE, built from the specification
itself and from nothing else (no field name is interpreted):

``reports_zeros``     every gated numeric field reads 0 and every text field is empty, with the
                      harness's PASS line: a program that measured nothing and said so politely;
``stops_before_pass`` every gated field matches and the PASS line is absent: a program that died
                      after reporting.

A mutation the gate does not refuse is returned by name. A configuration either fixes its
expectations or lists the name under ``accept_undetected_mutations``, where it is visible debt
rather than an unknown hole.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

MUTATIONS = ("reports_zeros", "stops_before_pass")
ACCEPT_KEY = "accept_undetected_mutations"


def _satisfying(expected: Any) -> str:
    """A printed value the expectation accepts."""
    if isinstance(expected, Mapping):
        if "equals" in expected:
            return str(expected["equals"])
        radix = int(expected.get("radix", 10))
        bound = expected.get("min", expected.get("max", 0))
        return format(int(bound), "x") if radix == 16 else str(int(bound))
    return str(expected)


def undetected(gate_spec) -> list[str]:
    """Names of the mutations ``gate_spec`` would PASS. Empty means the gate can fail on both."""
    from . import functional_gate as FG

    names = list(gate_spec.expectations)
    line = FG.RESULT_PREFIX + " placeholder=1"

    def verdict(fields: dict[str, str], harness: str | None) -> str:
        parsed = FG.ParsedTranscript(
            fields=fields, result_lines=(line,), verdict=harness, verdict_line=harness, duplicate_fields={}
        )
        return FG.evaluate_transcript(parsed, gate_spec)[0]

    found: list[str] = []
    zeros = {name: ("" if isinstance(gate_spec.expectations[name], str) else "0") for name in names}
    for name, expected in gate_spec.expectations.items():
        if isinstance(expected, Mapping) and isinstance(expected.get("equals"), str):
            zeros[name] = ""
    if verdict(zeros, "PASS") == FG.STATUS_PASSED:
        found.append("reports_zeros")
    matching = {name: _satisfying(expected) for name, expected in gate_spec.expectations.items()}
    if verdict(matching, None) == FG.STATUS_PASSED:
        found.append("stops_before_pass")
    return found


def require_discriminating(gate_spec, document: Mapping[str, Any] | None = None) -> list[str]:
    """Raise unless every undetected mutation is declared; return the declared ones."""
    declared = [str(name) for name in ((document or {}).get(ACCEPT_KEY) or ())]
    unknown = sorted(set(declared) - set(MUTATIONS))
    if unknown:
        raise ValueError(f"{ACCEPT_KEY} names mutations nobody runs: {unknown} (known: {list(MUTATIONS)})")
    open_holes = [name for name in undetected(gate_spec) if name not in declared]
    if open_holes:
        raise ValueError(
            f"this functional gate cannot fail on {open_holes}: a program that measured nothing "
            f"(or stopped before its verdict) would pass it. Gate a field that must be NON-ZERO "
            f"(how many values were compared), require the PASS line, or declare the hole under "
            f"`{ACCEPT_KEY}`"
        )
    stale = sorted(set(declared) - set(undetected(gate_spec)))
    if stale:
        raise ValueError(f"{ACCEPT_KEY} lists {stale}, which this gate now detects; remove them")
    return declared
