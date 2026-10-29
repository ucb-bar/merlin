"""Conservative classification of retained mutated-query diagnostics, SDK-free.

An ordinary proof-obligation refusal is not a semantic counterexample. Unknown
execution/import/resource errors are unavailable, regardless of return code1.
"""

from pathlib import Path

EXPECTED = {
    "middle_yield_order": "body_group",
    "final_return_order": "suffixAccepted34",
    "middle_lane_iv": "body_group",
    "final_wrong_capture": "innerAccepted",
    "final_wrong_body": "innerAccepted",
    "middle_wrong_call": "body_group",
    "final_partial_effect_order": "innerAccepted",
    "final_missing_observer": "suffixAccepted34",
    "final_wrong_inner_cutoff": "outer_form",
}
RESOURCE_OR_INFRA = (
    "maximum number of heartbeats",
    "maximum recursion depth",
    "deterministic) timeout",
    "out of memory",
    "stack overflow",
    "internal error",
    "panic",
    "interrupted",
    "segmentation fault",
    "unknown module prefix",
    "could not resolve module",
    "failed to import",
    "failed to read",
    "object file",
    "invalid header",
    "unknown constant",
    "unknown identifier",
    "unexpected token",
)


def classify(label, returncode, stdout, stderr, query, path):
    if returncode != 1:
        return {"classification": "UNAVAILABLE", "reason": "no ordinary error exit1"}
    combined = (stdout + "\n" + stderr).lower()
    if any(marker in combined for marker in RESOURCE_OR_INFRA):
        return {"classification": "UNAVAILABLE", "reason": "resource/import/internal/unknown execution failure"}
    if label not in EXPECTED:
        return {"classification": "UNAVAILABLE", "reason": "no reviewed expected obligation for this mutation"}
    prefix = str(Path(path)) + ":"
    first = None
    for line in stdout.splitlines():
        if line.startswith(prefix) and ": error: " in line:
            rest = line[len(prefix) :]
            number, _, rest = rest.partition(":")
            column, _, message = rest.partition(":")
            if not number.isdigit() or not column.isdigit():
                return {"classification": "UNAVAILABLE", "reason": "unknown diagnostic location"}
            first = int(number), message.strip().removeprefix("error: ")
            break
    if first is None:
        return {"classification": "UNAVAILABLE", "reason": "missing actual bound-query diagnostic"}
    number, message = first
    namespace = []
    theorem = None
    lines = query.splitlines()
    if not 1 <= number <= len(lines):
        return {"classification": "UNAVAILABLE", "reason": "diagnostic outside bound source"}
    for line in lines[:number]:
        words = line.split()
        if words[:1] == ["namespace"]:
            namespace.append(words[1])
        elif words[:1] == ["end"] and namespace:
            namespace.pop()
        elif words[:1] == ["theorem"]:
            theorem = ".".join([*namespace, words[1]])
    if theorem is None or theorem.rsplit(".", 1)[-1] != EXPECTED[label]:
        return {"classification": "UNAVAILABLE", "reason": "a different obligation failed", "root": theorem}
    ordinary = (
        message.startswith("unsolved goals")
        or message.startswith("Tactic `rfl` failed:")
        or message.startswith("Type mismatch")
        or message.startswith("type mismatch")
        or message.startswith("Tactic `decide` proved that the proposition")
        and "is false" in stdout
    )
    if not ordinary:
        return {"classification": "UNAVAILABLE", "reason": "unrecognized proof diagnostic", "root": theorem}
    return {
        "classification": "LEAN_COMPONENT_OBLIGATION_REFUSAL",
        "root": theorem,
        "line": number,
        "diagnostic": message,
    }
