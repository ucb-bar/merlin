"""Conservative expected-obligation attribution; never acceptance authority."""

from pathlib import Path

from loop_transform_qualification_classifier import RESOURCE_OR_INFRA

MATERIALIZATION_CASES = (
    "final_wrong_seed",
    "final_wrong_tile_stride",
    "final_wrong_load_address",
    "final_wrong_store_address",
    "final_wrong_scalar_return",
    "final_wrong_observer_operand",
    "final_early_free",
    "final_missing_free",
    "final_alignment1",
    "final_visibility",
    "final_entry_renamed",
    "final_missing_unchanged_entry",
    "final_observer_arguments_order",
    "final_observer_reordered",
    "final_missing_observer",
)
EXPECTED = {name: "GraphMemoryAfterMaterialization.accepted" for name in MATERIALIZATION_CASES}
EXPECTED.update(
    {
        "source_wrong_seed": "GraphMemoryAfterWhole.source_term_exact",
        "source_wrong_scalar_return": "GraphMemoryAfterWhole.source_term_exact",
        "final_wrong_inner_cutoff": "GraphMemorySCFObligation.valid",
    }
)


def classify(label, returncode, stdout, stderr, query, path):
    if returncode != 1:
        return {"status": "UNAVAILABLE", "reason": "not an ordinary exit1"}
    combined = (stdout + "\n" + stderr).lower()
    if any(
        marker in combined
        for marker in (*RESOURCE_OR_INFRA, "stack space", "failed to open", "uncaught exception", "error while loading")
    ):
        return {"status": "UNAVAILABLE", "reason": "resource/import/internal/unknown execution failure"}
    if label not in EXPECTED:
        return {"status": "UNAVAILABLE", "reason": "no reviewed expected obligation"}
    prefix = str(Path(path)) + ":"
    errors = [line for line in stdout.splitlines() if line.startswith(prefix) and ": error: " in line]
    if not errors:
        return {"status": "UNAVAILABLE", "reason": "missing actual bound source diagnostic"}
    number, _, rest = errors[0][len(prefix) :].partition(":")
    column, _, message = rest.partition(":")
    if not number.isdigit() or not column.isdigit():
        return {"status": "UNAVAILABLE", "reason": "unrecognized source diagnostic"}
    line = int(number)
    lines = query.splitlines()
    if not 1 <= line <= len(lines):
        return {"status": "UNAVAILABLE", "reason": "source diagnostic outside bound query"}
    namespace, root = [], None
    for source in lines[:line]:
        words = source.split()
        if words[:1] == ["namespace"]:
            namespace.append(words[1])
        elif words[:1] == ["end"] and namespace and (len(words) == 1 or words[1] == namespace[-1]):
            namespace.pop()
        elif words[:1] == ["theorem"]:
            root = ".".join([*namespace, words[1]])
    if root != EXPECTED[label]:
        return {"status": "UNAVAILABLE", "reason": "a different obligation failed", "root": root}
    message = message.strip().removeprefix("error: ")
    if not (
        message.startswith(("unsolved goals", "Tactic `rfl` failed:", "Type mismatch", "type mismatch"))
        or message.startswith("Tactic `decide` proved that the proposition")
        and "is false" in stdout
    ):
        return {"status": "UNAVAILABLE", "reason": "unrecognized proof diagnostic", "root": root}
    return {"status": "EXPECTED_COMPONENT_OBLIGATION_REFUSAL", "root": root, "line": line, "diagnostic": message}
