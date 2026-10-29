"""Only actual control setup / generated outer add-min receive bounded roles.

The original body (including ordinary index arithmetic) stays opaque in both
source and inner clone, even when its opcode metadata matches a generated add.
"""

from __future__ import annotations

import copy


def bind(capture, fi, cut, controls):
    original = copy.deepcopy(capture["functions"])
    old_size = len(capture["registry"])
    control_map = {row["identity"]: row for row in controls}

    def eligible(stage, path):
        return (len(path) == 1 and path[0] < (3 if stage == 0 else 4)) or (
            stage == 1 and len(path) == 3 and path[:2] == (4, "body") and path[2] in (0, 1)
        )

    selected = set()

    def collect(rows, stage, path=()):
        for i, row in enumerate(rows):
            here = path + (i,)
            if "effect" in row and eligible(stage, here) and row["effect"] in control_map:
                selected.add(row["effect"])
            elif "loop" in row:
                collect(row["loop"]["body"], stage, here + ("body",))

    for stage, functions in enumerate(capture["functions"]):
        collect(functions[fi]["body"][cut:], stage)
    identities = {}
    result_controls = []
    for identity in sorted(selected):
        row = capture["registry"][identity]
        new = len(capture["registry"])
        identities[identity] = new
        capture["registry"].append(
            {
                "descriptor": {"role": "selected-bounded-scf-control-v1", "operation": row["descriptor"]},
                "inputs": row["inputs"],
                "outputs": row["outputs"],
            }
        )
        result_controls.append({**control_map[identity], "identity": new})
    plans = []
    for stage, functions in enumerate(capture["functions"]):
        occurrence = 0
        marks = []

        def retag(rows, path=()):
            nonlocal occurrence
            for i, row in enumerate(rows):
                here = path + (i,)
                if "effect" in row:
                    old = row["effect"]
                    if eligible(stage, here) and old in identities:
                        marks.append({"occurrence": occurrence, "original": old, "selected": identities[old]})
                        row["effect"] = identities[old]
                    occurrence += 1
                elif "loop" in row:
                    retag(row["loop"]["body"], here + ("body",))

        retag(functions[fi]["body"][cut:])
        plans.append(marks)
    return {
        "unroled_functions": original,
        "raw_registry_size": old_size,
        "plans": plans,
        "index_operations": result_controls,
    }
