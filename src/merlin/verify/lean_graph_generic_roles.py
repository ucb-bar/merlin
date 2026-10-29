"""Per-occurrence selected-control roles; ordinary events stay partial/opaque."""

from __future__ import annotations

import copy


def bind(capture, fi, cut, controls):
    original = copy.deepcopy(capture["functions"])
    old_size = len(capture["registry"])
    control_map = {row["identity"]: row for row in controls}
    selected = set()

    def collect(rows):
        for row in rows:
            if "effect" in row and row["effect"] in control_map:
                selected.add(row["effect"])
            elif "loop" in row:
                collect(row["loop"]["body"])

    for stage in capture["functions"]:
        collect(stage[fi]["body"][cut:])
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
    for stage in capture["functions"]:
        occurrence = 0
        marks = []

        def retag(rows):
            nonlocal occurrence
            for row in rows:
                if "effect" in row:
                    old = row["effect"]
                    if old in identities:
                        marks.append({"occurrence": occurrence, "original": old, "selected": identities[old]})
                        row["effect"] = identities[old]
                    occurrence += 1
                elif "loop" in row:
                    retag(row["loop"]["body"])

        retag(stage[fi]["body"][cut:])
        plans.append(marks)
    return {
        "unroled_functions": original,
        "raw_registry_size": old_size,
        "plans": plans,
        "index_operations": result_controls,
    }
