"""Show that audit_scalar.py detects wrong correspondences (non-vacuity check).

Graham independent review (agent), 2026-10-26. Mutated copies of the sealed
worked-example query/projection are written under probes/audit-sensitivity/
and audited; every mutation must yield a reported problem.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_scalar as A  # noqa: E402

OUT = A.D2 / "probes/audit-sensitivity"
LEAN = A.WORKED / "checked/kernel/GraphStateScalarInstance.lean"
PROJECTION = A.WORKED / "checked/projection/projection.json"


def lean_mutation(old, new, only_line_prefix):
    lines = LEAN.read_text().splitlines(keepends=True)
    hits = 0
    for k, line in enumerate(lines):
        if line.startswith(only_line_prefix) and old in line:
            lines[k] = line.replace(old, new, 1)
            hits += 1
            break
    assert hits == 1, (old, only_line_prefix)
    return "".join(lines)


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    mutations = {
        "after-operand-order": ("lean", lean_mutation("(.pure (.add 8) [3,4]", "(.pure (.add 8) [4,3]", "def after")),
        "after-upper-step-descriptor-swap": ("lean", lean_mutation("(.opaque 3 [] [.index] (.opaque 2",
                                                                   "(.opaque 2 [] [.index] (.opaque 3", "def after")),
        "registry-width": ("lean", lean_mutation("{inputs := [(.bits 8),.index]", "{inputs := [(.bits 16),.index]",
                                                 "def registry")),
        "middle-observer-operand": ("lean", lean_mutation("(.opaque 0 [1,2]", "(.opaque 0 [2,1]", "def middleRight")),
    }
    projection = json.loads(PROJECTION.read_text())
    projection["functions"][0]["after"][-1]["ret"] = [1]
    mutations["projection-record-return"] = ("projection", json.dumps(projection))
    rows = []
    for name, (kind, text) in mutations.items():
        folder = OUT / name
        folder.mkdir()
        paths = {"before": A.WORKED / "checked/before.mlir", "middle": A.WORKED / "checked/middle.mlir",
                 "after": A.WORKED / "checked/after.mlir", "lean": LEAN, "projection": PROJECTION}
        target = folder / ("GraphStateScalarInstance.lean" if kind == "lean" else "projection.json")
        target.write_text(text)
        paths[kind] = target
        result = A.audit_case(name, paths)
        rows.append({"mutation": name, "verdict": result["verdict"], "problems": result["problems"][:3],
                     "serializer_problems": result["serializer_problems"][:3],
                     "differential_mismatches": result["differential"]["mismatch_count"]})
    (OUT / "SENSITIVITY.json").write_text(json.dumps(rows, indent=2) + "\n")
    for row in rows:
        print(row["mutation"], row["verdict"], len(row["problems"]), len(row["serializer_problems"]),
              row["differential_mismatches"])


if __name__ == "__main__":
    main()
