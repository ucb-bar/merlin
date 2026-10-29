"""Summarize reviewer `#print axioms` output and compare with retained per-root audits.

Read-only over retained evidence. Writes axiom-summary.json next to this script.
Run: .venv/bin/python -I summarize.py
"""

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[5]
V = REPO / "out/artifacts/verification"
ALLOWED = {"propext", "Classical.choice", "Quot.sound"}


def parse(path):
    result = {}
    for line in path.read_text().splitlines():
        if not line.startswith("'"):
            continue
        name, _, rest = line[1:].partition("'")
        if "does not depend on any axioms" in rest:
            result[name] = []
        elif "depends on axioms:" in rest:
            inside = rest.split("[", 1)[1].rsplit("]", 1)[0]
            result[name] = [a.strip() for a in inside.split(",") if a.strip()]
    return result


def retained():
    rows = {}
    shape = json.loads((V / "graph-shape-normalization/paper-evidence/CHECKER_OUTPUT.json").read_text())
    rows["shape"] = shape["transitive_axioms"]
    memory = json.loads((V / "graph-memory-scf/memory-scf-mode-v1/AUDIT.json").read_text())
    rows["memory_scf"] = memory["main_roots"]
    declared = json.loads((V / "graph-regions-declared/qualification-v1/fresh-replay/0/AUDIT.json").read_text())
    rows["declared"] = declared
    pipeline = json.loads((V / "graph-state-pipeline/review/sealed-v1/AUDIT.json").read_text())
    rows["pipeline_union"] = {row["name"]: row["axioms"] for row in pipeline["actual_roots"]}
    return rows


def main():
    logs = HERE / "build/logs"
    mine = {}
    for path in sorted(logs.glob("print-*.stdout")) + sorted(logs.glob("witness-*.stdout")):
        mine[path.name] = parse(path)
    flat = {name: axioms for table in mine.values() for name, axioms in table.items()}
    violations = {name: axioms for name, axioms in flat.items() if set(axioms) - ALLOWED}
    comparisons = {}
    for row, table in retained().items():
        if row == "pipeline_union":
            continue
        compared = {}
        for name, axioms in table.items():
            key = name if name in flat else ("GraphMemorySCFObligation." + name if False else name)
            if key in flat:
                compared[name] = {"retained": sorted(axioms), "reviewer": sorted(flat[key]),
                                  "equal": sorted(axioms) == sorted(flat[key])}
        comparisons[row] = {
            "retained_roots": len(table),
            "rechecked_roots": len(compared),
            "mismatches": {k: v for k, v in compared.items() if not v["equal"]},
            "not_rechecked": sorted(set(table) - set(compared)),
        }
    module_rows = [line.split("\t") for line in (logs / "modules.tsv").read_text().splitlines()]
    summary = {
        "reviewer": "Graham independent review (agent)",
        "date": "2026-10-26",
        "lean": (logs / "lean-version.txt").read_text().strip(),
        "steps": len(module_rows),
        "nonzero_steps": [row for row in module_rows if row[1] != "0"],
        "theorems_printed": len(flat),
        "non_std3": violations,
        "per_file": mine,
        "retained_comparison": comparisons,
        "reviewer_audit": {
            path.name: [line for line in path.read_text().splitlines()
                        if line.split(":")[0] in {"roots", "theorems_in_scope", "axiom_use_counts", "non_std3_axioms",
                                                  "compiler_unsafe_rec_aux_of_safe_defs", "findings"}]
            for path in sorted(logs.glob("reviewer-audit-*.stdout"))
        },
        "inputs_sha256": {
            str(path.relative_to(REPO)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [HERE / "build-order.txt", HERE / "queries.txt", HERE / "queries-inhabited.txt"]
            + sorted((HERE / "print").glob("*.lean")) + sorted((HERE / "print-inhabited").glob("*.lean"))
            + sorted((HERE.parent / "witness").glob("*.lean")) + [HERE / "ReviewerAudit.lean"]
        },
    }
    (HERE / "axiom-summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: summary[k] for k in ("steps", "nonzero_steps", "theorems_printed", "non_std3")}))
    for row, value in comparisons.items():
        print(row, value["retained_roots"], value["rechecked_roots"], len(value["mismatches"]), value["not_rechecked"][:12])
        for name, diff in value["mismatches"].items():
            print("   MISMATCH", name, diff)


if __name__ == "__main__":
    main()
