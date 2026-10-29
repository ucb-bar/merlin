"""Check the transferred a26 report bytes and Markdown links without SDK access."""

import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.root.resolve()
    review = root / "out/artifacts/verification/graham-graph-review"
    execution = review / "execution-a26-20261026-v1"
    baseline = review / "baseline-a26-20261026-v1"
    issues = []

    def selected(path):
        candidate = (root / path).resolve()
        if not candidate.is_relative_to(root):
            raise ValueError("Input resolves outside selected checkout")
        return candidate

    def same(data, row):
        return len(data) == row["bytes"] and hashlib.sha256(data).hexdigest() == row["sha256"]

    publication = json.loads((execution / "PUBLICATION_INDEX.json").read_bytes())
    extensions = {}
    for row in publication["copied_files"]:
        path = selected(row["path"])
        if not path.is_file():
            issues.append({"path": row["path"], "error": "missing copied file"})
            continue
        data = path.read_bytes()
        if not same(data, row):
            issues.append({"path": row["path"], "error": "copied bytes changed"})
        if b"\0" in data or path.suffix in {".olean", ".ilean", ".ir", ".so", ".o", ".a", ".zip", ".pyc"}:
            issues.append({"path": row["path"], "error": "binary/native payload"})
        data.decode("utf-8")
        if path.suffix == ".json":
            json.loads(data)
        elif path.suffix == ".py":
            ast.parse(data, filename=str(path))
        extensions[path.suffix] = extensions.get(path.suffix, 0) + 1

    intake = json.loads((baseline / "BASELINE_AUDIT.json").read_bytes())
    for row in intake["review_inputs"]:
        data = subprocess.check_output(
            ["git", "show", row["commit"] + ":" + row["original_path"]], cwd=root
        )
        if not same(data, row):
            issues.append({"path": row["original_path"], "error": "historical canonical review mismatch"})

    manifest_checks = {}
    for directory, filename in (("agent-1", "MANIFEST.json"), ("agent-2", "OUTPUT_MANIFEST.json"),
                                ("agent-3", "OUTPUT_MANIFEST.json")):
        manifest = json.loads((execution / directory / filename).read_bytes())
        for row in manifest["files"]:
            path = selected(row["path"])
            if not path.is_file() or not same(path.read_bytes(), row):
                issues.append({"path": row["path"], "error": "worker evidence manifest mismatch"})
        manifest_checks[directory] = len(manifest["files"])

    documents = [review / "README.md", baseline / "README.md", execution / "README.md",
                 execution / "PUBLICATION.md", execution / "agent-1/README.md",
                 execution / "agent-2/README.md"]
    documents += [review / ("deliverable-" + str(i)) / "UPDATE-20261026.md" for i in (1, 2, 3)]
    documents += [root / "out/artifacts/verification/merlin-passes/progress/planner/graph-first-20261025/GRAHAM_HANDOFF.md"]
    link_count = 0
    for document in documents:
        for tail in document.read_text().split("](")[1:]:
            target = tail.partition(")")[0]
            if target.startswith(("https://", "http://", "#")):
                continue
            target = target.partition("#")[0]
            path = (document.parent / target).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                issues.append({"path": str(document.relative_to(root)), "error": "broken link", "target": target})
            link_count += 1

    summary = json.loads((execution / "agent-3/REPRODUCTION_SUMMARY.json").read_bytes())
    totals = summary["totals"]
    if totals["case_queries_selected"] != sum(totals[key] for key in (
            "case_queries_compiled_and_exactly_audited", "case_queries_failed", "case_queries_unexecuted")):
        issues.append({"error": "query counts do not reconcile"})
    with (execution / "agent-3/CASE_ROSTER.csv").open(newline="") as stream:
        roster = list(csv.DictReader(stream))
    if len(roster) != totals["case_queries_selected"] + totals["witness_queries_selected"]:
        issues.append({"error": "case/witness roster does not reconcile"})
    repair = summary["same_statement_tail_proof_successor"]
    data = (execution / "proposed-repairs/TailLoopAppendProofSuccessor.lean").read_bytes()
    if not same(data, repair["successor_source_pin"]):
        issues.append({"error": "portable tail repair differs from compiled source"})

    print(json.dumps({
        "schema": "merlin.graham.a26.publication-check.v1",
        "status": "PASS" if not issues else "FAIL",
        "copied_files_verified": len(publication["copied_files"]),
        "canonical_historical_inputs_verified": len(intake["review_inputs"]),
        "worker_manifest_files_verified": manifest_checks,
        "markdown_documents_checked": len(documents), "markdown_links_checked": link_count,
        "case_roster_rows": len(roster), "copied_file_extensions": extensions,
        "SDK_or_compiled_objects_read": False, "issues": issues,
    }, indent=2))
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
