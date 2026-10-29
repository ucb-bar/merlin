"""Inspect pinned graph source/data packets without importing project code.

Run from the repository root with python3 -I -S -B. Native tools and old
absolute receipt paths are never opened. This is an intake audit, not a proof.
"""

import collections
import datetime
import hashlib
import json
import pathlib
import subprocess


ROOT = pathlib.Path.cwd().resolve()
OUTPUT = pathlib.Path(__file__).resolve().parent
BASELINE = "c62f81a8a11347ff53b60101fa75280e69d002a6"
REVIEW = "5a7f872db33d378d6b0c4a294e4099e80efd4956"
REVIEW_BASE = "58dbc7c00acdb6770d440b773cc2954328c78472"
ARTIFACTS = pathlib.Path("out/artifacts/verification")
DEADLINE = ARTIFACTS / "merlin-passes/progress/planner/paper-deadline-20261026"
GATE = ARTIFACTS / "graph-pipeline-gate/development"
MEMORY = ARTIFACTS / "graph-shape-normalization/after-binding-deadline-v1/final-memory-after-v1"
STAGING = DEADLINE / "independent-reproduction/fresh-graph-staging-v1"
ROWS = []


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT)


def load(rel):
    return json.loads((ROOT / rel).read_bytes())


def check(label, rel, expected=None):
    rel = pathlib.Path(rel)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError("Only repository-relative input paths are admitted")
    p = ROOT / rel
    if not p.resolve().is_relative_to(ROOT):
        raise ValueError("Input resolves outside the selected checkout")
    if p.suffix not in {".py", ".lean", ".json", ".md", ".patch", ".mlir", ".csv", ".txt"}:
        raise ValueError("Non-source/data input refused: " + str(rel))
    row = {"label": label, "path": str(rel)}
    if not p.is_file():
        row["status"] = "MISSING"
    else:
        data = p.read_bytes()
        row.update(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
        row["status"] = "PRESENT"
        if expected:
            row["expected"] = expected
            size = expected.get("bytes", expected.get("size"))
            row["status"] = "MATCH" if (
                row["sha256"] == expected["sha256"] and (size is None or len(data) == size)
            ) else "MISMATCH"
    ROWS.append(row)
    return row


def summarize(rows):
    return dict(collections.Counter(r["status"] for r in rows))


def write_new(name, value):
    with (OUTPUT / name).open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


if git("rev-parse", "HEAD").decode().strip() != BASELINE:
    raise ValueError("Working tree is not at the selected baseline")

review_files = [
    "deliverable-1/coverage-review.md", "deliverable-1/gaps.md",
    "deliverable-1/requirements-audit.md", "deliverable-1/correctness-audit.md",
    "deliverable-1/coverage.csv", "deliverable-1/source-inventory.csv",
    "deliverable-1/final-review.json", "deliverable-2/ir-map.md",
    "deliverable-2/omissions-bugs.md", "deliverable-2/review-identities.json",
    "deliverable-3/assumption-review.md", "deliverable-3/manifest.json",
    "deliverable-3/source-identities.json",
]
review_inputs = []
for name in review_files:
    original = str(ARTIFACTS / "graham-graph-review" / name)
    data = git("show", REVIEW + ":" + original)
    destination = OUTPUT / "inputs/reviewer" / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as stream:
        stream.write(data)
    review_inputs.append({
        "commit": REVIEW, "original_path": original,
        "local_path": str(destination.relative_to(ROOT)),
        "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
    })

q_path = GATE / "publication-cut-v1/SOURCES.json"
q = load(q_path)
q_rows = []
for row in q["sources"]:
    q_rows.append(check("Q fixed snapshot", row["snapshot"], row))
    if row.get("owner") == "primary" or row.get("owner") is None:
        check("Q canonical source", row["path"], row)
for cut in q["source_cuts"]:
    check("Q inventory", cut["inventory"], {"sha256": cut["inventory_sha256"]})
    if "manifest" in cut:
        check("Q manifest", cut["manifest"], {"sha256": cut["manifest_sha256"]})

w_path = MEMORY / "publication-context-v1/MANIFEST.json"
w = load(w_path)
w_rows = [check("W owned source", row["path"], row) for row in w["owned_sources"]]
w_closures = {}
for name in ("source_inventory", "source_seal", "lean_closure", "python_closure", "queries"):
    ref = w[name]
    check("W " + name, ref["path"], ref)
    if name in ("lean_closure", "python_closure"):
        closure_rows = [check("W " + name + " canonical", r["path"], r) for r in load(ref["path"])]
        w_closures[name] = {"count": len(closure_rows), "canonical_status": summarize(closure_rows)}

spec_path = STAGING / "capsule/STAGING_SPEC.json"
spec = load(spec_path)
capsule_refs = {}
profiles = []
for profile in spec["profiles"]:
    plan_ref = profile["provenance"]["source_plan_ref"]
    check("staging source plan", STAGING / plan_ref["file"], plan_ref)
    plan = load(STAGING / plan_ref["file"])
    order = profile["dependency_order"]
    positions = {name: i for i, name in enumerate(order)}
    dag = plan["project_DAG"]
    dag_ok = len(positions) == len(order) and set(positions) == set(dag)
    dag_ok = dag_ok and all(
        dep in positions and positions[dep] < positions[name]
        for name, dependencies in dag.items() for dep in dependencies
    )
    refs = [r["ref"] for r in profile["sources"]]
    refs += [case[key] for case in profile["cases"] for key in ("query", "roots")]
    for ref in refs:
        prior = capsule_refs.setdefault(ref["file"], ref)
        if prior != ref:
            raise ValueError("Conflicting capsule identities")
    profiles.append({
        "id": profile["id"], "modules": len(profile["sources"]),
        "queries": len(profile["cases"]), "declared_DAG_topological": dag_ok,
        "kernel_compilation": "UNEXECUTED", "qualification": "UNEXECUTED",
    })
capsule_rows = [check("staging capsule payload", STAGING / "capsule" / r["file"], r)
                for r in capsule_refs.values()]

inline = GATE / "unroll-inline-v1"
inline_inventory = load(inline / "QUERY-INVENTORY.json")
inline_manifest = load(inline / "MANIFEST.json")
inline_index = {r["path"]: r for r in inline_manifest["rows"]}
inline_queries = []
for case in inline_inventory["cases"]:
    rel = inline_inventory["authoritative_examples"] + "/" + case["case"] + "/Query.lean"
    query_row = check("Q authoritative inline query", inline / rel, inline_index[rel])
    query_row["matches_query_inventory"] = query_row.get("sha256") == case["query_sha256"]
    inline_queries.append(query_row)

missing_links = []
for rel in (
    DEADLINE / "independent-reproduction/service-integration-v2/HANDOFF.md",
    DEADLINE / "independent-reproduction/host-consumer-bridge-v1/PREREQUISITES_AND_PLACEMENT.md",
):
    present_at = {}
    for revision in (BASELINE, REVIEW):
        present_at[revision] = subprocess.run(
            ["git", "cat-file", "-e", revision + ":" + str(rel)], cwd=ROOT,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        ).returncode == 0
    missing_links.append({"path": str(rel), "present_at": present_at})

for rel in (pathlib.Path("VERIFICATION_HANDOFF.md"), q_path, w_path, spec_path,
            pathlib.Path("src/merlin/verify/lean_graph_input_admission.py")):
    check("intake entrypoint", rel)

report = {
    "schema": "merlin.graham.a26-baseline-intake.v1",
    "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "checkout": str(ROOT), "baseline": BASELINE, "review_branch_tip": REVIEW,
    "reviewed_source_base": REVIEW_BASE,
    "common_ancestor": git("merge-base", BASELINE, REVIEW).decode().strip(),
    "review_branch_merged": False,
    "execution_scope": "Python stdlib and Git source/data inspection only",
    "new_Lean_compilation": False, "native_execution": False,
    "fresh_qualification": False, "human_approval": False,
    "old_runtime_hold": "UNCHANGED",
    "Q_fixed_snapshot_status": summarize(q_rows), "W_owned_status": summarize(w_rows),
    "W_closures": w_closures,
    "staging_capsule": {"unique_payloads": len(capsule_refs), "status": summarize(capsule_rows)},
    "profiles": profiles,
    "inline_queries": {"count": len(inline_queries), "status": summarize(inline_queries),
                       "inventory_hashes_match": all(r["matches_query_inventory"] for r in inline_queries)},
    "missing_integration_links": missing_links,
    "review_inputs": review_inputs,
    "checks_by_category": {label: summarize([r for r in ROWS if r["label"] == label])
                           for label in sorted({r["label"] for r in ROWS})},
    "file_checks": ROWS,
    "limits": [
        "Hash and declared-DAG checks establish source/data availability, not Lean correctness.",
        "Historical runtime/source/qualification identities are preserved and do not qualify new candidates.",
        "Absolute historical receipt/tool paths were treated as data and were never opened.",
        "The complete historical qualification bundles and all external SDK imports were not audited.",
        "Review findings at 58dbc7c00 require separate reconciliation against c62f81a8a.",
    ],
}
write_new("BASELINE_AUDIT.json", report)
print(json.dumps({k: report[k] for k in (
    "baseline", "common_ancestor", "Q_fixed_snapshot_status", "W_owned_status", "W_closures",
    "staging_capsule", "profiles", "inline_queries", "checks_by_category",
)}, indent=2))
