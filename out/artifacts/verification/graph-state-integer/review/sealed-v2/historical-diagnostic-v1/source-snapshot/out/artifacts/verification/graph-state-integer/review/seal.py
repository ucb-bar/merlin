"""Source/evidence seal after the ONE fixed batch. No native runs or publication."""

from __future__ import annotations
import difflib
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path.cwd().resolve()
sys.path.insert(0, str(ROOT / "src"))
from merlin.common.paths import artifacts_dir, build_dir
from merlin.verify import lean_graph_state as stable
from merlin.verify.lean_graph_state_integer_kernel import context, PRIOR
from merlin.verify.lean_graph_state_integer_receipt import validate, collect_refs, CLAIMS

BASE = artifacts_dir() / "verification/graph-state-integer"
QUAL = BASE / "qualification-v1"
PACK = BASE / "review/sealed-v1"
INITIAL = "c50d6cb07a2100b09806b5a105eebc9ba23a189d"


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def retain(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    return stable.file_ref(target)


def main():
    batch = json.loads((QUAL / "BATCH.json").read_text())
    plan = json.loads((QUAL / "PLAN.json").read_text())
    if batch["status"] != "complete_conditional_graph_qualification" or len(batch["originals"]) != 17 or len(batch["fresh_replays"]) != 17:
        raise ValueError("complete frozen originals and ONE replay batch required")
    if not stable.fresh([*plan["pins"], batch["plan"]]):
        raise ValueError("frozen identities changed; no promotion")
    context()
    PACK.mkdir(parents=True, exist_ok=False)
    owned = plan["owned_sources"]
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    snapshots, patch, incremental = [], "", ""
    for row in owned:
        path = Path(row["path"])
        relative = str(path.relative_to(ROOT))
        if stable.file_ref(path) != row:
            raise ValueError("owned source differs from freeze")
        snapshots.append({"original": row, "retained": retain(path, PACK / "sources" / relative)})
        content = path.read_text()
        initial = subprocess.run(["git", "show", INITIAL + ":" + relative], cwd=ROOT, text=True, capture_output=True)
        if initial.returncode == 0:
            raise ValueError("new additive integer ownership crossed initial existing source")
        patch += f"diff --git a/{relative} b/{relative}\nnew file mode 100644\n"
        patch += "".join(difflib.unified_diff([], content.splitlines(keepends=True), fromfile="/dev/null", tofile="b/" + relative))
        old = subprocess.run(["git", "show", head + ":" + relative], cwd=ROOT, text=True, capture_output=True)
        previous = old.stdout if old.returncode == 0 else ""
        if previous != content:
            incremental += f"diff --git a/{relative} b/{relative}\n" + ("new file mode 100644\n" if old.returncode else "")
            incremental += "".join(difflib.unified_diff(previous.splitlines(keepends=True), content.splitlines(keepends=True),
                                                       fromfile="/dev/null" if old.returncode else "a/" + relative, tofile="b/" + relative))
    (PACK / "owned.patch").write_text(patch)
    (PACK / "incremental-owned.patch").write_text(incremental)
    check = subprocess.run(["git", "apply", "--check", "--reverse", str(PACK / "owned.patch")], cwd=ROOT, text=True, capture_output=True)
    (PACK / "patch-check.stdout").write_text(check.stdout)
    (PACK / "patch-check.stderr").write_text(check.stderr)
    if check.returncode:
        raise ValueError("owned patch does not exactly describe current source bytes")
    closure = {r["path"]: r for r in plan["pins"]}
    audits, statements, originals = [], {}, []
    for row in batch["originals"]:
        receipt = json.loads(Path(row["receipt"]["path"]).read_text())
        report, capture, proof, _ = validate(receipt)
        for ref in collect_refs(receipt):
            if ref["path"] in closure and closure[ref["path"]] != ref:
                raise ValueError("conflicting exact dependency closure")
            closure[ref["path"]] = ref
        audits.append({"name": row["name"], "theorems": proof["audit"]["raw"]["theorems_audited"],
                       "axioms": proof["audit"]["raw"]["axioms_used"], "violations": proof["audit"]["raw"]["violations"],
                       "roots": receipt["roots"]})
        statements[row["name"]] = proof["audit"]["statements"]
        originals.append({**row, "query": receipt["query"], "object": receipt["object"], "mode": receipt["mode"], "model": receipt["model"]})
    prior = []
    for family in PRIOR:
        packet = artifacts_dir() / "verification" / family
        manifest = json.loads((packet / "MANIFEST.json").read_text())
        delivery = json.loads((packet / "delivery.json").read_text())
        for row in manifest["files"]:
            path = Path(row["path"])
            path = path if path.is_absolute() else ROOT / path
            if not path.is_file() or stable.file_ref(path)["sha256"] != row["sha256"] or path.stat().st_size != row["bytes"]:
                raise ValueError("prior sealed capsule changed: " + str(path))
        for row in delivery["owned_sources"]:
            if stable.file_ref(row["original"]["path"]) != row["original"]:
                raise ValueError("prior source changed")
            closure[row["original"]["path"]] = row["original"]
        prior.append({"packet": str(packet.relative_to(ROOT)), "manifest": stable.file_ref(packet / "MANIFEST.json"),
                      "files": len(manifest["files"]), "sources": len(delivery["owned_sources"]), "mismatches": 0})
    deps = []
    own_paths = {r["path"] for r in owned}
    reference = ROOT.parent / "merlin-graph-proof-intake"
    for ref in closure.values():
        path = Path(ref["path"])
        if path.suffix not in (".py", ".lean") or str(path) in own_paths:
            continue
        if path.is_relative_to(reference):
            owner, relative = "pinned4499", path.relative_to(reference)
        elif path.is_relative_to(ROOT) and str(path.relative_to(ROOT)).startswith(("src/", "merlin/", "packages/", "build_tools/")):
            owner, relative = "primary", path.relative_to(ROOT)
        else:
            continue
        deps.append({"original": ref, "location": owner, "retained": retain(path, PACK / "dependency-sources" / owner / relative)})
    def aliases(directory, names):
        skipped = []
        for name in names:
            if (Path(directory) / name).is_symlink():
                if not name.endswith("current"):
                    raise ValueError("unaccounted delivery symlink")
                skipped.append(name)
        return skipped
    for tree in ("qualification-v1", "paper-evidence"):
        shutil.copytree(BASE / tree, PACK / tree, symlinks=True, ignore=aliases)
    for name in ("qualify.py", "seal.py"):
        retain(BASE / "review" / name, PACK / "controllers" / name)
    retain(BASE / "active/dependency-check-once.json", PACK / "fixed-dependency-check-once.json")
    if (BASE / "review/source-gates.json").exists():
        retain(BASE / "review/source-gates.json", PACK / "source-gates.json")
    # Keep small first raw failures and exact attempt sources, never rerun them.
    attempts = build_dir() / "graph-verification/state-integer/development/attempts"
    for result_path in attempts.glob("*/*/result.json"):
        result = json.loads(result_path.read_text())
        if result["returncode"] == 0:
            continue
        for path in result_path.parent.iterdir():
            if path.is_file():
                retain(path, PACK / "first-failures" / "module-attempts" / path.relative_to(attempts))
    for name in ("first-actual", "first-chain/model-v1"):
        source = BASE / "development" / name
        if source.is_dir():
            # first-actual only is small; first-chain/model-v1 is a known ordinary
            # proof refusal, preserved separately from qualified later queries.
            shutil.copytree(source, PACK / "first-failures" / name, symlinks=True)
    write(PACK / "AUDIT.json", audits)
    write(PACK / "STATEMENTS.json", statements)
    write(PACK / "dependencies.json", {"bindings": list(closure.values()), "project_source_snapshots": deps,
                                       "prior_capsules": prior, "prior_owned_sources": sum(r["sources"] for r in prior),
                                       "full_LeanMLIR_selected": False, "sdk_written": False})
    delivery = {"status": "SEALED_CONDITIONAL_GRAPH_QUALIFICATION", "checkout": str(ROOT), "branch": branch,
                "observed_HEAD": head, "initial_additive_base": INITIAL, "incremental_patch_base": head,
                "owned_sources": snapshots, "full_owned_patch": stable.file_ref(PACK / "owned.patch"),
                "incremental_owned_patch": stable.file_ref(PACK / "incremental-owned.patch"),
                "dependency_source_snapshots": len(deps), "tests": batch["tests"], "originals": originals,
                "fresh_replays": batch["fresh_replays"], "positive_replay_batches": 1,
                "changed_originals": sum(any(r["changed"]) for r in originals),
                "unchanged_originals": sum(not any(r["changed"]) for r in originals),
                "claims": dict(CLAIMS), "old75_sources_and_capsules_preserved": prior,
                "selected_models": [], "broker_selected": False, "review_approved": False,
                "broad_structure_gate": "foreign zephyr_model.py ceiling failure; owned source checks clean"}
    write(PACK / "delivery.json", delivery)
    (PACK / "README.md").write_text((BASE / "paper-evidence/README.md").read_text())
    (PACK / "REPRODUCE.md").write_text("""# Reproduction and integrity

Use PLAN/BATCH, delivery.json, dependencies.json and MANIFEST.json as exact
source/profile/intake/roster selectors. Qualification is the single frozen
66-test suite supplying17 originals, followed by ONE retained fresh-process
replay batch of17; the duplicate positive development replay is excluded.
Forgeries use owned copies with readonly original query/object references;
they do not rerun duplicate original proofs. Negative fresh audit probes are
separate from the positive replay batch.

Existing Linux tool: pinned Lean4.35 absolute executable/hash and original
unchanged standard/project object, sidecar and host bindings in PLAN. No
elan/lake/download/SDK rebuild/repair/reselection. Only std3, no reflection.
Full Lean-MLIR/broker route is not selected. Do not rerun qualify.py over this
directory; it refuses an existing batch. A separately authorized reproduction
needs a new owned root and fixed selection, never mutation of this seal.

owned.patch is source-only/additive against INITIAL in delivery; the incremental
patch uses the separately recorded observed HEAD. Source snapshots and explicit
transitive project source dependencies accompany them. Apply only the reviewed
owned cut; do not wholesale-merge pinned4499 or restore old archive sources.
Coordinator alone owns staging/commits/push/integration. Hashing this packet or
git apply --check --reverse is read-only integrity, not requalification.

The receipt binds exact original paths, source bytes, semantic projection,
canonical model and claims, query/object, roots/statements/import/audit and
tool/profile identities. Replay reopens original buffers, reconstructs the
query, reproduces the object at its exact original source path into private
output, fresh-checks/audits and postchecks identities. Hashes are not issuer
authenticity or runtime delivery. Old absolute Linux receipts on a laptop are
historical evidence, not a portability fix or laptop qualification.

All75 prior sources/capsules/patches/failures, unknown SDK incident and elision
ZERO qualification remain. Independent review, primitive/representation and
poison/runtime/native/model/broker coverage remain separate obligations.
""")
    if not stable.fresh([*plan["pins"], batch["plan"]]):
        raise ValueError("identity changed during sealing")
    rows = []
    for path in sorted(PACK.rglob("*")):
        if path.is_file() and not path.is_symlink():
            ref = stable.file_ref(path)
            rows.append({**ref, "path": str(path.relative_to(ROOT))})
    write(PACK / "MANIFEST.json", {"schema": "merlin.graph-state-integer.sealed.v1", "files": rows,
                                  "review_approved": False, "native_or_model_fidelity_proved": False})
    print(json.dumps({"packet": str(PACK), "files": len(rows), "owned_sources": len(owned),
                      "manifest": stable.file_ref(PACK / "MANIFEST.json")}))


if __name__ == "__main__":
    main()
