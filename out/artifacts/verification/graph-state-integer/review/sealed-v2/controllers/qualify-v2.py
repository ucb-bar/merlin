"""NEW corrected-source v2 qualification; v1 is an immutable held diagnostic.

Known producer classification guard fixed; unchanged original SDK bindings.
ONE frozen integer qualification; suite originals then ONE positive replay batch.

No failed profile, unknown SDK binding or literal-timeout route is retried.
Negative forgery probes are distinct from the retained positive replay batch.
"""

from __future__ import annotations
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path.cwd().resolve()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "merlin/tests/ir")]
from merlin.common.paths import artifacts_dir
from merlin.common.owned_process import run_owned
from merlin.verify import lean_graph_state as stable
from merlin.verify.lean_graph_state_integer import author
from merlin.verify.lean_graph_state_integer_kernel import context
from merlin.verify.lean_graph_state_integer_proof import prepare, sources
from merlin.verify.lean_graph_state_integer_receipt import validate, load
from test_lean_graph_state_integer import produce, standalone_source, chain_source, nested_source, identity_source, REFERENCE
from test_lean_graph_state_integer_mutations import VALID
from test_lean_graph_state_integer_receipts import ATTACKS

BASE = artifacts_dir() / "verification/graph-state-integer/qualification-v2"
REVIEW = BASE.parent / "review"


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def fresh(rows):
    if not stable.fresh(rows):
        raise ValueError("frozen source/tool/intake differs; no retry/refresh/promotion")


def roster_inputs():
    cases = []
    for width in (1, 8, 16, 32, 64, 128, 256):
        cases.append((f"rules-i{width}", standalone_source(width), False, {}, "graph", True))
    for width, factor, tile, bits in ((8, 2, 3, 32), (32, 3, 2, 64), (64, 4, 2, 64)):
        cases.append((f"chain-i{width}", chain_source(width), True,
                      {"factor": factor, "tile": tile, "index_bits": bits}, "graph", True))
    for empty in (False, True):
        cases.append(("zero-trip" if empty else "nested", nested_source(empty=empty), False, {}, "graph", True))
    cases.append(("initialized-provider", chain_source(), True, {}, "initialized-symbolic-provider-v1", True))
    for kind in ("flagged", "index", "floating"):
        cases.append(("unchanged-" + kind, identity_source(kind), False, {}, "graph", False))
    return cases


def main():
    BASE.mkdir(parents=True, exist_ok=False)
    owned = sources()
    controllers = [stable.file_ref(REVIEW / name) for name in ("qualify-v2.py", "seal-v2.py")]
    selected = prepare()
    fixed = context()
    lock = stable.file_ref(ROOT / "merlin/lean/SPEC_LOCK")
    if lock["sha256"] != "a805dc3f0f78d2a95803463808f7d02f90ed0327b76124d4d14118f61328bc95":
        raise ValueError("preserved SPEC_LOCK mismatch")
    roster, runtime = [], {}
    for name, text, chain, settings, mode, changed in roster_inputs():
        captured = produce(BASE / "intake" / name, text, chain=chain, settings=settings)
        query, roots, model = author(captured, mode)
        expected = BASE / "intake" / name / "expected-query.lean"
        expected.write_text(query)
        actual = captured["each_stage_changed"]
        if actual != [changed] * (5 if chain else 1):
            raise ValueError("frozen changed/identity population differs: " + name)
        roster.append({"name": name, "buffers": captured["buffers"], "query": stable.file_ref(expected),
                       "roots": roots, "model": model, "mode": mode, "changed": actual})
        for row in [*captured["producer_sources"], captured["python_runtime"]["executable"], *captured["python_runtime"]["loaded"]]:
            if row["path"] in runtime and runtime[row["path"]] != row:
                raise ValueError("conflicting producer/runtime identities")
            runtime[row["path"]] = row
    # The receipt-protocol fixture is one extra original. Its ten forgery cases
    # reuse readonly query/object references, avoiding ten duplicate originals.
    duplicate = next(row for row in roster if row["name"] == "rules-i8")
    roster.append({**duplicate, "name": "receipt-protocol"})
    snapshots = []
    for row in [*owned, *controllers]:
        path = Path(row["path"])
        copy = BASE / "source-snapshot" / path.relative_to(ROOT)
        copy.parent.mkdir(parents=True, exist_ok=True)
        copy.write_bytes(path.read_bytes())
        snapshots.append(stable.file_ref(copy))
    host = {}
    for module in tuple(sys.modules.values()):
        path = getattr(module, "__file__", None)
        if path and Path(path).is_file():
            row = stable.file_ref(path)
            host[row["path"]] = row
    tests = sorted(str(p.relative_to(ROOT)) for p in ROOT.glob("merlin/tests/ir/test_lean_graph_state_integer*.py"))
    command = [sys.executable, "-m", "pytest", "-q", *tests, "-k", "not one_development_replay",
               "-o", "tmp_path_retention_policy=all", "--basetemp=" + str(BASE / "suite"), "--junitxml=" + str(BASE / "suite.xml")]
    collection = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", *tests,
                                 "-k", "not one_development_replay"], cwd=ROOT, capture_output=True, text=True, check=True)
    (BASE / "collection.stdout").write_text(collection.stdout)
    (BASE / "collection.stderr").write_text(collection.stderr)
    nodes = [line for line in collection.stdout.splitlines() if line.startswith("merlin/tests/") and "::" in line]
    if len(nodes) != 75:
        raise ValueError("exact frozen 75-test denominator required: " + str(len(nodes)))
    indexed = {}
    pins = [*owned, *controllers, lock, *snapshots, *runtime.values(), *host.values(),
            stable.file_ref(sys.executable), selected["manifest"], *fixed["bindings"],
            *[r["copy"] for r in selected["copies"]]]
    pins += [stable.file_ref(p) for p in sorted((BASE / "intake").rglob("*")) if p.is_file()]
    pins += [stable.file_ref(BASE / "collection.stdout"), stable.file_ref(BASE / "collection.stderr")]
    for relative in ("pyproject.toml", "conftest.py", "merlin/conftest.py", "merlin/tests/conftest.py"):
        path = ROOT / relative
        if path.is_file():
            pins.append(stable.file_ref(path))
    for row in pins:
        if row["path"] in indexed and indexed[row["path"]] != row:
            raise ValueError("conflicting frozen pin")
        indexed[row["path"]] = row
    pins = list(indexed.values())
    plan = {"schema": "merlin.graph-state-integer.fixed-qualification.v1", "owned_sources": owned,
            "controllers": controllers, "profile": selected["manifest"], "pins": pins, "roster": roster,
            "tests": command, "nodes": nodes, "valid_mutations": list(VALID), "coherent_forgeries": list(ATTACKS),
            "positive_replay_batches": 1, "suite_positive_replay_excluded": True, "review_approved": False,
            "full_LeanMLIR_selected": False, "old_failures_retried_or_promoted": False, "qualified": False}
    write(BASE / "PLAN.json", plan)
    plan_ref = stable.file_ref(BASE / "PLAN.json")
    state = {"schema": "merlin.graph-state-integer.qualification-batch.v1", "status": "running",
             "plan": plan_ref, "originals": [], "fresh_replays": [], "review_approved": False,
             "all_graph_claim": False, "positive_replay_batches": 1}
    write(BASE / "BATCH.json", state)
    fresh([*pins, plan_ref])
    print("FROZEN", len(pins), "pins;75 tests;17 originals/one replay batch", flush=True)
    with (BASE / "suite.stdout").open("x") as out, (BASE / "suite.stderr").open("x") as err:
        run = run_owned(command, cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT / "src") + os.pathsep + str(ROOT / "merlin/tests/ir"),
                                             "PYTHONDONTWRITEBYTECODE": "1", "MERLIN_REPO_ROOT": str(ROOT)},
                        stdout=out, stderr=err, timeout=900, check=False)
    state["suite"] = {"returncode": run.returncode, "stdout": stable.file_ref(BASE / "suite.stdout"), "stderr": stable.file_ref(BASE / "suite.stderr")}
    write(BASE / "BATCH.json", state)
    if run.returncode:
        state["status"] = "UNQUALIFIED_SUITE_FAILURE"
        write(BASE / "BATCH.json", state)
        raise RuntimeError("fixed suite failed; preserve first evidence, no retry")
    counts = {key: sum(int(s.get(key, 0)) for s in ET.parse(BASE / "suite.xml").getroot().iter("testsuite"))
              for key in ("tests", "failures", "errors", "skipped")}
    if counts != {"tests": 75, "failures": 0, "errors": 0, "skipped": 0}:
        raise ValueError("exact all-passed fixed denominator differs")
    fresh([*pins, plan_ref])
    available = list(roster)
    for path in sorted((BASE / "suite").glob("*/receipt.json")):
        if path.parent.is_symlink():
            continue
        receipt = load(path)
        report, captured, proof, _ = validate(receipt)
        matches = [row for row in available if
                   (row["name"] == "receipt-protocol") == path.parent.name.startswith("integer-receipt-protocol")
                   and
                   [r["sha256"] for r in row["buffers"]] == [r["sha256"] for r in receipt["originals"]]
                   and row["query"]["sha256"] == receipt["query"]["sha256"] and row["roots"] == receipt["roots"]
                   and row["mode"] == receipt["mode"] and row["model"] == receipt["model"]]
        if not matches:
            raise ValueError("measured original is outside frozen source/query population: " + str(path))
        intended = matches[0]
        available.remove(intended)
        row = {"name": intended["name"], "receipt": stable.file_ref(path), "originals": receipt["originals"],
               "audit_theorems": proof["audit"]["raw"]["theorems_audited"], "roots": receipt["roots"], "changed": report["changed"]}
        state["originals"].append(row)
        write(BASE / "BATCH.json", state)
    if available or len(state["originals"]) != 17:
        raise ValueError("frozen original population incomplete")
    print("ORIGINALS 17 exact; starting ONE fresh-process replay batch", flush=True)
    for row in state["originals"]:
        dest = BASE / "fresh-replay" / row["name"]
        launcher = "from merlin.verify.lean_graph_state_integer_receipt import replay; import sys,json; replay(sys.argv[1],sys.argv[2],reference=sys.argv[3],expected_originals=json.loads(sys.argv[4]))"
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.with_suffix(".stdout").open("x") as out, dest.with_suffix(".stderr").open("x") as err:
            run = run_owned([sys.executable, "-c", launcher, row["receipt"]["path"], str(dest), str(REFERENCE), json.dumps(row["originals"])],
                            cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT / "src"), "PYTHONDONTWRITEBYTECODE": "1"},
                            stdout=out, stderr=err, timeout=240, check=False)
        if run.returncode:
            state["status"] = "UNQUALIFIED_FRESH_REPLAY_FAILURE"
            write(BASE / "BATCH.json", state)
            raise RuntimeError("fixed fresh replay failed; no retry")
        state["fresh_replays"].append({"name": row["name"], "report": stable.file_ref(dest / "report.json")})
        write(BASE / "BATCH.json", state)
        fresh([*pins, plan_ref])
        print("REPLAY", row["name"], "checked", flush=True)
    fresh([*pins, plan_ref])
    state.update(status="complete_conditional_graph_qualification", tests=counts, fixed_postcheck_exact=True)
    write(BASE / "BATCH.json", state)
    print("COMPLETE", flush=True)


if __name__ == "__main__":
    main()
