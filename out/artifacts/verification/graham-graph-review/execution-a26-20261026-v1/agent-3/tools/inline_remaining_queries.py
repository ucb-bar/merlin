"""Audit independent inline queries after a separate witness-source failure."""

import importlib.util
import json
import pathlib
import time

tool = pathlib.Path(__file__).resolve().with_name("reproduce_profiles.py")
spec = importlib.util.spec_from_file_location("bounded_reproduction", tool)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
ROOT, REPORT, BUILD = runner.ROOT, runner.REPORT, runner.BUILD
profiles = json.loads((REPORT / "FROZEN_PROFILES.json").read_bytes())["profiles"]
profile = next(p for p in profiles if p["id"] == "Q-inline-complete-carry-v1")
first = json.loads((REPORT / "REPRODUCTION_RESULTS.json").read_bytes())
initial = next(p for p in first["profiles"] if p["id"] == profile["id"])
lean = first["selection"]["lean"]
intent = ROOT / initial["modules"][0]["log_directory"] / "INTENT.json"
env = json.loads(intent.read_bytes())["environment"]
objects = BUILD / "compiled" / profile["id"] / "library"
deadline = time.monotonic() + max(1, min(600, (REPORT / "EXECUTION_SELECTION.json").stat().st_mtime + 1200 - time.time()))
results = {"schema": "merlin.a26.inline-independent-query-results.v1", "profile": profile["id"],
           "witness_failure_preserved": "raw-audit-successor/Q-inline-complete-carry-v1/empty-Witness-compile",
           "witnesses_retried": False, "cases": [], "qualification": False}
for case in profile["cases"]:
    if case["id"] == "empty":
        continue
    row = {"id": case["id"], "status": "UNEXECUTED", "roots_requested": len(case["roots"]["support_roots"]) + 1}
    results["cases"].append(row)
    runner.preflight(case)
    source = ROOT / case["path"]
    destination = objects.parent / "independent-query-v1" / case["id"]
    destination.mkdir(parents=True, exist_ok=False)
    run = runner.execute([lean, "-j1", "-o", str(destination / "Query.olean"), source.name],
        source.parent, env, REPORT / "raw-inline-independent" / (case["id"] + "-compile"), deadline)
    row["compile"] = run
    if run["classification"] != "ACCEPTED":
        row["status"] = "QUERY_FAILED"
        results.setdefault("first_failure", run)
        if run["classification"] == "INFRASTRUCTURE_FAILURE":
            break
        continue
    row["object"] = {"path": str((destination / "Query.olean").relative_to(ROOT)), **runner.pin(destination / "Query.olean")}
    roots = case["roots"]["support_roots"] + [case["roots"]["final_root"]]
    audit = destination / "Audit.lean"
    with audit.open("x") as stream:
        stream.write("import Query\nset_option pp.universes true\n")
        for name in roots:
            stream.write("#check @" + name + "\n#print axioms " + name + "\n")
    audit_env = dict(env, LEAN_PATH=str(destination) + ":" + env["LEAN_PATH"])
    audit_run = runner.execute([lean, "-j1", audit.name], destination, audit_env,
                              REPORT / "raw-inline-independent" / (case["id"] + "-audit"), deadline)
    row["audit"] = audit_run
    if audit_run["classification"] == "ACCEPTED":
        row["axiom_audit"] = runner.audit_axioms((ROOT / audit_run["log_directory"] / "stdout.txt").read_text(), roots)
        row["status"] = "COMPILED_AND_EXACT_ROOTS_AUDITED" if row["axiom_audit"]["passed"] else "AXIOM_AUDIT_FAILED"
    else:
        row["status"] = "AUDIT_FAILED"
        results.setdefault("first_failure", audit_run)
    print(json.dumps({"case": case["id"], "status": row["status"]}), flush=True)
runner.write_new(REPORT / "INLINE_INDEPENDENT_QUERY_RESULTS.json", results)
