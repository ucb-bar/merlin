"""Correct only the reviewer's audit wrapper; preserve initial raw failures."""

import importlib.util
import json
import pathlib
import sys
import time

TOOL = pathlib.Path(__file__).resolve().with_name("reproduce_profiles.py")
spec = importlib.util.spec_from_file_location("bounded_reproduction", TOOL)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
ROOT, REPORT, BUILD = runner.ROOT, runner.REPORT, runner.BUILD
first = json.loads((REPORT / "REPRODUCTION_RESULTS.json").read_bytes())
selection = first["selection"]
lean = selection["lean"]
sdk = pathlib.Path(selection["sdk_root"])
profiles = json.loads((REPORT / "FROZEN_PROFILES.json").read_bytes())["profiles"]
first_by_id = {p["id"]: p for p in first["profiles"]}
results = {"schema": "merlin.a26.audit-wrapper-successor.v1",
           "original_failure": "Unknown option pp.width in reviewer's generated Audit.lean",
           "candidate_bytes_changed": False, "first_results": runner.pin(REPORT / "REPRODUCTION_RESULTS.json"),
           "profiles": [], "qualification": False, "public_gate_activation": False}
deadline = time.monotonic() + 900
for profile in profiles:
    initial = first_by_id[profile["id"]]
    if initial["status"] != "AUDIT_FAILED" or "Unknown option `pp.width`" not in initial["first_failure"].get("first_error_excerpt", ""):
        continue
    result = {"id": profile["id"], "status": "UNEXECUTED", "cases": []}
    results["profiles"].append(result)
    objects = BUILD / "compiled" / profile["id"] / "library"
    intent_file = ROOT / initial["modules"][0]["log_directory"] / "INTENT.json"
    env = json.loads(intent_file.read_bytes())["environment"]
    failed = False
    for case in profile["cases"]:
        row = {"id": case["id"], "expected_protocol": case["roots"].get("expected_protocol"),
               "final_root": case["roots"]["final_root"], "status": "UNEXECUTED", "units": []}
        result["cases"].append(row)
        if failed:
            row["reason"] = "Previous unexpected query/audit failure in this profile"
            continue
        units = [{k: case[k] for k in ("module", "path", "sha256", "bytes")} |
                 {"roots": case["roots"]["support_roots"] + [case["roots"]["final_root"]]}]
        if case["witness"]:
            units.append(case["witness"])
        for unit in units:
            runner.preflight(unit)
            src = ROOT / unit["path"]
            destination = objects.parent / "audit-successor-v1" / case["id"] / unit["module"]
            destination.mkdir(parents=True, exist_ok=False)
            old_objects = objects.parent / "queries" / case["id"] / unit["module"]
            old_object = old_objects / (unit["module"] + ".olean")
            previous = [u for c in initial["cases"] if c["id"] == case["id"]
                        for u in c.get("units", []) if u.get("role") == unit["module"] and u["classification"] == "ACCEPTED"]
            if previous:
                if runner.pin(old_object) != {k: previous[0]["output"][k] for k in ("sha256", "bytes")}:
                    raise ValueError("Same-batch query object drift")
                compile_result = {"classification": "REUSED_EXACT_SAME_BATCH_COMPILED_QUERY",
                                  "original_command_result": previous[0], "source": {k: unit[k] for k in ("path", "sha256", "bytes")}}
                imported = old_objects
            else:
                command = [lean, "-j1", "-o", str(destination / (unit["module"] + ".olean")), src.name]
                compile_result = runner.execute(command, src.parent, env,
                    REPORT / "raw-audit-successor" / profile["id"] / (case["id"] + "-" + unit["module"] + "-compile"), deadline)
                imported = destination
            record = {"unit": unit["module"], "compile": compile_result}
            row["units"].append(record)
            if compile_result["classification"] not in ("ACCEPTED", "REUSED_EXACT_SAME_BATCH_COMPILED_QUERY"):
                row["status"] = "QUERY_FAILED"
                result["first_failure"] = compile_result
                failed = True
                break
            audit = destination / "AuditSuccessor.lean"
            with audit.open("x") as stream:
                stream.write("import " + unit["module"] + "\nset_option pp.universes true\n")
                for name in unit["roots"]:
                    stream.write("#check @" + name + "\n#print axioms " + name + "\n")
            audit_env = dict(env, LEAN_PATH=str(imported) + ":" + env["LEAN_PATH"])
            audit_result = runner.execute([lean, "-j1", audit.name], destination, audit_env,
                REPORT / "raw-audit-successor" / profile["id"] / (case["id"] + "-" + unit["module"] + "-audit"), deadline)
            record["audit"] = audit_result
            if audit_result["classification"] == "ACCEPTED":
                text = (ROOT / audit_result["log_directory"] / "stdout.txt").read_text()
                record["axiom_audit"] = runner.audit_axioms(text, unit["roots"])
                if not record["axiom_audit"]["passed"]:
                    audit_result["classification"] = "AXIOM_AUDIT_FAILED"
            if audit_result["classification"] != "ACCEPTED":
                row["status"] = "AUDIT_FAILED"
                result["first_failure"] = audit_result
                failed = True
                break
        if not failed:
            row["status"] = "COMPILED_AND_EXACT_ROOTS_AUDITED"
    result["status"] = "UNEXPECTED_FAILURE" if failed else "COMPILED_AND_EXACT_ROOTS_AUDITED"
    runner.write_new(REPORT / "audit-successor-profile-results" / (profile["id"] + ".json"), result)
    print(json.dumps({"profile": profile["id"], "status": result["status"],
                      "cases_audited": sum(c["status"] == "COMPILED_AND_EXACT_ROOTS_AUDITED" for c in result["cases"])}), flush=True)
runner.write_new(REPORT / "AUDIT_SUCCESSOR_RESULTS.json", results)
