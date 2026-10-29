"""Summarize original candidates and reviewer-wrapper corrections separately."""

import collections
import csv
import hashlib
import json
import pathlib

ROOT = pathlib.Path.cwd().resolve()
REPORT = pathlib.Path(__file__).resolve().parent.parent
BUILD = ROOT / "out/build/verification-a26-20261026-v1/profiles"


def read(name):
    return json.loads((REPORT / name).read_bytes())


def pin(path):
    raw = path.read_bytes()
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def error_excerpt(run):
    if not run or not run.get("log_directory"):
        return ""
    text = (ROOT / run["log_directory"] / "stdout.txt").read_text(errors="replace")
    text += "\n" + (ROOT / run["log_directory"] / "stderr.txt").read_text(errors="replace")
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if "error:" in line:
            return "\n".join(lines[i:i + 28])
    return text[:2500]


frozen = read("FROZEN_PROFILES.json")
original = read("REPRODUCTION_RESULTS.json")
audits = read("AUDIT_SUCCESSOR_RESULTS.json")
inline = read("INLINE_INDEPENDENT_QUERY_RESULTS.json")
witnesses = read("INLINE_INDEPENDENT_WITNESS_RESULTS.json")
first_by_id = {p["id"]: p for p in original["profiles"]}
audit_by_id = {p["id"]: p for p in audits["profiles"]}
inline_by_id = {c["id"]: c for c in inline["cases"]}
witness_by_id = {c["id"]: c for c in witnesses["cases"]}
source_postcheck = []
case_rows = []
profile_rows = []
accepted_queries = []
root_occurrences = 0
for profile in frozen["profiles"]:
    initial = first_by_id[profile["id"]]
    for source in [*profile["sources"], *[u for case in profile["cases"]
                    for u in [case, *([case["witness"]] if case["witness"] else [])]]]:
        actual = pin(ROOT / source["path"])
        match = actual == {k: source[k] for k in ("bytes", "sha256")}
        source_postcheck.append({"profile": profile["id"], "path": source["path"], "match": match, **actual})
        if not match:
            raise ValueError("Frozen input changed during reproduction")
    p_row = {"id": profile["id"], "source_modules": len(profile["sources"]),
             "modules_attempted": len(initial["modules"]),
             "modules_compiled": sum(m["classification"] == "ACCEPTED" for m in initial["modules"]),
             "case_queries": len(profile["cases"]), "queries_compiled_and_audited": 0,
             "queries_failed": 0, "queries_unexecuted": 0,
             "witnesses_failed": 0, "first_source_failure": None}
    if initial["status"] == "SUPPORT_BUILD_FAILED":
        p_row["first_source_failure"] = {"module": initial["first_failure"]["module"],
            "source": initial["first_failure"]["source"],
            "log_directory": initial["first_failure"]["log_directory"],
            "excerpt": error_excerpt(initial["first_failure"])}
    for case in profile["cases"]:
        row = {"profile": profile["id"], "case": case["id"], "role": "QUERY",
               "source": {k: case[k] for k in ("path", "bytes", "sha256")},
               "expected_protocol": case["roots"].get("expected_protocol", "POSITIVE_SOURCE_CANDIDATE_UNOBSERVED"),
               "domain": case["roots"].get("domain"), "relation": case["roots"].get("relation"),
               "final_root": case["roots"]["final_root"], "status": "UNEXECUTED"}
        if profile["id"] == "Q-schedule-successor-v1" or (profile["id"] == "Q-inline-complete-carry-v1" and case["id"] == "empty"):
            effective = next(c for c in audit_by_id[profile["id"]]["cases"] if c["id"] == case["id"])
            unit = next(u for u in effective["units"] if u["unit"] == "Query")
            row["status"] = "COMPILED_AND_EXACT_ROOTS_AUDITED" if unit.get("axiom_audit", {}).get("passed") else "AUDIT_FAILED"
            row["audit"] = unit["audit"]
            row["axiom_audit"] = unit.get("axiom_audit")
            row["compile"] = unit["compile"]
            if row["status"] == "COMPILED_AND_EXACT_ROOTS_AUDITED":
                root_occurrences += unit["axiom_audit"]["roots_requested"]
                text = (ROOT / unit["audit"]["log_directory"] / "stdout.txt").read_text()
                lines = text.splitlines()
                start = next(i for i, line in enumerate(lines) if line.startswith("@" + row["final_root"] + " :"))
                end = next(i for i in range(start + 1, len(lines)) if "depends on axioms:" in lines[i] or "does not depend on any axioms" in lines[i])
                row["kernel_checked_final_type"] = "\n".join(lines[start:end])
                accepted_queries.append({k: row[k] for k in ("profile", "case", "final_root", "kernel_checked_final_type", "source")})
        elif profile["id"] == "Q-inline-complete-carry-v1":
            effective = inline_by_id[case["id"]]
            row["status"] = effective["status"]
            row["compile"] = effective["compile"]
            row["first_error"] = error_excerpt(effective["compile"])
        else:
            row["reason"] = "First support-source compilation failure prevents this query from importing its required module closure."
        if row["status"] == "COMPILED_AND_EXACT_ROOTS_AUDITED":
            p_row["queries_compiled_and_audited"] += 1
        elif row["status"] == "UNEXECUTED":
            p_row["queries_unexecuted"] += 1
        else:
            p_row["queries_failed"] += 1
        case_rows.append(row)
        if case["witness"]:
            if case["id"] == "empty":
                effective = next(c for c in audit_by_id[profile["id"]]["cases"] if c["id"] == "empty")
                run = next(u for u in effective["units"] if u["unit"] == "Witness")["compile"]
            else:
                run = witness_by_id[case["id"]]["compile"]
            witness = {"profile": profile["id"], "case": case["id"], "role": "WITNESS",
                       "source": {k: case["witness"][k] for k in ("path", "bytes", "sha256")},
                       "status": "WITNESS_SOURCE_FAILED", "compile": run, "first_error": error_excerpt(run)}
            case_rows.append(witness)
            p_row["witnesses_failed"] += 1
    profile_rows.append(p_row)

raw_results = []
for directory in REPORT.glob("raw*"):
    for path in directory.rglob("RESULT.json"):
        value = json.loads(path.read_bytes())
        raw_results.append({"path": str(path.relative_to(ROOT)), **value})
objects = []
for path in sorted(BUILD.rglob("*.olean")):
    objects.append({"path": str(path.relative_to(ROOT)), **pin(path)})
summary = {
    "schema": "merlin.a26.independent-proof-reproduction-summary.v1",
    "baseline": frozen["baseline"], "fresh_context": original["selection"],
    "profiles": profile_rows,
    "totals": {"profile_source_module_attempts": sum(p["modules_attempted"] for p in profile_rows),
               "profile_source_module_acceptances": sum(p["modules_compiled"] for p in profile_rows),
               "case_queries_selected": 38, "case_queries_compiled_and_exactly_audited": 4,
               "case_queries_failed": 4, "case_queries_unexecuted": 30,
               "witness_queries_selected": 5, "witness_queries_failed": 5,
               "original_query_root_audit_occurrences": root_occurrences,
               "native_commands": len(raw_results),
               "command_statuses": dict(collections.Counter(r["classification"] for r in raw_results))},
    "accepted_queries": accepted_queries,
    "same_statement_tail_proof_successor": read("TAIL_PROOF_SUCCESSOR_RESULT.json"),
    "frozen_inputs_postcheck": {"units": len(source_postcheck), "matches": sum(r["match"] for r in source_postcheck)},
    "qualification": False, "human_approval": False, "public_gate_activation": False,
    "limits": [
        "Schedule outcomes cover two actual changed cases and one identity case; these are not three verified passes.",
        "The exact conditional query roots and their requested supports were kernel checked. Projection, primitive/native fidelity and full qualification remain separate.",
        "The empty-inline theorem compiled, while all five independent non-vacuity witness files failed; no new non-vacuity approval is inferred.",
        "Both W profiles and tail remain blocked at original support source; all30 dependent case queries are explicitly unexecuted.",
        "The first audit-wrapper pp.width error and staging bootstrap/scanner issues were reviewer-tool failures, preserved separately from proof-candidate failures.",
        "Only the standalone same-statement tail lemma successor passed; the original tail module/profile was not repaired or requalified.",
        "Source/module/root counts are operational evidence counts and must not be converted to verified pass counts."
    ]}
for name, value in (("REPRODUCTION_SUMMARY.json", summary), ("CASE_ROSTER.json", case_rows),
                    ("FROZEN_INPUT_POSTCHECK.json", source_postcheck), ("OBJECT_MANIFEST.json", objects)):
    with (REPORT / name).open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
with (REPORT / "CASE_ROSTER.csv").open("x", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=("profile", "case", "role", "status", "final_root", "source_path", "source_sha256"))
    writer.writeheader()
    for row in case_rows:
        writer.writerow({k: row.get(k) for k in ("profile", "case", "role", "status", "final_root")} |
                        {"source_path": row["source"]["path"], "source_sha256": row["source"]["sha256"]})
print(json.dumps({"totals": summary["totals"], "frozen_postcheck": summary["frozen_inputs_postcheck"],
                  "objects": len(objects)}, indent=2))
