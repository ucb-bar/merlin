"""Bounded single-worker Lean source reproduction in an explicitly fresh SDK.

No frozen source is edited. Logs, first failures, and downstream unexecuted
cases survive. A successful build/audit does not grant compiler authority.
"""

import argparse
import datetime
import hashlib
import json
import os
import pathlib
import resource
import signal
import subprocess
import time

ROOT = pathlib.Path.cwd().resolve()
REPORT = pathlib.Path(__file__).resolve().parent.parent
BUILD = ROOT / "out/build/verification-a26-20261026-v1/profiles"
ALLOWED = {"propext", "Classical.choice", "Quot.sound"}


def pin(path):
    raw = path.read_bytes()
    return {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def write_new(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def tokens(raw):
    """Skip Lean comments/strings before the conservative source ban scan."""
    text = raw.decode("utf-8")
    position = 0
    while position < len(text):
        if text.startswith("--", position):
            end = text.find("\n", position)
            position = len(text) if end < 0 else end
        elif text.startswith("/-", position):
            depth = 1
            position += 2
            while depth and position < len(text):
                if text.startswith("/-", position):
                    depth += 1
                    position += 2
                elif text.startswith("-/", position):
                    depth -= 1
                    position += 2
                else:
                    position += 1
            if depth:
                raise ValueError("Unterminated source comment")
        elif text[position] == '"':
            position += 1
            while position < len(text):
                if text[position] == "\\":
                    position += 2
                elif text[position] == '"':
                    position += 1
                    break
                else:
                    position += 1
        elif text[position].isalnum() or text[position] == "_":
            start = position
            while position < len(text) and (text[position].isalnum() or text[position] in "_.'"):
                position += 1
            yield text[start:position]
        else:
            position += 1


def code_without_comments_or_strings(raw):
    text = raw.decode("utf-8")
    chars = list(text)
    position = 0

    def blank(start, end):
        for i in range(start, end):
            if chars[i] not in "\r\n":
                chars[i] = " "

    while position < len(text):
        start = position
        if text.startswith("--", position):
            end = text.find("\n", position)
            position = len(text) if end < 0 else end
            blank(start, position)
        elif text.startswith("/-", position):
            depth = 1
            position += 2
            while depth and position < len(text):
                if text.startswith("/-", position):
                    depth += 1
                    position += 2
                elif text.startswith("-/", position):
                    depth -= 1
                    position += 2
                else:
                    position += 1
            if depth:
                raise ValueError("Unterminated source comment")
            blank(start, position)
        elif text[position] == '"':
            position += 1
            while position < len(text):
                if text[position] == "\\":
                    position += 2
                elif text[position] == '"':
                    position += 1
                    break
                else:
                    position += 1
            blank(start, position)
        else:
            position += 1
    return "".join(chars)


def preflight(source):
    p = ROOT / source["path"]
    if not p.resolve().is_relative_to(BUILD) or p.is_symlink():
        raise ValueError("Profile source outside fresh staging root")
    if pin(p) != {key: source[key] for key in ("sha256", "bytes")}:
        raise ValueError("Frozen source changed: " + source["path"])
    forbidden = {"sorry", "admit", "axiom", "native_decide", "bv_decide", "ofReduceBool",
                 "implemented_by", "extern", "unsafe", "partial", "skipKernelTC",
                 "run_cmd", "elab", "macro_rules", "csimp"}
    found = sorted({word for word in tokens(p.read_bytes()) if word.split(".")[-1] in forbidden})
    # `Code.opaque` and `Instruction.initialize` are ordinary IR constructors.
    # Reserve these two checks for actual declaration command starts.
    for line in code_without_comments_or_strings(p.read_bytes()).splitlines():
        line = line.lstrip()
        for prefix in ("", "private ", "protected ", "noncomputable "):
            if any(line.startswith(prefix + word + " ") for word in ("opaque", "initialize")):
                found.append("opaque/initialize declaration command")
    if found:
        raise ValueError("Banned source tokens: " + repr(found) + " in " + source["path"])


def limits():
    resource.setrlimit(resource.RLIMIT_AS, (8 * 1024**3, 8 * 1024**3))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def execute(command, cwd, env, directory, deadline):
    directory.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    timeout = min(180.0, deadline - start)
    if timeout <= 0:
        return {"classification": "BATCH_BUDGET_EXHAUSTED", "command": command, "executed": False}
    intent = {"command": command, "cwd": str(cwd), "environment": env,
              "timeout_seconds": timeout, "utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    write_new(directory / "INTENT.json", intent)
    with (directory / "stdout.txt").open("wb") as stdout, (directory / "stderr.txt").open("wb") as stderr:
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=stdout, stderr=stderr,
                                   start_new_session=True, preexec_fn=limits)
        timed_out = False
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            process.terminate()
            try:
                code = process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                code = process.wait(timeout=5)
    output = (directory / "stdout.txt").read_text(errors="replace")
    error = (directory / "stderr.txt").read_text(errors="replace")
    infrastructure = code < 0 and not timed_out
    for marker in ("unknown Lean version", "incompatible module", "error loading shared libraries",
                   "has no executable code", "no executable code", "cannot load shared object",
                   "failed to load object file", "failed to open file", "unknown module prefix 'Init'",
                   "unknown module prefix 'Lean'", "libLean", "libleanshared"):
        if marker in output or marker in error:
            infrastructure = True
    classification = ("TIMEOUT" if timed_out else "INFRASTRUCTURE_FAILURE" if infrastructure
                      else "ACCEPTED" if code == 0 else "LEAN_SOURCE_OR_AUDIT_FAILURE")
    result = {"executed": True, "command": command, "exit_code": code,
              "seconds": time.monotonic() - start, "classification": classification,
              "log_directory": str(directory.relative_to(ROOT)),
              "stdout": pin(directory / "stdout.txt"), "stderr": pin(directory / "stderr.txt")}
    if code != 0:
        result["first_error_excerpt"] = (output + "\n" + error)[:10000]
    write_new(directory / "RESULT.json", result)
    return result


def audit_axioms(text, roots):
    observed = {}
    lines = iter(text.splitlines())
    for line in lines:
        if "depends on axioms:" in line:
            left, right = line.split("depends on axioms:", 1)
            name = left.strip().strip("'")
            values = right.strip()
            while values.startswith("[") and not values.endswith("]"):
                values += " " + next(lines).strip()
            if not (values.startswith("[") and values.endswith("]")):
                raise ValueError("Unknown axiom output: " + line)
            observed[name] = sorted(v.strip().partition(".{")[0]
                                    for v in values[1:-1].split(",") if v.strip())
        elif "does not depend on any axioms" in line:
            observed[line.split("does not depend on any axioms", 1)[0].strip().strip("'")] = []
    missing = [root for root in roots if root not in observed]
    forbidden = {root: names for root, names in observed.items() if set(names) - ALLOWED}
    return {"roots_requested": len(roots), "roots_observed": len(observed),
            "missing": missing, "forbidden": forbidden, "axioms": observed,
            "passed": not missing and not forbidden}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lean", type=pathlib.Path, required=True)
    parser.add_argument("--sdk-root", type=pathlib.Path, required=True)
    parser.add_argument("--admission", type=pathlib.Path, required=True)
    parser.add_argument("--batch-seconds", type=int, default=1200)
    args = parser.parse_args()
    sdk = args.sdk_root.resolve()
    lean = args.lean.resolve()
    if not sdk.is_relative_to(ROOT / "out/build/verification-a26-20261026-v1/fresh-sdk") or not lean.is_relative_to(sdk):
        raise ValueError("Only the new selected SDK executable is admitted")
    if not args.admission.resolve().is_relative_to(ROOT):
        raise ValueError("Admission evidence outside checkout")
    admission_data = json.loads(args.admission.read_bytes())
    if admission_data["status"] != "READY_FRESH_INDEPENDENT_CLI" or pathlib.Path(admission_data["lean"]).resolve() != lean:
        raise ValueError("Fresh admission does not select this executable")
    if admission_data["lean_sha256"] != pin(lean)["sha256"]:
        raise ValueError("Fresh compiler executable differs from admission")
    frozen = json.loads((REPORT / "FROZEN_PROFILES.json").read_bytes())
    selected = {"lean": str(lean), "lean_identity": pin(lean), "sdk_root": str(sdk),
                "admission": str(args.admission), "admission_pin": pin(args.admission),
                "profiles_pin": pin(REPORT / "FROZEN_PROFILES.json"),
                "single_worker": True, "module_timeout_seconds": 180,
                "batch_budget_seconds": args.batch_seconds}
    write_new(REPORT / "EXECUTION_SELECTION.json", selected)
    result_roster = {"selection": selected, "profiles": [], "qualification": False,
                     "human_approval": False, "public_gate_activation": False}
    deadline = time.monotonic() + args.batch_seconds
    stop_batch = False
    for profile in frozen["profiles"]:
        result = {"id": profile["id"], "modules": [], "cases": [], "status": "UNEXECUTED"}
        result_roster["profiles"].append(result)
        if stop_batch:
            result["reason"] = "Earlier fresh-runtime infrastructure failure"
            continue
        project = ROOT / profile["project_root"]
        objects = BUILD / "compiled" / profile["id"] / "library"
        objects.mkdir(parents=True, exist_ok=False)
        env = {"PATH": str(sdk / "bin") + ":/usr/bin:/bin", "HOME": os.environ["HOME"],
               "LEAN_PATH": str(objects) + ":" + str(sdk / "lib/lean"),
               "LEAN_SRC_PATH": str(project) + ":" + str(sdk / "src/lean"),
               "LD_LIBRARY_PATH": str(sdk / "lib") + ":" + str(sdk / "lib/lean"),
               "TMPDIR": str(BUILD / "scratch"), "LC_ALL": "C.UTF-8",
               "LEAN_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
        pathlib.Path(env["TMPDIR"]).mkdir(exist_ok=True)
        failed = False
        for i, source in enumerate(profile["sources"]):
            preflight(source)
            module = source["module"]
            output = objects / (module.replace(".", "/") + ".olean")
            output.parent.mkdir(parents=True, exist_ok=True)
            command = [str(lean), "-j1", "-o", str(output), module.replace(".", "/") + ".lean"]
            run = execute(command, project, env, REPORT / "raw" / profile["id"] / (f"module-{i:03d}-" + module), deadline)
            run.update(module=module, source={k: source[k] for k in ("path", "sha256", "bytes")})
            if run["classification"] == "ACCEPTED":
                run["output"] = {"path": str(output.relative_to(ROOT)), **pin(output)}
            result["modules"].append(run)
            if run["classification"] != "ACCEPTED":
                result["status"] = "SUPPORT_BUILD_FAILED"
                result["first_failure"] = run
                stop_batch = run["classification"] == "INFRASTRUCTURE_FAILURE"
                failed = True
                break
        for case in profile["cases"]:
            expected_protocol = case["roots"].get("expected_protocol", "POSITIVE_SOURCE_CANDIDATE_UNOBSERVED")
            expects_refusal = "REFUSAL" in expected_protocol
            case_result = {"id": case["id"], "query": case["path"], "domain": case["roots"]["domain"],
                           "relation": case["roots"].get("relation"),
                           "exact_final_declaration": case["roots"].get("exact_final_declaration"),
                           "expected_protocol": expected_protocol, "status": "UNEXECUTED"}
            result["cases"].append(case_result)
            if failed:
                case_result["reason"] = "Profile's first support-source failure"
                continue
            units = [{"module": case["module"], "path": case["path"], "sha256": case["sha256"],
                      "bytes": case["bytes"], "roots": case["roots"]["support_roots"] + [case["roots"]["final_root"]]}]
            if case["witness"]:
                units.append(case["witness"])
            unit_results = []
            case_result["units"] = unit_results
            for unit in units:
                preflight(unit)
                src = ROOT / unit["path"]
                outdir = objects.parent / "queries" / case["id"] / unit["module"]
                outdir.mkdir(parents=True, exist_ok=False)
                output = outdir / (unit["module"] + ".olean")
                command = [str(lean), "-j1", "-o", str(output), src.name]
                logbase = REPORT / "raw" / profile["id"] / ("query-" + case["id"] + "-" + unit["module"])
                run = execute(command, src.parent, env, logbase, deadline)
                run["role"] = unit["module"]
                unit_results.append(run)
                if run["classification"] != "ACCEPTED":
                    if expects_refusal and run["classification"] == "LEAN_SOURCE_OR_AUDIT_FAILURE":
                        case_result["status"] = "EXPECTED_BINDING_REFUSAL_OBSERVED"
                        result.setdefault("first_expected_refusal", run)
                        break
                    failed = True
                    stop_batch = run["classification"] == "INFRASTRUCTURE_FAILURE"
                    result["first_failure"] = run
                    result["status"] = "QUERY_FAILED"
                    case_result["status"] = "QUERY_FAILED"
                    break
                run["output"] = {"path": str(output.relative_to(ROOT)), **pin(output)}
                roots = unit["roots"]
                audit = outdir / "Audit.lean"
                with audit.open("x") as stream:
                    stream.write("import " + unit["module"] + "\nset_option pp.universes true\n")
                    for name in roots:
                        if not name or any(not (c.isalnum() or c in "_.'") for c in name):
                            raise ValueError("Unsafe audit root name")
                        stream.write("#check @" + name + "\n#print axioms " + name + "\n")
                audit_env = dict(env, LEAN_PATH=str(outdir) + ":" + env["LEAN_PATH"])
                audit_run = execute([str(lean), "-j1", audit.name], outdir, audit_env,
                                    logbase.with_name(logbase.name + "-audit"), deadline)
                run["audit"] = audit_run
                if audit_run["classification"] == "ACCEPTED":
                    audit_run["axiom_audit"] = audit_axioms((ROOT / audit_run["log_directory"] / "stdout.txt").read_text(), roots)
                    if not audit_run["axiom_audit"]["passed"]:
                        audit_run["classification"] = "AXIOM_AUDIT_FAILED"
                if audit_run["classification"] != "ACCEPTED":
                    failed = True
                    stop_batch = audit_run["classification"] == "INFRASTRUCTURE_FAILURE"
                    result["first_failure"] = audit_run
                    result["status"] = "AUDIT_FAILED"
                    case_result["status"] = "AUDIT_FAILED"
                    break
            if not failed and case_result["status"] == "UNEXECUTED":
                case_result["status"] = ("REFUSAL_CANDIDATE_COMPILED_AND_ROOTS_AUDITED" if expects_refusal
                                         else "COMPILED_AND_ROOTS_AUDITED")
        if not failed:
            result["status"] = "COMPILED_AND_ROOTS_AUDITED"
        write_new(REPORT / "profile-results" / (profile["id"] + ".json"), result)
        print(json.dumps({"profile": profile["id"], "status": result["status"],
                          "modules_attempted": len(result["modules"]),
                          "queries_audited": sum(c["status"] == "COMPILED_AND_ROOTS_AUDITED" for c in result["cases"])}), flush=True)
    write_new(REPORT / "REPRODUCTION_RESULTS.json", result_roster)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
