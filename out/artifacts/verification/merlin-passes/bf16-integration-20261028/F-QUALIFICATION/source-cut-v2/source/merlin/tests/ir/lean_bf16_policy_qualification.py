"""Retained MACC qualification commands: fixture export, actual Lean check.

A fixed source fixture roster is generated before checking. Native verification,
Lean acceptance/refusal, infrastructure failure and policy approval stay distinct.
Every child has a timeout; raw outputs and first failures stay in the new product.
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from merlin.common.paths import repo_root


def reference(path):
    raw = path.read_bytes()
    return {"path": str(path.resolve()), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def save(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def command(args, directory, timeout, *, cwd=None):
    save(
        directory / "INTENT.json",
        {"command": list(map(str, args)), "timeout_seconds": timeout, "cwd": str(cwd) if cwd else None},
    )
    try:
        result = subprocess.run(list(map(str, args)), capture_output=True, timeout=timeout, check=False, cwd=cwd)
        (directory / "stdout.txt").write_bytes(result.stdout)
        (directory / "stderr.txt").write_bytes(result.stderr)
        status = {"returncode": result.returncode, "timeout": False}
    except subprocess.TimeoutExpired as error:
        (directory / "stdout.txt").write_bytes(error.stdout or b"")
        (directory / "stderr.txt").write_bytes(error.stderr or b"")
        status = {"returncode": None, "timeout": True, "error": str(error)}
    save(directory / "PROCESS.json", status)
    return status


def fixtures(output, compiler):
    from lean_bf16_policy_cases import cases

    output.mkdir(parents=True, exist_ok=False)
    worker = repo_root() / "merlin/tests/ir/lean_bf16_policy_mlir.py"
    exporter = repo_root() / "src/merlin/verify/lean/mlir_export.py"
    rows = []
    save(
        output / "SOURCE-CUT.json",
        {
            "files": [
                reference(path)
                for path in [
                    Path(__file__),
                    worker,
                    exporter,
                    repo_root() / "merlin/tests/ir/lean_bf16_policy_cases.py",
                    repo_root() / "src/merlin/llvmlower/passes_xdsl.py",
                ]
            ],
            "human_policy_reviewed": False,
        },
    )
    version = output / "versions"
    version.mkdir()
    command([compiler, worker, "--version"], version, 30)
    for case in cases():
        directory = output / case.name
        directory.mkdir()
        row = {"name": case.name, "expected": case.expected, "sites": case.sites, "reason": case.reason}
        for name, text in [("before", case.before), ("after", case.after)]:
            source = directory / (name + ".mlir")
            source.write_text(text)
            log = directory / name
            log.mkdir()
            exported = directory / (name + ".json")
            process = command([compiler, worker, source, exported, exporter], log, 90)
            row[name] = {
                "source": reference(source),
                "process": process,
                "export": reference(exported) if process["returncode"] == 0 else None,
            }
        rows.append(row)
        save(
            output / "MATRIX.json", {"phase": "AUTHORITATIVE_MLIR_ONLY", "cases": rows, "human_policy_reviewed": False}
        )
    print(
        json.dumps(
            {"cases": len(rows), "valid_pairs": sum(all(r[k]["export"] for k in ["before", "after"]) for r in rows)}
        )
    )


def checks(output, inputs, checker):
    from merlin.verify import bf16_policy_gate as gate

    output.mkdir(parents=True, exist_ok=False)
    matrix = json.loads((inputs / "MATRIX.json").read_text())
    checker_ref = reference(checker)
    save(
        output / "SOURCE-CUT.json",
        {
            "checker": checker_ref,
            "matrix": reference(inputs / "MATRIX.json"),
            "lock": reference(repo_root() / "merlin/lean/SPEC_LOCK"),
        },
    )
    rows = []
    for row in matrix["cases"]:
        directory = output / row["name"]
        directory.mkdir()
        observed = {"name": row["name"], "expected": row["expected"], "sites": row["sites"]}
        if not all(row[k]["export"] for k in ["before", "after"]):
            observed["outcome"] = "AUTHORITATIVE_MLIR_REFUSAL_NOT_LEAN_TEST"
        else:
            for k in ["before", "after"]:
                for ref in [row[k]["source"], row[k]["export"]]:
                    if reference(Path(ref["path"])) != ref:
                        raise ValueError("fixture binding changed")
            process = command(
                [checker, "check-bf16-f32acc", row["before"]["export"]["path"], row["after"]["export"]["path"], "64"],
                directory,
                300,
            )
            observed["process"] = process
            try:
                report = json.loads((directory / "stdout.txt").read_text())
                save(directory / "report.json", report)
                if (
                    process["returncode"] != 1
                    or report["verdict"] != "abstracted"
                    or report["policy_reviewed"] is not False
                ):
                    raise ValueError("unexpected driver identity/exit/status")
                source = json.loads(Path(row["before"]["export"]["path"]).read_text())
                if row["expected"] == "outside_policy_domain":
                    try:
                        gate.validate_report(report, source)
                    except (gate.GateUnsupported, gate.GateRejected):
                        if report["candidates"] != 0:
                            raise ValueError("domain refusal unexpectedly contains admitted candidates")
                        observed["outcome"] = "DOMAIN_REFUSAL"
                    else:
                        observed["outcome"] = "UNEXPECTED_DOMAIN_ADMISSION"
                elif row["expected"] == "raw_policy_match":
                    roster = gate.validate_report(report, source)
                    observed.update(outcome="RAW_POLICY_MATCH", covered_sites=len(roster))
                elif report["pass_boundary"]["checked"] is False:
                    observed["outcome"] = "LEAN_POLICY_REFUSAL"
                else:
                    observed["outcome"] = "UNEXPECTED_LEAN_ACCEPTANCE"
            except Exception as error:
                observed.update(outcome="CHECK_OR_BINDING_FAILURE", error=type(error).__name__ + ": " + str(error))
        rows.append(observed)
        save(
            output / "RESULT.json", {"phase": "RAW_POLICY_QUALIFICATION", "cases": rows, "human_policy_reviewed": False}
        )
    if reference(checker) != checker_ref:
        raise ValueError("checker changed during qualification")
    print(json.dumps({kind: sum(r["outcome"] == kind for r in rows) for kind in sorted({r["outcome"] for r in rows})}))


def receipts(output, baseline, compiler, sdk, audits_from=None):
    from merlin.verify import bf16_policy_gate as gate
    from merlin.verify.lean import bf16_f32acc as policy
    from merlin.verify.lean import check
    from merlin.verify.pass_validation import CheckerTools

    output.mkdir(parents=True, exist_ok=False)
    CheckerTools(sdk, compiler)
    tools = {
        key: reference(path)
        for key, path in {
            "lake": sdk / "lake",
            "lean": sdk / "lean",
            "compiler_python": compiler,
            "checker": check.lean_exe(),
            "audit": check.lean_exe().with_name("merlin-lean-audit"),
        }.items()
    }
    sources = gate.source_bindings(repo_root())
    save(output / "SOURCE-CUT.json", {"sources": sources, "tools": tools, "baseline": reference(baseline)})
    work = output / "audit"
    work.mkdir()
    if audits_from is None:
        audited = gate.audit_candidate(tools, work)
    else:
        # Reuse this task's completed exact audit; native gate admission still
        # runs its own live audit. Every project/tool binding must stay exact.
        bound = json.loads((audits_from / "SOURCE-CUT.json").read_text())
        if bound["sources"] != sources or bound["tools"] != tools:
            raise ValueError("prior task audit source/tool identity changed")
        files = [audits_from / "audit" / name for name in ("closure.stdout", "statements.stdout")]
        closure = json.loads(files[0].read_text())
        statements = json.loads(files[1].read_text())["statements"]
        if closure["violations"] or set(statements) != set(gate.ROOTS):
            raise ValueError("prior exact audit refused")
        axioms = {}
        for index, name in enumerate(gate.ROOTS):
            file = audits_from / "audit" / f"root-{index}.stdout"
            files.append(file)
            row = json.loads(file.read_text())
            if row["decl"] != name or not set(row["axioms"]) <= gate.STD3:
                raise ValueError("prior exact-root audit refused")
            axioms[name] = row["axioms"]
        audited = {"closure": closure, "statements": statements, "axioms": axioms}
        save(
            output / "REUSED-TASK-AUDIT.json",
            {
                "source_cut": reference(audits_from / "SOURCE-CUT.json"),
                "raw_records": [reference(file) for file in files],
            },
        )
    # Segment roots are separate from the live single-boundary gate roots.
    for suffix in ("check_sound", "qualify_sound"):
        root = policy.SEGMENT[1] + "." + suffix
        directory = work / suffix
        directory.mkdir()
        status = command(
            [sdk / "lake", "env", tools["audit"]["path"], "--axioms", root, policy.SEGMENT[1]],
            directory,
            300,
            cwd=check.lean_root(),
        )
        if status["returncode"] != 0:
            raise ValueError("segment root audit unavailable")
        result = json.loads((directory / "stdout.txt").read_text())
        if result["decl"] != root or not set(result["axioms"]) <= gate.STD3:
            raise ValueError("segment root audit refused")
        audited["axioms"][root] = result["axioms"]
    save(output / "AUDIT.json", audited)

    def auditor(name, module):
        expected = gate.ROOTS.get(name)
        if expected is None and name in {policy.SEGMENT[1] + ".check_sound", policy.SEGMENT[1] + ".qualify_sound"}:
            expected = policy.SEGMENT[1]
        if expected != module or name not in audited["axioms"]:
            raise ValueError("unaudited exact theorem root")
        return audited["axioms"][name]

    outcomes = []
    for row in json.loads(baseline.read_text())["receipts"]:
        directory = output / row["name"]
        stages = [Path(path) for path in row["stage_directories"]]
        receipt = policy.check_stages(stages, directory, m2m_python=compiler, theorem_auditor=auditor)
        save(output / (row["name"] + "-receipt-result.json"), receipt)
        if receipt.get("verdict") == "unmeasured" or "cert" not in receipt:
            raise ValueError("receipt unavailable; first failure retained")
        report = json.loads(Path(receipt["cert"]["path"]).read_text())
        boundary = policy.boundary_of(report)
        expected = row["expected_sites"]
        local = boundary["pass_boundary"]
        if not (
            local["checked"]
            and local["semantic_match"]
            and local["site_qualification_passed"]
            and local["expected_sites"] == local["covered_sites"] == expected
        ):
            raise ValueError("actual boundary policy obligation refused")
        replayed = policy.replay(
            directory / "receipt.json",
            m2m_python=compiler,
            work=output / (row["name"] + "-replay"),
            theorem_auditor=auditor,
        )
        save(output / (row["name"] + "-replay.json"), replayed)
        if not replayed["fresh"] or not replayed["matches_recorded"]:
            raise ValueError("fresh actual receipt replay refused")
        outcomes.append(
            {
                "name": row["name"],
                "sites": expected,
                "receipt": reference(directory / "receipt.json"),
                "fresh_replay": True,
                "verdict": receipt["verdict"],
                "human_policy_reviewed": False,
            }
        )
        save(
            output / "RESULT.json",
            {
                "receipts": outcomes,
                "distinct_smolvla_sites": 24,
                "whole_model_verified": False,
                "human_policy_reviewed": False,
            },
        )
    if sources != gate.source_bindings(repo_root()):
        raise ValueError("project sources changed during qualification; retain candidate evidence")
    for ref in tools.values():
        if reference(Path(ref["path"])) != ref:
            raise ValueError("tool changed during qualification")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["fixtures", "checks", "receipts"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compiler-python", type=Path)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--checker", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--sdk-bin", type=Path)
    parser.add_argument("--audits-from", type=Path)
    args = parser.parse_args()
    if args.phase == "fixtures":
        fixtures(args.output, args.compiler_python)
    elif args.phase == "checks":
        checks(args.output, args.inputs, args.checker)
    else:
        receipts(args.output, args.baseline, args.compiler_python, args.sdk_bin, args.audits_from)


if __name__ == "__main__":
    main()
