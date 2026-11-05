"""Inventory and compile an explicit kernel denominator with ordinary compiler gates.

The built-in corpus is the existing family-window corpus, not every kernel in
the paper or every operation in a full model. Saved campaign records carry no
compiler authority. Audit checks retained bytes; a new run executes new gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

from merlin.common.paths import artifacts_dir, out_dir, repo_root

ROOT = repo_root()
SCRIPT = ROOT / "build_tools/scripts/qualify_pass_cutoff.py"
SUCCESS = {"lowered_checked", "lowered_no_changed_selected_boundary"}


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def file_ref(path):
    path = Path(path).absolute()
    return {"path": str(path), "sha256": digest(path), "bytes": path.stat().st_size}


def check_ref(ref):
    if file_ref(Path(ref["path"])) != ref:
        raise ValueError("bound file changed: " + ref["path"])


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def source_bindings():
    """Freeze the compiler/checker tree and membership, beyond the input fixtures."""
    paths = {SCRIPT, ROOT / "merlin/tests/ir/lean_gate_chunks.py", ROOT / "merlin/lean/LEDGER.md"}
    for directory in ("src/merlin", "merlin/lean", "merlin/contract/schemas"):
        for folder, children, names in os.walk(ROOT / directory):
            children[:] = [name for name in children if name not in {"out", ".lake", "__pycache__"}]
            paths.update(
                path
                for name in names
                if (path := Path(folder) / name).suffix in {".py", ".lean", ".toml", ".json", ".c", ".h"}
            )
    paths.update(ROOT / "merlin/lean" / name for name in ("SPEC_LOCK", "lean-toolchain"))
    paths.update(ROOT / name for name in ("pyproject.toml", "uv.lock", ".env") if (ROOT / name).is_file())
    return [file_ref(path) for path in sorted(paths)]


def builtin_inputs():
    """Keep every authored capsule and every emitted window, including failures."""
    sys.path.insert(0, str(ROOT / "merlin/tests/ir"))
    import lean_gate_chunks as corpus

    from merlin.llvmlower.passes_xdsl import parse_mlir_text

    inputs, dependencies, errors, coverage = [], [], [], []
    for capsule in corpus.CAPSULES:
        source = ROOT / "merlin/contract/capsules/model" / capsule / "capsule.interface.mlir"
        try:
            dependencies.append(file_ref(source))
            module = parse_mlir_text(source.read_text())
            families = {family for op in module.walk() if (family := corpus._family(op)) is not None}
            chunks = corpus.chunks(capsule)
            covered = {
                family
                for chunk in chunks
                for op in parse_mlir_text(chunk.text).walk()
                if (family := corpus._family(op)) is not None
            }
            if not families or not chunks or not families <= covered:
                raise ValueError("family coverage is incomplete: " + str(sorted(families - covered)))
            coverage.append({"capsule": capsule, "families": sorted(families), "windows": len(chunks)})
            for chunk in chunks:
                for variant in ("text", "pruned"):
                    inputs.append((f"{chunk.name}/{variant}", getattr(chunk, variant)))
        except Exception as error:  # An absent capsule must remain in the denominator record.
            errors.append({"capsule": capsule, "error": type(error).__name__ + ": " + str(error)})
    return inputs, dependencies, errors, coverage


def external_inputs(path):
    document = json.loads(path.read_text())
    if document.get("schema") != "merlin.pass-cutoff.inputs.v1" or not document.get("inputs"):
        raise ValueError("external manifest needs schema merlin.pass-cutoff.inputs.v1 and nonempty inputs")
    inputs, dependencies, errors = [], [file_ref(path)], []
    for number, row in enumerate(document["inputs"]):
        try:
            source = (path.parent / row["path"]).absolute()
            ref = file_ref(source)
            if ref["sha256"] != row["sha256"]:
                raise ValueError("external source hash differs")
            inputs.append((row["id"], source.read_text()))
            dependencies.append(ref)
        except (OSError, ValueError, KeyError, TypeError) as error:
            errors.append({"declared_input": row, "position": number, "error": str(error)})
    return inputs, dependencies, errors, [{"external_scope": document.get("scope", "operator-authored inputs")}]


def inventory(destination, *, external=None, routes=("xdsl",)):
    from merlin.verify.pass_validation import DEFAULT_CHECKS

    destination.mkdir(parents=True, exist_ok=False)
    try:
        inputs, dependencies, errors, coverage = builtin_inputs() if external is None else external_inputs(external)
        names = [name for name, _ in inputs]
        if any(type(name) is not str or not name for name in names) or len(set(names)) != len(names):
            raise ValueError("case identifiers must be nonempty and unique")
        if not routes or len(set(routes)) != len(routes) or not set(routes) <= {"xdsl", "textual"}:
            raise ValueError("routes must be a nonempty unique xdsl/textual selection")
        cases = []
        for name, source in inputs:
            for route in routes:
                number = len(cases)
                path = destination / f"{number:04d}.mlir"
                path.write_text(source)
                cases.append(
                    {
                        "id": name + "/" + route,
                        "input": file_ref(path),
                        "route": route,
                        "selectors": sorted(DEFAULT_CHECKS),
                    }
                )
        manifest = {
            "schema": "merlin.pass-cutoff.inventory.v1",
            "scope": "external_authored_inputs" if external else "retained_model_family_windows",
            "status": "incomplete" if errors or not cases else "frozen",
            "cases": cases,
            "errors": errors,
            "coverage": coverage,
            "selectors": sorted(DEFAULT_CHECKS),
            "sources": source_bindings(),
            "dependencies": dependencies,
            "native_execution": False,
            "paper_corpus_complete": False,
        }
    except Exception as error:
        manifest = {
            "schema": "merlin.pass-cutoff.inventory.v1",
            "status": "incomplete",
            "cases": [],
            "errors": [{"error": type(error).__name__ + ": " + str(error)}],
        }
    write(destination / "inventory.json", manifest)
    return manifest


def validate_manifest(path, expected=None):
    if expected is not None:
        check_ref(expected)
        if str(path.absolute()) != expected["path"]:
            raise ValueError("different inventory path")
    manifest = json.loads(path.read_text())
    if manifest.get("schema") != "merlin.pass-cutoff.inventory.v1" or manifest.get("status") != "frozen":
        raise ValueError("complete frozen inventory required")
    cases = manifest["cases"]
    ids = [case["id"] for case in cases]
    if not cases or len(ids) != len(set(ids)):
        raise ValueError("empty or duplicate case denominator")
    if source_bindings() != manifest["sources"]:
        raise ValueError("compiler/checker source cut changed; create a new inventory")
    for ref in [*manifest["dependencies"], *(case["input"] for case in cases)]:
        check_ref(ref)
    from merlin.verify.pass_validation import DEFAULT_CHECKS

    if manifest["selectors"] != sorted(DEFAULT_CHECKS):
        raise ValueError("default selector roster changed")
    if any(case["selectors"] != manifest["selectors"] or case["route"] not in {"xdsl", "textual"} for case in cases):
        raise ValueError("case route or selector profile differs")
    return manifest


def _lower(source, destination, *, textual, validation=None):
    from merlin.llvmlower.lower import lower_model
    from merlin.verify.compiler_transforms import Options

    return lower_model(
        source,
        destination,
        targets=(),
        textual=textual,
        pass_validation=validation,
        formal_transforms=Options(),
        static_arena=False,
    )


def summarize_case(baseline, checked, jobs):
    """No error-parity, skipped evidence or saved accepted string establishes success."""
    result = {
        "baseline": baseline,
        "checked": checked,
        "jobs": jobs,
        "positive_checked_boundaries": 0,
        "completed_jobs": len(jobs),
        "accepted_jobs": sum(job["verdict"] == "accepted" for job in jobs),
        "changed_jobs": sum(bool(job.get("changed")) for job in jobs),
    }
    if baseline["status"] != "lowered":
        result["status"] = "baseline_lowering_failed"
    elif checked["status"] != "lowered":
        result["status"] = checked["status"]
    elif any(job["verdict"] != "accepted" for job in jobs):
        result["status"] = "gate_not_accepted"
    else:
        try:
            check_ref(baseline["output"])
            check_ref(checked["output"])
            for job in jobs:
                if len(job["bindings"]) < 4:
                    raise ValueError("job lacks its request, verdict and exact boundary artifacts")
                for ref in job["bindings"]:
                    check_ref(ref)
                if not job["changed"] or job["bindings"][2]["sha256"] == job["bindings"][3]["sha256"]:
                    raise ValueError("unchanged job cannot receive positive rewrite credit")
        except (OSError, ValueError, KeyError) as error:
            result.update(status="unavailable", error=str(error))
            return result
        if baseline["output"]["sha256"] != checked["output"]["sha256"]:
            result["status"] = "llvm_mismatch"
        else:
            result["status"] = "lowered_checked" if jobs else "lowered_no_changed_selected_boundary"
            result["positive_checked_boundaries"] = len(jobs)
    return result


def compile_case(case, destination, tools, selectors, *, job_timeout):
    from merlin.verify.gate_errors import GateUnavailable, GateUnsupported
    from merlin.verify.pass_validation import PassValidation
    from merlin.verify.speculative_gate import GateFailure, SpeculativeGate

    source = Path(case["input"]["path"]).read_text()
    worker = SpeculativeGate(destination / "gate", job_timeout=job_timeout)
    validation = PassValidation(worker, tools, frozenset(selectors))
    arms = {}
    try:
        for name, selected in (("baseline", None), ("checked", validation)):
            try:
                result = _lower(source, destination / name, textual=case["route"] == "textual", validation=selected)
                if not result.ll_path.is_file() or not result.ll_path.stat().st_size:
                    raise OSError("compiler returned no nonempty LLVM artifact")
                arms[name] = {"status": "lowered", "output": file_ref(result.ll_path), "stats": result.stats}
            except (Exception, KeyboardInterrupt) as error:
                if isinstance(error, KeyboardInterrupt):
                    raise
                status = "lowering_failed"
                if isinstance(error, GateFailure):
                    status = "gate_" + error.report["verdict"]
                elif isinstance(error, GateUnsupported):
                    status = "gate_unsupported"
                elif isinstance(error, (GateUnavailable, OSError, TimeoutError)):
                    status = "unavailable"
                arms[name] = {"status": status, "error": type(error).__name__ + ": " + str(error)}
    finally:
        worker.close()
    jobs = []
    for value in worker.verdicts:
        folder = worker.directory / value["job"]
        record = {
            "check": value["check"],
            "label": value["label"],
            "verdict": value["verdict"],
            "changed": False,
            "bindings": [],
            "detail": value.get("detail", {}),
        }
        try:
            required = [folder / name for name in ("request.json", "verdict.json", "before.mlir", "after.mlir")]
            evidence = sorted(path for path in folder.rglob("*") if path.is_file() and path not in required)
            bindings = [file_ref(path) for path in (*required, *evidence)]
            record.update(bindings=bindings, changed=bindings[2]["sha256"] != bindings[3]["sha256"])
        except OSError as error:
            record.update(verdict="unavailable", error=str(error))
        jobs.append(record)
    return summarize_case(arms["baseline"], arms["checked"], jobs)


def child(manifest_path, destination, number, manifest_sha, *, job_timeout):
    from merlin.verify.pass_validation import CheckerTools

    manifest = validate_manifest(manifest_path)
    if digest(manifest_path) != manifest_sha:
        raise ValueError("inventory bytes changed before child launch")
    case = manifest["cases"][number]
    tools = CheckerTools(Path(os.environ["MERLIN_LEAN_SDK_BIN"]), Path(os.environ["MERLIN_COMPILER_PYTHON"]))
    result = compile_case(case, destination, tools, manifest["selectors"], job_timeout=job_timeout)
    validate_manifest(manifest_path)
    result.update(id=case["id"], input=case["input"], selectors=manifest["selectors"])
    write(destination / "case.json", result)


def summary(manifest, records):
    expected = [case["id"] for case in manifest["cases"]]
    actual = [record["id"] for record in records]
    complete = bool(expected) and actual == expected and len(set(actual)) == len(actual)
    counts = dict(Counter(record["status"] for record in records))
    return {
        "expected_cases": len(expected),
        "recorded_cases": len(records),
        "complete": complete,
        "all_lowered": complete and all(record["status"] in SUCCESS for record in records),
        "counts": counts,
        "positive_checked_boundaries": sum(r.get("positive_checked_boundaries", 0) for r in records),
        "completed_jobs": sum(r.get("completed_jobs", 0) for r in records),
        "accepted_jobs": sum(r.get("accepted_jobs", 0) for r in records),
        "changed_jobs": sum(r.get("changed_jobs", 0) for r in records),
        "jobs_by_check": dict(Counter(job["check"] for r in records for job in r.get("jobs", []))),
        "jobs_by_label": dict(Counter(job["label"] for r in records for job in r.get("jobs", []))),
        "native_execution": False,
        "paper_corpus_complete": False,
        "saved_record_authority": False,
    }


def runtime_bindings():
    """Retain selected tools and effective environment without claiming a dependency closure."""
    from merlin.verify.lean.check import lean_exe

    paths = [Path(sys.executable), lean_exe(), lean_exe().with_name("merlin-lean-audit")]
    missing = []
    for key, children in (("MERLIN_LEAN_SDK_BIN", ("lean", "lake")), ("MERLIN_COMPILER_PYTHON", ("",))):
        if not os.environ.get(key):
            missing.append(key + " is unset")
        else:
            base = Path(os.environ[key])
            paths.extend(base / name if name else base for name in children)
    references = []
    for path in paths:
        try:
            references.append(file_ref(path))
        except OSError as error:
            missing.append(str(error))
    keys = (
        "PATH",
        "PYTHONPATH",
        "C_INCLUDE_PATH",
        "LD_LIBRARY_PATH",
        "MERLIN_COMPILER_PYTHON",
        "MERLIN_LEAN_SDK_BIN",
        "MERLIN_CLANG",
        "MERLIN_MLIR_INSTALL",
        "MERLIN_COMPILER_DRIVER",
    )
    return {
        "files": references,
        "missing": missing,
        "python_version": sys.version,
        "environment": {key: os.environ.get(key) for key in keys},
        "dependency_closure_qualified": False,
    }


def run(manifest_path, destination, *, timeout=1200, job_timeout=900):
    from merlin.common.owned_process import run as owned_run

    manifest = validate_manifest(manifest_path)
    bound = file_ref(manifest_path)
    destination.mkdir(parents=True, exist_ok=False)
    records, references = [], []
    environment = dict(os.environ)
    # Explicit Options() and static_arena=False fix transform selection for both arms.
    if any(
        value
        for key, value in environment.items()
        if key.startswith("MERLIN_VALIDATE") or key.startswith("MERLIN_FORMAL")
    ):
        raise ValueError("remove ambient validation/formal-transform options before this fixed-profile campaign")
    context = runtime_bindings()
    for number, case in enumerate(manifest["cases"]):
        folder = destination / f"{number:04d}"
        folder.mkdir()
        try:
            validate_manifest(manifest_path, bound)
            proc = owned_run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "_case",
                    "--inventory",
                    str(manifest_path),
                    "--output",
                    str(folder),
                    "--number",
                    str(number),
                    "--manifest-sha",
                    bound["sha256"],
                    "--job-timeout",
                    str(job_timeout),
                ],
                cwd=ROOT,
                env=environment,
                timeout=timeout,
                text=True,
            )
            (folder / "stdout.log").write_text(proc.stdout)
            (folder / "stderr.log").write_text(proc.stderr)
            if proc.returncode != 0:
                raise RuntimeError(f"case process exited {proc.returncode}; see retained stderr.log")
            record = json.loads((folder / "case.json").read_text())
            if record["id"] != case["id"]:
                raise ValueError("case result has a different identity")
            validate_manifest(manifest_path, bound)
        except Exception as error:
            record = {
                "id": case["id"],
                "status": "unavailable",
                "positive_checked_boundaries": 0,
                "error": type(error).__name__ + ": " + str(error),
            }
            if isinstance(error, subprocess.TimeoutExpired):
                for name, value in (("stdout", error.stdout), ("stderr", error.stderr)):
                    (folder / (name + ".log")).write_text(
                        value.decode(errors="replace") if isinstance(value, bytes) else value or ""
                    )
            write(folder / "failure.json", record)
        records.append(record)
        references.append(
            file_ref(
                folder / ("case.json" if (folder / "case.json").is_file() and record.get("input") else "failure.json")
            )
        )
        report = {
            "schema": "merlin.pass-cutoff.campaign.v1",
            "inventory": bound,
            "records": references,
            "results": records,
            "summary": summary(manifest, records),
            "runtime": context,
            "profile": {
                "targets": [],
                "formal_transforms": "disabled",
                "static_arena": False,
                "case_timeout_seconds": timeout,
                "job_timeout_seconds": job_timeout,
            },
        }
        write(destination / "campaign.json", report)
        print(json.dumps({"case": case["id"], "status": record["status"]}), flush=True)
    if runtime_bindings() != context:
        report["runtime_changed"] = True
        report["summary"]["all_lowered"] = False
        write(destination / "campaign.json", report)
    return report


def audit(path):
    report = json.loads(path.read_text())
    if report.get("schema") != "merlin.pass-cutoff.campaign.v1":
        raise ValueError("campaign schema differs")
    if report.get("runtime_changed"):
        raise ValueError("tool or environment changed during campaign")
    if runtime_bindings() != report["runtime"]:
        raise ValueError("tool or environment differs from tested campaign")
    check_ref(report["inventory"])
    manifest = validate_manifest(Path(report["inventory"]["path"]))
    records = []
    for number, ref in enumerate(report["records"]):
        check_ref(ref)
        record = json.loads(Path(ref["path"]).read_text())
        if "baseline" in record:
            case = manifest["cases"][number]
            if record["input"] != case["input"] or record["selectors"] != case["selectors"]:
                raise ValueError("case input or selector binding differs")
            recomputed = summarize_case(record["baseline"], record["checked"], record["jobs"])
            if any(
                recomputed[key] != record[key]
                for key in ("status", "positive_checked_boundaries", "completed_jobs", "accepted_jobs", "changed_jobs")
            ):
                raise ValueError("retained result differs from actual artifact bindings")
        elif record["status"] != "unavailable":
            raise ValueError("a case without compiler records cannot claim a lowering outcome")
        records.append(record)
    if records != report["results"] or summary(manifest, records) != report["summary"]:
        raise ValueError("campaign summary or denominator differs")
    return {"retained_bytes_consistent": True, "fresh_proof": False, **report["summary"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("inventory", "run", "audit", "_case"))
    parser.add_argument("--output", type=Path, default=artifacts_dir() / "verification/pass-cutoff")
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--external-inputs", type=Path)
    parser.add_argument("--routes", nargs="+", default=["xdsl"])
    parser.add_argument("--timeout", type=float, default=1200)
    parser.add_argument("--job-timeout", type=float, default=900)
    parser.add_argument("--number", type=int)
    parser.add_argument("--manifest-sha")
    args = parser.parse_args()
    if args.mode in {"run", "_case"} and args.inventory is None:
        parser.error("--inventory is required for run and _case")
    if args.mode == "_case" and (args.number is None or args.manifest_sha is None):
        parser.error("_case requires --number and --manifest-sha")
    if args.timeout <= 0 or args.job_timeout <= 0:
        parser.error("timeouts must be positive")
    output = args.output.absolute()
    if not output.resolve().is_relative_to(out_dir().resolve()):
        parser.error("generated output must be beneath the configured out root")
    if args.mode == "inventory":
        result = inventory(output, external=args.external_inputs, routes=tuple(args.routes))
        success = result["status"] == "frozen"
        result = {
            "status": result["status"],
            "cases": len(result["cases"]),
            "errors": result["errors"],
            "inventory": str(output / "inventory.json"),
        }
    elif args.mode == "run":
        result = run(args.inventory.absolute(), output, timeout=args.timeout, job_timeout=args.job_timeout)
        success = result["summary"]["all_lowered"]
    elif args.mode == "audit":
        result = audit(output / "campaign.json")
        success = result["complete"]
    else:
        child(args.inventory.absolute(), output, args.number, args.manifest_sha, job_timeout=args.job_timeout)
        return
    print(json.dumps(result.get("summary", result), indent=2))
    raise SystemExit(0 if success else 1)


if __name__ == "__main__":
    main()
