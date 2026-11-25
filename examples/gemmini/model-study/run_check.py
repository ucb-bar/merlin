#!/usr/bin/env python3
"""Run ordinary model_check with a separately pinned candidate Python environment.

The evaluator uses the selected checkout. Only the frozen compiler entrypoint
receives the archived compiler library. Numerical and instruction checks remain
the ordinary model_check checks. No FPGA job is submitted by this driver.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

from merlin.common import owned_process


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--preparation-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    raw = args.preparation.read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.preparation_sha256:
        raise ValueError("selected preparation record changed")
    selection = json.loads(raw)
    if selection["status"] != "prepared":
        raise ValueError("selected compiler preparation did not complete")
    arguments = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
    if not arguments or arguments[0] not in {"replay-kernel", "check", "bundle-verify"}:
        raise ValueError("select an ordinary model_check replay, check or bundle verification")
    if arguments[0] != "bundle-verify" and not selection.get("routing_artifacts", {}).get("ready"):
        raise ValueError("selected routing artifacts are missing or unverified; preserve the bundle and recover them separately")
    forbidden = {"--submission", "--models", "--bundle", "--work", "--out"}
    if any(token.split("=", 1)[0] in forbidden for token in arguments):
        raise ValueError("compiler, bundle, models and output come only from the pinned preparation")
    compiler = Path(selection["compiler"]["path"])
    spec = importlib.util.spec_from_file_location("model_study_preparation", Path(__file__).with_name("prepare.py"))
    preparation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(preparation)

    def verify():
        preparation.verify_compiler(compiler, Path(selection["compiler"]["inventory"]["path"]),
                                    selection["compiler"]["source_sha256"])
        preparation.tree_pin(Path(selection["compiler_library"]["path"]), selection["compiler_library"]["source_sha256"])
        for row in [selection["models"], selection["bundle_manifest"], selection["python"],
                    *selection.get("selected_metadata", [])]:
            if preparation.pin(row["path"]) != row:
                raise ValueError("selected study metadata changed")
        runtime = selection["runtime"]["observed"]
        for row in [*runtime["modules"].values(), *runtime.get("solver_libraries", [])]:
            if preparation.sha(row["path"]) != row.get("source_sha256", row.get("sha256")):
                raise ValueError("selected proof runtime changed")

    verify()
    python = Path(selection["runtime"]["argv"][0])
    output = args.output.resolve()
    if output.exists() or not 1 <= args.timeout <= 14400:
        raise ValueError("select a fresh output and timeout in 1..14400 seconds")
    for root in (compiler, Path(selection["bundle_manifest"]["path"]).parent,
                 Path(selection["compiler_library"]["path"])):
        if output.is_relative_to(root):
            raise ValueError("output overlaps a frozen input")
    output.mkdir(parents=True)
    wrapper_dir = output / "compiler-python"
    wrapper_dir.mkdir()
    origins = output / "compiler-imports"
    origins.mkdir()
    bootstrap = output / "candidate-python.py"
    bootstrap.write_bytes(Path(__file__).with_name("candidate_python.py").read_bytes())
    # The manifest declares python3. Resolve only its known original entrypoint
    # through this wrapper, preserving the evaluator's environment for all else.
    settings = {"python": str(python), "entrypoint": str(compiler / "compiler.py"),
                "pythonpath": os.pathsep.join([str(compiler / "compiler_library"),
                                                selection["compiler_library"]["path"]]),
                "compiler_roots": [str(compiler), selection["compiler_library"]["path"]],
                "dependency_roots": sorted({str(Path(row["path"]).parent.parent)
                                            for row in selection["runtime"]["observed"]["modules"].values()}),
                "origins": str(origins)}
    settings_file = output / "candidate-environment.json"
    settings_file.write_text(json.dumps(settings, indent=2) + "\n")
    wrapper = wrapper_dir / "python3"
    wrapper.write_text("#!/usr/bin/python3\nimport os,sys\nfrom pathlib import Path\n"
                       + "selection=" + repr(settings) + "\n"
                       + "env=dict(os.environ)\n"
                       + "args=sys.argv[1:]\n"
                       + "if any(not a.startswith('-') and str(Path(a).resolve()) == selection['entrypoint'] for a in sys.argv[1:]):\n"
                       + " env['PYTHONPATH']=selection['pythonpath']+os.pathsep+os.pathsep.join(selection['dependency_roots'])\n"
                       + " args=['-S'," + repr(str(bootstrap)) + "," + repr(str(settings_file)) + ",*args]\n"
                       + "env['PYTHONDONTWRITEBYTECODE']='1'\n"
                       + "os.execve(selection['python'],[selection['python'],'-B',*args],env)\n")
    wrapper.chmod(0o755)
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    environment["PATH"] = str(wrapper_dir) + os.pathsep + environment.get("PATH", "")
    temporary = output / "tmp"
    temporary.mkdir()
    environment["TMPDIR"] = str(temporary)
    environment["MERLIN_OUT_ROOT"] = str(output / "out")
    command = [str(python), "-B", "-m", "merlin.model_check", *arguments,
               "--models", selection["models"]["path"], "--bundle",
               str(Path(selection["bundle_manifest"]["path"]).parent),
               "--work", str(output / "work"), "--out", str(output / "report.json")]
    if arguments[0] != "bundle-verify":
        command += ["--submission", str(compiler)]
    record = {"schema": "merlin.model_study.invocation.v1", "argv": command,
              "preparation": {"path": str(args.preparation.resolve()), "sha256": args.preparation_sha256},
              "compiler_source_sha256": selection["compiler"]["source_sha256"],
              "compiler_python_wrapper": {"path": str(wrapper), "sha256": hashlib.sha256(wrapper.read_bytes()).hexdigest()},
              "candidate_bootstrap": preparation.pin(bootstrap), "candidate_configuration": preparation.pin(settings_file),
              "candidate_environment": settings, "timeout_s": args.timeout,
              "status": "running", "hardware_execution": "not_run"}
    record_path = output / "invocation.json"
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    try:
        result = owned_process.run(command, env=environment, cwd=str(output),
                                   capture_output=True, timeout=args.timeout)
        (output / "stdout").write_bytes(result.stdout)
        (output / "stderr").write_bytes(result.stderr)
        record.update(status="completed", returncode=result.returncode)
        returncode = result.returncode
    except Exception as error:
        (output / "stdout").write_bytes(getattr(error, "output", b"") or b"")
        (output / "stderr").write_bytes(getattr(error, "stderr", b"") or b"")
        record.update(status="interrupted_or_failed", error=f"{type(error).__name__}: {error}")
        returncode = 1
    if args.preparation.read_bytes() != raw:
        record.update(status="inputs_changed")
        returncode = 1
    try:
        verify()
    except Exception as error:
        record.update(status="inputs_changed", error=str(error))
        returncode = 1
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"status": record["status"], "returncode": returncode, "output": str(output)}))
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
