"""Exact-root standard-three audits for the private integer proof-producing checker.

Copies byte-identical project objects; never rebuilds old proofs or the SDK.
Fresh replay must reproduce both theorem statements and actual imported objects.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

from merlin.common.owned_process import run_owned
from merlin.common.paths import build_dir, repo_root

from . import lean_graph_state as stable
from . import lean_graph_state_integer_kernel as kernel
from .lean_graph_state_pipeline_kernel import exact, write
from .lean_graph_state_pipeline_proof import auditor_ref

MODULES = (
    "GraphStateIntegerRepresentation",
    "GraphStateIntegerOperations",
    "GraphStateIntegerExpressions",
    "GraphStateIntegerCode",
    "GraphStateIntegerProgram",
    "GraphStateIntegerTrace",
    "GraphStateIntegerModels",
    "GraphStateIntegerCountermodels",
)


def own_objects():
    rows = []
    for name in MODULES:
        source = stable.file_ref(repo_root() / f"src/merlin/verify/lean/Merlin/{name}.lean")
        selected = []
        for path in (build_dir() / kernel.BUILD / "development/attempts" / name).glob("*/result.json"):
            value = json.loads(path.read_text())
            if value["returncode"] == 0 and value["source"]["sha256"] == source["sha256"]:
                selected.append((path, value))
        if len({value["object"]["sha256"] for _, value in selected}) != 1:
            raise ValueError("exact successful integer object selection required: " + name)
        path, value = selected[-1]
        exact(value["object"])
        rows.append({"source": source, "object": value["object"], "record": stable.file_ref(path)})
    return rows


def sources():
    root = repo_root()
    paths = [*root.glob("src/merlin/verify/lean_graph_state_integer*.py")]
    paths += list(root.glob("src/merlin/verify/lean/Merlin/GraphStateInteger*.lean"))
    paths += list(root.glob("merlin/tests/ir/test_lean_graph_state_integer*.py"))
    return [stable.file_ref(path) for path in sorted(paths)]


def prepare():
    fixed = kernel.context()
    own = own_objects()
    selection = {"own": own, "sources": sources(), "auditor": auditor_ref(), "arguments": ["-j1"]}
    key = hashlib.sha256(json.dumps(selection, sort_keys=True).encode()).hexdigest()
    directory = build_dir() / kernel.BUILD / "profiles" / key
    lib = directory / "lib"
    originals = [*fixed["objects"], *[row["object"] for row in own]]
    copies = [
        {"original": row, "copy": {**row, "path": str(lib / "Merlin" / Path(row["path"]).name)}} for row in originals
    ]
    ready = directory / "ready.json"
    data = {"schema": "merlin.graph-state-integer.profile.v1", "selection": selection, "copies": copies}
    if not ready.exists():
        (lib / "Merlin").mkdir(parents=True, exist_ok=True)
        for row in copies:
            exact(row["original"])
            if not Path(row["copy"]["path"]).exists():
                shutil.copyfile(row["original"]["path"], row["copy"]["path"])
            exact(row["copy"])
        write(ready, data)
    if json.loads(ready.read_text()) != data:
        raise ValueError("integer profile copy roster/selection differs")
    for row in copies:
        exact(row["original"])
        exact(row["copy"])
    return {**data, "manifest": stable.file_ref(ready), "fixed": fixed, "lib": lib}


def execute(arguments, directory, environment, stdout, stderr):
    kernel.guard()
    try:
        with stdout.open("x") as out, stderr.open("x") as err:
            result = run_owned(
                [str(stable.LEAN), "-j1", *map(str, arguments)],
                cwd=directory,
                env=environment,
                stdout=out,
                stderr=err,
                timeout=60,
                check=False,
            )
    except subprocess.TimeoutExpired as error:
        write(
            build_dir() / kernel.BUILD / "LIMIT_BLOCKER.json",
            {"arguments": list(map(str, arguments)), "cleanup": str(error.__notes__), "retry": False},
        )
        raise
    if result.returncode < 0:
        write(
            build_dir() / kernel.BUILD / "NATIVE_BLOCKER.json",
            {
                "arguments": list(map(str, arguments)),
                "returncode": result.returncode,
                "stdout": stable.file_ref(stdout),
                "stderr": stable.file_ref(stderr),
                "cause": "UNKNOWN",
                "retry": False,
            },
        )
    return result


def imported_bindings(raw, profile, target):
    allowed = {row["path"]: row for row in profile["fixed"]["bindings"]}
    allowed.update({row["copy"]["path"]: row["copy"] for row in profile["copies"]})
    allowed[target["path"]] = target
    selected = []
    for item in raw["imports"]:
        path = str(Path(item["olean"]).resolve())
        if path not in allowed:
            write(
                build_dir() / kernel.BUILD / "IMPORT_BLOCKER.json",
                {"path": path, "reason": "outside original selected binding closure", "retry": False},
            )
            raise ValueError("new import has no original fixed provenance: " + path)
        exact(allowed[path])
        selected.append(allowed[path])
    return selected


def check_query(text, destination, roots):
    profile = prepare()
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    module = "GraphStateIntegerInstance"
    source = destination / (module + ".lean")
    source.write_text(text)
    target = source.with_suffix(".olean")
    environment = {
        **os.environ,
        "LEAN_PATH": os.pathsep.join((str(destination), str(profile["lib"]))),
        "LEAN_SYSROOT": str(stable.LEAN.parent.parent),
        "LEAN_NUM_THREADS": "1",
        "PATH": str(stable.LEAN.parent) + os.pathsep + os.environ.get("PATH", ""),
    }
    checked = execute(
        [source, "-o", target], destination, environment, destination / "kernel.stdout", destination / "kernel.stderr"
    )
    report = {
        "schema": "merlin.graph-state-integer.kernel.v1",
        "status": "refused" if checked.returncode > 0 else "unavailable",
        "returncode": checked.returncode,
        "source": stable.file_ref(source),
        "roots": roots,
        "stdout": stable.file_ref(destination / "kernel.stdout"),
        "stderr": stable.file_ref(destination / "kernel.stderr"),
        "profile": profile["manifest"],
        "qualified": False,
        "audit": {"status": "not_run"},
    }
    if checked.returncode == 0:
        report["object"] = stable.file_ref(target)
        measured = {}
        for name, args in (("audit", [module]), ("statements", ["--statements", ",".join(roots), module])):
            run = execute(
                ["--run", profile["selection"]["auditor"]["path"], *args],
                destination,
                environment,
                destination / (name + ".stdout"),
                destination / (name + ".stderr"),
            )
            if run.returncode:
                report.update(status="unavailable", phase=name, phase_returncode=run.returncode)
                write(destination / "report.json", report)
                return report
            measured[name] = json.loads((destination / (name + ".stdout")).read_text())
        raw = measured["audit"]
        if raw["violations"] or set(raw["axioms_used"]) - {"propext", "Classical.choice", "Quot.sound"}:
            raise ValueError("integer query violates standard-three policy")
        if set(measured["statements"]["statements"]) != set(roots):
            raise ValueError("exact selected integer theorem statement roster differs")
        report["audit"] = {
            "status": "checked",
            "raw": raw,
            "statements": measured["statements"]["statements"],
            "imported_objects": imported_bindings(raw, profile, report["object"]),
            "auditor": profile["selection"]["auditor"],
            "buffers": [
                stable.file_ref(destination / (name + "." + extension))
                for name in ("audit", "statements")
                for extension in ("stdout", "stderr")
            ],
        }
        report["status"] = "conditional_graph_checked"
    kernel.context()
    for row in profile["copies"]:
        exact(row["copy"])
    for row in profile["selection"]["sources"]:
        exact(row)
    exact(report["source"])
    if report.get("object"):
        exact(report["object"])
    write(destination / "report.json", report)
    return report
