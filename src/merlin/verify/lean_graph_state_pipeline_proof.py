"""Fixed unaffected standalone pipeline proof-object profile and exact-root audit.

Foreign sources/objects and toolchain files are read-only. Own compiled objects
are copied with exact source/object identities; no failed profile is retried or
rehash-selected. Kernel/audit execution and Python report transport remain trust.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

from merlin.common.owned_process import run_owned
from merlin.common.paths import artifacts_dir, build_dir, repo_root

from . import lean_graph_state as stable
from .lean_graph_state_pipeline_kernel import ARTIFACT, BUILD, dependencies, exact, write

MODULES = (
    "GraphStatePipelineFailure",
    "GraphStatePipelineControl",
    "GraphStatePipelineIndex",
    "GraphStatePipelineScope",
    "GraphStatePipelineFold",
    "GraphStatePipelineSegments",
    "GraphStatePipelineLoops",
    "GraphStatePipelineProgram",
    "GraphStatePipelineLiterals",
)


def guard(query=None):
    base = build_dir() / BUILD
    for name in ("NATIVE_BLOCKER.json", "UNCONFIRMED_CONTEXT.json"):
        if (base / name).exists():
            raise RuntimeError("pipeline native context blocked; no retries or promotion: " + str(base / name))
    for limited in sorted(base.glob("LIMIT*.json")):
        record = json.loads(limited.read_text())
        notes = ast.literal_eval(record["cleanup"])
        cleanup = ast.literal_eval(notes[0].partition(": ")[2])
        if (
            record["cause"] != "60-second selected command limit"
            or not cleanup["direct_child_reaped"]
            or cleanup["errors"]
        ):
            raise RuntimeError("unconfirmed limit cleanup; independent native context unavailable")
        # Preserve the first exact failure, permanently refusing its bytes. A
        # known, reaped per-query budget limit does not retry or qualify that
        # route when an independent structural query is checked in a fresh process.
        failed = stable.file_ref(Path(record["arguments"][0]))
        if query is not None and (
            hashlib.sha256(query.encode()).hexdigest() == failed["sha256"]
            or ("namespace Witness\n" in query and "namespace Execution\n" in query)
        ):
            raise RuntimeError("limited literal trace route is immutable/unqualified; retry refused")


def fixed_context():
    guard()
    data = dependencies()
    try:
        for row in data["bindings"]:
            exact(row)
    except (ValueError, OSError) as error:
        write(
            build_dir() / BUILD / "UNCONFIRMED_CONTEXT.json",
            {"status": "UNQUALIFIED", "reason": str(error), "refreshed": False, "retry_permitted": False},
        )
        raise
    return data


def own_objects():
    result = []
    for name in MODULES:
        source = stable.file_ref(repo_root() / f"src/merlin/verify/lean/Merlin/{name}.lean")
        candidates = []
        for path in (build_dir() / BUILD / "development/attempts" / name).glob("*/result.json"):
            row = json.loads(path.read_text())
            if row["returncode"] == 0 and row["source"]["sha256"] == source["sha256"]:
                candidates.append((path, row))
        distinct = {row["object"]["sha256"] for _, row in candidates}
        if len(distinct) != 1:
            raise ValueError("one exact successful own source/object selection required: " + name)
        path, row = candidates[-1]
        exact(row["object"])
        result.append(
            {
                "module": "Merlin." + name,
                "source": source,
                "object": row["object"],
                "development_record": stable.file_ref(path),
            }
        )
    return result


def auditor_ref():
    # Expected auditor identity is taken from the immutable scalar delivery's
    # already-qualified source list, not from an unexplained current file hash.
    packet = artifacts_dir() / "verification/graph-state/review/sealed-v2/delivery.json"
    rows = json.loads(packet.read_text())["owned_sources"]
    selected = [r["original"] for r in rows if r["original"]["path"].endswith("/Merlin/GraphStateAudit.lean")]
    if len(selected) != 1:
        raise ValueError("exact sealed standard-three auditor source required")
    exact(selected[0])
    return selected[0]


def prepare():
    context = fixed_context()
    own = own_objects()
    auditor = auditor_ref()
    selection = {
        "own": own,
        "fixed_context": stable.file_ref(artifacts_dir() / ARTIFACT / "active/dependency-check-once.json"),
        "auditor": auditor,
        "arguments": ["-j1"],
    }
    key = hashlib.sha256(json.dumps(selection, sort_keys=True).encode()).hexdigest()
    directory = build_dir() / BUILD / "proof-profiles" / key
    lib = directory / "lib"
    ready = directory / "ready.json"
    originals = [*context["foreign_objects"]]
    originals += [r for r in context["bindings"] if r["path"].startswith(context["scalar_library"] + "/")]
    originals += [r["object"] for r in own]
    expected = [{"original": r, "copy": {**r, "path": str(lib / "Merlin" / Path(r["path"]).name)}} for r in originals]
    if not ready.exists():
        (lib / "Merlin").mkdir(parents=True, exist_ok=True)
        copies = []
        for row in originals:
            exact(row)
            target = lib / "Merlin" / Path(row["path"]).name
            if not target.exists():
                shutil.copyfile(row["path"], target)
            observed = stable.file_ref(target)
            if any(observed[k] != row[k] for k in ("sha256", "bytes")):
                raise ValueError("byte-identical profile dependency copy differs")
            copies.append({"original": row, "copy": observed})
        write(
            ready,
            {
                "schema": "merlin.graph-state-pipeline.fixed-proof-profile.v1",
                "selection": selection,
                "library": str(lib),
                "copies": copies,
                "foreign_rebuild_performed": False,
                "qualified": False,
            },
        )
    data = json.loads(ready.read_text())
    if (
        set(data) != {"schema", "selection", "library", "copies", "foreign_rebuild_performed", "qualified"}
        or data["selection"] != selection
        or data["copies"] != expected
        or data["library"] != str(lib)
        or data["foreign_rebuild_performed"] is not False
        or data["qualified"] is not False
    ):
        raise ValueError("fixed proof profile identity/copy closure differs")
    for row in data["copies"]:
        exact(row["copy"])
        exact(row["original"])
    return {**data, "manifest": stable.file_ref(ready), "directory": directory, "lib": lib, "fixed": context}


def execute(arguments, directory, environment, stdout, stderr):
    guard(
        Path(arguments[0]).read_text()
        if Path(arguments[0]).is_file() and Path(arguments[0]).suffix == ".lean"
        else None
    )
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
        limit_path = build_dir() / BUILD / "LIMIT_BLOCKER.json"
        if limit_path.exists():
            limit_path = limit_path.with_name("LIMIT_" + stable.file_ref(Path(arguments[0]))["sha256"] + ".json")
        if limit_path.exists():
            raise RuntimeError("limited query route may not be retried") from error
        write(
            limit_path,
            {
                "status": "UNQUALIFIED_LIMIT",
                "cause": "60-second selected command limit",
                "arguments": list(map(str, arguments)),
                "stdout": stable.file_ref(stdout),
                "stderr": stable.file_ref(stderr),
                "cleanup": str(error.__notes__),
                "retry_permitted": False,
            },
        )
        raise
    if result.returncode < 0:
        write(
            build_dir() / BUILD / "NATIVE_BLOCKER.json",
            {
                "status": "UNQUALIFIED_NATIVE_SIGNAL",
                "cause": "UNKNOWN",
                "arguments": list(map(str, arguments)),
                "returncode": result.returncode,
                "stdout": stable.file_ref(stdout),
                "stderr": stable.file_ref(stderr),
                "retry_permitted": False,
            },
        )
    return result


def imported_bindings(raw, profile, query_object):
    allowed = {r["path"]: r for r in profile["fixed"]["bindings"]}
    allowed.update({r["copy"]["path"]: r["copy"] for r in profile["copies"]})
    allowed[query_object["path"]] = query_object
    actual = []
    for imported in raw["imports"]:
        path = Path(imported["olean"]).resolve()
        key = str(path)
        if key not in allowed:
            write(
                build_dir() / BUILD / "UNCONFIRMED_CONTEXT.json",
                {
                    "status": "UNQUALIFIED",
                    "reason": "import outside fixed bindings",
                    "path": key,
                    "identity_adopted": False,
                    "retry_permitted": False,
                },
            )
            raise ValueError("unconfirmed import object context; no rehash-to-promote")
        exact(allowed[key])
        actual.append(allowed[key])
    return actual


def check_query(text, destination, roots, *, module="GraphStatePipelineInstance"):
    guard(text)
    profile = prepare()
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
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
    record = {
        "schema": "merlin.graph-state-pipeline.kernel.v1",
        "status": "refused" if checked.returncode > 0 else "unavailable",
        "returncode": checked.returncode,
        "source": stable.file_ref(source),
        "stdout": stable.file_ref(destination / "kernel.stdout"),
        "stderr": stable.file_ref(destination / "kernel.stderr"),
        "profile": profile["manifest"],
        "roots": roots,
        "qualified": False,
        "audit": {"status": "not_run"},
    }
    if checked.returncode == 0:
        record["object"] = stable.file_ref(target)
        measured = {}
        for name, arguments in (("audit", [module]), ("statements", ["--statements", ",".join(roots), module])):
            run = execute(
                ["--run", profile["selection"]["auditor"]["path"], *arguments],
                destination,
                environment,
                destination / (name + ".stdout"),
                destination / (name + ".stderr"),
            )
            if run.returncode:
                record["status"] = "unavailable"
                record["phase"] = name
                record["phase_returncode"] = run.returncode
                write(destination / "report.json", record)
                return record
            measured[name] = json.loads((destination / (name + ".stdout")).read_text())
        raw = measured["audit"]
        if raw["violations"] or set(raw["axioms_used"]) - {"propext", "Classical.choice", "Quot.sound"}:
            raise ValueError("actual transitive proof violates standard-three policy")
        if set(measured["statements"]["statements"]) != set(roots):
            raise ValueError("exact root statement inventory differs")
        imports = imported_bindings(raw, profile, record["object"])
        record["audit"] = {
            "status": "checked",
            "raw": raw,
            "statements": measured["statements"]["statements"],
            "imported_objects": imports,
            "auditor": profile["selection"]["auditor"],
            "buffers": [
                stable.file_ref(destination / (name + "." + suffix))
                for name in ("audit", "statements")
                for suffix in ("stdout", "stderr")
            ],
        }
        record["status"] = "conditional_graph_checked"
    fixed_context()
    for row in profile["copies"]:
        exact(row["copy"])
    exact(record["source"])
    if record.get("object"):
        exact(record["object"])
    write(destination / "report.json", record)
    return record
