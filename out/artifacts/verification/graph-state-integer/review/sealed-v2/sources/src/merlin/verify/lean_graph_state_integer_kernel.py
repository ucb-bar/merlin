"""Private exact-integer proof development on the sealed pure-core bindings.

All old sources, objects, profiles and failures are read-only. A new unknown
native fault or binding mismatch holds this context; no refresh or retry.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

from merlin.common.owned_process import run_owned
from merlin.common.paths import artifacts_dir, build_dir

from . import lean_graph_state as stable
from . import lean_graph_state_pipeline_proof as pipeline
from .lean_graph_state_pipeline_kernel import exact, write

ARTIFACT = "verification/graph-state-integer"
BUILD = "graph-verification/state-integer"
PRIOR = (
    "graph-state/review/sealed-v2",
    "graph-state-chain/review/sealed-packed-v1",
    "graph-state-elision/review/sealed-diagnostic-v1",
    "graph-state-scalar/review/sealed-v1",
    "graph-state-pipeline/review/sealed-v1",
)


def guard():
    base = build_dir() / BUILD
    if any(base.glob("*BLOCKER*.json")):
        raise RuntimeError("integer development context held; inspect immutable first failure")


def context():
    guard()
    path = artifacts_dir() / ARTIFACT / "active/dependency-check-once.json"
    if path.exists():
        value = json.loads(path.read_text())
        if not value["exact"]:
            raise RuntimeError("integer fixed context affected/unconfirmed")
    else:
        old = pipeline.fixed_context()
        prior, receipts = [], []
        for name in PRIOR:
            packet = artifacts_dir() / "verification" / name / "delivery.json"
            receipts.append(stable.file_ref(packet))
            prior += [row["original"] for row in json.loads(packet.read_text())["owned_sources"]]
        own = pipeline.own_objects()
        rows = [*old["bindings"], *prior, *receipts]
        rows += [row[key] for row in own for key in ("source", "object", "development_record")]
        indexed = {}
        for row in rows:
            if row["path"] in indexed and indexed[row["path"]] != row:
                raise ValueError("conflicting original dependency provenance")
            indexed[row["path"]] = row
        mismatches = []
        for row in indexed.values():
            try:
                exact(row)
            except (OSError, ValueError) as error:
                mismatches.append(str(error))
        value = {
            "schema": "merlin.graph-state-integer.fixed-context.v1",
            "exact": not mismatches,
            "mismatches": mismatches,
            "bindings": list(indexed.values()),
            "prior_owned_sources": prior,
            "old_context": stable.file_ref(
                artifacts_dir() / "verification/graph-state-pipeline/active/dependency-check-once.json"
            ),
            "objects": [*old["foreign_objects"]]
            + [row for row in old["bindings"] if row["path"].startswith(old["scalar_library"] + "/")]
            + [row["object"] for row in own],
            "incident_target_in_closure": old["target_in_selected_closure"],
            "sdk_written": False,
            "new_expected_hashes_adopted": False,
            "qualified": False,
        }
        write(path, value)
    try:
        for row in value["bindings"]:
            exact(row)
    except (OSError, ValueError) as error:
        write(build_dir() / BUILD / "BINDING_BLOCKER.json", {"reason": str(error), "retry": False})
        raise
    if not value["exact"]:
        raise RuntimeError("integer fixed context mismatch; source-only work continues")
    return value


def environment():
    selected = context()
    lib = build_dir() / BUILD / "development/lib"
    (lib / "Merlin").mkdir(parents=True, exist_ok=True)
    for row in selected["objects"]:
        target = lib / "Merlin" / Path(row["path"]).name
        if not target.exists():
            shutil.copyfile(row["path"], target)
        exact({**row, "path": str(target)})
    return {**os.environ, "LEAN_PATH": str(lib), "LEAN_NUM_THREADS": "1"}


def invoke(source):
    source = Path(source).resolve()
    if not source.name.startswith("GraphStateInteger") or source.suffix != ".lean":
        raise ValueError("owned integer proof source required")
    env = environment()
    content = source.read_bytes()
    imports = []
    for line in content.decode().splitlines():
        if line.startswith("import Merlin.GraphStateInteger"):
            name = line.removeprefix("import Merlin.")
            imports.append(stable.file_ref(build_dir() / BUILD / "development/lib/Merlin" / (name + ".olean")))
    selection = {"source_sha256": hashlib.sha256(content).hexdigest(), "owned_imports": imports}
    key = hashlib.sha256(json.dumps(selection, sort_keys=True).encode()).hexdigest()
    directory = build_dir() / BUILD / "development/attempts" / source.stem / key
    directory.mkdir(parents=True, exist_ok=False)
    copy = directory / source.name
    copy.write_bytes(content)
    write(directory / "selection.json", selection)
    target = build_dir() / BUILD / "development/lib/Merlin" / (source.stem + ".olean")
    try:
        with (directory / "stdout").open("x") as out, (directory / "stderr").open("x") as err:
            result = run_owned(
                [str(stable.LEAN), "-j1", str(copy), "-o", str(target)],
                cwd=directory,
                env=env,
                stdout=out,
                stderr=err,
                timeout=60,
                check=False,
            )
    except subprocess.TimeoutExpired as error:
        write(
            build_dir() / BUILD / "LIMIT_BLOCKER.json",
            {"source": stable.file_ref(copy), "cleanup": str(error.__notes__), "retry": False},
        )
        raise
    record = {
        "returncode": result.returncode,
        "source": stable.file_ref(copy),
        "stdout": stable.file_ref(directory / "stdout"),
        "stderr": stable.file_ref(directory / "stderr"),
        "object": stable.file_ref(target) if result.returncode == 0 else None,
        "qualified": False,
    }
    write(directory / "result.json", record)
    if result.returncode < 0:
        write(build_dir() / BUILD / "NATIVE_BLOCKER.json", {**record, "cause": "UNKNOWN", "retry": False})
    context()
    return record
