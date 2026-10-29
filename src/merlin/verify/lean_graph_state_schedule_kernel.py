"""Private scheduling proof context on original unaffected core bindings.

All95 old sources/objects/capsules are read-only. No old context is invoked in
order to repair/adopt a changed binding. Native failure holds this new context.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

from merlin.common.owned_process import run_owned
from merlin.common.paths import artifacts_dir, build_dir, repo_root

from . import lean_graph_state as stable
from .lean_graph_state_integer_proof import own_objects as integer_objects
from .lean_graph_state_pipeline_kernel import exact, write

ARTIFACT = "verification/graph-state-schedule"
BUILD = "graph-verification/state-schedule"
INTEGER_MANIFEST = "5a2552cd32bba11b06acc43c2218abed0f8ace4639d99deb814623eb947194fa"


def guard():
    hold = artifacts_dir() / ARTIFACT / "active/SELECTED_RUNTIME_HOLD.json"
    if hold.exists():
        raise RuntimeError("SHARED SELECTED RUNTIME HOLD: no SDK reads, native launch, retry or alternate profile")
    if any((build_dir() / BUILD).glob("*BLOCKER*.json")):
        raise RuntimeError("schedule context held; no retry/repair/promotion")


def context():
    guard()
    path = artifacts_dir() / ARTIFACT / "active/dependency-check-once.json"
    if path.exists():
        data = json.loads(path.read_text())
    else:
        prior = artifacts_dir() / "verification/graph-state-integer/active/dependency-check-once.json"
        old = json.loads(prior.read_text())
        metadata = json.loads((path.parent / "metadata-incident-membership-once.json").read_text())
        if metadata["clang_named_selected_bindings"]:
            raise RuntimeError("suspect auxiliary module in selected context; do not reread or retry it")
        packet = artifacts_dir() / "verification/graph-state-integer/review/sealed-v2"
        manifest = stable.file_ref(packet / "MANIFEST.json")
        if manifest["sha256"] != INTEGER_MANIFEST:
            raise ValueError("accepted sealed integer source intake differs")
        delivery = json.loads((packet / "delivery.json").read_text())
        own = integer_objects()
        rows = [*old["bindings"], stable.file_ref(prior), manifest, stable.file_ref(packet / "delivery.json")]
        rows += [r["original"] for r in delivery["owned_sources"]]
        rows += [r[key] for r in own for key in ("source", "object", "record")]
        indexed = {}
        for row in rows:
            if row["path"] in indexed and indexed[row["path"]] != row:
                raise ValueError("conflicting original schedule dependency provenance")
            indexed[row["path"]] = row
        mismatches = []
        for row in indexed.values():
            try:
                exact(row)
            except (OSError, ValueError) as error:
                mismatches.append(str(error))
        data = {
            "schema": "merlin.graph-state-schedule.fixed-context.v1",
            "exact": not mismatches,
            "mismatches": mismatches,
            "bindings": list(indexed.values()),
            "objects": [*old["objects"], *[r["object"] for r in own]],
            "prior_owned_sources": [*old["prior_owned_sources"], *[r["original"] for r in delivery["owned_sources"]]],
            "Std_Time_target_selected": old["incident_target_in_closure"],
            "auxiliary_clang_selected": False,
            "new_expected_hashes_adopted": False,
            "SDK_written": False,
            "qualification_issued": False,
        }
        write(path, data)
    if not data["exact"]:
        raise RuntimeError("original selected context affected/unconfirmed; source work only")
    try:
        for row in data["bindings"]:
            exact(row)
    except (OSError, ValueError) as error:
        write(build_dir() / BUILD / "BINDING_BLOCKER.json", {"reason": str(error), "retry": False})
        raise
    return data


def environment():
    data = context()
    lib = build_dir() / BUILD / "development/lib"
    (lib / "Merlin").mkdir(parents=True, exist_ok=True)
    for row in data["objects"]:
        target = lib / "Merlin" / Path(row["path"]).name
        if not target.exists():
            shutil.copyfile(row["path"], target)
        exact({**row, "path": str(target)})
    return {**os.environ, "LEAN_PATH": str(lib), "LEAN_NUM_THREADS": "1"}


def invoke(source):
    source = Path(source).resolve()
    if not source.name.startswith("GraphStateSchedule") or source.suffix != ".lean":
        raise ValueError("owned scheduling source required")
    env = environment()
    content = source.read_bytes()
    imports = []
    for line in content.decode().splitlines():
        if line.startswith("import Merlin.GraphStateSchedule"):
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
                cwd=directory, env=env, stdout=out, stderr=err, timeout=60, check=False,
            )
    except subprocess.TimeoutExpired as error:
        write(build_dir() / BUILD / "LIMIT_BLOCKER.json", {"source": stable.file_ref(copy), "cleanup": str(error.__notes__), "retry": False})
        raise
    record = {"returncode": result.returncode, "source": stable.file_ref(copy),
              "stdout": stable.file_ref(directory / "stdout"), "stderr": stable.file_ref(directory / "stderr"),
              "object": stable.file_ref(target) if result.returncode == 0 else None, "qualified": False}
    write(directory / "result.json", record)
    if result.returncode < 0:
        write(build_dir() / BUILD / "NATIVE_BLOCKER.json", {**record, "cause": "UNKNOWN", "retry": False})
    context()
    return record
