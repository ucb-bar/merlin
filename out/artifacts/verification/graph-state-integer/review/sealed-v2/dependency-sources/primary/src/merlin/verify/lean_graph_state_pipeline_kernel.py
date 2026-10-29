"""Private pipeline checks on unchanged sealed scalar/W objects and core bindings.

Neither foreign modules nor toolchain files are rebuilt. A changed binding or
unknown native signal stops this context; old evidence cannot replace it.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

from merlin.common.owned_process import run_owned
from merlin.common.paths import artifacts_dir, build_dir, repo_root

from . import lean_graph_state as stable

BUILD = "graph-verification/state-pipeline"
ARTIFACT = "verification/graph-state-pipeline"
W_FREEZE_SHA = "9d129bd7e01e80911ef3335c3d19d826a37481a2e31bca6f2469d7b2afb13cca"
MODULES = ("GraphStatePipelineFailure", "GraphStatePipelineControl")


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def exact(ref):
    path = Path(ref["path"])
    before = path.stat()
    observed = stable.file_ref(path)
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError("unstable selected binding: " + str(path))
    if observed != ref:
        raise ValueError("fixed binding differs; no adoption/repair: " + str(path))


def dependencies():
    """One retained inspection against preexisting expected identities, not repinning."""
    active = artifacts_dir() / ARTIFACT / "active/dependency-check-once.json"
    blocked = build_dir() / BUILD / "NATIVE_BLOCKER.json"
    if blocked.exists():
        raise RuntimeError("pipeline native context blocked; no retries: " + str(blocked))
    if active.exists():
        data = json.loads(active.read_text())
        if not data["exact"]:
            raise RuntimeError("pipeline selected dependency context affected/unconfirmed")
        return data
    freeze_path = artifacts_dir() / "verification/graph-loops-unroll/unroll-mode-v1/SOURCE_FREEZE.json"
    freeze_ref = stable.file_ref(freeze_path)
    if freeze_ref["sha256"] != W_FREEZE_SHA:
        raise ValueError("sealed W source freeze identity differs")
    freeze = json.loads(freeze_path.read_text())
    incident_path = artifacts_dir() / "verification/graph-state-scalar/review/sealed-v1/shared-binding-check-once.json"
    incident = json.loads(incident_path.read_text())
    if (
        incident["target_in_actual_proof_import_object_closure"]
        or not incident["all_recorded_object_tool_bindings_exact"]
    ):
        raise ValueError("existing scalar fixed context affected/unconfirmed")
    scalar_sources = [
        stable.file_ref(repo_root() / ("src/merlin/verify/lean/Merlin/" + name + ".lean"))
        for name in (
            "GraphStateScalarContract",
            "GraphStateScalarNormalize",
            "GraphStateScalarSound",
            "GraphStateScalarTyping",
            "GraphStateScalarInitialized",
            "GraphStateScalar",
            "GraphStateScalarSteps",
        )
    ]
    ready = []
    for path in sorted((build_dir() / "graph-verification/state-scalar/kernel").glob("*/ready.json")):
        data = json.loads(path.read_text())
        if data["sources"] == scalar_sources:
            ready.append((path, data))
    if len(ready) != 1:
        raise ValueError("unique unchanged sealed scalar object profile required")
    ready_path, scalar = ready[0]
    w = json.loads(Path(freeze["selection"]["path"]).read_text())
    closure = json.loads(Path(w["proof"]["build_record"]).read_text())
    rows = [
        freeze_ref,
        freeze["selection"],
        *freeze["proof_dependencies"],
        *freeze["lean_loaded_files"],
        stable.file_ref(ready_path),
        *scalar["files"],
        *scalar_sources,
        *incident["fixed_standard_object_bindings"],
        *incident["fixed_tool_bindings"],
    ]
    indexed, mismatches = {}, []
    for row in rows:
        if row["path"] in indexed and indexed[row["path"]] != row:
            mismatches.append({"path": row["path"], "reason": "conflicting preexisting pins"})
        indexed[row["path"]] = row
    target = incident["target"]
    target_present = target in indexed
    if target_present:
        mismatches.append({"path": target, "reason": "incident module in selected closure"})
    for row in indexed.values():
        try:
            exact(row)
        except (ValueError, OSError) as error:
            mismatches.append({"path": row["path"], "reason": str(error)})
    data = {
        "schema": "merlin.graph-state-pipeline.dependency-check.v1",
        "exact": not mismatches,
        "mismatches": mismatches,
        "target_in_selected_closure": target_present,
        "bindings": list(indexed.values()),
        "scalar_ready": str(ready_path),
        "scalar_library": str(ready_path.parent / "lib/lean"),
        "w_library": w["proof"]["library"],
        "foreign_objects": [attempt["object"] for attempt in closure["attempts"]],
        "sidecars_and_host_from_sealed_W_freeze": True,
        "reads_per_unique_selected_binding": 1,
        "toolchain_written_or_rebuilt": False,
        "identity_refreshed_to_promote": False,
        "qualification_issued": False,
    }
    write(active, data)
    if mismatches:
        raise RuntimeError("fixed dependency check failed; source/proof/binding work only: " + str(active))
    return data


def environment():
    deps = dependencies()
    for row in deps["bindings"]:
        exact(row)
    own = build_dir() / BUILD / "development/lib"
    (own / "Merlin").mkdir(parents=True, exist_ok=True)
    originals = [*deps["foreign_objects"]]
    originals += [row for row in deps["bindings"] if row["path"].startswith(deps["scalar_library"] + "/")]
    for row in originals:
        destination = own / "Merlin" / Path(row["path"]).name
        if not destination.exists():
            shutil.copyfile(row["path"], destination)
        if stable.file_ref(destination)["sha256"] != row["sha256"]:
            raise ValueError("byte-identical dependency copy changed")
    return {**os.environ, "LEAN_PATH": str(own), "LEAN_NUM_THREADS": "1"}


def invoke(source, *, module=None):
    """Retain each new source attempt separately; native faults freeze the context."""
    env = environment()
    source = Path(source).resolve()
    content = source.read_bytes()
    key = hashlib.sha256(content).hexdigest()
    directory = build_dir() / BUILD / "development/attempts" / source.stem / key
    directory.mkdir(parents=True, exist_ok=False)
    copy = directory / source.name
    copy.write_bytes(content)
    target = build_dir() / BUILD / "development/lib/Merlin" / (source.stem + ".olean")
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
    record = {
        "returncode": result.returncode,
        "source": stable.file_ref(copy),
        "stdout": stable.file_ref(directory / "stdout"),
        "stderr": stable.file_ref(directory / "stderr"),
        "object": stable.file_ref(target) if result.returncode == 0 else None,
        "phase": "development-only; not qualification",
        "qualified": False,
    }
    write(directory / "result.json", record)
    if result.returncode < 0:
        write(build_dir() / BUILD / "NATIVE_BLOCKER.json", {**record, "cause": "UNKNOWN", "retry_permitted": False})
    return record
