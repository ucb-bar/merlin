"""Private pinned pure-kernel elision profile; all previous source/profiles read-only."""

from __future__ import annotations

import hashlib
import json
import os
from functools import cache
from pathlib import Path

from merlin.common.owned_process import run_owned
from merlin.common.paths import build_dir, repo_root

from . import lean_graph_state as stable

MODULES = ("GraphStateElisionContract", "GraphStateElision")


@cache
def prepare():
    base = stable.prepare()
    sources = [stable.file_ref(repo_root() / f"src/merlin/verify/lean/Merlin/{name}.lean") for name in MODULES]
    key = hashlib.sha256(
        json.dumps({"base": base["manifest"], "sources": sources}, sort_keys=True).encode()
    ).hexdigest()
    directory = build_dir() / "graph-verification/state-elision/kernel" / key
    lib = directory / "lib/lean"
    marker = directory / "ready.json"
    if not marker.exists():
        (lib / "Merlin").mkdir(parents=True, exist_ok=True)
        objects = []
        for path in sorted((base["lib"] / "Merlin").glob("*.olean")):
            original = stable.file_ref(path)
            target = lib / "Merlin" / path.name
            target.write_bytes(path.read_bytes())
            if stable.file_ref(target)["sha256"] != original["sha256"]:
                raise ValueError("base proof object changed during exact private retention")
            objects.append(original)
        for name, record in zip(MODULES, sources, strict=True):
            source = directory / f"sources/Merlin/{name}.lean"
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_bytes(Path(record["path"]).read_bytes())
            with (directory / f"{name}.stdout").open("x") as out, (directory / f"{name}.stderr").open("x") as err:
                result = run_owned(
                    [str(stable.LEAN), str(source), "-o", str(lib / f"Merlin/{name}.olean")],
                    cwd=directory / "sources",
                    env={**os.environ, "LEAN_PATH": str(lib)},
                    stdout=out,
                    stderr=err,
                    timeout=60,
                    check=False,
                )
            if result.returncode:
                raise RuntimeError(f"elision pure-kernel profile unavailable: {name}; first buffers at {directory}")
        files = [stable.file_ref(path) for path in sorted(directory.rglob("*")) if path.is_file()]
        stable._write(
            marker,
            {
                "schema": "merlin.graph-state-elision.kernel-profile.v1",
                "identity": key,
                "sources": sources,
                "base_objects": objects,
                "base_manifest": base["manifest"],
                "lean": base["lean"],
                "files": files,
            },
        )
    data = json.loads(marker.read_text())
    if (
        data["sources"] != sources
        or data["base_manifest"] != base["manifest"]
        or not stable.fresh([*sources, *data["base_objects"], *data["files"], data["lean"], data["base_manifest"]])
    ):
        raise ValueError("elision profile source/object/tool identity changed")
    return {**data, "directory": directory, "lib": lib, "manifest": stable.file_ref(marker)}


def check_query(text, destination, *, module="GraphStateElisionInstance", roots=None):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    frozen = prepare()
    source = destination / f"{module}.lean"
    source.write_text(text)
    with (destination / "kernel.stdout").open("x") as out, (destination / "kernel.stderr").open("x") as err:
        compiled = run_owned(
            [str(stable.LEAN), str(source), "-o", str(source.with_suffix(".olean"))],
            cwd=destination,
            env={**os.environ, "LEAN_PATH": str(frozen["lib"])},
            stdout=out,
            stderr=err,
            timeout=60,
        )
    names = roots or [
        "Merlin.GraphState.Elision.check_sound",
        "Merlin.GraphState.Elision.context_preserved",
        "Merlin.GraphStateElisionInstance.accepted",
        "Merlin.GraphStateElisionInstance.actual_elision",
    ]
    audit = (
        stable.audit_claim(destination, frozen, module, names) if compiled.returncode == 0 else {"status": "not_run"}
    )
    if not stable.fresh(
        [
            *frozen["sources"],
            *frozen["base_objects"],
            *frozen["files"],
            frozen["lean"],
            frozen["base_manifest"],
            frozen["manifest"],
            *audit.get("imported_objects", []),
        ]
    ):
        raise ValueError("actual selected source/proof/import/tool changed during elision check")
    result = {
        "schema": "merlin.graph-state-elision.kernel-check.v1",
        "query": stable.file_ref(source),
        "status": "conditional_model_checked"
        if compiled.returncode == 0 and audit["status"] == "checked"
        else "unavailable",
        "returncode": compiled.returncode,
        "profile": frozen["manifest"],
        "audit": audit,
        "public_acceptance_issued": False,
        "admission_approved": False,
        "native_fidelity_proved": False,
        "source_success_proved": False,
        "full_LeanMLIR_route": "unavailable: no matching selected manifest",
    }
    stable._write(destination / "report.json", result)
    return result
