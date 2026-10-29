"""Private pure-kernel profile for additive chain/receipt roots; stable v2 is read-only."""

from __future__ import annotations

import hashlib
import json
import os
from functools import cache
from pathlib import Path

from merlin.common.owned_process import run_owned
from merlin.common.paths import build_dir, repo_root

from . import lean_graph_state as stable

MODULES = ("GraphStateChain", "GraphStateReceiptBytes", "GraphStateReceipt")


@cache
def prepare():
    base = stable.prepare()
    originals = [stable.file_ref(repo_root() / f"src/merlin/verify/lean/Merlin/{name}.lean") for name in MODULES]
    key = hashlib.sha256(
        json.dumps({"base": base["manifest"], "new_sources": originals}, sort_keys=True).encode()
    ).hexdigest()
    directory = build_dir() / "graph-verification/state-chain/kernel" / key
    lib = directory / "lib/lean"
    marker = directory / "ready.json"
    if not marker.exists():
        (lib / "Merlin").mkdir(parents=True, exist_ok=True)
        # Lean resolves a namespace under one import root. Retain the exact
        # existing core objects in this private profile; never rebuild/edit v2.
        copied = []
        for path in sorted((base["lib"] / "Merlin").glob("*.olean")):
            identity = stable.file_ref(path)
            target = lib / "Merlin" / path.name
            target.write_bytes(path.read_bytes())
            if stable.file_ref(target)["sha256"] != identity["sha256"]:
                raise ValueError("stable proof object changed while retaining private import closure")
            copied.append(identity)
        for name, record in zip(MODULES, originals, strict=True):
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
                raise RuntimeError(
                    f"private chain proof build unavailable: {name}; retained first buffers at {directory}"
                )
        identities = [stable.file_ref(path) for path in sorted(directory.rglob("*")) if path.is_file()]
        stable._write(
            marker,
            {
                "new_sources": originals,
                "base_manifest": base["manifest"],
                "base_objects": copied,
                "files": identities,
                "selected_lean": base["lean"],
                "identity": key,
            },
        )
    record = json.loads(marker.read_text())
    if (
        record["new_sources"] != originals
        or record["base_manifest"] != base["manifest"]
        or not stable.fresh(
            [
                *record["new_sources"],
                *record["base_objects"],
                *record["files"],
                record["base_manifest"],
                record["selected_lean"],
            ]
        )
    ):
        raise ValueError("private source/proof/tool import identity changed")
    return {**record, "directory": directory, "lib": lib, "manifest": stable.file_ref(marker)}


def check_query(source_text, destination, *, module, roots):
    """Check the real generated module and audit its actual imported closure."""
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    frozen = prepare()
    source = destination / f"{module}.lean"
    source.write_text(source_text)
    with (destination / "kernel.stdout").open("x") as out, (destination / "kernel.stderr").open("x") as err:
        checked = run_owned(
            [str(stable.LEAN), str(source), "-o", str(source.with_suffix(".olean"))],
            cwd=destination,
            env={**os.environ, "LEAN_PATH": str(frozen["lib"])},
            stdout=out,
            stderr=err,
            timeout=60,
            check=False,
        )
    audit = stable.audit_claim(destination, frozen, module, roots) if checked.returncode == 0 else {"status": "not_run"}
    if not stable.fresh(
        [
            *frozen["new_sources"],
            *frozen["base_objects"],
            *frozen["files"],
            frozen["base_manifest"],
            frozen["selected_lean"],
            frozen["manifest"],
            *audit.get("imported_objects", []),
        ]
    ):
        raise ValueError("private chain profile or selected imported proof changed during check")
    result = {
        "schema": "merlin.graph-state-chain.kernel.v1",
        "query": stable.file_ref(source),
        "status": "kernel_model_checked" if checked.returncode == 0 and audit["status"] == "checked" else "unavailable",
        "profile": frozen["manifest"],
        "audit": audit,
        "public_acceptance_issued": False,
        "full_LeanMLIR_route": "unavailable: no matching selected manifest",
    }
    stable._write(destination / "report.json", result)
    return result
