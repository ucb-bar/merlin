"""Actual shared-pass schedule binding plus kernel-checked exchange composition.

This leaf extends StateEffects.preserveExchanges and the existing effect-schedule
projection boundary. It issues no existing broker acceptance or deserialized proof
authority. Totality/frame/locality and IR projection trust stay explicit.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from functools import cache
from pathlib import Path

from merlin.common.owned_process import run_owned
from merlin.common.paths import build_dir, repo_root

REFERENCE_REVISION = "4499eebb959c0f92cefd92cc6d80646ba66474ed"
LEAN = Path("/home/jeremy/.elan/toolchains/leanprover--lean4---v4.35.0/bin/lean")
LEAN_SHA256 = "bf8d54e4714cc4b03d3f6bb34c83b7202b87e49c8bfcbff6895c085bb90ceb38"
SOURCES = ("StateEffects", "GraphStateContract", "GraphState", "GraphStateConfiguration", "GraphStateModels")


def host_runtime():
    """Identify trusted execution support separately from kernel proof objects."""
    import merlin.common.owned_process
    import merlin.common.paths

    return {
        "python": file_ref(sys.executable),
        "python_version": sys.version,
        "support_sources": [file_ref(module.__file__) for module in (merlin.common.owned_process, merlin.common.paths)],
    }


def file_ref(path):
    path = Path(path).resolve()
    before = path.stat()
    data = path.read_bytes()
    after = path.stat()
    if (before.st_mtime_ns, before.st_size) != (after.st_mtime_ns, after.st_size) or len(data) != after.st_size:
        raise ValueError(f"unstable input bytes: {path}")
    return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def _write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def fresh(records):
    return all(file_ref(row["path"]) == row for row in records)


@cache
def prepare():
    root = repo_root()
    if file_ref(LEAN)["sha256"] != LEAN_SHA256:
        raise ValueError("selected standalone Lean bytes differ from coordinator intake")
    source = root / "src/merlin/verify/lean/Merlin"
    refs = [file_ref(source / f"{name}.lean") for name in SOURCES]
    identity = hashlib.sha256(
        json.dumps({"sources": refs, "lean": file_ref(LEAN)}, sort_keys=True).encode()
    ).hexdigest()
    directory = build_dir() / "graph-verification/state/kernel" / identity
    lib = directory / "lib/lean"
    marker = directory / "ready.json"
    if not marker.exists():
        (lib / "Merlin").mkdir(parents=True, exist_ok=True)
        (directory / "sources/Merlin").mkdir(parents=True, exist_ok=True)
        for name, ref in zip(SOURCES, refs, strict=True):
            target = directory / f"sources/Merlin/{name}.lean"
            target.write_bytes(Path(ref["path"]).read_bytes())
            command = [str(LEAN), str(target), "-o", str(lib / f"Merlin/{name}.olean")]
            with (directory / f"{name}.stdout").open("x") as out, (directory / f"{name}.stderr").open("x") as err:
                result = run_owned(
                    command,
                    cwd=directory / "sources",
                    env={**os.environ, "LEAN_PATH": str(lib)},
                    stdout=out,
                    stderr=err,
                    timeout=60,
                    check=False,
                )
            if result.returncode:
                raise RuntimeError(f"standalone shared proof build unavailable: {name}; buffers at {directory}")
        sealed = [file_ref(path) for path in sorted(directory.rglob("*")) if path.is_file()]
        _write(marker, {"sources": refs, "files": sealed, "lean": file_ref(LEAN), "identity": identity})
    data = json.loads(marker.read_text())
    if data["sources"] != refs or not fresh([*data["sources"], *data["files"], data["lean"]]):
        raise ValueError("proof source/object/tool identity changed")
    return {**data, "directory": directory, "lib": lib, "manifest": file_ref(marker)}


def _naturals(value):
    if type(value) is not list or any(type(item) is not int or item < 0 for item in value):
        raise ValueError("effect certificate needs exact nonnegative integer lists")
    return value


def emit_query(projection):
    nodes = projection["nodes"]
    if type(nodes) is not list or not nodes or len(nodes) > 8192:
        raise ValueError("complete nonempty bounded actual operation roster required")
    if _naturals([node["operation_index"] for node in nodes]) != list(range(len(nodes))):
        raise ValueError("actual operation roster is incomplete or reordered")
    effects = []
    cells = projection["cells"]
    if type(cells) is not list or any(type(cell) is not str for cell in cells) or len(set(cells)) != len(cells):
        raise ValueError("complete unique actual state-cell roster required")
    for node in nodes:
        reads, writes = _naturals(node["reads"]), _naturals(node["writes"])
        if max([*reads, *writes], default=-1) >= len(cells):
            raise ValueError("footprint refers to an absent actual state cell")
        if len(set(reads)) != len(reads) or len(set(writes)) != len(writes):
            raise ValueError("duplicate footprint cell")
        effects.append("{reads := " + str(reads) + ",writes := " + str(writes) + "}")
    target = _naturals(projection["order"])
    if sorted(target) != list(range(len(nodes))):
        raise ValueError("target drops or duplicates an actual operation")
    exchanges = projection["exchanges"]
    if type(exchanges) is not list or len(exchanges) > 65536:
        raise ValueError("exchange budget exceeded")
    moves = []
    for move in exchanges:
        values = [move["position"], move["first"], move["second"]]
        _naturals(values)
        moves.append("⟨" + ",".join(map(str, values)) + "⟩")
    return (
        "import Merlin.GraphState\nnamespace Merlin.GraphStateInstance\n"
        "def footprints : List Merlin.StateEffects.Footprint := [" + ",".join(effects) + "]\n"
        "def target : List Nat := " + str(target) + "\n"
        "def exchanges : List Merlin.GraphState.Exchange := [" + ",".join(moves) + "]\n"
        "theorem accepted : Merlin.GraphState.check footprints target exchanges = true := by decide\n"
        "theorem actual_schedule : Merlin.GraphState.Preserves footprints target :=\n"
        "  Merlin.GraphState.check_sound accepted\n"
        "#print axioms accepted\n#print axioms actual_schedule\nend Merlin.GraphStateInstance\n"
    )


def audit_claim(destination, frozen, module="GraphStateInstance", statements=None):
    """Inspect the actual imported proof closure; retain raw first audit buffers."""
    destination = Path(destination)
    auditor = repo_root() / "src/merlin/verify/lean/Merlin/GraphStateAudit.lean"
    environment = {
        **os.environ,
        "PATH": str(LEAN.parent) + os.pathsep + os.environ.get("PATH", ""),
        "LEAN_PATH": os.pathsep.join((str(destination), str(frozen["lib"]))),
    }
    names = statements or [
        "Merlin.StateEffects.preserveExchanges",
        "Merlin.GraphState.check_sound",
        "Merlin.GraphStateInstance.accepted",
        "Merlin.GraphStateInstance.actual_schedule",
    ]
    results = {}
    for label, arguments in (("audit", [module]), ("statements", ["--statements", ",".join(names), module])):
        with (destination / f"{label}.stdout").open("x") as out, (destination / f"{label}.stderr").open("x") as err:
            checked = run_owned(
                [str(LEAN), "--run", str(auditor), *arguments],
                cwd=destination,
                env=environment,
                stdout=out,
                stderr=err,
                timeout=60,
                check=False,
            )
        if checked.returncode:
            return {"status": "unavailable", "phase": label, "auditor": file_ref(auditor)}
        results[label] = json.loads((destination / f"{label}.stdout").read_text())
    imported = [file_ref(row["olean"]) for row in results["audit"]["imports"]]
    if results["audit"]["violations"]:
        raise ValueError("selected graph proof closure violates std3-only policy")
    return {
        "status": "checked",
        "auditor": file_ref(auditor),
        "raw": results["audit"],
        "statements": results["statements"]["statements"],
        "imported_objects": imported,
        "buffers": [
            file_ref(destination / f"{label}.{suffix}") for label in results for suffix in ("stdout", "stderr")
        ],
    }


def retain_sources(destination, records):
    """Retain the exact measured code, with live-source identities kept separate."""
    saved = []
    for index, record in enumerate(records):
        path = Path(record["path"])
        retained = Path(destination) / f"source-closure/{index:03d}-{path.name}"
        retained.parent.mkdir(parents=True, exist_ok=True)
        retained.write_bytes(path.read_bytes())
        identity = file_ref(retained)
        if any(identity[key] != record[key] for key in ("sha256", "bytes")):
            raise ValueError("source changed while retaining the measured code closure")
        saved.append({"original": record, "retained": identity})
    return saved


def check_boundary(
    before, after, destination, *, reference_root, contract, original_contract_sha256=None, expected_before_sha256=None
):
    """Reopen the actual IR and rederive its projection, then check ONE composed theorem.

    Original contract identity, when supplied, comes from the original host. This
    development leaf cannot itself admit an effect description or issue broker custody.
    """
    before, after, contract = map(Path, (before, after, contract))
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    inputs = [file_ref(path) for path in (before, after, contract)]
    owned = [
        file_ref(path)
        for path in (
            __file__,
            repo_root() / "src/merlin/verify/lean_graph_state_capture.py",
            repo_root() / "src/merlin/verify/lean/Merlin/GraphStateAudit.lean",
        )
    ]
    if expected_before_sha256 is not None and inputs[0]["sha256"] != expected_before_sha256:
        raise ValueError("actual source differs from original selected source")
    retained = []
    for original, identity, name in zip(
        (before, after, contract), inputs, ("before.mlir", "after.mlir", "contract.json"), strict=True
    ):
        copied = destination / name
        copied.write_bytes(original.read_bytes())
        copied_identity = file_ref(copied)
        if any(copied_identity[key] != identity[key] for key in ("sha256", "bytes")):
            raise ValueError("source bytes changed while retaining exact projection inputs")
        retained.append(copied_identity)
    generator = repo_root() / "src/merlin/verify/lean_graph_state_capture.py"
    environment = {
        **os.environ,
        "PYTHONPATH": str(Path(reference_root).resolve() / "src"),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    command = [
        sys.executable,
        str(generator),
        "--reference",
        str(Path(reference_root).resolve()),
        "--source",
        retained[0]["path"],
        "--target",
        retained[1]["path"],
        "--contract",
        retained[2]["path"],
        "--destination",
        str(destination / "projection"),
    ]
    with (destination / "projection.stdout").open("x") as out, (destination / "projection.stderr").open("x") as err:
        produced = run_owned(
            command, cwd=reference_root, env=environment, stdout=out, stderr=err, timeout=60, check=False
        )
    base = {
        "schema": "merlin.graph-state.composed-boundary.v1",
        "public_acceptance_issued": False,
        "source_contract_approved": False,
        "native_fidelity_proved": False,
        "pipeline_broker_integration": "pending exact shared interface and matching selected manifest",
        "inputs": inputs,
        "retained_inputs": retained,
        "owned_generator": file_ref(generator),
        "host_runtime": host_runtime(),
    }
    if produced.returncode:
        result = {**base, "status": "unavailable", "reason": "pinned producer/projection subprocess failed"}
        _write(destination / "report.json", result)
        return result
    capture = json.loads((destination / "projection/capture.json").read_text())
    if len(capture["inputs"]) != 2 or any(
        current[key] != original[key]
        for current, original in zip(capture["inputs"], retained[:2], strict=True)
        for key in ("sha256", "bytes")
    ):
        raise ValueError("actual projection is not bound to retained original input buffers")
    if original_contract_sha256 is not None and capture["contract_sha256"] != original_contract_sha256:
        raise ValueError("effect contract differs from original selected semantic identity")
    base.update(
        {
            "projection": file_ref(destination / "projection/capture.json"),
            "producer_revision": capture["producer_revision"],
            "contract_sha256": capture["contract_sha256"],
            "projection_trust": capture["projection_trust"],
            "producer_external_runtime": capture.get("external_runtime"),
        }
    )
    if not capture["projection_checked"]:
        result = {**base, "status": "unsupported", "reason": capture["reason"]}
        _write(destination / "report.json", result)
        return result
    frozen = prepare()
    query = destination / "GraphStateInstance.lean"
    query.write_text(emit_query(capture["projection"]))
    lean_command = [str(LEAN), str(query), "-o", str(destination / "GraphStateInstance.olean")]
    with (destination / "kernel.stdout").open("x") as out, (destination / "kernel.stderr").open("x") as err:
        checked = run_owned(
            lean_command,
            cwd=destination,
            env={**os.environ, "LEAN_PATH": str(frozen["lib"])},
            stdout=out,
            stderr=err,
            timeout=60,
            check=False,
        )
    audit = audit_claim(destination, frozen) if checked.returncode == 0 else {"status": "not_run"}
    closure = retain_sources(destination, [*owned, *capture["producer_sources"], *frozen["sources"]])
    if not fresh(
        [
            *owned,
            *inputs,
            *retained,
            *capture["producer_sources"],
            capture["owned_generator"],
            *frozen["sources"],
            *frozen["files"],
            frozen["lean"],
            frozen["manifest"],
            *audit.get("imported_objects", []),
            *base["host_runtime"]["support_sources"],
            base["host_runtime"]["python"],
            *capture["external_runtime"]["loaded_sources"],
            capture["external_runtime"]["python"],
        ]
    ):
        raise ValueError("IR, semantic source, proof object or tool changed during check")
    result = {
        **base,
        "status": "graph_model_checked" if checked.returncode == 0 and audit["status"] == "checked" else "unavailable",
        "composed_kernel_theorem_checked": checked.returncode == 0,
        "theorem": "Merlin.GraphStateInstance.actual_schedule",
        "existing_root": "Merlin.StateEffects.preserveExchanges",
        "graph_checker_root": "Merlin.GraphState.check_sound",
        "query": file_ref(query),
        "proof_profile": frozen["manifest"],
        "actual_operations": len(capture["projection"]["nodes"]),
        "actual_exchanges": len(capture["projection"]["exchanges"]),
        "source_projection_assumptions": capture["projection"]["assumptions"],
        "effect_semantics_sha256": capture["effect_semantics_sha256"],
        "numerical_gates_discharged": False,
        "transitive_audit": audit,
        "source_closure": closure,
        "full_LeanMLIR_route": "unavailable: no matching selected manifest",
    }
    _write(destination / "report.json", result)
    return result
