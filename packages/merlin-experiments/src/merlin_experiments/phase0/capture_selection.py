"""Pre-execution selection for one fresh, source-closed CPU M2M capture.

The selection is created before its run directory exists and is supplied again
by exact digest to issuance and derivation. A reproducible sandbox replay binds
these selected bytes to a capture. Selection and replay alone grant no Phase 0
admission; the separate, policy-restricted sealed-M2M attestation may qualify a
fresh verified replay. Neither establishes that the target compiler ran the model.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from merlin.common import strict_json
from merlin_experiments.capture_execution import sealed_m2m
from merlin_experiments.capture_execution.sealed_static import (
    _bwrap_binary,
    _canonical_path,
    _digest,
    _file_digest,
    _json,
)

SCHEMA = "merlin.phase0.capture_selection.v1"
SCHEMA_V2 = "merlin.phase0.capture_selection.v2"
MEMBER = "capture-selection.json"


def _sha(value: str) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _libraries(plan: dict) -> list[dict[str, Any]]:
    return [
        {"path": name, "bytes": Path(name).stat().st_size, "sha256": _file_digest(Path(name))}
        for name in plan["system_libs"]
    ]


def _selected_bytes(plan: dict, run_dir: Path, bwrap: Path) -> dict:
    output = run_dir / "capture"
    command = sealed_m2m._command_v2(
        output, dtype=plan["dtype"], recipe=plan.get("recipe") is not None, options=plan.get("worker_options")
    )
    full_inputs = plan.get("schema") == sealed_m2m.SCHEMA_V3
    return {
        "schema": SCHEMA_V2 if full_inputs else SCHEMA,
        "status": "preselected_before_capture",
        "run_dir": str(run_dir),
        "plan": plan,
        "plan_sha256": _digest(_json(plan)),
        "checkpoint": (
            next(row for row in plan["selected_inputs"] if row["role"] == "checkpoint")
            if full_inputs
            else {"kind": "none"}
        ),
        "system_libraries": _libraries(plan),
        "bwrap": {"path": str(bwrap), "sha256": _file_digest(bwrap)},
        "issuer_source_sha256": _file_digest(Path(sealed_m2m.__file__)),
        "sandbox_policy_sha256": sealed_m2m._policy(
            command,
            output,
            replayable_logs=True,
            loader_env=plan.get("loader_env") if full_inputs else None,
            timeout_seconds=plan.get("execution_timeout_seconds", sealed_m2m._TIMEOUT_SECONDS),
        ),
        "phase0_admission": "not_granted",
    }


def select(
    *,
    m2m_root: Path,
    frozen_origin: dict | None = None,
    workload_root: Path,
    worker: Path,
    venv: Path,
    schemas_root: Path,
    run_dir: Path,
    output_dir: Path,
    dtype: str = "fp32",
    recipe: Path | None = None,
    checkpoint: Path | None = None,
    checkpoint_guest_member: str | None = None,
    extra_inputs: dict[str, Path] | None = None,
    loader_env: dict[str, str | None] | None = None,
    execution_timeout_seconds: int | None = None,
    bwrap_binary: Path | None = None,
    worker_options: dict | None = None,
) -> dict:
    """Write one owner-only selection before any capture output exists.

    V1 selects checkpoint-free loaders; it may bind an operator-selected execution timeout
    (120..14400 s), which issue and the attestation replay then both use, and otherwise keeps
    the historical fixed 120 s and its historical bytes. V2 inventories one explicit checkpoint,
    any additional input files, and all declared loader environment reads before
    the sandbox sees them. Neither selection grants compilation admission.
    """
    if checkpoint is not None and checkpoint_guest_member is None:
        raise ValueError("selected checkpoint requires an exact guest member")
    if checkpoint is None and any(value is not None for value in (checkpoint_guest_member, extra_inputs, loader_env)):
        raise ValueError("declared loader environment and extra inputs require a selected checkpoint")
    run = _canonical_path(Path(run_dir), exists=False)
    destination = _canonical_path(Path(output_dir), exists=False)
    if run.exists() or destination.exists():
        raise ValueError("capture run and selection output must both be fresh")
    if run == destination or run.is_relative_to(destination) or destination.is_relative_to(run):
        raise ValueError("capture output and selection output may not overlap")
    if not run.parent.is_dir() or not destination.parent.is_dir():
        raise ValueError("capture and selection output parents must already exist")
    plan = sealed_m2m.prepare_plan(
        m2m_root=m2m_root,
        frozen_origin=frozen_origin,
        workload_root=workload_root,
        worker=worker,
        venv=venv,
        schemas_root=schemas_root,
        dtype=dtype,
        recipe=recipe,
        worker_options=worker_options,
        checkpoint=checkpoint,
        checkpoint_guest_member=checkpoint_guest_member,
        extra_inputs=extra_inputs,
        loader_env=loader_env,
        execution_timeout_seconds=execution_timeout_seconds,
    )
    selected_inputs = [
        Path(plan[name])
        for name in ("m2m_root", "workload_root", "worker", "merlin_root", "schemas_root", "venv", "base")
    ]
    if plan.get("recipe"):
        selected_inputs.append(Path(plan["recipe"]["path"]))
    if plan["schema"] == sealed_m2m.SCHEMA_V3:
        selected_inputs.extend(Path(row["source"]) for row in plan["selected_inputs"])
    if any(
        output == item or output.is_relative_to(item) or item.is_relative_to(output)
        for output in (run, destination)
        for item in selected_inputs
    ):
        raise ValueError("capture run or selection output overlaps a selected input")
    bwrap = _bwrap_binary(bwrap_binary)
    selected = _selected_bytes(plan, run, bwrap)
    raw = _json(selected) + b"\n"
    destination.mkdir(mode=0o700, parents=False, exist_ok=False)
    path = destination / MEMBER
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o400)
    destination.chmod(0o700)
    return {"path": str(path), "sha256": _digest(raw), "schema": selected["schema"], "phase0_admission": "not_granted"}


def load(path: Path, *, expected_sha256: str) -> dict:
    """Open only the independently chosen selection bytes, never receipt.plan."""
    if not _sha(expected_sha256):
        raise ValueError("capture selection requires an independently supplied SHA-256")
    selected_path = _canonical_path(Path(path), exists=True)
    if selected_path.name != MEMBER or not selected_path.is_file() or selected_path.is_symlink():
        raise ValueError("capture selection must name an ordinary capture-selection.json")
    if (
        selected_path.stat().st_uid != os.getuid()
        or selected_path.parent.stat().st_uid != os.getuid()
        or selected_path.stat().st_mode & 0o077
        or selected_path.parent.stat().st_mode & 0o077
    ):
        raise ValueError("capture selection must remain owner-only")
    raw = selected_path.read_bytes()
    if _digest(raw) != expected_sha256:
        raise ValueError("capture selection bytes differ from the pre-execution identity")
    try:
        selected = strict_json.loads(raw)
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError("capture selection is unreadable") from exc
    if (
        not isinstance(selected, dict)
        or selected.get("schema") not in {SCHEMA, SCHEMA_V2}
        or selected.get("status") != "preselected_before_capture"
        or selected.get("phase0_admission") != "not_granted"
        or not isinstance(selected.get("plan"), dict)
        or selected.get("plan_sha256") != _digest(_json(selected.get("plan")))
    ):
        raise ValueError("capture selection has an unsupported or inconsistent policy")
    if selected["schema"] == SCHEMA:
        if selected.get("checkpoint") != {"kind": "none"} or selected["plan"].get("schema") != sealed_m2m.SCHEMA:
            raise ValueError("v1 capture selection has an unsupported checkpoint or issuer policy")
    elif selected["plan"].get("schema") != sealed_m2m.SCHEMA_V3 or selected.get("checkpoint") != next(
        (row for row in selected["plan"].get("selected_inputs", []) if row.get("role") == "checkpoint"), None
    ):
        raise ValueError("v2 capture selection lacks its exact selected checkpoint")
    return selected


def issue(path: Path, *, expected_sha256: str) -> Path:
    """Execute only a fresh run selected by the owner-controlled artifact."""
    selected = load(path, expected_sha256=expected_sha256)
    run = _canonical_path(Path(selected["run_dir"]), exists=False)
    if run.exists():
        raise ValueError("capture run already exists; a selection cannot adopt historical output")
    plan = selected["plan"]
    current = sealed_m2m.prepare_plan(
        m2m_root=Path(plan["m2m_root"]),
        frozen_origin=plan.get("frozen_origin"),
        workload_root=Path(plan["workload_root"]),
        worker=Path(plan["worker"]),
        venv=Path(plan["venv"]),
        schemas_root=Path(plan["schemas_root"]),
        dtype=plan["dtype"],
        recipe=Path(plan["recipe"]["path"]) if plan.get("recipe") else None,
        max_snapshot_bytes=plan["max_snapshot_bytes"],
        worker_options=plan.get("worker_options"),
        **(
            {
                "checkpoint": Path(selected["checkpoint"]["source"]),
                "checkpoint_guest_member": selected["checkpoint"]["guest_member"],
                "extra_inputs": {
                    row["guest_member"]: Path(row["source"])
                    for row in plan["selected_inputs"]
                    if row["role"] == "input"
                },
                "loader_env": plan["loader_env"],
                "execution_timeout_seconds": plan["execution_timeout_seconds"],
            }
            if selected["schema"] == SCHEMA_V2
            else (
                {"execution_timeout_seconds": plan["execution_timeout_seconds"]}
                if "execution_timeout_seconds" in plan
                else {}
            )
        ),
    )
    bwrap = _bwrap_binary(Path(selected["bwrap"]["path"]))
    if current != plan or _selected_bytes(current, run, bwrap) != selected:
        raise ValueError("selected capture source, runtime, checkpoint absence, tool or sandbox policy changed")
    if _digest(Path(path).read_bytes()) != expected_sha256:
        raise ValueError("capture selection changed before execution")
    receipt = sealed_m2m.issue(
        plan,
        run,
        bwrap_binary=bwrap,
        capture_selection_sha256=expected_sha256,
        selected_system_libraries=selected["system_libraries"],
        selected_bwrap_sha256=selected["bwrap"]["sha256"],
    )
    if _digest(Path(path).read_bytes()) != expected_sha256:
        raise ValueError("capture selection changed during execution")
    return receipt


def verify(path: Path, *, expected_sha256: str, model_path: Path) -> dict:
    """Read-only independent binding of selected bytes to a fresh replay.

    The returned record intentionally remains nonadmissible. The separately
    issued sealed-M2M attestation checks these bytes again before granting
    source-closure admission under its explicit operator policy.
    """
    selected = load(path, expected_sha256=expected_sha256)
    run = _canonical_path(Path(selected["run_dir"]), exists=True)
    if run.stat().st_uid != os.getuid() or run.stat().st_mode & 0o077:
        raise ValueError("selected sealed capture run must remain owner-only")
    model = _canonical_path(Path(model_path), exists=True)
    full_capture = selected["schema"] == SCHEMA_V2
    if full_capture and (model != run / "capture" or not model.is_dir()):
        raise ValueError("selected v3 capture must name its complete output root")
    if not full_capture and (model != run / "capture/model.mlir" or not model.is_file()):
        raise ValueError("selected capture model is not this fresh run's exact output")
    pending = run / "sealed_m2m_pending.json"
    if pending.is_symlink() or not pending.is_file():
        raise ValueError("selected sealed M2M receipt is absent or indirect")
    receipt = strict_json.loads(pending.read_bytes())
    if (
        receipt.get("schema") != (sealed_m2m.SCHEMA_V3 if selected["schema"] == SCHEMA_V2 else sealed_m2m.SCHEMA)
        or receipt.get("capture_selection_sha256") != expected_sha256
        or receipt.get("plan") != selected["plan"]
        or receipt.get("policy_sha256") != selected["sandbox_policy_sha256"]
        or receipt.get("bwrap_sha256") != selected["bwrap"]["sha256"]
        or receipt.get("issuer_sha256") != selected["issuer_source_sha256"]
    ):
        raise ValueError("sealed capture does not bind the independently selected plan and policy")
    guest_root = run / "snapshots/guest-root"
    if [library["path"] for library in selected["system_libraries"]] != selected["plan"]["system_libs"]:
        raise ValueError("selected system libraries differ from the planned runtime")
    for library in selected["system_libraries"]:
        source = Path(library["path"])
        if not source.is_absolute() or ".." in source.parts:
            raise ValueError("selected system library path is unsafe")
        copied = guest_root / source.relative_to("/")
        if (
            copied.is_symlink()
            or not copied.is_file()
            or copied.stat().st_size != library["bytes"]
            or _file_digest(copied) != library["sha256"]
        ):
            raise ValueError("sealed runtime differs from preselected system library bytes")
    replay = sealed_m2m.replay_verify(run, bwrap_binary=Path(selected["bwrap"]["path"]))
    if (
        replay.get("status") != "verified_sandbox_replay"
        or replay.get("sealed_source_closure_replayed") is not True
        or replay.get("receipt_sha256") != _file_digest(pending)
    ):
        raise ValueError("fresh sandbox replay did not verify the selected capture")
    materialized = run / "capture/capture_receipt.json"
    result = {
        "schema": "merlin.phase0.preselected_capture_replay.v1",
        "status": "verified_preselected_replay",
        "selection_sha256": expected_sha256,
        "sealed_receipt_sha256": replay["receipt_sha256"],
        "source_closure_verified": False,
        "phase0_admission": "not_granted",
    }
    if full_capture:
        evidence = receipt.get("materialized") or {}
        result["capture_kind"] = evidence.get("kind")
        result["capture_tree_sha256"] = sealed_m2m._snapshot_tree(model)["sha256"]
        if evidence.get("kind") == "session" and isinstance(evidence.get("programs"), list):
            result.update(
                session_contract_sha256=evidence["session_contract_sha256"],
                session_receipt_sha256=evidence["session_receipt_sha256"],
                programs=evidence["programs"],
                integer_contractions=evidence.get("integer_contractions", 0),
            )
        elif evidence.get("kind") == "single":
            result.update(
                model_sha256=_file_digest(model / "model.mlir"),
                capture_receipt_sha256=_file_digest(model / "capture_receipt.json"),
                integer_contractions=evidence.get("integer_contractions", 0),
            )
        else:
            raise ValueError("selected v3 capture lacks a complete saved program or session")
    else:
        result.update(model_sha256=_file_digest(model), capture_receipt_sha256=_file_digest(materialized))
    if _digest(Path(path).read_bytes()) != expected_sha256:
        raise ValueError("capture selection changed during replay")
    return result


def attest(path: Path, *, expected_sha256: str, output: Path) -> dict:
    """Replay one issued selected capture and write its sealed execution attestation.

    A v1 selection (checkpoint-free loader) yields the ``merlin.sealed_m2m_cpu`` attestation of its
    ``model.mlir``; a v2 selection (explicit checkpoint, extra inputs and loader environment) yields
    the v3 attestation of the complete capture root, single program or saved session. Either is
    issued only from a fresh :func:`verify` replay of these exact selected bytes, and is written to a
    fresh owner-only file outside the selected run, so the attestation never alters what it attests.
    """
    from .capture_execution_attestation import attest_sealed_m2m, attest_sealed_m2m_v3

    selected = load(path, expected_sha256=expected_sha256)
    run = _canonical_path(Path(selected["run_dir"]), exists=True)
    destination = _canonical_path(Path(output), exists=False)
    if destination.exists() or destination.is_symlink():
        raise ValueError("capture attestation output must be fresh")
    if not destination.parent.is_dir():
        raise ValueError("capture attestation output parent must already exist")
    selection_path = _canonical_path(Path(path), exists=True)
    if destination.is_relative_to(run / "capture") or destination.is_relative_to(run / "snapshots"):
        raise ValueError("capture attestation output may not enter the attested capture or its snapshots")
    if destination.parent == selection_path.parent:
        raise ValueError("capture attestation output may not enter the owner-only selection directory")
    if selected["schema"] == SCHEMA_V2:
        capture = run / "capture"
        replay = verify(path, expected_sha256=expected_sha256, model_path=capture)
        document = attest_sealed_m2m_v3(replay, selection_path=selection_path, capture_path=capture)
    else:
        model = run / "capture/model.mlir"
        replay = verify(path, expected_sha256=expected_sha256, model_path=model)
        document = attest_sealed_m2m(replay, selection_path=selection_path, model_path=model)
    raw = _json(document) + b"\n"
    with destination.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    destination.chmod(0o400)
    if _digest(Path(path).read_bytes()) != expected_sha256:
        raise ValueError("capture selection changed during attestation")
    return {
        "capture_execution_attestation": str(destination),
        "capture_execution_attestation_sha256": _digest(raw),
        "issuer": document["issuer"],
        "selection_sha256": expected_sha256,
        "capture": document["capture"],
    }
