"""Pre-execution selection for one fresh, source-closed CPU M2M capture.

The selection is created before its run directory exists and is supplied again
by exact digest to issuance and derivation. A reproducible sandbox replay binds
these selected bytes to a capture. Selection and replay alone grant no Phase 0
admission; the separate, policy-restricted sealed-M2M attestation may qualify a
fresh verified replay. Neither establishes that the target compiler ran the model.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from merlin.common import record_formats as strict_json
from merlin.experiments.capture_execution import m2m_plan, m2m_sandbox, sealed_m2m
from merlin.experiments.capture_execution import sealed_static as native_integer
from merlin.experiments.capture_execution.sealed_static import (
    _bwrap_binary,
    _canonical_path,
    _digest,
    _file_digest,
    _json,
)

# ---- capture_selection -------------------------------------------------------------------------

SCHEMA = "merlin.phase0.capture_selection.v1"
SCHEMA_V2 = "merlin.phase0.capture_selection.v2"
MEMBER = "capture-selection.json"


def _native_batch(plan):
    return (
        plan.get("schema") == m2m_sandbox.SCHEMA_V3
        and native_integer.selected(plan.get("worker_options"))
        and plan.get("selected_inputs") == []
        and plan.get("loader_env") == {}
        and plan.get("loader_env_reads") == []
    )


def _sha(value: str) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _libraries(plan: dict) -> list[dict[str, Any]]:
    return [
        {"path": name, "bytes": Path(name).stat().st_size, "sha256": _file_digest(Path(name))}
        for name in plan["system_libs"]
    ]


def _selected_bytes(plan: dict, run_dir: Path, bwrap: Path) -> dict:
    output = run_dir / "capture"
    command = m2m_sandbox._command_v2(
        output,
        dtype=plan["dtype"],
        recipe=plan.get("recipe") is not None,
        options=plan.get("worker_options"),
        worker_member=m2m_sandbox._worker_member(plan),
    )
    full_inputs = plan.get("schema") == m2m_sandbox.SCHEMA_V3
    return {
        "schema": SCHEMA_V2 if full_inputs else SCHEMA,
        "status": "preselected_before_capture",
        "run_dir": str(run_dir),
        "plan": plan,
        "plan_sha256": _digest(_json(plan)),
        "checkpoint": (
            next(row for row in plan["selected_inputs"] if row["role"] == "checkpoint")
            if full_inputs and not _native_batch(plan)
            else {"kind": "none"}
        ),
        "system_libraries": _libraries(plan),
        "bwrap": {"path": str(bwrap), "sha256": _file_digest(bwrap)},
        "issuer_source_sha256": _file_digest(Path(sealed_m2m.__file__)),
        "sandbox_policy_sha256": m2m_sandbox._policy(
            command,
            output,
            replayable_logs=True,
            loader_env=plan.get("loader_env") if full_inputs else None,
            timeout_seconds=plan.get("execution_timeout_seconds", m2m_sandbox._TIMEOUT_SECONDS),
            **({"runtime_reuse": plan["runtime_reuse"]} if "runtime_reuse" in plan else {}),
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
    runtime_reuse: dict | None = None,
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
    if (
        checkpoint is None
        and any(value is not None for value in (checkpoint_guest_member, extra_inputs, loader_env))
        and not (
            native_integer.selected(worker_options)
            and checkpoint_guest_member is None
            and not extra_inputs
            and not loader_env
        )
    ):
        raise ValueError("declared loader environment and extra inputs require a selected checkpoint")
    run = _canonical_path(Path(run_dir), exists=False)
    destination = _canonical_path(Path(output_dir), exists=False)
    if run.exists() or destination.exists():
        raise ValueError("capture run and selection output must both be fresh")
    if run == destination or run.is_relative_to(destination) or destination.is_relative_to(run):
        raise ValueError("capture output and selection output may not overlap")
    if not run.parent.is_dir() or not destination.parent.is_dir():
        raise ValueError("capture and selection output parents must already exist")
    plan = m2m_plan.prepare_plan(
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
        **({"runtime_reuse": runtime_reuse} if runtime_reuse is not None else {}),
    )
    selected_inputs = [
        Path(plan[name])
        for name in ("m2m_root", "workload_root", "worker", "merlin_root", "schemas_root", "venv", "base")
    ]
    if plan.get("recipe"):
        selected_inputs.append(Path(plan["recipe"]["path"]))
    if "runtime_reuse" in plan:
        selected_inputs.append(Path(plan["runtime_reuse"]["entry"]))
    if plan["schema"] == m2m_sandbox.SCHEMA_V3:
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
        if selected.get("checkpoint") != {"kind": "none"} or selected["plan"].get("schema") != m2m_sandbox.SCHEMA:
            raise ValueError("v1 capture selection has an unsupported checkpoint or issuer policy")
    elif _native_batch(selected["plan"]):
        if selected.get("checkpoint") != {"kind": "none"}:
            raise ValueError("native batch selection has an unsupported checkpoint")
    elif selected["plan"].get("schema") != m2m_sandbox.SCHEMA_V3 or selected.get("checkpoint") != next(
        (row for row in selected["plan"].get("selected_inputs", []) if row.get("role") == "checkpoint"), None
    ):
        raise ValueError("v2 capture selection lacks its exact selected checkpoint")
    return selected


def issue(path: Path, *, expected_sha256: str) -> Path:
    """Execute only a fresh run selected by the owner-controlled artifact."""
    return _issue(path, expected_sha256=expected_sha256)[0]


def issue_and_attest(
    path: Path, *, expected_sha256: str, output: Path, cpus: tuple[frozenset[int], frozenset[int]]
) -> dict:
    """Issue a selected capture with its attestation replay beside it, then write the attestation.

    The same as :func:`issue` followed by :func:`attest`, except that the replay execution runs
    concurrently with issuance on the second of two disjoint CPU sets
    (:func:`m2m_plan.issue_and_replay`). Every check :func:`attest` makes runs unchanged; only
    the replay's start moves earlier.
    """
    _attestation_destination(path, expected_sha256=expected_sha256, output=output, run_exists=False)
    receipt, replay = _issue(path, expected_sha256=expected_sha256, cpus=cpus)
    result = attest(path, expected_sha256=expected_sha256, output=output, _replay=replay)
    issue_cpus, replay_cpus = cpus
    return {
        "sealed_receipt": str(receipt),
        **result,
        "replay_execution": {
            "mode": "concurrent_disjoint_cpus",
            "issue_cpus": sorted(issue_cpus),
            "replay_cpus": sorted(replay_cpus),
        },
    }


def _issue(
    path: Path, *, expected_sha256: str, cpus: tuple[frozenset[int], frozenset[int]] | None = None
) -> tuple[Path, dict | None]:
    selected = load(path, expected_sha256=expected_sha256)
    run = _canonical_path(Path(selected["run_dir"]), exists=False)
    if run.exists():
        raise ValueError("capture run already exists; a selection cannot adopt historical output")
    plan = selected["plan"]
    current = m2m_plan.prepare_plan(
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
        **({"runtime_reuse": plan["runtime_reuse"]} if "runtime_reuse" in plan else {}),
        **(
            {
                "checkpoint": None if _native_batch(plan) else Path(selected["checkpoint"]["source"]),
                "checkpoint_guest_member": None if _native_batch(plan) else selected["checkpoint"]["guest_member"],
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
    selected_execution = {
        "bwrap_binary": bwrap,
        "capture_selection_sha256": expected_sha256,
        "selected_system_libraries": selected["system_libraries"],
        "selected_bwrap_sha256": selected["bwrap"]["sha256"],
    }
    if cpus is None:
        receipt, replay = m2m_plan.issue(plan, run, **selected_execution), None
    else:
        receipt, replay = m2m_plan.issue_and_replay(plan, run, cpus=cpus, **selected_execution)
    if _digest(Path(path).read_bytes()) != expected_sha256:
        raise ValueError("capture selection changed during execution")
    return receipt, replay


def verify(path: Path, *, expected_sha256: str, model_path: Path, _replay: dict | None = None) -> dict:
    """Read-only independent binding of selected bytes to a fresh replay.

    The returned record intentionally remains nonadmissible. The separately
    issued sealed-M2M attestation checks these bytes again before granting
    source-closure admission under its explicit operator policy. ``_replay`` is
    only :func:`issue_and_attest`'s concurrent replay result, from
    :func:`m2m_plan.issue_and_replay`; every other caller replays now.
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
        receipt.get("schema") != (m2m_sandbox.SCHEMA_V3 if selected["schema"] == SCHEMA_V2 else m2m_sandbox.SCHEMA)
        or receipt.get("capture_selection_sha256") != expected_sha256
        or receipt.get("plan") != selected["plan"]
        or receipt.get("policy_sha256") != selected["sandbox_policy_sha256"]
        or receipt.get("bwrap_sha256") != selected["bwrap"]["sha256"]
        or receipt.get("issuer_sha256") != selected["issuer_source_sha256"]
    ):
        raise ValueError("sealed capture does not bind the independently selected plan and policy")
    guest_root = (
        Path(selected["plan"]["runtime_reuse"]["entry"])
        if "runtime_reuse" in selected["plan"]
        else run / "snapshots/guest-root"
    )
    if [library["path"] for library in selected["system_libraries"]] != selected["plan"]["system_libs"]:
        raise ValueError("selected system libraries differ from the planned runtime")
    for library in selected["system_libraries"]:
        source = Path(library["path"])
        if not source.is_absolute() or ".." in source.parts:
            raise ValueError("selected system library path is unsafe")
        if "runtime_reuse" in selected["plan"]:
            from merlin.experiments.capture_execution import readonly_runtime

            copied = readonly_runtime.file_path(selected["plan"]["runtime_reuse"], source.relative_to("/").as_posix())
        else:
            copied = guest_root / source.relative_to("/")
        if (
            copied.is_symlink()
            or not copied.is_file()
            or copied.stat().st_size != library["bytes"]
            or _file_digest(copied) != library["sha256"]
        ):
            raise ValueError("sealed runtime differs from preselected system library bytes")
    replay = sealed_m2m.replay_verify(run, bwrap_binary=Path(selected["bwrap"]["path"])) if _replay is None else _replay
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
        result["capture_tree_sha256"] = m2m_plan._snapshot_tree(model)["sha256"]
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


def _attestation_destination(
    path: Path, *, expected_sha256: str, output: Path, run_exists: bool = True
) -> tuple[dict, Path, Path, Path]:
    """The selection, its run, a fresh attestation destination and the selection path, checked."""
    selected = load(path, expected_sha256=expected_sha256)
    run = _canonical_path(Path(selected["run_dir"]), exists=run_exists)
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
    return selected, run, destination, selection_path


def attest(path: Path, *, expected_sha256: str, output: Path, _replay: dict | None = None) -> dict:
    """Replay one issued selected capture and write its sealed execution attestation.

    A v1 selection (checkpoint-free loader) yields the ``merlin.sealed_m2m_cpu`` attestation of its
    ``model.mlir``; a v2 selection (explicit checkpoint, extra inputs and loader environment) yields
    the v3 attestation of the complete capture root, single program or saved session. Either is
    issued only from a fresh :func:`verify` replay of these exact selected bytes, and is written to a
    fresh owner-only file outside the selected run, so the attestation never alters what it attests.
    """
    from merlin.development.testgen.sealed_capture.attestation import attest_sealed_m2m, attest_sealed_m2m_v3

    selected, run, destination, selection_path = _attestation_destination(
        path, expected_sha256=expected_sha256, output=output
    )
    if selected["schema"] == SCHEMA_V2:
        capture = run / "capture"
        replay = verify(path, expected_sha256=expected_sha256, model_path=capture, _replay=_replay)
        document = attest_sealed_m2m_v3(replay, selection_path=selection_path, capture_path=capture)
    else:
        model = run / "capture/model.mlir"
        replay = verify(path, expected_sha256=expected_sha256, model_path=model, _replay=_replay)
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


# ---- capture_recipe ----------------------------------------------------------------------------
# Select exactly one verified software-scoped capture recipe from frozen evidence.


def _selected_capture_recipe(evidence_root: Path, *, target: str, operand_dtype: str, accumulator_dtype: str) -> dict:
    """Read exactly one frozen SW-scoped recipe for a live integer model capture.

    Hardware-only recipe derivation is intentionally not a fallback: it can quantize families
    the selected software spec explicitly left on the host.
    """
    from merlin.common import target_data as quant_formats
    from merlin.compiler.quantization import recipe as quant_recipe

    index_path = evidence_root / "software" / "quantization-recipes.json"
    index = json.loads(index_path.read_bytes())
    if index.get("schema") != "merlin.phase0.capture_recipes.v1" or index.get("target") != target:
        raise ValueError("frozen capture-recipe index has wrong schema or target")
    wanted_operand = quant_formats.get(operand_dtype).name
    wanted_accumulator = quant_formats.get(accumulator_dtype).name
    matches = []
    for row in index.get("recipes") or []:
        rel = Path(str(row.get("path") or ""))
        if rel.is_absolute() or not rel.parts or ".." in rel.parts or rel.parts[0] != "software":
            raise ValueError("frozen capture recipe has an unsafe relative path")
        path = evidence_root / rel
        if path.is_symlink() or not path.is_file():
            raise ValueError("frozen capture recipe is absent or symlinked")
        raw = path.read_bytes()
        actual = hashlib.sha256(raw).hexdigest()
        if actual != row.get("sha256") or path.stem != actual:
            raise ValueError("frozen capture recipe bytes differ from their index")
        recipe = json.loads(raw)
        if recipe.get("target") != target or recipe.get("status") != quant_recipe.DERIVED:
            raise ValueError("frozen capture recipe has wrong target or unresolved status")
        digest = quant_recipe.digest(recipe)
        if recipe.get("recipe_sha256") != digest or row.get("recipe_sha256") != digest:
            raise ValueError("frozen capture recipe semantic digest differs from its index")
        if (
            quant_formats.get(recipe["weight"]["dtype"]).name == wanted_operand
            and quant_formats.get(recipe["activation"]["dtype"]).name == wanted_operand
            and quant_formats.get(recipe["accumulator_dtype"]).name == wanted_accumulator
        ):
            matches.append(recipe)
    if len(matches) != 1:
        raise ValueError(
            f"frozen evidence has {len(matches)} unambiguous SW-scoped recipes for "
            f"{wanted_operand}/{wanted_accumulator}; refusing a hardware-only default"
        )
    return matches[0]


def recipe_outputs(candidates, *, target: str, software_review, encode) -> dict[str, bytes]:
    """Frozen-evidence members for derived capture recipes: one file per recipe and their index.

    A reviewed model variant of a methodology recipe (``candidate["model_variants"]``) gets its own
    file, indexed under ``model_variants`` apart from ``recipes`` so that the model-independent
    selection (:func:`_selected_capture_recipe`) stays unambiguous.
    """
    outputs: dict[str, bytes] = {}
    recipes, variants = [], []

    def member(recipe) -> tuple[str, str]:
        payload = encode(recipe)
        digest = hashlib.sha256(payload).hexdigest()
        path = f"software/quantization-recipes/{digest}.json"
        outputs[path] = payload
        return path, digest

    for candidate in candidates:
        path, digest = member(candidate["recipe"])
        recipes.append(
            {key: value for key, value in candidate.items() if key not in ("recipe", "model_variants")}
            | {"path": path, "sha256": digest, "recipe_sha256": candidate["recipe"]["recipe_sha256"]}
        )
        for variant in candidate.get("model_variants") or ():
            path, digest = member(variant["recipe"])
            variants.append(
                {"format_id": candidate["format_id"], "model": variant["model"], "name": variant["name"]}
                | {"path": path, "sha256": digest, "recipe_sha256": variant["recipe"]["recipe_sha256"]}
            )
    outputs["software/quantization-recipes.json"] = encode(
        {
            "schema": "merlin.phase0.capture_recipes.v1",
            "target": target,
            "software_review": software_review,
            "recipes": recipes,
            **({"model_variants": variants} if variants else {}),
            "qualification": "diagnostic transformation inputs; calibrate and verify actual precision before admission",
        }
    )
    return outputs


def _with_selected_model_recipe(
    entry: dict,
    *,
    evidence_root: Path | None,
    target: str,
    binding,
) -> dict:
    """Use the frozen SW-scoped recipe when an integer model has no explicit choice."""
    if (
        evidence_root is None
        or (entry.get("kind") != "model" and entry.get("op") != "model")
        or entry.get("materialized_capture")
        or entry.get("quant_recipe")
        or entry.get("quant_scheme")
        or not binding.integer
    ):
        return entry
    return {
        **entry,
        "quant_recipe": _selected_capture_recipe(
            evidence_root,
            target=target,
            operand_dtype=str(entry.get("operand_dtype") or binding.operand_dtype),
            accumulator_dtype=str(binding.accum_dtype),
        ),
    }


def _prepare_model_capture_entry(
    entry: dict,
    *,
    evidence_root: Path | None,
    target: str,
    binding,
    capture=None,
) -> dict:
    """Check a selected static model capture tool before any capsule writer starts."""
    if entry.get("kind") != "model" and entry.get("op") != "model":
        return entry
    if entry.get("materialized_capture"):
        return entry

    from merlin.capture import pytorch as source_pytorch
    from merlin.capture import receipts as source_numerics
    from merlin.capture import receipts as source_receipts

    try:
        selected = _with_selected_model_recipe(entry, evidence_root=evidence_root, target=target, binding=binding)
    except Exception:  # noqa: BLE001 -- the writer records this entry's recipe failure as before
        return entry
    recipe = selected.get("quant_recipe") or source_numerics.derived_recipe(
        getattr(binding, "target", None), str(selected.get("operand_dtype") or binding.operand_dtype)
    )
    if source_receipts._static_pt2e_model(
        "model",
        scheme=None if recipe is not None else selected.get("quant_scheme"),
        recipe=recipe,
        already_quantized=selected.get("capture_quantization") == "already_materialized",
    ):
        capture = capture if capture is not None else source_pytorch.PytorchRefSource()
        integerizer = capture.m2m_dir / "m2m/capture/pt2e_integerize.py"
        if capture.available() and not integerizer.is_file():
            raise ValueError(
                f"selected model2MLIR checkout {capture.m2m_dir} lacks {integerizer}; "
                "static int8 model capture requires m2m.capture.pt2e_integerize"
            )
        if capture.available() and not entry.get("micro_model"):
            # The PT2E quantizer imports these from the selected checkout only
            # after source capsules start writing. The derived micro-model is
            # the sole exception: its graph is checked at capture time for
            # complete ATen provenance and absence of BatchNorm before Merlin
            # permits a missing fold API. Every other static model must have
            # the API before any writer runs.
            probe = subprocess.run(
                [
                    str(capture.python),
                    "-B",
                    "-c",
                    "import sys; sys.path.insert(0, sys.argv[1]); "
                    "from m2m.capture.trace import "
                    "pt2e_conv_bn_fold_candidates, attach_pt2e_conv_bn_folds",
                    str(capture.m2m_dir),
                ],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            if probe.returncode:
                raise ValueError(
                    f"selected model2MLIR checkout {capture.m2m_dir} lacks the required "
                    "m2m.capture.trace PT2E fold-provenance API "
                    "(pt2e_conv_bn_fold_candidates, attach_pt2e_conv_bn_folds); "
                    "select a compatible checkout and interpreter before generating capsules"
                )
    return selected
