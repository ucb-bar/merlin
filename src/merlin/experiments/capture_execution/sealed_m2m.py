"""One bounded, independently replayable CPU Model2MLIR capture.

This is a *process and byte-closure* proof for a selected CPU workload, not a
general Python purity theorem and not Phase 0 admission.  The guest has an
empty filesystem root apart from copied runtime/source, private devices and
temporary/output mounts; it has no network or host checkout mount.  Reviewers
must separately qualify the workload, framework numerics and Phase 0 policy.
"""

from __future__ import annotations

import ast
import json as json
import os
import secrets
import shutil
import stat
import subprocess
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

from merlin.common import record_formats as strict_json
from merlin.experiments.capture_execution import (
    readonly_runtime,
)
from merlin.experiments.capture_execution import (
    sealed_python as sealed_failure,
)
from merlin.experiments.capture_execution import (
    sealed_static as native_integer,
)
from merlin.experiments.capture_execution.m2m_origin import (
    M2MOriginError,
    git_origin,
    verify_frozen_receipt,
    verify_frozen_selector,
)
from merlin.experiments.capture_execution.precision_staging import STAGING_API_REQUIREMENTS as STAGING_API_REQUIREMENTS
from merlin.experiments.capture_execution.precision_staging import output_staging_error as output_staging_error
from merlin.experiments.capture_execution.python_preflight import _loader_env_reads
from merlin.experiments.capture_execution.sealed_python import (
    _FLAGS,
    _TIMEOUT_SECONDS,
    _snapshot_rows,
    _verify_replayed_output,
    _verify_staged_selection,
)
from merlin.experiments.capture_execution.sealed_static import (
    _bwrap_binary,
    _canonical_path,
    _digest,
    _file_digest,
    _json,
    _tree,
)

# ---- sealed_m2m --------------------------------------------------------------------------------


def _issue(
    plan: dict[str, Any],
    run_dir: Path,
    *,
    bwrap_binary: Path | None = None,
    capture_selection_sha256: str | None = None,
    selected_system_libraries: list[dict[str, Any]] | None = None,
    selected_bwrap_sha256: str | None = None,
    replay_cpus: tuple[frozenset[int], frozenset[int]] | None = None,
) -> tuple[Path, dict[str, Any] | None]:
    from merlin.experiments.capture_execution.m2m_plan import (
        _snapshot_tree,
        _stage_source,
        _validate_snapshots,
        prepare_plan,
    )
    from merlin.experiments.capture_execution.m2m_sandbox import (
        SCHEMA,
        SCHEMA_V3,
        SealedM2MError,
        _command_v2,
        _execute,
        _execute_beside,
        _probe_sandbox,
        _worker_member,
    )

    if plan.get("schema") not in {SCHEMA, SCHEMA_V3} or plan.get("status") != "plan_only":
        raise SealedM2MError("unsupported M2M plan")
    if capture_selection_sha256 is not None and (
        not isinstance(capture_selection_sha256, str)
        or len(capture_selection_sha256) != 64
        or any(character not in "0123456789abcdef" for character in capture_selection_sha256)
    ):
        raise SealedM2MError("capture selection requires an exact SHA-256 identity")
    if capture_selection_sha256 is not None and (selected_system_libraries is None or selected_bwrap_sha256 is None):
        raise SealedM2MError("selected capture requires antecedent system-library and bubblewrap bytes")
    if selected_system_libraries is not None and (
        not isinstance(selected_system_libraries, list)
        or [row.get("path") for row in selected_system_libraries] != plan.get("system_libs")
    ):
        raise SealedM2MError("selected system-library roster differs from the plan")
    selected = prepare_plan(
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
                "checkpoint": next(
                    (Path(row["source"]) for row in plan["selected_inputs"] if row["role"] == "checkpoint"), None
                ),
                "checkpoint_guest_member": next(
                    (row["guest_member"] for row in plan["selected_inputs"] if row["role"] == "checkpoint"), None
                ),
                "extra_inputs": {
                    row["guest_member"]: Path(row["source"])
                    for row in plan["selected_inputs"]
                    if row["role"] == "input"
                },
                "loader_env": plan["loader_env"],
                "execution_timeout_seconds": plan["execution_timeout_seconds"],
            }
            if plan["schema"] == SCHEMA_V3
            else (
                {"execution_timeout_seconds": plan["execution_timeout_seconds"]}
                if "execution_timeout_seconds" in plan
                else {}
            )
        ),
    )
    if selected != plan:
        raise SealedM2MError("selected M2M plan changed")
    run_dir = _canonical_path(run_dir, exists=False)
    # The source venv is read-only.  The copy and capture live on the selected
    # run filesystem, so only that destination's capacity is relevant here.
    from merlin.experiments.capture_execution.runtime_store import snapshot_space_requirement

    reuse = plan.get("runtime_reuse")
    if reuse is not None:
        readonly_runtime.verify(plan, reuse, run_dir.parent)
        required_bytes, verified_cache = plan["estimate_bytes"] - reuse["runtime_bytes"] + 64 * 1024**2, None
    else:
        required_bytes, verified_cache = snapshot_space_requirement(plan, run_dir.parent, selected_system_libraries)
    if required_bytes > shutil.disk_usage(run_dir.parent).free:
        raise SealedM2MError("normalized snapshot bytes exceed selected run filesystem free space")
    inputs = [
        Path(plan[key])
        for key in ("m2m_root", "workload_root", "worker", "merlin_root", "schemas_root", "venv", "base")
    ]
    if plan.get("recipe"):
        inputs.append(Path(plan["recipe"]["path"]))
    if reuse is not None:
        inputs.append(Path(reuse["entry"]))
    if plan["schema"] == SCHEMA_V3:
        inputs.extend(Path(row["source"]) for row in plan["selected_inputs"])
    if any(run_dir == path or run_dir.is_relative_to(path) or path.is_relative_to(run_dir) for path in inputs):
        raise SealedM2MError("run directory overlaps a selected input")
    bwrap = _bwrap_binary(bwrap_binary)
    if selected_bwrap_sha256 is not None and _file_digest(bwrap) != selected_bwrap_sha256:
        raise SealedM2MError("bubblewrap bytes differ from the pre-execution selection")
    _probe_sandbox(bwrap)
    run_dir.mkdir(mode=0o700, parents=False, exist_ok=False)
    source = run_dir / "snapshots/source"
    runtime = run_dir / "snapshots/guest-root"
    source.mkdir(parents=True)
    runtime.mkdir()
    _stage_source(plan, source)
    # The selected runtime (venv, base interpreter, system libraries) is identical across captures, so
    # it is materialized once into a content-addressed store and hard-linked into this private guest
    # root. Every byte is still re-verified against the plan before anything executes.
    from merlin.experiments.capture_execution.runtime_store import link_runtime

    if reuse is not None:
        readonly_runtime.stage(reuse, runtime)
    else:
        link_runtime(plan, runtime, verified_cache=verified_cache, selected_system_libraries=selected_system_libraries)
    output = run_dir / "capture"
    for name in ("source", "capture-out", "tmp", "dev"):
        (runtime / name).mkdir()
    (runtime / output.relative_to("/")).mkdir(parents=True)
    _validate_snapshots(
        source,
        runtime,
        output,
        schema=plan["schema"],
        worker_member=_worker_member(plan),
        **({"runtime_reuse": reuse} if reuse is not None else {}),
    )
    # Rehash the copied bytes against the antecedent selection, not mutable
    # host paths: a source change after prepare_plan must never be executed.
    source_rows, runtime_rows = _tree(source), readonly_runtime.effective_rows(reuse) if reuse else _tree(runtime)
    selected_schemas = _verify_staged_selection(
        plan, source, Path(reuse["entry"]) if reuse else runtime, source_rows=source_rows, runtime_rows=runtime_rows
    )
    if _file_digest(source / _worker_member(plan)) != plan["worker_sha256"]:
        raise SealedM2MError("worker snapshot differs from selected bytes")
    if plan.get("recipe") and _file_digest(source / "inputs/quant_recipe.json") != plan["recipe"]["sha256"]:
        raise SealedM2MError("selected recipe snapshot differs from selected bytes")
    for index, name in enumerate(plan["system_libs"]):
        member = Path(name).relative_to("/").as_posix()
        copied = readonly_runtime.file_path(reuse, member) if reuse else runtime / member
        expected = selected_system_libraries[index] if selected_system_libraries is not None else None
        if expected is not None:
            if copied.stat().st_size != expected.get("bytes") or _file_digest(copied) != expected.get("sha256"):
                raise SealedM2MError(f"system ELF snapshot differs from the pre-execution selection: {name}")
        elif _file_digest(copied) != _file_digest(Path(name)):
            raise SealedM2MError(f"system ELF snapshot differs from selected bytes: {name}")
    source_digest = _snapshot_rows(source_rows)
    runtime_digest = _snapshot_tree(runtime) if reuse else _snapshot_rows(runtime_rows)
    del source_rows, runtime_rows
    if source_digest["bytes"] + runtime_digest["bytes"] > plan["max_snapshot_bytes"]:
        raise SealedM2MError("actual snapshot bytes exceed selected cap")
    output.mkdir()
    command = _command_v2(
        output,
        dtype=plan["dtype"],
        recipe=plan.get("recipe") is not None,
        options=plan.get("worker_options"),
        worker_member=_worker_member(plan),
    )
    if selected_bwrap_sha256 is not None and _file_digest(bwrap) != selected_bwrap_sha256:
        raise SealedM2MError("bubblewrap bytes changed before sandbox execution")
    replayable_logs = capture_selection_sha256 is not None
    loader_env = plan.get("loader_env") if plan["schema"] == SCHEMA_V3 else None
    if replay_cpus is None:
        process = _execute(
            bwrap,
            runtime,
            source,
            output,
            command,
            output,
            replayable_logs=replayable_logs,
            loader_env=loader_env,
            timeout_seconds=plan.get("execution_timeout_seconds", _TIMEOUT_SECONDS),
            **({"runtime_reuse": reuse} if reuse is not None else {}),
        )
        with sealed_failure.publication(run_dir, replay=None, plan=plan, selection_sha256=capture_selection_sha256):
            return _write_pending(
                plan=plan,
                run_dir=run_dir,
                source=source,
                runtime=runtime,
                output=output,
                command=command,
                process=process,
                bwrap=bwrap,
                source_digest=source_digest,
                runtime_digest=runtime_digest,
                selected_schemas=selected_schemas,
                replayable_logs=replayable_logs,
                loader_env=loader_env,
                capture_selection_sha256=capture_selection_sha256,
                selected_bwrap_sha256=selected_bwrap_sha256,
            ), None
    # The replay's own empty output, made as replay_verify makes it: inside the run, with the
    # issued output's mode, bound at the same guest path.
    replay_output = Path(tempfile.mkdtemp(prefix="m2m-replay-", dir=run_dir))
    with sealed_failure.publication(
        run_dir, replay=replay_output, plan=plan, selection_sha256=capture_selection_sha256
    ):
        replay_output.chmod(output.stat().st_mode & 0o777)
        process, replay_process = _execute_beside(
            bwrap,
            runtime,
            source,
            (output, replay_output),
            command,
            output,
            cpus=replay_cpus,
            replayable_logs=replayable_logs,
            loader_env=loader_env,
            timeout_seconds=plan.get("execution_timeout_seconds", _TIMEOUT_SECONDS),
            **({"runtime_reuse": reuse} if reuse is not None else {}),
        )
        receipt = _write_pending(
            plan=plan,
            run_dir=run_dir,
            source=source,
            runtime=runtime,
            output=output,
            command=command,
            process=process,
            bwrap=bwrap,
            source_digest=source_digest,
            runtime_digest=runtime_digest,
            selected_schemas=selected_schemas,
            replayable_logs=replayable_logs,
            loader_env=loader_env,
            capture_selection_sha256=capture_selection_sha256,
            selected_bwrap_sha256=selected_bwrap_sha256,
        )
        return receipt, replay_verify(run_dir, bwrap_binary=bwrap, _executed=(replay_process, replay_output))


def _write_pending(
    *,
    plan: dict[str, Any],
    run_dir: Path,
    source: Path,
    runtime: Path,
    output: Path,
    command: tuple[str, ...],
    process: dict[str, Any],
    bwrap: Path,
    source_digest: dict[str, Any],
    runtime_digest: dict[str, Any],
    selected_schemas: Any,
    replayable_logs: bool,
    loader_env: dict[str, str | None] | None,
    capture_selection_sha256: str | None,
    selected_bwrap_sha256: str | None,
) -> Path:
    """Check the executed capture and write its pending receipt (shared by both issue modes)."""
    from merlin.experiments.capture_execution.m2m_plan import _materialized_v2, _materialized_v3, _snapshot_tree
    from merlin.experiments.capture_execution.m2m_sandbox import (
        _V2_SCOPE,
        _V3_SCOPE,
        SCHEMA_V3,
        SealedM2MError,
        _policy,
    )

    materialized = (
        _materialized_v3(output, source, output, plan)
        if plan["schema"] == SCHEMA_V3
        else _materialized_v2(output, source, output, plan)
    )
    if (_snapshot_tree(source), _snapshot_tree(runtime)) != (source_digest, runtime_digest):
        raise SealedM2MError("sealed source or runtime changed during capture")
    if plan.get("runtime_reuse") is not None:
        readonly_runtime.verify(plan, plan["runtime_reuse"], runtime)
    if selected_bwrap_sha256 is not None and _file_digest(bwrap) != selected_bwrap_sha256:
        raise SealedM2MError("bubblewrap bytes changed during sandbox execution")
    receipt = run_dir / "sealed_m2m_pending.json"
    payload = {
        "schema": plan["schema"],
        "status": "pending_replay",
        "issuer_sha256": _file_digest(Path(__file__)),
        "nonce": secrets.token_hex(16),
        "plan": plan,
        "command": list(command),
        "policy_sha256": _policy(
            command,
            output,
            replayable_logs=replayable_logs,
            loader_env=loader_env,
            timeout_seconds=plan.get("execution_timeout_seconds", _TIMEOUT_SECONDS),
            **({"runtime_reuse": plan["runtime_reuse"]} if "runtime_reuse" in plan else {}),
        ),
        "source": source_digest,
        "guest_root": runtime_digest,
        "output": _snapshot_tree(output),
        "schemas": selected_schemas,
        "process": process,
        "materialized": materialized,
        "bwrap_sha256": _file_digest(bwrap),
        "scope": _V3_SCOPE if plan["schema"] == SCHEMA_V3 else _V2_SCOPE,
    }
    if capture_selection_sha256 is not None:
        payload["capture_selection_sha256"] = capture_selection_sha256
    with receipt.open("xb") as stream:
        stream.write(_json(payload) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    receipt.chmod(0o444)
    return receipt


def replay_verify(
    run_dir: Path,
    *,
    bwrap_binary: Path | None = None,
    _executed: tuple[dict[str, Any], Path] | None = None,
) -> dict[str, Any]:
    """Independent replay from exact private bytes; no Phase 0 status is granted.

    ``_executed`` is only for :func:`issue_and_replay`: the process record and fresh output of a
    replay that ran beside issuance, compared here exactly as a replay run now would be.
    """
    from merlin.experiments.capture_execution.m2m_plan import (
        _declared_loader_env,
        _materialized,
        _materialized_v2,
        _materialized_v3,
        _snapshot_tree,
        _validate_snapshots,
        _verify_selected_input,
    )
    from merlin.experiments.capture_execution.m2m_sandbox import (
        _ARCHIVED_SELECTED_V2_ISSUER_SHA256,
        _ARCHIVED_V3_ISSUER_SHA256,
        _FLOAT_DTYPES,
        _HISTORICAL_V2_ISSUER_SHA256,
        _LAUNCH_PREFIX,
        _LAUNCH_SUFFIX,
        _PRE_CACHE_NORMALIZATION_V3_ISSUER_SHA256,
        _PRE_CONCURRENT_REPLAY_ISSUER_SHA256,
        _PRE_DYNAMIC_REFERENCE_ISSUER_SHA256,
        _PRE_EXTERNAL_REPLAY_SCRATCH_ISSUER_SHA256,
        _PRE_FAILURE_DIAGNOSTICS_ISSUER_SHA256,
        _PRE_FROZEN_ORIGIN_ISSUER_SHA256,
        _PRE_INVENTORY_REUSE_ISSUER_SHA256,
        _PRE_NATIVE_INTEGER_ISSUER_SHA256,
        _PRE_PUBLICATION_DIAGNOSTICS_ISSUER_SHA256,
        _PRE_REPLAY_DIAGNOSTICS_ISSUER_SHA256,
        _PRE_SELECTED_THREADS_ISSUER_SHA256,
        _PRE_V3_ISSUER_SHA256,
        _STRICT_JSON_PREDECESSOR_V2_ISSUER_SHA256,
        _STRICT_JSON_SELECTED_ISSUER_SHA256,
        _V1_ISSUER_SHA256,
        _V1_SCOPE,
        _V2_SCOPE,
        _V3_SCOPE,
        SCHEMA,
        SCHEMA_V1,
        SCHEMA_V3,
        SealedM2MError,
        _command,
        _command_v2,
        _execute,
        _policy,
        _selected_timeout_valid,
        _worker_member,
    )

    run_dir = _canonical_path(run_dir, exists=True)
    receipt = run_dir / "sealed_m2m_pending.json"
    if receipt.is_symlink() or not receipt.is_file():
        raise SealedM2MError("pending M2M receipt is absent or indirect")
    try:
        doc = strict_json.loads(receipt.read_bytes())
    except (ValueError, UnicodeDecodeError) as exc:
        raise SealedM2MError("pending M2M receipt is unreadable") from exc
    plan = doc.get("plan") or {}
    schema = doc.get("schema")
    if schema == SCHEMA_V1:
        expected_issuer = _V1_ISSUER_SHA256
        expected_template = _digest((_LAUNCH_PREFIX + _LAUNCH_SUFFIX).encode())
        expected_command = _command(run_dir / "capture")
        expected_scope = _V1_SCOPE
    elif schema in {SCHEMA, SCHEMA_V3}:
        dtype, recipe = plan.get("dtype"), plan.get("recipe")
        is_native = native_integer.selected(plan.get("worker_options"))
        if is_native:
            try:
                native_integer.require_selection(plan["worker_options"], dtype=dtype, recipe=recipe)
            except ValueError as error:
                raise SealedM2MError(str(error)) from error
        elif dtype not in _FLOAT_DTYPES | {"int8"} or (recipe is not None) != (dtype == "int8"):
            raise SealedM2MError("pending M2M receipt has an unsupported dtype or recipe selection")
        if not _selected_timeout_valid(schema, plan.get("execution_timeout_seconds")):
            raise SealedM2MError("pending capture has no selected bounded execution timeout")
        selected_digest = doc.get("capture_selection_sha256")
        if selected_digest is not None and (
            not isinstance(selected_digest, str)
            or len(selected_digest) != 64
            or any(character not in "0123456789abcdef" for character in selected_digest)
        ):
            raise SealedM2MError("pending M2M receipt has a malformed capture selection identity")
        expected_issuer = _file_digest(Path(__file__))
        expected_template = _digest(
            _command_v2(
                Path("/capture-out"),
                dtype=dtype,
                recipe=recipe is not None,
                options=plan.get("worker_options"),
                worker_member=_worker_member(plan),
            )[-1].encode()
        )
        expected_command = _command_v2(
            run_dir / "capture",
            dtype=dtype,
            recipe=recipe is not None,
            options=plan.get("worker_options"),
            worker_member=_worker_member(plan),
        )
        expected_scope = _V3_SCOPE if schema == SCHEMA_V3 else _V2_SCOPE
    else:
        raise SealedM2MError("pending M2M receipt has an unsupported schema")
    replayable_logs = schema in {SCHEMA, SCHEMA_V3} and doc.get("capture_selection_sha256") is not None
    loader_env = plan.get("loader_env") if schema == SCHEMA_V3 else None
    supported_issuer = (
        {expected_issuer, _PRE_V3_ISSUER_SHA256, _HISTORICAL_V2_ISSUER_SHA256}
        if schema == SCHEMA and doc.get("capture_selection_sha256") is None
        else (
            {expected_issuer, _PRE_V3_ISSUER_SHA256, _STRICT_JSON_PREDECESSOR_V2_ISSUER_SHA256}
            if schema == SCHEMA
            else (
                {expected_issuer, _ARCHIVED_V3_ISSUER_SHA256, _PRE_CACHE_NORMALIZATION_V3_ISSUER_SHA256}
                if schema == SCHEMA_V3
                else {expected_issuer}
            )
        )
    )
    if schema in {SCHEMA, SCHEMA_V3} and doc.get("capture_selection_sha256") is not None:
        supported_issuer.add(_STRICT_JSON_SELECTED_ISSUER_SHA256)
    if schema == SCHEMA and doc.get("capture_selection_sha256") is not None:
        supported_issuer.add(_ARCHIVED_SELECTED_V2_ISSUER_SHA256)
    if schema in {SCHEMA, SCHEMA_V3}:
        supported_issuer.add(_PRE_INVENTORY_REUSE_ISSUER_SHA256)
        supported_issuer.add(_PRE_CONCURRENT_REPLAY_ISSUER_SHA256)
        supported_issuer.add(_PRE_EXTERNAL_REPLAY_SCRATCH_ISSUER_SHA256)
        supported_issuer.add(_PRE_SELECTED_THREADS_ISSUER_SHA256)
        if not native_integer.selected(plan.get("worker_options")):
            supported_issuer.add(_PRE_NATIVE_INTEGER_ISSUER_SHA256)
        supported_issuer.add(_PRE_FAILURE_DIAGNOSTICS_ISSUER_SHA256)
        supported_issuer.add(_PRE_REPLAY_DIAGNOSTICS_ISSUER_SHA256)
        supported_issuer.add(_PRE_PUBLICATION_DIAGNOSTICS_ISSUER_SHA256)
    if schema in {SCHEMA, SCHEMA_V3} and "frozen_origin" not in plan:
        supported_issuer.add(_PRE_FROZEN_ORIGIN_ISSUER_SHA256)
    if native_integer.selected(plan.get("worker_options")) or "runtime_reuse" in plan:
        supported_issuer = {expected_issuer}
    if schema in {SCHEMA, SCHEMA_V3} and doc.get("capture_selection_sha256") is not None:
        supported_issuer.add(_PRE_DYNAMIC_REFERENCE_ISSUER_SHA256)
    if (
        doc.get("status") != "pending_replay"
        or doc.get("issuer_sha256") not in supported_issuer
        or not isinstance(doc.get("nonce"), str)
        or len(doc["nonce"]) != 32
        or plan.get("schema") != schema
        or plan.get("status") != "plan_only"
        or plan.get("command_template_sha256") != expected_template
        or doc.get("command") != list(expected_command)
        or doc.get("policy_sha256")
        != _policy(
            expected_command,
            run_dir / "capture",
            replayable_logs=replayable_logs,
            loader_env=loader_env,
            timeout_seconds=plan.get("execution_timeout_seconds", _TIMEOUT_SECONDS),
            **({"runtime_reuse": plan["runtime_reuse"]} if "runtime_reuse" in plan else {}),
        )
        or doc.get("scope") != expected_scope
    ):
        raise SealedM2MError("pending M2M receipt has an unsupported policy")
    source, runtime, output = (run_dir / "snapshots/source", run_dir / "snapshots/guest-root", run_dir / "capture")
    reuse = plan.get("runtime_reuse")
    if reuse is not None:
        if plan.get("readonly_runtime_verifier_sha256") != _file_digest(Path(readonly_runtime.__file__)):
            raise SealedM2MError("selected read-only runtime verifier source changed")
        readonly_runtime.verify(plan, reuse, runtime)
    _validate_snapshots(
        source,
        runtime,
        output,
        schema=schema,
        **({"worker_member": _worker_member(plan)} if schema != SCHEMA_V1 else {}),
        **({"runtime_reuse": reuse} if reuse is not None else {}),
    )
    source_rows, runtime_rows = _tree(source), _tree(runtime)
    if (_snapshot_rows(source_rows), _snapshot_rows(runtime_rows), _snapshot_tree(output)) != (
        doc.get("source"),
        doc.get("guest_root"),
        doc.get("output"),
    ):
        raise SealedM2MError("sealed M2M snapshot or capture bytes differ")
    if schema in {SCHEMA, SCHEMA_V3}:
        runtime_view = Path(reuse["entry"]) if reuse else runtime
        view_rows = readonly_runtime.effective_rows(reuse) if reuse else runtime_rows
        selected_trees = plan.get("selected_trees") or {}
        if not isinstance(selected_trees, dict):
            raise SealedM2MError("v2 plan lacks exact selected source/runtime tree identities")
        commit = plan.get("m2m_commit")
        if (
            not isinstance(commit, str)
            or len(commit) != 40
            or any(character not in "0123456789abcdef" for character in commit)
        ):
            raise SealedM2MError("v2 plan lacks a pinned M2M revision identity")
        selected_roots = {
            "venv": runtime_view / "opt/capture-venv",
            "m2m": source / "m2m-src/m2m",
            "workload": source / "workload",
            "schemas": source / "merlin-src/merlin/_data/schemas",
        }
        base = plan.get("base")
        if (
            not isinstance(base, str)
            or not base.startswith("/")
            or base == "/"
            or Path(base).as_posix() != base
            or ".." in Path(base).parts
        ):
            raise SealedM2MError("selected base Python path is absent or unsafe")
        selected_roots["base"] = runtime_view / base.lstrip("/")
        if set(selected_trees) != set(selected_roots) | {"merlin"}:
            raise SealedM2MError("v2 plan lacks exact selected source/runtime tree identities")
        if plan.get("frozen_origin") is not None:
            if schema != SCHEMA:
                raise SealedM2MError("frozen M2M origin is not a v2 source selection")
            try:
                if (
                    verify_frozen_receipt(
                        plan["frozen_origin"],
                        (source / "m2m-origin.json").read_bytes(),
                        Path(plan["m2m_root"]),
                        selected_trees["m2m"],
                    )
                    != commit
                ):
                    raise SealedM2MError("frozen M2M origin revision differs from selected plan")
            except (M2MOriginError, OSError) as exc:
                raise SealedM2MError("frozen M2M origin receipt differs from replayed source") from exc
        # The external-schema policy adds the selected schema tree to the
        # Merlin package after its original selected-tree digest was taken.
        if plan.get("schemas_root") == str(Path(str(plan.get("merlin_root"))) / "_data/schemas"):
            selected_roots["merlin"] = source / "merlin-src/merlin"
        for name, path in selected_roots.items():
            root, rows = (runtime_view, view_rows) if path.is_relative_to(runtime_view) else (source, source_rows)
            if _snapshot_rows(rows, path.relative_to(root).as_posix()) != selected_trees[name]:
                raise SealedM2MError(f"selected {name} bytes differ from the v2 plan")
        if _file_digest(source / _worker_member(plan)) != plan.get("worker_sha256"):
            raise SealedM2MError("selected Merlin worker bytes differ from the v2 plan")
        if selected_trees["schemas"] != doc.get("schemas"):
            raise SealedM2MError("selected Merlin schema bytes differ from the v2 receipt")
        if schema == SCHEMA_V3:
            if not isinstance(plan.get("loader_env"), dict) or not isinstance(plan.get("selected_inputs"), list):
                raise SealedM2MError("v3 plan lacks declared environment or checkpoint inputs")
            declared, reads = _declared_loader_env((source / "workload/loader.py").read_text(), plan["loader_env"])
            if declared != plan["loader_env"] or reads != plan.get("loader_env_reads"):
                raise SealedM2MError("v3 loader environment declaration changed")
            selected_rows = plan["selected_inputs"]
            if native_integer.selected(plan.get("worker_options")):
                if selected_rows or plan["loader_env"] or reads:
                    raise SealedM2MError(
                        "native integer session requires no checkpoint, extra input or loader environment"
                    )
            elif sum(row.get("role") == "checkpoint" for row in selected_rows) != 1:
                raise SealedM2MError("v3 plan must select exactly one checkpoint")
            for row in selected_rows:
                _verify_selected_input(row, source / "inputs" / row["guest_member"])
    materialized = (
        _materialized(output, source, output)
        if schema == SCHEMA_V1
        else (
            _materialized_v3(output, source, output, plan)
            if schema == SCHEMA_V3
            else _materialized_v2(output, source, output, plan)
        )
    )
    if materialized != doc.get("materialized"):
        raise SealedM2MError("materialized M2M receipt differs")
    bwrap = _bwrap_binary(bwrap_binary)
    del source_rows, runtime_rows
    if _file_digest(bwrap) != doc.get("bwrap_sha256"):
        raise SealedM2MError("bubblewrap bytes differ from M2M receipt")

    def compare(process: dict[str, Any], replay: Path) -> None:
        if process != doc.get("process"):
            raise SealedM2MError("fresh M2M process output differs")
        _verify_replayed_output(replay, source, output, doc, schema, plan)

    if _executed is not None:
        process, replay = _executed
        if replay.parent != run_dir or not replay.name.startswith("m2m-replay-") or replay.is_symlink():
            raise SealedM2MError("concurrent M2M replay output is not this run's fresh replay directory")
        compare(process, replay)
    else:
        # The host scratch directory is mounted at the original output path.
        # Keep it outside chattr-protected captures when the operator selects TMPDIR.
        with tempfile.TemporaryDirectory(prefix="m2m-replay-", dir=os.environ.get("TMPDIR") or run_dir) as temporary:
            replay = Path(temporary)
            replay.chmod(output.stat().st_mode & 0o777)
            compare(
                _execute(
                    bwrap,
                    runtime,
                    source,
                    replay,
                    expected_command,
                    output,
                    replayable_logs=replayable_logs,
                    loader_env=loader_env,
                    timeout_seconds=plan.get("execution_timeout_seconds", _TIMEOUT_SECONDS),
                    **({"runtime_reuse": reuse} if reuse is not None else {}),
                ),
                replay,
            )
    if (_snapshot_tree(source), _snapshot_tree(runtime), _snapshot_tree(output)) != (
        doc["source"],
        doc["guest_root"],
        doc["output"],
    ):
        raise SealedM2MError("sealed M2M evidence changed during replay")
    if reuse is not None:
        readonly_runtime.verify(plan, reuse, runtime)
    result = {
        "schema": schema,
        "status": "verified_sandbox_replay",
        "sealed_source_closure_replayed": True,
        "scope": (
            "selected FP32 CPU workload in copied empty-root Python runtime only"
            if schema == SCHEMA_V1
            else "selected CPU workload in copied empty-root Python runtime only"
        ),
        "phase0_admission": "not_granted",
        "receipt_sha256": _file_digest(receipt),
    }
    if schema in {SCHEMA, SCHEMA_V3}:
        result["capture_dtype"] = plan["dtype"]
    return result
