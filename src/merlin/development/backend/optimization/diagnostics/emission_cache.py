"""Whole-model compiler emission analysis, exact baseline reuse and readiness evidence.

Analysis consumes explicit contract resources and admitted immutable model inputs.
It emits compiler artifacts but never runs a whole-model simulator.
"""

from __future__ import annotations

import copy
import json
import math
import os
import shutil
import subprocess
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from merlin.analysis.optimization.guidance.agent_brief import inspect_compiler_package
from merlin.analysis.optimization.guidance.emission_brief import guidance_for_emission_analysis
from merlin.analysis.perf.measurement.execution_policy import (
    FULL_GRAPH_STATIC_ANALYSIS_MAX_SECONDS,
    ITERATION_MAX_SECONDS,
)
from merlin.common.digest import hash_tree
from merlin.common.digest import sha256_bytes as _sha256
from merlin.development.backend.optimization import contracts as CONTRACTS
from merlin.development.backend.optimization.author_interface import stage_inputs as INPUTS
from merlin.development.backend.optimization.campaign import identities as SI
from merlin.development.backend.optimization.contracts import StageGateError, _is_sha256
from merlin.development.backend.optimization.contracts import canonical_json as _canonical_json
from merlin.development.backend.optimization.contracts import write_json as _write_json
from merlin.development.backend.optimization.diagnostics import emission_diagnostics as ED

# ---- emission_analysis -------------------------------------------------------------------------

if TYPE_CHECKING:
    from merlin.development.targetgen.packaging import loading as OR


def seed_baseline_emission_cache_from_run(
    *,
    cache_binding: Mapping[str, Any],
    seed_run: Path,
    baseline: Path,
    sentinels: Sequence[INPUTS.StageE2ESentinel],
    target: str,
    compiler_api_schema: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Import only exact successful baseline-emission bytes from a prior host-owned run."""
    from merlin.development.targetgen.packaging import loading as OR

    seed_run = Path(seed_run).resolve()
    if seed_run.is_symlink() or not seed_run.is_dir():
        raise ValueError("baseline emission seed run is absent or linked")
    experiment = CONTRACTS.mapping_file(seed_run / "global_iterations/experiment.json")
    dependencies = (experiment.get("optimization_baseline") or {}).get("compiler_dependencies")
    baseline_sha256 = hash_tree(baseline)["sha256"]
    if (
        experiment.get("target") != target
        or experiment.get("optimization_baseline_sha256") != baseline_sha256
        or not isinstance(dependencies, Mapping)
        or SI.compiler_dependency_content_sha256(dependencies) != cache_binding.get("compiler_dependencies_sha256")
    ):
        raise ValueError("baseline emission seed run compiler, target or dependencies differ")
    by_capsule = {sentinel.capsule_sha256: sentinel for sentinel in sentinels}
    if len(by_capsule) != len(sentinels):
        raise ValueError("baseline emission seed portfolio identities are not unique")
    package = OR.load_package(baseline)
    entrypoints = OR.analysis_emission_entrypoints(package)
    imported: dict[str, dict[str, Any]] = {}
    workers = seed_run / "host_analysis_workers"
    for worker in sorted(workers.iterdir()) if workers.is_dir() else ():
        required = [worker / name for name in ("request.json", "baseline_emission.json", "baseline_lowered.mlir")]
        command_buffer_path = worker / "compiler_scratch/baseline/command_buffer.json"
        if any(path.is_symlink() or not path.is_file() for path in (*required, command_buffer_path)):
            continue
        request = CONTRACTS.mapping_file(required[0])
        sentinel_record = request.get("sentinel") or {}
        capsule_sha256 = sentinel_record.get("capsule_sha256")
        sentinel = by_capsule.get(capsule_sha256)
        if sentinel is None or capsule_sha256 in imported:
            continue
        baseline_value = request.get("baseline")
        worker_baseline = Path(baseline_value) if isinstance(baseline_value, str) else None
        if (
            request.get("kwargs", {}).get("target") != target
            or worker_baseline is None
            or not worker_baseline.is_absolute()
            or worker_baseline.is_symlink()
            or not worker_baseline.is_dir()
            or hash_tree(worker_baseline)["sha256"] != baseline_sha256
        ):
            raise ValueError("baseline emission seed worker changed compiler or target")
        source_root = Path(sentinel.frozen_source_path)
        descriptor = CONTRACTS.mapping_file(source_root / "capsule.yaml", yaml_file=True)
        interface = source_root / str(descriptor.get("interface_mlir") or "capsule.interface.mlir")
        copied_interface = worker / "compiler_scratch/baseline/interface.mlir"
        if (
            copied_interface.is_symlink()
            or not copied_interface.is_file()
            or copied_interface.read_bytes() != interface.read_bytes()
        ):
            raise ValueError("baseline emission seed worker source bytes differ")
        emission = CONTRACTS.mapping_file(required[1])
        rows = emission.get("entrypoints")
        if (
            emission.get("schema") != "compiler_emission_diagnostics_v1"
            or not isinstance(rows, list)
            or [row.get("command") for row in rows] != list(entrypoints)
            or any(row.get("returncode") != 0 for row in rows)
        ):
            raise ValueError("baseline emission seed worker did not complete exact entrypoints")
        lowered_text = required[2].read_text(encoding="utf-8")
        command_buffer_text = command_buffer_path.read_text(encoding="utf-8")
        command_buffer = json.loads(command_buffer_text)
        if not isinstance(command_buffer, Mapping) or command_buffer.get("declined") is not None:
            raise ValueError("baseline emission seed worker produced no admitted command buffer")
        ED.validate_whole_program_schema(command_buffer, compiler_api_schema, arm="seed")
        identity = baseline_emission_cache_identity(
            baseline_sha256=baseline_sha256,
            capsule_sha256=capsule_sha256,
            source_sha256=_sha256(interface.read_bytes()),
            target=target,
            compiler_dependencies_sha256=str(cache_binding["compiler_dependencies_sha256"]),
            compiler_api_schema=compiler_api_schema,
            entrypoints=entrypoints,
        )
        elapsed = required[2].stat().st_mtime - required[0].stat().st_mtime
        worker_receipt = CONTRACTS.mapping_file(worker / "receipt.json") if (worker / "receipt.json").is_file() else {}
        analysis_status = worker_receipt.get("status")
        analysis_wall = worker_receipt.get("wall_seconds") if analysis_status in ("completed", "timeout") else None
        imported[capsule_sha256] = store_baseline_emission_cache(
            cache_binding,
            identity,
            lowered_text=lowered_text,
            command_buffer_text=command_buffer_text,
            emission_wall_seconds=max(0.0, elapsed),
            observed_analysis_wall_seconds=(
                float(analysis_wall)
                if isinstance(analysis_wall, (int, float))
                and not isinstance(analysis_wall, bool)
                and math.isfinite(analysis_wall)
                and analysis_wall >= 0
                else None
            ),
            observed_analysis_status=(
                analysis_status
                if isinstance(analysis_wall, (int, float))
                and not isinstance(analysis_wall, bool)
                and math.isfinite(analysis_wall)
                and analysis_wall >= 0
                else None
            ),
        )
    missing = sorted(set(by_capsule) - imported.keys())
    if missing:
        raise ValueError(f"baseline emission seed run lacks portfolio members: {missing}")
    return [imported[sentinel.capsule_sha256] for sentinel in sentinels]


def baseline_emission_cache_identity(
    *,
    baseline_sha256: str,
    capsule_sha256: str,
    source_sha256: str,
    target: str,
    compiler_dependencies_sha256: str,
    compiler_api_schema: Mapping[str, Any] | None,
    entrypoints: Sequence[str],
) -> dict[str, Any]:
    """Bind a reusable compiler emission to every input that can affect its bytes."""
    if not all(
        _is_sha256(value) for value in (baseline_sha256, capsule_sha256, source_sha256, compiler_dependencies_sha256)
    ):
        raise ValueError("baseline emission cache identity requires exact SHA-256 inputs")
    schema = dict(compiler_api_schema) if compiler_api_schema is not None else None
    if schema is not None and (not _is_sha256(schema.get("sha256")) or not isinstance(schema.get("path"), str)):
        raise ValueError("baseline emission cache requires an exact compiler API schema")
    schema_identity = None if schema is None else {"filename": Path(schema["path"]).name, "sha256": schema["sha256"]}
    if not target or not entrypoints or any(not isinstance(name, str) or not name for name in entrypoints):
        raise ValueError("baseline emission cache target or entrypoint identity is incomplete")
    return {
        "schema": "baseline_emission_cache_identity_v1",
        "baseline_sha256": baseline_sha256,
        "capsule_sha256": capsule_sha256,
        "source_sha256": source_sha256,
        "target": target,
        "compiler_dependencies_sha256": compiler_dependencies_sha256,
        "compiler_api_schema": schema_identity,
        "entrypoints": list(entrypoints),
    }


def _baseline_emission_cache_root(binding: Mapping[str, Any]) -> Path:
    root_value = binding.get("root")
    dependency_sha = binding.get("compiler_dependencies_sha256")
    if not isinstance(root_value, str) or not _is_sha256(dependency_sha):
        raise StageGateError("baseline emission cache binding is incomplete")
    root = Path(root_value)
    if not root.is_absolute() or str(root.resolve()) != str(root):
        raise StageGateError("baseline emission cache root must be an absolute resolved path")
    if root.exists() and (root.is_symlink() or not root.is_dir()):
        raise StageGateError("baseline emission cache root is not a real directory")
    return root


def load_baseline_emission_cache(binding: Mapping[str, Any], identity: Mapping[str, Any]) -> dict[str, Any] | None:
    """Load exact cached emitted bytes, refusing any present but inconsistent entry."""
    root = _baseline_emission_cache_root(binding)
    key = CONTRACTS.document_sha256(identity)
    entry = root / key
    if not entry.exists():
        return None
    if entry.is_symlink() or not entry.is_dir():
        raise StageGateError("baseline emission cache entry is not a real directory")
    paths = {
        name: entry / filename
        for name, filename in {
            "receipt": "receipt.json",
            "lowered": "lowered.mlir",
            "command_buffer": "command_buffer.json",
        }.items()
    }
    if any(path.is_symlink() or not path.is_file() for path in paths.values()):
        raise StageGateError("baseline emission cache entry is incomplete or linked")
    try:
        receipt = json.loads(paths["receipt"].read_text(encoding="utf-8"))
        lowered = paths["lowered"].read_text(encoding="utf-8")
        command_buffer = paths["command_buffer"].read_text(encoding="utf-8")
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise StageGateError(f"baseline emission cache entry is unreadable: {exc}") from exc
    observed_analysis_status = receipt.get("observed_analysis_status")
    if (
        receipt.get("schema") != "baseline_emission_cache_entry_v1"
        or receipt.get("key") != key
        or receipt.get("identity") != dict(identity)
        or receipt.get("lowered_sha256") != _sha256(lowered.encode("utf-8"))
        or receipt.get("command_buffer_sha256") != _sha256(command_buffer.encode("utf-8"))
        or not isinstance(receipt.get("emission_wall_seconds"), (int, float))
        or isinstance(receipt.get("emission_wall_seconds"), bool)
        or not math.isfinite(receipt["emission_wall_seconds"])
        or receipt["emission_wall_seconds"] < 0
        or observed_analysis_status not in (None, "completed", "timeout")
    ):
        raise StageGateError("baseline emission cache identity or artifact digest changed")
    observed_analysis = receipt.get("observed_analysis_wall_seconds")
    if observed_analysis is not None and (
        isinstance(observed_analysis, bool)
        or not isinstance(observed_analysis, (int, float))
        or not math.isfinite(observed_analysis)
        or observed_analysis < 0
    ):
        raise StageGateError("baseline emission cache analysis-cost observation is malformed")
    return {
        "identity": dict(identity),
        "key": key,
        "lowered_text": lowered,
        "command_buffer_text": command_buffer,
        "lowered_sha256": receipt["lowered_sha256"],
        "command_buffer_sha256": receipt["command_buffer_sha256"],
        "emission_wall_seconds": float(receipt["emission_wall_seconds"]),
        "observed_analysis_wall_seconds": (float(observed_analysis) if observed_analysis is not None else None),
        "observed_analysis_status": observed_analysis_status,
        "source": "host_persistent_exact_emission_cache",
    }


def baseline_emission_cache_observation(
    binding: Mapping[str, Any], identity: Mapping[str, Any]
) -> dict[str, Any] | None:
    """Read bounded allocation metadata; emitted bytes are rehashed before actual reuse."""
    root = _baseline_emission_cache_root(binding)
    key = CONTRACTS.document_sha256(identity)
    entry = root / key
    if not entry.exists():
        return None
    receipt_path = entry / "receipt.json"
    artifact_paths = (entry / "lowered.mlir", entry / "command_buffer.json")
    if (
        entry.is_symlink()
        or not entry.is_dir()
        or receipt_path.is_symlink()
        or not receipt_path.is_file()
        or any(path.is_symlink() or not path.is_file() for path in artifact_paths)
    ):
        raise StageGateError("baseline emission cache observation is incomplete or linked")
    receipt = CONTRACTS.mapping_file(receipt_path)
    emission = receipt.get("emission_wall_seconds")
    analysis = receipt.get("observed_analysis_wall_seconds")
    analysis_status = receipt.get("observed_analysis_status")
    if (
        receipt.get("schema") != "baseline_emission_cache_entry_v1"
        or receipt.get("key") != key
        or receipt.get("identity") != dict(identity)
        or not _is_sha256(receipt.get("lowered_sha256"))
        or not _is_sha256(receipt.get("command_buffer_sha256"))
        or isinstance(emission, bool)
        or not isinstance(emission, (int, float))
        or not math.isfinite(emission)
        or emission < 0
        or analysis_status not in (None, "completed", "timeout")
        or (
            analysis is not None
            and (
                isinstance(analysis, bool)
                or not isinstance(analysis, (int, float))
                or not math.isfinite(analysis)
                or analysis < 0
            )
        )
    ):
        raise StageGateError("baseline emission cache observation identity changed")
    return {
        "key": key,
        "identity": dict(identity),
        "emission_wall_seconds": float(emission),
        "observed_analysis_wall_seconds": (float(analysis) if analysis is not None else None),
        "observed_analysis_status": analysis_status,
    }


def store_baseline_emission_cache(
    binding: Mapping[str, Any],
    identity: Mapping[str, Any],
    *,
    lowered_text: str,
    command_buffer_text: str,
    emission_wall_seconds: float,
    observed_analysis_wall_seconds: float | None = None,
    observed_analysis_status: str | None = None,
) -> dict[str, Any]:
    """Atomically retain one exact baseline emission for future launches."""
    if (
        not isinstance(emission_wall_seconds, (int, float))
        or isinstance(emission_wall_seconds, bool)
        or not math.isfinite(emission_wall_seconds)
        or emission_wall_seconds < 0
    ):
        raise ValueError("baseline emission wall time must be finite and nonnegative")
    if observed_analysis_wall_seconds is not None and (
        isinstance(observed_analysis_wall_seconds, bool)
        or not isinstance(observed_analysis_wall_seconds, (int, float))
        or not math.isfinite(observed_analysis_wall_seconds)
        or observed_analysis_wall_seconds < 0
    ):
        raise ValueError("baseline analysis wall time must be finite and nonnegative")
    if observed_analysis_status not in (None, "completed", "timeout"):
        raise ValueError("baseline analysis status is invalid")
    if (observed_analysis_wall_seconds is None) != (observed_analysis_status is None):
        raise ValueError("baseline analysis observation requires both wall time and status")
    root = _baseline_emission_cache_root(binding)
    root.mkdir(parents=True, exist_ok=True)
    if root.is_symlink() or not root.is_dir():
        raise StageGateError("baseline emission cache root changed during creation")
    key = CONTRACTS.document_sha256(identity)
    existing = load_baseline_emission_cache(binding, identity)
    if existing is not None:
        if existing["lowered_sha256"] != _sha256(lowered_text.encode("utf-8")) or existing[
            "command_buffer_sha256"
        ] != _sha256(command_buffer_text.encode("utf-8")):
            raise StageGateError("same baseline emission cache identity produced different bytes")
        return existing
    temporary = Path(tempfile.mkdtemp(prefix=f".{key}.", dir=root))
    try:
        lowered_sha = _sha256(lowered_text.encode("utf-8"))
        buffer_sha = _sha256(command_buffer_text.encode("utf-8"))
        (temporary / "lowered.mlir").write_text(lowered_text, encoding="utf-8")
        (temporary / "command_buffer.json").write_text(command_buffer_text, encoding="utf-8")
        _write_json(
            temporary / "receipt.json",
            {
                "schema": "baseline_emission_cache_entry_v1",
                "key": key,
                "identity": dict(identity),
                "lowered_sha256": lowered_sha,
                "command_buffer_sha256": buffer_sha,
                "emission_wall_seconds": float(emission_wall_seconds),
                "observed_analysis_wall_seconds": (
                    float(observed_analysis_wall_seconds) if observed_analysis_wall_seconds is not None else None
                ),
                "observed_analysis_status": observed_analysis_status,
                "scope": "compiler emission only; host verification is rerun under the current policy",
            },
        )
        for path in temporary.iterdir():
            path.chmod(0o444)
        temporary.chmod(0o555)
        try:
            temporary.rename(root / key)
        except OSError:
            # POSIX may report EEXIST or ENOTEMPTY when another writer won the
            # atomic directory rename.  Only accept that race after reloading
            # and comparing the complete exact entry.
            cached = load_baseline_emission_cache(binding, identity)
            if cached is None or (
                cached["lowered_sha256"] != lowered_sha or cached["command_buffer_sha256"] != buffer_sha
            ):
                raise StageGateError("concurrent baseline emission cache entry disagrees")
            return cached
        return load_baseline_emission_cache(binding, identity) or {}
    finally:
        if temporary.exists():
            temporary.chmod(0o700)
            shutil.rmtree(temporary)
