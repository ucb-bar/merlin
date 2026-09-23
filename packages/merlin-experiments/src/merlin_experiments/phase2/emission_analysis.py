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

from merlin.benchharness import hash_tree
from merlin.common.digest import sha256_bytes as _sha256
from merlin.perf.agent_guidance import guidance_for_emission_analysis, inspect_compiler_package
from merlin.perf.execution_policy import FULL_GRAPH_STATIC_ANALYSIS_MAX_SECONDS, ITERATION_MAX_SECONDS

from . import contracts as CONTRACTS
from . import emission_diagnostics as ED
from . import stage_inputs as INPUTS
from . import static_identity as SI
from .broker_evidence import _is_sha256
from .contracts import StageGateError
from .contracts import canonical_json as _canonical_json
from .contracts import write_json as _write_json

if TYPE_CHECKING:
    from merlin.targetgen import oot_runner as OR


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
    from merlin.targetgen import oot_runner as OR

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


def analyze_whole_model_emission(
    baseline: Path,
    candidate: Path,
    sentinel: INPUTS.StageE2ESentinel,
    *,
    contract_root: Path,
    timeout_s: int,
    peak_macs_per_cycle: int | None,
    achievable_macs_per_cycle: float | None,
    target: str,
    global_plan_verifier: Callable[..., Mapping[str, Any]] | None = None,
    artifact_sink: Callable[[Mapping[str, Any]], None] | None = None,
    baseline_artifacts: Mapping[str, Any] | None = None,
    emit_pair_runner: Callable[..., tuple[int, str, str]] | None = None,
    machine_artifact_auditor: Callable[..., Mapping[str, Any]] | None = None,
    machine_build_policy_identity: Mapping[str, Any] | None = None,
    host_verifier_policy_sha256: str | None = None,
    compiler_api_schema: Mapping[str, Any] | None = None,
    baseline_emission_cache: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Emit and compare the fixed complete-model sentinel without running a simulator.

    The host owns both compiler invocations, so the candidate cannot substitute a capsule or mutate
    the frozen baseline.  This is a fast global structural screen; warm measured cycles remain the
    only timing verdict.
    """
    from merlin.perf.artifact_activity import analyze_artifact_activity  # noqa: PLC0415
    from merlin.perf.model_placement import (  # noqa: PLC0415
        captured_global_graph,
        contraction_placement,
        prepare_captured_source,
    )
    from merlin.targetgen import oot_runner as OR  # noqa: PLC0415
    from merlin.targetgen import trace_check as TCK  # noqa: PLC0415
    from merlin.targetgen.rocc import decode as RD  # noqa: PLC0415

    analysis_started = time.monotonic()
    run_pair = emit_pair_runner or emit_pair
    candidate_before = hash_tree(Path(candidate))["sha256"]
    source = Path(sentinel.frozen_source_path)
    if source.is_symlink() or not source.is_dir():
        raise StageGateError("frozen whole-model sentinel is absent or linked")
    if CONTRACTS.exact_tree_record(source)["sha256"] != sentinel.capsule_sha256:
        raise StageGateError("frozen whole-model sentinel bytes changed")
    descriptor = CONTRACTS.mapping_file(source / "capsule.yaml", yaml_file=True)
    interface = source / str(descriptor.get("interface_mlir") or "capsule.interface.mlir")
    if interface.is_symlink() or not interface.is_file():
        raise StageGateError("frozen whole-model sentinel has no real interface MLIR")
    source_text = interface.read_text(encoding="utf-8")
    baseline_sha256 = hash_tree(Path(baseline))["sha256"]
    identical_compilers = candidate_before == baseline_sha256
    baseline_package = OR.load_package(Path(baseline))
    candidate_package = OR.load_package(Path(candidate))
    baseline_cache_identity = None
    cached_baseline_emission = None
    if baseline_emission_cache is not None and baseline_artifacts is None:
        baseline_cache_identity = baseline_emission_cache_identity(
            baseline_sha256=baseline_sha256,
            capsule_sha256=sentinel.capsule_sha256,
            source_sha256=_sha256(source_text.encode("utf-8")),
            target=target,
            compiler_dependencies_sha256=str(baseline_emission_cache.get("compiler_dependencies_sha256", "")),
            compiler_api_schema=compiler_api_schema,
            entrypoints=OR.analysis_emission_entrypoints(baseline_package),
        )
        cached_baseline_emission = load_baseline_emission_cache(baseline_emission_cache, baseline_cache_identity)
    # Divide the bounded analysis budget by the subprocesses we will actually launch.  An
    # optional one-pass bundle counts once; a legacy pair counts twice; a retained baseline
    # counts zero; and an exact candidate/optimization-baseline seed reuses the baseline arm.
    baseline_entrypoints = (
        0
        if baseline_artifacts is not None or cached_baseline_emission is not None
        else len(OR.analysis_emission_entrypoints(baseline_package))
    )
    candidate_entrypoints = 0 if identical_compilers else len(OR.analysis_emission_entrypoints(candidate_package))
    emitted_entrypoints = baseline_entrypoints + candidate_entrypoints
    # This path compiles and statically audits a complete graph; it does not execute it.  Its host
    # deadline is intentionally distinct from the ten-minute reduced-witness simulator ceiling.
    analysis_budget = min(int(FULL_GRAPH_STATIC_ANALYSIS_MAX_SECONDS), int(timeout_s))
    per_entrypoint_timeout = max(1, analysis_budget // max(1, emitted_entrypoints))

    def require_not_declined(payload: str, arm: str) -> None:
        # Some compiler entrypoints return success while emitting a structured
        # decline and an empty function. That is not a valid comparison arm.
        buffer = json.loads(payload)
        if not isinstance(buffer, Mapping):
            raise StageGateError(f"whole-model {arm} command buffer is not an object")
        if buffer.get("declined") is not None:
            raise StageGateError(
                f"whole-model {arm} lowering declined; no structural or cycle comparison is admissible: "
                + json.dumps(buffer["declined"], sort_keys=True)[:2000]
            )
        if compiler_api_schema is not None:
            ED.validate_whole_program_schema(buffer, compiler_api_schema, arm=arm)

    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR") or None) as raw:
        scratch = Path(raw)
        baseline_identity = {
            "baseline_sha256": baseline_sha256,
            "capsule_sha256": sentinel.capsule_sha256,
            "target": target,
        }
        if baseline_artifacts is not None:
            if (
                baseline_artifacts.get("identity") != baseline_identity
                or _sha256(baseline_artifacts["lowered_text"].encode("utf-8"))
                != baseline_artifacts.get("lowered_sha256")
                or _sha256(baseline_artifacts["command_buffer_text"].encode("utf-8"))
                != baseline_artifacts.get("command_buffer_sha256")
            ):
                raise StageGateError("retained frozen baseline artifact identity changed")
            base_rc = 0
            base_llvm, base_buffer = (baseline_artifacts["lowered_text"], baseline_artifacts["command_buffer_text"])
            baseline_emission_source = "retained_verified_baseline_artifacts"
            baseline_emission_wall_seconds = None
        elif cached_baseline_emission is not None:
            base_rc = 0
            base_llvm = cached_baseline_emission["lowered_text"]
            base_buffer = cached_baseline_emission["command_buffer_text"]
            baseline_emission_source = cached_baseline_emission["source"]
            baseline_emission_wall_seconds = cached_baseline_emission["emission_wall_seconds"]
        else:
            emission_started = time.monotonic()
            base_rc, base_llvm, base_buffer = run_pair(
                baseline_package, interface, scratch, "baseline", per_entrypoint_timeout
            )
            baseline_emission_wall_seconds = time.monotonic() - emission_started
            baseline_emission_source = "compiler_executed"
        if base_rc != 0 or not base_buffer:
            detail_path = scratch / "emission_baseline.json"
            details = json.loads(detail_path.read_text(encoding="utf-8")) if detail_path.is_file() else {}
            raise StageGateError(
                "whole-model baseline emission failed; candidate was not invoked "
                f"(baseline_rc={base_rc}, baseline_buffer={bool(base_buffer)}): " + json.dumps(details, sort_keys=True)
            )
        if base_rc == 0 and base_buffer:
            require_not_declined(base_buffer, "baseline")
            if (
                baseline_emission_cache is not None
                and baseline_cache_identity is not None
                and cached_baseline_emission is None
                and baseline_artifacts is None
            ):
                cached_baseline_emission = store_baseline_emission_cache(
                    baseline_emission_cache,
                    baseline_cache_identity,
                    lowered_text=base_llvm,
                    command_buffer_text=base_buffer,
                    emission_wall_seconds=float(baseline_emission_wall_seconds),
                )
        baseline_plan_binding = {
            "schema": "baseline_global_plan_evidence_binding_v1",
            "source_sha256": _sha256(source_text.encode("utf-8")),
            "lowered_sha256": _sha256(base_llvm.encode("utf-8")),
            "command_buffer_sha256": _sha256(base_buffer.encode("utf-8")),
            "compiler_sha256": baseline_identity["baseline_sha256"],
            "host_verifier_policy_sha256": host_verifier_policy_sha256,
        }
        baseline_plan_cached = None
        if (
            baseline_artifacts is not None
            and _is_sha256(host_verifier_policy_sha256)
            and (
                "verified_global_plan_emission" in baseline_artifacts
                or "global_plan_evidence_binding" in baseline_artifacts
            )
        ):
            proof = baseline_artifacts.get("verified_global_plan_emission")
            expected_binding = {**baseline_plan_binding, "evidence_sha256": CONTRACTS.document_sha256(proof)}
            if (
                not isinstance(proof, Mapping)
                or baseline_artifacts.get("global_plan_evidence_binding") != expected_binding
            ):
                raise StageGateError("retained baseline global-plan evidence binding changed")
            if proof.get("status") == "verified" and any(
                proof.get(key) != value
                for key, value in {
                    "source_sha256": baseline_plan_binding["source_sha256"],
                    "candidate_sha256": baseline_plan_binding["compiler_sha256"],
                    "candidate_lowered_sha256": baseline_plan_binding["lowered_sha256"],
                    "candidate_command_buffer_sha256": baseline_plan_binding["command_buffer_sha256"],
                }.items()
            ):
                raise StageGateError("retained baseline verified plan contradicts its artifact binding")
            baseline_plan_cached = copy.deepcopy(dict(proof))
        if identical_compilers:
            cand_rc, cand_llvm, cand_buffer = base_rc, base_llvm, base_buffer
        else:
            cand_rc, cand_llvm, cand_buffer = run_pair(
                candidate_package, interface, scratch, "candidate", per_entrypoint_timeout
            )
        if cand_rc == 0 and cand_buffer:
            require_not_declined(cand_buffer, "candidate")
        if base_rc != 0 or cand_rc != 0 or not base_buffer or not cand_buffer:
            failures = {}
            for tag in ("baseline", "candidate"):
                detail_path = scratch / f"emission_{tag}.json"
                if detail_path.is_file():
                    failures[tag] = json.loads(detail_path.read_text(encoding="utf-8"))
            raise StageGateError(
                "whole-model emission failed "
                f"(baseline_rc={base_rc}, candidate_rc={cand_rc}, "
                f"baseline_buffer={bool(base_buffer)}, candidate_buffer={bool(cand_buffer)}): "
                + json.dumps(failures, sort_keys=True)
            )
        baseline_json = scratch / "whole_baseline.json"
        candidate_json = scratch / "whole_candidate.json"
        baseline_json.write_text(base_buffer, encoding="utf-8")
        candidate_json.write_text(cand_buffer, encoding="utf-8")
        diagnostics = ED.analyze_command_buffers(
            baseline_json,
            candidate_json,
            peak_macs_per_cycle=peak_macs_per_cycle,
            achievable_macs_per_cycle=achievable_macs_per_cycle,
            target=target,
        )
        diagnostics["emission_execution"] = {
            "schema": "whole_model_emission_execution_v1",
            "identical_compiler_trees": identical_compilers,
            "retained_baseline_reused": baseline_artifacts is not None,
            "candidate_reused_baseline_artifacts": identical_compilers,
            "launched_entrypoint_count": emitted_entrypoints,
            "baseline_entrypoints": baseline_entrypoints,
            "candidate_entrypoints": candidate_entrypoints,
            "per_entrypoint_timeout_seconds": per_entrypoint_timeout,
            "analysis_budget_seconds": analysis_budget,
            "baseline_emission_source": baseline_emission_source,
            "baseline_emission_cache_key": (
                cached_baseline_emission.get("key") if cached_baseline_emission is not None else None
            ),
            "baseline_emission_measured_wall_seconds": baseline_emission_wall_seconds,
        }
        baseline_buffer = json.loads(base_buffer)
        candidate_buffer = json.loads(cand_buffer)
        expected = descriptor.get("expected") or {}
        cand_lowered_module = None
        base_lowered_module = None
        try:
            if identical_compilers:
                # Identical bytes imply identical IR and instruction semantics. Parse the exact
                # artifact once, and either decode it once or reuse the already-bound baseline
                # trace. Shallow arm rebinding keeps the large instruction vector shared in memory.
                base_lowered_module = RD._parse_module(base_llvm)
                cand_lowered_module = base_lowered_module
                if baseline_artifacts is not None:
                    shared_trace = baseline_artifacts["decoded_trace"]
                else:
                    shared_trace = (
                        RD.decode_module(base_lowered_module, source="immutable_optimization_baseline", target=target)
                        if base_lowered_module is not None
                        else RD._decode_by_text_scan(base_llvm, source="immutable_optimization_baseline", target=target)
                    )
                base_trace = (
                    {**shared_trace, "source": "immutable_optimization_baseline"}
                    if isinstance(shared_trace, Mapping)
                    else shared_trace
                )
                cand_trace = (
                    {**shared_trace, "source": "live_phase2_candidate"}
                    if isinstance(shared_trace, Mapping)
                    else shared_trace
                )
            else:
                if baseline_artifacts is not None:
                    base_trace = baseline_artifacts["decoded_trace"]
                else:
                    base_lowered_module = RD._parse_module(base_llvm)
                    base_trace = (
                        RD.decode_module(base_lowered_module, source="immutable_optimization_baseline", target=target)
                        if base_lowered_module is not None
                        else RD._decode_by_text_scan(base_llvm, source="immutable_optimization_baseline", target=target)
                    )
                # The complete candidate module can be large. Parse its exact bytes once and share
                # the host-owned in-memory IR with instruction decoding and global-plan verification.
                cand_lowered_module = RD._parse_module(cand_llvm)
                cand_trace = (
                    RD.decode_module(cand_lowered_module, source="live_phase2_candidate", target=target)
                    if cand_lowered_module is not None
                    else RD._decode_by_text_scan(cand_llvm, source="live_phase2_candidate", target=target)
                )
            base_has_stream = bool(base_trace.get("instructions"))
            cand_has_stream = bool(cand_trace.get("instructions"))
            if not cand_has_stream:
                trace_conformance = {
                    "status": "UNKNOWN",
                    "reason": (
                        "the candidate lowered artifact contains no target instruction "
                        "stream; absence cannot prove encoding, residency, or dispatch"
                    ),
                    "baseline_has_target_stream": base_has_stream,
                    "candidate_has_target_stream": False,
                    "introduced_candidate_findings": (
                        ["conformance: candidate removed the target instruction stream"] if base_has_stream else []
                    ),
                }
            else:
                base_check = TCK.check(base_trace, expected, baseline_buffer)
                cand_check = TCK.check(cand_trace, expected, candidate_buffer)
                base_residency = TCK.residency_findings(base_trace)
                cand_residency = TCK.residency_findings(cand_trace)
                movement_bound = TCK.movement_bound_for(target)
                base_movement = TCK.movement_findings(base_trace, movement_bound)
                cand_movement = TCK.movement_findings(cand_trace, movement_bound)
                base_findings = {f"conformance: {value}" for value in base_check["violations"]}
                base_findings.update(f"residency: {value}" for value in base_residency)
                base_findings.update(f"movement: {value}" for value in base_movement)
                cand_findings = {f"conformance: {value}" for value in cand_check["violations"]}
                cand_findings.update(f"residency: {value}" for value in cand_residency)
                cand_findings.update(f"movement: {value}" for value in cand_movement)
                trace_conformance = {
                    "status": "checked",
                    "baseline": {
                        "advisory_status": base_check["status"],
                        "finding_count": len(base_findings),
                        "residency_reload_findings": base_residency,
                        "movement_width_findings": base_movement,
                        "drives_accelerator": bool(TCK.drives_accelerator(base_trace)),
                    },
                    "candidate": {
                        "advisory_status": cand_check["status"],
                        "finding_count": len(cand_findings),
                        "residency_reload_findings": cand_residency,
                        "movement_width_findings": cand_movement,
                        "drives_accelerator": bool(TCK.drives_accelerator(cand_trace)),
                    },
                    "introduced_candidate_findings": sorted(cand_findings - base_findings),
                    "comparison_policy": (
                        "differential against the immutable optimization comparison baseline; advisory trace "
                        "findings are not a numeric-correctness verdict"
                    ),
                }
        except Exception as exc:  # noqa: BLE001 - absent decode evidence is explicit, never clean
            base_trace = cand_trace = None
            trace_conformance = {
                "status": "UNKNOWN",
                "reason": f"target trace diagnosis failed: {type(exc).__name__}: {str(exc)[:200]}",
                "introduced_candidate_findings": [],
            }

        def _artifact(trace: Mapping[str, Any] | None, arm: str) -> dict[str, Any]:
            if trace is None:
                return {"status": "UNKNOWN", "arm": arm, "reason": "the lowered target trace was unavailable"}
            try:
                return analyze_artifact_activity(
                    trace, target=target, op=str((descriptor.get("operation") or {}).get("op") or "model")
                )
            except Exception as exc:  # noqa: BLE001 - an unreadable semantic map is UNKNOWN
                return {
                    "status": "UNKNOWN",
                    "arm": arm,
                    "reason": (
                        f"emitted artifact activity could not be lifted: {type(exc).__name__}: {str(exc)[:200]}"
                    ),
                }

        base_activity = _artifact(base_trace, "baseline")
        cand_activity = _artifact(cand_trace, "candidate")
        issued_delta: dict[str, int | float] = {}
        base_issued = base_activity.get("issued")
        cand_issued = cand_activity.get("issued")
        if isinstance(base_issued, Mapping) and isinstance(cand_issued, Mapping):
            for key in sorted(set(base_issued).intersection(cand_issued)):
                left, right = base_issued[key], cand_issued[key]
                if (
                    isinstance(left, (int, float))
                    and not isinstance(left, bool)
                    and isinstance(right, (int, float))
                    and not isinstance(right, bool)
                ):
                    issued_delta[key] = right - left
        diagnostics["target_artifact_activity"] = {
            "baseline": base_activity,
            "candidate": cand_activity,
            "candidate_minus_baseline_issued": issued_delta,
        }
        diagnostics["trace_conformance"] = trace_conformance

        prepared_source_analysis = None
        prepared_source_failure = None
        try:
            # Placement, graph capture and plan verification all consume the same immutable source.
            # Parse and outline it once; each consumer still performs its own semantic checks.
            prepared_source_analysis = prepare_captured_source(interface)
        except Exception as exc:  # Existing per-audit fallbacks below retain fail-closed evidence.
            prepared_source_failure = exc

        def _placement(buffer: Mapping[str, Any]) -> dict[str, Any]:
            params = buffer.get("params")
            params = params if isinstance(params, Mapping) else {}
            rows = params.get("lane_placement")
            if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
                return {"status": "UNKNOWN", "reason": "compiler emitted no region-to-lane map"}
            try:
                return contraction_placement(
                    interface,
                    rows,
                    target=target,
                    entry=descriptor.get("entry"),
                    prepared_source=prepared_source_analysis,
                )
            except Exception as exc:  # noqa: BLE001 - incomplete placement is explicit evidence
                return {
                    "status": "UNKNOWN",
                    "reason": (f"MAC-weighted placement could not be derived: {type(exc).__name__}: {str(exc)[:200]}"),
                }

        diagnostics["model_contraction_placement"] = {
            "baseline": _placement(baseline_buffer),
            "candidate": _placement(candidate_buffer),
            "comparison_basis": (
                "exact captured contraction MACs by compiler-declared lane; non-contraction cycles remain unpriced"
            ),
        }
        try:
            if prepared_source_analysis is None and prepared_source_failure is not None:
                raise prepared_source_failure
            diagnostics["captured_logical_graph"] = captured_global_graph(
                interface, prepared_source=prepared_source_analysis
            )
        except Exception as exc:  # noqa: BLE001 - report missing graph coverage explicitly
            diagnostics["captured_logical_graph"] = {
                "status": "UNKNOWN",
                "reason": f"whole-graph outlining failed: {type(exc).__name__}: {str(exc)[:200]}",
            }
        if baseline_plan_cached is not None:
            baseline_plan = baseline_plan_cached
        else:
            try:
                if global_plan_verifier is not None:
                    baseline_plan = dict(
                        global_plan_verifier(
                            candidate=Path(baseline),
                            interface=interface,
                            lowered_text=base_llvm,
                            command_buffer=baseline_buffer,
                            logical_graph=diagnostics["captured_logical_graph"],
                            candidate_sha256=baseline_identity["baseline_sha256"],
                        )
                    )
                else:
                    from merlin.perf.compiler_plan_evidence import verify_compiler_global_plan

                    baseline_plan = verify_compiler_global_plan(
                        source_text=source_text,
                        lowered_text=base_llvm,
                        command_buffer=baseline_buffer,
                        candidate_sha256=baseline_identity["baseline_sha256"],
                        command_buffer_sha256=baseline_plan_binding["command_buffer_sha256"],
                        parsed_lowered_module=base_lowered_module,
                        prepared_source_analysis=prepared_source_analysis,
                    )
            except Exception as exc:
                baseline_plan = {
                    "status": "UNKNOWN",
                    "reason": f"host baseline global-plan verifier failed: {type(exc).__name__}: {exc}",
                }
        # Bind the actual JSON document retained across the worker boundary. Verifier
        # task maps have integer keys; JSON turns them into strings, whose canonical
        # order differs for task 10 versus task 2. Hashing pre-transport Python maps
        # made an unchanged multi-task proof fail its own next-iteration binding.
        baseline_plan = json.loads(_canonical_json(baseline_plan))
        baseline_plan_binding["evidence_sha256"] = CONTRACTS.document_sha256(baseline_plan)
        diagnostics["verified_baseline_global_plan_emission"] = baseline_plan
        diagnostics["baseline_global_plan_evidence_binding"] = baseline_plan_binding
        if identical_compilers:
            # The baseline proof is already bound to these exact compiler, source, lowered,
            # command-buffer and graph digests. Reusing it avoids a second whole-module walk while
            # preserving every candidate readiness binding (the compiler digest is identical).
            diagnostics["verified_global_plan_emission"] = copy.deepcopy(baseline_plan)
        else:
            try:
                if global_plan_verifier is not None:
                    diagnostics["verified_global_plan_emission"] = dict(
                        global_plan_verifier(
                            candidate=Path(candidate),
                            interface=interface,
                            lowered_text=cand_llvm,
                            command_buffer=candidate_buffer,
                            logical_graph=diagnostics["captured_logical_graph"],
                            candidate_sha256=candidate_before,
                        )
                    )
                else:
                    from merlin.perf.compiler_plan_evidence import verify_compiler_global_plan

                    diagnostics["verified_global_plan_emission"] = verify_compiler_global_plan(
                        source_text=source_text,
                        lowered_text=cand_llvm,
                        command_buffer=candidate_buffer,
                        candidate_sha256=candidate_before,
                        command_buffer_sha256=_sha256(cand_buffer.encode("utf-8")),
                        parsed_lowered_module=cand_lowered_module,
                        prepared_source_analysis=prepared_source_analysis,
                    )
            except Exception as exc:  # noqa: BLE001 - incomplete host verification is explicit
                diagnostics["verified_global_plan_emission"] = {
                    "status": "UNKNOWN",
                    "reason": f"host global-plan verifier failed: {type(exc).__name__}: {exc}",
                }
        # The first command-buffer pass intentionally treats representation declarations as UNKNOWN.
        # Only now, after the host has inspected and identity-bound the lowered artifact, may its
        # exact materialized transition count and typed load/store bytes replace that unknown.
        from merlin.perf.command_buffer_diagnostics import representation_activity  # noqa: PLC0415

        for arm, buffer, proof in (
            ("baseline", baseline_buffer, baseline_plan),
            ("candidate", candidate_buffer, diagnostics["verified_global_plan_emission"]),
        ):
            row = diagnostics["arms"].get(arm)
            if isinstance(row, dict):
                row["representation_activity"] = representation_activity(
                    buffer,
                    physical_transition_evidence=(
                        proof.get("physical_transition_evidence") if isinstance(proof, Mapping) else None
                    ),
                )

        # Cache source-owned STATIC instruction presence while parsed IR is already
        # available. Reduced-source actions must not reparse the complete model.
        def task_instructions(text, raw_buffer, buffer, module, trace, proof, *, baseline_arm=False):
            try:
                from merlin.perf.task_instruction_evidence import (
                    digest,
                    summarize_task_instructions,
                    target_instruction_facts,
                    task_instruction_binding,
                )

                facts = target_instruction_facts(target)
                binding_args = dict(
                    source_text=source_text,
                    lowered_text=text,
                    command_buffer_text=raw_buffer,
                    verified_plan=proof,
                    target_facts=facts,
                    host_policy_sha256=host_verifier_policy_sha256,
                )
                binding = task_instruction_binding(**binding_args)
                cached = (baseline_artifacts or {}).get("task_instruction_evidence") if baseline_arm else None
                # A CACHE WRITTEN BEFORE THE DECLARED SET EXISTED IS NOT A HIT. Its binding and its
                # digest both still agree with themselves, so it would be reused forever and the
                # baseline arm would silently carry no declared instruction set while the candidate
                # arm carried one -- a difference between arms that came from the cache, not the
                # program. Recompute instead; the binding check stays exactly as strict.
                if (
                    isinstance(cached, Mapping)
                    and cached.get("binding") == binding
                    and cached.get("declared_instruction_set")
                    and (baseline_artifacts or {}).get("task_instruction_evidence_sha256") == digest(cached)
                ):
                    return copy.deepcopy(dict(cached))
                return summarize_task_instructions(
                    **binding_args,
                    command_buffer=buffer,
                    parsed_module=module,
                    decoded_trace=trace,
                    decode_module=lambda parsed: RD.decode_module(parsed, target=target),
                )
            except Exception as exc:
                return {
                    "status": "UNKNOWN",
                    "route_correspondence": "UNKNOWN",
                    "timing_calibration_admissible": False,
                    "reason": f"static task instruction ownership unavailable: {type(exc).__name__}: {str(exc)[:200]}",
                }

        baseline_task_instructions = task_instructions(
            base_llvm, base_buffer, baseline_buffer, base_lowered_module, base_trace, baseline_plan, baseline_arm=True
        )
        candidate_task_instructions = (
            copy.deepcopy(baseline_task_instructions)
            if identical_compilers
            else task_instructions(
                cand_llvm,
                cand_buffer,
                candidate_buffer,
                cand_lowered_module,
                cand_trace,
                diagnostics["verified_global_plan_emission"],
            )
        )
        diagnostics["task_instruction_evidence"] = {
            "baseline": baseline_task_instructions,
            "candidate": candidate_task_instructions,
        }
        try:
            from merlin.perf.context_probe import extract_queued_movement_context

            verified = diagnostics["verified_global_plan_emission"].get("status") == "verified"
            diagnostics["queued_movement_context"] = extract_queued_movement_context(
                cand_trace,
                target=target,
                artifact_sha256=_sha256(cand_llvm.encode("utf-8")),
                artifact_text=cand_llvm,
                command_buffer=candidate_buffer if verified else None,
                parsed_module=cand_lowered_module,
                max_commands=32,
                max_motifs=4,
            )
        except Exception as exc:
            diagnostics["queued_movement_context"] = {
                "status": "UNKNOWN",
                "calibration_admissible": False,
                "reason": f"queued-context extraction unavailable: {type(exc).__name__}: {str(exc)[:200]}",
            }

        def machine_activity(text: str, arm: str) -> dict[str, Any]:
            digest = _sha256(text.encode("utf-8"))
            cached = (baseline_artifacts or {}).get("machine_artifact_activity") if arm == "baseline" else None
            if (
                isinstance(cached, Mapping)
                and cached.get("source_sha256") == digest
                and machine_build_policy_identity is not None
                and cached.get("build_policy_identity") == machine_build_policy_identity
            ):
                return dict(cached)
            if machine_artifact_auditor is None:
                return {"status": "UNKNOWN", "reason": "answer-masked machine auditor unavailable"}
            try:
                remaining = min(60.0, timeout_s - (time.monotonic() - analysis_started))
                if remaining <= 0:
                    raise TimeoutError("no remaining whole-model analysis budget")
                result = dict(machine_artifact_auditor(text, arm=arm, timeout_s=remaining))
                if result.get("source_sha256") != digest:
                    raise ValueError("machine audit source does not match the current emitted artifact")
                if (
                    machine_build_policy_identity is None
                    or result.get("build_policy_identity") != machine_build_policy_identity
                ):
                    raise ValueError("machine audit build policy changed during compilation")
                return result
            except Exception as exc:
                return {
                    "status": "UNKNOWN",
                    "source_sha256": digest,
                    "build_policy_identity": machine_build_policy_identity,
                    "failed_attempt_retained": True,
                    "reason": f"machine audit unavailable: {type(exc).__name__}: {str(exc)[:200]}",
                }

        baseline_machine_activity = machine_activity(base_llvm, "baseline")
        candidate_machine_activity = (
            copy.deepcopy(baseline_machine_activity)
            if identical_compilers
            else machine_activity(cand_llvm, "candidate")
        )
        diagnostics["machine_artifact_activity"] = {
            "baseline": baseline_machine_activity,
            "candidate": candidate_machine_activity,
        }
        # WHAT THE MACHINE OFFERS AND WHY IT WAS REFUSED. Both keys are read by
        # `agent_guidance.guidance_for_emission_analysis` below -- `declared_capability_unused` and
        # `declared_capability_refused` -- and until this line nothing in production wrote either,
        # so two complete findings never fired once and the agent's only view of the instruction set
        # was a count of what it had already emitted. They are set BEFORE the brief is built,
        # because the brief is what carries them to the agent.
        diagnostics["isa_capability_utilization"] = ED.isa_capability_utilization(cand_llvm, target=target)
        diagnostics["capability_refusals"] = ED.capability_refusals(
            prepared_source_analysis, target=target, datapath=ED.declared_capsule_datapath(descriptor)
        )
        # A CYCLE FLOOR UNDER THE STRUCTURAL DELTA. See `iteration_cost_plane`: the comparable
        # signal in the capsule grade never crossed into this loop.
        from merlin.common.provenance import load_artifacts
        from merlin.perf.gate_phase import STATUS_INCOMPLETE, configured_phase

        try:
            cost_phase = configured_phase("cost_plane", declaration=contract_root / "gate_phases.yaml")
        except Exception as exc:
            diagnostics["cost_plane"] = {
                "schema": "cost_plane_verdict_v1",
                "status": STATUS_INCOMPLETE,
                "reason": (
                    f"the cost plane could not be computed for this iteration: {type(exc).__name__}: {str(exc)[:200]}"
                ),
                "blocking": False,
                "admitted": False,
                "measured_cycles": None,
                "floor_cycles": None,
            }
        else:
            registry_error = None
            try:
                artifacts = load_artifacts(contract_root / "hardware_pins.yaml")
            except Exception as exc:
                artifacts = {}
                registry_error = {
                    "name": None,
                    "config": None,
                    "reason": (
                        "the device these design keys name could not be resolved: "
                        f"{type(exc).__name__}: {str(exc)[:200]}"
                    ),
                }
            diagnostics["cost_plane"] = ED.iteration_cost_plane(
                descriptor,
                target=target,
                arms=diagnostics.get("arms") or {},
                phase=cost_phase,
                artifacts=artifacts,
            )
            if (
                registry_error is not None
                and "device" in diagnostics["cost_plane"]
                and (diagnostics["cost_plane"].get("design") or {}).get("hw_config")
            ):
                diagnostics["cost_plane"]["device"] = registry_error
        emission_digests = {
            "baseline_lowered_sha256": _sha256(base_llvm.encode("utf-8")),
            "candidate_lowered_sha256": _sha256(cand_llvm.encode("utf-8")),
            "lowered_identical": base_llvm == cand_llvm,
            "baseline_command_buffer_sha256": _sha256(base_buffer.encode("utf-8")),
            "candidate_command_buffer_sha256": _sha256(cand_buffer.encode("utf-8")),
            "command_buffer_identical": base_buffer == cand_buffer,
        }
        from merlin.perf.candidate_comparison import decide_emitted_pair  # noqa: PLC0415

        diagnostics["candidate_decision"] = decide_emitted_pair(diagnostics, emission_digests)
        optimization_brief = guidance_for_emission_analysis(diagnostics, inspect_compiler_package(candidate))
        if artifact_sink is not None:
            artifact_sink(
                {
                    "lowered_text": cand_llvm,
                    "decoded_trace": cand_trace,
                    "parsed_lowered_module": cand_lowered_module,
                    "command_buffer": candidate_buffer,
                    "command_buffer_text": cand_buffer,
                    "interface": interface,
                    "candidate_sha256": candidate_before,
                    "candidate_lowered_sha256": _sha256(cand_llvm.encode("utf-8")),
                    "candidate_command_buffer_sha256": _sha256(cand_buffer.encode("utf-8")),
                    "task_instruction_evidence": diagnostics["task_instruction_evidence"]["candidate"],
                    "baseline_artifacts": {
                        "identity": baseline_identity,
                        "lowered_text": base_llvm,
                        "command_buffer_text": base_buffer,
                        "decoded_trace": base_trace,
                        "lowered_sha256": _sha256(base_llvm.encode("utf-8")),
                        "command_buffer_sha256": _sha256(base_buffer.encode("utf-8")),
                        "machine_artifact_activity": diagnostics["machine_artifact_activity"]["baseline"],
                        "verified_global_plan_emission": copy.deepcopy(baseline_plan),
                        "global_plan_evidence_binding": copy.deepcopy(baseline_plan_binding),
                        "task_instruction_evidence": diagnostics["task_instruction_evidence"]["baseline"],
                        "task_instruction_evidence_sha256": CONTRACTS.document_sha256(
                            diagnostics["task_instruction_evidence"]["baseline"]
                        ),
                    },
                }
            )
    candidate_after = hash_tree(Path(candidate))["sha256"]
    if candidate_before != candidate_after:
        raise StageGateError("candidate bytes changed during whole-model analysis")
    document = {
        "schema": "host_owned_whole_model_emission_analysis_v2",
        "candidate_sha256": candidate_after,
        "workload": {
            "capsule": sentinel.capsule,
            "capsule_sha256": sentinel.capsule_sha256,
            "required_lanes": list(sentinel.required_lanes),
            "required_tiers": list(sentinel.required_tiers),
        },
        "emission": emission_digests,
        "candidate_decision": diagnostics["candidate_decision"],
        "diagnostics": diagnostics,
        "optimization_brief": optimization_brief,
        "timing_status": "UNMEASURED",
        "next_measurement": {
            "scope": "separate_mechanism_equivalent_probe",
            "warmup_runs": 1,
            "measured_runs": 1,
            "primary_metric": "total_compute_cycles",
            "maximum_simulator_seconds": int(ITERATION_MAX_SECONDS),
        },
    }
    document["iteration_readiness"] = global_iteration_readiness(document)
    return document


def global_iteration_readiness(document: Mapping[str, Any]) -> dict[str, Any]:
    """Expose exactly which full-graph evidence is missing before a search measurement.

    This consumes host analysis, never candidate-provided readiness booleans. A parsed input graph
    is insufficient: the candidate must have a checked plan-to-emission receipt for that graph.
    The target adapter supplies that receipt after verifying the concrete candidate artifact.
    """
    diagnostics = document.get("diagnostics") or {}
    graph = diagnostics.get("captured_logical_graph") or {}
    arm = (diagnostics.get("arms") or {}).get("candidate") or {}
    plan = diagnostics.get("verified_global_plan_emission") or {}
    emission = document.get("emission") or {}
    blockers: list[str] = []
    if not _is_sha256(document.get("candidate_sha256")):
        blockers.append("candidate_digest_missing")
    if graph.get("status") != "verified" or not _is_sha256(graph.get("logical_dispatch_digest")):
        blockers.append("complete_logical_graph_unverified")
    if arm.get("status") != "emitted":
        blockers.append("whole_model_lowering_not_emitted")
    # Only a host adapter which has verified plan ownership, graph coverage, and artifact binding
    # may populate this field. The compiler's params/global_plan metadata is not copied here.
    if plan.get("status") != "verified":
        blockers.append("candidate_global_plan_emission_unverified")
    else:
        bindings = {
            "candidate_sha256": document.get("candidate_sha256"),
            "logical_dispatch_digest": graph.get("logical_dispatch_digest"),
            "candidate_lowered_sha256": emission.get("candidate_lowered_sha256"),
            "candidate_command_buffer_sha256": emission.get("candidate_command_buffer_sha256"),
        }
        if not _is_sha256(plan.get("plan_digest")) or any(
            not _is_sha256(value) or plan.get(key) != value for key, value in bindings.items()
        ):
            blockers.append("candidate_global_plan_binding_mismatch")
    return {
        "schema": "global_iteration_readiness_v1",
        "status": "ready_for_probe_admission" if not blockers else "blocked",
        "candidate_sha256": document.get("candidate_sha256"),
        "blockers": blockers,
        "probe_admission": "required_separately_for_each_measured_mechanism",
        "full_model_simulation_allowed": False,
        "micro_plateau_can_stop_global_search": False,
        "proof_scope": "structural graph/plan/artifact binding; authoring and probe admission only",
        "promotion_blockers": ["changed_region_semantic_qualification", "global_cost_evidence"],
    }


def emit_pair(package: OR.Package, interface: Path, scratch: Path, tag: str, timeout_s: int) -> tuple[int, str, str]:
    """One capsule's emitted artifacts under one compiler: (rc, lowered LLVM, command buffer)."""
    from merlin.targetgen import oot_runner as OR  # noqa: PLC0415

    buffer_path = scratch / f"cb_{tag}.json"
    rows = []
    results = []
    entrypoints = OR.analysis_emission_entrypoints(package)
    for name in entrypoints:
        destination = buffer_path if name in ("emit_command_buffer", "emit_analysis_bundle") else None
        try:
            result = OR.run_entrypoint(package, name, interface, destination, timeout=timeout_s)
        except (subprocess.TimeoutExpired, TimeoutError) as exc:
            stderr = getattr(exc, "stderr", None) or ""
            if isinstance(stderr, bytes):
                stderr = stderr.decode("utf-8", errors="replace")
            rows.append(
                {"command": name, "returncode": None, "exception": type(exc).__name__, "stderr_tail": stderr[-4096:]}
            )
            _write_json(
                scratch / f"emission_{tag}.json",
                {"schema": "compiler_emission_diagnostics_v1", "arm": tag, "entrypoints": rows},
            )
            raise
        rows.append({"command": name, "returncode": result.returncode, "stderr_tail": str(result.stderr or "")[-4096:]})
        _write_json(
            scratch / f"emission_{tag}.json",
            {"schema": "compiler_emission_diagnostics_v1", "arm": tag, "entrypoints": rows},
        )
        if result.returncode != 0:
            return result.returncode, "", ""
        results.append(result)
    target_result = results[0] if entrypoints == ("emit_analysis_bundle",) else results[1]
    return (0, target_result.stdout or "", buffer_path.read_text(encoding="utf-8") if buffer_path.is_file() else "")
