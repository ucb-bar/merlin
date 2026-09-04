"""Evidence-gated, non-agentic causal attribution for frozen paper studies.

Performance samples answer *whether* one complete session is faster.  They cannot answer *why*.
This module therefore never examines timing samples when producing an explanation.  A claim-ready
record can only be made from a pre-freeze evidence manifest containing both a frozen ablation and a
structural inspection.  Every component is content-addressed and bound to the exact compiler,
capture/session, and comparator bytes used by the matrix cell.
"""
from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping

import yaml


_SHA256_LEN = 64
_HEX = frozenset("0123456789abcdef")


class CausalEvidenceError(ValueError):
    """A declared causal-evidence artifact is malformed or not bound to the frozen study."""


def _is_sha(value: object) -> bool:
    text = str(value)
    return len(text) == _SHA256_LEN and all(char in _HEX for char in text)


def _canonical_sha(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _study_identity_sha(raw: Mapping[str, Any]) -> str:
    """Hash the frozen scientific contract without causal-manifest self-reference.

    ``status`` and ``frozen_at`` are lifecycle metadata.  The causal-attribution field itself is
    omitted because it contains the manifest digest that, in turn, contains this binding.
    Everything else—including claim thresholds, sessions, captures, packages, and sources—is part
    of the identity.
    """
    payload = copy.deepcopy(dict(raw))
    payload.pop("status", None)
    reporting = payload.get("reporting")
    if isinstance(reporting, dict):
        reporting.pop("causal_attribution", None)
    freeze = payload.get("freeze")
    if isinstance(freeze, dict):
        freeze.pop("frozen_at", None)
    return _canonical_sha(payload)


def _load_mapping(path: Path, *, label: str) -> dict[str, Any]:
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise CausalEvidenceError(f"cannot load {label}: {error}") from error
    if not isinstance(loaded, dict):
        raise CausalEvidenceError(f"{label} must be a mapping")
    return loaded


def _within(root: Path, value: object, *, label: str) -> Path:
    relative = Path(str(value))
    if relative.is_absolute() or ".." in relative.parts:
        raise CausalEvidenceError(f"{label} must be a relative path below the evidence manifest")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise CausalEvidenceError(f"{label} escapes the evidence manifest") from error
    if not resolved.is_file():
        raise CausalEvidenceError(f"{label} is absent: {resolved}")
    return resolved


def _backend(raw: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    rows = [row for row in raw.get("backends", ()) or ()
            if isinstance(row, Mapping) and str(row.get("name", "")) == name]
    if len(rows) != 1:
        raise CausalEvidenceError(f"frozen study has no unique backend {name!r}")
    return rows[0]


def _model(raw: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    rows = [row for row in raw.get("models", ()) or ()
            if isinstance(row, Mapping) and str(row.get("name", "")) == name]
    if len(rows) != 1:
        raise CausalEvidenceError(f"frozen study has no unique model {name!r}")
    return rows[0]


def _package_digest(backend: Mapping[str, Any], model: str, precision: str) -> str:
    options = backend.get("options", {}) or {}
    if backend.get("kind") == "external_runtime":
        packages = options.get("packages", {}) or {}
        row = (packages.get(model, {}) or {}).get(precision, {}) if isinstance(packages, Mapping) else {}
        value = row.get("sha256") if isinstance(row, Mapping) else None
    else:
        value = options.get("package_sha256")
    if not _is_sha(value):
        raise CausalEvidenceError(
            f"backend {backend.get('name')}/{model}/{precision} has no frozen package digest")
    return str(value)


def _source_digest(raw: Mapping[str, Any], backend: Mapping[str, Any]) -> str:
    kind = backend.get("kind")
    options = backend.get("options", {}) or {}
    if kind == "compiler":
        value = (raw.get("freeze", {}) or {}).get("compiler_source_sha256")
    elif kind == "external_runtime":
        value = options.get("framework_source_sha256")
    else:
        # Kernel swaps and frozen baselines both execute retained native code.  A package digest
        # alone identifies the binary, not the source against which a causal explanation was made.
        source_paths = options.get("source_paths")
        if not isinstance(source_paths, list) or not source_paths:
            raise CausalEvidenceError(
                f"backend {backend.get('name')} has no attested source_paths")
        value = options.get("kernel_source_sha256")
        if kind == "frozen_baseline" and _is_sha(value):
            from merlin.common.paths import repo_root
            from .freeze import sha256_paths
            paths = [Path(str(path)) for path in source_paths]
            paths = [path if path.is_absolute() else repo_root() / path for path in paths]
            try:
                actual = sha256_paths(paths)
            except (FileNotFoundError, OSError) as error:
                raise CausalEvidenceError(
                    f"backend {backend.get('name')} source_paths cannot be attested: {error}") from error
            if actual != value:
                raise CausalEvidenceError(
                    f"backend {backend.get('name')} source digest differs from source_paths")
    if not _is_sha(value):
        raise CausalEvidenceError(
            f"backend {backend.get('name')} has no frozen source digest")
    return str(value)


def expected_binding(raw: Mapping[str, Any], *, model: str, precision: str, core_count: int,
                     comparator: str) -> dict[str, Any]:
    """Canonical identity required for one ours-vs-comparator explanation.

    This intentionally contains no wall-clock number.  A manifest whose structural explanation is
    copied between a different capture, compiler package, or comparator cannot pass this check.
    """
    compiler_rows = [row for row in raw.get("backends", ()) or ()
                     if isinstance(row, Mapping) and row.get("kind") == "compiler"]
    if len(compiler_rows) != 1:
        raise CausalEvidenceError("frozen study must contain exactly one compiler backend")
    ours = compiler_rows[0]
    other = _backend(raw, comparator)
    model_row = _model(raw, model)
    artifacts = model_row.get("artifacts", {}) or {}
    artifact = artifacts.get(precision, {}) if isinstance(artifacts, Mapping) else {}
    capture_sha = artifact.get("sha256") if isinstance(artifact, Mapping) else None
    session_ids = (raw.get("freeze", {}) or {}).get("capture_session_identity_sha256", {}) or {}
    by_model = session_ids.get(model, {}) if isinstance(session_ids, Mapping) else {}
    session_sha = by_model.get(precision) if isinstance(by_model, Mapping) else None
    freeze = raw.get("freeze", {}) or {}
    binding = {
        "study_identity_sha256": _study_identity_sha(raw),
        "study_label": raw.get("label"),
        "target": raw.get("target"),
        "model": model,
        "checkpoint": model_row.get("checkpoint"),
        "fidelity": model_row.get("fidelity"),
        "precision": precision,
        "core_count": int(core_count),
        "comparator": comparator,
        "compiler_policy_sha256": freeze.get("policy_sha256"),
        "compiler_source_sha256": freeze.get("compiler_source_sha256"),
        "runtime_sha256": freeze.get("runtime_sha256"),
        "capture_sha256": capture_sha,
        "capture_session_identity_sha256": session_sha,
        "session_protocol_sha256": _canonical_sha(model_row.get("session")),
        "ours": {
            "backend": ours.get("name"), "kind": ours.get("kind"),
            "runtime": ours.get("runtime"), "quantization": ours.get("quantization"),
            "package_sha256": _package_digest(ours, model, precision),
            "source_sha256": _source_digest(raw, ours),
        },
        "comparator_backend": {
            "backend": other.get("name"), "kind": other.get("kind"),
            "runtime": other.get("runtime"), "quantization": other.get("quantization"),
            "package_sha256": _package_digest(other, model, precision),
            "source_sha256": _source_digest(raw, other),
        },
    }
    missing = [key for key in ("compiler_policy_sha256", "compiler_source_sha256", "runtime_sha256",
                               "capture_sha256", "capture_session_identity_sha256",
                               "session_protocol_sha256")
               if not _is_sha(binding[key])]
    if missing:
        raise CausalEvidenceError(f"causal binding has unresolved digests: {missing}")
    return binding


def _record_key(record: Mapping[str, Any]) -> tuple[str, str, int, str]:
    try:
        return (str(record["model"]), str(record["precision"]), int(record["core_count"]),
                str(record["comparator"]))
    except (KeyError, TypeError, ValueError) as error:
        raise CausalEvidenceError("causal evidence record lacks model/precision/core_count/comparator") from error


def _evidence_file(record: Mapping[str, Any], field: str, root: Path,
                   binding_sha: str, hasher: Callable[[list[Path]], str]) \
        -> tuple[dict[str, Any], Path]:
    value = record.get(field)
    value = _closed(value, {"path", "sha256"}, label=f"causal record {field}")
    path = _within(root, value.get("path"), label=f"{field}.path")
    digest = str(value.get("sha256", ""))
    if not _is_sha(digest) or hasher([path]) != digest:
        raise CausalEvidenceError(f"{field} digest does not match its retained artifact")
    payload = _load_mapping(path, label=field)
    expected_kind = f"frozen_{field}"
    if payload.get("kind") != expected_kind or payload.get("status") != "pass":
        raise CausalEvidenceError(f"{field} must be a passing {expected_kind} artifact")
    if payload.get("binding_sha256") != binding_sha:
        raise CausalEvidenceError(f"{field} is not bound to this compiler/session/comparator")
    return payload, path


def _retained_file(value: object, root: Path, *, label: str,
                   hasher: Callable[[list[Path]], str]) -> tuple[Path, str]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise CausalEvidenceError(
            f"{label} must be a closed retained path/digest mapping")
    path = _within(root, value.get("path"), label=f"{label}.path")
    digest = str(value.get("sha256", ""))
    if not _is_sha(digest) or hasher([path]) != digest:
        raise CausalEvidenceError(f"{label} digest does not match its retained artifact")
    return path, digest


def _closed(value: object, fields: set[str], *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CausalEvidenceError(f"{label} must be a mapping")
    extra, missing = sorted(set(value) - fields), sorted(fields - set(value))
    if extra or missing:
        raise CausalEvidenceError(
            f"{label} is closed; unrecognized={extra} missing={missing}")
    return value


def _execute_replay(generator: Path, action: str, source: Path,
                    expected: Mapping[str, Any]) -> Mapping[str, Any]:
    """Actually execute one canonical retained replay and compare its canonical bytes."""
    with tempfile.TemporaryDirectory(prefix="merlin-paper-replay-") as temp:
        output = Path(temp) / "result.json"
        command = ["python3", generator.name, action, source.name, str(output)]
        completed = subprocess.run(command, cwd=generator.parent, capture_output=True, text=True,
                                   timeout=60, check=False)
        if completed.returncode != 0:
            raise CausalEvidenceError(
                f"canonical {action} replay failed rc={completed.returncode}: "
                f"{completed.stderr[-500:]}")
        try:
            actual = json.loads(output.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise CausalEvidenceError(f"canonical {action} replay emitted invalid output") from error
        if action == "benchmark":
            actual_samples = actual.get("samples_ns")
            expected_samples = expected.get("samples_ns")
            if (not isinstance(actual_samples, list) or not actual_samples
                    or not isinstance(expected_samples, list) or not expected_samples):
                raise CausalEvidenceError("canonical benchmark replay lacks trusted samples")
            comparable_actual, comparable_expected = dict(actual), dict(expected)
            comparable_actual.pop("samples_ns", None)
            comparable_expected.pop("samples_ns", None)
            actual_median = sorted(actual_samples)[len(actual_samples) // 2]
            expected_median = sorted(expected_samples)[len(expected_samples) // 2]
            close = (actual_median <= expected_median * 4
                     and expected_median <= actual_median * 4)
        else:
            comparable_actual, comparable_expected, close = actual, expected, True
        if comparable_actual != comparable_expected or not close:
            suffix = "/executable/board receipts" if action == "benchmark" else ""
            raise CausalEvidenceError(
                f"canonical {action} replay differs from retained artifact{suffix}")
        return actual


def _validate_ablation(payload: Mapping[str, Any], path: Path, *, binding: Mapping[str, Any],
                       binding_sha: str,
                       hasher: Callable[[list[Path]], str]) -> dict[str, str]:
    root = path.parent
    _closed(payload, {"schema_version", "kind", "status", "binding_sha256", "changed",
                      "intervention",
                      "metric", "direction", "generator", "controller_pair", "control",
                      "treatment"},
            label="frozen ablation")
    if payload.get("schema_version") != 2:
        raise CausalEvidenceError("frozen ablation schema_version must be 2")
    intervention = _closed(payload.get("intervention"), {
        "id", "scope", "control", "treatment", "isolated_change",
    }, label="ablation intervention")
    expected_intervention = {
        "id": "runtime_dispatch_elimination_v1", "scope": "compiler_lowering",
        "control": "disabled", "treatment": "enabled",
        "isolated_change": "runtime_dispatch_sites",
    }
    if dict(intervention) != expected_intervention or payload.get("changed") != intervention["id"]:
        raise CausalEvidenceError(
            "frozen ablation lacks the predeclared isolated intervention contract")
    generator = _closed(payload.get("generator"),
                        {"kind", "agentic", "id", "source", "commands"},
                        label="ablation generator")
    if (not isinstance(generator, Mapping)
            or generator.get("kind") != "deterministic_non_agentic"
            or generator.get("agentic") is not False):
        raise CausalEvidenceError(
            "frozen ablation requires deterministic non-agentic generator provenance")
    generator_path, generator_sha = _retained_file(
        generator.get("source"), root, label="ablation.generator.source", hasher=hasher)
    from .paper_ablation_generator import (
        GENERATOR_ID,
        MEASUREMENT_TOOL_ID,
        TRUSTED_K1_PROBE_SOURCE_SHA256,
        observe,
        summarize,
    )

    trusted_generator = Path(__file__).with_name("paper_ablation_generator.py")
    if (generator.get("id") != GENERATOR_ID
            or generator_path.read_bytes() != trusted_generator.read_bytes()):
        raise CausalEvidenceError(
            "frozen ablation source is not the trusted non-agentic generator")
    commands = generator.get("commands")
    expected_commands = {
        "benchmark": ["python3", generator_path.name, "benchmark", "{contract}", "{raw_log}"],
        "observe": ["python3", generator_path.name, "observe", "{raw_log}", "{observation}"],
        "summarize": ["python3", generator_path.name, "summarize", "{observation}", "{result}"],
    }
    if commands != expected_commands:
        raise CausalEvidenceError(
            "non-agentic generator commands must use the canonical argv "
            f"{expected_commands!r}")

    pair_path, pair_sha = _retained_file(
        payload.get("controller_pair"), root, label="ablation.controller_pair", hasher=hasher)
    try:
        from .paper_ablation_generator import verify_causal_pair
        pair = verify_causal_pair(pair_path, expected_binding_sha256=binding_sha)
    except ValueError as error:
        raise CausalEvidenceError(
            f"controller-owned causal pair cannot be replayed: {error}") from error

    metric, direction = payload.get("metric"), payload.get("direction")
    if not isinstance(metric, str) or not metric.strip():
        raise CausalEvidenceError("frozen ablation metric must be nonempty")
    if direction not in {"lower_is_better", "higher_is_better"}:
        raise CausalEvidenceError("frozen ablation direction must be lower_is_better or higher_is_better")

    retained: dict[str, str] = {
        "generator_source_sha256": generator_sha,
        "controller_pair_sha256": pair_sha,
    }
    medians: dict[str, int] = {}
    replay_medians: dict[str, int] = {}
    functional_outputs: dict[str, str] = {}
    artifact_digests: dict[str, str] = {}
    for variant in ("control", "treatment"):
        arm = payload.get(variant)
        if not isinstance(arm, Mapping):
            raise CausalEvidenceError(
                f"frozen ablation requires retained {variant} artifact and result")
        arm = _closed(arm, {"artifact", "benchmark_contract", "raw_log",
                            "measurement_run", "observation", "result"},
                      label=f"ablation.{variant}")
        artifact = _closed(arm.get("artifact"), {
            "path", "sha256", "backend", "package_sha256", "source_sha256", "executable",
            "build_receipt"}, label=f"ablation.{variant}.artifact")
        _, artifact_sha = _retained_file(
            {"path": artifact["path"], "sha256": artifact["sha256"]}, root,
            label=f"ablation.{variant}.artifact", hasher=hasher)
        assert isinstance(artifact, Mapping)  # established by _retained_file
        backend = (binding["comparator_backend"] if variant == "control"
                   else binding["ours"])
        expected_artifact_identity = {
            "backend": backend["backend"],
            "package_sha256": backend["package_sha256"],
            "source_sha256": backend["source_sha256"],
        }
        if any(artifact.get(key) != value for key, value in expected_artifact_identity.items()):
            raise CausalEvidenceError(
                f"{variant} ablation artifact differs from its bound backend package/source")
        executable_path, executable_sha = _retained_file(
            artifact.get("executable"), root,
            label=f"ablation.{variant}.artifact.executable", hasher=hasher)
        build_path, build_sha = _retained_file(
            artifact.get("build_receipt"), root,
            label=f"ablation.{variant}.artifact.build_receipt", hasher=hasher)
        build = _load_mapping(build_path, label=f"ablation.{variant}.build_receipt")
        _closed(build, {"schema_version", "kind", "status", "backend", "package_sha256",
                        "source_sha256", "executable_sha256", "package", "source", "invocation"},
                label=f"ablation.{variant}.build_receipt")
        invocation = _closed(build.get("invocation"),
                             {"tool", "argv", "cwd", "environment", "timeout_seconds"},
                             label=f"ablation.{variant}.build_receipt.invocation")
        _retained_file(invocation.get("tool"), root,
                       label=f"ablation.{variant}.build_receipt.tool", hasher=hasher)
        _package_path, package_input_sha = _retained_file(
            build.get("package"), root,
            label=f"ablation.{variant}.build_receipt.package", hasher=hasher)
        _source_path, source_input_sha = _retained_file(
            build.get("source"), root,
            label=f"ablation.{variant}.build_receipt.source", hasher=hasher)
        if (build.get("schema_version") != 2
                or build.get("kind") != "paper_executable_build_receipt_v2"
                or build.get("status") != "pass"
                or build.get("backend") != backend["backend"]
                or build.get("package_sha256") != backend["package_sha256"]
                or build.get("source_sha256") != backend["source_sha256"]
                or package_input_sha != backend["package_sha256"]
                or source_input_sha != backend["source_sha256"]
                or build.get("executable_sha256") != executable_sha
                or not isinstance(invocation.get("argv"), list) or not invocation["argv"]
                or invocation["argv"][0] != "{tool}"
                or invocation["argv"].count("{output}") != 1
                or invocation["argv"].count("{package}") != 1
                or invocation["argv"].count("{source}") != 1
                or executable_path != _within(root, artifact["path"],
                                              label=f"ablation.{variant}.artifact.path")):
            raise CausalEvidenceError(
                f"{variant} executable differs from its exact build/package receipt")
        contract_path, contract_sha = _retained_file(
            arm.get("benchmark_contract"), root,
            label=f"ablation.{variant}.benchmark_contract", hasher=hasher)
        contract = _load_mapping(contract_path, label=f"ablation.{variant}.benchmark_contract")
        contract_fields = {
            "schema_version", "kind", "status", "binding_sha256", "variant", "target",
            "model", "precision", "core_count", "backend", "package_sha256", "source_sha256",
            "runtime_sha256", "capture_sha256", "capture_session_identity_sha256",
            "session_protocol_sha256", "artifact_sha256", "run_id", "metric", "direction",
            "executable", "build_receipt", "execution", "board_probe",
        }
        _closed(contract, contract_fields, label=f"ablation.{variant}.benchmark_contract")
        expected_contract_identity = {
            "schema_version": 2, "kind": "paper_ablation_benchmark_contract_v2",
            "status": "ready", "binding_sha256": binding_sha, "variant": variant,
            "target": binding["target"], "model": binding["model"],
            "precision": binding["precision"], "core_count": binding["core_count"],
            "backend": backend["backend"], "package_sha256": backend["package_sha256"],
            "source_sha256": backend["source_sha256"],
            "runtime_sha256": binding["runtime_sha256"],
            "capture_sha256": binding["capture_sha256"],
            "capture_session_identity_sha256": binding["capture_session_identity_sha256"],
            "session_protocol_sha256": binding["session_protocol_sha256"],
            "artifact_sha256": artifact_sha,
            "metric": metric, "direction": direction,
        }
        if any(contract.get(key) != value for key, value in expected_contract_identity.items()):
            raise CausalEvidenceError(
                f"{variant} benchmark contract differs from frozen cell/backend identity")
        if not str(contract.get("run_id", "")).strip():
            raise CausalEvidenceError(f"{variant} benchmark contract has no run_id")
        if (contract.get("executable") != artifact.get("executable")
                or contract.get("build_receipt") != artifact.get("build_receipt")):
            raise CausalEvidenceError(
                f"{variant} benchmark contract executable/build receipt differs")
        execution_contract = _closed(contract.get("execution"),
                                     {"argv", "cwd", "environment", "timeout_seconds",
                                      "warmup_iterations", "measured_iterations"},
                                     label=f"ablation.{variant}.execution contract")
        probe_contract = _closed(contract.get("board_probe"),
                                 {"authority", "source", "environment", "timeout_seconds"},
                                 label=f"ablation.{variant}.board probe contract")
        probe_source, _probe_source_sha = _retained_file(
            probe_contract.get("source"), root,
            label=f"ablation.{variant}.trusted K1 board probe source", hasher=hasher)
        if (probe_contract.get("authority") != "merlin_trusted_k1_csr_sysfs_probe_v1"
                or hashlib.sha256(probe_source.read_bytes()).hexdigest()
                != TRUSTED_K1_PROBE_SOURCE_SHA256):
            raise CausalEvidenceError(
                f"{variant} board probe differs from the separately shipped trusted K1 "
                "CSR/sysfs probe")
        declared_argv = execution_contract.get("argv")
        if (not isinstance(declared_argv, list) or not declared_argv
                or declared_argv[0] != "{executable}"):
            raise CausalEvidenceError(
                f"{variant} benchmark contract must invoke the bound executable first")
        raw_log_path, raw_log_sha = _retained_file(
            arm.get("raw_log"), root, label=f"ablation.{variant}.raw_log", hasher=hasher)
        raw_log = _load_mapping(raw_log_path, label=f"ablation.{variant}.raw_log")
        try:
            replay_observation = observe(
                raw_log, generator_source_sha256=generator_sha)
        except ValueError as error:
            raise CausalEvidenceError(
                f"{variant} raw ablation log cannot be replayed: {error}") from error
        if (raw_log.get("executable_sha256") != executable_sha
                or raw_log.get("build_receipt_sha256") != build_sha
                or raw_log.get("benchmark_contract_sha256") != _canonical_sha(contract)):
            raise CausalEvidenceError(
                f"{variant} raw log executable/build receipt differs from retained artifact")
        functional_outputs[variant] = str(raw_log.get("functional_stdout_sha256", ""))
        benchmark_replay = _execute_replay(generator_path, "benchmark", contract_path, raw_log)
        replay_values = benchmark_replay["samples_ns"]
        replay_medians[variant] = sorted(replay_values)[len(replay_values) // 2]
        observation_path, observation_sha = _retained_file(
            arm.get("observation"), root,
            label=f"ablation.{variant}.observation", hasher=hasher)
        observation = _load_mapping(
            observation_path, label=f"ablation.{variant}.observation")
        if observation != replay_observation:
            raise CausalEvidenceError(
                f"{variant} ablation observation differs from trusted raw-log replay")
        _execute_replay(generator_path, "observe", raw_log_path, observation)
        expected_observation = {
            "variant": variant, "binding_sha256": binding_sha,
            "target": binding["target"], "model": binding["model"],
            "precision": binding["precision"], "core_count": binding["core_count"],
            "artifact_sha256": artifact_sha, **expected_artifact_identity,
            "runtime_sha256": binding["runtime_sha256"],
            "capture_sha256": binding["capture_sha256"],
            "capture_session_identity_sha256": binding["capture_session_identity_sha256"],
            "session_protocol_sha256": binding["session_protocol_sha256"],
            "metric": metric, "direction": direction,
        }
        for key, value in expected_observation.items():
            if observation.get(key) != value:
                raise CausalEvidenceError(
                    f"{variant} ablation observation {key} differs from retained evidence")
        run_path, run_sha = _retained_file(
            arm.get("measurement_run"), root,
            label=f"ablation.{variant}.measurement_run", hasher=hasher)
        run = _load_mapping(run_path, label=f"ablation.{variant}.measurement_run")
        run_id = str(observation.get("run_id", "")).strip()
        execution_argv = raw_log.get("execution_argv")
        command_sha = _canonical_sha(execution_argv)
        if (not run_id or not isinstance(execution_argv, list) or not execution_argv
                or any(not isinstance(value, str) or not value for value in execution_argv)
                or observation.get("command_sha256") != command_sha
                or raw_log.get("executable_sha256") != executable_sha
                or raw_log.get("build_receipt_sha256") != build_sha
                or raw_log.get("benchmark_contract_sha256") != _canonical_sha(contract)
                or execution_argv != [str(executable_path), *declared_argv[1:]]
                or raw_log.get("board_probe_argv") != [
                    "merlin-trusted-k1-board-probe",
                    "--unit-test-json" if binding["target"] == "unit-test" else "--json"]):
            raise CausalEvidenceError(
                f"{variant} ablation raw log lacks a content-bound execution command/executable")
        expected_run = {
            "schema_version": 2, "kind": "frozen_ablation_measurement_run", "status": "pass",
            **expected_observation,
            "run_id": run_id, "command_sha256": command_sha,
            "build_receipt_sha256": build_sha,
            "benchmark_contract_sha256": _canonical_sha(contract),
            "executable_sha256": raw_log["executable_sha256"],
            "board_receipts_sha256": observation["board_receipts_sha256"],
            "raw_log_sha256": raw_log_sha, "observation_sha256": observation_sha,
            "tool": {
                "id": MEASUREMENT_TOOL_ID, "source_sha256": generator_sha,
                "command": ["python3", generator_path.name, "observe", raw_log_path.name,
                            observation_path.name],
            },
        }
        if set(run) != set(expected_run):
            raise CausalEvidenceError(
                f"{variant} measurement run contains unrecognized or missing fields")
        for key, value in expected_run.items():
            if run.get(key) != value:
                raise CausalEvidenceError(
                    f"{variant} measurement run {key} differs from bound raw evidence")
        result_path, result_sha = _retained_file(
            arm.get("result"), root, label=f"ablation.{variant}.result", hasher=hasher)
        result = _load_mapping(result_path, label=f"ablation.{variant}.result")
        try:
            replay = summarize(observation, generator_source_sha256=generator_sha)
        except ValueError as error:
            raise CausalEvidenceError(
                f"{variant} ablation observation cannot be replayed: {error}") from error
        if result != replay:
            raise CausalEvidenceError(
                f"{variant} ablation result differs from deterministic non-agentic replay")
        _execute_replay(generator_path, "summarize", observation_path, result)
        samples = result.get("samples")
        if (not isinstance(samples, list) or not samples
                or any(not isinstance(value, int) or isinstance(value, bool) or value <= 0
                       for value in samples)):
            raise CausalEvidenceError(
                f"{variant} ablation result requires positive integer samples")
        ordered = sorted(samples)
        median = (ordered[len(ordered) // 2] if len(ordered) % 2
                  else (ordered[len(ordered) // 2 - 1] + ordered[len(ordered) // 2]) // 2)
        if result.get("median") != median:
            raise CausalEvidenceError(
                f"{variant} ablation result median differs from retained samples")
        medians[variant] = median
        artifact_digests[variant] = artifact_sha
        if hashlib.sha256(executable_path.read_bytes()).hexdigest() != pair[
                f"{variant}_artifact_content_sha256"]:
            raise CausalEvidenceError(
                f"{variant} measured executable differs from the controller-generated typed pair")
        retained[f"{variant}_artifact_sha256"] = artifact_sha
        retained[f"{variant}_build_receipt_sha256"] = build_sha
        retained[f"{variant}_benchmark_contract_sha256"] = contract_sha
        retained[f"{variant}_raw_log_sha256"] = raw_log_sha
        retained[f"{variant}_measurement_run_sha256"] = run_sha
        retained[f"{variant}_observation_sha256"] = observation_sha
        retained[f"{variant}_result_sha256"] = result_sha
    if artifact_digests["control"] == artifact_digests["treatment"]:
        raise CausalEvidenceError("frozen ablation control and treatment artifacts must differ")
    if (not _is_sha(functional_outputs["control"])
            or functional_outputs["control"] != functional_outputs["treatment"]):
        raise CausalEvidenceError(
            "ablation control and treatment must have identical functional output/correctness")
    improved = (medians["treatment"] < medians["control"] if direction == "lower_is_better"
                else medians["treatment"] > medians["control"])
    replay_improved = (replay_medians["treatment"] < replay_medians["control"]
                       if direction == "lower_is_better"
                       else replay_medians["treatment"] > replay_medians["control"])
    if not improved or not replay_improved:
        raise CausalEvidenceError("retained treatment result does not improve over control")
    if pair["treatment_improved"] is not True:
        raise CausalEvidenceError(
            "controller-owned typed treatment did not reproduce a latency improvement")
    return retained


def _validate_structural(payload: Mapping[str, Any], path: Path, *,
                         binding: Mapping[str, Any], binding_sha: str,
                         ablation_sha: str, retained_ablation: Mapping[str, str],
                         hasher: Callable[[list[Path]], str]) -> tuple[dict[str, str], str, str]:
    """Replay a closed, non-narrative structural inspection into trusted why/how text."""
    allowed = {"schema_version", "kind", "status", "binding_sha256", "generator",
               "inspection", "result"}
    if set(payload) != allowed:
        raise CausalEvidenceError(
            "trusted structural evidence accepts only generator/inspection/result fields")
    root = path.parent
    if payload.get("schema_version") != 2:
        raise CausalEvidenceError("trusted structural evidence schema_version must be 2")
    generator = _closed(payload.get("generator"),
                        {"kind", "agentic", "id", "source", "command"},
                        label="structural.generator")
    if (not isinstance(generator, Mapping)
            or generator.get("kind") != "deterministic_non_agentic"
            or generator.get("agentic") is not False):
        raise CausalEvidenceError(
            "trusted structural evidence requires deterministic non-agentic provenance")
    generator_path, generator_sha = _retained_file(
        generator.get("source"), root, label="structural.generator.source", hasher=hasher)
    from .paper_ablation_generator import STRUCTURAL_TOOL_ID, inspect

    trusted_generator = Path(__file__).with_name("paper_ablation_generator.py")
    expected_command = [
        "python3", generator_path.name, "inspect", "{inspection}", "{result}"]
    if (generator.get("id") != STRUCTURAL_TOOL_ID
            or generator_path.read_bytes() != trusted_generator.read_bytes()
            or generator.get("command") != expected_command):
        raise CausalEvidenceError(
            "trusted structural generator source/argv differs from the verifier")
    inspection_path, inspection_sha = _retained_file(
        payload.get("inspection"), root, label="structural.inspection", hasher=hasher)
    inspection = _load_mapping(inspection_path, label="structural.inspection")
    inspection_keys = {
        "schema_version", "kind", "status", "binding_sha256", "ablation_sha256", "mechanism",
        "intervention_id",
        "control_artifact_sha256", "treatment_artifact_sha256",
        "control_package_sha256", "treatment_package_sha256", "control_source_sha256",
        "treatment_source_sha256", "control_build_receipt_sha256",
        "treatment_build_receipt_sha256", "control_measurement_run_sha256",
        "treatment_measurement_run_sha256", "control_artifact", "treatment_artifact",
    }
    if set(inspection) != inspection_keys:
        raise CausalEvidenceError(
            "trusted structural inspection contains unrecognized or narrative fields")
    expected = {
        "schema_version": 2, "kind": "frozen_structural_inspection_contract", "status": "pass",
        "binding_sha256": binding_sha, "ablation_sha256": ablation_sha,
        "intervention_id": "runtime_dispatch_elimination_v1",
        "control_artifact_sha256": retained_ablation["control_artifact_sha256"],
        "treatment_artifact_sha256": retained_ablation["treatment_artifact_sha256"],
        "control_package_sha256": binding["comparator_backend"]["package_sha256"],
        "treatment_package_sha256": binding["ours"]["package_sha256"],
        "control_source_sha256": binding["comparator_backend"]["source_sha256"],
        "treatment_source_sha256": binding["ours"]["source_sha256"],
        "control_build_receipt_sha256": retained_ablation[
            "control_build_receipt_sha256"],
        "treatment_build_receipt_sha256": retained_ablation[
            "treatment_build_receipt_sha256"],
        "control_measurement_run_sha256": retained_ablation[
            "control_measurement_run_sha256"],
        "treatment_measurement_run_sha256": retained_ablation[
            "treatment_measurement_run_sha256"],
    }
    for key, value in expected.items():
        if inspection.get(key) != value:
            raise CausalEvidenceError(
                f"structural inspection {key} differs from ablation/backend evidence")
    for variant in ("control", "treatment"):
        _analyzed_path, analyzed_sha = _retained_file(
            inspection.get(f"{variant}_artifact"), root,
            label=f"structural.inspection.{variant}_artifact", hasher=hasher)
        if analyzed_sha != retained_ablation[f"{variant}_artifact_sha256"]:
            raise CausalEvidenceError(
                f"structural {variant} artifact differs from measured executable")
    try:
        replay = inspect(inspection, root=root, generator_source_sha256=generator_sha)
    except ValueError as error:
        raise CausalEvidenceError(f"trusted structural inspection cannot be replayed: {error}") from error
    result_path, result_sha = _retained_file(
        payload.get("result"), root, label="structural.result", hasher=hasher)
    result = _load_mapping(result_path, label="structural.result")
    if result != replay:
        raise CausalEvidenceError(
            "structural result differs from trusted non-agentic replay")
    _execute_replay(generator_path, "inspect", inspection_path, result)
    return ({"structural_generator_source_sha256": generator_sha,
             "structural_inspection_sha256": inspection_sha,
             "structural_result_sha256": result_sha,
             "control_analyzed_artifact_sha256": replay[
                 "control_analyzed_artifact_sha256"],
             "treatment_analyzed_artifact_sha256": replay[
                 "treatment_analyzed_artifact_sha256"]},
            str(replay["why"]), str(replay["how"]))


def _validate_record(record: Mapping[str, Any], raw: Mapping[str, Any], root: Path,
                     hasher: Callable[[list[Path]], str]) -> dict[str, Any]:
    _closed(record, {"model", "precision", "core_count", "comparator", "binding",
                     "binding_sha256", "ablation", "structural"},
            label="causal evidence record")
    model, precision, cores, comparator = _record_key(record)
    expected = expected_binding(raw, model=model, precision=precision, core_count=cores,
                                comparator=comparator)
    binding = record.get("binding")
    if (isinstance(binding, Mapping)
            and binding.get("study_identity_sha256") != expected["study_identity_sha256"]):
        raise CausalEvidenceError(
            f"causal record {model}/{precision}/{cores}c/{comparator} study identity "
            "differs from frozen study")
    if binding != expected:
        raise CausalEvidenceError(
            f"causal record {model}/{precision}/{cores}c/{comparator} binding differs from frozen study")
    binding_sha = _canonical_sha(expected)
    if record.get("binding_sha256") != binding_sha:
        raise CausalEvidenceError("causal record binding_sha256 does not match its binding")
    ablation, ablation_path = _evidence_file(record, "ablation", root, binding_sha, hasher)
    structural, structural_path = _evidence_file(
        record, "structural", root, binding_sha, hasher)
    retained = _validate_ablation(
        ablation, ablation_path, binding=expected, binding_sha=binding_sha, hasher=hasher)
    if not str(ablation.get("changed", "")).strip():
        raise CausalEvidenceError("frozen ablation must state its declared changed treatment")
    structural_retained, why, how = _validate_structural(
        structural, structural_path, binding=expected, binding_sha=binding_sha,
        ablation_sha=str(record["ablation"]["sha256"]), retained_ablation=retained,
        hasher=hasher)
    return {
        "model": model, "precision": precision, "core_count": cores, "comparator": comparator,
        "binding": expected, "binding_sha256": binding_sha,
        "ablation": dict(record["ablation"]), "structural": dict(record["structural"]),
        "retained_ablation": {**retained, **structural_retained},
        "why": why, "how": how,
    }


def _validate_manifest(path: Path, raw: Mapping[str, Any],
                       hasher: Callable[[list[Path]], str]) -> dict[tuple[str, str, int, str], dict[str, Any]]:
    manifest = _load_mapping(path, label="causal evidence manifest")
    if manifest.get("schema_version") == 2:
        from .paper_full_model_ablation import FullModelAblationError, validate_manifest
        try:
            return validate_manifest(
                path, raw, expected_binding=expected_binding,
                study_identity_sha256=_study_identity_sha(raw), hasher=hasher)
        except FullModelAblationError as error:
            raise CausalEvidenceError(str(error)) from error
    _closed(manifest, {"schema_version", "records"}, label="causal evidence manifest")
    if int(manifest.get("schema_version", 0)) != 1:
        raise CausalEvidenceError("causal evidence manifest schema_version must be 1 or 2")
    # Schema v1 compares Merlin directly with a heterogeneous backend and retains a synthetic
    # micro-pair.  It remains useful for old unit fixtures, but it cannot isolate a K1 full-model
    # compiler intervention and therefore has no production claim authority.
    if raw.get("target") != "unit-test":
        raise CausalEvidenceError(
            "production causal claims require schema-v2 paired full-model Merlin evidence")
    records = manifest.get("records")
    if not isinstance(records, list):
        raise CausalEvidenceError("causal evidence manifest records must be a list")
    indexed: dict[tuple[str, str, int, str], dict[str, Any]] = {}
    for item in records:
        if not isinstance(item, Mapping):
            raise CausalEvidenceError("causal evidence records must be mappings")
        key = _record_key(item)
        if key in indexed:
            raise CausalEvidenceError(f"duplicate causal evidence record {key}")
        indexed[key] = _validate_record(item, raw, path.parent, hasher)
    return indexed


def freeze_causal_evidence(raw: dict[str, Any], *, root: Path,
                           hasher: Callable[[list[Path]], str]) -> None:
    """Resolve and verify an optional evidence manifest while freezing a study.

    No declaration means explanation records are explicitly unavailable (performance reporting still
    works, but cannot make a causal win claim).  A partially declared manifest fails closed.
    """
    reporting = raw.setdefault("reporting", {})
    declared = reporting.get("causal_attribution")
    if declared is None:
        reporting["causal_attribution"] = {
            "status": "unavailable",
            "reason": "no pre-freeze causal evidence manifest was declared",
        }
        return
    if not isinstance(declared, Mapping):
        raise CausalEvidenceError("reporting.causal_attribution must be a mapping")
    if declared.get("status") == "unavailable":
        _closed(declared, {"status", "reason"}, label="unavailable causal attribution")
        if not str(declared.get("reason", "")).strip():
            raise CausalEvidenceError("unavailable causal attribution requires a reason")
        reporting["causal_attribution"] = {"status": "unavailable",
                                             "reason": str(declared["reason"]).strip()}
        return
    _closed(declared, {"path"}, label="causal attribution declaration")
    value = Path(str(declared.get("path", "")))
    if not str(value) or str(value) == "unresolved":
        raise CausalEvidenceError("causal evidence manifest path is unresolved")
    path = value.resolve() if value.is_absolute() else (root / value).resolve()
    if not path.is_file():
        raise CausalEvidenceError(f"causal evidence manifest is absent: {path}")
    digest = hasher([path])
    records = _validate_manifest(path, raw, hasher)
    reporting["causal_attribution"] = {
        "status": "available", "schema_version": int(_load_mapping(
            path, label="causal evidence manifest")["schema_version"]),
        "path": str(path), "sha256": digest,
        "record_count": len(records),
    }


def _hasher(paths: list[Path]) -> str:
    # Local import avoids a module cycle while ``freeze.py`` imports this module.
    from .freeze import sha256_paths
    return sha256_paths(paths)


def _frozen_records(spec: Any) -> tuple[dict[tuple[str, str, int, str], dict[str, Any]], str | None]:
    config = (getattr(spec, "reporting", {}) or {}).get("causal_attribution")
    if not isinstance(config, Mapping) or config.get("status") != "available":
        return {}, "no frozen causal evidence manifest is available"
    path = Path(str(config.get("path", "")))
    if not path.is_file() or not _is_sha(config.get("sha256")):
        return {}, "frozen causal evidence manifest path or digest is invalid"
    if _hasher([path]) != config["sha256"]:
        return {}, "frozen causal evidence manifest digest changed after freeze"
    try:
        records = _validate_manifest(path, spec.canonical_dict(), _hasher)
    except CausalEvidenceError as error:
        return {}, str(error)
    if len(records) != int(config.get("record_count", -1)):
        return {}, "frozen causal evidence manifest record count changed after freeze"
    return records, None


def _result_matches(record: Mapping[str, Any], ours: Mapping[str, Any],
                    comparator: Mapping[str, Any], *, study_sha256: str) -> str | None:
    binding = record["binding"]
    for result, backend in ((ours, binding["ours"]), (comparator, binding["comparator_backend"])):
        if result.get("backend") != backend["backend"]:
            return "result backend differs from frozen causal binding"
        for key in ("runtime", "quantization"):
            if result.get(key) != backend[key]:
                return f"result {key} differs from frozen causal binding"
    for result in (ours, comparator):
        if (result.get("model"), result.get("precision"), int(result.get("core_count") or 0)) != (
                binding["model"], binding["precision"], binding["core_count"]):
            return "result matrix identity differs from frozen causal binding"
        provenance = result.get("provenance", {}) or {}
        if provenance.get("study_sha256") != study_sha256:
            return "result study_sha256 differs from the frozen study"
        for key in ("compiler_policy_sha256", "compiler_source_sha256", "runtime_sha256"):
            if provenance.get(key) != binding[key]:
                return f"result {key} differs from frozen causal binding"
        if result.get("artifact_sha256") != binding["capture_sha256"]:
            return "result capture digest differs from frozen causal binding"
        if provenance.get("capture_session_identity_sha256") != binding[
                "capture_session_identity_sha256"]:
            return "result capture session identity differs from frozen causal binding"
        if _canonical_sha(result.get("session")) != binding["session_protocol_sha256"]:
            return "result session protocol differs from frozen causal binding"
        for key in ("study_label", "target", "checkpoint", "fidelity"):
            if result.get(key) != binding[key]:
                return f"result {key} differs from frozen causal binding"

    ours_provenance = ours.get("provenance", {}) or {}
    if ours_provenance.get("package_sha256") != binding["ours"]["package_sha256"]:
        return "compiler result package digest differs from frozen causal binding"
    treatment_binary = record.get("treatment_binary_sha256")
    if treatment_binary is not None and ours_provenance.get("binary") != treatment_binary:
        return "compiler result binary differs from the paired full-model treatment"
    other_binding = binding["comparator_backend"]
    other_provenance = comparator.get("provenance", {}) or {}
    if other_binding["kind"] == "external_runtime":
        package_key, source_key = "framework_package_sha256", "framework_source_sha256"
    else:
        package_key, source_key = "package_sha256", "kernel_source_sha256"
    if other_provenance.get(package_key) != other_binding["package_sha256"]:
        return "comparator result package digest differs from frozen causal binding"
    if other_binding.get("source_sha256") is not None:
        if other_provenance.get(source_key) != other_binding["source_sha256"]:
            return "comparator result source digest differs from frozen causal binding"
    return None


def causal_record(spec: Any, ours: Mapping[str, Any], comparator: Mapping[str, Any]) -> dict[str, Any]:
    """Return a claim-ready record or a structured unavailable result, never a latency explanation."""
    comparator_name = str(comparator.get("backend", ""))
    base = {"comparator": comparator_name, "status": "unavailable"}
    records, unavailable = _frozen_records(spec)
    if unavailable:
        return {**base, "reason": unavailable}
    key = (str(ours.get("model", "")), str(ours.get("precision", "")),
           int(ours.get("core_count") or 0), comparator_name)
    record = records.get(key)
    if record is None:
        return {**base, "reason": "no frozen causal evidence record for this comparator cell"}
    try:
        study_sha256 = str(spec.sha256())
    except (AttributeError, TypeError):
        return {**base, "reason": "frozen study has no canonical study digest"}
    if not _is_sha(study_sha256):
        return {**base, "reason": "frozen study canonical digest is invalid"}
    mismatch = _result_matches(record, ours, comparator, study_sha256=study_sha256)
    if mismatch:
        return {**base, "reason": mismatch}
    return {
        "comparator": comparator_name, "status": "available",
        "why": record["why"], "how": record["how"],
        "evidence": {
            "binding_sha256": record["binding_sha256"],
            "ablation_sha256": record["ablation"]["sha256"],
            "structural_sha256": record["structural"]["sha256"],
            **record["retained_ablation"],
        },
    }


def attach_causal_attribution(spec: Any, results: list[dict[str, Any]]) -> None:
    """Attach one explicit available/unavailable record per measured comparator to compiler cells."""
    compiler = [backend for backend in spec.backends if backend.kind == "compiler"]
    if len(compiler) != 1:
        raise CausalEvidenceError("paper attribution requires one compiler backend")
    ours_name = compiler[0].name
    by_key = {(str(result.get("model")), str(result.get("backend")), str(result.get("precision")),
               int(result.get("core_count") or 0)): result for result in results}
    for result in results:
        if result.get("backend") != ours_name:
            continue
        records = []
        for backend in spec.backends:
            if backend.kind == "compiler":
                continue
            key = (str(result.get("model")), backend.name, str(result.get("precision")),
                   int(result.get("core_count") or 0))
            comparator = by_key.get(key)
            if comparator is None:
                records.append({"comparator": backend.name, "status": "unavailable",
                                "reason": "comparator result is absent from this paper run"})
            else:
                records.append(causal_record(spec, result, comparator))
        result["causal_attribution"] = {"schema_version": 1, "records": records}


__all__ = ["CausalEvidenceError", "attach_causal_attribution", "causal_record",
           "expected_binding", "freeze_causal_evidence"]
