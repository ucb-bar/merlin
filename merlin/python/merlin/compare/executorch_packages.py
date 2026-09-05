"""Build and register the complete FP32 ExecuTorch/XNNPACK paper package set.

This is the reproducible bridge between ``merlin-paper-capture`` and ``merlin-compare --freeze``.
Planning is the default and never imports a workload loader.  ``--execute`` builds all five immutable
packages sequentially, but does not publish a registration or an updated study unless every package
passes the same content, session, framework, and exporter/runtime identity checks used at freeze.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import yaml

from merlin.baselines import bundle as capture_bundle, executorch_session
from merlin.baselines.executorch import et_identity, et_venv_python
from merlin.baselines.executorch_session import (
    ExecuTorchSessionError,
    capture_session_identity,
    load_session_package,
    session_identity_sha256,
)
from merlin.common.artifacts import ProductDir, new_product
from merlin.common.paths import repo_root
from merlin.common.yaml import write_yaml

from .capture_workflow import _render_environment
from .freeze import sha256_paths
from .paper import BackendSpec, ModelSpec, PaperStudySpec
from .session import validate_capture_session


_SHA256_HEX = frozenset("0123456789abcdef")
_EXACT_ENV_KEYS = (
    "MERLIN_ET_VENV", "MERLIN_MODEL2MLIR", "MERLIN_M2M_DIR",
    "MERLIN_K1_TOOLCHAIN", "MERLIN_K1_TOOLCHAIN_ROOT",
)
_ALLOWED_INHERITED_ENV_KEYS = frozenset({
    "PATH", "LANG", "LC_ALL", "TZ", "TMPDIR", "SOURCE_DATE_EPOCH", "PYTHONHASHSEED",
    "CMAKE_GENERATOR",
})


class ExecuTorchPackagesNotReady(RuntimeError):
    """The five-package set cannot be planned, built, or atomically registered."""

    def __init__(self, reasons: list[str], output_dir: Path):
        super().__init__("; ".join(reasons))
        self.reasons = tuple(reasons)
        self.output_dir = output_dir


@dataclass(frozen=True)
class PackageTask:
    model: ModelSpec
    capture: Path
    capture_sha256: str
    capture_session_identity_sha256: str
    framework_source_sha256: str
    executorch_identity: dict[str, Any]
    model2mlir_identity: dict[str, Any]
    toolchain_identity: dict[str, Any]
    external_model_source: dict[str, Any] | None
    external_model_source_spec: dict[str, Any] | None
    environment: dict[str, str]
    output: Path
    work: Path
    command: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        command = list(self.command)
        return {
            "model": self.model.name,
            "capture": self.model.capture,
            "precision": "fp32",
            "variant": "fp32",
            "capture_path": str(self.capture),
            "capture_sha256": self.capture_sha256,
            "capture_session_identity_sha256": self.capture_session_identity_sha256,
            "framework_source_sha256": self.framework_source_sha256,
            "executorch_identity": dict(self.executorch_identity),
            "model2mlir_identity": dict(self.model2mlir_identity),
            "toolchain_identity": copy.deepcopy(self.toolchain_identity),
            "external_model_source": copy.deepcopy(self.external_model_source),
            "external_model_source_spec": copy.deepcopy(self.external_model_source_spec),
            "environment": dict(sorted(self.environment.items())),
            "environment_sha256": _json_sha256(self.environment),
            "output": str(self.output),
            "work": str(self.work),
            "command": command,
            "command_sha256": _json_sha256(command),
        }


PackageRunner = Callable[[PackageTask, dict[str, str], Path, Path], int]
PackageValidator = Callable[[PackageTask], dict[str, Any]]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _json_snapshot(path: Path) -> tuple[dict[str, Any], str]:
    """Parse and hash one immutable byte snapshot, never two reads of a mutable path."""
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    try:
        raw = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"JSON snapshot cannot be loaded: {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"JSON snapshot must contain a mapping: {path}")
    return raw, digest


def _study_snapshot(path: Path) -> tuple[PaperStudySpec, str]:
    """Parse and hash the exact same study bytes."""
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    try:
        raw = yaml.safe_load(payload.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ValueError(f"paper study snapshot cannot be loaded: {path}: {exc}") from exc
    return PaperStudySpec.parse(raw, source_path=path.resolve()), digest


def _is_sha256(value: object) -> bool:
    text = str(value)
    return len(text) == 64 and all(character in _SHA256_HEX for character in text)


def _resolve(path: object, *, relative_to: Path | None = None) -> Path:
    value = Path(str(path))
    return value.resolve() if value.is_absolute() else ((relative_to or repo_root()) / value).resolve()


def _external_backend(study: PaperStudySpec) -> BackendSpec:
    backends = [backend for backend in study.backends if backend.kind == "external_runtime"]
    if len(backends) != 1:
        raise ValueError(
            f"paper packaging requires exactly one external runtime, found {len(backends)}")
    backend = backends[0]
    if (backend.name != "executorch_xnnpack" or backend.runtime != "executorch"
            or backend.adapter != "executorch" or backend.precisions != ("fp32",)
            or backend.options.get("xnnpack") is not True):
        raise ValueError(
            "external runtime must be executorch_xnnpack/executorch, XNNPACK-enabled, FP32-only")
    return backend


def _registration(path: Path, study_path: Path, study_sha256: str,
                  study: PaperStudySpec) -> tuple[dict[str, Any], str | None, list[str]]:
    errors: list[str] = []
    if not path.is_file():
        return {}, None, [f"capture registration is absent: {path}"]
    try:
        raw, registration_sha256 = _json_snapshot(path)
    except (OSError, ValueError) as exc:
        return {}, None, [f"capture registration cannot be loaded: {exc}"]
    if raw.get("complete") is not True:
        errors.append("capture registration is not complete=true")
    registered_study = _resolve(raw.get("study", ""), relative_to=path.parent)
    if registered_study != study_path:
        errors.append(
            f"capture registration names another staged study: {registered_study}")
    expected_study_sha = str(raw.get("study_sha256", ""))
    if not _is_sha256(expected_study_sha) or expected_study_sha != study_sha256:
        errors.append(
            "capture registration staged-study digest differs: "
            f"registration={expected_study_sha} actual={study_sha256}")
    if raw.get("paper_inputs_sha256") != study.paper_inputs.get("sha256"):
        errors.append("capture registration paper-input digest differs from the staged study")

    expected = {(model.name, precision) for model in study.models for precision in model.precisions}
    rows = raw.get("captures")
    if not isinstance(rows, list):
        return raw, registration_sha256, [
            *errors, "capture registration captures must be a list"]
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            errors.append("capture registration contains a non-mapping capture row")
            continue
        key = (str(row.get("model", "")), str(row.get("precision", "")))
        if key in indexed:
            errors.append(f"capture registration repeats {key[0]}/{key[1]}")
        indexed[key] = row
    if set(indexed) != expected:
        errors.append(
            "capture registration does not contain exactly the staged 10-cell capture set: "
            f"expected={sorted(expected)} actual={sorted(indexed)}")
    for model in study.models:
        for precision, artifact in model.artifacts.items():
            row = indexed.get((model.name, precision))
            if row is None:
                continue
            if row.get("status") != "validated":
                errors.append(f"capture registration has non-validated {model.name}/{precision}")
            expected_path = _resolve(artifact.get("path", ""))
            actual_path = _resolve(row.get("path", ""), relative_to=path.parent)
            if actual_path != expected_path or row.get("sha256") != artifact.get("sha256"):
                errors.append(
                    f"capture registration path/digest differs for {model.name}/{precision}")
    return raw, registration_sha256, errors


def _paper_input_environments(study: PaperStudySpec) -> tuple[
        dict[str, dict[str, str]], list[str], Path, dict[str, Any]]:
    errors: list[str] = []
    bundle = _resolve(study.paper_inputs.get("path", ""))
    expected_digest = str(study.paper_inputs.get("sha256", ""))
    if not bundle.is_dir():
        return {}, [f"paper input bundle is absent: {bundle}"], bundle, {}
    actual_digest = sha256_paths([bundle])
    if not _is_sha256(expected_digest) or actual_digest != expected_digest:
        errors.append(
            f"paper input digest differs: study={expected_digest} actual={actual_digest}")
    record_path = bundle / "paper_inputs.json"
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {}, [*errors, f"paper input record cannot be loaded: {exc}"], bundle, {}
    records = record.get("models") if isinstance(record, dict) else None
    if not isinstance(records, dict) or set(records) != set(study.holdout_models):
        return {}, [*errors, "paper input record does not contain exactly the holdout roster"], bundle, {}
    environments: dict[str, dict[str, str]] = {}
    for model in study.models:
        try:
            environments[model.name] = _render_environment(records[model.name], bundle)
        except ValueError as exc:
            errors.append(f"{model.name}: {exc}")
    return environments, errors, bundle, records


def _framework_sources(backend: BackendSpec) -> tuple[list[Path], str]:
    values = backend.options.get("source_paths", ()) or ()
    if not isinstance(values, list) or not values:
        raise ValueError("ExecuTorch backend source_paths are absent")
    paths = [_resolve(value) for value in values]
    required_roots = {
        repo_root() / "third_party" / "baselines" / "executorch",
        repo_root() / "merlin" / "python" / "merlin",
    }
    if not required_roots <= set(paths):
        missing = sorted(str(path) for path in required_roots - set(paths))
        raise ValueError(
            "ExecuTorch framework_source_sha256 must cover the complete executed/imported "
            f"source closure; missing roots: {missing}")
    return paths, sha256_paths(paths)


def _compiler_identity(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise ValueError(f"SpacemiT compiler is absent: {path}")
    version = subprocess.run(
        [str(path), "--version"], capture_output=True, text=True, timeout=30)
    text = version.stdout.strip() or version.stderr.strip()
    if version.returncode or not text:
        raise ValueError(f"cannot record SpacemiT compiler version: {path}: {text}")
    return {"path": str(path.resolve()), "sha256": _file_sha256(path), "version": text}


def _toolchain_identity(root: str | Path | None = None) -> dict[str, Any]:
    """Resolve the exact compiler prefix without consulting repository ``.env`` state."""
    requested = str(root or os.environ.get("MERLIN_K1_TOOLCHAIN", "")).strip()
    if not requested:
        raise ValueError(
            "paper package preflight requires explicit MERLIN_K1_TOOLCHAIN or --k1-toolchain")
    candidate = Path(requested).resolve()
    roots = [candidate]
    if candidate.is_dir():
        roots.extend(sorted(candidate.glob("spacemit-toolchain-*")))
        roots.extend(sorted(candidate.glob("*/spacemit-toolchain-*")))
    resolved = next((value.resolve() for value in roots
                     if (value / "bin" / "clang").is_file()
                     and (value / "bin" / "clang++").is_file()), None)
    if resolved is None:
        raise ValueError(
            f"explicit SpacemiT toolchain contains no clang/clang++ prefix: {candidate}")
    return {
        "root": str(resolved),
        "c_compiler": _compiler_identity(resolved / "bin" / "clang"),
        "cxx_compiler": _compiler_identity(resolved / "bin" / "clang++"),
    }


def _git_sha(root: Path, where: str) -> str:
    revision = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True, text=True, timeout=30)
    value = revision.stdout.strip().lower() if revision.returncode == 0 else ""
    if len(value) != 40 or any(character not in _SHA256_HEX for character in value):
        raise ValueError(f"{where} has no full Git identity: {revision.stderr.strip()}")
    return value


def _require_external_source_closure(
        checkout: Path, source_root: Path, source_file: Path) -> None:
    """Reject an external source closure that can dereference outside its declared roots."""
    try:
        source_root.relative_to(checkout)
        source_file.relative_to(checkout)
        source_file.relative_to(source_root)
    except ValueError as exc:
        raise ValueError(
            "external model source escapes its declared source root/checkout") from exc
    if not source_root.is_dir() or not source_file.is_file():
        raise FileNotFoundError(
            f"external model source closure is absent: root={source_root} file={source_file}")
    for path in sorted(source_root.rglob("*"), key=lambda candidate: candidate.as_posix()):
        if not path.is_symlink():
            continue
        try:
            target = path.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise ValueError(
                f"external model source nested symlink cannot be resolved: {path}") from exc
        try:
            target.relative_to(source_root)
            target.relative_to(checkout)
        except ValueError as exc:
            raise ValueError(
                "external model source nested symlink escapes its declared "
                f"source root/checkout: {path} -> {target}") from exc


def _external_model_sources(
        backend: BackendSpec, environments: dict[str, dict[str, str]],
        records: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Observe every declared external model-code closure imported during export."""
    errors: list[str] = []
    declarations = backend.options.get("external_model_sources", {}) or {}
    if not isinstance(declarations, dict):
        return {}, ["external_model_sources must be a mapping"]
    unknown = set(declarations) - set(environments)
    if unknown:
        errors.append(f"external model source declarations name unknown models: {sorted(unknown)}")
    identities: dict[str, dict[str, Any]] = {}
    for model_name, value in declarations.items():
        if not isinstance(value, dict):
            errors.append(f"{model_name}: external model source declaration must be a mapping")
            continue
        environment_key = str(value.get("environment_key", ""))
        source_root_value = str(value.get("source_root", ""))
        source_file_value = str(value.get("source_file", ""))
        if (not environment_key or not source_root_value or not source_file_value
                or Path(source_root_value).is_absolute()
                or ".." in Path(source_root_value).parts
                or Path(source_file_value).is_absolute()
                or ".." in Path(source_file_value).parts):
            errors.append(f"{model_name}: external model source declaration is incomplete or unsafe")
            continue
        checkout_value = environments.get(model_name, {}).get(environment_key, "")
        checkout = Path(checkout_value).resolve() if checkout_value else Path()
        source_root = (checkout / source_root_value).resolve() if checkout_value else Path()
        source_file = (checkout / source_file_value).resolve() if checkout_value else Path()
        if not checkout_value:
            errors.append(
                f"{model_name}: external model source closure is absent: "
                f"checkout={checkout_value!r} root={source_root} file={source_file}")
            continue
        try:
            _require_external_source_closure(checkout, source_root, source_file)
        except (FileNotFoundError, ValueError) as exc:
            errors.append(f"{model_name}: {exc}")
            continue
        try:
            git_sha = _git_sha(checkout, f"{model_name} external model source")
            identity = {
                "environment_key": environment_key,
                "checkout": str(checkout),
                "git_sha": git_sha,
                "source_root": str(source_root),
                "source_tree_sha256": sha256_paths([source_root]),
                "source_file": str(source_file),
                "source_file_sha256": _file_sha256(source_file),
            }
        except (FileNotFoundError, ValueError) as exc:
            errors.append(str(exc))
            continue
        expected_git = str(value.get("git_sha", ""))
        expected_file = str(value.get("source_file_sha256", ""))
        if identity["git_sha"] != expected_git:
            errors.append(
                f"{model_name}: external model source Git identity differs: "
                f"declared={expected_git} actual={identity['git_sha']}")
        if identity["source_file_sha256"] != expected_file:
            errors.append(
                f"{model_name}: external model source file digest differs: "
                f"declared={expected_file} actual={identity['source_file_sha256']}")
        provenance = records.get(model_name, {}).get("provenance", {})
        checkpoint = provenance.get("checkpoint", {}) if isinstance(provenance, dict) else {}
        if (not isinstance(checkpoint, dict)
                or str(checkpoint.get("source_path", "")) != str(checkout)
                or str(checkpoint.get("source_revision", "")) != identity["git_sha"]
                or str(checkpoint.get("source_file", "")) != source_file_value
                or str(checkpoint.get("source_file_sha256", ""))
                != identity["source_file_sha256"]):
            errors.append(
                f"{model_name}: external model source differs from paper-input provenance")
        identities[model_name] = identity
    return identities, errors


def _exact_environment(model_environment: dict[str, str], model2mlir: Path,
                       toolchain: dict[str, Any]) -> dict[str, str]:
    exact = dict(model_environment)
    toolchain_root = str(toolchain.get("root", "unresolved"))
    exact.update({
        "MERLIN_ET_VENV": str(et_venv_python().absolute().parent.parent),
        "MERLIN_MODEL2MLIR": str(model2mlir),
        "MERLIN_M2M_DIR": str(model2mlir),
        "MERLIN_K1_TOOLCHAIN": toolchain_root,
        "MERLIN_K1_TOOLCHAIN_ROOT": toolchain_root,
    })
    return exact


def _model2mlir_identity(root: Path, study: PaperStudySpec) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    capture_source = root / "m2m" / "capture"
    loaders = {model.name: root / "workloads" / model.capture / "loader.py"
               for model in study.models}
    if not root.is_dir():
        return {}, [f"Model2MLIR root is absent: {root}"]
    missing = [str(path) for path in [capture_source, *loaders.values()] if not path.exists()]
    if missing:
        errors.append(f"Model2MLIR packaging sources are absent: {missing}")
    revision = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=30)
    git_sha = revision.stdout.strip().lower() if revision.returncode == 0 else ""
    if len(git_sha) != 40 or any(character not in _SHA256_HEX for character in git_sha):
        errors.append(f"Model2MLIR has no full Git identity: {revision.stderr.strip()}")
    return {
        "path": str(root), "git_sha": git_sha,
        "capture_source_sha256": (
            sha256_paths([capture_source]) if capture_source.is_dir() else "unresolved"),
        "loader_sha256": {
            name: sha256_paths([path]) if path.is_file() else "unresolved"
            for name, path in loaders.items()
        },
    }, errors


def _child_environment(exact: dict[str, str]) -> dict[str, str]:
    inherited = {
        key: value for key, value in os.environ.items()
        if key in _ALLOWED_INHERITED_ENV_KEYS
    }
    inherited.update(exact)
    return dict(sorted(inherited.items()))


def _tasks(study: PaperStudySpec, study_sha256: str, backend: BackendSpec,
           registration_path: Path, model2mlir: Path, product: ProductDir,
           toolchain_root: str | Path | None) -> tuple[
               list[PackageTask], list[str], dict[str, Any]]:
    errors: list[str] = []
    study_path = study.source_path
    if study_path is None:
        raise ValueError("paper study source path is absent")
    registration, registration_sha256, registration_errors = _registration(
        registration_path, study_path, study_sha256, study)
    errors.extend(registration_errors)
    environments, environment_errors, paper_inputs, paper_input_records = \
        _paper_input_environments(study)
    errors.extend(environment_errors)
    try:
        framework_paths, framework_digest = _framework_sources(backend)
    except (FileNotFoundError, ValueError) as exc:
        framework_paths, framework_digest = [], "unresolved"
        errors.append(str(exc))
    try:
        identity = et_identity().as_dict()
    except Exception as exc:  # ExecuTorch identity failures are a paper-package blocker.
        identity = {}
        errors.append(f"active ExecuTorch exporter/runtime identity is invalid: {exc}")
    model2mlir_identity, model2mlir_errors = _model2mlir_identity(model2mlir, study)
    errors.extend(model2mlir_errors)
    try:
        toolchain_identity = _toolchain_identity(toolchain_root)
    except (OSError, ValueError) as exc:
        toolchain_identity = {}
        errors.append(str(exc))
    external_sources, external_source_errors = _external_model_sources(
        backend, environments, paper_input_records)
    errors.extend(external_source_errors)
    source_declarations = backend.options.get("external_model_sources", {}) or {}
    if not isinstance(source_declarations, dict):
        source_declarations = {}

    packages = backend.options.get("packages")
    if not isinstance(packages, dict) or set(packages) != set(study.holdout_models):
        errors.append("ExecuTorch package map does not contain exactly the holdout roster")
        packages = {}
    actual_capture_digests: dict[tuple[str, str], str] = {}
    for model in study.models:
        for precision, artifact in model.artifacts.items():
            capture_path = _resolve(artifact.get("path", ""))
            expected_digest = str(artifact.get("sha256", ""))
            if not capture_path.is_dir() or not _is_sha256(expected_digest):
                errors.append(f"{model.name}/{precision}: staged capture path/digest is unresolved")
                continue
            actual_digest = sha256_paths([capture_path])
            actual_capture_digests[(model.name, precision)] = actual_digest
            if actual_digest != expected_digest:
                errors.append(
                    f"{model.name}/{precision}: capture digest differs: staged={expected_digest} "
                    f"actual={actual_digest}")
    tasks: list[PackageTask] = []
    for model in study.models:
        if set(model.precisions) != {study.primary_precision, study.control_precision}:
            errors.append(f"{model.name}: package workflow requires both declared study precisions")
        artifact = model.artifacts.get("fp32", {})
        capture_path = _resolve(artifact.get("path", ""))
        capture_digest = str(artifact.get("sha256", ""))
        if not capture_path.is_dir() or not _is_sha256(capture_digest):
            errors.append(f"{model.name}/fp32: staged capture path/digest is unresolved")
            continue
        actual_digest = actual_capture_digests.get((model.name, "fp32"))
        if actual_digest != capture_digest:
            errors.append(
                f"{model.name}/fp32: capture digest differs: staged={capture_digest} "
                f"actual={actual_digest}")
        session, session_errors = validate_capture_session(
            capture_path, model.session, expected_provenance=model.expected_provenance)
        errors.extend(f"{model.name}/fp32: {reason}" for reason in session_errors)
        try:
            session_digest = session_identity_sha256(capture_session_identity(session))
        except (ExecuTorchSessionError, ValueError) as exc:
            session_digest = "unresolved"
            errors.append(f"{model.name}/fp32: invalid capture session identity: {exc}")
        row = packages.get(model.name)
        package = row.get("fp32") if isinstance(row, dict) else None
        if not isinstance(row, dict) or set(row) != {"fp32"} or not isinstance(package, dict):
            errors.append(f"{model.name}: package map must contain exactly one fp32 row")
        elif (str(package.get("path", "")) != "unresolved"
              or str(package.get("sha256", "")) != "unresolved"):
            errors.append(
                f"{model.name}/fp32: package is already registered; immutable packages are not rebuilt")
        environment = _child_environment(_exact_environment(
            environments.get(model.name, {}), model2mlir, toolchain_identity))
        output = product.path / "packages" / f"{model.capture}_fp32"
        work = product.path / "work" / f"{model.capture}_fp32"
        command = (
            sys.executable, "-m", "merlin.baselines.executorch_session", "build",
            "--model", model.capture, "--variant", "fp32",
            "--session-contract", str(capture_path / "session_contract.yaml"),
            "--warmups", str(model.session.warmups),
            "--observations", str(model.session.observations),
            "--measurement-repeats", str(model.session.measurement_repeats),
            "--framework-source-sha256", framework_digest,
            "--build-invocation-environment-sha256", _json_sha256(environment),
            "--external-model-source-spec-json", json.dumps(
                source_declarations.get(model.name), sort_keys=True,
                separators=(",", ":")),
            "--output", str(output), "--work", str(work),
        )
        task_model2mlir_identity = {
            "path": str(model2mlir_identity.get("path", "")),
            "git_sha": str(model2mlir_identity.get("git_sha", "")),
            "loader_sha256": str(
                (model2mlir_identity.get("loader_sha256") or {}).get(model.name, "")),
            "capture_source_sha256": str(
                model2mlir_identity.get("capture_source_sha256", "")),
        }
        tasks.append(PackageTask(
            model, capture_path, capture_digest, session_digest, framework_digest,
            identity, task_model2mlir_identity, copy.deepcopy(toolchain_identity),
            copy.deepcopy(external_sources.get(model.name)),
            copy.deepcopy(source_declarations.get(model.name)),
            environment, output, work, command))
    if len(tasks) != 5 or {task.model.name for task in tasks} != set(study.holdout_models):
        errors.append(f"package plan has {len(tasks)} tasks, expected exactly five holdout models")
    evidence = {
        "capture_registration": {
            "path": str(registration_path),
            "sha256": registration_sha256,
            "complete": registration.get("complete"),
        },
        "paper_inputs": {"path": str(paper_inputs), "sha256": study.paper_inputs.get("sha256")},
        "framework_sources": [str(path) for path in framework_paths],
        "framework_source_sha256": framework_digest,
        "executorch_identity": identity,
        "model2mlir": model2mlir_identity,
        "toolchain_identity": toolchain_identity,
        "external_model_sources": external_sources,
    }
    return tasks, list(dict.fromkeys(errors)), evidence


def _default_runner(task: PackageTask, environment: dict[str, str],
                    stdout_path: Path, stderr_path: Path) -> int:
    with stdout_path.open("w", encoding="utf-8") as stdout, \
            stderr_path.open("w", encoding="utf-8") as stderr:
        proc = subprocess.run(
            list(task.command), cwd=repo_root(), env=environment, stdout=stdout, stderr=stderr)
    return int(proc.returncode)


def _validate_package(task: PackageTask) -> dict[str, Any]:
    digest = sha256_paths([task.output])
    package = load_session_package(task.output, expected_sha256=digest)
    mismatches: list[str] = []
    expected = {
        "model": (package.model, task.model.capture),
        "variant": (package.variant, "fp32"),
        "capture_sha256": (package.capture_sha256, task.capture_sha256),
        "capture_session_identity_sha256": (
            package.capture_session_identity_sha256, task.capture_session_identity_sha256),
        "framework_source_sha256": (
            package.framework_source_sha256, task.framework_source_sha256),
        "build_invocation_environment_sha256": (
            package.build_invocation_environment_sha256, _json_sha256(task.environment)),
        "warmups": (package.plan.warmups, task.model.session.warmups),
        "observations": (package.plan.observations, task.model.session.observations),
        "measurement_repeats": (
            package.plan.repeats, task.model.session.measurement_repeats),
    }
    mismatches.extend(
        f"{name}: package={actual!r} expected={wanted!r}"
        for name, (actual, wanted) in expected.items() if actual != wanted)
    metadata = json.loads((task.output / "session_package.json").read_text(encoding="utf-8"))
    environment = metadata.get("build_environment")
    embedded_identity = (environment.get("executorch_identity")
                         if isinstance(environment, dict) else None)
    if embedded_identity != task.executorch_identity:
        mismatches.append(
            "embedded ExecuTorch exporter/runtime identity differs from package preflight")
    if package.executorch_identity != embedded_identity:
        mismatches.append("loaded ExecuTorch identity differs from embedded metadata")
    if isinstance(environment, dict):
        embedded_model2mlir = environment.get("model2mlir_identity")
        embedded_toolchain = environment.get("toolchain_identity")
        embedded_external = environment.get("external_model_source")
        if embedded_model2mlir != task.model2mlir_identity:
            mismatches.append(
                "embedded Model2MLIR identity differs from package preflight")
        if package.model2mlir_identity != embedded_model2mlir:
            mismatches.append("loaded Model2MLIR identity differs from embedded metadata")
        if embedded_toolchain != task.toolchain_identity:
            mismatches.append("embedded toolchain identity differs from package preflight")
        if package.toolchain_identity != embedded_toolchain:
            mismatches.append("loaded toolchain identity differs from embedded metadata")
        if embedded_external != task.external_model_source:
            mismatches.append(
                "embedded external model source differs from package preflight")
        if package.external_model_source != embedded_external:
            mismatches.append(
                "loaded external model source differs from embedded metadata")
    else:
        embedded_model2mlir = None
        embedded_toolchain = None
        embedded_external = None
        mismatches.append("embedded package build environment is absent")
    if mismatches:
        raise ValueError("; ".join(mismatches))
    from merlin.compare.paper_model_object_builder import executorch_session_resources

    producer = executorch_session_resources(
        task.output / executorch_session.PAPER_COMPILER_INPUT, include_private=True)
    if producer.runner != package.runner:
        raise ValueError("paper producer receipt selects a different ExecuTorch session runner")
    return {
        "path": str(task.output.resolve()), "sha256": digest,
        "model": task.model.name, "capture": task.model.capture,
        "precision": "fp32", "variant": "fp32", "xnnpack": package.xnnpack,
        "capture_sha256": package.capture_sha256,
        "capture_session_identity_sha256": package.capture_session_identity_sha256,
        "framework_source_sha256": package.framework_source_sha256,
        "build_environment_sha256": package.build_environment_sha256,
        "build_invocation_environment_sha256":
            package.build_invocation_environment_sha256,
        "executorch_identity": embedded_identity,
        "model2mlir_identity": embedded_model2mlir,
        "toolchain_identity": embedded_toolchain,
        "external_model_source": embedded_external,
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        _write_json(temporary, value)
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _paper_measurement_templates(
    study: PaperStudySpec, backend: BackendSpec, tasks: list[PackageTask],
    results: list[dict[str, Any]], product: ProductDir,
) -> dict[str, dict[str, dict[str, dict[str, str]]]]:
    """Create receipt-bound sealed-executable templates for the production paper registry."""
    from merlin.compare.paper_measurement_controller import _EXECUTORCH_PRODUCTION_BUILD_ARGV
    from merlin.compare.paper_model_object_builder import (
        EXECUTORCH_RECIPE,
        executorch_session_resources,
        object_build_argv,
        regenerate_model_object,
    )

    root = product.path.resolve()
    tool_source = Path(tasks[0].toolchain_identity["c_compiler"]["path"]).resolve()
    tool = product.add_artifact("measurement-resources/build-tool")
    shutil.copy2(tool_source, tool)
    builder = Path(__file__).with_name("paper_model_object_builder.py")
    by_model: dict[str, dict[str, dict[str, dict[str, str]]]] = {}
    result_by_model = {str(row["model"]): row for row in results}
    task_by_model = {task.model.name: task for task in tasks}
    for model in study.models:
        task, result = task_by_model[model.name], result_by_model[model.name]
        package_root = Path(str(result["path"])).resolve()
        compiler_input = package_root / executorch_session.PAPER_COMPILER_INPUT
        resources = executorch_session_resources(compiler_input, include_private=True)
        runtime_artifact = resources.package_metadata
        runner = resources.runner
        receipt_path = product.add_artifact(
            f"measurement-resources/{model.name}/package-receipt.json")
        object_builder_sha = _file_sha256(builder)
        resource_hashes = {
            "compiler_input": _file_sha256(compiler_input),
            "model_object": _file_sha256(runner),
            "object_builder": object_builder_sha,
            "runner": _file_sha256(runner),
        }
        source_identity = _json_sha256(resource_hashes)
        with tempfile.TemporaryDirectory(prefix="merlin-executorch-receipt-") as temporary:
            verified = Path(temporary) / "sealed-runner"
            regeneration = regenerate_model_object(
                recipe=EXECUTORCH_RECIPE, registry_id="executorch_v1", target=study.target,
                compiler_input=compiler_input, tool=tool, output=verified,
                source_identity_sha256=task.framework_source_sha256,
                capture_sha256=task.capture_sha256,
                runtime_artifact_sha256=_file_sha256(runtime_artifact))
            if verified.read_bytes() != runner.read_bytes():
                raise ValueError("ExecuTorch sealed-runner reproduction differs")
        receipt = {
            "schema_version": 2, "kind": "paper_backend_package_receipt_v2",
            "status": "finalized", "registry_id": "executorch_v1",
            "build_adapter": "executorch_sealed_session_v1",
            "cell": {"model": model.name, "backend": backend.name, "precision": "fp32"},
            "package_identity_sha256": result["sha256"],
            "compiler_or_framework_source_sha256": task.framework_source_sha256,
            "capture_sha256": task.capture_sha256,
            "runtime_artifact_sha256": _file_sha256(runtime_artifact),
            "runner_source_sha256": _file_sha256(runner),
            "model_object_sha256": _file_sha256(runner),
            "compiler_input_sha256": _file_sha256(compiler_input),
            "object_builder_source_sha256": object_builder_sha,
            "object_recipe": EXECUTORCH_RECIPE,
            "object_build_argv": object_build_argv(EXECUTORCH_RECIPE),
            "generated_model_source_sha256": regeneration["generated_source_sha256"],
            "build_tool_sha256": _file_sha256(tool),
            "build_source_identity_sha256": source_identity,
            "build_argv": list(_EXECUTORCH_PRODUCTION_BUILD_ARGV),
            "result_executable_sha256": _file_sha256(runner),
            "finalized_at": _utc_now(),
        }
        _write_json(receipt_path, receipt)

        def ref(path: Path) -> dict[str, str]:
            return {"path": path.relative_to(root).as_posix(), "sha256": _file_sha256(path)}

        rows: dict[str, dict[str, str]] = {}
        for cores in study.core_counts:
            template_path = product.add_artifact(
                f"measurement-template-{model.name}-fp32-c{cores}.yaml")
            template = {
                "schema_version": 3, "kind": "paper_backend_measurement_template_v3",
                "status": "frozen", "registry_id": "executorch_v1",
                "backend_adapter": "executorch",
                "cell": {"model": model.name, "backend": backend.name,
                         "precision": "fp32", "core_count": cores},
                "resources": {
                    "package_receipt": ref(receipt_path),
                    "compiler_input": ref(compiler_input),
                    "model_object": ref(runner), "build_tool": ref(tool),
                    "runtime_artifact": ref(runtime_artifact),
                },
                "environment": {},
                "execution": {"mode": "rvv", "core_ids": list(range(cores)),
                              "require_worker_threads": True},
                "memory_policy": "mmap", "timeout_seconds": backend.options["timeout"],
            }
            template_path.write_text(yaml.safe_dump(template, sort_keys=True), encoding="utf-8")
            rows[str(cores)] = {"path": str(template_path), "sha256": _file_sha256(template_path)}
        by_model[model.name] = {"fp32": rows}
    return by_model


def _prepublication_reobserve(
        study_path: Path, study_sha256: str, registration_path: Path,
        model2mlir: Path, product: ProductDir, toolchain_root: str | Path | None,
        tasks: list[PackageTask], evidence: dict[str, Any]) -> list[str]:
    """Re-observe every mutable preflight input after the last long-running build."""
    errors: list[str] = []
    try:
        fresh_study, fresh_study_sha256 = _study_snapshot(study_path)
    except (OSError, ValueError) as exc:
        return [f"study changed before publication: {exc}"]
    if fresh_study_sha256 != study_sha256:
        errors.append(
            "study changed before publication: "
            f"preflight={study_sha256} actual={fresh_study_sha256}")
    try:
        backend = _external_backend(fresh_study)
        fresh_tasks, fresh_errors, fresh_evidence = _tasks(
            fresh_study, fresh_study_sha256, backend, registration_path,
            model2mlir, product, toolchain_root)
    except (OSError, ValueError) as exc:
        return [*errors, f"preflight input changed before publication: {exc}"]
    errors.extend(
        f"preflight input changed before publication: {error}" for error in fresh_errors)
    if [task.to_dict() for task in fresh_tasks] != [task.to_dict() for task in tasks]:
        errors.append("package task inputs or identities changed before publication")
    if fresh_evidence != evidence:
        errors.append("package-set evidence changed before publication")
    return list(dict.fromkeys(errors))


def materialize(study_path: str | Path, capture_registration: str | Path | None,
                model2mlir_root: str | Path, *, execute: bool = False,
                k1_toolchain: str | Path | None = None,
                runner: PackageRunner | None = None,
                validator: PackageValidator | None = None,
                product: ProductDir | None = None) -> Path:
    """Plan or build and atomically register all five FP32 ExecuTorch packages."""
    study_path = Path(study_path).resolve()
    study, study_sha256 = _study_snapshot(study_path)
    registration_path = (Path(capture_registration).resolve() if capture_registration is not None
                         else study_path.parent / "capture-registration.json")
    model2mlir = Path(model2mlir_root).resolve()
    product = product or new_product(
        "paper-executorch-packages", version=1, target=study.target,
        sources=[str(study_path), str(registration_path), str(model2mlir),
                 str(k1_toolchain or os.environ.get("MERLIN_K1_TOOLCHAIN", "unresolved"))])
    plan_path = product.add_artifact("package-plan.json")
    errors: list[str] = []
    if study.status != "draft":
        errors.append("ExecuTorch packages are pre-freeze artifacts and require a draft staged study")
    try:
        backend = _external_backend(study)
        tasks, task_errors, evidence = _tasks(
            study, study_sha256, backend, registration_path, model2mlir, product,
            k1_toolchain)
        errors.extend(task_errors)
    except (OSError, ValueError) as exc:
        tasks, evidence = [], {}
        errors.append(str(exc))
    plan: dict[str, Any] = {
        "version": 1, "mode": "execute" if execute else "preflight",
        "status": "blocked" if errors else "ready", "started_at": _utc_now(),
        "study": {"path": str(study_path), "sha256": study_sha256,
                  "status": study.status, "holdout_models": list(study.holdout_models)},
        "environment_policy": {
            "inherited_keys_allowed": sorted(_ALLOWED_INHERITED_ENV_KEYS),
            "all_other_inherited_keys_removed": True,
            "paper_input_environment_retained_exactly": True,
            "inherited_exact_keys_replaced": list(_EXACT_ENV_KEYS),
        },
        "evidence": evidence, "tasks": [task.to_dict() for task in tasks], "errors": errors,
    }
    _write_json(plan_path, plan)
    product.notes = f"ExecuTorch package {plan['status']}; execute={execute}; tasks={len(tasks)}"
    product.write_manifest()
    if errors:
        raise ExecuTorchPackagesNotReady(errors, product.path)
    if not execute:
        return product.path

    run_task = runner or _default_runner
    validate = validator or _validate_package
    results: list[dict[str, Any]] = []
    total_start_ns = time.monotonic_ns()
    failure: str | None = None
    for task in tasks:
        stdout_path = product.add_artifact(f"logs/{task.model.name}/fp32.stdout.log")
        stderr_path = product.add_artifact(f"logs/{task.model.name}/fp32.stderr.log")
        stdout_path.touch()
        stderr_path.touch()
        if task.output.exists() or task.work.exists():
            failure = f"refusing to overwrite package/build output for {task.model.name}"
            break
        started_at = _utc_now()
        started_ns = time.monotonic_ns()
        returncode = run_task(task, dict(task.environment), stdout_path, stderr_path)
        elapsed_ns = time.monotonic_ns() - started_ns
        result = {
            **task.to_dict(), "started_at": started_at, "finished_at": _utc_now(),
            "elapsed_ns": elapsed_ns, "returncode": returncode,
            "stdout": str(stdout_path), "stderr": str(stderr_path),
        }
        if returncode:
            result["status"] = "failed"
            failure = f"{task.model.name}/fp32: package command returned {returncode}"
        else:
            try:
                result.update(validate(task))
                result["status"] = "validated"
                product.add_artifact(str(task.output.relative_to(product.path)))
            except (ExecuTorchSessionError, OSError, ValueError, json.JSONDecodeError) as exc:
                result["status"] = "rejected"
                result["validation_error"] = str(exc)
                failure = f"{task.model.name}/fp32: {exc}"
        results.append(result)
        if failure:
            break

    plan["results"] = results
    plan["finished_at"] = _utc_now()
    plan["elapsed_ns"] = time.monotonic_ns() - total_start_ns
    if failure or len(results) != len(tasks):
        plan["status"] = "failed"
        plan["errors"] = [failure or "package set ended before all five packages completed"]
        _write_json(plan_path, plan)
        product.notes = f"ExecuTorch package build failed after {len(results)}/{len(tasks)} tasks"
        product.write_manifest()
        raise ExecuTorchPackagesNotReady(plan["errors"], product.path)

    reobservation_errors = _prepublication_reobserve(
        study_path, study_sha256, registration_path, model2mlir, product,
        k1_toolchain, tasks, evidence)
    if reobservation_errors:
        plan["status"] = "failed"
        plan["errors"] = reobservation_errors
        _write_json(plan_path, plan)
        product.notes = "ExecuTorch package inputs changed before publication"
        product.write_manifest()
        raise ExecuTorchPackagesNotReady(plan["errors"], product.path)

    # A set is published only after a final, set-wide content-address check. Earlier packages may
    # have been validated hours before the last build, so their recorded digest is not enough.
    for task, result in zip(tasks, results, strict=True):
        registered_path = Path(str(result.get("path", ""))).resolve()
        if registered_path != task.output.resolve():
            failure = f"{task.model.name}/fp32: validator returned another package path"
            break
        actual_digest = sha256_paths([registered_path])
        if result.get("sha256") != actual_digest:
            failure = (
                f"{task.model.name}/fp32: package changed after validation: "
                f"validated={result.get('sha256')} actual={actual_digest}")
            break
    if failure:
        plan["status"] = "failed"
        plan["errors"] = [failure]
        _write_json(plan_path, plan)
        product.notes = "ExecuTorch package set changed before atomic publication"
        product.write_manifest()
        raise ExecuTorchPackagesNotReady(plan["errors"], product.path)

    registered = {result["model"]: result for result in results}
    staged = copy.deepcopy(study.canonical_dict())
    backend_raw = next(row for row in staged["backends"] if row["name"] == "executorch_xnnpack")
    backend_raw["options"]["framework_source_sha256"] = tasks[0].framework_source_sha256
    for model in study.models:
        row = registered[model.name]
        backend_raw["options"]["packages"][model.name]["fp32"] = {
            "path": row["path"], "sha256": row["sha256"],
            "build_environment_sha256": row["build_environment_sha256"],
        }
    if validator is None:
        backend_raw["options"]["measurement_contracts"] = _paper_measurement_templates(
            study, backend, tasks, results, product)
    staged_path = product.path / "freeze-ready-study.yaml"
    staged_temp = product.path / ".freeze-ready-study.yaml.tmp"
    write_yaml(staged_temp, staged, header=(
        "Capture- and ExecuTorch-package-complete draft; pass to merlin-compare --freeze"))
    registration_output = product.add_artifact("package-registration.json")
    try:
        staged_payload = staged_temp.read_bytes()
        staged_sha256 = hashlib.sha256(staged_payload).hexdigest()
        os.replace(staged_temp, staged_path)
        _write_json_atomic(registration_output, {
            "version": 1, "complete": True,
            "input_study": str(study_path), "input_study_sha256": study_sha256,
            "capture_registration": str(registration_path),
            "capture_registration_sha256": evidence["capture_registration"]["sha256"],
            "study": str(staged_path), "study_sha256": staged_sha256,
            "framework_source_sha256": tasks[0].framework_source_sha256,
            "executorch_identity": tasks[0].executorch_identity,
            "model2mlir_identity": evidence["model2mlir"],
            "toolchain_identity": tasks[0].toolchain_identity,
            "external_model_sources": evidence["external_model_sources"],
            "packages": results,
        })
    except Exception:
        staged_temp.unlink(missing_ok=True)
        raise
    product.add_artifact("freeze-ready-study.yaml")
    plan["status"] = "complete"
    plan["freeze_ready_study"] = str(staged_path)
    plan["package_registration"] = str(registration_output)
    _write_json(plan_path, plan)
    product.notes = "complete immutable ExecuTorch/XNNPACK FP32 package set; validated=5/5"
    product.write_manifest()
    return product.path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="merlin-paper-executorch-packages",
        description="Preflight or build/register all five FP32 ExecuTorch/XNNPACK packages.")
    parser.add_argument("--study", type=Path, required=True,
                        help="capture-complete staged-study.yaml")
    parser.add_argument("--capture-registration", type=Path, default=None,
                        help="defaults to capture-registration.json beside --study")
    parser.add_argument("--model2mlir", type=Path, default=capture_bundle.model2mlir_root())
    parser.add_argument(
        "--k1-toolchain", type=Path,
        help="exact SpacemiT compiler prefix; otherwise explicit MERLIN_K1_TOOLCHAIN only")
    parser.add_argument("--execute", action="store_true",
                        help="build all five packages; default only writes a timestamped preflight")
    args = parser.parse_args(argv)
    try:
        output = materialize(
            args.study, args.capture_registration, args.model2mlir, execute=args.execute,
            k1_toolchain=args.k1_toolchain)
    except ExecuTorchPackagesNotReady as exc:
        print(f"merlin-paper-executorch-packages: BLOCKED — {exc}")
        print(f"  {exc.output_dir / 'package-plan.json'}")
        return 2
    print(f"merlin-paper-executorch-packages: wrote {output}")
    print(f"  {output / 'package-plan.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
