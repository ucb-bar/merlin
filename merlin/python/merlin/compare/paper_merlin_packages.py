"""Produce the complete non-ExecuTorch paper backend package set.

This module is deliberately a *handoff* between backend-specific compiler producers and the
paper measurement registry.  It never guesses a toolchain from ``.env`` and it never relabels a
cached object as another backend.  Each backend producer must supply a closed MRLNSES2 compiler
input plus a small identity receipt.  The package phase clean-replays that input, links it with the
independently authorized compiler, and publishes all 50 core-count templates atomically.

The backend-specific lowering producers are intentionally outside this module.  In particular,
absence of a promoted-policy, hand-v0, XNNPACK-routing, or OpenBLAS-routing producer receipt is a
hard package-set blocker rather than a reason to fall back to the generic Merlin object.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from merlin.common.artifacts import ProductDir, new_product
from merlin.common.paths import repo_root
from merlin.common.yaml import write_yaml

from .capture_workflow import CaptureWorkflowNotReady
from .freeze import sha256_paths
from .paper import MatrixCell, PaperStudySpec
from .paper_model_object_builder import (
    MERLIN_RECIPE,
    merlin_session_resources,
    object_build_argv,
    regenerate_model_object,
    stage_compiler_input,
)
from .paper_toolchain_authority import (
    load_toolchain_authority,
    verify_build_tool,
    write_toolchain_authority,
)

_BACKENDS = ("hand_v0_int8", "merlin_frozen", "merlin_xnnpack", "merlin_openblas")
_PRODUCER_INPUT_KIND = "paper_merlin_backend_producer_input_v1"
_AUTHORITY_RECEIPT_KIND = "paper_toolchain_authority_issuance_receipt_v1"
_PACKAGE_SET_KIND = "paper_merlin_package_registration_v1"
_BUILD_ARGV = ["{tool}", "-O2", "-std=c11", "{source:runner}",
               "{source:model_object}", "-o", "{output}"]
_HEX = frozenset("0123456789abcdef")


class MerlinPackageSetNotReady(RuntimeError):
    """The complete non-ExecuTorch package set could not be published."""

    def __init__(self, reasons: Sequence[str], output_dir: Path):
        super().__init__("; ".join(reasons))
        self.reasons = tuple(reasons)
        self.output_dir = output_dir


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _is_sha(value: object) -> bool:
    text = str(value)
    return len(text) == 64 and all(character in _HEX for character in text)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical(value) + b"\n")


def _load_json(path: Path, label: str) -> Mapping[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} is absent or unsafe")
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is invalid JSON") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _closed(value: Mapping[str, Any], fields: set[str], label: str) -> Mapping[str, Any]:
    if set(value) != fields:
        raise ValueError(
            f"{label} differs from its closed schema: "
            f"extra={sorted(set(value) - fields)} missing={sorted(fields - set(value))}")
    return value


def _safe_component(value: str, label: str) -> str:
    if not value.isascii() or not value or not value.replace("_", "").isalnum():
        raise ValueError(f"{label} is not a safe package path component: {value!r}")
    return value


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return (value if value.is_absolute() else repo_root() / value).resolve()


def issue_toolchain_authority(*, output: str | Path, authority_id: str, target: str,
                              build_tool: str | Path) -> tuple[Path, Path]:
    """Issue an independently reviewable authority and a closed invocation receipt.

    The caller must name the compiler explicitly.  No environment variable, PATH lookup, or
    repository ``.env`` file participates in authority construction.
    """
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.with_suffix(output.suffix + ".receipt.json").exists():
        raise FileExistsError("toolchain authority output already exists")
    authority = write_toolchain_authority(
        output, authority_id=authority_id, target=target,
        build_tool=Path(build_tool).resolve())
    document = load_toolchain_authority(authority, expected_target=target)
    receipt = output.with_suffix(output.suffix + ".receipt.json")
    _write_json(receipt, {
        "schema_version": 1,
        "kind": _AUTHORITY_RECEIPT_KIND,
        "status": "issued_for_independent_review",
        "authority": {"path": authority.name, "sha256": _sha(authority)},
        "authority_id": authority_id,
        "target": target,
        "tool": dict(document["tool"]),
        "issued_at": _utc_now(),
        "environment_sources": [],
    })
    return authority, receipt


def _verify_authority_receipt(authority_path: Path, receipt_path: Path,
                              *, target: str) -> Mapping[str, Any]:
    authority = load_toolchain_authority(authority_path, expected_target=target)
    receipt = _closed(_load_json(receipt_path, "toolchain authority issuance receipt"), {
        "schema_version", "kind", "status", "authority", "authority_id", "target",
        "tool", "issued_at", "environment_sources",
    }, "toolchain authority issuance receipt")
    ref = receipt["authority"]
    if not isinstance(ref, Mapping) or set(ref) != {"path", "sha256"}:
        raise ValueError("toolchain authority receipt has an invalid authority reference")
    if (receipt["schema_version"] != 1
            or receipt["kind"] != _AUTHORITY_RECEIPT_KIND
            or receipt["status"] != "issued_for_independent_review"
            or receipt["authority_id"] != authority["authority_id"]
            or receipt["target"] != target
            or receipt["tool"] != authority["tool"]
            or receipt["environment_sources"] != []
            or Path(str(ref["path"])).name != authority_path.name
            or ref["sha256"] != _sha(authority_path)):
        raise ValueError("toolchain authority issuance receipt differs from the authority")
    return authority


def _capture_registration(study_path: Path, registration_path: Path) -> Mapping[str, Any]:
    registration = _load_json(registration_path, "capture registration")
    if (registration.get("version") != 1 or registration.get("complete") is not True
            or registration.get("study_sha256") != _sha(study_path)
            or Path(str(registration.get("study", ""))).resolve() != study_path):
        raise ValueError("capture registration does not bind the staged study")
    captures = registration.get("captures")
    if not isinstance(captures, list):
        raise ValueError("capture registration has no capture rows")
    return registration


def _validate_registered_captures(study: PaperStudySpec,
                                  registration: Mapping[str, Any]) -> None:
    rows = registration["captures"]
    by_key: dict[tuple[str, str], Mapping[str, Any]] = {}
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise ValueError("capture registration row is not a mapping")
        key = (str(raw.get("model", "")), str(raw.get("precision", "")))
        if key in by_key:
            raise ValueError(f"capture registration duplicates {key[0]}/{key[1]}")
        by_key[key] = raw
    expected = {(model.name, precision) for model in study.models
                for precision in model.precisions}
    if set(by_key) != expected:
        raise ValueError("capture registration does not contain the exact study capture set")
    for model in study.models:
        for precision in model.precisions:
            artifact = model.artifacts[precision]
            path = Path(str(artifact.get("path", ""))).resolve()
            row = by_key[(model.name, precision)]
            if (row.get("status") != "validated" or row.get("path") != str(path)
                    or row.get("sha256") != artifact.get("sha256")
                    or not path.is_dir() or path.is_symlink()
                    or sha256_paths([path]) != artifact.get("sha256")):
                raise ValueError(
                    f"capture registration/study bytes differ for {model.name}/{precision}")


def _validate_promoted_campaign_binding(registration: Mapping[str, Any],
                                        promoted: Path) -> None:
    frozen = registration.get("host_campaign_freeze")
    if not isinstance(frozen, Mapping):
        raise ValueError("capture registration omits the promoted CPU-host campaign freeze")
    expected_path = Path(str(frozen.get("selected_compiler_package", ""))).resolve()
    if expected_path != promoted or promoted.is_symlink() or not promoted.is_dir():
        raise ValueError("promoted compiler differs from the capture-authorizing campaign")
    from merlin.benchharness.host_agent import (
        _submission_package_digest,
        _submission_source_digest,
    )
    manifest = yaml.safe_load((promoted / "manifest.yaml").read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping):
        raise ValueError("promoted compiler manifest is not a mapping")
    relative = Path(str(manifest.get("policy", "")))
    policy = (promoted / relative).resolve()
    if (relative.is_absolute() or ".." in relative.parts
            or not policy.is_relative_to(promoted) or policy.is_symlink()
            or not policy.is_file()
            or _sha(policy) != frozen.get("selected_policy_sha256")
            or _submission_package_digest(promoted) != frozen.get("runtime_sha256")
            or _submission_source_digest(promoted) != frozen.get("compiler_sha256")):
        raise ValueError("promoted compiler bytes differ from the capture-authorizing campaign")


def _required_cells(study: PaperStudySpec) -> tuple[MatrixCell, ...]:
    cells = tuple(cell for cell in study.matrix() if cell.backend.name in _BACKENDS)
    expected = {
        (model.name, backend.name, precision, core)
        for model in study.models
        for backend in study.backends if backend.name in _BACKENDS
        for precision in model.precisions if precision in backend.precisions
        for core in study.core_counts
    }
    actual = {(c.model.name, c.backend.name, c.precision, c.core_count) for c in cells}
    if actual != expected or len(cells) != len(expected):
        raise ValueError("study matrix does not produce the exact non-ExecuTorch cell set")
    return cells


def _package_dir(cell: MatrixCell) -> Path:
    value = cell.backend.options.get("package")
    if not isinstance(value, str) or value in {"", "unresolved"}:
        raise ValueError(f"{cell.backend.name}: backend package is unresolved")
    return _resolve(value)


def _backend_identity(cell: MatrixCell, promoted: Path) -> dict[str, Any]:
    if cell.backend.name == "merlin_frozen":
        if not promoted.is_dir() or promoted.is_symlink():
            raise ValueError(f"promoted compiler package is absent or unsafe: {promoted}")
        from merlin.benchharness.host_agent import (
            _submission_package_digest,
            _submission_source_digest,
        )
        manifest_path = promoted / "manifest.yaml"
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, Mapping) or set(manifest) != {
                "version", "build", "compiler", "policy"} or manifest["version"] != 1:
            raise ValueError("promoted compiler manifest differs from the closed submission schema")
        relative = Path(str(manifest["policy"]))
        policy = (promoted / relative).resolve()
        if (relative.is_absolute() or ".." in relative.parts
                or not policy.is_relative_to(promoted) or policy.is_symlink()
                or not policy.is_file()):
            raise ValueError("promoted compiler policy escapes or is absent")
        return {
            # freeze_study replaces the compiler backend package with this exact policy.
            "package_path": str(policy),
            "package_sha256": sha256_paths([policy]),
            "kernel_source_sha256": None,
            "run_id": "promoted_cpu_host_compiler",
            "dtype_strategy": cell.precision,
            "kernel_backend": None,
            "promoted_compiler_sha256": _submission_package_digest(promoted),
            "promoted_compiler_source_sha256": _submission_source_digest(promoted),
        }
    package = _package_dir(cell)
    if not package.is_dir() or package.is_symlink():
        raise ValueError(f"{cell.backend.name}: compiler package is absent or unsafe: {package}")
    # Loading is an important semantic check in addition to the content address.
    from merlin.mining.registry import load_rvv_package
    loaded = load_rvv_package(package)
    if cell.precision == "w8a8" and not loaded.is_int8:
        raise ValueError(f"{cell.key}: W8A8 cell names a non-int8 compiler package")
    if cell.precision == "fp32" and loaded.is_int8:
        raise ValueError(f"{cell.key}: FP32 cell names an int8 compiler package")
    sources = cell.backend.options.get("source_paths", ()) or ()
    source_paths = [_resolve(str(value)) for value in sources]
    missing = [str(path) for path in source_paths if not path.exists() or path.is_symlink()]
    if missing:
        raise ValueError(f"{cell.backend.name}: backend source closure is absent or unsafe: {missing}")
    return {
        "package_path": str(package),
        "package_sha256": sha256_paths([package]),
        "kernel_source_sha256": sha256_paths(source_paths) if source_paths else None,
        "run_id": loaded.run_id,
        "dtype_strategy": loaded.dtype_strategy,
        "kernel_backend": cell.backend.options.get("kernel_backend"),
        "promoted_compiler_sha256": None,
        "promoted_compiler_source_sha256": None,
    }


def _producer_receipt_path(root: Path, cell: MatrixCell) -> Path:
    return (root / _safe_component(cell.backend.name, "backend")
            / _safe_component(cell.model.name, "model")
            / _safe_component(cell.precision, "precision") / "producer-input.json")


def _producer_input(root: Path, cell: MatrixCell, *, identity: Mapping[str, Any],
                    source_identity: str, runtime_sha256: str) -> Path:
    receipt_path = _producer_receipt_path(root, cell)
    receipt = _closed(_load_json(receipt_path, f"{cell.key} producer input"), {
        "schema_version", "kind", "status", "cell", "compiler_input",
        "compiler_source_sha256", "compiler_package_sha256", "kernel_source_sha256",
        "capture_sha256", "runtime_artifact_sha256", "kernel_backend",
        "promoted_compiler_sha256", "promoted_compiler_source_sha256",
    }, f"{cell.key} producer input")
    ref = receipt["compiler_input"]
    if not isinstance(ref, Mapping) or set(ref) != {"path", "sha256"}:
        raise ValueError(f"{cell.key}: producer compiler_input reference is invalid")
    relative = Path(str(ref["path"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{cell.key}: producer compiler_input escapes its receipt")
    compiler_input = (receipt_path.parent / relative).resolve()
    if (not compiler_input.is_relative_to(receipt_path.parent.resolve())
            or compiler_input.is_symlink() or not compiler_input.is_file()
            or _sha(compiler_input) != ref["sha256"]):
        raise ValueError(f"{cell.key}: producer compiler_input identity differs")
    expected = {
        "model": cell.model.name, "backend": cell.backend.name, "precision": cell.precision,
    }
    if (receipt["schema_version"] != 1 or receipt["kind"] != _PRODUCER_INPUT_KIND
            or receipt["status"] != "producer_complete"
            or receipt["cell"] != expected
            or receipt["compiler_source_sha256"] != source_identity
            or receipt["compiler_package_sha256"] != identity["package_sha256"]
            or receipt["kernel_source_sha256"] != identity["kernel_source_sha256"]
            or receipt["capture_sha256"] != cell.model.artifacts[cell.precision]["sha256"]
            or receipt["runtime_artifact_sha256"] != runtime_sha256
            or receipt["kernel_backend"] != identity["kernel_backend"]
            or receipt["promoted_compiler_sha256"] != identity["promoted_compiler_sha256"]
            or receipt["promoted_compiler_source_sha256"]
            != identity["promoted_compiler_source_sha256"]):
        raise ValueError(f"{cell.key}: producer input is for a different frozen backend identity")
    # This resolves and validates the complete public producer graph without opening private I/O.
    merlin_session_resources(compiler_input)
    return compiler_input


def register_backend_producer_input(
        *, study_path: str | Path, promoted_compiler: str | Path,
        runtime_artifact: str | Path, producer_root: str | Path,
        backend: str, model: str, precision: str, compiler_input: str | Path) -> Path:
    """Deep-retain one completed backend producer graph and bind its frozen identity.

    This function does not lower a model and cannot turn a generic object into a backend result.
    It is the final operation of an actual backend producer after that producer has emitted the
    clean-replayable MRLNSES2 graph.
    """
    study = PaperStudySpec.from_yaml(Path(study_path).resolve())
    matches = [
        cell for cell in _required_cells(study)
        if (cell.backend.name, cell.model.name, cell.precision, cell.core_count)
        == (backend, model, precision, study.core_counts[0])
    ]
    if len(matches) != 1:
        raise ValueError("producer identity does not name exactly one required model/backend/precision")
    cell = matches[0]
    promoted = Path(promoted_compiler).resolve()
    runtime = Path(runtime_artifact).resolve()
    if runtime.is_symlink() or not runtime.is_file():
        raise ValueError("explicit runtime artifact is absent or unsafe")
    identity = _backend_identity(cell, promoted)
    source_identity = sha256_paths([
        repo_root() / "merlin/python/merlin", repo_root() / "pyproject.toml"])
    output = _producer_receipt_path(Path(producer_root).resolve(), cell)
    if output.exists() or output.parent.exists():
        raise FileExistsError(f"producer input output already exists: {output.parent}")
    output.parent.mkdir(parents=True)
    retained = stage_compiler_input(
        Path(compiler_input).resolve(), output.parent / "compiler-input/compiler-input.json",
        recipe=MERLIN_RECIPE)
    merlin_session_resources(retained)
    _write_json(output, {
        "schema_version": 1, "kind": _PRODUCER_INPUT_KIND,
        "status": "producer_complete",
        "cell": {"model": model, "backend": backend, "precision": precision},
        "compiler_input": {"path": retained.relative_to(output.parent).as_posix(),
                           "sha256": _sha(retained)},
        "compiler_source_sha256": source_identity,
        "compiler_package_sha256": identity["package_sha256"],
        "kernel_source_sha256": identity["kernel_source_sha256"],
        "capture_sha256": cell.model.artifacts[cell.precision]["sha256"],
        "runtime_artifact_sha256": _sha(runtime),
        "kernel_backend": identity["kernel_backend"],
        "promoted_compiler_sha256": identity["promoted_compiler_sha256"],
        "promoted_compiler_source_sha256": identity["promoted_compiler_source_sha256"],
    })
    _producer_input(
        Path(producer_root).resolve(), cell, identity=identity,
        source_identity=source_identity, runtime_sha256=_sha(runtime))
    return output


def _copy_file(source: Path, destination: Path) -> Path:
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"package resource is absent or unsafe: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination


def _link(tool: Path, runner: Path, model_object: Path, output: Path, timeout: int) -> None:
    completed = subprocess.run(
        [str(tool), "-O2", "-std=c11", str(runner), str(model_object), "-o", str(output)],
        capture_output=True, timeout=timeout, check=False,
        env={"LANG": "C", "LC_ALL": "C", "TZ": "UTC", "PATH": ""},
        cwd=output.parent, stdin=subprocess.DEVNULL, close_fds=True)
    if completed.returncode or not output.is_file():
        detail = completed.stderr.decode("utf-8", errors="replace")[-1000:]
        raise ValueError(f"package result link failed ({completed.returncode}): {detail}")


def _build_package(cell: MatrixCell, *, root: Path, compiler_input: Path,
                   identity: Mapping[str, Any], source_identity: str,
                   runtime_artifact: Path, authority_path: Path,
                   authority: Mapping[str, Any]) -> dict[str, Any]:
    package = root / cell.backend.name / cell.model.name / cell.precision
    package.mkdir(parents=True, exist_ok=False)
    staged_input = stage_compiler_input(
        compiler_input, package / "compiler-input/compiler-input.json", recipe=MERLIN_RECIPE)
    resources = merlin_session_resources(staged_input)
    model_object = package / "resources/model-object.o"
    regeneration = regenerate_model_object(
        recipe=MERLIN_RECIPE, registry_id="merlin_compile_v1", target="k1",
        compiler_input=staged_input, tool=Path(str(authority["tool"]["path"])),
        output=model_object, source_identity_sha256=source_identity,
        capture_sha256=str(cell.model.artifacts[cell.precision]["sha256"]),
        runtime_artifact_sha256=_sha(runtime_artifact),
        timeout_seconds=float(cell.backend.options["timeout"]))
    runner = _copy_file(resources.runner_source, package / "resources/session-runner.c")
    tool = _copy_file(Path(str(authority["tool"]["path"])), package / "resources/build-tool")
    tool.chmod(tool.stat().st_mode | 0o100)
    runtime = _copy_file(runtime_artifact, package / "resources/runtime-artifact")
    verify_build_tool(
        tool, authority_path=authority_path, authority_sha256=_sha(authority_path), target="k1",
        expected_identity_sha256=str(authority["tool"]["identity_sha256"]))
    executable = package / "resources/expected-cell"
    _link(tool, runner, model_object, executable, int(cell.backend.options["timeout"]))
    from . import paper_model_object_builder
    builder = Path(paper_model_object_builder.__file__).resolve()
    builder_sha = _sha(builder)
    hashes = {
        "compiler_input": _sha(staged_input), "model_object": _sha(model_object),
        "runner": _sha(runner), "build_tool": _sha(tool),
        "session_descriptor": resources.descriptor.sha256,
    }
    receipt = package / "resources/package-receipt.json"
    _write_json(receipt, {
        "schema_version": 3, "kind": "paper_backend_package_receipt_v3",
        "status": "finalized", "registry_id": "merlin_compile_v1",
        "build_adapter": "merlin_session_abi_c_v1",
        "cell": {"model": cell.model.name, "backend": cell.backend.name,
                 "precision": cell.precision},
        "package_identity_sha256": identity["package_sha256"],
        "compiler_or_framework_source_sha256": source_identity,
        "capture_sha256": cell.model.artifacts[cell.precision]["sha256"],
        "runtime_artifact_sha256": _sha(runtime),
        "runner_source_sha256": hashes["runner"],
        "model_object_sha256": hashes["model_object"],
        "compiler_input_sha256": hashes["compiler_input"],
        "object_builder_source_sha256": builder_sha,
        "object_recipe": MERLIN_RECIPE,
        "object_build_argv": object_build_argv(MERLIN_RECIPE),
        "generated_model_source_sha256": regeneration["generated_source_sha256"],
        "build_tool_sha256": hashes["build_tool"],
        "build_source_identity_sha256": _canonical_sha({
            "compiler_input": hashes["compiler_input"],
            "model_object": hashes["model_object"],
            "object_builder": builder_sha, "runner": hashes["runner"],
        }),
        "build_argv": _BUILD_ARGV,
        "result_executable_sha256": _sha(executable), "finalized_at": _utc_now(),
        "session_protocol": "MRLNSES2",
        "session_descriptor_sha256": hashes["session_descriptor"],
    })
    common_resources = {
        "package_receipt": receipt, "compiler_input": staged_input,
        "model_object": model_object, "build_tool": tool, "runtime_artifact": runtime,
    }
    return {"package": package, "resources": common_resources, "identity": dict(identity),
            "receipt": receipt, "executable": executable, "regeneration": regeneration}


def _write_template(package: Path, cell: MatrixCell, resources: Mapping[str, Path]) -> Path:
    core = int(cell.core_count)
    path = package / f"template-{core}c.yaml"
    refs: dict[str, dict[str, str]] = {
        name: {"path": value.relative_to(path.parent).as_posix(), "sha256": _sha(value)}
        for name, value in resources.items()
    }
    _write_json(path.with_suffix(".identity.json"), {
        "cell": {"model": cell.model.name, "backend": cell.backend.name,
                 "precision": cell.precision, "core_count": core},
        "resources_sha256": _canonical_sha(refs),
    })
    template = {
        "schema_version": 3, "kind": "paper_backend_measurement_template_v3",
        "status": "frozen", "registry_id": "merlin_compile_v1",
        "backend_adapter": "merlin_compile",
        "cell": {"model": cell.model.name, "backend": cell.backend.name,
                 "precision": cell.precision, "core_count": core},
        "resources": refs, "environment": {},
        "execution": {"mode": "rvv", "core_ids": list(range(core)),
                      "require_worker_threads": core > 1},
        "memory_policy": str(cell.model.memory["policies"][0]),
        "timeout_seconds": int(cell.backend.options["timeout"]),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(template, sort_keys=True), encoding="utf-8")
    return path


def _validate_template(path: Path, cell: MatrixCell, *, identity: Mapping[str, Any],
                       source_identity: str, regeneration: Mapping[str, object]) -> None:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{cell.key}: emitted template is not a mapping")
    _closed(value, {
        "schema_version", "kind", "status", "registry_id", "backend_adapter", "cell",
        "resources", "environment", "execution", "memory_policy", "timeout_seconds",
    }, f"{cell.key} emitted template")
    expected_cell = {"model": cell.model.name, "backend": cell.backend.name,
                     "precision": cell.precision, "core_count": cell.core_count}
    if (value["schema_version"] != 3
            or value["kind"] != "paper_backend_measurement_template_v3"
            or value["status"] != "frozen"
            or value["registry_id"] != "merlin_compile_v1"
            or value["backend_adapter"] != "merlin_compile"
            or value["cell"] != expected_cell):
        raise ValueError(f"{cell.key}: emitted template identity differs")
    resources = value["resources"]
    if not isinstance(resources, Mapping) or set(resources) != {
            "package_receipt", "compiler_input", "model_object", "build_tool",
            "runtime_artifact"}:
        raise ValueError(f"{cell.key}: emitted template resource set differs")
    resolved: dict[str, Path] = {}
    for name, raw in resources.items():
        if not isinstance(raw, Mapping) or set(raw) != {"path", "sha256"}:
            raise ValueError(f"{cell.key}: emitted resource {name} has an invalid reference")
        relative = Path(str(raw["path"]))
        candidate = (path.parent / relative).resolve()
        if (relative.is_absolute() or ".." in relative.parts
                or not candidate.is_relative_to(path.parent.resolve())
                or candidate.is_symlink() or not candidate.is_file()
                or _sha(candidate) != raw["sha256"]):
            raise ValueError(f"{cell.key}: emitted resource {name} differs")
        resolved[str(name)] = candidate
    session = merlin_session_resources(resolved["compiler_input"])
    from . import paper_model_object_builder
    from .paper_contract_registry import _package_receipt
    resource_hashes = {
        "compiler_input": _sha(resolved["compiler_input"]),
        "model_object": _sha(resolved["model_object"]),
        "build_tool": _sha(resolved["build_tool"]),
        "runner": _sha(session.runner_source),
        "object_builder": _sha(Path(paper_model_object_builder.__file__).resolve()),
        "session_descriptor": session.descriptor.sha256,
    }
    _package_receipt(
        resolved["package_receipt"], registry_id="merlin_compile_v1", cell=cell,
        source_identity=source_identity, package_identity=str(identity["package_sha256"]),
        runtime_artifact_sha256=_sha(resolved["runtime_artifact"]),
        resource_hashes=resource_hashes, target="k1", regeneration=regeneration)


def materialize(study_path: str | Path, capture_registration_path: str | Path,
                promoted_compiler: str | Path, producer_inputs: str | Path,
                runtime_artifact: str | Path, toolchain_authority: str | Path,
                authority_receipt: str | Path, *, execute: bool = False,
                product: ProductDir | None = None) -> Path:
    """Plan or atomically publish the complete 25-package/50-template Merlin set."""
    study_path = Path(study_path).resolve()
    registration_path = Path(capture_registration_path).resolve()
    promoted = Path(promoted_compiler).resolve()
    producer_root = Path(producer_inputs).resolve()
    runtime = Path(runtime_artifact).resolve()
    authority_path = Path(toolchain_authority).resolve()
    authority_receipt_path = Path(authority_receipt).resolve()
    study = PaperStudySpec.from_yaml(study_path)
    product = product or new_product(
        "paper-merlin-packages", version=1, target=study.target,
        sources=[str(study_path), str(registration_path), str(promoted),
                 str(producer_root), str(runtime), str(authority_path),
                 str(authority_receipt_path)])
    errors: list[str] = []
    cells = _required_cells(study)
    try:
        if study.status != "draft":
            raise ValueError("Merlin package production requires a capture-complete draft study")
        registration = _capture_registration(study_path, registration_path)
        _validate_registered_captures(study, registration)
        _validate_promoted_campaign_binding(registration, promoted)
        authority = _verify_authority_receipt(
            authority_path, authority_receipt_path, target=study.target)
        verify_build_tool(
            str(authority["tool"]["path"]), authority_path=authority_path,
            authority_sha256=_sha(authority_path), target=study.target,
            expected_identity_sha256=str(authority["tool"]["identity_sha256"]))
        if runtime.is_symlink() or not runtime.is_file():
            raise ValueError("explicit runtime artifact is absent or unsafe")
    except (OSError, ValueError) as exc:
        errors.append(str(exc))
        authority = {}

    source_identity = sha256_paths([
        repo_root() / "merlin/python/merlin", repo_root() / "pyproject.toml"])
    identities: dict[tuple[str, str], dict[str, Any]] = {}
    inputs: dict[tuple[str, str, str], Path] = {}
    package_cells: dict[tuple[str, str, str], MatrixCell] = {}
    for cell in cells if not errors else ():
        key = (cell.backend.name, cell.model.name, cell.precision)
        if key in package_cells:
            continue
        package_cells[key] = cell
        try:
            identity_key = (cell.backend.name, cell.precision)
            if identity_key not in identities:
                identities[identity_key] = _backend_identity(cell, promoted)
            identity = identities[identity_key]
            inputs[key] = _producer_input(
                producer_root, cell, identity=identity, source_identity=source_identity,
                runtime_sha256=_sha(runtime))
        except (OSError, ValueError) as exc:
            errors.append(str(exc))

    expected_packages = len({(c.backend.name, c.model.name, c.precision) for c in cells})
    expected_templates = len(cells)
    plan = {
        "schema_version": 1, "kind": "paper_merlin_package_plan_v1",
        "mode": "execute" if execute else "preflight",
        "status": "blocked" if errors else "ready",
        "study": {"path": str(study_path), "sha256": _sha(study_path)},
        "capture_registration": {"path": str(registration_path),
                                 "sha256": _sha(registration_path)},
        "promoted_compiler": {"path": str(promoted),
                              "sha256": sha256_paths([promoted]) if promoted.exists() else None},
        "producer_inputs": str(producer_root),
        "runtime_artifact": {"path": str(runtime),
                             "sha256": _sha(runtime) if runtime.is_file() else None},
        "toolchain_authority": {"path": str(authority_path),
                                "sha256": _sha(authority_path) if authority_path.is_file() else None},
        "authority_receipt": {"path": str(authority_receipt_path),
                              "sha256": (_sha(authority_receipt_path)
                                         if authority_receipt_path.is_file() else None)},
        "compiler_source_sha256": source_identity,
        "required_backends": list(_BACKENDS),
        "required_packages": expected_packages,
        "required_templates": expected_templates,
        "producer_inputs_validated": len(inputs),
        "errors": list(dict.fromkeys(errors)),
    }
    plan_path = product.add_artifact("package-plan.json")
    _write_json(plan_path, plan)
    product.notes = (f"non-ExecuTorch package plan {plan['status']}; "
                     f"producer inputs={len(inputs)}/{expected_packages}")
    product.write_manifest()
    if errors:
        raise MerlinPackageSetNotReady(plan["errors"], product.path)
    if not execute:
        return product.path

    staging = product.path / ".package-set.staging"
    published = product.path / "package-set"
    if staging.exists() or published.exists():
        raise FileExistsError("package-set staging or publication path already exists")
    staging.mkdir(parents=True)
    built: dict[tuple[str, str, str], dict[str, Any]] = {}
    try:
        for key, base_cell in sorted(package_cells.items()):
            identity = identities[(base_cell.backend.name, base_cell.precision)]
            built[key] = _build_package(
                base_cell, root=staging / "packages", compiler_input=inputs[key],
                identity=identity, source_identity=source_identity,
                runtime_artifact=runtime, authority_path=authority_path, authority=authority)
        refs: dict[str, dict[str, dict[str, dict[str, str]]]] = {}
        template_rows: list[dict[str, Any]] = []
        for cell in cells:
            key = (cell.backend.name, cell.model.name, cell.precision)
            package = built[key]
            template = _write_template(package["package"], cell, package["resources"])
            _validate_template(
                template, cell, identity=package["identity"], source_identity=source_identity,
                regeneration=package["regeneration"])
            # The staged tree is renamed as one unit.  Bind the final absolute path now; no study
            # carrying a staging path is ever published.
            ref = {"path": str(published / template.relative_to(staging)),
                   "sha256": _sha(template)}
            refs.setdefault(cell.backend.name, {}).setdefault(
                cell.model.name, {}).setdefault(cell.precision, {})[str(cell.core_count)] = ref
            template_rows.append({"cell": cell.key, **ref})
        if len(built) != expected_packages or len(template_rows) != expected_templates:
            raise ValueError("built package/template count differs from the complete matrix")

        staged = copy.deepcopy(study.canonical_dict())
        for backend in staged["backends"]:
            if backend["name"] not in _BACKENDS:
                continue
            original = next(row for row in study.backends if row.name == backend["name"])
            representative = next(c for c in cells if c.backend.name == backend["name"])
            identity = identities[(backend["name"], representative.precision)]
            backend["options"]["measurement_contracts"] = refs[backend["name"]]
            if backend["name"] == "merlin_frozen":
                backend["options"]["package"] = identity["package_path"]
                backend["options"]["package_sha256"] = identity["package_sha256"]
            elif original.options.get("package") not in {None, "unresolved"}:
                backend["options"]["package_sha256"] = identity["package_sha256"]
            if identity["kernel_source_sha256"] is not None:
                backend["options"]["kernel_source_sha256"] = identity["kernel_source_sha256"]
        ready_staging = staging / "package-ready-study.yaml"
        write_yaml(ready_staging, staged, header=(
            "Capture-complete draft with all non-ExecuTorch package templates; "
            "ExecuTorch registration remains a separate prerequisite"))
        registration_staging = staging / "package-registration.json"
        _write_json(registration_staging, {
            "schema_version": 1, "kind": _PACKAGE_SET_KIND, "complete": True,
            "study": "package-ready-study.yaml", "study_sha256": _sha(ready_staging),
            "source_study_sha256": _sha(study_path),
            "capture_registration_sha256": _sha(registration_path),
            "toolchain_authority_sha256": _sha(authority_path),
            "compiler_source_sha256": source_identity,
            "packages": [
                {"backend": key[0], "model": key[1], "precision": key[2],
                 "receipt_sha256": _sha(value["receipt"]),
                 "executable_sha256": _sha(value["executable"])}
                for key, value in sorted(built.items())
            ],
            "templates": sorted(template_rows, key=lambda row: row["cell"]),
            "completed_at": _utc_now(),
        })
        # Reparse before the single atomic publication rename.
        parsed = PaperStudySpec.from_yaml(ready_staging)
        if len(_required_cells(parsed)) != expected_templates:
            raise ValueError("package-ready study no longer has the complete matrix")
        os.replace(staging, published)
    except Exception:
        # Keep failed bytes as diagnostics, never under the publication name.
        raise
    plan["status"] = "complete"
    plan["package_set"] = str(published)
    plan["package_registration"] = str(published / "package-registration.json")
    plan["package_ready_study"] = str(published / "package-ready-study.yaml")
    _write_json(plan_path, plan)
    product.notes = (f"complete non-ExecuTorch package set; packages={expected_packages}; "
                     f"templates={expected_templates}")
    product.write_manifest()
    return product.path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="merlin-paper-merlin-packages")
    commands = parser.add_subparsers(dest="command", required=True)
    authority = commands.add_parser(
        "issue-authority", help="issue the explicit compiler authority and review receipt")
    authority.add_argument("--output", type=Path, required=True)
    authority.add_argument("--authority-id", required=True)
    authority.add_argument("--target", default="k1")
    authority.add_argument("--build-tool", type=Path, required=True)
    packages = commands.add_parser(
        "packages", help="plan or build the complete non-ExecuTorch package set")
    packages.add_argument("--study", type=Path, required=True)
    packages.add_argument("--capture-registration", type=Path, required=True)
    packages.add_argument("--promoted-compiler", type=Path, required=True)
    packages.add_argument("--producer-inputs", type=Path, required=True)
    packages.add_argument("--runtime-artifact", type=Path, required=True)
    packages.add_argument("--toolchain-authority", type=Path, required=True)
    packages.add_argument("--authority-receipt", type=Path, required=True)
    packages.add_argument("--execute", action="store_true")
    register = commands.add_parser(
        "register-producer-input",
        help="deep-retain one completed backend producer's MRLNSES2 graph")
    register.add_argument("--study", type=Path, required=True)
    register.add_argument("--promoted-compiler", type=Path, required=True)
    register.add_argument("--runtime-artifact", type=Path, required=True)
    register.add_argument("--producer-inputs", type=Path, required=True)
    register.add_argument("--backend", choices=_BACKENDS, required=True)
    register.add_argument("--model", required=True)
    register.add_argument("--precision", choices=("w8a8", "fp32"), required=True)
    register.add_argument("--compiler-input", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "issue-authority":
        authority_path, receipt = issue_toolchain_authority(
            output=args.output, authority_id=args.authority_id, target=args.target,
            build_tool=args.build_tool)
        print(f"merlin-paper-merlin-packages: wrote {authority_path}")
        print(f"merlin-paper-merlin-packages: wrote {receipt}")
        return 0
    if args.command == "register-producer-input":
        output = register_backend_producer_input(
            study_path=args.study, promoted_compiler=args.promoted_compiler,
            runtime_artifact=args.runtime_artifact, producer_root=args.producer_inputs,
            backend=args.backend, model=args.model, precision=args.precision,
            compiler_input=args.compiler_input)
        print(f"merlin-paper-merlin-packages: wrote {output}")
        return 0
    try:
        output = materialize(
            args.study, args.capture_registration, args.promoted_compiler,
            args.producer_inputs, args.runtime_artifact, args.toolchain_authority,
            args.authority_receipt, execute=args.execute)
    except (CaptureWorkflowNotReady, MerlinPackageSetNotReady) as exc:
        print(f"merlin-paper-merlin-packages: BLOCKED — {exc}")
        print(f"evidence: {exc.output_dir}")
        return 2
    print(f"merlin-paper-merlin-packages: wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
