"""Closed production registry for controller measurement contracts.

Backend templates are content-addressed by the frozen study.  They contain only runtime/build
resources; study identity, cell identity, session, thresholds, artifact identity, and provenance
are re-derived here from :class:`PaperStudySpec` and cannot be injected by a live caller.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

from merlin.common.paths import repo_root

from .paper import MatrixCell, PaperStudySpec
from .paper_model_object_builder import (
    EXECUTORCH_RECIPE,
    MERLIN_RECIPE,
    executorch_session_resources,
    expected_recipe,
    merlin_session_resources,
    object_build_argv,
    regenerate_model_object,
    stage_compiler_input,
)
from .paper_toolchain_authority import load_toolchain_authority, verify_build_tool

_REGISTRY = MappingProxyType({
    ("merlin_compile", "compiler"): "merlin_compile_v1",
    ("merlin_compile", "kernel_swap"): "merlin_compile_v1",
    ("merlin_compile", "frozen_baseline"): "merlin_compile_v1",
    ("executorch", "external_runtime"): "executorch_v1",
})
_IDENTITY_FIELDS = (
    "timestamp", "git_sha", "study_label", "target", "model", "checkpoint", "fidelity",
    "backend", "runtime", "precision", "quantization", "core_count",
)
_BUILD_ADAPTERS = {
    "merlin_compile_v1": "merlin_model_abi_c_v1",
    "executorch_v1": "executorch_model_abi_c_v1",
}
_MERLIN_BUILD_ADAPTER = "merlin_session_abi_c_v1"
_MERLIN_SESSION_PROTOCOL = "MRLNSES2"
_BUILD_ARGV = ["{tool}", "-O2", "-std=c11", "{source:runner}",
               "{source:model_object}", "-o", "{output}"]
_EXECUTORCH_BUILD_ARGV = [
    "verify_executorch_sealed_session", "{source:model_object}", "{output}"]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _closed(value: object, fields: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    extra, missing = sorted(set(value) - fields), sorted(fields - set(value))
    if extra or missing:
        raise ValueError(f"{label} is closed; unrecognized={extra} missing={missing}")
    return value


def _template_ref(cell: MatrixCell) -> tuple[Path, Mapping[str, Any]]:
    contracts = cell.backend.options.get("measurement_contracts")
    if not isinstance(contracts, Mapping):
        raise ValueError(
            f"{cell.key}: frozen backend lacks measurement_contracts; live execution is blocked")
    try:
        ref = _closed(
            contracts[cell.model.name][cell.precision][str(cell.core_count)],
            {"path", "sha256"}, f"{cell.key} measurement template ref")
    except (KeyError, TypeError) as error:
        raise ValueError(f"{cell.key}: frozen measurement template is absent") from error
    path = Path(str(ref["path"]))
    path = path if path.is_absolute() else repo_root() / path
    path = path.resolve()
    if not path.is_file() or _sha(path) != ref["sha256"]:
        raise ValueError(f"{cell.key}: frozen measurement template digest differs")
    return path, ref


def _resolve_ref(root: Path, ref: object, label: str) -> Path:
    ref = _closed(ref, {"path", "sha256"}, label)
    relative = Path(str(ref["path"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label} escapes its frozen template")
    source = (root / relative).resolve()
    try:
        source.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"{label} escapes its frozen template") from error
    if not source.is_file() or source.is_symlink() or _sha(source) != ref["sha256"]:
        raise ValueError(f"{label} digest differs")
    return source


def _copy_ref(root: Path, ref: object, destination: Path, label: str,
              staging: Path) -> dict[str, str]:
    source = _resolve_ref(root, ref, label)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return {"path": destination.relative_to(staging).as_posix(), "sha256": _sha(destination)}


def _copy_study_ref(ref: object, destination: Path, label: str,
                    staging: Path) -> dict[str, str]:
    """Retain an I/O authority file named by the canonical frozen study."""
    ref = _closed(ref, {"path", "sha256"}, label)
    source = Path(str(ref["path"]))
    source = source if source.is_absolute() else repo_root() / source
    source = source.resolve()
    if not source.is_file() or source.is_symlink() or _sha(source) != ref["sha256"]:
        raise ValueError(f"{label} digest differs")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return {"path": destination.relative_to(staging).as_posix(), "sha256": _sha(destination)}


def _measurement_io(spec: PaperStudySpec, cell: MatrixCell) -> Mapping[str, Any]:
    try:
        value = spec.freeze["measurement_io"][cell.backend.name][cell.model.name][cell.precision]
    except (KeyError, TypeError) as error:
        raise ValueError(f"{cell.key}: canonical freeze lacks measurement_io") from error
    value = _closed(value, {"artifact", "inputs", "session_manifest", "reference_output",
                            "generation_receipt"},
                    f"{cell.key} measurement_io")
    if not isinstance(value["inputs"], Mapping) or not value["inputs"]:
        raise ValueError(f"{cell.key}: canonical measurement inputs are absent")
    return value


def _oracle(cell: MatrixCell) -> dict[str, Any]:
    metric = cell.model.quality["metric"]
    kind = "int64_top1" if metric == "top1_agreement" else "float32_cosine"
    threshold = cell.model.quality.get("cosine_min")
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
        raise ValueError(f"{cell.key}: model quality threshold is unresolved")
    return {"kind": kind, "metric": metric, "threshold": float(threshold),
            "scope": "trajectory", "steps": cell.model.session.observations}


def _package_receipt(path: Path, *, registry_id: str, cell: MatrixCell,
                     source_identity: str, package_identity: str,
                     runtime_artifact_sha256: str, resource_hashes: Mapping[str, str],
                     target: str, regeneration: Mapping[str, object]
                     ) -> Mapping[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    fields = {
        "schema_version", "kind", "status", "registry_id", "build_adapter", "cell",
        "package_identity_sha256", "compiler_or_framework_source_sha256",
        "capture_sha256", "runtime_artifact_sha256", "runner_source_sha256",
        "model_object_sha256", "compiler_input_sha256", "object_builder_source_sha256",
        "object_recipe", "object_build_argv", "generated_model_source_sha256",
        "build_tool_sha256", "build_source_identity_sha256", "build_argv",
        "result_executable_sha256", "finalized_at",
    }
    merlin_session = isinstance(document, Mapping) and document.get("object_recipe") == MERLIN_RECIPE
    if merlin_session:
        fields.update({"session_protocol", "session_descriptor_sha256"})
    value = _closed(document, fields, "backend package receipt")
    expected_cell = {"model": cell.model.name, "backend": cell.backend.name,
                     "precision": cell.precision}
    expected_source_set = _canonical_sha({
        "compiler_input": resource_hashes["compiler_input"],
        "model_object": resource_hashes["model_object"],
        "object_builder": resource_hashes["object_builder"],
        "runner": resource_hashes["runner"],
    })
    expected_version = 3 if merlin_session else 2
    expected_kind = ("paper_backend_package_receipt_v3" if merlin_session
                     else "paper_backend_package_receipt_v2")
    executorch_sealed = value.get("object_recipe") == EXECUTORCH_RECIPE
    expected_adapter = (_MERLIN_BUILD_ADAPTER if merlin_session else
                        "executorch_sealed_session_v1" if executorch_sealed else
                        _BUILD_ADAPTERS[registry_id])
    if (value["schema_version"] != expected_version or value["kind"] != expected_kind
            or value["status"] != "finalized" or value["registry_id"] != registry_id
            or value["build_adapter"] != expected_adapter
            or value["cell"] != expected_cell
            or value["package_identity_sha256"] != package_identity
            or value["compiler_or_framework_source_sha256"] != source_identity
            or value["capture_sha256"] != cell.model.artifacts[cell.precision]["sha256"]
            or value["runtime_artifact_sha256"] != runtime_artifact_sha256
            or value["runner_source_sha256"] != resource_hashes["runner"]
            or value["model_object_sha256"] != resource_hashes["model_object"]
            or value["compiler_input_sha256"] != resource_hashes["compiler_input"]
            or value["object_builder_source_sha256"] != resource_hashes["object_builder"]
            or value["object_recipe"] != expected_recipe(registry_id, target)
            or value["object_build_argv"] != object_build_argv(str(value["object_recipe"]))
            or value["generated_model_source_sha256"] != regeneration["generated_source_sha256"]
            or value["model_object_sha256"] != regeneration["model_object_sha256"]
            or value["build_tool_sha256"] != resource_hashes["build_tool"]
            or value["build_source_identity_sha256"] != expected_source_set
            or value["build_argv"] != (
                _EXECUTORCH_BUILD_ARGV if executorch_sealed else _BUILD_ARGV)
            or not isinstance(value["finalized_at"], str)
            or not value["finalized_at"].strip()
            or not isinstance(value["result_executable_sha256"], str)
            or len(value["result_executable_sha256"]) != 64):
        raise ValueError(f"{cell.key}: backend package receipt linkage differs")
    if merlin_session and (
            value["session_protocol"] != _MERLIN_SESSION_PROTOCOL
            or value["session_descriptor_sha256"] != resource_hashes["session_descriptor"]):
        raise ValueError(f"{cell.key}: Merlin package session ABI linkage differs")
    return value


def build_registered_contract(spec: PaperStudySpec, cell: MatrixCell, *, run_id: str,
                              timestamp: str, git_sha: str, staging_dir: Path,
                              base_result: Mapping[str, Any]) -> Path:
    """Materialize the one contract selected by the frozen backend registry."""
    registry_id = _REGISTRY.get((cell.backend.adapter, cell.backend.kind))
    if registry_id is None:
        raise ValueError(
            f"{cell.key}: backend adapter/kind is not in the production measurement registry")
    template_path, _ = _template_ref(cell)
    template = _closed(yaml.safe_load(template_path.read_text(encoding="utf-8")), {
        "schema_version", "kind", "status", "registry_id", "backend_adapter", "cell",
        "resources", "environment", "execution", "memory_policy", "timeout_seconds",
    }, "backend measurement template")
    template_cell = _closed(template["cell"], {"model", "backend", "precision", "core_count"},
                            "backend measurement template cell")
    expected_cell = {"model": cell.model.name, "backend": cell.backend.name,
                     "precision": cell.precision, "core_count": cell.core_count}
    oracle = _oracle(cell)
    if (template["schema_version"] != 3
            or template["kind"] != "paper_backend_measurement_template_v3"
            or template["status"] != "frozen" or template["registry_id"] != registry_id
            or template["backend_adapter"] != cell.backend.adapter
            or dict(template_cell) != expected_cell
            or template["memory_policy"] not in cell.model.memory["policies"]
            or template["timeout_seconds"] != cell.backend.options.get("timeout")):
        raise ValueError(f"{cell.key}: frozen measurement template differs from study semantics")
    provenance = {key: value for key, value in base_result["provenance"].items()
                  if key not in {"vlen_bits", "vlen_source", "board_conditions"}}
    if registry_id == "merlin_compile_v1":
        provenance["package_sha256"] = cell.backend.options.get("package_sha256")
        if cell.backend.options.get("kernel_source_sha256") is not None:
            provenance["kernel_source_sha256"] = cell.backend.options["kernel_source_sha256"]
        source_identity = provenance.get("compiler_source_sha256")
        package_identity = provenance.get("package_sha256")
    else:
        try:
            frozen_package = cell.backend.options["packages"][cell.model.name][cell.precision]
        except (KeyError, TypeError) as error:
            raise ValueError(f"{cell.key}: frozen ExecuTorch package is absent") from error
        provenance["framework_source_sha256"] = cell.backend.options.get(
            "framework_source_sha256")
        provenance["framework_package_sha256"] = frozen_package.get("sha256")
        provenance["external_runtime_protocol"] = "executorch_session_package_v3"
        source_identity = provenance.get("framework_source_sha256")
        package_identity = provenance.get("framework_package_sha256")
    if not all(isinstance(value, str) and len(value) == 64
               for value in (source_identity, package_identity)):
        raise ValueError(f"{cell.key}: source/package freeze identity is unresolved")
    authority_value = str(spec.freeze.get("toolchain_authority_path", ""))
    authority_digest = str(spec.freeze.get("toolchain_authority_sha256", ""))
    if not authority_value or len(authority_digest) != 64:
        raise ValueError(f"{cell.key}: independent toolchain authority is unresolved")
    authority_path = Path(authority_value)
    authority_path = (authority_path if authority_path.is_absolute()
                      else repo_root() / authority_path).resolve()
    authority = load_toolchain_authority(
        authority_path, expected_sha256=authority_digest, expected_target=spec.target)

    staging_dir.mkdir(parents=True, exist_ok=False)
    root = template_path.parent
    retained_study = staging_dir / "frozen-study.yaml"
    retained_study.write_text(yaml.safe_dump(spec.canonical_dict(), sort_keys=True),
                              encoding="utf-8")
    retained_template = staging_dir / "backend-template.yaml"
    shutil.copy2(template_path, retained_template)

    def retain(ref: object, name: str, label: str) -> dict[str, str]:
        return _copy_ref(root, ref, staging_dir / name, label, staging_dir)

    # Validate and rebuild the package from its compiler/framework input before resolving the
    # private measurement I/O mapping.  Neither the object builder nor linker receives benchmark
    # inputs or eager-reference paths.
    resources = _closed(template["resources"], {
        "package_receipt", "compiler_input", "model_object", "build_tool",
        "runtime_artifact"},
        "backend template resources")
    package_receipt_ref = retain(resources["package_receipt"],
                                 "build/inputs/package_receipt.json", "package receipt")
    recipe = expected_recipe(registry_id, spec.target)
    compiler_input_source = _resolve_ref(root, resources["compiler_input"], "compiler input")
    compiler_input_destination = staging_dir / "build/inputs/compiler_input.json"
    stage_compiler_input(
        compiler_input_source, compiler_input_destination, recipe=recipe,
        include_private=recipe != EXECUTORCH_RECIPE)
    compiler_input_ref = {
        "path": compiler_input_destination.relative_to(staging_dir).as_posix(),
        "sha256": _sha(compiler_input_destination),
    }
    model_object_ref = retain(resources["model_object"], "build/sources/model_object.o",
                              "frozen model object")
    build_tool_ref = retain(resources["build_tool"], "build/tool", "build tool")
    authority_destination = staging_dir / "build/inputs/toolchain-authority.json"
    shutil.copy2(authority_path, authority_destination)
    toolchain_authority_ref = {
        "path": authority_destination.relative_to(staging_dir).as_posix(),
        "sha256": _sha(authority_destination),
    }
    build_tool_identity = verify_build_tool(
        staging_dir / build_tool_ref["path"], authority_path=authority_destination,
        authority_sha256=authority_digest, target=spec.target,
        expected_identity_sha256=str(authority["tool"]["identity_sha256"]))
    template_artifact_ref = retain(
        resources["runtime_artifact"], "build/inputs/template_runtime_artifact",
        "template runtime artifact")
    session_resources = (merlin_session_resources(compiler_input_destination)
                         if recipe == MERLIN_RECIPE else None)
    executorch_resources = (executorch_session_resources(compiler_input_destination)
                            if recipe == EXECUTORCH_RECIPE else None)
    runner_source = (session_resources.runner_source if session_resources is not None else
                     executorch_resources.runner if executorch_resources is not None else
                     Path(__file__).with_name("paper_model_abi_runner.c"))
    runner_destination = staging_dir / "build/sources/runner.c"
    shutil.copy2(runner_source, runner_destination)
    runner_ref = {"path": runner_destination.relative_to(staging_dir).as_posix(),
                  "sha256": _sha(runner_destination)}
    object_builder_source = Path(__file__).with_name("paper_model_object_builder.py")
    object_builder_destination = staging_dir / "build/inputs/object_builder.py"
    shutil.copy2(object_builder_source, object_builder_destination)
    object_builder_ref = {
        "path": object_builder_destination.relative_to(staging_dir).as_posix(),
        "sha256": _sha(object_builder_destination),
    }
    resource_hashes = {"runner": runner_ref["sha256"],
                       "model_object": model_object_ref["sha256"],
                       "compiler_input": compiler_input_ref["sha256"],
                       "object_builder": object_builder_ref["sha256"],
                       "build_tool": build_tool_ref["sha256"]}
    session_descriptor_ref = None
    if session_resources is not None:
        descriptor_destination = staging_dir / "build/inputs/session_descriptor.json"
        shutil.copy2(session_resources.descriptor_path, descriptor_destination)
        session_descriptor_ref = {
            "path": descriptor_destination.relative_to(staging_dir).as_posix(),
            "sha256": _sha(descriptor_destination),
        }
        resource_hashes["session_descriptor"] = session_descriptor_ref["sha256"]
    with tempfile.TemporaryDirectory(prefix="merlin-paper-contract-object-") as temporary:
        derived_object = Path(temporary) / "model_object.o"
        regeneration = regenerate_model_object(
            recipe=recipe, registry_id=registry_id,
            target=spec.target,
            compiler_input=staging_dir / compiler_input_ref["path"],
            tool=staging_dir / build_tool_ref["path"], output=derived_object,
            source_identity_sha256=str(source_identity),
            capture_sha256=cell.model.artifacts[cell.precision]["sha256"],
            runtime_artifact_sha256=template_artifact_ref["sha256"],
            timeout_seconds=template["timeout_seconds"])
        if _sha(derived_object) != model_object_ref["sha256"]:
            raise ValueError(
                f"{cell.key}: supplied model object differs from registry regeneration")
    package_receipt_path = staging_dir / package_receipt_ref["path"]
    parsed_package = _package_receipt(
        package_receipt_path, registry_id=registry_id, cell=cell,
        source_identity=str(source_identity), package_identity=str(package_identity),
        runtime_artifact_sha256=template_artifact_ref["sha256"],
        resource_hashes=resource_hashes, target=spec.target, regeneration=regeneration)
    with tempfile.TemporaryDirectory(prefix="merlin-paper-contract-link-") as temporary:
        rebuilt = Path(temporary) / "cell_executable"
        if recipe == EXECUTORCH_RECIPE:
            shutil.copy2(staging_dir / model_object_ref["path"], rebuilt)
        else:
            completed = subprocess.run([
                str(staging_dir / build_tool_ref["path"]), "-O2", "-std=c11",
                str(runner_destination), str(staging_dir / model_object_ref["path"]),
                "-o", str(rebuilt),
            ], capture_output=True, timeout=template["timeout_seconds"], check=False)
            if completed.returncode or not rebuilt.is_file():
                raise ValueError(f"{cell.key}: independently rebuilt package result differs")
        if _sha(rebuilt) != parsed_package["result_executable_sha256"]:
            raise ValueError(f"{cell.key}: independently rebuilt package result differs")

    # Only after source regeneration and result relinking succeed may private inputs and eager
    # references be resolved from the freeze.
    io = _measurement_io(spec, cell)
    artifact = _copy_study_ref(
        io["artifact"], staging_dir / "runtime/artifact", "frozen runtime artifact", staging_dir)
    inputs = {name: _copy_study_ref(ref, staging_dir / f"runtime/{name}",
                                    f"frozen runtime input {name}", staging_dir)
              for name, ref in io["inputs"].items()}
    session_manifest = _copy_study_ref(
        io["session_manifest"], staging_dir / "runtime/session-manifest.json",
        "frozen session manifest", staging_dir)
    reference_output = _copy_study_ref(
        io["reference_output"], staging_dir / "runtime/reference-output",
        "frozen reference output", staging_dir)
    io_receipt = _copy_study_ref(
        io["generation_receipt"], staging_dir / "runtime/measurement-io-receipt.json",
        "frozen measurement I/O receipt", staging_dir)
    if template_artifact_ref["sha256"] != artifact["sha256"]:
        raise ValueError(f"{cell.key}: template runtime artifact differs from frozen I/O")
    io_raw = json.loads(
        (staging_dir / io_receipt["path"]).read_text(encoding="utf-8"))
    io_fields = {
            "schema_version", "kind", "status", "cell", "package_receipt_sha256",
            "artifact_sha256", "input_sha256", "session_manifest_sha256",
            "reference_output_sha256", "reference_authority", "observations", "generated_at",
            "capture_sha256", "input_source_sha256", "eager_reference_source_sha256",
            "eager_reference_key",
        }
    if session_resources is not None:
        io_fields.update({"session_protocol", "session_descriptor_sha256"})
    io_document = _closed(io_raw, io_fields, "measurement I/O generation receipt")
    if (io_document["schema_version"] != (2 if session_resources is not None else 1)
            or io_document["kind"] != (
                "paper_measurement_io_generation_receipt_v2" if session_resources is not None
                else "paper_measurement_io_generation_receipt_v1")
            or io_document["status"] != "finalized"
            or io_document["cell"] != {"model": cell.model.name,
                                        "backend": cell.backend.name,
                                        "precision": cell.precision}
            or io_document["package_receipt_sha256"] != package_receipt_ref["sha256"]
            or io_document["artifact_sha256"] != artifact["sha256"]
            or io_document["input_sha256"] != {name: ref["sha256"]
                                                for name, ref in inputs.items()}
            or io_document["session_manifest_sha256"] != session_manifest["sha256"]
            or io_document["reference_output_sha256"] != reference_output["sha256"]
            or io_document["reference_authority"] != "eager_fp32"
            or io_document["observations"] != cell.model.session.observations
            or io_document["capture_sha256"] != cell.model.artifacts[cell.precision]["sha256"]
            or not all(isinstance(io_document[field], str) and io_document[field]
                       for field in ("input_source_sha256", "eager_reference_source_sha256",
                                     "eager_reference_key"))):
        raise ValueError(f"{cell.key}: measurement I/O receipt linkage differs")
    if session_resources is not None and (
            io_document["session_protocol"] != _MERLIN_SESSION_PROTOCOL
            or io_document["session_descriptor_sha256"] != session_resources.descriptor.sha256):
        raise ValueError(f"{cell.key}: measurement I/O session descriptor differs")
    if recipe == EXECUTORCH_RECIPE:
        # The producer barrier above verified only public build products.  Retain private streams
        # only after the canonical measurement-I/O receipt has been resolved and checked.
        stage_compiler_input(
            compiler_input_source, compiler_input_destination, recipe=recipe,
            include_private=True)
    build_sources = {"runner": runner_ref, "model_object": model_object_ref}
    build_inputs = {"package_receipt": package_receipt_ref,
                    "compiler_input": compiler_input_ref,
                    "object_builder": object_builder_ref}
    if session_descriptor_ref is not None:
        build_inputs["session_descriptor"] = session_descriptor_ref
    build_inputs["model_artifact"] = _copy_study_ref(
        io["artifact"], staging_dir / "build/inputs/model_artifact",
        "frozen build model artifact", staging_dir)
    probe_source = Path(__file__).with_name("paper_k1_board_probe.c")
    probe_destination = staging_dir / "paper_k1_board_probe.c"
    shutil.copy2(probe_source, probe_destination)
    build = {
        "study_sha256": spec.sha256(), "cell": expected_cell,
        "frozen_provenance_sha256": _canonical_sha(provenance),
        "model_artifact_sha256": base_result["artifact_sha256"],
        "source_identity_sha256": parsed_package["build_source_identity_sha256"],
        "package_identity_sha256": package_identity,
        "expected_executable_sha256": parsed_package["result_executable_sha256"],
        "tool": build_tool_ref,
        "toolchain_authority": toolchain_authority_ref,
        "build_tool_identity_sha256": build_tool_identity,
        "sources": build_sources, "inputs": build_inputs,
        "argv": list(_EXECUTORCH_BUILD_ARGV if recipe == EXECUTORCH_RECIPE else _BUILD_ARGV),
        "cwd": ".",
        "environment": {"PATH": "/usr/bin:/bin"},
        "timeout_seconds": template["timeout_seconds"],
    }
    contract = {
        "schema_version": 2, "kind": "paper_measurement_contract_v2", "status": "ready",
        "registry_id": registry_id, "backend_adapter": cell.backend.adapter,
        "study_spec": {"path": retained_study.relative_to(staging_dir).as_posix(),
                       "sha256": _sha(retained_study)},
        "backend_template": {"path": retained_template.relative_to(staging_dir).as_posix(),
                             "sha256": _sha(retained_template)},
        "study_sha256": spec.sha256(), "run_id": run_id, "target": spec.target,
        "cell": expected_cell,
        "result_identity": {key: base_result[key] for key in _IDENTITY_FIELDS},
        "session": cell.model.session.to_dict(), "frozen_provenance": provenance,
        "artifact_sha256": base_result["artifact_sha256"],
        "artifact": artifact,
        "inputs": inputs,
        "session_manifest": session_manifest,
        "measurement_io_receipt": io_receipt,
        "reference_output": reference_output,
        "oracle": dict(oracle), "build": build,
        "argv": (["{executable}", "{package_root}", "{core_count}", "{observation}"]
                 if recipe == EXECUTORCH_RECIPE else
                 ["{executable}", "{artifact}"] if session_resources is not None else
                 ["{executable}", "{artifact}", "{observation}",
                  *[f"{{input:{name}}}" for name in sorted(inputs)]]), "cwd": ".",
        "environment": dict(template["environment"]), "execution": dict(template["execution"]),
        "memory_policy": template["memory_policy"],
        "timeout_seconds": template["timeout_seconds"],
        "warmup_iterations": cell.model.session.warmups,
        "measured_iterations": cell.model.session.measurement_repeats,
        "timing": {key: base_result["timing"][key] for key in (
            "unit", "sample_unit", "scope", "timed_stages", "excluded_stages", "stage_samples")},
        "board_probe_source": {"path": probe_destination.relative_to(staging_dir).as_posix(),
                               "sha256": _sha(probe_destination)},
    }
    if session_descriptor_ref is not None:
        contract["session_abi"] = {
            "protocol": _MERLIN_SESSION_PROTOCOL,
            "descriptor": session_descriptor_ref,
        }
    # The K1 matrix transport accepts one canonical contract name.  Keeping the producer and
    # transport names identical prevents a prepare-only workflow from renaming an authority file
    # after it has been constructed and validated.
    contract_path = staging_dir / "measurement_contract.yaml"
    contract_path.write_text(yaml.safe_dump(contract, sort_keys=True), encoding="utf-8")
    return contract_path


__all__ = ["build_registered_contract"]
