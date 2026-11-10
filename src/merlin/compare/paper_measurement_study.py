"""Reopen a paper measurement against its exact frozen study and package."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from .paper_measurement_protocol import (
    _EXECUTORCH_PRODUCTION_BUILD_ARGV,
    _PRODUCTION_BUILD_ARGV,
    _bound_file,
    _canonical_sha,
    _closed,
    _is_sha,
    _sha_file,
)
from .paper_model_object_builder import (
    EXECUTORCH_RECIPE,
    executorch_session_resources,
    expected_recipe,
    object_build_argv,
)
from .paper_toolchain_authority import load_toolchain_authority


def _validate_study_root(root: Path, contract: Mapping[str, Any]) -> None:
    """Re-derive production identity from the retained canonical frozen study."""
    study_path = _bound_file(root, contract["study_spec"], "canonical frozen study")
    template_path = _bound_file(root, contract["backend_template"], "frozen backend template")
    if contract["registry_id"] == "unit_test_v1":
        unit = _closed(
            yaml.safe_load(study_path.read_text(encoding="utf-8")),
            {
                "schema_version",
                "kind",
                "status",
                "study_sha256",
                "cell",
            },
            "unit-test study root",
        )
        if (
            unit["schema_version"] != 1
            or unit["kind"] != "paper_unit_test_study_root_v1"
            or unit["status"] != "frozen"
            or unit["study_sha256"] != contract["study_sha256"]
            or unit["cell"] != contract["cell"]
            or contract["target"] != "unit-test"
        ):
            raise ValueError("unit-test study root differs from measurement contract")
        return

    from .paper import PaperStudySpec

    spec = PaperStudySpec.from_yaml(study_path)
    if spec.status != "frozen" or spec.sha256() != contract["study_sha256"]:
        raise ValueError("canonical study is not the exact frozen paper study")
    matches = [
        cell
        for cell in spec.matrix()
        if {
            "model": cell.model.name,
            "backend": cell.backend.name,
            "precision": cell.precision,
            "core_count": cell.core_count,
        }
        == contract["cell"]
    ]
    if len(matches) != 1:
        raise ValueError("measurement cell is not unique in canonical frozen study")
    cell = matches[0]
    expected_registry = "executorch_v1" if cell.backend.adapter == "executorch" else "merlin_compile_v1"
    contracts = cell.backend.options.get("measurement_contracts", {})
    try:
        template_digest = contracts[cell.model.name][cell.precision][str(cell.core_count)]["sha256"]
    except (KeyError, TypeError) as error:
        raise ValueError("canonical study has no frozen backend template for the cell") from error
    try:
        io = _closed(
            spec.freeze["measurement_io"][cell.backend.name][cell.model.name][cell.precision],
            {"artifact", "inputs", "session_manifest", "reference_output", "generation_receipt"},
            "canonical measurement_io",
        )
    except (KeyError, TypeError) as error:
        raise ValueError("canonical study has no frozen measurement I/O for the cell") from error
    identity = contract["result_identity"]
    expected_identity = {
        "study_label": spec.label,
        "target": spec.target,
        "model": cell.model.name,
        "checkpoint": cell.model.checkpoint,
        "fidelity": cell.model.fidelity,
        "backend": cell.backend.name,
        "runtime": cell.backend.runtime,
        "precision": cell.precision,
        "quantization": cell.backend.quantization,
        "core_count": cell.core_count,
    }
    provenance, build = contract["frozen_provenance"], contract["build"]
    if (
        contract["registry_id"] != expected_registry
        or contract["backend_adapter"] != cell.backend.adapter
        or contract["session"] != cell.model.session.to_dict()
        or contract["artifact_sha256"] != cell.model.artifacts[cell.precision]["sha256"]
        or any(identity[key] != value for key, value in expected_identity.items())
        or contract["backend_template"]["sha256"] != template_digest
        or provenance.get("study_sha256") != spec.sha256()
        or build["model_artifact_sha256"] != cell.model.artifacts[cell.precision]["sha256"]
    ):
        raise ValueError("measurement contract differs from canonical frozen study cell")
    authority_digest = str(spec.freeze.get("toolchain_authority_sha256", ""))

    template = _closed(
        yaml.safe_load(template_path.read_text(encoding="utf-8")),
        {
            "schema_version",
            "kind",
            "status",
            "registry_id",
            "backend_adapter",
            "cell",
            "resources",
            "environment",
            "execution",
            "memory_policy",
            "timeout_seconds",
        },
        "frozen backend template",
    )
    template_resources = _closed(
        template["resources"],
        {"package_receipt", "compiler_input", "model_object", "build_tool", "runtime_artifact"},
        "frozen backend resources",
    )

    def ref_sha(value: object, label: str) -> str:
        ref = _closed(value, {"path", "sha256"}, label)
        if not _is_sha(ref["sha256"]):
            raise ValueError(f"{label} digest is invalid")
        return str(ref["sha256"])

    def refs_sha(value: object, label: str) -> dict[str, str]:
        if not isinstance(value, Mapping):
            raise ValueError(f"{label} must be a mapping")
        return {str(name): ref_sha(ref, f"{label}.{name}") for name, ref in value.items()}

    metric = cell.model.quality["metric"]
    expected_oracle = {
        "kind": "int64_top1" if metric == "top1_agreement" else "float32_cosine",
        "metric": metric,
        "threshold": float(cell.model.quality["cosine_min"]),
        "scope": "trajectory",
        "steps": cell.model.session.observations,
    }
    expected_io = {
        "artifact": ref_sha(io["artifact"], "study measurement artifact"),
        "inputs": refs_sha(io["inputs"], "study measurement inputs"),
        "session_manifest": ref_sha(io["session_manifest"], "study session manifest"),
        "reference_output": ref_sha(io["reference_output"], "study reference output"),
        "generation_receipt": ref_sha(io["generation_receipt"], "study I/O receipt"),
    }
    contract_io = {
        "artifact": ref_sha(contract["artifact"], "artifact"),
        "inputs": refs_sha(contract["inputs"], "inputs"),
        "session_manifest": ref_sha(contract["session_manifest"], "session manifest"),
        "reference_output": ref_sha(contract["reference_output"], "reference output"),
        "generation_receipt": ref_sha(contract["measurement_io_receipt"], "I/O receipt"),
    }
    if contract_io != expected_io or contract["oracle"] != expected_oracle:
        raise ValueError("measurement I/O or oracle differs from canonical frozen study")

    expected_resources = {
        "package_receipt": ref_sha(build["inputs"]["package_receipt"], "package receipt"),
        "compiler_input": ref_sha(build["inputs"]["compiler_input"], "compiler input"),
        "model_object": ref_sha(build["sources"]["model_object"], "model object"),
        "build_tool": ref_sha(build["tool"], "build tool"),
        "runtime_artifact": expected_io["artifact"],
    }
    retained_authority_sha = ref_sha(build["toolchain_authority"], "independent toolchain authority")
    if retained_authority_sha != authority_digest:
        raise ValueError("measurement build differs from frozen toolchain authority")
    authority_path = _bound_file(root, build["toolchain_authority"], "independent toolchain authority")
    authority = load_toolchain_authority(
        authority_path, expected_sha256=authority_digest, expected_target=contract["target"]
    )
    if build["build_tool_identity_sha256"] != authority["tool"]["identity_sha256"]:
        raise ValueError("measurement build-tool identity differs from frozen authority")
    frozen_resources = {name: ref_sha(ref, f"template resource {name}") for name, ref in template_resources.items()}
    runner_digest = ref_sha(build["sources"]["runner"], "registry runner source")
    session_abi = contract.get("session_abi")
    compiler_input_path = _bound_file(root, build["inputs"]["compiler_input"], "compiler input")
    if session_abi is not None:
        from .paper_model_object_builder import merlin_session_resources

        merlin_resources = merlin_session_resources(compiler_input_path)
        shipped_runner = merlin_resources.runner_source
        descriptor_digest = ref_sha(session_abi["descriptor"], "MRLNSES2 descriptor")
        if descriptor_digest != _sha_file(merlin_resources.descriptor_path) or session_abi["descriptor"] != build[
            "inputs"
        ].get("session_descriptor"):
            raise ValueError("production MRLNSES2 descriptor differs from compiler input")
    elif (
        expected_registry == "executorch_v1"
        and expected_recipe(expected_registry, contract["target"]) == EXECUTORCH_RECIPE
    ):
        merlin_resources = None
        descriptor_digest = None
        executorch_resources = executorch_session_resources(compiler_input_path)
        shipped_runner = executorch_resources.runner
    else:
        merlin_resources = None
        descriptor_digest = None
        shipped_runner = Path(__file__).with_name("paper_model_abi_runner.c")
    if runner_digest != _sha_file(shipped_runner):
        raise ValueError("production runner differs from registry-owned source")
    package_path = _bound_file(root, build["inputs"]["package_receipt"], "package receipt")
    package_document = json.loads(package_path.read_text(encoding="utf-8"))
    package_fields = {
        "schema_version",
        "kind",
        "status",
        "registry_id",
        "build_adapter",
        "cell",
        "package_identity_sha256",
        "compiler_or_framework_source_sha256",
        "capture_sha256",
        "runtime_artifact_sha256",
        "runner_source_sha256",
        "model_object_sha256",
        "compiler_input_sha256",
        "object_builder_source_sha256",
        "object_recipe",
        "object_build_argv",
        "generated_model_source_sha256",
        "build_tool_sha256",
        "build_source_identity_sha256",
        "build_argv",
        "result_executable_sha256",
        "finalized_at",
    }
    if session_abi is not None:
        package_fields.update({"session_protocol", "session_descriptor_sha256"})
    package = _closed(package_document, package_fields, "backend package receipt")
    io_receipt_path = _bound_file(root, contract["measurement_io_receipt"], "measurement I/O generation receipt")
    io_document = json.loads(io_receipt_path.read_text(encoding="utf-8"))
    io_fields = {
        "schema_version",
        "kind",
        "status",
        "cell",
        "package_receipt_sha256",
        "artifact_sha256",
        "input_sha256",
        "session_manifest_sha256",
        "reference_output_sha256",
        "reference_authority",
        "observations",
        "generated_at",
        "capture_sha256",
        "input_source_sha256",
        "eager_reference_source_sha256",
        "eager_reference_key",
    }
    if session_abi is not None:
        io_fields.update({"session_protocol", "session_descriptor_sha256"})
    io_receipt = _closed(io_document, io_fields, "measurement I/O generation receipt")
    expected_package_source = (
        spec.freeze.get("compiler_source_sha256")
        if expected_registry == "merlin_compile_v1"
        else cell.backend.options.get("framework_source_sha256")
    )
    expected_package_identity = (
        cell.backend.options.get("package_sha256")
        if expected_registry == "merlin_compile_v1"
        else cell.backend.options["packages"][cell.model.name][cell.precision]["sha256"]
    )
    expected_source_identity = _canonical_sha(
        {
            "compiler_input": expected_resources["compiler_input"],
            "model_object": expected_resources["model_object"],
            "object_builder": ref_sha(build["inputs"]["object_builder"], "object builder"),
            "runner": runner_digest,
        }
    )
    shipped_builder = Path(__file__).with_name("paper_model_object_builder.py")
    if ref_sha(build["inputs"]["object_builder"], "object builder") != _sha_file(shipped_builder):
        raise ValueError("production object builder differs from registry-owned source")
    expected_package_cell = {"model": cell.model.name, "backend": cell.backend.name, "precision": cell.precision}
    expected_runtime_argv = (
        ["{executable}", "{package_root}", "{core_count}", "{observation}"]
        if (
            expected_registry == "executorch_v1"
            and expected_recipe(expected_registry, contract["target"]) == EXECUTORCH_RECIPE
        )
        else ["{executable}", "{artifact}"]
        if session_abi is not None
        else [
            "{executable}",
            "{artifact}",
            "{observation}",
            *[f"{{input:{name}}}" for name in sorted(contract["inputs"])],
        ]
    )
    contract_template_projection = {
        "registry_id": contract["registry_id"],
        "backend_adapter": contract["backend_adapter"],
        "cell": contract["cell"],
        "environment": contract["environment"],
        "execution": contract["execution"],
        "memory_policy": contract["memory_policy"],
        "timeout_seconds": contract["timeout_seconds"],
        "resources": expected_resources,
    }
    frozen_template_projection = {
        "registry_id": template["registry_id"],
        "backend_adapter": template["backend_adapter"],
        "cell": template["cell"],
        "environment": template["environment"],
        "execution": template["execution"],
        "memory_policy": template["memory_policy"],
        "timeout_seconds": template["timeout_seconds"],
        "resources": frozen_resources,
    }
    if (
        template["schema_version"] != 3
        or template["kind"] != "paper_backend_measurement_template_v3"
        or template["status"] != "frozen"
        or contract_template_projection != frozen_template_projection
    ):
        raise ValueError("measurement execution/build plan differs from frozen backend template")
    if (
        package["schema_version"] != (3 if session_abi is not None else 2)
        or package["kind"]
        != ("paper_backend_package_receipt_v3" if session_abi is not None else "paper_backend_package_receipt_v2")
        or package["status"] != "finalized"
        or package["registry_id"] != expected_registry
        or package["build_adapter"]
        != (
            "merlin_session_abi_c_v1"
            if session_abi is not None
            else (
                "executorch_sealed_session_v1"
                if (expected_registry == "executorch_v1" and package["object_recipe"] == EXECUTORCH_RECIPE)
                else "executorch_model_abi_c_v1"
                if expected_registry == "executorch_v1"
                else "merlin_model_abi_c_v1"
            )
        )
        or package["cell"] != expected_package_cell
        or package["package_identity_sha256"] != expected_package_identity
        or package["compiler_or_framework_source_sha256"] != expected_package_source
        or package["capture_sha256"] != cell.model.artifacts[cell.precision]["sha256"]
        or package["runtime_artifact_sha256"] != expected_io["artifact"]
        or package["runner_source_sha256"] != runner_digest
        or package["model_object_sha256"] != expected_resources["model_object"]
        or package["compiler_input_sha256"] != expected_resources["compiler_input"]
        or package["object_builder_source_sha256"] != _sha_file(shipped_builder)
        or package["object_recipe"] != expected_recipe(expected_registry, contract["target"])
        or package["object_build_argv"] != object_build_argv(str(package["object_recipe"]))
        or package["build_tool_sha256"] != expected_resources["build_tool"]
        or package["build_source_identity_sha256"] != expected_source_identity
        or package["build_argv"]
        != (
            _EXECUTORCH_PRODUCTION_BUILD_ARGV
            if (expected_registry == "executorch_v1" and package["object_recipe"] == EXECUTORCH_RECIPE)
            else _PRODUCTION_BUILD_ARGV
        )
        or package["result_executable_sha256"] != build["expected_executable_sha256"]
        or build["source_identity_sha256"] != expected_source_identity
        or build["argv"]
        != (
            _EXECUTORCH_PRODUCTION_BUILD_ARGV
            if (expected_registry == "executorch_v1" and package["object_recipe"] == EXECUTORCH_RECIPE)
            else _PRODUCTION_BUILD_ARGV
        )
        or contract["argv"] != expected_runtime_argv
    ):
        raise ValueError("package receipt does not bind the exact registry-owned build/result")
    if session_abi is not None and (
        package["session_protocol"] != "MRLNSES2" or package["session_descriptor_sha256"] != descriptor_digest
    ):
        raise ValueError("package receipt does not bind the MRLNSES2 descriptor")
    if (
        io_receipt["schema_version"] != (2 if session_abi is not None else 1)
        or io_receipt["kind"]
        != (
            "paper_measurement_io_generation_receipt_v2"
            if session_abi is not None
            else "paper_measurement_io_generation_receipt_v1"
        )
        or io_receipt["status"] != "finalized"
        or io_receipt["cell"] != expected_package_cell
        or io_receipt["package_receipt_sha256"] != expected_resources["package_receipt"]
        or io_receipt["artifact_sha256"] != expected_io["artifact"]
        or io_receipt["input_sha256"] != expected_io["inputs"]
        or io_receipt["session_manifest_sha256"] != expected_io["session_manifest"]
        or io_receipt["reference_output_sha256"] != expected_io["reference_output"]
        or io_receipt["reference_authority"] != "eager_fp32"
        or io_receipt["observations"] != cell.model.session.observations
        or io_receipt["capture_sha256"] != cell.model.artifacts[cell.precision]["sha256"]
        or not all(
            isinstance(io_receipt[field], str) and io_receipt[field]
            for field in ("input_source_sha256", "eager_reference_source_sha256", "eager_reference_key")
        )
    ):
        raise ValueError("measurement I/O receipt is not downstream of frozen package build")
    if session_abi is not None and (
        io_receipt["session_protocol"] != "MRLNSES2"
        or io_receipt["session_descriptor_sha256"] != merlin_resources.descriptor.sha256
    ):
        raise ValueError("measurement I/O receipt does not bind the MRLNSES2 descriptor")
    if expected_registry == "merlin_compile_v1":
        if (
            provenance.get("compiler_policy_sha256") != spec.freeze.get("policy_sha256")
            or provenance.get("compiler_source_sha256") != spec.freeze.get("compiler_source_sha256")
            or provenance.get("runtime_sha256") != spec.freeze.get("runtime_sha256")
            or build["package_identity_sha256"] != cell.backend.options.get("package_sha256")
        ):
            raise ValueError("Merlin contract differs from frozen compiler/runtime/package")
    else:
        try:
            package_digest = cell.backend.options["packages"][cell.model.name][cell.precision]["sha256"]
        except (KeyError, TypeError) as error:
            raise ValueError("canonical ExecuTorch package is absent") from error
        if build["package_identity_sha256"] != package_digest:
            raise ValueError("ExecuTorch contract differs from frozen framework/package")
