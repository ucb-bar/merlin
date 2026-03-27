from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .model import (
    AccessModel,
    DeploymentProfile,
    ExecutionModel,
    ISAContract,
    ISAExposure,
    MemoryModel,
    NumericContract,
    OperationCoverage,
    OperationSpec,
    PlatformCapabilities,
    RegisterConstraints,
    RuntimeContract,
    TargetCapabilities,
    TargetIdentity,
    TileModel,
    VerificationContract,
)

CAPABILITY_TOP_LEVEL_KEYS = {
    "schema_version",
    "target",
    "platform",
    "execution_model",
    "isa",
    "operations",
    "geometry",
    "memory",
    "numeric",
    "runtime",
    "verification",
    "access",
    "references",
}

DEPLOYMENT_TOP_LEVEL_KEYS = {
    "schema_version",
    "deployment",
}

EXECUTION_KINDS = {
    "cpu_only",
    "vector_cpu_extension",
    "matrix_coprocessor",
    "rocc_accelerator",
    "structured_npu",
    "simt_gpu",
    "mixed",
}

ISA_EXPOSURE_KINDS = {
    "llvm_intrinsics",
    "llvm_backend_features",
    "custom_exporter",
    "text_isa",
    "hal_runtime",
    "none",
}

COMPILER_RECOVERY_STAGES = {
    "post_global_optimization",
    "llvmcpu_codegen",
    "text_isa_export",
    "runtime_only",
}

DEPLOYMENT_MODES = {
    "board",
    "firesim",
    "bare-metal",
    "simulator",
    "runtime-driver",
    "host-emulation",
}

ACCESS_MODELS = {
    "local_device",
    "remote_device",
    "cloud_service",
    "simulator",
    "board",
    "lab_hardware",
}

AVAILABILITY_CLASSES = {
    "always_local",
    "lab_hardware",
    "cloud_token_required",
    "sdk_install_required",
}

VERIFICATION_GATES = {
    "compile_only",
    "device_query",
    "smoke_run",
    "profile_run",
}


def load_capability_spec(path: str | Path) -> TargetCapabilities:
    source_path = Path(path)
    data = _load_yaml(source_path)
    errors = _validate_capability_payload(data)
    if errors:
        raise ValueError(_format_errors(source_path, errors))
    return _normalize_capability_payload(data, source_path)


def load_deployment_overlay(path: str | Path) -> DeploymentProfile:
    source_path = Path(path)
    data = _load_yaml(source_path)
    errors = _validate_deployment_payload(data)
    if errors:
        raise ValueError(_format_errors(source_path, errors))
    return _normalize_deployment_payload(data, source_path)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a YAML mapping at the document root")
    return data


def _validate_capability_payload(data: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    _reject_unknown_keys(data, CAPABILITY_TOP_LEVEL_KEYS, "root", errors)
    _require_map(data, "target", errors)
    _require_map(data, "platform", errors)
    _require_map(data, "execution_model", errors)
    _require_map(data, "isa", errors)
    _require_map(data, "operations", errors)
    _require_map(data, "geometry", errors)
    _require_map(data, "memory", errors)
    _require_map(data, "numeric", errors)
    _require_map(data, "runtime", errors)
    _require_map(data, "verification", errors)
    _require_map(data, "access", errors)
    _require_int(data, "schema_version", errors)

    target = _map_or_empty(data.get("target"))
    for key in ("name", "display_name", "vendor", "maturity"):
        _require_string(target, key, errors, "target")

    platform = _map_or_empty(data.get("platform"))
    _require_string(platform, "host_isa", errors, "platform")
    _require_string_list(platform, "operating_systems", errors, "platform")
    _require_string_list(platform, "environments", errors, "platform")

    execution = _map_or_empty(data.get("execution_model"))
    _require_string(execution, "kind", errors, "execution_model", allowed=EXECUTION_KINDS)
    _require_string(execution, "attachment", errors, "execution_model")
    _require_string(execution, "submission_model", errors, "execution_model")
    _require_string(
        execution,
        "compiler_recovery_stage",
        errors,
        "execution_model",
        allowed=COMPILER_RECOVERY_STAGES,
    )

    isa = _map_or_empty(data.get("isa"))
    _require_string(isa, "base", errors, "isa")
    _require_string_list(isa, "features", errors, "isa")
    exposure = _map_or_empty(isa.get("exposure"))
    _require_string(exposure, "kind", errors, "isa.exposure", allowed=ISA_EXPOSURE_KINDS)
    _require_bool(exposure, "needs_llvm_backend_changes", errors, "isa.exposure")
    _require_bool(exposure, "needs_new_intrinsics", errors, "isa.exposure")
    _require_bool(exposure, "needs_new_feature_bits", errors, "isa.exposure")
    state_model = _map_or_empty(isa.get("state_model"))
    _require_string(state_model, "kind", errors, "isa.state_model")
    register_constraints = _map_or_empty(isa.get("register_constraints"))
    _require_bool(register_constraints, "even_alignment_required", errors, "isa.register_constraints")
    _require_bool(register_constraints, "vl_dependent", errors, "isa.register_constraints")

    operations = _map_or_empty(data.get("operations"))
    compute = operations.get("compute")
    if not isinstance(compute, list) or not compute:
        errors.append("operations.compute: expected a non-empty list")
    else:
        for index, op in enumerate(compute):
            prefix = f"operations.compute[{index}]"
            if not isinstance(op, dict):
                errors.append(f"{prefix}: expected a mapping")
                continue
            _require_string(op, "name", errors, prefix)
            _require_bool(op, "native", errors, prefix)
            if "type_triples" in op and not _is_string_list(op["type_triples"]):
                errors.append(f"{prefix}.type_triples: expected a list of strings")
    _require_string_list(operations, "movement", errors, "operations")
    _require_string_list(operations, "synchronization", errors, "operations")

    geometry = _map_or_empty(data.get("geometry"))
    vector = _map_or_empty(geometry.get("vector"))
    if "vlen_bits" in vector and not isinstance(vector["vlen_bits"], int):
        errors.append("geometry.vector.vlen_bits: expected an integer")
    if "dlen_bits" in vector and not isinstance(vector["dlen_bits"], int):
        errors.append("geometry.vector.dlen_bits: expected an integer")
    compute_array = _map_or_empty(geometry.get("compute_array"))
    if "kind" in compute_array and not isinstance(compute_array["kind"], str):
        errors.append("geometry.compute_array.kind: expected a string")
    for key in ("native_tile_options", "preferred_tiles"):
        if key in geometry and not _is_int_matrix(geometry[key]):
            errors.append(f"geometry.{key}: expected a list of integer tile lists")

    memory = _map_or_empty(data.get("memory"))
    spaces = memory.get("spaces")
    if not isinstance(spaces, list) or not spaces:
        errors.append("memory.spaces: expected a non-empty list")
    preferred_layouts = memory.get("preferred_layouts", {})
    if preferred_layouts and not isinstance(preferred_layouts, dict):
        errors.append("memory.preferred_layouts: expected a mapping")
    packing = memory.get("packing", {})
    if packing and not isinstance(packing, dict):
        errors.append("memory.packing: expected a mapping")

    numeric = _map_or_empty(data.get("numeric"))
    _require_string_list(numeric, "legal_type_triples", errors, "numeric")
    if "quantization" in numeric and not isinstance(numeric["quantization"], dict):
        errors.append("numeric.quantization: expected a mapping")
    _require_string_list(numeric, "rounding_modes", errors, "numeric")
    _require_bool(numeric, "saturation", errors, "numeric")

    runtime = _map_or_empty(data.get("runtime"))
    _require_bool(runtime, "required", errors, "runtime")
    _require_string(runtime, "executable_format", errors, "runtime")
    driver = _map_or_empty(runtime.get("driver"))
    if "backend_id" in driver and driver["backend_id"] is not None and not isinstance(driver["backend_id"], str):
        errors.append("runtime.driver.backend_id: expected a string or null")
    if "uri_schemes" in driver and not _is_string_list(driver["uri_schemes"]):
        errors.append("runtime.driver.uri_schemes: expected a list of strings")
    if "synchronization" in runtime and not isinstance(runtime["synchronization"], dict):
        errors.append("runtime.synchronization: expected a mapping")

    verification = _map_or_empty(data.get("verification"))
    for key in ("golden_model", "simulator", "rtl", "perf_counters"):
        _require_map(verification, key, errors, "verification")
        section = _map_or_empty(verification.get(key))
        _require_bool(section, "available", errors, f"verification.{key}")
    if "tolerances" in verification and not isinstance(verification["tolerances"], dict):
        errors.append("verification.tolerances: expected a mapping")

    access = _map_or_empty(data.get("access"))
    _require_string(access, "model", errors, "access", allowed=ACCESS_MODELS)
    _require_string_list(access, "sdk_requirements", errors, "access")
    _require_string(
        access,
        "credential_requirements",
        errors,
        "access",
    )
    _require_string(
        access,
        "availability_class",
        errors,
        "access",
        allowed=AVAILABILITY_CLASSES,
    )
    gates = access.get("verification_gates")
    if not _is_string_list(gates):
        errors.append("access.verification_gates: expected a list of strings")
    else:
        for gate in gates:
            if gate not in VERIFICATION_GATES:
                errors.append(f"access.verification_gates: unexpected value {gate!r}")

    if "references" in data and not _is_string_list(data["references"]):
        errors.append("references: expected a list of strings")

    return errors


def _validate_deployment_payload(data: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    _reject_unknown_keys(data, DEPLOYMENT_TOP_LEVEL_KEYS, "root", errors)
    _require_int(data, "schema_version", errors)
    _require_map(data, "deployment", errors)
    deployment = _map_or_empty(data.get("deployment"))
    _require_string(deployment, "name", errors, "deployment")
    _require_string(deployment, "mode", errors, "deployment", allowed=DEPLOYMENT_MODES)
    _require_string(deployment, "build_profile", errors, "deployment")
    if (
        "compile_target" in deployment
        and deployment["compile_target"] is not None
        and not isinstance(deployment["compile_target"], str)
    ):
        errors.append("deployment.compile_target: expected a string or null")
    if (
        "compile_hw" in deployment
        and deployment["compile_hw"] is not None
        and not isinstance(deployment["compile_hw"], str)
    ):
        errors.append("deployment.compile_hw: expected a string or null")
    if (
        "hardware_recipe" in deployment
        and deployment["hardware_recipe"] is not None
        and not isinstance(deployment["hardware_recipe"], str)
    ):
        errors.append("deployment.hardware_recipe: expected a string or null")
    for key in ("chipyard", "runtime", "extra"):
        value = deployment.get(key)
        if value is not None and not isinstance(value, dict):
            errors.append(f"deployment.{key}: expected a mapping")
    return errors


def _normalize_capability_payload(data: dict[str, Any], source_path: Path) -> TargetCapabilities:
    target = data["target"]
    platform = data["platform"]
    execution = data["execution_model"]
    isa = data["isa"]
    exposure = isa["exposure"]
    register_constraints = isa.get("register_constraints", {})
    operations = data["operations"]
    geometry = data["geometry"]
    compute_array = geometry.get("compute_array", {})
    vector = geometry.get("vector", {})
    runtime = data["runtime"]
    driver = runtime.get("driver", {})
    verification = data["verification"]
    access = data["access"]

    return TargetCapabilities(
        schema_version=int(data["schema_version"]),
        identity=TargetIdentity(
            name=target["name"],
            display_name=target["display_name"],
            vendor=target["vendor"],
            maturity=target["maturity"],
        ),
        platform=PlatformCapabilities(
            host_isa=platform["host_isa"],
            operating_systems=list(platform["operating_systems"]),
            environments=list(platform["environments"]),
        ),
        execution=ExecutionModel(
            kind=execution["kind"],
            attachment=execution["attachment"],
            submission_model=execution["submission_model"],
            compiler_recovery_stage=execution["compiler_recovery_stage"],
        ),
        isa=ISAContract(
            base=isa["base"],
            features=list(isa["features"]),
            exposure=ISAExposure(
                kind=exposure["kind"],
                needs_llvm_backend_changes=bool(exposure["needs_llvm_backend_changes"]),
                needs_new_intrinsics=bool(exposure["needs_new_intrinsics"]),
                needs_new_feature_bits=bool(exposure["needs_new_feature_bits"]),
            ),
            state_model=isa["state_model"]["kind"],
            register_constraints=RegisterConstraints(
                accumulator_register_file=register_constraints.get("accumulator_register_file"),
                even_alignment_required=bool(register_constraints["even_alignment_required"]),
                vl_dependent=bool(register_constraints["vl_dependent"]),
            ),
        ),
        operations=OperationCoverage(
            compute=[
                OperationSpec(
                    name=op["name"],
                    native=bool(op["native"]),
                    type_triples=list(op.get("type_triples", [])),
                )
                for op in operations["compute"]
            ],
            movement=list(operations["movement"]),
            synchronization=list(operations["synchronization"]),
        ),
        tiles=TileModel(
            compute_array_kind=compute_array.get("kind"),
            vector_vlen_bits=vector.get("vlen_bits"),
            vector_dlen_bits=vector.get("dlen_bits"),
            native_tile_options=[list(tile) for tile in geometry.get("native_tile_options", [])],
            preferred_tiles=[list(tile) for tile in geometry.get("preferred_tiles", [])],
        ),
        memory=MemoryModel(
            spaces=list(data["memory"]["spaces"]),
            preferred_layouts=dict(data["memory"].get("preferred_layouts", {})),
            packing=dict(data["memory"].get("packing", {})),
        ),
        numeric=NumericContract(
            legal_type_triples=list(data["numeric"]["legal_type_triples"]),
            quantization=dict(data["numeric"].get("quantization", {})),
            rounding_modes=list(data["numeric"]["rounding_modes"]),
            saturation=bool(data["numeric"]["saturation"]),
        ),
        runtime=RuntimeContract(
            required=bool(runtime["required"]),
            executable_format=runtime["executable_format"],
            driver_backend_id=driver.get("backend_id"),
            uri_schemes=list(driver.get("uri_schemes", [])),
            synchronization=dict(runtime.get("synchronization", {})),
        ),
        verification=VerificationContract(
            golden_model=dict(verification["golden_model"]),
            simulator=dict(verification["simulator"]),
            rtl=dict(verification["rtl"]),
            perf_counters=dict(verification["perf_counters"]),
            tolerances=dict(verification.get("tolerances", {})),
        ),
        access=AccessModel(
            model=access["model"],
            sdk_requirements=list(access["sdk_requirements"]),
            credential_requirements=access["credential_requirements"],
            availability_class=access["availability_class"],
            verification_gates=list(access["verification_gates"]),
        ),
        references=list(data.get("references", [])),
        capability_path=str(source_path),
    )


def _normalize_deployment_payload(data: dict[str, Any], source_path: Path) -> DeploymentProfile:
    deployment = data["deployment"]
    return DeploymentProfile(
        name=deployment["name"],
        mode=deployment["mode"],
        build_profile=deployment["build_profile"],
        compile_target=deployment.get("compile_target"),
        compile_hw=deployment.get("compile_hw"),
        hardware_recipe=deployment.get("hardware_recipe"),
        chipyard=dict(deployment.get("chipyard", {})),
        runtime=dict(deployment.get("runtime", {})),
        extra=dict(deployment.get("extra", {})),
        source_path=str(source_path),
    )


def _format_errors(source_path: Path, errors: list[str]) -> str:
    return f"{source_path} failed validation:\n- " + "\n- ".join(errors)


def _map_or_empty(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def _reject_unknown_keys(data: dict[str, Any], allowed: set[str], context: str, errors: list[str]) -> None:
    for key in data:
        if key not in allowed:
            errors.append(f"{context}: unexpected key '{key}'")


def _require_map(data: dict[str, Any], key: str, errors: list[str], context: str = "root") -> None:
    if not isinstance(data.get(key), dict):
        errors.append(f"{context}.{key}: expected a mapping")


def _require_int(data: dict[str, Any], key: str, errors: list[str], context: str = "root") -> None:
    if not isinstance(data.get(key), int):
        errors.append(f"{context}.{key}: expected an integer")


def _require_bool(data: dict[str, Any], key: str, errors: list[str], context: str) -> None:
    if not isinstance(data.get(key), bool):
        errors.append(f"{context}.{key}: expected a boolean")


def _require_string(
    data: dict[str, Any],
    key: str,
    errors: list[str],
    context: str,
    *,
    allowed: set[str] | None = None,
) -> None:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        errors.append(f"{context}.{key}: expected a non-empty string")
        return
    if allowed is not None and value not in allowed:
        errors.append(f"{context}.{key}: expected one of {sorted(allowed)!r}")


def _require_string_list(data: dict[str, Any], key: str, errors: list[str], context: str) -> None:
    value = data.get(key)
    if not _is_string_list(value):
        errors.append(f"{context}.{key}: expected a list of strings")


def _is_string_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) and item for item in value)


def _is_int_matrix(value: Any) -> bool:
    return isinstance(value, list) and all(
        isinstance(row, list) and row and all(isinstance(cell, int) for cell in row) for row in value
    )
