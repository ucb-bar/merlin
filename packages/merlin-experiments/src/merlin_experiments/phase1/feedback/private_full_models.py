"""Operator-private, build-only qualification of complete validation models.

The declaration is an operator input, never a candidate package file.  A result is
scoped to the exact frozen capture, selected software and hardware declarations,
and the current submission bytes.  Full-model simulation is deliberately a later
qualification; representative model capsules retain their independent L3 grade.
"""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Mapping, Sequence
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from typing import Any

import yaml

from merlin.compile.model_execution_inputs import file_sha256, strict_tree_sha256
from merlin_experiments.phase1.feedback import private_bucketize_support as bucketize_support
from merlin_experiments.phase1.feedback import private_compilation_inputs as compilation_support
from merlin_experiments.phase1.feedback import private_control_support as control_support
from merlin_experiments.phase1.feedback import private_data_movement as data_movement
from merlin_experiments.phase1.feedback import private_f32_maximum_support as maximum_support
from merlin_experiments.phase1.feedback import private_host_source_dispatch as host_source_dispatch
from merlin_experiments.phase1.feedback import private_index_host_support as index_support
from merlin_experiments.phase1.feedback import private_integer_reduction_support as integer_support
from merlin_experiments.phase1.feedback import private_linalg_support as linalg_support
from merlin_experiments.phase1.feedback import private_linkage_support as linkage_support
from merlin_experiments.phase1.feedback import private_linked_elf_selection as linked_policy
from merlin_experiments.phase1.feedback import private_literal_arange_admission as arange_support
from merlin_experiments.phase1.feedback import private_ordered_scan_support as ordered_scan_support
from merlin_experiments.phase1.feedback import private_pure_stage_support as pure_stage
from merlin_experiments.phase1.feedback import private_source_freeze as source_freeze_api
from merlin_experiments.phase1.feedback.private_capture_roster import (
    captured_input_provenance as _captured_input_provenance,
)
from merlin_experiments.phase1.feedback.private_capture_roster import captured_programs as _captured_programs
from merlin_experiments.phase1.feedback.private_device_audit import (
    audit_built_device_host_compute as _audit_built_device_host_compute,
)
from merlin_experiments.phase1.feedback.private_device_audit import (
    require_static_build_inputs as _require_static_build_inputs,
)
from merlin_experiments.phase1.feedback.private_group_provenance import (
    eligible_source_identity as _eligible_source_identity,
)
from merlin_experiments.phase1.feedback.private_group_provenance import (
    join_routed_source_groups as _join_routed_source_groups,
)
from merlin_experiments.phase1.feedback.private_group_provenance import (
    routed_kernel_symbols as _routed_kernel_symbols,
)
from merlin_experiments.phase1.feedback.private_prebuilt_receipt import load_diagnostic_receipt
from merlin_experiments.phase1.feedback.private_source_support_join import (
    linked_source_support_complete as _linked_source_support_complete,
)

SCHEMA = "merlin.phase1.private_full_models.v1"
RESULT_SCHEMA = "merlin.phase1.private_full_model_build_gate.v14"
BUILD_BOARD_SCOPE = "static_memory_layout_and_host_ISA_only; no board execution"
TRANSPOSE_DATA_SUPPORT_SCOPE = data_movement.SCOPE
_transpose_data_support = data_movement.prove_transpose_source
_verify_transpose_data_support = data_movement.verify_transpose
_CONTROL_DATA_SUPPORT = frozenset(
    {
        "arith.constant",
        "tensor.collapse_shape",
        "tensor.concat",
        "tensor.dim",
        "tensor.empty",
        "tensor.expand_shape",
        "tensor.extract",
        "tensor.extract_slice",
        "tensor.from_elements",
        "tensor.generate",
        "tensor.insert",
        "tensor.insert_slice",
        "tensor.pad",
        "tensor.splat",
        "memref.alloc",
        "memref.alloca",
        "memref.cast",
        "memref.copy",
        "memref.dealloc",
        "memref.dim",
        "memref.expand_shape",
        "memref.get_global",
        "memref.global",
        "memref.load",
        "memref.reinterpret_cast",
        "memref.store",
        "memref.subview",
        "scf.condition",
        "scf.execute_region",
        "scf.for",
        "scf.if",
        "scf.while",
    }
)
_EXPLICIT_HARDWARE_EXCLUSIONS = frozenset(
    {
        "undeclared_family",
        "input_dtype",
        "weight_dtype",
        "operand_pair",
        "result_dtype",
        "rank",
        "batch",
        "layout",
        "form",
    }
)


class SourceAdmissionError(ValueError):
    """Count-only public failure with copied, operator-private source diagnostics."""

    def __init__(self, unresolved: Sequence[Mapping[str, Any]]) -> None:
        super().__init__(f"source has {len(unresolved)} unaccounted or unjustified operation signature(s)")
        self.__unresolved = deepcopy(tuple(unresolved))

    @property
    def _private_unresolved(self) -> tuple[dict[str, Any], ...]:
        return deepcopy(self.__unresolved)


def _public_failure_reason(exc: Exception) -> str:
    kind = "ValueError" if isinstance(exc, SourceAdmissionError) else type(exc).__name__
    return f"{kind}: {exc}"


def requirements_for(descriptor: str | Path) -> tuple[str, ...]:
    """Read the target's required full-model identities, without private input paths."""
    if not Path(descriptor).is_file():
        return ()
    document = yaml.safe_load(Path(descriptor).read_bytes())
    gate = ((document or {}).get("phase1_gates") or {}).get("private_full_models")
    if not isinstance(gate, Mapping) or gate.get("required") is not True:
        return ()
    names = gate.get("models")
    if not isinstance(names, list) or not names or any(not isinstance(name, str) or not name for name in names):
        raise ValueError("target private full-model gate has no valid model roster")
    if len(names) != len(set(names)):
        raise ValueError("target private full-model gate repeats a model")
    return tuple(names)


def program_requirements_for(descriptor: str | Path) -> dict[str, tuple[str, ...]]:
    """Target declaration chooses the complete stage roster, never the private spec."""
    if not Path(descriptor).is_file():
        return {}
    document = yaml.safe_load(Path(descriptor).read_bytes()) or {}
    gate = (document.get("phase1_gates") or {}).get("private_full_models") or {}
    models = requirements_for(descriptor)
    if not models:
        return {}
    programs = gate.get("programs")
    if not isinstance(programs, Mapping) or set(programs) != set(models):
        raise ValueError("target private full-model program roster is incomplete")
    result = {}
    for model in models:
        names = programs[model]
        if (
            not isinstance(names, list)
            or not names
            or any(not isinstance(name, str) or Path(name).name != name or name in {".", ".."} for name in names)
            or len(set(names)) != len(names)
        ):
            raise ValueError(f"target {model} program roster is malformed")
        result[model] = tuple(names)
    return result


def loader_env_requirements_for(descriptor: str | Path) -> dict[str, dict[str, Any]]:
    """Generic data-driven loader scope checks; no model name is interpreted in code."""
    if not Path(descriptor).is_file():
        return {}
    document = yaml.safe_load(Path(descriptor).read_bytes()) or {}
    gate = (document.get("phase1_gates") or {}).get("private_full_models") or {}
    models = requirements_for(descriptor)
    if not models:
        return {}
    expected = gate.get("required_loader_env") or {}
    forbidden = gate.get("forbidden_loader_env") or {}
    roles = gate.get("required_selected_roles") or {}
    workloads = gate.get("source_workload_dirs") or {}
    dtypes = gate.get("deployment_dtypes") or {}
    # An opt-in obligation: a descriptor that does not declare it does not require integer contractions.
    integer_required = gate["require_integer_contractions"] if "require_integer_contractions" in gate else False
    if type(integer_required) is not bool:
        raise ValueError("target integer-contraction requirement must be boolean")
    if any(not isinstance(value, Mapping) for value in (expected, forbidden, roles, workloads, dtypes)):
        raise ValueError("target loader-env obligations are malformed")
    if (
        any(set(value) != set(models) for value in (roles, workloads, dtypes))
        or set(expected) - set(models)
        or set(forbidden) - set(models)
    ):
        raise ValueError("target loader-env obligations name an unrequired model")
    result = {}
    for model in models:
        positive, negative = expected.get(model) or {}, forbidden.get(model) or []
        if not isinstance(positive, Mapping) or not isinstance(negative, list):
            raise ValueError(f"target {model} loader-env obligations are malformed")
        if any(not isinstance(k, str) or not isinstance(v, str) for k, v in positive.items()) or any(
            not isinstance(k, str) for k in negative
        ):
            raise ValueError(f"target {model} loader-env obligations contain a non-string")
        selected_roles, workload, dtype = roles[model], workloads[model], dtypes[model]
        if (
            not isinstance(selected_roles, list)
            or any(not isinstance(role, str) or not role for role in selected_roles)
            or not isinstance(workload, str)
            or Path(workload).name != workload
            or not isinstance(dtype, str)
            or not dtype
        ):
            raise ValueError(f"target {model} source/deployment obligation is malformed")
        result[model] = {
            "required": dict(positive),
            "forbidden": tuple(negative),
            "selected_roles": tuple(selected_roles),
            "workload_dir": workload,
            "deployment_dtype": dtype,
            "require_integer_contractions": integer_required,
        }
    return result


def private_input_paths(
    private_spec: str | Path,
    *,
    target: str,
    required_models: Sequence[str],
    scope_requirements: Mapping[str, Mapping[str, Any]] | None = None,
    source_freeze: Mapping[str, Any] | None = None,
    source_freeze_root: str | Path | None = None,
) -> list[dict[str, str]]:
    """Derive sandbox denials from the operator's selected validation inputs.

    These are not grants.  In particular a future capture output may be absent
    when authoring starts; its preselected run root remains a denied location.
    """
    spec = Path(private_spec).absolute()
    document = yaml.safe_load(spec.read_bytes())
    if not isinstance(document, Mapping) or document.get("schema") != SCHEMA or document.get("target") != target:
        raise ValueError("operator-private full-model specification is malformed")
    rows = document.get("models")
    if (
        not isinstance(rows, list)
        or len(rows) != len(required_models)
        or any(not isinstance(row, Mapping) for row in rows)
        or [row.get("id") for row in rows] != list(required_models)
    ):
        raise ValueError("operator-private full-model roster is malformed")
    resolved_sources = source_freeze_api.resolve_optional(spec, source_freeze, source_freeze_root, target=target)
    denied: dict[str, str] = {}

    def add(value: Any, kind: str) -> None:
        if not isinstance(value, str) or not value:
            return
        path = Path(value)
        if not path.is_absolute() or ".." in path.parts or str(path) == "/":
            raise ValueError("private validation path must be absolute and non-broad")
        denied[str(path)] = kind

    add(str(spec), "file")
    for row in rows:
        _expected_input_provenance(row)
        for key in (
            "capture_selection",
            "capture",
            "capture_execution_attestation",
            "recipe_derivation_root",
            "rtl_facts",
            "host_package",
            "software_spec",
            "capability_contract",
            "host_capabilities",
            "board_catalog",
            "host_dts",
        ):
            value = row.get(key)
            if not isinstance(value, str) or not Path(value).is_absolute() or ".." in Path(value).parts:
                raise ValueError(f"private model {row['id']} requires an absolute {key} path")
        selection = _file(spec, row.get("capture_selection"), row.get("capture_selection_sha256"))
        from merlin_experiments.phase0 import capture_selection as capture_selection_api

        selected = capture_selection_api.load(selection, expected_sha256=row["capture_selection_sha256"])
        if not isinstance(selected, Mapping) or not isinstance(selected.get("plan"), Mapping):
            raise ValueError("private capture preselection has no source plan")
        plan = selected["plan"]
        _require_selected_input_bindings(row, plan)
        derivation_root = Path(row["recipe_derivation_root"])
        _file(spec, str(derivation_root / "evidence-manifest.json"), row.get("recipe_derivation_manifest_sha256"))
        _recipe_derivation(
            spec,
            row,
            plan,
            target=target,
            software_sha256=file_sha256(_authored_file(spec, row, "software_spec", resolved_sources)),
            capability_sha256=file_sha256(
                _file(spec, row["capability_contract"], row.get("capability_contract_sha256"))
            ),
            facts_sha256=file_sha256(_file(spec, row["rtl_facts"], row.get("rtl_facts_sha256"))),
            host_capabilities_sha256=file_sha256(_authored_file(spec, row, "host_capabilities", resolved_sources)),
        )
        add(str(derivation_root), "dir")
        if scope_requirements is not None:
            obligation = scope_requirements[row["id"]]
            loader_env = plan.get("loader_env") or {}
            selected_roles = {
                entry.get("role") for entry in plan.get("selected_inputs") or [] if isinstance(entry, Mapping)
            }
            if (
                selected.get("schema") != capture_selection_api.SCHEMA_V2
                or Path(str(plan.get("workload_root") or "")).name != obligation["workload_dir"]
                or row.get("deployment_dtype") != obligation["deployment_dtype"]
                or not set(obligation["selected_roles"]).issubset(selected_roles)
                or not isinstance(loader_env, Mapping)
                or any(loader_env.get(key) != value for key, value in obligation["required"].items())
                or any(loader_env.get(key) is not None for key in obligation["forbidden"])
            ):
                raise ValueError("private preselection is not the declared complete model source")
        add(str(selection.parent), "dir")
        add(selected.get("run_dir"), "dir")
        # Only the selected validation workload source is private. The common
        # M2M library, Merlin worker, target software contract and RTL facts
        # remain available through their independently reviewed public grants.
        for key in ("workload_root",):
            add(plan.get(key), "dir")
        for checkpoint in (selected.get("checkpoint"), plan.get("checkpoint")):
            if isinstance(checkpoint, Mapping):
                add(checkpoint.get("path"), "file")
        for selected_input in plan.get("selected_inputs") or []:
            if not isinstance(selected_input, Mapping):
                raise ValueError("private capture preselection input is malformed")
            kind = selected_input.get("kind")
            if kind not in {"file", "tree"}:
                raise ValueError("private selected input has an unsupported kind")
            source_path = Path(str(selected_input.get("source") or ""))
            add(str(source_path), "dir" if kind == "tree" else "file")
            if kind == "tree":
                for member in source_path.rglob("*"):
                    if member.is_symlink():
                        target_path = member.resolve(strict=True)
                        if not target_path.is_file():
                            raise ValueError("selected input tree has a non-file indirect member")
                        add(str(target_path), "file")
        for key in ("capture_execution_attestation",):
            value = row.get(key)
            if isinstance(value, str) and value:
                add(str(Path(value).parent), "dir")
    if resolved_sources is not None:
        add(str(source_freeze_root), "dir")
        for original, _field in resolved_sources:
            add(original, "file")
    return [{"path": path, "kind": kind} for path, kind in sorted(denied.items())]


_file = source_freeze_api.pinned_file
_authored_file = source_freeze_api.authored_file


def _expected_input_provenance(row: Mapping[str, Any]) -> dict[str, bool | None]:
    """The frozen operator's claim about selected inputs, not a target-name rule."""
    value = row.get("input_provenance")
    if (
        not isinstance(value, Mapping)
        or set(value) != {"paper_ready", "synthetic_inputs"}
        or any(item is not None and type(item) is not bool for item in value.values())
    ):
        raise ValueError("operator-private model needs explicit input-provenance expectations")
    return dict(value)


def _require_selected_input_bindings(row: Mapping[str, Any], plan: Mapping[str, Any]) -> None:
    """Freeze the entire selected source-input roster, including transitive models.

    This exact operator-private declaration is not an authoring or public
    descriptor field. It binds guest identity and content without embedding
    validation tensors, model shapes, or source paths in the agent surface.
    """
    selected = plan.get("selected_inputs")
    expected = row.get("selected_input_bindings")
    if not isinstance(selected, list) or not selected or not isinstance(expected, list):
        raise ValueError("complete model needs an exact private selected-input roster")
    observed = []
    for entry in selected:
        if not isinstance(entry, Mapping):
            raise ValueError("selected source input is malformed")
        kind = entry.get("kind")
        digest = entry.get("sha256") if kind == "file" else (entry.get("tree") or {}).get("sha256")
        guest = entry.get("guest_member")
        role = entry.get("role")
        if (
            kind not in {"file", "tree"}
            or not isinstance(role, str)
            or not role
            or not isinstance(guest, str)
            or not guest
            or Path(guest).is_absolute()
            or ".." in Path(guest).parts
            or not isinstance(digest, str)
            or len(digest) != 64
        ):
            raise ValueError("selected source input has no safe content identity")
        observed.append({"role": role, "kind": kind, "guest_member": guest, "sha256": digest})
    if len({entry["guest_member"] for entry in observed}) != len(observed) or expected != observed:
        raise ValueError("private selected-input roster omits or changes a source dependency")


def _tree(spec: Path, value: Any, digest: Any) -> Path:
    if not isinstance(value, str) or not value or not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("private model tree needs a path and SHA256")
    path = Path(value)
    path = path if path.is_absolute() else spec.parent / path
    if strict_tree_sha256(path)["sha256"] != digest:
        raise ValueError(f"private model tree changed: {path}")
    return path.resolve(strict=True)


def _capture_tree_bindings(capture: Path, pinned_strict: Any, attested_sealed: Any) -> dict[str, str]:
    """Check independent compiler and sealed-issuer identities of one capture."""
    from merlin_experiments.phase0.capture_execution_attestation import sealed_m2m_tree_snapshot

    strict = strict_tree_sha256(capture)["sha256"]
    if not isinstance(pinned_strict, str) or pinned_strict != strict:
        raise ValueError("full-model capture differs from an explicitly pinned compiler tree")
    sealed = sealed_m2m_tree_snapshot(capture)["sha256"]
    if not isinstance(attested_sealed, str) or attested_sealed != sealed:
        raise ValueError("verified source-execution attestation names another complete capture")
    return {"compiler_strict_tree_sha256": strict, "issuer_sealed_tree_sha256": sealed}


def _recipe_derivation(
    spec: Path,
    row: Mapping[str, Any],
    plan: Mapping[str, Any],
    *,
    target: str,
    software_sha256: str,
    capability_sha256: str,
    facts_sha256: str,
    host_capabilities_sha256: str,
) -> tuple[dict[str, str], dict, dict]:
    """Bind the selected capture recipe to canonical current-spec derivation.

    The Phase 0 export is a diagnostic transformation input, not an admission
    or a model-validation result.  Its independently checked source snapshots
    must match the very contracts used by this build, and its derived recipe
    must match the sealed capture preselection's actual recipe bytes. Return
    its resolved software and selected host views for independent source
    admission screening; never reinterpret the raw authored YAML here.
    """
    from merlin.targetgen.quant_recipe import digest as recipe_digest
    from merlin_experiments.phase0.evidence import load_exported_evidence

    root_value = row.get("recipe_derivation_root")
    if not isinstance(root_value, str) or not Path(root_value).is_absolute() or ".." in Path(root_value).parts:
        raise ValueError("private model needs an absolute recipe derivation root")
    root = Path(root_value)
    manifest = _file(spec, str(root / "evidence-manifest.json"), row.get("recipe_derivation_manifest_sha256"))
    evidence = load_exported_evidence(root)
    if evidence.target != target:
        raise ValueError("recipe derivation names another target")
    expected_sources = {
        "software-spec": software_sha256,
        "target-contract": capability_sha256,
        "rtl-facts": facts_sha256,
    }
    for role, digest in expected_sources.items():
        snapshots = [source for source in evidence.source_snapshots if source.role == role]
        if len(snapshots) != 1 or snapshots[0].sha256 != digest:
            raise ValueError(f"recipe derivation does not bind selected {role} bytes")
    host_profiles = evidence.host_capabilities
    if not isinstance(host_profiles, Mapping):
        raise ValueError("recipe derivation has no selected host profiles")
    host_sources = [
        source
        for source in evidence.source_snapshots
        if source.role.startswith("host-capability-spec:") and source.sha256 == host_capabilities_sha256
    ]
    if len(host_sources) != 1 or host_sources[0].role.removeprefix("host-capability-spec:") not in host_profiles:
        raise ValueError("recipe derivation does not bind selected host-capability bytes")
    selected_host_name = host_sources[0].role.removeprefix("host-capability-spec:")
    artifacts = dict(evidence.archived_artifacts)
    index_raw = artifacts.get("software/quantization-recipes.json")
    if index_raw is None:
        raise ValueError("recipe derivation has no verified recipe index")
    index = json.loads(index_raw)
    if (
        not isinstance(index, Mapping)
        or index.get("schema") != "merlin.phase0.capture_recipes.v1"
        or index.get("target") != target
        or index.get("software_review") != "reviewed"
        or not isinstance(index.get("recipes"), list)
    ):
        raise ValueError("recipe derivation has no reviewed target recipe index")
    selected = plan.get("recipe")
    if not isinstance(selected, Mapping):
        raise ValueError("complete model has no selected capture recipe")
    selected_file = _file(spec, selected.get("path"), selected.get("sha256"))
    candidates = [
        entry
        for entry in index["recipes"]
        if isinstance(entry, Mapping)
        and entry.get("sha256") == selected.get("sha256")
        and entry.get("recipe_sha256") == selected.get("recipe_sha256")
    ]
    if len(candidates) != 1:
        raise ValueError("selected capture recipe is absent from current-spec derivation")
    member = candidates[0].get("path")
    if not isinstance(member, str) or not member.startswith("software/quantization-recipes/"):
        raise ValueError("current-spec recipe has no safe indexed member")
    raw = artifacts.get(member)
    if raw is None or file_sha256(selected_file) != candidates[0]["sha256"]:
        raise ValueError("selected capture recipe differs from current-spec derived bytes")
    derived = json.loads(raw)
    if (
        not isinstance(derived, Mapping)
        or derived.get("target") != target
        or derived.get("recipe_sha256") != candidates[0]["recipe_sha256"]
        or recipe_digest(derived) != candidates[0]["recipe_sha256"]
        or raw != selected_file.read_bytes()
    ):
        raise ValueError("selected capture recipe has no verified current-spec content identity")
    proof = {
        "evidence_manifest_sha256": file_sha256(manifest),
        "recipe_sha256": candidates[0]["sha256"],
        "recipe_semantic_sha256": candidates[0]["recipe_sha256"],
        "scope": "current-spec diagnostic recipe derivation only; no model or hardware admission",
    }
    return proof, evidence.software_spec, {selected_host_name: host_profiles[selected_host_name]}


def _selected_admission_views(
    software: Any,
    profiles: Any,
    *,
    target: str,
    host_package: Path,
    host_package_sha256: str,
    host_capabilities_path: Path,
    host_capabilities_sha256: str,
) -> tuple[dict, dict]:
    """Use only a resolved exported spec and the exact selected host profile."""
    from merlin.targetgen.host_capabilities import validate_host_capabilities
    from merlin.targetgen.software_spec import validate_software_spec

    if not isinstance(software, Mapping) or not isinstance(software.get("operations"), list):
        raise ValueError("selected software admission view is not a normalized specification")
    checked_software = validate_software_spec(dict(software), target=target, source="verified Phase 0 export")
    if (
        checked_software != software
        or checked_software.get("status") != "reviewed"
        or any("hardware" in operation for operation in checked_software["operations"])
    ):
        raise ValueError("selected software admission view is not a reviewed resolved specification")
    if (
        not isinstance(profiles, Mapping)
        or host_capabilities_path.is_symlink()
        or file_sha256(host_capabilities_path) != host_capabilities_sha256
        or strict_tree_sha256(host_package)["sha256"] != host_package_sha256
    ):
        raise ValueError("selected host profile has no pinned package and capability document")
    document = yaml.safe_load(host_capabilities_path.read_bytes())
    if not isinstance(document, dict) or document.get("status") != "reviewed":
        raise ValueError("selected host profile has no reviewed capability document")
    compiler = document.get("compiler")
    if not isinstance(compiler, Mapping) or not isinstance(compiler.get("dtype_strategy"), str):
        raise ValueError("selected host profile has no precision lane")
    dtype_strategy = compiler["dtype_strategy"]
    validate_host_capabilities(document, package_sha256=host_package_sha256, dtype_strategy=dtype_strategy)
    if len(profiles) != 1:
        raise ValueError("selected host profile is not unique")
    profile_name, profile = next(iter(profiles.items()))
    if (
        not isinstance(profile_name, str)
        or not profile_name
        or not isinstance(profile, Mapping)
        or profile.get("status") != "reviewed"
        or profile.get("package_sha256") != host_package_sha256
        or profile.get("capability_spec_sha256") != host_capabilities_sha256
        or profile.get("dtype_strategy") != dtype_strategy
        or profile.get("capability_spec") != document
    ):
        raise ValueError("selected host profile differs from the pinned package or capability document")
    return checked_software, {profile_name: dict(profile)}


def _noncompute_support(row: Mapping[str, Any]) -> bool:
    """Only known control/data support may skip independent compute admission."""
    if row.get("disposition") != "support_required":
        return False
    operation = row.get("mlir_operation")
    if not isinstance(operation, str):
        raise ValueError("support-required operation has no source operation identity")
    if operation in _CONTROL_DATA_SUPPORT:
        return True
    if operation.startswith("linalg."):
        return False
    raise ValueError(f"support-required source operation has no audited lowering class: {operation}")


def _source_obligations(
    capture: Path,
    target: str,
    software: Mapping,
    capability: Mapping,
    host: Mapping,
    selected_index_observation: Mapping[str, Any] | None = None,
) -> dict:
    """Recompute eligibility over the actual source, independently of the candidate route."""
    from merlin.common import mlir_query as mq
    from merlin.frontends.capture_normalization import normalize_capture_mlir
    from merlin.targetgen import application_inventory as AI
    from merlin.targetgen import model_coverage as MC
    from merlin.targetgen import operation_accounting as OA
    from merlin.targetgen.eligibility import capability_map_from_contract, is_eligible
    from merlin.xdsl_dialects.lowering import compute_groups as CG
    from merlin_experiments.phase1.feedback.private_pool_support import prove_pool_source

    source = capture / "model.mlir"
    checked = AI.verify_capture_receipt(source)
    if checked.get("status") != "verified_materialized":
        raise ValueError(f"full-model capture receipt is not verified: {checked.get('errors')}")
    text, _ = normalize_capture_mlir(source.read_text(encoding="utf-8"))
    module = mq.parse(text)
    groups = CG.form_groups(module, target)
    plan = CG.plan(module, target, groups=groups)
    CG.require_explained(plan)
    owners = {id(member): group for group in groups for member in group.members}
    descriptors = MC.regions_from_module(module)
    operations = MC.region_ops(module)
    if len(descriptors) != len(operations):
        raise ValueError("source linalg inventory is incomplete")
    cap_map = capability_map_from_contract(dict(capability))
    source_sha = file_sha256(source)
    normalized_sha = sha256(text.encode("utf-8")).hexdigest()
    inventory = AI._application_operation_inventory(  # noqa: PLC2701 -- trusted exact inventory primitive
        source,
        target,
        cap_map,
        capability_contract=dict(capability),
        software_spec=dict(software),
        host_capabilities=dict(host),
    )
    bounded_control = control_support.prove_if_selected(
        module,
        inventory,
        raw_sha256=source_sha,
        normalized_sha256=normalized_sha,
        selected_observation=selected_index_observation,
    )
    direct_return_support = pure_stage.prove_if_empty(
        capture, module, host, source_sha, normalized_sha, inventory, descriptors
    )
    transpose_support = _transpose_data_support(
        module,
        inventory,
        raw_sha256=source_sha,
        normalized_sha256=normalized_sha,
    )
    generic_copy_support = data_movement.prove_generic_copy_source(
        module,
        inventory,
        raw_sha256=source_sha,
        normalized_sha256=normalized_sha,
    )
    parsed = tuple(mq.walk(module))
    source_rows = data_movement.source_inventory_by_ordinal(parsed, inventory, source_sha, normalized_sha)
    ordered_scan = ordered_scan_support.begin(
        source, source_sha, normalized_sha, checked["receipt_sha256"], selected_index_observation
    )
    ordered_scan_ordinals = ordered_scan_support.admit(
        ordered_scan, parsed, source_rows, software, capability, cap_map, host
    )
    linalg = linalg_support.begin(source_sha, normalized_sha, len(parsed), selected_index_observation)
    linkage = linkage_support.begin(source_sha, normalized_sha, len(parsed))
    arange = arange_support.begin(source_sha, normalized_sha, len(parsed), selected_index_observation)
    integer_reductions = integer_support.begin(source_sha, normalized_sha, len(parsed), selected_index_observation)
    maximum = maximum_support.begin(
        source_sha, normalized_sha, checked["receipt_sha256"], parsed, selected_index_observation
    )
    index_host = index_support.begin(
        capture, parsed, source_sha, normalized_sha, checked["receipt_sha256"], selected_index_observation
    )
    bucketize = bucketize_support.begin(capture, parsed, source_sha, normalized_sha, selected_index_observation)
    source_ordinals = {id(op): ordinal for ordinal, op in enumerate(parsed)}
    proven_ordinals = [*transpose_support["ordinals"], *generic_copy_support["ordinals"]]
    if len(proven_ordinals) != len(set(proven_ordinals)):
        raise ValueError("source movement proofs overlap")
    proven_movement = {id(parsed[ordinal]) for ordinal in proven_ordinals}
    eligible = []
    for index, (op, descriptor) in enumerate(zip(operations, descriptors, strict=True)):
        if id(op) in proven_movement:
            # Typed movement requires linking; it waives no compute admission.
            continue
        verdict = is_eligible(descriptor, cap_map)
        if verdict.undetermined:
            row = source_rows[source_ordinals[id(op)]]
            if row.get("disposition") not in {"host_required", "unclassified"}:
                raise ValueError(f"source operation {index} has unknown hardware eligibility")
            admission = OA.admit_operation_row(
                row,
                software_spec=dict(software),
                capability_contract=dict(capability),
                capability_map=cap_map,
                host_capabilities=dict(host),
                source_operations=tuple(parsed[ordinal] for ordinal in row["ordinals"]),
                source_context=host_source_dispatch.source_context(row, index_host, selected_index_observation),
            )
            observed = admission["observed_admission_signature"]
            hardware = admission["hardware_admission"]
            if (
                hardware.get("status") == "unsupported"
                and hardware.get("basis") == "selected_capability_contract"
                and hardware.get("refusal") in _EXPLICIT_HARDWARE_EXCLUSIONS
                and isinstance(observed.get("family"), str)
                and isinstance(observed.get("operand_dtype"), str)
                and isinstance(observed.get("ordered_result_dtypes"), list)
                and observed["ordered_result_dtypes"]
                and all(isinstance(dtype, str) and dtype for dtype in observed["ordered_result_dtypes"])
            ):
                # Source-joined hardware exclusion still requires reviewed host admission below.
                continue
            raise ValueError(f"source operation {index} has unknown hardware eligibility")
        if not verdict.eligible:
            continue
        group = owners.get(id(op))
        if group is None or group.placement == CG.HOST:
            raise ValueError(f"eligible source operation {index} has no accelerator group")
        eligible.append(group.index)

    pool_support = prove_pool_source(
        capture,
        module,
        inventory,
        raw_sha256=source_sha,
        normalized_sha256=normalized_sha,
    )
    unresolved = []
    support_lowering = 0
    for row in inventory["signatures"]:
        if ordered_scan_ordinals.intersection(row["ordinals"]):
            if not set(row["ordinals"]) <= ordered_scan_ordinals:
                raise ValueError("source ordered scan row mixes proved and unproved operations")
            # Only the reviewed root's source-proved closed body belongs to its admission.
            continue
        if row["disposition"] in {"structural", "component"}:
            continue
        if control_support.assertion_row_proven(row, bounded_control):
            continue
        if row["disposition"] == "support_required":
            if row["mlir_operation"] == "linalg.transpose":
                # The typed yield-only permutation still requires lowering and linking below.
                support_lowering += row["count"]
                continue
            elif row["mlir_operation"] == "linalg.generic" and row.get("semantic_family") == "movement":
                # The typed projected copy proof still requires the exact linked build.
                support_lowering += row["count"]
                continue
            elif _noncompute_support(row):
                # These are control/data-support instructions, not independent
                # compute decisions.  The real whole-program build below must
                # lower and link them; an unknown support op is not waived.
                support_lowering += row["count"]
                continue
            # A linalg movement is computation-carrying and its accelerator
            # eligibility/placement was checked in the source roster above.
        admission = OA.admit_operation_row(
            row,
            software_spec=dict(software),
            capability_contract=dict(capability),
            capability_map=cap_map,
            host_capabilities=dict(host),
            source_operations=tuple(parsed[ordinal] for ordinal in row["ordinals"]),
            source_context=host_source_dispatch.source_context(row, index_host, selected_index_observation),
        )
        accelerator = admission["accelerator_admission"]
        host_decision = admission["host_admission"]
        if accelerator["status"] == "admitted":
            # The exact source linalg inventory above must carry this demand.  A
            # standalone eligible op absent from grouping cannot disappear into host.
            if not row["mlir_operation"].startswith("linalg."):
                unresolved.append({"row": row, "admission": admission, "reason": "eligible operation is not outlined"})
        elif host_decision["status"] != "admitted" or host_decision.get("reviewed") is not True:
            unresolved.append(
                {"row": row, "admission": admission, "reason": "host operation lacks exact reviewed admission"}
            )
        else:
            host_source_dispatch.record(
                linalg,
                integer_reductions,
                index_host,
                row,
                host_decision,
                parsed,
                source_rows,
                bounded_control,
                maximum=maximum,
            )
            linkage_support.record(linkage, row, host_decision, source_rows, linalg)
            arange_support.record(arange, capture, row, host_decision, parsed, source_rows)
            bucketize_support.record(bucketize, row, host_decision, source_rows)
    if unresolved:
        raise SourceAdmissionError(unresolved)
    integer_support.verify_source(integer_reductions, source)
    maximum_support.verify_source(maximum, source)
    index_support.verify_source(index_host, capture)
    group_metrics = {}
    group_provenance = []
    seen_regions = set()
    for group in groups:
        if group.index not in eligible:
            continue
        region, nodes = _eligible_source_identity(group.root)
        if region in seen_regions:
            raise ValueError("eligible source groups repeat a provenance region")
        seen_regions.add(region)
        group_provenance.append(
            {"source_group": group.index, "source_region_id": region, "source_node_ids": list(nodes)}
        )
        values = (value for member in reversed(group.members) for value in member.results)
        for value in values:
            shape, _dtype = mq.type_shape_dtype(value.type)
            elements = 1
            for extent in shape:
                elements *= extent
            n_bytes = mq.value_bytes(value.type)
            if shape and elements > 0 and n_bytes > 0 and n_bytes % elements == 0:
                group_metrics[group.index] = {"elements": elements, "element_bytes": n_bytes // elements}
                break
        if group.index not in group_metrics:
            raise ValueError(f"eligible source group {group.index} has no exact static output extent")
    result = {
        "source_sha256": source_sha,
        "normalized_source_sha256": normalized_sha,
        "capture_receipt_sha256": checked["receipt_sha256"],
        "n_source_operations": inventory["n_operations"],
        "n_linalg_regions": len(descriptors),
        "n_groups": len(groups),
        "n_support_lowering_operations": support_lowering,
        "transpose_data_support": transpose_support,
        "generic_copy_data_support": generic_copy_support,
        "pool_value_support": pool_support,
        "linalg_host_support": linalg,
        linkage_support.FIELD: linkage,
        "literal_arange_host_support": arange,
        integer_support.FIELD: integer_reductions,
        maximum_support.FIELD: maximum,
        index_support.FIELD: index_host,
        ordered_scan_support.FIELD: ordered_scan,
        bucketize_support.FIELD: bucketize,
        "eligible_groups": sorted(set(eligible)),
        "eligible_group_provenance": group_provenance,
        "eligible_group_metrics": group_metrics,
        "host_groups": [group.index for group in groups if group.placement == CG.HOST],
        "host_justification": "source group reasons and exact reviewed host/software admissions",
    }
    control_support.attach_source_record(result, bounded_control)
    if direct_return_support is not None:
        result["direct_return_support"] = direct_return_support
    return result


def _linked_symbols(elf: Path, symbols: set[str]) -> None:
    from merlin.llvmlower import toolchain

    command = [str(toolchain.nm()), "--defined-only", "--extern-only", str(elf)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=120, check=True)
    defined = {line.split()[-1] for line in result.stdout.splitlines() if line.split()}
    missing = sorted(symbols - defined)
    if missing:
        raise ValueError(f"linked ELF omits {len(missing)} routed device symbol(s): {missing[:8]}")


def _verify_compiled_program(
    receipt: Mapping[str, Any],
    *,
    program: str,
    capture_path: Path | None = None,
    source: dict[str, Any],
    stage_tree: Mapping[str, Any],
    package_digest: Mapping[str, Any],
    target: str,
    board: str,
    catalog: Path,
    dts: Path,
    device_selected: bool,
    host_package: Path | None = None,
    host_package_tree_sha256: str | None = None,
    require_compilation_recipe: bool = True,
) -> dict[str, Any]:
    """Canonical post-build checks shared by freshly built and diagnostic images."""
    from merlin.llvmlower.device_offload import BY_GROUP

    expected_route = "device_requested_dispatch_unverified" if device_selected else "host_baseline"
    if (
        receipt.get("status") != "compiled"
        or receipt.get("execution_route") != expected_route
        or receipt.get("inputs", {}).get("capture_tree") != stage_tree
    ):
        raise ValueError(f"{program} did not capture/lower/codegen/link the selected source")
    _require_static_build_inputs(receipt, target=target, board=board, catalog=catalog, dts=dts)
    if device_selected and (receipt.get("inputs", {}).get("device") or {}).get("package_tree") != package_digest:
        raise ValueError(f"{program} build used a different candidate compiler tree")
    output = receipt.get("output") or {}
    elf = Path(str(output.get("elf") or ""))
    if not elf.is_file() or file_sha256(elf) != output.get("elf_sha256"):
        raise ValueError(f"{program} linked ELF is absent or changed")
    compilation_binding = compilation_support.verify(output, elf, required=require_compilation_recipe)
    sidecar_sha = None
    linked = 0
    host_compute_audit = []
    if device_selected:
        sidecar_record = output.get("device_sidecar") or {}
        sidecar = Path(str(sidecar_record.get("path") or ""))
        if not sidecar.is_file() or file_sha256(sidecar) != sidecar_record.get("sha256"):
            raise ValueError(f"{program} candidate device sidecar is absent or changed")
        emitted = json.loads(sidecar.read_text(encoding="utf-8"))
        if emitted.get("granularity") != BY_GROUP or emitted.get("device") != target:
            raise ValueError(f"{program} emitted another device or a partial route")
        assigned = emitted.get("routed")
        if not assigned or emitted.get("skipped"):
            raise ValueError(f"{program} linked roster differs from eligible source groups")
        _join_routed_source_groups(assigned, source)
        symbols = _routed_kernel_symbols(emitted)
        _linked_symbols(elf, symbols)
        host_compute_audit = _audit_built_device_host_compute(elf.parent, assigned, source, target)
        sidecar_sha, linked = sidecar_record["sha256"], len(assigned)
    linked_build = {
        "capture_tree_sha256": stage_tree["sha256"],
        "elf_sha256": output["elf_sha256"],
        "candidate_tree_sha256": package_digest["sha256"],
    }
    transpose_support = source["transpose_data_support"]
    if transpose_support["status"] != "source_structural_data_support_pending_build":
        raise ValueError(f"{program} has no pending source-bound transpose support proof")
    transpose_support["status"] = "source_structural_data_support_linked"
    transpose_support["linked_build"] = linked_build
    generic_copy_support = source["generic_copy_data_support"]
    if generic_copy_support["status"] != "source_structural_data_support_pending_build":
        raise ValueError(f"{program} has no pending source-bound generic copy proof")
    generic_copy_support["status"] = "source_structural_data_support_linked"
    generic_copy_support["linked_build"] = dict(linked_build)
    from merlin_experiments.phase1.feedback.private_pool_support import LINKED, PENDING

    pool_support = source["pool_value_support"]
    if pool_support["status"] != PENDING:
        raise ValueError(f"{program} has no pending source-bound pool-value proof")
    pool_support["status"] = LINKED
    pool_support["linked_build"] = dict(linked_build)
    pure_stage.link_direct_return(source, linked_build)
    index_lowering = control_support.link_selected_build(source, receipt, linked_build)
    linalg_support.link(source, index_lowering, linked_build, capture_path=capture_path)
    arange_support.link(source, index_lowering, linked_build)
    integer_support.link(source, index_lowering, linked_build)
    maximum_support.link(source, index_lowering, linked_build, capture_path=capture_path)
    index_support.link(source, index_lowering, linked_build, capture_path=capture_path)
    ordered_scan_support.link(source, index_lowering, linked_build)
    bucketize_support.link(source, index_lowering, linked_build, capture_path=capture_path)
    result = {
        "program": program,
        "status": "capture_lower_codegen_link_verified",
        "capture_tree_sha256": stage_tree["sha256"],
        "source_sha256": source["source_sha256"],
        "elf_sha256": output["elf_sha256"],
        "sidecar_sha256": sidecar_sha,
        "linked_device_groups": linked,
        "static_host_compute_audit": host_compute_audit,
        "candidate_tree_sha256": package_digest["sha256"],
        "index_lowering": index_lowering,
        "compilation_recipe": compilation_binding,
    }
    linkage_support.link_compiled(
        source, result, receipt, host_package, host_package_tree_sha256, catalog, board, dts, linked_build
    )
    return result


def run(
    submission: str | Path,
    private_spec: str | Path,
    *,
    target: str,
    required_models: Sequence[str],
    required_programs: Mapping[str, Sequence[str]],
    loader_env_requirements: Mapping[str, Mapping[str, Any]],
    out: str | Path,
    diagnostic_model: str | None = None,
    prebuilt_receipts: Mapping[str, str | Path] | None = None,
    source_freeze: Mapping[str, Any] | None = None,
    source_freeze_root: str | Path | None = None,
    linked_elf_admission=None,
) -> dict[str, Any]:
    """Build the frozen roster, or inspect one prebuilt model without producer attribution."""
    from merlin.compile.baremetal_model import compile_saved_model
    from merlin.compile.model_execution_inputs import selected_firrtl
    from merlin.compile.route_before_build import plan_before_build
    from merlin.llvmlower.device_offload import BY_GROUP
    from merlin.targetgen import target_registry

    instruction_selection = linked_policy.freeze(linked_elf_admission, target=target)
    spec = Path(private_spec)
    if spec.is_symlink() or not spec.is_file():
        raise ValueError("operator-private full-model specification is absent or indirect")
    spec_bytes = spec.read_bytes()
    spec_digest = sha256(spec_bytes).hexdigest()
    document = yaml.safe_load(spec_bytes)
    frozen_authored = source_freeze_api.resolve_optional(
        spec.absolute(), source_freeze, source_freeze_root, target=target
    )
    rows = document.get("models") if isinstance(document, Mapping) else None
    names = (
        [row.get("id") for row in rows]
        if isinstance(rows, list) and all(isinstance(row, Mapping) for row in rows)
        else []
    )
    if (
        not isinstance(document, Mapping)
        or document.get("schema") != SCHEMA
        or document.get("target") != target
        or len(names) != len(set(names))
        or set(names) != set(required_models)
        or len(names) != len(required_models)
        or set(required_programs) != set(required_models)
        or set(loader_env_requirements) != set(required_models)
    ):
        raise ValueError("private full-model specification differs from required target roster")
    diagnostic = diagnostic_model is not None
    if not diagnostic and frozen_authored is None:
        raise ValueError("canonical private full-model builds require a run-owned authored-source freeze")
    if diagnostic:
        if diagnostic_model not in required_models or not isinstance(prebuilt_receipts, Mapping):
            raise ValueError("prebuilt diagnostic must select one required model and its receipts")
        if set(prebuilt_receipts) != set(required_programs[diagnostic_model]):
            raise ValueError("prebuilt diagnostic receipts differ from the complete selected model")
    elif prebuilt_receipts is not None:
        raise ValueError("prebuilt receipts cannot enter the canonical full-model build")
    package = Path(submission)
    package_digest = strict_tree_sha256(package)
    base = Path(out)
    base.mkdir(mode=0o700, parents=True, exist_ok=True)
    results = []
    for row in rows:
        name = row["id"]
        if diagnostic and name != diagnostic_model:
            continue
        item: dict[str, Any] = {"model": name, "status": "fail", "checks": {}}
        results.append(item)
        try:
            identity = row.get("source_identity") or {}
            if identity != {"model_id": name, "scope": "complete_network", "role": "private_validation"}:
                raise ValueError("model has no reviewed complete-network validation identity")
            from merlin_experiments.phase0 import capture_selection as capture_selection_api

            selection_path = _file(spec, row.get("capture_selection"), row.get("capture_selection_sha256"))
            selection = capture_selection_api.load(selection_path, expected_sha256=row["capture_selection_sha256"])
            selected_env = (selection.get("plan") or {}).get("loader_env") or {}
            env_requirements = loader_env_requirements[name]
            selected_plan = selection.get("plan") or {}
            _require_selected_input_bindings(row, selected_plan)
            selected_roles = {
                entry.get("role") for entry in selected_plan.get("selected_inputs") or [] if isinstance(entry, Mapping)
            }
            if (
                selection.get("schema") != capture_selection_api.SCHEMA_V2
                or Path(str(selected_plan.get("workload_root") or "")).name != env_requirements["workload_dir"]
                or not set(env_requirements["selected_roles"]).issubset(selected_roles)
                or row.get("deployment_dtype") != env_requirements["deployment_dtype"]
            ):
                raise ValueError("preselected source, checkpoint or deployment differs from the full model")
            if (
                not isinstance(selected_env, Mapping)
                or any(selected_env.get(key) != value for key, value in env_requirements["required"].items())
                or any(selected_env.get(key) is not None for key in env_requirements["forbidden"])
            ):
                raise ValueError("preselected loader environment permits a truncated or diagnostic model")
            capture = Path(str(row.get("capture") or ""))
            capture = capture if capture.is_absolute() else spec.parent / capture
            if capture.is_symlink() or capture.resolve() != Path(selection["run_dir"]) / "capture":
                raise ValueError("full-model capture is not the preselected sealed run output")
            from merlin_experiments.phase0.capture_execution_attestation import require_verified_execution

            attestation_path = Path(str(row.get("capture_execution_attestation") or ""))
            attestation_path = attestation_path if attestation_path.is_absolute() else spec.parent / attestation_path
            if attestation_path.is_symlink() or not attestation_path.is_file():
                raise ValueError("preselected capture has no safe post-execution attestation")
            attestation_sha = file_sha256(attestation_path)
            if row.get("capture_execution_attestation_sha256") not in (None, attestation_sha):
                raise ValueError("capture execution attestation differs from an explicitly pinned digest")
            attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
            require_verified_execution(attestation)
            attested_selection = attestation.get("selection") or {}
            if (
                Path(str(attested_selection.get("path") or "")).resolve() != selection_path
                or attested_selection.get("sha256") != row["capture_selection_sha256"]
            ):
                raise ValueError("capture execution attestation names another preselection")
            attested_capture = attestation.get("capture") or {}
            capture_identity = _capture_tree_bindings(
                capture, row.get("capture_tree_sha256"), attested_capture.get("capture_tree_sha256")
            )
            item["checks"]["capture_identity"] = capture_identity
            expected_stages = tuple(required_programs[name])
            captured_kind = "session" if (capture / "session-receipt.json").exists() else "single"
            if (
                attestation.get("issuer") != "merlin.sealed_m2m_cpu.v3"
                or attested_capture.get("kind") != captured_kind
                or Path(str(attested_capture.get("capture_path") or "")).resolve() != capture
            ):
                raise ValueError("verified source-execution attestation names another complete capture")
            if captured_kind == "single":
                if (
                    expected_stages != ("model",)
                    or attested_capture.get("model_sha256") != file_sha256(capture / "model.mlir")
                    or attested_capture.get("receipt_sha256") != file_sha256(capture / "capture_receipt.json")
                ):
                    raise ValueError("attested single-network bytes differ from the full model")
            elif (
                [stage.get("name") for stage in (attested_capture.get("programs") or [])] != list(expected_stages)
                or attested_capture.get("session_contract_sha256") != file_sha256(capture / "session_contract.yaml")
                or attested_capture.get("session_receipt_sha256") != file_sha256(capture / "session-receipt.json")
            ):
                raise ValueError("attested session omits a required full-model program")
            if env_requirements["require_integer_contractions"] and (
                type(attested_capture.get("integer_contractions")) is not int
                or attested_capture["integer_contractions"] < 1
            ):
                raise ValueError("capture has no independently attested integerized accelerator computation")
            programs = _captured_programs(capture, expected_stages)
            item["checks"]["input_provenance"] = _captured_input_provenance(programs, _expected_input_provenance(row))
            facts_path = _file(spec, row.get("rtl_facts"), row.get("rtl_facts_sha256"))
            facts = selected_firrtl(facts_path, target=target, config=str(row.get("rtl_config") or ""))
            ambient_facts = os.environ.get("MERLIN_RTL_FACTS", "").strip()
            if not ambient_facts or file_sha256(Path(ambient_facts)) != facts["sha256"]:
                raise ValueError("formal source and placement readers are not bound to selected RTL facts")
            host_package = _tree(spec, row.get("host_package"), row.get("host_package_tree_sha256"))
            software_path = _authored_file(spec, row, "software_spec", frozen_authored)
            capability_path = _file(spec, row.get("capability_contract"), row.get("capability_contract_sha256"))
            host_path = _authored_file(spec, row, "host_capabilities", frozen_authored)
            catalog = _file(spec, row.get("board_catalog"), row.get("board_catalog_sha256"))
            dts = _file(spec, row.get("host_dts"), row.get("host_dts_sha256"))
            board_catalog_sha256 = file_sha256(catalog)
            board_dts_sha256 = file_sha256(dts)
            recipe_proof, exported_software, exported_host_profiles = _recipe_derivation(
                spec,
                row,
                selected_plan,
                target=target,
                software_sha256=file_sha256(software_path),
                capability_sha256=file_sha256(capability_path),
                facts_sha256=facts["sha256"],
                host_capabilities_sha256=file_sha256(host_path),
            )
            item["checks"]["recipe_derivation"] = recipe_proof
            software, host = _selected_admission_views(
                exported_software,
                exported_host_profiles,
                target=target,
                host_package=host_package,
                host_package_sha256=row["host_package_tree_sha256"],
                host_capabilities_path=host_path,
                host_capabilities_sha256=file_sha256(host_path),
            )
            capability = yaml.safe_load(capability_path.read_bytes())
            if not isinstance(capability, Mapping):
                raise ValueError("selected software, capability or host declaration is malformed")
            if software.get("target") != target or capability.get("name") != target:
                raise ValueError("selected software or capability declaration names another target")
            selected_index_observation = control_support.selected_build_observation(
                host_package, catalog, str(row["board"]), target
            )
            with target_registry.observed_contract(target, dict(capability), source_path=capability_path):
                sources = {
                    program: _source_obligations(stage, target, software, capability, host, selected_index_observation)
                    for program, stage in programs
                }
            if not any(source["eligible_groups"] for source in sources.values()):
                raise ValueError("full model has no independently eligible accelerator computation")
            item["checks"]["source"] = sources
            compiled = []
            diagnostic_receipts = {}
            for program, stage in programs:
                source = sources[program]
                stage_tree = strict_tree_sha256(stage)
                device = None
                if source["eligible_groups"]:
                    with target_registry.observed_contract(target, dict(capability), source_path=capability_path):
                        routed = plan_before_build(
                            target,
                            (stage / "model.mlir").read_text(encoding="utf-8"),
                            datapath=str(row["deployment_dtype"]),
                            device_package=str(package),
                            model=f"{name}:{program}",
                            capture=stage / "model.mlir",
                            granularity=BY_GROUP,
                            linked_elf_admission=linked_elf_admission,
                        )
                    device = routed.get("device_routing")
                    if device is None or routed.get("offload") is None:
                        raise ValueError(
                            f"{program} has no buildable accelerator route: {routed.get('device_routing_why')}"
                        )
                linked_policy.require_route(device, instruction_selection)
                if diagnostic:
                    receipt_path = Path(prebuilt_receipts[program])
                    receipt = load_diagnostic_receipt(
                        receipt_path,
                        capture=stage,
                        host_package=host_package,
                        candidate=package,
                        host_package_tree=strict_tree_sha256(host_package),
                        candidate_tree=package_digest,
                        arena_mb=int(row["arena_mb"]),
                        target=target,
                        device_selected=device is not None,
                        selected_rtl_facts=facts,
                    )
                    diagnostic_receipts[program] = {
                        "path": str(receipt_path),
                        "sha256": file_sha256(receipt_path),
                        "historical_rtl_facts_identity": (
                            "unavailable_in_prebuilt_receipt"
                            if receipt["inputs"].get("rtl_facts_identity") is None
                            else "recorded_exact_selected"
                        ),
                    }
                else:
                    with target_registry.observed_contract(target, dict(capability), source_path=capability_path):
                        math_symbols = linkage_support.symbols(source)
                        receipt = compile_saved_model(
                            capture=stage,
                            package=host_package,
                            board_catalog=catalog,
                            board=str(row["board"]),
                            dts=dts,
                            output=base / name / program,
                            target=target,
                            run="none",
                            arena_mb=int(row["arena_mb"]),
                            device=device,
                            **({"math_archive_symbols": math_symbols} if math_symbols else {}),
                        )
                linked_policy.require_route(device, instruction_selection)
                compiled.append(
                    _verify_compiled_program(
                        receipt,
                        program=program,
                        capture_path=stage,
                        source=source,
                        stage_tree=stage_tree,
                        package_digest=package_digest,
                        target=target,
                        board=str(row["board"]),
                        catalog=catalog,
                        dts=dts,
                        device_selected=device is not None,
                        host_package=host_package,
                        host_package_tree_sha256=row["host_package_tree_sha256"],
                        require_compilation_recipe=not diagnostic,
                    )
                )
            if (
                _capture_tree_bindings(
                    capture, row.get("capture_tree_sha256"), attested_capture.get("capture_tree_sha256")
                )
                != capture_identity
            ):
                raise ValueError("private model changed during candidate build")
            if selected_firrtl(facts_path, target=target, config=facts["config"]) != facts:
                raise ValueError("selected RTL facts changed during candidate build")
            if strict_tree_sha256(package) != package_digest:
                raise ValueError("candidate compiler changed during model build")
            if diagnostic and any(
                file_sha256(Path(entry["path"])) != entry["sha256"] for entry in diagnostic_receipts.values()
            ):
                raise ValueError("prebuilt diagnostic receipt changed during static verification")
            pinned_files = (
                (selection_path, row["capture_selection_sha256"]),
                (attestation_path, attestation_sha),
                (facts_path, row["rtl_facts_sha256"]),
                (software_path, row["software_spec_sha256"]),
                (capability_path, row["capability_contract_sha256"]),
                (host_path, row["host_capabilities_sha256"]),
                (catalog, row["board_catalog_sha256"]),
                (dts, row["host_dts_sha256"]),
            )
            if any(file_sha256(path) != digest for path, digest in pinned_files):
                raise ValueError("reviewed model input changed during linked-build verification")
            if strict_tree_sha256(host_package)["sha256"] != row["host_package_tree_sha256"]:
                raise ValueError("selected host package changed during linked-build verification")
            if (
                _recipe_derivation(
                    spec,
                    row,
                    selected_plan,
                    target=target,
                    software_sha256=file_sha256(software_path),
                    capability_sha256=file_sha256(capability_path),
                    facts_sha256=facts["sha256"],
                    host_capabilities_sha256=file_sha256(host_path),
                )[0]
                != recipe_proof
            ):
                raise ValueError("selected recipe derivation changed during linked-build verification")
            item["checks"]["build"] = {
                "programs": compiled,
                "linked_device_groups": sum(entry["linked_device_groups"] for entry in compiled),
                "candidate_tree_sha256": package_digest["sha256"],
                "static_build_board": str(row["board"]),
                "static_build_board_catalog_source": str(catalog),
                "static_build_board_catalog_sha256": board_catalog_sha256,
                "static_build_board_dts_source": str(dts),
                "static_build_board_dts_sha256": board_dts_sha256,
                "static_build_board_scope": BUILD_BOARD_SCOPE,
                "accelerator_rtl_facts_sha256": facts["sha256"],
                "accelerator_rtl_config": facts["config"],
                "status": "capture_lower_codegen_link_verified",
            }
            if diagnostic:
                item["checks"]["prebuilt_receipts"] = diagnostic_receipts
            linked_policy.unchanged(instruction_selection)
            item["status"] = "diagnostic_static_checks_passed" if diagnostic else "pass"
        except Exception as exc:  # noqa: BLE001 -- one model's refusal must not hide the others
            item["reason"] = _public_failure_reason(exc)
    result = {
        "schema": RESULT_SCHEMA,
        "target": target,
        "private_spec_sha256": spec_digest,
        "candidate_tree_sha256": package_digest["sha256"],
        "required_models": list(required_models),
        "required_programs": {name: list(required_programs[name]) for name in required_models},
        "models": results,
        "passed": not diagnostic and bool(results) and all(item["status"] == "pass" for item in results),
        "scope": (
            "diagnostic selected prebuilt model only; producer and toolchain unbound; not a full-model gate"
            if diagnostic
            else "complete captured networks, static source and linked-ELF evidence; full-model execution deferred"
        ),
        "full_model_numerical_equivalence": "not_run",
        "paper_accuracy": "not_claimed",
    }
    if file_sha256(spec) != spec_digest:
        raise ValueError("operator-private full-model specification changed during verification")
    result.update(source_freeze_api.claim_binding(spec.absolute(), source_freeze, source_freeze_root, target=target))
    if diagnostic:
        result["mode"] = "prebuilt_diagnostic"
        result["diagnostic_model"] = diagnostic_model
        result["diagnostic_passed"] = len(results) == 1 and results[0]["status"] == "diagnostic_static_checks_passed"
        result["producer_binding"] = "unavailable_in_prebuilt_receipt"
    report = base / "private_full_model_gate.json"
    with os.fdopen(os.open(report, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    return result


def _complete_capture_identity(checks: Mapping[str, Any]) -> bool:
    identity = checks.get("capture_identity")
    if not isinstance(identity, Mapping) or set(identity) != {
        "compiler_strict_tree_sha256",
        "issuer_sealed_tree_sha256",
    }:
        return False
    return all(
        isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)
        for value in identity.values()
    )


def complete(
    record: Mapping[str, Any] | None,
    *,
    required_models: Sequence[str],
    required_programs: Mapping[str, Sequence[str]],
    candidate_sha256: str,
) -> bool:
    """Revalidate the claim-bearing roster and exact current submission identity."""
    if (
        not isinstance(record, Mapping)
        or record.get("schema") != RESULT_SCHEMA
        or record.get("passed") is not True
        or record.get("mode") == "prebuilt_diagnostic"
    ):
        return False
    if record.get("full_model_numerical_equivalence") != "not_run" or record.get("paper_accuracy") != "not_claimed":
        return False
    if not source_freeze_api.valid_binding(record.get("authored_source_freeze"), record.get("private_spec_sha256")):
        return False
    rows = record.get("models")
    if not isinstance(rows, list) or len(rows) != len(required_models):
        return False
    if any(
        not isinstance(row, Mapping)
        or not isinstance(row.get("checks"), Mapping)
        or not _complete_capture_identity(row["checks"])
        or not isinstance(row["checks"].get("input_provenance"), Mapping)
        or set(row["checks"]["input_provenance"]) != set(required_programs.get(row.get("model"), ()))
        or any(
            not isinstance(provenance, Mapping)
            or type(provenance.get("paper_ready")) not in {bool, type(None)}
            or type(provenance.get("synthetic_inputs")) not in {bool, type(None)}
            or not isinstance(provenance.get("meta_sha256"), str)
            or len(provenance["meta_sha256"]) != 64
            or provenance.get("scope") != "input provenance only; no paper accuracy or full-model numerical result"
            for provenance in row["checks"]["input_provenance"].values()
        )
        or not isinstance(row["checks"].get("build"), Mapping)
        or not _linked_source_support_complete(
            row["checks"], required_programs.get(row.get("model"), ()), candidate_sha256
        )
        or not isinstance(row["checks"].get("recipe_derivation"), Mapping)
        or row["checks"]["recipe_derivation"].get("scope")
        != "current-spec diagnostic recipe derivation only; no model or hardware admission"
        or any(
            not isinstance(row["checks"]["recipe_derivation"].get(key), str)
            or len(row["checks"]["recipe_derivation"][key]) != 64
            for key in ("evidence_manifest_sha256", "recipe_sha256", "recipe_semantic_sha256")
        )
        for row in rows
    ):
        return False
    return (
        record.get("candidate_tree_sha256") == candidate_sha256
        and record.get("required_models") == list(required_models)
        and record.get("required_programs") == {name: list(required_programs[name]) for name in required_models}
        and {row.get("model") for row in rows if isinstance(row, Mapping) and row.get("status") == "pass"}
        == set(required_models)
        and all(
            (row.get("checks") or {}).get("build", {}).get("candidate_tree_sha256") == candidate_sha256
            and (row.get("checks") or {}).get("build", {}).get("status") == "capture_lower_codegen_link_verified"
            and isinstance((row.get("checks") or {}).get("build", {}).get("static_build_board"), str)
            and all(
                isinstance((row.get("checks") or {}).get("build", {}).get(key), str)
                and Path((row.get("checks") or {}).get("build", {})[key]).is_absolute()
                for key in ("static_build_board_catalog_source", "static_build_board_dts_source")
            )
            and (row.get("checks") or {}).get("build", {}).get("static_build_board_scope") == BUILD_BOARD_SCOPE
            and isinstance((row.get("checks") or {}).get("build", {}).get("accelerator_rtl_config"), str)
            and bool((row.get("checks") or {}).get("build", {}).get("accelerator_rtl_config"))
            and all(
                isinstance((row.get("checks") or {}).get("build", {}).get(key), str)
                and len((row.get("checks") or {}).get("build", {})[key]) == 64
                for key in (
                    "static_build_board_catalog_sha256",
                    "static_build_board_dts_sha256",
                    "accelerator_rtl_facts_sha256",
                )
            )
            and type((row.get("checks") or {}).get("build", {}).get("linked_device_groups")) is int
            and (row.get("checks") or {}).get("build", {}).get("linked_device_groups") > 0
            and isinstance((row.get("checks") or {}).get("build", {}).get("programs"), list)
            and (row.get("checks") or {}).get("build", {}).get("linked_device_groups")
            == sum(
                entry["linked_device_groups"]
                for entry in (row.get("checks") or {}).get("build", {}).get("programs", [])
                if isinstance(entry, Mapping) and type(entry.get("linked_device_groups")) is int
            )
            # Every program states its own linked count: one that omits it is not summed as zero.
            and all(
                isinstance(entry, Mapping)
                and type(entry.get("linked_device_groups")) is int
                and compilation_support.complete(entry)
                for entry in (row.get("checks") or {}).get("build", {}).get("programs", [])
            )
            and [entry.get("program") for entry in (row.get("checks") or {}).get("build", {}).get("programs", [])]
            == list(required_programs[row["model"]])
            and all(
                entry.get("status") == "capture_lower_codegen_link_verified"
                and entry.get("candidate_tree_sha256") == candidate_sha256
                and isinstance(entry.get("elf_sha256"), str)
                and len(entry["elf_sha256"]) == 64
                and type(entry.get("linked_device_groups")) is int
                and isinstance(entry.get("static_host_compute_audit"), list)
                and len(entry["static_host_compute_audit"]) == entry["linked_device_groups"]
                and all(
                    isinstance(audit, Mapping)
                    and audit.get("verdict") == "clean_static_host_compute_audit"
                    and isinstance(audit.get("artifact_sha256"), str)
                    and len(audit["artifact_sha256"]) == 64
                    and isinstance(audit.get("object_sha256"), str)
                    and len(audit["object_sha256"]) == 64
                    and isinstance(audit.get("audit"), Mapping)
                    and isinstance(audit["audit"].get("budget"), Mapping)
                    and isinstance(audit["audit"].get("groups"), list)
                    and bool(audit["audit"]["groups"])
                    and all(group.get("verdict") == "clean" for group in audit["audit"]["groups"])
                    for audit in entry["static_host_compute_audit"]
                )
                for entry in (row.get("checks") or {}).get("build", {}).get("programs", [])
            )
            for row in rows
        )
    )
