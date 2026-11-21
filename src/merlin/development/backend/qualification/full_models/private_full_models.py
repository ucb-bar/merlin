"""Build-only qualification against operator-private frozen model inputs.

Full-model execution and representative-capsule L3 grades remain independent.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from typing import Any

import yaml

from merlin.development.backend.qualification.full_models import audits as program_checks
from merlin.development.backend.qualification.full_models import (
    private_compilation_inputs as compilation_support,
)
from merlin.development.backend.qualification.full_models import (
    private_source_optimizations as source_optimizations,
)
from merlin.development.backend.qualification.full_models import private_static_products as static_products
from merlin.development.backend.qualification.full_models.audits import (
    audit_built_device_host_compute as _audit_built_device_host_compute,
)
from merlin.development.backend.qualification.full_models.audits import load_diagnostic_receipt
from merlin.development.backend.qualification.full_models.audits import (
    require_linked_symbols as _linked_symbols,
)
from merlin.development.backend.qualification.full_models.audits import (
    require_static_build_inputs as _require_static_build_inputs,
)
from merlin.development.backend.qualification.full_models.completion import (
    BUILD_BOARD_SCOPE,
    RESULT_SCHEMA,
    _complete_capture_identity,
    compile_saved_model,
    complete,
)
from merlin.development.backend.qualification.full_models.private_capture_roster import (
    captured_input_provenance as _captured_input_provenance,
)
from merlin.development.backend.qualification.full_models.private_capture_roster import (
    captured_programs as _captured_programs,
)
from merlin.development.backend.qualification.full_models.private_model_inputs import (
    _capture_tree_bindings as _verify_capture_bindings,
)
from merlin.development.backend.qualification.full_models.private_model_inputs import (
    _expected_input_provenance,
    _require_selected_input_bindings,
    _selected_admission_views,
)
from merlin.development.backend.qualification.full_models.private_model_inputs import (
    _recipe_derivation as _derive_recipe_binding,
)
from merlin.development.backend.qualification.full_models.private_model_inputs import (
    _tree as _verify_input_tree,
)
from merlin.development.backend.qualification.full_models.private_source_preflight import (
    _notify as _source_progress,
)
from merlin.development.backend.qualification.full_models.private_source_preflight import (
    source_result as _source_result,
)
from merlin.development.backend.qualification.provenance import private_model_math_policy as math_policy
from merlin.development.backend.qualification.provenance import private_source_demand as source_demand
from merlin.development.backend.qualification.provenance import private_source_freeze as source_freeze_api
from merlin.development.backend.qualification.provenance.groups import (
    convolution_input_preparations,
    source_group_roster,
    verify_routed_source_coverage,
)
from merlin.development.backend.qualification.provenance.groups import (
    eligible_source_identity as _eligible_source_identity,
)
from merlin.development.backend.qualification.provenance.groups import (
    join_routed_source_groups as _join_routed_source_groups,
)
from merlin.development.backend.qualification.provenance.groups import (
    routed_kernel_symbols as _routed_kernel_symbols,
)
from merlin.development.backend.qualification.provenance.private_source_demand import (
    exact_source_groups as _exact_source_groups,
)
from merlin.development.backend.qualification.source_witnesses import bucketize as bucketize_support
from merlin.development.backend.qualification.source_witnesses import control_guards as control_support
from merlin.development.backend.qualification.source_witnesses import data_movement
from merlin.development.backend.qualification.source_witnesses import integer_reductions as integer_support
from merlin.development.backend.qualification.source_witnesses import linalg_bodies as linalg_support
from merlin.development.backend.qualification.source_witnesses import ordered_scan as ordered_scan_support
from merlin.development.backend.qualification.source_witnesses import (
    ordered_scan_admission as _merlin_ordered_scan_admission_3,
)
from merlin.development.backend.qualification.source_witnesses import (
    private_f32_maximum_support as maximum_support,
)
from merlin.development.backend.qualification.source_witnesses import (
    private_f32_reduction_support as f32_sum_support,
)
from merlin.development.backend.qualification.source_witnesses import (
    private_host_source_dispatch as host_source_dispatch,
)
from merlin.development.backend.qualification.source_witnesses import (
    private_index_host_support as index_support,
)
from merlin.development.backend.qualification.source_witnesses import (
    private_linkage_support as linkage_support,
)
from merlin.development.backend.qualification.source_witnesses import (
    private_linked_elf_selection as linked_policy,
)
from merlin.development.backend.qualification.source_witnesses import (
    private_literal_arange_admission as arange_support,
)
from merlin.development.backend.qualification.source_witnesses import pure_stages as pure_stage
from merlin.development.backend.qualification.source_witnesses.integer_reductions import (
    contraction_zero_init_fill as _contraction_zero_init_fill,
)
from merlin.development.backend.qualification.source_witnesses.private_source_support_join import (
    linked_source_support_complete as _linked_source_support_complete,
)
from merlin.runtime.execution.model_inputs import file_sha256, strict_tree_sha256

# ---- private_full_models -----------------------------------------------------------------------

complete.__module__ = __name__

_complete_capture_identity.__module__ = __name__

SCHEMA = "merlin.phase1.private_full_models.v1"

SOURCE_PREFLIGHT_SCOPE = "compiler-independent private inputs and source obligations only; no build or gate claim"

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


def declares_full_models(descriptor: str | Path) -> bool:
    """Whether the target declares a required private full-model roster (the full-model Phase 1 flow)."""
    return bool(requirements_for(descriptor))


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
        from merlin.development.testgen.sealed_capture import selection as capture_selection_api

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


def _tree(spec: Path, value: Any, digest: Any) -> Path:
    return _verify_input_tree(spec, value, digest, tree_digest=strict_tree_sha256)


def _capture_tree_bindings(capture: Path, pinned_strict: Any, attested_sealed: Any) -> dict[str, str]:
    return _verify_capture_bindings(capture, pinned_strict, attested_sealed, tree_digest=strict_tree_sha256)


def _recipe_derivation(spec: Path, row: Mapping[str, Any], plan: Mapping[str, Any], **bindings):
    return _derive_recipe_binding(spec, row, plan, pinned_file=_file, file_digest=file_sha256, **bindings)


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


@host_source_dispatch.prepared_host_admission
def _source_obligations(
    capture: Path,
    target: str,
    software: Mapping,
    capability: Mapping,
    host: Mapping,
    selected_index_observation: Mapping[str, Any] | None = None,
    *,
    facts_sha256: str | None = None,
    numeric_policy_sha256: str | None = None,
    progress=None,
) -> dict:
    """Recompute eligibility over the actual source, independently of the candidate route."""
    from merlin.compiler.frontend.patterns.capture_normalization import normalize_capture_mlir
    from merlin.compiler.ir import mlir_query as mq
    from merlin.compiler.lower.device.source_compute_groups import form_source_groups
    from merlin.compiler.partition.groups import compute_groups as CG
    from merlin.compiler.partition.groups import plan_compute_groups as _merlin_plan_compute_groups_2
    from merlin.development.backend.qualification.source_witnesses.pooling import prove_pool_source
    from merlin.development.targetgen.applications import demand_inventory as _merlin_demand_inventory_1
    from merlin.development.targetgen.applications import inventory as AI
    from merlin.development.targetgen.applications import operation_accounting as OA
    from merlin.development.targetgen.conformance import model_coverage as MC
    from merlin.target.capabilities.eligibility import capability_map_from_contract, is_eligible

    source = capture / "model.mlir"
    _source_progress(progress, "capture_receipt")
    checked = AI.verify_capture_receipt(source)
    if checked.get("status") != "verified_materialized":
        raise ValueError(f"full-model capture receipt is not verified: {checked.get('errors')}")
    _source_progress(progress, "normalization")
    text, _ = normalize_capture_mlir(source.read_text(encoding="utf-8"))
    _source_progress(progress, "parse")
    module = (prepared_source := _merlin_demand_inventory_1.PreparedApplicationSource(source, text)).module
    _source_progress(progress, "group_formation")
    groups, exact_float_refusals = _exact_source_groups(module, target, capture, form_source_groups)
    plan = _merlin_plan_compute_groups_2.plan(module, target, groups=groups)
    _merlin_plan_compute_groups_2.require_explained(plan)
    owners = {id(member): group for group in groups for member in group.members}
    descriptors = MC.regions_from_module(module)
    operations = MC.region_ops(module)
    if len(descriptors) != len(operations):
        raise ValueError("source linalg inventory is incomplete")
    cap_map = capability_map_from_contract(dict(capability))
    source_sha = file_sha256(source)
    normalized_sha = sha256(text.encode("utf-8")).hexdigest()
    raw_verdicts = tuple(is_eligible(descriptor, cap_map) for descriptor in descriptors)
    _source_progress(progress, "source_demand")
    demand = source_demand.derive(
        groups,
        operations,
        raw_verdicts,
        source_sha256=source_sha,
        normalized_sha256=normalized_sha,
        capability=capability,
        software=software,
        facts_sha256=facts_sha256,
        numeric_policy_sha256=numeric_policy_sha256,
        source_operations=tuple(mq.walk(module)),
    )
    _source_progress(progress, "operation_inventory")
    inventory = _merlin_demand_inventory_1._application_operation_inventory(  # noqa: PLC2701 -- trusted exact inventory primitive
        source,
        target,
        cap_map,
        capability_contract=dict(capability),
        software_spec=dict(software),
        host_capabilities=host,
        prepared_source=prepared_source,
    )
    _source_progress(progress, "source_support")
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
    ordered_scan = _merlin_ordered_scan_admission_3.begin(
        source, source_sha, normalized_sha, checked["receipt_sha256"], selected_index_observation
    )
    ordered_scan_ordinals = _merlin_ordered_scan_admission_3.admit(
        ordered_scan, parsed, source_rows, software, capability, cap_map, host
    )
    linalg = linalg_support.begin(source_sha, normalized_sha, len(parsed), selected_index_observation)
    linkage = linkage_support.begin(source_sha, normalized_sha, len(parsed))
    arange = arange_support.begin(source_sha, normalized_sha, len(parsed), selected_index_observation)
    integer_reductions = integer_support.begin(source_sha, normalized_sha, len(parsed), selected_index_observation)
    f32_reductions = f32_sum_support.begin(source_sha, normalized_sha, len(parsed), selected_index_observation)
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
    eligible = [group.index for group in groups if group.placement != CG.HOST]
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
                host_capabilities=host,
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
    source_demand.require_raw_reconciliation(demand)
    pool_support = prove_pool_source(
        capture,
        module,
        inventory,
        raw_sha256=source_sha,
        normalized_sha256=normalized_sha,
    )
    _source_progress(progress, "host_admission")
    unresolved = []
    support_lowering = 0
    for row_index, row in enumerate(inventory["signatures"]):
        if row_index % 128 == 0:
            _source_progress(progress, "host_admission", current=row_index, total=len(inventory["signatures"]))
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
            if _contraction_zero_init_fill(row, parsed):
                support_lowering += row["count"]
                continue
            if row["mlir_operation"] == "linalg.transpose":
                # The typed yield-only permutation still requires lowering and linking below.
                support_lowering += row["count"]
                continue
            elif row["mlir_operation"] == "linalg.generic" and row.get("semantic_family") == "movement":
                # The typed projected copy proof still requires the exact linked build.
                support_lowering += row["count"]
                continue
            elif _noncompute_support(row):
                # The real build must lower/link these control/data-support
                # instructions; an unknown support op is not waived.
                support_lowering += row["count"]
                continue
            # A linalg movement is computation-carrying and its accelerator
            # eligibility/placement was checked in the source roster above.
        admission = OA.admit_operation_row(
            row,
            software_spec=dict(software),
            capability_contract=dict(capability),
            capability_map=cap_map,
            host_capabilities=host,
            source_operations=tuple(parsed[ordinal] for ordinal in row["ordinals"]),
            source_context=host_source_dispatch.source_context(row, index_host, selected_index_observation),
        )
        accelerator = admission["accelerator_admission"]
        host_decision = admission["host_admission"]
        hardware = admission["hardware_admission"]
        if accelerator["status"] == "admitted":
            # The exact source linalg inventory above must carry this demand.  A
            # standalone eligible op absent from grouping cannot disappear into host.
            if not row["mlir_operation"].startswith("linalg."):
                unresolved.append({"row": row, "admission": admission, "reason": "eligible operation is not outlined"})
        elif (
            hardware.get("status") != "unsupported"
            or hardware.get("basis") != "selected_capability_contract"
            or hardware.get("refusal") not in _EXPLICIT_HARDWARE_EXCLUSIONS
        ):
            unresolved.append(
                {"row": row, "admission": admission, "reason": "host compute is not proved hardware-unsupported"}
            )
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
                f32_reductions=f32_reductions,
            )
            linkage_support.record(linkage, row, host_decision, source_rows, linalg)
            arange_support.record(arange, capture, row, host_decision, parsed, source_rows)
            bucketize_support.record(bucketize, row, host_decision, source_rows)
    if unresolved:
        raise SourceAdmissionError(unresolved)
    integer_support.verify_source(integer_reductions, source)
    f32_sum_support.verify_source(f32_reductions, source)
    maximum_support.verify_source(maximum, source)
    index_support.verify_source(index_host, capture)
    group_roster = source_group_roster(groups, eligible, host_placement=CG.HOST, module=module)
    result = {
        "source_sha256": source_sha,
        "normalized_source_sha256": normalized_sha,
        "capture_receipt_sha256": checked["receipt_sha256"],
        "n_source_operations": inventory["n_operations"],
        "n_linalg_regions": len(descriptors),
        "n_groups": len(groups),
        "n_exact_float_epilogue_refusals": exact_float_refusals,
        "n_support_lowering_operations": support_lowering,
        "transpose_data_support": transpose_support,
        "generic_copy_data_support": generic_copy_support,
        "pool_value_support": pool_support,
        "linalg_host_support": linalg,
        linkage_support.FIELD: linkage,
        "literal_arange_host_support": arange,
        integer_support.FIELD: integer_reductions,
        f32_sum_support.FIELD: f32_reductions,
        maximum_support.FIELD: maximum,
        index_support.FIELD: index_host,
        ordered_scan_support.FIELD: ordered_scan,
        bucketize_support.FIELD: bucketize,
        **group_roster,
        "host_groups": [group.index for group in groups if group.placement == CG.HOST],
        "host_justification": "source group reasons and exact reviewed host/software admissions",
        "accelerator_source_demand": demand,
    }
    control_support.attach_source_record(result, bounded_control)
    if direct_return_support is not None:
        result["direct_return_support"] = direct_return_support
    return result
