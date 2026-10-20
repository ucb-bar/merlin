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

from merlin.compile.model_execution_inputs import file_sha256, strict_tree_sha256
from merlin_experiments.phase1.feedback import private_bucketize_support as bucketize_support
from merlin_experiments.phase1.feedback import private_compilation_inputs as compilation_support
from merlin_experiments.phase1.feedback import private_control_support as control_support
from merlin_experiments.phase1.feedback import private_data_movement as data_movement
from merlin_experiments.phase1.feedback import private_f32_maximum_support as maximum_support
from merlin_experiments.phase1.feedback import private_f32_reduction_support as f32_sum_support
from merlin_experiments.phase1.feedback import private_host_source_dispatch as host_source_dispatch
from merlin_experiments.phase1.feedback import private_index_host_support as index_support
from merlin_experiments.phase1.feedback import private_integer_reduction_support as integer_support
from merlin_experiments.phase1.feedback import private_linalg_support as linalg_support
from merlin_experiments.phase1.feedback import private_linkage_support as linkage_support
from merlin_experiments.phase1.feedback import private_linked_elf_selection as linked_policy
from merlin_experiments.phase1.feedback import private_literal_arange_admission as arange_support
from merlin_experiments.phase1.feedback import private_model_math_policy as math_policy
from merlin_experiments.phase1.feedback import private_ordered_scan_support as ordered_scan_support
from merlin_experiments.phase1.feedback import private_pure_stage_support as pure_stage
from merlin_experiments.phase1.feedback import private_source_demand as source_demand
from merlin_experiments.phase1.feedback import private_source_freeze as source_freeze_api
from merlin_experiments.phase1.feedback.private_capture_roster import (
    captured_input_provenance as _captured_input_provenance,
)
from merlin_experiments.phase1.feedback.private_capture_roster import captured_programs as _captured_programs
from merlin_experiments.phase1.feedback.private_contraction_support import (
    contraction_zero_init_fill as _contraction_zero_init_fill,
)
from merlin_experiments.phase1.feedback.private_device_audit import (
    audit_built_device_host_compute as _audit_built_device_host_compute,
)
from merlin_experiments.phase1.feedback.private_device_audit import require_linked_symbols as _linked_symbols
from merlin_experiments.phase1.feedback.private_device_audit import (
    require_static_build_inputs as _require_static_build_inputs,
)
from merlin_experiments.phase1.feedback.private_group_provenance import (
    convolution_input_preparations,
    source_group_roster,
    verify_routed_source_coverage,
)
from merlin_experiments.phase1.feedback.private_group_provenance import (
    eligible_source_identity as _eligible_source_identity,  # noqa: F401 -- preserve compatibility alias
)
from merlin_experiments.phase1.feedback.private_group_provenance import (
    join_routed_source_groups as _join_routed_source_groups,  # noqa: F401 -- preserve source-join compatibility alias
)
from merlin_experiments.phase1.feedback.private_group_provenance import (
    routed_kernel_symbols as _routed_kernel_symbols,
)
from merlin_experiments.phase1.feedback.private_source_support_join import (
    linked_source_support_complete as _linked_source_support_complete,
)

from . import private_program_checks as program_checks
from . import private_static_products as static_products
from .private_completed_build import compile_saved_model
from .private_model_inputs import _capture_tree_bindings as _verify_capture_bindings
from .private_model_inputs import _expected_input_provenance as _expected_input_provenance
from .private_model_inputs import _recipe_derivation as _derive_recipe_binding
from .private_model_inputs import _require_selected_input_bindings as _require_selected_input_bindings
from .private_model_inputs import _selected_admission_views as _selected_admission_views
from .private_model_inputs import _tree as _verify_input_tree
from .private_source_preflight import _notify as _source_progress
from .private_source_preflight import source_result as _source_result

SCHEMA = "merlin.phase1.private_full_models.v1"
RESULT_SCHEMA = "merlin.phase1.private_full_model_build_gate.v15"
BUILD_BOARD_SCOPE = "static_memory_layout_and_host_ISA_only; no board execution"
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


def _exact_source_groups(module, target: str, capture: Path, form) -> tuple[list, int]:
    """Source groups whose absorbed float epilogue is PROVED to round as the capture does.

    A group that takes a converted accumulator through a stored float bias and a quantize is an
    integer readout only when the transition proof over its stored weights says so
    (:mod:`merlin.xdsl_dialects.lowering.group_exact_float`). An unproved group is formed again with
    its float chain left on the host, so the gate never demands an accelerator epilogue that would
    compute something other than the reference. Returns the groups and how many were refused.
    """
    from merlin.llvmlower.source_compute_groups import form_exact_groups

    return form_exact_groups(module, target, capture, form)


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
    from merlin.common import mlir_query as mq
    from merlin.frontends.capture_normalization import normalize_capture_mlir
    from merlin.llvmlower.source_compute_groups import form_source_groups
    from merlin.targetgen import application_inventory as AI
    from merlin.targetgen import model_coverage as MC
    from merlin.targetgen import operation_accounting as OA
    from merlin.targetgen.eligibility import capability_map_from_contract, is_eligible
    from merlin.xdsl_dialects.lowering import compute_groups as CG
    from merlin_experiments.phase1.feedback.private_pool_support import prove_pool_source

    source = capture / "model.mlir"
    _source_progress(progress, "capture_receipt")
    checked = AI.verify_capture_receipt(source)
    if checked.get("status") != "verified_materialized":
        raise ValueError(f"full-model capture receipt is not verified: {checked.get('errors')}")
    _source_progress(progress, "normalization")
    text, _ = normalize_capture_mlir(source.read_text(encoding="utf-8"))
    _source_progress(progress, "parse")
    module = (prepared_source := AI.PreparedApplicationSource(source, text)).module
    _source_progress(progress, "group_formation")
    groups, exact_float_refusals = _exact_source_groups(module, target, capture, form_source_groups)
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
    inventory = AI._application_operation_inventory(  # noqa: PLC2701 -- trusted exact inventory primitive
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
    optional_linked = 0
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
        joined = verify_routed_source_coverage(emitted, source)
        symbols = _routed_kernel_symbols(emitted)
        _linked_symbols(elf, symbols)
        source["conv_input_preparation"] = convolution_input_preparations(emitted, source)
        host_compute_audit = _audit_built_device_host_compute(
            elf.parent, assigned, source, target, host_artifacts=output.get("host_artifacts")
        )
        sidecar_sha = sidecar_record["sha256"]
        linked = len(set(joined.values()) & set(source["eligible_groups"]))
        optional_linked = len(set(joined.values()) & set(source.get("optional_groups", [])))
    linked_build = {
        "capture_tree_sha256": stage_tree["sha256"],
        "elf_sha256": output["elf_sha256"],
        "candidate_tree_sha256": package_digest["sha256"],
    }
    source_demand.link(source, linked_build, linked)
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
    f32_sum_support.link(source, index_lowering, linked_build)
    maximum_support.link(source, index_lowering, linked_build, capture_path=capture_path)
    index_support.link(source, index_lowering, linked_build, capture_path=capture_path)
    ordered_scan_support.link(source, index_lowering, linked_build)
    bucketize_support.link(source, index_lowering, linked_build, capture_path=capture_path)
    result = {
        "program": program,
        "board": board,
        "status": "capture_lower_codegen_link_verified",
        "capture_tree_sha256": stage_tree["sha256"],
        "source_sha256": source["source_sha256"],
        "elf_sha256": output["elf_sha256"],
        "sidecar_sha256": sidecar_sha,
        "linked_device_groups": linked,
        "linked_optional_device_groups": optional_linked,
        "linked_total_device_groups": linked + optional_linked,
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
    selected_model: str | None = None,
    diagnostic_model: str | None = None,
    prebuilt_receipts: Mapping[str, str | Path] | None = None,
    source_freeze: Mapping[str, Any] | None = None,
    source_freeze_root: str | Path | None = None,
    build_options: Mapping[str, Mapping[str, Mapping[str, Any]]] | None = None,
    linked_elf_admission=None,
    native_result_format_selection=None,
    source_only: bool = False,
    progress=None,
    retained_products=None,
) -> dict[str, Any]:
    """Build the frozen roster or all programs of one canonical/diagnostic partition.
    Build options select per-program full readback/profiling. Partitions retain
    the full roster/source freeze and cannot pass the full gate; source-only takes no compiler.
    Typed semantic refusals retain failed rows; identity/route/unexpected errors abort.
    """
    from merlin.compile.model_execution_inputs import selected_firrtl
    from merlin.compile.route_before_build import plan_before_build
    from merlin.llvmlower.device_offload import BY_GROUP
    from merlin.targetgen import target_registry

    instruction_selection = linked_policy.freeze(linked_elf_admission, target=target)
    if native_result_format_selection is not None:
        from .private_result_format_selection import FrozenSelection

        if (
            type(native_result_format_selection) is not FrozenSelection
            or native_result_format_selection.target != target
        ):
            raise ValueError("native result format selection has no matching original run owner")
        native_result_format_selection.verify()
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
    static_products.require_selection(
        retained_products, selected_model=selected_model, diagnostic=diagnostic, source_only=source_only)
    if selected_model is not None and (
        not isinstance(selected_model, str)
        or selected_model not in required_models
        or diagnostic
        or prebuilt_receipts is not None
    ):
        raise ValueError("canonical model partition requires one declared model and no diagnostic or prebuilt inputs")
    if not diagnostic and frozen_authored is None:
        raise ValueError("canonical private full-model builds require a run-owned authored-source freeze")
    if diagnostic:
        if diagnostic_model not in required_models or not isinstance(prebuilt_receipts, Mapping):
            raise ValueError("prebuilt diagnostic must select one required model and its receipts")
        if set(prebuilt_receipts) != set(required_programs[diagnostic_model]):
            raise ValueError("prebuilt diagnostic receipts differ from the complete selected model")
    elif prebuilt_receipts is not None:
        raise ValueError("prebuilt receipts cannot enter the canonical full-model build")
    if source_only and (diagnostic or submission is not None):
        raise ValueError("source preflight takes no candidate compiler or prebuilt diagnostic")
    package = None if source_only else Path(submission)
    package_digest = None if source_only else strict_tree_sha256(package)
    base = Path(out)
    base.mkdir(mode=0o700, parents=True, exist_ok=True)
    results = []
    native_result_ceilings = {}
    for row in rows:
        name = row["id"]
        if (diagnostic and name != diagnostic_model) or (selected_model is not None and name != selected_model):
            continue
        item: dict[str, Any] = {"model": name, "status": "fail", "checks": {}}
        results.append(item)
        _source_progress(progress, "model_inputs", model=name)
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
            if retained_products is not None:
                retained_products.require(name, expected_stages, package_digest["sha256"], spec_digest)
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
            program_options = {
                program: dict(((build_options or {}).get(name) or {}).get(program) or {}) for program, _ in programs
            }
            selected_math = math_policy.bind(document, spec, row, target, program_options, item["checks"])
            from merlin.llvmlower.device_ir_scalability import DeviceIRBudget

            program_ir_budgets = {}
            for program, options in program_options.items():
                selected_budget = options.pop("device_ir_budget", None)
                if selected_budget is not None:
                    if type(selected_budget) is not dict:
                        raise ValueError("device_ir_budget must be explicit original build policy data")
                    program_ir_budgets[program] = DeviceIRBudget(**selected_budget)
            program_boards = {program: opts.pop("board", row["board"]) for program, opts in program_options.items()}
            if any(not isinstance(board, str) or not board for board in program_boards.values()):
                raise ValueError("every program requires an explicit valid selected build board")
            selected_index_observations = {
                program: control_support.selected_build_observation(host_package, catalog, board, target)
                for program, board in program_boards.items()
            }
            from contextlib import nullcontext

            from merlin.targetgen.eligibility import original_result_format_scope

            original_contract_bytes, original_facts_bytes = capability_path.read_bytes(), facts_path.read_bytes()
            native_result_ceiling = None
            if native_result_format_selection is not None:
                binding = (sha256(original_contract_bytes).hexdigest(), sha256(original_facts_bytes).hexdigest())
                if binding not in native_result_ceilings:
                    native_result_ceilings[binding] = native_result_format_selection.issue(
                        original_contract_bytes=original_contract_bytes,
                        original_facts_bytes=original_facts_bytes,
                    )
                native_result_ceiling = native_result_ceilings[binding]
            native_scope = (
                original_result_format_scope(
                    native_result_ceiling,
                    target=target,
                    original_contract_bytes=original_contract_bytes,
                    original_facts_bytes=original_facts_bytes,
                )
                if native_result_ceiling is not None
                else nullcontext()
            )
            with (
                target_registry.observed_contract(target, dict(capability), source_path=capability_path),
                native_scope,
            ):
                sources, unresolved_demands = {}, {}
                for program, stage in programs:
                    try:
                        _source_progress(progress, "program_source", model=name, program=program)

                        def observe(event):
                            _source_progress(progress, model=name, program=program, **event)

                        sources[program] = _source_obligations(
                            stage,
                            target,
                            software,
                            capability,
                            host,
                            selected_index_observations[program],
                            facts_sha256=facts["sha256"],
                            numeric_policy_sha256=recipe_proof["recipe_semantic_sha256"],
                            **({"progress": observe} if progress is not None else {}),
                        )
                        if native_result_ceiling is not None:
                            sources[program]["accelerator_source_demand"]["native_result_format_ceiling"] = (
                                native_result_ceiling.record()
                            )
                    except source_demand.SourceDemandIncomplete as error:
                        unresolved_demands[program] = error.receipt
                if unresolved_demands:
                    item["checks"]["source_demand_refusals"] = unresolved_demands
                    raise ValueError(
                        f"original accelerator source demand is incomplete for {len(unresolved_demands)} program(s)"
                    )
            if not any(source["eligible_groups"] for source in sources.values()):
                raise ValueError("full model has no independently eligible accelerator computation")
            item["checks"]["source"] = sources
            math_policy.require_sources(selected_math, sources)
            if source_only:
                item["status"] = "source_inputs_verified"
                _source_progress(progress, "model_complete", model=name, status=item["status"])
                continue
            compiled = []
            diagnostic_receipts = {}

            def revalidate_program_inputs():
                current_capture = _capture_tree_bindings(
                    capture, row.get("capture_tree_sha256"), attested_capture.get("capture_tree_sha256")
                )
                if current_capture != capture_identity:
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
                    (spec, spec_digest),
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
                current_recipe = _recipe_derivation(
                    spec,
                    row,
                    selected_plan,
                    target=target,
                    software_sha256=file_sha256(software_path),
                    capability_sha256=file_sha256(capability_path),
                    facts_sha256=facts["sha256"],
                    host_capabilities_sha256=file_sha256(host_path),
                )[0]
                if current_recipe != recipe_proof:
                    raise ValueError("selected recipe derivation changed during linked-build verification")
                linked_policy.unchanged(instruction_selection)
                if native_result_ceiling is not None:
                    native_result_ceiling.revalidate_sources()
                math_policy.recheck(selected_math)

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
                            require_all_accelerator_groups=True,
                        )
                    device = routed.get("device_routing")
                    if device is None or routed.get("offload") is None:
                        raise ValueError(
                            f"{program} has no buildable accelerator route: {routed.get('device_routing_why')}"
                        )
                    if program in program_ir_budgets:
                        from dataclasses import replace

                        device = replace(device, device_ir_budget=program_ir_budgets[program])
                source_demand.request(source, device)
                linked_policy.require_route(device, instruction_selection)
                if diagnostic or retained_products is not None:
                    receipt, diagnostic_receipts[program] = static_products.acquire(
                        retained_products, program,
                        diagnostic_path=Path(prebuilt_receipts[program]) if diagnostic else None,
                        directory=base / name / program / "linked-policy",
                        capture=stage, host_package=host_package, candidate=package, package_digest=package_digest,
                        arena_mb=int(row["arena_mb"]), target=target, device=device, facts=facts,
                        options=program_options[program], math_policy=selected_math["policy"],
                    )
                else:
                    _source_progress(progress, "program_build", model=name, program=program)
                    with target_registry.observed_contract(target, dict(capability), source_path=capability_path):
                        receipt = compile_saved_model(
                            capture=stage,
                            package=host_package,
                            board_catalog=catalog,
                            board=program_boards[program],
                            dts=dts,
                            output=base / name / program,
                            target=target,
                            run="none",
                            arena_mb=int(row["arena_mb"]),
                            device=device,
                            respect_captured_quantization_scope=True,
                            completed_build_reuse=document.get("completed_build_reuse"),
                            build_context={"capability": dict(capability), "facts": facts},
                            **linkage_support.build_options(source, selected_math["policy"]),
                            **program_options[program],
                        )
                linked_policy.require_route(device, instruction_selection)
                compiled.append(
                    program_checks.verify(
                        _verify_compiled_program,
                        receipt,
                        revalidate=revalidate_program_inputs,
                        program=program,
                        capture_path=stage,
                        source=source,
                        stage_tree=stage_tree,
                        package_digest=package_digest,
                        target=target,
                        board=program_boards[program],
                        catalog=catalog,
                        dts=dts,
                        device_selected=device is not None,
                        host_package=host_package,
                        host_package_tree_sha256=row["host_package_tree_sha256"],
                        require_compilation_recipe=not diagnostic,
                    )
                )
                check_stage = program_checks.progress_stage(compiled[-1])
                _source_progress(progress, check_stage, model=name, program=program)
            revalidate_program_inputs()
            item["checks"]["build"] = {
                "programs": compiled,
                "linked_device_groups": sum(entry["linked_device_groups"] for entry in compiled),
                "candidate_tree_sha256": package_digest["sha256"],
                "static_build_board": (
                    next(iter(program_boards.values()))
                    if len(set(program_boards.values())) == 1
                    else "per_program_selection"
                ),
                "static_build_program_boards": program_boards,
                "static_build_board_catalog_source": str(catalog),
                "static_build_board_catalog_sha256": board_catalog_sha256,
                "static_build_board_dts_source": str(dts),
                "static_build_board_dts_sha256": board_dts_sha256,
                "static_build_board_scope": BUILD_BOARD_SCOPE,
                "accelerator_rtl_facts_sha256": facts["sha256"],
                "accelerator_rtl_config": facts["config"],
                "status": program_checks.status(compiled),
            }
            if diagnostic:
                item["checks"]["prebuilt_receipts"] = diagnostic_receipts
            program_checks.require_complete(compiled)
            item["status"] = "diagnostic_static_checks_passed" if diagnostic else "pass"
        except Exception as exc:  # noqa: BLE001 -- one model's refusal must not hide the others
            item["reason"] = _public_failure_reason(exc)
            _source_progress(progress, "model_complete", model=name, status="fail")
            if source_only and isinstance(exc, SourceAdmissionError):
                item["operator_unresolved"] = json.loads(json.dumps(exc._private_unresolved, default=str))
    result = {
        "schema": RESULT_SCHEMA,
        "target": target,
        "private_spec_sha256": spec_digest,
        "candidate_tree_sha256": package_digest["sha256"] if package_digest else None,
        "required_models": list(required_models),
        "required_programs": {name: list(required_programs[name]) for name in required_models},
        "models": results,
        "passed": (
            selected_model is None
            and not diagnostic
            and not source_only
            and bool(results)
            and all(item["status"] == "pass" for item in results)
        ),
        "scope": (
            "diagnostic selected prebuilt model only; producer and toolchain unbound; not a full-model gate"
            if diagnostic
            else "complete captured networks, static source and linked-ELF evidence; full-model execution deferred"
        ),
        "full_model_numerical_equivalence": "not_run",
        "paper_accuracy": "not_claimed",
    }
    if native_result_format_selection is not None:
        native_result_format_selection.verify()
        for ceiling in native_result_ceilings.values():
            ceiling.revalidate_sources()
        result["native_result_format_selection"] = {
            "selection_sha256": native_result_format_selection.sha256,
            "scope": "original source-bound negative native output exclusions only; no positive support",
            "proofs": [ceiling.record() for ceiling in native_result_ceilings.values()],
        }
    if file_sha256(spec) != spec_digest:
        raise ValueError("operator-private full-model specification changed during verification")
    result.update(source_freeze_api.claim_binding(spec.absolute(), source_freeze, source_freeze_root, target=target))
    if diagnostic:
        result["mode"] = "prebuilt_diagnostic"
        result["diagnostic_model"] = diagnostic_model
        result["diagnostic_passed"] = len(results) == 1 and results[0]["status"] == "diagnostic_static_checks_passed"
        result["producer_binding"] = "unavailable_in_prebuilt_receipt"
    if selected_model is not None:
        result.update(
            mode="canonical_model_partition",
            selected_model=selected_model,
            partition_passed=len(results) == 1 and results[0]["status"] == "pass",
            scope="one complete declared model, canonical source and linked-ELF evidence; full roster gate incomplete",
        )
    if retained_products is not None:
        result["retained_product_revalidation"] = retained_products.record()
    if source_only:
        verified = bool(results) and all(item["status"] == "source_inputs_verified" for item in results)
        _source_result(result, verified, selected_model, SOURCE_PREFLIGHT_SCOPE)
    report = (
        base / "private_full_model_source_preflight.json"
        if source_only
        else base / "private_full_model_gate.json"
        if selected_model is None
        else base / f"private_full_model_partition-{sha256(selected_model.encode()).hexdigest()}.json"
    )
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
        or record.get("mode") in {"prebuilt_diagnostic", "canonical_model_partition"}
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
