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
from pathlib import Path
from typing import Any

import yaml

from merlin.compile.model_execution_inputs import file_sha256, strict_tree_sha256

SCHEMA = "merlin.phase1.private_full_models.v1"
RESULT_SCHEMA = "merlin.phase1.private_full_model_build_gate.v1"
BUILD_BOARD_SCOPE = "static_memory_layout_and_host_ISA_only; no board execution"
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
            software_sha256=file_sha256(_file(spec, row["software_spec"], row.get("software_spec_sha256"))),
            capability_sha256=file_sha256(
                _file(spec, row["capability_contract"], row.get("capability_contract_sha256"))
            ),
            facts_sha256=file_sha256(_file(spec, row["rtl_facts"], row.get("rtl_facts_sha256"))),
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
    return [{"path": path, "kind": kind} for path, kind in sorted(denied.items())]


def _file(spec: Path, value: Any, digest: Any) -> Path:
    if not isinstance(value, str) or not value or not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("private model input needs a path and SHA256")
    path = Path(value)
    path = path if path.is_absolute() else spec.parent / path
    if path.is_symlink() or not path.is_file() or file_sha256(path) != digest:
        raise ValueError(f"private model input is absent, indirect, or changed: {path}")
    return path.resolve(strict=True)


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


def _recipe_derivation(
    spec: Path,
    row: Mapping[str, Any],
    plan: Mapping[str, Any],
    *,
    target: str,
    software_sha256: str,
    capability_sha256: str,
    facts_sha256: str,
) -> dict[str, str]:
    """Bind the selected capture recipe to canonical current-spec derivation.

    The Phase 0 export is a diagnostic transformation input, not an admission
    or a model-validation result.  Its independently checked source snapshots
    must match the very contracts used by this build, and its derived recipe
    must match the sealed capture preselection's actual recipe bytes.
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
    return {
        "evidence_manifest_sha256": file_sha256(manifest),
        "recipe_sha256": candidates[0]["sha256"],
        "recipe_semantic_sha256": candidates[0]["recipe_sha256"],
        "scope": "current-spec diagnostic recipe derivation only; no model or hardware admission",
    }


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


def _source_obligations(capture: Path, target: str, software: Mapping, capability: Mapping, host: Mapping) -> dict:
    """Recompute eligibility over the actual source, independently of the candidate route."""
    from merlin.common import mlir_query as mq
    from merlin.frontends.capture_normalization import normalize_capture_mlir
    from merlin.targetgen import application_inventory as AI
    from merlin.targetgen import model_coverage as MC
    from merlin.targetgen import operation_accounting as OA
    from merlin.targetgen.eligibility import capability_map_from_contract, is_eligible
    from merlin.xdsl_dialects.lowering import compute_groups as CG

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
    if not descriptors or len(descriptors) != len(operations):
        raise ValueError("source linalg inventory is empty or incomplete")
    cap_map = capability_map_from_contract(dict(capability))
    eligible = []
    for index, (op, descriptor) in enumerate(zip(operations, descriptors, strict=True)):
        verdict = is_eligible(descriptor, cap_map)
        if verdict.undetermined:
            raise ValueError(f"source operation {index} has unknown hardware eligibility")
        if not verdict.eligible:
            continue
        group = owners.get(id(op))
        if group is None or group.placement == CG.HOST:
            raise ValueError(f"eligible source operation {index} has no accelerator group")
        eligible.append(group.index)

    inventory = AI._application_operation_inventory(  # noqa: PLC2701 -- trusted exact inventory primitive
        source,
        target,
        cap_map,
        capability_contract=dict(capability),
        software_spec=dict(software),
        host_capabilities=dict(host),
    )
    unresolved = []
    support_lowering = 0
    for row in inventory["signatures"]:
        if row["disposition"] in {"structural", "component"}:
            continue
        if row["disposition"] == "support_required":
            if _noncompute_support(row):
                # These are control/data-support instructions, not independent
                # compute decisions.  The real whole-program build below must
                # lower and link them; an unknown support op is not waived.
                support_lowering += 1
                continue
            # A linalg movement is computation-carrying and its accelerator
            # eligibility/placement was checked in the source roster above.
        admission = OA.admit_operation_row(
            row,
            software_spec=dict(software),
            capability_contract=dict(capability),
            capability_map=cap_map,
            host_capabilities=dict(host),
        )
        accelerator = admission["accelerator_admission"]
        host_decision = admission["host_admission"]
        if accelerator["status"] == "admitted":
            # The exact source linalg inventory above must carry this demand.  A
            # standalone eligible op absent from grouping cannot disappear into host.
            if not row["mlir_operation"].startswith("linalg."):
                unresolved.append({"ordinals": row["ordinals"], "reason": "eligible operation is not outlined"})
        elif host_decision["status"] != "admitted" or host_decision.get("reviewed") is not True:
            unresolved.append({"ordinals": row["ordinals"], "reason": "host operation lacks exact reviewed admission"})
    if unresolved:
        raise ValueError(f"source has {len(unresolved)} unaccounted or unjustified operation signature(s)")
    group_metrics = {}
    for group in groups:
        if group.index not in eligible:
            continue
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
    return {
        "source_sha256": file_sha256(source),
        "capture_receipt_sha256": checked["receipt_sha256"],
        "n_source_operations": inventory["n_operations"],
        "n_groups": len(groups),
        "n_support_lowering_operations": support_lowering,
        "eligible_groups": sorted(set(eligible)),
        "eligible_group_metrics": group_metrics,
        "host_groups": [group.index for group in groups if group.placement == CG.HOST],
        "host_justification": "source group reasons and exact reviewed host/software admissions",
    }


def _linked_symbols(elf: Path, symbols: set[str]) -> None:
    from merlin.llvmlower import toolchain

    command = [str(toolchain.nm()), "--defined-only", "--extern-only", str(elf)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=120, check=True)
    defined = {line.split()[-1] for line in result.stdout.splitlines() if line.split()}
    missing = sorted(symbols - defined)
    if missing:
        raise ValueError(f"linked ELF omits {len(missing)} routed device symbol(s): {missing[:8]}")


def _require_static_build_inputs(
    receipt: Mapping[str, Any], *, target: str, board: str, catalog: Path, dts: Path
) -> None:
    inputs = receipt.get("inputs") or {}
    if (
        not isinstance(inputs, Mapping)
        or inputs.get("target") != target
        or inputs.get("board") != board
        or inputs.get("board_catalog") != str(catalog)
        or inputs.get("dts") != str(dts)
        or inputs.get("board_catalog_sha256") != file_sha256(catalog)
        or inputs.get("dts_sha256") != file_sha256(dts)
        or inputs.get("run") != "none"
    ):
        raise ValueError("linked image used another board or implied model execution")


def _audit_built_device_host_compute(
    build: Path, assigned: Sequence[Mapping[str, Any]], source: Mapping[str, Any], target: str
) -> list[dict[str, Any]]:
    """Audit the package's actual per-group build inputs, not a fresh diagnostic emission.

    The trusted whole-model builder saves each package stdout as ``.device.mlir``
    before translating and compiling it to the neighboring linked object.  We
    read those exact files and the independently sourced group output extents.
    All defined functions are judged as accelerator work; an extra helper
    cannot self-label as a permitted host group.
    """
    from xdsl.context import Context
    from xdsl.dialects import builtin, func, llvm
    from xdsl.parser import Parser

    from merlin.llvmlower.device_shim import kernel_abi_for
    from merlin.verify import host_compute_audit as HA

    abi = kernel_abi_for(target)
    if abi is None:
        raise ValueError("selected device has no readable kernel ABI for host-compute audit")
    device_dir = build / "device"
    if device_dir.is_symlink() or not device_dir.is_dir():
        raise ValueError("linked build has no ordinary device-artifact directory")
    metrics = source["eligible_group_metrics"]
    records = []
    for entry in assigned:
        symbol, index = entry["symbol"], entry["group"]
        if not isinstance(symbol, str) or not symbol or Path(symbol).name != symbol or symbol in {".", ".."}:
            raise ValueError("routed group has no safe built-artifact stem")
        metric = metrics.get(index)
        if not isinstance(metric, Mapping):
            raise ValueError(f"routed group {index} has no source-derived output extent")
        artifact, llvm_ir, obj = (
            device_dir / f"{symbol}.device.mlir",
            device_dir / f"{symbol}.ll",
            device_dir / f"{symbol}.o",
        )
        for path in (artifact, llvm_ir, obj):
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"linked group {index} lacks its exact built artifact: {path}")
        context = Context(allow_unregistered=True)
        for dialect in (builtin.Builtin, llvm.LLVM, func.Func):
            context.load_dialect(dialect)
        module = Parser(context, artifact.read_text(encoding="utf-8")).parse_module()
        functions = HA._functions(module)  # noqa: PLC2701 -- shared independent host-code reader
        if abi.symbol not in functions:
            raise ValueError(f"linked group {index} has no body for its target ABI kernel")
        sites = [
            HA.GroupSite(
                group=index,
                placement=target,
                symbol=name,
                elements=metric["elements"],
                element_bytes=metric["element_bytes"],
            )
            for name in functions
        ]
        report = HA.audit(module, sites)
        HA.require_clean(report)
        if report.get("proven_clean") is not True or report.get("accelerator_groups_clean") != len(sites):
            raise ValueError(f"linked group {index} host-compute audit is unknown, not clean")
        from merlin.llvmlower import toolchain

        undefined = subprocess.run(
            [str(toolchain.nm()), "--undefined-only", "--extern-only", str(obj)],
            capture_output=True,
            text=True,
            timeout=120,
            check=True,
        )
        if undefined.stdout.strip():
            raise ValueError(f"linked group {index} calls an unaudited external host helper")
        records.append(
            {
                "group": index,
                "symbol": symbol,
                "defined_functions_audited": len(sites),
                "artifact_sha256": file_sha256(artifact),
                "llvm_sha256": file_sha256(llvm_ir),
                "object_sha256": file_sha256(obj),
                "verdict": "clean_static_host_compute_audit",
                "audit": {
                    "scope": (
                        "all defined LLVM functions of the exact built device artifact; command-scale static budget"
                    ),
                    "budget": report["budget"],
                    "groups": [
                        {
                            key: group.get(key)
                            for key in (
                                "symbol",
                                "verdict",
                                "elements",
                                "host_arithmetic",
                                "host_value_arithmetic",
                                "host_payload_bytes",
                                "arithmetic_per_element",
                                "value_arithmetic_per_element",
                                "payload_ratio",
                            )
                        }
                        for group in report["groups"]
                    ],
                },
            }
        )
    return records


def _captured_programs(capture: Path, expected: Sequence[str]) -> list[tuple[str, Path]]:
    """Use every program in the attested root session, never an operator-picked slice."""
    # A complete single-network capture may also carry a version-1 execution
    # session contract (for example an image stream). Only a root session
    # receipt identifies the version-2 multi-program capture protocol.
    if tuple(expected) == ("model",) and not (capture / "session-receipt.json").exists():
        if (capture / "stages").exists():
            raise ValueError("single-network declaration received an unbound stage directory")
        if any(
            path.is_symlink() or not path.is_file()
            for path in (capture / "model.mlir", capture / "capture_receipt.json")
        ):
            raise ValueError("single-network declaration has no ordinary source model and receipt")
        contract_path = capture / "session_contract.yaml"
        if contract_path.exists():
            if contract_path.is_symlink() or not contract_path.is_file():
                raise ValueError("single-network execution contract is indirect or malformed")
            contract = yaml.safe_load(contract_path.read_bytes())
            if not isinstance(contract, Mapping) or contract.get("version") != 1:
                raise ValueError("single-network execution contract is not version 1")
        return [("model", capture)]
    contract_path = capture / "session_contract.yaml"
    receipt_path = capture / "session-receipt.json"
    if any(path.is_symlink() or not path.is_file() for path in (contract_path, receipt_path)):
        raise ValueError("complete multi-program capture has no ordinary root session contract/receipt")
    contract = yaml.safe_load(contract_path.read_bytes())
    receipt = json.loads(receipt_path.read_bytes())
    if not isinstance(contract, Mapping) or not isinstance(receipt, Mapping):
        raise ValueError("complete session contract/receipt is malformed")
    names = list(expected)
    programs = contract.get("programs")
    observed = receipt.get("programs")
    if (
        contract.get("version") != 2
        or contract.get("stages") != names
        or not isinstance(programs, list)
        or not isinstance(observed, list)
        or receipt.get("schema") != "merlin.model_session_capture.v1"
        or receipt.get("session_contract_sha256") != file_sha256(contract_path)
        or len(programs) != len(names)
        or len(observed) != len(names)
        or (len(names) > 1 and not contract.get("bindings"))
    ):
        raise ValueError("root session does not bind the declared complete program roster")
    stage_root = capture / "stages"
    if (
        stage_root.is_symlink()
        or not stage_root.is_dir()
        or sorted(p.name for p in stage_root.iterdir()) != sorted(names)
    ):
        raise ValueError("capture contains missing or extra session stage directories")
    result = []
    for name, entry, attested in zip(names, programs, observed, strict=True):
        stage = stage_root / name
        if (
            not isinstance(entry, Mapping)
            or entry.get("name") != name
            or entry.get("bundle") != f"stages/{name}"
            or not isinstance(attested, Mapping)
            or attested.get("name") != name
            or attested.get("ok") is not True
            or attested.get("opaque") != 0
            or attested.get("receipt_sha256") != file_sha256(stage / "capture_receipt.json")
            or stage.is_symlink()
            or not stage.is_dir()
        ):
            raise ValueError(f"session stage {name} is unverified, opaque, or not contract-bound")
        result.append((name, stage))
    return result


def _captured_input_provenance(
    programs: Sequence[tuple[str, Path]], expected: Mapping[str, bool | None]
) -> dict[str, dict[str, Any]]:
    """Report the attested loader's input claim, without its private file path."""
    result = {}
    for name, stage in programs:
        meta_path = stage / "meta.json"
        if meta_path.is_symlink() or not meta_path.is_file():
            raise ValueError(f"{name} has no ordinary capture input-provenance record")
        meta = json.loads(meta_path.read_bytes())
        if not isinstance(meta, Mapping):
            raise ValueError(f"{name} has malformed capture input provenance")
        declared = meta.get("loader_provenance")
        if meta.get("loader_provenance_status") != "declared" or not isinstance(declared, Mapping):
            raise ValueError(f"{name} has no declared loader input provenance")
        paper_ready = meta.get("loader_paper_ready")
        synthetic_fields = [declared[key] for key in ("synthetic_inputs", "synthetic_tokens") if key in declared]
        if len(synthetic_fields) > 1 and synthetic_fields[0] is not synthetic_fields[1]:
            raise ValueError(f"{name} has conflicting synthetic-input declarations")
        synthetic = synthetic_fields[0] if synthetic_fields else None
        if paper_ready is not None and type(paper_ready) is not bool:
            raise ValueError(f"{name} has a malformed paper-readiness declaration")
        if synthetic is not None and type(synthetic) is not bool:
            raise ValueError(f"{name} has a malformed synthetic-input declaration")
        observed = {"paper_ready": paper_ready, "synthetic_inputs": synthetic}
        if any(observed.get(key) is not value for key, value in expected.items()):
            raise ValueError(f"{name} input provenance differs from the selected complete-model scope")
        input_source = declared.get("input_source", declared.get("token_source"))
        input_sha256 = declared.get("input_sha256", declared.get("token_sha256"))
        result[name] = {
            **observed,
            "input_source": input_source if isinstance(input_source, str) else None,
            "input_sha256": input_sha256 if isinstance(input_sha256, str) else None,
            "meta_sha256": file_sha256(meta_path),
            "scope": "input provenance only; no paper accuracy or full-model numerical result",
        }
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
) -> dict[str, Any]:
    """Build every frozen full model with the current candidate, without running a model simulator."""
    from merlin.compile.baremetal_model import compile_saved_model
    from merlin.compile.model_execution_inputs import selected_firrtl
    from merlin.compile.route_before_build import plan_before_build
    from merlin.llvmlower.device_offload import BY_GROUP
    from merlin.targetgen import target_registry

    spec = Path(private_spec)
    if spec.is_symlink() or not spec.is_file():
        raise ValueError("operator-private full-model specification is absent or indirect")
    document = yaml.safe_load(spec.read_bytes())
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
    package = Path(submission)
    package_digest = strict_tree_sha256(package)
    base = Path(out)
    base.mkdir(parents=True, exist_ok=True)
    results = []
    for row in rows:
        name = row["id"]
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
            capture_digest = strict_tree_sha256(capture)["sha256"]
            if row.get("capture_tree_sha256") not in (None, capture_digest):
                raise ValueError("full-model capture differs from an explicitly pinned tree")
            from merlin_experiments.phase0.capture_execution_attestation import require_verified_execution

            attestation_path = Path(str(row.get("capture_execution_attestation") or ""))
            attestation_path = attestation_path if attestation_path.is_absolute() else spec.parent / attestation_path
            if attestation_path.is_symlink() or not attestation_path.is_file():
                raise ValueError("preselected capture has no safe post-execution attestation")
            if row.get("capture_execution_attestation_sha256") not in (None, file_sha256(attestation_path)):
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
            expected_stages = tuple(required_programs[name])
            captured_kind = "session" if (capture / "session-receipt.json").exists() else "single"
            if (
                attestation.get("issuer") != "merlin.sealed_m2m_cpu.v3"
                or attested_capture.get("kind") != captured_kind
                or Path(str(attested_capture.get("capture_path") or "")).resolve() != capture
                or attested_capture.get("capture_tree_sha256") != capture_digest
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
            software_path = _file(spec, row.get("software_spec"), row.get("software_spec_sha256"))
            capability_path = _file(spec, row.get("capability_contract"), row.get("capability_contract_sha256"))
            host_path = _file(spec, row.get("host_capabilities"), row.get("host_capabilities_sha256"))
            catalog = _file(spec, row.get("board_catalog"), row.get("board_catalog_sha256"))
            dts = _file(spec, row.get("host_dts"), row.get("host_dts_sha256"))
            board_catalog_sha256 = file_sha256(catalog)
            board_dts_sha256 = file_sha256(dts)
            item["checks"]["recipe_derivation"] = _recipe_derivation(
                spec,
                row,
                selected_plan,
                target=target,
                software_sha256=file_sha256(software_path),
                capability_sha256=file_sha256(capability_path),
                facts_sha256=facts["sha256"],
            )
            software = yaml.safe_load(software_path.read_bytes())
            capability = yaml.safe_load(capability_path.read_bytes())
            host = yaml.safe_load(host_path.read_bytes())
            if any(not isinstance(value, Mapping) for value in (software, capability, host)):
                raise ValueError("selected software, capability or host declaration is malformed")
            if software.get("target") != target or capability.get("name") != target:
                raise ValueError("selected software or capability declaration names another target")
            if software.get("status") != "reviewed" or host.get("status") != "reviewed":
                raise ValueError("selected software or host declaration is not reviewed")
            with target_registry.observed_contract(target, dict(capability), source_path=capability_path):
                sources = {
                    program: _source_obligations(stage, target, software, capability, host)
                    for program, stage in programs
                }
            if not any(source["eligible_groups"] for source in sources.values()):
                raise ValueError("full model has no independently eligible accelerator computation")
            item["checks"]["source"] = sources
            compiled = []
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
                        )
                    device = routed.get("device_routing")
                    if device is None or routed.get("offload") is None:
                        raise ValueError(
                            f"{program} has no buildable accelerator route: {routed.get('device_routing_why')}"
                        )
                with target_registry.observed_contract(target, dict(capability), source_path=capability_path):
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
                    )
                expected_route = "device_requested_dispatch_unverified" if device is not None else "host_baseline"
                if (
                    receipt.get("status") != "compiled"
                    or receipt.get("execution_route") != expected_route
                    or receipt.get("inputs", {}).get("capture_tree") != stage_tree
                ):
                    raise ValueError(f"{program} did not capture/lower/codegen/link the selected source")
                _require_static_build_inputs(
                    receipt,
                    target=target,
                    board=str(row["board"]),
                    catalog=catalog,
                    dts=dts,
                )
                if (
                    device is not None
                    and (receipt.get("inputs", {}).get("device") or {}).get("package_tree") != package_digest
                ):
                    raise ValueError(f"{program} build used a different candidate compiler tree")
                output = receipt.get("output") or {}
                elf = Path(str(output.get("elf") or ""))
                if not elf.is_file() or file_sha256(elf) != output.get("elf_sha256"):
                    raise ValueError(f"{program} linked ELF is absent or changed")
                sidecar_sha = None
                linked = 0
                host_compute_audit = []
                if device is not None:
                    sidecar_record = output.get("device_sidecar") or {}
                    sidecar = Path(str(sidecar_record.get("path") or ""))
                    if not sidecar.is_file() or file_sha256(sidecar) != sidecar_record.get("sha256"):
                        raise ValueError(f"{program} candidate device sidecar is absent or changed")
                    emitted = json.loads(sidecar.read_text(encoding="utf-8"))
                    if emitted.get("granularity") != BY_GROUP or emitted.get("device") != target:
                        raise ValueError(f"{program} emitted another device or a partial route")
                    assigned = emitted.get("routed") or []
                    indices = [entry.get("group") for entry in assigned]
                    if (
                        not assigned
                        or any(type(index) is not int for index in indices)
                        or len(indices) != len(set(indices))
                        or sorted(indices) != source["eligible_groups"]
                        or emitted.get("skipped")
                    ):
                        raise ValueError(f"{program} linked roster differs from eligible source groups")
                    symbols = {entry.get("symbol") for entry in assigned}
                    if None in symbols or len(symbols) != len(assigned):
                        raise ValueError(f"{program} candidate dispatch symbols are missing or duplicated")
                    _linked_symbols(elf, symbols)
                    host_compute_audit = _audit_built_device_host_compute(
                        Path(str(output["elf"])).parent, assigned, source, target
                    )
                    sidecar_sha, linked = sidecar_record["sha256"], len(assigned)
                compiled.append(
                    {
                        "program": program,
                        "status": "capture_lower_codegen_link_verified",
                        "elf_sha256": output["elf_sha256"],
                        "sidecar_sha256": sidecar_sha,
                        "linked_device_groups": linked,
                        "static_host_compute_audit": host_compute_audit,
                        "candidate_tree_sha256": package_digest["sha256"],
                    }
                )
            if strict_tree_sha256(capture)["sha256"] != capture_digest:
                raise ValueError("private model changed during candidate build")
            if selected_firrtl(facts_path, target=target, config=facts["config"]) != facts:
                raise ValueError("selected RTL facts changed during candidate build")
            if strict_tree_sha256(package) != package_digest:
                raise ValueError("candidate compiler changed during model build")
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
            item["status"] = "pass"
        except Exception as exc:  # noqa: BLE001 -- one model's refusal must not hide the others
            item["reason"] = f"{type(exc).__name__}: {exc}"
    result = {
        "schema": RESULT_SCHEMA,
        "target": target,
        "private_spec_sha256": file_sha256(spec),
        "candidate_tree_sha256": package_digest["sha256"],
        "required_models": list(required_models),
        "required_programs": {name: list(required_programs[name]) for name in required_models},
        "models": results,
        "passed": bool(results) and all(item["status"] == "pass" for item in results),
        "scope": "complete captured networks, static source and linked-ELF evidence; full-model execution deferred",
        "full_model_numerical_equivalence": "not_run",
        "paper_accuracy": "not_claimed",
    }
    (base / "private_full_model_gate.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def complete(
    record: Mapping[str, Any] | None,
    *,
    required_models: Sequence[str],
    required_programs: Mapping[str, Sequence[str]],
    candidate_sha256: str,
) -> bool:
    """Revalidate the claim-bearing roster and exact current submission identity."""
    if not isinstance(record, Mapping) or record.get("schema") != RESULT_SCHEMA or record.get("passed") is not True:
        return False
    if record.get("full_model_numerical_equivalence") != "not_run" or record.get("paper_accuracy") != "not_claimed":
        return False
    rows = record.get("models")
    if not isinstance(rows, list) or len(rows) != len(required_models):
        return False
    if any(
        not isinstance(row, Mapping)
        or not isinstance(row.get("checks"), Mapping)
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
                isinstance(entry, Mapping) and type(entry.get("linked_device_groups")) is int
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
