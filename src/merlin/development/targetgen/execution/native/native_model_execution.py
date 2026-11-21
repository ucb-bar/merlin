"""Execute an emitted whole-model kernel on a frozen capsule's exact operands.

This is an evidence adapter, not a model compiler or a certification verdict.  The
caller supplies the candidate's already-emitted command buffer and LLVM-dialect
artifact; no reference compiler can substitute its own model implementation here.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from merlin.development.targetgen.execution.native.candidate_model import (
    bind_source_outputs,
    checked_source_output_values,
)
from merlin.development.targetgen.execution.native.native_model_values import (
    logical_values as _decode_logical_values,
)

# ---- native_model_execution --------------------------------------------------------------------


class NativeModelExecutionError(ValueError):
    """The frozen model, candidate artifact, and pointer ABI cannot be joined."""

    def __init__(self, detail: str, *, code: str | None = None):
        super().__init__(detail)
        self.code = code


def execute_independent_component(**arguments) -> dict:
    """Delegate only the explicit diagnostic protocol; no backend discovery."""
    from merlin.development.targetgen.execution.native.native_component_execution import execute_component

    return execute_component(**arguments)


def _digest(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return {"path": str(path), "sha256": digest.hexdigest(), "size_bytes": path.stat().st_size}


def _build_artifacts(output: Path) -> dict[str, dict[str, Any]]:
    """Pin partial compiler products too, so a failed build remains inspectable."""
    build = output / "build"
    if not build.is_dir() or build.is_symlink():
        return {}
    return {
        str(path.relative_to(build)): _digest(path)
        for path in sorted(build.rglob("*"))
        if path.is_file() and not path.is_symlink()
    }


def _read_pinned_payload(emission: Mapping[str, Any], role: str) -> bytes:
    pin = emission.get(role)
    if not isinstance(pin, Mapping) or not isinstance(pin.get("path"), str):
        raise NativeModelExecutionError(f"candidate {role} pin is absent")
    path = Path(pin["path"])
    if not path.is_absolute() or path.is_symlink() or path.resolve() != path or not path.is_file():
        raise NativeModelExecutionError(f"candidate {role} is not an ordinary absolute file")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != pin.get("sha256") or len(raw) != pin.get("size_bytes"):
        raise NativeModelExecutionError(f"candidate {role} bytes differ from emitted pin")
    return raw


def independent_frozen_source_eligibility(capsule: Mapping[str, Any], *, target: str) -> dict[str, Any]:
    """Build only the trusted eligibility denominator from frozen model source.

    No host-dispatch compile, outline, tile ledger, or candidate placement is
    consulted. The router is used to enumerate demands; its *decisions* are
    not native-execution evidence. This can run even when the legacy model
    grade fails before producing a coverage certificate.
    """
    from merlin.development.targetgen.execution.native.model_policy import census

    return census(capsule, target=target, error=NativeModelExecutionError)


def _logical_values(raw: bytes, *, tensor: str, dtype: str, shape: tuple[int, ...]) -> Any:
    return _decode_logical_values(raw, tensor=tensor, dtype=dtype, shape=shape, error_type=NativeModelExecutionError)


def _bind_inputs(
    command_buffer: Mapping[str, Any],
    capture_bundle: Path,
    *,
    target: str,
    rtl_facts: str | Path | None = None,
    board_config: str | None = None,
    input_storage: str = "inline",
):
    """Bind exact frozen entry leaves, never treating source-owned scratch as an input."""
    from merlin.development.targetgen.execution.native.native_model_input_storage import bind_entry_storage
    from merlin.development.targetgen.packaging.weight_packing import parse_arg_index, plan
    from merlin.development.targetgen.policies.form_regimes import tile_edge

    manifest_path = capture_bundle / "weights.safetensors.manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise NativeModelExecutionError(f"frozen runtime bundle lacks ordinary manifest {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise NativeModelExecutionError("frozen weight manifest is not a mapping")
    # Source entry order, not ABI spelling/order, identifies captured leaves.
    # Validate the complete source/plan join before accepting that mapping.
    from merlin.development.targetgen.oot_starterkit.plan import OUTPUT_WRITER_OWNERSHIP_FINDING as output_finding
    from merlin.development.targetgen.oot_starterkit.plan import source_operation_inventory, validate_mixed_program_plan
    from merlin.runtime.engine.command_buffer import whole_program_entry_bindings

    source_path = capture_bundle / "model.mlir"
    if source_path.is_symlink() or not source_path.is_file():
        raise NativeModelExecutionError("frozen model interface is absent or indirect")
    source_bytes = source_path.read_bytes()
    validation = validate_mixed_program_plan(source_bytes, command_buffer)
    findings = validation.get("findings", [])
    if not validation.get("ok"):
        raise NativeModelExecutionError(
            f"candidate source/entry plan is incomplete: {findings[:2]}",
            code="output_writer_source_result_ownership_unverified" if findings == [output_finding] else None,
        )
    entry = whole_program_entry_bindings(command_buffer)
    abi = command_buffer["kernel_abi"]["args"]
    read = {str(arg["tensor"]) for arg in abi if arg["access"] == "read"}
    if entry is None or set(entry) != read or len(entry) != len(read):
        raise NativeModelExecutionError("frozen source entry bindings differ from exact read-only pointer ABI")
    argument_indices = {name: index for index, name in enumerate(entry)}
    if any((named := parse_arg_index(name)) is not None and named != index for name, index in argument_indices.items()):
        raise NativeModelExecutionError("physical captured argument index differs from source argument index")
    # Prefer the explicitly selected RTL array's column count.  A named target
    # can otherwise resolve its own geometry through the existing capability
    # path, which refuses rather than guessing an unavailable fixed array.
    if rtl_facts is not None:
        facts_path = Path(rtl_facts)
        if facts_path.is_symlink() or not facts_path.is_file():
            raise NativeModelExecutionError("selected RTL facts must be an ordinary file")
        facts = json.loads(facts_path.read_text(encoding="utf-8"))
        if (facts.get("inputs") or {}).get("target") != target:
            raise NativeModelExecutionError("selected RTL facts name a different target")
        if board_config:
            from merlin.target.rtl.extraction.unit_correspondence import (
                ExecutionBindingError,
                execution_binding,
            )

            try:
                execution_binding(facts, config=board_config)
            except ExecutionBindingError as exc:
                raise NativeModelExecutionError("selected RTL facts name a different board config") from exc
        arrays = [
            row
            for row in ((facts.get("facts") or {}).get("arrays") or ())
            if isinstance(row, dict) and row.get("name") == "mesh"
        ]
        if len(arrays) != 1 or type(arrays[0].get("cols")) is not int or arrays[0]["cols"] <= 0:
            raise NativeModelExecutionError("selected RTL facts have no unique positive mesh column count")
        pitch = arrays[0]["cols"]
    else:
        pitch = tile_edge(target)
    packed = plan(command_buffer, row_pitch_elements=pitch, weight_manifest=manifest, argument_indices=argument_indices)
    inputs, source_report = bind_entry_storage(
        capture_bundle,
        packed,
        manifest,
        storage=input_storage,
        error_type=NativeModelExecutionError,
        decode=_logical_values,
    )
    # An ABI readwrite pointer is not necessarily a captured entry operand. A
    # source-produced intermediate is read by later tasks, but its initial
    # bytes belong to the submitted program, not to the frozen model capture.
    # Conversely, a true entry-bound readwrite pointer needs a carried-state
    # seed; this one-shot model runner has no such session contract. Check the
    # complete source/plan join before treating any mutable buffer as scratch.
    if set(inputs) != read or len(inputs) != len(read):
        raise NativeModelExecutionError("captured leaf binding does not cover the exact source entry ABI")

    from merlin.target.contract.harness.readback_policy import SourceOwnershipError, _source_owned_mutables

    try:
        mutable = set(_source_owned_mutables(command_buffer, source_operation_inventory(source_bytes)))
    except SourceOwnershipError as exc:
        raise NativeModelExecutionError(str(exc), code=exc.code) from exc
    if hashlib.sha256(source_path.read_bytes()).hexdigest() != validation["source_sha256"]:
        raise NativeModelExecutionError("frozen model interface changed during source binding")
    return inputs, {
        "pack": packed.to_dict(),
        "source": source_report.to_dict(),
        "weight_manifest": _digest(manifest_path),
        **({"input_storage": "streamed_exact_logical_bytes"} if input_storage == "streamed" else {}),
        "source_entry_binding": {
            "source_sha256": validation["source_sha256"],
            "entry_bindings": entry,
            "source_owned_mutables": sorted(mutable),
            "n_source_owned_mutable_intermediates": len(mutable),
            "scope": "source/plan initial-byte ownership, not compiled memory-dataflow or numerical proof",
        },
    }


def _build_service_for(
    target: str, *, source_owned_mutables: tuple[str, ...] | None = None, source_intake_protocol=None
):
    """Pin the host-selected target renderer and bypass the model importer for finished LLVM."""
    from merlin.runtime.backends import registry as backends
    from merlin.target.contract.build.services import BuildOnlyService, file_digest

    backend = backends.get_backend(target)
    root = Path(backend.__file__).resolve().parent
    providers = list(root.rglob("*.py"))
    extra = backend.build_source_paths() if callable(getattr(backend, "build_source_paths", None)) else ()
    recipe = backends._harness_build_recipe(backend, target=target)
    declared = [Path(__file__).resolve(), *providers, *map(Path, extra), *getattr(recipe, "header_dependencies", ())]
    if not declared or any(
        not p.is_absolute() or p.is_symlink() or p.resolve() != p or not p.is_file() for p in declared
    ):
        raise NativeModelExecutionError("selected target renderer has no stable source closure")
    sources = sorted(set(declared))
    if source_intake_protocol is not None:
        from merlin.development.targetgen.execution.native import (
            native_pointer_source_binding as source_intake,
        )

        source_intake.intake_limits(source_intake_protocol)
        sources = sorted(set((*sources, Path(source_intake.__file__).resolve())))

    def render(cb, *, inputs, readback_policy=None, blobs=None):
        policy = {"readback_policy": readback_policy} if readback_policy is not None else {}
        if source_intake_protocol is not None:
            import inspect

            if "host_storage_limit_bytes" not in inspect.signature(backend.render_harness).parameters:
                raise NativeModelExecutionError("selected backend lacks prospective host storage transport")
            source_intake.intake_limits(source_intake_protocol)
            policy["host_storage_limit_bytes"] = source_intake.LARGE_MAX_PAYLOAD_BYTES
        if blobs is not None:
            import inspect

            if "blobs" in inspect.signature(backend.render_harness).parameters:
                policy["blobs"] = blobs
        if not source_owned_mutables:
            return backend.render_harness(cb, target=target, inputs=inputs, **policy)
        return backend.render_harness(
            cb, target=target, inputs=inputs, source_owned_mutables=source_owned_mutables, **policy
        )

    return BuildOnlyService(
        target=target,
        recipe=recipe,
        renderer=render,
        source_pins=tuple((str(p), file_digest(p)) for p in sources),
    )


def _host_compute_report(command_buffer: Mapping[str, Any], lowered_mlir_text: str, *, entry_symbol: str):
    """Retain the historical import and monkeypatch seam for host-compute audits."""
    from merlin.development.targetgen.execution.native.dispatch_accounting import (
        _host_compute_report as accounting_report,
    )

    return accounting_report(command_buffer, lowered_mlir_text, entry_symbol=entry_symbol)


def audit_emitted_host_compute(emission: Mapping[str, Any] | None, *, entry_symbol: str) -> dict[str, Any]:
    """Reopen the exact runner-emitted bytes before admitting scoped host compute.

    A result-file claim of ``clean`` is never evidence.  Both files must still
    match the runner's digest, and the current independent analyzer must accept
    the actual command buffer and LLVM function.  Missing scope is unverified;
    known tensor arithmetic on an accelerator task is a violation.
    """
    from merlin.evaluation.host_compute_accounting import (
        HostComputeUnverified,
        HostComputeViolation,
        require_clean_host_compute,
    )

    try:
        if not isinstance(emission, Mapping):
            raise NativeModelExecutionError("candidate whole-program emission is absent")
        payloads = {role: _read_pinned_payload(emission, role) for role in ("command_buffer", "lowered_mlir")}
        cb = json.loads(payloads["command_buffer"].decode("utf-8"))
        if not isinstance(cb, dict) or (cb.get("kernel_abi") or {}).get("kind") != "whole_program":
            raise NativeModelExecutionError("candidate emission has no whole_program ABI")
        artifact = payloads["lowered_mlir"].decode("utf-8")
        report = _host_compute_report(cb, artifact, entry_symbol=entry_symbol)
        try:
            require_clean_host_compute(report)
        except HostComputeViolation as exc:
            return {"status": "violation", "detail": str(exc), "report": report.to_dict()}
        except HostComputeUnverified as exc:
            return {"status": "unverified", "detail": str(exc), "report": report.to_dict()}
        return {
            "status": "clean",
            "report": report.to_dict(),
            "candidate": {
                "command_buffer_sha256": hashlib.sha256(json.dumps(cb, sort_keys=True).encode()).hexdigest(),
                "lowered_mlir_sha256": hashlib.sha256(artifact.encode()).hexdigest(),
            },
        }
    except Exception as exc:  # noqa: BLE001 -- absent/mutated evidence can never become clean
        return {"status": "unverified", "detail": f"{type(exc).__name__}: {exc}"}


def audit_candidate_source_placement(
    emission: Mapping[str, Any] | None,
    certificate: Mapping[str, Any] | None,
    *,
    target: str,
) -> dict[str, Any]:
    """Join candidate task ownership to independent source-operation eligibility.

    The candidate chooses task kinds and source indices; it does *not* choose
    the frozen source or target eligibility.  The trusted runner's coverage
    certificate must describe those exact source bytes and every routable
    top-level operation in order.  A source index may belong to one task only.
    This is static placement accounting, never completed-dispatch evidence.
    """
    try:
        if not isinstance(emission, Mapping):
            raise NativeModelExecutionError("candidate whole-program emission is absent")
        source_path = _pinned_native_file(emission, "source_interface")
        source = _read_pinned_payload(emission, "source_interface")
        cb = json.loads(_read_pinned_payload(emission, "command_buffer").decode("utf-8"))
        if not isinstance(cb, dict) or (cb.get("kernel_abi") or {}).get("kind") != "whole_program":
            raise NativeModelExecutionError("candidate source plan has no whole_program ABI")
        if not isinstance(certificate, Mapping):
            raise NativeModelExecutionError("independent model coverage certificate is absent")
        # Recompute the denominator from the pinned frozen source and selected
        # reviewed contract. A serialized coverage row is not itself an oracle:
        # changing its eligible flags or newly captured operand formats must
        # not let a submitted compiler make its own placement denominator.
        import yaml

        declaration_path = _pinned_native_file(emission, "capsule_declaration")
        declaration = yaml.safe_load(declaration_path.read_text(encoding="utf-8"))
        if (
            not isinstance(declaration, dict)
            or not isinstance(declaration.get("interface_mlir"), str)
            or Path(declaration["interface_mlir"]).is_absolute()
            or ".." in Path(declaration["interface_mlir"]).parts
            or (declaration_path.parent / declaration["interface_mlir"]).resolve() != source_path
        ):
            raise NativeModelExecutionError("pinned interface is not the frozen capsule's declared source")
        fresh = independent_frozen_source_eligibility(
            {**declaration, "__dir__": str(declaration_path.parent)}, target=target
        )
        if dict(certificate) != fresh:
            raise NativeModelExecutionError("serialized source eligibility differs from fresh frozen-source census")
        source_sha = hashlib.sha256(source).hexdigest()
        if (
            certificate.get("target") != target
            or certificate.get("source_mlir_sha256") != source_sha
            or certificate.get("denominator_source") != "semantic_capabilities (independent eligibility oracle)"
            or certificate.get("source_inventory_scope") != "frozen_source_only_no_legacy_execution"
        ):
            raise NativeModelExecutionError("independent eligibility is not bound to this target and source")
        from merlin.target.registry import identity as target_registry

        contract_pin = certificate.get("eligibility_capability_contract")
        if not isinstance(contract_pin, Mapping):
            raise NativeModelExecutionError("independent eligibility lacks selected capability bytes")
        selected_contract = target_registry.resolve(target).capability_contract_path
        if (
            _pinned_native_file({"contract": contract_pin}, "contract") != selected_contract
            or not selected_contract.is_absolute()
        ):
            raise NativeModelExecutionError("eligibility names a different selected target capability contract")

        from merlin.compiler.frontend.linalg_mlir import parse_mlir_text
        from merlin.development.targetgen.applications.demand_inventory import indexed_demand_operations

        module = parse_mlir_text(source.decode("utf-8"))
        funcs = [op for op in module.body.block.ops if op.name == "func.func"]
        if len(funcs) != 1 or len(funcs[0].body.blocks) != 1:
            raise NativeModelExecutionError("frozen source has no unique single-block model entry")
        ops = [op for op in funcs[0].body.blocks[0].ops if op.name != "func.return"]

        def real_name(op: Any) -> str:
            return op.op_name.data if op.name == "builtin.unregistered" else op.name

        indexed = indexed_demand_operations(ops)
        regions = certificate.get("regions")
        if (
            not isinstance(regions, list)
            or certificate.get("n_regions") != len(regions)
            or len(regions) != len(indexed)
        ):
            raise NativeModelExecutionError("independent eligibility lacks a complete source-operation inventory")
        unknown_formats = sum(
            isinstance(row, Mapping) and row.get("precision_transform_required") is None for row in regions
        )
        conversion_obligations = sum(
            isinstance(row, Mapping) and row.get("precision_transform_required") is True for row in regions
        )
        precision = certificate.get("precision_transform_verification")
        if (
            type(certificate.get("n_unknown_capture_formats")) is not int
            or certificate["n_unknown_capture_formats"] != unknown_formats
            or type(certificate.get("n_precision_transform_obligations")) is not int
            or certificate["n_precision_transform_obligations"] != conversion_obligations
            or not isinstance(precision, Mapping)
            or precision.get("status") != "not_required"
            or unknown_formats
            or conversion_obligations
        ):
            raise NativeModelExecutionError(
                "independent source eligibility has unknown capture precision or an unverified precision transform"
            )
        from merlin.common.target_data import get as quant_format

        def same_observed_format(observed: Any, admitted: Any) -> bool:
            if not isinstance(observed, str) or not isinstance(admitted, str):
                return False
            try:
                return quant_format(observed).name == quant_format(admitted).name
            except (KeyError, ValueError):
                return False

        eligible: set[int] = set()
        source_regions: dict[int, str | None] = {}
        source_families: dict[int, str] = {}
        for (index, op), row in zip(indexed, regions, strict=True):
            if not isinstance(row, Mapping) or type(row.get("target_eligible")) is not bool:
                raise NativeModelExecutionError("independent source eligibility row is malformed")
            rid = getattr(op.attributes.get("prov.region_id"), "data", None)
            if row.get("region_id") != rid or row.get("carrier_op") != real_name(op):
                raise NativeModelExecutionError(f"independent source eligibility does not identify operation {index}")
            source_regions[index] = rid
            if row["target_eligible"]:
                if not same_observed_format(row.get("captured_input_format"), row.get("eligibility_input_format")):
                    raise NativeModelExecutionError(
                        f"eligible source operation {index} lacks exact observed input-format admission"
                    )
                weight = row.get("captured_weight_format")
                admitted_weight = row.get("eligibility_weight_format")
                if row.get("semantic_family") in {"contraction", "convolution"}:
                    if not same_observed_format(weight, admitted_weight):
                        raise NativeModelExecutionError(
                            f"eligible source operation {index} lacks exact observed weight-format admission"
                        )
                elif admitted_weight is not None and not same_observed_format(weight, admitted_weight):
                    raise NativeModelExecutionError(
                        f"eligible source operation {index} has an unverified observed weight format"
                    )
                if not isinstance(rid, str) or not rid:
                    raise NativeModelExecutionError(
                        f"eligible source operation {index} has no provenance region identity"
                    )
                if not isinstance(row.get("semantic_family"), str) or not row["semantic_family"]:
                    raise NativeModelExecutionError("eligible source operation lacks independent semantic family")
                source_families[index] = row["semantic_family"]
                eligible.add(index)
        for index, op in enumerate(ops):
            source_regions.setdefault(index, getattr(op.attributes.get("prov.region_id"), "data", None))

        receipt = (cb.get("params") or {}).get("global_program_plan")
        if not isinstance(receipt, Mapping):
            raise NativeModelExecutionError("candidate emitted no global_program_plan")
        problems: list[str] = []
        source_plan_validation = None
        if any(isinstance(row, Mapping) and row.get("source_precision_witness") for row in regions):
            # Parsed source types close only the *source* census. The existing
            # whole-program protocol checks every materialized source binding,
            # including entry/return values, against exact source type/shape;
            # candidate-volunteered source_values alone cannot establish that.
            # Lowered task tags are structural, not arithmetic equivalence.
            from merlin.development.targetgen.oot_starterkit.plan import validate_mixed_program_plan
            from merlin.target.contract.interfaces.abi import lowered_plan_abi

            source_plan_validation = validate_mixed_program_plan(
                source,
                cb,
                _read_pinned_payload(emission, "lowered_mlir").decode("utf-8"),
                lowered_abi=lowered_plan_abi(yaml.safe_load(selected_contract.read_text()), target=target),
            )
            if not source_plan_validation.get("ok"):
                problems.extend(f"candidate typed source plan: {issue}" for issue in source_plan_validation["findings"])
        if receipt.get("schema") != "mixed_program_plan_v1":
            problems.append("candidate global-plan schema is not mixed_program_plan_v1")
        if receipt.get("source_sha256") != source_sha:
            problems.append("candidate global plan identifies different source bytes")
        if receipt.get("source_op_count") != len(ops):
            problems.append("candidate global plan has wrong source-operation count")
        tasks = receipt.get("tasks")
        if not isinstance(tasks, list) or not tasks:
            problems.append("candidate global plan has no tasks")
            tasks = []
        owners: dict[int, str] = {}
        task_regions: list[dict[str, Any]] = []
        for ordinal, task in enumerate(tasks):
            if not isinstance(task, Mapping) or task.get("task_index") != ordinal:
                problems.append(f"candidate task {ordinal} has a missing or duplicate task identity")
                continue
            indices = task.get("source_op_indices")
            kind = task.get("kind")
            if not isinstance(indices, list) or not indices or not isinstance(kind, str) or not kind.strip():
                problems.append(f"candidate task {ordinal} lacks source indices or execution kind")
                continue
            derived: set[str] = set()
            for index in indices:
                if type(index) is not int or not 0 <= index < len(ops):
                    problems.append(f"candidate task {ordinal} names an absent source operation")
                elif index in owners:
                    problems.append(f"source operation {index} has duplicate candidate task owners")
                else:
                    owners[index] = kind
                    if source_regions[index] is not None:
                        derived.add(source_regions[index])
            task_regions.append(
                {
                    "task_index": ordinal,
                    "kind": kind,
                    "source_op_indices": list(indices),
                    "source_region_ids": sorted(derived),
                }
            )
            if "source_region_ids" in task and task["source_region_ids"] != sorted(derived):
                problems.append(f"candidate task {ordinal} region IDs disagree with frozen source")
        if set(owners) != set(range(len(ops))):
            problems.append("candidate tasks omit source operations")
        host_eligible = sorted(index for index in eligible if owners.get(index) == "host")
        if host_eligible:
            problems.append("independently eligible source operations are assigned to host tasks")
        if problems:
            return {
                "status": "violation",
                "problems": problems,
                "eligible_host_source_op_indices": host_eligible,
                "task_source_regions": task_regions,
                "source_sha256": source_sha,
            }
        return {
            "status": "clean",
            "source_sha256": source_sha,
            "n_source_operations": len(ops),
            "eligible_source_op_indices": sorted(eligible),
            "eligible_source_regions": [
                {
                    "source_op_index": index,
                    "region_id": source_regions[index],
                    "semantic_family": source_families[index],
                }
                for index in sorted(eligible)
            ],
            "eligible_host_source_op_indices": [],
            "task_source_regions": task_regions,
            **({"source_plan_validation": source_plan_validation} if source_plan_validation is not None else {}),
            "scope": "static source/task/eligibility join; no completed candidate dispatch proof",
        }
    except Exception as exc:  # noqa: BLE001 -- absent independent identity never permits placement
        return {"status": "unverified", "detail": f"{type(exc).__name__}: {exc}"}


def _pinned_native_file(record: Mapping[str, Any], role: str) -> Path:
    pin = record.get(role)
    if not isinstance(pin, Mapping) or not isinstance(pin.get("path"), str):
        raise NativeModelExecutionError(f"native {role} has no file pin")
    path = Path(pin["path"])
    if (
        not path.is_absolute()
        or path.is_symlink()
        or path.resolve() != path
        or not path.is_file()
        or _digest(path) != pin
    ):
        raise NativeModelExecutionError(f"native {role} bytes do not match their file pin")
    return path


def _frozen_model_policy(capsule_dir: Path, *, target: str, source_bound_groups: bool = False) -> dict[str, Any]:
    from merlin.development.targetgen.execution.native.model_policy import _frozen_model_policy as read_policy

    return read_policy(capsule_dir, target=target, source_bound_groups=source_bound_groups)


def _functional_engine(target: str):
    """Preserve the original direct-ELF tier selection before reopening bytes."""
    from merlin.runtime.backends.registry import get_backend
    from merlin.target.simulators.oracle_policy import oracle_tier_plan, selected_sim_via

    via = selected_sim_via(target)
    plan = oracle_tier_plan(target, via)
    if via != "chipyard" or "L2" not in plan.tiers:
        raise NativeModelExecutionError("selected target metadata has no direct-ELF functional L2 tier")
    backend = get_backend(target)
    from merlin.development.targetgen.execution.native.model_policy import functional_engine_bytes

    return functional_engine_bytes(backend, target=target)


def audit_candidate_static_tiers(
    emission: Mapping[str, Any] | None,
    certificate: Mapping[str, Any] | None,
    native: Mapping[str, Any] | None,
    *,
    target: str,
    entry_symbol: str,
) -> dict[str, Any]:
    """Recompute L0 source/IR structure and L1 exact-ELF ISA legality.

    These are candidate-artifact checks, not host reference numeric executions.
    An L0/L1 result never stands in for L2 or L3 functional execution.
    """
    from merlin.development.targetgen.packaging.bundle_harness import is_executable_emission

    tiers: dict[str, Any] = {}
    try:
        if not isinstance(emission, Mapping) or not isinstance(native, Mapping):
            raise NativeModelExecutionError("candidate emission/native build is absent")
        cb = json.loads(_read_pinned_payload(emission, "command_buffer").decode("utf-8"))
        lowered = _read_pinned_payload(emission, "lowered_mlir").decode("utf-8")
        source = _read_pinned_payload(emission, "source_interface").decode("utf-8")
        ok, reason = is_executable_emission(cb, artifact_text=lowered, entry_symbol=entry_symbol)
        if not ok:
            raise NativeModelExecutionError(reason)
        placement = audit_candidate_source_placement(emission, certificate, target=target)
        if placement.get("status") != "clean":
            raise NativeModelExecutionError("source placement is not independently verified")
        host = audit_emitted_host_compute(emission, entry_symbol=entry_symbol)
        if host.get("status") != "clean":
            raise NativeModelExecutionError("candidate task-scoped host compute is not clean")
        from xdsl.parser import Parser

        from merlin.analysis.programs.plan_evidence import verify_compiler_global_plan
        from merlin.compiler.ir.llvm_context import make_llvm_context

        context = make_llvm_context()
        module = Parser(context, lowered).parse_module()
        plan = verify_compiler_global_plan(
            source_text=source,
            lowered_text=lowered,
            command_buffer=cb,
            candidate_sha256=hashlib.sha256(lowered.encode()).hexdigest(),
            parsed_lowered_module=module,
        )
        if plan.get("status") != "verified" or (plan.get("control_flow") or {}).get("status") != "verified":
            raise NativeModelExecutionError("candidate source/ABI/task CFG is not verified")
        candidate = native.get("candidate") or {}
        if (
            candidate.get("command_buffer_sha256")
            != hashlib.sha256(json.dumps(cb, sort_keys=True).encode()).hexdigest()
            or candidate.get("lowered_mlir_sha256") != hashlib.sha256(lowered.encode()).hexdigest()
        ):
            raise NativeModelExecutionError("candidate build has different CB/LLVM bytes")
        elf = _pinned_native_file(native, "elf")
        policy = _frozen_model_policy(
            _pinned_native_file(native["source"], "capsule_declaration").parent, target=target
        )
        if native.get("frozen_policy") != policy:
            raise NativeModelExecutionError("frozen model instruction/tier policy changed after build")
        tiers["L0"] = {
            "status": "pass",
            "scope": "candidate source/CB/LLVM/ABI/task CFG and host-veto structure",
            "plan_digest": plan.get("plan_digest"),
            "elf": _digest(elf),
        }
    except Exception as exc:  # noqa: BLE001 -- missing structure cannot be credited
        tiers["L0"] = {"status": "unverified", "detail": f"{type(exc).__name__}: {exc}"}
        tiers["L1"] = {"status": "unverified", "detail": "L0 candidate identity/structure is incomplete"}
        return tiers
    from merlin.development.targetgen.execution.native.model_policy import audit_native_instruction_tier

    tiers["L1"] = audit_native_instruction_tier(
        native,
        target=target,
        module=module,
        policy=policy,
        allow_empty_instructions=(
            policy.get("public_zero_device_demand") is True and placement.get("eligible_source_op_indices") == []
        ),
    )
    return tiers
