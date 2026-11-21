"""Public source-bound model build using the coordinator's fixed host provider.

This adapter never consumes a candidate whole-program LLVM artifact. Existing
model gates must independently inspect its ordinary device/source/native proof;
a completed build or numeric match does not authorize a tier by itself.
"""

from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path

from merlin.compiler.build.owned_sources import OwnedHostModelCompiler
from merlin.compiler.frontend.inputs import MaterializedModelInputs
from merlin.compiler.lower.device.source_compute_groups import form_exact_groups
from merlin.development.targetgen.execution.host.dispatch_checks import (
    CandidateDispatchRefusal,
    _composition_preferences,
    _functional_engine_for_facts,
    _host_call_proof,
    _independently_eligible_group_ids,
    _llvm_function,
    _mandatory_source_commands,
    _native_engine_for_facts,
    _optimized_command_functs,
    _pinned,
    _require_instruction_classes,
    _resolve_weight_preparation,
    _same_owned_frozen_policy,
    _selected_facts,
    _selected_file,
    _stopped_before_receipt,
    _untagged_assert_comparisons,
    _verify_device_custody,
)
from merlin.development.targetgen.execution.host.selections import _numeric_full_output
from merlin.development.targetgen.execution.host.selections import group_profile as _group_profile
from merlin.runtime.execution.model_inputs import file_sha256

# ---- owned_host_model_execution ----------------------------------------------------------------


def _resolve_source_composition(module, groups, eligible_groups, *, preferences, eligible_source_regions=None):
    """Let the original source owner resolve an optional complete sibling route.

    The candidate cannot select roots, replace epilogues or certify an ABI. The
    source reader must cover every independently eligible group; actual member
    work, synchronous effects, objects, host calls and execution remain separate
    mandatory checks after the build.
    """
    if preferences.get("independent_heads", False):
        from merlin.development.targetgen.execution.host.selections import resolve_request

        resolved = resolve_request(
            module, groups, eligible_groups, preferences=preferences, eligible_source_regions=eligible_source_regions
        )
        if resolved is not None:
            return resolved
    if not preferences["shared_input_siblings"]:
        return None, {"status": "not_requested"}
    from hashlib import sha256

    from xdsl.dialects import func

    from merlin.compiler.dialects.xdsl_context import text
    from merlin.compiler.lower.device.sibling_source_route import SiblingSourceRoute
    from merlin.compiler.lower.host.borrowed_effects import BorrowedBufferEffects
    from merlin.compiler.lower.source.region import outline_source_siblings
    from merlin.compiler.partition.groups.compute_groups import HOST
    from merlin.compiler.schedule.shared_operands.shared_inputs import find_shared_input_groups
    from merlin.development.backend.qualification.provenance.groups import eligible_source_identity

    before = text(module, generic=True)
    try:
        outlined, _ = outline_source_siblings(module)
        shared = find_shared_input_groups(outlined.core)
        if len(shared) != 1:
            raise ValueError("source does not expose one complete checked sibling group")
        group = shared[0]
        group.verify_current(outlined.core)
        function = next(op for op in outlined.core.body.block.ops if isinstance(op, func.FuncOp))
        operations = tuple(function.body.block.ops)
        identities = [eligible_source_identity(operations[i]) for i in group.contraction_ordinals]
        roster = [(eligible_source_identity(g.root), g.index) for g in groups if g.placement != HOST]
        original = dict(roster)
        if len(original) != len(roster):
            raise ValueError("original source groups have ambiguous duplicate identities")
        if len(set(identities)) != len(identities) or any(identity not in original for identity in identities):
            raise ValueError("sibling source has missing or ambiguous original group identities")
        covered = [original[identity] for identity in identities]
        if not set(eligible_groups) <= set(covered):
            raise ValueError("sibling route does not cover every independently eligible source group")
        if eligible_source_regions is not None:
            original_function = next(op for op in module.body.block.ops if isinstance(op, func.FuncOp))
            original_indices = {id(op): index for index, op in enumerate(original_function.body.block.ops)}
            root_obligations = {
                (original_indices.get(id(g.root)), g.index, eligible_source_identity(g.root)[0], "contraction")
                for g in groups
                if g.index in covered
            }
            if (
                not isinstance(eligible_source_regions, list)
                or not eligible_source_regions
                or any(
                    not isinstance(row, dict)
                    or type(row.get("source_op_index")) is not int
                    or (
                        row["source_op_index"],
                        row.get("source_group"),
                        row.get("region_id"),
                        row.get("semantic_family"),
                    )
                    not in root_obligations
                    for row in eligible_source_regions
                )
            ):
                raise ValueError(
                    "sibling raw core cannot replace an independently required supporting source operation"
                )
    except ValueError as exc:
        if text(module, generic=True) != before:
            raise ValueError("source composition reader changed original source") from exc
        from merlin.development.targetgen.execution.host.selections import resolve_components

        try:
            return resolve_components(module, groups, eligible_groups, eligible_source_regions=eligible_source_regions)
        except ValueError as component_error:
            return None, {"status": "not_applicable", "reason": str(component_error), "baseline_retained": True}
    if text(module, generic=True) != before:
        raise ValueError("source composition reader changed original source")
    selected = SiblingSourceRoute(
        sha256(before.encode()).hexdigest(),
        BorrowedBufferEffects(
            "owned host requires synchronous noescape/no-release logical calls; "
            "actual implementation must independently pass dispatch and effect checks",
            True,
            True,
            True,
        ),
    )
    return selected, {
        "status": "source_selected_not_qualified",
        "source_ir_sha256": selected.source_ir_sha256,
        "original_source_groups": covered,
        "original_source_denominator_unchanged": True,
    }


def execute_owned_host_model(
    *,
    capsule,
    candidate_package,
    target,
    bundle,
    provenance,
    verify,
    output,
    selection_record,
    simulator,
    timeout,
    source_certificate=None,
    readback_policy=None,
    reachable_oracle_tiers=None,
    descriptor_effects=None,
):
    from merlin.development.backend.qualification.provenance import (
        private_instruction_coordinator as coordinator,
    )
    from merlin.development.backend.qualification.provenance import (
        private_instruction_declaration as declaration,
    )
    from merlin.development.targetgen.execution.host.model_audit import _execute_required_tiers
    from merlin.evaluation.capsule.models.runtime import _resolve_model_host_lane

    if not isinstance(selection_record, dict):
        raise ValueError("public model needs coordinator-selected owned host build inputs")
    selection_path = _selected_file({"selection": selection_record}, "selection")
    selection = json.loads(selection_path.read_text())
    if set(selection) != {"board_catalog", "dts", "board", "arena_mb", "instruction_declaration", "rtl_facts"}:
        raise ValueError("owned host build selection fields are incomplete or unexpected")
    catalog = _selected_file(selection, "board_catalog")
    dts = _selected_file(selection, "dts")
    instruction = _selected_file(selection, "instruction_declaration")
    facts = _selected_file(selection, "rtl_facts")
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise ValueError("owned host model needs a fresh output directory")
    from merlin.common.artifacts import new_product

    product = new_product(
        "owned-host-model",
        version=1,
        target=target,
        update_latest=False,
        sources=[str(output)],
        notes="Owned-host model build and source-bound native evidence custody",
    )
    output = product.path
    from merlin.target.contract.harness.readback_policy import FULL_VALUES_BIN, selected

    readback_policy = selected(readback_policy)
    if readback_policy is not None and readback_policy.transport != FULL_VALUES_BIN:
        raise ValueError("owned saved-model compiler needs its selected full binary value transport")
    prepared = coordinator.prepare(
        declaration.read(instruction, target=target),
        target=target,
        output=(output / "instruction_selection").absolute(),
    )
    attrs = (capsule.get("operation") or {}).get("attributes") or {}
    experiment, host, identity = _resolve_model_host_lane(target, attrs.get("compile_dtype", "fp32"))
    service = OwnedHostModelCompiler.select(
        host_package=host, target=target, board_catalog=catalog, board=selection["board"], dts=dts
    )
    verify()
    # Keep the validated bytes for independent durable-result auditing. This
    # copies the trusted materialization, never a candidate-supplied capture.
    retained = output / "runtime_inputs"
    shutil.copytree(bundle, retained)
    bundle = retained
    materialized = MaterializedModelInputs.bind(
        bundle,
        artifacts=provenance["bundle"]["artifacts"],
        validator_identity=provenance["tool"],
        source_identity=provenance["source"],
    )
    # Validate each required source operation before invoking the candidate, including its actual
    # group assignment. Supporting fusion is selected by Merlin's ordinary capability checks.
    from merlin.compiler.ir import mlir_query as mq
    from merlin.compiler.partition.groups import compute_groups as groups_api
    from merlin.compiler.partition.groups import (
        formation_compute_groups as _merlin_formation_compute_groups_1,
    )

    source_path = Path(bundle) / "model.mlir"
    module = mq.parse(source_path.read_text())
    groups, _ = form_exact_groups(module, target, bundle, _merlin_formation_compute_groups_1.form_groups)
    placement = _source_placement(capsule, source_certificate, module, groups, target=target, source_path=source_path)
    frozen_split = placement["frozen_placement"]
    preferences, preference_pin = _composition_preferences(candidate_package)
    sibling, composition_selection = _resolve_source_composition(
        module,
        groups,
        _independently_eligible_group_ids(placement, groups),
        preferences=preferences,
        eligible_source_regions=placement["eligible_source_regions"],
    )
    prepack_weights, preparation_selection = _resolve_weight_preparation(
        bundle,
        requested=preferences["prepack_weights"],
    )
    from merlin.compiler.lower.device.head import HeadBatchSourceRoute

    head_selection = sibling if type(sibling) is HeadBatchSourceRoute else None
    from merlin.development.targetgen.execution.host.source_composition_preferences import (
        host_preparation,
        require_host_preparation,
    )

    host_options, host_selection = host_preparation(
        preferences,
        target=target,
        dtype=attrs.get("compile_dtype", "fp32"),
        composed=sibling is not None,
        lane=(experiment, host, identity),
        descriptor_effects=descriptor_effects,
    )
    receipt = service.compile(
        capture=bundle,
        candidate_package=candidate_package,
        datapath=attrs.get("dtype"),
        linked_elf_admission=prepared.service,
        output=output / "model",
        arena_mb=selection["arena_mb"],
        timeout_s=timeout,
        run="none",
        rtl_facts=facts,
        materialized_inputs=materialized,
        # The original source audit requires every admitted accelerator group.
        require_all_accelerator_groups=True,
        sibling_source=None if head_selection is not None else sibling,
        **({"independent_heads": head_selection} if head_selection is not None else {}),
        prepack_weights=prepack_weights,
        share_exact_tensor_preparations=preferences.get("share_exact_tensor_preparations", False),
        dense_input_views=preferences.get("dense_input_views", False),
        op_profile=preferences.get("op_profile", False),
        group_profile=preferences.get("group_profile", False),
        **host_options,
    )
    verify()
    require_host_preparation(receipt, host_options, host_selection)
    audit = _audit_owned_host_device_build(
        bundle,
        receipt,
        target=target,
        capsule=capsule,
        certificate=source_certificate,
        sibling_source=sibling,
        descriptor_effects=descriptor_effects,
    )
    prepared.verify()
    for role in ("board_catalog", "dts", "instruction_declaration", "rtl_facts"):
        _selected_file(selection, role)
    _selected_file({"selection": selection_record}, "selection")
    record = {
        "status": "compiled_not_run",
        "scope": "Merlin-owned host model with candidate closed device groups; "
        "mandatory source/device/tier gates remain independent",
        "host_provider": identity,
        "build": receipt,
        "source_device_join": audit,
        "frozen_placement": frozen_split,
        "instruction_selection": prepared.record(),
        "materialization": materialized.record(),
        "selection_record": selection_record,
        "runtime_inputs": str(retained.absolute()),
        "readback_policy": readback_policy.record() if readback_policy is not None else None,
        "source_composition_selection": {
            **composition_selection,
            "preferences": preferences,
            "preference_file": preference_pin,
            "weight_preparation": preparation_selection,
            **({"host_preparation": host_selection} if host_selection else {}),
        },
    }
    record.update(
        _execute_required_tiers(
            capsule,
            record,
            target=target,
            facts=facts,
            timeout=timeout,
            reachable_oracle_tiers=reachable_oracle_tiers,
        )
    )
    verify()
    materialized.verify(bundle)
    prepared.verify()
    require_host_preparation(receipt, host_options, host_selection)
    if _composition_preferences(candidate_package) != (preferences, preference_pin):
        raise ValueError("source composition preferences changed during the original build")
    _selected_file({"selection": selection_record}, "selection")
    (output / "qualification.json").write_text(json.dumps(record, sort_keys=True, indent=2) + "\n")
    for artifact in ("qualification.json", "model", "runtime_inputs", "instruction_selection"):
        product.add_artifact(artifact)
    product.write_manifest()
    return record


def _frozen_declaration(capsule):
    """The pinned capsule's own declaration, never an in-memory copy a caller could have edited."""
    import yaml

    root = Path(capsule["__dir__"])
    declaration = yaml.safe_load((root / "capsule.yaml").read_text())
    if not isinstance(declaration, dict):
        raise ValueError("frozen model capsule declaration is malformed")
    return declaration, root


def _frozen_split(capsule, source_path, *, target, groups=None, eligible=None, module=None):
    """Bind Merlin's device groups (and, when given, fresh eligibility) to the frozen placement."""
    from merlin.compiler.ir import mlir_query as mq
    from merlin.compiler.partition.groups import compute_groups as groups_api
    from merlin.compiler.partition.groups import (
        formation_compute_groups as _merlin_formation_compute_groups_2,
    )
    from merlin.development.targetgen.execution.host import frozen_placement

    declaration, root = _frozen_declaration(capsule)
    frozen = frozen_placement.read(declaration, root)
    if module is None:
        module = mq.parse(Path(source_path).read_text())
    if groups is None:
        groups, _ = form_exact_groups(module, target, source_path, _merlin_formation_compute_groups_2.form_groups)
    return frozen_placement.require_split(
        frozen,
        source_sha256=file_sha256(source_path),
        device=frozen_placement.device_operations(groups, groups_api.HOST),
        eligible=eligible,
        source_operations=Counter(frozen_placement.operation_name(op) for op in mq.walk(module)),
    )


def _rederive_composition_selection(
    bundle, receipt, selection, *, module, groups, placement, target, compile_dtype="fp32", descriptor_effects=None
):
    """Reopen requests and resolve original owner authority during promotion."""
    from merlin.runtime.execution.model_inputs import strict_tree_sha256

    device = receipt.get("inputs", {}).get("device")
    if not isinstance(device, dict) or device.get("name") != target:
        raise ValueError("owned composition audit lost its selected candidate package")
    package = Path(device["package"])
    if (
        not package.is_absolute()
        or any(path.is_symlink() for path in (package, *package.parents))
        or strict_tree_sha256(package) != device.get("package_tree")
    ):
        raise ValueError("owned composition candidate package differs from the actual compiled snapshot")
    preferences, preference_pin = _composition_preferences(package)
    sibling, resolved = _resolve_source_composition(
        module,
        groups,
        _independently_eligible_group_ids(placement, groups),
        preferences=preferences,
        eligible_source_regions=placement["eligible_source_regions"],
    )
    prepack, preparation = _resolve_weight_preparation(bundle, requested=preferences["prepack_weights"])
    from merlin.development.targetgen.execution.host.source_composition_preferences import (
        host_preparation,
        require_host_preparation,
    )

    host_options, host_selection = host_preparation(
        preferences,
        target=target,
        dtype=compile_dtype,
        composed=sibling is not None,
        descriptor_effects=descriptor_effects,
    )
    expected = {
        **resolved,
        "preferences": preferences,
        "preference_file": preference_pin,
        "weight_preparation": preparation,
        **({"host_preparation": host_selection} if host_selection else {}),
    }
    require_host_preparation(receipt, host_options, host_selection)
    if (
        selection != expected
        or receipt["inputs"].get("prepack_weights", False) is not prepack
        or receipt["inputs"].get("share_exact_tensor_preparations", False)
        is not preferences.get("share_exact_tensor_preparations", False)
        or receipt["inputs"].get("dense_input_views", False) is not preferences.get("dense_input_views", False)
        or receipt["inputs"].get("op_profile", False) is not preferences.get("op_profile", False)
        or receipt["inputs"].get("group_profile", False) is not preferences.get("group_profile", False)
    ):
        raise ValueError("owned composition selection differs from independently resolved original source")
    return sibling


def _audit_owned_host_device_build(
    bundle,
    receipt,
    *,
    target,
    capsule=None,
    certificate=None,
    sibling_source=None,
    composition_selection=None,
    descriptor_effects=None,
):
    """Recompute original group identities and audit the actual linked objects."""
    from merlin.compiler.ir import mlir_query as mq
    from merlin.compiler.partition.groups import compute_groups as groups_api
    from merlin.compiler.partition.groups import (
        formation_compute_groups as _merlin_formation_compute_groups_3,
    )
    from merlin.compiler.partition.groups import plan_compute_groups as _merlin_plan_compute_groups_3
    from merlin.development.backend.qualification.full_models.audits import audit_built_device_host_compute
    from merlin.development.backend.qualification.full_models.private_full_models import _linked_symbols
    from merlin.development.backend.qualification.provenance.groups import (
        convolution_input_preparations,
        join_routed_source_groups,
        join_sibling_source_composition,
        require_intended_host_selection,
        routed_kernel_symbols,
        source_group_roster,
    )

    # Match the exact source passed to OwnedHostModelCompiler.plan_before_build.
    text = (Path(bundle) / "model.mlir").read_text()
    module = mq.parse(text)
    groups, _ = form_exact_groups(module, target, bundle, _merlin_formation_compute_groups_3.form_groups)
    _merlin_plan_compute_groups_3.require_explained(_merlin_plan_compute_groups_3.plan(module, target, groups=groups))
    placement = _source_placement(
        capsule, certificate, module, groups, target=target, source_path=Path(bundle) / "model.mlir"
    )
    eligible_groups = _independently_eligible_group_ids(placement, groups)
    source = source_group_roster(groups, eligible_groups, host_placement=groups_api.HOST, module=module)
    if composition_selection is not None:
        rederived = _rederive_composition_selection(
            bundle,
            receipt,
            composition_selection,
            module=module,
            groups=groups,
            placement=placement,
            target=target,
            compile_dtype=((capsule or {}).get("operation", {}).get("attributes") or {}).get("compile_dtype", "fp32"),
            descriptor_effects=descriptor_effects,
        )
        if sibling_source is not None and (rederived is None or sibling_source.record() != rederived.record()):
            raise ValueError("owned composition re-audit differs from the original typed caller selection")
        sibling_source = rederived
    output = receipt["output"]
    sidecar = output["device_sidecar"]
    path = Path(sidecar["path"])
    if file_sha256(path) != sidecar["sha256"]:
        raise ValueError("owned host device sidecar changed")
    emitted = json.loads(path.read_text())
    if emitted.get("device") != target:
        raise ValueError("owned host model omitted eligible device groups")
    assigned = emitted.get("routed")
    composition = None
    if sibling_source is None:
        joined = join_routed_source_groups(assigned, source)
    else:
        from hashlib import sha256

        from merlin.compiler.dialects.xdsl_context import text as ir_text
        from merlin.compiler.lower.device.head import HeadBatchSourceRoute
        from merlin.compiler.lower.device.sibling_source_route import SiblingSourceRoute

        if (
            type(sibling_source) not in (SiblingSourceRoute, HeadBatchSourceRoute)
            or sibling_source.source_ir_sha256 != sha256(ir_text(module, generic=True).encode()).hexdigest()
        ):
            raise ValueError("owned sibling source audit lost its original source selection")
        sibling_source.record()
        preparation = Path(output["elf"]).parent / "bundle_preparation.json"
        if type(sibling_source) is HeadBatchSourceRoute:
            from merlin.development.backend.qualification.provenance.composition import join_heads

            composition = join_heads(
                emitted,
                source,
                module=module,
                target=target,
                capture=bundle,
                preparation_receipt=preparation if receipt.get("inputs", {}).get("prepack_weights") else None,
                shared_input_siblings=sibling_source.shared_input_siblings,
                effects=sibling_source.effects,
                whole_batch=sibling_source.whole_batch,
                shared_rhs_row_views=sibling_source.shared_rhs_row_views,
            )
        elif sibling_source.partial:
            from merlin.development.backend.qualification.provenance.composition import join_components

            composition = join_components(
                emitted,
                source,
                module=module,
                target=target,
                effects=sibling_source.effects,
                capture=bundle,
                preparation_receipt=preparation if receipt.get("inputs", {}).get("prepack_weights") else None,
            )
        else:
            composition = join_sibling_source_composition(
                emitted,
                source,
                module=module,
                target=target,
                preparation_receipt=preparation if receipt.get("inputs", {}).get("prepack_weights") else None,
            )
        # One real call remains one call; every original member stays separately
        # listed in the composition and in the mandatory work denominator.
        joined = {}
    require_intended_host_selection(emitted.get("skipped", []))
    elf = Path(output["elf"])
    if file_sha256(elf) != output["elf_sha256"]:
        raise ValueError("owned host linked ELF changed")
    _linked_symbols(elf, routed_kernel_symbols(emitted))
    if composition is not None and composition.get("schema") in {
        "merlin.original_sibling_source_components.v1",
        "merlin.original_head_source_composition.v1",
    }:
        from merlin.development.backend.qualification.provenance.composition import source_subset

        ordinary = composition["ordinary_joined_groups"]
        source["conv_input_preparation"] = convolution_input_preparations(
            {**emitted, "routed": [row for row in assigned if row["group"] in ordinary]},
            source_subset(source, set(composition["composed_source_groups"])),
        )
    else:
        source["conv_input_preparation"] = (
            {} if composition is not None else convolution_input_preparations(emitted, source)
        )
    audits = audit_built_device_host_compute(
        elf.parent,
        assigned,
        source,
        target,
        diagnostics=elf.parent / "audit-diagnostics",
        host_artifacts=output.get("host_artifacts"),
        **({"source_composition": composition} if composition is not None else {}),
    )
    return {
        "source": source,
        "placement": placement,
        "joined_groups": joined,
        "static_device_host_compute": audits,
        "assigned": assigned,
        **({"composition": composition} if composition is not None else {}),
        "optional_source_groups_routed": sorted(
            set(composition["covered_source_groups"] if composition is not None else joined.values())
            & set(source["optional_groups"])
        ),
        "scope": "source identities joined to actual linked device objects; "
        "dynamic execution and required tiers remain separate",
    }


def _source_placement(capsule, certificate, module, groups, *, target, source_path):
    """Join original source operations to Merlin's independently selected groups."""
    import yaml

    from merlin.common.target_data import get as quant_format
    from merlin.compiler.partition.groups import compute_groups as groups_api
    from merlin.development.targetgen.execution.native.native_model_execution import (
        independent_frozen_source_eligibility,
    )

    root = Path(capsule["__dir__"])
    declaration = yaml.safe_load((root / "capsule.yaml").read_text())
    fresh = independent_frozen_source_eligibility({**declaration, "__dir__": str(root)}, target=target)
    if certificate is not None and certificate != fresh:
        raise ValueError("owned source certificate differs from frozen source")
    if fresh["source_mlir_sha256"] != file_sha256(source_path):
        raise ValueError("owned host routed another source")
    if fresh.get("n_precision_transform_obligations") != 0 or (fresh.get("precision_transform_verification") or {}).get(
        "status"
    ) not in {"not_required", "unknown_capture_format"}:
        raise ValueError("owned source has unverified captured precision obligations")
    # Layout/scalar host entries need not have an accelerator operand format.
    # Their frozen host body admission remains mandatory; unknown formats may
    # never authorize device work. The eligible rows are checked below against
    # actual captured input/weight types, without changing the certificate.
    if fresh.get("n_unknown_capture_formats"):
        from merlin.development.targetgen.execution.host.frozen_placement import read as read_frozen_placement

        read_frozen_placement(declaration, root)
    functions = [op for op in module.body.block.ops if op.name == "func.func"]
    if len(functions) != 1 or len(functions[0].body.blocks) != 1:
        raise ValueError("owned source has no unique single-block model entry")
    ops = [op for op in functions[0].body.block.ops if op.name != "func.return"]

    def name(op):
        return op.op_name.data if op.name == "builtin.unregistered" else op.name

    # Literal constants and empty tensor allocations have no semantic routing
    # demand; keep their original indices for device/source joins. Computation
    # and address/control operations still require independent inventory rows.
    def untagged_setup(op):
        if "prov.op" in op.attributes:
            return False
        if name(op) in {"arith.constant", "tensor.empty", "tensor.splat"}:
            return True
        stage = groups_api.classify(op)
        return stage is not None and stage.kind in {groups_api.VIEW, groups_api.MOVEMENT}

    from merlin.development.targetgen.applications.demand_inventory import indexed_demand_operations
    from merlin.development.targetgen.execution.host.frozen_placement import read as read_frozen_placement

    frozen = read_frozen_placement(declaration, root)
    assertion_support = _untagged_assert_comparisons(ops, frozen, groups)
    indexed = [
        (i, op)
        for i, op in indexed_demand_operations(ops)
        if not untagged_setup(op) and id(op) not in assertion_support
    ]
    rows = fresh.get("regions")
    if not isinstance(rows, list) or len(rows) != len(indexed) or fresh.get("n_regions") != len(rows):
        raise ValueError("owned source lacks complete operation eligibility")
    owner = {id(member): group for group in groups for member in group.members}
    eligible = []
    for (index, op), row in zip(indexed, rows, strict=True):
        rid = getattr(op.attributes.get("prov.region_id"), "data", None)
        if (
            type(row.get("target_eligible")) is not bool
            or row.get("region_id") != rid
            or row.get("carrier_op") != name(op)
        ):
            raise ValueError("owned source eligibility names another operation")
        if not row["target_eligible"]:
            group = owner.get(id(op))
            if row.get("eligibility_undetermined") and group is not None and group.placement != groups_api.HOST:
                raise ValueError("unknown captured precision cannot authorize supporting device work")
            continue
        for observed, admitted in ((row.get("captured_input_format"), row.get("eligibility_input_format")),):
            if (
                not isinstance(observed, str)
                or not isinstance(admitted, str)
                or quant_format(observed).name != quant_format(admitted).name
            ):
                raise ValueError("eligible owned source lacks observed input-format admission")
        if (
            row.get("semantic_family") in {"contraction", "convolution"}
            or row.get("eligibility_weight_format") is not None
        ):
            observed, admitted = row.get("captured_weight_format"), row.get("eligibility_weight_format")
            if (
                not isinstance(observed, str)
                or not isinstance(admitted, str)
                or quant_format(observed).name != quant_format(admitted).name
            ):
                raise ValueError("eligible owned source lacks observed weight-format admission")
        group = owner.get(id(op))
        if group is None or group.placement == groups_api.HOST or not isinstance(rid, str) or not rid:
            raise ValueError("eligible owned source was omitted or assigned to host")
        if not isinstance(row.get("semantic_family"), str) or not row["semantic_family"]:
            raise ValueError("eligible owned source has no independent semantic family")
        eligible.append(
            {
                "source_op_index": index,
                "region_id": rid,
                "semantic_family": row["semantic_family"],
                "source_group": group.index,
            }
        )
    if not eligible:
        raise ValueError("accelerated model has no independently eligible source operation")
    eligible_operations = Counter(name(ops[row["source_op_index"]]) for row in eligible)
    frozen = _frozen_split(
        capsule, source_path, target=target, groups=groups, eligible=dict(eligible_operations), module=module
    )
    return {
        "status": "clean",
        "frozen_placement": frozen,
        "source_sha256": fresh["source_mlir_sha256"],
        "n_source_operations": len(ops),
        "eligible_source_regions": eligible,
        "eligible_source_op_indices": [row["source_op_index"] for row in eligible],
        "certificate": fresh,
    }
