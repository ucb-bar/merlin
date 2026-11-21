"""Exact raw source regions through the existing Merlin host/device compiler.

Selection grants no numerical credit. Frozen placement, complete private inputs,
all live outputs and the native owned-host auditor remain mandatory.
"""

from __future__ import annotations

import contextlib
import json
import tempfile
from pathlib import Path

import yaml

# ---- owned_host_slice_execution ----------------------------------------------------------------


def selected_source_slice(capsule):
    if capsule.get("kind") != "model_slice" or not (capsule.get("semantic") or {}).get("must_accelerate"):
        return False
    # Source intake declares these raw linalg contracts explicitly. Ordinary
    # component interface capsules retain their existing kernel route.
    return (capsule.get("operation") or {}).get("op") in {"model", "host_island_seam"}


def require_slice_placement(capsule):
    from merlin.development.targetgen.execution.host import frozen_placement

    root = Path(capsule["__dir__"])
    declaration = yaml.safe_load((root / "capsule.yaml").read_bytes())
    screen = declaration.get("software_screen") or {}
    if screen.get("status") != "admitted":
        gaps = [
            {key: row.get(key) for key in ("operation", "mlir_operation", "status", "reason")}
            for row in screen.get("decisions", [])
            if row.get("status") != "admitted"
        ]
        raise ValueError("frozen Phase 0 placement is not admitted: " + json.dumps(gaps, sort_keys=True))
    return frozen_placement.read(declaration, root)


def materialize_source_slice(capsule, *, target, output):
    from merlin.development.targetgen.execution.host.leaf_execution import _materialize_typed_source
    from merlin.development.targetgen.execution.native.native_model_execution import (
        independent_frozen_source_eligibility,
    )

    root = Path(capsule["__dir__"])
    if any(path.is_symlink() for path in (root, *root.parents)):
        raise ValueError("source slice root is indirect")
    declaration = yaml.safe_load((root / "capsule.yaml").read_bytes())
    if not selected_source_slice(declaration):
        raise ValueError("source slice lacks its frozen Merlin host route")
    source_name = declaration.get("linalg_mlir")
    if (
        not isinstance(source_name, str)
        or Path(source_name).name != source_name
        or source_name != declaration.get("interface_mlir")
    ):
        raise ValueError("source slice requires one exact local linalg program")
    source = root / source_name
    if not isinstance(target, str) or not target:
        raise ValueError("source slice needs its trusted target selection")
    require_slice_placement(capsule)
    certificate = independent_frozen_source_eligibility({**declaration, "__dir__": str(root)}, target=target)
    if certificate.get("n_eligible", 0) <= 0:
        raise ValueError("source slice has no independently eligible accelerator work")
    return _materialize_typed_source(declaration, source, certificate, output=output)


@contextlib.contextmanager
def runtime_bundle(capsule, *, target, artifact_root=None):
    from merlin.development.targetgen.execution.host.leaf_execution import _verify_source_identity

    with tempfile.TemporaryDirectory(prefix="source-slice-") as temporary:
        output = Path(artifact_root) if artifact_root is not None else Path(temporary) / "bundle"
        bound, certificate, identity = materialize_source_slice(capsule, target=target, output=output)
        provenance = {
            "bundle": {"artifacts": json.loads(bound.identity_bytes)},
            "tool": json.loads(bound.validator_identity_bytes),
            "source": identity,
            "scope": "exact frozen connected source and all live outputs; no full-model equivalence",
        }

        def verify():
            _verify_source_identity(capsule, identity)
            bound.verify(bound.root)

        verify()
        yield bound.root, provenance, verify
        verify()


# ---- owned_host_model_selection ----------------------------------------------------------------
# Select submitted model compilation from frozen placement and fresh original source eligibility.


def owned_model_required(capsule, *, target):
    """Select a compiler route, granting no execution or numerical evidence.

    A whole-program capsule whose frozen Phase 0 placement gives the author accelerator work always
    takes the owned-host route: Merlin compiles its host part, and the route fails closed if the
    placement is unbound or disagrees with the build. It never falls back to the legacy graph.
    """
    from merlin.development.targetgen.execution.host.frozen_placement import requires_merlin_host

    if requires_merlin_host(capsule):
        return True
    if bool((capsule.get("semantic") or {}).get("must_accelerate")):
        return True
    if capsule.get("kind") != "model" and "on_mesh" not in ((capsule.get("lanes") or {}).get("require") or []):
        return False
    from merlin.development.targetgen.execution.native.native_model_execution import (
        independent_frozen_source_eligibility,
    )

    try:
        certificate = independent_frozen_source_eligibility(capsule, target=target)
    except Exception:
        return False
    eligible = certificate.get("n_eligible")
    return type(eligible) is int and eligible > 0


# ---- owned_host_numerics -----------------------------------------------------------------------
# Compare complete owned-model outputs against the frozen numerical policy.


def _numeric_full_output(capsule, record, console):
    """Use the unchanged public numeric policy on every typed forward result."""
    import numpy as np
    import yaml

    from merlin.development.targetgen.grading.golden import compare
    from merlin.development.targetgen.grading.golden_store import load_golden
    from merlin.development.targetgen.packaging.group_reference import owned_model_reference
    from merlin.model import read_full_outputs
    from merlin.runtime.harness.output_decoding import binary_console_diagnostics

    text = binary_console_diagnostics(console).decode("utf-8", errors="replace")
    if sum(line.strip() == "DONE" for line in text.splitlines()) != 1:
        raise ValueError("owned model full console has no unique successful completion")
    values, metrics = read_full_outputs(
        console, Path(record["runtime_inputs"]), record["build"]["output"]["build_hash"]
    )
    order = json.loads((Path(record["runtime_inputs"]) / "output_order.json").read_text())
    capsule_root = Path(capsule["__dir__"])
    declaration = yaml.safe_load((capsule_root / "capsule.yaml").read_text())
    golden = load_golden(capsule_root)
    expected = golden["outputs"]
    if len(order) != len(values) or set(order) != set(expected) or len(order) != len(set(order)):
        raise ValueError("actual owned model output roster differs from frozen full golden")
    observed = {}
    for name, value in zip(order, values, strict=True):
        shape = np.asarray(expected[name]).shape
        if value.size != np.asarray(expected[name]).size:
            raise ValueError("owned full result extent differs from frozen golden")
        observed[name] = value.reshape(shape).tolist()
    policy, source = declaration["numeric_policy"], str(golden.get("golden_source") or "independent_capsule")
    report = compare(expected, observed, policy, golden_source=source)
    # An eager mismatch may still be the reference the placed groups' declared bounds admit.
    report = owned_model_reference(
        report, expected, observed, policy, record=record, order=order, golden_source=source, compare=compare
    )
    return report, metrics


# ---- owned_host_diagnostics --------------------------------------------------------------------
# Coarse original-host profile projections; counter ticks are not performance scores.


def group_profile(record, console):
    """Coarse original-host diagnostics, never a functional-tick performance score."""
    if not record["build"]["inputs"].get("group_profile", False):
        return None
    from merlin.runtime.harness.output_decoding import binary_console_diagnostics
    from merlin.runtime.harness.output_observation import ReadbackError, parse_group_profile

    base = {
        "unit": "counter_ticks",
        "timing_qualified": False,
        "performance_ranking_eligible": False,
        "instrumented": True,
    }
    try:
        profile = parse_group_profile(binary_console_diagnostics(console).decode("utf-8", errors="replace"))
        if profile is None or profile["dropped"]:
            raise ValueError("missing or incomplete group observations")
    except (ReadbackError, ValueError) as exc:
        return {**base, "status": "unavailable", "detail": str(exc)}
    groups = profile["groups"]
    grouped = sum(row["cycles"] for row in groups)
    between = profile["tail_gap"] + sum(row["gap_before"] for row in groups)
    return {
        **base,
        "status": "observed",
        "executions": profile["calls"],
        "group_counter_ticks": grouped,
        "between_group_counter_ticks": between,
        "total_counter_ticks": grouped + between,
        "largest_groups": [
            {"index": row["index"], "counter_ticks": row["cycles"]}
            for row in sorted(groups, key=lambda row: row["cycles"], reverse=True)[:8]
        ],
        "scope": "Instrumented forward groups and gaps; gaps include orchestration and profiler overhead. "
        "Group membership does not itself establish CPU/device placement. "
        "Compare an uninstrumented control.",
    }


def operation_profile(record, console, *, selected_file):
    """Compact optional diagnostics; never turn functional ticks into cycles."""
    from merlin.runtime.harness.output_decoding import binary_console_diagnostics
    from merlin.runtime.harness.output_observation import OperationProfileError, summarize

    build = record["build"]
    if not build["inputs"].get("op_profile", False):
        return None
    table = selected_file(build["output"], "op_profile_table")
    payload = json.loads(table.read_text())
    text = binary_console_diagnostics(console).decode("utf-8", errors="replace")
    try:
        profile = summarize(payload, text)
    except OperationProfileError as exc:
        # Missing diagnostic observations do not erase independently checked
        # numerical results. Changed table identities still refuse above.
        return {"status": "unavailable", "detail": str(exc), "timing_qualified": False}
    return {
        "status": "observed",
        **{
            key: profile[key]
            for key in (
                "schema",
                "unit",
                "timing_qualified",
                "executions",
                "category_counter_ticks",
                "operation_counter_ticks",
                "inter_invocation_counter_ticks",
                "total_counter_ticks",
                "marks",
                "limitations",
            )
        },
        "table": build["output"]["op_profile_table"],
        "instrumented": True,
        "performance_ranking_eligible": False,
        "scope": "Instrumented operation intervals; functional Spike counters are not elapsed cycles. "
        "Compare an uninstrumented control and independently bind the timing engine.",
    }


# ---- source_component_selection ----------------------------------------------------------------
# Original-owner resolution of optional per-component source composition.


def resolve_components(module, groups, eligible_groups, *, eligible_source_regions=None):
    from hashlib import sha256

    from xdsl.dialects import func

    from merlin.compiler.dialects.xdsl_context import text
    from merlin.compiler.lower.device.sibling_components import prepare_sibling_components
    from merlin.compiler.lower.device.sibling_source_route import SiblingSourceRoute
    from merlin.compiler.lower.host.borrowed_effects import BorrowedBufferEffects
    from merlin.development.backend.qualification.provenance.groups import eligible_source_identity

    before = text(module, generic=True)
    effects = BorrowedBufferEffects(
        "owned host requires synchronous noescape/no-release logical calls; "
        "actual implementation must independently pass dispatch and effect checks",
        True,
        True,
        True,
    )
    original = {eligible_source_identity(g.root): g.index for g in groups if g.placement != "host"}
    if len(original) != sum(g.placement != "host" for g in groups) or not set(eligible_groups) <= set(
        original.values()
    ):
        raise ValueError("source components have missing or ambiguous original eligibility")
    plan = prepare_sibling_components(module, effects=effects)
    covered = []
    for component in plan.components:
        from merlin.compiler.lower.device.sibling_source_route import _command_buffer

        _, group = _command_buffer(component.core, "source_identity_only")
        function = next(op for op in component.core.body.block.ops if isinstance(op, func.FuncOp))
        ops = tuple(function.body.block.ops)
        for index in group.contraction_ordinals:
            identity = eligible_source_identity(ops[index])
            if identity not in original:
                raise ValueError("component member differs from original admitted source")
            covered.append(original[identity])
    if not covered or len(set(covered)) != len(covered):
        raise ValueError("source has no disjoint checked sibling components")
    if eligible_source_regions is not None:
        function = next(
            op for op in module.body.block.ops if isinstance(op, func.FuncOp) and op.sym_name.data == "forward"
        )
        indices = {id(op): i for i, op in enumerate(function.body.block.ops)}
        allowed = {
            (indices.get(id(g.root)), g.index, eligible_source_identity(g.root)[0], "contraction")
            for g in groups
            if g.index in covered
        }
        if any(
            (r.get("source_op_index"), r.get("source_group"), r.get("region_id"), r.get("semantic_family"))
            not in allowed
            for r in eligible_source_regions
            if r.get("source_group") in covered
        ):
            raise ValueError("component cannot replace independently required supporting source work")
    if text(module, generic=True) != before:
        raise ValueError("component resolution changed original source")
    return SiblingSourceRoute(sha256(before.encode()).hexdigest(), effects, partial=True), {
        "status": "source_selected_not_qualified",
        "source_ir_sha256": sha256(before.encode()).hexdigest(),
        "original_source_groups": sorted(eligible_groups),
        "composed_source_groups": covered,
        "unmatched_source_groups": sorted(set(eligible_groups) - set(covered)),
        "original_source_denominator_unchanged": True,
        "component_count": len(plan.components),
    }


# ---- source_head_selection ---------------------------------------------------------------------
# Resolve a head request from original eligible source identities only.


def resolve_request(module, groups, eligible_groups, *, preferences, eligible_source_regions):
    from dataclasses import replace

    try:
        selected, record = resolve_heads(
            module, groups, eligible_groups, eligible_source_regions=eligible_source_regions
        )
        rows = preferences.get("shared_rhs_row_views", False)
        if type(rows) is not bool:
            raise ValueError("shared RHS row-view request requires a Boolean")
        if rows:
            selected = replace(selected, shared_rhs_row_views=True)
            record = {
                **record,
                "shared_rhs_row_views": "source_selected_not_qualified",
                "effect_or_placement_grant": False,
            }
        whole = preferences.get("whole_head_batches", False)
        if type(whole) is not bool:
            raise ValueError("whole head batch request requires a Boolean")
        if whole:
            selected = replace(
                selected,
                whole_batch=True,
                effects=replace(
                    selected.effects,
                    qualification_identity="owned host requires synchronous noescape/no-release "
                    "whole-batch calls; every original product, ranked interface, actual bridge and object "
                    "remain independently checked; no target issue or storage capability supplied",
                ),
            )
            record = {
                **record,
                "whole_head_batches": "source_selected_not_qualified",
                "kernel_invocations": record["head_families"],
                "matrix_products": record["matrix_products"],
                "effect_or_placement_grant": False,
            }
        if preferences["shared_input_siblings"]:
            try:
                return resolve_joint(
                    module, groups, eligible_groups, selected, record, eligible_source_regions=eligible_source_regions
                )
            except ValueError as exc:
                record = {**record, "sibling_composition_unavailable": str(exc)}
        deferred = ["shared_input_siblings"] if preferences["shared_input_siblings"] else []
        return selected, {
            **record,
            "deferred_requests": deferred,
            "composition_order": (
                "whole_heads_first_siblings_deferred" if whole else "independent_heads_first_siblings_fallback_only"
            ),
        }
    except ValueError as exc:
        if not preferences["shared_input_siblings"]:
            return None, {
                "status": "not_applicable",
                "kind": "independent_heads",
                "reason": str(exc),
                "baseline_retained": True,
            }
    return None


def resolve_joint(module, groups, eligible_groups, selected, record, *, eligible_source_regions):
    """Resolve only disjoint remaining source roots after exact head refolding."""
    from dataclasses import replace

    from xdsl.dialects import func

    from merlin.compiler.lower.device.head import project_heads
    from merlin.compiler.lower.device.sibling_components import prepare_sibling_components
    from merlin.compiler.lower.device.sibling_source_route import _command_buffer, _text
    from merlin.development.backend.qualification.provenance.groups import eligible_source_identity

    before = _text(module)
    plan = prepare_sibling_components(project_heads(module).module, effects=selected.effects)
    original = {eligible_source_identity(g.root): g.index for g in groups if g.placement != "host"}
    covered = []
    for component in plan.components:
        _, group = _command_buffer(component.core, "source_identity_only")
        function = next(op for op in component.core.body.block.ops if type(op) is func.FuncOp)
        operations = tuple(function.body.block.ops)
        for index in group.contraction_ordinals:
            identity = eligible_source_identity(operations[index])
            if identity not in original:
                raise ValueError("joint sibling member lost original source identity")
            covered.append(original[identity])
    heads = set(record["composed_source_groups"])
    if (
        not covered
        or len(set(covered)) != len(covered)
        or heads.intersection(covered)
        or not set(covered) <= set(eligible_groups)
    ):
        raise ValueError("joint source needs disjoint admitted sibling components")
    function = next(op for op in module.body.block.ops if type(op) is func.FuncOp and op.sym_name.data == "forward")
    positions = {id(op): i for i, op in enumerate(function.body.block.ops)}
    allowed = {
        (positions[id(g.root)], g.index, eligible_source_identity(g.root)[0], "contraction")
        for g in groups
        if g.index in covered
    }
    supplied = [
        (r.get("source_op_index"), r.get("source_group"), r.get("region_id"), r.get("semantic_family"))
        for r in eligible_source_regions
        if r.get("source_group") in covered
    ]
    if len(supplied) != len(set(supplied)) or set(supplied) != allowed:
        raise ValueError("joint siblings need complete original contraction-only evidence")
    if _text(module) != before:
        raise ValueError("joint selection changed original source")
    all_covered = sorted(heads | set(covered))
    return replace(selected, shared_input_siblings=True), {
        **record,
        "kind": "independent_heads_and_shared_input_siblings",
        "composed_source_groups": all_covered,
        "sibling_source_groups": covered,
        "unmatched_source_groups": sorted(set(eligible_groups) - set(all_covered)),
        "component_count": len(plan.components),
        "deferred_requests": [],
        "composition_order": "checked_heads_then_disjoint_siblings_then_ordinary",
    }


def resolve_heads(module, groups, eligible_groups, *, eligible_source_regions=None):
    from xdsl.dialects import func

    from merlin.compiler.lower.device.head import HeadBatchSourceRoute, project_heads
    from merlin.compiler.lower.device.sibling_source_route import _sha, _text
    from merlin.compiler.lower.host.borrowed_effects import BorrowedBufferEffects
    from merlin.development.backend.qualification.provenance.groups import eligible_source_identity

    before = _text(module)
    projection = project_heads(module)
    original = {eligible_source_identity(g.root): g.index for g in groups if g.placement != "host"}
    if len(original) != sum(g.placement != "host" for g in groups) or not set(eligible_groups) <= set(
        original.values()
    ):
        raise ValueError("head source has ambiguous original admitted identities")
    covered = []
    for family in projection.members:
        for member in family["members"]:
            identity = member["source_region_id"], tuple(member["source_node_ids"])
            if identity not in original:
                raise ValueError("head source member is not an independently admitted original group")
            covered.append(original[identity])
    if not covered or len(set(covered)) != len(covered):
        raise ValueError("head source omits or duplicates original products")
    if not set(covered) <= set(eligible_groups):
        raise ValueError("head source includes a product outside independently admitted groups")
    if (
        type(eligible_source_regions) is not list
        or not eligible_source_regions
        or any(type(row) is not dict for row in eligible_source_regions)
    ):
        raise ValueError("head source requires complete original eligible-operation evidence")
    function = next(op for op in module.body.block.ops if type(op) is func.FuncOp and op.sym_name.data == "forward")
    positions = {id(op): i for i, op in enumerate(function.body.block.ops)}
    allowed = {
        (positions.get(id(g.root)), g.index, eligible_source_identity(g.root)[0], "contraction")
        for g in groups
        if g.index in covered
    }
    supplied = [
        (r.get("source_op_index"), r.get("source_group"), r.get("region_id"), r.get("semantic_family"))
        for r in eligible_source_regions
        if r.get("source_group") in covered
    ]
    if any(row not in allowed for row in supplied):
        raise ValueError("raw head batching cannot replace an independently required supporting operation")
    if len(supplied) != len(set(supplied)) or set(supplied) != allowed:
        raise ValueError("head source omits or duplicates original eligible-operation evidence")
    if _text(module) != before:
        raise ValueError("head selection changed original source")
    selected = HeadBatchSourceRoute(
        _sha(before),
        BorrowedBufferEffects(
            "owned host requires synchronous noescape/no-release per-matrix calls; "
            "actual source, stock slice bridge, device work and final object remain independently checked",
            True,
            True,
            True,
        ),
    )
    return selected, {
        "status": "source_selected_not_qualified",
        "kind": "independent_heads",
        "source_ir_sha256": selected.source_ir_sha256,
        "composed_source_groups": covered,
        "original_source_groups": sorted(eligible_groups),
        "unmatched_source_groups": sorted(set(eligible_groups) - set(covered)),
        "original_source_denominator_unchanged": True,
        "head_families": len(projection.members),
        "batch_abi_calls": len(projection.members),
        "matrix_products": len(covered),
        "accelerator_execution_count_reduced": False,
    }
