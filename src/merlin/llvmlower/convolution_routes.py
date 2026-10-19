"""Select bounded host window packing before ordinary dense discovery.

The budget counts distinct activation-panel payloads, not host RAM or device
memory. The deterministic policy meets that storage bound with the largest
legal panel found within the search budget. Timing and unknown padding costs
remain unknown; qualified device gathers are not inferred from dense support.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from math import prod
from pathlib import Path

from merlin.transforms.certificates import ir_digest
from merlin.transforms.pipeline_state import PassSpec, PipelineContext, PipelineSession, validate_pass_order


def implementation_paths():
    from . import im2col_pack

    return (Path(__file__), Path(im2col_pack.__file__))


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _padding_cost(value, element_bytes):
    """Count a declared zero-fill/insert; unknown producers remain opaque."""
    from xdsl.dialects import arith, builtin, tensor
    from xdsl.dialects.linalg.ops import FillOp

    unknown = {"status": "unknown", "transformation": "unchanged"}
    insert = value.owner
    if type(insert) is not tensor.InsertSliceOp or insert.attributes:
        return unknown
    fill = insert.dest.owner
    if type(fill) is not FillOp or fill.attributes or len(fill.inputs) != 1:
        return unknown
    zero = fill.inputs[0].owner
    if type(zero) is not arith.ConstantOp or type(zero.value) is not builtin.IntegerAttr or zero.value.value.data != 0:
        return unknown
    source_shape, target_shape = insert.source.type.get_shape(), value.type.get_shape()
    if (
        tuple(insert.static_sizes.get_values()) != tuple(source_shape)
        or tuple(insert.static_strides.get_values()) != (1,) * len(source_shape)
        or len(source_shape) != len(target_shape)
        or any(
            offset < 0 or offset + size > extent
            for offset, size, extent in zip(
                insert.static_offsets.get_values(),
                source_shape,
                target_shape,
                strict=True,
            )
        )
    ):
        return unknown
    return {
        "status": "source_payload_only",
        "transformation": "unchanged",
        "zero_fill_write_bytes": prod(target_shape) * element_bytes,
        "insert_read_bytes": prod(source_shape) * element_bytes,
        "insert_write_bytes": prod(source_shape) * element_bytes,
    }


def _legal_panel(match, width, selection):
    from merlin.compile.scheduling.dense_contracts import DenseSchedulingUnavailable
    from merlin.transforms.dense_planning import dense_blocks

    dense = selection.selected_contracts.dense
    for columns in (width, match.ow % width):
        if not columns:
            continue
        problem = dense.problem(match.f, match.k, columns)
        try:
            resources, _ = dense.bind(
                problem,
                selection.storage,
                selection.adapter,
                dtypes=dense.dtypes,
                hardware=selection.selected_contracts.hardware_identity,
                policy=selection.selected_contracts.numerical_policy,
                resources=selection.resources,
                reservations=selection.reservations,
            )
        except DenseSchedulingUnavailable:
            return False
        mt, _, nt = problem.tiles
        if min(mt, resources.max_m_blocks) * min(nt, resources.max_n_blocks) > resources.candidate_budget:
            return False
        blocks = dense_blocks(
            problem,
            resources,
            k_limit=lambda m, n: selection.storage.k_limit(problem, resources, m, n),
        )
        if not any(selection.storage.admits(problem, block, resources) for block in blocks):
            return False
    return True


def _select_width(match, selection):
    element_bytes = selection.selected_contracts.dense.problem(match.f, match.k, match.m).rhs_bytes
    maximum = min(match.ow, selection.max_convolution_packing_bytes // (match.k * element_bytes))
    for attempt, width in enumerate(range(maximum, 0, -1)):
        if attempt >= selection.max_convolution_candidates:
            return None, "candidate_budget"
        payload = match.k * (width + match.ow % width) * element_bytes
        if payload <= selection.max_convolution_packing_bytes and _legal_panel(match, width, selection):
            return width, None
    return None, "no_legal_panel_within_packing_budget"


def _signature(operation):
    return (
        operation.name,
        dict(operation.attributes),
        dict(operation.properties),
        tuple(operation.operands),
        tuple((value, value.type) for value in operation.results),
        tuple(operation.regions),
        tuple(operation.successors),
    )


class _GrowthBudgetExceeded(ValueError):
    pass


def normalize_convolution_source(module, selection, validate):
    """Return a checked private source or its unchanged legal full route."""
    from merlin.compile.scheduling.library_sources import shared_library_sources
    from merlin.compile.scheduling.selected_contracts import _parse

    from .im2col_pack import (
        bounded_source_identity,
        match_bounded_im2col,
        rewrite_bounded_match,
        verify_bounded_match,
    )
    from .source_dense import SourceDenseLowering

    if type(selection) is not SourceDenseLowering:
        raise ValueError("convolution normalization requires an original typed dense selection")
    selection.__post_init__()
    validate()
    source_identity = ir_digest(module)
    report = {
        "schema": "merlin.convolution_routes.v1",
        "source_sha256": source_identity,
        "selected_sha256": source_identity,
        "rewritten": 0,
        "routes": [],
        "declined": [],
        "policy": "largest resource-legal panel within explicit summed activation packing payload budget",
        "timing": "UNKNOWN; no profitability claim",
        "pipeline": [],
    }
    if selection.convolution_route == "full" or selection.max_convolution_packing_bytes is None:
        return module, report
    contracts = selection.selected_contracts
    if (
        contracts.dense is None
        or selection.borrowed_effects is None
        or selection.storage.accumulator is None
        or (
            not selection.private_ownership
            or not selection.external_buffers_disjoint
            or not selection.adapter.qualification_identity
            or not selection.adapter.synchronous_completion
            or not selection.adapter.exact_accumulator_continuation
        )
    ):
        report["declined"].append("qualified synchronous noescape dense ownership is unavailable")
        return module, report
    if any(store.capacity_bytes is None for store in contracts.dense.stores):
        report["declined"].append("dense physical capacity is unknown")
        return module, report
    selection.borrowed_effects.__post_init__()
    if _parse(contracts.facts_bytes, contracts.software_bytes, contracts.target)[4] != contracts.dense:
        raise ValueError("convolution resources differ from original selected declarations; reselect")

    def context():
        return PipelineContext(
            hardware=contracts.hardware_identity,
            numerical_policy=contracts.numerical_policy,
            resources=_digest(
                {
                    "storage": asdict(selection.storage),
                    "resources": None if selection.resources is None else asdict(selection.resources),
                    "reservations": [asdict(value) for value in selection.reservations],
                    "packing_bytes": selection.max_convolution_packing_bytes,
                    "candidates": selection.max_convolution_candidates,
                    "added_operations": selection.max_convolution_added_operations,
                    "route": selection.convolution_route,
                    "source_elements": selection.max_source_elements,
                    "islands": selection.max_islands,
                    "static_commands": selection.static_budget,
                    "outer_order": selection.outer_order,
                }
            ),
            execution=_digest(
                {
                    "effects": asdict(selection.borrowed_effects),
                    "dense": asdict(contracts.dense),
                    "adapter": {
                        name: getattr(selection.adapter, name)
                        for name in (
                            "qualification_identity",
                            "configuration_identity",
                            "numerical_policy_identity",
                            "synchronous_completion",
                            "exact_accumulator_continuation",
                            "index_bits",
                        )
                    },
                    "ownership": selection.private_ownership,
                    "external_buffers_disjoint": selection.external_buffers_disjoint,
                }
            ),
            library=_digest(
                {
                    **shared_library_sources(),
                    **{
                        "merlin.llvmlower." + path.stem: hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in implementation_paths()
                    },
                }
            ),
        )

    original_context = context()
    candidate = module.clone()
    session = PipelineSession(candidate, context=original_context, ir_level="tensor")
    specs = (
        PassSpec("convolution-source", 0, produces=("windows",), mutates_ir=False, ir_level="tensor"),
        PassSpec(
            "convolution-routes", 1, requires=("windows",), produces=("routes",), mutates_ir=False, ir_level="tensor"
        ),
        PassSpec(
            "convolution-panel-packing",
            2,
            requires=("routes",),
            produces=("realized",),
            invalidates=("*",),
            ir_level="tensor",
        ),
        PassSpec(
            "convolution-conformance",
            2,
            requires=("realized",),
            produces=("checked",),
            mutates_ir=False,
            ir_level="tensor",
        ),
    )
    validate_pass_order(specs)

    def run(spec, action):
        validate()
        contracts.revalidate()
        if context() != original_context:
            raise ValueError("convolution resource or numerical context changed; replan")
        session.run(spec, action)
        validate()
        contracts.revalidate()
        if context() != original_context:
            raise ValueError("convolution resource or numerical context changed; replan")

    def windows():
        found = []
        for operation in candidate.walk():
            if operation.name != "linalg.generic":
                continue
            match = match_bounded_im2col(
                operation, selected_contracts=contracts, max_elements=selection.max_source_elements
            )
            if match is not None:
                if len(found) >= selection.max_islands:
                    raise ValueError("convolution discovery exceeds its selected island budget")
                found.append(match)
        return {"windows": tuple(found)}

    def routes():
        plans = []
        for match in session.analysis("windows"):
            element_bytes = contracts.dense.problem(match.f, match.k, match.m).rhs_bytes
            full = match.k * match.m * element_bytes
            if selection.convolution_route == "auto" and full <= selection.max_convolution_packing_bytes:
                continue
            width, reason = _select_width(match, selection)
            if width is None:
                report["declined"].append(reason)
                continue
            plans.append((match, width, bounded_source_identity(match)))
        return {"routes": tuple(plans)}

    def realize():
        initial_count = sum(1 for _ in candidate.walk())
        realized = []
        for planned, width, _ in session.analysis("routes"):
            # Re-read later windows after any earlier checked SSA substitutions.
            match = match_bounded_im2col(
                planned.contraction,
                selected_contracts=contracts,
                max_elements=selection.max_source_elements,
            )
            if match is None or _select_width(match, selection)[0] != width:
                raise ValueError("convolution source changed during construction; replan")
            identity = bounded_source_identity(match)
            padding = _padding_cost(
                match.gather.inputs[0], contracts.dense.problem(match.f, match.k, match.m).rhs_bytes
            )
            before = set(candidate.walk())
            removable = {
                operation
                for root in (match.gather, match.collapse, match.expand, match.contraction)
                for operation in root.walk()
            }
            empty = match.gather.outputs[0].owner
            if getattr(empty, "name", None) == "tensor.empty" and len(tuple(empty.results[0].uses)) == 1:
                removable.add(empty)
            signatures = {operation: _signature(operation) for operation in before - removable}
            blocks = {
                block: (
                    tuple((argument, argument.type) for argument in block.args),
                    tuple(operation for operation in block.ops if operation not in removable),
                )
                for operation in before - removable
                for region in operation.regions
                for block in region.blocks
            }
            parent = match.contraction.parent_block()
            original_order = tuple(parent.ops)
            start, end = original_order.index(match.gather), original_order.index(match.contraction)
            output = match.contraction.results[0]
            bindings = (match.gather.inputs[0], match.contraction.inputs[0], match.contraction.outputs[0])
            before_identity = ir_digest(candidate)
            result = rewrite_bounded_match(match, width)
            if (result.source.input, result.source.weight, result.source.initializer) != bindings:
                raise ValueError("convolution construction changes its original SSA bindings")
            after = set(candidate.walk())
            expected_new = {operation for root in result.operations for operation in root.walk()}
            if after - before != expected_new or before - after - removable:
                raise ValueError("convolution construction has unaccounted operation membership")
            for block, (arguments, order) in blocks.items():
                if arguments != tuple((argument, argument.type) for argument in block.args) or order != tuple(
                    operation for operation in block.ops if operation in signatures
                ):
                    raise ValueError("convolution construction changes original block order or arguments")
            current_order = tuple(parent.ops)
            new_positions = [current_order.index(operation) for operation in result.operations]
            if any(
                current_order.index(operation) >= min(new_positions)
                for operation in original_order[:start]
                if operation in signatures
            ) or any(
                current_order.index(operation) <= max(new_positions)
                for operation in original_order[end + 1 :]
                if operation in signatures
            ):
                raise ValueError("convolution construction moves packing across the original source boundary")
            for operation, signature in signatures.items():
                current = _signature(operation)
                expected = (
                    *signature[:3],
                    tuple(result.result if value is output else value for value in signature[3]),
                    *signature[4:],
                )
                if current != expected:
                    raise ValueError("convolution construction changes an operation outside its checked region")
            realized.append((result, identity, bindings, padding, before_identity, ir_digest(candidate)))
            if sum(1 for _ in candidate.walk()) - initial_count > selection.max_convolution_added_operations:
                raise _GrowthBudgetExceeded("convolution construction exceeds its source operation budget")
        return {"realized": tuple(realized)}

    def check():
        rows = []
        for result, identity, bindings, padding, before_identity, after_identity in session.analysis("realized"):
            checked = verify_bounded_match(result, expected_source=identity, expected_bindings=bindings)
            width_bytes = contracts.dense.problem(result.source.f, result.source.k, result.source.m).rhs_bytes
            actual_bytes = sum(
                prod(arm.packing_empty.results[0].type.get_shape())
                * arm.packing_empty.results[0].type.element_type.width.data
                // 8
                for arm in result.arms
            )
            if actual_bytes > selection.max_convolution_packing_bytes:
                raise ValueError("realized convolution packing payload exceeds the selected budget")
            rows.append(
                {
                    "selected_width": result.width,
                    "source_identity": identity,
                    "before_ir_sha256": before_identity,
                    "after_ir_sha256": after_identity,
                    "conformance": checked,
                    "full_packing_payload_bytes": result.source.k * result.source.m * width_bytes,
                    "realized_distinct_panel_payload_bytes": actual_bytes,
                    "padding": padding,
                    "output_and_abi_scratch": "separate; not covered by activation packing budget",
                    "introduced_zero_fill_payload_bytes": result.source.f
                    * result.source.m
                    * contracts.dense.problem(result.source.f, result.source.k, result.source.m).accumulator_bytes,
                }
            )
        return {"checked": rows}

    try:
        for spec, action in zip(specs, (windows, routes, realize, check), strict=True):
            run(spec, action)
    except _GrowthBudgetExceeded as error:
        validate()
        report["declined"].append(str(error))
        return module, report
    rows = session.analysis("checked")
    report.update(
        rewritten=len(rows),
        routes=rows,
        pipeline=[asdict(record) for record in session.records],
        selected_sha256=ir_digest(candidate) if rows else source_identity,
    )
    return (candidate if rows else module), report
