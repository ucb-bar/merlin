"""Bounded mechanism questions from source domains and normalized storage effects.

Packing arithmetic describes conditional alternatives, not a legal rewrite or
the schedule a binary executed. Repeated reads retain the selected projector's
backing/alias authority; no opcode spelling supplies tensor or effect semantics.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import prod


@dataclass(frozen=True)
class ReductionLanes:
    """A selected adapter's physical reduction width and its exact evidence pointer."""

    width: int
    evidence: str

    def __post_init__(self):
        if type(self.width) is not int or self.width < 1 or not self.evidence:
            raise ValueError("reduction lanes require a positive width and selected evidence")


def packing_profiles(reduction, *, lanes=None):
    """Price contiguous suffix groupings without asserting their index/numeric legality."""
    if (not reduction or len(reduction) > 32
            or any(type(n) is not int or n < 1 for n in reduction)
            or sum(n.bit_length() for n in reduction) > 256):
        raise ValueError("packing analysis requires bounded positive reduction extents")
    useful = prod(reduction)
    profiles = []
    for suffix in range(1, len(reduction) + 1):
        groups, elements = prod(reduction[:-suffix]), prod(reduction[-suffix:])
        pieces = groups * (-(-elements // lanes.width)) if lanes else None
        physical = pieces * lanes.width if lanes else None
        profiles.append(dict(
            collapsed_suffix_axes=suffix, independent_groups=groups, elements_per_group=elements,
            pieces=pieces, physical_extent=physical,
            unused_lanes=None if physical is None else physical - useful,
            status="conditional_arithmetic" if lanes else "unknown_physical_width",
            legality="unknown", emitted_application="unknown",
        ))
    return dict(useful_reduction_extent=useful, reduction_extents=list(reduction),
                lanes=None if lanes is None else dict(width=lanes.width, evidence=lanes.evidence),
                alternatives=profiles,
                required_proof=["indexing and boundary padding under the new grouping",
                                "accumulation, rounding, saturation and overflow order",
                                "selected consumer and actual emitted reduction partition"],
                performance="unknown")


def source_mechanisms(module, *, evidence_id, lanes=None, max_operations=4096, max_witnesses=16):
    """Read actual yielded MACs and exact tensor SSA sharing, preserving unknown domains."""
    from xdsl.dialects.builtin import TensorType

    from merlin.perf.model_macs import observe_model_macs

    if max_operations < 1 or max_witnesses < 1:
        raise ValueError("mechanism analysis requires finite positive budgets")
    positions = {}
    for index, operation in enumerate(module.walk()):
        if index >= max_operations:
            return dict(status="unknown", reason="source_mechanism_operation_budget_exhausted",
                        findings=[], optimization_application="unknown")
        positions[operation] = index
    domains, uses, value_names = [], defaultdict(list), {}
    missing = []
    for operation, shape in observe_model_macs(module):
        if operation is None or shape.status != "derived":
            missing.append(dict(operation=None if operation is None else positions[operation], reason=shape.reason))
            continue
        witness = f"{evidence_id}#operation:{positions[operation]}"
        domains.append(dict(operation=positions[operation], source_op_index=shape.source_op_index,
                            witness=witness, dtypes=list(shape.dtypes), parallel_extents=list(shape.parallel),
                            packing=packing_profiles(shape.reduction, lanes=lanes)))
        # Tensor SSA is immutable; equal shapes or matching addresses are not SSA equality.
        # Output/init operands are deliberately excluded from shared input evidence.
        for operand_index, value in enumerate(tuple(operation.inputs)):
            if not isinstance(value.type, TensorType):
                continue
            value_names.setdefault(value, f"tensor:{len(value_names)}")
            uses[value].append(dict(operation=positions[operation], operand=operand_index, witness=witness))
    shared = [dict(value=value_names[value], shape=list(value.type.get_shape()),
                   dtype=str(value.type.get_element_type()), consumers=members[:max_witnesses],
                   consumers_total=len(members), consumers_omitted=max(0, len(members) - max_witnesses),
                   status="observed_immutable_ssa_sharing", storage_reuse="unknown",
                   required_proof=["source-to-selected-consumer binding", "load and preparation lifetime",
                                   "capacity and non-overwrite across all consumers"])
              for value, members in uses.items() if len({row["operation"] for row in members}) > 1]
    from merlin.kernels.source_product_views import source_product_views

    return dict(status="partial" if missing else "observed", origin="parsed_source_ir",
                reductions=domains[:max_witnesses], reductions_total=len(domains),
                reductions_omitted=max(0, len(domains) - max_witnesses),
                shared_inputs=shared[:max_witnesses], shared_inputs_total=len(shared),
                shared_inputs_omitted=max(0, len(shared) - max_witnesses),
                unknowns=missing[:max_witnesses], unknowns_total=len(missing),
                product_views=source_product_views(module, evidence_id=evidence_id, max_operations=max_operations),
                optimization_application="unknown", numerical_equivalence="unknown", performance="unknown")


def repeated_read_mechanisms(model, *, max_comparisons=100_000, max_witnesses=16):
    """Find invariant reads in observed affine recurrences without expanding them.

    All accesses to the same canonical backing participate in invalidation. A
    repeated external read is not evidence that the destination remains live.
    Completion, capacity, hidden effects and cross-buffer aliasing stay separate.
    """
    if max_comparisons < 1 or max_witnesses < 1:
        raise ValueError("reuse analysis requires finite positive budgets")
    stages = {stage.key: stage for stage in model.stages}
    findings, unknowns, comparisons = [], [], 0
    if any(not stage.accesses for stage in model.stages):
        unknowns.append("normalized_effect_roster_incomplete")
    for loop in model.loops:
        if (loop.status != "inferred" or loop.extent < 2 or any(key not in stages for key in loop.body)):
            unknowns.append("unsupported_or_unbound_recurrence")
            continue
        body = [stages[key] for key in loop.body]
        accesses = [(i, j, access) for i, stage in enumerate(body) for j, access in enumerate(stage.accesses)]
        for i, j, access in accesses:
            if access.mode != "read":
                continue
            if type(access.offset) is not int or access.extent is None or f"{i}:{j}" not in loop.strides:
                unknowns.append("read_extent_or_affine_recurrence_unresolved")
                continue
            if loop.strides[f"{i}:{j}"] != 0:
                continue
            conflict = False
            unresolved = not model.complete or model.truncated
            for wi, wj, write in accesses:
                comparisons += 1
                if comparisons > max_comparisons:
                    return dict(status="unknown", reason="reuse_comparison_budget_exhausted", findings=[],
                                optimization_application="unknown")
                if (write.space, write.buffer) != (access.space, access.buffer):
                    continue
                if write.mode not in ("write", "read_write"):
                    continue
                if write.unit != access.unit or type(write.offset) is not int or write.extent is None:
                    unresolved = True
                    continue
                stride = loop.strides.get(f"{wi}:{wj}")
                if type(stride) is not int:
                    unresolved = True
                    continue
                # An envelope may overapproximate gaps between writes. Treat
                # overlap as a refusal, never as proof the source changed.
                end = write.offset + (loop.extent - 1) * stride
                lo, hi = min(write.offset, end), max(write.offset, end) + write.extent
                conflict |= max(lo, access.offset) < min(hi, access.offset + access.extent)
            status = ("unknown" if unresolved else
                      "intervening_write_possible" if conflict else "observed_invariant_read")
            findings.append(dict(
                recurrence=loop.key, stage=body[i].key, access=j,
                backing=dict(space=access.space, buffer=access.buffer, unit=access.unit),
                offset=access.offset, extent=access.extent, repetitions=loop.extent,
                total_read_units=access.extent * loop.extent,
                repeated_read_units=access.extent * (loop.extent - 1), status=status,
                witness=f"schedule:{model.identity}#stages:{body[i].key}:access:{j}",
                scope="selected normalized backing/effects; observed finite recurrence",
                reuse_legality="unknown", optimization_application="unknown",
                required_proof=["complete effects and canonical alias mapping", "destination storage lifetime",
                                "capacity with simultaneous live values", "completion and dependency ordering"],
            ))
    return dict(status="partial" if unknowns or not model.complete else "observed",
                findings=findings[:max_witnesses], findings_total=len(findings),
                findings_omitted=max(0, len(findings) - max_witnesses), unknowns=unknowns[:max_witnesses],
                optimization_application="unknown", performance="unknown")
