"""Source-bound callable CPU implementation claims with explicit storage ceilings.

No native compilation occurs during discovery. Default zero workspace retains
its native obligation. Opt-in workspace derives exact original tensor values;
the caller ceiling is not an estimate or extra charge beyond source lifetimes.
Actual native ABI/effect and arena admission remain required before linking.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from merlin.compiler.backend.sourceproposal import SourceProposal, proposal_boundary, source_numerical_identity
from merlin.xdsl_dialects.lowering.global_plan import CycleInterval, RegionAlternative


def propose(source, *, resource, memory_domain, cuts=(), budget, preparation=None, workspace_limit=0):
    if preparation is not None:
        preparation.revalidate()
    source.revalidate()
    count = len(source.operations)
    if not count:
        raise ValueError("callable CPU discovery requires a nonempty source operation graph")
    if count > budget.max_source_operations:
        raise ValueError("CPU proposal discovery exceeded the source operation budget")
    for value in source.program.buffers.values():
        elements = 1
        for extent in value.shape:
            if type(extent) is not int or extent < 0:
                raise ValueError("callable CPU proposals require static storage extents")
            elements *= extent
        if elements > budget.max_tensor_elements:
            raise ValueError("CPU proposal discovery exceeded the tensor element budget")
    boundaries = sorted({0, count, *cuts})
    if any(type(n) is not int or not 0 <= n <= count for n in boundaries):
        raise ValueError("CPU proposal cuts must bind actual source operation positions")
    intervals = [(a, b, "piece") for a, b in zip(boundaries, boundaries[1:]) if (a, b) != (0, count)]
    intervals.append((0, count, "whole"))
    intervals = [(start, end, kind) for start, end, kind in intervals
                 if end - start <= budget.max_region_operations]
    if len(intervals) > budget.max_candidates:
        raise ValueError("CPU complement proposals exceed the selected candidate budget")
    result = []
    for start, end, kind in intervals:
        nodes = tuple(range(start, end))
        inputs, outputs = proposal_boundary(source, nodes)
        metadata = (("native_abi_admission", "required_before_control_or_link"),
                    ("tensor_workspace_limit", "0"), ("workspace_status", "conditional_not_observed"))
        if workspace_limit:
            from merlin.compiler.ir.boundary import live_values
            from merlin.compiler.lower.llvm.workspace import plan

            operations = tuple(source.operations[n] for n in nodes)
            _, live_outputs = live_values(operations)
            workspace, _ = plan(operations, live_outputs)
            metadata = (("native_abi_admission", "required_before_control_or_link"),
                        ("tensor_workspace_limit", str(workspace_limit)),
                        ("workspace_status", "source_backed_native_admission_pending"),
                        ("source_workspace", json.dumps(workspace.record() if workspace is not None else None,
                                                       sort_keys=True, separators=(",", ":"))))
            metadata = tuple(sorted(metadata))
        alternative = RegionAlternative(
            id=f"merlin.scalar/{0 if kind == 'piece' else 1}-{kind}/{start:06d}-{end:06d}",
            nodes=nodes, implementation="scalar.callable.v1", placement=resource,
            cycles=CycleInterval.unknown("no measured full-region CPU cost; storage ceiling is not an estimate"),
            inputs=inputs, outputs=outputs, scratch_bytes=0, metadata=metadata,
        )
        result.append(SourceProposal(
            alternative, "merlin.scalar", resource, memory_domain, "logical_tensor_v1",
            source.stamp.source_sha256, tuple(source.operation_ordinals[n] for n in nodes),
            source_numerical_identity(source, nodes),
        ))
    source.revalidate()
    return tuple(result)


@dataclass(frozen=True)
class WorkspaceProposals:
    limit: int

    def propose(self, source, *, resource, memory_domain, cuts=(), budget, preparation=None):
        if type(self.limit) is not int or self.limit <= 0:
            raise ValueError("CPU workspace proposals require an explicit positive caller ceiling")
        limit = self.limit
        result = propose(source, resource=resource, memory_domain=memory_domain, cuts=cuts, budget=budget,
                         preparation=preparation, workspace_limit=limit)
        if self.limit != limit:
            raise ValueError("original CPU workspace proposal ceiling changed")
        return result
