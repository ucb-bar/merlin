"""Diagnostic placement projection over native source groups; grants no admission."""

from collections import Counter
import math
import re

from merlin.common import mlir_query as mq
from merlin.xdsl_dialects.lowering import compute_groups as CG
from .private_source_member_hardware import observe


def _bytes(values):
    total = 0
    for value in values:
        typ = value.type
        if not hasattr(typ, "get_shape"):
            return None
        shape = typ.get_shape()
        match = re.fullmatch(r"(?:i|ui|si|f)(\d+)|bf(16)", str(typ.element_type))
        if not match or any(dim < 0 for dim in shape):
            return None
        bits = int(match.group(1) or match.group(2))
        total += (math.prod(shape) * bits + 7) // 8
    return total


def describe(groups, capability, *, demanded, unknown_host, missing_direct):
    """Explain the existing demand decision without reclassifying or relaxing it."""
    rows = []
    for group in groups:
        members = {id(op) for op in group.members}
        inputs, outputs = {}, {}
        identities, observations = [], []
        for op, stage in zip(group.members, group.stages, strict=True):
            attrs = op.attributes
            region = attrs.get("prov.region_id")
            nodes = attrs.get("prov.source_node_ids")
            identities.append({"operation": mq.op_name(op),
                               "region_id": getattr(region, "data", None),
                               "source_node_ids": [getattr(node, "data", None)
                                                   for node in getattr(nodes, "data", ())]})
            try:
                observations.append(observe(op, stage, capability))
            except ValueError:
                # This projection cannot introduce an admission rule for a
                # source grouping form the independent member observer lacks.
                observations.append({"status": "unknown", "refusal": "typed_member_observation_refused",
                                     "basis": "diagnostic_only"})
            for value in op.operands:
                if id(value.owner) not in members:
                    inputs[id(value)] = value
            for value in op.results:
                if any(id(use.operation) not in members for use in value.uses):
                    outputs[id(value)] = value
        required = group.index in demanded
        unresolved = group.index in unknown_host
        admitted_member = any(row.get("status") == "admitted" for row in observations)
        reason = ("required_device_region" if required else
                  "eligible_member_without_region_lowering" if unresolved and admitted_member else
                  "unresolved_host_placement" if unresolved else "source_checked_host_placement")
        rows.append({"source_group": group.index, "source_identities": identities,
                     "stages": list(group.stages), "placement": group.placement,
                     "required_device": required, "reason": reason,
                     "compiler_refusal": group.refusal, "compiler_gap": group.gap,
                     "member_hardware_observations": observations,
                     "mapping_scope": "composed_region" if len(group.members) > 1 else "standalone_region",
                     "standalone_equivalence": "not_proved_by_composed_group_admission",
                     "logical_input_bytes": _bytes(inputs.values()),
                     "logical_output_bytes": _bytes(outputs.values()),
                     "transferred_bytes": None, "host_time_cycles": None})
    return {"schema": "merlin.source_placement_feedback.v1", "authority": "diagnostic_only",
            "counts": dict(Counter(row["reason"] for row in rows)),
            "source_groups": len(groups), "missing_direct_operations": missing_direct,
            "byte_scope": "distinct static SSA boundary payloads per group; not physical traffic or cross-group unique bytes",
            "timing_scope": "not_measured", "rows": rows}
