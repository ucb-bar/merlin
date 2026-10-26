"""Shape-stratified source coverage and direct SSA boundaries for caller groups.

This is a planning census, not placement, equivalence, alias analysis, a runtime
allocator, or performance qualification. In particular a representative covers
a descriptive shape/profile only; each original numerical program still needs
its own source proof. The caller supplies the grouping/admission decision.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict

from merlin.common import mlir_query as mq


def _typed(value):
    shape, dtype = mq.type_shape_dtype(value.type)
    shape = None if shape is None else list(shape)
    bits = getattr(getattr(value.type, "element_type", value.type), "bitwidth", None)
    if not isinstance(bits, int):
        bits = {
            "int8": 8,
            "int16": 16,
            "int32": 32,
            "int64": 64,
            "fp16": 16,
            "fp32": 32,
            "fp64": 64,
            "bf16": 16,
            "i8": 8,
            "i16": 16,
            "i32": 32,
            "i64": 64,
            "f16": 16,
            "f32": 32,
            "f64": 64,
        }.get(dtype)
    size = None
    if shape is not None and all(isinstance(n, int) and n >= 0 for n in shape) and bits is not None:
        size = (math.prod(shape) * bits + 7) // 8
    return {"shape": shape, "dtype": dtype, "logical_bytes": size}


def _strings(op, key):
    attr = op.attributes.get(key)
    if attr is None:
        return []
    value = getattr(attr, "data", None)
    if isinstance(value, str):
        return [value]
    return [a.data for a in attr if isinstance(getattr(a, "data", None), str)] if hasattr(attr, "__iter__") else []


def _sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def plan_model_coverage(module, groups):
    """Describe current groups and exact direct cross-group SSA uses, without payloads.

    Group indices must be distinct and supplied in source sink order. Members must
    belong to this module, cannot overlap or contain another selected member,
    and roots must be members. Direct edges can cross arbitrary host groups;
    no view/copy is elided. Unowned producers/consumers remain explicit, so an
    incomplete grouping cannot turn into a claim of complete model coverage.
    Lexical first/last uses are not asynchronous completion or allocation bounds.
    """
    from xdsl.ir import Operation

    groups = tuple(groups)
    operations = tuple(module.walk())
    position = {id(op): i for i, op in enumerate(operations)}
    owner = {}
    index_order = {}
    rows = []
    for ordinal, group in enumerate(groups):
        if not isinstance(group.index, int) or group.index in index_order:
            raise ValueError("group indices must be distinct integers")
        index_order[group.index] = ordinal
        members = tuple(group.members)
        if not members or any(id(op) not in position for op in members):
            raise ValueError("group must contain current source operations")
        if tuple(position[id(op)] for op in members) != tuple(sorted(position[id(op)] for op in members)):
            raise ValueError("group members must retain source order")
        if group.root is not None and all(group.root is not op for op in members):
            raise ValueError("group root must be a current member")
        for op in members:
            if id(op) in owner:
                raise ValueError("group members overlap")
            owner[id(op)] = group.index
        nested = Counter(mq.op_name(child) for op in members for child in op.walk())
        signature = {
            "placement": group.placement,
            "stages": list(group.stages),
            "members": [
                {
                    "operation": mq.op_name(op),
                    "operands": [_typed(v) for v in op.operands],
                    "results": [_typed(v) for v in op.results],
                }
                for op in members
            ],
            "scalar_operation_counts": dict(sorted(nested.items())),
        }
        provenance = {}
        for key, name in (
            ("prov.region_id", "region_ids"),
            ("prov.source_node_ids", "source_node_ids"),
            ("prov.fqn", "fqns"),
            ("prov.op", "source_operations"),
        ):
            provenance[name] = sorted({s for op in members for s in _strings(op, key)})
        rows.append(
            {
                "index": group.index,
                "structural_signature": _sha(signature),
                **signature,
                **provenance,
                "root": None
                if group.root is None
                else {
                    "operation": mq.op_name(group.root),
                    "operands": [_typed(v) for v in group.root.operands],
                    "results": [_typed(v) for v in group.root.results],
                },
                "macs": group.macs,
                "elements": group.elements,
                "refusal": group.refusal,
                "reason": group.reason,
            }
        )

    def containing(op):
        while isinstance(op, Operation):
            if id(op) in owner:
                return owner[id(op)]
            op = op.parent_op()
        return None

    for op in operations:
        if id(op) in owner and containing(op.parent_op()) is not None:
            raise ValueError("selected member contains another selected member")
    sinks = [max(position[id(op)] for op in g.members) for g in groups]
    if sinks != sorted(sinks):
        raise ValueError("groups must retain source sink order")

    # Number SSA definitions by original walk/block order, never by mutable names.
    values = []
    for op in operations:
        for region in op.regions:
            for block in region.blocks:
                values.extend(block.args)
        values.extend(op.results)
    boundaries = []
    for value_index, value in enumerate(values):
        definition = value.owner if isinstance(value.owner, Operation) else value.owner.parent_op()
        producer = containing(definition)
        consumers = set()
        unowned = set()
        returned = False
        uses = []
        for use in value.uses:
            uses.append(position[id(use.operation)])
            group = containing(use.operation)
            if group is not None and group != producer:
                consumers.add(group)
            elif group is None:
                name = mq.op_name(use.operation)
                unowned.add(name)
                returned |= name == "func.return"
        if not consumers and not (producer is not None and unowned):
            continue
        ordered = sorted(consumers, key=index_order.__getitem__)
        boundaries.append(
            {
                "value": value_index,
                "producer_group": producer,
                "consumer_groups": ordered,
                "unowned_consumer_operations": sorted(unowned),
                "external_output": returned,
                **_typed(value),
                "definition_position": position.get(id(definition)),
                "last_use_position": max(uses),
                "lexical_first_group": producer if producer is not None else ordered[0],
                "lexical_last_group": ordered[-1] if ordered else producer,
                "alias_ownership": "not_established",
                "completion": "not_established",
            }
        )
    profile_members = defaultdict(list)
    for row in rows:
        profile_members[row["structural_signature"]].append(row["index"])
    profiles = [
        {"structural_signature": key, "representative_index": indices[0], "members": indices, "count": len(indices)}
        for key, indices in profile_members.items()
    ]
    return {
        "schema": "merlin.model_shape_coverage.v1",
        "source_ir_sha256": hashlib.sha256(str(module).encode()).hexdigest(),
        "scope": (
            "descriptive source profiles and direct SSA uses; "
            "no numerical substitution, allocation, admission or timing proof"
        ),
        "groups": rows,
        "profiles": profiles,
        "boundaries": boundaries,
        "placement_counts": dict(Counter(row["placement"] for row in rows)),
        "stage_counts": dict(Counter(stage for row in rows for stage in row["stages"])),
        "unowned_operation_counts": dict(Counter(mq.op_name(op) for op in operations if containing(op) is None)),
        "performance_qualification": False,
    }
