"""Complete original source demand, independent of the compiler's chosen route."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy

from merlin.common import mlir_query as mq
from merlin.common.digest import is_sha256
from merlin.verify.lean_contract import document_digest
from merlin.xdsl_dialects.lowering import compute_groups as CG
from merlin_experiments.phase1.feedback.private_group_provenance import eligible_source_identity
from merlin_experiments.phase1.feedback.private_source_member_hardware import observe as observe_member

SCHEMA = "merlin.private_source_device_demand.v1"
_EXCLUSIONS = frozenset({"undeclared_family", "input_dtype", "weight_dtype", "operand_pair", "result_dtype",
                         "rank", "batch", "layout", "form"})


def _metadata_only(member):
    """Only known shape/view operations with unchanged element type are glue."""
    name = mq.op_name(member)
    if name in {"tensor.dim", "tensor.rank", "memref.dim", "memref.rank"}:
        return True
    if name not in {"tensor.expand_shape", "tensor.collapse_shape", "tensor.cast", "tensor.reshape"}:
        return False
    if not member.operands or len(member.results) != 1 or member.regions:
        return False
    return mq.type_shape_dtype(member.operands[0].type)[1] == mq.type_shape_dtype(member.results[0].type)[1]


def _data_hardware_exclusion(member, capability):
    """A typed data transfer cannot become host glue by its grouping stage."""
    actual = CG.classify(member)
    stage = actual.kind if actual is not None else CG.MOVEMENT
    return _member_excluded(observe_member(member, stage, capability))


def _member_excluded(observation):
    return (type(observation) is dict and observation.get("status") == "unsupported"
            and observation.get("basis") == "actual_typed_source_and_selected_capability"
            and observation.get("refusal") in _EXCLUSIONS)


def _hardware_exclusion(group, capability):
    """Every actual member must be unsupported, independently of grouping hints."""
    checked = False
    for member, stage in zip(group.members, group.stages, strict=True):
        if _metadata_only(member):
            continue
        if not _member_excluded(observe_member(member, stage, capability)):
            return False
        checked = True
    return checked


def _source_value(value, ordinals):
    """Identify the actual SSA producer in this original parsed source."""
    owner = value.owner
    if id(owner) in ordinals:
        return {"producer": ordinals[id(owner)], "result": list(owner.results).index(value),
                "type": str(value.type)}
    parent = owner.parent_op()
    if parent is None or id(parent) not in ordinals:
        raise ValueError("raw hardware join has an SSA producer outside the original source")
    region = owner.parent
    return {"block_owner": ordinals[id(parent)], "region": list(parent.regions).index(region),
            "block": list(region.blocks).index(owner), "argument": value.index, "type": str(value.type)}


def _raw_hardware_joins(operations, verdicts, source_operations, capability, binding):
    unknown = [index for index, verdict in enumerate(verdicts) if verdict.undetermined]
    rows = []
    if unknown and source_operations:
        if (not all(is_sha256(binding.get(key)) for key in ("source_sha256", "normalized_source_sha256"))
            or not all(is_sha256(binding["declarations"].get(key))
                       for key in ("capability", "software", "rtl_facts", "numeric_policy"))):
            raise ValueError("raw hardware join requires complete original source and declaration byte bindings")
        parsed = tuple(source_operations)
        if (parsed[0].parent_op() is not None or tuple(mq.walk(parsed[0])) != parsed
            or len({id(operation) for operation in parsed}) != len(parsed)):
            raise ValueError("raw hardware join requires the complete original parsed source roster")
        ordinals = {id(operation): index for index, operation in enumerate(parsed)}
        for index in unknown:
            operation = operations[index]
            if id(operation) not in ordinals:
                raise ValueError("raw hardware join operation is absent from the original source")
            stage = CG.classify(operation)
            if stage is None:
                continue
            observed = observe_member(operation, stage.kind, capability)
            if not _member_excluded(observed):
                continue
            rows.append({"raw_region": index, "source_ordinal": ordinals[id(operation)],
                         "operation": mq.op_name(operation),
                         "operands": [_source_value(value, ordinals) for value in operation.operands],
                         "results": [str(value.type) for value in operation.results],
                         "actual_typed_hardware": observed})
    record = {"schema": "merlin.raw_source_hardware_reconciliation.v1", "binding": binding,
              "raw_unknown_regions": unknown, "n_source_operations": len(source_operations or ()), "rows": rows}
    return {**record, "sha256": document_digest(record)}


def _raw_reconciliation_complete(receipt):
    raw = receipt.get("n_raw_undetermined_regions", 0)
    record = receipt.get("raw_hardware_reconciliation")
    if record is None:
        return raw == 0 and receipt.get("n_unresolved_raw_regions") == 0
    if type(raw) is not int or raw < 0 or type(record) is not dict:
        return False
    expected = {"source_sha256": receipt.get("source_sha256"),
                "normalized_source_sha256": receipt.get("normalized_source_sha256"),
                "declarations": receipt.get("declarations")}
    rows, indices = record.get("rows"), record.get("raw_unknown_regions")
    if (record.get("schema") != "merlin.raw_source_hardware_reconciliation.v1"
        or record.get("binding") != expected
        or record.get("sha256") != document_digest({key: value for key, value in record.items() if key != "sha256"})
        or type(rows) is not list or type(indices) is not list or len(indices) != raw
        or any(type(index) is not int or index < 0 for index in indices)
        or len(set(indices)) != raw or len(rows) != raw
        or type(record.get("n_source_operations")) is not int or record["n_source_operations"] < 0
        or receipt.get("n_unresolved_raw_regions") != 0):
        return False
    joined = []
    for row in rows:
        if (type(row) is not dict or type(row.get("raw_region")) is not int
            or type(row.get("source_ordinal")) is not int
            or not 0 <= row["source_ordinal"] < record["n_source_operations"]
            or not isinstance(row.get("operation"), str)
            or type(row.get("operands")) is not list or type(row.get("results")) is not list
            or not _member_excluded(row.get("actual_typed_hardware") or {})):
            return False
        joined.append(row["raw_region"])
    return sorted(joined) == sorted(indices) and len({row["source_ordinal"] for row in rows}) == raw


def require_raw_reconciliation(receipt):
    """A zero effective count requires every retained original negative join."""
    if not _raw_reconciliation_complete(receipt):
        raise ValueError("raw source hardware eligibility is not completely reconciled")


class SourceDemandIncomplete(ValueError):
    def __init__(self, receipt):
        self.receipt = deepcopy(receipt)
        super().__init__(f"original source demand has {receipt['n_unknown_host_groups']} unresolved host groups "
                         f"and {receipt['n_missing_direct_groups']} ungrouped required or unresolved operations; "
                         f"{receipt['n_unresolved_raw_regions']} unresolved raw hardware regions")


def derive(groups, operations, verdicts, *, source_sha256, normalized_sha256, capability, software,
           facts_sha256, numeric_policy_sha256, source_operations=None):
    """Reconcile raw typed regions with source-derived dequantized groups.

    Group formation may recognize a legal integer region behind floating
    dequantization shells. Its entire admitted root belongs in demand even when
    the raw per-operation descriptor cannot express that composition.
    Compiler/form gaps and unproved readout exclusions are not host support.
    """
    if (len(operations) != len(verdicts) or len({id(operation) for operation in operations}) != len(operations)
        or [g.index for g in groups] != list(range(len(groups)))):
        raise ValueError("original source group/operation inventory is ambiguous")
    owners = {}
    for group in groups:
        if not group.members:
            raise ValueError("original source group has no operation members")
        for member in group.members:
            if id(member) in owners:
                raise ValueError("original source groups overlap")
            owners[id(member)] = group
    demand, provenance, host_reasons, unknown, missing = [], [], Counter(), [], []
    for group in groups:
        if group.placement != CG.HOST:
            if group.root is None or not isinstance(group.in_dtype, str) or not group.in_dtype:
                raise ValueError("original accelerator group has no root or typed input")
            region, nodes = eligible_source_identity(group.root)
            if any(row["source_region_id"] == region for row in provenance):
                raise ValueError("original accelerator source identities repeat")
            demand.append(group.index)
            provenance.append({"source_group": group.index, "source_region_id": region,
                               "source_node_ids": list(nodes)})
        elif all(_metadata_only(member) for member in group.members):
            host_reasons["shape_metadata_orchestration"] += 1
        elif (group.stages and all(stage in {CG.VIEW, CG.PAD, CG.MOVEMENT} for stage in group.stages)
              and all(_metadata_only(member) or _data_hardware_exclusion(member, capability)
                      for member in group.members)):
            host_reasons["selected_hardware:data_transfer_unsupported"] += 1
        elif _hardware_exclusion(group, capability):
            host_reasons["selected_hardware:actual_source_members_unsupported"] += 1
        else:
            host_reasons[f"unresolved:{group.refusal or 'missing_typed_refusal'}"] += 1
            unknown.append(group.index)
    for index, (operation, verdict) in enumerate(zip(operations, verdicts, strict=True)):
        if verdict.eligible:
            owner = owners.get(id(operation))
            if owner is None or owner.index not in demand:
                missing.append(index)
    data_missing = []
    for index, operation in enumerate(source_operations or ()):
        if id(operation) in owners or _metadata_only(operation):
            continue
        parent = operation.parent_op()
        while parent is not None and id(parent) not in owners:
            parent = parent.parent_op()
        if parent is not None:
            # Scalar/data components inside an already inventoried original
            # region belong to that root, not a second independent denominator.
            continue
        stage = CG.classify(operation)
        if ((stage is not None and stage.kind in {CG.PAD, CG.MOVEMENT, CG.VIEW})
            or mq.op_name(operation) == "memref.copy"):
            if not _data_hardware_exclusion(operation, capability):
                data_missing.append(index)
    declarations = {"capability": document_digest(dict(capability)), "software": document_digest(dict(software)),
                    "rtl_facts": facts_sha256, "numeric_policy": numeric_policy_sha256}
    binding = {"source_sha256": source_sha256, "normalized_source_sha256": normalized_sha256,
               "declarations": declarations}
    joins = _raw_hardware_joins(operations, verdicts, source_operations, capability, binding)
    receipt = {
        "schema": SCHEMA, "status": "source_demand_derived", "source_sha256": source_sha256,
        "normalized_source_sha256": normalized_sha256,
        "declarations": declarations,
        "n_source_groups": len(groups), "n_demanded_groups": len(demand), "n_requested_groups": 0,
        "n_lowered_groups": 0, "n_covered_groups": 0,
        "n_host_groups": len(groups) - len(demand), "host_reason_counts": dict(sorted(host_reasons.items())),
        "n_unknown_host_groups": len(unknown), "n_missing_direct_groups": len(missing) + len(data_missing),
        "n_raw_eligible_regions": sum(verdict.eligible for verdict in verdicts),
        "n_raw_undetermined_regions": sum(verdict.undetermined for verdict in verdicts),
        "n_unresolved_raw_regions": sum(verdict.undetermined for verdict in verdicts) - len(joins["rows"]),
        "raw_hardware_reconciliation": joins,
        "demanded_group_provenance": provenance,
        "scope": "all original source-admitted device roots and direct hardware-eligible operations; no host fallback",
    }
    if unknown or missing or data_missing or receipt["n_unresolved_raw_regions"]:
        receipt["status"] = "incomplete"
        raise SourceDemandIncomplete(receipt)
    return receipt


def request(source, device):
    from merlin.compile.complete_group_routing import select_complete_group
    from merlin.llvmlower.device_build import DeviceRouting

    receipt = source["accelerator_source_demand"]
    if receipt["n_demanded_groups"] and (
        type(device) is not DeviceRouting or device.require_complete_groups is not True
        or device.select is not select_complete_group
    ):
        raise ValueError("original source demand was not fully requested from the accelerator compiler")
    receipt["n_requested_groups"] = receipt["n_demanded_groups"]


def link(source, linked_build, covered):
    receipt = source["accelerator_source_demand"]
    require_raw_reconciliation(receipt)
    if (receipt["status"] != "source_demand_derived" or receipt["n_unresolved_raw_regions"] != 0
        or receipt["n_unknown_host_groups"] or receipt["n_missing_direct_groups"]
        or receipt["n_requested_groups"] != receipt["n_demanded_groups"]
        or covered != receipt["n_demanded_groups"]):
        raise ValueError("linked image does not cover complete original accelerator source demand")
    receipt.update(status="source_demand_linked", n_lowered_groups=covered, n_covered_groups=covered,
                   linked_build=dict(linked_build))


def linked_complete(source, entry, candidate_sha256):
    receipt = source.get("accelerator_source_demand")
    if (type(receipt) is not dict or receipt.get("schema") != SCHEMA or receipt.get("status") != "source_demand_linked"
        or not _raw_reconciliation_complete(receipt)
        or receipt.get("source_sha256") != source.get("source_sha256")
        or receipt.get("normalized_source_sha256") != source.get("normalized_source_sha256")
        or type(receipt.get("declarations")) is not dict
        or not all(is_sha256(receipt["declarations"].get(key))
                   for key in ("capability", "software", "rtl_facts", "numeric_policy"))
        or any(type(receipt.get(key)) is not int or receipt[key] < 0 for key in (
            "n_source_groups", "n_demanded_groups", "n_requested_groups", "n_lowered_groups",
            "n_covered_groups", "n_host_groups",
            "n_unknown_host_groups", "n_missing_direct_groups", "n_unresolved_raw_regions"))
        or any(receipt[key] for key in ("n_unknown_host_groups", "n_missing_direct_groups", "n_unresolved_raw_regions"))
        or receipt["n_source_groups"] != receipt["n_demanded_groups"] + receipt["n_host_groups"]
        or receipt["n_demanded_groups"] != receipt["n_covered_groups"]
        or receipt["n_requested_groups"] != receipt["n_demanded_groups"]
        or receipt["n_lowered_groups"] != receipt["n_demanded_groups"]
        or receipt["n_covered_groups"] != entry.get("linked_device_groups")
        or receipt.get("demanded_group_provenance") != source.get("eligible_group_provenance")):
        return False
    return receipt.get("linked_build") == {
        "capture_tree_sha256": entry.get("capture_tree_sha256"), "elf_sha256": entry.get("elf_sha256"),
        "candidate_tree_sha256": candidate_sha256,
    }
