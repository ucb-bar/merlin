"""Complete original source demand, independent of the compiler's chosen route."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy

from merlin.common import mlir_query as mq
from merlin.common.digest import is_sha256
from merlin.targetgen import eligibility as E
from merlin.verify.lean_contract import document_digest
from merlin.xdsl_dialects.lowering import compute_groups as CG
from merlin_experiments.phase1.feedback.private_group_provenance import eligible_source_identity

SCHEMA = "merlin.private_source_device_demand.v1"
_EXCLUSIONS = frozenset({"undeclared_family", "input_dtype", "weight_dtype", "operand_pair", "result_dtype",
                         "rank", "batch", "layout", "form"})


def _hardware_exclusion(group, capability):
    """A source reason must agree with the independently selected declaration."""
    if group.refusal not in _EXCLUSIONS or not isinstance(group.in_dtype, str) or not group.in_dtype:
        return False
    cap_map = E.capability_map_from_contract(dict(capability))
    checked = False
    for member, stage in zip(group.members, group.stages, strict=True):
        if stage in {CG.VIEW, CG.PAD, CG.MOVEMENT}:
            continue
        family = CG.FAMILY_OF_STAGE.get(stage)
        if family is None:
            return False
        shape = mq.type_shape_dtype(member.results[0].type)[0] if member.results else []
        verdict = E.is_eligible(
            E.RegionDescriptor(op=stage, family=family, in_dtype=group.in_dtype,
                               weight_dtype=group.weight_dtype, rank=len(shape) if shape else None),
            cap_map, undetermined=E.undetermined_families_from_contract(dict(capability)),
        )
        if verdict.eligible or verdict.undetermined or verdict.refusal not in _EXCLUSIONS:
            return False
        checked = True
    return checked


class SourceDemandIncomplete(ValueError):
    def __init__(self, receipt):
        self.receipt = deepcopy(receipt)
        super().__init__(f"original source demand has {receipt['n_unknown_host_groups']} unresolved host groups "
                         f"and {receipt['n_missing_direct_groups']} ungrouped eligible operations")


def derive(groups, operations, verdicts, *, source_sha256, normalized_sha256, capability, software,
           facts_sha256, numeric_policy_sha256):
    """Reconcile raw typed regions with source-derived dequantized groups.

    Group formation may recognize a legal integer region behind floating
    dequantization shells. Its entire admitted root belongs in demand even when
    the raw per-operation descriptor cannot express that composition.
    Compiler/form gaps and unproved readout exclusions are not host support.
    """
    if len(operations) != len(verdicts) or [g.index for g in groups] != list(range(len(groups))):
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
        elif group.stages and all(stage in {CG.VIEW, CG.PAD, CG.MOVEMENT} for stage in group.stages):
            host_reasons["data_or_shape_orchestration"] += 1
        elif _hardware_exclusion(group, capability):
            host_reasons[f"selected_hardware:{group.refusal}"] += 1
        else:
            host_reasons[f"unresolved:{group.refusal or 'missing_typed_refusal'}"] += 1
            unknown.append(group.index)
    for index, (operation, verdict) in enumerate(zip(operations, verdicts, strict=True)):
        if verdict.eligible:
            owner = owners.get(id(operation))
            if owner is None or owner.index not in demand:
                missing.append(index)
    receipt = {
        "schema": SCHEMA, "status": "source_demand_derived", "source_sha256": source_sha256,
        "normalized_source_sha256": normalized_sha256,
        "declarations": {"capability": document_digest(dict(capability)), "software": document_digest(dict(software)),
                         "rtl_facts": facts_sha256, "numeric_policy": numeric_policy_sha256},
        "n_source_groups": len(groups), "n_demanded_groups": len(demand), "n_requested_groups": 0,
        "n_lowered_groups": 0, "n_covered_groups": 0,
        "n_host_groups": len(groups) - len(demand), "host_reason_counts": dict(sorted(host_reasons.items())),
        "n_unknown_host_groups": len(unknown), "n_missing_direct_groups": len(missing),
        "n_raw_eligible_regions": sum(verdict.eligible for verdict in verdicts),
        "n_raw_undetermined_regions": sum(verdict.undetermined for verdict in verdicts),
        "n_unresolved_raw_regions": sum(verdict.undetermined for verdict in verdicts),
        "demanded_group_provenance": provenance,
        "scope": "all original source-admitted device roots and direct hardware-eligible operations; no host fallback",
    }
    if unknown or missing:
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
