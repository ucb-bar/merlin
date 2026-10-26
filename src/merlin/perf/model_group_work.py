"""Source work for every original group, independently of a target schedule.

The existing MAC-domain observer proves recurrences. The same affine-domain
reader counts scalar body instances without expanding loops. Neither source
count is a runtime cost: fusion, vectorization, preparation and storage may
change emitted work, and a timing provider needs its own complete scope.
"""

from collections import Counter
from hashlib import sha256
from math import prod

from merlin.perf.model_macs import _domain, _iterators, observe_model_macs
from merlin.xdsl_dialects._common import text


def summarize_group_work(module, groups, *, entry="forward"):
    """Observe current source groups; report unknown domains rather than zeros.

    Caller-owned placement and grouping are retained as descriptions, never
    inferred as execution/admission evidence. Every MAC operation found outside
    the supplied membership remains visible in the aggregate coverage record.
    """
    module.verify()
    before = text(module, generic=True)
    functions = [op for op in module.body.block.ops if op.name == "func.func"
                 and op.sym_name.data == entry and len(op.body.blocks) == 1]
    if len(functions) != 1:
        raise ValueError("source work requires one original single-block entry")
    block = functions[0].body.block
    positions = {op: i for i, op in enumerate(block.ops)}
    current = set(module.walk())
    groups = tuple(groups)
    indices = [group.index for group in groups]
    members = [op for group in groups for op in group.members]
    if (any(type(index) is not int or index < 0 for index in indices)
            or len(set(indices)) != len(indices) or len(set(members)) != len(members)
            or any(op not in current for op in members)):
        raise ValueError("source work needs distinct current groups and operation membership")
    regions = set()
    for group in groups:
        operations = {op for member in group.members for op in member.walk()}
        if regions & operations:
            raise ValueError("source work cannot count overlapping original group regions")
        regions.update(operations)
    proofs = observe_model_macs(module, entry=entry)
    owned = {op for member in members for op in member.walk()}
    unowned = [proof for op, proof in proofs if op not in owned]
    rows = []
    for group in groups:
        operations = {op for member in group.members for op in member.walk()}
        macs = [proof for op, proof in proofs if op in operations]
        unknown_macs = [proof.reason for proof in macs if proof.macs is None]
        source_scope = all(op.parent is block for op in group.members)
        if not source_scope:
            unknown_macs.append("source group invocation/control-flow multiplicity is unproved")
        arithmetic = Counter()
        unknown_scalar = []
        for op in group.members:
            if op.name == "builtin.unregistered":
                unknown_scalar.append({"operation_ordinal": positions.get(op),
                                       "reason": "opaque source operation has no scalar body domain"})
                continue
            if not op.name.startswith("linalg."):
                continue
            try:
                iterators = _iterators(op)
                if iterators is None or len(op.regions) != 1 or len(op.regions[0].blocks) != 1:
                    raise ValueError("source scalar body/domain is not exposed")
                if any(inner.regions for inner in op.regions[0].block.ops):
                    raise ValueError("nested scalar control-flow multiplicity is unproved")
                parallel, reduction, _ = _domain(op, iterators)
                instances = prod((*parallel, *reduction))
                if not source_scope:
                    raise ValueError("source group invocation/control-flow multiplicity is unproved")
                for inner in op.regions[0].block.ops:
                    if inner.name.startswith(("arith.", "math.")):
                        arithmetic[inner.name] += instances
            except (ValueError, AttributeError, IndexError, NotImplementedError) as error:
                unknown_scalar.append({"operation_ordinal": positions.get(op), "reason": str(error)})
        known_macs = sum(proof.macs for proof in macs if proof.macs is not None)
        rows.append({
            "index": group.index, "placement_description": group.placement,
            "source_operation_ordinals": [positions.get(op) for op in group.members],
            "source_macs": None if unknown_macs else known_macs,
            "known_source_macs_subtotal": known_macs,
            "unknown_mac_reasons": unknown_macs,
            "known_scalar_body_instances": dict(sorted(arithmetic.items())),
            "scalar_body_coverage_complete": not unknown_scalar,
            "unknown_scalar_domains": unknown_scalar,
            "requested_transfer_bytes": None, "runtime_host_instructions": None,
            "synchronization_cycles": None, "elapsed_cycles": None,
            "physical_array_utilization": None,
        })
    known = sum(row["known_source_macs_subtotal"] for row in rows)
    complete = not unowned and all(row["source_macs"] is not None for row in rows)
    if text(module, generic=True) != before:
        raise ValueError("source work observation changed the original module")
    return {
        "schema": "merlin.model_group_source_work.v1",
        "source_ir_sha256": sha256(before.encode()).hexdigest(), "entry": entry,
        "groups": rows, "source_macs": known if complete else None,
        "known_source_macs_subtotal": known, "mac_coverage_complete": complete,
        "unassigned_mac_operations": [
            {"operation": proof.op, "source_operation_ordinal": proof.source_op_index,
             "macs": proof.macs, "reason": proof.reason} for proof in unowned
        ],
        "meaning": "original source recurrences and scalar body instances; not emitted work, traffic or cycles",
        "execution_evidence": False, "placement_or_effect_admission": False,
        "timing_ranking_available": False,
    }
