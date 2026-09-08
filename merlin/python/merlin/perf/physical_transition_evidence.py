"""Host-owned evidence for simple source-bound, scalar LLVM layout copies.

No candidate code is imported or executed. This proves the copy's address and
bit-preserving dataflow plus a declared consumer-region use, not that the rest of
the consumer implements the source operation. Unsupported patterns stay UNKNOWN.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from math import prod
from typing import Any

from xdsl.dialects.builtin import TensorType
from xdsl.dialects.llvm import LLVM, GEP_USE_SSA_VAL
from xdsl.ir import Block
from xdsl.irdl.dominance import DominanceInfo

from merlin.frontends.linalg_mlir import make_context, parse_mlir_text
from merlin.perf.host_cfg_activity import analyze_host_cfg_activity
from merlin.perf.structural_transitions import StaticStridedLayout
from merlin.xdsl_dialects.lowering.integer_constant_eval import constant_integer

MARKERS = ("merlin.global_transition", "merlin.transition_source", "merlin.transition_buffer")


class _Refused(ValueError):
    pass


class _Unsupported(ValueError):
    pass


def _need(condition: bool, message: str, *, unsupported: bool = False) -> None:
    if not condition:
        raise (_Unsupported if unsupported else _Refused)(message)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _finalize(result: dict[str, Any]) -> dict[str, Any]:
    """Attach a content address and expose only artifact-verified encoding activity.

    A transition declaration is an emitter obligation, not evidence that the emitted program
    realizes it.  Consequently ``not_declared``, ``UNKNOWN`` and ``refused`` all retain unknown
    executed counts and bytes.  The exact totals below exist only after :func:`_verify_one` has
    checked the emitted CFG, typed load/store dataflow, address functions, allocation extents and
    dynamic access multiplicity for every transition.
    """
    transitions = result.get("transitions")
    rows = transitions if isinstance(transitions, list) else []
    verified = result.get("status") == "verified" and bool(rows) and all(
        isinstance(row, Mapping) and row.get("status") == "verified" for row in rows)
    read_bytes = write_bytes = 0
    if verified:
        for row in rows:
            load = row.get("load_payload_bytes")
            store = row.get("store_payload_bytes")
            if (type(load) is not int or load < 0 or type(store) is not int or store < 0
                    or row.get("materialized") is not True
                    or row.get("execution_multiplicity_verified") is not True):
                verified = False
                break
            read_bytes += load
            write_bytes += store
    result["encoding_activity"] = ({
        "status": "verified",
        "executed_transition_count": len(rows),
        "materialized_transition_count": len(rows),
        "physical_read_bytes": read_bytes,
        "physical_write_bytes": write_bytes,
        "physical_bytes": read_bytes + write_bytes,
        "basis": (
            "exact emitted CFG multiplicity, typed load/store dataflow and proved physical "
            "address functions"
        ),
    } if verified else {
        "status": "UNKNOWN",
        "executed_transition_count": None,
        "materialized_transition_count": None,
        "physical_read_bytes": None,
        "physical_write_bytes": None,
        "physical_bytes": None,
        "reason": (
            "encoding declarations do not prove materialization; every emitted transition must "
            "pass artifact verification"
        ),
    })
    body = dict(result)
    body.pop("receipt_sha256", None)
    result["receipt_sha256"] = _sha(_canonical(body))
    return result


def _tag(operation, name):
    attribute = operation.attributes.get(name)
    return getattr(attribute, "data", None)


def _root(value):
    seen = set()
    while not isinstance(value.owner, Block) and value.owner.name in {"llvm.getelementptr", "llvm.bitcast"}:
        _need(value not in seen, "cyclic pointer expression")
        seen.add(value)
        value = value.owner.operands[0]
    return value


def _index(row, key, size):
    value = row.get(key)
    _need(type(value) is int and 0 <= value < size, f"invalid source edge {key}")
    return value


def _verify_one(transition, source_ops, function, activity):
    ident = transition["id"]
    _need(all(isinstance(transition.get(key), str) and bool(transition[key].strip())
              for key in ("producer", "consumer", "buffer")),
          "copy requires nonempty producer, consumer and buffer identities")
    _need(transition.get("kind") == "static_strided_copy", "unsupported transition mechanism", unsupported=True)
    edge = transition.get("source_edge")
    _need(isinstance(edge, Mapping), "transition has no exact source edge")
    producer = source_ops[_index(edge, "producer_op_index", len(source_ops))]
    consumer = source_ops[_index(edge, "consumer_op_index", len(source_ops))]
    value = producer.results[_index(edge, "producer_result_index", len(producer.results))]
    operand = consumer.operands[_index(edge, "consumer_operand_index", len(consumer.operands))]
    _need(operand is value, "declared source producer does not feed the declared consumer operand")
    _need(isinstance(value.type, TensorType), "copy source must be a tensor", unsupported=True)
    element = value.type.get_element_type()
    bits = getattr(element, "bitwidth", None)
    _need(type(bits) is int and bits > 0 and bits % 8 == 0,
          "source scalar type is not a supported byte-addressable type", unsupported=True)
    width, dtype = bits // 8, str(element)
    shape = tuple(value.type.get_shape())
    _need(all(type(dim) is int and dim > 0 for dim in shape), "dynamic/empty copy domain is unsupported", unsupported=True)
    layouts = []
    for side in ("source", "destination"):
        spec, representation = transition.get(side+"_layout"), transition.get(side)
        _need(isinstance(spec, Mapping) and isinstance(representation, Mapping), "missing typed layout contract")
        layout = StaticStridedLayout(tuple(spec["shape"]), tuple(spec["strides_elements"]),
                                     spec["storage_bytes"], spec["offset_elements"])
        layout.validate(dtype)
        _need(layout.shape == shape, "layout logical domain differs from source")
        _need(representation == layout.representation(dtype=dtype, placement=representation["placement"]).to_dict(),
              "representation differs from source type or declared address function")
        layouts.append(layout)
    _need(transition["source"]["placement"] == transition["destination"]["placement"],
          "cross-address-space copy needs another verifier", unsupported=True)
    _need(transition.get("materializes") is True, "copy must explicitly declare materialization")

    blocks = list(function.body.blocks)
    positions = {operation: index for index, operation in enumerate(function.walk())}
    loops = {row["header_block"]: row for row in activity["loops"]}
    copies = [op for op in function.walk() if _tag(op, MARKERS[0]) == ident]
    _need(bool(copies), "declared copy emitted no owned operations")
    allowed = {"llvm.alloca", "llvm.mlir.constant", "llvm.add", "llvm.mul", "llvm.icmp",
               "llvm.br", "llvm.cond_br", "llvm.getelementptr", "llvm.load", "llvm.store"}
    _need(all(op.name in allowed for op in copies), "copy contains unsupported effects/operations", unsupported=True)
    loads, stores = [op for op in copies if op.name == "llvm.load"], [op for op in copies if op.name == "llvm.store"]
    _need(len(loads) == len(stores) == 1, "only one scalar load/store in a structured copy loop is supported", unsupported=True)
    load, store = loads[0], stores[0]
    _need(load.parent is store.parent, "copy load/store must share the proved execution block", unsupported=True)
    _need(store.operands[0] is load.results[0], "copy does not store exactly the loaded bits")
    _need(load.results[0].type == element, "emitted copy scalar type differs from the source")
    src, dst = _root(load.operands[0]), _root(store.operands[1])
    _need(src is not dst, "copy source and destination are the same allocation")
    _need(not isinstance(src.owner, Block) and not isinstance(dst.owner, Block)
          and src.owner.name == dst.owner.name == "llvm.alloca", "copy requires explicit allocation identities", unsupported=True)
    _need(_tag(src.owner, MARKERS[1]) == ident and _tag(dst.owner, MARKERS[2]) == ident,
          "actual pointer roots differ from declared source/destination ownership")
    _need(dst.owner in copies, "destination allocation is not owned by the copy")
    for marker, allocation in ((MARKERS[1], src.owner), (MARKERS[2], dst.owner)):
        _need([op for op in function.walk() if _tag(op, marker) == ident] == [allocation],
              "transition allocation identity is missing or ambiguous")
    body_index = blocks.index(load.parent)
    active_headers = {header for header, row in loops.items() if body_index in row["body_blocks"]}

    def affine(value, seen=frozenset()):
        _need(value not in seen and len(seen) < 128, "unbounded/cyclic copy address expression", unsupported=True)
        owner = value.owner
        if isinstance(owner, Block):
            header = blocks.index(owner)
            _need(header in active_headers, "address uses an induction value outside its active loop")
            loop = loops[header]
            _need(value.index == loop["induction_argument_index"] and loop["initial"] == 0 and loop["step"] == 1,
                  "copy induction form is unsupported", unsupported=True)
            constant, coefficients = 0, {header: 1}
        else:
            number = constant_integer(value)
            if number is not None:
                constant, coefficients = number, {}
            else:
                _need(owner.name in {"llvm.add", "llvm.mul"}, "copy address is not a supported affine expression", unsupported=True)
                a, left = affine(owner.operands[0], seen | {value})
                b, right = affine(owner.operands[1], seen | {value})
                if owner.name == "llvm.add":
                    constant = a+b
                    coefficients = {key: left.get(key, 0)+right.get(key, 0) for key in set(left) | set(right)}
                else:
                    _need(not left or not right, "non-affine copy address", unsupported=True)
                    constant = a*b
                    coefficients = ({key: coefficient*b for key, coefficient in left.items()} if left else
                                    {key: coefficient*a for key, coefficient in right.items()})
        integer_bits = getattr(value.type, "bitwidth", None)
        _need(type(integer_bits) is int and constant >= 0 and all(c >= 0 for c in coefficients.values()),
              "unsupported signed/negative address arithmetic", unsupported=True)
        upper = constant+sum(c*(loops[h]["trip_count"]-1) for h, c in coefficients.items())
        _need(upper < 1 << (integer_bits-1), "copy address arithmetic may wrap or sign-extend negatively")
        return constant, coefficients

    allocation_rows = []
    for layout, pointer, root in zip(layouts, (load.operands[0], store.operands[1]), (src, dst)):
        gep = pointer.owner
        _need(gep.name == "llvm.getelementptr" and len(gep.operands) == 2 and gep.operands[0] is root,
              "copy requires a direct, single-index allocation GEP", unsupported=True)
        _need(gep.properties.get("elem_type") == element, "GEP element scale differs from copied scalar type")
        raw = gep.properties.get("rawConstantIndices")
        _need(raw is not None and tuple(raw.get_values()) == (GEP_USE_SSA_VAL,),
              "unsupported compound GEP indexing", unsupported=True)
        offset, coefficients = affine(gep.operands[1])
        headers = sorted(coefficients)
        _need(set(headers) == active_headers, "copy address does not cover its complete loop domain")
        _need(tuple(loops[h]["trip_count"] for h in headers) == shape,
              "copy loop extents differ from logical source shape")
        _need(tuple(coefficients[h] for h in headers) == layout.strides_elements and offset == layout.offset_elements,
              "actual copy address differs from declared layout")
        allocation = next(row for row in activity["allocations"] if row["operation_index"] == positions[root.owner])
        _need(root.owner.properties.get("elem_type") == element and allocation["payload_bytes"] == layout.storage_bytes
              and allocation["execution_count"] == 1, "actual allocation type/extent/execution differs from layout")
        allocation_rows.append(allocation)
    multiplicity = activity["blocks"][body_index]["execution_count"]
    _need(multiplicity == prod(shape), "copy dynamic access multiplicity differs from logical domain")
    expected = {"scalar_load_payload": multiplicity*width, "scalar_store_payload": multiplicity*width,
                "destination_storage": allocation_rows[1]["payload_bytes"]}
    quantities = transition.get("quantities")
    _need(isinstance(quantities, list) and all(isinstance(item, Mapping) for item in quantities), "missing charged copy quantities")
    _need(len({item.get("name") for item in quantities}) == len(quantities), "duplicate charged copy quantity")
    for name, amount in expected.items():
        row = next((item for item in quantities if item.get("name") == name), {})
        _need(type(row.get("amount")) is int and row["amount"] == amount and row.get("unit") == "bytes"
              and row.get("basis") == "derived", f"charged {name} does not match actual emitted copy")

    # Follow every destination pointer alias. Unknown escapes cannot hide later writes.
    aliases, pending, consumers = {dst}, [dst], []
    while pending:
        pointer = pending.pop()
        for use in pointer.uses:
            op = use.operation
            if op.name in {"llvm.getelementptr", "llvm.bitcast"} and use.index == 0:
                for result in op.results:
                    if result not in aliases:
                        aliases.add(result)
                        pending.append(result)
            elif op.name == "llvm.load" and use.index == 0:
                if op not in copies:
                    _need(_tag(op, "merlin.structural_region") == transition.get("consumer"),
                          "converted allocation is read outside the declared consumer region")
                    consumers.append(op)
            elif op.name == "llvm.store" and use.index == 1:
                _need(op is store, "converted allocation has an unrelated writer")
            else:
                raise _Unsupported("converted allocation escapes through an unsupported pointer use")
    _need(bool(consumers), "declared consumer never reads the converted allocation")
    dominators = DominanceInfo(function.body)
    outer_headers = [h for h in active_headers if not any(
        h in loops[other]["body_blocks"] for other in active_headers if other != h)]
    for header in outer_headers:
        exit_block = blocks[header].last_op.successors[1]
        _need(all(dominators.dominates(exit_block, op.parent) for op in consumers),
              "consumer can read converted storage before copy completion")
    return {"id": ident, "status": "verified", "source_edge": dict(edge), "dtype": dtype,
            "element_bytes": width, "logical_shape": list(shape), "allocations": allocation_rows,
            "load_payload_bytes": expected["scalar_load_payload"],
            "store_payload_bytes": expected["scalar_store_payload"],
            "destination_storage_bytes": expected["destination_storage"],
            "physical_bytes": expected["scalar_load_payload"] + expected["scalar_store_payload"],
            "materialized": True, "execution_multiplicity_verified": True,
            "owned_operation_indices": [positions[op] for op in copies],
            "consumer_load_operation_indices": [positions[op] for op in consumers],
            "source_allocation_binding": "compiler-declared source identity; source arithmetic not independently proved",
            "consumer_binding": "converted allocation is read in declared region; consumer arithmetic not proved",
            "bit_preserving_copy": True, "fresh_disjoint_allocations": True}


def verify_physical_transitions(*, source_text: str, lowered_text: str,
                                command_buffer: Mapping[str, Any]) -> dict[str, Any]:
    """Return verified/refused/UNKNOWN/not_declared with exact artifact identities."""
    result = {"schema": "physical_transition_evidence_v1", "source_sha256": _sha(source_text),
              "lowered_sha256": _sha(lowered_text), "transitions": [], "problems": [],
              "scope": "simple scalar copy addresses, byte accounting and declared consumer-region binding",
              "numeric_equivalence": "copy bits only; arbitrary producer/consumer/model semantics unproved",
              "cycles": None, "dram_bytes": None, "full_model_executed": False}
    try:
        result["command_buffer_sha256"] = _sha(_canonical(command_buffer))
        context = make_context()
        context.load_dialect(LLVM)
        module = parse_mlir_text(lowered_text, context)
        module.verify()
        marked = set()
        for op in module.walk():
            for marker in MARKERS:
                if marker in op.attributes:
                    value = _tag(op, marker)
                    _need(isinstance(value, str) and bool(value.strip()), "malformed physical transition marker")
                    marked.add(value)
        params = command_buffer.get("params", {})
        _need(isinstance(params, Mapping), "malformed command buffer params")
        plan = params.get("global_program_plan", {})
        _need(isinstance(plan, Mapping), "malformed global program plan")
        declarations = plan.get("physical_transitions", [])
        _need(isinstance(declarations, list) and all(isinstance(row, Mapping) for row in declarations), "malformed transition declaration list")
        ids = [row.get("id") for row in declarations]
        _need(all(isinstance(ident, str) and bool(ident.strip()) for ident in ids) and len(set(ids)) == len(ids), "malformed/duplicate transition identities")
        _need(marked == set(ids), "declared transition identities and actual LLVM markers do not exactly match")
        if not declarations:
            result["status"] = "not_declared"
            return _finalize(result)
        _need(plan.get("source_sha256") == result["source_sha256"], "transition plan is not bound to the exact source")
        source = parse_mlir_text(source_text)
        functions = [op for op in source.body.block.ops if op.name == "func.func" and op.body.blocks]
        emitted = [op for op in module.body.block.ops if op.name == "llvm.func" and op.body.blocks]
        _need(len(functions) == len(emitted) == 1 and len(functions[0].body.blocks) == 1,
              "copy verifier requires one source entry and one emitted kernel", unsupported=True)
        source_ops = [op for op in functions[0].body.block.ops if op.name != "func.return"]
        activity = analyze_host_cfg_activity(emitted[0])
        _need(activity["status"] == "derived", "copy CFG multiplicity is unsupported: "+str(activity["problems"]), unsupported=True)
        for row in declarations:
            try:
                result["transitions"].append(_verify_one(row, source_ops, emitted[0], activity))
            except (_Refused, _Unsupported, KeyError, TypeError, ValueError, StopIteration) as error:
                status = "UNKNOWN" if isinstance(error, _Unsupported) else "refused"
                result["transitions"].append({"id": row.get("id"), "status": status, "problems": [str(error)]})
        states = {row["status"] for row in result["transitions"]}
        result["status"] = "refused" if "refused" in states else "UNKNOWN" if "UNKNOWN" in states else "verified"
    except (_Refused, KeyError, TypeError, ValueError) as error:
        result.update(status="UNKNOWN" if isinstance(error, _Unsupported) else "refused", problems=[str(error)])
    except Exception as error:
        result.update(status="UNKNOWN", problems=[f"unsupported analysis: {type(error).__name__}: {error}"])
    return _finalize(result)
