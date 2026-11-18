"""Bind an explicit read-only consumer to a proved segmented tensor input.

The provider accepts the element address map and must not retain or write the
input. Dense owner storage comes from a separately supplied fresh writer
contract. The existing fresh-writer rewrite validates those contracts on a
clone before any source edits, and must subsequently run on the accepted route.
This helper emits no ABI bridge or device code and removes no view operations.
"""

import json
from dataclasses import dataclass

from xdsl.dialects.builtin import FunctionType, StringAttr, SymbolRefAttr
from xdsl.dialects.func import CallOp

from merlin.compiler.lower.host.writers import rewrite_fresh_tensor_writers
from merlin.compiler.lower.source.dense_group import SegmentedRows, prove_segmented_matrix


@dataclass(frozen=True)
class SegmentedInputContract:
    """Provider acceptance of a bounded map, read-only and without retention."""

    symbol: str
    argument: int
    accepted_symbol: str
    address: SegmentedRows


def rewrite_segmented_inputs(module, contracts, *, fresh_writers):
    """Retarget selected arguments to their unchanged, typed dense SSA owners.

    Every selected call must have the exact accepted map and owner type. All
    selection and producer/consumer full-write checks precede mutation. Source
    numeric attributes and output ABI are copied verbatim to the accepted
    declaration. The caller must compile its accepted consumer and replace the
    consumer's fresh-writer contract with the accepted symbol before lowering.
    """
    contracts = tuple(contracts)
    if not contracts:
        return []
    writers = tuple(fresh_writers)
    producer_contracts = {contract.symbol: contract for contract in writers}
    if len(producer_contracts) != len(writers):
        raise ValueError("duplicate fresh writer contract")
    # Reuse the actual allocation/ABI/full-write validator, preserving source IR.
    rewrite_fresh_tensor_writers(module.clone(), writers)
    declarations = {op.sym_name.data: op for op in module.body.block.ops if op.name == "func.func"}
    selected = set()
    reserved = set(declarations)
    plans = []
    for contract in contracts:
        name = contract.symbol
        accepted = contract.accepted_symbol
        if name in selected:
            raise ValueError("duplicate segmented input selection")
        selected.add(name)
        if not accepted or accepted in reserved:
            raise ValueError("accepted consumer symbol collision")
        reserved.add(accepted)
        declaration = declarations.get(name)
        if declaration is None or declaration.body.blocks or name not in producer_contracts:
            raise ValueError("consumer requires an explicit bodyless fresh writer contract")
        types = tuple(declaration.function_type.inputs)
        index = contract.argument
        if type(index) is not int or not 0 <= index < len(types):
            raise ValueError("consumer input argument is outside its declaration")
        attrs = declaration.arg_attrs
        if attrs is None or getattr(attrs.data[index].data.get("bufferization.access"), "data", None) != "read":
            raise ValueError("segmented input requires declared read-only access")
        if not isinstance(contract.address, SegmentedRows):
            raise ValueError("consumer requires a typed segmented address contract")
        calls = [op for op in module.walk() if isinstance(op, CallOp) and op.callee.root_reference.data == name]
        if not calls:
            raise ValueError("selected consumer has no source call")
        matches = []
        owner_type = None
        for call in calls:
            proof = prove_segmented_matrix(call.arguments[index])
            if proof.address != contract.address:
                raise ValueError("consumer accepted map differs from the source view")
            owner = proof.owner.owner
            if not isinstance(owner, CallOp) or owner.callee.root_reference.data not in producer_contracts:
                raise ValueError("dense owner requires a supplied fresh writer producer")
            producer = producer_contracts[owner.callee.root_reference.data]
            if producer.allow_initialized_writers or owner.results[0] is not proof.owner:
                raise ValueError("dense owner requires a unique fresh producer result")
            if owner.arguments[producer.result_argument].type != proof.owner.type:
                raise ValueError("dense owner producer result type differs from its allocation")
            if owner_type is not None and owner_type != proof.owner.type:
                raise ValueError("consumer calls disagree on dense owner type")
            owner_type = proof.owner.type
            matches.append((call, proof))
        updated = list(types)
        updated[index] = owner_type
        plans.append((declaration, contract, updated, matches))
    report = []
    for declaration, contract, updated, matches in plans:
        accepted = declaration.clone()
        accepted.properties["sym_name"] = StringAttr(contract.accepted_symbol)
        accepted.properties["function_type"] = FunctionType.from_lists(updated, declaration.function_type.outputs)
        accepted.attributes["merlin.accepted_input_view"] = StringAttr(
            json.dumps(dict(argument=contract.argument, address=contract.address.__dict__), sort_keys=True)
        )
        declaration.parent.insert_op_before(accepted, declaration)
        owners = []
        for call, proof in matches:
            operands = list(call.operands)
            operands[contract.argument] = proof.owner
            call.operands = operands
            call.properties["callee"] = SymbolRefAttr(contract.accepted_symbol)
            owners.append(proof.owner.owner.callee.root_reference.data)
        report.append(
            dict(
                source_symbol=contract.symbol,
                accepted_symbol=contract.accepted_symbol,
                argument=contract.argument,
                calls=len(matches),
                owner_type=str(updated[contract.argument]),
                owner_producers=owners,
                address=contract.address.__dict__.copy(),
                numeric_attributes="Preserved verbatim",
                ownership="Fresh writer producer; unchanged read-only SSA owner",
                provider_obligations="Compile this accepted map; no input writes or retention; run fresh-writer conversion",
            )
        )
    module.verify()
    return report
