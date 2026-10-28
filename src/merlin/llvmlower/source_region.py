"""Extract exact current SSA spans for ordinary source-owned region builds.

Extraction grants no effect, offload, numerical or source-closure permission.
Every operation remains in source order; free operands and all externally
observed results become explicit region arguments/results. Original SSA values
are retained for the independent input binder and source oracle.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass


@dataclass(frozen=True)
class SourceSSARegion:
    module: object
    inputs: tuple
    outputs: tuple
    operations: tuple
    receipt: dict


@dataclass(frozen=True)
class SourceObservations:
    """Additional original SSA results for a caller-owned diagnostic build."""

    module: object
    receipt: dict


def append_source_observations(module, values, *, function="forward", max_elements=1 << 20):
    """Clone a source function and append bounded original tensor observations.

    Existing arguments, operations, result order and callees remain unchanged.
    Returning an intermediate extends its lifetime: this is a distinct diagnostic
    artifact, not evidence that its allocation or timing equals the original.
    No input/reference binding, source admission or execution authority is granted.
    """
    from math import prod

    from xdsl.dialects import func
    from xdsl.dialects.builtin import FunctionType, TensorType
    from xdsl.ir import OpResult

    from merlin.xdsl_dialects._common import text

    if (type(function) is not str or not function or type(values) is not tuple
            or not values or len(values) > 32 or len(set(values)) != len(values)
            or type(max_elements) is not int or max_elements < 1):
        raise ValueError("source observations require distinct bounded original SSA results")
    module.verify()
    before = text(module, generic=True)
    owners = [op for op in module.body.block.ops if type(op) is func.FuncOp
              and op.sym_name.data == function and len(op.body.blocks) == 1]
    if len(owners) != 1:
        raise ValueError("source observations require one original single-block function")
    owner = owners[0]
    block = owner.body.block
    terminator = block.last_op
    if type(terminator) is not func.ReturnOp:
        raise ValueError("source observations require the original return")
    original_results = tuple(terminator.operands)
    positions = {op: i for i, op in enumerate(block.ops)}
    records, elements = [], 0
    for value in values:
        if (not isinstance(value, OpResult) or value.owner.parent is not block
                or value in original_results or not isinstance(value.type, TensorType)):
            raise ValueError("observation is not a distinct top-level original tensor result")
        shape = tuple(value.type.get_shape())
        if any(d < 1 for d in shape):
            raise ValueError("source observations require positive static tensor extents")
        elements += prod(shape)
        if elements > max_elements:
            raise ValueError("source observations exceed the explicit element bound")
        records.append({"operation_ordinal": positions[value.owner],
                        "result_index": value.index, "type": str(value.type)})
    mapping = {}
    observed = module.clone(value_mapper=mapping)
    copied = next(op for op in observed.body.block.ops if type(op) is func.FuncOp
                  and op.sym_name.data == function)
    result_values = (*tuple(copied.body.block.last_op.operands), *(mapping[v] for v in values))
    copied.body.block.erase_op(copied.body.block.last_op)
    copied.body.block.add_op(func.ReturnOp(*result_values))
    copied.properties["function_type"] = FunctionType.from_lists(
        tuple(copied.function_type.inputs), [v.type for v in result_values]
    )
    observed.verify()
    if text(module, generic=True) != before:
        raise ValueError("source observation changed the original module")
    return SourceObservations(observed, {
        "schema": "merlin.original_ssa_observations.v1",
        "source_ir_sha256": hashlib.sha256(before.encode()).hexdigest(),
        "observed_ir_sha256": hashlib.sha256(text(observed, generic=True).encode()).hexdigest(),
        "function": function, "original_result_count": len(original_results),
        "observations": records, "observed_elements": elements,
        "original_operation_order_preserved": True,
        "original_results_preserved": True, "intermediate_lifetimes_extended": True,
        "effect_or_target_admission": False, "source_closure_verified": False,
        "target_execution": False, "timing_equivalence": False,
    })


def extract_source_region(
    module,
    operations,
    *,
    symbol="forward",
    capture_constants=False,
    capture_views=False,
    max_constant_bytes=1 << 20,
    capture_literal_splats=False,
    max_literal_splat_elements=1 << 20,
):
    """Clone one complete contiguous current block span, retaining every observer.

    This does not move computations, infer pure calls, select device groups or
    replace an epilogue. Direct callees retain their original definitions and
    declarations recursively. Unknown symbol-bearing operations are refused
    rather than silently importing a different global/runtime environment.

    Explicit literal-splat capture keeps original constant accumulator seeds in
    the source instead of turning them into unproved runtime inputs. Only the
    original scalar arith.constant -> tensor.splat chain qualifies. Its complete
    static logical extent is independently bounded; this is a source-analysis
    budget, not storage, alias, or device admission.
    """
    from xdsl.dialects import arith, func
    from xdsl.dialects.builtin import ModuleOp, SymbolRefAttr, UnitAttr
    from xdsl.ir import Operation
    from xdsl.traits import IsTerminator

    from merlin.xdsl_dialects._common import text
    from merlin.xdsl_dialects.lowering.outline import _free_values

    if (
        type(symbol) is not str
        or not symbol.isascii()
        or not symbol.isidentifier()
        or type(operations) is not tuple
        or not operations
        or len(set(operations)) != len(operations)
        or any(not isinstance(op, Operation) for op in operations)
    ):
        raise ValueError("source extraction requires a named immutable nonempty operation span")
    if (
        type(capture_constants) is not bool
        or type(capture_views) is not bool
        or type(max_constant_bytes) is not int
        or max_constant_bytes < 1
        or type(capture_literal_splats) is not bool
        or capture_literal_splats
        and not capture_constants
        or type(max_literal_splat_elements) is not int
        or max_literal_splat_elements < 1
    ):
        raise ValueError("source literal capture requires a Boolean choice and positive byte bound")
    module.verify()
    before = text(module, generic=True)
    block = operations[0].parent
    if block is None or any(op.parent is not block for op in operations):
        raise ValueError("source extraction requires one current block")
    current = tuple(block.ops)
    first, last = current.index(operations[0]), current.index(operations[-1])
    if current[first : last + 1] != operations or any(
        op.has_trait(IsTerminator, value_if_unregistered=False) for op in operations
    ):
        raise ValueError("source extraction requires a contiguous span without its source terminator")
    owner = operations[0].parent_op()
    while owner is not None and not isinstance(owner, func.FuncOp):
        owner = owner.parent_op()
    if owner is None:
        raise ValueError("source span has no original function owner")
    from merlin.transforms.tensor_views import _key as tensor_view_key

    prefix, seen, literal_splats = [], set(), set()
    splat_elements = 0

    def capture(value):
        nonlocal splat_elements
        op = value.owner
        if not isinstance(op, Operation) or op in seen or op in operations:
            return
        literal = capture_constants and type(op) is arith.ConstantOp
        if capture_literal_splats and op.name == "tensor.splat" and len(op.operands) == len(op.results) == 1:
            from xdsl.dialects.builtin import TensorType

            source = op.operands[0].owner
            result_type = op.results[0].type
            if (
                type(source) is arith.ConstantOp
                and not isinstance(op.operands[0].type, TensorType)
                and isinstance(result_type, TensorType)
                and op.operands[0].type == result_type.element_type
            ):
                import math

                shape = result_type.get_shape()
                if any(d < 0 for d in shape):
                    raise ValueError("literal splat capture requires complete static extents")
                splat_elements += math.prod(shape)
                if splat_elements > max_literal_splat_elements:
                    raise ValueError("original literal splats exceed the logical element bound")
                literal = True
                literal_splats.add(op)
        view = capture_views and tensor_view_key(op) is not None
        if not literal and not view:
            return
        seen.add(op)
        for operand in op.operands:
            capture(operand)
        prefix.append(op)

    for value in _free_values(operations):
        capture(value)
    constants = tuple(op for op in prefix if type(op) is arith.ConstantOp or op in literal_splats)
    if sum(len(text(op, generic=True).encode()) for op in constants) > max_constant_bytes:
        raise ValueError("original literal dependencies exceed the source capture byte bound")
    inputs = tuple(_free_values((*prefix, *operations)))
    members = {op for top in operations for op in top.walk()}
    outputs = tuple(v for op in operations for v in op.results if any(u.operation not in members for u in v.uses))
    if not outputs:
        raise ValueError("source span has no externally observed result")
    declarations = {op.sym_name.data: op for op in module.body.block.ops if isinstance(op, func.FuncOp)}
    callees, pending = {}, list(operations)
    while pending:
        for op in pending.pop().walk():
            refs = [
                value
                for value in (*op.attributes.values(), *op.properties.values())
                if isinstance(value, SymbolRefAttr)
            ]
            if isinstance(op, func.CallOp):
                if len(refs) != 1 or refs[0] != op.callee or op.callee.nested_references:
                    raise ValueError("source extraction requires a direct original callee")
                name = op.callee.root_reference.data
                if name == owner.sym_name.data or name == symbol or name not in declarations:
                    raise ValueError("source extraction has unresolved or recursive original linkage")
                if name not in callees:
                    callees[name] = declarations[name]
                    pending.append(declarations[name])
            elif refs:
                raise ValueError("source extraction cannot infer a symbol/global closure")
    from xdsl.ir import Block, Region

    body = Block(arg_types=[value.type for value in inputs])
    mapping = dict(zip(inputs, body.args, strict=True))
    for op in (*prefix, *operations):
        cloned = op.clone(value_mapper=mapping)
        body.add_op(cloned)
        mapping.update(zip(op.results, cloned.results, strict=True))
    body.add_op(func.ReturnOp(*(mapping[value] for value in outputs)))
    entry = func.FuncOp(symbol, ([v.type for v in inputs], [v.type for v in outputs]), Region(body))
    entry.attributes["llvm.emit_c_interface"] = UnitAttr()
    region = ModuleOp([entry, *(op.clone() for op in callees.values())])
    region.verify()
    if text(module, generic=True) != before:
        raise ValueError("source extraction changed the original graph")
    return SourceSSARegion(
        region,
        inputs,
        outputs,
        operations,
        {
            "schema": "merlin.current_ssa_source_region.v1",
            "source_ir_sha256": hashlib.sha256(before.encode()).hexdigest(),
            "source_owner": owner.sym_name.data,
            "source_operation_ordinals": list(range(first, last + 1)),
            "region_ir_sha256": hashlib.sha256(text(region, generic=True).encode()).hexdigest(),
            "captured_original_view_dependencies": [
                hashlib.sha256(text(op, generic=True).encode()).hexdigest() for op in prefix if op not in constants
            ],
            "captured_original_literal_dependencies": [
                hashlib.sha256(text(op, generic=True).encode()).hexdigest() for op in constants
            ],
            "input_types": [str(value.type) for value in inputs],
            "output_types": [str(value.type) for value in outputs],
            "input_binding": "original free SSA values in first-use order",
            "output_binding": "all externally observed original member results in source order",
            "operation_order_preserved": True,
            "original_graph_preserved": True,
            "effect_or_target_admission": False,
            "source_closure_verified": False,
            "target_execution": False,
        },
    )


@dataclass(frozen=True)
class SiblingContractionOutline:
    module: object
    core: object
    receipt: dict


def outline_sibling_contractions(module, roots, *, symbol="source_shared_core", max_source_elements=64 << 20):
    """Outline actual independent zero-seeded roots; preserve original epilogues.

    The normal immutable-tensor purity checker must admit every preparation
    moved before the first root. This makes no statement about physical loads,
    device calls or external effects. The returned host still defines its pure
    source callee. A separate target/ownership admission must replace it before
    any device executable can be claimed.
    """
    from merlin.xdsl_dialects._common import text

    module.verify()
    if type(roots) is not tuple or not 2 <= len(roots) <= 8 or len(set(roots)) != len(roots):
        raise ValueError("source siblings require two to eight distinct current roots")
    if type(symbol) is not str or not symbol.isascii() or not symbol.isidentifier():
        raise ValueError("source siblings require an explicit source symbol")
    walked = tuple(module.walk())
    if any(op not in walked for op in roots):
        raise ValueError("source sibling root is absent from the current module")
    before = text(module, generic=True)
    host = module.clone()
    cloned_walk = tuple(host.walk())
    selected = tuple(cloned_walk[walked.index(op)] for op in roots)
    plan = _plan_sibling_contractions(host, selected, symbol=symbol, max_source_elements=max_source_elements)
    _apply_sibling_contractions(plan)
    host.verify()
    if text(module, generic=True) != before:
        raise ValueError("source sibling outlining changed the original input module")
    return SiblingContractionOutline(
        host,
        plan.core,
        {
            "schema": "merlin.source_sibling_contraction_outline.v1",
            "source_ir_sha256": hashlib.sha256(before.encode()).hexdigest(),
            "host_ir_sha256": hashlib.sha256(text(host, generic=True).encode()).hexdigest(),
            "core_ir_sha256": hashlib.sha256(text(plan.core, generic=True).encode()).hexdigest(),
            **plan.receipt,
            "same_lhs_ssa": True,
            "original_epilogues_and_observers_preserved": True,
            "original_full_source_unchanged": True,
            "host_still_defines_original_integer_core": True,
            "target_implementation_or_effect_admission": False,
            "target_execution": False,
        },
    )


@dataclass(frozen=True)
class _SiblingContractionPlan:
    """An uncommitted rewrite of one caller-owned graph; no source admission."""

    host: object
    block: object
    selected: tuple
    inputs: tuple
    outputs: tuple
    hoisted: tuple
    callee: object
    core: object
    receipt: dict
    erased: tuple = ()
    relayouts: tuple = ()
    prepared_inputs: tuple = ()



def _sibling_tensor_operation_is_pure(op):
    """Keep source quantization markers while checking their typed lowering.

    The existing quant_ext lowering owns these exact software semantics. A
    detached local shadow establishes the same closed value-only operations;
    no marker, original use list or unrelated source group is rewritten here.
    Unknown fields/effects still refuse. This is not target readout permission.
    """
    from xdsl.dialects.builtin import ModuleOp, UnregisteredOp
    from .passes_xdsl import _DEQUANT_KINDS, _QUANT_KINDS, _dead_tensor_operation_is_pure, lower_quant_ext
    if _dead_tensor_operation_is_pure(op):
        return True
    name = getattr(getattr(op, "op_name", None), "data", None)
    allowed = {"quant_min", "quant_max", "output_dtype"}
    if name in _DEQUANT_KINDS:
        allowed |= {"axis", "group_size"}
    if (not isinstance(op, UnregisteredOp) or name not in (*_DEQUANT_KINDS, *_QUANT_KINDS)
            or op.regions or op.successors
            or any(key != "op_name__" and not key.startswith("prov.") for key in op.attributes)
            or set(op.properties) - allowed):
        return False
    cloned = op.clone()
    shadow = ModuleOp([cloned])
    try:
        if lower_quant_ext(shadow) != 1:
            return False
        return all(_dead_tensor_operation_is_pure(child) for child in shadow.body.block.ops)
    except (ValueError, TypeError, KeyError, IndexError):
        return False
    finally:
        shadow.drop_all_references()
        cloned.drop_all_references()


def _plan_sibling_contractions(host, selected, *, symbol, max_source_elements=64 << 20):
    """Check every refusal before editing an owned graph.

    The public single-outline and component-batch owners snapshot and verify
    their original source. Keeping this plan local avoids cloning and printing
    a full model for each disjoint component of that same transaction.
    """
    from xdsl.dialects import arith, func, tensor
    from xdsl.dialects.builtin import ModuleOp
    from xdsl.ir import Block, Region

    from merlin.compile.scheduling.dense_source import read_zero_seeded_integer_matmul
    from merlin.xdsl_dialects.lowering.outline import _free_values

    if type(selected) is not tuple or not 2 <= len(selected) <= 8 or len(set(selected)) != len(selected):
        raise ValueError("source siblings require two to eight distinct current roots")
    if type(symbol) is not str or not symbol.isascii() or not symbol.isidentifier():
        raise ValueError("source siblings require an explicit source symbol")
    for root in selected:
        owner = root
        while owner is not None and owner is not host:
            owner = owner.parent_op()
        if owner is not host:
            raise ValueError("source sibling root is absent from the current module")
    if any(getattr(op, "sym_name", None) is not None and op.sym_name.data == symbol for op in host.body.block.ops):
        raise ValueError("source sibling callee symbol already exists")
    block = selected[0].parent
    if block is None or any(op.parent is not block for op in selected):
        raise ValueError("source siblings require one current sequential block")
    current = tuple(block.ops)
    positions = {op: i for i, op in enumerate(current)}
    if tuple(sorted(selected, key=positions.get)) != selected:
        raise ValueError("source siblings must retain original root order")
    geometries = tuple(read_zero_seeded_integer_matmul(op, max_source_elements=max_source_elements) for op in selected)
    lhs = selected[0].operands[0]
    if any(
        op.operands[0] is not lhs or shape[:2] != geometries[0][:2]
        for op, shape in zip(selected, geometries, strict=True)
    ):
        raise ValueError("source siblings require identical current LHS SSA and reduction geometry")
    anchor = selected[0]
    hoisted, pending = set(), set()

    def preparation(value):
        from xdsl.ir import Operation

        op = value.owner
        if not isinstance(op, Operation) or op.parent is not block or positions[op] < positions[anchor]:
            return
        if op in selected or op in pending:
            raise ValueError("source sibling preparation depends on a selected result or cycle")
        if op in hoisted:
            return
        if not _sibling_tensor_operation_is_pure(op):
            raise ValueError("source sibling preparation has no closed immutable tensor semantics")
        pending.add(op)
        for operand in _free_values((op,)):
            preparation(operand)
        pending.remove(op)
        hoisted.add(op)

    for op in selected:
        for operand in op.operands:
            preparation(operand)
    # Check the entire crossed range as well: an unknown effect cannot be
    # crossed merely because the moved operands are immutable tensors.
    for op in current[positions[anchor] : positions[selected[-1]] + 1]:
        if op not in selected and not _sibling_tensor_operation_is_pure(op):
            raise ValueError("source sibling outline crosses an unproved operation effect")
    core_ops = []
    for op in selected:
        seed = op.operands[-1].owner
        if type(seed) is not tensor.SplatOp or len(seed.operands) != 1:
            raise ValueError("source sibling initializer must be its original exact zero splat")
        literal = seed.operands[0].owner
        if type(literal) is not arith.ConstantOp or literal.value.value.data != 0:
            raise ValueError("source sibling initializer is not an original literal zero")
        core_ops.extend(item for item in (literal, seed, op) if item not in core_ops)
    inputs = tuple(_free_values(tuple(core_ops)))
    outputs = tuple(op.results[0] for op in selected)
    if inputs != (lhs, *(op.operands[1] for op in selected)):
        raise ValueError("source sibling core has an unaccounted input or aliased member weight")
    body = Block(arg_types=[value.type for value in inputs])
    mapping = dict(zip(inputs, body.args, strict=True))
    for op in core_ops:
        cloned = op.clone(value_mapper=mapping)
        body.add_op(cloned)
        mapping.update(zip(op.results, cloned.results, strict=True))
    body.add_op(func.ReturnOp(*(mapping[value] for value in outputs)))
    callee = func.FuncOp(symbol, ([v.type for v in inputs], [v.type for v in outputs]), Region(body))
    core = ModuleOp([callee.clone()])
    core.verify()
    return _SiblingContractionPlan(
        host, block, selected, inputs, outputs, tuple(sorted(hoisted, key=positions.get)), callee, core,
        {
            "source_root_ordinals": [positions[op] for op in selected],
            "source_geometries": [list(shape) for shape in geometries],
            "source_hoisted_preparation_ordinals": sorted(positions[op] for op in hoisted),
        },
    )


def _apply_sibling_contractions(plan):
    """Commit a checked local plan; the owning transaction verifies the graph."""
    from xdsl.dialects import func

    host, block = plan.host, plan.block
    anchor = plan.selected[0]
    for op in plan.hoisted:
        block.detach_op(op)
        block.insert_op_before(op, anchor)
    for op in plan.prepared_inputs:
        block.insert_op_before(op, anchor)
    call = func.CallOp(plan.callee.sym_name.data, list(plan.inputs), list(plan.callee.function_type.outputs))
    block.insert_op_before(call, anchor)
    after = call
    for index, (original, result) in enumerate(zip(plan.outputs, call.results, strict=True)):
        relayout = plan.relayouts[index] if plan.relayouts else None
        for view in relayout(result) if relayout is not None else ():
            block.insert_op_after(view, after)
            after = view
            result = view.results[0]
        original.replace_all_uses_with(result)
    for op in reversed(plan.erased or plan.selected):
        block.erase_op(op)
    host.body.block.add_op(plan.callee)
