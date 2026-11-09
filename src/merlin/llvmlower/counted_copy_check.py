"""Closed static copy theorem over original tensor IR and emitted LLVM SSA.

This checks a complete counted scalar loop without unrolling or allocating
shaped data. Its theorem is conditional on explicitly selected original
software pointer objects. Machine code, physical allocation/resources/lifetime,
runtime equivalence and arbitrary compiler transformations remain unproved.
"""

from __future__ import annotations

from xdsl.dialects import builtin, func, llvm, tensor
from xdsl.dialects.linalg import ops as linalg
from xdsl.parser import Parser
from xdsl.utils.exceptions import ParseError, VerifyException
from xdsl.utils.lexer import Input
from xdsl.utils.mlir_lexer import MLIRLexer, MLIRTokenKind

from merlin.targetgen.contract.linalg_iface import make_linalg_context
from merlin.targetgen.contract.pointer_storage import OriginalPointerStorageContract
from merlin.targetgen.oot_starterkit.llvm_context import make_llvm_context

from .layout_observation import LLVMLayoutObservation

_FACETS = ("semantic_coverage", "index_bounds", "complete_output_coverage")


class _Unknown(ValueError):
    pass


class _Refuted(ValueError):
    pass


def _require(condition, detail, *, refuted=False):
    if not condition:
        raise (_Refuted if refuted else _Unknown)(detail)


def _integer_type(typ, bits):
    return (
        type(typ) is builtin.IntegerType
        and typ.width.data == bits
        and typ.signedness.data == builtin.Signedness.SIGNLESS
    )


def _parse(context, text):
    # Upstream dense splat parsing expands shaped storage before operation
    # verification. This closed scalar proof supports no aggregate literals.
    # Refuse those actual lexer tokens before parsing; do not rewrite the IR.
    lexer, previous = MLIRLexer(Input(text, "static-copy-source")), None
    while True:
        token = lexer.lex()
        if (
            previous is not None
            and previous.kind is MLIRTokenKind.BARE_IDENT
            and previous.text in {"dense", "dense_resource"}
            and token.kind is MLIRTokenKind.LESS
        ):
            raise _Unknown("aggregate literals are unsupported before shaped parser allocation")
        if token.kind is MLIRTokenKind.EOF:
            break
        previous = token
    module = Parser(context, text).parse_module()
    module.verify()
    return module


def _source_copy(text, storage):
    module = _parse(make_linalg_context(), text)
    _require(not module.properties, "original module carries unsupported properties")
    _require(set(module.attributes) <= {"prov.level"}, "original module carries unsupported semantic attributes")
    _require(len(module.body.block.ops) == 1, "original source has unsupported extra top-level operations")
    source = module.body.block.first_op
    _require(
        type(source) is func.FuncOp and len(source.body.blocks) == 1,
        "original source is not one defined tensor function",
    )
    _props(source, ("sym_name", "function_type"))
    slots = storage.slots
    _require(
        len(storage.original_abi.inputs) == len(storage.original_abi.outputs) == 1,
        "copy checker requires the complete single-input/single-output original roster",
    )
    _require(
        slots[0].shape == slots[1].shape and slots[0].element_bits == slots[1].element_bits,
        "original copy input/output geometry differs",
    )
    types = (*source.function_type.inputs, *source.function_type.outputs)
    _require(len(types) == 2, "original tensor function does not cover the declared complete ABI", refuted=True)
    for typ, slot in zip(types, slots, strict=True):
        _require(
            type(typ) is builtin.TensorType
            and typ.get_shape() == slot.shape
            and type(typ.encoding) is builtin.NoneAttr
            and _integer_type(typ.element_type, slot.element_bits),
            "original tensor type differs from the independently declared storage",
            refuted=True,
        )
    block = source.body.block
    ops = tuple(block.ops)
    _require(
        len(block.args) == 1 and type(ops[-1]) is func.ReturnOp and len(ops[-1].arguments) == 1,
        "original source lacks one complete tensor return",
    )
    _props(ops[-1], ())
    if len(ops) == 1:
        _require(ops[0].arguments[0] is block.args[0], "original return is not the complete input tensor")
        return
    _require(
        len(ops) == 3 and type(ops[0]) is tensor.EmptyOp and type(ops[1]) is linalg.CopyOp,
        "original source is outside the closed identity/copy semantics",
    )
    empty, copy, ret = ops
    _props(empty, ())
    _props(copy, ("operandSegmentSizes",))
    _require(
        not empty.operands
        and len(empty.results) == len(copy.results) == 1
        and tuple(copy.inputs) == (block.args[0],)
        and tuple(copy.outputs) == (empty.results[0],)
        and ret.arguments[0] is copy.results[0]
        and empty.results[0].type == types[1]
        and copy.results[0].type == types[1],
        "original copy does not produce the complete returned tensor",
    )
    region = copy.regions[0]
    _require(
        len(region.blocks) == 1 and len(region.block.args) == 2 and len(region.block.ops) == 1,
        "original copy contains an unsupported scalar body",
    )
    yielded = region.block.first_op
    _props(yielded, ())
    _require(
        tuple(arg.type for arg in region.block.args) == tuple(typ.element_type for typ in types),
        "original scalar copy body types differ from the complete tensor ABI",
    )
    _require(
        type(yielded) is linalg.YieldOp and tuple(yielded.operands) == (region.block.args[0],),
        "original scalar copy body does not yield its exact input",
    )


def _props(operation, allowed):
    _require(
        not operation.attributes and set(operation.properties) <= set(allowed),
        f"{operation.name} carries unsupported semantics or metadata",
    )


def _plain_flag(operation, name):
    value = operation.properties.get(name)
    _require(
        value is None or isinstance(value, builtin.IntegerAttr) and value.value.data == 0,
        f"{operation.name} has unsupported {name}",
    )


def _address(operation, *, base, iv, element_bits):
    _require(isinstance(operation, llvm.GEPOp), "loop has an unsupported address operation")
    _props(operation, ("elem_type", "rawConstantIndices", "noWrapFlags", "inbounds"))
    _plain_flag(operation, "noWrapFlags")
    _require(operation.properties.get("inbounds") is None, "inbounds pointer/provenance semantics remain unproved")
    _require(
        operation.ptr is base
        and tuple(operation.ssa_indices) == (iv,)
        and tuple(operation.rawConstantIndices.iter_values()) == (llvm.GEP_USE_SSA_VAL,)
        and _integer_type(operation.elem_type, element_bits),
        "emitted address does not index the original scalar element at the loop ordinal",
        refuted=True,
    )
    _require(
        type(operation.results[0].type) is llvm.LLVMPointerType
        and type(operation.results[0].type.addr_space) is builtin.NoneAttr,
        "address space or pointer ABI is unsupported",
    )


def _memory(operation, *, stride, alignment):
    allowed = ("ordering", "alignment")
    _props(operation, allowed)
    _plain_flag(operation, "ordering")
    assertion = operation.properties.get("alignment")
    _require(type(assertion) is builtin.IntegerAttr, "memory access has no explicit alignment assertion")
    asserted = assertion.value.data
    _require(
        asserted > 0 and asserted & (asserted - 1) == 0, "emitted memory alignment assertion is invalid", refuted=True
    )
    _require(
        alignment % asserted == 0 and stride % asserted == 0,
        "emitted memory alignment exceeds the original object/element guarantee",
        refuted=True,
    )


def _emitted_copy(text, storage, observed, entry_symbol):
    module = _parse(make_llvm_context(), text)
    _require(not module.properties, "LLVM module carries unsupported properties")
    _require(
        set(module.attributes) <= {"llvm.data_layout", "llvm.target_triple", "prov.level"},
        "LLVM module carries unsupported semantic attributes",
    )
    _require(len(module.body.block.ops) == 1, "LLVM module has unsupported extra definitions/declarations")
    function = module.body.block.first_op
    _require(
        isinstance(function, llvm.FuncOp) and function.sym_name.data == entry_symbol,
        "LLVM module lacks the exact selected pointer entry",
    )
    _props(function, ("sym_name", "function_type", "CConv", "linkage", "unnamed_addr", "visibility_"))
    _plain_flag(function, "unnamed_addr")
    _plain_flag(function, "visibility_")
    _require(
        not function.function_type.is_variadic
        and type(function.function_type.output) is llvm.LLVMVoidType
        and function.CConv.convention.data == "ccc"
        and function.linkage.linkage.data == "external"
        and len(function.function_type.inputs) == 2
        and all(
            type(typ) is llvm.LLVMPointerType and type(typ.addr_space) is builtin.NoneAttr
            for typ in function.function_type.inputs
        ),
        "emitted function is outside the selected original pointer ABI",
    )
    _require(len(function.body.blocks) == 3, "only one complete post-tested counted CFG is supported")
    entry, loop, exit_block = tuple(function.body.blocks)
    _require(
        len(entry.args) == 2 and len(loop.args) == 1 and not exit_block.args,
        "loop carries unsupported pointer/index state",
    )
    initial = tuple(entry.ops)
    _require(
        len(initial) == 4
        and all(type(op) is llvm.ConstantOp for op in initial[:3])
        and isinstance(initial[-1], llvm.BrOp),
        "entry does not contain the exact constant counted-loop initialization",
    )
    iv = loop.args[0]
    _require(type(iv.type) is builtin.IntegerType, "loop ordinal is not an integer")
    bits, count, element_bits = iv.type.width.data, storage.slots[0].element_count, storage.slots[0].element_bits
    _require(
        bits == observed.index_bits and observed.index_bits <= observed.pointer_bits,
        "loop ordinal width differs from the actual selected pointer-index layout",
    )
    # Use signed bounds conservatively for all accepted comparisons and GEPs.
    # This prevents induction/add truncation, signed index reinterpretation and
    # element-byte offset wrap for every ordinal without exploring iterations.
    _require(
        count <= (1 << (bits - 1)) - 1 and storage.slots[0].byte_extent <= (1 << (bits - 1)) - 1,
        "original element/byte extent cannot fit the selected signed index arithmetic",
        refuted=True,
    )
    constants = {}
    for op in initial[:3]:
        _props(op, ("value",))
        value = op.properties.get("value")
        _require(
            type(value) is builtin.IntegerAttr and _integer_type(op.results[0].type, bits),
            "loop initialization contains an unsupported constant type",
        )
        constants[op.results[0]] = value.value.data
    branch = initial[-1]
    _props(branch, ())
    _require(
        branch.successors[0] is loop and len(branch.arguments) == 1 and constants.get(branch.arguments[0]) == 0,
        "loop does not begin at original ordinal zero",
        refuted=True,
    )
    body = tuple(loop.ops)
    _require(
        len(body) == 7
        and isinstance(body[2], llvm.LoadOp)
        and isinstance(body[3], llvm.StoreOp)
        and type(body[4]) is llvm.AddOp
        and isinstance(body[5], llvm.ICmpOp)
        and isinstance(body[6], llvm.CondBrOp),
        "loop has unsupported operations, control flow or effects",
    )
    src, dst, load, store, increment, compare, cond = body
    _address(src, base=entry.args[0], iv=iv, element_bits=element_bits)
    _address(dst, base=entry.args[1], iv=iv, element_bits=element_bits)
    _memory(load, stride=observed.allocation_stride, alignment=storage.policy.tensor_alignment)
    _memory(store, stride=observed.allocation_stride, alignment=storage.policy.tensor_alignment)
    _require(
        load.ptr is src.results[0]
        and _integer_type(load.results[0].type, element_bits)
        and store.ptr is dst.results[0]
        and store.value is load.results[0],
        "emitted store is not the complete original input element",
        refuted=True,
    )
    _props(increment, ("overflowFlags",))
    _plain_flag(increment, "overflowFlags")
    _require(
        increment.operands[0] is iv and constants.get(increment.operands[1]) == 1,
        "loop does not advance exactly one original ordinal",
        refuted=True,
    )
    _props(compare, ("predicate",))
    _require(
        compare.operands[0] is increment.results[0] and constants.get(compare.operands[1]) == count,
        "loop stop bound does not cover the complete original output",
        refuted=True,
    )
    predicate = llvm.ICmpPredicateFlag.from_int(compare.predicate.value.data)
    _props(cond, ("operandSegmentSizes",))
    _require(cond.cond is compare.results[0], "loop branch does not consume its actual stop predicate", refuted=True)
    if predicate == llvm.ICmpPredicateFlag.EQ:
        terminated, again, exit_args, loop_args = (
            cond.then_block,
            cond.else_block,
            cond.then_arguments,
            cond.else_arguments,
        )
    elif predicate in (llvm.ICmpPredicateFlag.NE, llvm.ICmpPredicateFlag.SLT, llvm.ICmpPredicateFlag.ULT):
        again, terminated, loop_args, exit_args = (
            cond.then_block,
            cond.else_block,
            cond.then_arguments,
            cond.else_arguments,
        )
    else:
        raise _Unknown("loop uses an unsupported stop comparison")
    _require(
        terminated is exit_block and again is loop and not exit_args and tuple(loop_args) == (increment.results[0],),
        "loop control does not visit each original ordinal once",
        refuted=True,
    )
    final = tuple(exit_block.ops)
    _require(
        len(final) == 1 and type(final[0]) is llvm.ReturnOp and not final[0].operands,
        "exit contains unsupported effects or results",
    )
    _props(final[0], ())
    return {
        "iterations": count,
        "ordinal_interval": [0, count - 1],
        "byte_interval": [0, storage.slots[0].byte_extent - 1],
        "element_bits": element_bits,
    }


def check_counted_copy(*, original_source, emitted_llvm, storage, layout_observation, entry_symbol):
    """Derive limited static facets; unsupported representations retain UNKNOWN.

    Typed declarations are not independent provenance. The experiment owner
    must bind the original storage/source and ordinary compile artifacts before
    using these conditional theorem facets in its required denominator.
    """
    facts = None
    try:
        _require(type(storage) is OriginalPointerStorageContract, "original pointer storage is unavailable")
        storage.record()
        _require(
            type(layout_observation) is LLVMLayoutObservation, "actual native LLVM layout observation is unavailable"
        )
        layout_observation.verify()
        _require(
            layout_observation.integer_bits == storage.slots[0].element_bits,
            "layout queried a different original element width",
        )
        _require(
            layout_observation.allocation_stride == storage.slots[0].element_bits // 8,
            "actual LLVM element allocation stride differs from the original dense ABI",
            refuted=True,
        )
        _require(
            layout_observation.byte_order == storage.policy.byte_order,
            "selected compiler byte order differs from the original software ABI",
            refuted=True,
        )
        _source_copy(original_source, storage)
        facts = _emitted_copy(emitted_llvm, storage, layout_observation, entry_symbol)
        status, detail = "PROVED", "complete original scalar copy under explicit original pointer object preconditions"
    except _Refuted as error:
        status, detail = "REFUTED", str(error)
    except (ValueError, TypeError, IndexError, AttributeError, AssertionError, ParseError, VerifyException) as error:
        status, detail = "UNKNOWN", str(error)
    return {
        "schema": "merlin.counted_copy_static_check.v1",
        "status": status,
        "facets": {name: status for name in _FACETS},
        "detail": detail,
        "facts": facts,
        "scope": (
            "original tensor to emitted LLVM; resources, physical storage/lifetime, "
            "machine-code and runtime equivalence unproved"
        ),
    }
