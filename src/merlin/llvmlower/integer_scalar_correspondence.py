"""Conditional complete integer arith-to-LLVM correspondence, without codegen.

Every accepted body is pure, straight-line and defined under the explicitly
selected modular integer contract. Structural typed DAG equality proves the
ordered results for all bit patterns, rather than testing representative values.
Unsupported transformations retain UNKNOWN; differing DAGs alone are not a
counterexample. This establishes neither machine-code nor runtime authority.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from xdsl.context import Context
from xdsl.dialects import arith, builtin, func, llvm
from xdsl.parser import Parser
from xdsl.utils.exceptions import ParseError, VerifyException

from merlin.common.strict_json import loads
from merlin.targetgen.contract.mlir_source_admission import admit_mlir_source
from merlin.targetgen.oot_starterkit.llvm_context import make_llvm_context

from .integer_scalar_contract import IntegerScalarLimits, OriginalIntegerScalarSource
from .llvm_dialect_product import verify_llvm_dialect_product


def _require(condition, detail):
    if not condition:
        raise ValueError(detail)


def _read(path, maximum):
    path = Path(path).absolute()
    _require(
        path.resolve() == path and not any(item.is_symlink() for item in (path, *path.parents)),
        "scalar source/products require canonical ordinary files",
    )
    with path.open("rb") as stream:
        raw = stream.read(maximum + 1)
    _require(len(raw) <= maximum, "scalar source/product exceeds its selected byte limit")
    return raw


def _parse(raw, limits, *, emitted=False):
    text = raw.decode("utf-8")
    admit_mlir_source(
        text,
        max_source_bytes=limits.source_bytes,
        max_nesting=limits.nesting,
        max_integer_bits=limits.integer_bits,
        allow_dense=False,
        allow_dense_resource=False,
    )
    context = make_llvm_context() if emitted else Context()
    if not emitted:
        for dialect in (builtin.Builtin, func.Func, arith.Arith):
            context.load_dialect(dialect)
    module = Parser(context, text).parse_module()
    module.verify()
    _require(not module.attributes and not module.properties, "scalar module metadata is unsupported")
    return module


def _width(typ, limits):
    _require(
        type(typ) is builtin.IntegerType
        and typ.signedness.data is builtin.Signedness.SIGNLESS
        and 0 < typ.width.data <= limits.integer_bits,
        "only bounded signless integer scalar types are supported",
    )
    return typ.width.data


def _properties(op, allowed=()):
    _require(
        not op.attributes and not op.regions and not op.successors and set(op.properties) <= set(allowed),
        "unsupported scalar operation semantics: " + op.name,
    )


def _empty_flag(value, *, overflow=False):
    if value is None:
        return
    if overflow and type(value) is builtin.IntegerAttr:
        _require(value.type == builtin.i32 and value.value.data == 0, "integer overflow promises are unsupported")
    else:
        classes = (arith.IntegerOverflowAttr, llvm.OverflowAttr) if overflow else (llvm.FastMathAttr,)
        _require(type(value) in classes and not value.data, "nonempty scalar flags are unsupported")


class _Dags:
    def __init__(self):
        self.nodes = {}

    def intern(self, key):
        if key not in self.nodes:
            self.nodes[key] = len(self.nodes)
        return self.nodes[key]


_ARITHMETIC = {
    arith.AddiOp: "add",
    arith.SubiOp: "sub",
    arith.MuliOp: "mul",
    arith.AndIOp: "and",
    arith.OrIOp: "or",
    arith.XOrIOp: "xor",
    llvm.AddOp: "add",
    llvm.SubOp: "sub",
    llvm.MulOp: "mul",
    llvm.AndOp: "and",
    llvm.OrOp: "or",
    llvm.XOrOp: "xor",
}


def _dag(function, *, inputs, outputs, limits, dags, emitted):
    _require(len(function.body.blocks) == 1, "scalar control flow/loops are unsupported")
    block = function.body.block
    operations = tuple(block.ops)
    _require(0 < len(operations) <= limits.operations, "scalar operation roster exceeds its selected limit")
    _require(
        tuple(_width(value.type, limits) for value in block.args) == inputs,
        "scalar argument types/order differ from the complete original ABI",
    )
    values = {
        value: dags.intern(("argument", ordinal, bits))
        for ordinal, (value, bits) in enumerate(zip(block.args, inputs, strict=True))
    }
    for index, op in enumerate(operations):
        _require(all(value in values for value in op.operands), "scalar operand lacks a preceding definition")
        args = tuple(values[value] for value in op.operands)
        if type(op) is (llvm.ReturnOp if emitted else func.ReturnOp):
            _properties(op)
            _require(
                index == len(operations) - 1 and not op.results, "scalar return is not the final complete boundary"
            )
            _require(
                tuple(_width(value.type, limits) for value in op.operands) == outputs,
                "scalar return omits/changes original ordered output slots",
            )
            return args
        _require(len(op.results) == 1, "scalar DAG operation has unsupported result arity")
        width = _width(op.results[0].type, limits)
        if type(op) in {arith.ConstantOp, llvm.ConstantOp}:
            _properties(op, ("value",))
            constant = op.properties.get("value")
            _require(
                not args and type(constant) is builtin.IntegerAttr and constant.type == op.results[0].type,
                "scalar constant does not retain its exact typed integer value",
            )
            key = ("constant", width, constant.value.data % (1 << width))
        elif type(op) in _ARITHMETIC:
            _properties(op, ("overflowFlags",))
            _empty_flag(op.properties.get("overflowFlags"), overflow=True)
            _require(
                len(args) == 2 and all(value.type == op.results[0].type for value in op.operands),
                "scalar arithmetic loses operand/result width identity",
            )
            key = (_ARITHMETIC[type(op)], width, args)
        elif type(op) in {arith.CmpiOp, llvm.ICmpOp}:
            _properties(op, ("predicate",))
            _require(
                len(args) == 2 and width == 1 and op.operands[0].type == op.operands[1].type,
                "scalar comparison loses typed predicate operands",
            )
            operand_bits = _width(op.operands[0].type, limits)
            predicate = op.properties.get("predicate")
            _require(
                type(predicate) is builtin.IntegerAttr and predicate.type == builtin.i64,
                "scalar comparison lacks an exact integer predicate",
            )
            predicates = arith.CMPI_COMPARISON_OPERATIONS if type(op) is arith.CmpiOp else llvm.ALL_ICMP_FLAGS
            ordinal = predicate.value.data
            _require(0 <= ordinal < len(predicates), "scalar predicate is unsupported")
            key = ("compare", operand_bits, str(predicates[ordinal]), args)
        elif type(op) in {arith.SelectOp, llvm.SelectOp}:
            _properties(op, ("fastmathFlags",) if emitted else ())
            _empty_flag(op.properties.get("fastmathFlags"))
            _require(
                len(args) == 3
                and op.operands[0].type == builtin.i1
                and all(value.type == op.results[0].type for value in op.operands[1:]),
                "scalar selection changes its condition/value types",
            )
            key = ("select", width, args)
        else:
            raise ValueError("unsupported scalar operation: " + op.name)
        values[op.results[0]] = dags.intern(key)
    raise ValueError("scalar body has no complete return")


def _source(module, abi, limits, dags):
    members = tuple(module.body.block.ops)
    _require(
        len(members) == 1 and type(members[0]) is func.FuncOp, "original scalar source needs one complete function"
    )
    function = members[0]
    _require(
        function.sym_name.data == abi.entry_symbol
        and set(function.properties) <= {"sym_name", "function_type"}
        and set(function.attributes) <= {"llvm.emit_c_interface"}
        and all(type(value) is builtin.UnitAttr for value in function.attributes.values()),
        "original scalar function identity/attributes are unsupported",
    )
    inputs, outputs = (tuple(slot.bits for slot in slots) for slots in (abi.inputs, abi.outputs))
    _require(
        tuple(_width(typ, limits) for typ in function.function_type.inputs) == inputs
        and tuple(_width(typ, limits) for typ in function.function_type.outputs) == outputs,
        "original scalar signature differs from its complete ordered public ABI",
    )
    return _dag(function, inputs=inputs, outputs=outputs, limits=limits, dags=dags, emitted=False)


def _function(function, name, inputs, outputs, limits):
    _require(
        isinstance(function, llvm.FuncOp) and function.sym_name.data == name, "emitted scalar entry identity differs"
    )
    _require(
        set(function.properties) <= {"sym_name", "function_type", "CConv", "linkage", "visibility_"}
        and set(function.attributes) <= {"llvm.emit_c_interface"}
        and all(type(value) is builtin.UnitAttr for value in function.attributes.values()),
        "emitted scalar function carries unsupported metadata",
    )
    visibility = function.properties.get("visibility_")
    _require(
        visibility is None or type(visibility) is builtin.IntegerAttr and visibility.value.data == 0,
        "emitted scalar visibility is unsupported",
    )
    typ = function.function_type
    _require(
        not typ.is_variadic and function.CConv.convention.data == "ccc" and function.linkage.linkage.data == "external",
        "scalar calling convention/linkage is unsupported",
    )
    _require(len(outputs) == 1, "aggregate/repeated scalar results remain unsupported; original roster retained")
    _require(
        tuple(_width(value, limits) for value in typ.inputs) == inputs and _width(typ.output, limits) == outputs[0],
        "emitted scalar ABI does not cover original ordered slots",
    )


def _wrapper(wrapper, abi, inputs, outputs, limits):
    _function(wrapper, abi.c_interface_symbol, inputs, outputs, limits)
    _require(len(wrapper.body.blocks) == 1, "scalar C-interface CFG is unsupported")
    block = wrapper.body.block
    _require(
        tuple(_width(value.type, limits) for value in block.args) == inputs,
        "scalar wrapper block arguments differ from the original ABI",
    )
    operations = tuple(block.ops)
    _require(
        len(operations) == 2 and type(operations[0]) is llvm.CallOp and type(operations[1]) is llvm.ReturnOp,
        "scalar C-interface must be one exact call and return",
    )
    call, returned = operations
    _properties(call, ("CConv", "TailCallKind", "callee", "fastmathFlags", "op_bundle_sizes", "operandSegmentSizes"))
    _properties(returned)
    _empty_flag(call.properties.get("fastmathFlags"))
    _require(
        call.CConv.convention.data == "ccc" and call.TailCallKind.data is llvm.TailCallKind.NONE,
        "scalar wrapper call convention/tail semantics are unsupported",
    )
    _require(
        call.callee.root_reference.data == abi.entry_symbol and not tuple(call.callee.nested_references),
        "scalar wrapper calls a different original entry",
    )
    _require(not tuple(call.op_bundle_sizes.iter_values()), "scalar wrapper operand bundles are unsupported")
    _require(
        tuple(call.operands) == tuple(block.args)
        and len(call.results) == 1
        and tuple(returned.operands) == tuple(call.results)
        and not returned.results,
        "scalar wrapper changes ordered argument or returned result identity",
    )
    _require(
        tuple(_width(value.type, limits) for value in call.results) == outputs,
        "scalar wrapper call result differs from the original ABI",
    )


def check_integer_scalar_correspondence(*, original, retained_product, limits):
    """Recompute one source-exact conditional facet, never issue qualification.

    The caller owns independent original source/ABI/numeric selection. Typed
    contracts are declarations, not authority. Every call reopens live products;
    source preparation, candidate applicability and all runtime gates remain
    separate. No tensor, descriptor, pointer or physical ABI is interpreted.
    """
    binding, facts, selected_limits, retained_binding = None, None, None, None
    try:
        _require(type(limits) is IntegerScalarLimits, "explicit scalar reader limits are missing")
        selected_limits = limits.record()
        _require(type(original) is OriginalIntegerScalarSource, "typed original scalar source is missing")
        binding = original.record(limits)
        raw = _read(original.path, limits.source_bytes)
        _require(hashlib.sha256(raw).hexdigest() == original.sha256, "original scalar source bytes changed")
        receipt_raw = _read(retained_product, limits.receipt_bytes)
        retained_binding = {
            "path": str(Path(retained_product).absolute()),
            "sha256": hashlib.sha256(receipt_raw).hexdigest(),
        }
        preliminary = loads(receipt_raw)
        _require(type(preliminary) is dict, "retained scalar product is not an object")
        source_raw = _read(preliminary["source"]["path"], limits.source_bytes)
        module_raw = _read(preliminary["llvm_dialect"]["path"], limits.source_bytes)
        # Bound the receipt owner's other direct products before its full live
        # replay. This limits this reader, not native compiler memory or time.
        for name in ("runner", "translated_llvm_ir", "producer"):
            _read(preliminary[name]["path"], limits.source_bytes)
        _read(preliminary["invocation"]["path"], limits.receipt_bytes)
        verify_llvm_dialect_product(retained_product)
        dags = _Dags()
        source = _source(_parse(raw, limits), original.abi, limits, dags)
        current = _source(_parse(source_raw, limits), original.abi, limits, dags)
        _require(source == current, "original and actual translation-input scalar DAGs differ; equivalence unproved")
        module = _parse(module_raw, limits, emitted=True)
        members = tuple(module.body.block.ops)
        _require(
            len(members) == 2 and all(isinstance(op, llvm.FuncOp) for op in members),
            "complete emitted scalar source and C-interface function roster is unavailable",
        )
        by_name = {op.sym_name.data: op for op in members}
        _require(
            set(by_name) == {original.abi.entry_symbol, original.abi.c_interface_symbol},
            "emitted scalar function roster differs from the original ABI",
        )
        inputs, outputs = (tuple(slot.bits for slot in slots) for slots in (original.abi.inputs, original.abi.outputs))
        function = by_name[original.abi.entry_symbol]
        _function(function, original.abi.entry_symbol, inputs, outputs, limits)
        emitted = _dag(function, inputs=inputs, outputs=outputs, limits=limits, dags=dags, emitted=True)
        _require(source == emitted, "ordered typed source/emitted scalar DAGs differ; equivalence unproved")
        _wrapper(by_name[original.abi.c_interface_symbol], original.abi, inputs, outputs, limits)
        _require(_read(original.path, limits.source_bytes) == raw, "original scalar source changed during checking")
        _require(
            _read(retained_product, limits.receipt_bytes) == receipt_raw,
            "retained scalar receipt changed during checking",
        )
        verify_llvm_dialect_product(retained_product)
        facts = {"input_widths": list(inputs), "output_widths": list(outputs), "typed_nodes": len(dags.nodes)}
        status, detail = "PROVED", "complete ordered modular-integer DAG and exact scalar C-interface call/return"
    except (
        ValueError,
        OSError,
        TypeError,
        KeyError,
        IndexError,
        AttributeError,
        ParseError,
        VerifyException,
        RecursionError,
    ) as error:
        status, detail = "UNKNOWN", str(error)
    return {
        "schema": "merlin.integer_scalar_correspondence.v1",
        "status": status,
        "facet": "integer_scalar_source_to_llvm",
        "original": binding,
        "retained_product": retained_binding,
        "limits": selected_limits,
        "facts": facts,
        "detail": detail,
        "scope": (
            "conditional original IR correspondence only; object/link/ABI realization, "
            "effects, resources and runtime unproved"
        ),
    }
