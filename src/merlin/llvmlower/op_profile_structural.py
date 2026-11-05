"""Typed operation boundaries for the existing, explicitly selected profiler.

Every top-level operation receives an interval, including effect-only calls and
stores. Nested regions are charged to their enclosing operation. Marker calls
can affect optimization and allocation placement; paired uninstrumented timing
remains necessary. This does not infer CPU/device ownership from an op name.
"""

from __future__ import annotations

from io import StringIO

from xdsl.dialects import arith, func
from xdsl.dialects.builtin import ArrayAttr, IntegerAttr, StringAttr, i32
from xdsl.printer import Printer

from merlin.common.mlir_query import op_name


def _parse_text(text):
    from xdsl.dialects.vector import Vector

    from merlin.frontends.linalg_mlir import make_context, parse_mlir_text

    context = make_context()
    # The shared frontend already owns the prepared memref dialect registry.
    context.load_dialect(Vector)
    return parse_mlir_text(text, ctx=context)


def _operation_record(op, mid):
    from .op_profile import _PROV_KEYS, _elem_count

    result_types = [str(value.type) for value in op.results]
    result_type = result_types[0] if result_types else None
    row = {
        "id": mid, "mlir_op": op_name(op), "result_type": result_type,
        "result_types": result_types, "elems": _elem_count(result_type),
        "callee": ("@" + op.callee.root_reference.data if isinstance(op, func.CallOp) else None),
    }
    for key in _PROV_KEYS:
        value = op.attributes.get(key, op.properties.get(key))
        row[key.split(".", 1)[1]] = value.data if isinstance(value, StringAttr) else None
    for key in ("prov.source_node_ids", "prov.origin_node_ids"):
        value = op.attributes.get(key, op.properties.get(key))
        if isinstance(value, ArrayAttr) and all(isinstance(x, StringAttr) for x in value):
            row[key.split(".", 1)[1]] = [x.data for x in value]
    if not row.get("family") and op_name(op) == "linalg.generic":
        row["body_ops"] = sorted({op_name(child) for region in op.regions for child in region.walk()
                                  if op_name(child).startswith(("arith.", "math."))})
    return row


def verify_instrumentation(original, selected, table, *, function="forward"):
    """Reconstruct actual marker boundaries and retain the complete original IR.

    This checks the emitted source, not runtime counter semantics or later
    machine scheduling. Neither input module is mutated.
    """
    from .op_profile import MARK_SYM, OpProfileError

    original.verify()
    selected.verify()
    restored = selected.clone()
    functions = [op for op in restored.body.block.ops
                 if isinstance(op, func.FuncOp) and op.sym_name.data == function]
    sources = [op for op in original.body.block.ops
               if isinstance(op, func.FuncOp) and op.sym_name.data == function]
    declarations = [op for op in restored.body.block.ops
                    if isinstance(op, func.FuncOp) and op.sym_name.data == MARK_SYM]
    if (len(functions) != 1 or len(sources) != 1 or len(declarations) != 1
            or not declarations[0].is_structurally_equivalent(func.FuncOp.external(MARK_SYM, [i32], []))
            or len(functions[0].body.blocks) != 1 or len(sources[0].body.blocks) != 1):
        raise OpProfileError("profile source, selected function or marker ABI differs")
    ops, source_ops = list(functions[0].body.block.ops), list(sources[0].body.block.ops)
    if not source_ops or not isinstance(source_ops[-1], func.ReturnOp) or len(ops) != 3 * len(source_ops):
        raise OpProfileError("profile has incomplete operation boundaries")
    actual_table = []
    block = functions[0].body.block
    for mid in range(len(source_ops)):
        constant, marker, retained = ops[3 * mid:3 * mid + 3]
        if (not isinstance(constant, arith.ConstantOp) or constant.attributes
                or set(constant.properties) != {"value"} or not isinstance(constant.value, IntegerAttr)
                or constant.value.type != i32 or constant.value.value.data != mid
                or not isinstance(marker, func.CallOp) or marker.attributes
                or set(marker.properties) != {"callee"} or marker.callee.string_value() != MARK_SYM
                or tuple(marker.arguments) != (constant.result,) or marker.results
                or len(tuple(constant.result.uses)) != 1):
            raise OpProfileError("profile marker id, order, uses or callback differs")
        if mid != len(source_ops) - 1:
            actual_table.append(_operation_record(retained, mid))
        block.erase_op(marker)
        block.erase_op(constant)
    restored.body.block.erase_op(declarations[0])
    # Structural float equality can equate +/-0; retain canonical typed metadata
    # as well as the complete original operations, uses, regions and interfaces.
    def metadata(module):
        return tuple(tuple(tuple(sorted((key, str(value)) for key, value in fields.items()))
                           for fields in (op.attributes, op.properties)) for op in module.walk())

    if not original.is_structurally_equivalent(restored) or metadata(original) != metadata(restored):
        raise OpProfileError("profile changed original source operations, effects or metadata")
    if table != actual_table:
        raise OpProfileError("profile table differs from actual instrumented operations")


def instrument_module(module, *, function: str = "forward"):
    """Return a new instrumented module and complete top-level operation table.

    Refuse unsupported control flow and marker collisions before cloning. The
    original module, including cached read-only parses, is never modified.
    """
    from .op_profile import MARK_SYM, OpProfileError

    candidates = [op for op in module.body.block.ops if isinstance(op, func.FuncOp) and op.sym_name.data == function]
    if len(candidates) != 1:
        raise OpProfileError(f"expected one top-level func.func @{function}")
    fn = candidates[0]
    if len(fn.body.blocks) != 1:
        raise OpProfileError("operation profiling requires one function body block")
    ops = list(fn.body.block.ops)
    if not ops or not isinstance(ops[-1], func.ReturnOp):
        raise OpProfileError("operation profiling requires a final func.return")
    if any(isinstance(op, func.ReturnOp) for op in ops[:-1]):
        raise OpProfileError("operation profiling cannot follow multiple returns")
    if len(ops) == 1:
        raise OpProfileError(f"@{function} has no top-level ops to instrument")
    for op in module.walk():
        if isinstance(op, func.FuncOp) and op.sym_name.data == MARK_SYM:
            raise OpProfileError("profiler marker symbol already exists")
        if isinstance(op, func.CallOp) and op.callee.root_reference.data == MARK_SYM:
            raise OpProfileError("module already calls the profiler marker")
    module.verify()

    selected = module.clone()
    fn = next(op for op in selected.body.block.ops if isinstance(op, func.FuncOp) and op.sym_name.data == function)
    ops = list(fn.body.block.ops)
    table = []
    for mid, op in enumerate(ops):
        constant = arith.ConstantOp(IntegerAttr(mid, i32))
        constant.result.name_hint = f"prof_id_{mid}"
        marker = func.CallOp(MARK_SYM, [constant.result], [])
        fn.body.block.insert_ops_before([constant, marker], op)
        if isinstance(op, func.ReturnOp):
            continue
        table.append(_operation_record(op, mid))
    selected.body.block.insert_op_before(func.FuncOp.external(MARK_SYM, [i32], []), fn)
    selected.verify()
    verify_instrumentation(module, selected, table, function=function)
    return selected, table


def instrument_text(mlir_text: str, *, function: str = "forward"):
    """Parse either registered custom or generic MLIR and print verified generic IR."""
    module, table = instrument_module(_parse_text(mlir_text), function=function)
    stream = StringIO()
    Printer(stream=stream, print_generic_format=True).print_op(module)
    return stream.getvalue() + "\n", table
