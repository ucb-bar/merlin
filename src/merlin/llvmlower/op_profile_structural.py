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
from merlin.llvmlower.dense_literals import DenseLiteralError, preserve_raw_dense_bits


def _parse_text(text):
    from xdsl.dialects.vector import Vector
    from xdsl.utils.exceptions import ParseError

    from merlin.frontends.linalg_mlir import make_context, parse_mlir_text

    from .op_profile import OpProfileError

    context = make_context()
    # The shared frontend already owns the prepared memref dialect registry.
    context.load_dialect(Vector)
    try:
        return parse_mlir_text(text, ctx=context)
    except ParseError as exc:
        # An op this context has no format for (a named linalg op it does not register, say) means the
        # module cannot be instrumented -- which is the profiler's refusal to make, not a crash of the
        # image build that asked for it. Callers already degrade on OpProfileError.
        raise OpProfileError(f"cannot parse the module to instrument it: {' '.join(str(exc).split())[:400]}") from exc


def instrument_module(module, *, function: str = "forward"):
    """Return a new instrumented module and complete top-level operation table.

    Refuse unsupported control flow and marker collisions before cloning. The
    original module, including cached read-only parses, is never modified.
    """
    from .op_profile import _PROV_KEYS, MARK_SYM, OpProfileError, _elem_count

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
        result_types = [str(value.type) for value in op.results]
        result_type = result_types[0] if result_types else None
        row = {
            "id": mid,
            "mlir_op": op_name(op),
            "result_type": result_type,
            "result_types": result_types,
            "elems": _elem_count(result_type),
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
            row["body_ops"] = sorted(
                {
                    op_name(child)
                    for region in op.regions
                    for child in region.walk()
                    if op_name(child).startswith(("arith.", "math."))
                }
            )
        table.append(row)
    selected.body.block.insert_op_before(func.FuncOp.external(MARK_SYM, [i32], []), fn)
    selected.verify()
    return selected, table


def instrument_text(mlir_text: str, *, function: str = "forward"):
    """Parse either registered custom or generic MLIR and print verified generic IR."""
    module, table = instrument_module(_parse_text(mlir_text), function=function)
    stream = StringIO()
    Printer(stream=stream, print_generic_format=True).print_op(module)
    try:
        return preserve_raw_dense_bits(mlir_text, stream.getvalue()), table
    except DenseLiteralError as exc:
        from .op_profile import OpProfileError

        raise OpProfileError(f"cannot safely reprint dense constants: {exc}") from exc
