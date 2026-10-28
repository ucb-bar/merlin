"""Dense readonly offsets through an original source-owned borrowed wrapper.

The wrapper and complete stock C bridge must be independently reopened first.
Only input descriptors change; fresh output storage and alias checks remain.
"""


def bind(module, host, selected, groups):
    from xdsl.dialects import builtin, func
    from xdsl.rewriter import Rewriter

    from merlin.frontends.linalg_mlir import layout_to_buffer_type
    from merlin.xdsl_dialects._common import text

    from .device_input_views import _dense_tensor
    from .sibling_source_route import component_hosts, require_host

    require_host(host, selected)
    module.verify()
    functions = {op.sym_name.data: op for op in module.body.block.ops if isinstance(op, func.FuncOp)}
    calls = {}
    for operation in module.walk():
        if isinstance(operation, func.CallOp):
            calls.setdefault(operation.callee.root_reference.data, []).append(operation)
    result, retained, symbols = {}, {}, set()
    for component in component_hosts(host):
        symbol, name = component.device_symbol, component.writer.symbol
        symbols.add(symbol)
        wrapper = functions.get(name)
        borrowed = functions.get(component.writer.borrowed_symbol)
        original = next(
            op for op in component.module.body.block.ops if isinstance(op, func.FuncOp) and op.sym_name.data == name
        )
        original_borrowed = next(
            op for op in component.module.body.block.ops
            if isinstance(op, func.FuncOp) and op.sym_name.data == component.writer.borrowed_symbol)
        if (
            wrapper is None
            or borrowed is None
            or borrowed.body.blocks
            or text(wrapper, generic=True) != text(original, generic=True)
            or text(borrowed, generic=True) != text(original_borrowed, generic=True)
        ):
            raise ValueError("dense sibling inputs lost the exact original borrowed writer")
        _buffers, call = groups[symbol]
        indices = []
        attrs = list(wrapper.arg_attrs)
        borrowed_types = list(borrowed.function_type.inputs)
        for i, row in enumerate(call):
            typ = tuple(wrapper.function_type.inputs)[i]
            if (
                row["role"].startswith("out_")
                or len(row["shape"]) != 2
                or attrs[i].data != {"bufferization.access": builtin.StringAttr("read")}
            ):
                continue
            if not calls.get(name) or not all(_dense_tensor(c.arguments[i]) for c in calls[name]):
                retained.setdefault(name, []).append(i)
                continue
            layout = builtin.StridedLayoutAttr([typ.get_shape()[1], 1], None)
            buffer_type = builtin.MemRefType(typ.element_type, typ.get_shape(), layout=layout)
            adapters = [
                op
                for op in wrapper.body.block.ops
                if op.name == "bufferization.to_buffer" and op.operands[0] is wrapper.body.block.args[i]
            ]
            if len(adapters) != 1 or "read_only" not in adapters[0].properties:
                raise ValueError("readonly sibling layout lacks its actual original input descriptor")
            replacement = layout_to_buffer_type().build(
                operands=[wrapper.body.block.args[i]],
                result_types=[buffer_type],
                properties={"read_only": builtin.UnitAttr()},
            )
            Rewriter.replace_op(adapters[0], replacement)
            attrs[i] = builtin.DictionaryAttr({**attrs[i].data, "bufferization.buffer_layout": layout})
            borrowed_types[i] = buffer_type
            indices.append(i)
        wrapper.properties["arg_attrs"] = builtin.ArrayAttr(attrs)
        borrowed.properties["function_type"] = builtin.FunctionType.from_lists(
            borrowed_types, borrowed.function_type.outputs
        )
        if indices:
            result[name] = indices
    module.verify()
    return result, retained, symbols
