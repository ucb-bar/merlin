"""Preserve caller-owned tensor inputs across native host bufferization.

MLIR treats function arguments as writable unless told otherwise. Functional
tensor updates may therefore overwrite an input after conversion to memrefs.
Public C-interface tensor arguments default to read-only storage here. Explicit
``bufferization.writable`` declarations retain their upstream meaning; memref
arguments retain their explicit memory effects.
"""


def bind_runner(source: str) -> str:
    """Apply the input boundary before passes in every native runner variant."""
    marker = "# __MERLIN_INSPECT_PARSED__"
    if source.count(marker) != 1:
        raise ValueError("tensor input ownership requires one parsed-module checkpoint")
    return source.replace(marker, RUNNER_PRELUDE + "\n_preserve_tensor_inputs(module, ctx)\n" + marker)


# Executed in the selected compiler environment, which owns the native bindings.
RUNNER_PRELUDE = r"""
def _preserve_tensor_inputs(module, ctx):
    with ctx:
        for view in module.body.operations:
            op = view.operation
            if op.name != "func.func" or "llvm.emit_c_interface" not in op.attributes:
                continue
            if ("sym_visibility" in op.attributes
                    and ir.StringAttr(op.attributes["sym_visibility"]).value == "private"):
                continue
            if not op.regions[0].blocks:
                continue
            inputs = ir.FunctionType(ir.TypeAttr(op.attributes["function_type"]).value).inputs
            existing = (ir.ArrayAttr(op.attributes["arg_attrs"])
                        if "arg_attrs" in op.attributes else None)
            rows = []
            changed = False
            for index, value_type in enumerate(inputs):
                attrs = ({item.name: item.attr for item in ir.DictAttr(existing[index])}
                         if existing is not None else {})
                if (isinstance(value_type, (ir.RankedTensorType, ir.UnrankedTensorType))
                        and "bufferization.writable" not in attrs):
                    attrs["bufferization.writable"] = ir.BoolAttr.get(False)
                    changed = True
                rows.append(ir.DictAttr.get(attrs))
            if changed:
                op.attributes["arg_attrs"] = ir.ArrayAttr.get(rows)
"""
