"""Make direct user tensor results explicit device movement candidates."""


def expose_argument_results(module, report):
    from xdsl.dialects.builtin import TensorType

    from merlin.llvmlower._ir import copy

    functions = [op for op in module.walk() if op.name == "func.func" and op.sym_name.data == "forward"]
    if len(functions) != 1:
        return 0
    function = functions[0]
    returned = function.body.block.last_op
    if returned.name != "func.return":
        return 0
    indices = {
        i for row in report.get("cases", ()) if row.get("status") == "bundled" for i in row.get("output_indices", ())
    }
    values = list(returned.operands)
    changed = 0
    for index in sorted(indices):
        if not 0 <= index < len(values):
            continue
        value = values[index]
        if value not in function.body.block.args or not isinstance(value.type, TensorType):
            continue
        shape = value.type.get_shape()
        if any(n <= 0 for n in shape):
            continue
        operations = []
        values[index] = copy(value, operations)
        function.body.block.insert_ops_before(operations, returned)
        changed += 1
    returned.operands = values
    return changed
