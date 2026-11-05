"""Explicit scalar host-rounding calls with caller-selected CPU support.

Keep this optional rewrite after fusion when possible. The shared helper
symbol is caller-selectable for historical ABI compatibility.
"""

from xdsl.dialects import func
from xdsl.dialects.builtin import Float32Type
from xdsl.ir import Region

from merlin.runtime.host_arithmetic import _identifier


def rewrite_roundeven(module, *, symbol: str = "merlin_host_roundevenf") -> int:
    """Replace scalar binary32 roundeven only; retain all other formats."""
    _identifier(symbol)
    selected = [
        op
        for op in module.walk()
        if op.name == "math.roundeven" and len(op.results) == 1 and isinstance(op.results[0].type, Float32Type)
    ]
    if not selected:
        return 0
    if any(op.name == "func.func" and op.sym_name.data == symbol for op in module.walk()):
        raise ValueError("host rounding helper symbol already exists")
    for op in selected:
        replacement = func.CallOp(symbol, list(op.operands), [op.results[0].type])
        replacement.attributes.update(op.attributes)
        op.parent.insert_op_before(replacement, op)
        op.results[0].replace_all_uses_with(replacement.results[0])
        op.parent.erase_op(op)
    ty = Float32Type()
    module.body.block.add_op(func.FuncOp(symbol, ([ty], [ty]), Region(), visibility="private"))
    module.verify()
    return len(selected)
