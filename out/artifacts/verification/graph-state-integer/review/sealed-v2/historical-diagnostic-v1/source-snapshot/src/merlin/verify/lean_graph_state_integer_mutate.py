"""Independent actual-IR adverse edits, with compiler-validity evidence.

Malformed cases are distinctly labeled; no native failure is a semantic refusal.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def mutate(source, destination, kind):
    from xdsl.dialects import arith, builtin, func, scf
    from xdsl.rewriter import Rewriter

    from merlin.transforms.certificates import ir_text, parse_ir
    from merlin.transforms.normalization import signed_wrap

    module = parse_ir(source.read_text())
    module.verify()
    main = next(op for op in module.body.block.ops if type(op) is func.FuncOp and op.sym_name.data == "main")
    operations = list(main.walk())
    calls = [op for op in operations if type(op) is func.CallOp]
    loops = [op for op in operations if type(op) is scf.ForOp]
    malformed = False
    if kind in ("constant", "exposed-site"):
        call = next(op for op in calls if op.callee.root_reference.data == "watch")
        argument = call.operands[1 if kind == "exposed-site" else 0]
        op = argument.owner
        if type(op) is not arith.ConstantOp:
            raise ValueError("live actual folded constant expected")
        value = signed_wrap(op.value.value.data + 1, op.result.type.width.data)
        Rewriter.replace_op(op, arith.ConstantOp.from_int_and_width(value, op.result.type))
    elif kind == "operand":
        call = next(op for op in calls if op.callee.root_reference.data == "watch")
        call.operands = (main.body.block.args[0], *call.operands[1:])
    elif kind in ("operand-order", "observer-arguments"):
        call = next(op for op in reversed(calls) if op.callee.root_reference.data == "watch")
        call.operands = tuple(reversed(call.operands))
    elif kind == "opcode":
        op = next(op for op in operations if type(op) is arith.AddiOp)
        Rewriter.replace_op(op, arith.SubiOp(*op.operands))
    elif kind == "flags":
        op = next(op for op in operations if type(op) in (arith.AddiOp, arith.SubiOp, arith.MuliOp))
        op.properties["overflowFlags"] = arith.IntegerOverflowAttr((arith.IntegerOverflowFlag.NSW,))
    elif kind in ("carry-init", "yield-order", "capture", "step", "bound"):
        if kind == "carry-init":
            op = loops[0]
            op.operands = (*op.operands[:3], *tuple(reversed(op.iter_args)))
        elif kind == "yield-order":
            op = next(op for op in operations if type(op) is scf.YieldOp and len(op.operands) > 1)
            op.operands = tuple(reversed(op.operands))
        elif kind == "capture":
            call = next(op for op in calls if op.callee.root_reference.data == "watch")
            call.operands = (call.operands[0], main.body.block.args[0])
        else:
            argument = loops[0].step if kind == "step" else loops[0].ub
            op = argument.owner
            Rewriter.replace_op(op, arith.ConstantOp.from_int_and_width(op.value.value.data + 1, op.result.type))
    elif kind == "callee":
        call = next(op for op in calls if op.callee.root_reference.data == "watch")
        call.properties["callee"] = builtin.SymbolRefAttr("other")
    elif kind == "drop-observer":
        call = next(op for op in calls if op.callee.root_reference.data == "watch")
        call.parent_block().erase_op(call)
    elif kind == "return-order":
        op = next(op for op in operations if type(op) is func.ReturnOp)
        op.operands = tuple(reversed(op.operands))
    elif kind == "return-value":
        op = next(op for op in operations if type(op) is func.ReturnOp)
        constant = arith.ConstantOp.from_int_and_width(1, op.operands[0].type)
        op.parent_block().insert_op_before(constant, op)
        op.operands = (constant.result, *op.operands[1:])
    elif kind in ("function-context", "signature-width"):
        op = next(op for op in module.body.block.ops if type(op) is func.FuncOp and op.sym_name.data == "untouched")
        if kind == "signature-width":
            replacement = parse_ir("builtin.module {func.func @untouched(%x:i32)->i32{func.return %x:i32}}")
            function = replacement.body.block.first_op
            replacement.body.block.detach_op(function)
            Rewriter.replace_op(op, function)
        else:
            ret = op.body.block.last_op
            constant = arith.ConstantOp.from_int_and_width(1, op.body.block.args[0].type)
            op.body.block.insert_op_before(constant, ret)
            ret.operands = (constant.result,)
    elif kind in ("function-header", "module-header"):
        (main if kind == "function-header" else module).attributes["unmodeled"] = builtin.UnitAttr()
    elif kind in ("malformed-operand", "malformed-result", "malformed-region", "malformed-call"):
        malformed = True
        call = next(op for op in calls if op.callee.root_reference.data == "provider")
        if kind == "malformed-operand":
            call.operands = (*call.operands[:-1], main.body.block.args[0])
        elif kind == "malformed-result":
            call.results[0]._type = builtin.i64
        elif kind == "malformed-region":
            loops[0].body.block.args[0]._type = builtin.i8
        else:
            call.operands = tuple(call.operands[:-1])
    else:
        raise ValueError("unknown owned integer mutation")
    if not malformed:
        module.verify()
    text = ir_text(module)
    if text == ir_text(parse_ir(source.read_text())):
        raise ValueError("requested mutation made no actual change")
    destination.write_text(text)
    return {"kind": kind, "compiler_valid": not malformed, "malformed": malformed, "changed": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("kind")
    print(json.dumps(mutate(**vars(parser.parse_args()))))


if __name__ == "__main__":
    main()
