"""Compiler-valid adverse boundary mutations (test evidence, never proof authority)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def mutate(source, destination, kind):
    from xdsl.dialects import arith, builtin, func, scf
    from xdsl.rewriter import Rewriter

    from merlin.transforms.certificates import ir_text, parse_ir

    module = parse_ir(source.read_text())
    module.verify()
    main = next(f for f in module.body.block.ops if type(f) is func.FuncOp and f.sym_name.data == "main")
    operations = list(main.walk())
    loops = [op for op in operations if type(op) is scf.ForOp]
    if kind in ("step", "bound"):
        loop = loops[0]
        op = (loop.step if kind == "step" else loop.ub).owner
        Rewriter.replace_op(op, arith.ConstantOp.from_int_and_width(op.value.value.data + 1, op.result.type))
    elif kind == "offset":
        op = next(
            op
            for op in loops[0].body.walk()
            if type(op) is arith.ConstantOp and type(op.result.type) is builtin.IndexType
        )
        Rewriter.replace_op(op, arith.ConstantOp.from_int_and_width(op.value.value.data + 1, op.result.type))
    elif kind in ("operand", "operand-order", "opcode", "flags"):
        op = next(
            op
            for op in operations
            if type(op) is arith.AddiOp
            and type(op.result.type) is builtin.IntegerType
            and op.parent_block() is not main.body.block
        )
        if kind == "operand":
            op.operands = (main.body.block.args[1], op.operands[1])
        elif kind == "operand-order":
            op.operands = tuple(reversed(op.operands))
        elif kind == "opcode":
            Rewriter.replace_op(op, arith.SubiOp(*op.operands))
        else:
            op.properties["overflowFlags"] = arith.IntegerOverflowAttr((arith.IntegerOverflowFlag.NSW,))
    elif kind == "carry-init":
        loop = loops[0]
        loop.operands = (*loop.operands[:3], *tuple(reversed(loop.iter_args)))
    elif kind == "yield-order":
        op = next(op for op in operations if type(op) is scf.YieldOp and len(op.operands) > 1)
        op.operands = tuple(reversed(op.operands))
    elif kind == "callee":
        op = next(op for op in operations if type(op) is func.CallOp and op.callee.root_reference.data == "partial")
        op.properties["callee"] = builtin.SymbolRefAttr("other")
    elif kind == "drop-observer":
        op = next(
            op for op in operations if type(op) is func.CallOp and op.callee.root_reference.data == "final_observe"
        )
        op.parent_block().erase_op(op)
    elif kind == "drop-inner-observer":
        op = next(op for op in operations if type(op) is func.CallOp and op.callee.root_reference.data == "watch")
        op.parent_block().erase_op(op)
    elif kind == "observer-arguments":
        op = next(
            op for op in operations if type(op) is func.CallOp and op.callee.root_reference.data == "final_observe"
        )
        op.operands = tuple(reversed(op.operands))
    elif kind == "return-order":
        op = next(op for op in operations if type(op) is func.ReturnOp)
        op.operands = tuple(reversed(op.operands))
    elif kind == "min-to-max":
        op = next(op for op in operations if type(op) is arith.MinUIOp)
        Rewriter.replace_op(op, arith.MaxUIOp(*op.operands))
    elif kind == "min-operands":
        op = next(op for op in operations if type(op) is arith.MinUIOp)
        op.operands = (op.operands[0], loops[0].step)
    elif kind == "inner-control":
        inner = loops[1]
        inner.operands = (loops[0].lb, *inner.operands[1:])
    elif kind == "function-context":
        target = next(f for f in module.body.block.ops if type(f) is func.FuncOp and f.sym_name.data == "untouched")
        ret = target.body.block.last_op
        constant = arith.ConstantOp.from_int_and_width(1, target.body.block.args[0].type)
        target.body.block.insert_op_before(constant, ret)
        ret.operands = (constant.result,)
    elif kind == "function-header":
        main.attributes["unmodeled"] = builtin.UnitAttr()
    elif kind == "module-header":
        module.attributes["unmodeled"] = builtin.UnitAttr()
    else:
        raise ValueError("unknown test mutation: " + kind)
    module.verify()
    text = ir_text(module)
    original = ir_text(parse_ir(source.read_text()))
    if text == original:
        raise ValueError("mutation did not change actual IR")
    destination.write_text(text)
    return {"compiler_valid": True, "changed": True, "kind": kind}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("kind")
    args = vars(parser.parse_args())
    print(json.dumps(mutate(**args)))


if __name__ == "__main__":
    main()
