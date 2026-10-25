"""Actual over-rank pass fixtures and independent valid loop references."""

from io import StringIO

from xdsl.dialects import arith, func
from xdsl.dialects.builtin import BFloat16Type, Float16Type, ModuleOp, TensorType, f32, f64
from xdsl.dialects.linalg import ops as linalg
from xdsl.ir import Block, Region
from xdsl.printer import Printer

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.passes_xdsl import collapse_overrank_matmul


def render(module):
    stream = StringIO()
    Printer(stream=stream, print_generic_format=True).print_op(module)
    return stream.getvalue()


def fixture(batch=(2, 3), k=4, n=5, kind="f32", nested=False):
    elt = {"f32": f32, "f64": f64, "f16": Float16Type(), "bf16": BFloat16Type()}[kind]
    lhs, rhs, out = [TensorType(elt, dims) for dims in ((*batch, k), (k, n), (*batch, n))]
    block = Block(arg_types=[lhs, rhs, out])
    op = linalg.MatmulOp(inputs=block.args[:2], outputs=[block.args[2]], res=[out])
    block.add_ops([op, func.ReturnOp(op.results[0])])
    module = ModuleOp([func.FuncOp("subject", ([lhs, rhs, out], [out]), Region(block))])
    if nested:
        # Nested region captures all three enclosing arguments; complete module is still checked.
        text = render(module)
        original = parse_mlir_text(text)
        from xdsl.dialects import scf
        from xdsl.dialects.builtin import i1

        fn = next(o for o in original.walk() if o.name == "func.func")
        body = fn.body.block
        condition = arith.ConstantOp.from_int_and_width(1, i1)
        branches = []
        for _ in range(2):
            branch = Block()
            mat = linalg.MatmulOp(inputs=body.args[:2], outputs=[body.args[2]], res=[out])
            branch.add_ops([mat, scf.YieldOp(mat.results[0])])
            branches.append(Region(branch))
        choose = scf.IfOp(condition.result, [out], *branches)
        for old in list(body.ops):
            body.erase_op(old, safe_erase=False)
        body.add_ops([condition, choose, func.ReturnOp(choose.results[0])])
        module = original
    return render(module)


def actual(source):
    module = parse_mlir_text(source)
    before = render(module)
    count = collapse_overrank_matmul(module)
    module.verify()
    return before, render(module), count


def ty(shape, kind):
    return "tensor<" + "".join(f"{d}x" for d in shape) + kind + ">"


def reference(batch, k, n, kind="f32"):
    """Independent valid tensor/scf linear kernel; no linalg generic or invalid named op."""
    left, right, out = [ty(dims, kind) for dims in ((*batch, k), (k, n), (*batch, n))]
    rank = len(batch) + 1
    lines = [
        f"module {{ func.func @subject(%a: {left}, %b: {right}, %unused: {out}) -> {out} {{",
        "%c0 = arith.constant 0 : index",
        "%c1 = arith.constant 1 : index",
        f"%zero = arith.constant 0.0 : {kind}",
        f"%initial = tensor.empty() : {out}",
    ]
    for axis, size in enumerate((*batch, n, k)):
        lines.append(f"%s{axis} = arith.constant {size} : index")
    for axis in range(rank):
        prev = "%initial" if axis == 0 else f"%t{axis - 1}"
        lines.append(
            f"%r{axis} = scf.for %i{axis} = %c0 to %s{axis} step %c1 iter_args(%t{axis} = {prev}) -> ({out}) {{"
        )
    aindex = ", ".join([*(f"%i{d}" for d in range(len(batch))), "%k"])
    lines += [
        f"%sum = scf.for %k = %c0 to %s{rank} step %c1 iter_args(%acc = %zero) -> ({kind}) {{",
        f"%av = tensor.extract %a[{aindex}] : {left}",
        f"%bv = tensor.extract %b[%k, %i{len(batch)}] : {right}",
        f"%p = arith.mulf %av, %bv : {kind}",
        f"%s = arith.addf %acc, %p : {kind}",
        f"scf.yield %s : {kind}",
        "}",
    ]
    indices = ", ".join(f"%i{d}" for d in range(rank))
    lines += [f"%set = tensor.insert %sum into %t{rank - 1}[{indices}] : {out}", f"scf.yield %set : {out}", "}"]
    for axis in reversed(range(rank - 1)):
        lines += [f"scf.yield %r{axis + 1} : {out}", "}"]
    lines += [f"return %r0 : {out}", "}}"]
    return "\n".join(lines)
