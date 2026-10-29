"""Actual producer inputs for the rank-one buffer profile; no saved proof hints."""

from merlin.transforms import loops, normalization, planning
from merlin.transforms.certificates import ir_text
from merlin.verify.buffer_reuse_bridge import parse


def source(extent=3, *, rank=1, computed=False):
    shape = (extent, *((2,) * (rank - 1)))
    kind = "memref<" + "x".join(map(str, shape)) + "xi32>"
    declarations = [
        "%zero = arith.constant 0 : index",
        "%one = arith.constant 1 : index",
        *(f"%bound{i} = arith.constant {size} : index" for i, size in enumerate(shape)),
    ]
    body = []
    for epoch in range(2):
        indices = [f"%iv{epoch}_{i}" for i in range(rank)]
        body += [
            f"%buffer{epoch} = memref.alloc() {{alignment = 16 : i64}} : {kind}",
            f'%view{epoch} = "memref.cast"(%buffer{epoch}) : ({kind}) -> {kind}',
            f"%dim{epoch} = memref.dim %view{epoch}, %zero : {kind}",
            *(f"scf.for {iv} = %zero to %bound{i} step %one {{" for i, iv in enumerate(indices)),
        ]
        value = f"%seed{epoch}"
        if computed:
            value = f"%computed{epoch}"
            body.append(f"{value} = arith.addi %seed{epoch}, %seed{epoch} : i32")
        body += [
            f"memref.store {value}, %view{epoch}[{', '.join(indices)}] : {kind}",
            *("scf.yield\n}" for _ in indices),
            f"%first{epoch} = memref.load %view{epoch}[{', '.join(['%zero'] * rank)}] : {kind}",
            f"%observed{epoch} = func.call @touch(%first{epoch}, %caller) : (i32, memref<1xi32>) -> i32",
            f"memref.dealloc %buffer{epoch} : {kind}",
        ]
    return f"""builtin.module {{
      func.func @kernel(%seed0: i32, %seed1: i32, %caller: memref<1xi32>) -> (i32, index) {{
        {chr(10).join(declarations + body)}
        %sum = arith.addi %observed0, %observed1 : i32
        func.return %sum, %dim1 : i32, index
      }}
      func.func @touch(%value: i32, %caller: memref<1xi32>) -> i32 {{
        %zero = arith.constant 0 : index
        %old = memref.load %caller[%zero] : memref<1xi32>
        %new = arith.addi %old, %value : i32
        memref.store %new, %caller[%zero] : memref<1xi32>
        func.return %new : i32
      }}
      func.func @untouched(%value: i32) -> i32 {{ func.return %value : i32 }}
    }}"""


def chain(extent=3, *, tiled=False, tile=2, rank=1, computed=False):
    module = parse(source(extent, rank=rank, computed=computed))
    result = [ir_text(module)]
    normalization.normalize_shapes(module)
    result.append(ir_text(module))
    plan = planning.plan_storage(module)
    if not plan.witnesses:
        raise ValueError("actual storage producer declined: " + repr(plan.unknown))
    planning.reuse_storage(module, plan)
    result.append(ir_text(module))
    if tiled:
        loops.segment_loops(module, (tile,), index_bits=64)
        result.append(ir_text(module))
        if result[-2] == result[-1]:
            raise ValueError("actual tiling producer did not change the requested buffer")
    module.verify()
    return tuple(result)


def mutate_target(buffers, mutation):
    """Return verifier-valid raw target changes, without editing source hints."""
    from xdsl.dialects import arith, builtin, func, memref, scf
    from xdsl.rewriter import Rewriter

    module = parse(buffers[-1])
    function = next(op for op in module.body.block.ops if type(op) is func.FuncOp and op.sym_name.data == "kernel")
    outer = next(op for op in function.body.block.ops if type(op) is scf.ForOp)
    point = next(op for op in outer.body.block.ops if type(op) is scf.ForOp)
    constants = [op for op in function.body.block.ops if type(op) is arith.ConstantOp]
    zero = next(op.result for op in constants if op.value.value.data == 0)
    one = next(op.result for op in constants if op.value.value.data == 1)
    if mutation == "point_lower":
        point.operands = (zero, point.ub, point.step, *point.iter_args)
    elif mutation == "point_upper":
        point.operands = (point.lb, outer.ub, point.step, *point.iter_args)
    elif mutation == "point_stride":
        point.operands = (point.lb, point.ub, outer.step, *point.iter_args)
    elif mutation == "outer_stride":
        outer.operands = (outer.lb, outer.ub, one, *outer.iter_args)
    elif mutation == "add_operand":
        add = next(op for op in outer.body.block.ops if type(op) is arith.AddiOp)
        add.operands = (outer.step, outer.step)
    elif mutation in ("store_seed", "store_buffer", "store_index"):
        store = next(op for op in point.body.block.ops if type(op) is memref.StoreOp)
        if mutation == "store_seed":
            store.operands = (function.body.block.args[1], *store.operands[1:])
        elif mutation == "store_index":
            store.operands = (*store.operands[:2], zero)
        else:
            # An additional same-shaped allocation is an actual effect, not an
            # alias accepted on the basis of a forged storage plan.
            original = next(op for op in function.body.block.ops if type(op) is memref.AllocOp)
            extra = original.clone()
            function.body.block.insert_op_before(extra, outer)
            store.operands = (store.operands[0], extra.memref, *store.operands[2:])
    elif mutation == "callee":
        helper = next(op for op in module.body.block.ops if type(op) is func.FuncOp and op.sym_name.data == "touch")
        addition = next(op for op in helper.walk() if type(op) is arith.AddiOp)
        Rewriter.replace_op(addition, arith.SubiOp(addition.lhs, addition.rhs))
    elif mutation == "shape":
        bound = next(op for op in constants if op.value.value.data == 3)
        Rewriter.replace_op(bound, arith.ConstantOp.from_int_and_width(2, builtin.IndexType()))
    else:
        raise ValueError("unknown fixed rank-one mutation")
    module.verify()
    return (*buffers[:-1], ir_text(module))
