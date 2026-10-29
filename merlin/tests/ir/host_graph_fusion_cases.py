"""Actual connected host graphs with retained producer fanout and ordered outputs."""

import hashlib

from merlin.llvmlower.graph_optimizations import GraphOptimizationContract, optimize_host_graph


def source(*, shape=(2, 3), width=32, context=False):
    typ = "tensor<" + "x".join(map(str, shape)) + f"xi{width}>"
    dims = ",".join(f"d{i}" for i in range(len(shape)))
    maps = ",".join([f"affine_map<({dims})->({dims})>"] * 3)
    iters = ",".join(['"parallel"'] * len(shape))
    stages = [("addi", "%x,%y"), ("muli", "%a0,%y"), ("subi", "%a0,%x")]
    returned = "%a1,%a2,%a0"
    if context:
        stages = [("addi", "%x,%y"), ("muli", "%a0,%y"), ("addi", "%a1,%x"), ("subi", "%a1,%a0"), ("addi", "%a2,%a3")]
        returned = "%a4,%a1,%a0"
    body = []
    for i, (primitive, operands) in enumerate(stages):
        body.append(f"""%e{i}=tensor.empty():{typ}
 %a{i}=linalg.generic {{indexing_maps=[{maps}],iterator_types=[{iters}]}}
 ins({operands}:{typ},{typ}) outs(%e{i}:{typ}) {{
 ^bb0(%v:i{width},%w:i{width},%init:i{width}):
 %s=arith.{primitive} %v,%w:i{width}
 linalg.yield %s:i{width}
 }} -> {typ}""")
    return (
        f"module {{func.func @forward(%x:{typ},%y:{typ})->({typ},{typ},{typ}){{"
        + "\n".join(body)
        + f"\n return {returned}:{typ},{typ},{typ}}}}}"
    )


def contract(raw, *, context=False):
    group = tuple(range(2, 8)) if context else tuple(range(6))
    return GraphOptimizationContract(hashlib.sha256(raw.encode()).hexdigest(), "forward", fusion_groups=(group,))


def pair(**options):
    raw = source(**options)
    result, _ = optimize_host_graph(raw, contract(raw, context=options.get("context", False)))
    return raw, result


def mutate(raw, fault):
    from xdsl.dialects import arith
    from xdsl.rewriter import Rewriter

    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.xdsl_dialects._common import text

    module = parse_mlir_text(raw)
    functions = {op.sym_name.data: op for op in module.body.block.ops}
    entry = functions["forward"]
    calls = [op for op in entry.body.block.ops if op.name == "func.call"]
    call = next(
        op
        for op in calls
        if sum(child.name == "linalg.generic" for child in functions[op.callee.root_reference.data].walk()) > 1
    )
    body = functions[call.callee.root_reference.data]
    if fault == "arithmetic":
        op = next(op for op in body.walk() if op.name == "arith.muli")
        Rewriter.replace_op(op, arith.SubiOp(*op.operands))
    elif fault == "call-input":
        call.operands = tuple(reversed(call.operands))
    elif fault == "callee-output":
        term = body.body.block.last_op
        term.operands = tuple(reversed(term.operands))
    elif fault == "public-output":
        term = entry.body.block.last_op
        term.operands = (*term.operands[:-1], term.operands[0])
    elif fault == "suffix":
        later = calls[calls.index(call) + 1]
        op = next(op for op in functions[later.callee.root_reference.data].walk() if op.name == "arith.addi")
        Rewriter.replace_op(op, arith.SubiOp(*op.operands))
    else:
        raise AssertionError(fault)
    module.verify()
    return text(module, generic=True)
