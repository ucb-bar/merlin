"""Expose scalar IEEE operation boundaries as closed tensor pointwise nodes."""


def expose_pointwise_primitives(module, *, integer_boundaries=False):
    from xdsl.dialects import tensor
    from xdsl.dialects.builtin import TensorType
    from xdsl.dialects.linalg import ops as L
    from xdsl.rewriter import Rewriter

    from merlin.llvmlower._ir import identity, point

    allowed = {
        "arith.constant",
        "arith.addf",
        "arith.subf",
        "arith.mulf",
        "arith.divf",
        "arith.negf",
        "arith.extf",
        "arith.truncf",
        "arith.sitofp",
        "arith.uitofp",
        "arith.bitcast",
        "arith.cmpf",
        "arith.cmpi",
        "arith.select",
        "math.fma",
        "arith.extui",
        "arith.extsi",
        "arith.trunci",
        "arith.andi",
        "arith.ori",
        "arith.xori",
        "arith.shli",
        "arith.shrui",
        "arith.extui",
        "arith.trunci",
        "arith.cmpi",
        "arith.cmpf",
        "arith.select",
        "arith.addi",
        "arith.subi",
        "arith.muli",
        "arith.divsi",
        "arith.divui",
        "arith.remsi",
        "arith.remui",
        "arith.floordivsi",
        "arith.minsi",
        "arith.minui",
        "arith.maxsi",
        "arith.maxui",
        "linalg.index",
        "linalg.yield",
    }
    changed = 0
    for op in list(module.walk()):
        if op.name != "linalg.generic" or len(op.results) != 1:
            continue
        if any(it.data != L.IteratorType.PARALLEL for it in op.get_iterator_types()):
            continue
        body = op.body.block
        integer_casts = {"arith.extsi", "arith.extui", "arith.trunci"}
        closed_integer = (
            integer_boundaries
            and "merlin.exact_host_arithmetic" not in op.attributes
            and any(child.name in integer_casts for child in body.ops)
            and all(
                child.name
                in integer_casts | {"arith.constant", "arith.addi", "arith.subi", "arith.muli", "linalg.yield"}
                for child in body.ops
            )
        )
        if integer_boundaries and not closed_integer:
            # This late pass must not expose generated IEEE integer RNE or
            # quotient plumbing as new data arithmetic/device producers.
            continue
        if (
            not any(
                child.name
                in {
                    "arith.addf",
                    "arith.subf",
                    "arith.mulf",
                    "arith.divf",
                    "arith.negf",
                    "arith.extf",
                    "arith.truncf",
                    "arith.sitofp",
                    "arith.uitofp",
                    "math.fma",
                    "arith.divsi",
                    "arith.divui",
                    "arith.remsi",
                    "arith.remui",
                    "arith.floordivsi",
                }
                for child in body.ops
            )
            and not closed_integer
        ):
            # Keep generated integer RNE/bit plumbing intact, including its
            # SSA-bound finite grid. Only source floating boundaries split.
            continue
        if any(child.name not in allowed for child in body.ops):
            continue
        arithmetic = [child for child in body.ops if child.name not in {"arith.constant", "linalg.yield"}]
        scalar_literal_operand = (
            len(arithmetic) == 1
            and arithmetic[0].name in {"arith.addf", "arith.subf", "arith.mulf", "arith.divf"}
            and any(getattr(value.owner, "name", None) == "arith.constant" for value in arithmetic[0].operands)
        )
        if len(arithmetic) <= 1 and not scalar_literal_operand:
            continue
        # A closed word predicate has no floating arithmetic boundary to expose.
        # In particular, literal casts belong to that predicate, not its data path.
        dependent = set(body.args)
        numeric = {
            "math.fma",
            "arith.addf",
            "arith.subf",
            "arith.mulf",
            "arith.divf",
            "arith.negf",
            "arith.extf",
            "arith.truncf",
            "arith.sitofp",
            "arith.uitofp",
            "arith.divsi",
            "arith.divui",
            "arith.remsi",
            "arith.remui",
            "arith.floordivsi",
        }
        has_numeric_data = False
        for child in body.ops:
            if any(value in dependent for value in child.operands):
                dependent.update(child.results)
                has_numeric_data |= child.name in numeric
        if not has_numeric_data and not closed_integer:
            continue
        shape = op.results[0].type.get_shape()
        out_map = identity(len(shape))
        maps = list(op.get_indexing_maps())
        if maps[-1] != out_map:
            continue
        values = {arg: (value, maps[i]) for i, (arg, value) in enumerate(zip(body.args, op.inputs))}
        operations = []
        valid = True
        for child in body.ops:
            if child.name == "linalg.yield":
                continue
            if child.name == "arith.constant":
                literal = child.clone()
                splat = tensor.SplatOp(literal.results[0], [], TensorType(literal.results[0].type, shape))
                operations.extend([literal, splat])
                values[child.results[0]] = splat.result, out_map
                continue
            if any(value not in values for value in child.operands):
                valid = False
                break
            inputs = [values[value][0] for value in child.operands]
            result = point(
                inputs,
                TensorType(child.results[0].type, shape),
                lambda args, emit: emit(child.clone(value_mapper=dict(zip(child.operands, args)))),
                operations,
                maps=[values[value][1] for value in child.operands] + [out_map],
                iterators=op.get_iterator_types(),
            )
            result.owner.attributes.update(op.attributes)
            values[child.results[0]] = result, out_map
        if not valid or body.last_op.operands[0] not in values:
            continue
        result, result_map = values[body.last_op.operands[0]]
        if result_map != out_map or result.type != op.results[0].type:
            continue
        Rewriter.replace_op(op, operations, [result])
        changed += 1
    module.verify()
    return changed
