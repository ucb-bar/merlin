"""Exact radix-256 integer producers with explicit modular host recombination.

Balanced digits are computed from the immutable source tensors at runtime.
Only marked signed-byte/i32 producers are device work. The original storage
width determines each host shift/add, so wraparound is retained at every source
operation boundary. No floating conversion or output-dependent specialization
is used. Smaller input limb counts require a bound on the complete input bytes.
"""

from __future__ import annotations

from math import prod


def rewrite_integer_components(module, *, limb_attribute, input_bounds=(), known_bounds=None):
    import json

    from xdsl.dialects import arith
    from xdsl.dialects.builtin import IntegerAttr, IntegerType, StringAttr, TensorType, i8, i32
    from xdsl.dialects.linalg import ops as L
    from xdsl.ir import Block
    from xdsl.rewriter import Rewriter

    from merlin.llvmlower._ir import emitter, generic, mac, point, splat

    functions = [op for op in module.walk() if op.name == "func.func" and op.sym_name.data == "forward"]
    if len(functions) != 1:
        raise ValueError("integer components require one forward function")
    bounds = dict(zip(functions[0].body.block.args, input_bounds))
    bounds.update(known_bounds or {})
    plans = []
    from merlin.llvmlower.additive_regions import expose_reduce_domains
    from merlin.llvmlower.integer_width_boundaries import preserve_integer_width_boundaries

    expose_reduce_domains(module)
    preserve_integer_width_boundaries(module, mathematical_sources=bounds)

    def indices(digits):
        return list(range(digits)) or [0]

    def width(value):
        typ = value.type
        return (
            typ.element_type.width.data
            if isinstance(typ, TensorType) and isinstance(typ.element_type, IntegerType)
            else 0
        )

    def absolute(value):
        if value in bounds:
            return bounds[value]
        op = value.owner
        if getattr(op, "name", None) == "arith.constant":
            attr = op.properties.get("value", op.attributes.get("value"))
            if hasattr(attr, "get_values"):
                return max((abs(int(v)) for v in attr.get_values()), default=0)
        return 1 << (width(value) - 1) if width(value) else None

    def component_width(value):
        bits = width(value)
        return bits == 1 or (bits >= 8 and bits & (bits - 1) == 0)

    def count(value, *, mathematical=False):
        bits = width(value)
        magnitude = absolute(value)
        digits, capacity = 1, 127
        while magnitude > capacity:
            digits += 1
            capacity = 256 * capacity + 127
        return digits if mathematical else min((bits + 7) // 8, digits)

    def digit(source, index, operations):
        if width(source) == 8 and index == 0:
            return source
        wide = IntegerType(max(16, 2 * width(source)))

        def scalar(args, emit):
            remaining = emit((arith.ExtUIOp if width(source) == 1 else arith.ExtSIOp)(args[0], wide))
            if index:
                # Balanced radix carries below this digit are the geometric
                # offset 128*(1+256+...+256**(index-1)). Widening before the
                # addition preserves the carry above the source storage width.
                offset = emit(arith.ConstantOp(IntegerAttr(128 * ((1 << (8 * index)) - 1) // 255, wide)))
                remaining = emit(arith.AddiOp(remaining, offset))
                shift = emit(arith.ConstantOp(IntegerAttr(8 * index, wide)))
                remaining = emit(arith.ShRSIOp(remaining, shift))
            return emit(arith.TruncIOp(remaining, i8))

        return point([source], TensorType(i8, source.type.get_shape()), scalar, operations)

    def mark(producer, entries):
        from merlin.llvmlower.additive_regions import positive_zero_seed

        entries = list(entries)
        covered = {entry["operand"] for entry in entries}
        zero_exponent = 8 * entries[0]["index"] + entries[0]["exponent"] if entries else 0
        for index, value in enumerate(producer.inputs):
            if (
                index not in covered
                and width(value) == 8
                and not producer.body.block.args[index].uses
                and positive_zero_seed(value)
            ):
                # An unread literal byte tensor still supplies its static loop
                # domain. The plan must cover that operand as the exact one-byte
                # zero representation at the data plane's binary exponent.
                # Scaling zero by that exponent is exact for every limb index.
                entries.append(dict(operand=index, index=0, count=1, exponent=zero_exponent))
        plan = dict(version=1, family="F2", radix=256, inputs=entries)
        producer.attributes[limb_attribute] = StringAttr(json.dumps(plan, sort_keys=True))
        plans.append(plan)

    def weighted(result, component, exponent, operations):
        element = result.type.element_type

        def scalar(args, emit):
            value = args[1]
            if element.width.data > 32:
                value = emit(arith.ExtSIOp(value, element))
            elif element.width.data < 32:
                value = emit(arith.TruncIOp(value, element))
            if exponent:
                shift = emit(arith.ConstantOp(IntegerAttr(exponent, element)))
                value = emit(arith.ShLIOp(value, shift))
            return emit(arith.AddiOp(args[0], value))

        return point([result, component], result.type, scalar, operations)

    def affine(op):
        if op.name != "linalg.generic" or any(it.data != L.IteratorType.PARALLEL for it in op.get_iterator_types()):
            return None
        block = op.body.block
        expressions = {}
        for i, arg in enumerate(block.args[: len(op.inputs)]):
            source = op.inputs[i].owner
            while getattr(source, "name", None) in {"tensor.splat", "linalg.fill"}:
                source = source.operands[0].owner
            attr = (
                (source.properties.get("value") or source.attributes.get("value"))
                if getattr(source, "name", None) == "arith.constant"
                else None
            )
            expressions[arg] = ({}, attr.value.data) if isinstance(attr, IntegerAttr) else ({i: 1}, 0)
        from merlin.llvmlower.additive_regions import positive_zero_seed

        for arg, seed in zip(block.args[len(op.inputs) :], op.outputs):
            if positive_zero_seed(seed):
                # With no reduction dimensions this is one read of a literal
                # seed per output, including the rank-zero additive identity.
                expressions[arg] = ({}, 0)
        for child in block.ops:
            if child.name == "linalg.yield":
                return expressions.get(child.operands[0])
            if child.name == "arith.constant":
                attr = child.properties.get("value")
                if not isinstance(attr, IntegerAttr):
                    return None
                expressions[child.results[0]] = ({}, attr.value.data)
                continue
            if child.name in {"arith.extsi", "arith.extui", "arith.trunci"}:
                # Unexposed/compound casts cannot be polynomial substitutions:
                # extension observes the source ring's sign/modular boundary.
                return None
            if child.name not in {"arith.addi", "arith.subi", "arith.muli"}:
                return None
            if child.results[0].type.width.data < op.results[0].type.element_type.width.data:
                return None
            if any(v not in expressions for v in child.operands):
                return None
            (a, ac), (b, bc) = [expressions[v] for v in child.operands]
            if child.name == "arith.muli":
                if a and b:
                    return None
                coefficient, constant = (bc, ac * bc) if a else (ac, ac * bc)
                terms = {i: n * coefficient for i, n in (a or b).items()}
            else:
                sign = -1 if child.name == "arith.subi" else 1
                terms = a.copy()
                for i, n in b.items():
                    terms[i] = terms.get(i, 0) + sign * n
                constant = ac + sign * bc
            expressions[child.results[0]] = ({i: n for i, n in terms.items() if n}, constant)
        return None

    from merlin.kernels.shapes import observe_contractions

    contractions = dict(observe_contractions(module))

    def bilinear(op):
        if op not in contractions or len(op.operands) != 3:
            return False
        if any(not component_width(value) for value in op.operands[:2]):
            return False
        body = op.body.block
        if any(child.name not in {"arith.extsi", "arith.muli", "arith.addi", "linalg.yield"} for child in body.ops):
            return False
        yielded = body.last_op.operands[0].owner
        if getattr(yielded, "name", None) != "arith.addi" or body.args[2] not in yielded.operands:
            return False
        product_value = next((v for v in yielded.operands if v is not body.args[2]), None)
        product = getattr(product_value, "owner", None)
        if getattr(product, "name", None) != "arith.muli":
            return False
        if product.results[0].type != body.args[2].type:
            return False
        leaves = []
        for value in product.operands:
            while getattr(value.owner, "name", None) == "arith.extsi":
                value = value.owner.operands[0]
            leaves.append(value)
        if set(leaves) != set(body.args[:2]):
            return False
        typ = op.results[0].type
        bits = width(op.results[0])
        operations, result = [], op.operands[2]
        counts = [count(v, mathematical=bits > width(v)) for v in op.operands[:2]]
        digits = []
        for source, n in zip(op.operands[:2], counts, strict=True):
            values = {}
            for index in indices(n):
                values[index] = digit(source, index, operations)
            digits.append(values)
        maps = list(op.get_indexing_maps())
        for j, lhs in digits[0].items():
            for k, rhs in digits[1].items():
                exponent = 8 * (j + k)
                if exponent >= bits:
                    continue
                seed = splat(0, i32, typ.get_shape(), operations)
                producer = mac([lhs, rhs], seed, maps, op.get_iterator_types())
                mark(
                    producer,
                    [dict(operand=i, index=index, count=counts[i], exponent=0) for i, index in enumerate((j, k))],
                )
                operations.append(producer)
                result = weighted(result, producer.results[0], exponent, operations)
        bound = prod(contractions[op].reduction) * absolute(op.operands[0]) * absolute(op.operands[1])
        bounds[result] = min(bound + absolute(op.operands[2]), 1 << (bits - 1))
        Rewriter.replace_op(op, operations, [result])
        return True

    def sum_reduction(op):
        if op.name != "linalg.generic" or len(op.results) != 1 or len(op.outputs) != 1:
            return False
        if not any(it.data == L.IteratorType.REDUCTION for it in op.get_iterator_types()):
            return False
        block = op.body.block
        if len(list(block.ops)) != 2 or block.first_op.name != "arith.addi":
            return False
        if list(block.last_op.operands) != list(block.first_op.results):
            return False
        accumulator = block.args[-1]
        used = list(block.first_op.operands)
        if accumulator not in used:
            return False
        data = used[1] if used[0] is accumulator else used[0]
        if data not in block.args[:-1]:
            return False
        source_index = list(block.args).index(data)
        # Window dimensions may be supplied by unused shape-only operands.
        # Their maps determine the iteration domain but their values are unread.
        source, seed = op.inputs[source_index], op.outputs[0]
        if not component_width(source):
            return False
        bits = width(op.results[0])
        digits = count(source, mathematical=bits > width(source))
        operations, result = [], seed
        for index in indices(digits):
            if 8 * index >= bits:
                continue
            limb = digit(source, index, operations)
            zero = splat(0, i32, result.type.get_shape(), operations)
            inputs = [
                limb if j == source_index else splat(0, i8, original_input.type.get_shape(), operations)
                for j, original_input in enumerate(op.inputs)
            ]
            body = Block(arg_types=[i8] * len(inputs) + [i32])
            emit = emitter(body)
            body.add_op(L.YieldOp(emit(arith.AddiOp(emit(arith.ExtSIOp(body.args[source_index], i32)), body.args[-1]))))
            producer = generic(inputs, zero, body, op.get_indexing_maps(), op.get_iterator_types())
            mark(
                producer,
                [dict(operand=source_index, index=index, count=digits, exponent=0)],
            )
            operations.append(producer)
            result = weighted(result, producer.results[0], 8 * index, operations)
        extent = prod(source.type.get_shape()) // max(1, prod(result.type.get_shape()))
        bounds[result] = min(extent * absolute(source) + absolute(seed), 1 << (bits - 1))
        Rewriter.replace_op(op, operations, [result])
        return True

    original = list(module.walk())
    for op in original:
        if len(op.results) != 1 or not component_width(op.results[0]):
            continue
        if "merlin.exact_host_arithmetic" in op.attributes or any(
            k.startswith("prov.quant.") or k == limb_attribute for k in op.attributes
        ):
            continue
        if "merlin.exact_additive" in op.attributes:
            from merlin.llvmlower.additive_regions import data_sources, retype_additive

            sources = data_sources(op, lambda typ: isinstance(typ, IntegerType) and typ.width.data >= 128)
            if not sources:
                continue
            digits = max(count(value, mathematical=True) for value in sources)
            operations = []
            result = splat(0, op.results[0].type.element_type, op.results[0].type.get_shape(), operations)
            for index in range(digits):
                replacements, carriers = {}, []
                for source in sources:
                    limb = digit(source, index, operations)

                    def extend(args, emit):
                        return emit(arith.ExtSIOp(args[0], i32))

                    carrier = point([limb], TensorType(i32, source.type.get_shape()), extend, operations)
                    replacements[source] = carrier
                    carriers.append(carrier)
                lifted = retype_additive(op, replacements, i32, floating=False)
                if lifted is None:
                    break
                producer_ops, producer = lifted
                operations.extend(producer_ops)
                mark(
                    producer,
                    [
                        dict(operand=list(producer.inputs).index(v), index=index, count=digits, exponent=0)
                        for v in carriers
                    ],
                )
                result = weighted(result, producer.results[0], 8 * index, operations)
            else:
                bounds[result] = absolute(op.results[0])
                Rewriter.replace_op(op, operations, [result])
                continue
        if bilinear(op) or sum_reduction(op):
            continue
        expression = affine(op)
        if expression is None:
            continue
        terms, bias = expression
        if not terms or any(not component_width(op.inputs[i]) for i in terms):
            continue
        typ = op.results[0].type
        bits = width(op.results[0])
        # Constants are ring elements too. Large coefficients need their own
        # signed-byte planes; no coefficient or product is truncated to a byte.
        bias = (bias + (1 << (bits - 1))) % (1 << bits) - (1 << (bits - 1))
        operations = []
        result = splat(bias, typ.element_type, typ.get_shape(), operations)
        for i, coefficient in terms.items():
            source = op.inputs[i]
            digits = count(source, mathematical=bits > width(source))
            planes = [(0, coefficient)]
            if abs(coefficient) * 128 >= (1 << 31):
                remaining = (coefficient + (1 << (bits - 1))) % (1 << bits) - (1 << (bits - 1))
                planes = []
                position = 0
                while remaining:
                    digit_coefficient = (remaining + 128) % 256 - 128
                    if digit_coefficient:
                        planes.append((position, digit_coefficient))
                    remaining = (remaining - digit_coefficient) // 256
                    position += 1
            for index, plane, factor in (
                (index, plane, factor) for index in indices(digits) for plane, factor in planes
            ):
                exponent = 8 * (index + plane)
                if exponent >= bits:
                    continue
                limb = digit(source, index, operations)
                seed = splat(0, i32, typ.get_shape(), operations)
                body = Block(arg_types=[i8, i32])
                emit = emitter(body)
                extended = emit(arith.ExtSIOp(body.args[0], i32))
                body.add_op(L.YieldOp(emit(arith.MuliOp(extended, emit(arith.ConstantOp(IntegerAttr(factor, i32)))))))
                maps = list(op.get_indexing_maps())
                producer = generic([limb], seed, body, [maps[i], maps[-1]], op.get_iterator_types())
                mark(
                    producer,
                    [dict(operand=0, index=index, count=digits, exponent=0)],
                )
                operations.append(producer)
                result = weighted(result, producer.results[0], exponent, operations)
        bound = abs(bias) + sum(abs(c) * absolute(op.inputs[i]) for i, c in terms.items())
        bounds[result] = min(bound, 1 << (bits - 1))
        Rewriter.replace_op(op, operations, [result])
    module.verify()
    return plans
