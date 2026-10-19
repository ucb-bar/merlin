"""Checked interval proof for complete typed integer tensor SSA programs.

This reconstructs source arithmetic, including every reduction prefix and live
output. It neither chooses quantization nor certifies target instructions. The
separate admission owner must bind its selected software and executable effects.
"""

from __future__ import annotations

from math import prod

from xdsl.dialects import builtin
from xdsl.ir.affine import AffineConstantExpr, AffineDimExpr

from merlin.common import mlir_query as mq

SCHEMA = "merlin.connected_integer_source_bounds.v1"


def check_arithmetic_rule(proof, rule):
    """Check source ranges against an independently verified target panel scope.

    This function does not grant the rule or prove that a compiler realizes it.
    The host-selected realization owner supplies those separate obligations.
    """
    keys = {"operand_bits", "accumulator_bits", "primitive_mac_bits", "max_primitive_k", "accumulation"}
    if (
        not isinstance(rule, dict)
        or set(rule) != keys
        or rule["accumulation"] != "exact_i32_before_readout"
        or rule["accumulator_bits"] != 32
        or any(type(rule[k]) is not int or not 1 <= rule[k] <= 64 for k in keys - {"accumulation", "max_primitive_k"})
        or type(rule["max_primitive_k"]) is not int
        or not 1 <= rule["max_primitive_k"] <= 2**31
    ):
        raise ValueError("connected integer target has no supported independently selected arithmetic scope")
    checked = []
    for row in proof["operations"]:
        if row.get("kind") != "multiply_accumulate":
            continue
        if (
            row["operand_bits"] != [rule["operand_bits"]] * 2
            or row["initial_interval"] != [0, 0]
            or len(row["product_intervals"]) != 1
            or row["scalar_operations"] != ["arith.extsi", "arith.extsi", "arith.muli", "arith.addi", "linalg.yield"]
        ):
            raise ValueError("source contraction is not the selected zero-initialized signed primitive recurrence")
        product = row["product_intervals"][0]
        k = min(row["reduction_extent"], rule["max_primitive_k"])
        panel = min(0, k * product[0]), max(0, k * product[1])
        _fit(panel, rule["primitive_mac_bits"])
        _fit(row["prefix_interval"], rule["accumulator_bits"])
        checked.append(
            dict(
                source_operation=row["ordinal"],
                full_k=row["reduction_extent"],
                max_primitive_k=k,
                primitive_prefix_interval=list(panel),
                primitive_mac_bits=rule["primitive_mac_bits"],
                full_prefix_interval=row["prefix_interval"],
                accumulator_bits=rule["accumulator_bits"],
            )
        )
    if not checked:
        raise ValueError("connected source has no selected primitive arithmetic to check")
    return checked


def _bits(typ):
    element = typ.get_element_type() if isinstance(typ, builtin.ShapedType) else typ
    if not isinstance(element, builtin.IntegerType) or element.width.data not in {1, 8, 32, 64}:
        raise ValueError("connected integer proof requires explicit supported integral types")
    return element.width.data


def _shape(value):
    typ = value.type
    if not isinstance(typ, builtin.TensorType) or any(d < 1 for d in typ.get_shape()):
        raise ValueError("connected integer tensor proof requires positive static tensor types")
    _bits(typ)
    return tuple(typ.get_shape())


def _fit(interval, bits):
    if (
        interval is None
        or len(interval) != 2
        or any(type(n) is not int for n in interval)
        or interval[0] > interval[1]
        or interval[0] < -(1 << (bits - 1))
        or interval[1] >= 1 << (bits - 1)
    ):
        raise ValueError(f"connected integer source interval may overflow signed i{bits}: {interval}")
    return interval


def _union(intervals):
    if not intervals or any(row is None for row in intervals):
        raise ValueError("connected integer source reads an undefined tensor")
    return min(row[0] for row in intervals), max(row[1] for row in intervals)


def _constant(op):
    attr = op.properties.get("value")
    if not isinstance(attr, builtin.IntegerAttr):
        raise ValueError("connected integer source has a nonintegral or dense constant")
    return int(attr.value.data)


def _scalar(op, env):
    name = op.name
    args = [env[v] for v in op.operands]
    if any(row is None for row in args):
        raise ValueError("connected integer scalar operation reads undefined storage")
    if name == "arith.constant":
        result = (_constant(op),) * 2
    elif name in {"arith.extsi", "arith.trunci", "arith.index_cast"}:
        result = args[0]
    elif name == "arith.addi":
        result = args[0][0] + args[1][0], args[0][1] + args[1][1]
    elif name == "arith.subi":
        result = args[0][0] - args[1][1], args[0][1] - args[1][0]
    elif name == "arith.muli":
        products = [a * b for a in args[0] for b in args[1]]
        result = min(products), max(products)
    elif name in {"arith.minsi", "arith.maxsi"}:
        fn = min if name == "arith.minsi" else max
        result = fn(args[0][0], args[1][0]), fn(args[0][1], args[1][1])
    elif name == "arith.floordivsi":
        if args[1][0] != args[1][1] or args[1][0] < 1:
            raise ValueError("integer source floor division needs an exact positive divisor")
        result = args[0][0] // args[1][0], args[0][1] // args[1][0]
    elif name == "arith.cmpi":
        return (0, 1)
    elif name == "arith.select":
        result = _union(args[1:])
        comparison = op.operands[0].owner
        if comparison.name == "arith.cmpi":
            x, bound = comparison.operands
            predicate = comparison.properties["predicate"].value.data
            if op.operands[1] is bound and op.operands[2] is x and env[bound][0] == env[bound][1]:
                if predicate in {2, 4}:  # signed x<bound ? bound:x; signed x>bound ? bound:x
                    fn = max if predicate == 2 else min
                    result = fn(env[x][0], env[bound][0]), fn(env[x][1], env[bound][0])
    else:
        raise ValueError(f"connected integer scalar operation unsupported: {name}")
    return _fit(result, _bits(op.results[0].type))


def _loops(op):
    maps = [a.data for a in op.properties["indexing_maps"].data]
    kinds = [a.data.value for a in op.properties["iterator_types"].data]
    if len(maps) != len(op.operands) or any(m.num_symbols or m.num_dims != len(kinds) for m in maps):
        raise ValueError("integer generic has incomplete loop-to-operand maps")
    sizes = {}
    for operand, mapping in zip(op.operands, maps, strict=True):
        shape = _shape(operand)
        if len(mapping.results) != len(shape):
            raise ValueError("integer generic map rank differs from its typed operand")
        for axis, extent in zip(mapping.results, shape, strict=True):
            if isinstance(axis, AffineDimExpr):
                if axis.position in sizes and sizes[axis.position] != extent:
                    raise ValueError("integer generic loop extents disagree")
                sizes[axis.position] = extent
            elif not isinstance(axis, AffineConstantExpr) or axis.value != 0 or extent != 1:
                raise ValueError("integer generic map is not an exact projected permutation")
    if set(sizes) != set(range(len(kinds))) or any(k not in {"parallel", "reduction"} for k in kinds):
        raise ValueError("integer generic has unknown or unbounded loop geometry")
    return kinds, sizes, maps


def _generic(op, values):
    if len(op.results) != 1 or len(op.regions) != 1 or len(op.regions[0].blocks) != 1:
        raise ValueError("integer generic requires one complete result and one scalar block")
    kinds, sizes, maps = _loops(op)
    if op.results[0].type != op.operands[-1].type:
        raise ValueError("integer generic output differs from its complete typed initializer")
    block = op.regions[0].block
    if len(block.args) != len(op.operands) or block.last_op.name != "linalg.yield":
        raise ValueError("integer generic scalar ABI is incomplete")
    env = dict(zip(block.args, [values[v] for v in op.operands], strict=True))
    reductions = [i for i, kind in enumerate(kinds) if kind == "reduction"]
    record = dict(
        kind="pointwise",
        result_bits=_bits(op.results[0].type),
        input_shapes=[list(_shape(v)) for v in op.operands[:-1]],
        output_shape=list(_shape(op.results[0])),
        scalar_operations=[scalar.name for scalar in block.ops],
        scalar_literals=[_constant(scalar) for scalar in block.ops if scalar.name == "arith.constant"],
    )
    if reductions:
        # A closed linear reduction, not a guessed final-result bound. The
        # accumulator appears exactly once, as an operand of its final addition.
        yielded = block.last_op.operands[0]
        add, acc = yielded.owner, block.args[-1]
        if add.name != "arith.addi" or list(add.operands).count(acc) != 1:
            raise ValueError("integer reduction is not a checked additive recurrence")
        if any(acc in scalar.operands for scalar in block.ops if scalar is not add):
            raise ValueError("integer reduction accumulator participates in an unsupported recurrence")
        if any(isinstance(axis, AffineDimExpr) and axis.position in reductions for axis in maps[-1].results):
            raise ValueError("integer reduction output varies along its reduction dimension")
        initial = env[acc]
        if initial is None:
            raise ValueError("integer reduction begins from undefined storage")
        env[acc] = (0, 0)
        for scalar in block.ops:
            if scalar is block.last_op:
                break
            if len(scalar.results) != 1 or scalar.regions:
                raise ValueError("integer reduction contains unsupported nested scalar effects")
            env[scalar.results[0]] = _scalar(scalar, env)
        addend = env[yielded]
        count = prod(sizes[i] for i in reductions)
        interval = initial[0] + count * addend[0], initial[1] + count * addend[1]
        prefix = min(initial[0], interval[0]), max(initial[1], interval[1])
        _fit(prefix, _bits(op.results[0].type))
        products = [scalar for scalar in block.ops if scalar.name == "arith.muli"]
        record.update(
            kind="multiply_accumulate" if products else "sum_reduction",
            reduction_extent=count,
            initial_interval=list(initial),
            addend_interval=list(addend),
            prefix_interval=list(prefix),
            product_intervals=[list(env[scalar.results[0]]) for scalar in products],
            operand_bits=[_bits(v.type) for v in block.args[:-1]],
        )
    else:
        for scalar in block.ops:
            if scalar is block.last_op:
                break
            if len(scalar.results) != 1 or scalar.regions:
                raise ValueError("integer pointwise generic contains unsupported effects")
            env[scalar.results[0]] = _scalar(scalar, env)
        interval = env[block.last_op.operands[0]]
        _fit(interval, _bits(op.results[0].type))
    return interval, record


def _transpose(op, args):
    """Only the complete pure typed permutation; never initializer contents."""
    permutation = op.properties.get("permutation")
    body = op.regions[0].block if len(op.regions) == 1 and len(op.regions[0].blocks) == 1 else None
    if (
        len(op.operands) != 2 or len(op.results) != 1 or args[0] is None
        or not isinstance(permutation, builtin.DenseArrayBase)
        or any(not key.startswith("prov.") and key != "permutation"
               for key in {**op.attributes, **op.properties})
        or body is None or len(body.args) != 2 or len(body.ops) != 1
        or body.last_op.name != "linalg.yield" or list(body.last_op.operands) != [body.args[0]]
        or any(not key.startswith("prov.") for key in {**body.last_op.attributes, **body.last_op.properties})
        or body.args[0].type != op.operands[0].type.get_element_type()
        or body.args[1].type != op.operands[1].type.get_element_type()
        or op.operands[1].type != op.results[0].type
        or _bits(op.operands[0].type) != _bits(op.results[0].type)
    ):
        raise ValueError("integer transpose is not a complete pure typed source permutation")
    source, result = _shape(op.operands[0]), _shape(op.results[0])
    axes = list(permutation.get_values())
    if sorted(axes) != list(range(len(source))) or tuple(source[axis] for axis in axes) != result:
        raise ValueError("integer transpose permutation differs from complete tensor geometry")
    return args[0], dict(kind="layout_permutation", permutation=axes,
                         input_shape=list(source), output_shape=list(result),
                         input_bits=_bits(op.operands[0].type), result_bits=_bits(op.results[0].type))


def derive(module, input_intervals):
    """Prove the complete closed source; target arithmetic remains unverified."""
    module.verify()
    functions = list(mq.walk(module, "func.func"))
    if len(functions) != 1 or list(module.body.block.ops) != functions or len(functions[0].regions[0].blocks) != 1:
        raise ValueError("connected integer source needs one closed single-block function")
    block = functions[0].regions[0].block
    if len(block.args) != len(input_intervals):
        raise ValueError("connected integer source input ownership or cardinality differs")
    values, records, owners = {}, [], {}
    for position, (argument, interval) in enumerate(zip(block.args, input_intervals, strict=True)):
        _shape(argument)
        values[argument] = _fit(tuple(interval), _bits(argument.type))
        owners[argument] = dict(argument=position)
    outputs = None
    for index, op in enumerate(block.ops):
        name = op.name
        args = [values[v] for v in op.operands]
        record = dict(ordinal=index, operation=name, inputs=[owners[v] for v in op.operands])
        if name == "func.return":
            if op is not block.last_op or not args or any(a is None for a in args):
                raise ValueError("connected integer source has incomplete live output ownership")
            outputs = [list(a) for a in args]
            continue
        if len(op.results) != 1:
            raise ValueError("connected integer source has an unsupported result/effect ABI")
        if name == "linalg.generic":
            interval, arithmetic = _generic(op, values)
            record.update(arithmetic)
        elif name == "linalg.transpose":
            interval, layout = _transpose(op, args)
            record.update(layout)
        elif name == "arith.constant":
            interval = _scalar(op, values)
        elif name == "tensor.empty":
            if op.operands:
                raise ValueError("integer empty has a dynamic extent")
            interval = None
        elif name == "tensor.splat":
            interval = args[0]
        elif name in {"tensor.expand_shape", "tensor.collapse_shape", "tensor.cast"}:
            if len(args) != 1 or prod(_shape(op.operands[0])) != prod(_shape(op.results[0])):
                raise ValueError("integer layout operation changes the complete element count")
            interval = args[0]
        elif name == "tensor.concat":
            interval = _union(args)
        elif name == "tensor.extract_slice":
            if len(args) != 1:
                raise ValueError("integer slice has unproved dynamic indices")
            interval = args[0]
        elif name == "tensor.pad":
            body = op.regions[0].block if len(op.regions) == 1 else None
            if (
                body is None
                or body.last_op.name != "tensor.yield"
                or any(scalar.name not in {"arith.constant", "tensor.yield"} for scalar in body.ops)
            ):
                raise ValueError("integer pad has an unsupported value/effect body")
            constant = body.last_op.operands[0].owner
            interval = _union([args[0], (_constant(constant),) * 2])
        else:
            raise ValueError(f"connected integer tensor operation unsupported: {name}")
        if isinstance(op.results[0].type, builtin.TensorType):
            _shape(op.results[0])
        if interval is not None:
            _fit(interval, _bits(op.results[0].type))
        values[op.results[0]] = interval
        owners[op.results[0]] = dict(operation=index, result=0)
        records.append({**record, "interval": list(interval) if interval is not None else None})
    if outputs is None or not any(r.get("kind") == "multiply_accumulate" for r in records):
        raise ValueError("connected integer program lacks complete outputs or accelerator contraction work")
    return dict(
        schema=SCHEMA,
        status="source_arithmetic_proven",
        input_intervals=[list(a) for a in input_intervals],
        input_abi=[dict(shape=list(_shape(v)), bits=_bits(v.type)) for v in block.args],
        operations=records,
        output_intervals=outputs,
        output_owners=[owners[v] for v in block.last_op.operands],
        output_abi=[dict(shape=list(_shape(v)), bits=_bits(v.type)) for v in block.last_op.operands],
        target_arithmetic_status="UNVERIFIED: independent selected executable arithmetic scope required",
        scope="all typed source operations and additive reduction prefixes; no target effects or performance proof",
    )
