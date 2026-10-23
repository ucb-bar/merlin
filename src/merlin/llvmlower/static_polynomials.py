"""Closed static scalar regions as bounded degree-two rational polynomials.

Only static index/control expressions are evaluated. Tensor payloads remain SSA
atoms; no observed output, host floating execution, or provenance selects terms.
The caller must separately authorize reassociation and prove finite packing.
"""

from fractions import Fraction
from math import prod


class PolynomialRefusal(ValueError):
    pass


def analyze_static_polynomial(
    op, *, work_limit=1_000_000, require_bilinear=True, statistics=None, ignore_float_rounding=False
):
    from xdsl.dialects.builtin import FloatAttr, IntegerAttr
    from xdsl.ir.affine import AffineDimExpr

    from merlin.targetgen.closed_scalar import predicate_holds

    if op.name != "linalg.generic" or len(op.results) != 1:
        raise PolynomialRefusal("one generic result required")
    shape = op.results[0].type.get_shape()
    maps = list(op.get_indexing_maps())
    kinds = [it.data.value for it in op.get_iterator_types()]
    output_dimensions = maps[-1].data.results
    if (
        prod(shape) > 4096
        or any(not isinstance(e, AffineDimExpr) for e in output_dimensions)
        or sorted(e.position for e in output_dimensions) != [i for i, k in enumerate(kinds) if k == "parallel"]
    ):
        raise PolynomialRefusal("bounded projected output map required")
    extents = [None] * len(kinds)
    for value, amap in zip(op.operands, maps, strict=True):
        for expression, extent in zip(amap.data.results, value.type.get_shape(), strict=True):
            if isinstance(expression, AffineDimExpr):
                position = expression.position
                if extents[position] not in (None, extent):
                    raise PolynomialRefusal("inconsistent static iteration dimension")
                extents[position] = extent
    if any(n is None or n <= 0 for n in extents) or prod(extents) > work_limit:
        raise PolynomialRefusal("unproved or over-budget static iteration domain")
    atoms, atom_ids, work, floating_work = [], {}, 0, 0

    def atom(source, index):
        from merlin.xdsl_dialects.lowering.group_numerics import _constant_of

        uniform = _constant_of(source)
        if uniform is not None:
            try:
                value = Fraction.from_float(float(uniform))
            except (ValueError, OverflowError) as exc:
                raise PolynomialRefusal("nonfinite tensor literal") from exc
            return {(): value} if value else {}
        key = source, tuple(index)
        if key not in atom_ids:
            atom_ids[key] = len(atoms)
            atoms.append(key)
        return {(atom_ids[key],): Fraction(1)}

    def literal(value):
        owner = value.owner
        if getattr(owner, "name", None) != "arith.constant":
            raise PolynomialRefusal("unbound scalar capture")
        attr = owner.properties.get("value", owner.attributes.get("value"))
        if isinstance(attr, FloatAttr):
            try:
                return {(): Fraction.from_float(float(attr.value.data))} if attr.value.data else {}
            except (ValueError, OverflowError) as exc:
                raise PolynomialRefusal("nonfinite literal") from exc
        if isinstance(attr, IntegerAttr):
            return int(attr.value.data)
        raise PolynomialRefusal("unsupported literal")

    def add(a, b, sign=1):
        if not isinstance(a, dict) or not isinstance(b, dict):
            raise PolynomialRefusal("data/control type mismatch")
        result = a.copy()
        for key, coefficient in b.items():
            result[key] = result.get(key, Fraction(0)) + sign * coefficient
        return {key: value for key, value in result.items() if value}

    def multiply(a, b):
        result = {}
        for ka, ca in a.items():
            for kb, cb in b.items():
                key = tuple(sorted((*ka, *kb)))
                if len(key) > 2:
                    raise PolynomialRefusal("polynomial degree exceeds two")
                result[key] = result.get(key, Fraction(0)) + ca * cb
        return {key: value for key, value in result.items() if value}

    def block(region_block, arguments, environment, point):
        nonlocal work, floating_work
        env = dict(environment)
        env.update(zip(region_block.args, arguments, strict=True))

        def get(value):
            return env[value] if value in env else literal(value)

        for child in region_block.ops:
            work += 1
            if work > work_limit:
                raise PolynomialRefusal("static polynomial work budget exceeded")
            name = child.name
            floating_work += name in {"arith.addf", "arith.subf", "arith.mulf", "arith.divf"}
            if getattr(child.properties.get("fastmath"), "data", frozenset()):
                raise PolynomialRefusal("exact static polynomials refuse floating fastmath flags")
            if name in {"scf.yield", "linalg.yield"}:
                return tuple(get(v) for v in child.operands)
            args = [] if name in {"tensor.extract", "scf.for", "scf.if"} else [get(v) for v in child.operands]
            if name == "arith.constant":
                result = literal(child.results[0])
            elif name == "linalg.index":
                result = point[child.dim.value.data]
            elif name == "tensor.extract":
                source = child.operands[0]
                index = tuple(get(v) for v in child.operands[1:])
                if any(
                    type(i) is not int or not 0 <= i < n for i, n in zip(index, source.type.get_shape(), strict=True)
                ):
                    raise PolynomialRefusal("unproved tensor address")
                result = atom(source, index)
            elif name == "scf.for":
                lower, upper, step = [get(v) for v in child.operands[:3]]
                if any(type(v) is not int for v in (lower, upper, step)) or step <= 0:
                    raise PolynomialRefusal("loop bounds must be static positive-step integers")
                carried = tuple(get(v) for v in child.operands[3:])
                for index in range(lower, upper, step):
                    carried = block(child.body.block, (index, *carried), env, point)
                env.update(zip(child.results, carried, strict=True))
                continue
            elif name == "scf.if":
                condition = get(child.operands[0])
                if type(condition) is not int:
                    raise PolynomialRefusal("data-dependent branch")
                results = block(child.regions[0 if condition else 1].block, (), env, point)
                env.update(zip(child.results, results, strict=True))
                continue
            elif name in {"arith.addf", "arith.subf"}:
                result = add(*args, sign=-1 if name == "arith.subf" else 1)
            elif name == "arith.mulf":
                result = multiply(*args)
            elif ignore_float_rounding and name == "math.fma":
                result = add(multiply(args[0], args[1]), args[2])
            elif name == "arith.divf":
                divisor = args[1]
                if set(divisor) != {()} or not divisor[()]:
                    raise PolynomialRefusal("division requires a nonzero rational literal")
                result = {key: value / divisor[()] for key, value in args[0].items()}
            elif ignore_float_rounding and name in {"arith.sitofp", "arith.uitofp"} and type(args[0]) is int:
                from merlin.targetgen.exact_arithmetic import round_binary

                value = args[0]
                if name == "arith.uitofp":
                    value %= 1 << child.operands[0].type.width.data
                token = str(child.results[0].type)
                value = round_binary(
                    value, {"f16": 16, "bf16": 16, "f32": 32, "f64": 64}[token], bfloat16=token == "bf16"
                )
                result = {(): Fraction(value)} if value else {}
            elif name == "arith.extf" or (ignore_float_rounding and name == "arith.truncf"):
                result = args[0]
            elif name == "arith.negf":
                result = {key: -value for key, value in args[0].items()}
            elif ignore_float_rounding and name == "math.absf" and set(args[0]) <= {()}:
                result = {(): abs(args[0][()])} if args[0] else {}
            elif ignore_float_rounding and name == "arith.cmpf" and all(set(v) <= {()} for v in args):
                a, b = (v.get((), Fraction(0)) for v in args)
                result = int(predicate_holds(child.predicate.value.data, (a > b) - (a < b), floating=True))
            elif name == "arith.cmpi":
                if any(type(v) is not int for v in args):
                    raise PolynomialRefusal("data-dependent predicate")
                a, b = args
                if child.predicate.value.data >= 6:
                    # Unsigned order places all negative index values after
                    # nonnegative values, independently of the index width.
                    # Same-sign ordering agrees with mathematical integers.
                    if (a < 0) != (b < 0):
                        a, b = int(a < 0), int(b < 0)
                result = int(predicate_holds(child.predicate.value.data, (a > b) - (a < b)))
            elif name == "arith.select":
                if type(args[0]) is not int:
                    raise PolynomialRefusal("data-dependent select")
                result = args[1] if args[0] else args[2]
            elif name in {
                "arith.addi",
                "arith.subi",
                "arith.muli",
                "arith.divui",
                "arith.remui",
                "arith.andi",
                "arith.ori",
                "arith.minsi",
                "arith.maxsi",
                "arith.minui",
                "arith.maxui",
            }:
                if any(type(v) is not int for v in args):
                    raise PolynomialRefusal("non-index integer expression")
                a, b = args
                result = {
                    "arith.addi": lambda: a + b,
                    "arith.subi": lambda: a - b,
                    "arith.muli": lambda: a * b,
                    "arith.divui": lambda: a // b,
                    "arith.remui": lambda: a % b,
                    "arith.andi": lambda: a & b,
                    "arith.ori": lambda: a | b,
                    "arith.minsi": lambda: min(a, b),
                    "arith.maxsi": lambda: max(a, b),
                    "arith.minui": lambda: min((a, b), key=lambda v: (v < 0, v)),
                    "arith.maxui": lambda: max((a, b), key=lambda v: (v < 0, v)),
                }[name]()
            else:
                raise PolynomialRefusal("unsupported static polynomial operation " + name)
            if len(child.results) != 1:
                raise PolynomialRefusal("one scalar result required")
            env[child.results[0]] = result
        raise PolynomialRefusal("missing region terminator")

    from itertools import product

    output_points = list(product(*(range(n) for n in shape)))
    rows_by_index = {
        point: atom(op.outputs[0], point) if op.body.block.args[-1].uses else {} for point in output_points
    }
    for point in product(*(range(n) for n in extents)):
        output_index = maps[-1].data.eval(point, ())
        arguments = []
        for i, (value, amap) in enumerate(zip(op.operands, maps, strict=True)):
            if i == len(op.inputs):
                arguments.append(rows_by_index[output_index])
            elif not op.body.block.args[i].uses:
                arguments.append({})
            else:
                coordinates = amap.data.eval(point, ())
                if any(
                    not 0 <= coordinate < extent
                    for coordinate, extent in zip(coordinates, value.type.get_shape(), strict=True)
                ):
                    raise PolynomialRefusal("static operand address exceeds its tensor shape")
                arguments.append(atom(value, coordinates))
        (result,) = block(op.body.block, arguments, {}, point)
        if not isinstance(result, dict):
            raise PolynomialRefusal("floating polynomial required")
        rows_by_index[output_index] = result
    rows = [rows_by_index[point] for point in output_points]
    if require_bilinear and not any(len(key) == 2 for row in rows for key in row):
        raise PolynomialRefusal("no bilinear data producer")
    if statistics is not None:
        statistics.update(static_work=work, floating_work=floating_work)
    return atoms, rows
