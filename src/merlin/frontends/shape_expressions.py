"""Bounded typed integer/size programs; declarations supply no source proof.

Index arithmetic is deliberately limited to width-independent comparisons and
selection. Fixed-width words retain modular arithmetic and signedness. A
consumer must independently reconstruct these nodes from its executable IR.
"""

import hashlib
import json

MAX_SIZE = (1 << 63) - 1
SCHEMA = "logical-size-program.v1"
PREDICATES = ("eq", "ne", "slt", "sle", "sgt", "sge", "ult", "ule", "ugt", "uge")
BINARY = {"addi", "subi", "muli", "andi", "ori", "xori", "divui", "divsi", "floordivsi", "remui", "remsi"}
CASTS = {"extui", "extsi", "trunci", "index_castui", "index_cast"}


def signed(value, width):
    return value - (1 << width) if value >= 1 << (width - 1) else value


def _scalar(op, values, types, type_, predicate=None):
    if op == "cmpi":
        a, b = values
        if predicate.startswith("s"):
            a, b = (signed(value, types[0]) for value in values)
        relation = predicate if predicate in ("eq", "ne") else predicate[1:]
        return int({"eq": a == b, "ne": a != b, "lt": a < b, "le": a <= b, "gt": a > b, "ge": a >= b}[relation])
    if op == "select":
        return values[1] if values[0] else values[2]
    if op in CASTS:
        value = values[0]
        if op in ("extsi", "index_cast") and types[0] != "index":
            value = signed(value, types[0])
        if type_ == "index":
            if not 0 <= value <= MAX_SIZE:
                raise ValueError("shape expression does not fit a nonnegative logical size")
            return value
        return value % (1 << type_)
    a, b = values
    if op == "addi":
        value = a + b
    elif op == "subi":
        value = a - b
    elif op == "muli":
        value = a * b
    elif op == "andi":
        value = a & b
    elif op == "ori":
        value = a | b
    elif op == "xori":
        value = a ^ b
    else:
        if op in ("divsi", "floordivsi", "remsi"):
            a, b = signed(a, type_), signed(b, type_)
            if op != "remsi" and a == -(1 << (type_ - 1)) and b == -1:
                raise ValueError("shape expression signed division overflows")
        if b == 0:
            raise ValueError("shape expression division has a zero divisor")
        if op == "divui":
            value = a // b
        elif op == "remui":
            value = a % b
        elif op == "floordivsi":
            value = a // b
        else:
            quotient = (-1 if (a < 0) != (b < 0) else 1) * (abs(a) // abs(b))
            value = quotient if op == "divsi" else a - quotient * b
    return value % (1 << type_)


def _join(intervals):
    result = []
    for low, high in sorted((a, b) for a, b in intervals if a <= b):
        if result and low <= result[-1][1] + 1:
            result[-1] = result[-1][0], max(result[-1][1], high)
        else:
            result.append((low, high))
    return result if len(result) <= 8 else [(result[0][0], result[-1][1])]


def _intersect(left, right):
    return _join((max(a, c), min(b, d)) for a, b in left for c, d in right)


def _wrap(low, high, width):
    modulus = 1 << width
    if high - low >= modulus - 1:
        return [(0, modulus - 1)]
    if low // modulus == high // modulus:
        return [(low % modulus, high % modulus)]
    return [(0, high % modulus), (low % modulus, modulus - 1)]


def _signed_intervals(intervals, width):
    half, modulus = 1 << (width - 1), 1 << width
    return _join(
        [(low, min(high, half - 1)) for low, high in intervals if low < half]
        + [(max(low, half) - modulus, high - modulus) for low, high in intervals if high >= half]
    )


def _allowed(predicate, value, width):
    modulus = 1 << width
    if predicate == "eq":
        return [(value, value)]
    if predicate == "ne":
        return [(0, value - 1), (value + 1, modulus - 1)]
    is_signed = predicate.startswith("s")
    bound = signed(value, width) if is_signed else value
    low, high = (-(modulus >> 1), (modulus >> 1) - 1) if is_signed else (0, modulus - 1)
    relation = predicate[1:]
    if relation == "lt":
        high = bound - 1
    elif relation == "le":
        high = bound
    elif relation == "gt":
        low = bound + 1
    else:
        low = bound
    return _join(_wrap(low, high, width)) if low <= high else []


class Program:
    def __init__(self, nodes=None):
        self.nodes = dict(nodes or {})
        self.visits = 0

    def _visit(self, depth):
        self.visits += 1
        if depth >= 64 or self.visits > 16384 or len(self.nodes) > 4096:
            raise ValueError("shape expression program exceeds its finite source budget")

    def type(self, name):
        row = self.nodes[name]
        if row["op"] == "axis":
            return "index"
        if row["op"] == "cmpi":
            return 1
        return row["type"]

    def intern(self, op, *, args=(), type_=None, value=None, axis=None, predicate=None):
        row = {"op": op}
        if op == "axis":
            row["axis"] = axis
        else:
            if op != "cmpi":
                row["type"] = type_
            if op == "const":
                row["value"] = value
            else:
                row["args"] = list(args)
            if op == "cmpi":
                row["predicate"] = predicate
        if op not in ("axis", "const") and all(self.nodes[name]["op"] == "const" for name in args):
            try:
                number = _scalar(
                    op,
                    [self.nodes[name]["value"] for name in args],
                    [self.type(name) for name in args],
                    1 if op == "cmpi" else type_,
                    predicate,
                )
                return self.intern("const", type_=1 if op == "cmpi" else type_, value=number)
            except ValueError:
                pass
        if op in ("addi", "subi", "muli") and args:
            right = self.nodes[args[1]]
            if right["op"] == "const" and right["value"] == (1 if op == "muli" else 0):
                return args[0]
        name = "e_" + hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()[:32]
        if name in self.nodes and self.nodes[name] != row:
            raise ValueError("shape expression identifier collision")
        self.nodes[name] = row
        if len(self.nodes) > 4096:
            raise ValueError("shape expression program exceeds its finite node budget")
        return name

    def dimension(self, value):
        if type(value) is int:
            return self.intern("const", type_="index", value=value)
        if value in self.nodes:
            return value
        return self.intern("axis", axis=value)

    def export(self, roots):
        self.visits = 0
        result, pending = {}, list(roots)
        while pending:
            name = pending.pop()
            if name in result:
                continue
            self._visit(0)
            result[name] = self.nodes[name]
            pending.extend(self.nodes[name].get("args", []))
        return {"schema": SCHEMA, "nodes": result}

    def evaluate(self, name, axes, depth=0):
        if depth == 0:
            self.visits, self._values = 0, {}
        self._visit(depth)
        if name in self._values:
            return self._values[name]
        row = self.nodes[name]
        op = row["op"]
        if op == "axis":
            result = axes[row["axis"]]
        elif op == "const":
            result = row["value"]
        elif op == "select":
            args = row["args"]
            condition = self.evaluate(args[0], axes, depth + 1)
            result = self.evaluate(args[1] if condition else args[2], axes, depth + 1)
        else:
            args = row["args"]
            values = [self.evaluate(item, axes, depth + 1) for item in args]
            result = _scalar(op, values, [self.type(item) for item in args], self.type(name), row.get("predicate"))
        self._values[name] = result
        return result

    def bounds(self, name, axis_bounds, restrictions=None, depth=0):
        if depth == 0:
            self.visits, self._ranges = 0, {}
        self._visit(depth)
        restrictions = restrictions or {}
        key = name, tuple(sorted((node, tuple(intervals)) for node, intervals in restrictions.items()))
        if key in self._ranges:
            return self._ranges[key]
        row, result = self.nodes[name], None
        op = row["op"]
        if op == "axis":
            interval = axis_bounds(row["axis"])
            result = None if interval is None else [tuple(interval)]
        elif op == "const":
            result = [(row["value"], row["value"])]
        elif op == "select":
            condition = self.bounds(row["args"][0], axis_bounds, restrictions, depth + 1)
            branches = []
            for truth, argument in ((1, row["args"][1]), (0, row["args"][2])):
                if condition is not None and not any(low <= truth <= high for low, high in condition):
                    continue
                refined = self._assume(row["args"][0], truth, restrictions)
                interval = self.bounds(argument, axis_bounds, refined, depth + 1)
                if interval is None:
                    return None
                branches.extend(interval)
            result = _join(branches)
        else:
            intervals = [self.bounds(item, axis_bounds, restrictions, depth + 1) for item in row["args"]]
            if any(item is None for item in intervals):
                return None
            types = [self.type(item) for item in row["args"]]
            result = self._operation_bounds(row, intervals, types)
        if result is not None and name in restrictions:
            result = _intersect(result, restrictions[name])
        self._ranges[key] = result
        return result

    def _assume(self, condition, truth, restrictions):
        row = self.nodes[condition]
        if row["op"] != "cmpi":
            return restrictions
        left, right = row["args"]
        predicate = row["predicate"]
        if self.nodes[right]["op"] != "const":
            if self.nodes[left]["op"] != "const":
                return restrictions
            left, right = right, left
            predicate = {
                "eq": "eq",
                "ne": "ne",
                "slt": "sgt",
                "sle": "sge",
                "sgt": "slt",
                "sge": "sle",
                "ult": "ugt",
                "ule": "uge",
                "ugt": "ult",
                "uge": "ule",
            }[predicate]
        if not truth:
            predicate = {
                "eq": "ne",
                "ne": "eq",
                "slt": "sge",
                "sle": "sgt",
                "sgt": "sle",
                "sge": "slt",
                "ult": "uge",
                "ule": "ugt",
                "ugt": "ule",
                "uge": "ult",
            }[predicate]
        allowed = _allowed(predicate, self.nodes[right]["value"], 64 if self.type(left) == "index" else self.type(left))
        return {**restrictions, left: _intersect(restrictions[left], allowed) if left in restrictions else allowed}

    def _operation_bounds(self, row, intervals, types):
        op, type_ = row["op"], 1 if row["op"] == "cmpi" else row["type"]
        if op == "cmpi":
            a, b = intervals
            if row["args"][0] == row["args"][1]:
                value = int(row["predicate"] in ("eq", "sle", "sge", "ule", "uge"))
                return [(value, value)]
            if all(len(item) == 1 and item[0][0] == item[0][1] for item in intervals):
                value = _scalar(op, [item[0][0] for item in intervals], types, 1, row["predicate"])
                return [(value, value)]
            if row["predicate"] == "eq" and not _intersect(a, b):
                return [(0, 0)]
            if row["predicate"] == "ne" and not _intersect(a, b):
                return [(1, 1)]
            predicate = row["predicate"]
            if predicate not in ("eq", "ne"):
                if predicate.startswith("s"):
                    a, b = _signed_intervals(a, types[0]), _signed_intervals(b, types[0])
                if a and b:
                    low_a, high_a, low_b, high_b = a[0][0], a[-1][1], b[0][0], b[-1][1]
                    relation = predicate[1:]
                    true = {"lt": high_a < low_b, "le": high_a <= low_b, "gt": low_a > high_b, "ge": low_a >= high_b}[
                        relation
                    ]
                    false = {"lt": low_a >= high_b, "le": low_a > high_b, "gt": high_a <= low_b, "ge": high_a < low_b}[
                        relation
                    ]
                    if true or false:
                        return [(int(true), int(true))]
            return [(0, 1)]
        if op in CASTS:
            source = intervals[0]
            if op in ("extsi", "index_cast") and types[0] != "index":
                source = _signed_intervals(source, types[0])
            if type_ == "index":
                return source if all(0 <= low <= high <= MAX_SIZE for low, high in source) else None
            return _join(piece for low, high in source for piece in _wrap(low, high, type_))
        left, right = intervals
        if op in ("addi", "subi", "muli", "divsi", "floordivsi", "remsi"):
            left, right = _signed_intervals(left, type_), _signed_intervals(right, type_)
        result = []
        for a, b in left:
            for c, d in right:
                if op == "addi":
                    low, high = a + c, b + d
                elif op == "subi":
                    low, high = a - d, b - c
                elif op == "muli":
                    low, high = min(a * c, a * d, b * c, b * d), max(a * c, a * d, b * c, b * d)
                elif op in ("divui", "divsi", "floordivsi", "remui", "remsi"):
                    if c <= 0 <= d or op in ("divsi", "floordivsi") and a <= -(1 << (type_ - 1)) <= b and c <= -1 <= d:
                        return None
                    if op == "remui":
                        low, high = 0, min(b, d - 1)
                    elif op == "remsi":
                        low, high = -max(abs(c), abs(d)) + 1, max(abs(c), abs(d)) - 1
                    else:
                        values = [
                            _scalar(op, [x % (1 << type_), y % (1 << type_)], [type_, type_], type_)
                            for x in (a, b)
                            for y in (c, d)
                        ]
                        if op != "divui":
                            values = [signed(value, type_) for value in values]
                        low, high = min(values), max(values)
                elif op == "andi" and c == d:
                    low, high = 0, c
                else:
                    low, high = 0, (1 << type_) - 1
                result.extend(_wrap(low, high, type_))
        return _join(result)

    @classmethod
    def load(cls, program, axes, roots):
        if (
            not isinstance(program, dict)
            or set(program) != {"schema", "nodes"}
            or program["schema"] != SCHEMA
            or not isinstance(program["nodes"], dict)
            or len(program["nodes"]) > 4096
        ):
            raise ValueError("runtime shapes require an explicit bounded typed expression program")
        result = cls(program["nodes"])
        visited, pending = set(), set()

        def check(name, depth=0):
            result._visit(depth)
            if not isinstance(name, str) or name not in result.nodes or name in pending:
                raise ValueError("shape expression has an unknown or cyclic dependency")
            if name in visited:
                return result.type(name)
            pending.add(name)
            row = result.nodes[name]
            if not isinstance(row, dict) or not isinstance(row.get("op"), str):
                raise ValueError("shape expression needs a complete typed operation")
            op = row["op"]
            expected = (
                {"op", "axis"}
                if op == "axis"
                else {"op", "type", "value"}
                if op == "const"
                else ({"op", "predicate", "args"} if op == "cmpi" else {"op", "type", "args"})
            )
            if set(row) != expected:
                raise ValueError("shape expression has unknown or missing semantic fields")
            type_ = 1 if op == "cmpi" else "index" if op == "axis" else row["type"]
            if type_ != "index" and (type(type_) is not int or not 1 <= type_ <= 4096):
                raise ValueError("shape expression word width exceeds its source domain")
            if op == "axis":
                if not isinstance(row["axis"], str) or row["axis"] not in axes:
                    raise ValueError("shape expression axis lacks its public input binding")
            elif op == "const":
                if type(row["value"]) is not int or not 0 <= row["value"] <= (
                    MAX_SIZE if type_ == "index" else (1 << type_) - 1
                ):
                    raise ValueError("shape expression literal is outside its typed word domain")
            else:
                args = row["args"]
                if not isinstance(args, list) or len(args) != (3 if op == "select" else 1 if op in CASTS else 2):
                    raise ValueError("shape expression operand roster differs from its operation")
                types = [check(argument, depth + 1) for argument in args]
                valid = False
                if op == "cmpi":
                    valid = (
                        row["predicate"] in PREDICATES
                        and types[0] == types[1]
                        and (types[0] != "index" or not row["predicate"].startswith("s"))
                    )
                elif op == "select":
                    valid = types == [1, type_, type_]
                elif op in BINARY:
                    valid = type_ != "index" and types == [type_, type_]
                elif op in ("extui", "extsi", "trunci"):
                    valid = (
                        type_ != "index"
                        and type(types[0]) is int
                        and (types[0] > type_ if op == "trunci" else types[0] < type_)
                    )
                elif op in ("index_castui", "index_cast"):
                    valid = (type_ == "index") != (types[0] == "index") and (op == "index_castui" or type_ == "index")
                if not valid:
                    raise ValueError("shape expression operation/type/signedness is unsupported")
            expected_name = "e_" + hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()[:32]
            if name != expected_name:
                raise ValueError("shape expression identifier differs from its complete definition")
            pending.remove(name)
            visited.add(name)
            return type_

        for root in roots:
            if check(root) != "index":
                raise ValueError("result dimensions require a logical index expression")
        if visited != set(result.nodes):
            raise ValueError("shape expression program has unaccounted nodes")
        result.visits = 0
        return result

    def equivalent(self, left, right, axis_bounds, *, assertions=()):
        """Prove equality over actual source enclosures/assertions, never samples.

        The solver's index encoding holds the entire logical-size domain and
        every fixed source word. It is not the target's native index width.
        Only UNSAT supplies a proof; timeout/unknown remains a refusal.
        """
        try:
            import z3
        except ImportError as error:
            raise ValueError("derived shape proofs require the installed z3-solver extra") from error
        logical_width = (
            max(
                [
                    MAX_SIZE.bit_length() + 1,
                    *[row["type"] for row in self.nodes.values() if type(row.get("type")) is int],
                ]
            )
            + 1
        )
        solver = z3.Solver()
        solver.set(timeout=2000, rlimit=1000000, max_memory=128)
        memo, axes = {}, {}

        def cast(value, width, signed_cast=False):
            if value.size() > width:
                return z3.Extract(width - 1, 0, value)
            if value.size() < width:
                return (z3.SignExt if signed_cast else z3.ZeroExt)(width - value.size(), value)
            return value

        def expression(name, depth=0):
            self._visit(depth)
            if name in memo:
                return memo[name]
            row, type_ = self.nodes[name], self.type(name)
            op = row["op"]
            width = logical_width if type_ == "index" else type_
            valid = z3.BoolVal(True)
            if op == "axis":
                axis = row["axis"]
                if axis not in axes:
                    axes[axis] = z3.BitVec(axis, logical_width)
                    bounds = axis_bounds(axis) or (0, MAX_SIZE)
                    solver.add(z3.UGE(axes[axis], bounds[0]), z3.ULE(axes[axis], bounds[1]))
                value = axes[axis]
            elif op == "const":
                value = z3.BitVecVal(row["value"], width)
            else:
                arguments = [expression(item, depth + 1) for item in row["args"]]
                values = [item[0] for item in arguments]
                valid = z3.And(*[item[1] for item in arguments])
                if op == "select":
                    condition = values[0] != 0
                    value = z3.If(condition, values[1], values[2])
                    valid = z3.And(arguments[0][1], z3.If(condition, arguments[1][1], arguments[2][1]))
                elif op in CASTS:
                    value = cast(values[0], width, op in ("extsi", "index_cast"))
                    if type_ == "index":
                        valid = z3.And(valid, z3.ULE(value, MAX_SIZE))
                elif op == "cmpi":
                    a, b = values
                    predicate = row["predicate"]
                    condition = {
                        "eq": a == b,
                        "ne": a != b,
                        "slt": a < b,
                        "sle": a <= b,
                        "sgt": a > b,
                        "sge": a >= b,
                        "ult": z3.ULT(a, b),
                        "ule": z3.ULE(a, b),
                        "ugt": z3.UGT(a, b),
                        "uge": z3.UGE(a, b),
                    }[predicate]
                    value = z3.If(condition, z3.BitVecVal(1, 1), z3.BitVecVal(0, 1))
                else:
                    a, b = values
                    if op == "addi":
                        value = a + b
                    elif op == "subi":
                        value = a - b
                    elif op == "muli":
                        value = a * b
                    elif op == "andi":
                        value = a & b
                    elif op == "ori":
                        value = a | b
                    elif op == "xori":
                        value = a ^ b
                    else:
                        valid = z3.And(valid, b != 0)
                        if op in ("divsi", "floordivsi"):
                            valid = z3.And(valid, z3.Not(z3.And(a == -(1 << (width - 1)), b == -1)))
                        if op == "divui":
                            value = z3.UDiv(a, b)
                        elif op == "remui":
                            value = z3.URem(a, b)
                        elif op == "remsi":
                            value = z3.SRem(a, b)
                        elif op == "divsi":
                            value = a / b
                        else:
                            quotient, remainder = a / b, z3.SRem(a, b)
                            adjust = z3.And(remainder != 0, (a < 0) != (b < 0))
                            value = quotient - z3.If(adjust, z3.BitVecVal(1, width), z3.BitVecVal(0, width))
            memo[name] = value, valid
            return memo[name]

        self.visits = 0
        for assertion in assertions:
            value, valid = expression(assertion)
            solver.add(valid, value == 1)
        a, a_valid = expression(left)
        b, b_valid = expression(right)
        solver.add(z3.Not(z3.And(a_valid, b_valid, a == b)))
        verdict = solver.check()
        if verdict == z3.unknown:
            raise ValueError("derived shape equivalence was not proved within its bounded solver profile")
        return verdict == z3.unsat
