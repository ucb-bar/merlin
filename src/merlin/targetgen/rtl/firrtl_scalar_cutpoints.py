"""Bounded unsigned combinational observations at explicit FIRRTL source cuts.

Static aggregate projection, priority connects and UInt truncation are retained.
Unbound inputs/registers/memories never become zero or next-state values. This
local two-state observation proves no reachability, clocking, protocol or timing.
"""

from __future__ import annotations

from copy import deepcopy

from . import firrtl_struct as F


def _field(typ, selectors):
    for selector in selectors:
        if not isinstance(typ, tuple) or not typ:
            raise ValueError("scalar cone type is unavailable")
        if isinstance(selector, int):
            if typ[0] != "vec" or not 0 <= selector < typ[2]:
                raise ValueError("scalar cone index is outside its original vector")
            typ = typ[1]
        else:
            if typ[0] != "bundle":
                raise ValueError("scalar cone field lacks an original aggregate")
            rows = [row[2] for row in typ[1] if row[0] == selector]
            if len(rows) != 1:
                raise ValueError("scalar cone field is absent or ambiguous")
            typ = rows[0]
    if (
        not isinstance(typ, tuple)
        or len(typ) != 2
        or typ[0] != "uint"
        or type(typ[1]) is not int
        or not 1 <= typ[1] <= 256
    ):
        raise ValueError("scalar cone requires bounded unsigned scalar leaves")
    return typ[1]


class ScalarCutpoints:
    """Evaluate existing parsed source; original caller owns file/source custody."""

    def __init__(self, module, *, max_nodes=8192, max_depth=128, modules=None):
        if type(max_nodes) is not int or not 1 <= max_nodes <= 65536:
            raise ValueError("scalar cone needs a finite node budget")
        if type(max_depth) is not int or not 1 <= max_depth <= 256:
            raise ValueError("scalar cone needs an explicit finite depth budget")
        self.module, self.max_nodes, self.max_depth = deepcopy(module), max_nodes, max_depth
        if modules is not None and (not isinstance(modules, dict) or len(modules) > 256):
            raise ValueError("scalar cone original module-type roster exceeds its finite budget")
        self.modules = deepcopy(modules or {})
        self.drivers = {}
        for row in self.module.connects:
            self.drivers.setdefault(row.lhs, []).append(row)

    def evaluate(self, cuts, outputs):
        if not isinstance(cuts, dict) or not 1 <= len(cuts) <= 256 or not isinstance(outputs, (list, tuple)):
            raise ValueError("scalar cone needs explicit bounded cuts and outputs")
        if not 1 <= len(outputs) <= 256:
            raise ValueError("scalar cone output roster exceeds its finite budget")
        self.cuts = {F.parse_expr(key) if isinstance(key, str) else key: value for key, value in cuts.items()}
        if len(self.cuts) != len(cuts) or any(type(key) is not F.Ref for key in self.cuts):
            raise ValueError("scalar cone cuts require distinct original references")
        self.widths, self.values, self.active_widths, self.active_values = {}, {}, set(), set()
        for key, value in self.cuts.items():
            if type(value) is not int or not 0 <= value < 1 << self._width(key):
                raise ValueError("scalar cone cut does not fit its original unsigned width")
        return {name: self._value(F.parse_expr(name)) for name in outputs}

    def _static(self, ref):
        if not ref.dynamic:
            return ref
        indices = iter(ref.dynamic)
        selectors = tuple(self._value(next(indices)) if part is None else part for part in ref.segments)
        return F.Ref(ref.root, selectors)

    def _guard(self, row):
        for guard in row.guards:
            value = self._value(guard.cond)
            if value not in (0, 1):
                raise ValueError("scalar cone driver guard is not Boolean")
            if bool(value) != guard.positive:
                return False
        return True

    def _width(self, expression, selectors=()):
        key = expression, selectors
        if key in self.widths:
            return self.widths[key]
        if key in self.active_widths or len(self.active_widths) > self.max_depth or len(self.widths) >= self.max_nodes:
            raise ValueError("scalar cone width is cyclic or oversized")
        self.active_widths.add(key)
        try:
            value = self._width_impl(expression, selectors)
            if type(value) is not int or not 1 <= value <= 256:
                raise ValueError("scalar cone result width exceeds its finite unsigned scope")
            self.widths[key] = value
            return value
        finally:
            self.active_widths.remove(key)

    def _width_impl(self, expression, selectors):
        if type(expression) is F.Ref:
            expression = self._static(expression)
            path = (*expression.segments, *selectors)
            if expression.root in self.module.nodes:
                return self._width(self.module.nodes[expression.root][0], path)
            declaration = self.module.decls.get(expression.root)
            if declaration is not None and declaration[0] == "inst":
                child = self.modules.get(declaration[1])
                if child is None or not path or not isinstance(path[0], str):
                    raise ValueError("scalar cone instance cut needs its original child port type")
                return _field(child.port_type(path[0]), path[1:])
            if declaration is None or declaration[0] not in {"reg", "regreset", "wire", "input", "output"}:
                raise ValueError("scalar cone has no original scalar declaration")
            return _field(F.parse_type(declaration[1]), path)
        if type(expression) is F.Lit:
            if selectors or expression.kind != "UInt" or expression.value < 0:
                raise ValueError("scalar cone literal is outside unsigned scope")
            width = expression.width or max(1, expression.value.bit_length())
            if expression.value >= 1 << width:
                raise ValueError("scalar cone literal exceeds its original declared width")
            return width
        if type(expression) is not F.Prim:
            raise ValueError("scalar cone width primitive is unavailable")
        op, args = expression.op, expression.args
        arities = {
            "mux": 3,
            "bits": 3,
            "head": 2,
            "tail": 2,
            "pad": 2,
            "shl": 2,
            "shr": 2,
            "eq": 2,
            "neq": 2,
            "lt": 2,
            "leq": 2,
            "gt": 2,
            "geq": 2,
            "add": 2,
            "sub": 2,
            "and": 2,
            "or": 2,
            "xor": 2,
            "asUInt": 1,
            "not": 1,
            "cat": 2,
            "mul": 2,
            "div": 2,
            "rem": 2,
        }
        if op not in arities or len(args) != arities[op]:
            raise ValueError("unsupported or malformed unsigned scalar cone primitive: " + op)
        if op == "mux":
            if len(args) != 3:
                raise ValueError("malformed scalar cone mux")
            return max(self._width(arg, selectors) for arg in args[1:])
        if selectors:
            raise ValueError("scalar cone projection reaches a scalar primitive")
        if op in {"bits", "head", "tail", "pad", "shl", "shr"}:
            immediate = args[1:]
            if any(type(arg) is not F.IntArg or arg.value < 0 for arg in immediate):
                raise ValueError("scalar cone shift/extract requires finite literal arguments")
            base = self._width(args[0])
            count = immediate[0].value
            if op == "bits":
                if len(immediate) != 2 or not 0 <= immediate[1].value <= count < base:
                    raise ValueError("scalar cone extract is outside original width")
                return count - immediate[1].value + 1
            if op == "head":
                if not 0 < count <= base:
                    raise ValueError("scalar cone head exceeds original width")
                return count
            return {"tail": base - count, "pad": max(base, count), "shl": base + count, "shr": max(1, base - count)}[op]
        widths = [self._width(arg) for arg in args]
        if op in {"eq", "neq", "lt", "leq", "gt", "geq"} and len(args) == 2:
            return 1
        if op in {"add", "sub"} and len(args) == 2:
            return max(widths) + 1
        if op in {"and", "or", "xor"} and len(args) == 2:
            return max(widths)
        if op in {"asUInt", "not"} and len(args) == 1:
            return widths[0]
        if op in {"cat", "mul"} and len(args) == 2:
            return sum(widths)
        if op == "div" and len(args) == 2:
            return widths[0]
        if op == "rem" and len(args) == 2:
            return min(widths)
        raise ValueError("unsupported unsigned scalar cone primitive: " + op)

    def _value(self, expression, selectors=()):
        key = expression, selectors
        if key in self.values:
            return self.values[key]
        if key in self.active_values or len(self.active_values) > self.max_depth or len(self.values) >= self.max_nodes:
            raise ValueError("scalar cone value is cyclic or exceeds its node budget")
        self.active_values.add(key)
        try:
            width = self._width(expression, selectors)
            value = self._value_impl(expression, selectors) & ((1 << width) - 1)
            self.values[key] = value
            return value
        finally:
            self.active_values.remove(key)

    def _value_impl(self, expression, selectors):
        if type(expression) is F.Lit:
            return expression.value
        if type(expression) is F.Ref:
            expression = self._static(expression)
            ref = F.Ref(expression.root, (*expression.segments, *selectors))
            if ref in self.cuts:
                return self.cuts[ref]
            if expression.root in self.module.nodes:
                return self._value(self.module.nodes[expression.root][0], ref.segments)
            declaration = self.module.decls.get(ref.root)
            if declaration is None or declaration[0] not in {"wire", "output"}:
                raise ValueError("scalar cone reached unbound input, state or memory")
            for size in range(len(ref.segments), -1, -1):
                rows = self.drivers.get(F.Ref(ref.root, ref.segments[:size]), [])
                if rows:
                    active = [row for row in rows if self._guard(row)]
                    if not active:
                        raise ValueError("scalar cone wire has no active driver")
                    return self._value(active[-1].rhs, ref.segments[size:])
            raise ValueError("scalar cone wire has no original driver")
        if type(expression) is not F.Prim:
            raise ValueError("scalar cone value primitive is unavailable")
        op, args = expression.op, expression.args
        if op == "mux":
            condition = self._value(args[0])
            if condition not in (0, 1):
                raise ValueError("scalar cone mux condition is not Boolean")
            return self._value(args[1 if condition else 2], selectors)
        if selectors:
            raise ValueError("scalar cone projection reaches scalar arithmetic")
        values = [arg.value if type(arg) is F.IntArg else self._value(arg) for arg in args]
        if op in {"bits", "head", "tail", "pad", "shl", "shr"}:
            if op == "bits":
                return values[0] >> values[2]
            if op == "head":
                return values[0] >> (self._width(args[0]) - values[1])
            if op in {"tail", "pad"}:
                return values[0]
            return values[0] << values[1] if op == "shl" else values[0] >> values[1]
        if op == "cat":
            return (values[0] << self._width(args[1])) | values[1]
        if op in {"div", "rem"} and values[1] == 0:
            raise ValueError("scalar cone division by zero is unresolved")
        if op in {"asUInt", "not"}:
            return values[0] if op == "asUInt" else ~values[0]
        left, right = values
        operations = {
            "add": lambda: left + right,
            "sub": lambda: left - right,
            "mul": lambda: left * right,
            "div": lambda: left // right,
            "rem": lambda: left % right,
            "and": lambda: left & right,
            "or": lambda: left | right,
            "xor": lambda: left ^ right,
            "eq": lambda: int(left == right),
            "neq": lambda: int(left != right),
            "lt": lambda: int(left < right),
            "leq": lambda: int(left <= right),
            "gt": lambda: int(left > right),
            "geq": lambda: int(left >= right),
        }
        if op not in operations:
            raise ValueError("unsupported scalar cone value primitive")
        return operations[op]()
