"""Bound a selected original byte-count cone by its decoded positive rows and columns.

The target selects the meaning of the byte-count sink. Constants and widths come from FIRRTL.
This proves an upper bound, including every mux alternative, and grants no command acceptance,
arithmetic-format or scheduling evidence.
"""

from collections import Counter
from itertools import product

from . import firrtl_struct as F


def derive(module, layouts, selection, isa, classes, invalidated):
    if type(selection) is not dict or set(selection) != {"byte_count", "rows", "cols"}:
        raise ValueError("transfer bound needs an original byte-count sink and decoded extents")
    variables = []
    for axis in ("rows", "cols"):
        field = selection[axis]
        if type(field) is not dict or set(field) != {"bundle", "name"}:
            raise ValueError("transfer bound lacks a selected source extent")
        matches = [row for row in layouts["fields"]
                   if (row["bundle"], row["name"]) == (field["bundle"], field["name"])]
        if len(matches) != 1 or matches[0]["module"] != module.name or matches[0]["status"] != "derived":
            raise ValueError("transfer extent differs from the selected controller source")
        source = matches[0]
        reference = F.parse_expr(source["field"])
        declaration = module.decls.get(reference.root) if isinstance(reference, F.Ref) else None
        typ = F.field_type(F.parse_type(declaration[1]), reference.segments) if declaration else None
        if typ != ("uint", source["width"]):
            raise ValueError("transfer extent lacks its original unsigned field type")
        for klass in classes:
            decoded = [step for step in isa["OPERAND_FIELDS"][klass]["fields"] if step["name"] == axis]
            if (len(decoded) != 1 or decoded[0].get("operand") != "rs2"
                    or (decoded[0].get("offset"), decoded[0].get("width")) != (source["offset"], source["width"])):
                raise ValueError("transfer bound extent differs from the original operand decoder")
        variables.append(reference)
    sink = F.parse_expr(selection["byte_count"])
    if any(not isinstance(ref, F.Ref) or ref.dynamic for ref in [sink, *variables]):
        raise ValueError("transfer bound needs exact static source references")

    def combine(left, right, multiply):
        if len(left) * len(right) > 32:
            raise ValueError("transfer bound has too many source alternatives")
        results = []
        for a, b in product(left, right):
            value = Counter()
            if multiply:
                for (ar, ac), av in a.items():
                    for (br, bc), bv in b.items():
                        value[ar + br, ac + bc] += av * bv
            else:
                value.update(a)
                value.update(b)
            if len(value) > 16 or any(coefficient > 1 << 20 for coefficient in value.values()):
                raise ValueError("transfer bound polynomial exceeds finite source limits")
            results.append(value)
        return results

    def upper(expr, active=()):
        if len(active) >= 128 or expr in active:
            raise ValueError("transfer byte-count source is cyclic or oversized")
        if isinstance(expr, F.Ref) and any(
            expr.root == ref.root and expr.segments[:len(ref.segments)] == ref.segments for ref in invalidated
        ):
            raise ValueError("transfer bound uses invalidated source data")
        if expr == variables[0]:
            return [Counter({(1, 0): 1})]
        if expr == variables[1]:
            return [Counter({(0, 1): 1})]
        if isinstance(expr, F.Lit) and expr.kind == "UInt":
            return [Counter({(0, 0): expr.value})]
        active = (*active, expr)
        if isinstance(expr, F.Ref):
            if expr.root in module.nodes and not expr.segments:
                value = module.nodes[expr.root][0]
            else:
                drivers = [row for row in module.connects if row.lhs == expr]
                if len(drivers) != 1 or drivers[0].guards:
                    raise ValueError("transfer byte count has ambiguous or conditional source drivers")
                value = drivers[0].rhs
            return upper(value, active)
        if isinstance(expr, F.Prim):
            if expr.op == "mux" and len(expr.args) == 3:
                alternatives = upper(expr.args[1], active) + upper(expr.args[2], active)
                if len(alternatives) > 32:
                    raise ValueError("transfer bound has too many mux alternatives")
                return alternatives
            if expr.op in {"mul", "add"} and len(expr.args) == 2:
                return combine(upper(expr.args[0], active), upper(expr.args[1], active), expr.op == "mul")
            if expr.op in {"pad", "tail", "bits"}:
                # Dropping high/low bits of a nonnegative unsigned quantity never increases it.
                # Keeping the wider original value is a conservative upper bound.
                return upper(expr.args[0], active)
            if (expr.op in {"shr", "shl"} and len(expr.args) == 2
                    and isinstance(expr.args[1], F.IntArg) and 0 <= expr.args[1].value <= 16):
                scale = 1 << expr.args[1].value
                return [Counter({key: (coefficient * scale if expr.op == "shl"
                                       else (coefficient + scale - 1) // scale)
                                 for key, coefficient in row.items()}) for row in upper(expr.args[0], active)]
        raise ValueError("transfer byte count has an unproved source operation")

    alternatives = upper(sink)
    if any(any(r > 1 or c > 1 for r, c in row) for row in alternatives):
        raise ValueError("transfer byte count is not bounded linearly by decoded rows and columns")
    # Rows and columns are independently required positive by the consuming copy proof. Every
    # constant, row-only and column-only term is therefore bounded by that same rows*columns product.
    maximum = max(sum(row.values()) for row in alternatives)
    if not 1 <= maximum <= 256:
        raise ValueError("transfer element byte bound is absent or outside the finite proof range")
    return maximum
