"""Check counted-schedule integer intermediates without target loading or emission.

The same conservative interval check serves shared verification and C codegen.
It never expands dynamic iterations and grants no target address/window or
completion semantics. Unsupported expressions remain a barrier.
"""

from __future__ import annotations

from .expr import Const, Expr, Scaled, Select, Sum, Var
from .kernel import Call, Loop, Ptr


class IntegerDomainError(ValueError):
    """A selected signed index domain cannot represent every intermediate."""


def check_integer_domain(kernel, bits):
    """Conservative bounds for every emitted integer intermediate, without expansion."""
    if type(bits) is not int or bits not in (32, 64):
        raise IntegerDomainError("explicit signed 32-bit or 64-bit index domain required")
    limit = (1 << (bits - 1)) - 1

    def bounds(expression, domains):
        if isinstance(expression, Const):
            result = expression.value, expression.value
        elif isinstance(expression, Var):
            if expression.name not in domains:
                raise IntegerDomainError(f"unbound integer index variable: {expression.name}")
            result = domains[expression.name]
        elif isinstance(expression, Sum):
            lo = hi = 0
            for term in expression.terms:
                term_lo, term_hi = bounds(term, domains)
                lo, hi = lo + term_lo, hi + term_hi
                if lo < -limit - 1 or hi > limit:
                    raise IntegerDomainError(
                        f"integer sum intermediate may overflow the selected signed {bits}-bit C index domain"
                    )
            result = lo, hi
        elif isinstance(expression, Scaled):
            lo, hi = bounds(expression.expr, domains)
            result = tuple(sorted((lo * expression.factor, hi * expression.factor)))
        elif isinstance(expression, Select):
            if expression.var not in domains:
                raise IntegerDomainError(f"unbound integer index variable: {expression.var}")
            lo, hi = domains[expression.var]
            parts = []
            if lo <= expression.equals <= hi:
                parts.append(
                    bounds(expression.then, {**domains, expression.var: (expression.equals, expression.equals)})
                )
            if lo != hi or lo != expression.equals:
                parts.append(bounds(expression.other, domains))
            result = min(a for a, _b in parts), max(b for _a, b in parts)
        else:
            raise IntegerDomainError("unknown integer expression in selected C index domain")
        if result[0] < -limit - 1 or result[1] > limit:
            raise IntegerDomainError(f"integer expression may overflow the selected signed {bits}-bit C index domain")
        return result

    def walk(body, domains):
        for statement in body:
            if isinstance(statement, Loop):
                if not 0 < statement.extent <= limit:
                    raise IntegerDomainError(f"loop extent exceeds the selected signed {bits}-bit C index domain")
                walk(statement.body, {**domains, statement.var: (0, statement.extent - 1)})
            elif isinstance(statement, Call):
                for _name, value in statement.args:
                    if isinstance(value, Expr):
                        bounds(value, domains)
                    elif isinstance(value, Ptr):
                        bounds(value.offset, domains)
            else:
                raise IntegerDomainError("unknown statement in selected integer index domain")

    walk(kernel.body, {})
