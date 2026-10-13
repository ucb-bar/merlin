"""Bounded validity implications for exact selected FIRRTL value paths.

Uninterpreted predicates remain independent Boolean atoms. This can prove a
masking relation, but never assumes a guard is true or a selector is in range.
"""

from __future__ import annotations

import itertools
import weakref

from merlin.targetgen.rtl import firrtl_struct as F

UNDEFINED = "__merlin_result_format_undefined"
_TERMS = weakref.WeakValueDictionary()
_SERIAL = itertools.count()


class _Term:
    def __init__(self, op, args):
        self.op, self.args, self.serial = op, args, next(_SERIAL)

    def __repr__(self):
        return "BooleanTerm(" + str(self.serial) + ")"


def _term(op, *args):
    key = (op, args)
    if key not in _TERMS:
        _TERMS[key] = value = _Term(op, args)
        return value
    return _TERMS[key]


def negate(value):
    if isinstance(value, bool):
        return not value
    return value.args[0] if value.op == "not" else _term("not", value)


def conjunction(*values):
    terms = set()
    for value in values:
        if value is False:
            return False
        if value is True:
            continue
        terms.update(value.args if value.op == "and" else (value,))
    if any(negate(term) in terms for term in terms):
        return False
    if not terms:
        return True
    ordered = tuple(sorted(terms, key=lambda term: term.serial))
    return ordered[0] if len(ordered) == 1 else _term("and", *ordered)


def disjunction(*values):
    return negate(conjunction(*(negate(value) for value in values)))


def _restrict(value, atom, truth, cache):
    if isinstance(value, bool):
        return value
    if value in cache:
        return cache[value]
    if value.op == "atom":
        return truth if value == atom else value
    result = (
        negate(_restrict(value.args[0], atom, truth, cache))
        if value.op == "not"
        else conjunction(*(_restrict(term, atom, truth, cache) for term in value.args))
    )
    cache[value] = result
    return result


def possible(value):
    """Bounded propositional satisfiability; exhaustion refuses the proof."""
    work, seen = [value], set()
    while work:
        current = work.pop()
        if current is True:
            return True
        if current is False or current in seen:
            continue
        seen.add(current)
        if len(seen) > 4096:
            raise ValueError("output validity implication exceeds its bounded graph")
        todo, walked, atom = [current], set(), None
        while todo:
            term = todo.pop()
            if isinstance(term, bool) or term in walked:
                continue
            walked.add(term)
            if term.op == "atom":
                atom = term
                break
            todo.extend(term.args)
        if atom is None:
            raise ValueError("output validity contains an unexplained formula")
        work.extend((_restrict(current, atom, False, {}), _restrict(current, atom, True, {})))
    return False


def parse_module(name, lines):
    """Retain invalidation as an explicit undefined value, with its guards."""
    rewritten = []
    for number, raw in lines:
        if UNDEFINED in raw:
            raise ValueError("selected FIRRTL uses the reserved undefined marker")
        active, _locator = F.split_locator(raw)
        statement = active.strip()
        if statement.startswith("invalidate "):
            prefix = raw[: len(raw) - len(raw.lstrip())]
            raw = prefix + "connect " + statement[len("invalidate ") :] + ", " + UNDEFINED
        rewritten.append((number, raw))
    return F.parse_module(name, rewritten)


class BooleanPaths:
    def __init__(self, modules, project):
        self.modules, self.project = modules, project
        self.cache, self.active = {}, set()

    def guards(self, module, guards, trail):
        return conjunction(
            *(
                self.read(module, guard.cond, trail) if guard.positive else negate(self.read(module, guard.cond, trail))
                for guard in guards
            )
        )

    def _width(self, module_name, expression):
        module = self.modules[module_name]
        if isinstance(expression, F.Lit):
            return expression.width
        if isinstance(expression, F.Prim):
            if expression.op == "bits" and len(expression.args) == 3:
                high, low = expression.args[1:]
                if isinstance(high, F.IntArg) and isinstance(low, F.IntArg) and high.value >= low.value:
                    return high.value - low.value + 1
            if expression.op in {"eq", "neq", "lt", "leq", "gt", "geq", "andr", "orr", "xorr"}:
                return 1
            if expression.op in {"and", "or", "xor", "not"}:
                widths = [self._width(module_name, item) for item in expression.args]
                return max(widths) if all(width is not None for width in widths) else None
        if isinstance(expression, F.Ref):
            if expression.root in module.nodes and not expression.segments:
                return self._width(module_name, module.nodes[expression.root][0])
            declaration = module.decls.get(expression.root)
            if declaration is not None:
                if declaration[0] == "inst" and expression.segments:
                    child = self.modules.get(declaration[1])
                    if child is not None:
                        return self._width(child.name, F.Ref(str(expression.segments[0]), expression.segments[1:]))
                elif declaration[0] in {"wire", "reg", "regreset", "input", "output"}:
                    typ = F.field_type(F.parse_type(declaration[1]), expression.segments)
                    return typ[1] if typ is not None and typ[0] in {"uint", "sint"} else None
        return None

    def read(self, module_name, expression, trail=()):
        key = (module_name, expression, trail)
        atom = _term("atom", key)
        if key in self.cache:
            return self.cache[key]
        if key in self.active or len(self.cache) > 10000:
            return atom  # unconstrained state is not assumed to be true/false
        self.active.add(key)
        try:
            result = self._read(module_name, expression, trail, atom)
            self.cache[key] = result
            return result
        finally:
            self.active.remove(key)

    def _read(self, module_name, expression, trail, atom):
        module = self.modules[module_name]
        if isinstance(expression, F.Lit):
            return bool(expression.value) if expression.width == 1 else atom
        if isinstance(expression, F.Prim):
            arguments = expression.args
            if expression.op == "not" and self._width(module_name, expression) == 1:
                return negate(self.read(module_name, arguments[0], trail))
            if expression.op in {"and", "or"} and self._width(module_name, expression) == 1:
                terms = [self.read(module_name, value, trail) for value in arguments]
                return conjunction(*terms) if expression.op == "and" else disjunction(*terms)
            if expression.op in {"eq", "neq"} and len(arguments) == 2:
                for left, right in (arguments, arguments[::-1]):
                    if isinstance(right, F.Lit) and right.width == 1 and self._width(module_name, left) == 1:
                        value = self.read(module_name, left, trail)
                        return value if bool(right.value) == (expression.op == "eq") else negate(value)
            if expression.op == "mux":
                selector, yes, no = (self.read(module_name, value, trail) for value in arguments)
                return disjunction(conjunction(selector, yes), conjunction(negate(selector), no))
        if isinstance(expression, F.Ref) and not expression.dynamic:
            if expression.root in module.nodes and not expression.segments:
                return self.read(module_name, module.nodes[expression.root][0], trail)
            drivers = [
                (connect, self.project(expression, connect.lhs, connect.rhs))
                for connect in module.connects_to(expression.root)
            ]
            drivers = [(connect, value) for connect, value in drivers if value is not None]
            if drivers:
                current = atom
                for connect, value in drivers:
                    guard = self.guards(module_name, connect.guards, trail)
                    current = disjunction(
                        conjunction(guard, self.read(module_name, value, trail)), conjunction(negate(guard), current)
                    )
                return current
            declaration = module.decls.get(expression.root)
            if declaration is not None and declaration[0] in {"input", "output"} and trail:
                parent, instance = trail[-1]
                return self.read(parent, F.Ref(instance, (expression.root, *expression.segments)), trail[:-1])
        return atom
