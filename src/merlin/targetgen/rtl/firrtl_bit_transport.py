"""Exact known-bit transport through selected FIRRTL wires and held registers.

This local observation covers every data-driver alternative and preserves
aggregate indices, signed extension and actual sink truncation. It does not
prove validity, initialization, register history, clocking or timing. A direct
self-hold contributes no new data; other feedback and unknown values refuse.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

from . import firrtl_struct as F


@dataclass(frozen=True)
class BitValue:
    signed: bool
    bits: tuple  # LSB first; integer literals or (original_source, bit_index).


def _resize(value, width, signed):
    if type(width) is not int or not 0 < width <= 256:
        raise ValueError("selected bit transport width is unresolved or oversized")
    extension = value.bits[-1] if value.signed else 0
    return BitValue(signed, (value.bits + (extension,) * max(0, width - len(value.bits)))[:width])


def _type(module, reference, modules):
    declaration = module.decls.get(reference.root)
    if declaration is None or reference.dynamic:
        raise ValueError("selected bit transport reference lacks a static declaration")
    kind, text, _line = declaration
    if kind == "inst":
        callee = modules.get(text)
        if callee is None or not reference.segments:
            raise ValueError("selected instance port needs its original module type")
        result = F.field_type(callee.port_type(reference.segments[0]), reference.segments[1:])
    elif kind in {"input", "output", "wire", "reg"}:
        result = F.field_type(F.parse_type(text), reference.segments)
    else:
        raise ValueError("selected bit transport reaches unsupported state or memory")
    if result is None or result[0] not in {"sint", "uint"} or type(result[1]) is not int or not 0 < result[1] <= 256:
        raise ValueError("selected bit transport requires exact scalar integer leaves")
    return result


def observe_bit_transport(module, destination, *, sources, modules=None, max_nodes=2048):
    """Return all exact data alternatives at one original selected scalar sink.

    `sources` selects original interface/instance cuts; their types come from
    declarations. Names do not assign a command, arithmetic or resource role.
    Caller still owns every temporal/definedness premise and file custody.
    """
    modules = {} if modules is None else dict(modules)
    sources = {F.parse_expr(value) if type(value) is str else value for value in sources}
    destination = F.parse_expr(destination) if type(destination) is str else destination
    if (
        type(max_nodes) is not int
        or not 1 <= max_nodes <= 65536
        or not sources
        or len(sources) > 256
        or any(type(value) is not F.Ref for value in sources)
    ):
        raise ValueError("selected bit transport needs bounded explicit original source cuts")
    visited, cache = set(), {}

    def observe(expression, active=()):
        if len(active) >= 128:
            raise ValueError("selected bit transport exceeds its depth bound")
        if expression in cache:
            return cache[expression]
        if expression in active:
            # Only a literal direct self-hold is omitted below. A multi-node
            # loop cannot supply a new original value without temporal proof.
            raise ValueError("selected bit transport has a nontrivial feedback cycle")
        visited.add(expression)
        if len(visited) > max_nodes:
            raise ValueError("selected bit transport exceeds its node bound")
        nested = (*active, expression)
        if type(expression) is F.Lit:
            if type(expression.width) is not int or not 0 < expression.width <= 256:
                raise ValueError("selected literal width is unresolved or oversized")
            values = {
                BitValue(
                    expression.kind == "SInt", tuple((expression.value >> bit) & 1 for bit in range(expression.width))
                )
            }
        elif type(expression) is F.Ref:
            if expression in sources:
                kind, width = _type(module, expression, modules)
                values = {BitValue(kind == "sint", tuple((expression.text(), bit) for bit in range(width)))}
            elif not expression.segments and expression.root in module.nodes:
                values = observe(module.nodes[expression.root][0], nested)
            else:
                drivers = []
                text = expression.text()
                for row in module.connects:
                    prefix = row.lhs.text()
                    if text == prefix:
                        rhs = row.rhs
                    elif text.startswith((prefix + "[", prefix + ".")):
                        if type(row.rhs) is not F.Ref:
                            raise ValueError("selected aggregate driver is not an exact alias")
                        rhs = F.parse_expr(row.rhs.text() + text[len(prefix) :])
                    else:
                        continue
                    if rhs == expression and module.decls[expression.root][0] == "reg":
                        continue
                    drivers.append(rhs)
                if not drivers:
                    raise ValueError("selected transport has no original data-producing driver")
                kind, width = _type(module, expression, modules)
                values = {_resize(value, width, kind == "sint") for rhs in drivers for value in observe(rhs, nested)}
        elif type(expression) is F.Prim:
            op, args = expression.op, expression.args
            if op in {"asSInt", "asUInt"} and len(args) == 1:
                values = {BitValue(op == "asSInt", value.bits) for value in observe(args[0], nested)}
            elif op == "cat" and len(args) == 2:
                values = {
                    BitValue(False, low.bits + high.bits)
                    for high, low in product(observe(args[0], nested), observe(args[1], nested))
                }
            elif op == "bits" and len(args) == 3 and all(type(value) is F.IntArg for value in args[1:]):
                high, low = (value.value for value in args[1:])
                operands = observe(args[0], nested)
                if any(not 0 <= low <= high < len(value.bits) for value in operands):
                    raise ValueError("selected bit extraction exceeds its original source")
                values = {BitValue(False, value.bits[low : high + 1]) for value in operands}
            elif op == "pad" and len(args) == 2 and type(args[1]) is F.IntArg:
                values = {
                    _resize(value, max(len(value.bits), args[1].value), value.signed)
                    for value in observe(args[0], nested)
                }
            else:
                raise ValueError("selected transport contains arithmetic or an unsupported producer")
        else:
            raise ValueError("selected transport reaches an unknown value producer")
        if len(values) > 64 or any(not 0 < len(value.bits) <= 256 for value in values):
            raise ValueError("selected transport alternatives or bit widths exceed bounds")
        cache[expression] = frozenset(values)
        return cache[expression]

    return observe(destination)
