"""Bounded source-local emission dependencies and observable scalar effects.

External macro data is an explicit conditional premise, never a compiler or
runtime environment observation. All original definitions and inactive regions
are checked. Opaque bodies and effects outside this exact domain refuse.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects.builtin import (
    ArrayAttr,
    Builtin,
    IntegerAttr,
    IntegerType,
    StringAttr,
    SymbolRefAttr,
    UnregisteredAttr,
)
from xdsl.parser import Parser
from xdsl.utils.exceptions import ParseError

from .hw_observations import _attribute, _name


@dataclass(frozen=True)
class MacroValue:
    symbol: str
    defined: bool
    value: int | None


@dataclass(frozen=True)
class OriginalSourceEffect:
    ordinal: int
    operation: str
    clock_input: int
    compile_enabled: bool
    predicate_endpoint: tuple[str, int] | None
    branch_predicate_endpoints: tuple[tuple[tuple[str, int], bool], ...]
    format_string: str | None


@dataclass(frozen=True)
class SourceEffectPhase:
    phase: int
    effect_ordinal: int
    operation: str
    status: str
    rising_edge: bool
    predicate: int | None
    branch_predicates: tuple[int, ...]
    file_descriptor: int | None


@dataclass(frozen=True)
class SourceEmissionObservation:
    environment_sha256: str
    original_macro_premises: tuple[MacroValue, ...]
    resolved_macros: tuple[MacroValue, ...]
    original_fragment_symbols: tuple[str, ...]
    original_definition_ordinals: tuple[int, ...]
    fragment_operations: tuple[tuple[int, str], ...]
    original_effects: tuple[OriginalSourceEffect, ...]
    phases: tuple[SourceEffectPhase, ...]
    compiler_runtime_environment_correspondence: str = "UNKNOWN"
    source_effect_scheduling_correspondence: str = "UNKNOWN"


@dataclass(frozen=True)
class _Effect:
    operation: object
    clock: object
    enabled: bool
    branches: tuple[tuple[object, bool], ...]
    predicate: object | None
    file_descriptor: object | None
    format_string: str | None


@dataclass(frozen=True)
class _Emission:
    environment_sha256: str
    original_macros: tuple[MacroValue, ...]
    macros: tuple[MacroValue, ...]
    fragments: tuple[str, ...]
    definitions: tuple[int, ...]
    fragment_operations: tuple[tuple[int, str], ...]
    scalar_operations: tuple[object, ...]
    macro_constants: tuple[tuple[object, int], ...]
    effects: tuple[_Effect, ...]
    operation_ordinals: dict


def _fields(op, allowed):
    if set(op.attributes) & set(op.properties) or (set(op.attributes) | set(op.properties)) - {*allowed, "op_name__"}:
        raise ValueError("Source emission operation fields are unsupported.")


def _identifier(value):
    return (
        type(value) is str
        and bool(value)
        and (value[0].isascii() and (value[0].isalpha() or value[0] == "_"))
        and all(char.isascii() and (char.isalnum() or char in "_$") for char in value[1:])
    )


def _flat_reference(value):
    if (
        not isinstance(value, SymbolRefAttr)
        or value.nested_references.data
        or not _identifier(value.root_reference.data)
    ):
        raise ValueError("Source emission symbol reference is unsupported.")
    return value.root_reference.data


def _ifdef_symbol(op):
    value = _attribute(op, "cond")
    if not isinstance(value, UnregisteredAttr) or value.attr_name.data != "sv.macro.ident":
        raise ValueError("Source emission macro condition is unsupported.")
    reference = value.value.data.strip()
    context = Context(allow_unregistered=True)
    context.load_dialect(Builtin)
    try:
        # Parse a real symbol attribute, then require its canonical complete
        # spelling. No opaque macro text is treated as a typed expression.
        attribute = Parser(context, reference).parse_attribute()
    except ParseError:
        raise ValueError("Source emission macro condition is unsupported.") from None
    if str(attribute) != reference:
        raise ValueError("Source emission macro condition is unsupported.")
    return _flat_reference(attribute)


def _block(op, index, *, regions):
    if len(op.regions) == regions and regions == 2 and index == 1 and not op.regions[index].blocks:
        return None
    if len(op.regions) != regions or len(op.regions[index].blocks) != 1 or op.regions[index].block.args:
        raise ValueError("Source emission region membership is unsupported.")
    return op.regions[index].block


def _macro_body(value, macros):
    if not isinstance(value, StringAttr):
        raise ValueError("Source emission macro body is unresolved.")
    body = value.data.strip(" \t\r\n")
    if body in {"0", "1"}:
        return int(body)
    if body.startswith("(") and body.endswith(")"):
        body = body[1:-1].strip(" \t\r\n")
    if body.startswith("`") and _identifier(body[1:]) and body[1:] in macros:
        dependency = macros[body[1:]]
        return dependency.value if dependency.defined else None
    raise ValueError("Source emission macro body is unresolved.")


def prepare_source_emission(parsed, module, fragments, environment, bound, source_sha256):
    """Resolve the complete original container under explicit external premises."""
    if type(environment) is not str or len(environment.encode("utf-8")) > bound.source_bytes:
        raise ValueError("Source emission macro premise exceeds its byte budget.")

    def unique_mapping(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Source emission macro premise fields are duplicated.")
            result[key] = value
        return result

    try:
        document = json.loads(environment, object_pairs_hook=unique_mapping)
    except (ValueError, RecursionError):
        raise ValueError("Source emission macro premise is unavailable.") from None
    if (
        type(document) is not dict
        or set(document) != {"schema", "source_sha256", "macros"}
        or document["schema"] != "merlin.source_macro_environment.v1"
        or document["source_sha256"] != source_sha256
        or type(document["macros"]) is not list
        or len(document["macros"]) > bound.nodes
    ):
        raise ValueError("Source emission macro premise differs from the selected source.")
    operations = tuple(parsed.walk())
    if len(operations) > bound.nodes:
        raise ValueError("Source emission container exceeds its node budget.")
    ordinals = {op: ordinal for ordinal, op in enumerate(operations)}
    _fields(parsed, ())
    top = _block(parsed, 0, regions=1)
    module_name = _attribute(module, "sym_name")
    if not isinstance(module_name, StringAttr):
        raise ValueError("Source emission definition membership is incomplete or duplicated.")
    symbols, declarations, definitions = {module_name.data: module}, [], []
    for op in top.ops:
        if op is module:
            continue
        kind = _name(op)
        if kind not in {"sv.macro.decl", "emit.fragment"}:
            raise ValueError("Source emission container operation is unsupported.")
        _fields(op, ("sym_name",))
        symbol = _attribute(op, "sym_name")
        if not isinstance(symbol, StringAttr) or not _identifier(symbol.data) or symbol.data in symbols:
            raise ValueError("Source emission definition membership is incomplete or duplicated.")
        if op.operands or op.results:
            raise ValueError("Source emission definition operands are unsupported.")
        symbols[symbol.data] = op
        definitions.append(ordinals[op])
        if kind == "sv.macro.decl":
            if op.regions:
                raise ValueError("Source emission macro declaration is unsupported.")
            declarations.append(symbol.data)
        else:
            _block(op, 0, regions=1)
    rows = document["macros"]
    if len(operations) * max(1, len(declarations)) > bound.bit_work:
        raise ValueError("Source emission dependency work exceeds its budget.")
    if len(rows) != len(declarations):
        raise ValueError("Source emission macro premise roster is incomplete.")
    macros = {}
    for symbol, row in zip(declarations, rows, strict=True):
        if (
            type(row) is not dict
            or set(row) != {"symbol", "defined", "value"}
            or row["symbol"] != symbol
            or type(row["defined"]) is not bool
            or (row["defined"] and (type(row["value"]) is not int or row["value"] not in {0, 1}))
            or (not row["defined"] and row["value"] is not None)
        ):
            raise ValueError("Source emission macro premise roster is incomplete.")
        macros[symbol] = MacroValue(symbol, row["defined"], row["value"])
    original_macros = tuple(macros.values())
    fragment_ops = []

    def fragment_block(block, state, depth=0):
        if depth >= 64:
            raise ValueError("Source emission dependency nesting exceeds its budget.")
        if block is None:
            return
        for op in block.ops:
            kind = _name(op)
            if op.operands or op.results:
                raise ValueError("Source emission fragment operands are unsupported.")
            if kind == "sv.verbatim":
                _fields(op, ("format_string", "symbols"))
                text, refs = _attribute(op, "format_string"), _attribute(op, "symbols")
                if (
                    op.regions
                    or not isinstance(text, StringAttr)
                    or not isinstance(refs, ArrayAttr)
                    or refs.data
                    or not text.data.isascii()
                    or any(
                        line.strip(" \t\r") and not line.lstrip(" \t").startswith("//")
                        for line in text.data.split("\n")
                    )
                ):
                    raise ValueError("Source emission opaque text semantics are unresolved.")
            elif kind == "sv.macro.def":
                _fields(op, ("macroName", "format_string", "symbols"))
                symbol = _flat_reference(_attribute(op, "macroName"))
                refs = _attribute(op, "symbols")
                if op.regions or symbol not in state or not isinstance(refs, ArrayAttr) or refs.data:
                    raise ValueError("Source emission macro definition is unresolved.")
                state[symbol] = MacroValue(symbol, True, _macro_body(_attribute(op, "format_string"), state))
            elif kind == "sv.ifdef":
                _fields(op, ("cond",))
                symbol = _ifdef_symbol(op)
                if symbol not in state:
                    raise ValueError("Source emission macro definition is unresolved.")
                branch_states = []
                for index in (0, 1):
                    branch_state = dict(state)
                    fragment_block(_block(op, index, regions=2), branch_state, depth + 1)
                    branch_states.append(branch_state)
                state.update(branch_states[0 if state[symbol].defined else 1])
            else:
                raise ValueError("Source emission fragment operation is unsupported.")

    # Check unused fragment bodies too. They do not mutate the selected emitted
    # environment, and an inactive unsupported body still refuses.
    for symbol, op in symbols.items():
        if _name(op) == "emit.fragment":
            fragment_ops.extend((ordinals[child], _name(child)) for child in op.walk())
            fragment_block(op.regions[0].block, dict(macros))
    for symbol in fragments or ():
        definition = symbols.get(symbol)
        if definition is None or _name(definition) != "emit.fragment":
            raise ValueError("Source emission fragment definition is unavailable.")
        fragment_block(definition.regions[0].block, macros)
    scalars, constants, effects = [], [], []

    def local_block(block, enabled=True, clock=None, branches=(), depth=0):
        if depth >= 64:
            raise ValueError("Source emission effect nesting exceeds its budget.")
        if block is None:
            return
        for op in block.ops:
            kind = _name(op)
            if kind in {"seq.firreg", "seq.to_clock", "hw.output"} and block is module.regions[0].block:
                continue
            if kind == "sv.ifdef":
                _fields(op, ("cond",))
                symbol = _ifdef_symbol(op)
                if op.operands or op.results or symbol not in macros:
                    raise ValueError("Source emission macro condition is unresolved.")
                for index in (0, 1):
                    local_block(
                        _block(op, index, regions=2),
                        enabled and macros[symbol].defined == (index == 0),
                        clock,
                        branches,
                        depth + 1,
                    )
            elif kind == "sv.always":
                _fields(op, ("events",))
                events = _attribute(op, "events")
                if (
                    clock is not None
                    or len(op.operands) != 1
                    or op.operands[0].type != IntegerType(1)
                    or op.results
                    or not isinstance(events, ArrayAttr)
                    or len(events) != 1
                    or not isinstance(events.data[0], IntegerAttr)
                    or events.data[0].type != IntegerType(32)
                    or events.data[0].value.data != 0
                ):
                    raise ValueError("Source emission effect clock event is unsupported.")
                local_block(_block(op, 0, regions=1), enabled, op.operands[0], branches, depth + 1)
            elif kind == "sv.if":
                _fields(op, ())
                if clock is None or len(op.operands) != 1 or op.operands[0].type != IntegerType(1) or op.results:
                    raise ValueError("Source emission effect predicate is unsupported.")
                for index in (0, 1):
                    local_block(
                        _block(op, index, regions=2),
                        enabled,
                        clock,
                        (*branches, (op.operands[0], index == 0)),
                        depth + 1,
                    )
            elif kind == "sim.fatal":
                _fields(op, ())
                # The reviewed public SimToSV lowering uses this declared
                # synthesis macro. Its presence/value is never defaulted.
                synthesis = macros.get("SYNTHESIS")
                if (
                    clock is not None
                    or len(op.operands) != 2
                    or str(op.operands[0].type) != "!seq.clock"
                    or op.operands[1].type != IntegerType(1)
                    or op.results
                    or op.regions
                    or synthesis is None
                ):
                    raise ValueError("Source emission termination semantics are unresolved.")
                effects.append(
                    _Effect(op, op.operands[0], enabled and not synthesis.defined, branches, op.operands[1], None, None)
                )
            elif kind == "sv.fwrite":
                _fields(op, ("format_string",))
                text = _attribute(op, "format_string")
                if (
                    clock is None
                    or len(op.operands) != 1
                    or op.operands[0].type != IntegerType(32)
                    or op.results
                    or op.regions
                    or not isinstance(text, StringAttr)
                    or "%" in text.data
                ):
                    raise ValueError("Source emission output effect semantics are unresolved.")
                effects.append(_Effect(op, clock, enabled, branches, None, op.operands[0], text.data))
            elif kind == "sv.macro.ref":
                _fields(op, ("macroName",))
                symbol = _flat_reference(_attribute(op, "macroName"))
                macro = macros.get(symbol)
                if (
                    op.operands
                    or len(op.results) != 1
                    or op.results[0].type != IntegerType(1)
                    or op.regions
                    or macro is None
                    or not macro.defined
                    or macro.value is None
                ):
                    raise ValueError("Source emission macro value is unresolved.")
                scalars.append(op)
                constants.append((op.results[0], macro.value))
            elif not op.regions and len(op.results) == 1:
                scalars.append(op)
            else:
                raise ValueError("Source emission local operation semantics are unsupported.")

    local_block(module.regions[0].block)
    return _Emission(
        hashlib.sha256(environment.encode("utf-8")).hexdigest(),
        original_macros,
        tuple(macros.values()),
        tuple(fragments or ()),
        tuple(definitions),
        tuple(fragment_ops),
        tuple(scalars),
        tuple(constants),
        tuple(effects),
        ordinals,
    )
