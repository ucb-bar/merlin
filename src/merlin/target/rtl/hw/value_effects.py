"""Bounded source-local emission dependencies and observable scalar effects.

External macro data is an explicit conditional premise, never a compiler or
runtime environment observation. All original definitions and inactive regions
are checked. Opaque bodies and effects outside this exact domain refuse.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

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
from xdsl.ir import BlockArgument, OpResult
from xdsl.parser import Parser
from xdsl.utils.exceptions import ParseError

from merlin.target.contract.interfaces.grammar import admit_mlir_source
from merlin.target.rtl.hw.evaluation import _expression
from merlin.target.rtl.hw.graph import _attribute, _module_name, _name, parse_generic_hw
from merlin.target.rtl.hw.hw_hierarchy_bindings import _binding, _parameters, _static_signature
from merlin.target.rtl.hw.instance_ports import _COMBINATIONAL

# ---- hw_state_effects --------------------------------------------------------------------------


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


# ---- hw_value_bindings -------------------------------------------------------------------------
# Bounded exact source value connectivity, without endpoint semantic roles.
#
# Selections identify original SSA slots in an exact source and occurrence. A
# graph may follow defined combinational module bindings. State, clocks, memory,
# parameterized/external bodies and unsupported values stay explicit cuts. The
# complete original state/effect membership of visited definitions is retained.

SCHEMA = "merlin.hw_value_bindings.v1"
_STATE = {"seq.firreg", "seq.compreg", "seq.compreg.ce", "seq.shiftreg"}
_CLOCK = {"seq.to_clock", "seq.from_clock"}
_PURE = _COMBINATIONAL | {"comb.replicate", "comb.shru"}
_KINDS = {"module_input", "module_output", "operation_result", "operation_operand"}


@dataclass(frozen=True)
class ValueBindingLimits:
    source_bytes: int
    modules: int
    operations: int
    occurrences: int
    port_bindings: int
    selections: int
    nodes: int
    scalar_bits: int
    bit_work: int
    hierarchy_depth: int
    expression_depth: int
    metadata_bytes: int

    def __post_init__(self):
        if any(type(value) is not int or value <= 0 for value in asdict(self).values()):
            raise ValueError("Value bindings require explicit positive limits.")


@dataclass(frozen=True)
class OriginalValueSelection:
    source_sha256: str
    occurrence: tuple[str, ...]
    module: str
    kind: str
    ordinal: int
    slot: int
    type: str

    def __post_init__(self):
        if (
            type(self.source_sha256) is not str
            or len(self.source_sha256) != 64
            or any(char not in "0123456789abcdef" for char in self.source_sha256)
            or type(self.occurrence) is not tuple
            or not self.occurrence
            or any(type(name) is not str or not name for name in self.occurrence)
            or type(self.module) is not str
            or not self.module
            or type(self.kind) is not str
            or self.kind not in _KINDS
            or type(self.ordinal) is not int
            or self.ordinal < 0
            or type(self.slot) is not int
            or self.slot < 0
            or type(self.type) is not str
            or not self.type
            or (self.kind in {"module_input", "module_output"} and self.slot != 0)
        ):
            raise ValueError("Value binding selection has unsupported identity fields.")


def prepare_value_bindings(text, *, root, selections, limits):
    """Parse exact selected bytes; selection descriptors grant no source roles."""
    if type(limits) is not ValueBindingLimits or type(text) is not str or len(text) > limits.source_bytes:
        raise ValueError("Value binding source exceeds its explicit parse budget.")
    if (
        type(root) is not str
        or not root
        or type(selections) is not tuple
        or not selections
        or len(selections) > limits.selections
        or any(type(row) is not OriginalValueSelection for row in selections)
        or len(set(selections)) != len(selections)
    ):
        raise ValueError("Value bindings require an exact nonempty selection roster.")
    admit_mlir_source(
        text,
        max_source_bytes=limits.source_bytes,
        max_nesting=64,
        max_integer_bits=max(64, limits.scalar_bits),
        allow_dense=False,
        allow_dense_resource=False,
    )
    identity = hashlib.sha256(text.encode()).hexdigest()
    if any(row.source_sha256 != identity for row in selections):
        raise ValueError("Value binding source identity differs from its selections.")
    result = hierarchical_value_bindings(
        parse_generic_hw(text, reject_dense_literals=True), root=root, selections=selections, limits=limits
    )
    result["source_sha256"] = identity
    result["source_byte_binding"] = "exact supplied source bytes"
    return result


def hierarchical_value_bindings(parsed, *, root, selections, limits):
    """Export structural dependencies only, preserving all original source cuts."""
    if (
        type(limits) is not ValueBindingLimits
        or type(root) is not str
        or not root
        or type(selections) is not tuple
        or not selections
        or len(selections) > limits.selections
        or any(type(row) is not OriginalValueSelection for row in selections)
        or len(set(selections)) != len(selections)
    ):
        raise ValueError("Value bindings require explicit limits and original selections.")
    modules, operation_count = {}, 0
    for op in parsed.walk():
        operation_count += 1
        if operation_count > limits.operations:
            raise ValueError("Value binding source exceeds its complete operation budget.")
        if _name(op) in {"hw.module", "hw.module.extern"}:
            name = _module_name(op)
            if name in modules or len(modules) >= limits.modules:
                raise ValueError("Value binding module membership differs or exceeds its budget.")
            modules[name] = op
    if root not in modules or _name(modules[root]) != "hw.module" or _parameters(modules[root]):
        raise ValueError("Value binding root body is unavailable or parameterized.")

    definitions, operation_paths, block_paths = {}, {}, {}
    for name, module in modules.items():
        ports = _static_signature(module)
        block = module.regions[0].block if _name(module) == "hw.module" else None
        ops = tuple(block.ops) if block is not None else ()
        children, names = {}, set()
        for ordinal, op in enumerate(ops):
            if _name(op) == "hw.instance":
                info = _binding(op, modules)
                if info[0] in names or op.regions:
                    raise ValueError("Value binding instance membership is ambiguous.")
                names.add(info[0])
                children[op] = ordinal, info
        definitions[name] = ports, block, ops, {op: i for i, op in enumerate(ops)}, children
        operation_paths[name], block_paths[name] = {}, {}

        def locate(original_block, path):
            block_paths[name][original_block] = path
            for ordinal, child in enumerate(original_block.ops):
                child_path = path + (ordinal,)
                operation_paths[name][child] = child_path
                for region_ordinal, region in enumerate(child.regions):
                    for block_ordinal, nested in enumerate(region.blocks):
                        locate(nested, child_path + (region_ordinal, block_ordinal))

        if block is not None:
            locate(block, ())

    sizes = {}

    def size(module, ancestors=()):
        if module in ancestors or len(ancestors) >= limits.hierarchy_depth:
            raise ValueError("Value binding hierarchy is recursive or exceeds its depth budget.")
        if module in sizes:
            count, ports, height = sizes[module]
        else:
            count, ports, height = 1, 0, 1
            for _, (_, (_, callee, signature, _)) in definitions[module][4].items():
                child_count, child_ports, child_height = size(callee, ancestors + (module,))
                count += child_count
                ports += len(signature) + child_ports
                height = max(height, child_height + 1)
                if count > limits.occurrences or ports > limits.port_bindings:
                    raise ValueError("Value binding hierarchy exceeds its pre-expansion budget.")
            sizes[module] = count, ports, height
        if len(ancestors) + height > limits.hierarchy_depth:
            raise ValueError("Value binding hierarchy exceeds its complete depth budget.")
        return count, ports, height

    # Validate unused definition edges too; no malformed body is silently dropped.
    for module in modules:
        size(module)
    expected, expected_ports, height = size(root)
    if expected > limits.occurrences or expected_ports > limits.port_bindings:
        raise ValueError("Value binding hierarchy exceeds its pre-expansion budget.")
    frames, by_path, child_frames = [], {}, {}

    def expand(module, path, parent=None, incoming=None, boundary=None):
        index = len(frames)
        if path in by_path:
            raise ValueError("Value binding occurrence identity is ambiguous.")
        by_path[path] = index
        ports, block, _, _, children = definitions[module]
        stop = boundary or (
            "external_body" if block is None else "parameterized_body" if _parameters(modules[module]) else None
        )
        frames.append(
            {
                "id": index,
                "module": module,
                "path": list(path),
                "parent": parent,
                "stop": stop,
                "original_ports": [asdict(port) for port in ports],
                "incoming": incoming,
            }
        )
        for op, (_, (name, callee, _, child_stop)) in children.items():
            child_frames[index, op] = expand(callee, path + (name,), index, op, child_stop)
        return index

    expand(root, (root,))
    if len(frames) != expected:
        raise ValueError("Value binding complete occurrence membership differs.")
    totals = {
        "modules": len(modules),
        "operations": operation_count,
        "occurrences": expected,
        "port_bindings": expected_ports,
        "hierarchy_depth": height,
        "nodes": 0,
        "bit_work": 0,
        "metadata_bytes": 0,
    }

    def metadata(op):
        attrs = {key: str(value) for key, value in op.attributes.items()}
        props = {key: str(value) for key, value in op.properties.items()}
        totals["metadata_bytes"] += sum(
            len(key.encode()) + len(value.encode()) for key, value in (*attrs.items(), *props.items())
        )
        if totals["metadata_bytes"] > limits.metadata_bytes:
            raise ValueError("Value binding metadata exceeds its aggregate budget.")
        return {"original_attributes": attrs, "original_properties": props, "location": str(op.location)}

    def reference(module, value):
        _, block, _, indices, _ = definitions[module]
        if isinstance(value, BlockArgument) and value.block is block:
            return {"kind": "module_input", "ordinal": value.index, "type": str(value.type)}
        if isinstance(value, OpResult) and value.owner.parent is block:
            return {
                "kind": "operation_result",
                "ordinal": indices[value.owner],
                "slot": value.index,
                "type": str(value.type),
            }
        if isinstance(value, OpResult) and value.owner in operation_paths[module]:
            return {
                "kind": "nested_operation_result",
                "path": list(operation_paths[module][value.owner]),
                "slot": value.index,
                "type": str(value.type),
            }
        if isinstance(value, BlockArgument) and value.block in block_paths[module]:
            return {
                "kind": "nested_block_argument",
                "path": list(block_paths[module][value.block]),
                "ordinal": value.index,
                "type": str(value.type),
            }
        raise ValueError("Value binding original SSA ownership is unavailable.")

    def width(value):
        if str(value.type) == "!seq.clock":
            return None
        if not isinstance(value.type, IntegerType) or value.type != IntegerType(value.type.width.data):
            return None
        bits = value.type.width.data
        if not 0 < bits <= limits.scalar_bits:
            raise ValueError("Value binding scalar exceeds its explicit width budget.")
        return bits

    nodes, visiting, visited_modules = {}, set(), set()

    def trace(frame_id, value, depth=0):
        key = frame_id, value
        if key in nodes:
            return nodes[key]["id"]
        if key in visiting or depth >= limits.expression_depth:
            raise ValueError("Value binding expression is cyclic or exceeds its depth budget.")
        if totals["nodes"] >= limits.nodes:
            raise ValueError("Value binding expression exceeds its node budget.")
        bits = width(value)
        totals["bit_work"] += bits or 0
        if totals["bit_work"] > limits.bit_work:
            raise ValueError("Value binding expression exceeds its bit-work budget.")
        row = {"id": totals["nodes"], "frame": frame_id, "type": str(value.type), "width": bits}
        totals["nodes"] += 1
        frame = frames[frame_id]
        module = frame["module"]
        visited_modules.add(module)
        _, block, _, indices, _ = definitions[module]
        row["original_value"] = reference(module, value)
        visiting.add(key)
        if frame["stop"] is not None:
            row.update(kind="opaque_instance_result", stop=frame["stop"])
        elif str(value.type) == "!seq.clock":
            row.update(kind="clock_value")
        elif isinstance(value, BlockArgument) and value.block is block:
            parent = frame["parent"]
            if parent is None:
                row.update(kind="root_input")
            else:
                row.update(
                    kind="instance_input_binding",
                    operands=[trace(parent, frame["incoming"].operands[value.index], depth + 1)],
                )
        elif isinstance(value, OpResult) and value.owner.parent is block:
            op, kind = value.owner, _name(value.owner)
            row.update(
                operation=kind,
                **metadata(op),
                original_operand_values=[reference(module, operand) for operand in op.operands],
                original_result_types=[str(result.type) for result in op.results],
                original_regions=len(op.regions),
            )
            if kind == "hw.instance":
                child = child_frames[frame_id, op]
                callee = frames[child]["module"]
                stop = frames[child]["stop"]
                if stop is not None:
                    row.update(kind="opaque_instance_result", child_frame=child, stop=stop)
                else:
                    output = definitions[callee][1].last_op.operands[value.index]
                    row.update(
                        kind="instance_output_binding", child_frame=child, operands=[trace(child, output, depth + 1)]
                    )
            elif kind in _STATE:
                row.update(kind="state_result")
            elif kind in _CLOCK:
                row.update(kind="clock_value")
            elif kind in {"seq.firmem.read_port", "seq.firmem.read_write_port"}:
                row.update(kind="memory_read_result")
            elif kind in _PURE and bits is not None and not op.regions and len(op.results) == 1:
                widths = [width(operand) for operand in op.operands]
                if any(bit is None for bit in widths):
                    row.update(kind="unsupported_result")
                else:
                    try:
                        expression, parameter = _expression(op, widths, bits, conditional_logic=True)
                    except ValueError as error:
                        row.update(kind="unsupported_result", reason=str(error))
                    else:
                        totals["bit_work"] += sum(widths)
                        if totals["bit_work"] > limits.bit_work:
                            raise ValueError("Value binding expression exceeds its bit-work budget.")
                        row.update(
                            kind="combinational",
                            expression=expression,
                            parameter=parameter,
                            operands=[trace(frame_id, operand, depth + 1) for operand in op.operands],
                        )
            else:
                row.update(kind="unsupported_result")
        else:
            row.update(kind="unsupported_result")
        visiting.remove(key)
        nodes[key] = row
        return row["id"]

    selected = []
    for selection in selections:
        frame_id = by_path.get(selection.occurrence)
        if frame_id is None or frames[frame_id]["module"] != selection.module:
            raise ValueError("Value binding selected occurrence differs from original source.")
        _, block, ops, _, _ = definitions[selection.module]
        if block is None or frames[frame_id]["stop"] is not None:
            raise ValueError("Value binding selected body is opaque.")
        try:
            if selection.kind == "module_input":
                value = block.args[selection.ordinal]
            elif selection.kind == "module_output":
                value = block.last_op.operands[selection.ordinal]
            elif selection.kind == "operation_result":
                value = ops[selection.ordinal].results[selection.slot]
            else:
                value = ops[selection.ordinal].operands[selection.slot]
        except IndexError:
            raise ValueError("Value binding selected original SSA slot is unavailable.") from None
        if str(value.type) != selection.type:
            raise ValueError("Value binding selected original type differs.")
        selected.append({"selection": asdict(selection), "frame": frame_id, "value": trace(frame_id, value)})

    members = []
    for module in modules:
        if module not in visited_modules:
            continue
        ports, block, ops, _, _ = definitions[module]
        retained = []
        for ordinal, op in enumerate(ops):
            retained.append(
                {
                    "ordinal": ordinal,
                    "operation": _name(op),
                    **metadata(op),
                    "operands": [reference(module, value) for value in op.operands],
                    "result_types": [str(value.type) for value in op.results],
                    "regions": [
                        [[str(v.type) for v in block.args] for block in region.blocks] for region in op.regions
                    ],
                    "nested_operations": [
                        {
                            "operation": _name(child),
                            **metadata(child),
                            "path": list(operation_paths[module][child]),
                            "operands": [reference(module, v) for v in child.operands],
                            "result_types": [str(v.type) for v in child.results],
                            "regions": [
                                [[str(v.type) for v in block.args] for block in region.blocks]
                                for region in child.regions
                            ],
                        }
                        for child in op.walk()
                        if child is not op
                    ],
                }
            )
        members.append(
            {
                "module": module,
                "original_ports": [asdict(port) for port in ports],
                "operation_count": len(ops),
                **metadata(modules[module]),
                "complete_operation_members": retained,
                "member_semantics": "UNKNOWN",
            }
        )
    public_frames = []
    for frame in frames:
        row = {key: value for key, value in frame.items() if key != "incoming"}
        op = frame["incoming"]
        if op is not None:
            parent_module = frames[frame["parent"]]["module"]
            row["incoming_binding"] = {
                "ordinal": definitions[parent_module][3][op],
                **metadata(op),
                "operands": [reference(parent_module, value) for value in op.operands],
                "result_types": [str(value.type) for value in op.results],
            }
        public_frames.append(row)
    return {
        "schema": SCHEMA,
        "root": root,
        "limits": asdict(limits),
        "cost": totals,
        "frames": public_frames,
        "selections": selected,
        "expressions": sorted(nodes.values(), key=lambda row: row["id"]),
        "visited_definition_members": members,
        "unknowns": [
            "source_member_semantic_roles",
            "unvisited_expression_semantics",
            "state_clock_memory_and_opaque_cut_semantics",
            "source_and_runtime_correspondence",
            "getter_return_sample_event_custody",
            "physical_units_and_complete_costs",
        ],
        "scope": "conditional typed SSA connectivity only",
        "admission_authority": False,
        "source_byte_binding": "not reopened by parsed-graph API",
    }
