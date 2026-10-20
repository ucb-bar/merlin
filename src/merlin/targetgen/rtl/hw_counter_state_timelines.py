"""Complete bounded local state timelines behind an explicitly selected getter.

Every original FirReg next/reset is evaluated from the same pre-edge state with
the shared two-state expression evaluator. The getter must concatenate complete
selected registers without dropping or duplicating bits. Observed changes are
data, not elapsed units, event custody, reachability or physical timer authority.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass

from xdsl.dialects.builtin import (
    ArrayAttr,
    DictionaryAttr,
    IntegerAttr,
    IntegerType,
    LocationAttr,
    Signedness,
    StringAttr,
    SymbolRefAttr,
    UnregisteredAttr,
)
from xdsl.ir import ParametrizedAttribute
from xdsl.utils.exceptions import ParseError, VerifyException

from merlin.targetgen.contract.mlir_source_admission import admit_mlir_source

from .hw_combinational import (
    EvaluationLimits,
    PreparedCombinationalObservation,
    ScalarPort,
    _Expression,
    _expression,
    _width,
)
from .hw_counter_intervals import _UNKNOWN, CounterInterval
from .hw_graph import parse_generic_hw
from .hw_observations import _attribute, _name
from .hw_state_effects import (
    OriginalSourceEffect,
    SourceEffectPhase,
    SourceEmissionObservation,
    prepare_source_emission,
)
from .ports import _hw_port_entries


@dataclass(frozen=True)
class StateTimelineLimits:
    expressions: EvaluationLimits
    registers: int
    intervals: int

    def __post_init__(self):
        if (
            type(self.expressions) is not EvaluationLimits
            or self.expressions.scalar_bits > 4096
            or any(type(value) is not int or not 0 < value < 1 << 63 for value in (self.registers, self.intervals))
            or any(value >= 1 << 63 for value in asdict(self.expressions).values())
        ):
            raise ValueError("State timeline limits are unavailable.")


@dataclass(frozen=True)
class StateGetterSelection:
    """Data-only source endpoints, ordered as the actual getter's MSB-first leaves."""

    source_sha256: str
    module: str
    register_ordinals: tuple[int, ...]
    getter_output: str
    clock_input: str

    def __post_init__(self):
        if (
            type(self.source_sha256) is not str
            or len(self.source_sha256) != 64
            or any(c not in "0123456789abcdef" for c in self.source_sha256)
            or any(type(value) is not str or not value for value in (self.module, self.getter_output, self.clock_input))
            or type(self.register_ordinals) is not tuple
            or not self.register_ordinals
            or any(type(value) is not int or not 0 <= value < 1 << 63 for value in self.register_ordinals)
        ):
            raise ValueError("State getter selection is unavailable.")


@dataclass(frozen=True)
class TimelinePort:
    name: str
    type: str
    width: int


@dataclass(frozen=True)
class SourceLocation:
    """Typed canonical location assembly; selected source bytes remain authoritative."""

    kind: str
    assembly: str


@dataclass(frozen=True)
class ModuleSourceMetadata:
    symbol_visibility: str | None
    result_locations: tuple[SourceLocation, ...] | None
    emission_fragments: tuple[str, ...] | None


@dataclass(frozen=True)
class OriginalState:
    ordinal: int
    name: str
    width: int
    clock_input: str
    reset_present: bool
    random_initialization_offset: int | None


@dataclass(frozen=True)
class StatePhaseSample:
    ordinal: int
    inputs: tuple[int, ...]
    states: tuple[int, ...]
    outputs: tuple[int, ...]


@dataclass(frozen=True)
class LocalStateTransition:
    phase: int
    rising_registers: tuple[int, ...]
    reset_registers: tuple[int, ...]
    before: tuple[int, ...]
    after: tuple[int, ...]


@dataclass(frozen=True)
class LocalGetterInterval:
    start: int
    end: int
    observed_modular_delta: int
    rising_edges: int
    reset_edges: int
    held_edges: int
    observed_unit_changes: int
    nonunit_changes: int
    observed_wrap_changes: int
    unit_increments: None = None


@dataclass(frozen=True)
class StateTimelineObservation:
    source_sha256: str
    selection: StateGetterSelection
    input_ports: tuple[TimelinePort, ...]
    output_ports: tuple[TimelinePort, ...]
    states: tuple[OriginalState, ...]
    getter_width: int
    samples: tuple[StatePhaseSample, ...]
    transitions: tuple[LocalStateTransition, ...]
    intervals: tuple[LocalGetterInterval, ...]
    module_metadata: ModuleSourceMetadata
    unknowns: tuple[str, ...] = (*_UNKNOWN, "unit_increment_meaning_for_general_state_updates")
    source_emission: SourceEmissionObservation | None = None


@dataclass(frozen=True)
class _PreparedEffect:
    source: OriginalSourceEffect
    predicate: int | None
    branches: tuple[tuple[int, bool], ...]
    file_descriptor: int | None


@dataclass(frozen=True)
class _Prepared:
    evaluator: PreparedCombinationalObservation
    inputs: tuple[TimelinePort, ...]
    outputs: tuple[TimelinePort, ...]
    states: tuple[OriginalState, ...]
    clocks: tuple[int, ...]
    register_clocks: tuple[int, ...]
    primary_clock: int
    getter: int
    selected_states: tuple[int, ...]
    updates: tuple[tuple[int, int | None, int | None], ...]
    output_indices: tuple[int, ...]
    module_metadata: ModuleSourceMetadata
    emission: object | None = None
    effects: tuple[_PreparedEffect, ...] = ()


def _module_metadata(module, output_count, bound, *, explicit_emission=False):
    visibility = _attribute(module, "sym_visibility")
    if visibility is not None and (
        not isinstance(visibility, StringAttr) or visibility.data not in {"public", "private", "nested"}
    ):
        raise ValueError("State timeline symbol visibility is unsupported.")
    locations = _attribute(module, "result_locs")
    if locations is not None and (
        not isinstance(locations, ArrayAttr)
        or len(locations) != output_count
        or any(not isinstance(location, LocationAttr) for location in locations)
    ):
        raise ValueError("State timeline result locations differ from the original output roster.")
    if locations is not None:
        # Location aliases can share a DAG whose expanded printer form is much
        # larger than its input spelling. Price every occurrence before printing.
        stack = [(location, 0) for location in locations]
        visits = string_chars = 0
        while stack:
            attribute, depth = stack.pop()
            visits += 1
            if visits > bound.nodes or depth > 64:
                raise ValueError("State timeline location metadata exceeds its node budget.")
            if isinstance(attribute, ParametrizedAttribute):
                stack.extend((child, depth + 1) for child in attribute.parameters)
            elif isinstance(attribute, ArrayAttr):
                if len(attribute) + visits + len(stack) > bound.nodes:
                    raise ValueError("State timeline location metadata exceeds its node budget.")
                stack.extend((child, depth + 1) for child in attribute)
            elif isinstance(attribute, StringAttr):
                string_chars += len(attribute.data)
            elif isinstance(attribute, DictionaryAttr):
                if len(attribute.data) + visits + len(stack) > bound.nodes:
                    raise ValueError("State timeline location metadata exceeds its node budget.")
                string_chars += sum(len(key) for key in attribute.data)
                stack.extend((child, depth + 1) for child in attribute.data.values())
            if string_chars > bound.source_bytes:
                raise ValueError("State timeline location metadata exceeds its source byte budget.")
        try:
            for location in locations:
                location.verify()
        except VerifyException:
            raise ValueError("State timeline result location metadata is unsupported.") from None
    fragments = _attribute(module, "emit.fragments")
    if fragments is not None and (
        not isinstance(fragments, ArrayAttr)
        or any(
            not isinstance(fragment, SymbolRefAttr)
            or fragment.nested_references.data
            or not fragment.root_reference.data
            for fragment in fragments
        )
        or len({fragment.root_reference.data for fragment in fragments}) != len(fragments)
    ):
        raise ValueError("State timeline emission dependency metadata is unsupported.")
    metadata = ModuleSourceMetadata(
        visibility.data if visibility is not None else None,
        tuple(SourceLocation(location.name, str(location)) for location in locations)
        if locations is not None
        else None,
        tuple(fragment.root_reference.data for fragment in fragments) if fragments is not None else None,
    )
    # Emit inserts each referenced body before the module. A well-typed symbol
    # reference is a required semantic dependency, not a disposable location.
    if metadata.emission_fragments and not explicit_emission:
        raise ValueError("State timeline emission dependencies are unresolved.")
    return metadata


def _ports(module, block, output, bound):
    typ = _attribute(module, "module_type")
    if not isinstance(typ, UnregisteredAttr) or typ.attr_name.data != "hw.modty":
        raise ValueError("State timeline module type is unavailable.")
    entries = _hw_port_entries("(" + typ.value.data + ")")
    if entries is None:
        raise ValueError("State timeline port roster is unavailable.")
    inputs, outputs, names = [], [], set()
    for entry in entries:
        head, separator, declared = entry.partition(":")
        words = head.split()
        if not separator or len(words) != 2 or words[0] not in {"input", "output"} or words[1] in names:
            raise ValueError("State timeline port semantics are unsupported.")
        names.add(words[1])
        ports, values = (inputs, block.args) if words[0] == "input" else (outputs, output.operands)
        if len(ports) >= len(values):
            raise ValueError("State timeline port membership is incomplete.")
        value = values[len(ports)]
        bits = 1 if str(value.type) == "!seq.clock" else _width(value, bound)
        if declared.strip() != str(value.type):
            raise ValueError("State timeline declared port types differ from the source.")
        ports.append(TimelinePort(words[1], str(value.type), bits))
        if len(inputs) + len(outputs) > bound.nodes:
            raise ValueError("State timeline ports exceed their node budget.")
    if len(inputs) != len(block.args) or len(outputs) != len(output.operands) or not outputs:
        raise ValueError("State timeline port membership is incomplete.")
    return tuple(inputs), tuple(outputs)


def _prepare(text, selection, limits, *, macro_environment=None):
    bound = limits.expressions
    if type(text) is not str or len(text) > bound.source_bytes:
        raise ValueError("State timeline source exceeds its parse budget.")
    try:
        admit_mlir_source(
            text,
            max_source_bytes=bound.source_bytes,
            max_nesting=64,
            max_integer_bits=max(64, bound.scalar_bits),
            allow_dense=False,
            allow_dense_resource=False,
        )
        parsed = parse_generic_hw(text, reject_dense_literals=True)
    except (ValueError, ParseError, VerifyException):
        raise ValueError("State timeline source grammar is unsupported.") from None
    sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if sha != selection.source_sha256:
        raise ValueError("State timeline source bytes differ from the selection.")
    matches = [
        op
        for op in parsed.walk()
        if _name(op) == "hw.module" and _attribute(op, "sym_name") == StringAttr(selection.module)
    ]
    if len(matches) != 1:
        raise ValueError("State timeline module membership is incomplete.")
    module = matches[0]
    if (
        (set(module.attributes) | set(module.properties))
        - {"op_name__", "sym_name", "module_type", "parameters", "sym_visibility", "result_locs", "emit.fragments"}
        or set(module.attributes) & set(module.properties)
        or len(module.regions) != 1
        or len(module.regions[0].blocks) != 1
    ):
        raise ValueError("State timeline module semantics are unsupported.")
    parameters = _attribute(module, "parameters")
    if parameters is not None and (not isinstance(parameters, ArrayAttr) or parameters.data):
        raise ValueError("State timeline module parameters are unresolved.")
    block = module.regions[0].block
    children = tuple(block.ops)
    if len(children) + len(block.args) > bound.nodes:
        raise ValueError("State timeline source exceeds its node budget.")
    output = block.last_op
    if (
        output is None
        or _name(output) != "hw.output"
        or output.results
        or output.regions
        or (set(output.attributes) | set(output.properties)) - {"op_name__"}
        or set(output.attributes) & set(output.properties)
    ):
        raise ValueError("State timeline output membership is incomplete.")
    inputs, outputs = _ports(module, block, output, bound)
    metadata = _module_metadata(module, len(outputs), bound, explicit_emission=macro_environment is not None)
    emission = (
        prepare_source_emission(parsed, module, metadata.emission_fragments, macro_environment, bound, sha)
        if macro_environment is not None
        else None
    )
    if len(inputs) + len(outputs) + len(children) > bound.nodes:
        raise ValueError("State timeline original roster exceeds its node budget.")
    registers = tuple((ordinal, op) for ordinal, op in enumerate(children) if _name(op) == "seq.firreg")
    if not registers or len(registers) > limits.registers or len(selection.register_ordinals) > len(registers):
        raise ValueError("State timeline register roster exceeds its bound or is incomplete.")
    if len(set(selection.register_ordinals)) != len(selection.register_ordinals):
        raise ValueError("State getter register membership is duplicated.")
    input_names, output_names = tuple(port.name for port in inputs), tuple(port.name for port in outputs)
    if selection.clock_input not in input_names or selection.getter_output not in output_names:
        raise ValueError("State timeline endpoint membership is incomplete.")
    primary_clock, getter = input_names.index(selection.clock_input), output_names.index(selection.getter_output)
    roots = {value: index for index, value in enumerate(block.args) if str(value.type) == "!seq.clock"}
    clock_ops = set()
    for op in children:
        if _name(op) == "seq.to_clock":
            if (
                op.regions
                or len(op.operands) != 1
                or len(op.results) != 1
                or (set(op.attributes) | set(op.properties)) - {"op_name__"}
                or set(op.attributes) & set(op.properties)
                or op.operands[0] not in block.args
                or op.operands[0].type != IntegerType(1)
                or str(op.results[0].type) != "!seq.clock"
            ):
                raise ValueError("State timeline clock expression is unsupported.")
            roots[op.results[0]] = block.args.index(op.operands[0])
            clock_ops.add(op)
    clocks = tuple(sorted(set(roots.values())))
    if primary_clock not in clocks:
        raise ValueError("State timeline selected clock differs from the source.")
    states, register_clocks = [], []
    for ordinal, op in registers:
        if (
            op.regions
            or len(op.results) != 1
            or len(op.operands) not in {2, 4}
            or (set(op.attributes) | set(op.properties))
            - {"op_name__", "name", "sv.namehint", "firrtl.random_init_start"}
            or set(op.attributes) & set(op.properties)
            or not isinstance(_attribute(op, "name"), StringAttr)
            or (_attribute(op, "sv.namehint") is not None and not isinstance(_attribute(op, "sv.namehint"), StringAttr))
        ):
            raise ValueError("State timeline register semantics are unsupported.")
        width = _width(op.results[0], bound)
        if op.operands[0].type != op.results[0].type or op.operands[1] not in roots:
            raise ValueError("State timeline register clock or next type is unsupported.")
        if len(op.operands) == 4 and (
            op.operands[2].type != IntegerType(1) or op.operands[3].type != op.results[0].type
        ):
            raise ValueError("State timeline reset types differ from the source.")
        random = _attribute(op, "firrtl.random_init_start")
        if random is not None and (
            not isinstance(random, IntegerAttr)
            or random.type != IntegerType(64, Signedness.UNSIGNED)
            or not 0 <= random.value.data < 1 << 64
        ):
            raise ValueError("State timeline initialization metadata is unsupported.")
        clock_index = roots[op.operands[1]]
        states.append(
            OriginalState(
                ordinal,
                _attribute(op, "name").data,
                width,
                input_names[clock_index],
                len(op.operands) == 4,
                random.value.data if random is not None else None,
            )
        )
        register_clocks.append(clock_index)
    states = tuple(states)
    state_values = tuple(op.results[0] for _, op in registers)
    state_indices = {value: index for index, value in enumerate(state_values)}
    ordinals = tuple(state.ordinal for state in states)

    getter_seen = set()

    def getter_leaves(value, depth=0):
        if value in getter_seen or len(getter_seen) >= bound.nodes:
            raise ValueError("State getter dependencies are duplicated or exceed their budget.")
        getter_seen.add(value)
        if value in state_indices:
            return (state_indices[value],)
        op = value.owner
        if depth >= 64 or op not in children or _name(op) != "comb.concat":
            raise ValueError("State getter does not retain complete original register bits.")
        return tuple(index for operand in op.operands for index in getter_leaves(operand, depth + 1))

    selected = getter_leaves(output.operands[getter])
    if (
        tuple(ordinals[index] for index in selected) != selection.register_ordinals
        or len(set(selected)) != len(selected)
        or any(register_clocks[index] != primary_clock for index in selected)
    ):
        raise ValueError("State getter register or clock membership differs from the selection.")
    if outputs[getter].width != sum(states[index].width for index in selected):
        raise ValueError("State getter width differs from the complete original registers.")
    indices = {value: index for index, value in enumerate((*block.args, *state_values))}
    for value, index in roots.items():
        indices[value] = index
    depths = {value: 0 for value in indices}
    expressions, visiting = [], set()
    base = len(inputs) + len(states)
    work = sum(port.width for port in (*inputs, *outputs)) + sum(state.width for state in states)
    widths = [port.width for port in inputs] + [state.width for state in states]
    if work > bound.bit_work:
        raise ValueError("State timeline typed roster exceeds its bit-work budget.")

    def trace(value):
        nonlocal work
        if value in indices:
            return indices[value]
        if value in visiting or len(visiting) >= 64:
            raise ValueError("State timeline expression dependencies are cyclic or too deep.")
        op = value.owner
        if op not in scalar_ops or op in clock_ops or op is output or op.regions or len(op.results) != 1:
            raise ValueError("State timeline expression semantics are unsupported.")
        visiting.add(value)
        bits = _width(value, bound)
        if value in macro_constants:
            operand_widths = []
            kind, parameter = "hw.constant", macro_constants[value]
        elif _name(op) == "seq.from_clock":
            if (
                bits != 1
                or len(op.operands) != 1
                or op.operands[0] not in roots
                or str(op.operands[0].type) != "!seq.clock"
                or (set(op.attributes) | set(op.properties)) - {"op_name__", "sv.namehint"}
                or set(op.attributes) & set(op.properties)
                or (
                    _attribute(op, "sv.namehint") is not None
                    and not isinstance(_attribute(op, "sv.namehint"), StringAttr)
                )
            ):
                raise ValueError("State timeline clock value expression is unsupported.")
            operand_widths = [1]
            kind, parameter = "seq.from_clock", None
        else:
            operand_widths = [_width(operand, bound) for operand in op.operands]
            try:
                kind, parameter = _expression(op, operand_widths, bits, conditional_logic=True)
            except ValueError:
                raise ValueError("State timeline expression semantics are unsupported.") from None
        operands = tuple(trace(operand) for operand in op.operands)
        depth = 1 + max((depths[operand] for operand in op.operands), default=0)
        if depth > 64:
            raise ValueError("State timeline expression dependencies are cyclic or too deep.")
        work += bits + sum(operand_widths)
        if work > bound.bit_work:
            raise ValueError("State timeline expression work exceeds its budget.")
        indices[value] = base + len(expressions)
        depths[value] = depth
        expressions.append(_Expression(kind, bits, operands, parameter))
        widths.append(bits)
        visiting.remove(value)
        return indices[value]

    register_ops = {op for _, op in registers}
    scalar_ops = (
        set(emission.scalar_operations) if emission is not None else set(children) - register_ops - clock_ops - {output}
    )
    macro_constants = dict(emission.macro_constants) if emission is not None else {}
    # The shared lossless parser owns SSA region visibility. Preserve its whole
    # original walk order when preparing newly supported nested expressions.
    scalar_order = tuple(op for op in module.walk() if op in scalar_ops)
    for op in scalar_order:
        if op not in register_ops | clock_ops | {output}:
            if op.regions or len(op.results) != 1:
                raise ValueError("State timeline expression semantics are unsupported.")
            trace(op.results[0])
    updates = []
    for _, op in registers:
        reset = reset_value = None
        if len(op.operands) == 4:
            if op.operands[3] in block.args or _name(op.operands[3].owner) != "hw.constant":
                raise ValueError("State timeline reset value semantics are unsupported.")
            reset, reset_value = (trace(value) for value in op.operands[2:])
        updates.append((trace(op.operands[0]), reset, reset_value))
    output_indices = tuple(trace(value) for value in output.operands)
    prepared_effects = []
    if emission is not None:

        def endpoint(value):
            return (
                ("input", block.args.index(value))
                if value in block.args
                else ("operation", emission.operation_ordinals[value.owner])
            )

        for effect in emission.effects:
            clock = effect.clock
            if clock not in roots and clock.owner in scalar_ops and _name(clock.owner) == "seq.from_clock":
                clock = clock.owner.operands[0]
            if clock in roots:
                clock_index = roots[clock]
            elif clock in block.args and block.args.index(clock) in clocks:
                clock_index = block.args.index(clock)
            else:
                raise ValueError("State timeline observable effect clock is unsupported.")
            source = OriginalSourceEffect(
                emission.operation_ordinals[effect.operation],
                _name(effect.operation),
                clock_index,
                effect.enabled,
                endpoint(effect.predicate) if effect.predicate is not None else None,
                tuple((endpoint(value), required) for value, required in effect.branches),
                effect.format_string,
            )
            prepared_effects.append(
                _PreparedEffect(
                    source,
                    trace(effect.predicate) if effect.predicate is not None else None,
                    tuple((trace(value), required) for value, required in effect.branches),
                    trace(effect.file_descriptor) if effect.file_descriptor is not None else None,
                )
            )
    prepared_effects = tuple(prepared_effects)
    refs = tuple(
        dict.fromkeys(
            (
                *[value for update in updates for value in update if value is not None],
                *output_indices,
                *[
                    value
                    for effect in prepared_effects
                    for value in (effect.predicate, effect.file_descriptor)
                    if value is not None
                ],
                *[value for effect in prepared_effects for value, _ in effect.branches],
            )
        )
    )
    internal = tuple(ScalarPort(f"${index}", width) for index, width in enumerate(widths[:base]))
    evaluator = PreparedCombinationalObservation(
        sha,
        selection.module,
        internal,
        tuple(ScalarPort(str(index), widths[index]) for index in refs),
        tuple(expressions),
        refs,
        work,
        bound,
    )
    return _Prepared(
        evaluator,
        inputs,
        outputs,
        states,
        clocks,
        tuple(register_clocks),
        primary_clock,
        getter,
        selected,
        tuple(updates),
        output_indices,
        metadata,
        emission,
        prepared_effects,
    )


def observe_state_getter_timeline(
    text: str,
    *,
    selection: StateGetterSelection,
    limits: StateTimelineLimits,
    expected_phases: int,
    samples: tuple[StatePhaseSample, ...],
    intervals: tuple[CounterInterval, ...],
    macro_environment: str | None = None,
) -> StateTimelineObservation:
    """Reparse every original state update; compare complete post-phase samples.

    Initial state is conditional observed data. All clock roots start LOW; the
    selected clock alternates LOW/HIGH. Other roots may hold, with rising edges only
    at HIGH phases. Reset priority and all state updates use pre-edge values. A
    numeric +1 does not establish a unit event: writes can produce that same delta.
    """
    if type(selection) is not StateGetterSelection or type(limits) is not StateTimelineLimits:
        raise ValueError("State timeline selection is unavailable.")
    if (
        type(expected_phases) is not int
        or expected_phases < 2
        or expected_phases % 2
        or type(samples) is not tuple
        or len(samples) != expected_phases
        or expected_phases > limits.expressions.cases
        or type(intervals) is not tuple
        or not intervals
        or len(intervals) > limits.intervals
    ):
        raise ValueError("State timeline phase or interval roster is incomplete.")
    prepared = _prepare(text, selection, limits, macro_environment=macro_environment)
    if len(prepared.effects) * expected_phases > limits.expressions.nodes:
        raise ValueError("State timeline observable effect roster exceeds its budget.")
    if expected_phases * 2 * prepared.evaluator.per_case_bit_work > limits.expressions.bit_work:
        raise ValueError("State timeline exceeds its bit-work budget.")
    state_ports = tuple(ScalarPort(str(state.ordinal), state.width) for state in prepared.states)
    for ordinal, sample in enumerate(samples):
        if type(sample) is not StatePhaseSample or type(sample.ordinal) is not int or sample.ordinal != ordinal:
            raise ValueError("State timeline phase membership is incomplete.")
        for values, ports in (
            (sample.inputs, prepared.inputs),
            (sample.states, state_ports),
            (sample.outputs, prepared.outputs),
        ):
            if (
                type(values) is not tuple
                or len(values) != len(ports)
                or any(
                    type(value) is not int or not 0 <= value < 1 << port.width
                    for value, port in zip(values, ports, strict=True)
                )
            ):
                raise ValueError("State timeline sample differs from the complete original typed roster.")
        if sample.inputs[prepared.primary_clock] != ordinal % 2 or (
            ordinal == 0 and any(sample.inputs[index] for index in prepared.clocks)
        ):
            raise ValueError("State timeline original clock phases are incomplete.")
        if ordinal:
            previous = samples[ordinal - 1]
            if any(
                sample.inputs[index] != previous.inputs[index]
                for index in prepared.clocks
                if sample.inputs[index] != ordinal % 2
            ):
                raise ValueError("State timeline secondary clock phases are inconsistent.")
            if ordinal % 2 and any(
                value != previous.inputs[index]
                for index, value in enumerate(sample.inputs)
                if index not in prepared.clocks
            ):
                raise ValueError("State timeline edge inputs are unstable.")
    for interval in intervals:
        if (
            type(interval) is not CounterInterval
            or type(interval.start) is not int
            or type(interval.end) is not int
            or not 0 <= interval.start < interval.end < expected_phases
        ):
            raise ValueError("State timeline interval endpoints are unavailable.")

    def evaluate(sample, states):
        return prepared.evaluator.evaluate(
            (dict(zip((port.name for port in prepared.evaluator.inputs), (*sample.inputs, *states), strict=True)),)
        )[0]

    states, transitions, prefixes = samples[0].states, [], [(0, 0, 0, 0, 0, 0)]
    effect_phases = []
    domain = 1 << prepared.outputs[prepared.getter].width
    for ordinal, sample in enumerate(samples):
        before, reset_registers, rising = states, [], []
        values = evaluate(sample, before)
        for effect in prepared.effects:
            edge = bool(
                ordinal
                and not samples[ordinal - 1].inputs[effect.source.clock_input]
                and sample.inputs[effect.source.clock_input]
            )
            predicate = values[str(effect.predicate)] if effect.predicate is not None else None
            branches = tuple(values[str(value)] for value, _ in effect.branches)
            active = (
                effect.source.compile_enabled
                and edge
                and (predicate is None or bool(predicate))
                and all(bool(value) == required for value, (_, required) in zip(branches, effect.branches, strict=True))
            )
            effect_phases.append(
                SourceEffectPhase(
                    ordinal,
                    effect.source.ordinal,
                    effect.source.operation,
                    "triggered" if active else "inactive",
                    edge,
                    predicate,
                    branches,
                    values[str(effect.file_descriptor)] if effect.file_descriptor is not None else None,
                )
            )
            if active and effect.source.operation == "sim.fatal" and ordinal + 1 < expected_phases:
                raise ValueError("State timeline continues after a triggered termination.")
        prior_getter = values[str(prepared.output_indices[prepared.getter])]
        after = list(before)
        if ordinal:
            for index, (next_index, reset, reset_value) in enumerate(prepared.updates):
                clock = prepared.register_clocks[index]
                if not samples[ordinal - 1].inputs[clock] and sample.inputs[clock]:
                    rising.append(prepared.states[index].ordinal)
                    active_reset = reset is not None and bool(values[str(reset)])
                    if active_reset:
                        reset_registers.append(prepared.states[index].ordinal)
                    after[index] = values[str(reset_value if active_reset else next_index)]
        states = tuple(after)
        values = evaluate(sample, states)
        observed = tuple(values[str(index)] for index in prepared.output_indices)
        if states != sample.states or observed != sample.outputs:
            raise ValueError("State timeline differs from the actual source transitions.")
        transitions.append(LocalStateTransition(ordinal, tuple(rising), tuple(reset_registers), before, states))
        selected_reset = any(prepared.states[index].ordinal in reset_registers for index in prepared.selected_states)
        edge = bool(
            ordinal
            and not samples[ordinal - 1].inputs[prepared.primary_clock]
            and sample.inputs[prepared.primary_clock]
        )
        after_getter = observed[prepared.getter]
        delta = (after_getter - prior_getter) % domain
        flags = (
            edge,
            selected_reset,
            edge and not selected_reset and delta == 0,
            edge and not selected_reset and delta == 1,
            edge and not selected_reset and delta not in {0, 1},
            edge and not selected_reset and delta == 1 and after_getter < prior_getter,
        )
        prefixes.append(tuple(left + int(right) for left, right in zip(prefixes[-1], flags, strict=True)))
    results = []
    for interval in intervals:
        counts = tuple(
            right - left for left, right in zip(prefixes[interval.start + 1], prefixes[interval.end + 1], strict=True)
        )
        first, last = (samples[index].outputs[prepared.getter] for index in (interval.start, interval.end))
        results.append(LocalGetterInterval(interval.start, interval.end, (last - first) % domain, *counts))
    return StateTimelineObservation(
        selection.source_sha256,
        selection,
        prepared.inputs,
        prepared.outputs,
        prepared.states,
        prepared.outputs[prepared.getter].width,
        samples,
        tuple(transitions),
        tuple(results),
        prepared.module_metadata,
        source_emission=SourceEmissionObservation(
            prepared.emission.environment_sha256,
            prepared.emission.original_macros,
            prepared.emission.macros,
            prepared.emission.fragments,
            prepared.emission.definitions,
            prepared.emission.fragment_operations,
            tuple(effect.source for effect in prepared.effects),
            tuple(effect_phases),
        )
        if prepared.emission is not None
        else None,
    )
