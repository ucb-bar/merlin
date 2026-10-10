"""Bounded source-local observations of guarded unit counter transitions.

The selected FirReg, clock conversion, update and getter are rederived from the
actual typed source. Complete post-evaluation LOW/HIGH rows are checked with
the existing two-state expression evaluator. These observations establish no
sample custody, physical clock unit, elapsed cost or admission authority.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass

from xdsl.dialects.builtin import ArrayAttr, IntegerType, StringAttr
from xdsl.utils.exceptions import ParseError, VerifyException

from merlin.targetgen.contract.mlir_source_admission import admit_mlir_source

from .hw_combinational import (
    EvaluationLimits,
    PreparedCombinationalObservation,
    ScalarPort,
    _Expression,
    _expression,
    _ports,
    _width,
)
from .hw_graph import parse_generic_hw
from .hw_observations import _attribute, _integer, _name

_UNKNOWN = (
    "native_source_sdk_and_selected_hardware_correspondence",
    "sample_producer_integrity_and_phase_completeness",
    "initial_state_reachability_and_external_events",
    "getter_execution_and_interval_endpoint_correspondence",
    "physical_clock_units_frequency_and_loaded_image",
    "startup_return_sample_publication_and_parent_readback_costs",
    "complete_stage_composition_and_cold_warm_reuse",
    "independent_held_group_predictive_qualification",
)


@dataclass(frozen=True)
class CounterIntervalLimits:
    expressions: EvaluationLimits
    intervals: int

    def __post_init__(self):
        if (
            type(self.expressions) is not EvaluationLimits
            or type(self.intervals) is not int
            or not 0 < self.intervals < 1 << 63
            or self.expressions.scalar_bits > 4096
            or any(value >= 1 << 63 for value in asdict(self.expressions).values())
        ):
            raise ValueError("Counter observation limits are unavailable.")


@dataclass(frozen=True)
class CounterEndpointSelection:
    """Explicit OOT source endpoints; a declaration is not source authority."""

    source_sha256: str
    module: str
    register_ordinal: int
    getter_output: str
    clock_input: str

    def __post_init__(self):
        if (
            type(self.source_sha256) is not str
            or len(self.source_sha256) != 64
            or any(c not in "0123456789abcdef" for c in self.source_sha256)
            or any(type(value) is not str or not value for value in (self.module, self.getter_output, self.clock_input))
            or type(self.register_ordinal) is not int
            or not 0 <= self.register_ordinal < 1 << 63
        ):
            raise ValueError("Counter source endpoint selection is unavailable.")


@dataclass(frozen=True)
class CounterPhaseSample:
    ordinal: int
    inputs: tuple[int, ...]
    outputs: tuple[int, ...]


@dataclass(frozen=True)
class CounterInterval:
    start: int
    end: int


@dataclass(frozen=True)
class CounterTransition:
    phase: int
    rising_edge: bool
    reset: bool
    increment: bool
    before: int
    after: int
    wrap: bool


@dataclass(frozen=True)
class ObservedCounterInterval:
    start: int
    end: int
    observed_modular_delta: int
    rising_edges: int
    reset_edges: int
    held_edges: int
    wraps: int
    unit_increments: int | None


@dataclass(frozen=True)
class CounterIntervalObservation:
    """Data only; successful local evaluation never constructs a timer owner."""

    source_sha256: str
    selection: CounterEndpointSelection
    input_ports: tuple[ScalarPort, ...]
    output_ports: tuple[ScalarPort, ...]
    counter_width: int
    reset_present: bool
    samples: tuple[CounterPhaseSample, ...]
    transitions: tuple[CounterTransition, ...]
    intervals: tuple[ObservedCounterInterval, ...]
    unknowns: tuple[str, ...] = _UNKNOWN


@dataclass(frozen=True)
class _Prepared:
    evaluator: PreparedCombinationalObservation
    inputs: tuple[ScalarPort, ...]
    outputs: tuple[ScalarPort, ...]
    clock_index: int
    getter_index: int
    next_index: int
    reset_index: int | None
    reset_value_index: int | None
    output_indices: tuple[int, ...]
    width: int


def _prepare(text, selection, limits):
    bound = limits.expressions
    if type(text) is not str or len(text) > bound.source_bytes:
        raise ValueError("Counter source exceeds its parse budget.")
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
        raise ValueError("Counter source grammar is unsupported.") from None
    source_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if source_sha != selection.source_sha256:
        raise ValueError("Counter source bytes differ from the selection.")
    matches = [
        op
        for op in parsed.walk()
        if _name(op) == "hw.module" and _attribute(op, "sym_name") == StringAttr(selection.module)
    ]
    if len(matches) != 1:
        raise ValueError("Counter module membership is incomplete.")
    module = matches[0]
    if (set(module.attributes) | set(module.properties)) - {
        "op_name__",
        "sym_name",
        "module_type",
        "parameters",
    } or set(module.attributes) & set(module.properties):
        raise ValueError("Counter module semantics are unsupported.")
    if len(module.regions) != 1 or len(module.regions[0].blocks) != 1:
        raise ValueError("Counter module structure is unsupported.")
    parameters = _attribute(module, "parameters")
    if parameters is not None and (not isinstance(parameters, ArrayAttr) or parameters.data):
        raise ValueError("Counter module parameters are unresolved.")
    block = module.regions[0].block
    children = tuple(block.ops)
    if len(children) + len(block.args) > bound.nodes:
        raise ValueError("Counter source exceeds its node budget.")
    output = block.last_op
    if (
        output is None
        or _name(output) != "hw.output"
        or output.results
        or output.regions
        or (set(output.attributes) | set(output.properties)) - {"op_name__"}
    ):
        raise ValueError("Counter output membership is incomplete.")
    inputs, outputs = _ports(module, block, output, bound)
    if len(inputs) + len(outputs) + len(children) > bound.nodes:
        raise ValueError("Counter port and source roster exceeds its node budget.")
    selected = [op for op in children if _name(op) == "seq.firreg"]
    if (
        len(selected) != 1
        or selection.register_ordinal >= len(children)
        or children[selection.register_ordinal] is not selected[0]
    ):
        raise ValueError("Counter register membership differs from the selection.")
    register = selected[0]
    if (
        register.regions
        or len(register.results) != 1
        or len(register.operands) not in {2, 4}
        or (set(register.attributes) | set(register.properties)) - {"op_name__", "name"}
        or set(register.attributes) & set(register.properties)
        or not isinstance(_attribute(register, "name"), StringAttr)
    ):
        raise ValueError("Counter register semantics are unsupported.")
    current = register.results[0]
    width = _width(current, bound)
    if register.operands[0].type != current.type or str(register.operands[1].type) != "!seq.clock":
        raise ValueError("Counter register operand types are inconsistent.")
    if len(register.operands) == 4 and (
        register.operands[2].type != IntegerType(1) or register.operands[3].type != current.type
    ):
        raise ValueError("Counter reset operand types are inconsistent.")
    clocks = [op for op in children if _name(op) == "seq.to_clock"]
    if len(clocks) != 1:
        raise ValueError("Counter clock expression is unsupported.")
    clock = clocks[0]
    input_names = tuple(port.name for port in inputs)
    output_names = tuple(port.name for port in outputs)
    if selection.clock_input not in input_names or selection.getter_output not in output_names:
        raise ValueError("Counter endpoint membership is incomplete.")
    clock_index = input_names.index(selection.clock_input)
    getter_index = output_names.index(selection.getter_output)
    if (
        clock.regions
        or len(clock.operands) != 1
        or len(clock.results) != 1
        or (set(clock.attributes) | set(clock.properties)) - {"op_name__"}
        or set(clock.attributes) & set(clock.properties)
        or clock.operands[0] is not block.args[clock_index]
        or clock.operands[0].type != IntegerType(1)
        or clock.results[0] is not register.operands[1]
        or output.operands[getter_index] is not current
    ):
        raise ValueError("Counter clock or getter relation differs from the original source.")
    state_name = "$counter_state"
    if state_name in input_names:
        raise ValueError("Counter input identity is unsupported.")
    indices = {value: index for index, value in enumerate((*block.args, current))}
    depths = {value: 0 for value in indices}
    expressions, visiting = [], set()
    work = sum(port.width for port in (*inputs, *outputs)) + width

    def trace(value):
        nonlocal work
        if value in indices:
            return indices[value]
        if value in visiting:
            raise ValueError("Counter expression dependencies are cyclic.")
        if len(visiting) >= 64:
            raise ValueError("Counter expression depth exceeds its budget.")
        op = value.owner
        if op not in children or op in {register, clock, output} or op.regions or len(op.results) != 1:
            raise ValueError("Counter expression semantics are unsupported.")
        visiting.add(value)
        bits = _width(value, bound)
        operand_widths = [_width(operand, bound) for operand in op.operands]
        try:
            kind, parameter = _expression(op, operand_widths, bits, conditional_logic=True)
        except ValueError:
            raise ValueError("Counter expression semantics are unsupported.") from None
        operands = tuple(trace(operand) for operand in op.operands)
        depth = 1 + max((depths[operand] for operand in op.operands), default=0)
        if depth > 64:
            raise ValueError("Counter expression depth exceeds its budget.")
        work += bits + sum(operand_widths)
        if work > bound.bit_work:
            raise ValueError("Counter expression work exceeds its budget.")
        indices[value] = len(indices)
        depths[value] = depth
        expressions.append(_Expression(kind, bits, operands, parameter))
        visiting.remove(value)
        return indices[value]

    # Every original expression is checked, including unused source operations.
    for op in children:
        if op not in {register, clock, output}:
            if op.regions or len(op.results) != 1:
                raise ValueError("Counter expression semantics are unsupported.")
            trace(op.results[0])
    next_index = trace(register.operands[0])
    output_indices = tuple(trace(value) for value in output.operands)
    reset_index = reset_value_index = None
    if len(register.operands) == 4:
        reset_index, reset_value_index = (trace(value) for value in register.operands[2:])
        value = register.operands[3]
        if value in block.args or _name(value.owner) != "hw.constant":
            raise ValueError("Counter reset value semantics are unsupported.")

    state_dependencies = {}

    def contains_state(value):
        if value is current:
            return True
        if value in block.args:
            return False
        if value not in state_dependencies:
            state_dependencies[value] = any(contains_state(operand) for operand in value.owner.operands)
        return state_dependencies[value]

    if reset_index is not None and contains_state(register.operands[2]):
        raise ValueError("Counter reset dependencies are unsupported.")

    update_kinds = {}

    def unit_update(value):
        if value is current:
            return False
        if value in update_kinds:
            return update_kinds[value]
        if value in block.args:
            raise ValueError("Counter update is not a supported guarded unit increment.")
        op = value.owner
        if _name(op) == "comb.add" and len(op.operands) == 2:
            other = next((operand for operand in op.operands if operand is not current), None)
            if (
                current in op.operands
                and other is not None
                and other not in block.args
                and _name(other.owner) == "hw.constant"
                and (_integer(other.owner, "value") & ((1 << width) - 1)) == 1
            ):
                update_kinds[value] = True
                return True
        if _name(op) == "comb.mux" and not contains_state(op.operands[0]):
            left, right = (unit_update(operand) for operand in op.operands[1:])
            update_kinds[value] = left or right
            return update_kinds[value]
        raise ValueError("Counter update is not a supported guarded unit increment.")

    if not unit_update(register.operands[0]):
        raise ValueError("Counter update contains no unit increment.")
    refs = (next_index, *([reset_index, reset_value_index] if reset_index is not None else []), *output_indices)
    evaluator = PreparedCombinationalObservation(
        source_sha,
        selection.module,
        (*inputs, ScalarPort(state_name, width)),
        tuple(
            ScalarPort(str(index), _width((*block.args, current)[index], bound))
            if index < len(inputs) + 1
            else ScalarPort(str(index), expressions[index - len(inputs) - 1].width)
            for index in refs
        ),
        tuple(expressions),
        tuple(refs),
        work,
        bound,
    )
    return _Prepared(
        evaluator,
        inputs,
        outputs,
        clock_index,
        getter_index,
        next_index,
        reset_index,
        reset_value_index,
        output_indices,
        width,
    )


def observe_counter_intervals(
    text: str,
    *,
    selection: CounterEndpointSelection,
    limits: CounterIntervalLimits,
    expected_phases: int,
    samples: tuple[CounterPhaseSample, ...],
    intervals: tuple[CounterInterval, ...],
) -> CounterIntervalObservation:
    """Reparse source and check every post-phase value before interval arithmetic.

    Rows begin LOW, alternate LOW/HIGH, and retain every original input/output.
    Nonclock inputs must be stable from LOW to HIGH. The first observed register
    value is a conditional boundary, not a proved initialized/reachable state.
    Interval arithmetic is unknown across reset and never assigns clock units.
    """
    if type(selection) is not CounterEndpointSelection or type(limits) is not CounterIntervalLimits:
        raise ValueError("Counter observation selection is unavailable.")
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
        raise ValueError("Counter phase or interval roster is incomplete.")
    prepared = _prepare(text, selection, limits)
    if expected_phases * 2 * prepared.evaluator.per_case_bit_work > limits.expressions.bit_work:
        raise ValueError("Counter timeline exceeds its bit-work budget.")
    for ordinal, sample in enumerate(samples):
        if type(sample) is not CounterPhaseSample or type(sample.ordinal) is not int or sample.ordinal != ordinal:
            raise ValueError("Counter phase membership is incomplete.")
        for values, ports in ((sample.inputs, prepared.inputs), (sample.outputs, prepared.outputs)):
            if (
                type(values) is not tuple
                or len(values) != len(ports)
                or any(
                    type(value) is not int or not 0 <= value < 1 << port.width
                    for value, port in zip(values, ports, strict=True)
                )
            ):
                raise ValueError("Counter sample differs from the complete original typed roster.")
        if sample.inputs[prepared.clock_index] != ordinal % 2:
            raise ValueError("Counter clock phases are incomplete.")
        if ordinal % 2 and any(
            value != samples[ordinal - 1].inputs[index]
            for index, value in enumerate(sample.inputs)
            if index != prepared.clock_index
        ):
            raise ValueError("Counter edge inputs are unstable.")
    for interval in intervals:
        if (
            type(interval) is not CounterInterval
            or type(interval.start) is not int
            or type(interval.end) is not int
            or not 0 <= interval.start < interval.end < expected_phases
        ):
            raise ValueError("Counter interval endpoints are unavailable.")

    def evaluate(sample, state):
        values = dict(zip((port.name for port in prepared.inputs), sample.inputs, strict=True))
        values[prepared.evaluator.inputs[-1].name] = state
        return prepared.evaluator.evaluate((values,))[0]

    transitions, prefixes = [], [(0, 0, 0, 0, 0)]
    state = samples[0].outputs[prepared.getter_index]
    domain = 1 << prepared.width
    for ordinal, sample in enumerate(samples):
        prior = state
        values = evaluate(sample, state)
        edge, reset, increment, wrap = bool(ordinal % 2), False, False, False
        if ordinal and edge:
            reset = prepared.reset_index is not None and bool(values[str(prepared.reset_index)])
            state = values[str(prepared.reset_value_index if reset else prepared.next_index)]
            increment = not reset and state != prior
            if not reset and state not in {prior, (prior + 1) % domain}:
                raise ValueError("Counter transition differs from the derived unit update.")
            wrap = increment and state < prior
            values = evaluate(sample, state)
        observed = tuple(values[str(index)] for index in prepared.output_indices)
        if observed != sample.outputs:
            raise ValueError("Counter timeline differs from the actual source transition.")
        transitions.append(CounterTransition(ordinal, edge, reset, increment, prior, state, wrap))
        before = prefixes[-1]
        prefixes.append(
            tuple(
                left + int(right)
                for left, right in zip(
                    before, (edge, reset, edge and not reset and not increment, wrap, increment), strict=True
                )
            )
        )
    results = []
    for interval in intervals:
        counts = tuple(
            right - left for left, right in zip(prefixes[interval.start + 1], prefixes[interval.end + 1], strict=True)
        )
        first, last = (samples[index].outputs[prepared.getter_index] for index in (interval.start, interval.end))
        results.append(
            ObservedCounterInterval(
                interval.start, interval.end, (last - first) % domain, *counts[:4], None if counts[1] else counts[4]
            )
        )
    return CounterIntervalObservation(
        selection.source_sha256,
        selection,
        prepared.inputs,
        prepared.outputs,
        prepared.width,
        prepared.reset_index is not None,
        samples,
        tuple(transitions),
        tuple(results),
    )
