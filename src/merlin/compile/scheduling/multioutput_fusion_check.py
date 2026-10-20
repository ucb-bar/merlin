"""Reconstruct one private producer and both actual physical consumers.

This checks an original-qualified composition, not a capability declaration.
Every instruction still has the existing single-result scalar/effect semantics.
The shared producer's physical byte generation, work and every output are
checked together before any final-output projection.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from math import prod

from merlin.common.digest import sha256_file
from merlin.sched.ir import Kernel, TensorArg

from .compact_dense_group import _callback, _same_callback
from .configuration import verify_configuration_state
from .fusion import _elementwise
from .fusion_semantics import UnsupportedElementwise
from .program_limits import AddressWindow
from .spill_conformance import _Memory, _Piece
from .temporary_storage import StorageAdapter, _problem


@dataclass(frozen=True)
class MultiOutputCompositionLaw:
    """Original execution/removal premises for a closed instruction composition.

    Identifiers do not establish these premises. The original caller selects
    qualified implementations and exact decoded instruction/semantic-key roles.
    The checker separately proves physical retention and source correspondence.
    """

    configuration_identity: str
    numerical_policy_identity: str
    adapter_qualification_identity: str
    qualification_identity: str
    producer: tuple[str, str]
    epilogues: tuple[tuple[str, str], tuple[str, str]]
    complete_projected_effects: bool = field(kw_only=True)
    synchronous_private_composition: bool = field(kw_only=True)
    suppressed_intermediate_transfers_no_faults: bool = field(kw_only=True)
    suppressed_intermediate_transfers_unobserved: bool = field(kw_only=True)

    def __post_init__(self):
        identities = (
            self.configuration_identity,
            self.numerical_policy_identity,
            self.adapter_qualification_identity,
            self.qualification_identity,
        )
        if any(type(value) is not str or not value.strip() for value in identities):
            raise ValueError("multi-output composition requires original nonempty qualification identities")
        if (
            type(self.epilogues) is not tuple
            or len(self.epilogues) != 2
            or any(
                type(role) is not tuple or len(role) != 2 or any(type(value) is not str or not value for value in role)
                for role in (self.producer, *self.epilogues)
            )
            or any(type(value) is not bool for value in self.flags())
        ):
            raise ValueError("multi-output law requires exact instruction/semantic roles and explicit Boolean premises")

    def flags(self):
        return (
            self.complete_projected_effects,
            self.synchronous_private_composition,
            self.suppressed_intermediate_transfers_no_faults,
            self.suppressed_intermediate_transfers_unobserved,
        )

    def admits(self, adapter):
        self.__post_init__()
        if type(adapter) is not StorageAdapter or (
            self.configuration_identity,
            self.numerical_policy_identity,
            self.adapter_qualification_identity,
        ) != (adapter.configuration_identity, adapter.numerical_policy_identity, adapter.qualification_identity):
            raise ValueError("multi-output law differs from the original selected adapter")
        if not all(self.flags()):
            raise UnsupportedElementwise(
                "multi-output execution/removal law is unavailable; retain original dispatches"
            )


@dataclass(frozen=True)
class MultiOutputBindings:
    inputs: tuple[AddressWindow, ...]
    outputs: tuple[AddressWindow, AddressWindow]
    byte_orders: tuple[str, ...]
    external_buffers_disjoint: bool

    def __post_init__(self):
        if (
            type(self.inputs) is not tuple
            or not self.inputs
            or type(self.outputs) is not tuple
            or len(self.outputs) != 2
            or any(type(window) is not AddressWindow for window in (*self.inputs, *self.outputs))
            or type(self.byte_orders) is not tuple
            or len(self.byte_orders) != len(self.inputs) + 2
            or any(order not in ("little", "big") for order in self.byte_orders)
            or type(self.external_buffers_disjoint) is not bool
        ):
            raise ValueError("multi-output bindings require original ordered typed windows and explicit byte layouts")


def _state(adapter, configuration_adapter):
    callbacks = (adapter.project, adapter.rewrite)
    if configuration_adapter is not None:
        callbacks = (*callbacks, configuration_adapter.project)
    return (
        (adapter.configuration_identity, adapter.numerical_policy_identity, adapter.qualification_identity),
        tuple(_callback(function) for function in callbacks),
        tuple((row[3], sha256_file(row[3])) for row in map(_callback, callbacks)),
    )


def _current(adapter, configuration_adapter, snapshot):
    current = _state(adapter, configuration_adapter)
    callbacks = (adapter.project, adapter.rewrite)
    if configuration_adapter is not None:
        callbacks = (*callbacks, configuration_adapter.project)
    if (
        any(path.is_symlink() or not path.is_file() or sha256_file(path) != digest for path, digest in snapshot[2])
        or current[0] != snapshot[0]
        or any(not _same_callback(function, frozen) for function, frozen in zip(callbacks, snapshot[1], strict=True))
    ):
        raise ValueError("multi-output original projection identity or code changed")


def _source_state(source):
    return (
        source.shape,
        asdict(source.producer),
        tuple(asdict(stage) for stage in source.epilogues),
        source.inputs,
        source.outputs,
        source.input_indices,
        tuple(str(value.type) for value in (*source.inputs, *source.outputs)),
    )


def _arguments(source):
    return (
        *(
            TensorArg(f"input{i}", source.shape, str(value.type.element_type), "read")
            for i, value in enumerate(source.inputs)
        ),
        *(
            TensorArg(f"output{i}", source.shape, str(value.type.element_type), "write")
            for i, value in enumerate(source.outputs)
        ),
    )


def _checker_functions():
    return (
        verify_multioutput_fusion,
        _state,
        _current,
        _source_state,
        _arguments,
        _checker_functions,
        _callback,
        _same_callback,
        _elementwise,
        _problem,
        _Memory.read,
        _Memory.write,
        _Memory.__init__,
        MultiOutputCompositionLaw.__post_init__,
        MultiOutputCompositionLaw.flags,
        MultiOutputCompositionLaw.admits,
        MultiOutputBindings.__post_init__,
    )


def verify_multioutput_fusion(
    source,
    kernel,
    objects,
    banks,
    adapter,
    bindings,
    composition_law,
    *,
    external_windows,
    private_ownership,
    budgets,
    configuration_adapter=None,
    role_adapter=None,
    virtual_private_addresses=False,
):
    """Read the actual whole instruction program, never a claimed fusion plan.

    The first domain uses synchronous single-result arithmetic, private P and
    direct final epilogue writes. Exact input-staging copies may have explicit
    waits. P cannot migrate to a copied backing between its two consumers.
    """
    if type(bindings) is not MultiOutputBindings or type(composition_law) is not MultiOutputCompositionLaw:
        raise ValueError("multi-output checking requires original typed bindings and law")
    bindings.__post_init__()
    role_adapter = adapter if role_adapter is None else role_adapter
    composition_law.admits(role_adapter)
    if not bindings.external_buffers_disjoint:
        raise UnsupportedElementwise("multi-output external ownership is unavailable")
    if type(kernel) is not Kernel or kernel.args != _arguments(source):
        raise ValueError("multi-output kernel argument geometry differs from the complete original source")
    same_callback, same_code = _same_callback, _same_callback.__code__
    function_reader, function_code = _checker_functions, _checker_functions.__code__
    functions = tuple(_callback(function) for function in function_reader())
    original_kernel, original_law = kernel.digest(), asdict(composition_law)
    snapshot = _state(adapter, configuration_adapter)
    role_snapshot = _state(role_adapter, None)
    source_state = _source_state(source)

    def current():
        if (
            _same_callback is not same_callback
            or same_callback.__code__ is not same_code
            or _checker_functions is not function_reader
            or function_reader.__code__ is not function_code
            or any(
                not same_callback(function, frozen)
                for function, frozen in zip(function_reader(), functions, strict=True)
            )
        ):
            raise ValueError("multi-output checker implementation function or code changed")
        _current(adapter, configuration_adapter, snapshot)
        _current(role_adapter, None, role_snapshot)
        if (
            kernel.digest() != original_kernel
            or asdict(composition_law) != original_law
            or _source_state(source) != source_state
        ):
            raise ValueError("multi-output original kernel or law changed during projection")

    original_project = adapter.project

    def project(call):
        current()
        effect = original_project(call)
        current()
        return effect

    checked_adapter = replace(adapter, project=project)
    bank_map, effects, *_ = _problem(
        kernel,
        objects,
        banks,
        checked_adapter,
        external_windows,
        private_ownership,
        budgets.max_calls,
        budgets.max_objects,
    )
    current()
    if type(virtual_private_addresses) is not bool:
        raise ValueError("multi-output address phase must be an explicit Boolean")
    if not virtual_private_addresses and any(
        access.space in bank_map
        and (
            access.start < bank_map[access.space].start or access.start + access.size > bank_map[access.space].capacity
        )
        for effect in effects
        for access in effect.accesses
    ):
        raise ValueError("multi-output actual private access exceeds its physical bank bounds")
    if configuration_adapter is not None:
        verify_configuration_state(kernel, configuration_adapter, max_calls=budgets.max_calls)
        current()
    count = prod(source.shape)
    stages = (source.producer, *source.epilogues)
    widths = tuple(int(value.type.element_type.width.data) for value in (*source.inputs, *source.outputs))
    windows = (*bindings.inputs, *bindings.outputs)
    if len(bindings.inputs) != len(source.inputs) or any(
        window.size != count * bits // 8 for window, bits in zip(windows, widths, strict=True)
    ):
        raise ValueError("multi-output external bindings change complete logical extents")
    if tuple(window.space for window in windows) != tuple(argument.name for argument in kernel.args):
        raise ValueError("multi-output external windows must identify actual canonical tensor arguments")
    fixed = {window.space: window for window in external_windows}
    for index, window in enumerate(windows):
        bound = fixed.get(window.space)
        if bound is None or window.start < bound.start or window.start + window.size > bound.start + bound.size:
            raise ValueError("multi-output binding exceeds an original selected external window")
        if any(
            window.space == earlier.space
            and window.start < earlier.start + earlier.size
            and earlier.start < window.start + window.size
            for earlier in windows[:index]
        ):
            raise UnsupportedElementwise("multi-output shifted or output aliases are outside this composition")
    memory = _Memory()

    def key(access):
        return ("private", bank_map[access.space].backing) if access.space in bank_map else ("external", access.space)

    for index, window in enumerate(bindings.inputs):
        memory.write(key(window), window.start, window.size, (_Piece(("input", index), 0, window.size),))
    coverage = [[], [], []]
    generations, pending = {}, {}

    def input_value(access, input_index, bits, order, elements):
        parts = memory.read(key(access), access.start, access.size)
        if len(parts) != 1:
            raise ValueError("multi-output arithmetic reads partial or overwritten provenance")
        part = parts[0]
        if input_index >= 0:
            width = bits // 8
            if (
                part.identity != ("input", input_index)
                or part.offset % width
                or order != bindings.byte_orders[input_index]
            ):
                raise ValueError("multi-output instruction reads the wrong original input or byte layout")
            return part.offset // width, None
        if part.identity[0] != "producer" or part.offset != 0:
            raise ValueError("epilogue does not read one complete actual producer generation")
        generation = generations[part.identity[1]]
        if (key(access), access.start, access.size) != generation[1]:
            raise ValueError("epilogues do not retain the same actual physical producer bytes")
        if generation[2] != elements or order != generation[3]:
            raise ValueError("epilogue changes its producer's complete fragment")
        return generation[0], part.identity[1]

    for ordinal, (call, effect) in enumerate(zip(kernel.body, effects, strict=True)):
        current()
        if effect.completion == "wait":
            for token in effect.waits:
                destination, parts = pending.pop(token)
                memory.write(key(destination), destination.start, destination.size, parts)
            continue
        if effect.exact_copy:
            read, write = effect.accesses
            # Only original input staging is admitted; copying P cannot stand
            # in for its qualified retained physical generation.
            parts = memory.read(key(read), read.start, read.size)
            if write.space not in bank_map or any(part.identity[0] != "input" for part in parts):
                raise UnsupportedElementwise("multi-output copies may only stage unchanged original inputs")
            if effect.completion == "asynchronous":
                pending[effect.token] = write, parts
            else:
                memory.write(key(write), write.start, write.size, parts)
            continue
        if not effect.accesses and configuration_adapter is not None:
            state = configuration_adapter.project(call)
            current()
            if state is not None and state.pure_write:
                continue
        if effect.completion != "synchronous":
            raise UnsupportedElementwise("multi-output arithmetic requires synchronous completion")
        graph, reads, write = _elementwise(effect, bank_map, budgets.max_scalar_operations, budgets.max_scalar_ir_bytes)
        role = 0
        if write.space not in bank_map:
            matches = [
                i
                for i, window in enumerate(bindings.outputs)
                if write.space == window.space
                and window.start <= write.start
                and write.start + write.size <= window.start + window.size
            ]
            if len(matches) != 1:
                raise ValueError("multi-output instruction has an undeclared external write")
            role = matches[0] + 1
        declared = composition_law.producer if role == 0 else composition_law.epilogues[role - 1]
        raw = effect if role_adapter is adapter else role_adapter.project(call)
        current()
        if (
            raw is None
            or replace(effect, semantic_key=raw.semantic_key, elementwise=raw.elementwise) != raw
            or (call.instr, raw.semantic_key) != declared
            or graph != stages[role].graph
        ):
            raise ValueError("multi-output actual instruction changes its qualified source scalar role")
        indices = source.input_indices[role]
        if len(reads) != len(indices):
            raise ValueError("multi-output actual inputs differ from the original scalar stage")
        elements = effect.elementwise.elements
        logical = [
            input_value(access, input_index, bits, order, elements)
            for access, input_index, bits, order in zip(
                reads, indices, graph.input_bits, effect.elementwise.byte_orders[:-1], strict=True
            )
        ]
        if not logical or any(start != logical[0][0] for start, _ in logical):
            raise ValueError("multi-output instruction changes logical element coordinates")
        start = logical[0][0]
        if not 0 <= start < start + elements <= count:
            raise ValueError("multi-output instruction exceeds its complete logical domain")
        coverage[role].append((start, start + elements))
        if role == 0:
            if any(generation is not None for _, generation in logical):
                raise ValueError("multi-output producer reads another producer generation")
            generations[ordinal] = (
                start,
                (key(write), write.start, write.size),
                elements,
                effect.elementwise.byte_orders[-1],
            )
            parts = (_Piece(("producer", ordinal), 0, write.size),)
        else:
            generation_ids = [generation for _, generation in logical if generation is not None]
            if len(generation_ids) != 1:
                raise ValueError("multi-output epilogue must consume its exact common producer once")
            window = bindings.outputs[role - 1]
            bits = graph.output_bits // 8
            if (
                write.start != window.start + start * bits
                or effect.elementwise.byte_orders[-1] != bindings.byte_orders[len(bindings.inputs) + role - 1]
            ):
                raise ValueError("multi-output epilogue writes the wrong logical output coordinates")
            parts = (_Piece(("output", role, start, elements, generation_ids[0]), 0, write.size),)
        memory.write(key(write), write.start, write.size, parts)
    if pending:
        raise ValueError("multi-output input movement remains incomplete")
    for intervals in coverage:
        cursor = 0
        for start, end in sorted(intervals):
            if start != cursor:
                raise ValueError("multi-output producer/epilogue duplicated or omitted logical work")
            cursor = end
        if cursor != count:
            raise ValueError("multi-output producer/epilogue does not cover the complete source")
    for role, window in enumerate(bindings.outputs, 1):
        cursor = 0
        bits = stages[role].graph.output_bits // 8
        for part in memory.read(key(window), window.start, window.size):
            identity = part.identity
            if identity[0] != "output" or identity[1] != role or part.offset or identity[2] != cursor:
                raise ValueError("multi-output final result is undefined, overwritten or omitted")
            cursor += identity[3]
            if part.size != identity[3] * bits:
                raise ValueError("multi-output final result has an incomplete typed extent")
        if cursor != count:
            raise ValueError("multi-output final result omits source elements")
    current()
    return dict(
        schema="merlin.physical_multioutput_fusion.v1",
        source_shape=list(source.shape),
        law=asdict(composition_law),
        producer_evaluations=count,
        epilogue_evaluations=[count, count],
        producer_instructions=len(coverage[0]),
        epilogue_instructions=[len(c) for c in coverage[1:]],
        actual_commands=len(effects),
        output_elements=[count, count],
        requested_read_bytes=sum(access.size for effect in effects for access in effect.accesses if not access.writes),
        requested_write_bytes=sum(access.size for effect in effects for access in effect.accesses if access.writes),
        actual_schedule_sha256=kernel.digest(),
        address_scope="virtual_private_source" if virtual_private_addresses else "actual_physical",
        timing="UNKNOWN",
    )
