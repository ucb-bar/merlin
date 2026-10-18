"""Check source pointwise arithmetic through allocation, fusion and encoding.

Targets supply the original instruction sequence and qualified implementations.
Merlin reads current typed source, checks that sequence, runs the existing
scheduler, then independently reconstructs the final bytes' scalar dataflow.
This bounded family admits complete contiguous integer vectors, not opaque
backend output, floating reassociation, broadcasts or asynchronous arithmetic.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from math import prod

from xdsl.dialects import builtin, func
from xdsl.dialects.linalg.ops import YieldOp
from xdsl.ir import Block, Region

from merlin.sched.ir import Kernel
from merlin.transforms.certificates import ir_digest, ir_text
from merlin.transforms.dense_ir import checked_static_linalg_shell
from merlin.transforms.pipeline_state import PassSpec, PipelineContext, PipelineSession, validate_pass_order

from .configuration import verify_configuration_state
from .fusion_semantics import ScalarBudgetExceeded, ScalarGraph, UnsupportedElementwise, scalar_graph, term
from .library_sources import shared_library_sources
from .program_limits import AddressWindow
from .schedule_pipeline import ScheduleBudgets, ScheduleCompilation, compile_schedule
from .selected_contracts import SelectedSchedulingContracts
from .spill_conformance import _Memory, _Piece
from .temporary_storage import _project


@dataclass(frozen=True)
class PointwiseSource:
    shape: tuple[int, ...]
    scalar_ir: bytes
    graph: ScalarGraph


def read_pointwise_source(operation, *, max_elements, max_operations, max_ir_bytes):
    """Retain every typed scalar operation, constant and conversion in order."""
    if type(max_elements) is not int or max_elements <= 0:
        raise ValueError("pointwise source requires a positive logical element budget")
    if type(max_operations) is not int or max_operations <= 0 or type(max_ir_bytes) is not int or max_ir_bytes <= 0:
        raise ValueError("pointwise source requires positive scalar operation and byte budgets")
    try:
        shell = checked_static_linalg_shell(operation)
    except ValueError as error:
        raise UnsupportedElementwise(str(error)) from error
    if any(d <= 0 for d in shell.shape) or prod(shell.shape) * len(shell.args) > max_elements:
        raise UnsupportedElementwise("pointwise source has empty extents or exceeds its logical element budget")
    body = shell.body
    if len(body) > max_operations + 1:
        raise ScalarBudgetExceeded("pointwise scalar source exceeds its selected operation budget")
    if (
        not body
        or type(body[-1]) is not YieldOp
        or len(body[-1].operands) != 1
        or body[-1].attributes
        or body[-1].properties
    ):
        raise UnsupportedElementwise("pointwise source needs one plain explicit scalar yield")
    block = Block(arg_types=[arg.type for arg in shell.args[:-1]])
    values = dict(zip(shell.args[:-1], block.args, strict=True))
    for operation in body[:-1]:
        if operation.regions or operation.successors or any(arg not in values for arg in operation.operands):
            raise UnsupportedElementwise("pointwise source reads its initializer, captured state or an unknown region")
        clone = operation.clone(value_mapper=values)
        block.add_op(clone)
    returned = body[-1].operands[0]
    if returned not in values:
        raise UnsupportedElementwise("pointwise source yields its initializer or an external value")
    block.add_op(func.ReturnOp(values[returned]))
    function = func.FuncOp.from_region("source", [arg.type for arg in block.args], [returned.type], Region(block))
    module = builtin.ModuleOp([function])
    module.verify()
    source = ir_text(module).encode()
    return PointwiseSource(shell.shape, source, scalar_graph(source, max_operations, max_ir_bytes))


@dataclass(frozen=True)
class PointwiseBindings:
    """Original runtime's contiguous tensor bindings, in source operand order.

    Distinct external spaces must be physically disjoint, as selected storage
    declares. Exact same-window input aliases are allowed; shifted aliases and
    input/output overlap are refused. This premise needs runtime qualification,
    not deduction from different tensor SSA names. Byte order is explicit for
    every input followed by the result. All buffers are borrowed for one call.
    """

    inputs: tuple[AddressWindow, ...]
    output: AddressWindow
    byte_orders: tuple[str, ...]
    external_buffers_disjoint: bool

    def __post_init__(self):
        if (
            type(self.inputs) is not tuple
            or any(type(w) is not AddressWindow for w in self.inputs)
            or type(self.output) is not AddressWindow
            or type(self.byte_orders) is not tuple
            or len(self.byte_orders) != len(self.inputs) + 1
            or any(order not in ("little", "big") for order in self.byte_orders)
            or type(self.external_buffers_disjoint) is not bool
        ):
            raise ValueError("pointwise bindings require typed windows, byte layouts and an explicit alias premise")


def _verify_vectors(source, kernel, bindings, banks, external, adapter, budgets, configuration_adapter=None):
    """Reconstruct whole-vector expressions using the existing interval memory."""
    if not bindings.external_buffers_disjoint:
        raise UnsupportedElementwise("pointwise runtime has not established distinct external backing ownership")
    count = prod(source.shape)
    graph = source.graph
    windows = (*bindings.inputs, bindings.output)
    widths = (*graph.input_bits, graph.output_bits)
    if len(windows) != len(widths) or any(w.size != count * bits // 8 for w, bits in zip(windows, widths)):
        raise ValueError("pointwise bindings differ from the source's complete typed tensor extents")
    bank_map = {bank.name: bank for bank in banks}
    external_map = {window.space: window for window in external}

    def key(access):
        return ("private", bank_map[access.space].backing) if access.space in bank_map else ("external", access.space)

    def overlaps(left, right):
        return (
            key(left) == key(right) and left.start < right.start + right.size and right.start < left.start + left.size
        )

    for i, window in enumerate(windows):
        bound = external_map.get(window.space)
        if bound is None or window.start < bound.start or window.start + window.size > bound.start + bound.size:
            raise ValueError("pointwise source binding exceeds the selected external window")
        for j, earlier in enumerate(windows[:i]):
            if overlaps(window, earlier) and (
                i == len(windows) - 1
                or window != earlier
                or (widths[i], bindings.byte_orders[i]) != (widths[j], bindings.byte_orders[j])
            ):
                raise UnsupportedElementwise("pointwise source has shifted, reinterpreted or output aliases")

    memory, arguments = _Memory(), []
    for window, bits, order in zip(bindings.inputs, graph.input_bits, bindings.byte_orders):
        expression = term(("source-vector", key(window), window.start, window.size, bits, order))
        identity = ("scalar-vector", expression, bits, order)
        memory.write(key(window), window.start, window.size, (_Piece(identity, 0, window.size),))
        arguments.append(expression)

    def read(access, bits=None, order=None):
        parts = memory.read(key(access), access.start, access.size)
        if (
            len(parts) != 1
            or parts[0].offset != 0
            or parts[0].size != access.size
            or parts[0].identity[0] != "scalar-vector"
        ):
            raise ValueError("pointwise instruction reads undefined, partial or overwritten vector data")
        part = parts[0]
        if bits is not None and (part.identity[2:] != (bits, order) or access.size != count * bits // 8):
            raise ValueError("pointwise instruction changes a source vector's width, extent or byte layout")
        return part

    effects = _project(kernel, adapter, budgets.max_calls)
    state = None
    if configuration_adapter is not None:
        state = verify_configuration_state(kernel, configuration_adapter, max_calls=budgets.max_calls)
    for call, effect in zip(kernel.body, effects, strict=True):
        if effect.completion != "synchronous":
            raise UnsupportedElementwise("source pointwise arithmetic requires qualified synchronous completion")
        accesses = effect.accesses
        if not accesses and configuration_adapter is not None:
            configuration = configuration_adapter.project(call)
            if configuration is not None and configuration.pure_write:
                continue  # Selected ISA proves only persistent idempotent writes.
        for access in accesses:
            if access.space not in bank_map and (
                not access.writes
                and not any(
                    access.space == w.space
                    and w.start <= access.start
                    and access.start + access.size <= w.start + w.size
                    for w in bindings.inputs
                )
                or access.writes
                and (access.space, access.start, access.size)
                != (bindings.output.space, bindings.output.start, bindings.output.size)
            ):
                raise ValueError("pointwise program has an undeclared external read or state write")
        if not accesses or not accesses[-1].writes or any(a.writes for a in accesses[:-1]):
            raise UnsupportedElementwise("pointwise instructions need ordered inputs and one complete result")
        if effect.exact_copy:
            part = read(accesses[0])
        elif effect.elementwise is not None:
            semantic = effect.elementwise
            actual = scalar_graph(semantic.function_ir, budgets.max_scalar_operations, budgets.max_scalar_ir_bytes)
            if semantic.elements != count or len(accesses) != len(actual.input_bits) + 1:
                raise ValueError("pointwise instruction does not cover the complete source domain")
            inputs = [
                read(a, bits, order).identity[1]
                for a, bits, order in zip(accesses[:-1], actual.input_bits, semantic.byte_orders[:-1], strict=True)
            ]
            if accesses[-1].size != count * actual.output_bits // 8:
                raise ValueError("pointwise instruction result has an incorrect complete extent")
            part = _Piece(
                ("scalar-vector", actual.expression(inputs), actual.output_bits, semantic.byte_orders[-1]),
                0,
                accesses[-1].size,
            )
        else:
            raise UnsupportedElementwise("pointwise program contains unmodeled arithmetic or execution effects")
        output = accesses[-1]
        memory.write(key(output), output.start, output.size, (part,))
    returned = read(bindings.output, graph.output_bits, bindings.byte_orders[-1])
    if returned.identity[1] != graph.expression(arguments):
        raise ValueError("realized pointwise arithmetic differs from the complete typed source")
    return {
        "actual_commands": len(effects),
        "elements": count,
        "requested_read_bytes": sum(a.size for e in effects for a in e.accesses if not a.writes),
        "requested_write_bytes": sum(a.size for e in effects for a in e.accesses if a.writes),
        "source_scalar_sha256": sha256(source.scalar_ir).hexdigest(),
        "result_expression": returned.identity[1],
        "configuration": state,
    }


@dataclass(frozen=True)
class PointwiseCompilation:
    source: builtin.ModuleOp
    operation: object
    operation_ordinal: int
    source_sha256: str
    library_sources: tuple
    selected_contracts: SelectedSchedulingContracts
    compiled: ScheduleCompilation
    checked_images: tuple[bytes, ...]
    checked_schedule_sha256: str
    source_passes: tuple
    conformance: dict

    def revalidate(self):
        self.selected_contracts.revalidate()
        if (
            ir_digest(self.source) != self.source_sha256
            or tuple(sorted(shared_library_sources().items())) != self.library_sources
            or not any(i == self.operation_ordinal and op is self.operation for i, op in enumerate(self.source.walk()))
            or self.compiled.kernel.digest() != self.checked_schedule_sha256
            or tuple(self.compiled.program.segments) != self.checked_images
        ):
            raise ValueError("pointwise source, library, schedule or encoded program changed; replan")

    def images(self):
        """Return only source-checked final images while all inputs stay current."""
        self.revalidate()
        return self.checked_images


def compile_source_pointwise(
    module,
    operation,
    kernel,
    objects,
    adapter,
    encoding,
    limits,
    *,
    selected_contracts,
    bindings,
    budgets=ScheduleBudgets(),
    banks=None,
    reservations=(),
    private_ownership=False,
    fusion_emitters=None,
    fusion_qualification_identity=None,
    configuration_qualification_identity=None,
    max_source_elements=64 << 20,
    pass_order=(),
):
    """Use the ordinary scheduler with source admission and final-byte checking.

    UnsupportedElementwise leaves the caller's original source route available.
    An incorrect provider sequence raises ValueError before any result escapes.
    This entry neither invents instruction selection nor qualifies execution.
    """
    if (
        type(module) is not builtin.ModuleOp
        or type(selected_contracts) is not SelectedSchedulingContracts
        or selected_contracts.storage is None
        or selected_contracts.fusion is None
        or type(bindings) is not PointwiseBindings
        or type(budgets) is not ScheduleBudgets
        or not any(op is operation for op in module.walk())
    ):
        raise ValueError(
            "pointwise compilation requires current typed source, bindings and selected storage/scalar contracts"
        )
    module.verify()
    selected_contracts.revalidate()
    library = tuple(sorted(shared_library_sources().items()))
    context = PipelineContext(
        hardware=selected_contracts.hardware_identity,
        numerical_policy=selected_contracts.numerical_policy,
        execution=adapter.qualification_identity,
        library=term(library),
        resources=term(
            (
                asdict(bindings),
                asdict(budgets),
                max_source_elements,
                [asdict(b) for b in banks] if banks is not None else None,
                [asdict(r) for r in reservations],
                private_ownership,
                fusion_qualification_identity,
                configuration_qualification_identity,
            )
        ),
    )
    session = PipelineSession(module, context=context, ir_level="linalg", max_repairs=0)
    specs = (
        PassSpec("source-admission", 0, produces=("pointwise_source",), mutates_ir=False, ir_level="linalg"),
        PassSpec(
            "schedule-lowering",
            1,
            requires=("pointwise_source",),
            produces=("compiled",),
            mutates_ir=False,
            ir_level="linalg",
        ),
        PassSpec(
            "source-conformance",
            2,
            requires=("pointwise_source", "compiled"),
            produces=("conformance",),
            mutates_ir=False,
            ir_level="linalg",
        ),
    )
    by_name = {spec.name: spec for spec in specs}
    order = pass_order or tuple(by_name)
    if len(order) != len(by_name) or set(order) != set(by_name):
        raise ValueError("pointwise pass order must name every source pass exactly once")
    validate_pass_order(tuple(by_name[name] for name in order))
    configuration_adapter = (
        selected_contracts.configuration_adapter(configuration_qualification_identity)
        if configuration_qualification_identity is not None
        else None
    )
    physical_banks, fixed, external, projector = selected_contracts.storage.bind(
        selected_contracts.isa,
        adapter,
        banks=banks,
        reservations=reservations,
        external_windows=None,
        private_ownership=private_ownership,
        configuration_adapter=configuration_adapter,
    )
    projector, _, _ = selected_contracts.fusion.bind(
        projector,
        emitters=fusion_emitters,
        qualification_identity=fusion_qualification_identity,
        max_operations=budgets.max_scalar_operations,
        max_ir_bytes=budgets.max_scalar_ir_bytes,
    )

    def lower():
        source = session.analysis("pointwise_source")
        _verify_vectors(source, kernel, bindings, physical_banks, external, projector, budgets, configuration_adapter)
        return {
            "compiled": compile_schedule(
                kernel,
                objects,
                physical_banks,
                adapter,
                encoding,
                limits,
                budgets=budgets,
                reservations=fixed,
                external_windows=external,
                private_ownership=private_ownership,
                selected_contracts=selected_contracts,
                fusion_emitters=fusion_emitters,
                fusion_qualification_identity=fusion_qualification_identity,
                configuration_qualification_identity=configuration_qualification_identity,
            )
        }

    def final():
        compiled = session.analysis("compiled")
        calls = []
        for image in compiled.program.segments:
            decoded = encoding.decode_calls(image)
            selected_contracts.verify_image(image, decoded)
            calls.extend(decoded)
        actual = Kernel(kernel.name, kernel.args, tuple(calls), kernel.attrs)
        return {
            "conformance": _verify_vectors(
                session.analysis("pointwise_source"),
                actual,
                bindings,
                physical_banks,
                external,
                projector,
                budgets,
                configuration_adapter,
            )
        }

    actions = {
        "source-admission": lambda: {
            "pointwise_source": read_pointwise_source(
                operation,
                max_elements=max_source_elements,
                max_operations=budgets.max_scalar_operations,
                max_ir_bytes=budgets.max_scalar_ir_bytes,
            )
        },
        "schedule-lowering": lower,
        "source-conformance": final,
    }
    for name in order:
        selected_contracts.revalidate()
        if tuple(sorted(shared_library_sources().items())) != library:
            raise ValueError("pointwise shared library changed during compilation; replan")
        session.run(by_name[name], actions[name])
    compiled = session.analysis("compiled")
    result = PointwiseCompilation(
        module,
        operation,
        next(i for i, op in enumerate(module.walk()) if op is operation),
        session.stamp().source_sha256,
        library,
        selected_contracts,
        compiled,
        tuple(compiled.program.segments),
        compiled.kernel.digest(),
        tuple(session.records),
        session.analysis("conformance"),
    )
    result.revalidate()
    return result
