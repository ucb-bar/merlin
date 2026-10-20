"""Bind typed source matmul semantics to the existing compact dense scheduler.

Targets still supply physical storage assignments and qualified instruction
adapters. Merlin derives the problem from current source, verifies the scalar
recurrence and initialization, and retains source custody through C emission.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from xdsl.dialects import arith, builtin, tensor
from xdsl.dialects.linalg.ops import FillOp, YieldOp

from merlin.transforms.certificates import ir_digest
from merlin.transforms.dense_ir import recognize_signed_integer_matmul
from merlin.transforms.pipeline_state import (
    AnalysisStamp,
    PassSpec,
    PipelineContext,
    PipelineSession,
    validate_pass_order,
)

from .compact_dense import CompactDenseResult, schedule_compact_dense
from .library_sources import shared_library_sources
from .selected_contracts import SelectedSchedulingContracts


class DenseSourceUnavailable(ValueError):
    """Unsupported source semantics; a caller may retain its existing legal route."""


def _digest(value):
    import hashlib
    import json

    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _selection(selected):
    return _digest((selected.record(), asdict(selected.dense)))


def _problem(operation, selected, element_budget):
    try:
        lhs, rhs, init = recognize_signed_integer_matmul(operation, selected.dense.dtypes)
    except ValueError as error:
        raise DenseSourceUnavailable(str(error)) from error
    shapes = tuple(tuple(v.type.get_shape()) for v in (lhs, rhs, init))
    if any(len(shape) != 2 or any(type(d) is not int or d <= 0 for d in shape) for shape in shapes):
        raise DenseSourceUnavailable("dense source requires positive static rank-two tensor shapes")
    (m, k), (rhs_k, n), output = shapes
    if rhs_k != k or output != (m, n):
        raise DenseSourceUnavailable("dense source operand extents disagree with the complete reduction domain")
    if m * k + k * n + m * n > element_budget:
        raise DenseSourceUnavailable("dense source exceeds the selected logical element budget")
    fill = init.owner
    if (
        type(fill) is not FillOp
        or len(fill.inputs) != 1
        or len(fill.outputs) != 1
        or len(fill.results) != 1
        or fill.results[0] is not init
        or fill.attributes
        or set(fill.properties) - {"operandSegmentSizes"}
    ):
        raise DenseSourceUnavailable("dense source requires an explicit pure zero fill initializer")
    constant = fill.inputs[0].owner
    if (
        type(constant) is not arith.ConstantOp
        or constant.attributes
        or set(constant.properties) != {"value"}
        or type(constant.value) is not builtin.IntegerAttr
        or constant.value.value.data != 0
        or constant.results[0].type != init.type.element_type
    ):
        raise DenseSourceUnavailable("dense source initializer is not an exact integer zero")
    if len(fill.regions) != 1 or len(fill.regions[0].blocks) != 1 or len(fill.regions[0].block.args) != 2:
        raise DenseSourceUnavailable("dense source zero fill has an unsupported body")
    body = list(fill.regions[0].block.ops)
    if (
        len(body) != 1
        or type(body[0]) is not YieldOp
        or list(body[0].operands) != [fill.regions[0].block.args[0]]
        or body[0].attributes
        or body[0].properties
    ):
        raise DenseSourceUnavailable("dense source zero fill does not yield its zero input")
    empty = fill.outputs[0].owner
    if type(empty) is not tensor.EmptyOp or empty.attributes or empty.properties or empty.operands:
        raise DenseSourceUnavailable("dense source requires a fresh tensor destination before initialization")
    return selected.dense.problem(m, k, n)


def read_dense_source(operation, *, selected_contracts, max_source_elements=64 << 20):
    """Match current exact signed dense source before asking a target for policy.

    This establishes neither runtime ownership nor executable target support.
    Compilation repeats admission and binds the complete original module.
    """
    if type(selected_contracts) is not SelectedSchedulingContracts or selected_contracts.dense is None:
        raise DenseSourceUnavailable("selected software has no exact dense scheduling contract")
    if type(max_source_elements) is not int or max_source_elements <= 0:
        raise ValueError("dense source compilation needs a positive original element budget")
    selected_contracts.revalidate()
    result = _problem(operation, selected_contracts, max_source_elements)
    selected_contracts.revalidate()
    return result
def read_source_dense(operation, *, selected_contracts, max_source_elements=64 << 20):
    """Read supported scalar semantics without starting a scheduling session.

    This uses the same source admission as compilation. A model discovery pass
    can decline unrelated operations before verifying or scheduling the complete
    module; compilation independently repeats admission on the current source.
    """
    if type(selected_contracts) is not SelectedSchedulingContracts or selected_contracts.dense is None:
        raise DenseSourceUnavailable("selected software specification has no dense scheduling capability")
    if type(max_source_elements) is not int or max_source_elements <= 0:
        raise ValueError("dense source reading needs a positive original element budget")
    selected_contracts.revalidate()
    return _problem(operation, selected_contracts, max_source_elements)


@dataclass(frozen=True)
class DenseSourceResult:
    source: builtin.ModuleOp
    operation: object
    operation_ordinal: int
    source_stamp: AnalysisStamp
    lowered_stamp: AnalysisStamp
    selected_contracts: SelectedSchedulingContracts
    selection_identity: str
    compiled: CompactDenseResult
    source_passes: tuple
    output_operation: object = None
    output_operation_ordinal: int | None = None
    output_readout: object = None
    checked_readout: str | None = None
    readout_qualification_identity: str | None = None

    @property
    def operations(self):
        return (self.operation,) if self.output_operation is None else (self.operation, self.output_operation)

    @property
    def output_dtype(self):
        return self.compiled.kernel.args[-1].dtype

    def revalidate(self):
        self.selected_contracts.revalidate()
        if (
            ir_digest(self.source) != self.source_stamp.source_sha256
            or _selection(self.selected_contracts) != self.selection_identity
            or _digest(shared_library_sources()) != self.source_stamp.context.library
            or not any(i == self.operation_ordinal and op is self.operation for i, op in enumerate(self.source.walk()))
            or type(self.compiled) is not CompactDenseResult
            or self.compiled.checked_emission != self.lowered_stamp
            or self.compiled.kernel.digest() != self.lowered_stamp.source_sha256
            or self.output_operation is not None
            and not any(
                i == self.output_operation_ordinal and op is self.output_operation
                for i, op in enumerate(self.source.walk())
            )
            or (None if self.output_readout is None else self.output_readout.identity) != self.checked_readout
            or self.compiled.output_readout != self.output_readout
            or (self.output_operation is None) != (self.output_readout is None)
            or self.output_readout is not None
            and self.source_stamp.context.execution
            != (
                "dense.readout.execution.v1:"
                + _digest(
                    (self.readout_qualification_identity, self.output_readout.identity, self.output_operation_ordinal)
                )
            )
        ):
            raise ValueError("dense source, selected contracts or shared library changed; replan")
        self.compiled.revalidate()

    def emit_c(self, instruction_set, *, symbol):
        """Emit the checked loop kernel only while its original source stays current."""
        self.revalidate()
        text = self.compiled.emit_c(instruction_set, symbol=symbol)
        self.revalidate()
        return text


def schedule_source_dense(
    module,
    operation,
    *,
    selected_contracts,
    storage,
    adapter,
    name,
    static_budget,
    external_buffers_disjoint,
    resources=None,
    reservations=(),
    outer_order=("m", "n"),
    max_source_elements=64 << 20,
    verification_timeout_ms=2000,
    max_repairs=1,
    pass_order=(),
    schedule_pass_order=(),
    output_operation=None,
    readout_qualification_identity=None,
):
    """Compile one current source operation; missing capability keeps fallback explicit.

    No shape, model name or provenance label selects the transformation. The
    scalar recurrence, source extents, zero initialization and selected formats
    must agree. Physical/runtime disjointness is an original-caller premise,
    never inferred from different tensor SSA names. Returns None only when the
    existing bounded scheduler cannot realize a legal candidate.
    ``pass_order`` selects source stages; ``schedule_pass_order`` reaches the
    compact scheduler's independent prerequisite and realization checks.
    """
    if (
        type(module) is not builtin.ModuleOp
        or type(selected_contracts) is not SelectedSchedulingContracts
        or selected_contracts.dense is None
    ):
        raise ValueError("dense source compilation requires an original module and selected dense contracts")
    if type(schedule_pass_order) is not tuple:
        raise ValueError("dense schedule pass order requires an explicit tuple")
    if type(max_source_elements) is not int or max_source_elements <= 0:
        raise ValueError("dense source compilation needs a positive original element budget")
    if not any(op is operation for op in module.walk()):
        raise ValueError("dense source operation is not in the selected module")
    module.verify()
    selected_contracts.revalidate()
    output_readout = None
    if output_operation is not None:
        output_readout = read_source_dense_readout(
            operation,
            output_operation,
            selected_contracts=selected_contracts,
            qualification_identity=readout_qualification_identity,
            max_source_elements=max_source_elements,
        )
    elif readout_qualification_identity is not None:
        raise ValueError("dense output qualification requires an original output operation")
    output_ordinal = (
        None if output_operation is None else next(i for i, op in enumerate(module.walk()) if op is output_operation)
    )
    selection = _selection(selected_contracts)
    context = PipelineContext(
        hardware=selected_contracts.hardware_identity,
        numerical_policy=selected_contracts.numerical_policy,
        execution=(
            adapter.qualification_identity
            if output_readout is None
            else "dense.readout.execution.v1:"
            + _digest((readout_qualification_identity, output_readout.identity, output_ordinal))
        ),
        resources=_digest(
            (
                asdict(storage),
                asdict(resources) if resources else None,
                [asdict(r) for r in reservations],
                static_budget,
                max_source_elements,
                external_buffers_disjoint,
                outer_order,
                verification_timeout_ms,
                max_repairs,
                schedule_pass_order,
                None if output_readout is None else asdict(output_readout),
                readout_qualification_identity,
            )
        ),
        library=_digest(shared_library_sources()),
    )
    session = PipelineSession(module, context=context, ir_level="linalg", max_repairs=0)
    specifications = (
        PassSpec("source-admission", 0, produces=("semantic_problem",), mutates_ir=False, ir_level="linalg"),
        PassSpec(
            "dense-lowering",
            1,
            requires=("semantic_problem",),
            produces=("compiled_dense",),
            mutates_ir=False,
            ir_level="linalg",
        ),
    )
    order = pass_order or tuple(spec.name for spec in specifications)
    by_name = {spec.name: spec for spec in specifications}
    if len(order) != len(by_name) or set(order) != set(by_name):
        raise ValueError("dense source pass order must name each source pass exactly once")
    validate_pass_order(tuple(by_name[name] for name in order))

    def fresh():
        selected_contracts.revalidate()
        if _selection(selected_contracts) != selection or _digest(shared_library_sources()) != context.library:
            raise ValueError("dense source selection changed during compilation; replan")

    def lower():
        fresh()
        result = schedule_compact_dense(
            session.analysis("semantic_problem"),
            resources,
            storage,
            adapter,
            name=name,
            dtypes=selected_contracts.dense.dtypes,
            static_budget=static_budget,
            external_buffers_disjoint=external_buffers_disjoint,
            reservations=reservations,
            outer_order=outer_order,
            verification_timeout_ms=verification_timeout_ms,
            max_repairs=max_repairs,
            selected_contracts=selected_contracts,
            pass_order=schedule_pass_order,
            output_readout=output_readout,
            readout_qualification_identity=readout_qualification_identity,
        )
        fresh()
        return {"compiled_dense": result}

    for step in order:
        session.run(
            by_name[step],
            (
                lambda: {
                    "semantic_problem": read_source_dense(
                        operation,
                        selected_contracts=selected_contracts,
                        max_source_elements=max_source_elements,
                    )
                }
            )
            if step == "source-admission"
            else lower,
        )
    compiled = session.analysis("compiled_dense")
    if compiled is None:
        return None
    ordinal = next(i for i, op in enumerate(module.walk()) if op is operation)
    result = DenseSourceResult(
        module,
        operation,
        ordinal,
        session.stamp(),
        compiled.checked_emission,
        selected_contracts,
        selection,
        compiled,
        tuple(session.records),
        output_operation,
        output_ordinal,
        output_readout,
        None if output_readout is None else output_readout.identity,
        readout_qualification_identity,
    )
    result.revalidate()
    return result


def read_source_dense_readout(
    operation, output_operation, *, selected_contracts, qualification_identity, max_source_elements=64 << 20
):
    """Match one original unary integer output computation to one declared readout.

    Extra accumulator consumers, runtime captures and unsupported source keep
    the existing dense plus CPU route. A declaration is never qualification.
    """
    from xdsl.dialects.linalg.ops import GenericOp

    from .fusion_semantics import UnsupportedElementwise, term
    from .pointwise_source import read_pointwise_source

    if (
        type(selected_contracts) is not SelectedSchedulingContracts
        or selected_contracts.dense is None
        or type(output_operation) is not GenericOp
    ):
        raise DenseSourceUnavailable("dense output readout requires a declared typed pointwise source")
    readout = selected_contracts.dense.output_readout
    if readout is None or type(qualification_identity) is not str or not qualification_identity.strip():
        raise DenseSourceUnavailable("dense output readout lacks a declared independently qualified implementation")
    if (
        operation.parent_block() is None
        or operation.parent_block() is not output_operation.parent_block()
        or len(tuple(operation.results[0].uses)) != 1
        or next(iter(operation.results[0].uses)).operation is not output_operation
        or tuple(output_operation.inputs) != (operation.results[0],)
    ):
        raise DenseSourceUnavailable("dense readout requires one same-block consumer of the complete accumulator")
    between = operation.next_op
    while between is not output_operation:
        if (
            between is None
            or type(between) not in (tensor.EmptyOp, arith.ConstantOp)
            or between.attributes
            or between.regions
            or between.successors
        ):
            raise DenseSourceUnavailable("dense output readout cannot cross unknown operations or execution effects")
        between = between.next_op
    try:
        source = read_pointwise_source(
            output_operation,
            max_elements=max_source_elements,
            max_operations=128,
            max_ir_bytes=65536,
            allow_constant_captures=True,
        )
    except UnsupportedElementwise as error:
        raise DenseSourceUnavailable(str(error)) from error
    expected = readout.graph()
    if (
        source.graph.input_bits != (32,)
        or source.graph.output_bits != 8
        or source.graph.expression((term("dense_accumulator"),)) != expected.expression((term("dense_accumulator"),))
    ):
        raise DenseSourceUnavailable("dense source output differs from the selected exact integer readout")
    return readout
