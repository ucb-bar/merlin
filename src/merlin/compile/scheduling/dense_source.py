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
        ):
            raise ValueError("dense source, selected contracts or shared library changed; replan")

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
    selection = _selection(selected_contracts)
    context = PipelineContext(
        hardware=selected_contracts.hardware_identity,
        numerical_policy=selected_contracts.numerical_policy,
        execution=adapter.qualification_identity,
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
        )
        fresh()
        return {"compiled_dense": result}

    for step in order:
        session.run(
            by_name[step],
            (lambda: {"semantic_problem": _problem(operation, selected_contracts, max_source_elements)})
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
    )
    result.revalidate()
    return result
