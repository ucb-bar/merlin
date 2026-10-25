"""Bind typed source matmul semantics to the existing compact dense scheduler.

Targets still supply physical storage assignments and qualified instruction
adapters. Merlin derives the problem from current source, verifies the scalar
recurrence and initialization, and retains source custody through C emission.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from xdsl.dialects import arith, builtin, tensor

from merlin.transforms.certificates import ir_digest
from merlin.transforms.dense_ir import static_zero_integer_matmul_geometry
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


def _source_geometry(operation, dtypes, element_budget, *, allow_provenance=False):
    try:
        return static_zero_integer_matmul_geometry(
            operation, dtypes, max_source_elements=element_budget
        )
    except ValueError as error:
        raise DenseSourceUnavailable(str(error)) from error


def read_zero_seeded_integer_matmul(operation, *, dtypes=("i8", "i8", "i32"), max_source_elements=64 << 20):
    """Read tensor semantics and geometry without selecting hardware or a schedule."""
    if type(max_source_elements) is not int or max_source_elements <= 0:
        raise ValueError("dense source reading needs a positive original element budget")
    detached = operation.clone()
    try:
        for op in detached.walk():
            for key in tuple(op.attributes):
                if key.startswith("prov."):
                    del op.attributes[key]
        return _source_geometry(detached, dtypes, max_source_elements, allow_provenance=True)
    finally:
        detached.drop_all_references()


def _problem(operation, selected, element_budget):
    return selected.dense.problem(*_source_geometry(operation, selected.dense.dtypes, element_budget))


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
    readout_operations: tuple = ()
    readout_operation_ordinals: tuple[int, ...] = ()
    source_element_budget: int = 64 << 20

    @property
    def operations(self):
        return (
            (self.operation, *self.readout_operations)
            if self.readout_operations
            else ((self.operation,) if self.output_operation is None else (self.operation, self.output_operation))
        )

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
            or bool(self.readout_operations) != bool(self.readout_operation_ordinals)
            or len(self.readout_operations) != len(self.readout_operation_ordinals)
            or self.readout_operations
            and self.readout_operations[-1] is not self.output_operation
            or any(
                not any(i == ordinal and op is member for i, op in enumerate(self.source.walk()))
                for member, ordinal in zip(self.readout_operations, self.readout_operation_ordinals, strict=True)
            )
            or (None if self.output_readout is None else self.output_readout.identity) != self.checked_readout
            or self.compiled.output_readout != self.output_readout
            or (self.output_operation is None) != (self.output_readout is None)
            or self.output_readout is not None
            and self.source_stamp.context.execution
            != (
                ("dense.readout.execution.v2:" if self.readout_operations else "dense.readout.execution.v1:")
                + _digest(
                    (
                        self.readout_qualification_identity,
                        self.output_readout.identity,
                        self.readout_operation_ordinals if self.readout_operations else self.output_operation_ordinal,
                    )
                )
            )
        ):
            raise ValueError("dense source, selected contracts or shared library changed; replan")
        self.compiled.revalidate()
        problem = _problem(self.operation, self.selected_contracts, self.source_element_budget)
        expected = (
            ((problem.m, problem.k), self.selected_contracts.dense.dtypes[0], "read"),
            ((problem.k, problem.n), self.selected_contracts.dense.dtypes[1], "read"),
            ((problem.m, problem.n), self.output_dtype, "write"),
        )
        output_dtype = (
            self.selected_contracts.dense.dtypes[2] if self.output_readout is None else self.output_readout.output_dtype
        )
        if (
            self.compiled.selected_contracts is None
            or _selection(self.compiled.selected_contracts) != self.selection_identity
            or tuple((arg.shape, arg.dtype, arg.access) for arg in self.compiled.kernel.args) != expected
            or self.output_dtype != output_dtype
        ):
            raise ValueError("compiled dense program differs from current source geometry or selection; replan")
        if self.output_readout is not None:
            read_source_dense_readout_chain(
                self.operation,
                self.readout_operations or (self.output_operation,),
                selected_contracts=self.selected_contracts,
                qualification_identity=self.readout_qualification_identity,
                max_source_elements=self.source_element_budget,
            )

    def emit_c(self, instruction_set, *, symbol):
        """Emit the checked loop kernel only while its original source stays current."""
        self.revalidate()
        text = self.compiled.emit_c(instruction_set, symbol=symbol)
        self.revalidate()
        return text

    def emit_llvm(self, builder, lower_call):
        """Consume source-bound compact IR with the same custody as C emission.

        Targets bind the original logical arguments and implement instructions;
        Merlin retains tails, reduction initialization/continuation and stores.
        Any source/contract mutation during callbacks invalidates the result;
        discard the caller's partial builder on failure.
        """
        self.revalidate()
        result = self.compiled.emit_llvm(builder, lower_call)
        self.revalidate()
        return {**result, "source_sha256": self.source_stamp.source_sha256,
                "source_operation_ordinal": self.operation_ordinal,
                "selected_contracts_identity": self.selection_identity}


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
    output_operations=(),
    readout_qualification_identity=None,
    schedule_plan=None,
):
    """Compile one current source operation; missing capability keeps fallback explicit.

    No shape, model name or provenance label selects the transformation. The
    scalar recurrence, source extents, zero initialization and selected formats
    must agree. Physical/runtime disjointness is an original-caller premise,
    never inferred from different tensor SSA names. Returns None only when the
    existing bounded scheduler cannot realize a legal candidate.
    ``pass_order`` selects source stages; ``schedule_pass_order`` reaches the
    compact scheduler's independent prerequisite and realization checks.
    An optional typed ``schedule_plan`` is bound to this unchanged source module's
    digest automatically; it cannot supply a different scalar recurrence.
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
    if type(output_operations) is not tuple or output_operations and output_operation is not None:
        raise ValueError("select one original output operation or an explicit tuple of output operations")
    readout_operations = output_operations
    readout_ordinals = tuple(
        next((i for i, op in enumerate(module.walk()) if op is member), -1) for member in readout_operations
    )
    if any(ordinal < 0 for ordinal in readout_ordinals):
        raise ValueError("dense readout operation is not in the selected module")
    output_readout = None
    if readout_operations:
        output_readout = read_source_dense_readout_chain(
            operation,
            readout_operations,
            selected_contracts=selected_contracts,
            qualification_identity=readout_qualification_identity,
            max_source_elements=max_source_elements,
        )
        output_operation = readout_operations[-1]
    elif output_operation is not None:
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
            else ("dense.readout.execution.v2:" if readout_operations else "dense.readout.execution.v1:")
            + _digest(
                (
                    readout_qualification_identity,
                    output_readout.identity,
                    readout_ordinals if readout_operations else output_ordinal,
                )
            )
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
            schedule_plan=schedule_plan,
            source_identity=None if schedule_plan is None else session.stamp().source_sha256,
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
        readout_operations,
        readout_ordinals,
        max_source_elements,
    )
    result.revalidate()
    return result


def read_source_dense_readout(
    operation, output_operation, *, selected_contracts, qualification_identity, max_source_elements=64 << 20
):
    """Match one original unary output computation to its selected exact readout."""
    return read_source_dense_readout_chain(
        operation,
        (output_operation,),
        selected_contracts=selected_contracts,
        qualification_identity=qualification_identity,
        max_source_elements=max_source_elements,
    )


def read_source_dense_readout_chain(
    operation,
    output_operations,
    *,
    selected_contracts,
    qualification_identity,
    max_source_elements=64 << 20,
    max_stages=16,
    max_scalar_operations=128,
    max_scalar_ir_bytes=65536,
):
    """Compose complete unary integer stages and match the original physical recipe.

    Each source stage retains its exact typed casts, clamps and arithmetic.
    No intermediate may have another observer. This checks correspondence to
    a separately selected readout, not equivalence under reassociation or an
    approximate conversion. Runtime captures, broadcasts and unknown effects
    preserve the original dense-plus-source route.
    """
    from xdsl.dialects.linalg.ops import GenericOp

    from .fusion_semantics import UnsupportedElementwise, term
    from .pointwise_source import read_pointwise_source

    if (
        type(selected_contracts) is not SelectedSchedulingContracts
        or selected_contracts.dense is None
        or type(output_operations) is not tuple
        or not output_operations
        or any(type(op) is not GenericOp for op in output_operations)
    ):
        raise DenseSourceUnavailable("dense output readout requires a declared typed pointwise source chain")
    if any(type(x) is not int or x <= 0 for x in (max_stages, max_scalar_operations, max_scalar_ir_bytes)):
        raise ValueError("dense readout chain requires positive stage, scalar operation and byte budgets")
    if len(output_operations) > max_stages:
        raise DenseSourceUnavailable("dense output readout chain exceeds its explicit stage budget")
    selected_contracts.revalidate()
    readout = selected_contracts.dense.output_readout
    if readout is None or type(qualification_identity) is not str or not qualification_identity.strip():
        raise DenseSourceUnavailable("dense output readout lacks a declared independently qualified implementation")
    previous = operation
    expression = term("dense_accumulator")
    bits, operation_count, byte_count = 32, 0, 0
    shape = tuple(operation.results[0].type.get_shape())
    for output in output_operations:
        uses = tuple(previous.results[0].uses)
        if (
            previous.parent_block() is None
            or previous.parent_block() is not output.parent_block()
            or len(uses) != 1
            or uses[0].operation is not output
            or tuple(output.inputs) != (previous.results[0],)
        ):
            raise DenseSourceUnavailable("dense readout requires one same-block consumer of each complete intermediate")
        between = previous.next_op
        while between is not output:
            if (
                between is None
                or type(between) not in (tensor.EmptyOp, arith.ConstantOp)
                or between.attributes
                or between.regions
                or between.successors
            ):
                raise DenseSourceUnavailable(
                    "dense output readout cannot cross unknown operations or execution effects"
                )
            between = between.next_op
        try:
            source = read_pointwise_source(
                output,
                max_elements=max_source_elements,
                max_operations=max_scalar_operations,
                max_ir_bytes=max_scalar_ir_bytes,
                allow_constant_captures=True,
            )
        except UnsupportedElementwise as error:
            raise DenseSourceUnavailable(str(error)) from error
        operation_count += len(source.graph.operations)
        byte_count += len(source.scalar_ir)
        if operation_count > max_scalar_operations or byte_count > max_scalar_ir_bytes:
            raise DenseSourceUnavailable("dense output readout chain exceeds its aggregate scalar budget")
        if source.shape != shape or source.graph.input_bits != (bits,):
            raise DenseSourceUnavailable("dense readout stage changes coordinates or complete typed input")
        expression = source.graph.expression((expression,))
        bits, previous = source.graph.output_bits, output
    expected = readout.graph()
    if (
        bits != expected.output_bits
        or expected.input_bits != (32,)
        or expression != expected.expression((term("dense_accumulator"),))
    ):
        raise DenseSourceUnavailable("dense source output differs from the selected exact integer readout")
    selected_contracts.revalidate()
    return readout
