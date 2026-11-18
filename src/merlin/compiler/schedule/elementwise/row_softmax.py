"""Complete-domain row normalization in the existing counted schedule IR.

The source selects the rounded exponential, lane accumulation order, final
binary tree and normalization operation. Hardware selects chunk limits and
private, addressable storage. A chunk never owns a denominator. Unknown costs
select the largest legal chunk; this is a command-count policy, not a cycle
estimate. Encoders and actual-operand projectors require independent qualification.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from math import lcm

from merlin.compiler.kernel_ir.ir import (
    DTYPE_BYTES,
    Call,
    Const,
    Expr,
    Kernel,
    Loop,
    Ptr,
    TensorArg,
    Var,
    check_structure,
)
from merlin.compiler.schedule.dense.compact import DenseMemoryView
from merlin.compiler.schedule.elementwise.row_softmax_checks import RowSoftmaxCompilation, RowSoftmaxNumerics
from merlin.compiler.semantic.allocation_types import CandidateGraph, Reservation, StorageBank, Value
from merlin.compiler.semantic.rules import AddressConstraint
from merlin.compiler.semantic.search.allocation import allocate, check_assignment

# ---- row_softmax -------------------------------------------------------------------------------


@dataclass(frozen=True)
class RowSoftmaxSource:
    source: object
    operations: tuple
    shape: tuple[int, ...]
    numerics: RowSoftmaxNumerics


def read_source_rows(final, *, numerical_policy, assume_finite_inputs, max_source_elements=64 << 20):
    """Match the complete existing pinned source algorithm, including its tree.

    This first source family is the explicit f32 CPU algorithm already provided
    by cpu_softmax. It supplies no source exponential, lane count or mask from
    hardware facts. Other source algorithms retain their original route.
    Finite admission applies to this island's actual input, not its producers.
    """
    from math import prod

    from xdsl.dialects import builtin
    from xdsl.dialects.linalg.ops import GenericOp
    from xdsl.ir import Block

    from merlin.compiler.lower.options.nonlinear import (
        CPU_SOFTMAX_F32_AVX2_POLICY,
        build_cpu_softmax_lastdim,
        cpu_softmax_row_numerics,
    )
    from merlin.compiler.schedule.elementwise.row_softmax_checks import RowSoftmaxUnavailable

    if type(max_source_elements) is not int or max_source_elements <= 0:
        raise ValueError("source rows require a positive logical element budget")
    if type(assume_finite_inputs) is not bool:
        raise ValueError("source rows require an explicit finite-input premise")
    if numerical_policy != CPU_SOFTMAX_F32_AVX2_POLICY or not assume_finite_inputs:
        raise RowSoftmaxUnavailable("source rows lack the supported original numerical/finite-input policy")
    if type(final) is not GenericOp or len(final.results) != 1 or final.parent is None:
        raise RowSoftmaxUnavailable("source rows require an attached complete normalization result")
    typ = final.results[0].type
    if (
        not isinstance(typ, builtin.TensorType)
        or typ.element_type != builtin.f32
        or not isinstance(typ.encoding, builtin.NoneAttr)
    ):
        raise RowSoftmaxUnavailable("source rows require the original plain f32 tensor layout")
    shape = tuple(typ.get_shape())
    if not shape or any(n <= 0 for n in shape) or shape[-1] < 8 or prod(shape) > max_source_elements:
        raise RowSoftmaxUnavailable("source rows have unsupported extents or exceed their logical element budget")
    # A fresh argument avoids adding artificial detached uses to the live IR.
    argument = Block(arg_types=[typ]).args[0]
    expected, _ = build_cpu_softmax_lastdim(argument, backend_policy=numerical_policy, assume_finite_inputs=True)
    try:
        block = tuple(final.parent.ops)
        stop = block.index(final) + 1
        if stop < len(expected):
            raise RowSoftmaxUnavailable("source row algorithm is incomplete")
        actual = block[stop - len(expected) : stop]
        positions = [
            (i, j) for i, op in enumerate(expected) for j, value in enumerate(op.operands) if value is argument
        ]
        if any(j >= len(actual[i].operands) for i, j in positions):
            raise RowSoftmaxUnavailable("source row input bindings differ from the complete algorithm")
        inputs = {actual[i].operands[j] for i, j in positions}
        if len(inputs) != 1 or (source := next(iter(inputs))).type != typ or source.owner in actual:
            raise RowSoftmaxUnavailable("source row input bindings differ from the complete algorithm")
        context = {argument: source}
        for reference, realized in zip(expected, actual, strict=True):
            # Structural equivalence omits result types and can equate signed
            # floating zeros. Preserve every typed field and nested result.
            if not reference.is_structurally_equivalent(realized, context) or any(
                tuple(v.type for v in left.results) != tuple(v.type for v in right.results)
                or any(
                    tuple(sorted((key, str(value)) for key, value in a.items()))
                    != tuple(sorted((key, str(value)) for key, value in b.items()))
                    for a, b in zip(
                        (left.attributes, left.properties),
                        (right.attributes, right.properties),
                        strict=True,
                    )
                )
                for left, right in zip(reference.walk(), realized.walk(), strict=True)
            ):
                raise RowSoftmaxUnavailable("source row arithmetic, types, initialization or reduction order differs")
        members = {node for op in actual for node in op.walk()}
        published = set(final.results)
        defined = (
            value
            for node in members
            for value in (
                *node.results,
                *(arg for region in node.regions for block in region.blocks for arg in block.args),
            )
        )
        if any(use.operation not in members for value in defined if value not in published for use in value.uses):
            raise RowSoftmaxUnavailable("source row intermediate has an escaping observation")
        return RowSoftmaxSource(
            source, actual, shape, cpu_softmax_row_numerics(backend_policy=numerical_policy, assume_finite_inputs=True)
        )
    finally:
        for op in expected:
            op.drop_all_references()


@dataclass(frozen=True)
class RowSoftmaxSourceCompilation:
    module: object
    source: RowSoftmaxSource
    operation_ordinal: int
    source_stamp: object
    compiled: RowSoftmaxCompilation
    compiled_stamp: object
    source_passes: tuple

    def revalidate(self):
        from math import prod

        from merlin.compiler.schedule.library_sources import shared_library_sources
        from merlin.compiler.transforms.certificates import ir_digest

        library = hashlib.sha256(json.dumps(shared_library_sources(), sort_keys=True).encode()).hexdigest()
        if (
            ir_digest(self.module) != self.source_stamp.source_sha256
            or library != self.source_stamp.context.library
            or not any(
                i == self.operation_ordinal and op is self.source.operations[-1]
                for i, op in enumerate(self.module.walk())
            )
            or not self.compiled.passes
            or self.compiled.passes[-1].after != self.compiled_stamp
            or self.compiled.kernel.digest() != self.compiled.passes[-1].after.source_sha256
            or self.compiled_stamp.context.numerical_policy != self.source.numerics.identity
        ):
            raise ValueError("row source, numerical algorithm, schedule or shared library changed; replan")
        self.compiled.revalidate()
        if (
            read_source_rows(
                self.source.operations[-1],
                numerical_policy=self.source_stamp.context.numerical_policy,
                assume_finite_inputs=True,
                max_source_elements=prod(self.source.shape),
            )
            != self.source
        ):
            raise ValueError("row source input, island membership or numerical algorithm changed; replan")

    def emit_c(self, instruction_set, *, symbol):
        self.revalidate()
        emitted = self.compiled.emit_c(instruction_set, symbol=symbol)
        self.revalidate()
        return emitted


def compile_source_rows(
    module,
    final,
    *,
    selected_contracts,
    adapter,
    numerical_policy,
    assume_finite_inputs=False,
    private_ownership=False,
    disjoint_arguments=False,
    resources=None,
    banks=None,
    role_banks=None,
    reservations=(),
    chunk_elements=None,
    timeout_ms=2000,
    max_source_elements=64 << 20,
    pass_order=(),
):
    """Bind current typed source to the existing complete-row compiler.

    Target instruction selection, realization and independently qualified runtime
    bindings remain explicit. Unsupported source raises RowSoftmaxUnavailable
    before encoding, leaving the caller's original source route available.
    """
    from math import prod

    from xdsl.dialects import builtin

    from merlin.compiler.schedule.elementwise.row_softmax_checks import RowSoftmaxProblem
    from merlin.compiler.schedule.library_sources import shared_library_sources
    from merlin.compiler.schedule.pipeline.selected_contracts import SelectedSchedulingContracts
    from merlin.compiler.transforms.pipeline_state import (
        PassSpec,
        PipelineContext,
        PipelineSession,
        validate_pass_order,
    )

    if (
        type(module) is not builtin.ModuleOp
        or type(selected_contracts) is not SelectedSchedulingContracts
        or selected_contracts.row_reduction is None
        or not any(op is final for op in module.walk())
    ):
        raise ValueError("source rows require current typed IR and selected complete-row RTL/software contracts")
    module.verify()
    selected_contracts.revalidate()

    def identity(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

    sources = shared_library_sources()
    context = PipelineContext(
        hardware=selected_contracts.hardware_identity,
        numerical_policy=numerical_policy,
        library=identity(sources),
        execution=adapter.qualification_identity,
        resources=identity(
            dict(
                resources=asdict(resources) if resources is not None else None,
                banks=[asdict(b) for b in banks] if banks is not None else None,
                role_banks=role_banks,
                reservations=[asdict(r) for r in reservations],
                chunk_elements=chunk_elements,
                timeout_ms=timeout_ms,
                max_source_elements=max_source_elements,
                assume_finite_inputs=assume_finite_inputs,
                private_ownership=private_ownership,
                disjoint_arguments=disjoint_arguments,
            )
        ),
    )
    specs = (
        PassSpec("row_source_admission", 0, produces=("source_rows",), mutates_ir=False, ir_level="linalg"),
        PassSpec(
            "row_source_lowering",
            1,
            requires=("source_rows",),
            produces=("compiled_rows",),
            mutates_ir=False,
            ir_level="linalg",
        ),
    )
    if (
        type(pass_order) is not tuple
        or pass_order
        and (len(pass_order) != len(specs) or set(pass_order) != {spec.name for spec in specs})
    ):
        raise ValueError("source row ordering must name every source pass exactly once")
    specs = tuple(next(spec for spec in specs if spec.name == name) for name in pass_order) if pass_order else specs
    validate_pass_order(specs)
    session = PipelineSession(module, context=context, ir_level="linalg", max_repairs=0)

    def lower():
        source = session.analysis("source_rows")
        return {
            "compiled_rows": compile_row_softmax(
                RowSoftmaxProblem(prod(source.shape[:-1]), source.shape[-1]),
                source.numerics,
                resources,
                banks,
                role_banks,
                adapter,
                selected_contracts=selected_contracts,
                reservations=reservations,
                private_ownership=private_ownership,
                disjoint_arguments=disjoint_arguments,
                chunk_elements=chunk_elements,
                timeout_ms=timeout_ms,
            )
        }

    actions = {
        "row_source_admission": lambda: {
            "source_rows": read_source_rows(
                final,
                numerical_policy=numerical_policy,
                assume_finite_inputs=assume_finite_inputs,
                max_source_elements=max_source_elements,
            )
        },
        "row_source_lowering": lower,
    }
    for spec in specs:
        selected_contracts.revalidate()
        if sources != shared_library_sources():
            raise ValueError("row source compiler changed during compilation; replan")
        session.run(spec, actions[spec.name])
    compiled = session.analysis("compiled_rows")
    result = RowSoftmaxSourceCompilation(
        module,
        session.analysis("source_rows"),
        next(i for i, op in enumerate(module.walk()) if op is final),
        session.stamp(),
        compiled,
        compiled.passes[-1].after,
        tuple(session.records),
    )
    result.revalidate()
    return result


def _choose_chunk(problem, resources, explicit):
    from merlin.compiler.schedule.elementwise.row_softmax_checks import RowSoftmaxUnavailable

    if resources.max_static_calls < 7:
        raise RowSoftmaxUnavailable("row softmax static command budget cannot hold its required seven phases")
    limit = min(problem.columns, resources.max_chunk_elements)
    chunk = (limit // resources.granularity) * resources.granularity
    if resources.partial_chunks and limit == problem.columns:
        chunk = limit
    elif not chunk and resources.partial_chunks:
        chunk = limit
    if explicit is not None:
        chunk = explicit
    elif chunk and (not resources.partial_chunks or resources.max_static_calls < 11) and problem.columns % chunk:
        # A bounded descending search avoids scanning a tensor-sized candidate
        # domain. The primitive granularity is a legal conservative fallback
        # when it divides the complete row. No timing model is implied.
        candidates = (chunk - i * resources.granularity for i in range(64))
        chunk = next(
            (n for n in candidates if n > 0 and problem.columns % n == 0),
            resources.granularity if problem.columns % resources.granularity == 0 else 0,
        )
    if type(chunk) is not int or chunk <= 0 or chunk > limit:
        raise RowSoftmaxUnavailable("no legal row chunk within declared resources")
    if chunk % resources.granularity and (chunk != problem.columns or not resources.partial_chunks):
        raise RowSoftmaxUnavailable("row chunk violates declared primitive granularity")
    if problem.columns % chunk and not resources.partial_chunks:
        raise RowSoftmaxUnavailable("row tail requires declared partial-chunk support")
    if 7 + 4 * bool(problem.columns % chunk) > resources.max_static_calls:
        raise RowSoftmaxUnavailable("row softmax chosen tail exceeds the static command budget")
    return chunk


def _schedule(problem, numerics, storage, chunk, adapter):
    from merlin.compiler.schedule.elementwise.row_softmax_checks import _step

    row = Var("row")
    body = []

    def call(kind, column, count):
        return adapter.encode(_step(kind, row, column, count, problem, numerics, storage))

    def chunks(kind):
        trips, tail = divmod(problem.columns, chunk)
        var = kind + "_chunk"
        if trips:
            body.append(Loop(var, trips, (call(kind, Var(var) * chunk, chunk),)))
        if tail:
            body.append(call(kind, Const(trips * chunk), tail))

    body.append(call("init_maximum", Const(0), 1))
    chunks("maximum")
    chunks("exponential")
    body.append(call("init_lanes", Const(0), numerics.lanes))
    chunks("sum")
    body.append(call("finish", Const(0), numerics.lanes))
    chunks("normalize")
    return Kernel(
        "row_softmax",
        tuple(
            TensorArg(name, (problem.rows, problem.columns), numerics.dtype, access)
            for name, access in (("input", "read"), ("output", "write"))
        ),
        (Loop("row", problem.rows, tuple(body)),),
    )


def compile_row_softmax(
    problem,
    numerics,
    resources,
    banks,
    role_banks,
    adapter,
    *,
    reservations=(),
    private_ownership=False,
    disjoint_arguments=False,
    chunk_elements=None,
    timeout_ms=2000,
    pass_order=(),
    selected_contracts=None,
):
    """Execute admission, tiling, allocation and verified dependency scheduling.

    Explicit chunks are Phase-2 alternatives through the same pipeline, selected
    source algorithm and checks. Invalid explicit ordering fails before any
    encoder/allocation runs. Refusal retains the caller's original source route.
    This synchronous schedule-to-C path is not final machine-byte conformance
    or ordinary full-model target integration.
    """
    from merlin.compiler.schedule.elementwise.row_softmax_checks import (
        RowSoftmaxCompilation,
        _admit,
        _storage_problem,
        allocate_row_softmax_storage,
        verify_row_softmax,
    )
    from merlin.compiler.schedule.pipeline.selected_contracts import SelectedSchedulingContracts
    from merlin.compiler.schedule.pipeline.stages import shared_library_sources
    from merlin.compiler.transforms.pipeline_state import (
        PassSpec,
        PipelineContext,
        PipelineSession,
        validate_pass_order,
    )

    specs = (
        PassSpec("row_semantics", 0, produces=("row_source",), mutates_ir=False, ir_level="schedule"),
        PassSpec(
            "row_tiling", 2, requires=("row_source",), produces=("row_chunk",), mutates_ir=False, ir_level="schedule"
        ),
        PassSpec(
            "row_allocation",
            3,
            requires=("row_source", "row_chunk"),
            produces=("row_storage",),
            mutates_ir=False,
            ir_level="schedule",
        ),
        PassSpec(
            "row_schedule",
            4,
            requires=("row_source", "row_chunk", "row_storage"),
            produces=("row_realization",),
            invalidates=("*",),
            ir_level="schedule",
        ),
    )
    if type(pass_order) is not tuple:
        raise ValueError("row reduction pass ordering must be an explicit immutable sequence")
    if pass_order:
        if len(pass_order) != len(specs) or set(pass_order) != {s.name for s in specs}:
            raise ValueError("explicit row reduction ordering must contain every pass exactly once")
        specs = tuple(next(s for s in specs if s.name == name) for name in pass_order)
    validate_pass_order(specs)
    source_adapter = adapter
    binding = None
    checked_selection = None
    if selected_contracts is not None:
        if type(selected_contracts) is not SelectedSchedulingContracts or selected_contracts.row_reduction is None:
            raise ValueError("row scheduling requires explicitly selected complete-row RTL/software contracts")
        selected_contracts.revalidate()
        checked_selection = hashlib.sha256(json.dumps(selected_contracts.record(), sort_keys=True).encode()).hexdigest()
        resources, banks, role_banks, reservations, adapter = selected_contracts.row_reduction.bind(
            selected_contracts,
            problem,
            numerics,
            resources,
            banks,
            role_banks,
            adapter,
            reservations=reservations,
            private_ownership=private_ownership,
        )
        binding = selected_contracts.record(
            scopes=(
                "row_source_algorithm",
                "row_chunk_and_index_limits",
                "row_state_windows_and_reservations",
                "complete_phase_operands_and_effects",
            )
        )
    _admit(problem, numerics, resources, adapter, private_ownership, disjoint_arguments)
    _storage_problem(problem, numerics, banks, role_banks)
    if type(reservations) is not tuple or any(type(r) is not Reservation for r in reservations):
        raise ValueError("row softmax reservations must be explicit immutable intervals")

    def identity(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

    def resource_inputs():
        return dict(
            problem=asdict(problem),
            resources=asdict(resources),
            banks=[asdict(b) for b in banks],
            role_banks=role_banks,
            reservations=[asdict(r) for r in reservations],
            chunk_elements=chunk_elements,
            private_ownership=private_ownership,
            disjoint_arguments=disjoint_arguments,
            index_bits=adapter.index_bits,
        )

    def execution_inputs():
        fields = (
            "qualification_identity",
            "configuration_identity",
            "numerical_identity",
            "index_bits",
            "synchronous_completion",
            "exact_source_order",
            "addressable_state",
        )
        return dict(
            selected_contracts=binding,
            adapters={
                name: {key: getattr(value, key) for key in fields}
                for name, value in (("source", source_adapter), ("bound", adapter))
            },
        )

    def execution_identity():
        return (
            "row.execution.v3:"
            + identity(execution_inputs())
            + f":selection={checked_selection or 'unselected'}:index_bits={adapter.index_bits}"
        )

    sources = shared_library_sources()
    context = PipelineContext(
        hardware=adapter.configuration_identity,
        numerical_policy=numerics.identity,
        resources=identity(resource_inputs()),
        execution=execution_identity(),
        library=identity(sources),
    )

    def verify_ir(value):
        if type(value) is not Kernel or check_structure(value):
            raise ValueError("row scheduling produced invalid counted IR")

    session = PipelineSession(
        Kernel("row_softmax", (), ()),
        context=context,
        ir_level="schedule",
        max_repairs=0,
        fingerprint=lambda value: value.digest(),
        verify_ir=verify_ir,
    )

    def scheduled():
        assigned = session.analysis("row_storage")
        chunk = session.analysis("row_chunk")
        session.module = _schedule(problem, numerics, assigned, chunk, adapter)
        report = verify_row_softmax(
            session.module,
            problem,
            numerics,
            resources,
            assigned,
            banks,
            role_banks,
            adapter,
            reservations=reservations,
            private_ownership=private_ownership,
            disjoint_arguments=disjoint_arguments,
        )
        return {"row_realization": report}

    actions = {
        "row_semantics": lambda: {"row_source": numerics.identity},
        "row_tiling": lambda: {"row_chunk": _choose_chunk(problem, resources, chunk_elements)},
        "row_allocation": lambda: {
            "row_storage": allocate_row_softmax_storage(
                problem, numerics, banks, role_banks, reservations=reservations, timeout_ms=timeout_ms
            )
        },
        "row_schedule": scheduled,
    }
    storage = None

    def fresh():
        if (
            context.resources != identity(resource_inputs())
            or context.numerical_policy != numerics.identity
            or context.hardware != adapter.configuration_identity
            or context.execution != execution_identity()
        ):
            raise ValueError("row source, numerical policy, resource or execution inputs changed; replan")
        if sources != shared_library_sources():
            raise ValueError("row shared library changed during compilation; replan")
        if selected_contracts is not None:
            selected_contracts.revalidate()
            if identity(selected_contracts.record()) != checked_selection:
                raise ValueError("row selected RTL/software contracts changed during compilation; replan")

    for spec in specs:
        fresh()
        session.run(spec, actions[spec.name])
        fresh()
        if spec.name == "row_allocation":
            storage = session.analysis("row_storage")
    report = {
        **session.analysis("row_realization"),
        "pipeline": {
            "schema": "merlin.complete_row_pipeline.v1",
            "context": asdict(context),
            "kernel_sha256": session.module.digest(),
            "library_sources_sha256": sources,
            "selected_contracts": binding,
            "index_bits": adapter.index_bits,
        },
    }
    return RowSoftmaxCompilation(
        session.module,
        storage,
        report,
        adapter.index_bits,
        tuple(session.records),
        selected_contracts,
        adapter.index_bits,
        checked_selection,
    )
