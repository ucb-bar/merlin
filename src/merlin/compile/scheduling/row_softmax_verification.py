"""Reconstruct complete row reductions from actual projected counted calls.

This checker admits one bounded grammar, independently of selected chunks:
row loop; initialized full maximum; full exponentials; initialized source lanes;
full ordered sum; completed tree; full normalization. Affine byte operands and
constant chunk counts establish coverage without enumerating tensor elements.
It proves the schedule under the qualified primitive contract, not that an
arbitrary primitive implementation obeys that contract.
"""

from __future__ import annotations

from dataclasses import asdict

from merlin.sched.ir import Call, Const, Loop, Ptr, Scaled, Sum, TensorArg, Var
from merlin.sched.ir.kernel import DTYPE_BYTES, check_structure

from .row_softmax import (
    PHASE_ROLES,
    RowSoftmaxStep,
    _admit,
    verify_row_softmax_storage,
)


def _affine(expression):
    if isinstance(expression, Const):
        return {"": expression.value} if expression.value else {}
    if isinstance(expression, Var):
        return {expression.name: 1}
    if isinstance(expression, Scaled):
        return {
            key: value * expression.factor
            for key, value in _affine(expression.expr).items()
            if value * expression.factor
        }
    if isinstance(expression, Sum):
        result = {}
        for term in expression.terms:
            for key, value in _affine(term).items():
                result[key] = result.get(key, 0) + value
        return {key: value for key, value in result.items() if value}
    raise ValueError("row softmax operands require supported affine byte addressing")


def _equal(actual, expected):
    if _affine(actual) != _affine(expected):
        raise ValueError("row softmax actual address/coordinate does not cover its source domain")


def verify_row_softmax(
    kernel,
    problem,
    numerics,
    resources,
    storage,
    banks,
    role_banks,
    adapter,
    *,
    reservations=(),
    private_ownership=False,
    disjoint_arguments=False,
):
    """Validate actual phases, operands, domains, state lifetimes and requests."""
    _admit(problem, numerics, resources, adapter, private_ownership, disjoint_arguments)
    verify_row_softmax_storage(problem, numerics, storage, banks, role_banks, reservations)
    if check_structure(kernel) or kernel.attrs:
        raise ValueError("row softmax requires a valid kernel without unknown attributes")
    expected_args = tuple(
        TensorArg(name, (problem.rows, problem.columns), numerics.dtype, access)
        for name, access in (("input", "read"), ("output", "write"))
    )
    if kernel.args != expected_args or len(kernel.body) != 1 or type(kernel.body[0]) is not Loop:
        raise ValueError("row softmax requires complete disjoint source/output tensors and one row loop")
    outer = kernel.body[0]
    if outer.extent != problem.rows:
        raise ValueError("row softmax row loop omits or duplicates complete rows")
    row = Var(outer.var)
    body = outer.body
    index = 0
    static_calls = 0
    commands = reads = writes = 0
    phase_counts = {}
    width = DTYPE_BYTES[numerics.dtype]

    def projected(call, kind, column, count, instances):
        nonlocal static_calls, commands, reads, writes
        if (
            type(call) is not Call
            or call.unit
            or call.produces
            or call.awaits
            or call.stages
            or call.sets
            or call.assumes
        ):
            raise ValueError("unknown execution/completion/configuration effects are row reduction barriers")
        actual = adapter.project(call)
        if type(actual) is not RowSoftmaxStep or actual.kind != kind:
            raise ValueError("row softmax actual phase order is incomplete or reordered")
        if type(actual.count) is not int or actual.count != count or count <= 0:
            raise ValueError("row softmax primitive count changed its domain")
        if actual.numerical_identity != numerics.identity or actual.mask != problem.mask:
            raise ValueError("row softmax actual numerical algorithm or mask changed")
        _equal(actual.row, row)
        _equal(actual.column, column)
        if type(actual.local) is not tuple or len(actual.local) != len(PHASE_ROLES[kind]):
            raise ValueError("row softmax actual intermediate state roles changed")
        for (space, offset), role in zip(actual.local, PHASE_ROLES[kind], strict=True):
            view = storage[role]
            if space != view.backing:
                raise ValueError("row softmax intermediate uses the wrong physical backing")
            _equal(offset, Const(view.base * width) + (column * width if role == 1 else Const(0)))
        for pointer, name, present in (
            (actual.source, "input", kind in ("maximum", "exponential")),
            (actual.output, "output", kind == "normalize"),
        ):
            if present:
                if type(pointer) is not Ptr or pointer.tensor != name:
                    raise ValueError("row softmax actual external source/output binding changed")
                _equal(pointer.offset, (row * problem.columns + column) * width)
            elif pointer is not None:
                raise ValueError("row softmax has an undeclared external effect")
        static_calls += 1
        if static_calls > resources.max_static_calls:
            raise ValueError("row softmax exceeds its static command budget")
        multiplicity = instances * problem.rows
        commands += multiplicity
        phase_counts[kind] = phase_counts.get(kind, 0) + multiplicity
        # Logical requested payloads of the declared primitive, including state.
        # Physical/native register traffic and hidden effects are not inferred.
        lane = numerics.lanes
        r, w = {
            "init_maximum": (0, 1),
            "maximum": (count + 1, 1),
            "exponential": (count + 1, count),
            "init_lanes": (0, lane),
            "sum": (count + lane, lane),
            "finish": (lane, 1),
            "normalize": (count + 1, count),
        }[kind]
        reads += multiplicity * width * r
        writes += multiplicity * width * w

    def singleton(kind, count):
        nonlocal index
        if index >= len(body):
            raise ValueError("row softmax lost a required initialization/completion phase")
        projected(body[index], kind, Const(0), count, 1)
        index += 1

    def domain(kind):
        nonlocal index
        if index >= len(body):
            raise ValueError("row softmax lost a complete reduction domain")
        statement = body[index]
        covered = 0
        full_chunk = None
        if type(statement) is Loop:
            if len(statement.body) != 1 or type(statement.body[0]) is not Call:
                raise ValueError("row chunk requires one declared primitive with no hidden nested work")
            actual = adapter.project(statement.body[0])
            if type(actual) is not RowSoftmaxStep or type(actual.count) is not int or actual.count <= 0:
                raise ValueError("row chunk requires a positive actual constant extent")
            full_chunk = actual.count
            covered = statement.extent * full_chunk
            if (
                full_chunk > resources.max_chunk_elements
                or full_chunk % resources.granularity
                and (full_chunk != problem.columns or not resources.partial_chunks)
                or covered > problem.columns
            ):
                raise ValueError("row chunk violates capacity, granularity or complete domain")
            projected(statement.body[0], kind, Var(statement.var) * full_chunk, full_chunk, statement.extent)
            index += 1
        remaining = problem.columns - covered
        if remaining:
            if (
                not resources.partial_chunks
                or remaining > resources.max_chunk_elements
                or (full_chunk is not None and remaining >= full_chunk)
            ):
                raise ValueError("row softmax incomplete domain or unsupported tail")
            if index >= len(body):
                raise ValueError("row softmax lost its final reduction tail")
            projected(body[index], kind, Const(covered), remaining, 1)
            index += 1

    singleton("init_maximum", 1)
    domain("maximum")
    domain("exponential")
    singleton("init_lanes", numerics.lanes)
    domain("sum")
    singleton("finish", numerics.lanes)
    domain("normalize")
    if index != len(body):
        raise ValueError("row softmax has extra effects after its complete normalization")
    # The shared IR check bounds every emitted intermediate,
    # not just the final tensor byte extent. Exercise the same check before
    # publishing a scheduling receipt, including local addresses and loop ends.
    from merlin.sched.ir.integer_domain import check_integer_domain

    check_integer_domain(kernel, adapter.index_bits)
    return {
        "schema": "complete_row_softmax.v1",
        "kernel_sha256": kernel.digest(),
        "numerical_identity": numerics.identity,
        "configuration_identity": adapter.configuration_identity,
        "qualification_identity": adapter.qualification_identity,
        "problem": asdict(problem),
        "resources": asdict(resources),
        "static_calls": static_calls,
        "executed_commands": commands,
        "requested_read_bytes": reads,
        "requested_write_bytes": writes,
        "phase_commands": phase_counts,
        "private_state_bytes": sum(v.capacity * v.unit_bytes for v in storage),
        "physical_high_water_bytes": {
            b.backing: max((v.base + v.capacity) * v.unit_bytes for v in storage if v.backing == b.backing)
            for b in banks
            if any(v.backing == b.backing for v in storage)
        },
        "scope": "actual projected counted calls; independent primitive and final-byte qualification required",
        "timing": None,
    }
