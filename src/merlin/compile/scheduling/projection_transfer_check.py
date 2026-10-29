"""Independently join checked virtual transfers to projected final operands.

This finite evidence family compares original Kernel rectangles with operands
read from emitted LLVM. Neither a caller's pointer declaration nor matching
shapes alone supplies the join. Existing source/dense/effect/readout contracts
and normal native qualification remain independent requirements.
"""

from dataclasses import dataclass

from merlin.perf.finite_pointer_trace import ArgumentAddress

from .projection_views import SharedProjectionViews


@dataclass(frozen=True)
class VirtualProjectionTransfer:
    role: str
    row: int
    column: int
    rows: int
    columns: int
    local_row: int


@dataclass(frozen=True)
class PhysicalProjectionTransfer:
    role: str
    pointer: ArgumentAddress
    rows: int
    columns: int
    row_stride_bytes: int
    element_bytes: int
    local_row: int
    readout: tuple = ()


def verify_projection_transfers(
    views,
    expected,
    observed,
    *,
    lhs_argument,
    rhs_arguments,
    output_arguments,
    lhs_element_bytes,
    output_readouts,
    source_module,
    max_transfers=100_000,
    max_output_rows=100_000,
):
    """Compare every final rectangle, exact pointer identity, pitch and readout.

    Expected transfers must be enumerated from the actual independently checked
    source Kernel. Observed transfers must come from the emitted command/config
    operands. This observer supplies no physical alias grant and does not infer
    persistent configuration or completion from issue order.
    """
    if (
        type(views) is not SharedProjectionViews
        or type(expected) is not tuple
        or type(observed) is not tuple
        or type(max_transfers) is not int
        or not 1 <= max_transfers <= 100_000
        or len(expected) != len(observed)
        or not len(expected) <= max_transfers
        or any(type(e) is not VirtualProjectionTransfer for e in expected)
        or any(type(o) is not PhysicalProjectionTransfer for o in observed)
        or type(rhs_arguments) is not tuple
        or type(output_arguments) is not tuple
        or len(rhs_arguments) != len(views.source.widths)
        or len(output_arguments) != len(views.source.widths)
        or type(output_readouts) is not tuple
        or len(output_readouts) != len(views.source.widths)
        or any(type(r) is not tuple or not r for r in output_readouts)
        or type(lhs_element_bytes) is not int
        or lhs_element_bytes <= 0
        or type(max_output_rows) is not int
        or not 1 <= max_output_rows <= 100_000
        or views.source.m > max_output_rows
    ):
        raise ValueError("bounded actual source/physical transfer sequences and complete original ABI required")
    views.verify_current(source_module)
    arguments = (lhs_argument, *rhs_arguments, *output_arguments)
    if any(type(i) is not int or i < 0 for i in arguments) or len(set(arguments)) != len(arguments):
        raise ValueError("original projection ABI pointer identities must be distinct")
    outputs = {}
    counts = {"lhs": 0, "weight": 0, "output": 0}
    requested_bytes = dict.fromkeys(counts, 0)
    for e, o in zip(expected, observed, strict=True):
        if (
            e.role not in counts
            or e.role != o.role
            or type(o.pointer) is not ArgumentAddress
            or type(o.pointer.argument) is not int
            or type(o.pointer.offset) is not int
            or any(
                type(v) is not int
                for v in (
                    e.row,
                    e.column,
                    e.rows,
                    e.columns,
                    e.local_row,
                    o.rows,
                    o.columns,
                    o.local_row,
                    o.row_stride_bytes,
                    o.element_bytes,
                )
            )
            or e.local_row < 0
            or (e.rows, e.columns, e.local_row) != (o.rows, o.columns, o.local_row)
        ):
            raise ValueError("actual projection transfer role, local address or extent differs from checked source")
        if e.role == "lhs":
            if not (
                0 <= e.row < views.source.m
                and 0 < e.rows <= views.source.m - e.row
                and 0 <= e.column < views.source.k
                and 0 < e.columns <= views.source.k - e.column
            ):
                raise ValueError("original shared LHS rectangle exceeds source allocation")
            pointer = ArgumentAddress(lhs_argument, (e.row * views.source.k + e.column) * lhs_element_bytes)
            stride, unit, readout = views.source.k * lhs_element_bytes, lhs_element_bytes, ()
        else:
            r = views.rectangle(e.role, e.row, e.column, e.rows, e.columns)
            pointer = ArgumentAddress(
                (rhs_arguments if e.role == "weight" else output_arguments)[r.member], r.byte_offset
            )
            stride = r.row_stride_bytes
            unit = views.weight_element_bytes if e.role == "weight" else views.output_element_bytes[r.member]
            readout = () if e.role == "weight" else output_readouts[r.member]
            if e.role == "output":
                for row in range(e.row, e.row + e.rows):
                    outputs.setdefault((r.member, row), []).append((r.column, r.column + e.columns))
        if o.pointer != pointer or o.row_stride_bytes != stride or o.element_bytes != unit or o.readout != readout:
            raise ValueError(
                "final pointer/byte offset, physical pitch, element format or readout differs from original member"
            )
        counts[e.role] += 1
        requested_bytes[e.role] += o.rows * o.columns * o.element_bytes
    for member, width in enumerate(views.source.widths):
        for row in range(views.source.m):
            cursor = 0
            for first, last in sorted(outputs.get((member, row), ())):
                if first != cursor:
                    raise ValueError("projection outputs overlap or leave an original live output uncovered")
                cursor = last
            if cursor != width:
                raise ValueError("projection leaves an original live output uncovered")
    m, k, columns = views.source.m, views.source.k, views.columns
    logical_bytes = dict(
        lhs=m * k * lhs_element_bytes,
        weight=k * columns * views.weight_element_bytes,
        output=m * sum(width * unit for width, unit in zip(views.source.widths, views.output_element_bytes)),
    )
    return dict(
        schema="merlin.projection_transfer_check.v1",
        status="pass",
        transfers=counts,
        output_elements=views.source.m * views.columns,
        source_contraction_macs=m * k * columns,
        logical_tensor_bytes=logical_bytes,
        requested_transfer_bytes=requested_bytes,
        shared_input_request_ratio=dict(numerator=requested_bytes["lhs"], denominator=logical_bytes["lhs"]),
        traffic_scope=(
            "complete descriptor payload bytes; logical tensors counted once each, not cache misses, "
            "physical bus traffic or a DRAM lower bound"
        ),
        elapsed_cycles=None,
        utilization=None,
        source_identity=views.source.source_ir_sha256,
        source="actual checked virtual Kernel versus projected emitted operands",
        alias_and_local_lifetime="original source/runtime and dense effect proofs remain required",
        configuration_and_completion=(
            "readout snapshots checked; actual context selection and completion remain adapter obligations"
        ),
        native_execution=False,
    )


def source_projection_transfers(kernel, project, *, max_calls=100_000):
    """Enumerate transfers from the ACTUAL checked Kernel's counted indices.

    The original source adapter projects each concrete original Call and its
    evaluated arguments; arbitrary effects cannot silently become a transfer.
    Existing source/dense verification must still revalidate the Kernel. This
    bounded observer is useful for small complete final-code projection checks,
    not a replacement for compact proofs on huge schedules.
    """
    from merlin.sched.ir import Call, Const, Expr, Kernel, Loop, Ptr
    from merlin.sched.ir.expr import evaluate
    from merlin.sched.ir.kernel import check_structure

    if (
        type(kernel) is not Kernel
        or check_structure(kernel)
        or not callable(project)
        or type(max_calls) is not int
        or not 1 <= max_calls <= 100_000
    ):
        raise ValueError("bounded actual source Kernel and original projection callback required")
    out = []
    calls = 0
    iterations = 0

    def body(statements, env):
        nonlocal calls, iterations
        for statement in statements:
            if type(statement) is Loop:
                for i in range(statement.extent):
                    iterations += 1
                    if iterations > max_calls:
                        raise ValueError("actual source loop enumeration budget exhausted")
                    body(statement.body, {**env, statement.var: i})
            elif type(statement) is Call:
                calls += 1
                if calls > max_calls:
                    raise ValueError("actual source transfer enumeration budget exhausted")
                values = {
                    name: evaluate(v, env)
                    if isinstance(v, Expr)
                    else Ptr(v.tensor, Const(evaluate(v.offset, env)))
                    if type(v) is Ptr
                    else v
                    for name, v in statement.args
                }
                projected = project(statement, values)
                if projected is not None:
                    if type(projected) is not VirtualProjectionTransfer:
                        raise ValueError("original source effect projection returned an untyped transfer")
                    out.append(projected)
            else:
                raise ValueError("original source Kernel contains unknown counted control")

    body(kernel.body, {})
    return tuple(out)
