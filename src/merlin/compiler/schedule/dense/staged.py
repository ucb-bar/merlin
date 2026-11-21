"""Finite staged contraction proposals under independently checked effects.

Unlike dense-v1, a descriptor issue is not synchronous completion. The target
must independently bind these logical effects to its current emitted program;
this library supplies no target lookup, adapter admission or timing authority.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass

# ---- staged_dense ------------------------------------------------------------------------------


@dataclass(frozen=True)
class StagedDenseContract:
    hardware: str
    software: str
    numerical_policy: str
    execution: str
    dim: int
    operand_rows: int
    accumulator_rows: int
    operand_bits: int
    primitive_bits: int
    accumulator_bits: int
    preload_visibility: str
    fence_completion: str
    readout: str
    external_buffers_disjoint: bool
    rhs_immutable: bool
    retained_weights: bool
    schema: str = "merlin.staged_dense.v1"

    def __post_init__(self):
        if self.schema != "merlin.staged_dense.v1":
            raise ValueError("unknown staged dense contract version")
        if any(
            type(value) is not str or not value or len(value) > 512
            for value in (self.hardware, self.software, self.numerical_policy, self.execution)
        ):
            raise ValueError("staged dense needs exact selected hardware/software/numerical/execution identities")
        if (
            any(
                type(value) is not int or value <= 0
                for value in (
                    self.dim,
                    self.operand_rows,
                    self.accumulator_rows,
                    self.operand_bits,
                    self.primitive_bits,
                    self.accumulator_bits,
                )
            )
            or self.dim > 256
        ):
            raise ValueError("staged dense dimensions and arithmetic widths must be explicit and bounded")
        if (
            self.primitive_bits < 2 * self.operand_bits + (self.dim - 1).bit_length()
            or self.accumulator_bits < self.primitive_bits
        ):
            raise ValueError("staged primitive arithmetic cannot contain the selected full signed panel")
        if (
            self.preload_visibility != "read_at_compute_consume"
            or self.fence_completion != "data_commands_preserve_pending_descriptor"
            or self.readout != "raw_accumulator_identity"
        ):
            raise ValueError("unsupported staged descriptor, completion or readout semantics")
        if any(
            type(value) is not bool
            for value in (self.external_buffers_disjoint, self.rhs_immutable, self.retained_weights)
        ):
            raise ValueError("staged ownership and retention require explicit Boolean premises")


@dataclass(frozen=True)
class StagedDenseProblem:
    m: int
    k: int
    n: int

    def __post_init__(self):
        if any(type(value) is not int or value <= 0 for value in (self.m, self.k, self.n)):
            raise ValueError("staged source dimensions must be positive static extents")


@dataclass(frozen=True)
class OperandLoad:
    role: str
    source_row: int
    source_col: int
    local_row: int
    rows: int
    cols: int


@dataclass(frozen=True)
class AccumulatorZero:
    local_row: int
    rows: int
    cols: int


@dataclass(frozen=True)
class DescriptorIssue:
    rhs_row: int | None
    accumulator_row: int
    m: int
    k: int
    n: int
    continuation: bool


@dataclass(frozen=True)
class ComputeConsume:
    lhs_row: int
    m: int
    k: int
    retain_weights: bool = False


@dataclass(frozen=True)
class AccumulatorReadout:
    accumulator_row: int
    output_row: int
    output_col: int
    rows: int
    cols: int


@dataclass(frozen=True)
class CompletionFence:
    pass


StagedDenseOperation = (
    OperandLoad | AccumulatorZero | DescriptorIssue | ComputeConsume | AccumulatorReadout | CompletionFence
)


@dataclass(frozen=True)
class StagedDenseRewrite:
    name: str
    implementation: Callable

    def __post_init__(self):
        if type(self.name) is not str or not self.name or len(self.name) > 128 or not callable(self.implementation):
            raise ValueError("staged rewrites need a bounded name and an executable implementation")


def eliminate_redundant_rhs_loads(operations, problem, contract):
    """Keep an immutable panel only while its exact private cells remain live.

    This does not allocate another panel, prefetch, remove a fence or pretend a
    pending descriptor completed. LHS loads invalidate aliased operand cells.
    """
    if not contract.rhs_immutable or not contract.external_buffers_disjoint:
        return operations
    cells, result = {}, []
    for operation in operations:
        if type(operation) is OperandLoad:
            stride = problem.k if operation.role == "lhs" else problem.n
            values = {
                (operation.local_row + r, c): (
                    operation.role,
                    (operation.source_row + r) * stride + operation.source_col + c,
                )
                for r in range(operation.rows)
                for c in range(operation.cols)
            }
            if operation.role == "rhs" and all(cells.get(cell) == value for cell, value in values.items()):
                continue
            cells.update(values)
        result.append(operation)
    return tuple(result)


def rewrite_staged_dense(operations, *, problem, contract, rewrites=(), max_commands=1_000_000, max_products=4 << 20):
    """Apply actual ordered candidate code; independently check every result.

    The original target owner must later bind and check the final emitted code.
    A successful logical proposal is not that final executable evidence.
    """

    if (
        type(operations) is not tuple
        or type(problem) is not StagedDenseProblem
        or type(contract) is not StagedDenseContract
        or type(rewrites) is not tuple
        or any(type(rewrite) is not StagedDenseRewrite for rewrite in rewrites)
    ):
        raise ValueError("staged source effects and ordered rewrites require explicit immutable tuples")
    original = operations
    context = (asdict(problem), asdict(contract))
    original_effects = [(type(operation).__name__, asdict(operation)) for operation in original]
    reports = [
        verify_staged_dense(
            original, problem=problem, contract=contract, max_commands=max_commands, max_products=max_products
        )
    ]
    for rewrite in rewrites:
        proposal = rewrite.implementation(operations, problem, contract)
        if context != (asdict(problem), asdict(contract)) or original_effects != [
            (type(operation).__name__, asdict(operation)) for operation in original
        ]:
            raise ValueError("candidate staged rewrite changed original source, effects or selected contracts")
        if type(proposal) is not tuple:
            raise ValueError("candidate staged rewrite must return actual immutable operations")
        reports.append(
            verify_staged_dense(
                proposal, problem=problem, contract=contract, max_commands=max_commands, max_products=max_products
            )
        )
        operations = proposal
    return operations, {
        "schema": "merlin.checked_staged_dense_proposal.v1",
        "rewrite_order": [rewrite.name for rewrite in rewrites],
        "before_commands": len(original),
        "after_commands": len(operations),
        "checks": reports,
        "final_emitted_program_check": "required from original qualified target owner",
        "timing_authority": "NONE",
    }


def order_output_tiles_n_before_m(operations, problem, contract):
    """Move complete output tiles to column-major traversal without new storage.

    Each movable tile includes its zero, full reduction, readout and following
    completion fence. No operation inside a tile is reordered, and no fence is
    removed. A later RHS-residency rewrite can use the resulting panel locality.
    If earlier rewrites removed a needed load, the independent checker must
    reject the resulting proposal; pass order is therefore observable.
    """
    del problem
    if not contract.external_buffers_disjoint or not contract.rhs_immutable:
        return operations
    if type(operations) is not tuple:
        raise ValueError("output-tile ordering requires immutable staged effects")
    cursor = 0
    while cursor < len(operations) and type(operations[cursor]) is CompletionFence:
        cursor += 1
    prefix = operations[:cursor]
    blocks = []
    while cursor < len(operations):
        if type(operations[cursor]) is CompletionFence:
            if not all(type(op) is CompletionFence for op in operations[cursor:]):
                raise ValueError("output-tile ordering cannot detach an inter-tile effect")
            break
        start = cursor
        if type(operations[cursor]) is not AccumulatorZero:
            raise ValueError("output-tile ordering requires complete independently zeroed tiles")
        cursor += 1
        while cursor < len(operations) and type(operations[cursor]) is not AccumulatorReadout:
            if type(operations[cursor]) is AccumulatorZero:
                raise ValueError("output-tile ordering cannot split overlapping accumulator lifetimes")
            cursor += 1
        if cursor + 1 >= len(operations) or type(operations[cursor + 1]) is not CompletionFence:
            raise ValueError("output-tile ordering requires complete readout and completion fence")
        readout = operations[cursor]
        cursor += 2
        blocks.append(((readout.output_col, readout.output_row), operations[start:cursor]))
    keys = [key for key, _ in blocks]
    if len(keys) != len(set(keys)):
        raise ValueError("output-tile ordering cannot duplicate a logical output tile")
    ordered = sorted(blocks, key=lambda block: block[0])
    return (*prefix, *(op for _, block in ordered for op in block), *operations[cursor:])


# ---- staged_dense_verification -----------------------------------------------------------------
# Independent finite staged data/descriptor/accumulator state induction.
#
# The target supplies checked effect projection from current actual emitted
# operands. No plan claim, pass implementation or configuration label proves
# these effects. This checker provides no asynchronous overlap or timing model.


def verify_staged_dense(operations, *, problem, contract, max_commands=1_000_000, max_products=4 << 20):
    if type(problem) is not StagedDenseProblem or type(contract) is not StagedDenseContract:
        raise ValueError("staged verification requires exact source and selected effect contracts")
    if not contract.external_buffers_disjoint:
        raise ValueError("staged verification lacks original external source/output ownership proof")
    if type(operations) is not tuple or any(
        type(value) is not int or value <= 0 for value in (max_commands, max_products)
    ):
        raise ValueError("staged verification requires bounded immutable actual effects")
    if len(operations) > max_commands or problem.m * problem.k * problem.n > max_products:
        raise ValueError("staged source exceeds its independent command/product checking budget")
    operands, accumulators, outputs = {}, {}, set()
    pending, stationary = None, None
    products = computes = loads = retained = fences_pending = 0
    cells_visited = 0

    def rectangle(row, rows, cols, capacity):
        if (
            any(type(value) is not int for value in (row, rows, cols))
            or row < 0
            or not 0 < rows <= contract.dim
            or not 0 < cols <= contract.dim
            or row + rows > capacity
        ):
            raise ValueError("actual staged operand rectangle exceeds selected primitive or assigned storage")

    for operation in operations:
        if type(operation) in (OperandLoad, AccumulatorZero, AccumulatorReadout):
            if type(operation.rows) is int and type(operation.cols) is int:
                cells_visited += operation.rows * operation.cols
            if cells_visited > 4 * max_products:
                raise ValueError("staged transfer/readout effects exceed independent checking budget")
        if type(operation) is CompletionFence:
            fences_pending += pending is not None
        elif type(operation) is OperandLoad:
            if operation.role not in {"lhs", "rhs"}:
                raise ValueError("staged load has unknown source arithmetic ownership")
            rectangle(operation.local_row, operation.rows, operation.cols, contract.operand_rows)
            rows, cols = (problem.m, problem.k) if operation.role == "lhs" else (problem.k, problem.n)
            if (
                any(type(value) is not int or value < 0 for value in (operation.source_row, operation.source_col))
                or operation.source_row + operation.rows > rows
                or operation.source_col + operation.cols > cols
            ):
                raise ValueError("actual staged DMA exceeds its current source tensor")
            for r in range(operation.rows):
                for c in range(operation.cols):
                    operands[operation.local_row + r, c] = (
                        operation.role,
                        (operation.source_row + r) * cols + operation.source_col + c,
                    )
            loads += 1
        elif type(operation) is AccumulatorZero:
            rectangle(operation.local_row, operation.rows, operation.cols, contract.accumulator_rows)
            for r in range(operation.rows):
                for c in range(operation.cols):
                    accumulators[operation.local_row + r, c] = None
        elif type(operation) is DescriptorIssue:
            if pending is not None or type(operation.continuation) is not bool:
                raise ValueError("actual staged descriptor overwrites pending ownership or has unknown continuation")
            rectangle(operation.accumulator_row, operation.m, operation.n, contract.accumulator_rows)
            if type(operation.k) is not int or not 0 < operation.k <= contract.dim:
                raise ValueError("staged descriptor reduction exceeds selected primitive")
            if operation.rhs_row is not None:
                rectangle(operation.rhs_row, operation.k, operation.n, contract.operand_rows)
            pending = operation
        elif type(operation) is ComputeConsume:
            if (
                pending is None
                or (operation.m, operation.k) != (pending.m, pending.k)
                or type(operation.retain_weights) is not bool
            ):
                raise ValueError("actual staged compute has no matching pending descriptor")
            rectangle(operation.lhs_row, operation.m, operation.k, contract.operand_rows)
            if operation.retain_weights:
                if not contract.retained_weights or stationary is None or stationary[:2] != (pending.k, pending.n):
                    raise ValueError("actual retained weights lack matching independently established ownership")
                weights = stationary[2]
                retained += 1
            else:
                if pending.rhs_row is None:
                    raise ValueError("actual staged preload has no RHS source cells")
                weights = tuple(
                    tuple(operands.get((pending.rhs_row + k, n)) for n in range(pending.n)) for k in range(pending.k)
                )
                stationary = pending.k, pending.n, weights
            for r in range(operation.m):
                for n in range(pending.n):
                    a = [operands.get((operation.lhs_row + r, k)) for k in range(operation.k)]
                    b = [weights[k][n] for k in range(operation.k)]
                    if None in a or None in b or a[0][0] != "lhs" or b[0][0] != "rhs":
                        raise ValueError("actual staged compute reads unowned/clobbered source cells")
                    m, k0 = divmod(a[0][1], problem.k)
                    kb, out_n = divmod(b[0][1], problem.n)
                    if (
                        kb != k0
                        or k0 + operation.k > problem.k
                        or a != [("lhs", m * problem.k + k0 + k) for k in range(operation.k)]
                        or b != [("rhs", (k0 + k) * problem.n + out_n) for k in range(operation.k)]
                    ):
                        raise ValueError("actual staged product cells differ from current source geometry")
                    location = pending.accumulator_row + r, n
                    if (
                        location not in accumulators
                        or accumulators[location] != (None if k0 == 0 else (m, out_n, k0))
                        or (k0 > 0 and not pending.continuation)
                    ):
                        raise ValueError(
                            "actual staged compute loses zero initialization or exact accumulator continuation"
                        )
                    accumulators[location] = m, out_n, k0 + operation.k
                    products += operation.k
            pending = None
            computes += 1
        elif type(operation) is AccumulatorReadout:
            rectangle(operation.accumulator_row, operation.rows, operation.cols, contract.accumulator_rows)
            if (
                any(type(value) is not int or value < 0 for value in (operation.output_row, operation.output_col))
                or operation.output_row + operation.rows > problem.m
                or operation.output_col + operation.cols > problem.n
            ):
                raise ValueError("actual staged readout exceeds current complete source outputs")
            for r in range(operation.rows):
                for c in range(operation.cols):
                    target = operation.output_row + r, operation.output_col + c
                    if (
                        accumulators.get((operation.accumulator_row + r, c)) != (*target, problem.k)
                        or target in outputs
                    ):
                        raise ValueError("actual staged readout omits, duplicates or changes source products")
                    outputs.add(target)
        else:
            raise ValueError("unknown staged effects cannot be erased or treated as a fence")
    if pending is not None or len(outputs) != problem.m * problem.n or products != problem.m * problem.k * problem.n:
        raise ValueError("actual staged final completion leaves pending descriptors or incomplete required source work")
    return {
        "status": "verified_finite_staged_source_effects",
        "products": products,
        "outputs": len(outputs),
        "computes": computes,
        "operand_loads": loads,
        "retained_computes": retained,
        "fences_preserving_pending_descriptor": fences_pending,
        "timing_authority": "NONE",
    }


# ---- periodic_dense ----------------------------------------------------------------------------
# Counted two-window dense proposals with exact bounded macro checks.
#
# The caller supplies the primitive shape and explicit staged execution contract.
# Paired descriptor/compute macros and common drains retain ordered source dots.
# The checker proves complete loop domains using piecewise affine identities;
# it neither selects target instructions nor grants execution/timing qualification.


def build_periodic_staged_dense(problem, contract, *, prefetch=True):
    """Two private panel-pair windows, with a bounded two-step periodic body.

    This target-neutral macro IR has an explicit paired descriptor/compute
    operation and full-drain operations. It creates no target qualification.
    """
    from merlin.compiler.kernel_ir.ir import Call, Kernel, Loop, TensorArg, Var, as_expr, select

    if (
        type(problem) is not StagedDenseProblem
        or type(contract) is not StagedDenseContract
        or type(prefetch) is not bool
    ):
        raise ValueError("typed periodic source/contract and explicit prefetch policy required")
    d = contract.dim
    m, k, n = problem.m, problem.k, problem.n
    mt, kt, nt = ((x + d - 1) // d for x in (m, k, n))
    pairs, remainder = divmod(kt - 1, 2)
    mi, ni, pi = Var("mi"), Var("ni"), Var("pi")
    mr = select("mi", mt - 1, m - (mt - 1) * d, d)
    nr = select("ni", nt - 1, n - (nt - 1) * d, d)
    lastk = k - (kt - 1) * d

    def call(name, **args):
        return Call(name, tuple((key, as_expr(value)) for key, value in args.items()))

    fence = call("drain")

    def loads(index, slot, kr):
        return (
            call("load_lhs", source_row=mi * d, source_col=index * d, local_row=slot * 2 * d, rows=mr, cols=kr),
            call("load_rhs", source_row=index * d, source_col=ni * d, local_row=slot * 2 * d + d, rows=kr, cols=nr),
        )

    def compute(slot, kr, continuation):
        return call(
            "paired_compute",
            lhs_row=slot * 2 * d,
            rhs_row=slot * 2 * d + d,
            acc_row=0,
            rows=mr,
            k=kr,
            cols=nr,
            continuation=continuation,
        )

    def stage(index, slot, cont, next_kr):
        c = compute(slot, d, cont)
        future = loads(index + 1, 1 - slot, next_kr)
        return (c, *future, fence) if prefetch else (c, fence, *future, fence)

    body = [*loads(0, 0, lastk if kt == 1 else d), fence]
    if pairs:
        next_tail = select("pi", pairs - 1, lastk, d) if kt % 2 else as_expr(d)
        period = (*stage(pi * 2, 0, select("pi", 0, 0, 1), d), *stage(pi * 2 + 1, 1, 1, next_tail))
        body.append(Loop("pi", pairs, period))
    if remainder:
        body.extend(stage(as_expr(2 * pairs), 0, int(pairs > 0), lastk))
    body.extend(
        (
            compute((kt - 1) % 2, lastk, int(kt > 1)),
            fence,
            call("store", acc_row=0, output_row=mi * d, output_col=ni * d, rows=mr, cols=nr),
            fence,
        )
    )
    return Kernel(
        "periodic_staged_dense",
        (
            TensorArg("lhs", (m, k), "i8", "read"),
            TensorArg("rhs", (k, n), "i8", "read"),
            TensorArg("output", (m, n), "i32", "write"),
        ),
        (Loop("mi", mt, (Loop("ni", nt, tuple(body)),)),),
    )


def verify_periodic_staged_dense(
    kernel, problem, contract, *, prefetch=True, max_static_calls=64, max_affine_basis=512
):
    """Induct over actual two-step loop IR without expanding dynamic commands.

    Each macro operand is proved equal over its complete loop domain. Equality
    is exact piecewise-affine algebra: equality-select exceptional points split
    the domain; each remaining affine region is checked on a Cartesian affine
    basis (at most two distinct values per axis). This is not sampled testing.
    The period's two window transitions establish the induction step, while
    separate prologue/remainder/epilogue checks establish both boundaries.
    """
    from itertools import product

    from merlin.compiler.kernel_ir.ir import (
        Call,
        Const,
        Kernel,
        Loop,
        Scaled,
        Select,
        Sum,
        TensorArg,
        Var,
        as_expr,
        check_structure,
        evaluate,
        select,
    )

    if (
        type(kernel) is not Kernel
        or type(problem) is not StagedDenseProblem
        or type(contract) is not StagedDenseContract
        or type(prefetch) is not bool
    ):
        raise ValueError("periodic verification requires exact typed artifacts")
    if any(type(x) is not int or x <= 0 for x in (max_static_calls, max_affine_basis)):
        raise ValueError("explicit static proof budgets required")
    d = contract.dim
    m, k, n = problem.m, problem.k, problem.n
    if (
        (contract.operand_bits, contract.accumulator_bits) != (8, 32)
        or contract.operand_rows < 4 * d
        or contract.accumulator_rows < d
        or not contract.external_buffers_disjoint
    ):
        raise ValueError("two disjoint operand-pair windows and complete raw output required")
    if max(m * k, k * n, m * n * 4) >= 1 << 63:
        raise ValueError("counted host byte offsets exceed signed i64")
    mt, kt, nt = ((x + d - 1) // d for x in (m, k, n))
    pairs, remainder = divmod(kt - 1, 2)
    expected = (
        TensorArg("lhs", (m, k), "i8", "read"),
        TensorArg("rhs", (k, n), "i8", "read"),
        TensorArg("output", (m, n), "i32", "write"),
    )
    if kernel.args != expected or kernel.attrs or check_structure(kernel):
        raise ValueError("complete periodic source/IR boundary differs")

    def loop(statement, name, extent):
        if type(statement) is not Loop or statement.var != name or statement.extent != extent:
            raise ValueError("actual loop bound differs from complete source partition")
        return statement.body

    if len(kernel.body) != 1:
        raise ValueError("unexpected work outside output loops")
    outer = loop(kernel.body[0], "mi", mt)
    if len(outer) != 1:
        raise ValueError("unexpected work outside column loop")
    body = loop(outer[0], "ni", nt)
    mi, ni, pi = Var("mi"), Var("ni"), Var("pi")
    domains = {"mi": mt, "ni": nt}
    mr = select("mi", mt - 1, m - (mt - 1) * d, d)
    nr = select("ni", nt - 1, n - (nt - 1) * d, d)
    lastk = k - (kt - 1) * d
    proved = sites = 0

    def equal(actual, expected, scope):
        nonlocal proved
        a, b = as_expr(actual), as_expr(expected)
        cuts = {name: set() for name in scope}

        def inspect(value):
            if type(value) is Const:
                if type(value.value) is not int:
                    raise ValueError("noninteger constant")
            elif type(value) is Var:
                if value.name not in scope:
                    raise ValueError("unbound operand index")
            elif type(value) is Sum:
                for child in value.terms:
                    inspect(child)
            elif type(value) is Scaled:
                if type(value.factor) is not int:
                    raise ValueError("nonaffine scale")
                inspect(value.expr)
            elif type(value) is Select:
                if value.var not in scope or type(value.equals) is not int:
                    raise ValueError("unknown equality partition")
                cuts[value.var].add(value.equals)
                inspect(value.then)
                inspect(value.other)
            else:
                raise ValueError("operand outside piecewise affine language")

        inspect(a)
        inspect(b)
        choices = []
        for name, extent in scope.items():
            exceptions = sorted(x for x in cuts[name] if 0 <= x < extent)
            basis = []
            cursor = 0
            while cursor < extent and len(basis) < 2:
                if cursor not in cuts[name]:
                    basis.append(cursor)
                cursor += 1
            choices.append((name, tuple(exceptions + basis)))
        count = 1
        for _, points in choices:
            count *= len(points)
        if count > max_affine_basis:
            raise ValueError("affine proof partition budget exceeded")
        for values in product(*(points for _, points in choices)):
            env = dict(zip((name for name, _ in choices), values, strict=True))
            if evaluate(a, env) != evaluate(b, env):
                raise ValueError("actual operand violates periodic source/lifetime invariant")
        proved += 1

    def call(statement, name, args, scope):
        nonlocal sites
        if (
            type(statement) is not Call
            or statement.instr != name
            or any(
                (
                    statement.unit,
                    statement.produces,
                    statement.awaits,
                    statement.stages,
                    statement.sets,
                    statement.assumes,
                )
            )
        ):
            raise ValueError("unexpected periodic instruction or unmodeled effect")
        if len(statement.args) != len(args) or set(dict(statement.args)) != set(args):
            raise ValueError("actual macro operand roster differs")
        for key, value in statement.args:
            equal(value, args[key], scope)
        sites += 1
        if sites > max_static_calls:
            raise ValueError("static macro budget exceeded")

    def load_pair(rows, index, slot, kr, scope):
        if len(rows) != 2:
            raise ValueError("incomplete panel pair")
        call(
            rows[0],
            "load_lhs",
            dict(source_row=mi * d, source_col=index * d, local_row=slot * 2 * d, rows=mr, cols=kr),
            scope,
        )
        call(
            rows[1],
            "load_rhs",
            dict(source_row=index * d, source_col=ni * d, local_row=slot * 2 * d + d, rows=kr, cols=nr),
            scope,
        )

    def compute(statement, slot, kr, cont, scope):
        call(
            statement,
            "paired_compute",
            dict(lhs_row=slot * 2 * d, rhs_row=slot * 2 * d + d, acc_row=0, rows=mr, k=kr, cols=nr, continuation=cont),
            scope,
        )

    def stage(rows, index, slot, cont, nextkr, scope):
        if len(rows) != (4 if prefetch else 5):
            raise ValueError("periodic stage boundary differs")
        compute(rows[0], slot, d, cont, scope)
        if prefetch:
            load_pair(rows[1:3], index + 1, 1 - slot, nextkr, scope)
            call(rows[3], "drain", {}, scope)
        else:
            call(rows[1], "drain", {}, scope)
            load_pair(rows[2:4], index + 1, 1 - slot, nextkr, scope)
            call(rows[4], "drain", {}, scope)

    load_pair(body[:2], as_expr(0), 0, lastk if kt == 1 else d, domains)
    call(body[2], "drain", {}, domains)
    cursor = 3
    stage_size = 4 if prefetch else 5
    if pairs:
        period = loop(body[cursor], "pi", pairs)
        cursor += 1
        if len(period) != 2 * stage_size:
            raise ValueError("not one complete two-window period")
        scope = {**domains, "pi": pairs}
        tail = select("pi", pairs - 1, lastk, d) if kt % 2 else as_expr(d)
        stage(period[:stage_size], pi * 2, 0, select("pi", 0, 0, 1), d, scope)
        stage(period[stage_size:], pi * 2 + 1, 1, 1, tail, scope)
    if remainder:
        stage(body[cursor : cursor + stage_size], as_expr(2 * pairs), 0, int(pairs > 0), lastk, domains)
        cursor += stage_size
    if len(body) != cursor + 4:
        raise ValueError("unexpected work outside final stage")
    compute(body[cursor], (kt - 1) % 2, lastk, int(kt > 1), domains)
    call(body[cursor + 1], "drain", {}, domains)
    call(body[cursor + 2], "store", dict(acc_row=0, output_row=mi * d, output_col=ni * d, rows=mr, cols=nr), domains)
    call(body[cursor + 3], "drain", {}, domains)
    return {
        "schema": "merlin.periodic_staged_dense_check.v1",
        "status": "proved_periodic_macro_effects_conditional_on_target_contract",
        "kernel_sha256": kernel.digest(),
        "static_calls": sites,
        "operand_equalities": proved,
        "dynamic_products": m * k * n,
        "complete_outputs": m * n,
        "periods_per_output_tile": pairs,
        "induction": (
            "ready current window -> paired compute plus disjoint next-panel loads -> "
            "common drain -> ready next window and ordered accumulator prefix"
        ),
        "source_partition": (
            "Cartesian output tiles with exact tails; K=2*periods+remainder+1; "
            "first overwrite then ordered continuation"
        ),
        "prefetch": prefetch,
        "max_pending_data_operations": 3 if prefetch and kt > 1 else 2,
        "timing_authority": "NONE",
        "target_binding": "actual emitted macros and command fields require independent native qualification",
    }


__all__ = [
    "build_periodic_staged_dense",
    "verify_periodic_staged_dense",
]
