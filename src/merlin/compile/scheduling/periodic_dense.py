"""Counted two-window dense proposals with exact bounded macro checks.

The caller supplies the primitive shape and explicit staged execution contract.
Paired descriptor/compute macros and common drains retain ordered source dots.
The checker proves complete loop domains using piecewise affine identities;
it neither selects target instructions nor grants execution/timing qualification.
"""

from __future__ import annotations

from merlin.compile.scheduling.staged_dense import StagedDenseContract, StagedDenseProblem

__all__ = ["build_periodic_staged_dense", "verify_periodic_staged_dense"]


def build_periodic_staged_dense(problem, contract, *, prefetch=True):
    """Two private panel-pair windows, with a bounded two-step periodic body.

    This target-neutral macro IR has an explicit paired descriptor/compute
    operation and full-drain operations. It creates no target qualification.
    """
    from merlin.sched.ir import Call, Kernel, Loop, TensorArg, Var, as_expr, select

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

    from merlin.sched.ir import (
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
        evaluate,
        select,
    )
    from merlin.sched.ir.kernel import check_structure

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
