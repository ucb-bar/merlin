"""Construct exact checking projections for dense panel epochs."""

from __future__ import annotations

from dataclasses import replace

from merlin.compiler.kernel_ir.ir import (
    Call,
    Const,
    Kernel,
    Loop,
    Ptr,
    Scaled,
    Select,
    Sum,
    Var,
    add,
    check_integer_domain,
    check_structure,
)
from merlin.compiler.schedule.blocks import Compute, Load, Preload
from merlin.compiler.schedule.dense_panel_ir import DRAIN as DRAIN
from merlin.compiler.schedule.dense_panel_ir import _calls as _calls


def _fission_noninterfering_epoch_loads(kernel, adapter):
    """Exact checking representation AFTER actual epoch noninterference proof.

    Stable filtering distributes through positive counted loops: each original
    command and its full domain occurs once in its original phase subsequence.
    Within a mixed epoch all actual DMA destinations were independently proved
    disjoint from each other and every compute read. Assigned operand/ACC views
    are separately owned. Thus loads commute with computations, while every
    reduction, descriptor pair and same-phase command order remains intact.
    Actual emission MUST retain the original Kernel; this is no new target law.
    """
    changes = []

    def phase(body, kinds):
        result = []
        for statement in body:
            if type(statement) is Loop:
                selected = phase(statement.body, kinds)
                if selected:
                    result.append(replace(statement, body=selected))
            elif type(adapter.project(statement)) in kinds:
                result.append(statement)
        return tuple(result)

    def walk(body):
        result = []
        for statement in body:
            if type(statement) is Loop:
                if statement.body and statement.body[-1] == Call(DRAIN, ()):
                    content = statement.body[:-1]
                    kinds = {type(adapter.project(c)) for c in _calls(content)}
                    if kinds == {Load, Preload, Compute}:
                        separated = (*phase(content, {Preload, Compute}), *phase(content, {Load}), Call(DRAIN, ()))
                        if separated != statement.body:
                            changes.append(statement.var)
                            statement = replace(statement, body=separated)
                else:
                    statement = replace(statement, body=walk(statement.body))
            result.append(statement)
        return tuple(result)

    rewritten = replace(kernel, body=walk(kernel.body))
    if check_structure(rewritten):
        raise ValueError("stable phase fission changed actual source domains or descriptor scope")
    return rewritten, changes


def _collapse_compute_radices(kernel, adapter):
    """Exact checking-only bijection for adjacent pure compute loop scopes.

    Every operand must depend on the two indices solely as outer*B+inner,
    where B is the complete inner extent. Flattening then preserves the full
    lexicographic call stream, including physical descriptor adjacency. This
    supplies no commutation or target capability; actual epoch hazards must
    already have been proved before using this checking representation.
    """
    from merlin.compiler.kernel_ir.ir import Expr, free_vars
    from merlin.compiler.schedule.tile_load_coalescing import _plain

    if type(kernel) is not Kernel or check_structure(kernel):
        raise ValueError("compute radix checking requires unshadowed complete counted scopes")
    changes = []

    def pure(body):
        if len(body) == 1 and type(body[0]) is Loop:
            return pure(body[0].body)
        if len(body) != 2 or any(type(c) is not Call or c.instr == DRAIN or not _plain(c) for c in body):
            return False
        try:
            return [type(adapter.project(c)) for c in body] == [Preload, Compute]
        except (ValueError, KeyError):
            return False

    def collapse(outer, inner):
        names = {outer.var, inner.var}

        def form(value):
            if type(value) is Const:
                return value.value, {}
            if type(value) is Var:
                return 0, {value: 1}
            if type(value) is Select:
                # A select on either joined coordinate carries tail/guard
                # information which this affine inverse cannot represent.
                if names & free_vars(value):
                    raise ValueError("mixed-radix dependent guard or partial tail")
                form(value.then)
                form(value.other)
                return 0, {value: 1}
            if type(value) is Scaled:
                constant, terms = form(value.expr)
                return constant * value.factor, {v: n * value.factor for v, n in terms.items()}
            if type(value) is Sum:
                constant, terms = 0, {}
                for item in value.terms:
                    c, ts = form(item)
                    constant += c
                    for atom, coefficient in ts.items():
                        terms[atom] = terms.get(atom, 0) + coefficient
                return constant, {v: n for v, n in terms.items() if n}
            raise ValueError("unknown compute scalar expression")

        def expression(value):
            constant, terms = form(value)
            a, b = terms.pop(Var(outer.var), 0), terms.pop(Var(inner.var), 0)
            if a != inner.extent * b or any(names & free_vars(atom) for atom in terms):
                raise ValueError("scalar operand distinguishes mixed-radix coordinates")
            return add(
                Const(constant), *(atom * coefficient for atom, coefficient in terms.items()), Var(outer.var) * b
            )

        def operand(value):
            if isinstance(value, Expr):
                return expression(value)
            if type(value) is Ptr:
                return replace(value, offset=expression(value.offset))
            if value is None or type(value) in (int, float, str, bool):
                return value
            raise ValueError("unknown compute scalar operand")

        def body(statements):
            result = []
            for statement in statements:
                if type(statement) is Loop:
                    if statement.var in names:
                        raise ValueError("mixed-radix rewrite would capture a descendant binder")
                    result.append(replace(statement, body=body(statement.body)))
                else:
                    result.append(replace(statement, args=tuple((key, operand(v)) for key, v in statement.args)))
            return tuple(result)

        extent = outer.extent * inner.extent
        if any(type(v) is not int or v <= 0 for v in (outer.extent, inner.extent)) or extent >= 1 << (
            adapter.index_bits - 1
        ):
            raise ValueError("joined counted extent exceeds original signed index domain")
        return replace(outer, extent=extent, body=body(inner.body))

    def walk(body):
        result = []
        for statement in body:
            if type(statement) is Loop:
                statement = replace(statement, body=walk(statement.body))
                if len(statement.body) == 1 and type(inner := statement.body[0]) is Loop and pure(inner.body):
                    try:
                        joined = collapse(statement, inner)
                    except ValueError:
                        pass
                    else:
                        changes.append(
                            dict(
                                outer=statement.var,
                                inner=inner.var,
                                outer_extent=statement.extent,
                                inner_extent=inner.extent,
                                joined_extent=joined.extent,
                                forward="t=outer*inner_extent+inner",
                                inverse="outer=t//inner_extent; inner=t%inner_extent",
                                proof="every scalar operand: outer_coefficient=inner_extent*inner_coefficient",
                            )
                        )
                        statement = joined
            result.append(statement)
        return tuple(result)

    rewritten = replace(kernel, body=walk(kernel.body))
    if check_structure(rewritten):
        raise ValueError("compute radix checking changed original binder custody")
    if changes:
        check_integer_domain(rewritten, adapter.index_bits)
    return rewritten, changes


# Preserve the original public lookup and pickle identities.
for _definition in (
    _fission_noninterfering_epoch_loads,
    _collapse_compute_radices,
):
    _definition.__module__ = "merlin.compiler.schedule.dense_panel_pipeline"
del _definition
