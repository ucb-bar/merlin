"""Bind exact completed producer rectangles after counted input pipelining.

This consumes the original producer's independently checked nonoverlapping
readouts. It does not move a command or invent completion. Every selected epoch
ends with its original full drain, including all future operand staging.
"""

from dataclasses import replace
from itertools import product
from math import prod

from merlin.sched.ir import Call, Const, Loop
from merlin.sched.ir.expr import free_vars

from .dense_contracts import TiledIntegerStore
from .dense_panel_pipeline import DRAIN
from .dense_readout_overlap import _affine_window_bounds
from .panel_tile_loads import _offset_form


def compose_completed_groups(kernel, project, *, rows, columns, row_span, column_span,
                             rewrite, end_bound):
    """Return original control with one consumer per proved complete Q1 epoch."""
    found = 0

    def difference(value, base):
        # Cancel the enclosing period/column terms before interval bounds;
        # independent bounds on (period + local) - period lose correlation.
        constant, terms = _offset_form(value + base * -1)
        result = Const(constant)
        for term, coefficient in terms.items():
            result += term * coefficient
        return result

    def observations(body, local):
        result = []
        for statement in body:
            if type(statement) is Loop:
                if statement.var in local:
                    raise ValueError('completed producer reuses an index scope')
                result.extend(observations(statement.body, {**local, statement.var: (0, statement.extent - 1)}))
            elif statement != Call(DRAIN, ()):
                operation = project(statement)
                if type(operation) is TiledIntegerStore:
                    result.append((operation, local))
        return result

    def origin(observed, field, outer):
        common, lower = None, None
        for operation, local in observed:
            constant, terms = _offset_form(getattr(operation, field))
            global_terms, local_terms = {}, {}
            for term, coefficient in terms.items():
                names = free_vars(term)
                if names <= outer.keys():
                    global_terms[term] = coefficient
                elif names <= local.keys():
                    local_terms[term] = coefficient
                else:
                    raise ValueError('completed producer has a mixed or unbound group origin')
            if common is not None and common != global_terms:
                raise ValueError('completed producer readouts disagree on their group origin')
            common = global_terms
            value = Const(constant)
            for term, coefficient in local_terms.items():
                value += term * coefficient
            low = _affine_window_bounds(value, local)[0]
            lower = low if lower is None else min(lower, low)
        value = Const(lower)
        for term, coefficient in common.items():
            value += term * coefficient
        return value

    def walk(body, outer):
        nonlocal found
        result = []
        for statement in body:
            if type(statement) is not Loop:
                result.append(statement)
                continue
            if statement.var in outer:
                raise ValueError('completed producer has ambiguous counted control')
            observed = observations(statement.body, {statement.var: (0, 0)}) if (
                statement.extent == 1 and statement.body and statement.body[-1] == Call(DRAIN, ())
                and not statement.iter_args and not statement.yields and not statement.results) else []
            if not observed:
                result.append(replace(statement, body=walk(statement.body,
                    {**outer, statement.var: (0, statement.extent - 1)})))
                continue
            base_row, base_column = (origin(observed, field, outer) for field in ('dram_row', 'dram_col'))
            height, width, area = 0, 0, 0
            for operation, local in observed:
                domains = {**outer, **local}
                height = max(height, end_bound(difference(operation.dram_row, base_row), operation.rows, domains))
                width = max(width, end_bound(difference(operation.dram_col, base_column), operation.cols, domains))
                # Only one bounded local output panel is enumerated. The
                # inference's outer M/N/period domains stay symbolic. Together
                # with the checked producer's no-overlap proof, exact area and
                # bounds establish a complete rectangle, including ragged rows.
                if prod(high - low + 1 for low, high in local.values()) > 4096:
                    raise ValueError('completed output panel exceeds its local checking budget')
                for values in product(*(range(low, high + 1) for low, high in local.values())):
                    selected = dict(outer, **dict(zip(local, ((v, v) for v in values))))
                    h, w = (_affine_window_bounds(expr, selected) for expr in (operation.rows, operation.cols))
                    if h[0] != h[1] or w[0] != w[1] or min(h[0], w[0]) <= 0:
                        raise ValueError('completed Q1 view needs a separate exact tail epoch')
                    area += h[0] * w[0]
            if (not 0 < height <= row_span or width != column_span or area != height * width
                    or _affine_window_bounds(base_row, outer)[0] < 0
                    or _affine_window_bounds(base_column, outer)[0] < 0
                    or end_bound(base_row, Const(height), outer) > rows
                    or end_bound(base_column, Const(width), outer) > columns):
                raise ValueError('completed Q1 readouts do not cover one exact original panel')
            contents = rewrite(statement.body, base_row, base_column,
                               {**outer, statement.var: (0, 0)}, height)
            result.append(replace(statement, body=(*contents, Call('connected_loaded_sum_group',
                (('output_row', base_row), ('output_column', base_column),
                 ('rows', Const(height)), ('columns', Const(column_span)))))))
            found += len(observed)
        return tuple(result)

    return walk(kernel.body, {}), found
