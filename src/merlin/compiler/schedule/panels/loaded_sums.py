"""A completed quantized producer panel followed by its original loaded sum.

Q1 is materialized in an explicitly owned bounded byte workspace. Only a real
full producer drain permits its reader and ACC reuse. The existing loaded-sum
owner retains Q2, its three phase drains and numerical proof. This alternative
does not commute quantization, saturation or activation with the addition.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from itertools import product
from math import prod

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
    check_structure,
    free_vars,
)
from merlin.compiler.schedule.dense.compact import _digest
from merlin.compiler.schedule.dense.contracts import TiledIntegerStore
from merlin.compiler.schedule.dense.loaded_sums import LoadedSumEpochPlan, emit_loaded_sum_epochs
from merlin.compiler.schedule.panels.pipeline import DRAIN, _calls, _loops, _remap
from merlin.compiler.schedule.panels.readouts import _affine_window_bounds
from merlin.compiler.schedule.panels.tile_loads import _offset_form

# ---- dense_loaded_sum_composition --------------------------------------------------------------


def _window_end_bound(start, count, domains):
    """Bound correlated final-tile extents without expanding counted loops."""
    from itertools import product
    from math import prod

    from merlin.compiler.schedule.panels.readouts import _affine_window_bounds

    cuts = {}

    def visit(expr):
        if type(expr) is Select:
            if expr.var not in domains:
                raise ValueError("connected window has an unbound tail condition")
            cuts.setdefault(expr.var, set()).add(expr.equals)
            visit(expr.then)
            visit(expr.other)
        elif type(expr) is Scaled:
            visit(expr.expr)
        elif type(expr) is Sum:
            for term in expr.terms:
                visit(term)

    visit(start)
    visit(count)
    parts = {}
    for name, equalities in cuts.items():
        low, high = domains[name]
        selected = []
        for at in sorted(v for v in equalities if low <= v <= high):
            if low < at:
                selected.append((low, at - 1))
            selected.append((at, at))
            low = at + 1
        if low <= high:
            selected.append((low, high))
        parts[name] = selected
    if prod(map(len, parts.values())) > 64:
        raise ValueError("connected window tail conditions exceed the static checking budget")
    return max(
        _affine_window_bounds(start + count, dict(domains, **dict(zip(parts, values))))[1]
        for values in product(*parts.values())
    )


@dataclass(frozen=True)
class CheckedLoadedSumView:
    """Retain the actual source, projection and scalar proof behind each view."""

    plan: LoadedSumEpochPlan
    kernel: Kernel
    project: object
    scalar_proof: dict

    def revalidate(self):
        if type(self.plan) is not LoadedSumEpochPlan:
            raise ValueError("connected sum requires an original checked loaded-sum plan")
        self.plan.revalidate(
            self.kernel, self.plan.layout, self.plan.contract, self.project, scalar_proof=self.scalar_proof
        )


@dataclass(frozen=True)
class QuantizedSumCompositionAdmission:
    source_identity: str
    qualification_identity: str
    sole_intermediate_consumer: bool = False
    immutable_skip: bool = False
    private_workspace_disjoint: bool = False
    producer_completion_closes_workspace_and_ACC: bool = False
    consumer_preserves_stationary_operands: bool = False
    original_configuration_restored: bool = False
    real_full_completion: bool = False
    consumer_preserves_prefetched_inputs: bool = False

    def require(self):
        values = asdict(self)
        if (
            any(type(v) is not str or not v.strip() for k, v in values.items() if k.endswith("identity"))
            or any(
                type(v) is not bool or not v
                for k, v in values.items()
                if not k.endswith("identity") and k != "consumer_preserves_prefetched_inputs"
            )
            or type(self.consumer_preserves_prefetched_inputs) is not bool
        ):
            raise ValueError(
                "quantized sum composition needs source observer, workspace, preserving effects and real completion admission"
            )


def _completed_epoch_mode(producer, group_path, admission):
    """Retain the checked input-placement owner, including through load wrappers."""
    from merlin.compiler.schedule.panels.ready_schedule import DenseReadyReadoutResult

    owner = producer
    while not isinstance(owner, DenseReadyReadoutResult) and hasattr(owner, "original"):
        owner = owner.original
    if not isinstance(owner, DenseReadyReadoutResult) or owner.input_pipeline_row_offset is None:
        return False
    if not admission.consumer_preserves_prefetched_inputs:
        raise ValueError("connected prefetched inputs need an explicit consumer preserving-effects admission")
    records = owner.realization.get("input_pipeline", ())
    if not records or any(tuple(record["path"]) != group_path for record in records):
        raise ValueError("connected prefetched group differs from its original checked input placement")
    return True


def _compose_kernel(
    kernel,
    project,
    *,
    group_path,
    row_group,
    column_group,
    rows,
    columns,
    tile_rows,
    tile_columns,
    block_rows,
    block_columns,
    completed_epochs=False,
):
    """Exact syntactic proposal; original producer/consumer proofs remain required."""
    if (
        type(kernel) is not Kernel
        or check_structure(kernel)
        or not callable(project)
        or type(group_path) is not tuple
        or not group_path
        or group_path[-1] != row_group
        or row_group == column_group
        or column_group not in group_path[:-1]
        or any(type(v) is not str or not v for v in group_path)
        or any(
            type(v) is not int or v <= 0 for v in (rows, columns, tile_rows, tile_columns, block_rows, block_columns)
        )
        or type(completed_epochs) is not bool
        or columns % (tile_columns * block_columns)
    ):
        raise ValueError("staged sum needs a complete source N group and exact counted row group")
    row_span, column_span = tile_rows * block_rows, tile_columns * block_columns
    extent = (rows + row_span - 1) // row_span
    row_tiles = (rows + tile_rows - 1) // tile_rows
    full_groups, partial_tiles = divmod(row_tiles, block_rows)
    expected = sum(type(project(call)) is TiledIntegerStore for call in _calls(kernel.body) if call != Call(DRAIN, ()))
    found = 0

    def rewrite(body, base_row, base_column, domains, height):
        from merlin.compiler.schedule.panels.readouts import _affine_window_bounds
        from merlin.compiler.schedule.panels.tile_loads import _offset_form

        def local(expr, base):
            constant, terms = _offset_form(expr + base * -1)
            result = Const(constant)
            for term, coefficient in terms.items():
                result += term * coefficient
            return result

        result = []
        for statement in body:
            if type(statement) is Loop:
                if statement.var in domains:
                    raise ValueError("connected producer must have unique counted index scopes")
                result.append(
                    replace(
                        statement,
                        body=rewrite(
                            statement.body,
                            base_row,
                            base_column,
                            {**domains, statement.var: (0, statement.extent - 1)},
                            height,
                        ),
                    )
                )
            elif statement == Call(DRAIN, ()):
                result.append(statement)
            else:
                op = project(statement)
                if type(op) is TiledIntegerStore:
                    row, column = local(op.dram_row, base_row), local(op.dram_col, base_column)
                    # Translation by one common group origin preserves the
                    # original producer's independently checked exact cover.
                    # Outer group indices must cancel, so no other panel can
                    # address this owned temporary window.
                    try:
                        r, c, h, w = (_affine_window_bounds(v, domains) for v in (row, column, op.rows, op.cols))
                    except ValueError as error:
                        raise ValueError("connected readout escapes its owned source group") from error
                    if (
                        min(r[0], c[0]) < 0
                        or min(h[0], w[0]) <= 0
                        or _window_end_bound(row, op.rows, domains) > height
                        or _window_end_bound(column, op.cols, domains) > column_span
                    ):
                        raise ValueError("connected readout exceeds its private quantized workspace")
                    result.append(
                        replace(
                            statement,
                            instr="connected_q1_readout",
                            args=(*statement.args, ("workspace_row", row), ("workspace_column", column)),
                        )
                    )
                else:
                    result.append(statement)
        return tuple(result)

    def actual(body, base_row, height, *, counted):
        # Peeling changes only control representation: the independently
        # checked producer still owns every original readout and row count.
        # In particular, the final workspace may contain fewer primitive
        # tiles; the sum must not consume the unwritten rest of its capacity.
        domains = {column_group: (0, columns // column_span - 1)}
        if counted:
            domains[row_group] = (0, full_groups - 1)
        return (
            *rewrite(body, base_row, Var(column_group) * column_span, domains, height),
            Call(DRAIN, ()),
            Call(
                "connected_loaded_sum_group",
                (
                    ("output_row", base_row),
                    ("output_column", Var(column_group) * column_span),
                    ("rows", Const(height)),
                    ("columns", Const(column_span)),
                ),
            ),
        )

    def group(loop):
        nonlocal found
        if loop.extent != (full_groups if partial_tiles else extent) or loop.iter_args or loop.yields or loop.results:
            raise ValueError("staged sum source row group differs from its original complete domain")
        readouts = [
            call for call in _calls(loop.body) if call != Call(DRAIN, ()) and type(project(call)) is TiledIntegerStore
        ]
        if not readouts:
            raise ValueError("staged sum group has no original quantized producer readout")
        found += len(readouts)
        if partial_tiles:
            return (replace(loop, body=actual(loop.body, Var(row_group) * row_span, row_span, counted=True)),)
        tail = rows % row_span
        if not tail:
            return (replace(loop, body=actual(loop.body, Var(row_group) * row_span, row_span, counted=True)),)
        result = []
        if extent > 1:
            result.append(
                replace(
                    loop, extent=extent - 1, body=actual(loop.body, Var(row_group) * row_span, row_span, counted=True)
                )
            )
        last = _remap(loop.body, row_group, None, 1, extent - 1)
        result.extend(actual(last, Const((extent - 1) * row_span), tail, counted=False))
        return tuple(result)

    def walk(body, path):
        nonlocal found
        matches = [i for i, s in enumerate(body) if type(s) is Loop and s.var == path[0]]
        if len(matches) != 1:
            raise ValueError("staged sum group path must name one exact original counted loop")
        index = matches[0]
        statement = body[index]
        if statement.var == column_group and statement.extent != columns // column_span:
            raise ValueError("connected source column group differs from its complete original domain")
        if len(path) == 1 and partial_tiles:
            # compact_dense emits the full counted groups followed by one
            # explicit peeled group. Keep its original commands in place and
            # translate only its readouts, after independently bounding every
            # one against the exact remaining logical height.
            tail_body = body[index + 1 :]
            readouts = [
                call
                for call in _calls(tail_body)
                if call != Call(DRAIN, ()) and type(project(call)) is TiledIntegerStore
            ]
            if not readouts or any(loop.var == row_group for loop in _loops(tail_body)):
                raise ValueError("staged sum requires one explicit original peeled producer group")
            found += len(readouts)
            replacement = (
                *group(statement),
                *actual(tail_body, Const(full_groups * row_span), rows - full_groups * row_span, counted=False),
            )
            return (*body[:index], *replacement)
        replacement = group(statement) if len(path) == 1 else (replace(statement, body=walk(statement.body, path[1:])),)
        return (*body[:index], *replacement, *body[index + 1 :])

    body = kernel.body
    if completed_epochs:
        body, found = compose_completed_groups(
            kernel,
            project,
            rows=rows,
            columns=columns,
            row_span=row_span,
            column_span=column_span,
            rewrite=rewrite,
            end_bound=_window_end_bound,
        )
    elif (
        columns == column_span
        and group_path[0] == column_group
        and not any(loop.var == column_group for loop in _loops(body))
    ):
        # A checked producer may eliminate a one-trip outer column loop.
        # Restoring that exact unit domain changes no source command or
        # order, and binds the consumer's column origin to zero. Nonunit or
        # ambiguously named source domains still require their real loop.
        body = (Loop(column_group, 1, body),)
    if not completed_epochs:
        body = walk(body, group_path)
    if found != expected:
        raise ValueError("staged sum cannot leave any original intermediate readout outside its owned group")
    if any(a.name in ("skip", "quantized_workspace") for a in kernel.args):
        raise ValueError("connected source argument names collide with original producer")
    output = kernel.tensor("output")
    if output.shape != (rows, columns) or output.dtype != "i8" or output.access != "write":
        raise ValueError("staged sum needs the complete original signed-i8 producer exit")
    actual = replace(
        kernel,
        body=body,
        args=(
            *kernel.args,
            TensorArg("skip", (rows, columns), "i8", "read"),
            TensorArg("quantized_workspace", (row_span, column_span), "i8", "readwrite"),
        ),
    )
    if check_structure(actual):
        raise ValueError("staged sum changed original source index scopes")
    return actual


@dataclass(frozen=True)
class DenseLoadedSumComposition:
    source: object
    producer: object
    kernel: Kernel
    problem: object
    block: object
    project: object
    group_path: tuple
    row_group: str
    column_group: str
    consumers: tuple[CheckedLoadedSumView, ...]
    admission: QuantizedSumCompositionAdmission
    identity: str

    def revalidate(self):
        from merlin.compiler.schedule.panels.pipeline import DensePanelPipelineResult
        from merlin.compiler.schedule.panels.ready_schedule import DenseReadyReadoutResult
        from merlin.compiler.schedule.panels.tile_loads import PanelTileLoadResult

        if type(self.producer) not in (DensePanelPipelineResult, DenseReadyReadoutResult, PanelTileLoadResult):
            raise ValueError("connected loaded sum requires an original checked producer result")
        self.producer.revalidate()
        self.admission.require()
        from merlin.compiler.lower.device.outline import QuantizedSumCommandSource, QuantizedSumSource

        if type(self.source) not in (QuantizedSumSource, QuantizedSumCommandSource):
            raise ValueError("connected loaded sum requires its current original quantized source edge")
        self.source.revalidate()
        if self.source.identity != self.admission.source_identity or self.source.receipt["shape"] != [
            self.problem.m,
            self.problem.n,
        ]:
            raise ValueError("connected source binding differs from the actual producer domain")
        from merlin.compiler.schedule.dense.contracts import DenseTiledIntegerReadout

        owner = self.producer
        while not hasattr(owner, "output_readout"):
            owner = owner.original
        readout = owner.output_readout
        entry = self.source.receipt["entries"][0]
        if type(readout) is not DenseTiledIntegerReadout:
            raise ValueError("connected producer must retain its original completed INT8 Q1")
        wanted = DenseTiledIntegerReadout.from_source_scales(
            entry.get("acc_scale_by_column", entry.get("acc_scale")),
            columns=self.problem.n,
            tile_columns=self.problem.tile_n,
            relu="relu" in entry["epilogue"],
        )
        if (
            readout.column_scale_bits != wanted.column_scale_bits
            or readout.relu != wanted.relu
            or bool(owner.bias_source) != ("bias_add" in entry["epilogue"])
        ):
            raise ValueError("connected producer changed the original Q1 coefficients, bias or activation")
        expected = _compose_kernel(
            self.producer.kernel,
            self.project,
            group_path=self.group_path,
            row_group=self.row_group,
            column_group=self.column_group,
            rows=self.problem.m,
            columns=self.problem.n,
            tile_rows=self.problem.tile_m,
            tile_columns=self.problem.tile_n,
            block_rows=self.block.m,
            block_columns=self.block.n,
            completed_epochs=_completed_epoch_mode(self.producer, self.group_path, self.admission),
        )
        if expected != self.kernel or self.identity != _digest(
            (
                self.producer.kernel.digest(),
                self.kernel.digest(),
                asdict(self.problem),
                asdict(self.block),
                self.group_path,
                self.row_group,
                self.column_group,
                [_consumer_identity(c) for c in self.consumers],
                asdict(self.admission),
            )
        ):
            raise ValueError("connected loaded sum source, proposal or original consumer changed")
        shapes = {(min(self.problem.m, self.block.m * self.problem.tile_m), self.block.n * self.problem.tile_n)}
        if self.problem.m % (self.block.m * self.problem.tile_m):
            shapes.add((self.problem.m % (self.block.m * self.problem.tile_m), self.block.n * self.problem.tile_n))
        if (
            any(type(c) is not CheckedLoadedSumView for c in self.consumers)
            or len(self.consumers) != len(shapes)
            or {(c.plan.layout.rows, c.plan.layout.columns) for c in self.consumers} != shapes
            or len({c.plan.contract for c in self.consumers}) != 1
        ):
            raise ValueError(
                "connected loaded sum needs each exact full/tail source view and one unchanged scalar contract"
            )
        for consumer in self.consumers:
            consumer.revalidate()
            entry = self.source.receipt["entries"][1]
            scales = entry.get("source_scales")
            proof = consumer.scalar_proof
            if (
                type(scales) is not list
                or len(scales) != 3
                or proof.get("schema") != "merlin.ordered_loaded_operand_sum.v1"
                or proof.get("source")
                != dict(
                    lhs_scale=scales[0], rhs_scale=scales[1], output_scale=scales[2], relu="relu" in entry["epilogue"]
                )
                or proof.get("max_error_lsb") != 0
            ):
                raise ValueError("connected loaded sum changed the original ordered residual numerical stages")

    def emit(self, builder, producer_lower, consumer_lower, complete):
        from merlin.compiler.schedule.loops.counted_loops import emit_index_effect_program

        self.revalidate()
        if not all(callable(v) for v in (producer_lower, consumer_lower, complete)):
            raise ValueError("connected loaded sum needs actual effects and real completion callbacks")

        def lower(kind, fields):
            if kind != "connected_loaded_sum_group":
                producer_lower(kind, fields)
                return
            selected = [
                c.plan
                for c in self.consumers
                if (c.plan.layout.rows, c.plan.layout.columns) == (fields["rows"], fields["columns"])
            ]
            if len(selected) != 1:
                raise ValueError("actual connected group differs from its checked consumer shape")
            consumer_lower("begin_consumer", fields)

            def consume(action, values):
                if action == "complete_phase":
                    complete()
                else:
                    consumer_lower(
                        action,
                        dict(
                            values,
                            output_row=builder.op("add", fields["output_row"], values["row"]),
                            output_column=builder.op("add", fields["output_column"], values["column"]),
                        ),
                    )

            emit_loaded_sum_epochs(selected[0], builder, consume)
            consumer_lower("end_consumer", fields)

        report = emit_index_effect_program(
            self.kernel, builder, lower, complete_epoch=complete, completion_instruction=DRAIN
        )
        self.revalidate()
        return dict(
            index_program=report,
            producer_kernel_sha256=self.producer.kernel.digest(),
            composed_kernel_sha256=self.kernel.digest(),
            workspace_shape=list(self.kernel.tensor("quantized_workspace").shape),
            ordered_stages=["original Q1", "real completion", "original loaded sum", "original Q2/ReLU"],
            completion="original producer drains plus real Q1 handoff and all original loaded-sum phase drains",
        )


def _consumer_identity(view):
    if type(view) is not CheckedLoadedSumView:
        raise ValueError("original loaded-sum source view required")
    view.revalidate()
    return (view.plan.identity, view.kernel.digest(), view.scalar_proof)


def compose_dense_loaded_sum(
    source, producer, problem, block, *, project, group_path, row_group, column_group, consumers, admission
):
    if type(admission) is not QuantizedSumCompositionAdmission or type(consumers) is not tuple:
        raise ValueError("explicit source/effect admission and checked loaded-sum views required")
    admission.require()
    producer.revalidate()
    kernel = _compose_kernel(
        producer.kernel,
        project,
        group_path=group_path,
        row_group=row_group,
        column_group=column_group,
        rows=problem.m,
        columns=problem.n,
        tile_rows=problem.tile_m,
        tile_columns=problem.tile_n,
        block_rows=block.m,
        block_columns=block.n,
        completed_epochs=_completed_epoch_mode(producer, group_path, admission),
    )
    identity = _digest(
        (
            producer.kernel.digest(),
            kernel.digest(),
            asdict(problem),
            asdict(block),
            group_path,
            row_group,
            column_group,
            [_consumer_identity(c) for c in consumers],
            asdict(admission),
        )
    )
    result = DenseLoadedSumComposition(
        source,
        producer,
        kernel,
        problem,
        block,
        project,
        group_path,
        row_group,
        column_group,
        consumers,
        admission,
        identity,
    )
    result.revalidate()
    return result


# ---- dense_loaded_sum_groups -------------------------------------------------------------------
# Bind exact completed producer rectangles after counted input pipelining.
#
# This consumes the original producer's independently checked nonoverlapping
# readouts. It does not move a command or invent completion. Every selected epoch
# ends with its original full drain, including all future operand staging.


def compose_completed_groups(kernel, project, *, rows, columns, row_span, column_span, rewrite, end_bound):
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
                    raise ValueError("completed producer reuses an index scope")
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
                    raise ValueError("completed producer has a mixed or unbound group origin")
            if common is not None and common != global_terms:
                raise ValueError("completed producer readouts disagree on their group origin")
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
                raise ValueError("completed producer has ambiguous counted control")
            observed = (
                observations(statement.body, {statement.var: (0, 0)})
                if (
                    statement.extent == 1
                    and statement.body
                    and statement.body[-1] == Call(DRAIN, ())
                    and not statement.iter_args
                    and not statement.yields
                    and not statement.results
                )
                else []
            )
            if not observed:
                result.append(
                    replace(statement, body=walk(statement.body, {**outer, statement.var: (0, statement.extent - 1)}))
                )
                continue
            base_row, base_column = (origin(observed, field, outer) for field in ("dram_row", "dram_col"))
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
                    raise ValueError("completed output panel exceeds its local checking budget")
                for values in product(*(range(low, high + 1) for low, high in local.values())):
                    selected = dict(outer, **dict(zip(local, ((v, v) for v in values))))
                    h, w = (_affine_window_bounds(expr, selected) for expr in (operation.rows, operation.cols))
                    if h[0] != h[1] or w[0] != w[1] or min(h[0], w[0]) <= 0:
                        raise ValueError("completed Q1 view needs a separate exact tail epoch")
                    area += h[0] * w[0]
            if (
                not 0 < height <= row_span
                or width != column_span
                or area != height * width
                or _affine_window_bounds(base_row, outer)[0] < 0
                or _affine_window_bounds(base_column, outer)[0] < 0
                or end_bound(base_row, Const(height), outer) > rows
                or end_bound(base_column, Const(width), outer) > columns
            ):
                raise ValueError("completed Q1 readouts do not cover one exact original panel")
            contents = rewrite(statement.body, base_row, base_column, {**outer, statement.var: (0, 0)}, height)
            result.append(
                replace(
                    statement,
                    body=(
                        *contents,
                        Call(
                            "connected_loaded_sum_group",
                            (
                                ("output_row", base_row),
                                ("output_column", base_column),
                                ("rows", Const(height)),
                                ("columns", Const(column_span)),
                            ),
                        ),
                    ),
                )
            )
            found += len(observed)
        return tuple(result)

    return walk(kernel.body, {}), found
