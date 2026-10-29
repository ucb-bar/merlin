"""Bind counted virtual-N dense commands to separate original projection tensors.

Ordered loop partitions keep every command and completion in source order.
Only physical weight/output rectangles gain explicit member and local-column
operands. The original dense owner must still check products, storage, readout
and completion; this adapter supplies no target encoder or execution grant.
"""

from dataclasses import dataclass, replace

from merlin.sched.ir import Call, Const, Kernel, Loop, Scaled, Select, Sum, check_structure
from merlin.sched.ir.integer_domain import check_integer_domain

from .block_schedule import Load, Store
from .dense_contracts import DenseTiledIntegerReadout, TiledIntegerStore
from .dense_panel_pipeline import DRAIN
from .dense_readout_overlap import _affine_window_bounds
from .loop_schedule import LoopTransform, propose_loop_transforms
from .projection_provenance import projection_compute_member
from .projection_views import SharedProjectionViews

_MEMBER = "projection_member"
_COLUMN = "projection_local_column"
_WIDTH = "projection_row_columns"


def _check_rows(row, rows, domains, height):
    """Keep exact tail equality predicates while bounding complete rectangles."""
    selectors = set()

    def collect(expr):
        if type(expr) is Select:
            selectors.add((expr.var, expr.equals))
            collect(expr.then)
            collect(expr.other)
        elif type(expr) is Sum:
            for term in expr.terms:
                collect(term)
        elif type(expr) is Scaled:
            collect(expr.expr)

    collect(row)
    collect(rows)
    pieces = [domains]
    for name, value in sorted(selectors):
        refined = []
        for domain in pieces:
            low, high = domain[name]
            for first, last in (
                (low, min(high, value - 1)),
                (max(low, value), min(high, value)),
                (max(low, value + 1), high),
            ):
                if first <= last:
                    refined.append({**domain, name: (first, last)})
        pieces = refined
        if len(pieces) > 256:
            raise ValueError("projection row-tail partition exceeds its static proof budget")
    for domain in pieces:
        if (
            _affine_window_bounds(row, domain)[0] < 0
            or _affine_window_bounds(rows, domain)[0] <= 0
            or _affine_window_bounds(row + rows, domain)[1] > height
        ):
            raise ValueError("actual projection rectangle exceeds an original source/result row domain")


def _readouts(views, readouts, virtual_readout):
    if not readouts and virtual_readout is None:
        if any(unit != 4 for unit in views.output_element_bytes):
            raise ValueError("scaled projection outputs require every original readout")
        return
    if (
        type(readouts) is not tuple
        or len(readouts) != len(views.source.widths)
        or any(type(r) is not DenseTiledIntegerReadout for r in readouts)
        or type(virtual_readout) is not DenseTiledIntegerReadout
        or any(unit != 1 for unit in views.output_element_bytes)
    ):
        raise ValueError("projection readouts require complete original and virtual signed-i8 descriptors")
    for width, readout in zip(views.source.widths, readouts, strict=True):
        readout.__post_init__()
        if (
            readout.columns != width
            or readout.tile_columns != views.tile_columns
            or readout.relu != virtual_readout.relu
        ):
            raise ValueError("projection member shape, scale tile or activation differs")
    virtual_readout.__post_init__()
    if (
        virtual_readout.columns != views.columns
        or virtual_readout.tile_columns != views.tile_columns
        or virtual_readout.column_scale_bits != tuple(word for r in readouts for word in r.column_scale_bits)
    ):
        raise ValueError("virtual projection readout changes an original column coefficient")


def _bind(kernel, views, adapter, paths, panel_tiles, readouts, virtual_readout, budget):
    if (
        type(kernel) is not Kernel
        or check_structure(kernel)
        or type(paths) is not tuple
        or not 1 <= len(paths) <= 16
        or any(type(path) is not tuple or not path for path in paths)
        or len(set(paths)) != len(paths)
        or type(panel_tiles) is not int
        or panel_tiles not in views.aligned_panel_tiles(4096)
        or type(budget) is not int
        or not 1 <= budget <= 65536
    ):
        raise ValueError("projection binding needs counted commands and aligned explicit column paths/panels")
    if any(a != b and b[: len(a)] == a for a in paths for b in paths):
        raise ValueError("projection column partitions must be independent loops")
    _readouts(views, readouts, virtual_readout)
    expected_shapes = (
        (views.source.m, views.source.k),
        (views.source.k, views.columns),
        (views.source.m, views.columns),
    )
    expected_dtypes = ("i8", "i8", "i8" if virtual_readout is not None else "i32")
    if (
        kernel.attrs
        or len(kernel.args) != 3
        or tuple(argument.shape for argument in kernel.args) != expected_shapes
        or tuple(argument.dtype for argument in kernel.args) != expected_dtypes
    ):
        raise ValueError("virtual dense domain or precision differs from the complete original projections")
    width = panel_tiles * views.tile_columns
    cuts = tuple(column // width for column in views.boundaries[1:-1])
    if len(cuts) > 15:
        raise ValueError("projection partition exceeds the ordinary finite loop partition budget")
    requests = tuple(LoopTransform("partition", path, cuts=cuts) for path in paths)
    actual = propose_loop_transforms(kernel, requests, index_bits=adapter.index_bits, max_static_operations=budget)
    sites = []

    def walk(body, domains, path=()):
        result = []
        for ordinal, statement in enumerate(body):
            if type(statement) is Loop:
                if statement.iter_args or statement.yields or statement.results:
                    raise ValueError("projection binding cannot erase carried completion")
                result.append(
                    replace(
                        statement,
                        body=walk(
                            statement.body,
                            {**domains, statement.var: (0, statement.extent - 1)},
                            (*path, statement.var),
                        ),
                    )
                )
                continue
            if (
                type(statement) is not Call
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
                or {_MEMBER, _COLUMN, _WIDTH}.intersection(name for name, _ in statement.args)
            ):
                raise ValueError("projection binding requires unannotated original integer commands")
            if statement == Call(DRAIN, ()):
                result.append(statement)
                continue
            operation = adapter.project(statement)
            role = (
                "weight"
                if type(operation) is Load and operation.role == "weight"
                else "output"
                if type(operation) in (Store, TiledIntegerStore)
                else None
            )
            if role is None:
                result.append(statement)
                continue
            column = _affine_window_bounds(operation.dram_col, domains)
            columns = _affine_window_bounds(operation.cols, domains)
            member = projection_compute_member(views.boundaries, column, columns)
            height = views.source.k if role == "weight" else views.source.m
            _check_rows(operation.dram_row, operation.rows, domains, height)
            if type(operation) is TiledIntegerStore:
                if virtual_readout is None or operation.readout != virtual_readout.identity:
                    raise ValueError("actual projection store changed its original scalar descriptor")
                # The original dense owner independently binds readout_tile to
                # output columns. This adapter retains that exact operand.
            elif role == "output" and virtual_readout is not None:
                raise ValueError("raw projection store cannot replace a scaled original result")
            local = operation.dram_col + (-views.boundaries[member])
            result.append(
                replace(
                    statement,
                    args=(
                        *statement.args,
                        (_MEMBER, Const(member)),
                        (_COLUMN, local),
                        (_WIDTH, Const(views.source.widths[member])),
                    ),
                )
            )
            sites.append(
                dict(
                    path=path,
                    ordinal=ordinal,
                    role=role,
                    member=member,
                    virtual_column_bounds=column,
                    columns_bounds=columns,
                    original_row_columns=views.source.widths[member],
                )
            )
        return tuple(result)

    actual = replace(actual, body=walk(actual.body, {}))
    if not sites or not any(site["role"] == "output" for site in sites) or check_structure(actual):
        raise ValueError("projection binding has no complete original physical output sites")
    check_integer_domain(actual, adapter.index_bits)
    return actual, sites


@dataclass(frozen=True)
class ProjectionDenseBinding:
    """Exact counted pointer mapping; the original dense owner remains required."""

    original_kernel: Kernel
    kernel: Kernel
    views: SharedProjectionViews
    source_module: object
    adapter: object
    column_loop_paths: tuple
    panel_tiles: int
    readouts: tuple
    virtual_readout: object
    static_budget: int
    report: dict

    def revalidate(self):
        self.views.verify_current(self.source_module)
        expected, _ = _bind(
            self.original_kernel,
            self.views,
            self.adapter,
            self.column_loop_paths,
            self.panel_tiles,
            self.readouts,
            self.virtual_readout,
            self.static_budget,
        )
        if self.kernel != expected or self.kernel.digest() != self.report["kernel_sha256"]:
            raise ValueError("projection original source, counted commands or physical member mapping changed")

    def emit_index_program(self, control, lower_effect, *, complete_epoch):
        from .llvm_program import emit_index_effect_program

        self.revalidate()
        receipt = emit_index_effect_program(
            self.kernel, control, lower_effect, complete_epoch=complete_epoch, completion_instruction=DRAIN
        )
        self.revalidate()
        return receipt


def bind_dense_projection(
    kernel,
    views,
    adapter,
    *,
    source_module,
    column_loop_paths,
    panel_tiles,
    readouts=(),
    virtual_readout=None,
    static_budget=4096,
):
    """Partition original column loops and retain separate physical ABI tensors.

    ``column_loop_paths`` name complete virtual-N panel loops; each original
    member must contain an integer number of selected panels. Ordinary ordered
    loop partitioning preserves all calls, operands and drains. Every resulting
    physical rectangle is bounded symbolically inside one member, with no
    dynamic command expansion. The target consumes the three added operands to
    select that member's original pointer, local column and row pitch. The
    original virtual fields/readout tile stay present for independent checking.
    No concatenated tensor, source-output alias, effect or completion is granted.
    Revalidate the original checked dense owner before and after emission too.
    """
    if type(views) is not SharedProjectionViews:
        raise ValueError("dense projection requires current original source views")
    views.verify_current(source_module)
    actual, sites = _bind(
        kernel, views, adapter, column_loop_paths, panel_tiles, readouts, virtual_readout, static_budget
    )
    report = dict(
        schema="merlin.counted_dense_projection.v1",
        original_kernel_sha256=kernel.digest(),
        kernel_sha256=actual.digest(),
        views=views.report(),
        panel_tiles=panel_tiles,
        column_loop_paths=column_loop_paths,
        physical_sites=sites,
        command_order="ordinary ordered partition; every original effect and completion retained",
        host_tensor_copy_bytes=0,
        dynamic_command_expansion=False,
        dense_source_storage_readout_and_execution="original checked owner and target obligations",
    )
    return ProjectionDenseBinding(
        kernel,
        actual,
        views,
        source_module,
        adapter,
        column_loop_paths,
        panel_tiles,
        readouts,
        virtual_readout,
        static_budget,
        report,
    )
