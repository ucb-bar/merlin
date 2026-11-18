"""Transport original sibling root identities onto source-bound compute work.

These records carry source identity, never an execution or coverage grant.
The caller still checks actual arithmetic, storage, epochs and physical work.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import gcd

from xdsl.dialects.builtin import ArrayAttr, StringAttr

from merlin.compiler.kernel_ir.ir import (
    Call,
    Const,
    Kernel,
    Loop,
    Scaled,
    Select,
    Sum,
    check_integer_domain,
    check_structure,
)
from merlin.compiler.schedule.blocks import Load, Store
from merlin.compiler.schedule.dense.contracts import DenseTiledIntegerReadout, TiledIntegerStore
from merlin.compiler.schedule.loops.counted_loops import LoopTransform, propose_loop_transforms
from merlin.compiler.schedule.panels.pipeline import DRAIN
from merlin.compiler.schedule.panels.readouts import _affine_window_bounds
from merlin.compiler.schedule.shared_operands.shared_inputs import SharedInputGroup

# ---- projection_provenance ---------------------------------------------------------------------

_KEYS = frozenset({"prov.region_id", "prov.source_node_ids"})


def projection_provenance_attributes(attributes):
    """Read the closed optional CB identity pair, without silently losing fields.

    The interface's existing flat string/list syntax cannot quote punctuation
    inside identities. Refuse unrepresentable strings instead of changing them.
    """
    if type(attributes) is not dict:
        raise ValueError("projection provenance requires a plain attribute dictionary")
    if not attributes:
        return {}
    if set(attributes) != _KEYS:
        raise ValueError("projection provenance needs exactly original region and node identities")

    def identity(value):
        if type(value) is not str or not 0 < len(value) <= 4096 or any(ord(c) < 32 or c in '"\\,[]{}' for c in value):
            raise ValueError("projection source identity cannot round-trip through the selected interface")
        return value

    region = identity(attributes["prov.region_id"])
    nodes = attributes["prov.source_node_ids"]
    if type(nodes) is not list or not 1 <= len(nodes) <= 256 or any(type(node) is not str for node in nodes):
        raise ValueError("projection provenance requires a bounded original node list")
    nodes = [identity(node) for node in nodes]
    if len(set(nodes)) != len(nodes):
        raise ValueError("projection provenance repeats an original node identity")
    return {"prov.region_id": region, "prov.source_node_ids": nodes}


def shared_projection_provenance(module, group):
    """Re-read identities from the exact current typed roots, in member order.

    Fully unannotated source remains unannotated. Partial/duplicate identity
    records refuse; a local normalized operation ordinal is never synthesized.
    """
    from xdsl.dialects import func

    from merlin.compiler.schedule.shared_operands.shared_inputs import SharedInputGroup

    if type(group) is not SharedInputGroup:
        raise ValueError("projection provenance requires the checked typed source group")
    group.verify_current(module)
    function = next(op for op in module.body.block.ops if type(op) is func.FuncOp)
    operations = tuple(function.body.block.ops)
    roots = tuple(operations[index] for index in group.contraction_ordinals)
    if not any(_KEYS.intersection(root.attributes) for root in roots):
        return ()
    members = []
    for root in roots:
        region = root.attributes.get("prov.region_id")
        nodes = root.attributes.get("prov.source_node_ids")
        if (
            type(region) is not StringAttr
            or type(nodes) is not ArrayAttr
            or any(type(node) is not StringAttr for node in nodes)
        ):
            raise ValueError("every projection member needs its original typed region and nodes")
        members.append(
            projection_provenance_attributes(
                {
                    "prov.region_id": region.data,
                    "prov.source_node_ids": [node.data for node in nodes],
                }
            )
        )
    if len({member["prov.region_id"] for member in members}) != len(members):
        raise ValueError("projection members have ambiguous duplicate original regions")
    return tuple(members)


def projection_compute_member(boundaries, column_bounds, width_bounds):
    """Resolve a real compute interval to ONE original member, conservatively.

    Bounds come from the consumed counted/SCF indices, not receipt assertions.
    Unknown, negative, crossing or ambiguous intervals cannot be attributed.
    """
    if (
        type(boundaries) is not tuple
        or len(boundaries) < 3
        or boundaries[0] != 0
        or any(type(v) is not int for v in boundaries)
        or any(a >= b for a, b in zip(boundaries, boundaries[1:]))
    ):
        raise ValueError("compute attribution needs the complete original column partition")
    for bounds in (column_bounds, width_bounds):
        if (
            type(bounds) is not tuple
            or len(bounds) != 2
            or any(type(v) is not int for v in bounds)
            or bounds[0] > bounds[1]
        ):
            raise ValueError("compute attribution needs proved counted index bounds")
    if column_bounds[0] < 0 or width_bounds[0] <= 0:
        raise ValueError("compute attribution has a negative or empty interval")
    members = [
        i
        for i, (start, stop) in enumerate(zip(boundaries, boundaries[1:]))
        if start <= column_bounds[0] and column_bounds[1] + width_bounds[1] <= stop
    ]
    if len(members) != 1:
        raise ValueError("actual compute interval crosses original projection members")
    return members[0]


def projection_ir_attributes(attributes):
    """Build the exact LLVM/typed-target identity attributes, without a grant."""
    attributes = projection_provenance_attributes(attributes)
    if not attributes:
        return {}
    return {
        "prov.region_id": StringAttr(attributes["prov.region_id"]),
        "prov.source_node_ids": ArrayAttr(StringAttr(node) for node in attributes["prov.source_node_ids"]),
    }


# ---- projection_views --------------------------------------------------------------------------
# Source-checked column views for sibling contractions of one immutable input.
#
# The virtual weight/output matrices are index spaces, never allocated tensors.
# Each rectangle must stay inside one original member. The caller still owns
# physical pointer aliases, instruction effects, per-result scalar readouts and
# completion; this mapping establishes none of those target permissions.


@dataclass(frozen=True)
class ProjectionRectangle:
    member: int
    row: int
    column: int
    rows: int
    columns: int
    row_stride_bytes: int
    byte_offset: int


@dataclass(frozen=True)
class SharedProjectionViews:
    source: SharedInputGroup
    weight_element_bytes: int
    output_element_bytes: tuple[int, ...]
    tile_columns: int

    def __post_init__(self):
        if type(self.source) is not SharedInputGroup:
            raise ValueError("projection views need an original shared-input source group")
        widths = self.source.widths
        if (
            not 2 <= len(widths) <= 128
            or any(type(v) is not int or v <= 0 for v in (self.source.m, self.source.k, *widths))
            or type(self.weight_element_bytes) is not int
            or self.weight_element_bytes != 1
            or type(self.tile_columns) is not int
            or self.tile_columns <= 0
            or any(width % self.tile_columns for width in widths)
            or sum(widths) >= 1 << 63
            or self.source.k * sum(widths) >= 1 << 63
            or self.source.m * sum(widths) * 4 >= 1 << 63
            or type(self.output_element_bytes) is not tuple
            or len(self.output_element_bytes) != len(widths)
            or any(type(v) is not int or v <= 0 for v in self.output_element_bytes)
            or any(self.source.k * width * self.weight_element_bytes >= 1 << 63 for width in widths)
            or any(self.source.m * width * unit >= 1 << 63 for width, unit in zip(widths, self.output_element_bytes))
        ):
            raise ValueError("projection views need bounded whole-column tiles and every original output width")

    @property
    def columns(self):
        return sum(self.source.widths)

    @property
    def boundaries(self):
        result = [0]
        for width in self.source.widths:
            result.append(result[-1] + width)
        return tuple(result)

    def verify_current(self, module):
        self.__post_init__()
        self.source.verify_current(module)

    def aligned_panel_tiles(self, maximum):
        """Panel widths that divide every original member, in increasing order.

        A selected panel and every transfer inside it must obey these member
        boundaries. This is not permission to widen an independently encoded
        physical transfer or to fuse scalar readouts across a boundary.
        """
        self.__post_init__()
        if type(maximum) is not int or not 1 <= maximum <= 4096:
            raise ValueError("explicit finite panel search bound required")
        common = 0
        for width in self.source.widths:
            common = gcd(common, width // self.tile_columns)
        return tuple(n for n in range(1, min(common, maximum) + 1) if common % n == 0)

    def rectangle(self, kind, row, column, rows, columns):
        """Map an actual complete rectangle to exactly one original tensor.

        The source and destination element widths are independent. A crossing
        rectangle refuses instead of silently assuming packed adjacent arrays.
        """
        self.__post_init__()
        if kind not in ("weight", "output") or any(type(v) is not int for v in (row, column, rows, columns)):
            raise ValueError("projection rectangle needs exact integer coordinates and a declared role")
        height = self.source.k if kind == "weight" else self.source.m
        if not (0 <= row < height and 0 < rows <= height - row and 0 <= column < self.columns and columns > 0):
            raise ValueError("projection rectangle exceeds the original logical domain")
        starts = self.boundaries
        member = next(i for i in range(len(starts) - 1) if starts[i] <= column < starts[i + 1])
        local = column - starts[member]
        width = self.source.widths[member]
        if columns > width - local:
            raise ValueError("projection rectangle crosses an original tensor boundary")
        unit = self.weight_element_bytes if kind == "weight" else self.output_element_bytes[member]
        return ProjectionRectangle(member, row, local, rows, columns, width * unit, (row * width + local) * unit)

    def report(self):
        self.__post_init__()
        return {
            "schema": "merlin.shared_projection_views.v1",
            "source": self.source.record(),
            "shape_m_k_n": [self.source.m, self.source.k, self.columns],
            "column_boundaries": list(self.boundaries),
            "weight_element_bytes": self.weight_element_bytes,
            "output_element_bytes": list(self.output_element_bytes),
            "tile_columns": self.tile_columns,
            "source_weights_or_outputs_concatenated": False,
            "host_payload_copy_bytes": 0,
            "scalar_readouts": "separate original per-member obligations",
            "physical_alias_and_completion": "caller obligations; no target grant",
        }


def plan_shared_projection_views(module, group, *, weight_element_bytes, output_element_bytes, tile_columns):
    """Bind a current source group to explicit per-tensor byte views."""
    result = SharedProjectionViews(group, weight_element_bytes, output_element_bytes, tile_columns)
    result.verify_current(module)
    return result


# ---- projection_dense --------------------------------------------------------------------------
# Bind counted virtual-N dense commands to separate original projection tensors.
#
# Ordered loop partitions keep every command and completion in source order.
# Only physical weight/output rectangles gain explicit member and local-column
# operands. The original dense owner must still check products, storage, readout
# and completion; this adapter supplies no target encoder or execution grant.

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
        from merlin.compiler.schedule.loops.counted_loops import emit_index_effect_program

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
