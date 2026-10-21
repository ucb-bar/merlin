"""Compact rectangular dense schedules in the existing counted-loop kernel IR.

Merlin owns panel blocking, retention, tails and source reduction order. An
explicit adapter translates typed moves/computes and projects their *actual*
operands back. Instruction implementation and physical row mapping remain OOT.
No target lookup, opaque output inference, or latency model is used here.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from merlin.transforms.pipeline_state import AnalysisStamp

from merlin.sched.ir import Call, Const, Expr, Kernel, Loop, TensorArg, Var, select
from merlin.sched.ir.kernel import DTYPE_BYTES, check_structure
from merlin.transforms.dense_planning import DenseBlock, DenseChoice, DenseProblem, DenseResources, choose_dense_block

from . import block_schedule
from .block_schedule import AccumulatorZero, Compute, Load, LoadGroup, PanelTransferLimits, Preload, Store

DenseOperation = Load | LoadGroup | AccumulatorZero | Preload | Compute | Store


@dataclass(frozen=True)
class DenseLoadGrouping:
    """Original-qualified blocked wide-copy semantics, beyond affine copying.

    Tuple entries follow (lhs, rhs). Strides, alignments and destination block
    pitches are independent original declarations, not compute-tile defaults.
    Source base alignments constrain the actual runtime pointers. The selected
    mapping copies adjacent source column tiles to stacked local row blocks,
    with no other effects and synchronous completion of every component.
    """

    limits: PanelTransferLimits
    source_strides: tuple[int, int]
    source_base_alignments: tuple[int, int]
    destination_block_rows: tuple[int, int]
    max_components: int
    mapping: str
    exact_blocked_mapping: bool
    qualification_identity: str

    def __post_init__(self):
        if type(self.limits) is not PanelTransferLimits:
            raise ValueError("load grouping requires selected rectangular transfer limits")
        PanelTransferLimits.__post_init__(self.limits)
        if any(
            type(pair) is not tuple or len(pair) != 2 or any(type(x) is not int or not 0 < x < 2**63 for x in pair)
            for pair in (self.source_strides, self.source_base_alignments, self.destination_block_rows)
        ):
            raise ValueError("load grouping requires explicit source strides, base alignments and block pitches")
        if (
            type(self.max_components) is not int
            or not 2 <= self.max_components <= 128
            or type(self.mapping) is not str
            or self.mapping != "stacked_column_tiles"
            or type(self.exact_blocked_mapping) is not bool
            or type(self.qualification_identity) is not str
            or not self.qualification_identity.strip()
            or len(self.qualification_identity) > 512
        ):
            raise ValueError("load grouping requires a bounded separately qualified exact blocked mapping law")


@dataclass(frozen=True)
class DenseMemoryView:
    """Assigned rows in an explicitly identified physical backing.

    ``base``/``capacity``/``alignment`` are in this view's row units. Different
    row widths may alias a backing only through their explicit byte mapping.
    These are usable assignments after packing/DMA/other reservations.
    """

    name: str
    backing: str
    unit_bytes: int
    base: int
    capacity: int
    alignment: int = 1

    def __post_init__(self):
        if any(not isinstance(value, str) or not value for value in (self.name, self.backing)):
            raise ValueError("dense storage needs explicit view and backing identities")
        if any(type(x) is not int or x <= 0 for x in (self.unit_bytes, self.capacity, self.alignment)):
            raise ValueError("dense storage units, capacity and alignment must be positive")
        if type(self.base) is not int or self.base < 0 or self.base % self.alignment:
            raise ValueError("dense storage base is negative or unaligned")


@dataclass(frozen=True)
class DenseLoopAdapter:
    """Public instruction boundary. Neither function is selected by its name.

    Projection must decode the actual call operands, not retained requests.
    The declaration is conditional on the adapter's independent qualification;
    synthetic adapters establish only the shared algorithm's scope.
    """

    encode: Callable[[DenseOperation], Call]
    project: Callable[[Call], DenseOperation]
    configuration_identity: str
    numerical_policy_identity: str
    synchronous_completion: bool
    exact_accumulator_continuation: bool
    index_bits: int
    qualification_identity: str | None = None
    #: Original semantics preserve PE weights through loads, stores and
    #: descriptor-only preloads; a fresh compute replaces that resident owner.
    retained_weights: bool = field(default=False, kw_only=True)
    #: Original semantics establish an exact synchronous zero write to every
    #: addressed accumulator element, independently of operand memory.
    zero_initialization: bool = field(default=False, kw_only=True)
    load_grouping: DenseLoadGrouping | None = field(default=None, kw_only=True)

    def __post_init__(self):
        if self.load_grouping is not None:
            if type(self.load_grouping) is not DenseLoadGrouping:
                raise ValueError("dense load grouping requires an original typed law")
            DenseLoadGrouping.__post_init__(self.load_grouping)
        if not callable(self.encode) or not callable(self.project):
            raise ValueError("dense lowering requires an encoder and actual-operand projection")
        if any(
            not isinstance(value, str) or not value
            for value in (self.configuration_identity, self.numerical_policy_identity)
        ):
            raise ValueError("dense lowering requires selected configuration and numerical policy")
        if any(
            type(x) is not bool
            for x in (
                self.synchronous_completion,
                self.exact_accumulator_continuation,
                self.retained_weights,
                self.zero_initialization,
            )
        ):
            raise ValueError("dense completion and continuation declarations must be Boolean")
        if self.qualification_identity is not None and (
            type(self.qualification_identity) is not str or not self.qualification_identity.strip()
        ):
            raise ValueError("dense qualification identity must be nonempty or unknown")
        if type(self.index_bits) is not int or self.index_bits not in (32, 64):
            raise ValueError("dense C index arithmetic requires a selected signed 32 or 64-bit representation")


def _adapter_guard(adapter):
    """Retain actual adapter permissions and callback bindings across callbacks."""
    from . import schedule_pipeline

    reader, reader_code = schedule_pipeline._rewrite_snapshot, schedule_pipeline._rewrite_snapshot.__code__
    validator, validator_code = DenseLoopAdapter.__post_init__, DenseLoopAdapter.__post_init__.__code__
    group_validator, group_code = DenseLoadGrouping.__post_init__, DenseLoadGrouping.__post_init__.__code__
    limit_validator, limit_code = PanelTransferLimits.__post_init__, PanelTransferLimits.__post_init__.__code__
    limit_integers = block_schedule._transfer_integers
    limit_integer_code = limit_integers.__code__
    limit_integer_defaults = reader((limit_integers.__defaults__, limit_integers.__kwdefaults__))
    original_callbacks = (adapter.encode, adapter.project)

    def snapshot():
        if (
            schedule_pipeline._rewrite_snapshot is not reader
            or reader.__code__ is not reader_code
            or DenseLoopAdapter.__post_init__ is not validator
            or validator.__code__ is not validator_code
            or DenseLoadGrouping.__post_init__ is not group_validator
            or group_validator.__code__ is not group_code
            or PanelTransferLimits.__post_init__ is not limit_validator
            or limit_validator.__code__ is not limit_code
            or block_schedule._transfer_integers is not limit_integers
            or limit_integers.__code__ is not limit_integer_code
            or reader((limit_integers.__defaults__, limit_integers.__kwdefaults__)) != limit_integer_defaults
        ):
            raise ValueError("dense adapter snapshot implementation changed")
        validator(adapter)
        callbacks = []
        for callback in (adapter.encode, adapter.project):
            target = getattr(callback, "__func__", callback)
            if not hasattr(target, "__code__"):
                target = getattr(type(target), "__call__", target)
            callbacks.append(
                (
                    id(callback),
                    id(getattr(callback, "__self__", None)),
                    id(target),
                    getattr(target, "__code__", None),
                    reader((getattr(target, "__defaults__", None), getattr(target, "__kwdefaults__", None))),
                )
            )
        return (
            tuple(callbacks),
            adapter.configuration_identity,
            adapter.numerical_policy_identity,
            adapter.qualification_identity,
            adapter.synchronous_completion,
            adapter.exact_accumulator_continuation,
            adapter.index_bits,
            (type(adapter.retained_weights), adapter.retained_weights),
            (type(adapter.zero_initialization), adapter.zero_initialization),
            reader(adapter.load_grouping),
        )

    original = snapshot()

    def fresh():
        if len(original_callbacks) != 2 or snapshot() != original:
            raise ValueError("dense adapter permissions or callbacks changed; replan")

    return fresh


@dataclass(frozen=True)
class DenseLoopStorage:
    lhs: DenseMemoryView
    rhs: DenseMemoryView
    accumulator: DenseMemoryView | None

    def __post_init__(self):
        views = (self.lhs, self.rhs) + (() if self.accumulator is None else (self.accumulator,))
        if len({view.name for view in views}) != len(views):
            raise ValueError("dense live storage roles require distinct view identities")

    def row_counts(self, problem: DenseProblem, block: DenseBlock):
        return (
            block.m * block.k * problem.tile_m,
            block.k * block.n * problem.tile_k,
            block.m * block.n * problem.tile_m,
        )

    def k_limit(self, problem: DenseProblem, resources: DenseResources, m: int, n: int):
        return min(
            self.lhs.capacity // (m * problem.tile_m),
            self.rhs.capacity // (n * problem.tile_k),
            resources.operand_bytes
            // (m * problem.tile_m * self.lhs.unit_bytes + n * problem.tile_k * self.rhs.unit_bytes),
        )

    def admits(self, problem: DenseProblem, block: DenseBlock, resources: DenseResources):
        if self.accumulator is None:
            return False
        views = (self.lhs, self.rhs, self.accumulator)
        widths = (
            problem.tile_k * problem.lhs_bytes,
            problem.tile_n * problem.rhs_bytes,
            problem.tile_n * problem.accumulator_bytes,
        )
        rows = self.row_counts(problem, block)
        if any(
            stride % view.alignment for stride, view in zip((problem.tile_m, problem.tile_k, problem.tile_m), views)
        ):
            return False
        if any(width > view.unit_bytes or count > view.capacity for width, view, count in zip(widths, views, rows)):
            return False
        used = tuple(view.unit_bytes * count for view, count in zip(views, rows))
        if used[0] + used[1] > resources.operand_bytes or used[2] > resources.accumulator_bytes:
            return False
        intervals = [
            (view.backing, view.base * view.unit_bytes, (view.base + view.capacity) * view.unit_bytes) for view in views
        ]
        return not any(
            a[0] == b[0] and a[1] < b[2] and b[1] < a[2] for i, a in enumerate(intervals) for b in intervals[i + 1 :]
        )


@dataclass(frozen=True)
class CompactDenseResult:
    kernel: Kernel
    choice: DenseChoice
    realization: dict
    index_bits: int
    selected_contracts: object = None
    checked_emission: AnalysisStamp | None = None
    checked_index_bits: int | None = None
    checked_selection: str | None = None
    output_readout: object = None
    checked_readout: str | None = None
    _adapter_check: Callable | None = field(default=None, repr=False, compare=False, kw_only=True)

    def revalidate(self):
        from .selected_contracts import SelectedSchedulingContracts

        if self._adapter_check is not None:
            self._adapter_check()
        if self.selected_contracts is not None:
            if type(self.selected_contracts) is not SelectedSchedulingContracts:
                raise ValueError("dense selected RTL/software contracts changed; reselect and replan")
            self.selected_contracts.revalidate()
        selection = None if self.selected_contracts is None else _digest(self.selected_contracts.record())
        if self._adapter_check is not None:
            self._adapter_check()
        if selection != self.checked_selection:
            raise ValueError("dense selected RTL/software contracts changed; reselect and replan")
        from .dense_contracts import DenseIntegerReadout

        if (
            self.output_readout is not None
            and type(self.output_readout) is not DenseIntegerReadout
            or (None if self.output_readout is None else self.output_readout.identity) != self.checked_readout
        ):
            raise ValueError("dense output readout changed; replan before emission")
        # Diagnostic dictionaries may be edited or removed. Emission custody
        # belongs to an immutable current-IR stamp instead.
        stamp = self.checked_emission
        if (
            stamp is None
            or self.kernel.digest() != stamp.source_sha256
            or _digest(_dense_library_sources()) != stamp.context.library
            or type(self.index_bits) is not int
            or self.index_bits != self.checked_index_bits
            or stamp.context.execution is None
            or not stamp.context.execution.endswith(
                f":selection={selection or 'unselected'}:index_bits={self.index_bits}"
                + ("" if self.checked_readout is None else f":readout={self.checked_readout}")
            )
        ):
            raise ValueError("dense checked kernel or shared library changed; replan before emission")

    def emit_c(self, instruction_set, *, symbol):
        """Use the existing emitter with the selected, checked loop index width.

        Target instruction implementations still own their argument widths and
        address windows. This emits C, not a machine conformance receipt.
        """
        from merlin.sched.codegen import emit_c_function

        self.revalidate()
        source = emit_c_function(self.kernel, instruction_set, symbol=symbol, index_bits=self.index_bits)
        self.revalidate()
        return source


@dataclass(frozen=True)
class _Group:
    var: str | None
    trips: int
    base: Expr
    blocks: int
    contains_last: bool


def _groups(total, block, name):
    full, tail = divmod(total, block)
    parts = []
    if full:
        parts.append(_Group(name, full, Var(name) * block, block, not tail))
    if tail:
        parts.append(_Group(None, 1, Const(full * block), tail, True))
    return tuple(parts)


def _edge(group, local, logical_extent, primitive):
    tail = logical_extent % primitive
    if not tail or not group.contains_last:
        return Const(primitive)
    last = (
        Const(tail if local.value == group.blocks - 1 else primitive)
        if isinstance(local, Const)
        else select(local, group.blocks - 1, tail, primitive)
    )
    return last if group.var is None else select(group.var, group.trips - 1, last, primitive)


def _first(group, local):
    if isinstance(group.base, Const):
        return select(local, 0, 0, 1) if group.base.value == 0 else Const(1)
    return select(group.var, 0, select(local, 0, 0, 1), 1)


def _static_calls(body):
    for statement in body:
        if isinstance(statement, Loop):
            yield from _static_calls(statement.body)
        else:
            yield statement


def _load_group_calls(operation, count, *, full_width, possible_tail, call, grouping):
    """Group bounded full column tiles; retain primitive tail transfers."""
    maximum = min(grouping.max_components, grouping.limits.max_row_bytes // full_width)
    eligible = count - int(possible_tail)
    result = []
    cursor = 0
    while cursor < eligible:
        size = min(maximum, eligible - cursor)
        # An unavailable wide width does not disable the original primitive.
        size = max(1, size)
        loads = tuple(operation(Const(i)) for i in range(cursor, cursor + size))
        result.append(call(LoadGroup(loads) if size > 1 else loads[0]))
        cursor += size
    if possible_tail:
        result.append(call(operation(Const(count - 1))))
    return tuple(result)


def _load_grouping_admission(problem, storage, adapter):
    law = adapter.load_grouping
    if type(law) is not DenseLoadGrouping:
        return False
    DenseLoadGrouping.__post_init__(law)
    declared = (
        type(law) is DenseLoadGrouping
        and law.exact_blocked_mapping is True
        and bool(adapter.qualification_identity)
        and law.source_strides == (problem.k * problem.lhs_bytes, problem.n * problem.rhs_bytes)
        and all(value % law.limits.address_alignment == 0 for value in law.source_base_alignments)
        and law.destination_block_rows == (problem.tile_m, problem.tile_k)
    )
    if not declared:
        return False
    limits = law.limits
    # This first grouping family requires alignment for every full tile origin
    # and every physical row. More specialized aligned subdomains retain the
    # primitive route rather than sending an unproved group to its encoder.
    for height, width, stride, view in (
        (min(problem.m, problem.tile_m), problem.tile_k * problem.lhs_bytes, law.source_strides[0], storage.lhs),
        (min(problem.k, problem.tile_k), problem.tile_n * problem.rhs_bytes, law.source_strides[1], storage.rhs),
    ):
        if (
            height > limits.max_rows
            or stride > limits.max_source_stride_bytes
            or view.unit_bytes > limits.max_destination_stride_bytes
            or stride % limits.address_alignment
            or view.unit_bytes % limits.address_alignment
            or width % limits.address_alignment
            or width % limits.row_bytes_multiple
            or (view.base * view.unit_bytes) % limits.address_alignment
        ):
            return False
    return True


def _emit_lhs_panel(problem, block, storage, call, nest, mg, kg, *, grouping=None):
    def operation(i, reduction):
        m, k = mg.base + i, kg.base + reduction
        return Load(
            "lhs",
            (m, k),
            m * problem.tile_m,
            k * problem.tile_k,
            _edge(mg, i, problem.m, problem.tile_m),
            _edge(kg, reduction, problem.k, problem.tile_k),
            storage.lhs.base + (i * block.k + reduction) * problem.tile_m,
            1,
        )

    def lhs(i, reduction):
        return (call(operation(i, reduction)),)

    if grouping is not None:
        return nest(
            (mg.blocks,),
            lambda i: _load_group_calls(
                lambda reduction: operation(i, reduction),
                kg.blocks,
                full_width=problem.tile_k * problem.lhs_bytes,
                possible_tail=kg.contains_last and bool(problem.k % problem.tile_k),
                call=call,
                grouping=grouping,
            ),
        )

    return nest((mg.blocks, kg.blocks), lhs)


def _emit_dense_tile_group(
    problem,
    block,
    storage,
    call,
    variable,
    nest,
    mg,
    ng,
    *,
    output_readout=None,
    include_lhs=True,
    reuse_pe_weights=False,
    grouping=None,
):
    kt = problem.tiles[1]
    statements = []
    for kg in _groups(kt, block.k, variable("panel")):

        def rhs_operation(reduction, j):
            k, n = kg.base + reduction, ng.base + j
            return Load(
                "weight",
                (k, n),
                k * problem.tile_k,
                n * problem.tile_n,
                _edge(kg, reduction, problem.k, problem.tile_k),
                _edge(ng, j, problem.n, problem.tile_n),
                storage.rhs.base + (reduction * block.n + j) * problem.tile_k,
                1,
            )

        def rhs(reduction, j):
            return (call(rhs_operation(reduction, j)),)

        def compute(reduction, i, j, *, retained=False, rows=None):
            m, k, n = mg.base + i, kg.base + reduction, ng.base + j
            rows = _edge(mg, i, problem.m, problem.tile_m) if rows is None else rows
            depth = _edge(kg, reduction, problem.k, problem.tile_k)
            cols = _edge(ng, j, problem.n, problem.tile_n)
            row = storage.accumulator.base + (i * block.n + j) * problem.tile_m
            return (
                call(
                    Preload(
                        None if retained else storage.rhs.base + (reduction * block.n + j) * problem.tile_k,
                        (k, n),
                        row,
                        _first(kg, reduction),
                        rows,
                        cols,
                        depth,
                    )
                ),
                call(
                    Compute(
                        storage.lhs.base + (i * block.k + reduction) * problem.tile_m,
                        (m, k),
                        rows,
                        depth,
                        not retained,
                    )
                ),
            )

        def stationary(reduction, j):
            statements = compute(reduction, Const(0), j)
            if mg.blocks > 1:
                name = variable("retained_m")
                index = Var(name)
                rows = _edge(replace(mg, blocks=mg.blocks - 1), index, problem.m, problem.tile_m)
                statements += (Loop(name, mg.blocks - 1, compute(reduction, index + 1, j, retained=True, rows=rows)),)
            return statements

        body = (
            *(_emit_lhs_panel(problem, block, storage, call, nest, mg, kg, grouping=grouping) if include_lhs else ()),
            *(
                nest((kg.blocks, ng.blocks), rhs)
                if grouping is None
                else nest(
                    (kg.blocks,),
                    lambda reduction: _load_group_calls(
                        lambda j: rhs_operation(reduction, j),
                        ng.blocks,
                        full_width=problem.tile_n * problem.rhs_bytes,
                        possible_tail=ng.contains_last and bool(problem.n % problem.tile_n),
                        call=call,
                        grouping=grouping,
                    ),
                )
            ),
            *(
                nest((kg.blocks, ng.blocks), stationary)
                if reuse_pe_weights
                else nest((kg.blocks, mg.blocks, ng.blocks), compute)
            ),
        )
        statements.extend((Loop(kg.var, kg.trips, body),) if kg.var is not None else body)

    def store(i, j):
        return (
            call(
                Store(
                    (mg.base + i) * problem.tile_m,
                    (ng.base + j) * problem.tile_n,
                    _edge(mg, i, problem.m, problem.tile_m),
                    _edge(ng, j, problem.n, problem.tile_n),
                    storage.accumulator.base + (i * block.n + j) * problem.tile_m,
                    output_readout.identity if output_readout is not None else None,
                )
            ),
        )

    statements.extend(nest((mg.blocks, ng.blocks), store))
    return tuple(statements)


def emit_compact_dense(
    problem: DenseProblem,
    block: DenseBlock,
    storage: DenseLoopStorage,
    adapter: DenseLoopAdapter,
    *,
    name: str,
    dtypes: tuple[str, str, str],
    outer_order: tuple[str, str] = ("m", "n"),
    output_readout=None,
    reuse_pe_weights=False,
    group_dma=False,
):
    """Emit bounded panel nests without enumerating tensor elements or tiles.

    The supported family starts a zero-initialized reduction on its first
    compute and keeps each output accumulator through the complete K domain.
    Bias/other seeds retain their original route until explicitly implemented.
    Explicit PE reuse keeps one weight panel resident across the current M
    block. Its original adapter qualification must cover the retained forms.
    """
    if problem.bias:
        raise ValueError("compact dense family requires an explicit zero seed; bias initialization is unavailable")
    if type(reuse_pe_weights) is not bool:
        raise ValueError("PE weight reuse selection must be Boolean")
    if type(group_dma) is not bool:
        raise ValueError("DMA grouping selection must be Boolean")
    if group_dma and not _load_grouping_admission(problem, storage, adapter):
        raise ValueError("DMA grouping requires original separately qualified exact blocked mapping semantics")
    if reuse_pe_weights and (not adapter.retained_weights or not adapter.qualification_identity):
        raise ValueError("PE weight reuse requires original qualified retained-weight semantics")
    adapter_check = _adapter_guard(adapter)
    if any(type(value) is not int or value <= 0 for value in (block.m, block.k, block.n)):
        raise ValueError("compact dense block dimensions must be positive integers")
    if not adapter.synchronous_completion or not adapter.exact_accumulator_continuation:
        raise ValueError("compact dense family needs synchronous completion and exact accumulator continuation")
    if storage.accumulator is None:
        raise ValueError("compact dense family has no declared addressable accumulator storage")
    if outer_order not in (("m", "n"), ("n", "m")):
        raise ValueError("compact dense outer order must keep the complete source reduction inside each output block")
    if tuple(DTYPE_BYTES.get(t) for t in dtypes) != (problem.lhs_bytes, problem.rhs_bytes, problem.accumulator_bytes):
        raise ValueError("dense operand and accumulator dtypes disagree with their physical widths")
    mt, kt, nt = problem.tiles
    serial = 0

    def variable(prefix):
        nonlocal serial
        serial += 1
        return prefix + str(serial)

    def call(operation):
        adapter_check()
        result = adapter.encode(operation)
        adapter_check()
        if type(result) is not Call:
            raise ValueError("dense adapter must emit an actual kernel instruction call")
        return result

    def nest(extents, action):
        names = tuple(variable("tile") for _ in extents)
        body = tuple(action(*(Var(name) for name in names)))
        for var, extent in reversed(tuple(zip(names, extents))):
            body = (Loop(var, extent, body),)
        return body

    def tile_group(mg, ng):
        return _emit_dense_tile_group(
            problem,
            block,
            storage,
            call,
            variable,
            nest,
            mg,
            ng,
            output_readout=output_readout,
            reuse_pe_weights=reuse_pe_weights,
            grouping=adapter.load_grouping if group_dma else None,
        )

    groups = {"m": _groups(mt, block.m, "outer_m"), "n": _groups(nt, block.n, "outer_n")}

    def outer(level, selected):
        if level == 2:
            return tile_group(selected["m"], selected["n"])
        axis = outer_order[level]
        body = []
        for group in groups[axis]:
            nested = outer(level + 1, {**selected, axis: group})
            body.extend((Loop(group.var, group.trips, nested),) if group.var is not None else nested)
        return tuple(body)

    kernel = Kernel(
        name,
        (
            TensorArg("lhs", (problem.m, problem.k), dtypes[0], "read"),
            TensorArg("rhs", (problem.k, problem.n), dtypes[1], "read"),
            TensorArg(
                "output",
                (problem.m, problem.n),
                dtypes[2] if output_readout is None else output_readout.output_dtype,
                "write",
            ),
        ),
        outer(0, {}),
    )
    problems = check_structure(kernel)
    if problems:
        raise ValueError("invalid compact dense structure: " + "; ".join(problems))
    return kernel


def schedule_compact_dense(
    problem: DenseProblem,
    resources: DenseResources | None,
    storage: DenseLoopStorage,
    adapter: DenseLoopAdapter,
    *,
    name: str,
    dtypes: tuple[str, str, str],
    static_budget: int,
    external_buffers_disjoint: bool,
    outer_order: tuple[str, str] = ("m", "n"),
    verification_timeout_ms: int = 2000,
    max_repairs: int = 1,
    selected_contracts=None,
    reservations=(),
    pass_order=(),
    output_readout=None,
    readout_qualification_identity=None,
    reuse_pe_weights=False,
    group_dma=False,
):
    """Ordinary scheduling entry: choose, emit and independently verify output.

    No candidate preserves the caller's existing legal route. A checking-budget
    refusal may try one simple streaming loop; every address and dependency is
    rechecked. A concrete invariant failure never triggers fallback. The explicit
    disjointness premise belongs to the original source/runtime boundary, not
    to an optimization candidate. Missing decoder/completion facts diagnose.
    With selected contracts, geometry, formats and assigned windows are checked
    against both pinned documents; resources=None derives available budgets.
    Reservations name canonical backing bytes after upstream allocation.
    """
    if type(external_buffers_disjoint) is not bool or not external_buffers_disjoint:
        raise ValueError("dense scheduling cannot reorder possibly aliased external buffers")
    if type(reuse_pe_weights) is not bool:
        raise ValueError("PE weight reuse selection must be Boolean")
    if type(group_dma) is not bool:
        raise ValueError("DMA grouping selection must be Boolean")
    if group_dma and not _load_grouping_admission(problem, storage, adapter):
        return None
    original_adapter_check = _adapter_guard(adapter)
    if type(max_repairs) is not int or max_repairs not in (0, 1):
        raise ValueError("compact dense supports a bounded zero or one repair")

    from merlin.transforms.pipeline_state import (
        AnalysisStamp,
        PassSpec,
        PipelineContext,
        PipelineSession,
        validate_pass_order,
    )

    from .compact_verification import DenseVerificationUnavailable, verify_compact_dense
    from .dense_contracts import DenseIntegerReadout
    from .selected_contracts import SelectedSchedulingContracts, _parse

    if output_readout is not None:
        if (
            type(output_readout) is not DenseIntegerReadout
            or selected_contracts is None
            or type(readout_qualification_identity) is not str
            or not readout_qualification_identity.strip()
        ):
            raise ValueError("dense output readout requires original selected semantics and independent qualification")
        original_dense = _parse(
            selected_contracts.facts_bytes, selected_contracts.software_bytes, selected_contracts.target
        )[4]
        if original_dense is None or output_readout != original_dense.output_readout:
            raise ValueError("dense output readout differs from original software/RTL declaration")
    elif readout_qualification_identity is not None:
        raise ValueError("dense readout qualification requires a selected output transformation")
    checked_readout = None if output_readout is None else output_readout.identity

    if selected_contracts is not None:
        if type(selected_contracts) is not SelectedSchedulingContracts or selected_contracts.dense is None:
            raise ValueError("compact dense requires a selected dense scheduling contract")
        selected_contracts.revalidate()
        original_adapter_check()
        resources, adapter = selected_contracts.dense.bind(
            problem,
            storage,
            adapter,
            dtypes=dtypes,
            hardware=selected_contracts.hardware_identity,
            policy=selected_contracts.numerical_policy,
            resources=resources,
            reservations=reservations,
            output_readout=output_readout,
        )
    elif resources is None or reservations:
        raise ValueError("dense resource derivation and reservations require selected RTL/software contracts")
    original_adapter_check()
    adapter_check = _adapter_guard(adapter)

    def adapters_fresh():
        original_adapter_check()
        adapter_check()

    sources = _dense_library_sources()
    checked_selection = None if selected_contracts is None else _digest(selected_contracts.record())
    binding = (
        None
        if selected_contracts is None
        else selected_contracts.record(
            scopes=(
                "dense_geometry",
                "dense_storage_windows",
                "dense_numerics_and_execution",
                "dense_instruction_bindings",
            )
        )
    )
    if output_readout is not None:
        binding["output_readout"] = {
            "identity": checked_readout,
            "output_dtype": output_readout.output_dtype,
            "qualification_identity": readout_qualification_identity,
            "scope": "declared exact unary integer output; independent implementation qualification required",
        }
    context = PipelineContext(
        hardware=adapter.configuration_identity,
        numerical_policy=adapter.numerical_policy_identity,
        resources=_digest((asdict(resources), asdict(storage), tuple(asdict(r) for r in reservations))),
        execution="dense.execution.v3:"
        + _digest(
            (
                binding,
                adapter.qualification_identity,
                adapter.synchronous_completion,
                adapter.exact_accumulator_continuation,
                adapter.index_bits,
                *(((adapter.retained_weights, reuse_pe_weights),) if reuse_pe_weights else ()),
                *(((asdict(adapter.load_grouping), group_dma),) if group_dma else ()),
            )
        )
        + (
            ""
            if output_readout is None
            else f":readout={checked_readout}:qualification={readout_qualification_identity}"
        )
        + f":selection={checked_selection or 'unselected'}:index_bits={adapter.index_bits}"
        + ("" if checked_readout is None else f":readout={checked_readout}"),
        library=_digest(sources),
    )
    state = {
        "problem": asdict(problem),
        "resources": asdict(resources),
        "storage": asdict(storage),
        "dtypes": dtypes,
        "outer_order": outer_order,
        "name": name,
        "static_budget": static_budget,
        "external_buffers_disjoint": external_buffers_disjoint,
        "kernel": None,
        **({"reuse_pe_weights": True} if reuse_pe_weights else {}),
        **({"group_dma": True} if group_dma else {}),
        **({} if output_readout is None else {"output_readout": asdict(output_readout)}),
    }

    def fingerprint(current):
        return _digest({**current, "kernel": None if current["kernel"] is None else current["kernel"].digest()})

    def structure(current):
        if current["kernel"] is not None and check_structure(current["kernel"]):
            raise ValueError("dense pipeline requires structurally valid kernel IR")

    session = PipelineSession(
        state,
        context=context,
        ir_level="compact_dense",
        max_repairs=max_repairs,
        fingerprint=fingerprint,
        verify_ir=structure,
    )
    specs = (
        PassSpec("admission", 0, produces=("legality",), mutates_ir=False, ir_level="compact_dense"),
        PassSpec(
            "blocking", 2, requires=("legality",), produces=("choice",), mutates_ir=False, ir_level="compact_dense"
        ),
        PassSpec(
            "emission", 5, requires=("choice",), produces=("emitted",), invalidates=("*",), ir_level="compact_dense"
        ),
        PassSpec(
            "realization",
            5,
            requires=("emitted",),
            produces=("realization",),
            mutates_ir=False,
            ir_level="compact_dense",
        ),
    )
    names = tuple(spec.name for spec in specs)
    if type(pass_order) is not tuple or (
        pass_order and (len(pass_order) != len(names) or set(pass_order) != set(names))
    ):
        raise ValueError("explicit dense pass order must name each pass exactly once")
    ordered = tuple({spec.name: spec for spec in specs}[name] for name in (pass_order or names))
    validate_pass_order(ordered)

    def fresh():
        original_adapter_check()
        adapter_check()
        if (None if output_readout is None else output_readout.identity) != checked_readout:
            raise ValueError("dense output readout changed during compilation; replan")
        if selected_contracts is not None:
            selected_contracts.revalidate()
            original_adapter_check()
            adapter_check()
            if output_readout is not None and selected_contracts.dense.output_readout != output_readout:
                raise ValueError("dense selected output readout changed during compilation; replan")
            if _digest(selected_contracts.record()) != checked_selection:
                raise ValueError("dense selected RTL/software contracts changed during compilation; replan")
            original_adapter_check()
            adapter_check()
        if sources != _dense_library_sources():
            raise ValueError("dense shared library changed during compilation; replan")

    def emit(block):
        return emit_compact_dense(
            problem,
            block,
            storage,
            adapter,
            name=name,
            dtypes=dtypes,
            outer_order=outer_order,
            output_readout=output_readout,
            reuse_pe_weights=reuse_pe_weights,
            group_dma=group_dma,
        )

    repairs = []
    search = resources
    for attempt in range(max_repairs + 1):
        choice = None
        unavailable = None

        def admission():
            if not adapter.synchronous_completion or not adapter.exact_accumulator_continuation:
                raise ValueError("compact dense family needs synchronous completion and exact accumulator continuation")
            return {"legality": {"selected_contracts": binding}}

        def blocking():
            nonlocal choice
            choice = choose_dense_block(
                problem,
                search,
                static_budget=static_budget,
                price=lambda block: sum(1 for _ in _static_calls(emit(block).body)),
                admits=lambda block: storage.admits(problem, block, resources),
                require_reuse=False,
                k_limit=lambda m, n: storage.k_limit(problem, resources, m, n),
            )
            return {"choice": choice}

        def emission():
            state["kernel"] = emit(session.analysis("choice").block)
            return {"emitted": state["kernel"].digest()}

        def realization():
            nonlocal unavailable
            try:
                report = verify_compact_dense(
                    state["kernel"],
                    problem,
                    resources,
                    storage,
                    adapter,
                    static_budget=static_budget,
                    timeout_ms=verification_timeout_ms,
                    output_readout=output_readout,
                )
            except DenseVerificationUnavailable as exc:
                unavailable = exc
                report = None
            return {"realization": report}

        actions = dict(admission=admission, blocking=blocking, emission=emission, realization=realization)
        for spec in ordered:
            fresh()
            session.run(spec, actions[spec.name])
            fresh()
            if spec.name == "blocking" and choice is None:
                return None
        if unavailable is not None:
            if attempt == max_repairs or isinstance(unavailable.__cause__, ImportError):
                raise unavailable
            repairs.append({"discarded_kernel_sha256": state["kernel"].digest(), "reason": str(unavailable)})
            session.repair(stage=0, reason=str(unavailable))
            state["kernel"] = None
            search = replace(resources, max_m_blocks=1, max_n_blocks=1, max_k_blocks=1)
        else:
            report = session.analysis("realization")
            report["repairs"] = repairs
            report["pipeline"] = {
                "schema": "merlin.compact_dense_pipeline.v1",
                "context": asdict(context),
                "kernel_sha256": state["kernel"].digest(),
                "library_sources_sha256": sources,
                "selected_contracts": binding,
                "passes": [asdict(record) for record in session.records],
                "repair_rounds": session.repair_round,
            }
            return CompactDenseResult(
                state["kernel"],
                choice,
                report,
                adapter.index_bits,
                selected_contracts,
                AnalysisStamp(state["kernel"].digest(), context, "compact_dense"),
                adapter.index_bits,
                checked_selection,
                output_readout,
                checked_readout,
                _adapter_check=adapters_fresh,
            )


def _digest(value):
    import hashlib
    import json

    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _dense_library_sources():
    from .schedule_pipeline import shared_library_sources

    return shared_library_sources()
