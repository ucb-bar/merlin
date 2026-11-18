"""Check a synchronous tiled loaded sum without enumerating its dynamic tiles.

This narrow checker admits two complete row-major inputs and one complete output.
An adapter projects actual instruction operands/effects. Its numerical proof and
synchronous implementation remain independently owned target obligations; this
module grants neither arithmetic equivalence nor execution qualification.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass

from merlin.compiler.kernel_ir.ir import DTYPE_BYTES, Call, Const, Expr, Kernel, Loop, Var, select

# ---- compact_loaded_sum ------------------------------------------------------------------------


def _identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _sha(value):
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


@dataclass(frozen=True)
class LoadedSumLayout:
    rows: int
    columns: int
    tile_rows: int
    tile_columns: int
    accumulator_base_bytes: int
    accumulator_row_bytes: int
    accumulator_capacity_bytes: int
    input_dtype: str = "i8"
    accumulator_dtype: str = "i32"
    output_dtype: str = "i8"
    index_bits: int = 64

    def validate(self):
        positive = (
            self.rows,
            self.columns,
            self.tile_rows,
            self.tile_columns,
            self.accumulator_row_bytes,
            self.accumulator_capacity_bytes,
        )
        if any(type(n) is not int or n <= 0 for n in positive):
            raise ValueError("loaded sum requires positive static extents and storage")
        if type(self.accumulator_base_bytes) is not int or self.accumulator_base_bytes < 0:
            raise ValueError("loaded sum requires a nonnegative private accumulator base")
        if any(t not in ("i8", "i16", "i32") for t in (self.input_dtype, self.accumulator_dtype, self.output_dtype)):
            raise ValueError("loaded sum requires explicit signed integer storage widths")
        if DTYPE_BYTES[self.accumulator_dtype] < DTYPE_BYTES[self.input_dtype]:
            raise ValueError("loaded sum accumulator is narrower than its operands")
        unit = DTYPE_BYTES[self.accumulator_dtype]
        if (
            self.accumulator_base_bytes % unit
            or self.accumulator_row_bytes % unit
            or self.accumulator_row_bytes < self.tile_columns * unit
        ):
            raise ValueError("private accumulator rows overlap or have unaligned storage")
        height, width = min(self.rows, self.tile_rows), min(self.columns, self.tile_columns)
        end = self.accumulator_base_bytes + (height - 1) * self.accumulator_row_bytes + width * unit
        if end > self.accumulator_capacity_bytes:
            raise ValueError("loaded sum private accumulator tile exceeds its capacity")
        if type(self.index_bits) is not int or self.index_bits not in (32, 64):
            raise ValueError("loaded sum needs explicit signed index width")
        largest = max(
            self.rows * self.columns * DTYPE_BYTES[self.input_dtype],
            self.rows * self.columns * DTYPE_BYTES[self.output_dtype],
            end,
            self.tile_rows,
            self.tile_columns,
        )
        if largest >= 1 << (self.index_bits - 1):
            raise ValueError("loaded sum byte addressing exceeds the selected signed index width")


@dataclass(frozen=True)
class LoadedSumContract:
    """Original scalar proof and three exact adapter parameter bindings.

    Parameters are opaque integer payloads (for example scale bits), never
    interpreted as target instructions here. The original owner must prove
    that each binding implements this very scalar proof before execution.
    """

    scalar_proof_sha256: str
    parameters: tuple[tuple[tuple[str, int], ...], ...]
    implementation_identity: str
    private_ownership: bool
    external_buffers_disjoint: bool

    def validate(self):
        if not _sha(self.scalar_proof_sha256):
            raise ValueError("loaded sum requires the exact original scalar proof identity")
        if (
            type(self.parameters) is not tuple
            or len(self.parameters) != 3
            or any(
                type(p) is not tuple
                or any(
                    type(item) is not tuple
                    or len(item) != 2
                    or type(item[0]) is not str
                    or not item[0]
                    or type(item[1]) is not int
                    for item in p
                )
                or len({item[0] for item in p}) != len(p)
                for p in self.parameters
            )
        ):
            raise ValueError("loaded sum requires three exact immutable integer parameter bindings")
        if type(self.implementation_identity) is not str or not self.implementation_identity:
            raise ValueError("loaded sum needs an explicit implementation identity")
        if self.private_ownership is not True or self.external_buffers_disjoint is not True:
            raise ValueError("loaded sum needs original private ownership and external disjointness")


@dataclass(frozen=True)
class LoadedSumEffect:
    """Actual logical effect decoded by the original instruction adapter.

    initialize reads lhs and overwrites the private rectangle; accumulate reads
    rhs and reads/adds/writes that rectangle; store reads it and writes output.
    All three must finish before the next effect begins. Coordinates are in
    logical elements; the private base and row stride are in bytes.
    """

    kind: str
    tensor: str
    row: Expr
    column: Expr
    rows: Expr
    columns: Expr
    accumulator_base_bytes: Expr
    accumulator_row_bytes: int
    scalar_proof_sha256: str
    parameters: tuple[tuple[str, int], ...]
    completion: str


def verify_compact_loaded_sum(
    kernel: Kernel,
    layout: LoadedSumLayout,
    contract: LoadedSumContract,
    project: Callable[[Call], LoadedSumEffect],
    *,
    scalar_proof: dict,
):
    """Prove the complete canonical two-loop cover and private tile lifetimes.

    Work is constant in the dynamic tile count: exactly three static effects
    are projected. Unknown loop/effect forms refuse; no finite expansion or
    sampling substitutes for verification. The scalar proof is byte-bound,
    not established here. Projectors must decode actual operands, not requests.
    """
    if type(layout) is not LoadedSumLayout or type(contract) is not LoadedSumContract:
        raise ValueError("explicit loaded sum layout and contract required")
    layout.validate()
    contract.validate()
    if type(scalar_proof) is not dict or _identity(scalar_proof) != contract.scalar_proof_sha256:
        raise ValueError("loaded sum scalar proof changed or differs from the original contract")
    if type(kernel) is not Kernel or kernel.attrs or not callable(project):
        raise ValueError("loaded sum requires a plain current counted kernel and effect projector")
    signature = tuple((a.name, a.shape, a.dtype, a.access) for a in kernel.args)
    shape = (layout.rows, layout.columns)
    if signature != (
        ("lhs", shape, layout.input_dtype, "read"),
        ("rhs", shape, layout.input_dtype, "read"),
        ("output", shape, layout.output_dtype, "write"),
    ):
        raise ValueError("loaded sum must cover both complete original inputs and its sole output")
    mt = (layout.rows + layout.tile_rows - 1) // layout.tile_rows
    nt = (layout.columns + layout.tile_columns - 1) // layout.tile_columns
    if len(kernel.body) != 1 or type(kernel.body[0]) is not Loop:
        raise ValueError("loaded sum needs one outer counted row loop")
    outer = kernel.body[0]
    if outer.extent != mt or type(outer.extent) is not int or len(outer.body) != 1 or type(outer.body[0]) is not Loop:
        raise ValueError("loaded sum row loop does not cover the original source")
    inner = outer.body[0]
    if (
        inner.extent != nt
        or type(inner.extent) is not int
        or outer.var == inner.var
        or type(outer.var) is not str
        or not outer.var
        or type(inner.var) is not str
        or not inner.var
        or len(inner.body) != 3
        or any(type(call) is not Call for call in inner.body)
    ):
        raise ValueError("loaded sum needs three ordered effects in each complete column tile")
    digest = kernel.digest()
    layout_before, contract_before = asdict(layout), asdict(contract)
    effects = []
    row = Var(outer.var) * layout.tile_rows
    column = Var(inner.var) * layout.tile_columns
    height = select(outer.var, mt - 1, (layout.rows - 1) % layout.tile_rows + 1, layout.tile_rows)
    width = select(inner.var, nt - 1, (layout.columns - 1) % layout.tile_columns + 1, layout.tile_columns)
    for index, (call, kind, tensor) in enumerate(
        zip(inner.body, ("initialize", "accumulate", "store"), ("lhs", "rhs", "output"), strict=True)
    ):
        if call != Call(call.instr, call.args):
            raise ValueError("loaded sum cannot hide async, configuration or staging metadata")
        effect = project(call)
        expected = LoadedSumEffect(
            kind,
            tensor,
            row,
            column,
            height,
            width,
            Const(layout.accumulator_base_bytes),
            layout.accumulator_row_bytes,
            contract.scalar_proof_sha256,
            contract.parameters[index],
            "synchronous",
        )
        if type(effect) is not LoadedSumEffect or effect != expected:
            raise ValueError("actual loaded sum effect differs in order, source cover, tail, storage or scalar proof")
        effects.append(asdict(effect))
    if (
        kernel.digest() != digest
        or asdict(layout) != layout_before
        or asdict(contract) != contract_before
        or _identity(scalar_proof) != contract.scalar_proof_sha256
    ):
        raise ValueError("loaded sum source, contract or scalar proof changed during projection")
    return dict(
        schema="merlin.compact_loaded_sum.v1",
        kernel_sha256=digest,
        scalar_proof_sha256=contract.scalar_proof_sha256,
        layout=layout_before,
        contract=contract_before,
        tile_grid=[mt, nt],
        dynamic_calls=3 * mt * nt,
        checked_static_effects=3,
        output_elements=layout.rows * layout.columns,
        complete_input_output_cover=True,
        synchronous_private_tile_reuse=True,
        proof=(
            "Disjoint clipped row and column tile intervals partition the original rectangle; "
            "each tile initializes, accumulates and stores the identical live rectangle "
            "in order before synchronous reuse"
        ),
        target_execution="not_established",
        numerical_equivalence="original scalar proof owner",
        effect_projection_sha256=_identity(effects),
    )


# ---- loaded_sum_epochs -------------------------------------------------------------------------
# Bounded disjoint loaded-sum tiles with complete drains between phases.
#
# The original compact checker still proves the scalar parameters and complete
# source cover. This consumer only groups independent tiles into private windows:
# all initializations finish before additions, all additions finish before stores,
# and all stores finish before those windows are reused. No ordering between
# pending loads to the same accumulator cell is assumed.

_FLAGS = (
    "disjoint_loads_pending",
    "disjoint_stores_pending",
    "ordered_configuration_capture",
    "finite_queue_progress",
    "complete_device_drain",
    "exclusive_private_state",
)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class LoadedSumEpochLaw:
    """Original target premises; names and scalar proofs do not establish them.

    Pending operations within one phase touch disjoint private rectangles.
    Configuration is initialized before issue and captured in queue order.
    Finite queues backpressure issue, and the callback performs a real complete
    device drain at every phase boundary and before return.
    """

    hardware_identity: str
    numerical_policy_identity: str
    qualification_identity: str
    disjoint_loads_pending: bool = False
    disjoint_stores_pending: bool = False
    ordered_configuration_capture: bool = False
    finite_queue_progress: bool = False
    complete_device_drain: bool = False
    exclusive_private_state: bool = False

    def require(self, contract):
        if (
            type(contract) is not LoadedSumContract
            or self.hardware_identity != contract.implementation_identity
            or self.numerical_policy_identity != contract.scalar_proof_sha256
            or any(
                type(v) is not str or not v
                for v in (self.hardware_identity, self.numerical_policy_identity, self.qualification_identity)
            )
            or any(getattr(self, flag) is not True for flag in _FLAGS)
        ):
            raise ValueError("loaded-sum epochs require complete original target ordering and ownership premises")


@dataclass(frozen=True)
class LoadedSumEpochPlan:
    layout: LoadedSumLayout
    contract: LoadedSumContract
    source_kernel_sha256: str
    source_proof_sha256: str
    max_live_tiles: int
    live_tiles: int
    target_law: LoadedSumEpochLaw
    flatten_rows: bool = False

    def __post_init__(self):
        if type(self.layout) is not LoadedSumLayout or type(self.contract) is not LoadedSumContract:
            raise ValueError("loaded-sum epochs require original typed source storage and scalar contract")
        self.layout.validate()
        self.contract.validate()
        if type(self.target_law) is not LoadedSumEpochLaw:
            raise ValueError("loaded-sum epochs require original target law")
        self.target_law.require(self.contract)
        for digest in (self.source_kernel_sha256, self.source_proof_sha256):
            if type(digest) is not str or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("loaded-sum epochs require complete original source proof identities")
        if type(self.flatten_rows) is not bool:
            raise ValueError("loaded-sum row flattening requires an explicit Boolean")
        if (
            type(self.max_live_tiles) is not int
            or not 0 < self.max_live_tiles < 1 << 63
            or type(self.live_tiles) is not int
            or self.live_tiles != min(self.grouped_extent, self.capacity_tiles, self.max_live_tiles)
        ):
            raise ValueError("loaded-sum epochs exceed bounded disjoint private capacity")
        if self.layout.accumulator_base_bytes + self.live_tiles * self.tile_bytes >= 1 << (self.layout.index_bits - 1):
            raise ValueError("loaded-sum epoch private addresses exceed the selected signed index width")
        if self.live_tiles < 1:
            raise ValueError("loaded-sum epochs have no complete private tile")

    @property
    def tile_bytes(self):
        return self.layout.tile_rows * self.layout.accumulator_row_bytes

    @property
    def capacity_tiles(self):
        return (self.layout.accumulator_capacity_bytes - self.layout.accumulator_base_bytes) // self.tile_bytes

    @property
    def column_tiles(self):
        return (self.layout.columns + self.layout.tile_columns - 1) // self.layout.tile_columns

    @property
    def row_tiles(self):
        return (self.layout.rows + self.layout.tile_rows - 1) // self.layout.tile_rows

    @property
    def grouped_extent(self):
        return self.column_tiles * (self.row_tiles if self.flatten_rows else 1)

    @property
    def identity(self):
        return _digest(asdict(self))

    def report(self):
        self.__post_init__()
        l = self.layout
        mt = (l.rows + l.tile_rows - 1) // l.tile_rows
        groups = ((self.grouped_extent + self.live_tiles - 1) // self.live_tiles) * (1 if self.flatten_rows else mt)
        return dict(
            schema="merlin.loaded_sum_epoch_plan.v1",
            plan_sha256=self.identity,
            source_kernel_sha256=self.source_kernel_sha256,
            source_proof_sha256=self.source_proof_sha256,
            live_tiles=self.live_tiles,
            private_interval_bytes=[
                l.accumulator_base_bytes,
                l.accumulator_base_bytes + self.live_tiles * self.tile_bytes,
            ],
            output_elements=l.rows * l.columns,
            source_tiles=mt * self.column_tiles,
            flatten_rows=self.flatten_rows,
            groups=groups,
            complete_phase_drains=3 * groups,
            phase_order=["initialize", "complete", "accumulate", "complete", "store", "complete"],
            source_readout_parameters=self.contract.parameters,
            scope="bounded source/storage cover; target premises and runtime qualification are separate",
        )

    def revalidate(self, kernel, layout, contract, project, *, scalar_proof):
        checked = verify_compact_loaded_sum(kernel, layout, contract, project, scalar_proof=scalar_proof)
        if (
            layout != self.layout
            or contract != self.contract
            or kernel.digest() != self.source_kernel_sha256
            or _digest(checked) != self.source_proof_sha256
        ):
            raise ValueError("loaded-sum source, scalar parameters or original cover changed")
        self.__post_init__()
        return checked


def plan_loaded_sum_epochs(
    kernel, layout, contract, project, *, scalar_proof, target_law, max_live_tiles, flatten_rows=False
):
    """Consume the complete compact proof before grouping independent tiles.

    By default each group stays within one source row tile. ``flatten_rows``
    permits a group to cross row tiles while retaining the same private bound,
    scalar parameters, complete source cover and drains before slot reuse.
    """
    checked = verify_compact_loaded_sum(kernel, layout, contract, project, scalar_proof=scalar_proof)
    if type(max_live_tiles) is not int or not 0 < max_live_tiles < 1 << 63:
        raise ValueError("loaded-sum epochs need an explicit positive construction bound")
    tile_bytes = layout.tile_rows * layout.accumulator_row_bytes
    capacity = (layout.accumulator_capacity_bytes - layout.accumulator_base_bytes) // tile_bytes
    nt = (layout.columns + layout.tile_columns - 1) // layout.tile_columns
    if type(flatten_rows) is not bool:
        raise ValueError("loaded-sum row flattening requires an explicit Boolean")
    extent = nt * ((layout.rows + layout.tile_rows - 1) // layout.tile_rows if flatten_rows else 1)
    return LoadedSumEpochPlan(
        layout,
        contract,
        kernel.digest(),
        _digest(checked),
        max_live_tiles,
        min(extent, capacity, max_live_tiles),
        target_law,
        flatten_rows,
    )


def emit_loaded_sum_epochs(plan, builder, lower):
    """Emit counted control and every actual effect, including phase completion.

    The callback owns target encodings. ``complete_phase`` must fully drain the
    device; it cannot be a marker or delay. No callback may touch another slot,
    hide a configuration change or retain external aliases. Final output remains
    fully synchronous. Full and tail groups require bounded static construction.
    """
    if (
        type(plan) is not LoadedSumEpochPlan
        or not callable(lower)
        or any(not callable(getattr(builder, name, None)) for name in ("op", "cmp", "select", "loop"))
    ):
        raise ValueError("loaded-sum epochs require a checked plan, counted builder and target callbacks")
    plan.__post_init__()
    identity = plan.identity
    l = plan.layout
    mt = (l.rows + l.tile_rows - 1) // l.tile_rows
    nt = plan.column_tiles
    whole, tail = divmod(plan.grouped_extent, plan.live_tiles)

    def group(first, count, fixed_m=None):
        for index, kind in enumerate(("initialize", "accumulate", "store")):

            def slot_body(slot, kind=kind, index=index):
                tile = builder.op("add", first, slot)
                m = builder.op("udiv", tile, nt) if plan.flatten_rows else fixed_m
                n = builder.op("urem", tile, nt) if plan.flatten_rows else tile
                height = builder.select(builder.cmp("eq", m, mt - 1), (l.rows - 1) % l.tile_rows + 1, l.tile_rows)
                width = builder.select(
                    builder.cmp("eq", n, nt - 1), (l.columns - 1) % l.tile_columns + 1, l.tile_columns
                )
                lower(
                    kind,
                    dict(
                        row=builder.op("mul", m, l.tile_rows),
                        column=builder.op("mul", n, l.tile_columns),
                        rows=height,
                        columns=width,
                        local_byte=builder.op(
                            "add", l.accumulator_base_bytes, builder.op("mul", slot, plan.tile_bytes)
                        ),
                        accumulator_row_bytes=l.accumulator_row_bytes,
                        parameters=plan.contract.parameters[index],
                    ),
                )

            builder.loop(0, count, 1, slot_body)
            lower(
                "complete_phase",
                dict(
                    phase=kind,
                    tiles=count,
                    local_first_byte=l.accumulator_base_bytes,
                    local_extent_bytes=count * plan.tile_bytes,
                ),
            )

    def groups(m=None):
        if whole:
            builder.loop(0, whole, 1, lambda n: group(builder.op("mul", n, plan.live_tiles), plan.live_tiles, m))
        if tail:
            group(whole * plan.live_tiles, tail, m)

    if plan.flatten_rows:
        groups()
    else:
        builder.loop(0, mt, 1, groups)
    plan.__post_init__()
    if plan.identity != identity:
        raise ValueError("loaded-sum plan changed during actual emission")
    return dict(**plan.report(), completion="real full-device drain between every phase and before private reuse")
