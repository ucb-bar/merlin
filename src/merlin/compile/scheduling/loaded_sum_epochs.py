"""Bounded disjoint loaded-sum tiles with complete drains between phases.

The original compact checker still proves the scalar parameters and complete
source cover. This consumer only groups independent tiles into private windows:
all initializations finish before additions, all additions finish before stores,
and all stores finish before those windows are reused. No ordering between
pending loads to the same accumulator cell is assumed.
"""

from dataclasses import asdict, dataclass
import hashlib
import json

from .compact_loaded_sum import LoadedSumContract, LoadedSumLayout, verify_compact_loaded_sum


_FLAGS = ('disjoint_loads_pending', 'disjoint_stores_pending', 'ordered_configuration_capture',
          'finite_queue_progress', 'complete_device_drain', 'exclusive_private_state')


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


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
        if (type(contract) is not LoadedSumContract
            or self.hardware_identity != contract.implementation_identity
            or self.numerical_policy_identity != contract.scalar_proof_sha256
            or any(type(v) is not str or not v for v in (self.hardware_identity,
                self.numerical_policy_identity, self.qualification_identity))
            or any(getattr(self, flag) is not True for flag in _FLAGS)):
            raise ValueError('loaded-sum epochs require complete original target ordering and ownership premises')


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
            raise ValueError('loaded-sum epochs require original typed source storage and scalar contract')
        self.layout.validate()
        self.contract.validate()
        if type(self.target_law) is not LoadedSumEpochLaw:
            raise ValueError('loaded-sum epochs require original target law')
        self.target_law.require(self.contract)
        for digest in (self.source_kernel_sha256, self.source_proof_sha256):
            if type(digest) is not str or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
                raise ValueError('loaded-sum epochs require complete original source proof identities')
        if type(self.flatten_rows) is not bool:
            raise ValueError('loaded-sum row flattening requires an explicit Boolean')
        if (type(self.max_live_tiles) is not int or not 0 < self.max_live_tiles < 1 << 63
            or type(self.live_tiles) is not int or self.live_tiles != min(self.grouped_extent,
                self.capacity_tiles, self.max_live_tiles)):
            raise ValueError('loaded-sum epochs exceed bounded disjoint private capacity')
        if self.layout.accumulator_base_bytes + self.live_tiles * self.tile_bytes >= 1 << (self.layout.index_bits - 1):
            raise ValueError('loaded-sum epoch private addresses exceed the selected signed index width')
        if self.live_tiles < 1:
            raise ValueError('loaded-sum epochs have no complete private tile')

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
        return dict(schema='merlin.loaded_sum_epoch_plan.v1', plan_sha256=self.identity,
            source_kernel_sha256=self.source_kernel_sha256, source_proof_sha256=self.source_proof_sha256,
            live_tiles=self.live_tiles, private_interval_bytes=[l.accumulator_base_bytes,
                l.accumulator_base_bytes + self.live_tiles * self.tile_bytes],
            output_elements=l.rows * l.columns, source_tiles=mt * self.column_tiles,
            flatten_rows=self.flatten_rows, groups=groups, complete_phase_drains=3 * groups,
            phase_order=['initialize', 'complete', 'accumulate', 'complete', 'store', 'complete'],
            source_readout_parameters=self.contract.parameters,
            scope='bounded source/storage cover; target premises and runtime qualification are separate')

    def revalidate(self, kernel, layout, contract, project, *, scalar_proof):
        checked = verify_compact_loaded_sum(kernel, layout, contract, project, scalar_proof=scalar_proof)
        if (layout != self.layout or contract != self.contract or kernel.digest() != self.source_kernel_sha256
            or _digest(checked) != self.source_proof_sha256):
            raise ValueError('loaded-sum source, scalar parameters or original cover changed')
        self.__post_init__()
        return checked


def plan_loaded_sum_epochs(kernel, layout, contract, project, *, scalar_proof, target_law, max_live_tiles,
                           flatten_rows=False):
    """Consume the complete compact proof before grouping independent tiles.

    By default each group stays within one source row tile. ``flatten_rows``
    permits a group to cross row tiles while retaining the same private bound,
    scalar parameters, complete source cover and drains before slot reuse.
    """
    checked = verify_compact_loaded_sum(kernel, layout, contract, project, scalar_proof=scalar_proof)
    if type(max_live_tiles) is not int or not 0 < max_live_tiles < 1 << 63:
        raise ValueError('loaded-sum epochs need an explicit positive construction bound')
    tile_bytes = layout.tile_rows * layout.accumulator_row_bytes
    capacity = (layout.accumulator_capacity_bytes - layout.accumulator_base_bytes) // tile_bytes
    nt = (layout.columns + layout.tile_columns - 1) // layout.tile_columns
    if type(flatten_rows) is not bool:
        raise ValueError('loaded-sum row flattening requires an explicit Boolean')
    extent = nt * ((layout.rows + layout.tile_rows - 1) // layout.tile_rows if flatten_rows else 1)
    return LoadedSumEpochPlan(layout, contract, kernel.digest(), _digest(checked), max_live_tiles,
                              min(extent, capacity, max_live_tiles), target_law, flatten_rows)


def emit_loaded_sum_epochs(plan, builder, lower):
    """Emit counted control and every actual effect, including phase completion.

    The callback owns target encodings. ``complete_phase`` must fully drain the
    device; it cannot be a marker or delay. No callback may touch another slot,
    hide a configuration change or retain external aliases. Final output remains
    fully synchronous. Full and tail groups require bounded static construction.
    """
    if type(plan) is not LoadedSumEpochPlan or not callable(lower) or any(
        not callable(getattr(builder, name, None)) for name in ('op', 'cmp', 'select', 'loop')):
        raise ValueError('loaded-sum epochs require a checked plan, counted builder and target callbacks')
    plan.__post_init__()
    identity = plan.identity
    l = plan.layout
    mt = (l.rows + l.tile_rows - 1) // l.tile_rows
    nt = plan.column_tiles
    whole, tail = divmod(plan.grouped_extent, plan.live_tiles)

    def group(first, count, fixed_m=None):
        for index, kind in enumerate(('initialize', 'accumulate', 'store')):
            def slot_body(slot, kind=kind, index=index):
                tile = builder.op('add', first, slot)
                m = builder.op('udiv', tile, nt) if plan.flatten_rows else fixed_m
                n = builder.op('urem', tile, nt) if plan.flatten_rows else tile
                height = builder.select(builder.cmp('eq', m, mt - 1),
                                        (l.rows - 1) % l.tile_rows + 1, l.tile_rows)
                width = builder.select(builder.cmp('eq', n, nt - 1),
                                       (l.columns - 1) % l.tile_columns + 1, l.tile_columns)
                lower(kind, dict(row=builder.op('mul', m, l.tile_rows),
                    column=builder.op('mul', n, l.tile_columns), rows=height, columns=width,
                    local_byte=builder.op('add', l.accumulator_base_bytes,
                                          builder.op('mul', slot, plan.tile_bytes)),
                    accumulator_row_bytes=l.accumulator_row_bytes,
                    parameters=plan.contract.parameters[index]))
            builder.loop(0, count, 1, slot_body)
            lower('complete_phase', dict(phase=kind, tiles=count,
                local_first_byte=l.accumulator_base_bytes, local_extent_bytes=count * plan.tile_bytes))

    def groups(m=None):
        if whole:
            builder.loop(0, whole, 1, lambda n: group(builder.op('mul', n, plan.live_tiles), plan.live_tiles, m))
        if tail:
            group(whole * plan.live_tiles, tail, m)

    if plan.flatten_rows:
        groups()
    else:
        builder.loop(0, mt, 1, groups)
    plan.__post_init__()
    if plan.identity != identity:
        raise ValueError('loaded-sum plan changed during actual emission')
    return dict(**plan.report(), completion='real full-device drain between every phase and before private reuse')
