"""Source-checked, two-window dense panel issue and explicit completion epochs.

The ordinary dense verifier checks the actual serial projection of every source
product. A separate epoch checker proves that only disjoint future DMA writes
cross current compute reads, that accumulator writes in an epoch are disjoint,
and that every epoch drains. The target must independently qualify those issue,
configuration and full-drain semantics; this module supplies no target answers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from itertools import product
from math import prod

from merlin.sched.ir import Call, Const, Kernel, Loop, Ptr, Scaled, Select, Sum, Var, check_structure
from merlin.sched.ir.expr import add, mul, select
from merlin.sched.ir.integer_domain import check_integer_domain

from .block_schedule import Compute, InitializeAccumulator, Load, Preload, Store
from .compact_dense import DenseBiasSource, DenseLoopStorage, _digest, emit_compact_dense
from .dense_contracts import DenseIntegerReadout, DenseTiledIntegerReadout, TiledIntegerStore

DRAIN = "merlin_dense_panel_drain"
ISSUE_ORDERS = ("compute_then_prefetch", "prefetch_rhs_between_columns")


@dataclass(frozen=True)
class DensePanelOverlapAdmission:
    """Original selected implementation premises, never inferred from capacity.

    Qualification must cover ordered issue/backpressure, disjoint load/execute
    overlap, load configuration snapshotting, paired preload/compute consumption,
    disjoint output compute issue and full-device visibility before drain return.
    Bytes/identities alone are not qualification. The normal execution owner
    supplies and freezes this contract; candidate IR cannot grant itself one.
    Issued counts are finite compiler epoch budgets; pending counts are selected
    physical queue bounds. Qualified ready/valid backpressure enforces pending
    capacity while a larger epoch issues, without fabricated completion tokens.
    """

    hardware_identity: str
    numerical_policy_identity: str
    qualification_identity: str
    max_issued_compute_pairs: int
    max_issued_loads: int
    pending_compute_pairs: int
    pending_loads: int
    execution: str = "ordered_issue_disjoint_load_compute_full_drain"
    load_configuration: str = "ordered_snapshot_before_data_issue"
    descriptor: str = "preload_immediately_paired_with_compute"
    max_issued_readouts: int = 1
    pending_readouts: int = 1
    readout_configuration: str = "separate_drained_readout"

    def __post_init__(self):
        if (
            any(
                type(v) is not str or not v
                for v in (self.hardware_identity, self.numerical_policy_identity, self.qualification_identity)
            )
            or any(
                type(v) is not int or v <= 0
                for v in (
                    self.max_issued_compute_pairs,
                    self.max_issued_loads,
                    self.pending_compute_pairs,
                    self.pending_loads,
                    self.max_issued_readouts,
                    self.pending_readouts,
                )
            )
            or self.execution != "ordered_issue_disjoint_load_compute_full_drain"
            or self.load_configuration != "ordered_snapshot_before_data_issue"
            or self.descriptor != "preload_immediately_paired_with_compute"
            or self.readout_configuration not in ("separate_drained_readout", "ordered_snapshot_before_store_issue")
            or self.max_issued_readouts > 1
            and self.readout_configuration != "ordered_snapshot_before_store_issue"
        ):
            raise ValueError("panel overlap requires explicit selected issue, configuration and completion premises")


@dataclass(frozen=True)
class DenseReadoutConfigurationSnapshot:
    """Target-owned capture of each exact tiled readout's numerical state.

    The selected implementation must consume configuration in store order and
    retain that store's scale/activation through backpressure and completion.
    This does not permit combining differently configured stores, changing
    coefficients, or issuing a readout before its accumulator producer.
    """

    hardware_identity: str
    numerical_policy_identity: str
    qualification_identity: str
    readout_identity: str
    execution: str = "ordered_per_store_snapshot_until_completion"

    def validate_for(self, adapter, readout):
        if (any(type(value) is not str or not value.strip() for value in (
                self.hardware_identity, self.numerical_policy_identity,
                self.qualification_identity, self.readout_identity))
                or self.hardware_identity != adapter.configuration_identity
                or self.numerical_policy_identity != adapter.numerical_policy_identity
                or type(readout) is not DenseTiledIntegerReadout
                or self.readout_identity != readout.identity
                or self.execution != "ordered_per_store_snapshot_until_completion"):
            raise ValueError("varying readout requires its exact selected configuration snapshot mapping")
        readout.__post_init__()


@dataclass(frozen=True)
class DenseReadoutComputeAdmission:
    """Additional target conjunction; pure readout/reduction laws do not imply it.

    A store reads a completed ACC region while computation writes a disjoint
    region. Accepted requests retain their numerical configuration, finite
    queues apply backpressure, and one full drain closes both lifetimes before
    either region is reused. The source owner must qualify this conjunction.
    """

    hardware_identity: str
    numerical_policy_identity: str
    qualification_identity: str
    store_qualification_identity: str
    max_issued_readouts: int
    pending_readouts: int
    execution: str = "disjoint_accumulator_readout_compute_full_drain"
    configuration_snapshot: DenseReadoutConfigurationSnapshot | None = None

    def validate_for(self, adapter, original):
        if (any(type(value) is not str or not value.strip() for value in
                (self.hardware_identity, self.numerical_policy_identity,
                 self.qualification_identity, self.store_qualification_identity))
                or any(type(value) is not int or value <= 0 for value in
                       (self.max_issued_readouts, self.pending_readouts))
                or self.execution != "disjoint_accumulator_readout_compute_full_drain"
                or self.hardware_identity != adapter.configuration_identity
                or self.numerical_policy_identity != adapter.numerical_policy_identity
                or self.hardware_identity != original.hardware_identity
                or self.numerical_policy_identity != original.numerical_policy_identity
                or self.max_issued_readouts > original.max_issued_readouts
                or self.pending_readouts != original.pending_readouts
                or original.readout_configuration != "ordered_snapshot_before_store_issue"):
            raise ValueError("readout/compute needs the selected separate snapshot and completion conjunction")


@dataclass(frozen=True)
class DenseReadyReadoutAdmission:
    """Separate causal ACC producer-to-readout completion capability.

    Ordered issue alone is insufficient. The selected implementation must keep
    overlapping stores dependent on producer completion and pending ACC writes,
    snapshot each readout, apply finite-queue backpressure, and fully drain before
    ownership reuse. No compiler-chosen cycle spacing is part of this contract.
    Varying tiled numerical state additionally needs ``configuration_snapshot``
    bound to that complete original readout; uniform-only callers grant none.
    Concurrent disjoint input loads additionally require the separately selected
    ``disjoint_load_qualification_identity`` for the complete LD/compute/readout
    conjunction. Its absence keeps the original refusal of mixed load epochs.
    """

    hardware_identity: str
    numerical_policy_identity: str
    qualification_identity: str
    store_qualification_identity: str
    max_issued_readouts: int
    pending_readouts: int
    execution: str = "ordered_accumulator_raw_readout_full_drain"
    configuration_snapshot: DenseReadoutConfigurationSnapshot | None = None
    disjoint_load_qualification_identity: str | None = None

    def validate_for(self, adapter, original):
        if self.execution != "ordered_accumulator_raw_readout_full_drain":
            raise ValueError("causal readout requires the separate producer completion execution law")
        if (self.disjoint_load_qualification_identity is not None
                and (type(self.disjoint_load_qualification_identity) is not str
                     or not self.disjoint_load_qualification_identity.strip())):
            raise ValueError("ready readout/load overlap needs its separate selected execution conjunction")
        DenseReadoutComputeAdmission(
            self.hardware_identity, self.numerical_policy_identity,
            self.qualification_identity, self.store_qualification_identity,
            self.max_issued_readouts, self.pending_readouts,
            configuration_snapshot=self.configuration_snapshot).validate_for(adapter, original)


def _calls(body):
    for statement in body:
        if type(statement) is Loop:
            yield from _calls(statement.body)
        else:
            yield statement


def _shift(value, old, new, scale, offset):
    """Substitute old = scale*new+offset, including exact equality predicates."""
    if type(value) is Ptr:
        return replace(value, offset=_shift(value.offset, old, new, scale, offset))
    if type(value) is Var:
        return (Const(offset) if new is None else Var(new) * scale + offset) if value.name == old else value
    if type(value) is Sum:
        return add(*(_shift(v, old, new, scale, offset) for v in value.terms))
    if type(value) is Scaled:
        return mul(_shift(value.expr, old, new, scale, offset), value.factor)
    if type(value) is Select:
        yes, no = (_shift(v, old, new, scale, offset) for v in (value.then, value.other))
        if value.var != old:
            return select(value.var, value.equals, yes, no)
        if new is None:
            return yes if value.equals == offset else no
        quotient, remainder = divmod(value.equals - offset, scale)
        return no if remainder else select(new, quotient, yes, no)
    return value


def _remap(body, old, new, scale=1, offset=0):
    result = []
    for statement in body:
        if type(statement) is Loop:
            result.append(replace(statement, body=_remap(statement.body, old, new, scale, offset)))
        else:

            def values(items):
                return tuple((key, _shift(value, old, new, scale, offset)) for key, value in items)

            result.append(
                replace(
                    statement,
                    args=values(statement.args),
                    sets=values(statement.sets),
                    assumes=values(statement.assumes),
                )
            )
    return tuple(result)


def _windows(problem, block, storage, resources, offsets=None):
    if block.k != 1 or storage.accumulator is None:
        raise ValueError("panel overlap requires one primitive K slice and assigned private accumulators")
    if block.m > resources.max_m_blocks or block.n > resources.max_n_blocks or block.k > resources.max_k_blocks:
        raise ValueError("panel block exceeds original supported domain")
    from merlin.transforms.dense_planning import dense_blocks

    canonical = dense_blocks(problem, replace(resources, max_m_blocks=block.m, max_n_blocks=block.n, max_k_blocks=1))
    if block not in canonical:
        raise ValueError("panel block differs from original geometry/count/resource derivation")
    rows = (block.m * problem.tile_m, block.n * problem.tile_k)
    views = (storage.lhs, storage.rhs)
    if any(2 * count > view.capacity for count, view in zip(rows, views)):
        raise ValueError("two complete panel windows exceed original assigned rows")
    if 2 * sum(count * view.unit_bytes for count, view in zip(rows, views)) > resources.operand_bytes:
        raise ValueError("two complete padded panels exceed original operand byte budget")
    if not storage.admits(problem, block, resources):
        raise ValueError("panel block disagrees with original resource/window admission")
    if offsets is None:
        offsets = ((0, 0), rows)
    if (
        type(offsets) is not tuple
        or len(offsets) != 2
        or any(type(slot) is not tuple or len(slot) != 2 for slot in offsets)
        or any(type(offset) is not int or offset < 0 for slot in offsets for offset in slot)
    ):
        raise ValueError("panel windows require two explicit LHS/RHS nonnegative row-offset pairs")
    slots = []
    for slot in offsets:
        narrowed = []
        for view, count, offset in zip(views, rows, slot, strict=True):
            if offset + count > view.capacity or (view.base + offset) % view.alignment:
                raise ValueError("panel window escapes or misaligns the original assigned view")
            narrowed.append(replace(view, base=view.base + offset, capacity=count))
        slots.append(DenseLoopStorage(*narrowed, storage.accumulator))
    assigned = [view for slot in slots for view in (slot.lhs, slot.rhs)] + [storage.accumulator]
    for index, left in enumerate(assigned):
        for right in assigned[index + 1 :]:
            if (
                left.backing == right.backing
                and left.base * left.unit_bytes < (right.base + right.capacity) * right.unit_bytes
                and right.base * right.unit_bytes < (left.base + left.capacity) * left.unit_bytes
            ):
                raise ValueError("panel windows overlap simultaneously live physical byte intervals")
    return tuple(slots)


def select_dense_panel_window_offsets(problem, block, resources, storage, *, banks, max_candidates=4096):
    """Select disjoint bank sets for two slots inside existing assignments.

    Banks are caller-selected canonical BYTE intervals (exclusive capacity),
    derived from its independently checked physical mapping. They guide a
    placement proposal only: capacity never grants overlap or timing semantics.
    The ordinary source and actual-band lifetime checks still check all emitted
    addresses. Search visits physical bank alternatives, never dynamic tiles.
    """
    from math import lcm

    from merlin.semantic_compiler.allocation_types import StorageBank

    if (
        type(banks) is not tuple
        or not banks
        or any(type(bank) is not StorageBank or bank.unit != "bytes" for bank in banks)
        or len({bank.name for bank in banks}) != len(banks)
        or type(max_candidates) is not int
        or max_candidates <= 0
    ):
        raise ValueError("panel bank selection requires selected byte intervals and an explicit search budget")
    for index, left in enumerate(banks):
        for right in banks[index + 1 :]:
            if left.backing == right.backing and left.start < right.capacity and right.start < left.capacity:
                raise ValueError("selected physical banks have overlapping canonical byte intervals")
    _windows(problem, block, storage, resources)
    choices = []
    for view, count in zip((storage.lhs, storage.rhs), (block.m * problem.tile_m, block.n * problem.tile_k)):
        role = []
        for bank in banks:
            if bank.backing != view.backing:
                continue
            alignment = lcm(view.unit_bytes * view.alignment, bank.alignment)
            low = max(view.base * view.unit_bytes, bank.start)
            start = ((low + alignment - 1) // alignment) * alignment
            end = min((view.base + view.capacity) * view.unit_bytes, bank.capacity)
            if start + count * view.unit_bytes <= end:
                role.append((bank.name, start // view.unit_bytes - view.base))
        choices.append(sorted(role, key=lambda item: (item[1], item[0])))
    examined = 0
    # Both complete operand sets must use disjoint physical bank identities;
    # separating LHS alone is insufficient if future RHS still hits active LHS.
    for lhs0, rhs0, lhs1, rhs1 in product(choices[0], choices[1], choices[0], choices[1]):
        examined += 1
        if examined > max_candidates:
            raise ValueError("panel bank-placement search budget exhausted; no physical capacity conclusion")
        if {lhs0[0], rhs0[0]} & {lhs1[0], rhs1[0]}:
            continue
        offsets = ((lhs0[1], rhs0[1]), (lhs1[1], rhs1[1]))
        _windows(problem, block, storage, resources, offsets)
        return offsets
    raise ValueError("no two disjoint operand bank sets fit the original assigned views")


def build_dense_panel_pipeline(
    problem,
    block,
    resources,
    storage,
    adapter,
    *,
    name,
    dtypes,
    prefetch=True,
    outer_order=("m", "n"),
    output_readout=None,
    bias_source=None,
    issue_order="compute_then_prefetch",
    window_offsets=None,
):
    """Construct counted panel overlap; this is an unadmitted proposal.

    Two windows retain block-M/block-N operand reuse. Increasing K and complete
    accumulators remain inside each output block. A serialized control has the
    identical data calls and addresses, with an extra drain between compute and
    next-panel movement. No image or dynamic instruction domain is expanded.
    """
    if any(dtype not in ("i8", "i16", "i32", "i64") for dtype in dtypes) or type(prefetch) is not bool:
        raise ValueError("panel family requires explicit signed integer source and issue policy")
    if type(issue_order) is not str or issue_order not in ISSUE_ORDERS:
        raise ValueError("unsupported executable panel issue order")
    slots = _windows(problem, block, storage, resources, window_offsets)
    originals = tuple(
        emit_compact_dense(
            problem,
            block,
            s,
            adapter,
            name=name,
            dtypes=dtypes,
            outer_order=outer_order,
            output_readout=output_readout,
            bias_source=bias_source,
        )
        for s in slots
    )
    serial = 0

    def epoch(body):
        nonlocal serial
        serial += 1
        return Loop("dense_epoch_" + str(serial), 1, (*body, Call(DRAIN, ())))

    def kinds(body):
        return {type(adapter.project(call)) for call in _calls(body)}

    def pair(before, alternate):
        if len(before) != len(alternate):
            raise ValueError("alternate physical window changed the panel control structure")
        result = []
        for left, right in zip(before, alternate, strict=True):
            if type(left) is not type(right):
                raise ValueError("alternate physical window changed the panel control structure")
            if type(left) is not Loop:
                if (
                    type(adapter.project(left)) not in (Store, TiledIntegerStore, InitializeAccumulator)
                    or left != right
                ):
                    raise ValueError("panel pipeline requires structured source calls")
                result.append(epoch((left,)))
                continue
            if (left.var, left.extent) != (right.var, right.extent):
                raise ValueError("alternate physical window changed the source domain")
            types = kinds(left.body)
            if types <= {Store, TiledIntegerStore, InitializeAccumulator}:
                result.append(replace(left, body=pair(left.body, right.body)))
                continue
            if types == {Load, Preload, Compute}:
                load_parts, compute_parts = [], []
                for source in (left, right):
                    loads, computes, computing = [], [], False
                    for child in source.body:
                        selected = kinds((child,))
                        if selected == {Load} and not computing:
                            loads.append(child)
                        elif selected == {Preload, Compute}:
                            computing = True
                            computes.append(child)
                        else:
                            raise ValueError("panel must contain complete load prefixes then paired compute loops")
                    if not loads or not computes:
                        raise ValueError("incomplete panel body")
                    load_parts.append(tuple(loads))
                    compute_parts.append(tuple(computes))
                variable = left.var
                period = "dense_period_" + str(serial + 1)
                if period in {loop.var for loop in _loops(originals[0].body)}:
                    raise ValueError("pipeline period variable collides with original control")

                def loads(slot, new, scale, offset):
                    return _remap(load_parts[slot], variable, new, scale, offset)

                def computes(slot, new, scale, offset):
                    return _remap(compute_parts[slot], variable, new, scale, offset)

                def stage(slot, new, scale, offset):
                    current = computes(slot, new, scale, offset)
                    future = loads(1 - slot, new, scale, offset + 1)
                    if issue_order == "prefetch_rhs_between_columns":
                        # Decode actual canonical K=1/M/N streams, transpose
                        # independent outputs, zip one RHS tile after each M
                        # column. The independent source/epoch checks retain
                        # responsibility for every actual operand and reduction.
                        def chain(body, depth):
                            loops = []
                            for _ in range(depth):
                                if len(body) != 1 or type(body[0]) is not Loop:
                                    raise ValueError("interleaved panel needs complete canonical counted streams")
                                loops.append(body[0])
                                body = body[0].body
                            return loops, body

                        compute_loops, pairs = chain(current, 3)
                        if len(future) != 2:
                            raise ValueError("interleaved panel needs separate complete LHS and RHS streams")
                        lhs_loops, lhs_calls = chain(future[:1], 2)
                        rhs_loops, rhs_calls = chain(future[1:], 2)
                        ck, cm, cn = compute_loops
                        lm, lk = lhs_loops
                        rk, rn = rhs_loops
                        if (
                            (ck.extent, lk.extent, rk.extent) != (1, 1, 1)
                            or cm.extent != lm.extent
                            or cn.extent != rn.extent
                            or kinds(lhs_calls) != {Load}
                            or kinds(rhs_calls) != {Load}
                            or {adapter.project(c).role for c in lhs_calls} != {"lhs"}
                            or {adapter.project(c).role for c in rhs_calls} != {"weight"}
                            or [type(adapter.project(c)) for c in pairs] != [Preload, Compute]
                        ):
                            raise ValueError("interleaved panel streams differ from exact primitive K/M/N domains")
                        column_compute = replace(cm, body=_remap(pairs, ck.var, None, 1, 0))
                        column_load = _remap(_remap(rhs_calls, rk.var, None, 1, 0), rn.var, cn.var)
                        if prefetch:
                            columns = replace(cn, body=(column_compute, *column_load))
                            return (epoch((*future[:1], columns)),)
                        columns = replace(cn, body=(epoch((column_compute,)), epoch(column_load)))
                        return (epoch(future[:1]), columns)
                    return (epoch((*current, *future)),) if prefetch else (epoch(current), epoch(future))

                count, tail = divmod(left.extent - 1, 2)
                result.append(epoch(loads(0, None, 1, 0)))
                if count:
                    result.append(Loop(period, count, (*stage(0, period, 2, 0), *stage(1, period, 2, 1))))
                if tail:
                    result.extend(stage(0, None, 1, 2 * count))
                result.append(epoch(computes((left.extent - 1) % 2, None, 1, left.extent - 1)))
            else:
                result.append(replace(left, body=pair(left.body, right.body)))
        return tuple(result)

    kernel = replace(originals[0], body=pair(originals[0].body, originals[1].body))
    if check_structure(kernel):
        raise ValueError("panel pipeline produced invalid control/completion scopes")
    check_integer_domain(kernel, adapter.index_bits)
    return kernel


def _loops(body):
    for statement in body:
        if type(statement) is Loop:
            yield statement
            yield from _loops(statement.body)


def dense_panel_rhs_stripes(kernel, adapter, *, factor, retain_short_tails=False):
    """Propose typed split/fission of actual mixed compute/RHS column loops.

    A stripe retains each physical preload/compute pair, executes its current
    output columns, then issues their future RHS loads. The unchanged epoch
    drain stays outside the stripe. This exposes load-only inner loops to the
    ordinary coalescer; the final expanded-band/source checker owns legality.
    Serialized per-column epochs and compute-only/grouped streams refuse: the
    helper never crosses a completion or substitutes a different issue policy.
    With explicit ``retain_short_tails``, mixed loops shorter than one factor
    keep their original paired compute/load order. Only complete stripes receive
    split/fission requests; the final actual owner must still verify all tails.
    """
    from .loop_schedule import striped_fission_transforms

    if (type(kernel) is not Kernel or check_structure(kernel) or type(factor) is not int or not 2 <= factor <= 64
            or type(retain_short_tails) is not bool):
        raise ValueError("panel stripes require actual checked loop structure and an explicit factor 2..64")
    paths, short_paths = [], []

    def walk(body, prefix=(), inside_epoch=False):
        for statement in body:
            if type(statement) is not Loop:
                continue
            path = (*prefix, statement.var)
            drained = bool(statement.body and statement.body[-1] == Call(DRAIN, ()))
            if inside_epoch and len(statement.body) == 2:
                compute, load = statement.body
                if (
                    type(compute) is Loop
                    and len(compute.body) == 2
                    and all(type(call) is Call for call in compute.body)
                    and type(load) is Call
                    and load.instr != DRAIN
                    and [type(adapter.project(call)) for call in compute.body] == [Preload, Compute]
                    and type(operation := adapter.project(load)) is Load
                    and operation.role == "weight"
                ):
                    (short_paths if retain_short_tails and statement.extent < factor else paths).append(path)
            walk(statement.body, path, inside_epoch or drained)

    walk(kernel.body)
    if not paths:
        if short_paths:
            return ()
        raise ValueError("panel stripes require a mixed current-compute/future-RHS loop inside one drained epoch")
    return striped_fission_transforms(kernel, tuple(paths), factor=factor, cut=1)


def _serial_projection(kernel):
    def body(statements):
        return tuple(
            replace(s, body=body(s.body)) if type(s) is Loop else s
            for s in statements
            if not (type(s) is Call and s.instr == DRAIN)
        )

    return replace(kernel, body=body(kernel.body))


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
    from merlin.sched.ir import Expr
    from merlin.sched.ir.expr import free_vars

    from .tile_load_coalescing import _plain

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
            return add(Const(constant), *(atom * coefficient for atom, coefficient in terms.items()),
                       Var(outer.var) * b)

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
        if (any(type(v) is not int or v <= 0 for v in (outer.extent, inner.extent))
                or extent >= 1 << (adapter.index_bits - 1)):
            raise ValueError("joined counted extent exceeds original signed index domain")
        return replace(outer, extent=extent, body=body(inner.body))

    def walk(body):
        result = []
        for statement in body:
            if type(statement) is Loop:
                statement = replace(statement, body=walk(statement.body))
                if (len(statement.body) == 1 and type(inner := statement.body[0]) is Loop
                        and pure(inner.body)):
                    try:
                        joined = collapse(statement, inner)
                    except ValueError:
                        pass
                    else:
                        changes.append(dict(outer=statement.var, inner=inner.var,
                                            outer_extent=statement.extent, inner_extent=inner.extent,
                                            joined_extent=joined.extent,
                                            forward="t=outer*inner_extent+inner",
                                            inverse="outer=t//inner_extent; inner=t%inner_extent",
                                            proof="every scalar operand: outer_coefficient="
                                            "inner_extent*inner_coefficient"))
                        statement = joined
            result.append(statement)
        return tuple(result)

    rewritten = replace(kernel, body=walk(kernel.body))
    if check_structure(rewritten):
        raise ValueError("compute radix checking changed original binder custody")
    if changes:
        check_integer_domain(rewritten, adapter.index_bits)
    return rewritten, changes


def verify_dense_panel_pipeline(
    kernel,
    problem,
    block,
    resources,
    storage,
    adapter,
    admission,
    *,
    static_budget,
    timeout_ms=5000,
    output_readout=None,
    bias_source=None,
):
    """Check all source products plus actual epoch read/write/completion hazards."""
    if (
        type(admission) is not DensePanelOverlapAdmission
        or admission.hardware_identity != adapter.configuration_identity
        or admission.numerical_policy_identity != adapter.numerical_policy_identity
    ):
        raise ValueError("panel overlap differs from original selected execution/numerical capability")
    admission.__post_init__()
    _windows(problem, block, storage, resources)
    return _verify_dense_epochs(
        kernel, problem, resources, storage, adapter, admission,
        static_budget=static_budget, timeout_ms=timeout_ms,
        output_readout=output_readout, bias_source=bias_source,
    )


def _verify_dense_epochs(
    kernel, problem, resources, storage, adapter, admission, *, static_budget,
    timeout_ms, output_readout=None, bias_source=None, ordered_accumulator_commits=False,
    source_phase_fission=False, readout_compute_admission=None, readout_source_projection=None,
):
    """Shared proof engine; public owners validate their distinct typed laws.

    Ordered accumulator commitment is an explicit additional implementation
    premise, never inferred from storage, issue order or this private switch.
    Operand/load disjointness and the complete serial source proof always run.
    """
    import time

    import z3

    if readout_compute_admission is not None:
        if type(readout_compute_admission) not in (DenseReadoutComputeAdmission, DenseReadyReadoutAdmission):
            raise ValueError("typed additional readout/compute conjunction required")
        readout_compute_admission.validate_for(adapter, admission)
        snapshot = readout_compute_admission.configuration_snapshot
        varying = (type(output_readout) is DenseTiledIntegerReadout
                   and len(set(output_readout.column_scale_bits)) != 1)
        if varying and snapshot is None:
            raise ValueError("mixed per-tile readout configuration has no selected conjunction")
        if snapshot is not None:
            if type(snapshot) is not DenseReadoutConfigurationSnapshot:
                raise ValueError("typed exact readout configuration snapshot required")
            snapshot.validate_for(adapter, output_readout)

    from .compact_verification import _project, verify_compact_dense

    if (
        check_structure(kernel)
        or type(static_budget) is not int
        or static_budget <= 0
        or sum(1 for _ in _calls(kernel.body)) > static_budget
        or type(timeout_ms) is not int
        or timeout_ms <= 0
    ):
        raise ValueError("invalid panel pipeline structure or original checking/command budget")
    check_integer_domain(kernel, adapter.index_bits)
    deadline = time.monotonic() + timeout_ms / 1000
    sites = _project(_serial_projection(kernel), adapter, fold_singleton_loops=False)
    epochs = []
    serial = 0

    def structure(body, ancestry=(), inside=False):
        nonlocal serial
        for statement in body:
            if type(statement) is Loop:
                serial += 1
                scope = (serial, statement.var, statement.extent)
                drains = [s for s in statement.body if type(s) is Call and s.instr == DRAIN]
                if drains:
                    if (
                        inside
                        or statement.extent != 1
                        or len(drains) != 1
                        or statement.body[-1] is not drains[0]
                        or drains[0] != Call(DRAIN, ())
                    ):
                        raise ValueError("each panel epoch needs one final explicit full drain")
                    epochs.append((*ancestry, scope))
                    structure(statement.body[:-1], (*ancestry, scope), True)
                else:
                    structure(statement.body, (*ancestry, scope), inside)
            elif not inside:
                raise ValueError("every movement, compute and readout must be inside a drained panel epoch")

    structure(kernel.body)
    if not epochs:
        raise ValueError("panel pipeline has no explicit completion epochs")
    views = {"lhs": storage.lhs, "weight": storage.rhs, "acc": storage.accumulator}

    def expression(value, env):
        if type(value) is Const:
            return z3.IntVal(value.value)
        if type(value) is int:
            return z3.IntVal(value)
        if type(value) is Var:
            return env[value.name]
        if type(value) is Sum:
            return z3.Sum(*(expression(v, env) for v in value.terms))
        if type(value) is Scaled:
            return expression(value.expr, env) * value.factor
        if type(value) is Select:
            return z3.If(env[value.var] == value.equals, expression(value.then, env), expression(value.other, env))
        raise ValueError("unknown panel address expression")

    def domain(site, tag):
        env = {name: z3.Int(f"{tag}:{scope}:{name}") for scope, name, _ in site.scopes}
        return env, [z3.And(env[name] >= 0, env[name] < extent) for _, name, extent in site.scopes]

    def interval(view, row, rows, env):
        start = expression(row, env) * view.unit_bytes
        return view.backing, start, start + expression(rows, env) * view.unit_bytes

    def prove_disjoint(a, b, outer, *, exclude_same=False):
        sa, va, ra, na = a
        sb, vb, rb, nb = b
        if va.backing != vb.backing:
            return
        ae, ac = domain(sa, "a")
        be, bc = domain(sb, "b")
        same = [ae[name] == be[name] for _, name, _ in outer]
        if exclude_same and sa.ordinal == sb.ordinal:
            local = [ae[name] != be[name] for scope, name, _ in sa.scopes if scope not in {s[0] for s in outer}]
            if not local:
                return
            same.append(z3.Or(*local))
        _, al, ah = interval(va, ra, na, ae)
        _, bl, bh = interval(vb, rb, nb, be)
        remaining = int((deadline - time.monotonic()) * 1000)
        if remaining <= 0:
            raise ValueError("panel epoch checking budget exhausted")
        solver = z3.Solver()
        solver.set(timeout=remaining)
        solver.add(*ac, *bc, *same, al < bh, bl < ah)
        if solver.check() != z3.unsat:
            raise ValueError("panel epoch has overlapping in-flight operand or accumulator accesses")

    def prove_producer_before_readout(read, write, outer):
        # Each overlapping producer INSTANCE must already be issued before the
        # store. The new target law supplies the causal commit/readiness edge;
        # this source proof never substitutes issue for hardware completion.
        reader, rv, rr, rn = read
        writer, wv, wr, wn = write
        if rv.backing != wv.backing:
            return
        re, rc = domain(reader, "readout")
        we, wc = domain(writer, "producer")
        same = [re[name] == we[name] for _, name, _ in outer]
        _, rl, rh = interval(rv, rr, rn, re)
        _, wl, wh = interval(wv, wr, wn, we)

        def position(site, env):
            return tuple(z3.IntVal(v) if type(v) is int else env[v[1]] for v in site.path)

        left, right = position(writer, we), position(reader, re)
        before = z3.Or(*(z3.And(*(left[j] == right[j] for j in range(i)), left[i] < right[i])
                         for i in range(min(len(left), len(right)))))
        remaining = int((deadline - time.monotonic()) * 1000)
        if remaining <= 0:
            raise ValueError("causal readout epoch checking budget exhausted")
        solver = z3.Solver()
        solver.set(timeout=remaining)
        solver.add(*rc, *wc, *same, rl < wh, wl < rh, z3.Not(before))
        if solver.check() != z3.unsat:
            raise ValueError("overlapping ACC producer must precede readout; later writes/reuse are unqualified")

    def prove_assigned(access):
        site, view, row, rows = access
        env, constraints = domain(site, "assigned")
        _, low, high = interval(view, row, rows, env)
        solver = z3.Solver()
        remaining = int((deadline - time.monotonic()) * 1000)
        if remaining <= 0:
            raise ValueError("panel assigned-access checking budget exhausted")
        solver.set(timeout=remaining)
        solver.add(*constraints, z3.Or(low < view.base * view.unit_bytes,
                                      high > (view.base + view.capacity) * view.unit_bytes,
                                      high <= low))
        if solver.check() != z3.unsat:
            raise ValueError("actual panel access escapes its original assigned private view")

    summaries = []
    for outer in epochs:
        marker = outer[-1][0]
        selected = [s for s in sites if marker in {scope for scope, _, _ in s.scopes}]
        loads = []
        reads = []
        writes = []
        stores = 0
        store_reads = []
        initializations = 0
        pairs = 0
        load_count = 0
        pending = None
        for site in selected:
            op = site.operation
            local = prod(extent for scope, _, extent in site.scopes if scope not in {s[0] for s in outer})
            if type(op) is Load:
                if pending is not None:
                    raise ValueError("epoch splits a physical preload/compute pair")
                load_count += local
                loads.append((site, views[op.role], op.row, op.rows))
            elif type(op) is Preload:
                if pending is not None:
                    raise ValueError("epoch overwrites an unconsumed preload descriptor")
                pending = site
            elif type(op) is Compute:
                if pending is None or pending.scopes != site.scopes:
                    raise ValueError("epoch has an unpaired compute")
                pre = pending.operation
                pairs += local
                reads.extend(
                    ((site, storage.lhs, op.input_row, op.rows), (site, storage.rhs, pre.weight_row, pre.weight_rows))
                )
                writes.append((site, storage.accumulator, pre.accumulator_row, pre.rows))
                pending = None
            elif type(op) in (Store, TiledIntegerStore):
                stores += local
                store_reads.append((site, storage.accumulator, op.accumulator_row, op.rows))
            elif type(op) is InitializeAccumulator:
                initializations += local
            else:
                raise ValueError("unknown panel effect")
        if (
            pending is not None
            or stores
            and (loads or writes or initializations)
            and readout_compute_admission is None
            or stores > admission.max_issued_readouts
            or initializations
            and (loads or writes or stores)
            or initializations > 1
        ):
            raise ValueError("initialization or readout exceeds the selected separate completion epoch")
        if stores and (loads or writes):
            if not writes or initializations or stores > readout_compute_admission.max_issued_readouts:
                raise ValueError("mixed readout epoch needs disjoint compute and its explicit issued store bound")
            if (type(readout_compute_admission) is DenseReadyReadoutAdmission and loads
                    and readout_compute_admission.disjoint_load_qualification_identity is None):
                raise ValueError("causal-ready readout has no mixed DMA-load execution conjunction")
            for read in store_reads:
                for load in loads:
                    prove_disjoint(read, load, outer)
            for read in store_reads:
                for write in writes:
                    if type(readout_compute_admission) is DenseReadyReadoutAdmission:
                        prove_producer_before_readout(read, write, outer)
                    else:
                        prove_disjoint(read, write, outer)
        if pairs > admission.max_issued_compute_pairs or load_count > admission.max_issued_loads:
            raise ValueError("panel epoch exceeds original selected in-flight command bounds")
        for access in (*loads, *reads, *writes, *store_reads):
            prove_assigned(access)
        for i, a in enumerate(loads):
            for b in loads[i:]:
                prove_disjoint(a, b, outer, exclude_same=True)
        for a in reads:
            for b in loads:
                prove_disjoint(a, b, outer)
        if not ordered_accumulator_commits:
            for i, a in enumerate(writes):
                for b in writes[i:]:
                    prove_disjoint(a, b, outer, exclude_same=True)
        summaries.append(
            {
                "compute_pairs": pairs,
                "future_or_initial_loads": load_count,
                "readout": bool(stores),
                "readout_commands": stores,
                "disjoint_readout_compute": bool(
                    stores and writes and type(readout_compute_admission) is DenseReadoutComputeAdmission),
                "causal_ready_readout_compute": bool(
                    stores and writes and type(readout_compute_admission) is DenseReadyReadoutAdmission),
                "initialization": bool(initializations),
                "issue_order": tuple(type(site.operation).__name__ for site in selected),
                "completion": DRAIN,
                "awaits": "all issued movement, operand consumption, accumulator writes and readout "
                "before window reuse",
            }
        )
    # Every ACTUAL epoch and physical interval above is checked first. Only the
    # exact typed inverse of this one counted transformation may then restore
    # its original source representation. Arbitrary kernels/callbacks cannot
    # supply a checking projection.
    source_base, readout_projection = kernel, None
    if readout_source_projection is not None:
        from .dense_readout_overlap import DenseReadoutCheckingProjection

        if readout_compute_admission is None:
            raise ValueError("readout source projection needs exact typed transformation and selected conjunction")
        if type(readout_source_projection) is not DenseReadoutCheckingProjection:
            from .dense_ready_readout import DenseReadyReadoutCheckingProjection
            if (type(readout_source_projection) is not DenseReadyReadoutCheckingProjection
                    or type(readout_compute_admission) is not DenseReadyReadoutAdmission):
                raise ValueError("readout source projection needs exact typed transformation and selected conjunction")
        elif type(readout_compute_admission) is not DenseReadoutComputeAdmission:
            raise ValueError("paired disjoint readout projection cannot claim a different execution law")
        source_base, readout_projection = readout_source_projection.checking_kernel(kernel, adapter, storage)
    source_kernel, phase_changes = (
        _fission_noninterfering_epoch_loads(source_base, adapter) if source_phase_fission else (source_base, [])
    )
    phase_kernel_sha256 = source_kernel.digest()
    source_kernel, radix_changes = (
        _collapse_compute_radices(source_kernel, adapter) if source_phase_fission else (source_kernel, [])
    )
    remaining = int((deadline - time.monotonic()) * 1000)
    if remaining <= 0:
        raise ValueError("panel source checking budget exhausted")
    source = verify_compact_dense(
        _serial_projection(source_kernel),
        problem,
        resources,
        storage,
        adapter,
        static_budget=static_budget,
        timeout_ms=remaining,
        output_readout=output_readout,
        bias_source=bias_source,
    )
    return {
        "schema": "merlin.dense_panel_pipeline.v1",
        "kernel_sha256": kernel.digest(),
        "source": source,
        **({"readout_checking_projection": readout_projection} if readout_projection is not None else {}),
        **({"readout_compute_admission": asdict(readout_compute_admission)}
           if readout_compute_admission is not None else {}),
        **({"source_phase_fission": {
            "actual_kernel_sha256": kernel.digest(), "checking_kernel_sha256": source_kernel.digest(),
            "epochs": phase_changes,
            "phase_fission_kernel_sha256": phase_kernel_sha256,
            "compute_radix_collapses": radix_changes,
            "premise": "all actual per-epoch load writes pairwise disjoint and disjoint from every compute read",
            "preserves": "all actual commands, complete counted domains, per-phase order and full epoch completion",
            "emission": "original actual Kernel unchanged",
        }} if source_phase_fission else {}),
        "epochs": summaries,
        "admission": asdict(admission),
        "dynamic_epoch_drains": sum(prod(extent for _, _, extent in scopes) for scopes in epochs),
        "overlapped_compute_pairs": sum(
            prod(extent for _, _, extent in scopes) * summary["compute_pairs"]
            for scopes, summary in zip(epochs, summaries)
            if summary["future_or_initial_loads"]
        ),
        "qualification": "conditional original implementation premises; no independent physical qualification",
        "target_execution": "not_established",
    }


@dataclass(frozen=True)
class DensePanelPipelineResult:
    """Checked actual program; execution qualification stays with its owner."""

    kernel: Kernel
    realization: dict
    source_identity: str
    admission: DensePanelOverlapAdmission
    checked_kernel_sha256: str
    checked_admission_sha256: str
    checked_library_sources: tuple
    source_module: object = None
    selected_contracts: object = None
    output_readout: object = None
    bias_source: object = None
    readout_qualification_identity: str | None = None
    checked_boundary_sha256: str | None = None

    def revalidate(self):
        from merlin.transforms.certificates import ir_digest

        from .library_sources import shared_library_sources

        if (
            self.output_readout is not None
            and type(self.output_readout) not in (DenseIntegerReadout, DenseTiledIntegerReadout)
            or self.bias_source is not None
            and type(self.bias_source) is not DenseBiasSource
        ):
            raise ValueError("panel original readout or bias boundary changed")
        if (
            self.kernel.digest() != self.checked_kernel_sha256
            or _digest(asdict(self.admission)) != self.checked_admission_sha256
            or tuple(sorted(shared_library_sources(version=14).items())) != self.checked_library_sources
            or self.source_module is not None
            and ir_digest(self.source_module) != self.source_identity
            or _digest(
                (
                    None if self.output_readout is None else asdict(self.output_readout),
                    None if self.bias_source is None else asdict(self.bias_source),
                    self.readout_qualification_identity,
                )
            )
            != self.checked_boundary_sha256
        ):
            raise ValueError("panel source, checked emitted program, execution premises or library changed")
        if self.selected_contracts is not None:
            self.selected_contracts.revalidate()

    def emit_llvm(self, builder, lower_call):
        """Emit the checked actual commands, including every explicit drain.

        The callback must realize the selected ordered issue and full completion
        semantics. This does not turn a callback into physical qualification.
        """
        from .llvm_program import FnBuilderIndexAdapter, emit_index_program

        self.revalidate()
        receipt = emit_index_program(self.kernel, FnBuilderIndexAdapter(builder), lower_call)
        self.revalidate()
        return receipt

    def emit_index_program(self, control, lower_effect, *, complete_epoch):
        """Lower checked effects and explicit full-completion boundaries separately.

        ``lower_effect`` issues the actual instruction represented by each call;
        it must preserve the selected operand/configuration mapping. Only the
        schedule's explicit drains invoke ``complete_epoch``. That callback must
        emit a real full device completion, never a delay or an issue ticket.
        Both callbacks and target execution premises remain caller obligations.
        This entry neither chooses a schedule nor removes completion boundaries.
        """
        from .llvm_program import emit_index_effect_program

        self.revalidate()
        receipt = emit_index_effect_program(self.kernel, control, lower_effect,
            complete_epoch=complete_epoch, completion_instruction=DRAIN)
        self.revalidate()
        return receipt

    def materialize_scf(self):
        from .scf_commands import materialize_scf_commands

        self.revalidate()
        result = materialize_scf_commands(self.kernel)
        self.revalidate()
        return result


def schedule_dense_panel_pipeline(
    problem,
    resources,
    storage,
    adapter,
    admission,
    *,
    name,
    dtypes,
    schedule_plan,
    source_identity,
    external_buffers_disjoint,
    static_budget=128,
    timeout_ms=5000,
    output_readout=None,
    bias_source=None,
    readout_qualification_identity=None,
):
    """Consume typed block/layout/pipeline/loop decisions; check actual effects.

    Source identity/context binding confers no numerical or execution capability.
    The original source-aware caller and independently qualified admission retain
    those obligations. Every explicit decision executes or refuses; there is no
    alternate block/serialized fallback on verification failure.
    """
    from .library_sources import shared_library_sources
    from .loop_schedule import propose_loop_transforms
    from .schedule_ir import check_dense_schedule_commands, read_dense_schedule_plan

    if external_buffers_disjoint is not True:
        raise ValueError("panel overlap requires original runtime external-buffer disjointness")
    if output_readout is not None:
        if (
            type(output_readout) not in (DenseIntegerReadout, DenseTiledIntegerReadout)
            or type(readout_qualification_identity) is not str
            or not readout_qualification_identity.strip()
        ):
            raise ValueError("panel readout requires original independent execution qualification")
    elif readout_qualification_identity is not None:
        raise ValueError("readout qualification requires an original typed output readout")
    decision = read_dense_schedule_plan(
        schedule_plan,
        problem,
        resources,
        storage,
        adapter,
        source_identity=source_identity,
        dtypes=dtypes,
        static_budget=static_budget,
        output_readout=output_readout,
        bias_source=bias_source,
    )
    if decision.pipeline_prefetch is None:
        raise ValueError("panel pipeline service requires an explicit schedule.panel_pipeline decision")
    if decision.pipeline_kind == "retained_lhs_segmented_k":
        raise ValueError("segmented decisions require the separate combined-capability scheduling service")
    sources = tuple(sorted(shared_library_sources(version=14).items()))
    retained = decision.pipeline_kind == "retained_full_k"
    kernel = (
        emit_compact_dense(
            problem,
            decision.block,
            decision.storage,
            adapter,
            name=name,
            dtypes=dtypes,
            outer_order=decision.outer_order,
            output_readout=output_readout,
            bias_source=bias_source,
        )
        if retained
        else build_dense_panel_pipeline(
            problem,
            decision.block,
            resources,
            decision.storage,
            adapter,
            name=name,
            dtypes=dtypes,
            prefetch=decision.pipeline_prefetch,
            outer_order=decision.outer_order,
            issue_order=decision.pipeline_issue_order,
            window_offsets=decision.pipeline_window_offsets,
            output_readout=output_readout,
            bias_source=bias_source,
        )
    )
    if decision.loop_transforms:
        kernel = propose_loop_transforms(kernel, decision.loop_transforms, index_bits=adapter.index_bits)
    if retained:
        kernel = complete_dense_retained_reduction(
            kernel, problem, decision.block, resources, decision.storage, adapter
        )
    check_dense_schedule_commands(decision, sum(1 for _ in _calls(kernel.body)), static_budget)
    verifier = verify_dense_retained_reduction if retained else verify_dense_panel_pipeline
    report = verifier(
        kernel,
        problem,
        decision.block,
        resources,
        decision.storage,
        adapter,
        admission,
        static_budget=static_budget,
        timeout_ms=timeout_ms,
        output_readout=output_readout,
        bias_source=bias_source,
    )
    report["schedule_ir"] = {
        "sha256": decision.plan_sha256,
        "text": decision.plan_text,
        "source_identity": source_identity,
        "pipeline_issue_order": decision.pipeline_issue_order,
        "pipeline_kind": decision.pipeline_kind,
        "prefetch": decision.pipeline_prefetch,
        "pipeline_window_offsets": decision.pipeline_window_offsets,
        "loop_transforms": [asdict(request) for request in decision.loop_transforms],
        "decisions_consumed": ("block_tiles", "outer_order", "row_bases", "panel_pipeline")
        + (("pipeline_window_offsets",) if decision.pipeline_window_offsets is not None else ())
        + (("loop_transforms",) if decision.loop_transforms else ()),
    }
    result = DensePanelPipelineResult(
        kernel,
        report,
        source_identity,
        admission,
        kernel.digest(),
        _digest(asdict(admission)),
        sources,
        output_readout=output_readout,
        bias_source=bias_source,
        readout_qualification_identity=readout_qualification_identity,
        checked_boundary_sha256=_digest(
            (
                None if output_readout is None else asdict(output_readout),
                None if bias_source is None else asdict(bias_source),
                readout_qualification_identity,
            )
        ),
    )
    result.revalidate()
    return result


def schedule_source_dense_panel(
    module,
    operation,
    *,
    selected_contracts,
    storage,
    adapter,
    admission,
    name,
    schedule_plan,
    resources=None,
    reservations=(),
    external_buffers_disjoint,
    static_budget=128,
    timeout_ms=5000,
    output_operation=None,
    output_operations=(),
    readout_qualification_identity=None,
):
    """Original typed source entry; no model/capsule identity dispatch."""
    from xdsl.dialects.builtin import ModuleOp

    from merlin.transforms.certificates import ir_digest

    from .dense_source import read_source_dense, read_source_dense_readout, read_source_dense_readout_chain

    if type(module) is not ModuleOp or not any(op is operation for op in module.walk()):
        raise ValueError("panel source must be an original typed operation inside its current module")
    module.verify()
    selected_contracts.revalidate()
    identity = ir_digest(module)
    problem = read_source_dense(operation, selected_contracts=selected_contracts)
    if type(output_operations) is not tuple or output_operations and output_operation is not None:
        raise ValueError("select one original readout or an explicit ordered readout chain")
    outputs = output_operations or (() if output_operation is None else (output_operation,))
    if any(not any(op is member for op in module.walk()) for member in outputs):
        raise ValueError("panel readout must belong to the current original source module")
    readout = None
    if output_operations:
        readout = read_source_dense_readout_chain(
            operation,
            output_operations,
            selected_contracts=selected_contracts,
            qualification_identity=readout_qualification_identity,
        )
    elif output_operation is not None:
        readout = read_source_dense_readout(
            operation,
            output_operation,
            selected_contracts=selected_contracts,
            qualification_identity=readout_qualification_identity,
        )
    elif readout_qualification_identity is not None:
        raise ValueError("panel readout qualification requires an original output operation")
    resources, bound = selected_contracts.dense.bind(
        problem,
        storage,
        adapter,
        dtypes=selected_contracts.dense.dtypes,
        hardware=selected_contracts.hardware_identity,
        policy=selected_contracts.numerical_policy,
        resources=resources,
        reservations=reservations,
        output_readout=readout,
    )
    result = schedule_dense_panel_pipeline(
        problem,
        resources,
        storage,
        bound,
        admission,
        name=name,
        dtypes=selected_contracts.dense.dtypes,
        schedule_plan=schedule_plan,
        source_identity=identity,
        external_buffers_disjoint=external_buffers_disjoint,
        static_budget=static_budget,
        timeout_ms=timeout_ms,
        output_readout=readout,
        readout_qualification_identity=readout_qualification_identity,
    )
    result = replace(result, source_module=module, selected_contracts=selected_contracts)
    result.revalidate()
    return result


@dataclass(frozen=True)
class DenseRetainedChoice:
    """Unverified canonical residency alternative, never an execution grant.

    Counts include every full/tail family in the counted emitter after its
    invariant hoists. They exclude unchanged bias/readout transfers and do not
    predict physical DMA aggregation, target stalls, configuration or cycles.
    The consumer must execute the selected role/order/block and independently
    check final source products, addresses, effects and completion.
    """

    block: object
    role: str
    outer_order: tuple[str, str]
    lhs_loads: int
    rhs_loads: int
    requested_operand_bytes: int


def dense_retained_choices(problem, resources, storage, *, roles=("lhs", "weight")):
    """Enumerate both reusable residency families within original assignments.

    No memory is reassigned between roles. The existing bounded geometric
    search and all original partial-tile/block restrictions are retained. For
    LHS retention, the complete N groups share one invariant LHS stage; a
    separate partial N family still reloads it. RHS retention is symmetric in
    M. Thus even the traffic ranking accounts for remainder families without
    expanding the dynamic domain or assuming reuse across unrelated scopes.
    """
    from merlin.transforms.dense_planning import dense_blocks

    if (
        type(roles) is not tuple
        or not roles
        or len(set(roles)) != len(roles)
        or any(role not in ("lhs", "weight") for role in roles)
    ):
        raise ValueError("retained selection requires distinct explicit LHS/weight residency families")
    mt, kt, nt = problem.tiles
    blocks = dense_blocks(problem, resources, k_limit=lambda m, n: storage.k_limit(problem, resources, m, n))
    choices = []
    for block in blocks:
        if block.k != kt or not storage.admits(problem, block, resources):
            continue
        for role in roles:
            lhs_repeats = (
                int(bool(nt // block.n)) + int(bool(nt % block.n)) if role == "lhs" else (nt + block.n - 1) // block.n
            )
            rhs_repeats = (
                int(bool(mt // block.m)) + int(bool(mt % block.m))
                if role == "weight"
                else (mt + block.m - 1) // block.m
            )
            choices.append(
                DenseRetainedChoice(
                    block,
                    role,
                    ("m", "n") if role == "lhs" else ("n", "m"),
                    mt * kt * lhs_repeats,
                    kt * nt * rhs_repeats,
                    problem.m * problem.k * problem.lhs_bytes * lhs_repeats
                    + problem.k * problem.n * problem.rhs_bytes * rhs_repeats,
                )
            )
    return tuple(
        sorted(
            choices,
            key=lambda choice: (
                choice.requested_operand_bytes,
                choice.lhs_loads + choice.rhs_loads,
                -choice.block.m * choice.block.n,
                roles.index(choice.role),
                choice.block.m,
                choice.block.n,
            ),
        )
    )


@dataclass(frozen=True)
class DenseRetainedReductionAdmission:
    """Independent selected premises for read-only operands and ordered sums.

    Ordered command acceptance alone does not prove accumulator commit order.
    The selected implementation must preserve each overlapping accumulator
    update in issue order, including initialization and continuation. No operand
    overwrite, load/compute overlap, or generic RAW/WAR scoreboard is admitted.
    Initializers retain individual drains unless an explicit ordered load
    configuration snapshot and pairwise-disjoint ACC write admission is selected.
    Issued counts are compiler cost bounds, not invented physical capacities.
    """

    hardware_identity: str
    numerical_policy_identity: str
    qualification_identity: str
    max_issued_compute_pairs: int
    max_issued_loads: int
    pending_compute_pairs: int
    pending_loads: int
    execution: str = "ordered_accumulator_commits_read_only_operands_full_drain"
    load_configuration: str = "ordered_snapshot_before_data_issue"
    descriptor: str = "preload_immediately_paired_with_compute"
    max_issued_readouts: int = 1
    pending_readouts: int = 1
    readout_configuration: str = "separate_drained_readout"
    max_issued_initializations: int = 1
    initialization_configuration: str = "separate_drained_initialization"

    def __post_init__(self):
        if (
            any(
                type(v) is not str or not v.strip()
                for v in (self.hardware_identity, self.numerical_policy_identity, self.qualification_identity)
            )
            or any(
                type(v) is not int or v <= 0
                for v in (
                    self.max_issued_compute_pairs,
                    self.max_issued_loads,
                    self.pending_compute_pairs,
                    self.pending_loads,
                    self.max_issued_readouts,
                    self.pending_readouts,
                    self.max_issued_initializations,
                )
            )
            or self.execution != "ordered_accumulator_commits_read_only_operands_full_drain"
            or self.load_configuration != "ordered_snapshot_before_data_issue"
            or self.descriptor != "preload_immediately_paired_with_compute"
            or self.initialization_configuration not in ("separate_drained_initialization",
                "ordered_snapshot_before_disjoint_accumulator_issue")
            or self.max_issued_initializations > 1
            and self.initialization_configuration != "ordered_snapshot_before_disjoint_accumulator_issue"
            or self.readout_configuration not in ("separate_drained_readout", "ordered_snapshot_before_store_issue")
            or self.max_issued_readouts > 1
            and self.readout_configuration != "ordered_snapshot_before_store_issue"
        ):
            raise ValueError("retained reduction needs independent ordered commit, snapshot and completion premises")


def dense_retained_operand_hoists(kernel, adapter, *, role):
    """Propose actual invariant operand motion across canonical output groups.

    The input is the actual compact counted program, not a requested access
    list. Only full-K singleton panels are extracted. Every request executes
    against the preceding result; final source/lifetime verification is still
    required. Tails outside the reused output loop keep their original loads.
    """
    from .loop_schedule import LoopTransform, propose_loop_transforms

    if role not in ("lhs", "weight"):
        raise ValueError("retained hoist requires an explicit original operand role")
    crossed = "outer_n" if role == "lhs" else "outer_m"
    requests = []
    actual = kernel

    def candidate(body, path=()):
        for statement in body:
            if type(statement) is not Loop:
                continue
            current = (*path, statement.var)
            if statement.var == crossed:
                for index, panel in enumerate(statement.body):
                    if type(panel) is not Loop or panel.extent != 1:
                        continue
                    types = {type(adapter.project(c)) for c in _calls(panel.body)}
                    if types != {Load, Preload, Compute}:
                        continue
                    for load_index, child in enumerate(panel.body):
                        ops = tuple(adapter.project(c) for c in _calls((child,)))
                        if ops and all(type(op) is Load and op.role == role for op in ops):
                            return (
                                LoopTransform("hoist", (*current, panel.var), index=load_index),
                                LoopTransform("hoist", current, index=index),
                            )
            found = candidate(statement.body, current)
            if found is not None:
                return found
        return None

    while (selected := candidate(actual.body)) is not None:
        if len(requests) + len(selected) > 32:
            raise ValueError("retained operand hoists exceed the original finite transformation budget")
        actual = propose_loop_transforms(actual, selected, index_bits=adapter.index_bits)
        requests.extend(selected)
    return tuple(requests)


def _dense_retained_compute_nests(kernel, adapter):
    """Read independent M/K/N axes from original projected command operands."""
    from merlin.sched.ir.expr import free_vars

    streams = []

    def discover(body, path=()):
        for statement in body:
            if type(statement) is not Loop:
                continue
            kinds = {type(adapter.project(call)) for call in _calls((statement,))}
            if kinds != {Preload, Compute}:
                discover(statement.body, (*path, statement.var))
                continue
            chain, tail = [statement], statement.body
            while len(tail) == 1 and type(tail[0]) is Loop:
                chain.append(tail[0])
                tail = tail[0].body
            if (
                len(chain) < 3
                or any(loop.extent != 1 for loop in chain[:-3])
                or len(tail) != 2
                or any(type(call) is not Call for call in tail)
            ):
                raise ValueError("resident compute interchange requires a perfect three-axis paired nest")
            preload, compute = (adapter.project(call) for call in tail)
            if type(preload) is not Preload or type(compute) is not Compute:
                raise ValueError("resident compute interchange cannot split a physical descriptor/compute pair")
            loops = chain[-3:]
            variables = {loop.var for loop in loops}
            coordinates = (compute.input_block[0], compute.input_block[1], preload.weight_block[1])
            names = {}
            for axis, coordinate in zip(("m", "k", "n"), coordinates, strict=True):
                selected = variables & free_vars(coordinate)
                if len(selected) != 1:
                    raise ValueError("resident compute axes are not independent original source coordinates")
                names[axis] = selected.pop()
            if set(names.values()) != variables or variables & free_vars(preload.weight_block[0]) != {names["k"]}:
                raise ValueError("resident compute input/weight reduction axes disagree")
            current = [next(axis for axis, name in names.items() if name == loop.var) for loop in loops]
            streams.append(
                ((*path, *(loop.var for loop in chain[:-3])), names, current, {loop.var: loop.extent for loop in loops})
            )

    discover(kernel.body)
    if not streams:
        raise ValueError("resident compute order has no complete original compute nest to transform")
    return streams


def dense_retained_compute_nests(kernel, adapter):
    """Inspect actual source M/K/N axes of perfect paired compute nests.

    Returns tuples of (enclosing path, axis-to-variable map, current axis
    order, variable-to-extent map). These are proposal coordinates, not a
    source, storage, dependency or target execution proof. Unknown effects
    and noncanonical descriptor/compute nests refuse through the same reader
    used by executable interchange and array-weight retention.
    """
    return tuple(_dense_retained_compute_nests(kernel, adapter))


def dense_retained_compute_order(kernel, adapter, *, order):
    """Propose actual adjacent interchanges of complete resident compute nests.

    Axes come from projected input/weight block coordinates, not loop names or
    descriptive attributes. Only perfect three-axis nests of adjacent physical
    PRELOAD/COMPUTE pairs are handled; singleton surrounding scopes stay put.
    Each chosen axis occurs once, so each output still visits K in increasing
    order. This is a proposal for the unchanged source and completion verifiers,
    never permission to reorder an operand staging or external observer.
    """
    from .loop_schedule import LoopTransform, propose_loop_transforms

    if (
        type(order) is not tuple
        or len(order) != 3
        or any(type(axis) is not str for axis in order)
        or set(order) != {"m", "k", "n"}
    ):
        raise ValueError("resident compute order requires an explicit permutation of M/K/N")
    streams = _dense_retained_compute_nests(kernel, adapter)
    actual, requests = kernel, []
    for prefix, names, current, _extents in streams:
        for position, axis in enumerate(order):
            index = current.index(axis)
            while index > position:
                cursor = (*prefix, *(names[axis] for axis in current[:index]))
                request = LoopTransform("reorder", cursor)
                if len(requests) == 32:
                    raise ValueError("resident compute order exceeds original finite transformation budget")
                actual = propose_loop_transforms(actual, (request,), index_bits=adapter.index_bits)
                requests.append(request)
                current[index - 1], current[index] = current[index], current[index - 1]
                index -= 1
    return tuple(requests)


def dense_retained_compute_unroll(kernel, adapter, *, axis, factor, max_static_operations=4096):
    """Propose bounded unrolling of actual resident compute coordinates.

    Each request targets a semantic source axis, preserving the complete dynamic
    command sequence, descriptor adjacency and increasing reduction order. The
    ordinary typed schedule pipeline still proves source products, storage and
    completion afterward. This does not unroll loads, remove a drain, expand the
    full dynamic program, or infer a performance benefit from code size.
    """
    from .loop_schedule import LoopTransform, propose_loop_transforms

    if axis not in ("m", "k", "n") or type(factor) is not int or not 2 <= factor <= 64:
        raise ValueError("resident compute unroll requires a source axis and explicit factor 2..64")
    streams = _dense_retained_compute_nests(kernel, adapter)
    actual, requests = kernel, []
    for prefix, names, current, extents in streams:
        if factor > extents[names[axis]]:
            raise ValueError("resident compute unroll factor exceeds an original source-axis extent")
        cursor = (*prefix, *(names[value] for value in current[: current.index(axis) + 1]))
        request = LoopTransform("unroll", cursor, factor=factor)
        if len(requests) == 32:
            raise ValueError("resident compute unroll exceeds original finite transformation budget")
        actual = propose_loop_transforms(
            actual, (request,), index_bits=adapter.index_bits, max_static_operations=max_static_operations
        )
        requests.append(request)
    return tuple(requests)


def complete_dense_retained_reduction(kernel, problem, block, resources, storage, adapter):
    """Form pure staging/reduction epochs from actual already transformed IR.

    All K tiles must fit the original assigned windows. Consecutive pure loads
    drain once before any reader; a complete pure compute region drains once
    before reuse/readout. Initializers and stores retain individual completion
    boundaries. This function proposes completion; it grants no execution law.
    """
    if block.k != problem.tiles[1] or not storage.admits(problem, block, resources):
        raise ValueError("retained reduction requires full K inside the original assigned capacity")
    used = {statement.var for statement in _loops(kernel.body)}
    serial = 0

    def epoch(body):
        nonlocal serial
        serial += 1
        name = "dense_retained_epoch_" + str(serial)
        if name in used:
            raise ValueError("retained completion scope collides with original loop")
        used.add(name)
        return Loop(name, 1, (*body, Call(DRAIN, ())))

    def form(body):
        result, pending = [], []
        previous = None

        def flush():
            if pending:
                result.append(epoch(tuple(pending)))
                pending.clear()

        for statement in body:
            kinds = {type(adapter.project(c)) for c in _calls((statement,))}
            kind = "loads" if kinds == {Load} else "compute" if kinds == {Preload, Compute} else None
            if kind is not None:
                if previous != kind:
                    flush()
                pending.append(statement)
            else:
                flush()
                if type(statement) is Loop:
                    result.append(replace(statement, body=form(statement.body)))
                elif kinds in ({Store}, {TiledIntegerStore}, {InitializeAccumulator}):
                    result.append(epoch((statement,)))
                else:
                    raise ValueError("retained completion cannot classify an unknown or unpaired effect")
            previous = kind
        flush()
        return tuple(result)

    result = replace(kernel, body=form(kernel.body))
    if check_structure(result):
        raise ValueError("retained completion did not preserve closed structured IR")
    check_integer_domain(result, adapter.index_bits)
    return result


def verify_dense_retained_reduction(
    kernel,
    problem,
    block,
    resources,
    storage,
    adapter,
    admission,
    *,
    static_budget,
    timeout_ms=5000,
    output_readout=None,
    bias_source=None,
):
    """Prove source cover and pure-epoch hazards, with ordered sum commitment.

    Repeated accumulator writes are permitted only in compute-only regions
    under the separate selected ordered-commit premise. The unchanged complete
    source checker proves exact output identity, increasing K, initialization,
    continuations, source residency, aliases, tails and final observers.
    """
    import time

    import z3

    from .compact_verification import _project, verify_compact_dense

    if (
        type(admission) is not DenseRetainedReductionAdmission
        or admission.hardware_identity != adapter.configuration_identity
        or admission.numerical_policy_identity != adapter.numerical_policy_identity
    ):
        raise ValueError("retained reduction differs from selected ordered-commit capability")
    admission.__post_init__()
    if (
        block.k != problem.tiles[1]
        or not storage.admits(problem, block, resources)
        or check_structure(kernel)
        or type(static_budget) is not int
        or static_budget <= 0
        or sum(1 for _ in _calls(kernel.body)) > static_budget
        or type(timeout_ms) is not int
        or timeout_ms <= 0
    ):
        raise ValueError("retained reduction exceeds original full-K capacity, structure or checking bounds")
    check_integer_domain(kernel, adapter.index_bits)
    deadline = time.monotonic() + timeout_ms / 1000
    epochs, serial = [], 0

    def structure(body, ancestry=(), inside=False):
        nonlocal serial
        for statement in body:
            if type(statement) is Loop:
                serial += 1
                scope = (serial, statement.var, statement.extent)
                drains = [s for s in statement.body if type(s) is Call and s.instr == DRAIN]
                if drains:
                    if inside or statement.extent != 1 or len(drains) != 1 or statement.body[-1] != Call(DRAIN, ()):
                        raise ValueError("retained epoch requires exactly one final full drain")
                    epochs.append((*ancestry, scope))
                    structure(statement.body[:-1], (*ancestry, scope), True)
                else:
                    structure(statement.body, (*ancestry, scope), inside)
            elif not inside:
                raise ValueError("every retained effect must belong to a completed epoch")

    structure(kernel.body)
    if not epochs:
        raise ValueError("retained reduction has no completion epochs")
    sites = _project(_serial_projection(kernel), adapter, fold_singleton_loops=False)

    def expression(value, env):
        if type(value) is Const:
            return z3.IntVal(value.value)
        if type(value) is int:
            return z3.IntVal(value)
        if type(value) is Var:
            return env[value.name]
        if type(value) is Sum:
            return z3.Sum(*(expression(v, env) for v in value.terms))
        if type(value) is Scaled:
            return expression(value.expr, env) * value.factor
        if type(value) is Select:
            return z3.If(env[value.var] == value.equals, expression(value.then, env), expression(value.other, env))
        raise ValueError("unknown retained address expression")

    def disjoint(a, b, outer):
        def view(site):
            if type(site.operation) is InitializeAccumulator:
                return storage.accumulator
            return storage.lhs if site.operation.role == "lhs" else storage.rhs
        va, vb = view(a), view(b)
        if va.backing != vb.backing:
            return
        ae = {name: z3.Int(f"retained-a:{scope}:{name}") for scope, name, _ in a.scopes}
        be = {name: z3.Int(f"retained-b:{scope}:{name}") for scope, name, _ in b.scopes}
        constraints = [
            z3.And(env[name] >= 0, env[name] < extent)
            for site, env in ((a, ae), (b, be))
            for _, name, extent in site.scopes
        ]
        constraints.extend(ae[name] == be[name] for _, name, _ in outer)
        if a.ordinal == b.ordinal:
            local = [ae[name] != be[name] for scope, name, _ in a.scopes if scope not in {s[0] for s in outer}]
            if not local:
                return
            constraints.append(z3.Or(*local))
        al = expression(a.operation.row, ae) * va.unit_bytes
        bl = expression(b.operation.row, be) * vb.unit_bytes
        ah = al + expression(a.operation.rows, ae) * va.unit_bytes
        bh = bl + expression(b.operation.rows, be) * vb.unit_bytes
        remaining = int((deadline - time.monotonic()) * 1000)
        if remaining <= 0:
            raise ValueError("retained staging proof budget exhausted")
        solver = z3.Solver()
        solver.set(timeout=remaining)
        solver.add(*constraints, al < bh, bl < ah)
        if solver.check() != z3.unsat:
            raise ValueError("retained staging or initialization has overlapping in-flight destinations")

    summaries = []
    for outer in epochs:
        selected = [s for s in sites if outer[-1][0] in {scope for scope, _, _ in s.scopes}]
        kinds = {type(s.operation) for s in selected}
        if not selected or kinds not in (
            {Load},
            {Preload, Compute},
            {InitializeAccumulator},
            {Store},
            {TiledIntegerStore},
        ):
            raise ValueError("retained epoch must contain only staging, paired computation, initialization or readout")

        def count(site):
            return prod(extent for scope, _, extent in site.scopes if scope not in {s[0] for s in outer})

        loads = sum(count(s) for s in selected if type(s.operation) is Load)
        pairs = sum(count(s) for s in selected if type(s.operation) is Compute)
        if loads > admission.max_issued_loads or pairs > admission.max_issued_compute_pairs:
            raise ValueError("retained epoch exceeds original selected issued-command budget")
        if kinds == {Load}:
            if any(s.operation.role not in ("lhs", "weight") for s in selected):
                raise ValueError("retained staging has an unknown source role")
            for index, a in enumerate(selected):
                for b in selected[index:]:
                    disjoint(a, b, outer)
        elif kinds == {Preload, Compute}:
            if len(selected) % 2 or any(
                type(a.operation) is not Preload or type(b.operation) is not Compute or a.scopes != b.scopes
                for a, b in zip(selected[::2], selected[1::2], strict=True)
            ):
                raise ValueError("retained reduction must not split or overwrite a physical preload/compute pair")
        elif kinds == {InitializeAccumulator}:
            if sum(count(s) for s in selected) > admission.max_issued_initializations:
                raise ValueError("retained initializer exceeds its selected completion budget")
            for index, a in enumerate(selected):
                for b in selected[index:]:
                    disjoint(a, b, outer)
        elif kinds & {Store, TiledIntegerStore}:
            readouts = sum(count(s) for s in selected)
            if readouts > admission.max_issued_readouts:
                raise ValueError("retained readout exceeds selected store completion budget")
            if (readouts > 1 and type(output_readout) is DenseTiledIntegerReadout
                    and len(set(output_readout.column_scale_bits)) != 1):
                raise ValueError("varying per-tile readout needs an independent configuration snapshot mapping")
        summaries.append(
            dict(
                compute_pairs=pairs,
                future_or_initial_loads=loads,
                initialization=kinds == {InitializeAccumulator},
                readout=bool(kinds & {Store, TiledIntegerStore}),
                readout_commands=sum(count(s) for s in selected) if kinds & {Store, TiledIntegerStore} else 0,
                issue_order=tuple(type(s.operation).__name__ for s in selected),
                completion=DRAIN,
            )
        )
    remaining = int((deadline - time.monotonic()) * 1000)
    if remaining <= 0:
        raise ValueError("retained source proof budget exhausted")
    source = verify_compact_dense(
        _serial_projection(kernel),
        problem,
        resources,
        storage,
        adapter,
        static_budget=static_budget,
        timeout_ms=remaining,
        output_readout=output_readout,
        bias_source=bias_source,
    )
    return dict(
        schema="merlin.dense_retained_reduction.v1",
        kernel_sha256=kernel.digest(),
        source=source,
        epochs=summaries,
        admission=asdict(admission),
        dynamic_epoch_drains=sum(prod(extent for _, _, extent in scopes) for scopes in epochs),
        overlapped_compute_pairs=0,
        qualification="conditional independent ordered accumulator commits; physical execution not established",
        target_execution="not_established",
    )
