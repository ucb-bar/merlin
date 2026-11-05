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
from .library_sources import admission_point

from .dense_panel_ir import DRAIN as DRAIN
from .dense_panel_contracts import (
    DensePanelOverlapAdmission as DensePanelOverlapAdmission,
    DenseReadoutConfigurationSnapshot as DenseReadoutConfigurationSnapshot,
    DenseReadoutComputeAdmission as DenseReadoutComputeAdmission,
    DenseReadyReadoutAdmission as DenseReadyReadoutAdmission,
)
from .dense_panel_ir import (
    _calls as _calls,
    _loops as _loops,
    _serial_projection as _serial_projection,
)
from .dense_retained import (
    DenseRetainedChoice as DenseRetainedChoice,
    dense_m_panel_capacity as dense_m_panel_capacity,
    dense_retained_choices as dense_retained_choices,
    DenseRetainedReductionAdmission as DenseRetainedReductionAdmission,
    dense_retained_operand_hoists as dense_retained_operand_hoists,
    _dense_retained_compute_nests as _dense_retained_compute_nests,
    dense_retained_compute_nests as dense_retained_compute_nests,
    dense_retained_compute_order as dense_retained_compute_order,
    dense_retained_compute_unroll as dense_retained_compute_unroll,
    complete_dense_retained_reduction as complete_dense_retained_reduction,
    verify_dense_retained_reduction as verify_dense_retained_reduction,
)

ISSUE_ORDERS = ("compute_then_prefetch", "prefetch_rhs_between_columns")


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


@admission_point(
    "merlin.compile.scheduling.dense_panel_pipeline.schedule_dense_panel_pipeline",
    fallback_route="caller-selected original route; this consumer emits nothing on refusal",
)
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
class DenseMPanelAttempt:
    """A checked current schedule or an explicit construction refusal per shape."""

    result: DensePanelPipelineResult | None
    feedback: dict


@admission_point(
    "merlin.compile.scheduling.dense_panel_pipeline.schedule_dense_m_panels",
    fallback_route="caller-selected original route; this consumer emits nothing on refusal",
)
def schedule_dense_m_panels(problem, resources, storage, adapter, admission, *, n_tiles,
                           name, dtypes, source_identity, external_buffers_disjoint,
                           m_tiles=None, static_budget=128, timeout_ms=5000,
                           output_readout=None, bias_source=None, readout_qualification_identity=None):
    """Construct one counted invocation with RHS staging outside the M panels.

    An omitted M choice proposes the largest legal panel for the selected N
    width. It is a capacity choice, not a measured winner. The real scheduling
    dialect carries blocking, order and invariant hoists into the existing
    complete source/storage/epoch checker. It never concatenates thin kernels.
    A declined choice returns feedback before any alternative executes; errors
    in a concrete source or target verification propagate normally.
    """
    from .compact_dense import emit_compact_dense
    from .schedule_ir import make_dense_schedule_plan

    capacity = dense_m_panel_capacity(problem, resources, storage, n_tiles=n_tiles)
    feedback = dict(shape=dict(m=problem.m, k=problem.k, n=problem.n), capacity=capacity,
                    mechanism=capacity["mechanism"], status="not_applied", emitted="not_observed",
                    numerical_status="not_run", transfer_status="not_established")
    if m_tiles is not None and (type(m_tiles) is not int or m_tiles < 1):
        raise ValueError("explicit M panel must be a positive original tile extent")
    blockers = list(capacity["limiting_conditions"]) if not capacity["maximum_m_tiles"] else []
    if admission is None:
        blockers.append("selected_read_only_operand_and_ordered_reduction_law")
    elif type(admission) is not DenseRetainedReductionAdmission:
        raise ValueError("M panels require the original typed retained-reduction admission")
    if external_buffers_disjoint is not True:
        blockers.append("original_external_buffer_disjointness")
    if blockers:
        return DenseMPanelAttempt(None, dict(feedback, extension_blockers=blockers))
    issue_limit = min(admission.max_issued_compute_pairs // (problem.tiles[1] * n_tiles),
                      admission.max_issued_loads // problem.tiles[1] - n_tiles)
    feedback["selected_issue_m_tile_limit"] = max(0, issue_limit)
    if issue_limit < 1:
        return DenseMPanelAttempt(None, dict(feedback, extension_blockers=["selected_issue_limit"]))
    bounded = replace(resources, max_m_blocks=min(capacity["maximum_m_tiles"], issue_limit))
    choices = [choice for choice in dense_retained_choices(problem, bounded, storage, roles=("weight",),
                                                          n_choices=(n_tiles,))
               if choice.block.n == n_tiles and choice.block.m <= issue_limit
               and (m_tiles is None or choice.block.m == m_tiles)]
    if not choices:
        return DenseMPanelAttempt(None, dict(feedback, extension_blockers=[
            "requested_m_panel_exceeds_capacity_original_tail_domain_or_selected_issue_limit"]))
    choice = max(choices, key=lambda item: item.block.m)
    block = choice.block
    raw = emit_compact_dense(problem, block, storage, adapter, name=name, dtypes=dtypes,
                             outer_order=choice.outer_order, output_readout=output_readout, bias_source=bias_source)
    hoists = dense_retained_operand_hoists(raw, adapter, role="weight")
    plan = make_dense_schedule_plan(problem, resources, storage, adapter, source_identity=source_identity,
        dtypes=dtypes, static_budget=static_budget, block_tiles=(block.m, block.k, block.n),
        outer_order=choice.outer_order, loop_transforms=hoists, pipeline_kind="retained_full_k",
        pipeline_prefetch=False, output_readout=output_readout, bias_source=bias_source)
    result = schedule_dense_panel_pipeline(problem, resources, storage, adapter, admission, name=name,
        dtypes=dtypes, schedule_plan=plan, source_identity=source_identity,
        external_buffers_disjoint=True, static_budget=static_budget, timeout_ms=timeout_ms,
        output_readout=output_readout, bias_source=bias_source,
        readout_qualification_identity=readout_qualification_identity)
    actual = result.realization["source"]
    if actual["rhs_loads"] != choice.rhs_loads or actual["lhs_loads"] != choice.lhs_loads:
        raise ValueError("M-panel RHS reuse did not survive actual source and operand checking")
    feedback.update(status="checked_schedule_constructed", selected_m_tiles=block.m,
        selected_m_rows=block.m * problem.tile_m, counted_m_panels=(problem.tiles[0] + block.m - 1) // block.m,
        rhs_loads=actual["rhs_loads"], lhs_loads=actual["lhs_loads"],
        schedule_sha256=result.realization["schedule_ir"]["sha256"],
        checked_kernel_sha256=result.checked_kernel_sha256,
        extension_blockers=(capacity["limiting_conditions"] if block.m == capacity["maximum_m_tiles"] else
                            ["selected_issue_limit"] if block.m == issue_limit else ["original_tail_emitter_domain"]),
        tail_scope="full groups share RHS; separate partial M families retain their checked reloads")
    return DenseMPanelAttempt(result, feedback)
