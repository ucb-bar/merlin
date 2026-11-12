"""Construct dense panel windows, counted issue and RHS stripes."""

from __future__ import annotations

from dataclasses import replace
from itertools import product

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
    mul,
    select,
)
from merlin.compiler.schedule.blocks import Compute, InitializeAccumulator, Load, Preload, Store
from merlin.compiler.schedule.compact_dense import DenseLoopStorage, emit_compact_dense
from merlin.compiler.schedule.dense_contracts import TiledIntegerStore
from merlin.compiler.schedule.dense_panel_ir import DRAIN as DRAIN
from merlin.compiler.schedule.dense_panel_ir import _calls as _calls
from merlin.compiler.schedule.dense_panel_ir import _loops as _loops

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
    from merlin.compiler.transforms.dense_planning import dense_blocks

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

    from merlin.compiler.semantic.allocation_types import StorageBank

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
    from merlin.compiler.schedule.loop_schedule import striped_fission_transforms

    if (
        type(kernel) is not Kernel
        or check_structure(kernel)
        or type(factor) is not int
        or not 2 <= factor <= 64
        or type(retain_short_tails) is not bool
    ):
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


# Preserve the original public lookup and pickle identities.
for _definition in (
    _shift,
    _remap,
    _windows,
    select_dense_panel_window_offsets,
    build_dense_panel_pipeline,
    dense_panel_rhs_stripes,
):
    _definition.__module__ = "merlin.compiler.schedule.dense_panel_pipeline"
del _definition
