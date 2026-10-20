"""Retain one full-reduction input stripe across independent sibling outputs.

The source supplies immutable tensor value identity, exact products and zeros;
the caller selects primitive dimensions and already assigned private storage.
No target instruction, completion rule or physical source/output alias proof
is supplied. Actual target projection must pass the independent checker.
"""

from __future__ import annotations

from dataclasses import dataclass

from .block_schedule import Compute, Load, Preload, Store
from .compact_dense import DenseLoopStorage
from .shared_input_groups import SharedInputGroup


@dataclass(frozen=True)
class SharedInputEffect:
    """Actual logical effect, with distinct RHS/output ownership per sibling.

    ``member=None`` is reserved for reads of the common immutable input.
    A member index selects the source contraction ordinal in the checked group.
    """

    member: int | None
    operation: Load | Preload | Compute | Store


@dataclass(frozen=True)
class SharedInputBlock:
    m: int
    k: int
    n: int
    operand_bytes: int
    accumulator_bytes: int


@dataclass(frozen=True)
class SharedInputSchedule:
    group: SharedInputGroup
    primitive_tile: tuple[int, int, int]
    block: SharedInputBlock
    operations: tuple[SharedInputEffect, ...]
    candidates_considered: int
    realization: dict


def _ceil(value, unit):
    return (value + unit - 1) // unit


def _storage(problem, tile, block, storage, resources):
    """Check all assigned intervals, including padding and shared backings."""
    if type(storage) is not DenseLoopStorage or storage.accumulator is None:
        return False
    tm, tk, tn = tile
    kt = _ceil(problem.k, tk)
    views = storage.lhs, storage.rhs, storage.accumulator
    rows = block.m * kt * tm, block.k * block.n * tk, block.m * block.n * tm
    widths = tk, tn, tn * 4  # Derived from the checked source's i8/i8/i32 formats.
    if any(
        count > view.capacity or width > view.unit_bytes or stride % view.alignment
        for count, width, stride, view in zip(rows, widths, (tm, tk, tm), views)
    ):
        return False
    used = tuple(view.unit_bytes * count for view, count in zip(views, rows))
    if used[0] + used[1] > resources.operand_bytes or used[2] > resources.accumulator_bytes:
        return False
    intervals = [(v.backing, v.base * v.unit_bytes, (v.base + v.capacity) * v.unit_bytes) for v in views]
    return not any(
        a[0] == b[0] and a[1] < b[2] and b[1] < a[2]
        for index, a in enumerate(intervals)
        for b in intervals[index + 1 :]
    )


def schedule_shared_input(
    module,
    group,
    *,
    primitive_tile,
    resources,
    storage,
    selected_block=None,
    max_commands=1_000_000,
    max_rows_visited=16 << 20,
    max_source_operations=10_000,
    max_source_elements=64 << 20,
):
    """Construct and check a bounded raw-core sibling schedule, or decline.

    All K tiles of an M panel stay in LHS storage while distinct RHS panels
    stream through a bounded output wave. Each output retains increasing K,
    its own exact zero seed and its original result identity. Later epilogues
    remain separate. No weight/output concatenation or host packing is needed.

    ``selected_block=(M,K,N)`` exposes a legal choice to an external tuner.
    Without it, requested logical bytes, command count and storage break ties.
    Counts are not physical traffic/cycles. This finite family declines when
    a full-K LHS stripe does not fit; it emits no compact machine loop or fence.
    """
    from merlin.transforms.dense_planning import DenseResources

    from .shared_input_verification import verify_shared_input

    if type(group) is not SharedInputGroup or type(resources) is not DenseResources:
        raise ValueError("shared-input scheduling requires a current source group and selected resources")
    group.verify_current(module, max_source_operations=max_source_operations, max_source_elements=max_source_elements)
    if (
        type(primitive_tile) is not tuple
        or len(primitive_tile) != 3
        or any(type(v) is not int or v <= 0 for v in primitive_tile)
        or any(type(v) is not int or v <= 0 for v in (max_commands, max_rows_visited))
    ):
        raise ValueError("shared-input primitive and verification budgets must be positive and explicit")
    if selected_block is not None and (
        type(selected_block) is not tuple
        or len(selected_block) != 3
        or any(type(v) is not int or v <= 0 for v in selected_block)
    ):
        raise ValueError("selected sibling block must contain three positive tile counts")
    tm, tk, tn = primitive_tile
    mt, kt = _ceil(group.m, tm), _ceil(group.k, tk)
    nt = sum(_ceil(width, tn) for width in group.widths)
    if not resources.partial_output_tiles and (group.m % tm or any(n % tn for n in group.widths)):
        return None
    # Every member has its own output tile boundaries; padded virtual lanes
    # never become a source read, arithmetic product or physical output cell.
    bm_max, bn_max, bk_max = (
        min(mt, resources.max_m_blocks),
        min(nt, resources.max_n_blocks),
        min(kt, resources.max_k_blocks),
    )
    if bm_max * bn_max * bk_max > resources.candidate_budget:
        raise ValueError("shared-input blocking search exceeds its candidate budget")
    commands = mt * kt + _ceil(mt, bm_max) * nt * kt + 2 * mt * nt * kt + mt * nt
    if commands > max_commands:
        raise ValueError("shared-input schedule exceeds its command budget before materialization")
    candidates, considered = [], 0
    for bm in range(1, bm_max + 1):
        for bn in range(1, bn_max + 1):
            for bk in range(1, bk_max + 1):
                if selected_block is not None and (bm, bk, bn) != selected_block:
                    continue
                considered += 1
                if not resources.partial_output_blocks and (mt % bm or nt % bn):
                    continue
                lhs = bm * kt * tm * storage.lhs.unit_bytes if type(storage) is DenseLoopStorage else 0
                rhs = bk * bn * tk * storage.rhs.unit_bytes if type(storage) is DenseLoopStorage else 0
                acc = (
                    bm * bn * tm * storage.accumulator.unit_bytes
                    if type(storage) is DenseLoopStorage and storage.accumulator
                    else 0
                )
                block = SharedInputBlock(bm, bk, bn, lhs + rhs, acc)
                if not _storage(group, primitive_tile, block, storage, resources):
                    continue
                count = mt * kt + _ceil(mt, bm) * nt * kt + 2 * mt * nt * kt + mt * nt
                if count <= max_commands:
                    requested = group.m * group.k + _ceil(mt, bm) * group.k * sum(group.widths)
                    candidates.append(((requested, count, lhs + rhs + acc, bm, bn, bk), block))
    if not candidates:
        return None
    _, block = min(candidates, key=lambda item: item[0])
    bm, bk, bn = block.m, block.k, block.n
    lanes = tuple((member, n) for member, width in enumerate(group.widths) for n in range(_ceil(width, tn)))
    effects = []
    for m0 in range(0, mt, bm):
        for mi in range(m0, min(mt, m0 + bm)):
            for ki in range(kt):
                effects.append(
                    SharedInputEffect(
                        None,
                        Load(
                            "lhs",
                            (mi, ki),
                            mi * tm,
                            ki * tk,
                            min(tm, group.m - mi * tm),
                            min(tk, group.k - ki * tk),
                            storage.lhs.base + ((mi - m0) * kt + ki) * tm,
                            ki,
                        ),
                    )
                )
        for n0 in range(0, nt, bn):
            wave = lanes[n0 : n0 + bn]
            for k0 in range(0, kt, bk):
                for ki in range(k0, min(kt, k0 + bk)):
                    for slot, (member, ni) in enumerate(wave):
                        effects.append(
                            SharedInputEffect(
                                member,
                                Load(
                                    "weight",
                                    (ki, ni),
                                    ki * tk,
                                    ni * tn,
                                    min(tk, group.k - ki * tk),
                                    min(tn, group.widths[member] - ni * tn),
                                    storage.rhs.base + ((ki - k0) * bn + slot) * tk,
                                    ki,
                                ),
                            )
                        )
                for ki in range(k0, min(kt, k0 + bk)):
                    depth = min(tk, group.k - ki * tk)
                    for slot, (member, ni) in enumerate(wave):
                        cols = min(tn, group.widths[member] - ni * tn)
                        for mi in range(m0, min(mt, m0 + bm)):
                            rows = min(tm, group.m - mi * tm)
                            fresh = mi == m0
                            effects.append(
                                SharedInputEffect(
                                    member,
                                    Preload(
                                        storage.rhs.base + ((ki - k0) * bn + slot) * tk if fresh else None,
                                        (ki, ni) if fresh else None,
                                        storage.accumulator.base + ((mi - m0) * bn + slot) * tm,
                                        ki != 0,
                                        rows,
                                        cols,
                                        depth,
                                    ),
                                )
                            )
                            effects.append(
                                SharedInputEffect(
                                    member,
                                    Compute(
                                        storage.lhs.base + ((mi - m0) * kt + ki) * tm,
                                        (mi, ki),
                                        rows,
                                        depth,
                                        fresh,
                                    ),
                                )
                            )
            for slot, (member, ni) in enumerate(wave):
                for mi in range(m0, min(mt, m0 + bm)):
                    effects.append(
                        SharedInputEffect(
                            member,
                            Store(
                                mi * tm,
                                ni * tn,
                                min(tm, group.m - mi * tm),
                                min(tn, group.widths[member] - ni * tn),
                                storage.accumulator.base + ((mi - m0) * bn + slot) * tm,
                            ),
                        )
                    )
    effects = tuple(effects)
    realization = verify_shared_input(
        effects,
        module=module,
        group=group,
        primitive_tile=primitive_tile,
        resources=resources,
        storage=storage,
        max_commands=max_commands,
        max_rows_visited=max_rows_visited,
        max_source_operations=max_source_operations,
        max_source_elements=max_source_elements,
    )
    return SharedInputSchedule(group, primitive_tile, block, effects, considered, realization)
