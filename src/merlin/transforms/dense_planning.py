"""Bounded contraction blocking from explicit storage and emission constraints.

This module uses only the standard library so standalone compilers can carry
an exact, versioned copy. Instruction encoding, layouts, effects and numerical
legality belong to the caller. Counts describe requests, never device cycles.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from itertools import product


def _positive(**values: int) -> None:
    if any(type(value) is not int or value <= 0 for value in values.values()):
        raise ValueError("positive integer dimensions and resource bounds required: " + ", ".join(values))


def _ceil(value: int, unit: int) -> int:
    return (value + unit - 1) // unit


@dataclass(frozen=True)
class DenseProblem:
    """Logical contraction and the target's explicit primitive tile representation."""

    m: int
    k: int
    n: int
    tile_m: int
    tile_k: int
    tile_n: int
    lhs_bytes: int
    rhs_bytes: int
    accumulator_bytes: int
    bias: bool = False

    def __post_init__(self):
        _positive(**{key: value for key, value in vars(self).items() if key != "bias"})
        if type(self.bias) is not bool:
            raise ValueError("bias presence must be explicit Boolean")

    @property
    def tiles(self) -> tuple[int, int, int]:
        return _ceil(self.m, self.tile_m), _ceil(self.k, self.tile_k), _ceil(self.n, self.tile_n)


@dataclass(frozen=True)
class DenseResources:
    """Available bytes after reservations, plus the emitter's supported block domain.

    Capacities may describe a single usable bank rather than an entire memory.
    Shared backings and other live allocations must be accounted for by the
    adapter before constructing this view. It creates no asynchronous proof.
    """

    operand_bytes: int
    accumulator_bytes: int
    max_m_blocks: int = 8
    max_n_blocks: int = 8
    max_k_blocks: int = 1
    candidate_budget: int = 4096
    partial_output_tiles: bool = False
    partial_output_blocks: bool = False

    def __post_init__(self):
        _positive(
            **{
                key: value
                for key, value in vars(self).items()
                if key not in ("partial_output_tiles", "partial_output_blocks")
            }
        )
        if any(type(value) is not bool for value in (self.partial_output_tiles, self.partial_output_blocks)):
            raise ValueError("partial tile/block support must be explicit Boolean")


@dataclass(frozen=True)
class DenseBlock:
    """One legal geometric candidate; target instruction legality is checked separately."""

    m: int
    k: int
    n: int
    operand_bytes: int
    accumulator_bytes: int
    lhs_loads: int
    rhs_loads: int
    bias_loads: int
    output_tiles: int
    requested_operand_bytes: int

    @property
    def loads(self) -> int:
        return self.lhs_loads + self.rhs_loads + self.bias_loads


@dataclass(frozen=True)
class DenseChoice:
    block: DenseBlock
    static_commands: int
    candidates_considered: int
    cost_unit: str = "requested_operand_bytes_then_commands"


@dataclass(frozen=True)
class DenseEmission:
    """Observations reconstructed by an adapter from its emitted operations.

    Load/store counts include loop multiplicities; static commands count each
    instruction site once. High-water marks are exclusive byte offsets relative
    to the assigned operand and accumulator regions, after decoding addresses.
    These observations do not prove aliases, completion or numerical semantics.
    """

    static_commands: int
    lhs_loads: int
    rhs_loads: int
    bias_loads: int
    output_tiles: int
    operand_high_water: int
    accumulator_high_water: int

    def __post_init__(self):
        if any(type(value) is not int or value < 0 for value in vars(self).values()):
            raise ValueError("dense emission observations require nonnegative integers")


def verify_dense_emission(choice: DenseChoice, emission: DenseEmission, *, static_budget: int) -> None:
    """Require selected reuse and storage bounds to survive target emission.

    The adapter must inspect the actual operation stream, including loop bounds
    and memory addresses. Copying the planned counts into this report supplies
    no evidence. Instruction decoding and physical legality remain target-owned.
    A target's price may be a conservative static upper bound.
    """
    if type(static_budget) is not int or static_budget < 0:
        raise ValueError("static command budget must be a nonnegative integer")
    block = choice.block
    for name in ("lhs_loads", "rhs_loads", "bias_loads", "output_tiles"):
        if getattr(emission, name) != getattr(block, name):
            raise ValueError(f"dense transfer realization differs: {name}")
    if emission.static_commands > min(choice.static_commands, static_budget):
        raise ValueError("dense command realization exceeds selected bound")
    for role, used, reserved in (
        ("operand", emission.operand_high_water, block.operand_bytes),
        ("accumulator", emission.accumulator_high_water, block.accumulator_bytes),
    ):
        if not 0 < used <= reserved:
            raise ValueError(f"dense {role} realization exceeds selected storage or is missing")


def dense_blocks(
    problem: DenseProblem, resources: DenseResources, *, k_limit: Callable[[int, int], int] | None = None,
    n_choices: tuple[int, ...] | None = None,
) -> tuple[DenseBlock, ...]:
    """Enumerate bounded M/N choices without expanding tensors or dynamic loops.

    One largest resource-legal K panel is considered for each M/N choice. An
    optional physical ``k_limit(bm, bn)`` narrows the logical byte bound for
    separate stores, row padding or reserved windows; zero declines that M/N.
    The callback supplies a capacity limit, not a performance estimate. This
    initial family preserves increasing K per output and holds the complete
    output block in private accumulators. Other dataflows need other families.
    Explicit ``n_choices`` narrows the original supported widths and the
    candidate budget; it never extends the emitter's domain.
    """
    p, r = problem, resources
    mt, kt, nt = p.tiles
    widths = range(1, min(nt, r.max_n_blocks) + 1) if n_choices is None else n_choices
    if n_choices is not None and (type(n_choices) is not tuple or not n_choices
            or any(type(n) is not int or not 1 <= n <= min(nt, r.max_n_blocks) for n in n_choices)
            or len(set(n_choices)) != len(n_choices)):
        raise ValueError("dense N choices must be distinct original supported tile extents")
    if not r.partial_output_tiles and (p.m % p.tile_m or p.n % p.tile_n):
        return ()
    if min(mt, r.max_m_blocks) * len(widths) > r.candidate_budget:
        raise ValueError("dense blocking search exceeds its candidate budget")
    choices = []
    for bm in range(1, min(mt, r.max_m_blocks) + 1):
        for bn in widths:
            if not r.partial_output_blocks and (mt % bm or nt % bn):
                continue
            acc = bm * bn * p.tile_m * p.tile_n * p.accumulator_bytes
            per_k = bm * p.tile_m * p.tile_k * p.lhs_bytes + bn * p.tile_k * p.tile_n * p.rhs_bytes
            bk = min(kt, r.max_k_blocks, r.operand_bytes // per_k)
            if k_limit is not None:
                physical = k_limit(bm, bn)
                if type(physical) is not int or physical < 0:
                    raise ValueError("physical K limit must be a nonnegative integer")
                bk = min(bk, physical)
            if acc > r.accumulator_bytes or bk == 0:
                continue
            choices.append(
                DenseBlock(
                    bm,
                    bk,
                    bn,
                    per_k * bk,
                    acc,
                    mt * _ceil(nt, bn) * kt,
                    nt * _ceil(mt, bm) * kt,
                    mt * nt if p.bias else 0,
                    mt * nt,
                    p.m * p.k * p.lhs_bytes * _ceil(nt, bn) + p.k * p.n * p.rhs_bytes * _ceil(mt, bm),
                )
            )
    return tuple(choices)


def choose_dense_block(
    problem: DenseProblem,
    resources: DenseResources,
    *,
    static_budget: int,
    price: Callable[[DenseBlock], int | None],
    admits: Callable[[DenseBlock], bool] = lambda block: True,
    require_reuse: bool = True,
    k_limit: Callable[[int, int], int] | None = None,
) -> DenseChoice | None:
    """Choose a resource-legal candidate priced by its actual target emitter.

    ``None`` from ``price`` declines a candidate; zero is a valid count. A zero
    budget admits only a zero-command candidate. No result asks the caller to
    keep its existing legal streaming/loop path. This function emits no code.
    """
    if type(static_budget) is not int or static_budget < 0:
        raise ValueError("static command budget must be a nonnegative integer")
    if type(require_reuse) is not bool:
        raise ValueError("reuse requirement must be Boolean")
    mt, kt, nt = problem.tiles
    baseline_loads = mt * nt * (2 * kt + int(problem.bias))
    best = None
    considered = 0
    for block in dense_blocks(problem, resources, k_limit=k_limit):
        considered += 1
        if require_reuse and block.loads >= baseline_loads:
            continue
        if not admits(block):
            continue
        sites = price(block)
        if sites is None:
            continue
        if type(sites) is not int or sites < 0:
            raise ValueError("target price must be a nonnegative static command count or None")
        if sites > static_budget:
            continue
        key = (block.requested_operand_bytes, block.loads, sites, block.operand_bytes, block.m, block.n, -block.k)
        if best is None or key < best[0]:
            best = key, block, sites
    return None if best is None else DenseChoice(best[1], best[2], considered)


@dataclass(frozen=True)
class DenseGroupChoice:
    """Full-K shared-LHS choice, over the existing per-member dense blocks."""

    blocks: tuple[DenseBlock, ...]
    static_commands: int
    candidates_considered: int
    operand_bytes: int
    accumulator_bytes: int
    lhs_loads: int
    rhs_loads: int
    output_tiles: int
    requested_operand_bytes: int


def choose_dense_group(
    problems: tuple[DenseProblem, ...],
    resources: DenseResources,
    *,
    static_budget: int,
    price: Callable[[tuple[DenseBlock, ...]], int | None],
    admits: Callable[[tuple[DenseBlock, ...]], bool],
    k_limits: tuple[Callable[[int, int], int], ...],
    max_members: int = 8,
) -> DenseGroupChoice | None:
    """Intersect existing block candidates under one finite combined budget.

    All members keep the same complete LHS reduction resident. Their RHS and
    accumulator windows are separate. This does not grant retention, output
    aliasing, instruction semantics or member-interleaving permissions.
    """
    if (
        type(problems) is not tuple
        or type(max_members) is not int
        or not 2 <= len(problems) <= max_members <= 128
        or any(type(p) is not DenseProblem for p in problems)
        or type(resources) is not DenseResources
        or type(static_budget) is not int
        or static_budget < 0
        or type(k_limits) is not tuple
        or len(k_limits) != len(problems)
        or any(not callable(limit) for limit in k_limits)
    ):
        raise ValueError("dense grouping requires bounded typed members, resources and code budgets")
    first = problems[0]
    if any(
        (p.m, p.k, p.tile_m, p.tile_k, p.tile_n, p.lhs_bytes, p.rhs_bytes, p.accumulator_bytes, p.bias)
        != (
            first.m,
            first.k,
            first.tile_m,
            first.tile_k,
            first.tile_n,
            first.lhs_bytes,
            first.rhs_bytes,
            first.accumulator_bytes,
            False,
        )
        for p in problems
    ):
        raise ValueError("dense grouping requires common M/K, primitive, formats and exact zero seeds")
    domains = tuple(
        tuple(b for b in dense_blocks(p, resources, k_limit=limit) if b.k == p.tiles[1])
        for p, limit in zip(problems, k_limits, strict=True)
    )
    common_m = sorted(set.intersection(*(set(b.m for b in values) for values in domains)))
    combinations = []
    work = 0
    for bm in common_m:
        choices = tuple(tuple(b for b in values if b.m == bm) for values in domains)
        count = 1
        for values in choices:
            count *= len(values)
        work += count
        if work > resources.candidate_budget:
            return None
        combinations.append(choices)
    best = None
    for domains_at_m in combinations:
        for blocks in product(*domains_at_m):
            lhs = blocks[0].m * blocks[0].k * first.tile_m * first.tile_k * first.lhs_bytes
            operand = lhs + sum(
                b.k * b.n * p.tile_k * p.tile_n * p.rhs_bytes for p, b in zip(problems, blocks, strict=True)
            )
            accumulator = sum(b.accumulator_bytes for b in blocks)
            if operand > resources.operand_bytes or accumulator > resources.accumulator_bytes or not admits(blocks):
                continue
            commands = price(blocks)
            if commands is None:
                continue
            if type(commands) is not int or commands < 0:
                raise ValueError("dense group emission price must be a nonnegative count or unknown")
            if commands > static_budget:
                continue
            requested = first.m * first.k * first.lhs_bytes + sum(
                p.k * p.n * p.rhs_bytes * _ceil(p.tiles[0], b.m) for p, b in zip(problems, blocks, strict=True)
            )
            key = (requested, commands, operand + accumulator, blocks[0].m, tuple(b.n for b in blocks))
            choice = DenseGroupChoice(
                blocks,
                commands,
                work,
                operand,
                accumulator,
                first.tiles[0] * first.tiles[1],
                sum(b.rhs_loads for b in blocks),
                sum(b.output_tiles for b in blocks),
                requested,
            )
            if best is None or key < best[0]:
                best = key, choice
    return None if best is None else best[1]
