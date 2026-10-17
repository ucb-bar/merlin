"""Bounded contraction blocking from explicit storage and emission constraints.

This module uses only the standard library so standalone compilers can carry
an exact, versioned copy. Instruction encoding, layouts, effects and numerical
legality belong to the caller. Counts describe requests, never device cycles.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


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


def dense_blocks(problem: DenseProblem, resources: DenseResources) -> tuple[DenseBlock, ...]:
    """Enumerate bounded M/N choices without expanding tensors or dynamic loops.

    One largest resource-legal K panel is considered for each M/N choice. This
    initial family preserves increasing K per output and holds the complete
    output block in private accumulators. Other dataflows need other families.
    """
    p, r = problem, resources
    mt, kt, nt = p.tiles
    if not r.partial_output_tiles and (p.m % p.tile_m or p.n % p.tile_n):
        return ()
    if min(mt, r.max_m_blocks) * min(nt, r.max_n_blocks) > r.candidate_budget:
        raise ValueError("dense blocking search exceeds its candidate budget")
    choices = []
    for bm in range(1, min(mt, r.max_m_blocks) + 1):
        for bn in range(1, min(nt, r.max_n_blocks) + 1):
            if not r.partial_output_blocks and (mt % bm or nt % bn):
                continue
            acc = bm * bn * p.tile_m * p.tile_n * p.accumulator_bytes
            per_k = bm * p.tile_m * p.tile_k * p.lhs_bytes + bn * p.tile_k * p.tile_n * p.rhs_bytes
            bk = min(kt, r.max_k_blocks, r.operand_bytes // per_k)
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
    for block in dense_blocks(problem, resources):
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
