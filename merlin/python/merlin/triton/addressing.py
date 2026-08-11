"""Pointer -> tensor re-raising: the analysis that turns Triton addressing back into whole tensors.

A Triton kernel does not say "multiply these two matrices". It says "one program instance, given its
`program_id`, computes these addresses, loads through them, and stores through those". Merlin's
staged pipeline needs the opposite: tensor-typed function arguments feeding a contraction directly
(an operand that is not a block argument is a `KeyError` in interface materialization, and a
bufferized operand traces to nothing in contract inference). Recovering the first form from the
second is the hard part of the bridge; op-by-op translation is the easy part.

The recovery works on two ideas.

**Addresses are affine.** Every index-space value in TTIR is `constant + Σ cᵍ·program_id[g] +
Σ kᵈ·iota[d]` over the tile — `tt.make_range` introduces an iota, `tt.splat` broadcasts a scalar,
`tt.expand_dims`/`tt.broadcast` move iotas between dimensions, and `arith.addi`/`muli` combine them.
:class:`Affine` is that form, and anything that does not fit it is refused rather than approximated.

**The grid is normalized, not lowered.** The SPMD grid is not turned into threads, lanes, or a loop
— choosing target parallelism is Merlin's decision and baking one choice in here would make one
accelerator work and another impossible. Instead the grid is *enumerated* and the accesses of every
program instance are collected. If they tile a declared argument exactly — every element once, in
order — then the whole launch is equivalent to naming that argument as a tensor, and the grid has
disappeared into ordinary whole-function semantics without any parallelism decision being made.

That enumeration also settles masks, which is the point of doing it concretely rather than
symbolically. A masked tail is exactly what stops a grid of `ceil(N/BLOCK)` instances from running
past the end, so ignoring a mask turns coverage from exact into over-covering and the check fails.
The most dangerous bug in this area — a dropped mask, which still computes the right answer whenever
the block size happens to divide the extent — cannot survive it.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from .diagnostics import BridgeError

# Enumeration is O(elements accessed). Beyond this the analysis refuses instead of stalling; a
# kernel that big needs the symbolic version of this check, which is a deliberate later step.
MAX_ENUMERATED_ELEMENTS = 1 << 22


@dataclass(frozen=True)
class Affine:
    """An index-space value over a tile: ``const + Σ pid[g]·program_id[g] + Σ iota[d]·index[d]``.

    ``shape`` is the tile shape this value has in TTIR; a rank-0 shape is a scalar. Coefficients are
    kept as sparse maps so that "does not depend on the grid" is a checkable property (``not pid``)
    rather than a comparison against zeros.
    """

    shape: tuple[int, ...] = ()
    const: int = 0
    pid: Mapping[int, int] = field(default_factory=dict)
    iota: Mapping[int, int] = field(default_factory=dict)

    @property
    def is_constant(self) -> bool:
        return not self.pid and not self.iota

    def evaluate(self, program_id: tuple[int, int, int], index: tuple[int, ...]) -> int:
        v = self.const
        for axis, coef in self.pid.items():
            v += coef * program_id[axis]
        for dim, coef in self.iota.items():
            v += coef * index[dim]
        return v

    def splat_to(self, shape: tuple[int, ...]) -> Affine:
        """Broadcast a scalar across ``shape`` — the value is the same at every position."""
        if self.shape:
            raise BridgeError(f"cannot splat a rank-{len(self.shape)} value", op="tt.splat")
        return Affine(shape=shape, const=self.const, pid=dict(self.pid), iota={})

    def expand_dims(self, axis: int) -> Affine:
        """Insert a unit dimension at ``axis``; iotas at or past it shift up one dimension."""
        shape = self.shape[:axis] + (1,) + self.shape[axis:]
        iota = {(d + 1 if d >= axis else d): c for d, c in self.iota.items()}
        return Affine(shape=shape, const=self.const, pid=dict(self.pid), iota=iota)

    def broadcast_to(self, shape: tuple[int, ...]) -> Affine:
        """Stretch unit dimensions. A stretched dimension must carry no iota, or it would alias."""
        if len(shape) != len(self.shape):
            raise BridgeError(
                f"broadcast changes rank {len(self.shape)} -> {len(shape)}", op="tt.broadcast")
        for d, (have, want) in enumerate(zip(self.shape, shape)):
            if have != want and (have != 1 or self.iota.get(d)):
                raise BridgeError(
                    f"cannot broadcast dimension {d} from {have} to {want}", op="tt.broadcast")
        return Affine(shape=shape, const=self.const, pid=dict(self.pid), iota=dict(self.iota))

    def __add__(self, other: Affine) -> Affine:
        shape = self.shape or other.shape
        if self.shape and other.shape and self.shape != other.shape:
            raise BridgeError(f"cannot add index values of shapes {self.shape} and {other.shape}")
        return Affine(shape=shape, const=self.const + other.const,
                      pid=_merge(self.pid, other.pid), iota=_merge(self.iota, other.iota))

    def scaled(self, factor: int) -> Affine:
        return Affine(shape=self.shape, const=self.const * factor,
                      pid={k: v * factor for k, v in self.pid.items()},
                      iota={k: v * factor for k, v in self.iota.items()})

    def __mul__(self, other: Affine) -> Affine:
        """Affine × affine is only affine when one side is a constant — otherwise refuse."""
        if other.is_constant:
            return self.scaled(other.const).with_shape(self.shape or other.shape)
        if self.is_constant:
            return other.scaled(self.const).with_shape(self.shape or other.shape)
        raise BridgeError(
            "index expression is not affine: both operands of a multiply vary",
            hint="tile offsets must be `constant * program_id/iota` sums; a product of two varying "
                 "indices needs a real polyhedral analysis, which the bridge does not do")

    def with_shape(self, shape: tuple[int, ...]) -> Affine:
        return Affine(shape=shape, const=self.const, pid=dict(self.pid), iota=dict(self.iota))


def _merge(a: Mapping[int, int], b: Mapping[int, int]) -> dict[int, int]:
    out = dict(a)
    for k, v in b.items():
        out[k] = out.get(k, 0) + v
    return {k: v for k, v in out.items() if v}


@dataclass(frozen=True)
class Predicate:
    """A comparison of two index values — a Triton mask, kept exactly rather than approximated."""

    lhs: Affine
    rhs: Affine
    kind: str  # "slt" | "sle" | "sgt" | "sge" | "eq" | "ne"

    _OPS = {"slt": lambda a, b: a < b, "sle": lambda a, b: a <= b,
            "sgt": lambda a, b: a > b, "sge": lambda a, b: a >= b,
            "eq": lambda a, b: a == b, "ne": lambda a, b: a != b}

    def holds(self, program_id: tuple[int, int, int], index: tuple[int, ...]) -> bool:
        try:
            op = self._OPS[self.kind]
        except KeyError:
            raise BridgeError(f"unsupported mask comparison {self.kind!r}", op="arith.cmpi") from None
        return bool(op(self.lhs.evaluate(program_id, index), self.rhs.evaluate(program_id, index)))

    def conjunction(self, other: Predicate) -> Conjunction:
        return Conjunction((self, other))


@dataclass(frozen=True)
class Conjunction:
    """Several predicates that must all hold — `arith.andi` over masks."""

    terms: tuple[Predicate, ...]

    def holds(self, program_id: tuple[int, int, int], index: tuple[int, ...]) -> bool:
        return all(t.holds(program_id, index) for t in self.terms)


@dataclass(frozen=True)
class PointerTensor:
    """A tile of pointers: a declared pointer argument plus an affine element offset from its base."""

    base: str
    offset: Affine

    @property
    def shape(self) -> tuple[int, ...]:
        return self.offset.shape


def row_major_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    strides = [1] * len(shape)
    for d in range(len(shape) - 2, -1, -1):
        strides[d] = strides[d + 1] * shape[d + 1]
    return tuple(strides)


def _tile_indices(shape: tuple[int, ...]):
    """Every coordinate of a tile, in row-major order."""
    if not shape:
        yield ()
        return
    idx = [0] * len(shape)
    total = 1
    for d in shape:
        total *= d
    for _ in range(total):
        yield tuple(idx)
        for d in range(len(shape) - 1, -1, -1):
            idx[d] += 1
            if idx[d] < shape[d]:
                break
            idx[d] = 0


def whole_tensor_access(ptr: PointerTensor, *, shape: tuple[int, ...], grid: tuple[int, int, int],
                        mask: Predicate | Conjunction | None) -> str:
    """Verify that the launch reads/writes every element of ``shape`` exactly once, in order.

    Returns a short description of the recognized pattern for the capability report, or raises
    :class:`BridgeError` explaining precisely how the access differs. "In order" is required and not
    merely "covers": an access that is a permutation of the tensor (a transposed tile, say) covers
    it perfectly but means something different, so it is reported as its own case rather than
    silently accepted.
    """
    strides = row_major_strides(shape)
    total = strides[0] * shape[0] if shape else 1
    tile = ptr.shape
    tile_elems = 1
    for d in tile:
        tile_elems *= d
    visited = tile_elems * grid[0] * grid[1] * grid[2]
    if visited > MAX_ENUMERATED_ELEMENTS:
        raise BridgeError(
            f"access to {ptr.base!r} spans {visited} elements, above the {MAX_ENUMERATED_ELEMENTS} "
            "the bridge will enumerate",
            hint="verifying coverage concretely is what makes masks impossible to ignore; a kernel "
                 "this large needs the symbolic form of the check")

    order: list[int] = []
    for pz in range(grid[2]):
        for py in range(grid[1]):
            for px in range(grid[0]):
                pid = (px, py, pz)
                for index in _tile_indices(tile):
                    if mask is not None and not mask.holds(pid, index):
                        continue
                    order.append(ptr.offset.evaluate(pid, index))

    if order == list(range(total)):
        if grid == (1, 1, 1):
            return f"whole tensor{list(shape)}"
        return f"whole tensor{list(shape)} tiled by grid{list(grid)} into tile{list(tile)}"

    seen = set(order)
    if len(seen) != len(order):
        raise BridgeError(
            f"argument {ptr.base!r} is accessed more than once at some element — the bridge re-raises "
            "an argument to a single tensor value and cannot express overlapping access")
    if seen == set(range(total)):
        raise BridgeError(
            f"argument {ptr.base!r} is accessed in full but not in order (a permutation such as a "
            "transposed tile)",
            hint="express the permutation in the kernel, or wait for the bridge to emit an explicit "
                 "linalg.transpose for it")
    missing = sorted(set(range(total)) - seen)
    extra = sorted(seen - set(range(total)))
    detail = []
    if missing:
        detail.append(f"{len(missing)} element(s) never accessed (first: {missing[0]})")
    if extra:
        detail.append(f"{len(extra)} access(es) outside the declared shape (first: {extra[0]})")
    raise BridgeError(
        f"argument {ptr.base!r} declared {list(shape)} is not covered exactly by the launch: "
        + "; ".join(detail),
        hint="check the declared shape, the grid, and whether the kernel's mask bounds match the "
             "extent stated in the spec — an out-of-range access here usually means a missing mask")
