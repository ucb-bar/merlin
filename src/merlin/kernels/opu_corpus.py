"""Explicit corpus loading and integer reference semantics.

The caller supplies a selected corpus produced and reviewed outside this module.
No fixed case roster is bundled or synthesized here. Missing or empty input
cannot establish coverage; the reference arithmetic is independent of selection.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = [
    "Case",
    "REQUANT_SHIFT",
    "load_corpus",
    "reference",
    "requantize",
    "resolve_extent",
    "select",
]

#: int32 wrap boundary. Not a target fact — the accumulator width is derived from the hardware and passed
#: in; this is the modulus for the width the corpus is written at.
_ACC_BITS = 32


#: Extents may be written as an expression over ``tile`` — the target's derived logical tile edge —
#: instead of as a number. The swept extents (1, 2, 3, 7, 15, 16, 17, …) are deliberately absolute: they
#: name specific awkward values and mean nothing rescaled. But the COMPANION extent, held constant while
#: the other is swept, means "a full tile", and writing it as a literal silently ties the corpus to one
#: configuration: at ``vLen=256`` the tile edge is 32, so a literal 64 puts 18 of 31 cases — including
#: every named narrow-extent regression — outside a single tile, and at ``vLen=128`` it is worse.
#: Resolved structurally (``tile``, ``tile/2``, ``tile/4``), never by evaluating a string as code.
_TILE = "tile"


def resolve_extent(extent: "int | str", tile: int) -> int:
    """A concrete extent from a number or a ``tile``-relative expression.

    Supports ``tile``, ``tile/<divisor>`` and ``tile*<multiplier>``. Anything else raises rather than
    falling back to a default, because a silently mis-resolved extent produces a corpus that tests a
    different shape than it claims.

    ``tile*<multiplier>`` exists because without it the corpus could only ever describe extents at or
    BELOW one tile, and so could not state "several tile columns" or "a long reduction" at all. That
    was not a cosmetic limit: every narrow-M case here ran at ``n = tile`` and ``k = 64``, i.e. a
    single column block and the shortest useful reduction, so narrow M combined with many column
    blocks or a deep accumulation was certified nowhere -- which is exactly the regime a whole-model
    decode shape puts the unit in (M is the sequence length, N and K are the layer widths).

    Parsed structurally rather than by pattern, per the repo's no-regex rule.
    """
    if isinstance(extent, int):
        return extent
    text = str(extent).strip()
    head, sep, rhs = text.partition("/")
    if not sep:
        head, sep, rhs = text.partition("*")
        if head.strip() != _TILE:
            raise ValueError(
                f"unsupported extent expression {extent!r}; expected an int, 'tile', "
                f"'tile/<divisor>', or 'tile*<multiplier>'"
            )
        if not sep:
            return int(tile)
        try:
            mul = int(rhs)
        except ValueError as exc:
            raise ValueError(f"unsupported multiplier in {extent!r}") from exc
        if mul < 1:
            raise ValueError(f"multiplier must be >= 1 in {extent!r}")
        return int(tile) * mul
    if head.strip() != _TILE:
        raise ValueError(
            f"unsupported extent expression {extent!r}; expected an int, 'tile', "
            f"'tile/<divisor>', or 'tile*<multiplier>'"
        )
    try:
        d = int(rhs)
    except ValueError as exc:
        raise ValueError(f"unsupported divisor in {extent!r}") from exc
    if d < 1:
        raise ValueError(f"divisor must be >= 1 in {extent!r}")
    return max(1, int(tile) // d)


@dataclass(frozen=True)
class Case:
    """One shape the microkernel must be exactly right on, with why it is in the corpus.

    ``m`` / ``n`` / ``k`` are each an int or a ``tile``-relative expression; call :meth:`resolved` with
    the target's derived tile edge to get numbers.
    """

    name: str
    m: "int | str"
    n: "int | str"
    k: "int | str"
    why: str
    bias: bool = False
    #: Apply the int8 requant epilogue and judge the NARROWED output instead of the int32 accumulator.
    #: This is the L4 rung: a dispatch, not just a datapath.
    requant: bool = False
    seed: int = 0
    #: Amplitude of the generated operands. The full-range case needs the largest magnitudes the datapath
    #: can be handed; most cases do not, and a corpus where every case ran at full range would not
    #: distinguish a magnitude bug from a plain arithmetic one.
    amplitude: int = 8

    def resolved(self, tile: int) -> "Case":
        """This case with every extent resolved against ``tile``."""
        # EVERY field has to be carried. A new field forgotten here is dropped the moment a case is
        # resolved, which is always -- so the case would run without the property it was added for and
        # report a pass. `requant` was added after this method and is exactly that trap.
        return Case(
            name=self.name,
            m=resolve_extent(self.m, tile),
            n=resolve_extent(self.n, tile),
            k=resolve_extent(self.k, tile),
            why=self.why,
            bias=self.bias,
            requant=self.requant,
            seed=self.seed,
            amplitude=self.amplitude,
        )

    def fits_tile(self, tile: int) -> bool:
        """Whether both parallel extents fit ONE tile, i.e. whether a single-tile kernel can run it.

        Kept as a description of the shape, not as a run/defer decision — the emitter tiles M and N, so a
        case that does not fit one tile is perfectly runnable. :func:`select` takes the kernel's tiling
        ability as a parameter rather than reading it from here, because a predicate named after a shape is
        exactly the kind of thing that keeps being consulted after the capability it stood for has changed.
        """
        got = self.resolved(tile)
        return int(got.m) <= int(tile) and int(got.n) <= int(tile)

    def multiplier(self) -> np.ndarray | None:
        """Per-output-column requant multiplier, or None when this case has no epilogue.

        Per COLUMN rather than one scalar because that is what a quantised model actually has (a scale per
        output channel), and because a single multiplier would not catch an epilogue that applied the right
        arithmetic to the wrong column.

        Drawn from a stream OFFSET from the operand stream, so the multipliers are not correlated with the
        data they scale -- a multiplier that happened to track its own column's magnitude could hide a
        mis-indexed epilogue.
        """
        if not self.requant:
            return None
        if not isinstance(self.n, int):
            raise ValueError(f"case {self.name!r} has an unresolved N; call .resolved(tile) first")
        rng = np.random.default_rng(self.seed + 0x5EED)
        # Around half of 1 << REQUANT_SHIFT, so a typical accumulator lands inside int8 after the shift and
        # the clamp is exercised by the tails rather than by everything.
        half = 1 << (REQUANT_SHIFT - 1)
        return rng.integers(half // 4, half, size=(int(self.n),)).astype(np.int64)

    def operands(self) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """``(lhs, rhs, bias)`` — deterministic from ``seed`` so a failure is reproducible.

        The LHS is produced K-major (``k`` rows of ``m``), which is the layout the hardware forces on
        both operands: the expert kernel indexes ``at[k*M + i]`` and ``b[k*N + j]``. Handing back an
        M-major LHS here would hide the transpose/packing cost that the routing decision has to price.
        """
        if not all(isinstance(e, int) for e in (self.m, self.n, self.k)):
            raise ValueError(
                f"case {self.name!r} still has a tile-relative extent "
                f"(m={self.m!r}, n={self.n!r}, k={self.k!r}); call .resolved(tile) first — "
                "generating operands from an unresolved extent would silently test a "
                "different shape than the case names"
            )
        rng = np.random.default_rng(self.seed)
        lo, hi = -self.amplitude, self.amplitude
        lhs = rng.integers(lo, hi + 1, size=(self.k, self.m), dtype=np.int8)  # K-major
        rhs = rng.integers(lo, hi + 1, size=(self.k, self.n), dtype=np.int8)  # K-major
        bias = rng.integers(-1000, 1001, size=(self.n,)).astype(np.int32) if self.bias else None
        return lhs, rhs, bias


#: Right shift the requant epilogue applies. Fixed rather than swept: what is under test is whether the
#: readout, the multiply and the rounding are bit-exact on the unit, not which shift a model happens to use.
#: 15 keeps a full-range int8 product times a per-column multiplier inside int32 before the shift.
REQUANT_SHIFT = 15


def requantize(acc: np.ndarray, multiplier: np.ndarray, *, shift: int = REQUANT_SHIFT) -> np.ndarray:
    """The int8 epilogue: scale an int32 accumulator, round, clamp, narrow.

    This is what makes a contraction usable in a quantised model, and it is the half of a dispatch the
    datapath certification did not cover: everything before it stopped at the int32 readout. A kernel whose
    accumulator is right and whose epilogue rounds the other way produces a model that is subtly wrong
    everywhere rather than obviously wrong somewhere.

    ROUNDING IS HALF-UP ON THE SHIFTED MAGNITUDE, applied by adding half an LSB before an arithmetic shift.
    Stated because it is the one place a C implementation and a numpy one drift apart silently: an
    arithmetic right shift of a negative value rounds toward negative infinity, so `(x + half) >> s` and
    `round(x / 2**s)` disagree on exact halves of negative numbers. The C in the image does the same
    addition and the same shift, so the two agree by construction rather than by luck.
    """
    a = np.asarray(acc, dtype=np.int64)
    m = np.asarray(multiplier, dtype=np.int64)
    if m.ndim != 1 or m.shape[0] != a.shape[-1]:
        raise ValueError(f"multiplier must be one per output column; got {m.shape} for {a.shape}")
    scaled = a * m[None, :]
    scaled = (scaled + (1 << (int(shift) - 1))) >> int(shift)
    return np.clip(scaled, -128, 127).astype(np.int8)


def reference(
    lhs: np.ndarray, rhs: np.ndarray, bias: np.ndarray | None = None, *, acc_bits: int = _ACC_BITS
) -> np.ndarray:
    """The exact expected result: ``sum_k lhs[k, i] * rhs[k, j]`` accumulated in a wrapping int.

    Both operands are K-major, matching the hardware's own layout. Computed in int64 and wrapped once at
    the end rather than accumulated in the narrow type: numpy's int32 overflow is undefined-ish and
    warns, whereas the hardware's behaviour is a well-defined two's-complement wrap, and the two agree
    only if the wrap is applied deliberately.
    """
    a = np.asarray(lhs, dtype=np.int64)
    b = np.asarray(rhs, dtype=np.int64)
    if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[0]:
        raise ValueError(f"expected K-major (K, M) and (K, N); got {a.shape} and {b.shape}")
    acc = a.T @ b  # (M, N)
    if bias is not None:
        acc = acc + np.asarray(bias, dtype=np.int64)[None, :]
    return _wrap(acc, acc_bits)


def _wrap(values: np.ndarray, bits: int) -> np.ndarray:
    """Two's-complement wrap to ``bits``, which is what an accumulator with no saturation logic does."""
    span = 1 << bits
    half = span >> 1
    wrapped = (values.astype(object) + half) % span - half
    return wrapped.astype(np.int32 if bits == 32 else np.int64)


#: The reduction length at which int8 x int8 accumulation could first exceed a signed 32-bit
#: accumulator: ``(2**31 - 1) / 127**2``. Derived, not assumed, and it retires a stated hazard rather
#: than leaving it open: the RTL's missing saturation logic (an explicit ``TODO``) is UNREACHABLE on the
#: int8 datapath for any K a real contraction has -- the largest reduction in the workload census is
#: three orders of magnitude below this. So the corpus deliberately contains no wrap case: one would have
#: to use a K no model produces, and a case that claims to test wrapping while staying inside the range
#: is worse than no case at all. :func:`reference` still wraps, because the reference must match the
#: hardware wherever the hardware is defined, and it is unit-tested on synthetic wide values.
ACC_OVERFLOW_UNREACHABLE_ABOVE_K = ((1 << 31) - 1) // (127 * 127)


def load_corpus(path: "str | Path | None" = None) -> tuple[Case, ...]:
    """Read an explicitly selected corpus without substituting stock cases."""
    if path is None:
        raise ValueError("select a generated and reviewed corpus explicitly with path=...")
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"selected corpus is unavailable: {p}")
    raw = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("cases"), list) or not raw["cases"]:
        raise ValueError("selected corpus must contain a non-empty cases list; coverage is not established")
    return tuple(Case(**c) for c in raw["cases"])


def select(
    tile: int, *, path: "str | Path | None" = None, tiles_mn: bool = True, max_out_elems: int | None = None
) -> tuple[tuple[Case, ...], tuple[tuple[Case, str], ...]]:
    """``(runnable, [(skipped, reason)])`` for a target whose logical tile edge is ``tile``.

    Every case is resolved against ``tile`` first, then split by whether the kernel can actually run it. A
    case that cannot is returned WITH ITS REASON rather than dropped: a report that silently omitted the
    shapes out of reach would read as full coverage, which is the precise failure this corpus exists to
    prevent.

    **The deferral criterion is a property of the kernel, so it is a parameter.** It used to be hardcoded
    as "both parallel extents fit one tile", which was true of the single-tile kernel and is no longer true
    of anything: the emitter tiles M and N. Left as it was, every real workload shape would have been
    deferred with the reason "needs a kernel that tiles M and N" — a report claiming a limitation that had
    already been lifted, which is worse than claiming none, because it looks like diligence.
    ``tiles_mn=False`` restores the old behaviour for a kernel that genuinely cannot tile.

    ``max_out_elems`` defers cases whose output does not fit the image's result buffers. That is a real
    limit of the harness rather than of the datapath, and it is stated as such in the reason.

    ``K`` is never a reason to defer: the accumulator cannot overflow below
    :data:`ACC_OVERFLOW_UNREACHABLE_ABOVE_K`, and a case above it would be testing a reduction length no
    model produces.
    """
    runnable: list[Case] = []
    skipped: list[tuple[Case, str]] = []
    for case in load_corpus(path):
        got = case.resolved(tile)
        m, n, k = int(got.m), int(got.n), int(got.k)
        if not tiles_mn and (m > int(tile) or n > int(tile)):
            skipped.append((got, f"m={m} n={n} exceeds the {tile}x{tile} tile; needs a kernel that tiles M and N"))
        elif max_out_elems is not None and m * n > int(max_out_elems):
            skipped.append(
                (
                    got,
                    f"output is {m}x{n} = {m * n} int32 elements, over the image's "
                    f"{int(max_out_elems)}-element result buffers; a harness limit, not a "
                    f"datapath one",
                )
            )
        elif k > ACC_OVERFLOW_UNREACHABLE_ABOVE_K:
            skipped.append(
                (
                    got,
                    f"k={k} is at or above the accumulator-overflow bound "
                    f"{ACC_OVERFLOW_UNREACHABLE_ABOVE_K}, where the datapath's missing "
                    f"saturation becomes reachable; no model produces such a reduction",
                )
            )
        else:
            runnable.append(got)
    return tuple(runnable), tuple(skipped)
