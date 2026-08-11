"""The frozen shape corpus a matrix-extension microkernel must be exactly right on, and its reference.

This is the acceptance surface for the whole outer-product delta, and it carries more weight than a
corpus normally would: with no usable functional simulator for the unit and no FPGA bitstream, there is
no whole-model numerical oracle in this pass, so correctness rests here. It is also the exact surface a
prior integration escaped through — that one reported full coverage from static instruction counts while
its narrow-M case read past the end of an operand panel.

Three properties, each deliberate:

* **The reference wraps, and the wrap turns out to be unreachable.** The hardware's accumulator has no
  saturation logic (the RTL says so in a ``TODO``), so where it overflows it wraps, and the reference
  matches that. But the bound is derivable: int8 × int8 into int32 cannot exceed the accumulator below
  ``K ≈ 133,152`` (:data:`ACC_OVERFLOW_UNREACHABLE_ABOVE_K`), which is three orders of magnitude above
  the longest reduction in the workload census. So the missing saturation is a **retired** hazard on this
  datapath rather than an open one, and the corpus contains no wrap case: writing one would need a K no
  model produces, and a case that claims to exercise wrapping while staying inside the range is worse
  than none. The reference's wrap is unit-tested directly on synthetic wide values instead.
* **Comparison is exact.** Integer arithmetic has no rounding, so the gate is equality, not cosine and
  not a tolerance. An aggregate similarity gate has previously accepted a kernel that was over 1000%
  wrong on individual elements.
* **A narrow parallel extent is a named case, not a corner.** `M = 1` and `N = 1` are called out by
  name. The workload census found these are numerous and arithmetically negligible (49 of 91
  contractions in one model, together under 1% of its work), so they earn their place here for
  correctness, not for throughput — which is also why a cost model is expected to decline to route them.

The corpus is DATA (``merlin/tests/data/opu_shapes/corpus.json``) rather than a generator, so a case
cannot silently disappear when the enumeration logic changes, and a case that was ever certified stays in
the record.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["Case", "CORPUS_PATH", "load_corpus", "reference", "resolve_extent", "select",
           "write_corpus"]

#: int32 wrap boundary. Not a target fact — the accumulator width is derived from the hardware and passed
#: in; this is the modulus for the width the corpus is written at.
_ACC_BITS = 32


def corpus_path() -> Path:
    """The frozen corpus file. Under the test data root, because it is an input to a gate."""
    from ..common.paths import merlin_dir
    return Path(merlin_dir()) / "tests" / "data" / "opu_shapes" / "corpus.json"


CORPUS_PATH = "merlin/tests/data/opu_shapes/corpus.json"


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

    Supports ``tile`` and ``tile/<divisor>``. Anything else raises rather than falling back to a default,
    because a silently mis-resolved extent produces a corpus that tests a different shape than it claims.
    """
    if isinstance(extent, int):
        return extent
    text = str(extent).strip()
    head, sep, divisor = text.partition("/")
    if head.strip() != _TILE:
        raise ValueError(f"unsupported extent expression {extent!r}; expected an int, 'tile', "
                         f"or 'tile/<divisor>'")
    if not sep:
        return int(tile)
    try:
        d = int(divisor)
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
    seed: int = 0
    #: Amplitude of the generated operands. The full-range case needs the largest magnitudes the datapath
    #: can be handed; most cases do not, and a corpus where every case ran at full range would not
    #: distinguish a magnitude bug from a plain arithmetic one.
    amplitude: int = 8

    def resolved(self, tile: int) -> "Case":
        """This case with every extent resolved against ``tile``."""
        return Case(name=self.name, m=resolve_extent(self.m, tile), n=resolve_extent(self.n, tile),
                    k=resolve_extent(self.k, tile), why=self.why, bias=self.bias, seed=self.seed,
                    amplitude=self.amplitude)

    def fits_tile(self, tile: int) -> bool:
        """Whether both parallel extents fit one tile, i.e. whether a single-tile kernel can run it.

        A case that does not fit is not a failure and must not be reported as one — it needs a kernel
        that tiles, and until there is one it is honestly ``not_run`` for that target.
        """
        got = self.resolved(tile)
        return int(got.m) <= int(tile) and int(got.n) <= int(tile)

    def operands(self) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """``(lhs, rhs, bias)`` — deterministic from ``seed`` so a failure is reproducible.

        The LHS is produced K-major (``k`` rows of ``m``), which is the layout the hardware forces on
        both operands: the expert kernel indexes ``at[k*M + i]`` and ``b[k*N + j]``. Handing back an
        M-major LHS here would hide the transpose/packing cost that the routing decision has to price.
        """
        if not all(isinstance(e, int) for e in (self.m, self.n, self.k)):
            raise ValueError(f"case {self.name!r} still has a tile-relative extent "
                             f"(m={self.m!r}, n={self.n!r}, k={self.k!r}); call .resolved(tile) first — "
                             "generating operands from an unresolved extent would silently test a "
                             "different shape than the case names")
        rng = np.random.default_rng(self.seed)
        lo, hi = -self.amplitude, self.amplitude
        lhs = rng.integers(lo, hi + 1, size=(self.k, self.m), dtype=np.int8)   # K-major
        rhs = rng.integers(lo, hi + 1, size=(self.k, self.n), dtype=np.int8)   # K-major
        bias = (rng.integers(-1000, 1001, size=(self.n,)).astype(np.int32)
                if self.bias else None)
        return lhs, rhs, bias


def reference(lhs: np.ndarray, rhs: np.ndarray, bias: np.ndarray | None = None, *,
              acc_bits: int = _ACC_BITS) -> np.ndarray:
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
    acc = a.T @ b                                   # (M, N)
    if bias is not None:
        acc = acc + np.asarray(bias, dtype=np.int64)[None, :]
    return _wrap(acc, acc_bits)


def _wrap(values: np.ndarray, bits: int) -> np.ndarray:
    """Two's-complement wrap to ``bits``, which is what an accumulator with no saturation logic does."""
    span = 1 << bits
    half = span >> 1
    wrapped = (values.astype(object) + half) % span - half
    return wrapped.astype(np.int32 if bits == 32 else np.int64)


#: The frozen cases. Each `why` is the reason it may not be deleted. The companion extent is `tile`
#: rather than a number, so the same corpus is meaningful on every configuration of the unit.
_CASES: tuple[Case, ...] = (
    # --- the historical failure, by name -------------------------------------------------------
    Case("narrow_m_1", 1, _TILE, 64, "M=1 vecmat: the prior integration packed this to a single row "
         "and its operand load read past the end of the panel", seed=1),
    Case("narrow_n_1", _TILE, 1, 64, "N=1 matvec: the mirror case, where the column operand is one "
         "lane", seed=2),
    Case("narrow_both_1", 1, 1, 64, "both parallel extents 1: a rank-1 accumulate reduced to a dot "
         "product, where every length in play is 1", seed=3),
    # --- parallel extents around the tile -----------------------------------------------------
    *(Case(f"m_{m}", m, _TILE, 64, "M swept below, at and just over a plausible tile edge, and odd",
           seed=10 + m) for m in (2, 3, 7, 15, 16, 17, 31, 32)),
    *(Case(f"n_{n}", _TILE, n, 64, "N swept below, at and just over a plausible tile edge, and odd",
           seed=40 + n) for n in (2, 3, 7, 15, 16, 17, 31, 32)),
    # --- reduction extent ---------------------------------------------------------------------
    *(Case(f"k_{k}", "tile/2", "tile/2", k, "K tiny / odd / exact / one over / long: the reduction "
           "tail is harmless only if it is actually handled", seed=80 + k)
      for k in (1, 2, 3, 15, 16, 17, 255)),
    # --- non-square and asymmetric ------------------------------------------------------------
    Case("asymmetric_m_gt_n", _TILE, "tile/4", 24, "M != N, so an operand swap in the accumulate is "
         "NOT shape-safe: this is the case that would have caught the reversed operand order",
         seed=200),
    Case("asymmetric_n_gt_m", "tile/4", _TILE, 24, "the other asymmetry, so a swap fails in both "
         "directions", seed=201),
    # --- epilogue -----------------------------------------------------------------------------
    Case("bias", "tile/2", "tile/2", 32, "bias initialised into the accumulator by broadcast rather "
         "than added afterwards, which is the hardware's own init path", bias=True, seed=300),
    Case("bias_narrow_m", 1, "tile/2", 32, "bias broadcast with a single output row: the init writes "
         "all rows whether or not they are read", bias=True, seed=301),
    # --- accumulator behaviour ----------------------------------------------------------------
    Case("full_range_int8", "tile/2", "tile/2", 255, "full-range int8 operands over a long reduction: the largest "
         "magnitudes the datapath can be handed. It does NOT overflow, and cannot -- see "
         "ACC_OVERFLOW_UNREACHABLE_ABOVE_K, which is why no corpus case tests the missing saturation",
         seed=400, amplitude=127),
)

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
    """The frozen corpus, read from disk. Falls back to the in-module cases only when the file is
    absent, and says which happened via :func:`corpus_is_frozen`."""
    p = Path(path) if path is not None else corpus_path()
    if not p.is_file():
        return _CASES
    raw = json.loads(p.read_text(encoding="utf-8"))
    return tuple(Case(**c) for c in raw["cases"])


def select(tile: int, *, path: "str | Path | None" = None
           ) -> tuple[tuple[Case, ...], tuple[tuple[Case, str], ...]]:
    """``(runnable, [(skipped, reason)])`` for a target whose logical tile edge is ``tile``.

    Every case is resolved against ``tile`` first, then split by whether a SINGLE-TILE kernel can run it.
    A case that does not fit is returned with its reason rather than dropped: a certification report that
    silently omitted the shapes the kernel cannot yet reach would read as full coverage of the corpus,
    which is the precise shape of the failure this corpus exists to prevent.
    """
    runnable: list[Case] = []
    skipped: list[tuple[Case, str]] = []
    for case in load_corpus(path):
        got = case.resolved(tile)
        if case.fits_tile(tile):
            runnable.append(got)
        else:
            skipped.append((got, f"m={got.m} n={got.n} exceeds the {tile}x{tile} tile; needs a kernel "
                                 f"that tiles M and N"))
    return tuple(runnable), tuple(skipped)


def corpus_is_frozen(path: "str | Path | None" = None) -> bool:
    p = Path(path) if path is not None else corpus_path()
    return p.is_file()


def write_corpus(path: "str | Path | None" = None) -> Path:
    """Write the in-module cases out as the frozen file.

    Intended to be run once and reviewed, not regenerated in a build: the point of freezing is that a
    case cannot vanish because the enumeration changed.
    """
    p = Path(path) if path is not None else corpus_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "note": ("Frozen acceptance corpus for the outer-product delta. Cases are removed only with a "
                 "written reason; each `why` states what it is protecting. Comparison against "
                 "kernels.opu_corpus.reference is EXACT integer equality -- the accumulator does not "
                 "saturate, so the reference wraps."),
        "cases": [asdict(c) for c in _CASES],
    }
    p.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    return p
