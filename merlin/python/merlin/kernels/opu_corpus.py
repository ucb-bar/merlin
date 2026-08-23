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
  ``K ≈ 133,152`` (:data:`ACC_OVERFLOW_UNREACHABLE_ABOVE_K`), which is 14.4x the longest reduction in
  the workload census (``K = 9216``, a decode-shaped ffn down). That margin was three orders of
  magnitude while the census was spectformer alone; whole-model decode shapes cut it by 9x, so it is
  now asserted by the gate rather than described. So the missing saturation is a **retired** hazard on this
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

__all__ = ["Case", "CORPUS_PATH", "REQUANT_SHIFT", "load_corpus", "reference", "requantize",
           "resolve_extent", "select", "write_corpus"]

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
            raise ValueError(f"unsupported extent expression {extent!r}; expected an int, 'tile', "
                             f"'tile/<divisor>', or 'tile*<multiplier>'")
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
        raise ValueError(f"unsupported extent expression {extent!r}; expected an int, 'tile', "
                         f"'tile/<divisor>', or 'tile*<multiplier>'")
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
        return Case(name=self.name, m=resolve_extent(self.m, tile), n=resolve_extent(self.n, tile),
                    k=resolve_extent(self.k, tile), why=self.why, bias=self.bias,
                    requant=self.requant, seed=self.seed, amplitude=self.amplitude)

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
    # --- L4: the DISPATCH, not just the datapath ----------------------------------------------
    # Every case above is judged on the int32 accumulator, which is where the readout stops. A quantised
    # model does not stop there: it scales, rounds, clamps and narrows to int8, and a kernel whose
    # accumulator is right while its epilogue rounds the other way produces a model that is subtly wrong
    # everywhere rather than obviously wrong somewhere. These are judged on the NARROWED output.
    Case("requant_square", "tile/2", "tile/2", 32, "the epilogue on a square tile: the base case for "
         "scale/round/clamp/narrow being bit-exact against numpy", requant=True, seed=600),
    Case("requant_narrow_m", 1, "tile/2", 32, "epilogue with a single output row -- the readout writes "
         "one row and the narrowing must follow it, not the whole tile", requant=True, seed=601),
    Case("requant_narrow_n", "tile/2", 1, 32, "epilogue on a single output column: a per-column "
         "multiplier indexed wrongly is invisible when there is more than one column to average over",
         requant=True, seed=602),
    Case("requant_tail", 17, 33, 24, "epilogue across a short row tile AND a short column tile, so the "
         "narrowing is exercised where the tiling loop's bounds are ragged", requant=True, seed=603),
    Case("requant_saturating", "tile/2", "tile/2", 255, "full-range operands over a long reduction, so "
         "the accumulator is large and the clamp actually fires -- the cases above mostly do not reach "
         "it, and a clamp that never runs is a clamp that is not tested", requant=True, seed=604,
         amplitude=127),
    Case("requant_with_bias", "tile/2", "tile/2", 32, "bias broadcast into the accumulator AND the "
         "epilogue applied after it: the two halves of the dispatch composed, which is the L4 rung",
         bias=True, requant=True, seed=605),
    # --- the shapes a real workload actually asks for -----------------------------------------
    # Every case above is a PROBE: a swept extent, a degenerate edge, a named past failure. None of them
    # is a shape any model produces, so passing all of them says the kernel handles the awkward cases
    # without saying it handles the work. These five are the distinct signatures the compile path mints
    # for spectformer int8 (llvmlower/passes_opu on the prepared module), i.e. the contractions that
    # actually get routed, carrying ~88% of the model's arithmetic between them.
    #
    # They are ABSOLUTE, not tile-relative, on purpose: a workload's extents are a fact about the
    # workload and do not scale with the hardware. That means they exercise a genuinely different regime
    # from everything above -- M and N far beyond one tile, so the tiling loop runs hundreds of tiles and
    # its tail arithmetic is stressed at real magnitudes rather than at edge+1.
    #
    # K is reduced from the model's real reduction lengths to 32. That is a stated compromise, not an
    # oversight: the in-image scalar reference is O(M*N*K) and on cycle-accurate RTL it, not the kernel,
    # sets the run time -- at the model's K=256 one case alone is hours of Verilator. The tiling, the
    # tails, the pack layout and the readout are all independent of K, and the K sweep above covers K
    # itself (1, 2, 3, 15, 16, 17, 255) including the odd lengths the unrolled reduction peels. What
    # these cases add is the PARALLEL geometry at workload scale.
    Case("workload_ffn_up", 196, 1024, 32, "spectformer's FFN up-projection shape (12 of them in the "
         "model, the joint largest contributor): M=196 is not a multiple of any tile edge, so every "
         "tile column ends in a short row tile", seed=500),
    Case("workload_ffn_down", 196, 256, 32, "the FFN down-projection (12 of them): the same short M "
         "with a narrower N, so the tail lands at a different place in the tiling loop", seed=501),
    Case("workload_attn_proj", 196, 768, 32, "the fused QKV projection (8 of them): N=768 is a whole "
         "number of tiles at every edge, isolating the M tail from the N tail", seed=502),
    Case("workload_im2col", 256, 196, 32, "the patch-embedding contraction: the ONLY routed shape whose "
         "N is not a multiple of the operand alignment, so it is the case that would catch a "
         "right-operand row stride the datapath cannot fetch", seed=503),
    Case("workload_classifier", 1, 1000, 32, "the classifier head: M=1 at a large N, i.e. a vecmat over "
         "many tile columns. The M=1 probes above use a single tile; this one runs the tiling loop with "
         "one live row throughout, which is where a row-index bug in the readout would show", seed=504),
    # The same five contractions at the model's REAL reduction lengths. The K=32 versions above stay --
    # a certified case is never removed -- but they are reduced-depth stand-ins, and a corpus that only
    # ever ran the model's M and N could be cited as "the model's shapes pass" while no case had ever
    # performed the model's reduction. These close that gap.
    #
    # Why they can exist NOW and could not before: the stated blocker was the in-image scalar reference
    # at O(M*N*K) on cycle-accurate RTL. On the FPGA that argument no longer holds -- all five are 180M
    # MACs together, which is seconds, not the hours Verilator would have taken. The compromise was
    # honest when written and is simply obsolete; the cost that justified it was measured away.
    #
    # These are also the cases where a K-major packing bug can finally show. At K=32 the packed left
    # operand is a single tile deep at every edge, so a stride computed from the wrong extent still lands
    # inside the buffer; at K=1024 it does not.
    Case("workload_ffn_up_k256", 196, 1024, 256, "the FFN up-projection at the model's real reduction "
         "depth (12 of these, tied for the largest contributor at 51.4M MACs). The K=32 twin certifies "
         "the parallel geometry; this one certifies that geometry while the reduction actually runs to "
         "the model's length", seed=520),
    Case("workload_ffn_down_k1024", 196, 256, 1024, "the FFN down-projection at its real K -- the "
         "LONGEST reduction in the model. K=1024 is where an accumulator that is re-zeroed, or carried "
         "across a tile boundary it should not be, stops being maskable by a short reduction", seed=521),
    Case("workload_attn_proj_k256", 196, 768, 256, "the fused QKV projection at real depth: N=768 is a "
         "whole number of tiles at every edge, so a failure here is the reduction or the pack, never the "
         "N tail -- which is exactly what makes it diagnostic alongside the two above", seed=522),
    Case("workload_im2col_k768", 256, 196, 768, "patch embedding at real depth. Its N=196 is still not a "
         "multiple of the operand alignment, so this is the one case that exercises an unaligned "
         "right-operand row stride AND a long reduction at the same time", seed=523),
    Case("workload_classifier_k256", 1, 1000, 256, "the classifier head at real depth: M=1 with a "
         "reduction 8x longer than its twin. Measured on this part, the M=1 shape is the one the cost "
         "model should DECLINE to route (row-serial readout charges a full tile of rows for one live "
         "row), so it is here for correctness, not throughput", seed=524),
    # --- a DECODE-shaped whole model, which the workload cases above do not reach ---------------
    # Every workload_* case above comes from spectformer, whose M is 196 or 256 -- one or more whole
    # tiles. A decode shape inverts that: M is the SEQUENCE LENGTH (8), while N and K are the layer
    # widths, so the unit runs a permanently partial row panel across hundreds of column blocks and a
    # reduction up to 36x the corpus's baseline. The nearest existing case, workload_classifier_k256,
    # is narrow-M (1) over 1000 columns but only K=256. Gemma 2 2B at seq 8 routes 183 contractions in
    # this class and its whole-model output came back UNCORRELATED with golden (rank corr 0.006),
    # which is what these cases exist to have caught.
    Case("decode_qkv_proj", 8, 2048, 2304, "a decode-shaped QKV projection: M=8 against a tile edge of "
         "64 is a row panel that never fills, held across 32 column blocks at edge 64 and a reduction "
         "9x the corpus baseline", seed=530),
    Case("decode_ffn_up", 8, 9216, 2304, "the widest decode projection: 144 column blocks at edge 64 "
         "under the same unfilled row panel, so a fault that needs many tile-column iterations to "
         "show up has room to", seed=531),
    Case("decode_ffn_down_k9216", 8, 2304, 9216, "the DEEPEST reduction any model here lowers, at a "
         "partial row panel. K=9216 is also exactly the pad-tail scratch bound, so this is the case "
         "that sits on that boundary rather than comfortably inside it", seed=532),
    Case("decode_ffn_down_bias", 8, 2304, 9216, "the same shape with the bias broadcast init, which "
         "writes every tile row whether or not M fills it -- the init and the readout disagree about "
         "how many rows are live, and only a partial row panel can expose that", bias=True, seed=533),
    Case("decode_ffn_down_requant", 8, 2304, 9216, "the same shape with the full epilogue, which reads "
         "back only the rows M covers", bias=True, requant=True, seed=534),
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


def select(tile: int, *, path: "str | Path | None" = None, tiles_mn: bool = True,
           max_out_elems: int | None = None
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
            skipped.append((got, f"m={m} n={n} exceeds the {tile}x{tile} tile; needs a kernel that "
                                 f"tiles M and N"))
        elif max_out_elems is not None and m * n > int(max_out_elems):
            skipped.append((got, f"output is {m}x{n} = {m * n} int32 elements, over the image's "
                                 f"{int(max_out_elems)}-element result buffers; a harness limit, not a "
                                 f"datapath one"))
        elif k > ACC_OVERFLOW_UNREACHABLE_ABOVE_K:
            skipped.append((got, f"k={k} is at or above the accumulator-overflow bound "
                                 f"{ACC_OVERFLOW_UNREACHABLE_ABOVE_K}, where the datapath's missing "
                                 f"saturation becomes reachable; no model produces such a reduction"))
        else:
            runnable.append(got)
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
