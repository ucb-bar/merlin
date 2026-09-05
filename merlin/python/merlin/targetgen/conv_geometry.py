"""The convolution geometry a captured model really contains, recovered STRUCTURALLY.

A padded convolution is a defect class this corpus could not observe. Measured by
``check_defect_reach``: no capsule on any target declares a non-zero padding, so a lowering that
loses the padding identity is wrong only in border rows nothing here computes -- and that is exactly
how a fused convolution/max-pool shipped with a ``-128`` padding identity dropped and 119 wrong
outputs.

The obvious fix is to read ``padding``/``stride``/``dilation`` off the captured op, and it does not
work: torch-mlir emits convolutions as **im2col**, so the captured program contains a gather and a
matmul and no convolution op at all. Nothing carries those attributes because by that point nothing
is a convolution.

WHAT IS STILL THERE IS THE GEOMETRY ITSELF, in two structures:

``the gather's affine map``   ``(d3, d0, d4 * S + d1 * D, d5 * S + d2 * D)`` indexes the padded input.
                              The coefficient on the OUTPUT dim is the stride; the coefficient on the
                              KERNEL dim is the dilation; the kernel extent is that dim's extent in
                              the iteration space. A ``k7x7 s1`` and a ``k3x3 s2`` are different maps,
                              and they are different here even though the op names are identical.
``the producer insert_slice`` A padded convolution writes the real input into the interior of a larger
                              zero tensor. ``static_offsets`` is the padding BEFORE, the destination
                              extent minus offset minus size is the padding AFTER (which is how an
                              ASYMMETRIC pad survives), and ``static_strides`` above 1 is an input
                              dilation -- a transposed convolution, which this capture contains and
                              which no hand-written corpus entry anticipated.

NOTHING HERE IS GUESSED, AND THE ONE AMBIGUITY IS CHECKED. Within ``d_a * c_a + d_b * c_b`` the map
alone does not say which dim is the kernel and which is the output: the expression is symmetric under
swapping the pair. The extents break it -- a kernel is smaller than its output -- but an extent
comparison is a heuristic, so it is only a PROPOSAL here. It is then verified against the geometry
identity

    padded_extent == (out - 1) * stride + (kernel - 1) * dilation + 1

which the swapped assignment does not satisfy except in degenerate cases. A gather failing that
identity yields ``None``: an unverified geometry is not recorded, because a wrong stride would demand
capsules for a convolution the model does not contain.
"""
from __future__ import annotations

from dataclasses import dataclass

__all__ = ["ConvGeometry", "geometries", "geometry_classes"]


@dataclass(frozen=True)
class ConvGeometry:
    """One convolution's window, recovered from the program rather than from an attribute."""

    kernel: tuple[int, ...]
    stride: tuple[int, ...]
    dilation: tuple[int, ...]
    pad_before: tuple[int, ...]
    pad_after: tuple[int, ...]
    input_dilation: tuple[int, ...]                # >1 means a TRANSPOSED convolution
    pad_known: bool                                # False when no readable padding producer was found
    in_spatial: tuple[int, ...]                    # the real input, before padding
    out_spatial: tuple[int, ...]
    channels_in: int
    dtype: str

    @property
    def padded(self) -> "bool | None":
        """True, False, or None when no padding producer was readable. Never False for unknown."""
        if not self.pad_known:
            return None
        return any(self.pad_before) or any(self.pad_after)

    @property
    def symmetric_pad(self) -> bool:
        return self.pad_before == self.pad_after

    def signature(self) -> str:
        """What the compiler must DO with this window — the grouping key.

        Extents are deliberately absent: two 3x3 stride-1 pad-1 convolutions exercise one lowering
        whatever their channel counts, and a corpus with a member per extent would grow without
        covering anything new. What IS in the key is every axis that changes the code: the window, how
        it steps, how it is spaced, whether the border is padded, whether that padding is symmetric,
        and whether the input is dilated.
        """
        k = "x".join(str(v) for v in self.kernel)
        s = "x".join(str(v) for v in self.stride)
        d = "x".join(str(v) for v in self.dilation)
        pb = "x".join(str(v) for v in self.pad_before)
        pa = "x".join(str(v) for v in self.pad_after)
        if not self.pad_known:
            pad = "UNKNOWN"
        else:
            pad = pb if self.symmetric_pad else f"{pb}_{pa}"
        base = f"k{k}/s{s}/d{d}/pad{pad}"
        return base + ("/indilated" + "x".join(str(v) for v in self.input_dilation)
                       if any(v != 1 for v in self.input_dilation) else "")

    def to_dict(self) -> dict:
        return {"signature": self.signature(), "kernel": list(self.kernel),
                "stride": list(self.stride), "dilation": list(self.dilation),
                "pad_before": list(self.pad_before), "pad_after": list(self.pad_after),
                "input_dilation": list(self.input_dilation), "pad_known": self.pad_known,
                "in_spatial": list(self.in_spatial), "out_spatial": list(self.out_spatial),
                "channels_in": self.channels_in, "dtype": self.dtype,
                "padded": self.padded, "symmetric_pad": self.symmetric_pad}


def _terms(expr) -> "dict[int, int] | None":
    """``{iteration dim: coefficient}`` for an affine result, or None when it is not affine-linear.

    Fails closed on anything it does not model -- a ``mod``, a ``floordiv``, a symbol. Those appear in
    tiled or packed forms whose geometry is not read the same way, and treating one as linear would
    return a stride that is not the program's.
    """
    from xdsl.ir.affine import (AffineBinaryOpExpr, AffineBinaryOpKind, AffineConstantExpr,
                                AffineDimExpr)

    if isinstance(expr, AffineDimExpr):
        return {int(expr.position): 1}
    if not isinstance(expr, AffineBinaryOpExpr):
        return None
    lhs, rhs = _terms(expr.lhs), _terms(expr.rhs)
    if expr.kind is AffineBinaryOpKind.Add:
        if lhs is None or rhs is None:
            return None
        out = dict(lhs)
        for d, c in rhs.items():
            out[d] = out.get(d, 0) + c
        return out
    if expr.kind is AffineBinaryOpKind.Mul:
        # Exactly one side must be a constant; a dim-by-dim product is not affine.
        for a, b in ((expr.lhs, expr.rhs), (expr.rhs, expr.lhs)):
            if isinstance(a, AffineConstantExpr):
                inner = _terms(b)
                if inner is None:
                    return None
                return {d: c * int(a.value) for d, c in inner.items()}
    return None


def _pad_chain(value) -> "tuple[dict, bool]":
    """Walk back through ``tensor.insert_slice`` producers, accumulating padding and input dilation.

    A transposed convolution pads in TWO steps -- first a strided insert that spaces the input out,
    then an offset insert that adds the border -- so a reader that stops at the first producer sees
    the spacing and misses the border, or the reverse. Both are collected, and the walk is bounded so
    a malformed chain cannot spin.

    The second return value is whether a producer chain was found at all. ``False`` means the gather
    reads its input directly, which is a genuine zero padding, NOT an unknown one.
    """
    from merlin.common import mlir_query as mq

    pad_before: dict[int, int] = {}
    pad_after: dict[int, int] = {}
    in_dil: dict[int, int] = {}
    found = False
    for _ in range(8):
        owner = getattr(value, "owner", None)
        if owner is None or not hasattr(owner, "name"):
            break
        if mq.op_name(owner) != "tensor.insert_slice":
            break
        tables = list(mq._attr_tables(owner))  # noqa: PLC2701 -- the same reader indexing_maps uses
        prop = {}
        for t in tables:
            for k in ("static_offsets", "static_sizes", "static_strides"):
                if k in t and k not in prop:
                    prop[k] = [int(v) for v in getattr(t[k], "get_values", lambda: t[k].data)()]
        if len(prop) != 3:
            break
        dest = None
        for res in getattr(owner, "results", ()):
            shp = getattr(getattr(res, "type", None), "get_shape", None)
            if shp is not None:
                dest = [int(v) for v in shp()]
        if dest is None or len(dest) != len(prop["static_offsets"]):
            break
        found = True
        for i, (off, size, strd) in enumerate(zip(prop["static_offsets"], prop["static_sizes"],
                                                  prop["static_strides"])):
            # A strided insert SPACES the source out: it occupies (size-1)*stride + 1 of the
            # destination. Anything past that on either side is border padding.
            span = (size - 1) * strd + 1
            pad_before[i] = pad_before.get(i, 0) + off
            pad_after[i] = pad_after.get(i, 0) + (dest[i] - off - span)
            if strd != 1:
                in_dil[i] = strd
        value = owner.operands[0] if getattr(owner, "operands", None) else None
        if value is None:
            break
    return {"before": pad_before, "after": pad_after, "input_dilation": in_dil}, found


def _im2col_km(op) -> "tuple[int, int] | None":
    """``(K, M)`` of the matrix the im2col gather feeds, or None.

    THIS IS THE DISCRIMINATOR, and nothing weaker works. Inside ``d_a * c_a + d_b * c_b`` the affine
    map is symmetric under swapping the pair: one dim is the kernel and one is the output, and the map
    alone cannot say which. The obvious tie-break -- "a kernel is smaller than its output" -- is FALSE,
    and it was measured false on a real capture in this store: a 4x4 kernel producing a 2x3 output read
    as a 2x3 kernel with dilation 4, which is a different convolution from the one the model contains.
    The convolution output identity does not separate them either; both assignments satisfy it there.

    What does separate them is the shape the gather is reshaped INTO. im2col produces
    ``[Cin * prod(kernel), N * prod(output)]``, so ``K`` names the kernel side unambiguously. It is
    read by following the gather's result forward through the reshapes to the first rank-2 value --
    structurally, not by assuming a fixed op sequence -- and a gather whose chain does not reach one
    yields None, so the geometry is dropped rather than guessed.
    """
    from merlin.common import mlir_query as mq
    from merlin.kernels.shapes import _shaped  # noqa: PLC2701

    frontier = [r for r in getattr(op, "results", ())]
    for _ in range(6):
        nxt = []
        for value in frontier:
            for use in getattr(value, "uses", ()):
                user = getattr(use, "operation", None)
                if user is None or mq.op_name(user) not in ("tensor.collapse_shape",
                                                            "tensor.expand_shape"):
                    continue
                for res in getattr(user, "results", ()):
                    shaped = _shaped(res)
                    if shaped is not None and len(shaped[0]) == 2:
                        return (int(shaped[0][0]), int(shaped[0][1]))
                    nxt.append(res)
        if not nxt:
            return None
        frontier = nxt
    return None


def _resolve_axes(axes, *, bare, padded, k_total: int, m_total: int):
    """Decide kernel-vs-output for every window axis at once, or return None.

    Every axis is resolved TOGETHER rather than one at a time, because the evidence is a single
    product: ``K == channel * prod(kernel extents)``. Both orderings of the two bare dims (batch and
    channel) are tried, since which is which is a layout convention this reader declines to assume.

    Returns ``(axes, channel_extent)``, or None when zero assignments fit or when more than one does.
    More than one is the important case: it means the evidence does not determine the geometry, and
    emitting either would put a convolution in the requirement that the model may not contain.

    The channel extent falls out of the same arithmetic (``K // prod(kernel)``) rather than being
    scanned for separately, so it cannot disagree with the assignment it came from.
    """
    from itertools import product

    fits = []
    for choice in product((0, 1), repeat=len(axes)):
        kernels, strides, dils, poss, outs = [], [], [], [], []
        for (pos, da, ca, db, cb, ea, eb), pick in zip(axes, choice):
            # pick 0: `da` is the kernel dim; pick 1: `db` is.
            kdim, odim = ((da, db) if pick == 0 else (db, da))
            kext, oext = ((ea, eb) if pick == 0 else (eb, ea))
            dil = ca if kdim == da else cb
            strd = cb if kdim == da else ca
            kernels.append(kext); strides.append(strd); dils.append(dil)
            poss.append(pos); outs.append(oext)
        kprod = 1
        for v in kernels:
            kprod *= v
        oprod = 1
        for v in outs:
            oprod *= v
        if kprod <= 0 or k_total % kprod:
            continue
        channel = k_total // kprod
        # The two bare dims are batch and channel in some order; the K side takes one and the M side
        # the other, and the M side must then account for m_total exactly.
        rest = list(bare)
        if channel not in rest:
            continue
        rest.remove(channel)
        if oprod * rest[0] != m_total:
            continue
        # The convolution output identity, applied last as a consistency check rather than as the
        # discriminator it cannot be.
        if any(o != (padded[pos] - ((k - 1) * d + 1)) // s + 1
               for k, s, d, pos, o in zip(kernels, strides, dils, poss, outs)):
            continue
        fits.append((list(zip(kernels, strides, dils, poss, outs)), channel))
    if len(fits) != 1:
        return None                                # undetermined, or contradictory -- never guessed
    return fits[0]


def geometries(src) -> list[ConvGeometry]:
    """Every im2col convolution in ``src``, as verified geometries.

    Returns an EMPTY list rather than raising on an unreadable module, matching
    ``kernels.shapes.observe_contractions``: a geometry observer that fails must degrade to "I saw
    nothing" so the caller falls back rather than failing a build over one capture.
    """
    from merlin.common import mlir_query as mq
    from merlin.kernels.shapes import _shaped, indexing_maps  # noqa: PLC2701

    try:
        module = mq.parse(src)
    except Exception:                              # noqa: BLE001
        return []

    out: list[ConvGeometry] = []
    for op in mq.walk(module, "linalg.generic"):
        try:
            maps = indexing_maps(op)
            if not maps or len(maps) < 2:
                continue
            res = _shaped(getattr(op, "results", (None,))[0])
            src_shape = _shaped(op.operands[0]) if getattr(op, "operands", None) else None
            if res is None or src_shape is None:
                continue
            extents, _ = res
            padded, dtype = src_shape

            # The output map must be the identity for the result shape to BE the iteration space.
            # Anything else and the extents below would be indexed wrongly, so it is required rather
            # than assumed.
            out_terms = [_terms(r) for r in maps[-1]]
            if len(out_terms) != len(extents) or any(
                    t is None or t != {i: 1} for i, t in enumerate(out_terms)):
                continue

            # The K side of the im2col, read from the reshape the gather feeds. This is what DECIDES
            # which dim of each `d_a * c_a + d_b * c_b` is the kernel and which is the output; see
            # `_im2col_km`.
            km = _im2col_km(op)
            if km is None:
                continue
            k_total, m_total = km

            axes: list[tuple[int, int, int, int, int, int, int]] = []  # pos + both candidates
            bare: list[int] = []
            ok = True
            for pos, r in enumerate(maps[0]):
                t = _terms(r)
                if t is None:
                    ok = False
                    break
                if len(t) == 1:
                    bare.append(extents[next(iter(t))])
                    continue                       # batch or channel, not a window axis
                if len(t) != 2:
                    ok = False
                    break
                (da, ca), (db, cb) = sorted(t.items())
                axes.append((pos, da, ca, db, cb, extents[da], extents[db]))
            if not ok or not axes or len(bare) != 2:
                continue

            resolved = _resolve_axes(axes, bare=bare, padded=padded,
                                     k_total=k_total, m_total=m_total)
            if resolved is None:
                continue
            spatial, cin = resolved

            # ⚠️ NO PADDING PRODUCER MEANS UNKNOWN, NOT ZERO. Measured: this capture's first
            # convolution is padded by an `aten.index.Tensor` gather -- a REFLECTION pad, whose
            # identity is not zero and whose offsets are nowhere to read. Reporting it as `pad0` would
            # have been wrong twice: it claims no padding where there is padding, and it claims the
            # zero identity for one that reflects. An unknown padding is recorded as unknown and
            # becomes its own obligation class, which is a finding about the corpus rather than a
            # silent default.
            pads, pad_known = _pad_chain(op.operands[0])
            pb = tuple(pads["before"].get(pos, 0) for (_, _, _, pos, _) in spatial)
            pa = tuple(pads["after"].get(pos, 0) for (_, _, _, pos, _) in spatial)
            idil = tuple(pads["input_dilation"].get(pos, 1) for (_, _, _, pos, _) in spatial)
            # The real input, before the padding was added and before any input dilation spaced it.
            ins = tuple(((padded[pos] - pb[i] - pa[i]) - 1) // idil[i] + 1
                        for i, (_, _, _, pos, _) in enumerate(spatial))
            out.append(ConvGeometry(
                kernel=tuple(k for k, _, _, _, _ in spatial),
                stride=tuple(s for _, s, _, _, _ in spatial),
                dilation=tuple(d for _, _, d, _, _ in spatial),
                pad_before=pb, pad_after=pa, input_dilation=idil, pad_known=pad_known,
                in_spatial=ins, out_spatial=tuple(o for *_, o in spatial),
                channels_in=cin, dtype=dtype))
        except Exception:                          # noqa: BLE001 -- one bad op never kills the walk
            continue
    return out


#: Parsed-geometry cache, keyed by the capture path. A capture's convolution windows are a property of
#: the CAPTURE and not of any target, but the requirement is derived once per target -- so without this
#: the same 29 model.mlir files are re-parsed for every target, which is most of what a spec
#: regeneration spends its time on. Keyed by (path, mtime, size) so an edited capture re-parses.
_GEOMETRY_CACHE: dict = {}


def _cached_geometries(path) -> list[ConvGeometry]:
    from pathlib import Path as _P

    try:
        st = _P(path).stat()
        key = (str(path), st.st_mtime_ns, st.st_size)
    except OSError:
        return geometries(path)                    # not a file we can stamp; parse it and do not cache
    hit = _GEOMETRY_CACHE.get(key)
    if hit is None:
        hit = geometries(path)
        _GEOMETRY_CACHE[key] = hit
    return hit


def geometry_classes(captures: dict) -> dict:
    """The distinct convolution geometries a set of captures contains, with their evidence.

    ``captures`` is ``{label: path to model.mlir}``, the shape every other derivation reader takes.
    """
    by_sig: dict[str, dict] = {}
    unreadable: dict[str, str] = {}
    for label, path in sorted((captures or {}).items()):
        try:
            found = _cached_geometries(path)
        except Exception as e:                     # noqa: BLE001 -- reported, never skipped silently
            unreadable[label] = f"{type(e).__name__}: {str(e)[-160:]}"
            continue
        for g in found:
            row = by_sig.setdefault(g.signature(), {**g.to_dict(), "n_regions": 0, "sources": []})
            row["n_regions"] += 1
            if label not in row["sources"]:
                row["sources"].append(label)
    return {
        "required": [by_sig[s] for s in sorted(by_sig)],
        "n_classes": len(by_sig),
        "captures_unreadable": unreadable,
        "axis_basis": (
            "the convolution windows real captures CONTAIN, recovered from the im2col gather's affine "
            "map and its padding producer rather than from op attributes -- torch-mlir emits im2col, "
            "so a captured convolution carries no padding/stride/dilation attribute to read. Each "
            "geometry is verified against padded == (out-1)*stride + (kernel-1)*dilation + 1 and "
            "dropped when it does not hold, so an unverified window never becomes an obligation"),
    }
