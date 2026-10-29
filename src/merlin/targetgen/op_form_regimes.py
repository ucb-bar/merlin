"""Fact-derived form regimes and conservative witness-shape proposals.

Missing layout, transfer, reduction-axis or array evidence yields UNDERIVABLE.
Coverage is metadata, never evidence of runtime legality or numerical correctness.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from merlin.targetgen import semantic_families as sf

#: A regime whose boundary could not be derived. Not a regime — a recorded absence of one.
UNDERIVABLE = "UNDERIVABLE"

#: The regimes each axis contributes, in the order a corpus should acquire them (cheapest first).
#: Closed per axis: a new regime is a real distinction, not a spelling.
AXIS_REGIMES: dict[str, tuple[str, ...]] = {
    # How full the ragged tile is. The three classes are `conformance.classify_alignment`'s, so the
    # requirement and the thing that measures it cannot each carry their own definition.
    "occupancy": ("sub_tile", "partial", "aligned"),
    # Which operand the schedule will want resident. `square` is the symmetric case; `tall_thin` and
    # `wide` are the two ways M and N can be lopsided past the array edge.
    "aspect": ("square", "tall_thin", "wide"),
    # How many times the accumulator is re-entered. `single_pass` never leaves it; `multi_pass` does.
    "reduction_depth": ("single_pass", "multi_pass"),
    # Whether the live operand working set survives in the store, or must be re-streamed.
    "operand_capacity": ("fits", "spills"),
    # Whether the result set survives in the accumulator across the reduction.
    "accumulator_capacity": ("fits", "spills"),
    # Whether one operand run is one command or several.
    "transfer": ("single", "split"),
    # --- the WINDOW axes, for a contraction that slides one ------------------------------------
    # A convolution is not one cell. Each of these changes the lowering on its own: the kernel decides
    # how many products land in one accumulator pass, the stride decides whether the gather is
    # contiguous, the dilation decides whether it is strided at all, the padding decides whether a
    # border identity is ever computed, and the channel depth decides whether the reduction fills the
    # array. Measured: no capsule on any target declared a non-zero padding, and a fused
    # convolution/max-pool shipped with its padding identity dropped -- wrong only in the border rows
    # that nothing in the corpus computed.
    "window_kernel": ("point", "sub_tile", "tile_or_larger"),
    "window_stride": ("unit", "strided"),
    "window_dilation": ("undilated", "dilated"),
    "window_padding": ("unpadded", "padded"),
    "window_channels": ("sub_tile", "tile_or_deeper"),
}

#: The attribute names a capsule spells its window with — the capsule's own declared vocabulary, read
#: the same structural way ``interface_emit._NAMED_OP_OPERAND_KEYS`` names each op's operands. A capsule
#: declaring none of them has no window and is not scored on these axes.
#:
#: ⚠️ WHICH OPS ARE WINDOWED IS NOT DERIVABLE FROM THE CAPABILITY MANIFEST, and that is a real hole
#: rather than an omission here: :class:`compute_units.SemanticCapability` has ``ranks``, ``layouts``,
#: ``transpose``, ``batch`` and ``arbitrary_mnk``, and no window axis at all. So a target cannot say
#: "my contraction slides a window and these are the strides it takes", and the requirement has to
#: learn windowed-ness from the ops whose capsules declare one. Adding a ``window`` axis to
#: ``SemanticCapability`` is what would close it -- see ``opset_contract.required_op_cells``'s
#: ``axis_basis``, which records the weaker basis rather than presenting it as derived.
WINDOW_KEYS: dict[str, tuple[str, ...]] = {
    "window_kernel": ("kh", "kw"),
    "window_stride": ("stride",),
    "window_dilation": ("dilation",),
    "window_padding": ("padding",),
    "window_channels": ("ci",),
}

#: The window axes, in the order a report should read them.
WINDOW_AXES: tuple[str, ...] = tuple(WINDOW_KEYS)


@dataclass(frozen=True)
class Bound:
    """One hardware number a regime boundary sits on, with where it came from."""

    axis: str
    value: int | None
    source: str
    #: Why the value is absent. Set exactly when ``value`` is None — a bound with neither is a bug.
    reason: str = ""

    def __post_init__(self):
        if self.value is not None and (type(self.value) is not int or self.value <= 0 or not self.source):
            raise ValueError("derived bounds require a positive integer and a source")
        if self.value is None and not self.reason:
            raise ValueError("missing bounds require a reason")

    @property
    def derived(self) -> bool:
        return self.value is not None

    def to_dict(self) -> dict:
        out: dict = {"axis": self.axis, "value": self.value, "source": self.source}
        if self.reason:
            out["reason"] = self.reason
        return out


def boundary_witnesses(bound: Bound) -> dict[str, int | None]:
    """Adjacent positive source probes for an independently derived scalar.

    These comparisons establish arithmetic relations to the selected number.
    A tensor axis, physical capacity or tiling interpretation still needs its
    own evidence. A missing bound or impossible positive lower witness stays
    explicit rather than removing its requested comparison.
    """
    value = bound.value
    if value is None:
        return {"<": None, "=": None, ">": None}
    return {"<": value - 1 if value > 1 else None, "=": value, ">": value + 1}


@dataclass
class FormBounds:
    """Every regime boundary one target states, derived from its own facts."""

    target: str
    array_rows: Bound | None = None
    array_cols: Bound | None = None
    operand_rows: Bound | None = None
    accumulator_rows: Bound | None = None
    dma_max_bytes: Bound | None = None
    element_bits: Bound | None = None
    status: str = "unknown"
    notes: list[str] = field(default_factory=list)

    def bounds(self) -> dict[str, Bound]:
        return {
            name: bound
            for name, bound in (
                ("array_rows", self.array_rows),
                ("array_cols", self.array_cols),
                ("operand_rows", self.operand_rows),
                ("accumulator_rows", self.accumulator_rows),
                ("dma_max_bytes", self.dma_max_bytes),
                ("element_bits", self.element_bits),
            )
            if bound is not None
        }

    @property
    def tile_edge(self) -> int | None:
        """The edge a shape is tiled against. ``cols`` because that is the N the array presents."""
        for bound in (self.array_cols, self.array_rows):
            if bound is not None and bound.derived:
                return int(bound.value or 0) or None
        return None

    def to_dict(self) -> dict:
        return {
            "target": self.target,
            "status": self.status,
            "tile_edge": self.tile_edge,
            "bounds": {name: bound.to_dict() for name, bound in self.bounds().items()},
            "notes": list(self.notes),
        }


def _missing(axis: str, reason: str) -> Bound:
    return Bound(axis=axis, value=None, source="", reason=reason)


def bounds_for_target(
    target: str, *, dtype: str | None = None, facts: dict | None = None, contract: dict | None = None
) -> FormBounds:
    """Derive every regime boundary ``target`` states. Never raises; an absent bound says why.

    ``dtype`` selects the datapath whose element width turns row counts into extents; ``None`` takes
    the store's own declared element width.
    """
    out = FormBounds(target=target)
    try:
        from merlin.targetgen import address_space as AS

        space = AS.derive_address_space(target, facts=facts)
    except Exception as exc:  # noqa: BLE001 -- an underivable address space is reported, never assumed
        why = f"{type(exc).__name__}: {str(exc)[-160:]}"
        out.status = f"unresolvable: {why}"
        for axis in ("array_rows", "array_cols", "operand_rows", "accumulator_rows", "element_bits"):
            setattr(out, axis, _missing(axis, f"address space did not derive ({why})"))
        out.dma_max_bytes = _missing("dma_max_bytes", f"address space did not derive ({why})")
        out.notes.append(
            "no regime boundary could be derived for this target, so its form space is UNKNOWN rather "
            "than one regime per axis; generate the target's RTL facts before reading any cell count"
        )
        return out

    out.status = "resolved"
    src = f"rtl facts arrays[{space.array_name!r}]" if space.array_name else "rtl facts arrays[]"
    for name, value in (("array_rows", space.array_rows), ("array_cols", space.array_cols)):
        setattr(
            out,
            name,
            Bound(name, int(value), src) if value else _missing(name, "the facts declare no compute array"),
        )

    for name, resolver in (("operand_rows", AS.operand_store), ("accumulator_rows", AS.accumulator_store)):
        try:
            resolution = resolver(space, dtype=dtype) if name == "operand_rows" else resolver(space)
        except Exception as exc:  # noqa: BLE001
            setattr(out, name, _missing(name, f"{type(exc).__name__}: {str(exc)[-120:]}"))
            continue
        store = getattr(resolution, "store", None)
        rows = getattr(store, "total_rows", None) if store is not None else None
        if rows:
            setattr(out, name, Bound(name, int(rows), f"address_space.{resolver.__name__} ({resolution.basis})"))
        else:
            setattr(out, name, _missing(name, getattr(resolution, "reason", None) or "no store resolved this role"))
        if name == "operand_rows" and store is not None and out.element_bits is None:
            elems, row_bytes = getattr(store, "row_elems", None), getattr(store, "row_bytes", None)
            if elems and row_bytes:
                bits = AS.element_bits(dtype) if dtype else store.element_bits
                if bits:
                    out.element_bits = Bound("element_bits", bits, "address_space operand format")

    if out.element_bits is None:
        out.element_bits = _missing("element_bits", "no operand store resolved, so no datapath width")

    # The transfer bound is a MANIFEST declaration, not an RTL fact: the payload limit is an ABI
    # property (`memory_model.dma.max_transfer_bytes`, the same field the trace check reads), and a
    # target that does not declare it has not stated one. Absent is absent -- inventing a limit here
    # would mint a `split` regime no hardware asked for.
    try:
        from merlin.targetgen.target_experiment import load_capability_manifest

        selected = load_capability_manifest(target).contract if contract is None else contract
        dma = (selected.get("memory_model") or {}).get("dma") or {}
        limit = dma.get("max_transfer_bytes")
        if limit is not None and (type(limit) is not int or limit <= 0):
            raise ValueError("DMA limit must be positive integer bytes")
        out.dma_max_bytes = (
            Bound("dma_max_bytes", limit, "capability manifest memory_model.dma.max_transfer_bytes")
            if limit
            else _missing("dma_max_bytes", "the manifest declares no memory_model.dma.max_transfer_bytes")
        )
    except Exception as exc:  # noqa: BLE001
        out.dma_max_bytes = _missing("dma_max_bytes", f"manifest unreadable: {type(exc).__name__}")

    for bound in out.bounds().values():
        if not bound.derived:
            out.notes.append(f"{bound.axis}: {bound.reason}")
    return out


#: Which bound each axis needs before its regimes are real distinctions.
_AXIS_BOUND: dict[str, tuple[str, ...]] = {
    "occupancy": ("array_cols",),
    "aspect": ("array_rows", "array_cols"),
    "reduction_depth": ("array_cols",),
    "operand_capacity": ("operand_rows",),
    "accumulator_capacity": ("accumulator_rows",),
    "transfer": ("dma_max_bytes", "element_bits"),
}


@dataclass(frozen=True)
class FormRegime:
    """One named point on one form axis, and the hardware bound that makes it a distinct lowering."""

    axis: str
    regime: str
    bound: str = ""
    value: int | None = None
    reason: str = ""

    def key(self) -> str:
        return f"{self.axis}={self.regime}"

    def to_dict(self) -> dict:
        out = {"axis": self.axis, "regime": self.regime}
        if self.bound:
            out["bound"] = self.bound
        if self.value is not None:
            out["value"] = self.value
        if self.reason:
            out["reason"] = self.reason
        return out


def regimes_for_family(family: str, bounds: FormBounds) -> tuple[FormRegime, ...]:
    """The declared taxonomy for a family, conditioned on available target bounds.

    This does not prove every regime is reachable; synthesis must verify placement.

    An axis whose bound the target does not state yields a single :data:`UNDERIVABLE` regime carrying
    the reason, never its full regime list and never nothing. Emitting the full list would demand cells
    nobody can place; emitting nothing would report a hole as a hardware fact.
    """
    out: list[FormRegime] = []
    for axis in sf.form_axes(family):
        needed = _AXIS_BOUND.get(axis, ())
        missing = [n for n in needed if not (bounds.bounds().get(n) or _missing(n, "not derived")).derived]
        if missing:
            why = "; ".join(
                (bounds.bounds().get(n) or _missing(n, "not derived")).reason or "not derived" for n in missing
            )
            out.append(FormRegime(axis=axis, regime=UNDERIVABLE, bound=",".join(missing), reason=why))
            continue
        primary = bounds.bounds()[needed[0]]
        for regime in AXIS_REGIMES.get(axis, ()):
            out.append(FormRegime(axis=axis, regime=regime, bound=primary.axis, value=primary.value))
    # A SQUARE ARRAY DOES NOT REMOVE THE ASPECT AXIS, and that is worth stating because it looks as if
    # it should. `tall_thin` and `wide` are properties of the PROBLEM, not of the array: on a square
    # array an M x N of 256 x 16 still re-feeds the stationary operand sixteen times where 16 x 256
    # does not. What a square array removes is the asymmetry between the two, not either regime.
    return tuple(out)


# ---------------------------------------------------------------------------------------------------
# where a capsule sits in that space
# ---------------------------------------------------------------------------------------------------


def _extents(capsule: dict) -> tuple[int, ...]:
    """Every operand extent a capsule declares, from its own ``inputs[]`` shapes."""
    out: list[int] = []
    for tensor in capsule.get("inputs") or ():
        if not isinstance(tensor, dict) or tensor.get("role") not in ("input", "weight"):
            continue
        for dim in tensor.get("shape") or ():
            try:
                out.append(int(dim))
            except (TypeError, ValueError):
                continue
    return tuple(out)


def _mkn(capsule: dict) -> tuple[int | None, int | None, int | None]:
    operation = capsule.get("operation") or {}
    attrs = operation.get("attributes") or {}
    if has_window(capsule) or attrs.get("transpose_a") or attrs.get("transpose_b"):
        return None, None, None
    tensors = capsule.get("inputs") or ()
    by_name = {t.get("name"): t for t in tensors if isinstance(t, dict)}

    def operand(key, role):
        if key in attrs:
            return by_name.get(attrs[key])
        values = [t for t in tensors if t.get("role") == role]
        return values[0] if len(values) == 1 else None

    lhs, rhs = operand("lhs", "input"), operand("weight", "weight")
    a, b = (lhs or {}).get("shape"), (rhs or {}).get("shape")
    if not a or not b or len(a) != 2 or len(b) != 2:
        return None, None, None
    if any(type(d) is not int or d <= 0 for d in (*a, *b)):
        return None, None, None
    if "contraction" in sf.primitives_of(sf.from_op(operation.get("op")) or ""):
        return (a[0], a[1], b[1]) if a[1] == b[0] else (None, None, None)
    # A reduction's axis is not implied by rank or family membership.
    axis = attrs.get("axis")
    if axis in (-1, 1) and a == b:
        return a[0], None, a[1]
    return None, None, None


def classify_capsule(capsule: dict, bounds: FormBounds, *, family: str | None = None) -> dict:
    """``axis -> regime`` for where one capsule sits in its family's form space.

    An axis the capsule's own declaration cannot place it on is reported ``UNDERIVABLE`` for that
    capsule, not omitted: a capsule that cannot be placed is not a capsule that covers everything, and
    dropping it from the join would let a corpus look denser than it is.
    """
    fam = family or sf.from_op((capsule.get("operation") or {}).get("op")) or ""
    axes = sf.form_axes(fam)
    edge = bounds.tile_edge
    if not bounds.array_rows or not bounds.array_cols or bounds.array_rows.value != bounds.array_cols.value:
        edge = None
    out: dict[str, str] = {}
    extents = _extents(capsule)
    m, k, n = _mkn(capsule)

    for axis in axes:
        if axis == "occupancy":
            from merlin.targetgen.conformance import classify_alignment

            out[axis] = classify_alignment(extents, edge or 0) or UNDERIVABLE
        elif axis == "aspect":
            if m and n and edge:
                # Lopsided is measured against the ARRAY EDGE, not against a ratio constant: one side
                # needs at least two tiles more than the other before the schedule changes at all.
                if m >= n + edge:
                    out[axis] = "tall_thin"
                elif n >= m + edge:
                    out[axis] = "wide"
                else:
                    out[axis] = "square"
            else:
                out[axis] = UNDERIVABLE
        elif axis == "reduction_depth":
            # THE AXIS THE FAMILY ACTUALLY REDUCES OVER. A contraction's depth is K; a reduce-over-axis
            # family has no K at all, and reading one reported every shape as `single_pass` no matter
            # how deep its reduction was.
            depth = k if reduced_axis(fam) == "k" else n
            out[axis] = ("multi_pass" if depth > edge else "single_pass") if (depth and edge) else UNDERIVABLE
        elif axis in ("operand_capacity", "accumulator_capacity"):
            out[axis] = _capacity_regime(capsule, bounds, axis)
        elif axis == "transfer":
            out[axis] = _transfer_regime(capsule, bounds)
    out.update(window_regimes(capsule, bounds))
    return out


def _ints(value) -> tuple[int, ...]:
    """Every integer in a declared window value, whether it is a scalar or a per-axis list."""
    if isinstance(value, (list, tuple)):
        out: list[int] = []
        for item in value:
            out.extend(_ints(item))
        return tuple(out)
    try:
        return (int(value),)
    except (TypeError, ValueError):
        return ()


def has_window(capsule: dict) -> bool:
    """Does this capsule declare a sliding window at all? Structural: does it spell any window key."""
    attrs = (capsule.get("operation") or {}).get("attributes") or {}
    return any(key in attrs for keys in WINDOW_KEYS.values() for key in keys)


def window_regimes(capsule: dict, bounds: FormBounds) -> dict[str, str]:
    """``window axis -> regime`` for a capsule that declares a window; ``{}`` for one that does not.

    An axis the capsule leaves unstated is ``UNDERIVABLE`` rather than assumed at its identity value.
    "This convolution is unpadded" and "this convolution does not say" are different claims, and the
    padding defect above is exactly what the second one hides.
    """
    if not has_window(capsule):
        return {}
    attrs = (capsule.get("operation") or {}).get("attributes") or {}
    edge = bounds.tile_edge
    if not bounds.array_rows or not bounds.array_cols or bounds.array_rows.value != bounds.array_cols.value:
        edge = None
    out: dict[str, str] = {}
    for axis, keys in WINDOW_KEYS.items():
        values: list[int] = []
        stated = False
        for key in keys:
            if key in attrs:
                stated = True
                values.extend(_ints(attrs[key]))
        if not stated or not values:
            out[axis] = UNDERIVABLE
            continue
        peak = max(values)
        if axis == "window_kernel":
            if not edge:
                out[axis] = UNDERIVABLE
            elif peak <= 1:
                out[axis] = "point"
            elif peak < edge:
                out[axis] = "sub_tile"
            else:
                out[axis] = "tile_or_larger"
        elif axis == "window_stride":
            out[axis] = "unit" if peak <= 1 else "strided"
        elif axis == "window_dilation":
            out[axis] = "undilated" if peak <= 1 else "dilated"
        elif axis == "window_padding":
            out[axis] = "unpadded" if peak <= 0 else "padded"
        elif axis == "window_channels":
            out[axis] = UNDERIVABLE if not edge else ("sub_tile" if peak < edge else "tile_or_deeper")
    return out


def _capacity_regime(capsule: dict, bounds: FormBounds, axis: str) -> str:
    """Require a declared simultaneous working-set row count from layout analysis.

    Input extents do not specify banking, aliasing, residency or accumulator
    layout. Counting inputs as accumulator storage is particularly unsound.
    """
    quantity = "operand_rows" if axis == "operand_capacity" else "accumulator_rows"
    bound = bounds.bounds().get(quantity)
    evidence = (capsule.get("form_evidence") or {}).get(quantity)
    if bound is None or not bound.derived or not isinstance(evidence, dict):
        return UNDERIVABLE
    rows = evidence.get("value")
    if type(rows) is not int or rows < 0 or not evidence.get("source"):
        return UNDERIVABLE
    return "fits" if rows <= bound.value else "spills"


# ---------------------------------------------------------------------------------------------------
# a SMALL shape that lands in a regime
# ---------------------------------------------------------------------------------------------------


def _round_up(value: int, step: int) -> int:
    return ((int(value) + step - 1) // step) * step


def reduced_axis(family: str) -> str:
    """Which extent a family reduces over — ``"k"`` for a reduce-over-k, ``"n"`` for a reduce-over-axis.

    A contraction's depth is K, which its operands carry between them. Everything else that reduces
    does so along its own last axis, and its operands are two same-shaped ``[M, N]`` tensors with no K
    at all -- so reading K on one of those asks about an extent the capsule does not have. That is how a
    normalization's reduction-depth cell came back ``single_pass`` for every shape the chooser proposed:
    the chooser grew K and the classifier read a K the capsule never declared.
    """
    return "k" if "contraction" in sf.primitives_of(family) else "n"


def operand_rows_grow_with(family: str) -> str:
    """Which extent grows the live operand set most cheaply for ``family``.

    For a contraction, K: both operands carry it, so rows rise as ``2K`` while MACs rise as ``K`` alone.
    For a family whose operands are two ``[M, N]`` tensors there is no K, and M is the only axis that
    grows rows without also multiplying the output.
    """
    return "k" if "contraction" in sf.primitives_of(family) else "m"


def smallest_shape_for(axis: str, regime: str, bounds: FormBounds, *, family: str = "contraction") -> dict | None:
    """A small candidate ``(m, k, n)`` that lands on ``axis`` in ``regime``; ``None`` when no bound places it.

    SMALLEST, NOT EXTREME, and this is a cost rule with a measured reason. A regime is entered the
    moment its boundary is crossed: a capsule whose working set exceeds the operand store by one row
    exercises the spill path exactly as one that exceeds it sixteen-fold, and costs a thousandth of the
    certification time. The corpus already carries the other kind -- five capsules consume 73% of all L3
    cycles, and the worst is a ``[16384, 16] x [16, 16384]`` reduction whose K is an order of magnitude
    past anything a real model contains and 32x the largest model-derived shape.

    So every regime is entered by the minimum margin the boundary allows, and the boundary is the
    hardware's own number. Where two axes can both be satisfied by growing different extents, the one
    that grows the MAC count least is chosen -- a deep K is cheaper to spill the operand store with than
    a tall M, because M multiplies against N and K does not.
    """
    edge = bounds.tile_edge
    if not edge or not bounds.array_rows or bounds.array_rows.value != edge:
        return None
    if axis in ("operand_capacity", "accumulator_capacity", "transfer"):
        return None
    base = {"m": edge, "k": edge, "n": edge}

    if axis == "occupancy":
        if regime == "aligned":
            return base
        # `classify_alignment` takes the MIN nonzero remainder over every declared extent, so only the
        # intended axis may be ragged; the others stay whole tiles or the verdict is decided by them.
        if regime == "partial":  # ragged tile nearly full: remainder > edge//2
            return {**base, "m": edge + (edge - 1)}
        if regime == "sub_tile":  # ragged tile barely occupied: remainder <= edge//2
            return {**base, "m": edge + max(1, edge // 4)}
        return None

    if axis == "aspect":
        if regime == "square":
            return base
        if regime == "tall_thin":  # M at least one whole tile beyond N
            return {**base, "m": edge * 2, "n": edge}
        if regime == "wide":
            return {**base, "m": edge, "n": edge * 2}
        return None

    if axis == "reduction_depth":
        if regime == "single_pass":
            return base
        if regime == "multi_pass":  # one accumulation step past the array's own depth
            return {**base, reduced_axis(family): edge * 2}
        return None

    return None


def smallest_window_for(axis: str, regime: str, bounds: FormBounds) -> dict | None:
    """A small window attributes that land on one window axis in ``regime``.

    Every one of these is free in MAC terms -- a padded convolution costs no more than an unpadded one
    of the same extents -- so "smallest" here means the identity value on every axis but the one being
    exercised, which keeps each capsule's verdict attributable to a single change.
    """
    edge = bounds.tile_edge
    if not edge:
        return None
    base = {"kh": 1, "kw": 1, "stride": [1, 1], "padding": [0, 0, 0, 0], "dilation": [1, 1], "ci": edge}
    if axis == "window_kernel":
        if regime == "point":
            return {**base, "kh": 1, "kw": 1}
        if regime == "sub_tile":
            return {**base, "kh": 2, "kw": 2} if edge > 2 else None
        if regime == "tile_or_larger":
            return {**base, "kh": edge, "kw": edge}
        return None
    if axis == "window_stride":
        return base if regime == "unit" else ({**base, "stride": [2, 2]} if regime == "strided" else None)
    if axis == "window_dilation":
        return base if regime == "undilated" else ({**base, "dilation": [2, 2]} if regime == "dilated" else None)
    if axis == "window_padding":
        return base if regime == "unpadded" else ({**base, "padding": [1, 1, 1, 1]} if regime == "padded" else None)
    if axis == "window_channels":
        if regime == "sub_tile":
            return {**base, "ci": max(1, edge // 2)}
        if regime == "tile_or_deeper":
            return {**base, "ci": edge}
        return None
    return None


def _transfer_regime(capsule: dict, bounds: FormBounds) -> str:
    """Price declared contiguous requests, not a guessed tensor-axis length."""
    limit = bounds.dma_max_bytes
    evidence = (capsule.get("form_evidence") or {}).get("max_contiguous_bytes")
    if limit is None or not limit.derived or not isinstance(evidence, dict):
        return UNDERIVABLE
    size = evidence.get("value")
    if type(size) is not int or size <= 0 or not evidence.get("source"):
        return UNDERIVABLE
    return "single" if size <= limit.value else "split"
