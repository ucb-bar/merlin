"""Seeded shape-perturbed variants of every unique device kernel, stored and judged as sealed cases.

A bundle's recorded calls replay each kernel on its model's own shapes, and a sealed input changes
values, never shapes: a compiler that special-cased exactly the shapes it was shown would pass both.
This module states, for every unique kernel entry of every program, a few VARIANTS that keep the
kernel's FORM -- op, epilogue stages, window (kernel size, stride, padding, dilation), stationary
operand, operand/accumulator/output dtypes -- and change only its SIZES. Each variant is one of four
classes whose boundaries are derived from the target's own facts (array edge, operand store and
accumulator store, :func:`limits`); nothing here assumes an array size or a store capacity:

``sub_tile``
    every non-unit extent below one array edge (a convolution: its channels and its output pixels);
``aspect``
    the kernel's own extents, each moved by a non-zero offset smaller than one array edge, then scaled
    down together until the variant fits the work caps;
``accumulator_spill``
    the result holds more elements than the accumulator store, by a factor drawn from
    :data:`SPILL_RATIO`;
``operand_spill``
    the live operands (activation + stationary operand) hold more elements than the operand store, by
    a factor drawn from :data:`SPILL_RATIO`.

Rules every class keeps: a non-unit extent is never a multiple of the array edge and never the kernel's
own extent; a unit extent stays unit (a GEMV's single row is part of its form); a convolution's input
is never square and its padding stays smaller than its input; a batched form's batch count changes too.
Every variant does at most :data:`MAX_MACS` multiply-accumulates over at most
:data:`MAX_OPERAND_ELEMENTS` operand elements, and no exact accumulator can overflow on its operands.
``acc_scale`` moves by a relative step in :data:`SCALE_STEP`, rounded to :data:`SCALE_MANTISSA_BITS`
significant bits, so float32, the interface text and the reference all hold the same number exactly.
Operands are seeded integers over the operand dtype's whole range, with both extremes present (a bias
over an accumulator-sized range); a narrowing readout shrinks the weight range so a typical result is
not saturated. Expected results are :func:`merlin.model_check.kernel_reference.call_outputs`.

Everything is a function of ``(seed, interface sha256, class, attempt)``, so the order kernels are
visited in changes nothing. The rule (this docstring and the constants below) is public; the seed and
every variant live only in the private bundle (contract §8)::

    <bundle>/<model>/sealed_variants/blobs/<sha256>.npy
    <bundle>/<model>/sealed_variants/<program>/variants.json   entries, interfaces, call buffers, regimes
    <bundle>/<model>/sealed_variants/<program>/calls.json      calls.v1: one call per variant
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from merlin.model_check import catalog as CAT
from merlin.model_check import kernel_reference as KR

VARIANTS_SCHEMA = "merlin.model_check.variants.v1"
RULE_VERSION = "1"
#: The directory (under ``<bundle>/<model>/``) and the case-label prefix of the variants.
SOURCE = "sealed_variants"
CLASSES = ("sub_tile", "aspect", "accumulator_spill", "operand_spill")

MAX_MACS = 1 << 24
MAX_OPERAND_ELEMENTS = 1 << 22
#: A spill variant exceeds its store by this factor (low, high): just past the boundary, never extreme.
SPILL_RATIO = (1.25, 2.0)
#: |relative change| of a declared ``acc_scale`` (low, high); the sign is drawn.
SCALE_STEP = (0.005, 0.03)
SCALE_MANTISSA_BITS = 12
#: Elements of each full-range activation set to each of the dtype's two extremes.
EXTREME_ELEMENTS = 4
ATTEMPTS = 32

CONTRACTION_OPS = frozenset({"matmul", "linear", "fused_matmul_bias"})
BATCHED_OPS = frozenset({"gemv_batched"})
WINDOW_OPS = frozenset({"conv2d"})


class VariantError(RuntimeError):
    """A variant set cannot be derived (facts missing) or a stored one is malformed."""


# ------------------------------------------------------------------------------------------- limits


@dataclass(frozen=True)
class Limits:
    """The target facts the classes' boundaries sit on, each with where it came from."""

    edge: int
    operand_elements: int
    accumulator_elements: int
    sources: dict[str, str]

    def record(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def limits(target: str, operand_dtype: str | None) -> Limits:
    """Array edge, operand-store and accumulator-store capacity (in elements) from ``target``'s facts.

    Fails closed: a bound the facts do not state is an error, never a default.
    """
    from merlin.targetgen import address_space as AS
    from merlin.targetgen.op_form_regimes import bounds_for_target

    bounds = bounds_for_target(target, dtype=operand_dtype)
    edge = bounds.tile_edge
    if not edge or edge < 3:
        raise VariantError(f"{target}: no usable array edge in the facts ({bounds.notes})")
    space = AS.derive_address_space(target)
    capacity: dict[str, tuple[int, str]] = {}
    for role, resolution in (
        ("operand", AS.operand_store(space, dtype=operand_dtype)),
        ("accumulator", AS.accumulator_store(space)),
    ):
        store = resolution.store
        rows, elems = getattr(store, "total_rows", None), getattr(store, "row_elems", None)
        if not rows or not elems:
            raise VariantError(f"{target}: no {role} store capacity in the facts ({resolution.reason})")
        capacity[role] = (int(rows) * int(elems), f"address_space.{role}_store {store.name} ({resolution.basis})")
    return Limits(
        edge=int(edge),
        operand_elements=capacity["operand"][0],
        accumulator_elements=capacity["accumulator"][0],
        sources={
            "edge": bounds.array_cols.source if bounds.array_cols else "",
            "operand_elements": capacity["operand"][1],
            "accumulator_elements": capacity["accumulator"][1],
        },
    )


# ------------------------------------------------------------------------------------- draw helpers


def _rng(seed: int, *parts: Any) -> np.random.Generator:
    text = "|".join([RULE_VERSION, str(int(seed)), *(str(p) for p in parts)])
    return np.random.default_rng(int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little"))


def _off_edge(value: int, edge: int) -> int:
    value = max(2, int(value))
    return value + 1 if value % edge == 0 else value


def _between(rng: np.random.Generator, lo: int, hi: int, edge: int, avoid: int) -> int | None:
    """A uniform integer of ``[lo, hi]`` off the array edge and not ``avoid``; ``None`` if there is none."""
    choices = [v for v in range(max(2, lo), hi + 1) if v % edge and v != avoid]
    return int(rng.choice(choices)) if choices else None


def _tail(rng: np.random.Generator, edge: int, whole: tuple[int, int], avoid: int) -> int | None:
    """``a * edge + r`` with ``a`` in ``whole`` and ``0 < r < edge``: whole tiles plus a ragged one."""
    value = int(rng.integers(whole[0], whole[1] + 1)) * edge + int(rng.integers(1, edge))
    return value if value >= 2 and value != avoid else None


def _offset(rng: np.random.Generator, value: int, edge: int) -> int | None:
    """``value`` moved by a non-zero offset smaller than one edge, landing off the edge."""
    return _between(rng, value - (edge - 1), value + (edge - 1), edge, value)


def _spill_target(rng: np.random.Generator, capacity: int) -> int:
    return int(math.ceil(capacity * float(rng.uniform(*SPILL_RATIO))))


def _fits(cost: tuple[int, int]) -> bool:
    macs, elements = cost
    return macs <= MAX_MACS and elements <= MAX_OPERAND_ELEMENTS


def _over(cost: tuple[int, int]) -> float:
    macs, elements = cost
    return max(macs / MAX_MACS, elements / MAX_OPERAND_ELEMENTS, 1.0)


# --------------------------------------------------------------------------------------- contraction


@dataclass(frozen=True)
class _Contraction:
    """Extent keys of a contraction entry: batch (or None), rows, reduction, columns."""

    batch: str | None
    m: str
    k: str
    n: str

    @classmethod
    def of(cls, entry: Mapping[str, Any]) -> _Contraction | None:
        op = str(entry.get("op"))
        if op in CONTRACTION_OPS:
            keys = cls(None, "M", "K", "N")
        elif op in BATCHED_OPS:
            keys = cls("B", "M", "H" if "H" in entry else "K", "N")
        else:
            return None
        names = [k for k in (keys.batch, keys.m, keys.k, keys.n) if k]
        if not all(type(entry.get(k)) is int and entry[k] >= 1 for k in names):
            return None
        return keys

    def sizes(self, entry: Mapping[str, Any]) -> dict[str, int]:
        batch = int(entry[self.batch]) if self.batch else 1
        return {"B": batch, "M": int(entry[self.m]), "K": int(entry[self.k]), "N": int(entry[self.n])}


def _contraction_cost(s: Mapping[str, int]) -> tuple[int, int]:
    """``(MACs, operand elements)`` of a batched ``[M, K] x [K, N]``."""
    return s["B"] * s["M"] * s["K"] * s["N"], s["B"] * s["K"] * (s["M"] + s["N"])


def _batch(rng: np.random.Generator, batch: int) -> int:
    return 1 if batch == 1 else int(rng.choice([v for v in range(2, 2 * batch + 2) if v != batch]))


def _contraction_sizes(cls: str, s0: Mapping[str, int], lim: Limits, rng) -> dict[str, int] | None:
    edge = lim.edge
    out: dict[str, int | None] = {"B": _batch(rng, s0["B"])}
    if cls == "sub_tile":
        for d in ("M", "K", "N"):
            out[d] = 1 if s0[d] == 1 else _between(rng, 2, edge - 1, edge, s0[d])
    elif cls == "aspect":
        for d in ("M", "K", "N"):
            out[d] = 1 if s0[d] == 1 else _offset(rng, s0[d], edge)
        if any(v is None for v in out.values()):
            return None
        _shrink(out, ("M", "K", "N"), s0, edge, _contraction_cost)
    elif cls == "accumulator_spill":
        if s0["M"] == 1 and s0["N"] == 1:
            return None
        total = _spill_target(rng, lim.accumulator_elements)
        if s0["M"] == 1:
            out["M"], out["N"] = 1, _off_edge(total, edge)
        elif s0["N"] == 1:
            out["M"], out["N"] = _off_edge(total, edge), 1
        else:
            ratio = min(max(s0["M"] / s0["N"], 1 / edge), edge) * math.exp(float(rng.uniform(-0.5, 0.5)))
            out["M"] = _off_edge(round(math.sqrt(total * ratio)), edge)
            out["N"] = _off_edge(math.ceil(total / out["M"]), edge)
        out["K"] = 1 if s0["K"] == 1 else _tail(rng, edge, (1, 3), s0["K"])
    elif cls == "operand_spill":
        if s0["K"] == 1:
            return None
        out["M"] = 1 if s0["M"] == 1 else _tail(rng, edge, (0, 2), s0["M"])
        out["N"] = 1 if s0["N"] == 1 else _tail(rng, edge, (2, 8), s0["N"])
        if out["M"] is None or out["N"] is None:
            return None
        out["K"] = _off_edge(math.ceil(_spill_target(rng, lim.operand_elements) / (out["M"] + out["N"])), edge)
    else:  # pragma: no cover - CLASSES is closed
        raise VariantError(f"unknown class {cls!r}")
    if any(v is None for v in out.values()):
        return None
    return {k: int(v) for k, v in out.items()}


def _shrink(sizes: dict[str, int], keys: Sequence[str], s0: Mapping[str, int], edge: int, cost) -> None:
    """Scale the non-unit extents of ``keys`` down together until the work caps hold, keeping every rule."""
    scalable = [k for k in keys if sizes[k] >= edge]
    over = _over(cost(sizes))
    if over > 1.0 and scalable:
        factor = over ** (1.0 / len(scalable))
        for k in scalable:
            sizes[k] = _off_edge(sizes[k] / factor, edge)
    for _ in range(4 * ATTEMPTS):
        if _fits(cost(sizes)):
            break
        largest = max((k for k in keys if sizes[k] > 2), key=lambda k: sizes[k], default=None)
        if largest is None:
            break
        sizes[largest] = _off_edge(sizes[largest] * 0.85, edge)
    for k in keys:
        if sizes[k] == s0[k] and sizes[k] > 2:
            sizes[k] = sizes[k] - 1 if (sizes[k] - 1) % edge else sizes[k] - 2


def _contraction_valid(s: Mapping[str, int], s0: Mapping[str, int], edge: int) -> bool:
    for d in ("M", "K", "N"):
        if s0[d] == 1:
            if s[d] != 1:
                return False
        elif s[d] < 2 or s[d] % edge == 0 or s[d] == s0[d]:
            return False
    return s0["B"] == 1 or (s["B"] >= 2 and s["B"] != s0["B"])


# -------------------------------------------------------------------------------------------- window


@dataclass(frozen=True)
class _Window:
    """A convolution's fixed geometry: kernel, stride, padding (top, left, bottom, right), dilation."""

    kh: int
    kw: int
    stride: tuple[int, int]
    padding: tuple[int, int, int, int]
    dilation: tuple[int, int]

    @classmethod
    def of(cls, entry: Mapping[str, Any]) -> _Window | None:
        if str(entry.get("op")) not in WINDOW_OPS:
            return None
        try:
            window = cls(
                int(entry["kh"]),
                int(entry["kw"]),
                tuple(int(v) for v in entry.get("stride", [1, 1])),
                tuple(int(v) for v in entry.get("padding", [0, 0, 0, 0])),
                tuple(int(v) for v in entry.get("dilation", [1, 1])),
            )
        except (KeyError, TypeError, ValueError):
            return None
        if len(window.stride) != 2 or len(window.padding) != 4 or len(window.dilation) != 2:
            return None
        if not all(type(entry.get(k)) is int and entry[k] >= 1 for k in ("ci", "N", "Himg", "Wimg")):
            return None
        return window

    @property
    def taps(self) -> int:
        return self.kh * self.kw

    def out_dims(self, h: int, w: int) -> tuple[int, int]:
        from merlin.runtime.commandbuffer import conv_out_dims

        return conv_out_dims(h, w, self.kh, self.kw, self.stride, self.padding, self.dilation)

    def input_extent(self, rng, out: int, axis: int) -> int | None:
        """An input extent, larger than its padding, whose output extent is ``out`` (axis 0: rows)."""
        if axis == 0:
            k, s, d, pads = self.kh, self.stride[0], self.dilation[0], (self.padding[0], self.padding[2])
        else:
            k, s, d, pads = self.kw, self.stride[1], self.dilation[1], (self.padding[1], self.padding[3])
        least = (out - 1) * s + d * (k - 1) + 1 - sum(pads)
        choices = [v for v in range(least, least + s) if v >= 1 and v > max(pads)]
        return int(rng.choice(choices)) if choices else None


def _window_cost(s: Mapping[str, int], window: _Window) -> tuple[int, int]:
    """``(MACs, operand elements)``: output pixels x taps x channels; input + weight elements."""
    ho, wo = window.out_dims(s["H"], s["W"])
    return ho * wo * window.taps * s["ci"] * s["cout"], s["H"] * s["W"] * s["ci"] + window.taps * s["ci"] * s["cout"]


def _spatial(rng, window: _Window, s0: Mapping[str, int], ho: int, wo: int) -> tuple[int, int] | None:
    """Input extents for output extents ``(ho, wo)``; a unit input extent stays unit."""
    h = 1 if s0["H"] == 1 else window.input_extent(rng, ho, 0)
    w = 1 if s0["W"] == 1 else window.input_extent(rng, wo, 1)
    return None if h is None or w is None else (h, w)


def _channels(rng, s0: Mapping[str, int], edge: int, key: str, whole: tuple[int, int]) -> int | None:
    """A channel count in the kernel's own regime (below one edge stays below) with a ragged tile."""
    if s0[key] == 1:
        return 1
    if s0[key] < edge:
        return _between(rng, 2, edge - 1, edge, s0[key])
    return _tail(rng, edge, whole, s0[key])


def _window_sizes(cls: str, s0: Mapping[str, int], window: _Window, lim: Limits, rng) -> dict[str, int] | None:
    edge = lim.edge
    if cls == "sub_tile":
        ci = 1 if s0["ci"] == 1 else _between(rng, 2, edge - 1, edge, s0["ci"])
        cout = 1 if s0["cout"] == 1 else _between(rng, 2, edge - 1, edge, s0["cout"])
        ho = int(rng.integers(2, max(3, (edge - 1) // 2 + 1)))
        wo = int(rng.integers(2, max(3, (edge - 1) // ho + 1)))
        spatial = _spatial(rng, window, s0, ho, wo) if ho * wo < edge and ho != wo else None
    elif cls == "aspect":
        ci = 1 if s0["ci"] == 1 else _offset(rng, s0["ci"], edge)
        cout = 1 if s0["cout"] == 1 else _offset(rng, s0["cout"], edge)
        spatial = tuple(
            1
            if s0[key] == 1
            else s0[key] + int(rng.choice([v for v in range(-min(edge - 1, s0[key] // 2), edge) if v]))
            for key in ("H", "W")
        )
    elif cls == "accumulator_spill":
        cout = 1 if s0["cout"] == 1 else _tail(rng, edge, (1, 3), s0["cout"])
        ci = _channels(rng, s0, edge, "ci", (1, 2))
        if cout is None:
            return None
        rows = math.ceil(_spill_target(rng, lim.accumulator_elements) / cout)
        ho = max(1, round(math.sqrt(rows * math.exp(float(rng.uniform(-0.4, 0.4))))))
        wo = max(1, math.ceil(rows / ho))
        wo += int(ho == wo)
        wo += int((ho * wo) % edge == 0)
        spatial = _spatial(rng, window, s0, ho, wo)
    elif cls == "operand_spill":
        if s0["ci"] == 1:
            return None
        cout = 1 if s0["cout"] == 1 else _tail(rng, edge, (2, 8), s0["cout"])
        ho, wo = int(rng.integers(2, 7)), int(rng.integers(2, 7))
        spatial = _spatial(rng, window, s0, ho, wo + int(ho == wo))
        ci = None
        if spatial is not None and cout is not None:
            room = spatial[0] * spatial[1] + window.taps * cout
            ci = _off_edge(math.ceil(_spill_target(rng, lim.operand_elements) / room), edge)
    else:  # pragma: no cover - CLASSES is closed
        raise VariantError(f"unknown class {cls!r}")
    if spatial is None or ci is None or cout is None:
        return None
    sizes = {"ci": int(ci), "cout": int(cout), "H": int(spatial[0]), "W": int(spatial[1])}
    if cls == "aspect":
        _shrink_window(sizes, s0, window, edge)
    return sizes if _window_valid(sizes, s0, window, edge) else None


def _window_valid(s: Mapping[str, int], s0: Mapping[str, int], window: _Window, edge: int) -> bool:
    for key, pads in (("H", (window.padding[0], window.padding[2])), ("W", (window.padding[1], window.padding[3]))):
        if s0[key] == 1:
            if s[key] != 1:
                return False
        elif s[key] < 2 or s[key] <= max(pads) or s[key] == s0[key]:
            return False
    if s0["H"] > 1 and s0["W"] > 1 and s["H"] == s["W"]:
        return False
    ho, wo = window.out_dims(s["H"], s["W"])
    own = window.out_dims(s0["H"], s0["W"])
    if min(ho, wo) < 1 or (ho < 2 <= own[0]) or (wo < 2 <= own[1]) or (ho * wo > 1 and (ho * wo) % edge == 0):
        return False
    for key in ("ci", "cout"):
        if s0[key] == 1:
            if s[key] != 1:
                return False
        elif s[key] < 2 or s[key] % edge == 0 or s[key] == s0[key]:
            return False
    return True


def _shrink_window(s: dict[str, int], s0: Mapping[str, int], window: _Window, edge: int) -> None:
    """Scale an ``aspect`` convolution down to the caps: each input axis and each channel count together."""
    over = _over(_window_cost(s, window))
    if over > 1.0:
        keys = [k for k in ("H", "W") if s[k] > 2] + [k for k in ("ci", "cout") if s[k] >= edge]
        factor = over ** (1.0 / max(1, len(keys)))
        for key in keys:
            s[key] = _off_edge(s[key] / factor, edge) if key in ("ci", "cout") else max(2, int(s[key] / factor))
    for _ in range(4 * ATTEMPTS):
        if _fits(_window_cost(s, window)):
            break
        key = max((k for k in ("H", "W", "ci", "cout") if s[k] > 2), key=lambda k: s[k], default=None)
        if key is None:
            break
        s[key] = _off_edge(s[key] * 0.85, edge) if key in ("ci", "cout") else max(2, int(s[key] * 0.85))
    if s["H"] == s["W"] and s0["H"] > 1 and s0["W"] > 1:
        s["W"] += 1


# ------------------------------------------------------------------------------------------ scales


def _perturbed_scale(rng, value: float) -> float:
    """``value`` moved by a relative step in :data:`SCALE_STEP`, exactly representable in float32."""
    for _ in range(ATTEMPTS):
        step = float(rng.uniform(*SCALE_STEP)) * float(rng.choice([-1.0, 1.0]))
        moved = float(value) * (1.0 + step)
        quantum = 2.0 ** (math.floor(math.log2(abs(moved))) - (SCALE_MANTISSA_BITS - 1))
        exact = round(moved / quantum) * quantum
        if exact not in (0.0, float(value)) and float(np.float32(exact)) == exact:
            return exact
    raise VariantError(f"acc_scale {value!r}: no exactly representable step in {SCALE_STEP}")


# ----------------------------------------------------------------------------------------- variants


@dataclass
class Variant:
    """One shape-perturbed kernel: its stated program, interface, host call and the regimes it lands in."""

    symbol: str
    variant_of: str
    klass: str
    entry: dict[str, Any]
    signature: list[int]
    dtypes: list[str]
    call_buffers: list[dict[str, Any]]
    interface: str
    macs: int
    operand_elements: int
    regimes: dict[str, Any]

    @property
    def interface_sha256(self) -> str:
        return hashlib.sha256(self.interface.encode()).hexdigest()

    def record(self) -> dict[str, Any]:
        row = dataclasses.asdict(self)
        row["class"] = row.pop("klass")
        row["interface_sha256"] = self.interface_sha256
        return row

    @classmethod
    def from_record(cls, row: Mapping[str, Any]) -> Variant:
        names = {f.name for f in dataclasses.fields(cls)} - {"klass"}
        variant = cls(klass=str(row["class"]), **{k: row[k] for k in names})
        if variant.interface_sha256 != row.get("interface_sha256"):
            raise VariantError(f"{variant.symbol}: stored interface does not match its digest")
        return variant


def _signature_for(signature: Sequence[int], s0: Mapping[str, int], s: Mapping[str, int]) -> list[int] | None:
    """The variant's rewrite signature in the kernel's own convention, or ``None`` when it has none known."""
    sig = [int(v) for v in signature]
    if not sig:
        return []
    if not s0:
        return None
    for keys in (("M", "N", "K"), ("B", "M", "N", "K")):
        if sig == [s0[k] for k in keys]:
            return [s[k] for k in keys]
    return None


def _regimes(lim: Limits, *, m: int, k: int, n: int, batch: int, operand_elements: int) -> dict[str, Any]:
    from merlin.targetgen.conformance import classify_alignment

    return {
        "occupancy": classify_alignment([m, k, n], lim.edge),
        "reduction_depth": "multi_pass" if k > lim.edge else "single_pass",
        "accumulator_ratio": round(m * n / lim.accumulator_elements, 4),
        "operand_ratio": round(operand_elements / batch / lim.operand_elements, 4),
    }


def _class_holds(cls: str, extents: tuple[int, ...], regimes: Mapping[str, Any], edge: int) -> bool:
    """Whether a draw really is in ``cls``: its claim, checked on the drawn sizes.

    ``extents`` are the extents ``sub_tile`` bounds: a contraction's M, K, N; a convolution's output
    pixels and channels (its reduction also spans the taps, which the form fixes).
    """
    if cls == "sub_tile":
        return all(v < edge for v in extents)
    if cls == "accumulator_spill":
        return regimes["accumulator_ratio"] >= SPILL_RATIO[0]
    if cls == "operand_spill":
        return regimes["operand_ratio"] >= SPILL_RATIO[0]
    return True


def _restate(entry: Mapping[str, Any], cls: str, sizes: Mapping[str, int], rng) -> dict[str, Any]:
    out = dict(entry)
    contraction = _Contraction.of(entry)
    if contraction is not None:
        if contraction.batch:
            out[contraction.batch] = sizes["B"]
        out[contraction.m], out[contraction.k], out[contraction.n] = sizes["M"], sizes["K"], sizes["N"]
    else:
        out.update({"ci": sizes["ci"], "N": sizes["cout"], "Himg": sizes["H"], "Wimg": sizes["W"]})
    if out.get("acc_scale") is not None:
        out["acc_scale"] = _perturbed_scale(rng, float(out["acc_scale"]))
    out["source_reference"] = f"{SOURCE} {cls} of {entry.get('name', 'kernel')}"
    return out


def interface_builder(target: str, routing: Mapping[str, Any]) -> Callable[[Mapping[str, Any]], str]:
    """:func:`merlin.model_check.kernel_reference.interface_for` with the device binding derived once."""
    from merlin.compile.mesh import _mesh_tile_binding
    from merlin.targetgen import corpus_spec as CS

    binding = _mesh_tile_binding(
        target, routing.get("operand_dtype"), routing.get("accum_dtype"), numeric_policy=routing.get("numeric_policy")
    )

    def build(entry: Mapping[str, Any]) -> str:
        return CS.build(dict(entry), binding)[1]

    return build


def _skip(symbol: str, cls: str, reason: str) -> dict[str, str]:
    return {"variant_of": symbol, "class": cls, "reason": reason}


def variants_for(
    symbol: str,
    entry: Mapping[str, Any],
    *,
    signature: Sequence[int],
    call_buffers: Sequence[Mapping[str, Any]],
    dtypes: Sequence[str],
    interface: str,
    target: str,
    routing: Mapping[str, Any],
    lim: Limits,
    seed: int,
    name_of: Callable[[], str],
    interface_of: Callable[[Mapping[str, Any]], str] | None = None,
) -> tuple[list[Variant], list[dict[str, str]]]:
    """``(variants, skipped)``: one variant per class of :data:`CLASSES` for one kernel, where reachable.

    ``interface_of`` restates an entry as its interface (default :func:`interface_builder`).
    """
    from merlin.llvmlower.device_shim import logical_buffers

    interface_of = interface_of or interface_builder(target, routing)

    contraction, window = _Contraction.of(entry), _Window.of(entry)
    if contraction is None and window is None:
        return [], [_skip(symbol, "*", f"form {entry.get('op')!r} has no size rule")]
    if contraction is not None and "maxpool" in [str(s) for s in entry.get("epilogue") or ()]:
        return [], [_skip(symbol, "*", "a contraction's declared pool extent is fixed by its program")]
    own, _why = logical_buffers(interface)
    call = {str(b["role"]): (tuple(int(v) for v in b["shape"]), str(b["dtype"])) for b in call_buffers}
    if own is None or {b.role: (tuple(b.shape), b.dtype) for b in own} != call:
        return [], [_skip(symbol, "*", "the host call's layout is not the interface's")]
    if contraction is not None:
        s0 = contraction.sizes(entry)
    else:
        s0 = {"ci": entry["ci"], "cout": entry["N"], "H": entry["Himg"], "W": entry["Wimg"]}
    key = hashlib.sha256(interface.encode()).hexdigest()
    variants: list[Variant] = []
    skipped: list[dict[str, str]] = []
    for cls in CLASSES:
        made, reason = None, "no draw met the class's rules within the work caps"
        for attempt in range(ATTEMPTS):
            rng = _rng(seed, key, cls, attempt)
            if contraction is not None:
                sizes = _contraction_sizes(cls, s0, lim, rng)
                if sizes is None or not _contraction_valid(sizes, s0, lim.edge) or not _fits(_contraction_cost(sizes)):
                    continue
                macs, elements = _contraction_cost(sizes)
                mkn, batch = (sizes["M"], sizes["K"], sizes["N"]), sizes["B"]
                checked = mkn
                sig = _signature_for(signature, s0, sizes)
            else:
                sizes = _window_sizes(cls, s0, window, lim, rng)
                if sizes is None or not _fits(_window_cost(sizes, window)):
                    continue
                macs, elements = _window_cost(sizes, window)
                ho, wo = window.out_dims(sizes["H"], sizes["W"])
                mkn, batch = (ho * wo, window.taps * sizes["ci"], sizes["cout"]), 1
                checked = (ho * wo, sizes["ci"], sizes["cout"])
                sig = _signature_for(signature, {}, sizes)
            if sig is None:
                reason = f"rewrite signature {list(signature)} is in no known convention"
                break
            regimes = _regimes(lim, m=mkn[0], k=mkn[1], n=mkn[2], batch=batch, operand_elements=elements)
            if not _class_holds(cls, checked, regimes, lim.edge):
                continue
            restated = _restate(entry, cls, sizes, rng)
            restated["name"] = name = name_of()
            try:
                text = interface_of(restated)
            except Exception as error:  # noqa: BLE001 -- a draw the interface builder refuses is redrawn
                reason = f"interface: {type(error).__name__}: {str(error)[:160]}"
                continue
            buffers, why = logical_buffers(text)
            if buffers is None:
                reason = f"interface buffers: {why}"
                continue
            shapes = {b.role: list(b.shape) for b in buffers}
            if set(shapes) != set(call):
                reason = "the restated interface changed the call's roles"
                continue
            made = Variant(
                symbol=name,
                variant_of=symbol,
                klass=cls,
                entry=restated,
                signature=sig,
                dtypes=list(dtypes),
                call_buffers=[{**dict(b), "shape": shapes[str(b["role"])]} for b in call_buffers],
                interface=text,
                macs=int(macs),
                operand_elements=int(elements),
                regimes=regimes,
            )
            break
        if made is None:
            skipped.append(_skip(symbol, cls, reason))
        else:
            variants.append(made)
    return variants, skipped


# ----------------------------------------------------------------------------------------- operands


def _int_range(dtype: str) -> tuple[int, int] | None:
    """``(lo, hi)`` of an integer dtype token (``i8``, ``int32``, ``u8``), ``None`` for anything else."""
    token = str(dtype)
    for prefix, signed in (("uint", False), ("int", True), ("u", False), ("i", True)):
        digits = token[len(prefix) :]
        if token.startswith(prefix) and digits.isdigit():
            bits = int(digits)
            return (-(1 << (bits - 1)), (1 << (bits - 1)) - 1) if signed else (0, (1 << bits) - 1)
    return None


def _reduction(entry: Mapping[str, Any]) -> int:
    window = _Window.of(entry)
    if window is not None:
        return window.taps * int(entry["ci"])
    contraction = _Contraction.of(entry)
    return int(entry[contraction.k]) if contraction else 1


def _readout_scale(entry: Mapping[str, Any]) -> float:
    """The multiplier a readout applies to the accumulator (``acc_scale`` times a requant shift)."""
    stages = [str(s) for s in entry.get("epilogue") or ()]
    scale = float(entry["acc_scale"]) if "acc_scale" in stages and entry.get("acc_scale") is not None else 1.0
    if "requant" in stages and entry.get("requant_shift") is not None:
        scale *= 2.0 ** -int(entry["requant_shift"])
    return scale


def amplitudes(variant: Variant, *, accum_dtype: str) -> dict[str, int]:
    """Symmetric operand bounds for one variant: ``{"activation", "weight", "bias"}``.

    Full operand range by default. A narrowing readout (result dtype narrower than the accumulator)
    shrinks the weight range so a typical result is a quarter of the result range rather than
    saturated; every range then shrinks until no exact accumulator can reach half its range.
    """
    acc = _int_range(accum_dtype)
    if acc is None:
        raise VariantError(f"accumulator dtype {accum_dtype!r} is not an integer type")
    budget = acc[1] // 2
    by_role = {str(b["role"]).split("_")[0]: b for b in variant.call_buffers}
    span = _int_range(str(by_role["input"]["dtype"])) if "input" in by_role else None
    if span is None:
        raise VariantError(f"{variant.symbol}: the activation is not an integer operand")
    top = max(abs(span[0]), abs(span[1]))
    act, weight, k = top, top, max(1, _reduction(variant.entry))
    result = _int_range(str(by_role["out"]["dtype"])) if "out" in by_role else None
    if result is not None and result[1] < acc[1]:
        sigma = (result[1] + 1) / 4.0 / _readout_scale(variant.entry) / math.sqrt(k)
        weight = int(min(top, max(1, round(sigma * 3.0 / top))))
        if weight == 1:
            act = int(min(top, max(1, round(sigma * 3.0 / math.sqrt(2.0)))))
    if k * act * weight > budget:
        weight = max(1, budget // (k * act))
    if k * act * weight > budget:
        act = max(1, budget // (k * weight))
    bias = int(min(acc[1] - k * act * weight, max(1, round(4 * math.sqrt(k) * act * weight / 3.0))))
    return {"activation": int(act), "weight": int(weight), "bias": bias}


def operands(variant: Variant, *, seed: int, accum_dtype: str) -> list[np.ndarray]:
    """Seeded host-call operands for ``variant``, in call order (destinations zeroed)."""
    from merlin.runtime.linalg_numpy import storage_dtype

    bounds = amplitudes(variant, accum_dtype=accum_dtype)
    rng = _rng(seed, variant.interface_sha256, variant.klass, "operands")
    values: list[np.ndarray] = []
    for buffer in variant.call_buffers:
        role, shape, dtype = str(buffer["role"]), tuple(int(v) for v in buffer["shape"]), str(buffer["dtype"])
        store = storage_dtype(dtype)
        kind = role.split("_")[0]
        if kind == "out":
            values.append(np.zeros(shape, dtype=store))
            continue
        span = _int_range(dtype)
        rule_of = {"input": "activation", "weight": "weight", "bias": "bias"}
        if span is None or kind not in rule_of:
            raise VariantError(f"{variant.symbol}: no value rule for operand {role!r} ({dtype})")
        bound = bounds[rule_of[kind]]
        array = rng.integers(max(span[0], -bound), min(span[1], bound) + 1, size=shape, dtype=np.int64)
        if kind == "input" and bound >= max(abs(span[0]), abs(span[1])) and array.size >= 2 * EXTREME_ELEMENTS:
            flat = array.reshape(-1)
            picks = rng.choice(flat.size, size=2 * EXTREME_ELEMENTS, replace=False)
            flat[picks[:EXTREME_ELEMENTS]] = span[0]
            flat[picks[EXTREME_ELEMENTS:]] = span[1]
        values.append(array.astype(store))
    return values


# ------------------------------------------------------------------------------------------- storage


def root(bundle_root: Path, model: str) -> Path:
    return Path(bundle_root) / model / SOURCE


def scratch(bundle_root: Path, model: str) -> Path:
    """Where a replay builds and runs ``model``'s variants: beside the private bundle, never under ``--work``.

    The kernel interfaces, lowered objects, operand files and images of a variant replay restate every
    variant's shapes, so they stay exactly as private as the variants themselves (contract §8).
    """
    bundle = Path(bundle_root).resolve()
    return bundle.parent / f"{bundle.name}.{SOURCE}-work" / model


def _next_symbols(symbols: Iterable[str]) -> Callable[[], str]:
    """Names in the routed symbols' own ``<stem><index>`` style, after the catalog's last index."""
    stem, top = "", -1
    for symbol in symbols:
        head = symbol.rstrip("0123456789")
        stem = stem or head
        if head == stem and symbol[len(head) :]:
            top = max(top, int(symbol[len(head) :]))
    counter = [top + 1]

    def name() -> str:
        counter[0] += 1
        return f"{stem or 'kernel_'}{counter[0] - 1}"

    return name


def rule() -> dict[str, Any]:
    """The public rule parameters (everything but the seed)."""
    return {
        "version": RULE_VERSION,
        "classes": list(CLASSES),
        "max_macs": MAX_MACS,
        "max_operand_elements": MAX_OPERAND_ELEMENTS,
        "spill_ratio": list(SPILL_RATIO),
        "scale_step": list(SCALE_STEP),
        "scale_mantissa_bits": SCALE_MANTISSA_BITS,
        "extreme_elements": EXTREME_ELEMENTS,
    }


def build_program(
    catalog: CAT.ProgramCatalog,
    *,
    out_dir: Path,
    blobs: CAT.BlobStore,
    seed: int,
    model: str,
    program: str,
    lim: Limits | None = None,
) -> dict[str, Any]:
    """Variants of every unique kernel of one program, with operands and expected results, on disk.

    ``lim`` defaults to the catalog target's own :func:`limits`.
    """
    started = time.monotonic()
    if not catalog.interfaces:
        return {"kernels": 0, "variants": 0, "skipped": 0, "classes": {}, "host_only": True}
    routing = dict(catalog.routing)
    lim = lim or limits(catalog.target, routing.get("operand_dtype"))
    name_of = _next_symbols(catalog.interfaces)
    interface_of = interface_builder(catalog.target, routing)
    routed = {str(r.get("symbol")): r for r in catalog.sidecar.get("routed") or ()}
    variants: list[Variant] = []
    skipped: list[dict[str, str]] = []
    seen: set[str] = set()
    for symbol in sorted(catalog.interfaces, key=lambda s: (len(s), s)):
        interface = catalog.interfaces[symbol]
        digest = hashlib.sha256(interface.encode()).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        dtypes = (catalog.sidecar.get("dtypes") or {}).get(symbol) or (routed.get(symbol) or {}).get("dtypes")
        made, missed = variants_for(
            symbol,
            catalog.sidecar["entries"][symbol],
            signature=catalog.sidecar["signatures"][symbol],
            call_buffers=catalog.sidecar["call_buffers"][symbol],
            dtypes=list(dtypes or ()),
            interface=interface,
            target=catalog.target,
            routing=routing,
            lim=lim,
            seed=seed,
            name_of=name_of,
            interface_of=interface_of,
        )
        variants += made
        skipped += missed
    calls = []
    for variant in variants:
        values = operands(variant, seed=seed, accum_dtype=str(routing.get("accum_dtype")))
        roles = [str(b["role"]) for b in variant.call_buffers]
        expected = KR.call_outputs(variant.interface, values, roles)
        calls.append(
            {
                "symbol": variant.symbol,
                "ordinal": 0,
                "roles": roles,
                "operands": [blobs.put(v) for v in values],
                "expected": [blobs.put(np.asarray(r)) for r in expected],
            }
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = {
        "schema": VARIANTS_SCHEMA,
        "rule": rule(),
        "seed": int(seed),
        "model": model,
        "program": program,
        "target": catalog.target,
        "routing": routing,
        "limits": lim.record(),
        "variants": [v.record() for v in variants],
        "skipped": skipped,
    }
    (out_dir / "variants.json").write_text(json.dumps(doc, indent=1) + "\n")
    CAT.write_calls(out_dir / "calls.json", calls, model=model, program=program, input=SOURCE)
    classes: dict[str, int] = {}
    for variant in variants:
        classes[variant.klass] = classes.get(variant.klass, 0) + 1
    return {
        "kernels": len(seen),
        "variants": len(variants),
        "skipped": len(skipped),
        "classes": classes,
        "macs": sum(v.macs for v in variants),
        "seconds": round(time.monotonic() - started, 1),
    }


def build(bundle, model: str, programs: Sequence[str], *, seed: int, log: Callable[[str], None] = print) -> dict:
    """Variants for ``programs`` of ``model`` in ``bundle`` (a :class:`merlin.model_check.bundle.Bundle`)."""
    import fcntl

    directory = root(bundle.root, model)
    blobs = CAT.BlobStore(directory / "blobs")
    record: dict[str, Any] = {"model": model, "programs": {}}
    for program in programs:
        catalog = bundle.catalog(model, program)
        row = build_program(catalog, out_dir=directory / program, blobs=blobs, seed=seed, model=model, program=program)
        log(f"[variants] {model}/{program}: {json.dumps(row)}")
        record["programs"][program] = row
    manifest_path = bundle.root / "MANIFEST.json"
    with (bundle.root / ".manifest.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        manifest = json.loads(manifest_path.read_text())
        entry = manifest.setdefault("models", {}).setdefault(model, {"model": model, "programs": {}})
        state = entry.setdefault(SOURCE, {"rule": RULE_VERSION, "programs": {}})
        state["rule"] = RULE_VERSION
        for program, row in record["programs"].items():
            path = directory / program / "variants.json"
            state["programs"][program] = {
                "variants": row["variants"],
                "variants_sha256": hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None,
            }
        manifest_path.write_text(json.dumps(manifest, indent=1) + "\n")
    return record


# ---------------------------------------------------------------------------------------------- load


@dataclass
class VariantSet:
    """A program's stored variants: the catalog a replay builds them from, and the cases it replays."""

    program: str
    catalog: CAT.ProgramCatalog
    variants: dict[str, Variant]
    calls: list[dict[str, Any]]
    blobs: CAT.BlobStore
    regions: dict[str, set[str]]

    def cases(self, *, group: str | None = None) -> list:
        """Replay cases, all or those varying ``group`` (a varied symbol or its source region id)."""
        from merlin.model_check.replay import ReplayCase

        out = []
        for call in self.calls:
            variant = self.variants[call["symbol"]]
            names = {variant.variant_of, variant.symbol} | self.regions.get(variant.variant_of, set())
            if group is not None and group not in names:
                continue
            out.append(
                ReplayCase(
                    symbol=call["symbol"],
                    label=f"{SOURCE}/{self.program}/{call['symbol']}#{call['ordinal']}",
                    roles=tuple(call["roles"]),
                    operands=tuple(self.blobs.get(k) for k in call["operands"]),
                    expected=tuple(self.blobs.get(k) for k in call["expected"]),
                )
            )
        return out

    def annotate(self, kernels: Iterable[dict[str, Any]]) -> None:
        """Mark each replayed kernel row with the kernel it varies and its class."""
        for row in kernels:
            variant = self.variants.get(row.get("symbol"))
            if variant is not None:
                row["variant_of"], row["variant_class"] = variant.variant_of, variant.klass


def load(bundle_root: Path, model: str, program: str, base: CAT.ProgramCatalog) -> VariantSet | None:
    """The stored variants of ``program`` (``base`` is its catalog), or ``None`` when there are none."""
    directory = root(bundle_root, model) / program
    path = directory / "variants.json"
    if not path.is_file():
        return None
    doc = json.loads(path.read_text())
    if doc.get("schema") != VARIANTS_SCHEMA:
        raise VariantError(f"{path}: not a {VARIANTS_SCHEMA} document")
    variants = {row["symbol"]: Variant.from_record(row) for row in doc["variants"]}
    sidecar = {
        "signatures": {s: v.signature for s, v in variants.items()},
        "entries": {s: v.entry for s, v in variants.items()},
        "call_buffers": {s: v.call_buffers for s, v in variants.items()},
        "dtypes": {s: v.dtypes for s, v in variants.items()},
        "routed": [{"symbol": s, "dtypes": v.dtypes, "variant_of": v.variant_of} for s, v in variants.items()],
    }
    catalog = dataclasses.replace(base, sidecar=sidecar, interfaces={s: v.interface for s, v in variants.items()})
    regions: dict[str, set[str]] = {}
    for row in base.sidecar.get("routed") or ():
        regions.setdefault(str(row.get("symbol")), set()).add(str(row.get("source_region_id")))
    calls = CAT.read_calls(directory / "calls.json")["calls"]
    return VariantSet(program, catalog, variants, calls, CAT.BlobStore(root(bundle_root, model) / "blobs"), regions)
