"""Derived Phase 0 stage: capacity-scaled shape families for the Phase 2 performance cohort.

The form-perf and PK cohorts reach about M<=17, N<=32, K=128. Those shapes fit every on-chip store
many times over, so nothing in them is decided by tiling, blocking, DMA scheduling, double-buffering,
store reuse, epilogue placement or command-queue pressure -- which is exactly what Phase 2 is for.
Phase 2 iterates on a calibrated Spike timing model, not on an RTL simulator, so large shapes are
affordable; the RTL simulator sees only the short calibration-validation list
(:mod:`merlin_experiments.phase2.calibration_validation`).

This module derives the large shapes from the target's own RTL facts and from nothing else:

* array geometry, element widths and the stores' banks and capacities
  (:func:`merlin.perf.derived_bound.machine_from_facts`);
* the movement engine's read/write ports and in-flight windows (:mod:`merlin.perf.movement_peak`);
* per command class, its reservation entries and command-queue depth, and the dataflow modes the
  array's cell select chooses (``timing_structure.command_classes`` / ``dataflow``).

The template (``requires_capacity_scope`` in ``experiments/templates/phase0/performance.yaml``)
declares only POLICY: which family kinds, at which multiples of which capacity. Every extent comes
from a capacity read here. Each point records the regime it exercises, why (with the numbers that
put it there), and its staged roofline (:mod:`merlin.perf.staged_roofline`). Claims are per-regime
scaling fits (:mod:`merlin_experiments.phase2.claims.capacity`), never single points and never a
monotone "fewer is better": a change that removes commands can lose overlap and be slower, and the
fits are per regime so that is visible instead of averaged away.

Nothing here names a target. A target whose facts do not establish a family's inputs skips that
family with the reason; it is never given a default.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import isqrt
from typing import Any

SCHEMA = "merlin.phase0.capacity_cohort.v1"
#: The block a capacity-family member carries under ``performance``.
CAPACITY_BLOCK = "capacity"

FITS_DOUBLE, FITS_SINGLE, SPILLS = "fits_double", "fits_single", "spills"


class CapacityUnavailable(ValueError):
    """The facts do not establish an input a family needs; the family is skipped with this reason."""


# ------------------------------------------------------------------------------------------------
# the basis: every capacity a family is scaled by, read from the facts
# ------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Basis:
    tile_rows: int
    tile_cols: int
    operand_bytes: int
    accumulate_bytes: int
    ways: int
    operand_store_bytes: int
    accumulator_bytes: int
    read_bytes_per_cycle: float
    write_bytes_per_cycle: float
    command_windows: Mapping[str, int]
    dataflow: Mapping[str, Any]
    peak_macs_per_cycle: int
    machine: Any = field(repr=False, compare=False)
    ports: Mapping[str, Any] = field(repr=False, compare=False)
    provenance: Mapping[str, str] = field(default_factory=dict)

    @property
    def operand_loop_bytes(self) -> int:
        return self.operand_store_bytes // self.ways

    @property
    def accumulator_loop_bytes(self) -> int:
        return self.accumulator_bytes // self.ways

    def to_dict(self) -> dict[str, Any]:
        return {
            "tile_rows": self.tile_rows,
            "tile_cols": self.tile_cols,
            "operand_bytes": self.operand_bytes,
            "accumulate_bytes": self.accumulate_bytes,
            "ping_pong_ways": self.ways,
            "operand_store_bytes": self.operand_store_bytes,
            "accumulator_bytes": self.accumulator_bytes,
            "operand_loop_bytes": self.operand_loop_bytes,
            "accumulator_loop_bytes": self.accumulator_loop_bytes,
            "read_bytes_per_cycle": self.read_bytes_per_cycle,
            "write_bytes_per_cycle": self.write_bytes_per_cycle,
            "command_windows": dict(self.command_windows),
            "dataflow": dict(self.dataflow),
            "peak_macs_per_cycle": self.peak_macs_per_cycle,
            "in_flight": {
                name: {
                    "in_flight_bytes": row.get("in_flight_bytes"),
                    "binding_latency_cycles": row.get("binding_latency_cycles"),
                }
                for name, row in (self.ports.get("clients") or {}).items()
            },
            "provenance": dict(self.provenance),
        }

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()


def _command_windows(body: Mapping[str, Any]) -> tuple[dict[str, int], dict[str, str]]:
    classes = (body.get("timing_structure") or {}).get("command_classes") or {}
    if classes.get("status") != "derived":
        return {}, {"command_windows": "timing_structure.command_classes is not derived"}
    windows: dict[str, int] = {}
    for name, row in (classes.get("classes") or {}).items():
        entries = (row.get("entries") or {}) if isinstance(row, Mapping) else {}
        queue = (row.get("command_queue") or {}) if isinstance(row, Mapping) else {}
        value, depth = entries.get("value"), queue.get("entries_total")
        if entries.get("status") == "derived" and queue.get("status") == "derived":
            if type(value) is int and type(depth) is int and value > 0 and depth >= 0:
                # Commands that may be live in this class before issue stalls: the dispatch entries
                # plus the buffers the intact issued command crosses.
                windows[str(name)] = value + depth
    if not windows:
        return {}, {"command_windows": "no command class states both its entries and its queue depth"}
    return windows, {}


def derive_basis(target: str, facts: Mapping[str, Any]) -> Basis:
    """Every capacity the families scale by, or :class:`CapacityUnavailable` naming what is missing."""
    from merlin.perf.decompose import is_unknown
    from merlin.perf.derived_bound import machine_from_facts
    from merlin.perf.movement_peak import movement_peak

    machine = machine_from_facts(target, facts=dict(facts), measure_fill=False)
    needed = (
        "array_rows",
        "array_cols",
        "operand_bytes",
        "accumulate_bytes",
        "ping_pong_ways",
        "operand_store_bytes",
        "accumulate_store_bytes",
    )
    missing = {name: machine.refusals.get(name, "UNKNOWN") for name in needed if is_unknown(getattr(machine, name))}
    if missing or is_unknown(machine.peak_macs_per_cycle):
        raise CapacityUnavailable(f"the facts do not establish {sorted(missing)}: {missing}")
    ports = movement_peak(facts)
    if ports["status"] != "derived":
        raise CapacityUnavailable(f"the movement ports are not derived: {ports['unresolved']}")
    body = facts.get("facts") if "facts" in facts else facts
    windows, unresolved = _command_windows(body)
    dataflow = dict((body.get("timing_structure") or {}).get("dataflow") or {})
    ways = int(machine.ping_pong_ways)
    provenance = dict(machine.provenance)
    provenance.update(
        movement="facts.timing_structure.memory_clients (merlin.perf.movement_peak)",
        command_windows=(
            "timing_structure.command_classes: dispatch entries + command-queue entries per class"
            if windows
            else unresolved["command_windows"]
        ),
        dataflow=f"timing_structure.dataflow status={dataflow.get('status')!r}",
    )
    return Basis(
        tile_rows=int(machine.array_rows),
        tile_cols=int(machine.array_cols),
        operand_bytes=int(machine.operand_bytes),
        accumulate_bytes=int(machine.accumulate_bytes),
        ways=ways,
        # The machine reports the share one loop context sees; the store is that times its ways.
        operand_store_bytes=int(machine.operand_store_bytes) * ways,
        accumulator_bytes=int(machine.accumulate_store_bytes) * ways,
        read_bytes_per_cycle=float(ports["read_bytes_per_cycle"]),
        write_bytes_per_cycle=float(ports["write_bytes_per_cycle"]),
        command_windows=windows,
        dataflow=dataflow,
        peak_macs_per_cycle=int(machine.peak_macs_per_cycle),
        machine=machine,
        ports=ports,
        provenance=provenance,
    )


# ------------------------------------------------------------------------------------------------
# shape arithmetic
# ------------------------------------------------------------------------------------------------
def _round_tile(value: float, tile: int) -> int:
    """The positive multiple of ``tile`` nearest ``value`` (at least one tile)."""
    return max(int(tile), int(round(float(value) / tile)) * int(tile))


def _balanced_pair(elements: float, rows: int, cols: int) -> tuple[int, int]:
    """Tile-multiple ``(M, N)`` near-square with product near ``elements``: M from the root, N to match."""
    target = max(1.0, float(elements))
    m = _round_tile(target**0.5, rows)
    return m, _round_tile(target / m, cols)


def residency(basis: Basis, operand: int, accumulator: int) -> str:
    """The residency regime of a working set: both stores' share against one loop context and the store."""
    if operand <= basis.operand_loop_bytes and accumulator <= basis.accumulator_loop_bytes:
        return FITS_DOUBLE
    if operand <= basis.operand_store_bytes and accumulator <= basis.accumulator_bytes:
        return FITS_SINGLE
    return SPILLS


def _gemm_working_set(basis: Basis, m: int, k: int, n: int) -> tuple[int, int]:
    return (m * k + k * n) * basis.operand_bytes, m * n * basis.accumulate_bytes


def _fractions(spec: Mapping[str, Any], key: str, *, owner: str) -> list[float]:
    values = spec.get(key)
    if (
        not isinstance(values, list)
        or not values
        or any(isinstance(v, bool) or not isinstance(v, (int, float)) or v <= 0 for v in values)
    ):
        raise ValueError(f"{owner}: {key} must be a non-empty list of positive multiples")
    if len(set(values)) != len(values):
        raise ValueError(f"{owner}: {key} repeats a multiple")
    return [float(v) for v in values]


def _extent(token: Any, basis: Basis) -> int:
    from .sweeps import resolve_extent

    return resolve_extent(token, basis.tile_rows)


def _point(
    *,
    variant: str,
    regime: str,
    x: float,
    x_name: str,
    entry: dict[str, Any],
    work: list,
    result_bytes: int,
    why: str,
    working_set: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    return {
        "variant": variant,
        "regime": regime,
        "x": x,
        "x_name": x_name,
        "entry": entry,
        "work": work,
        "result_bytes": result_bytes,
        "why": why,
        "working_set": dict(working_set or {}),
    }


def _matmul(m: int, k: int, n: int, epilogue: Sequence[str] = ()) -> dict[str, Any]:
    return {"op": "matmul", "M": m, "K": k, "N": n, "epilogue": list(epilogue)}


def _gemm_point(basis: Basis, *, variant: str, regime: str | None, x_name: str, m: int, k: int, n: int, why: str):
    from merlin.perf.staged_roofline import Contraction

    operand, accumulator = _gemm_working_set(basis, m, k, n)
    band = residency(basis, operand, accumulator)
    x = {"useful_macs": m * k * n, "K": k, "N": n, "output_elements": m * n}[x_name]
    return _point(
        variant=variant,
        regime=regime or band,
        x=x,
        x_name=x_name,
        entry=_matmul(m, k, n),
        work=[Contraction(m, k, n)],
        result_bytes=basis.accumulate_bytes,
        why=(
            f"{why}; working set {operand} operand bytes ({operand / basis.operand_store_bytes:.3g}x the "
            f"{basis.operand_store_bytes}-byte operand store) and {accumulator} accumulator bytes "
            f"({accumulator / basis.accumulator_bytes:.3g}x the {basis.accumulator_bytes}-byte accumulator), "
            f"residency {band} at {basis.ways} loop context(s)"
        ),
        working_set={"operand_bytes": operand, "accumulator_bytes": accumulator, "residency": band},
    )


# ------------------------------------------------------------------------------------------------
# family kinds
# ------------------------------------------------------------------------------------------------
def _capacity_knee(spec, basis: Basis, owner: str) -> list[dict]:
    """Joint working sets at multiples of BOTH stores, plus one reduction tile either side of the knee."""
    points = []
    for fraction in _fractions(spec, "capacity_multiples", owner=owner):
        m, n = _balanced_pair(
            fraction * basis.accumulator_bytes / basis.accumulate_bytes, basis.tile_rows, basis.tile_cols
        )
        k = _round_tile(fraction * basis.operand_store_bytes / ((m + n) * basis.operand_bytes), basis.tile_rows)
        point = _gemm_point(
            basis,
            variant="joint",
            regime=None,
            x_name="useful_macs",
            m=m,
            k=k,
            n=n,
            why=(
                f"output tile sized to {fraction:g}x the accumulator and operand slab to {fraction:g}x "
                "the operand store"
            ),
        )
        point["label"] = f"x{fraction:g}"
        points.append(point)
    if spec.get("knee_neighbours"):
        m, n = _balanced_pair(basis.accumulator_bytes / basis.accumulate_bytes, basis.tile_rows, basis.tile_cols)
        k_cap = basis.operand_store_bytes // ((m + n) * basis.operand_bytes)
        # The deepest tile-multiple K strictly inside the store's knee, and the first one past it.
        below = ((k_cap - 1) // basis.tile_rows) * basis.tile_rows
        above = (k_cap // basis.tile_rows + 1) * basis.tile_rows
        for label, k in (("just_below", below), ("just_above", above)):
            point = _gemm_point(
                basis,
                variant="joint",
                regime=None,
                x_name="useful_macs",
                m=m,
                k=k,
                n=n,
                why=f"one reduction tile {label.replace('_', ' ')} the operand-store knee at K={k_cap}",
            )
            point["label"] = label
            points.append(point)
    return points


def _large_k(spec, basis: Basis, owner: str) -> list[dict]:
    """A resident output tile at one loop context's accumulator share, reduced over many panels."""
    side = isqrt(basis.accumulator_loop_bytes // basis.accumulate_bytes)
    m = max(basis.tile_rows, (side // basis.tile_rows) * basis.tile_rows)
    n = max(basis.tile_cols, (side // basis.tile_cols) * basis.tile_cols)
    k_cap = basis.operand_store_bytes / ((m + n) * basis.operand_bytes)
    return [
        _gemm_point(
            basis,
            variant="panels",
            regime=None,
            x_name="K",
            m=m,
            k=_round_tile(f * k_cap, basis.tile_rows),
            n=n,
            why=(
                f"output {m}x{n} fills one loop context's accumulator share; K at {f:g}x the depth "
                f"({k_cap:.0f}) at which its operand slab fills the operand store: "
                f"{_round_tile(f * k_cap, basis.tile_rows) // basis.tile_rows} reduction panels"
            ),
        )
        for f in _fractions(spec, "operand_store_multiples", owner=owner)
    ]


def _short_m(spec, basis: Basis, owner: str) -> list[dict]:
    """Short-M / large-N: the weight dominates, so each weight byte feeds at most M MACs."""
    k = _extent(spec.get("K"), basis)
    points = []
    for m_token in spec.get("M") or []:
        m = _extent(m_token, basis)
        for f in _fractions(spec, "weight_store_multiples", owner=owner):
            n = _round_tile(f * basis.operand_store_bytes / (k * basis.operand_bytes), basis.tile_cols)
            point = _gemm_point(
                basis,
                variant=f"m{m}",
                regime=None,
                x_name="N",
                m=m,
                k=k,
                n=n,
                why=(
                    f"M={m} streams {k}x{n} weights at {f:g}x the operand store; arithmetic intensity is at most "
                    f"{m} MAC per weight byte against a {basis.peak_macs_per_cycle} MAC/cycle array and a "
                    f"{basis.read_bytes_per_cycle:g} byte/cycle read port, so movement bounds it"
                ),
            )
            points.append(point)
    return points


def _tails(spec, basis: Basis, owner: str) -> list[dict]:
    """The same shapes with one extent one element off the array edge: padded work and partial beats."""
    points = []
    for f in _fractions(spec, "capacity_multiples", owner=owner):
        m, n = _balanced_pair(f * basis.accumulator_bytes / basis.accumulate_bytes, basis.tile_rows, basis.tile_cols)
        k = _round_tile(f * basis.operand_store_bytes / ((m + n) * basis.operand_bytes), basis.tile_rows)
        for variant, (dm, dk, dn) in (
            ("aligned", (0, 0, 0)),
            ("m_tail", (1, 0, 0)),
            ("k_tail", (0, 1, 0)),
            ("n_tail", (0, 0, -1)),
            ("all_tails", (1, 1, -1)),
        ):
            points.append(
                _gemm_point(
                    basis,
                    variant=variant,
                    regime=variant,
                    x_name="useful_macs",
                    m=m + dm,
                    k=k + dk,
                    n=n + dn,
                    why=(
                        f"{variant} at {f:g}x capacity: extents off the {basis.tile_rows}-wide array edge "
                        "pay padded MACs and partial beats"
                    ),
                )
            )
    return points


def _conv(spec, basis: Basis, owner: str) -> list[dict]:
    """Strided, padded, pointwise and stem convolutions with large channels and image sizes."""
    from merlin.perf.staged_roofline import Contraction

    points = []
    fractions = _fractions(spec, "image_store_multiples", owner=owner)
    for window in spec.get("windows") or []:
        name = str(window["name"])
        kh, kw, stride, pad = (int(window[key]) for key in ("kh", "kw", "stride", "pad"))
        ci, cout = _extent(window["ci"], basis), _extent(window["cout"], basis)
        for f in fractions:
            side = max(kh, isqrt(int(f * basis.operand_store_bytes / (ci * basis.operand_bytes))))
            ho, wo = (side + 2 * pad - kh) // stride + 1, (side + 2 * pad - kw) // stride + 1
            m, k = ho * wo, ci * kh * kw
            image = side * side * ci * basis.operand_bytes
            weight = k * cout * basis.operand_bytes
            accumulator = m * cout * basis.accumulate_bytes
            band = residency(basis, image + weight, accumulator)
            points.append(
                _point(
                    # One fit per window: the image sizes span residency on purpose, and the
                    # residency of each point travels in its working set.
                    variant=name,
                    regime=name,
                    x=m * k * cout,
                    x_name="useful_macs",
                    entry={
                        "op": "conv2d",
                        "ci": ci,
                        "Himg": side,
                        "Wimg": side,
                        "kh": kh,
                        "kw": kw,
                        "stride": [stride, stride],
                        "padding": [pad, pad, pad, pad],
                        "N": cout,
                        "epilogue": [],
                    },
                    work=[Contraction(m, k, cout, lhs_bytes=image)],
                    result_bytes=basis.accumulate_bytes,
                    why=(
                        f"{kh}x{kw}/s{stride}/p{pad} window, {ci} input channels on a {side}x{side} image "
                        f"({image} bytes, {f:g}x the operand store): im2col contraction {m}x{k}x{cout}, "
                        f"residency {band}"
                    ),
                    working_set={"operand_bytes": image + weight, "accumulator_bytes": accumulator, "residency": band},
                )
            )
    return points


def _epilogue(spec, basis: Basis, owner: str) -> list[dict]:
    """One reduction panel and a large output, so the readout path and its width decide the cost."""
    from merlin.perf.staged_roofline import Contraction

    k = _extent(spec.get("K"), basis)
    points = []
    for variant in spec.get("variants") or []:
        stages = list(variant.get("epilogue") or [])
        narrow = bool(set(stages) & {"acc_scale", "requant"})
        result_bytes = basis.operand_bytes if narrow else basis.accumulate_bytes
        for f in _fractions(spec, "accumulator_multiples", owner=owner):
            m, n = _balanced_pair(
                f * basis.accumulator_bytes / basis.accumulate_bytes, basis.tile_rows, basis.tile_cols
            )
            entry = _matmul(m, k, n, stages)
            entry.update(copy.deepcopy(variant.get("attributes") or {}))
            points.append(
                _point(
                    variant=str(variant["name"]),
                    regime=str(variant["name"]),
                    x=m * n,
                    x_name="output_elements",
                    entry=entry,
                    work=[Contraction(m, k, n)],
                    result_bytes=result_bytes,
                    why=(
                        f"epilogue {stages or ['none: full-width accumulator readout']} over a {m}x{n} output "
                        f"({f:g}x the accumulator) written at {result_bytes} byte(s) per element"
                    ),
                    working_set={"accumulator_bytes": m * n * basis.accumulate_bytes},
                )
            )
    return points


def _residency_chain(spec, basis: Basis, owner: str) -> list[dict]:
    """One weight reused by a chain of contractions: residency that persists across ops, or cannot."""
    from merlin.perf.staged_roofline import Contraction

    m = _extent(spec.get("M"), basis)
    n = _extent(spec.get("N"), basis)
    counts = [int(c) for c in spec.get("op_counts") or []]
    if len(counts) < 2 or any(c < 1 for c in counts):
        raise ValueError(f"{owner}: op_counts must list at least two positive chain lengths")
    points = []
    for f in _fractions(spec, "weight_store_multiples", owner=owner):
        k = _round_tile(f * basis.operand_loop_bytes / (n * basis.operand_bytes), basis.tile_rows)
        weight = k * n * basis.operand_bytes
        band = "resident" if weight + m * k * basis.operand_bytes <= basis.operand_loop_bytes else "reloaded"
        for count in counts:
            work = [Contraction(m, k, n, rhs_resident=(i > 0 and band == "resident")) for i in range(count)]
            points.append(
                _point(
                    variant=f"w{f:g}",
                    regime=band,
                    x=count,
                    x_name="op_count",
                    entry={
                        "op": "resident_reuse",
                        "weight": "W",
                        "K": k,
                        "N": n,
                        "matmuls": [{"lhs": f"A{i}", "out": f"Y{i}", "M": m} for i in range(count)],
                    },
                    work=work,
                    result_bytes=basis.accumulate_bytes,
                    why=(
                        f"{count} contraction(s) share a {k}x{n} weight ({weight} bytes, {f:g}x one loop context's "
                        f"operand share): the weight can stay {band} across ops"
                    ),
                    working_set={"operand_bytes": weight, "residency": band},
                )
            )
    return points


def _queue_pressure(spec, basis: Basis, owner: str) -> list[dict]:
    """One reduction panel per output tile, so issue, not arithmetic, bounds the stream."""
    if not basis.command_windows:
        raise CapacityUnavailable("no command class states its live window, so back-pressure is not placeable")
    name, window = min(basis.command_windows.items(), key=lambda item: (item[1], item[0]))
    points = []
    for f in _fractions(spec, "window_multiples", owner=owner):
        tiles = max(1, int(round(f * window)))
        m_tiles, n_tiles = _balanced_pair(tiles, 1, 1)
        m, n, k = m_tiles * basis.tile_rows, n_tiles * basis.tile_cols, basis.tile_rows
        point = _gemm_point(
            basis,
            variant="one_panel",
            regime="within_window" if tiles <= window else "back_pressure",
            x_name="output_elements",
            m=m,
            k=k,
            n=n,
            why=(
                f"{m_tiles * n_tiles} output tiles of one reduction panel each, {f:g}x the {window}-command live "
                f"window of class {name!r} (the smallest), so its queue "
                f"{'absorbs' if tiles <= window else 'back-pressures'} the stream"
            ),
        )
        points.append(point)
    return points


def _dataflow(spec, basis: Basis, owner: str) -> list[dict]:
    """Equal work at different aspect ratios, where weight- and output-stationary choices differ."""
    modes = set((basis.dataflow.get("modes") or {}).values())
    if basis.dataflow.get("status") != "derived" or len(modes) < 2 or not basis.dataflow.get("select_runtime_driven"):
        raise CapacityUnavailable("the facts do not show a runtime-selected choice between two dataflow modes")
    base = _extent(spec.get("base_extent"), basis)
    points = []
    for f in _fractions(spec, "work_multiples", owner=owner):
        tiles = max(1, int(round(f)))
        for variant, (m, k, n) in (
            ("rows_heavy", (base * tiles * tiles, basis.tile_rows, base)),
            ("depth_heavy", (base, basis.tile_rows * tiles * tiles, base)),
            ("balanced", (base * tiles, basis.tile_rows * tiles, base)),
        ):
            points.append(
                _gemm_point(
                    basis,
                    variant=variant,
                    regime=variant,
                    x_name="useful_macs",
                    m=m,
                    k=k,
                    n=n,
                    why=(
                        f"{variant}: modes {sorted(modes)} reuse a different operand in the array, so the "
                        "better mode differs by aspect"
                    ),
                )
            )
    return points


KINDS = {
    "capacity_knee": _capacity_knee,
    "large_k": _large_k,
    "short_m": _short_m,
    "tails": _tails,
    "conv": _conv,
    "epilogue": _epilogue,
    "residency_chain": _residency_chain,
    "queue_pressure": _queue_pressure,
    "dataflow": _dataflow,
}


# ------------------------------------------------------------------------------------------------
# the template declaration and the generated members
# ------------------------------------------------------------------------------------------------
def validate_capacity_scope_declaration(sweep: Mapping[str, Any], *, owner: str) -> dict[str, Any]:
    """The template family's declared kinds and roofline target, checked; the family owes no axes."""
    declared = sweep.get("requires_capacity_scope")
    if not isinstance(declared, Mapping):
        raise ValueError(f"{owner}: requires_capacity_scope must be a mapping")
    if sweep.get("axes"):
        raise ValueError(f"{owner}: a capacity-scope family takes its extents from the facts, not from axes")
    families = declared.get("families")
    if not isinstance(families, list) or not families:
        raise ValueError(f"{owner}: requires_capacity_scope.families must be a non-empty list")
    seen = set()
    for spec in families:
        if not isinstance(spec, Mapping) or spec.get("kind") not in KINDS:
            raise ValueError(f"{owner}: every capacity family names one of {sorted(KINDS)}")
        name = spec.get("name")
        if not isinstance(name, str) or not name or name in seen:
            raise ValueError(f"{owner}: every capacity family needs a unique name")
        seen.add(name)
        for key in ("exercises", "why"):
            if not isinstance(spec.get(key), str) or not spec[key].strip():
                raise ValueError(f"{owner}: capacity family {name} must state what it {key}")
    target = declared.get("roofline_target")
    within = target.get("attainment_min") if isinstance(target, Mapping) else None
    if isinstance(within, bool) or not isinstance(within, (int, float)) or not 0.0 < float(within) <= 1.0:
        raise ValueError(f"{owner}: requires_capacity_scope.roofline_target.attainment_min must be in (0, 1]")
    return dict(declared)


def derive_points(
    declaration: Mapping[str, Any], basis: Basis, *, owner: str, min_points: int
) -> tuple[list[dict], list[dict]]:
    """``(points, unavailable)``: every family's derived points, and what the facts cannot host.

    A regime that lands fewer than ``min_points`` distinct regime-variable values on this target
    cannot be fitted; its points are withheld and reported, so the claim is never decided on a group
    that cannot identify a rate and an intercept.
    """
    points, unavailable = [], []
    for spec in declaration["families"]:
        try:
            found = KINDS[spec["kind"]](spec, basis, f"{owner} family {spec['name']}")
        except CapacityUnavailable as exc:
            unavailable.append({"family": spec["name"], "kind": spec["kind"], "reason": str(exc)})
            continue
        groups: dict[tuple[str, str], set[float]] = {}
        unique, shapes = [], set()
        for point in found:
            shape = json.dumps(point["entry"], sort_keys=True)
            if shape in shapes:
                continue  # two policies landing on one shape measure it once
            shapes.add(shape)
            point["family"] = spec["name"]
            point["kind"] = spec["kind"]
            point["exercises"] = spec["exercises"]
            point.setdefault("label", f"{point['variant']}_{point['x_name']}{int(point['x'])}")
            groups.setdefault((point["variant"], point["regime"]), set()).add(float(point["x"]))
            unique.append(point)
        thin = {key for key, xs in groups.items() if len(xs) < min_points}
        for variant, regime in sorted(thin):
            unavailable.append(
                {
                    "family": spec["name"],
                    "kind": spec["kind"],
                    "variant": variant,
                    "regime": regime,
                    "reason": f"only {len(groups[(variant, regime)])} point(s) land in this regime on this target",
                }
            )
        points.extend(p for p in unique if (p["variant"], p["regime"]) not in thin)
    return points, unavailable


def _roofline(point: Mapping[str, Any], basis: Basis) -> dict[str, Any]:
    from merlin.perf.staged_roofline import staged_roofline

    doc = staged_roofline(point["work"], basis.machine, basis.ports, result_bytes=int(point["result_bytes"]))
    return {
        "status": doc["status"],
        "floor_cycles": doc["floor_cycles"],
        "limiter": doc["limiter"],
        "attainable_useful_macs_per_cycle": doc["attainable_useful_macs_per_cycle"],
        "compute": {k: doc["compute"].get(k) for k in ("useful_macs", "padded_macs", "padding_efficiency")},
        "movement": {k: doc["movement"].get(k) for k in ("unique_bytes", "requested_bytes", "physical_bytes")},
        "composition": doc["composition"],
        "unresolved": sorted(doc["unresolved"]),
    }


def capacity_family_entries(
    sweep: Mapping[str, Any],
    *,
    target: str,
    facts: Mapping[str, Any] | None,
    skipped: list | None = None,
) -> list[dict[str, Any]]:
    """Members of one capacity-scope template family, each a flat corpus entry with its regime."""
    family = str(sweep.get("id") or "")
    owner = f"performance sweep {family}"
    declaration = validate_capacity_scope_declaration(sweep, owner=owner)
    if facts is None:
        from merlin.targetgen.rtl import facts as rtl_facts

        facts = rtl_facts.load_facts(target)
    try:
        basis = derive_basis(target, facts)
    except CapacityUnavailable as exc:
        if skipped is not None:
            skipped.append({"family": family, "sweep": family, "status": "skipped_inapplicable", "reason": str(exc)})
        return []
    points, unavailable = derive_points(
        declaration, basis, owner=owner, min_points=min_points_per_regime(sweep.get("base") or {}, owner=owner)
    )
    for row in unavailable:
        if skipped is not None:
            skipped.append({"family": family, "sweep": family, "status": "skipped_inapplicable", **row})
    base = copy.deepcopy(sweep.get("base") or {})
    out = []
    digest = basis.digest()
    for index, point in enumerate(points):
        entry = copy.deepcopy(base)
        entry.update(copy.deepcopy(point["entry"]))
        entry["name"] = f"{family}{index:02d}_{point['family']}_{point.get('label') or point['variant']}"
        entry["source_role"] = "derived_sweep"
        entry["source_reference"] = f"{sweep.get('source_reference') or 'capacity family'}; {point['why']}"
        entry["performance"][CAPACITY_BLOCK] = {
            "schema": SCHEMA,
            "family": point["family"],
            "kind": point["kind"],
            "exercises": point["exercises"],
            "variant": point["variant"],
            "regime": point["regime"],
            "x": point["x"],
            "x_name": point["x_name"],
            "why": point["why"],
            "working_set": point["working_set"],
            "roofline": _roofline(point, basis),
            "roofline_target": dict(declaration["roofline_target"]),
            "basis_sha256": digest,
        }
        out.append(entry)
    return out


def scope_members(sweep, *, binding, evidence, requirement, entries, skipped, blocked) -> list[dict[str, Any]]:
    """Members of a derived-scope family: form-perf from the frozen requirement, capacity from the facts.

    Neither kind takes axes; both reach the same direct corpus path in ``sweeps.expand_sweeps``.
    """
    if sweep.get("requires_form_scope") is not None:
        from .form_perf import form_perf_entries

        selected, digest = requirement
        return form_perf_entries(
            sweep, selected, digest, source_window_entries=entries, skipped=skipped, blocked=blocked
        )
    return capacity_family_entries(
        sweep,
        target=str(getattr(binding, "target", "") or ""),
        facts=getattr(evidence, "refreshed_facts", None),
        skipped=skipped,
    )


def min_points_per_regime(base: Mapping[str, Any], *, owner: str) -> int:
    """The frozen contract's points-per-regime minimum, which generation must already satisfy."""
    cohort = (((base.get("performance") or {}).get("acceptance") or {}).get("cohort")) or {}
    value = cohort.get("min_points_per_regime")
    if isinstance(value, bool) or not isinstance(value, int) or value < 2:
        raise ValueError(f"{owner}: acceptance.cohort.min_points_per_regime must be an int >= 2")
    return value


def cohort_report(target: str, facts: Mapping[str, Any], sweep: Mapping[str, Any]) -> dict[str, Any]:
    """A readable account of the cohort a target's facts produce: basis, families, points and regimes."""
    owner = "capacity cohort report"
    declaration = validate_capacity_scope_declaration(sweep, owner=owner)
    basis = derive_basis(target, facts)
    points, unavailable = derive_points(
        declaration, basis, owner=owner, min_points=min_points_per_regime(sweep.get("base") or {}, owner=owner)
    )
    return {
        "schema": SCHEMA,
        "basis": basis.to_dict(),
        "basis_sha256": basis.digest(),
        "points": [
            {
                "family": p["family"],
                "variant": p["variant"],
                "label": p.get("label") or p["variant"],
                "regime": p["regime"],
                "x_name": p["x_name"],
                "x": p["x"],
                "entry": p["entry"]
                if p["entry"].get("op") != "resident_reuse"
                else {
                    **{k: v for k, v in p["entry"].items() if k != "matmuls"},
                    "chain": len(p["entry"]["matmuls"]),
                    "M": p["entry"]["matmuls"][0]["M"],
                },
                "roofline": {k: v for k, v in _roofline(p, basis).items() if k in ("floor_cycles", "limiter")},
                "why": p["why"],
            }
            for p in points
        ],
        "unavailable": unavailable,
    }
