"""Per-group HEADROOM: measured cycles against the same-machine reference, and two DERIVED bounds --
never a suggested fix, only where a group sits relative to what the target's own facts say is
achievable.

Two bounds, both roofline-simple by design (this is a per-group DIAGNOSTIC, not the tiling-search
achievable bound :mod:`merlin.analysis.perf.derived_bound` computes for a whole workload search):

* **compute bound** -- ``macs / mesh_macs_per_cycle``, where the mesh's rate is the array geometry
  :func:`merlin.analysis.perf.derived_bound.machine_from_facts` already derives from the target's own RTL
  facts (never a nameplate, never a literal here).
* **data-movement bound** -- ``moved_bytes / dram_bytes_per_cycle``, the same machine's own derived
  DMA rate. ``moved_bytes`` is the group's OWN tensors (image, weight, result) -- never the im2col
  matrix a convolution's unit never materializes, which would over-charge a k>1 tap ninefold (see
  :class:`merlin.analysis.perf.schedule_proxy.Contraction`'s own note on the same trap).

Every rate is read off :class:`merlin.analysis.perf.derived_bound.Machine`; a term whose fact is UNKNOWN drops
out rather than being defaulted, and the group's bound is then reported over whatever terms resolved
-- never silently zero, never a guess. An op this module has no MAC-bearing formula for (a residual
add, a window mean) gets no bound at all: fabricating one would be worse than omitting it.

A group's shape facts are first stated as a :class:`merlin.analysis.perf.schedule_proxy.Contraction`
(:func:`contraction_for`), so the bound above and the RANK signal beside it price the same work.
:func:`group_rank` is the validated schedule cost proxy's answer for that contraction -- array-tile
transactions of compute and movement, ordered against measured hardware -- reported beside the bound
and never folded into it: the proxy establishes an order between groups, not a cycle count.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from merlin.analysis.perf.decompose import is_unknown
from merlin.analysis.perf.derived_bound import Machine, _width_bits, machine_from_facts
from merlin.analysis.perf.schedule_proxy import Contraction, schedule_cost

__all__ = [
    "SCHEMA",
    "contraction_for",
    "group_bound",
    "group_headroom",
    "group_rank",
    "machine_for",
]

SCHEMA = "merlin_group_headroom_v1"

#: Term names, matching :mod:`merlin.analysis.perf.derived_bound`'s vocabulary so a reader of both never sees
#: the same idea spelled two ways.
COMPUTE_TERM = "compute"
TRAFFIC_TERM = "dram_traffic"


def _dtype_bytes(dtype: Any) -> int | None:
    bits = _width_bits(dtype)
    return None if bits is None else -(-bits // 8)  # ceil: a sub-byte width still occupies one byte


def contraction_for(op: Any, facts: Mapping[str, Any], *, label: str = "") -> Contraction | None:
    """One group's shape facts as the contraction it presents, or ``None`` when ``op`` states no
    MAC-bearing contraction this estimate covers, or a needed extent/dtype is missing.

    The footprints are the group's OWN tensors. A convolution's ``m`` is its output positions and its
    ``k`` the window times the input channels, but its ``lhs`` is the image -- never the im2col matrix
    -- and its contiguous output run is one output row, which is what separates a narrow-row stage
    from an equal-MAC one on a fixed-height array."""
    operand_bytes = _dtype_bytes(facts.get("operand_dtype"))
    output_bytes = _dtype_bytes(facts.get("output_dtype")) or operand_bytes
    if operand_bytes is None:
        return None
    if op == "matmul":
        try:
            m, k, n = int(facts["M"]), int(facts["K"]), int(facts["N"])
        except (KeyError, TypeError, ValueError):
            return None
        if min(m, k, n) <= 0:
            return None
        return Contraction(
            m=m,
            n=n,
            k=k,
            lhs_bytes=m * k * operand_bytes,
            rhs_bytes=k * n * operand_bytes,
            result_bytes=m * n * (output_bytes or operand_bytes),
            stream_run=m,
            label=label,
        )
    if op == "conv2d":
        try:
            co, ci = int(facts["N"]), int(facts["ci"])
            himg, wimg = int(facts["Himg"]), int(facts["Wimg"])
            kh, kw = int(facts["kh"]), int(facts["kw"])
            stride = list(facts.get("stride") or (1, 1))
            padding = list(facts.get("padding") or (0, 0, 0, 0))
            sh, sw = int(stride[0]), int(stride[1])
            pad_h, pad_w = int(padding[0]) + int(padding[2]), int(padding[1]) + int(padding[3])
        except (KeyError, TypeError, ValueError, IndexError):
            return None
        if min(co, ci, himg, wimg, kh, kw, sh, sw) <= 0:
            return None
        hout = (himg - kh + pad_h) // sh + 1
        wout = (wimg - kw + pad_w) // sw + 1
        if hout <= 0 or wout <= 0:
            return None
        # The unit's OWN tensors: the image and the weight, never the im2col matrix a native
        # convolution unit forms in its own address stream and no buffer ever holds whole.
        return Contraction(
            m=hout * wout,
            n=co,
            k=ci * kh * kw,
            lhs_bytes=himg * wimg * ci * operand_bytes,
            rhs_bytes=co * ci * kh * kw * operand_bytes,
            result_bytes=hout * wout * co * (output_bytes or operand_bytes),
            stream_run=wout,
            label=label,
        )
    return None


def machine_for(target: str) -> Machine:
    """The target's derived machine facts, without consulting the circuit for a fill/drain depth --
    this diagnostic's two terms (mesh rate, DMA rate) do not need one, and deriving it is the
    expensive half of :func:`merlin.analysis.perf.derived_bound.machine_from_facts`."""
    return machine_from_facts(target, measure_fill=False)


def group_bound(macs: int, moved_bytes: int, machine: Machine) -> dict[str, Any]:
    """The two derived bounds for one group's ``(macs, moved_bytes)``, each dropped when its own
    machine fact is UNKNOWN rather than defaulted. ``bound_cycles`` is the max of what resolved (the
    binding constraint), or ``None`` when neither did."""
    rows, cols, muls = machine.array_rows, machine.array_cols, machine.muls_per_element
    mesh_macs_per_cycle = (
        int(rows) * int(cols) * int(muls) if not (is_unknown(rows) or is_unknown(cols) or is_unknown(muls)) else None
    )
    compute_bound_cycles = macs / mesh_macs_per_cycle if mesh_macs_per_cycle else None
    dram_rate = machine.dram_bytes_per_cycle
    dram_bytes_per_cycle = None if is_unknown(dram_rate) or not dram_rate else dram_rate
    traffic_bound_cycles = moved_bytes / dram_bytes_per_cycle if dram_bytes_per_cycle else None
    terms = {
        name: cycles
        for name, cycles in ((COMPUTE_TERM, compute_bound_cycles), (TRAFFIC_TERM, traffic_bound_cycles))
        if cycles is not None
    }
    return {
        "macs": macs,
        "moved_bytes": moved_bytes,
        "mesh_macs_per_cycle": mesh_macs_per_cycle,
        "dram_bytes_per_cycle": dram_bytes_per_cycle,
        "compute_bound_cycles": compute_bound_cycles,
        "traffic_bound_cycles": traffic_bound_cycles,
        "bound_cycles": max(terms.values()) if terms else None,
        "limiter": max(terms, key=terms.get) if terms else None,
    }


def group_rank(work: Contraction, machine: Machine, *, target: str) -> dict[str, Any]:
    """The schedule cost proxy's RANK signal for one group, priced on ``machine``.

    ``transactions`` orders groups by what they ask the device for and is ``None`` when the machine's
    own facts do not ground a transaction (the reasons travel in ``unresolved``). It is never a cycle
    count: the proxy is validated as an order, and its magnitude error is large."""
    cost = schedule_cost(work, target=target, machine=machine, workload=work.name)
    return {
        "transactions": round(cost.transactions, 4) if cost.resolved else None,
        "terms": {name: round(value, 4) for name, value in cost.terms.items()},
        "regime": cost.regime,
        "unresolved": {name: cost.reasons[name] for name in cost.unresolved},
        "pricing": cost.pricing.provenance,
    }


def group_headroom(
    *,
    op: Any,
    shape_facts: Mapping[str, Any] | None,
    ours_cycles: Any,
    reference_cycles: Any,
    machine: Machine,
    target: str | None = None,
) -> dict[str, Any] | None:
    """One group's full headroom document, or ``None`` when its op/shape facts admit no MAC-bearing
    estimate. ``headroom_cycles`` is ``ours - reference`` (never derived, the measurement already
    states both); ``over_bound`` is ``ours / bound_cycles`` when the bound resolved. With ``target``,
    the document also carries the group's :func:`group_rank`."""
    if not isinstance(shape_facts, Mapping) or not isinstance(ours_cycles, int):
        return None
    work = contraction_for(op, shape_facts)
    if work is None:
        return None
    bound = group_bound(work.macs, work.moved_bytes, machine)
    document: dict[str, Any] = {"schema": SCHEMA, **bound}
    if isinstance(reference_cycles, int):
        document["headroom_cycles"] = ours_cycles - reference_cycles
    if bound["bound_cycles"]:
        document["over_bound"] = round(ours_cycles / bound["bound_cycles"], 4)
    if target is not None:
        document["rank"] = group_rank(work, machine, target=target)
    return document
