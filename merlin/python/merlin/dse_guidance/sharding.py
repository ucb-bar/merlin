"""Intra-operator sharding analysis — how each matmul can be split across N candidate units.

For ``C[M,N] = A[M,K] · B[K,N]`` there are three structural sharding axes, each with a different
communication signature:

* **M-sharding** — split output rows. Rows are independent, **no reduction**; the weight matrix B
  (K×N) is broadcast/duplicated to every unit; outputs are concatenated. Needs ``weight_broadcast``
  + ``output_partition_commit``.
* **N-sharding** — split output columns. Columns are independent, **no reduction**; the activation A
  (M×K) is broadcast/duplicated; outputs are concatenated. Needs ``activation_multicast`` +
  ``output_partition_commit``.
* **K-sharding** — split the reduction dimension. Each shard produces a **partial sum** (M×N) that
  must be merged; this adds cross-shard reduction traffic. Needs ``partial_sum_object`` +
  ``accumulator_merge`` (the high-communication mode).

For every matmul and each candidate unit count (2, 4, 8) this records whether the split is possible,
whether it leaves a tail (uneven split), the per-extra-shard byte cost (duplicated input bytes for
M/N, partial-sum bytes for K), the communication category, and the required communication/reduction
abstraction. Attention/conv sharding is reported ``unavailable`` because that structure is lowered
into the matmul projections in the flat capture (it is not invented). No speedup, no cycle claim.
"""
from __future__ import annotations

from dataclasses import dataclass

from merlin.dse_guidance.design_envelope import E_DERIVED, E_IR, E_NA, ELEMENT_BYTES

UNIT_COUNTS = (2, 4, 8)
ACC_BYTES = 4.0          # i32 / f32 accumulator (partial-sum element size)

# axis -> (reduction?, comm category, required abstractions)
_AXIS = {
    "M": (False, "low", ["weight_broadcast", "output_partition_commit"]),
    "N": (False, "low", ["activation_multicast", "output_partition_commit"]),
    "K": (True, "high", ["partial_sum_object", "accumulator_merge"]),
}


@dataclass
class ShardAxis:
    workload: str
    op_index: int
    shape_class: str
    semantic_class: str
    axis: str                       # M | N | K
    dim_size: int
    shardable: dict                 # unit -> bool
    has_tail: dict                  # unit -> bool
    reduction_required: bool
    comm_category: str
    per_extra_shard_bytes: int      # duplicated input (M/N) or partial-sum (K) bytes per extra shard
    required_abstractions: list


def shard_axes(shape) -> list[ShardAxis]:
    """The three sharding axes for one matmul-like operator (from its M/N/K geometry)."""
    M, N, K = shape.M, shape.N, shape.K
    elem = ELEMENT_BYTES.get(str(shape.dtype or "f32").strip().lower(), 4.0)
    sizes = {"M": M, "N": N, "K": K}
    # per-extra-shard byte cost: M dup weights(K*N), N dup activations(M*K), K partial sums(M*N*acc)
    extra = {"M": int(K * N * elem), "N": int(M * K * elem), "K": int(M * N * ACC_BYTES)}
    out = []
    for axis, dim in sizes.items():
        reduction, comm, absts = _AXIS[axis]
        out.append(ShardAxis(
            workload=shape.workload, op_index=shape.op_index, shape_class=shape.shape_class,
            semantic_class=shape.semantic_class, axis=axis, dim_size=dim,
            shardable={u: dim >= u for u in UNIT_COUNTS},
            has_tail={u: (dim % u != 0) for u in UNIT_COUNTS},
            reduction_required=reduction, comm_category=comm,
            per_extra_shard_bytes=extra[axis], required_abstractions=absts))
    return out


def all_shard_axes(shapes) -> list[ShardAxis]:
    out = []
    for s in shapes:
        out.extend(shard_axes(s))
    return out


# --------------------------------------------------------------------------- emitters

def sharding_csv(axes: list[ShardAxis]) -> str:
    from merlin.dse_guidance.corpus import _csv
    rows = []
    for a in axes:
        rows.append({
            "workload": a.workload, "op_index": a.op_index, "shape_class": a.shape_class,
            "semantic_class": a.semantic_class, "axis": a.axis, "dim_size": a.dim_size,
            "shardable_2": a.shardable[2], "shardable_4": a.shardable[4],
            "shardable_8": a.shardable[8],
            "tail_2": a.has_tail[2], "tail_4": a.has_tail[4], "tail_8": a.has_tail[8],
            "reduction_required": a.reduction_required, "comm_category": a.comm_category,
            "per_extra_shard_bytes": a.per_extra_shard_bytes,
            "required_abstractions": "; ".join(a.required_abstractions),
        })
    return _csv(rows, ["workload", "op_index", "shape_class", "semantic_class", "axis", "dim_size",
                       "shardable_2", "shardable_4", "shardable_8", "tail_2", "tail_4", "tail_8",
                       "reduction_required", "comm_category", "per_extra_shard_bytes",
                       "required_abstractions"])


def sharding_opportunities_yaml(by_workload: dict) -> dict:
    """Per-workload sharding summary: which axes split cleanly, and the abstractions implied."""
    from collections import Counter
    workloads = []
    needed_abstractions: set = set()
    for wl, axes in by_workload.items():
        clean8 = Counter()      # ops that shard 8-ways with NO tail, per axis
        tail8 = Counter()
        for a in axes:
            if a.shardable[8] and not a.has_tail[8]:
                clean8[a.axis] += 1
            elif a.shardable[8] and a.has_tail[8]:
                tail8[a.axis] += 1
            for ab in a.required_abstractions:
                needed_abstractions.add(ab)
        workloads.append({
            "workload": wl,
            "ops": len({a.op_index for a in axes}),
            "clean_8way_shards": {ax: clean8.get(ax, 0) for ax in ("M", "N", "K")},
            "tailed_8way_shards": {ax: tail8.get(ax, 0) for ax in ("M", "N", "K")},
            "evidence": E_IR})
    return {"sharding_opportunities": {
        "note": "intra-operator sharding geometry for matmuls. M/N sharding has NO reduction "
                "(broadcast input + concat output); K sharding requires partial sums + an "
                "accumulator merge (the high-communication mode). All byte costs are derived from "
                "recovered_from_ir shapes. No speedup/cycle claim.",
        "attention_sharding": {"value": "unavailable", "evidence": E_NA,
                               "reason": "attention is lowered into matmul projections; head / "
                                         "query / KV / sequence sharding structure is not visible"},
        "conv_sharding": {"value": "unavailable", "evidence": E_NA,
                          "reason": "no linalg.conv* ops in the captures"},
        "required_abstractions": sorted(needed_abstractions),
        "workloads": workloads}}


def report_md(by_workload: dict, axes_all: list[ShardAxis]) -> str:
    from collections import Counter
    L = ["# Intra-op sharding report\n",
         "> How each matmul can be split across 2/4/8 candidate units along M (rows), N (columns), "
         "or K (reduction). M/N sharding is reduction-free (broadcast + concat); K sharding needs "
         "partial sums + an accumulator merge. **Structural geometry only — no speedup, no cycle "
         "claim.**\n"]
    L.append("## Clean 8-way shardability (no tail) by axis\n")
    L.append("| workload | M (rows) | N (cols) | K (reduction) |")
    L.append("|---|---|---|---|")
    for wl, axes in by_workload.items():
        clean = Counter(a.axis for a in axes if a.shardable[8] and not a.has_tail[8])
        L.append(f"| {wl} | {clean.get('M',0)} | {clean.get('N',0)} | {clean.get('K',0)} |")
    L.append("")
    # which mode is cheapest structurally (no reduction) dominates
    nored = sum(1 for a in axes_all if not a.reduction_required and a.shardable[8])
    kshard = sum(1 for a in axes_all if a.reduction_required and a.shardable[8])
    L.append("## Findings\n")
    L.append(f"- **Reduction-free sharding dominates:** {nored} (op,axis) M/N opportunities split "
             f"without any cross-shard reduction — only `weight_broadcast`/`activation_multicast` + "
             f"`output_partition_commit`.")
    L.append(f"- **K-sharding is the high-communication mode:** {kshard} (op,axis) opportunities "
             f"would need a `partial_sum_object` + `accumulator_merge`; the partial-sum bytes are "
             f"in `sharding_table.csv`.")
    L.append("- **Attention / conv sharding:** `unavailable` — that structure is lowered into the "
             "matmul projections and is not invented.")
    L.append("\n**Caveat (structural, not realized):** these are sharding *geometries* and their "
             "byte costs. They are **not a speedup**, latency, or throughput claim, and assume no "
             "hardware.\n")
    return "\n".join(L)
