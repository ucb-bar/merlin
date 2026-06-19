"""Tool A (P20): Timeloop-native mapspace seeds from the recovered operator graph.

Turns the recovered contraction ops (linear-GEMM / attention / batched-matmul, each with M/N/K + operand
identities + multiplicity, from ``attribution``) into **Timeloop problem shapes** — the einsum / dimensions
/ data-space-projection schema a Timeloop+Accelergy DSE run ingests — plus the structural dataflow-candidate
space (weight/output/input-stationary) and operand bypass candidates.

This is a SEARCH-SPACE SEED for a future DSE tool, not DSE: no speedup / cycles / area / energy / mapping is
chosen. Magnitudes are structural-only. Every shape cites its source op (``prov.fqn`` + the inventory).

Grounding (P19 audit): attention contractions have BOTH operands as activations (no stationary weight) ->
weight_stationary is excluded; linear/batched have a stationary operand. K is the reduction dimension ->
spatially partitioning K needs a partial-sum datapath (flagged). What a *full* problem needs but the flat
capture lacks (the K denoise/decode loop, attention heads/seq, conv windows, bit-width) is recorded in
``mapspace_seed_gaps``.
"""
from __future__ import annotations

from collections import defaultdict

from merlin.dse_guidance import attribution as ATTR

# op_class -> (operand_b identity, dataflow candidates). Attention = both activations (no WS).
_OPB = {
    "linear_gemm": ("weight", ["weight_stationary", "output_stationary", "input_stationary"]),
    "batched_matmul": ("batched_weight_or_state",
                       ["weight_stationary", "output_stationary", "input_stationary"]),
    "attention_contraction": ("activation", ["output_stationary", "input_stationary"]),
}
_DATAFLOW_COL = ["workload", "op_class", "shape_id", "M", "N", "K", "multiplicity", "macs_per_instance",
                 "operand_b_identity", "dataflow_candidates", "legal_spatial_axes",
                 "legal_temporal_axes", "bypass_candidates", "k_reduction_needs_partial_sum", "fqn"]


def _distinct_shapes(workloads):
    """workloads = [(name, capture_dir)]; group contraction ops by (workload, op_class, M, K, N).
    Returns ordered list of dicts with multiplicity + macs + a representative fqn."""
    acc = defaultdict(lambda: {"mult": 0, "macs": 0, "fqn": ""})
    for name, cap in workloads:
        for r in ATTR.extract_matmuls(cap):                       # linear-GEMM (linalg.matmul)
            if r.M > 0 and r.K > 0 and r.N > 0:
                k = (name, "linear_gemm", r.M, r.K, r.N)
                a = acc[k]
                a["mult"] += 1
                a["macs"] += r.macs
                a["fqn"] = a["fqn"] or (r.fqn or "")
        for r in ATTR.extract_non_gemm_ops(cap):                 # attention / batched (from generics)
            if r.op_class in ("attention_contraction", "batched_matmul") and r.M > 0 and r.K > 0 and r.N > 0:
                k = (name, r.op_class, r.M, r.K, r.N)
                a = acc[k]
                a["mult"] += 1
                a["macs"] += r.macs
                a["fqn"] = a["fqn"] or (r.fqn or "")
    out = []
    for (name, oc, M, K, N), a in sorted(acc.items(), key=lambda kv: (kv[0][0], kv[0][1], -kv[1]["macs"])):
        out.append({"workload": name, "op_class": oc, "M": M, "K": K, "N": N,
                    "multiplicity": a["mult"], "macs": a["macs"], "fqn": a["fqn"]})
    return out


def _problem(s):
    """A Timeloop problem-shape dict for one distinct shape. Standard GEMM einsum
    Outputs[M,N] = Inputs[M,K] * Weights[K,N]; attention operand B is an activation (annotated)."""
    opb_id, _ = _OPB.get(s["op_class"], ("weight", ["output_stationary"]))
    return {
        "shape": {
            "name": "matmul",
            "dimensions": ["M", "N", "K"],
            "data-spaces": [
                {"name": "Weights", "projection": [[["K"]], [["N"]]], "operand_identity": opb_id},
                {"name": "Inputs", "projection": [[["M"]], [["K"]]], "operand_identity": "activation"},
                {"name": "Outputs", "projection": [[["M"]], [["N"]]], "read-write": True,
                 "operand_identity": "output"},
            ],
        },
        "instance": {"M": s["M"], "N": s["N"], "K": s["K"]},
    }


def problem_shapes(workloads) -> dict:
    """Timeloop-ingestible problem shapes (one per distinct contraction shape) + provenance. Returns the
    object to dump as ``timeloop_problem_shapes.yaml``."""
    shapes = _distinct_shapes(workloads)
    items = []
    for s in shapes:
        opb_id, dataflow = _OPB.get(s["op_class"], ("weight", ["output_stationary"]))
        items.append({
            "id": f"{s['workload']}__{s['op_class']}__{s['M']}x{s['N']}x{s['K']}",
            "workload": s["workload"], "op_class": s["op_class"],
            "multiplicity": s["multiplicity"], "macs_per_instance": s["M"] * s["N"] * s["K"],
            "problem": _problem(s),
            "dataflow_candidates": dataflow,
            "bypass_candidates": {"Weights": "possible", "Inputs": "possible", "Outputs": "possible"},
            "k_reduction_needs_partial_sum": True,
            "evidence": {"fqn": s["fqn"], "source": "operator_full_inventory.csv"},
            "note": "structural search-space seed; no mapping/perf chosen",
        })
    return {"timeloop_problem_shapes": {"count": len(items), "shapes": items}}


def dataflow_rows(workloads) -> list[dict]:
    """Flat CSV view: per distinct shape, the dataflow/bypass candidate space + legal axes."""
    rows = []
    for s in _distinct_shapes(workloads):
        opb_id, dataflow = _OPB.get(s["op_class"], ("weight", ["output_stationary"]))
        rows.append({
            "workload": s["workload"], "op_class": s["op_class"],
            "shape_id": f"{s['M']}x{s['N']}x{s['K']}", "M": s["M"], "N": s["N"], "K": s["K"],
            "multiplicity": s["multiplicity"], "macs_per_instance": s["M"] * s["N"] * s["K"],
            "operand_b_identity": opb_id, "dataflow_candidates": "|".join(dataflow),
            "legal_spatial_axes": "M|N|K_with_reduction", "legal_temporal_axes": "M|N|K",
            "bypass_candidates": "Weights|Inputs|Outputs", "k_reduction_needs_partial_sum": "True",
            "fqn": s["fqn"]})
    return rows
