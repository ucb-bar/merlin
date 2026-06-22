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


# ======================================================================================
# P21-aware mapspace seeds: the SAME problem shapes, but enriched with the loop facts the
# flat seed could not carry (see mapspace_seed_gaps.md). The loop-preserving capture gives
# us, per shape: the region role (prefix-once vs the x K repeated head), the IR-recovered
# trip count K (so the TRUE per-replan instance count is layers x K, not layers), and the
# loop-carried KV state as a resident-state node. Built from the committed artifacts
# (operator_shape_table.csv + loop_aware_contract.csv), so it is re-derivable + verifiable.
# Structural search-space seed only: no mapping / cycles / perf is chosen.
# ======================================================================================
_OPCLASS_FROM_KIND = {"matmul": "linear_gemm", "addmm": "linear_gemm", "linear": "linear_gemm",
                      "mm": "linear_gemm", "bmm": "batched_matmul", "baddbmm": "batched_matmul"}
_LOOP_SEED_COLS = ["workload", "op_class", "region_role", "M", "N", "K", "layers_in_capture",
                   "outer_loop_K", "K_source", "instances_per_replan", "macs_per_instance",
                   "macs_per_replan", "operand_b_identity", "dataflow_candidates",
                   "reduction_reuse_scope", "kv_state_bytes_workload", "k_reduction_needs_partial_sum",
                   "evidence"]


def _read_csv(path):
    import csv as _c
    from pathlib import Path
    p = Path(path)
    return list(_c.DictReader(p.read_text().splitlines())) if p.is_file() else []


def _loop_facts(cs_dir):
    """workload -> {K, K_source, kv_bytes, roles} from loop_aware_contract.csv (P21 IR facts)."""
    from pathlib import Path
    facts = {}
    for r in _read_csv(Path(cs_dir) / "loop_aware_contract.csv"):
        try:
            K = int(r.get("K_ir") or 0)
        except (TypeError, ValueError):
            K = 0
        kv = (r.get("kv_cache_bytes_ir") or "").strip()
        facts[r["workload"]] = {
            "K": max(K, 1), "K_present": K > 0,
            "kv_bytes": "" if kv in ("", "n/a") else kv,
            "roles": r.get("loop_carried_roles", ""),
        }
    return facts


def loop_aware_seed_rows(cs_dir) -> list[dict]:
    """Flat loop-aware seed: one row per (workload, op_class, region_role, distinct shape). The
    repeated-head shapes are multiplied by the IR-recovered K (instances_per_replan = layers x K);
    the once-prefix shapes keep layers x 1. KV state is attached per workload."""
    from collections import defaultdict
    from pathlib import Path
    facts = _loop_facts(cs_dir)
    acc = defaultdict(lambda: {"layers": 0, "fqn": ""})
    for r in _read_csv(Path(cs_dir) / "operator_shape_table.csv"):
        try:
            M, N, Kd = int(r["M"]), int(r["N"]), int(r["K"])
        except (TypeError, ValueError, KeyError):
            continue
        if M <= 0 or N <= 0 or Kd <= 0:
            continue
        oc = _OPCLASS_FROM_KIND.get((r.get("op_kind") or "").lower(), "linear_gemm")
        key = (r["workload"], oc, r.get("region_role") or "unknown", M, N, Kd)
        a = acc[key]
        a["layers"] += 1
        a["fqn"] = a["fqn"] or (r.get("prov_fqn") or "")
    rows = []
    for (w, oc, role, M, N, Kd), a in sorted(acc.items(),
                                             key=lambda kv: (kv[0][0], kv[0][2], -kv[0][3] * kv[0][4] * kv[0][5])):
        f = facts.get(w, {"K": 1, "K_present": False, "kv_bytes": "", "roles": ""})
        repeated = role in ("repeated_head", "decode_lm")
        outer = f["K"] if repeated else 1
        layers = a["layers"]
        mac_inst = M * N * Kd
        opb_id, dataflow = _OPB.get(oc, ("weight", ["output_stationary"]))
        rows.append({
            "workload": w, "op_class": oc, "region_role": role, "M": M, "N": N, "K": Kd,
            "layers_in_capture": layers, "outer_loop_K": outer,
            "K_source": "recovered_from_ir" if (repeated and f["K_present"]) else
                        ("not_repeated" if not repeated else "assumed_or_config"),
            "instances_per_replan": layers * outer,
            "macs_per_instance": mac_inst, "macs_per_replan": mac_inst * layers * outer,
            "operand_b_identity": opb_id, "dataflow_candidates": "|".join(dataflow),
            "reduction_reuse_scope": ("across_K (resident-eligible)" if repeated else "once_per_replan"),
            "kv_state_bytes_workload": f["kv_bytes"], "k_reduction_needs_partial_sum": "True",
            "evidence": "operator_shape_table.csv (shape+role) x loop_aware_contract.csv (K, KV; "
                        "recovered_from_ir scf.for)",
        })
    return rows


def loop_aware_problem_shapes(cs_dir) -> dict:
    """Timeloop-ingestible problem shapes, grouped per workload with the recovered OUTER K loop +
    the loop-carried KV state node — the structure a full Timeloop problem needs that the flat
    one-step seed lacked. Structural search-space seed; no mapping/perf chosen."""
    rows = loop_aware_seed_rows(cs_dir)
    facts = _loop_facts(cs_dir)
    by_w = {}
    for r in rows:
        w = r["workload"]
        wobj = by_w.setdefault(w, {
            "workload": w,
            "outer_loop_K": facts.get(w, {}).get("K", 1),
            "K_source": "recovered_from_ir" if facts.get(w, {}).get("K_present") else "assumed_or_config",
            "loop_carried_roles": facts.get(w, {}).get("roles", ""),
            "kv_state_bytes": facts.get(w, {}).get("kv_bytes", "") or "n/a",
            "shapes": [],
        })
        wobj["shapes"].append({
            "id": f"{w}__{r['op_class']}__{r['region_role']}__{r['M']}x{r['N']}x{r['K']}",
            "op_class": r["op_class"], "region_role": r["region_role"],
            "problem": {"shape": {"name": "matmul", "dimensions": ["M", "N", "K"]},
                        "instance": {"M": r["M"], "N": r["N"], "K": r["K"]}},
            "layers_in_capture": r["layers_in_capture"],
            "outer_loop_K": r["outer_loop_K"],
            "instances_per_replan": r["instances_per_replan"],
            "macs_per_replan": r["macs_per_replan"],
            "operand_b_identity": r["operand_b_identity"],
            "dataflow_candidates": r["dataflow_candidates"].split("|"),
            "reduction_reuse_scope": r["reduction_reuse_scope"],
            "k_reduction_needs_partial_sum": True,
        })
    workloads = [by_w[w] for w in sorted(by_w)]
    return {"loop_aware_mapspace_seeds": {
        "note": "P21-aware: repeated-head shapes carry the IR-recovered outer K loop (instances_per_"
                "replan = layers x K) + the loop-carried KV state node; once-prefix shapes run x1. "
                "Structural search-space seed; no mapping/cycles/perf chosen.",
        "count_workloads": len(workloads), "count_shapes": len(rows),
        "workloads": workloads}}
