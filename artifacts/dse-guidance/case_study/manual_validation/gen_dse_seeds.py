"""Generate the P19 Phase-3 DSE-paper seed artifacts from the committed (P19-corrected) case-study
inventory. Source-grounded: every op traces to source via prov.fqn -> operator_full_inventory.csv.
One-off generator (not a pipeline tool). Structural/requirements only; no perf claims.

Outputs (into this dir):
  mapspace_seed_candidates.yaml   (Timeloop-style: per distinct shape, loops/operands/dataflow)
  mapspace_seed_gaps.md
  operand_locality_targets.csv    (CADOSys-style: per region/operand bytes + reuse scope)
  dependence_resource_targets.csv (Aladdin-style: loops/state/deps preserved? recovered?)
  task_graph_targets.yaml         (MLDSE-style: compute/storage nodes + edges)
"""
from __future__ import annotations

import csv
import io
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
CS = HERE.parent  # case_study/


def _rows(name):
    p = CS / name
    return list(csv.DictReader(io.StringIO(p.read_text()))) if p.is_file() else []


def _int(x):
    try:
        return int(x)
    except (TypeError, ValueError):
        return 0


inv = _rows("operator_full_inventory.csv")
dm = _rows("data_movement_table.csv")
CONTRACTIONS = {"linear_gemm", "attention_contraction", "batched_matmul"}


def _yq(s):  # minimal yaml string quote
    return '"' + str(s).replace('"', "'") + '"'


# ---- A. mapspace_seed_candidates.yaml (per workload, op_class, distinct (M,K,N)) ----
seeds = defaultdict(lambda: {"mult": 0, "fqn": "", "macs": 0})
for r in inv:
    if r["op_class"] not in CONTRACTIONS:
        continue
    M, K, N = int(r["M"]), int(r["K"]), int(r["N"])
    if M <= 0 or K <= 0 or N <= 0:
        continue
    key = (r["workload"], r["op_class"], M, K, N)
    s = seeds[key]
    s["mult"] += 1
    s["macs"] += int(r["macs"])
    if not s["fqn"]:
        s["fqn"] = r["prov_fqn"]

L = ["# Timeloop-style mapspace seed candidates (P19 Phase 3) — structural; no perf claim",
     "# Per (workload, op_class, distinct shape). Attention operands are BOTH activations (no weight",
     "# reuse); linear/batched have a stationary weight. Loop axes + operand identities are recovered",
     "# from the MLIR; K-reduction spatial legality needs a partial-sum datapath (see boundary catalog).",
     "mapspace_seed_candidates:"]
for (wl, oc, M, K, N), s in sorted(seeds.items(), key=lambda kv: (kv[0][0], kv[0][1], -kv[1]["macs"])):
    attn = oc == "attention_contraction"
    operands = ("{A: activation, B: activation, C: output}" if attn
                else "{A: activation, B: weight, C: output}")
    dataflow = ("[output_stationary, input_stationary]" if attn
                else "[weight_stationary, output_stationary, input_stationary]")
    L += [f"  - workload: {wl}",
          f"    op_class: {oc}",
          f"    shape: {{M: {M}, K: {K}, N: {N}}}",
          f"    multiplicity: {s['mult']}        # identical-shape instances (layers)",
          f"    macs_total: {s['macs']}",
          '    equation: "C[M,N] += A[M,K] * B[K,N]"',
          "    loops: [M, N, K]",
          f"    operands: {operands}",
          "    legal_spatial_axes: [M, N, K_with_reduction]",
          "    legal_temporal_axes: [M, N, K]",
          "    bypass_candidates: {A: possible, B: possible, C: possible}",
          f"    dataflow_candidates: {dataflow}",
          f"    evidence: {{fqn_sample: {_yq(s['fqn'])}, source: operator_full_inventory.csv}}"]
(HERE / "mapspace_seed_candidates.yaml").write_text("\n".join(L) + "\n")

gaps = ["# Mapspace seed gaps (what a full Timeloop problem needs that the flat capture lacks)\n",
        "- **K denoise/decode loop trip count**: unrolled by torch.export -> seeds cover ONE step; the",
        "  outer K loop is config/assumed (MODEL_ARCH), not in the seed. Needs loop-preserving capture.",
        "- **Attention inner dims (heads, seq, head_dim)**: the recovered attention contraction folds",
        "  batch/heads into M; the per-head/seq mapspace axes need the SDPA op kept un-decomposed",
        "  (high-level capture) or a sidecar with heads/seq/head_dim.",
        "- **Conv spatial/channel loops**: conv ops are counted (n_conv) but MACs/loops not quantified",
        "  here (would need the conv window dims). 12 conv ops corpus-wide (patch embed).",
        "- **Operand bit-width / packed layout**: f32 in the flat capture; the qdq recapture exposes",
        "  per-channel int8 (quant_ext.dequantize) for a true low-bit mapspace.",
        "- **Fusion/epilogue**: bias-add (addmm) + activation are separate generics here; a fused-epilogue",
        "  mapspace needs the epilogue grouped onto the matmul (epilogue_pattern_table has the grouping).\n"]
(HERE / "mapspace_seed_gaps.md").write_text("\n".join(gaps))

# ---- B. operand_locality_targets.csv (per region/operand bytes + reuse scope) ----
loc = []
for r in dm:
    wl, reg = r["workload"], r["region"]
    inv_n = _int(r.get("invocations")) or 1
    scope_w = "across_K" if (reg == "repeated_head" and inv_n > 1) else "across_replan"
    wb = _int(r.get("weight_bytes"))
    ab = _int(r.get("activation_input_bytes"))
    ob = _int(r.get("output_bytes"))
    kv = _int(r.get("kv_bytes"))
    loc.append({"workload": wl, "region": reg, "operand": "weight", "bytes": wb,
                "reuse_scope": scope_w, "resident_candidate": "yes" if wb else "no",
                "evidence": "data_movement_table.csv"})
    loc.append({"workload": wl, "region": reg, "operand": "activation_in", "bytes": ab,
                "reuse_scope": "within_op", "resident_candidate": "no",
                "evidence": "data_movement_table.csv"})
    loc.append({"workload": wl, "region": reg, "operand": "output", "bytes": ob,
                "reuse_scope": "within_op", "resident_candidate": "no",
                "evidence": "data_movement_table.csv"})
    if kv:
        loc.append({"workload": wl, "region": reg, "operand": "kv_state", "bytes": kv,
                    "reuse_scope": "across_decode", "resident_candidate": "yes (KV cache)",
                    "evidence": "data_movement_table.csv (kv inputs)"})
buf = io.StringIO()
w = csv.DictWriter(buf, fieldnames=["workload", "region", "operand", "bytes", "reuse_scope",
                                    "resident_candidate", "evidence"])
w.writeheader()
[w.writerow(x) for x in loc]
(HERE / "operand_locality_targets.csv").write_text(buf.getvalue())

# ---- C. dependence_resource_targets.csv (per workload) ----
wls = sorted({r["workload"] for r in inv})
dep = []
for wl in wls:
    dep += [
        {"workload": wl, "feature": "K denoise/decode loop", "type": "loop",
         "preserved_in_export": "no (unrolled)", "preserved_in_mlir": "no",
         "merlin_recovers": "no (config K only)", "dse_relevance": "loop rolling / bounded command",
         "tool_needed": "loop-preserving capture"},
        {"workload": wl, "feature": "loop-carried latent / KV state", "type": "loop_carried_state",
         "preserved_in_export": "no", "preserved_in_mlir": "no",
         "merlin_recovers": "partial (KV as inputs for rdt2)", "dse_relevance": "state buffer sizing",
         "tool_needed": "loop/decode-preserving capture"},
        {"workload": wl, "feature": "SSA op producer->consumer deps", "type": "data_dependency",
         "preserved_in_export": "yes", "preserved_in_mlir": "yes",
         "merlin_recovers": "yes (matmul_dependencies)", "dse_relevance": "pipelining / critical path",
         "tool_needed": "none (recovered)"},
        {"workload": wl, "feature": "in-place / aliasing", "type": "aliasing",
         "preserved_in_export": "partial", "preserved_in_mlir": "no (functional tensors)",
         "merlin_recovers": "no", "dse_relevance": "buffer reuse / port pressure",
         "tool_needed": "memref-level lowering"},
    ]
buf = io.StringIO()
w = csv.DictWriter(buf, fieldnames=["workload", "feature", "type", "preserved_in_export",
                                    "preserved_in_mlir", "merlin_recovers", "dse_relevance", "tool_needed"])
w.writeheader()
[w.writerow(x) for x in dep]
(HERE / "dependence_resource_targets.csv").write_text(buf.getvalue())

# ---- D. task_graph_targets.yaml (per workload: compute/storage nodes + edges) ----
by_wl = defaultdict(Counter)
macs_wl = defaultdict(lambda: Counter())
for r in inv:
    by_wl[r["workload"]][r["op_class"]] += 1
    macs_wl[r["workload"]][r["op_class"]] += int(r["macs"])
T = ["# MLDSE-style task-graph seeds (P19 Phase 3) — node/edge types per workload; structural",
     "task_graph_targets:"]
for wl in wls:
    c = by_wl[wl]
    T += [f"  - workload: {wl}",
          "    compute_nodes:",
          f"      linear_gemm: {{count: {c.get('linear_gemm',0)}, macs: {macs_wl[wl].get('linear_gemm',0)}}}",
          f"      attention: {{count: {c.get('attention_contraction',0)}, macs: {macs_wl[wl].get('attention_contraction',0)}}}",
          f"      batched_matmul: {{count: {c.get('batched_matmul',0)}, macs: {macs_wl[wl].get('batched_matmul',0)}}}",
          f"      softmax: {{count: {c.get('softmax',0)}}}",
          f"      normalization: {{count: {c.get('normalization',0)}}}",
          f"      conv: {{count: {c.get('conv',0)}}}",
          "    storage_nodes: [resident_weights (across_K), activations (within_op), kv_state (across_decode, if AR)]",
          "    edges: [data_dependency (SSA, recovered), loop_carried (K, config-only), "
          "producer_consumer (recovered)]",
          "    sync_nodes: [softmax reduction barrier, norm reduction barrier]",
          "    evidence: operator_full_inventory.csv + matmul_dependencies (SSA)"]
(HERE / "task_graph_targets.yaml").write_text("\n".join(T) + "\n")

print("wrote mapspace_seed_candidates.yaml (%d seeds), mapspace_seed_gaps.md, "
      "operand_locality_targets.csv (%d rows), dependence_resource_targets.csv (%d rows), "
      "task_graph_targets.yaml (%d workloads)" % (len(seeds), len(loc), len(dep), len(wls)))
