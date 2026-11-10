"""Assess captured representation fidelity and capture-level ablations.

Evidence labels, numerical caveats and artifact schemas retain their original
meaning; these analyses establish no hardware performance qualification.
"""

from __future__ import annotations

from pathlib import Path

from .artifacts import _csv_rows
from .workloads import _signal_by_workload

# capture-fidelity feature rows (reuse fidelity.py vocab); status per workload reconstructed from
# committed artifacts + MODEL_ARCH (run-folder-only, no live topology object needed).
_FIDELITY_FEATURES = [
    ("op_shapes_MNK", "strong"),
    ("region_roles", "strong"),
    ("dtype_information", "strong"),
    # P18: attention bmm / softmax / norm are NOT erased — lowered to linalg.generic but re-parsed
    # (attention with real MACs). Only the KV *state across the decode loop* stays erased.
    ("attention_bmm_qkT_attnV", "recovered_attn"),
    ("softmax", "recovered_softmax"),
    ("normalization", "recovered_norm"),
    ("K_or_decode_loop", "loop"),
    ("kv_cache_state", "kv"),
    ("loop_carried_state", "loop_carried"),
    ("packed_lowbit_layout", "lowbit"),
    ("scale_metadata", "lowbit"),
    ("host_dispatch_count", "measured"),
    ("target_latency_cycles", "not_claimed"),
]


def capture_fidelity(cs_dir: Path) -> dict:
    """First-class result: which structural features the flat capture preserves vs erased, per
    workload. Attention/softmax/norm are RECOVERED (lowered to linalg.generic but re-parsed; P18);
    what stays erased is the K-loop, the KV *state* across the decode loop, and packed-lowbit/scales —
    the axes the residency & loop conclusions depend on."""
    from merlin.analysis.dse.guidance import fidelity as FID
    from merlin.analysis.dse.guidance import topology as TOP
    from merlin.analysis.dse.guidance.loop_recovery import recover_loop

    sig, wls = _signal_by_workload(cs_dir)
    wc = {r["workload"]: r for r in _csv_rows(Path(cs_dir) / "work_coverage_table.csv")}
    # P21-S1: loop-preserving captures (recaptures_loop/<w>/model.mlir, if present) recover K, the
    # loop-carried state and the KV cache directly from scf.for -> flip K/KV from assumed/erased.
    # Resolved through the corpus accessor (committed under experiments/benchmarks/ + out/ overflow), NOT
    # as a sibling of cs_dir (which broke silently once case_study moved under out/artifacts/).
    from merlin.analysis.dse.guidance.corpus import _recap_dir_in

    lr_by_w = {}
    for w in wls:
        mp = _recap_dir_in(w, "recaptures_loop") / "model.mlir"
        if mp.is_file():
            lr = recover_loop(mp, w)
            if lr.present:
                lr_by_w[w] = lr
    fam_to_class = {
        "flow_matching": TOP.CLASS_FLOW_MATCHING,
        "diffusion": TOP.CLASS_FLOW_MATCHING,
        "autoregressive_vla": TOP.CLASS_AUTOREGRESSIVE,
        "llm": TOP.CLASS_AUTOREGRESSIVE,
    }
    per_workload, matrix = {}, []
    for w in wls:
        s = sig[w]
        is_ar = s["family"] in ("autoregressive_vla", "llm")
        cls = fam_to_class.get(s["family"], TOP.CLASS_UNKNOWN)
        missing = []
        if s["has_repeated"]:
            missing.append("token_decode_loop" if is_ar else "denoise_loop")
        if is_ar:
            missing.append("kv_cache_growth")
        if s["has_backbone"] and s["has_repeated"]:
            missing.append("prefix_kv_reuse")
            missing.append("async_backbone_head_overlap")
        if s["H"] > 1:
            missing.append("action_chunk_horizon")
        if s["control_rate"]:
            missing.append("replan_deadline")
        seen = set()
        missing = [m for m in missing if not (m in seen or seen.add(m))]
        hidden = []
        for m in missing:
            for ax in FID.HIDDEN_AXES_BY_STRUCTURE.get(m, []):
                if ax not in hidden:
                    hidden.append(ax)
        severity = TOP.CLASS_SEVERITY.get(cls, "medium")
        if not s["has_repeated"] and not s["control_rate"]:
            severity = "low"
        recovered_loop = []
        lr = lr_by_w.get(w)
        if lr is not None:
            # the loop-preserving capture moves loop/KV out of "missing" -> recovered-from-IR
            for tag in ("token_decode_loop", "denoise_loop", "kv_cache_growth", "prefix_kv_reuse"):
                if tag in missing:
                    missing.remove(tag)
            severity = "low"
            recovered_loop = [
                f"K_loop(K={lr.K},IR)",
                f"repeated_region({lr.repeated_region_op_count}ops)",
                f"loop_carried_state({lr.n_iter_args} iter_args)",
            ]
            if lr.kv_cache_bytes:
                recovered_loop.append(f"kv_cache_state({lr.kv_cache_bytes}B,IR)")
        per_workload[w] = {
            "family": s["family"],
            "class": cls,
            "severity": severity,
            "missing": missing,
            "hidden_axes": hidden,
            "preserved": ["matmul_shapes", "dtype_information", "op_mix", "weight_sizes"],
            "recovered_from_loop_capture": recovered_loop,
        }
    for feat, kind in _FIDELITY_FEATURES:
        row = {"feature": feat}
        for w in wls:
            s = sig[w]
            is_ar = s["family"] in ("autoregressive_vla", "llm")
            wcr = wc.get(w, {})
            if kind == "strong":
                st = "strong"
            elif kind == "recovered_attn":
                n = int(wcr.get("n_attention_ops", 0) or 0)
                st = f"recovered ({n} ops)" if n else "n/a (attention-free here)"
            elif kind == "recovered_softmax":
                n = int(wcr.get("n_softmax", 0) or 0)
                st = "recovered" if n else "n/a"
            elif kind == "recovered_norm":
                n = int(wcr.get("n_normalization", 0) or 0)
                st = "recovered" if n else "n/a (norm as elementwise primitives)"
            elif kind == "loop":
                lr = lr_by_w.get(w)
                if lr is not None:
                    st = f"recovered (K={lr.K}, IR scf.for)"
                else:
                    st = "assumed (config K)" if s["has_repeated"] else "n/a"
            elif kind == "kv":
                lr = lr_by_w.get(w)
                if lr is not None and lr.kv_cache_bytes:
                    st = f"recovered ({lr.kv_cache_bytes} B, IR iter_arg)"
                elif lr is not None:
                    st = "n/a (prefix-KV invariant, closed-over)"
                else:
                    st = "erased" if is_ar else "n/a"  # KV STATE across the decode loop (not the bmm)
            elif kind == "loop_carried":
                lr = lr_by_w.get(w)
                if lr is not None:
                    roles = sorted({c.role for c in lr.carried_state})
                    st = f"recovered ({lr.n_iter_args} iter_args: {','.join(roles)})"
                else:
                    st = "erased (loop unrolled)" if s["has_repeated"] else "n/a"
            elif kind == "lowbit":
                st = "erased"  # all recaptures are dequantized f32
            elif kind == "measured":
                st = "measured (host)"
            else:
                st = "not_claimed"
            row[w] = st
        matrix.append(row)
    return {"workloads": wls, "per_workload": per_workload, "matrix": matrix}


# ---- Stage B: capture-level ablation over REAL multi-level recaptures (m2m flags) ----
# Level 0 = flat (committed recaptures/), high_level = m2m level="high-level" (linalg_ext.softmax),
# quant_qdq = m2m preserve_qdq int8 (quant_ext.dequantize). Loop-preserving is torch.export-blocked.
_ABLATION_LEVELS = [
    ("flat", "recaptures", "model.mlir"),
    ("high_level", "recaptures_levels", "model_highlevel.mlir"),
    ("quant_qdq", "recaptures_levels", "model_qdq.mlir"),
]


_ABLATION_OPS = [
    "linalg.matmul",
    "linalg.generic",
    "linalg_ext.softmax",
    "linalg_ext.layer_norm",
    "quant_ext.dequantize",
    "scf.for",
]


_ABLATION_COLS = ["workload", "level", "available"] + [o.replace(".", "_") for o in _ABLATION_OPS]


# DSE axes and how each becomes decidable as the capture level rises (categorical).
_ABLATION_AXES = [
    "attention_primitive",
    "softmax_reduction_unit",
    "packed_lowbit_weights",
    "scale_metadata",
    "bounded_loop_command",
    "kv_cache_object",
]


def capture_level_ablation(cs_dir: Path) -> dict:
    """Stage B: real capture-level ablation. Reads the committed op-count summary
    ``capture_level_ablation.csv`` (the raw multi-level recaptures are gitignored + regenerable) and
    derives which DSE axes move blocked -> decidable. Levels: flat (attention recovered by re-parsing
    generics, P18-A), high_level (attention/softmax as NAMED linalg_ext ops), quant_qdq (packed-lowbit
    + scales via quant_ext.dequantize). Loop/KV stay blocked at every level (torch.export unrolls
    loops). Empty if the summary is absent (no recaptures were generated)."""
    raw = _csv_rows(Path(cs_dir) / "capture_level_ablation.csv")
    if not raw:
        return {"workloads": [], "rows": [], "unlock": []}
    rows = []
    for r in raw:
        row = {"workload": r["workload"], "level": r["level"], "available": r["available"] == "True"}
        for o in _ABLATION_OPS:
            k = o.replace(".", "_")
            row[k] = int(r.get(k, 0) or 0)
        rows.append(row)
    wls = sorted({r["workload"] for r in rows})
    unlock = []
    for w in wls:
        by = {r["level"]: r for r in rows if r["workload"] == w}
        hl, qd = by.get("high_level", {}), by.get("quant_qdq", {})
        any_loop = any(by.get(lv, {}).get("scf_for") for lv in ("flat", "high_level", "quant_qdq"))
        unlock.append(
            {
                "workload": w,
                "attention_primitive": (
                    "named (linalg_ext) @high_level" if hl.get("linalg_ext_softmax") else "recovered (generic) @flat"
                ),
                "softmax_reduction_unit": (
                    "named @high_level" if hl.get("linalg_ext_softmax") else "recovered (generic) @flat"
                ),
                "packed_lowbit_weights": (
                    "decidable @quant_qdq" if qd.get("quant_ext_dequantize") else "blocked (no qdq capture)"
                ),
                "scale_metadata": (
                    "decidable @quant_qdq" if qd.get("quant_ext_dequantize") else "blocked (no qdq capture)"
                ),
                "bounded_loop_command": ("partial" if any_loop else "blocked (torch.export unrolls loops)"),
                "kv_cache_object": ("partial" if any_loop else "blocked (torch.export unrolls loops)"),
            }
        )
    return {"workloads": wls, "rows": rows, "unlock": unlock}


def _capture_fidelity_md(cf, scope) -> str:
    wls = cf["workloads"]
    L = [
        f"# Capture-fidelity matrix ({scope})\n",
        "> The likely central result: which structural features the flat capture preserves vs "
        "erased. `strong`=recovered from IR; `assumed (config K)`=loop count is a reference value, "
        "not captured; `erased`=lowered/dequantized away; `measured (host)`=real host measurement; "
        "`not_claimed`=needs a target design. Findings that depend on `assumed`/`erased` rows are "
        "capture-limited.\n",
        "| feature | " + " | ".join(wls) + " |",
        "|---|" + "|".join(["---"] * len(wls)) + "|",
    ]
    for row in cf["matrix"]:
        L.append(f"| {row['feature']} | " + " | ".join(row[w] for w in wls) + " |")
    L.append("\n**Per-workload DSE risk:**\n")
    for w in wls:
        d = cf["per_workload"][w]
        L.append(
            f"- **{w}** ({d['family']}, severity {d['severity']}): lost "
            f"{', '.join(d['missing']) or 'none'}; hides axes "
            f"{', '.join(d['hidden_axes']) or 'none'}"
        )
    return "\n".join(L) + "\n"


def _ablation_md(ab, scope) -> str:
    L = [
        f"# Capture-level ablation ({scope})\n",
        "> REAL multi-level recaptures (m2m flags). **flat** = the committed capture (attention "
        'recovered by re-parsing generics, P18-A); **high_level** (`level="high-level"`) emits '
        "attention/softmax as NAMED `linalg_ext` ops; **quant_qdq** (`preserve_qdq` int8) emits "
        "`quant_ext.dequantize` (packed-lowbit + scales). **Loop-preserving is torch.export-blocked** "
        "(no Level — the named frontier a TorchDynamo/`torch.cond` frontend would unlock).\n",
        "## Op vocabulary per level\n",
        "| workload | level | available | matmul | generic | linalg_ext.softmax | quant_ext.dequantize | scf.for |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in ab["rows"]:
        L.append(
            f"| {r['workload']} | {r['level']} | {r['available']} | {r['linalg_matmul']} | "
            f"{r['linalg_generic']} | {r['linalg_ext_softmax']} | {r['quant_ext_dequantize']} | "
            f"{r['scf_for']} |"
        )
    L += [
        "\n## Axis-unlock matrix (which DSE axes become decidable at which level)\n",
        "| workload | " + " | ".join(_ABLATION_AXES) + " |",
        "|---|" + "|".join(["---"] * len(_ABLATION_AXES)) + "|",
    ]
    for u in ab["unlock"]:
        L.append(f"| {u['workload']} | " + " | ".join(u[a] for a in _ABLATION_AXES) + " |")
    L.append(
        "\n**Loop-preserving capture is the blocked frontier**: `scf.for` is absent at every level "
        "(torch.export unrolls Python loops), so `bounded_loop_command` / `kv_cache_object` stay "
        "blocked — the one axis a richer (TorchDynamo / torch.cond) frontend would unlock.\n"
    )
    return "\n".join(L) + "\n"
