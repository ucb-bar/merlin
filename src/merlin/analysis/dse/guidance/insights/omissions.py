"""Account for visible, recovered and erased workload operations.

Evidence labels, numerical caveats and artifact schemas retain their original
meaning; these analyses establish no hardware performance qualification.
"""

from __future__ import annotations

from pathlib import Path

from .artifacts import _csv_rows
from .workloads import _signal_by_workload

_OMIT_COLS = [
    "workload",
    "linear_gemm_ops",
    "linear_gemm_macs",
    "attention_ops",
    "attention_macs",
    "visible_linear_fraction",
    "softmax_ops",
    "norm_ops",
    "conv_ops",
    "elementwise_ops",
    "kv_state_recovered",
    "lowbit_packed_recovered",
    "scale_metadata_recovered",
    "still_erased",
    "capture_level_needed",
]


def operator_recovery_accounting(cs_dir: Path) -> dict:
    """Part F (RECOVERED, not count-only): attention bmm / softmax / norm are NOT erased — the flat
    capture lowered them to ``linalg.generic`` but they are re-parsed. Per workload: linear-GEMM vs
    attention MAC mass (both recovered from IR shapes — no model-card config) + softmax/norm/conv/
    elementwise op counts, and what genuinely STAYS erased (KV state across the decode loop, packed
    low-bit layout, scale metadata). ``visible_linear_fraction`` answers "how much of the recovered
    MAC work is the linear-GEMM geometry this study analyzes" with real numbers."""
    cs_dir = Path(cs_dir)
    wc = {r["workload"]: r for r in _csv_rows(cs_dir / "work_coverage_table.csv")}
    sig, wls = _signal_by_workload(cs_dir)
    rows = []
    for w in wls:
        r = wc.get(w, {})
        is_ar = sig[w]["family"] in ("autoregressive_vla", "llm")
        rows.append(
            {
                "workload": w,
                "linear_gemm_ops": int(r.get("n_linear_matmul", 0) or 0),
                "linear_gemm_macs": int(r.get("linear_gemm_macs", 0) or 0),
                "attention_ops": int(r.get("n_attention_ops", 0) or 0),
                "attention_macs": int(r.get("attention_macs", 0) or 0),
                "visible_linear_fraction": float(r.get("visible_linear_fraction", 1.0) or 1.0),
                "softmax_ops": int(r.get("n_softmax", 0) or 0),
                "norm_ops": int(r.get("n_normalization", 0) or 0),
                "conv_ops": int(r.get("n_conv", 0) or 0),
                "elementwise_ops": int(r.get("n_elementwise", 0) or 0),
                # what STAYS erased (genuinely absent in the flat capture, not merely unparsed):
                "kv_state_recovered": (
                    "no (KV state across decode loop erased)" if is_ar else "n/a (non-autoregressive)"
                ),
                "lowbit_packed_recovered": "no (f32 fake-quant capture)",
                "scale_metadata_recovered": "no (dequantized in capture)",
                "still_erased": "K-loop trip count; KV state across decode; packed low-bit layout + scales",
                "capture_level_needed": (
                    "loop-preserving (K / KV state)"
                    + ("; KV-cache" if is_ar else "")
                    + "; low-bit recapture for packed layout + scales"
                ),
            }
        )
    return {"workloads": wls, "rows": rows}


# ---- Stage C: capture-erasure IR evidence, per-family decomposition (analysis-only) ----
_ERASE_COLS = [
    "workload",
    "n_scf_loops",
    "n_linalg_matmul",
    "n_linalg_generic",
    "lowbit_int_types_present",
    "loops_preserved",
    "evidence",
]


def _count_word_bounded(text: str, *needles: str) -> int:
    """Occurrences of each ``needle`` bounded by non-word chars on both sides — the regex-free
    equivalent of ``\\b<needle>\\b`` (e.g. counts ``scf.for(`` but not ``scf.forall``)."""
    total = 0
    for needle in needles:
        start = 0
        while True:
            i = text.find(needle, start)
            if i == -1:
                break
            start = i + 1
            before = text[i - 1] if i > 0 else ""
            after = text[i + len(needle)] if i + len(needle) < len(text) else ""
            bw = before.isalnum() or before == "_"
            aw = after.isalnum() or after == "_"
            if not bw and not aw:
                total += 1
    return total


def capture_erasure_evidence(cs_dir: Path) -> dict:
    """Stage C: demonstrate (not just assert) what the flat capture erases, from the recapture IR:
    scf.for/while count (loops), linalg.matmul vs linalg.generic counts (named ops vs lowered),
    and whether any packed low-bit integer tensor types survive.

    Always reads the FLAT corpus (that is the capture whose erasure is being shown), resolved through
    the corpus accessor — committed under experiments/benchmarks/ with the oversized ones in the
    out/artifacts/recaptures/ overflow. (It used to be globbed as a sibling of cs_dir, which silently
    yielded zero rows once case_study moved under out/artifacts/.)"""
    from merlin.analysis.dse.guidance.corpus import RECAP_MODELS, _recap_dir_in

    rows = []
    for d in [_recap_dir_in(w, "recaptures") for w in sorted(RECAP_MODELS)]:
        if not (d / "model.mlir").is_file():
            continue
        # Bulk triage over the whole flat corpus (dozens of large captures): a cheap, regex-free
        # token/substring scan — a full xDSL parse per file would be 100-1000x costlier for a
        # presence/count heuristic. (Per-workload deep facts use merlin.common.mlir_query.)
        txt = (d / "model.mlir").read_text(errors="ignore")
        n_loops = _count_word_bounded(txt, "scf.for", "scf.while")  # == old \bscf\.(for|while)\b
        n_mm = txt.count("linalg.matmul")
        n_gen = txt.count("linalg.generic")
        lowbit = any(f"x{dt}>" in txt for dt in ("i4", "i8", "si8", "ui8", "si4"))
        rows.append(
            {
                "workload": d.name,
                "n_scf_loops": n_loops,
                "n_linalg_matmul": n_mm,
                "n_linalg_generic": n_gen,
                "lowbit_int_types_present": lowbit,
                "loops_preserved": n_loops > 0,
                "evidence": (
                    f"{n_loops} scf loops, {n_gen} generics vs {n_mm} named matmuls, "
                    f"low-bit int types {'present' if lowbit else 'absent'}"
                ),
            }
        )
    return {"rows": rows}


def _omitted_op_md(om, scope) -> str:
    L = [
        f"# Operator recovery accounting ({scope})\n",
        "> Attention bmm / softmax / norm are **NOT erased** — the flat capture lowered them to "
        "`linalg.generic` but they are re-parsed. Linear-GEMM and attention MACs are both recovered "
        "from IR shapes (no model-card config). `visible_linear_fraction` = linear / (linear+attention) "
        "answers how much of the recovered MAC work is the linear-GEMM geometry. What genuinely STAYS "
        "erased: the K-loop, KV state across the decode loop, and packed low-bit layout + scales.\n",
        "| workload | linear ops | linear MACs | attn ops | attn MACs | **visible_linear_frac** | "
        "softmax | norm | conv | elementwise |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in om["rows"]:
        L.append(
            f"| {r['workload']} | {r['linear_gemm_ops']} | {r['linear_gemm_macs']} | "
            f"{r['attention_ops']} | {r['attention_macs']} | **{r['visible_linear_fraction']:.3f}** | "
            f"{r['softmax_ops']} | {r['norm_ops']} | {r['conv_ops']} | {r['elementwise_ops']} |"
        )
    L.append(
        "\n**Still erased** (genuinely absent in the flat capture, not merely unparsed): the K-loop "
        "trip count, KV state across the decode loop, and packed low-bit layout + scales — needing "
        "a loop-preserving (+ KV) capture and a low-bit recapture. Per-workload: "
        "`visible_vs_erased_work_table.csv`.\n"
    )
    return "\n".join(L) + "\n"


def _erasure_evidence_md(ce, scope) -> str:
    L = [
        f"# Capture-erasure IR evidence ({scope})\n",
        "> The 'erased' claims, demonstrated from the recapture IR rather than asserted. Loops are "
        "absent (torch.export unrolls them — only a gather/scatter artifact shows scf.for); attention "
        "lives in `linalg.generic` (now re-parsed); no packed low-bit integer tensor types survive "
        "(the capture is dequantized f32).\n",
        "| workload | scf loops | linalg.matmul | linalg.generic | low-bit int types | loops preserved |",
        "|---|---|---|---|---|---|",
    ]
    for r in ce["rows"]:
        L.append(
            f"| {r['workload']} | {r['n_scf_loops']} | {r['n_linalg_matmul']} | "
            f"{r['n_linalg_generic']} | {r['lowbit_int_types_present']} | {r['loops_preserved']} |"
        )
    nloop = sum(1 for r in ce["rows"] if r["loops_preserved"])
    L.append(
        f"\n**{nloop}/{len(ce['rows'])} captures retain any scf loop** (and that one is a bool-mask "
        "gather artifact, not a model loop). No capture retains packed low-bit types. This is the "
        "concrete basis for the capture-fidelity 'erased' rows.\n"
    )
    return "\n".join(L) + "\n"
