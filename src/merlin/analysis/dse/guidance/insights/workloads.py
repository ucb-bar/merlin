"""Summarize workload hotspots, abstraction coverage and corpus requirements.

Evidence labels, numerical caveats and artifact schemas retain their original
meaning; these analyses establish no hardware performance qualification.
"""

from __future__ import annotations

from pathlib import Path

from .artifacts import _csv_rows


def per_operator_hotspots(cs_dir: Path, scope: str = "all", k: int = 10) -> dict:
    """Deeper-per-operator signal: which few ops dominate compute / weight / tile-waste / reload.

    Reads operator_shape_table (per-op M/N/K, macs, weight bytes, shape class), joins the best
    achievable tile padding waste per op from tile_waste_table, and the region reload from
    data_movement_table. Top-k by each axis; structural quantities only. When ``scope`` is a single
    workload (not "all"), the tables are restricted to that workload so a per-network run reports
    that network's own hotspots, not the corpus-wide top."""
    cs_dir = Path(cs_dir)
    ops = _csv_rows(cs_dir / "operator_shape_table.csv")
    tw = _csv_rows(cs_dir / "tile_waste_table.csv")
    dm = _csv_rows(cs_dir / "data_movement_table.csv")
    if scope != "all":
        ops = [o for o in ops if o["workload"] == scope]
        tw = [r for r in tw if r["workload"] == scope]
        dm = [r for r in dm if r["workload"] == scope]
    # best (lowest) achievable tile padding waste per op over the *real* applicable tile primitives
    best: dict[tuple, float] = {}
    for r in tw:
        if r.get("primitive_kind") != "tile" or r.get("applicable") != "True":
            continue
        key = (r["workload"], r["op_index"])
        w = float(r["padding_waste"])
        if key not in best or w < best[key]:
            best[key] = w
    mac_total_wl: dict[str, int] = {}
    for o in ops:
        mac_total_wl[o["workload"]] = mac_total_wl.get(o["workload"], 0) + int(o["macs"])
    enriched = []
    for o in ops:
        wl = o["workload"]
        tot = mac_total_wl.get(wl, 0) or 1
        enriched.append(
            {
                "workload": wl,
                "op_index": o["op_index"],
                "prov_fqn": o["prov_fqn"],
                "op_kind": o["op_kind"],
                "shape_class": o["shape_class"],
                "M": o["M"],
                "N": o["N"],
                "K": o["K"],
                "macs": int(o["macs"]),
                "mac_fraction_of_workload": round(int(o["macs"]) / tot, 4),
                "rhs_weight_bytes": int(o["rhs_weight_bytes"]),
                "best_tile_padding_waste": best.get((wl, o["op_index"])),
                "is_tail_heavy": o["is_tail_heavy"],
            }
        )
    by_macs = sorted(enriched, key=lambda r: -r["macs"])[:k]
    by_weight = sorted(enriched, key=lambda r: -r["rhs_weight_bytes"])[:k]
    by_pad = sorted(
        [r for r in enriched if r["best_tile_padding_waste"] is not None], key=lambda r: -r["best_tile_padding_waste"]
    )[:k]
    by_reload = sorted(dm, key=lambda r: -int(r["avoidable_weight_reload"]))[:k]
    reload_rows = [
        {
            "workload": r["workload"],
            "region": r["region"],
            "avoidable_weight_reload": int(r["avoidable_weight_reload"]),
            "invocations": r["invocations"],
            "weight_bytes": int(r["weight_bytes"]),
        }
        for r in by_reload
    ]
    # long-form flat table for CSV: one row per (ranking, rank, entity)
    flat = []
    for tag, rows_ in (("macs", by_macs), ("weight_bytes", by_weight), ("padding_waste", by_pad)):
        for i, r in enumerate(rows_, 1):
            flat.append({"ranking": tag, "rank": i, **r, "region": "", "avoidable_weight_reload": ""})
    for i, r in enumerate(reload_rows, 1):
        flat.append(
            {
                "ranking": "avoidable_reload",
                "rank": i,
                "workload": r["workload"],
                "op_index": "",
                "prov_fqn": "",
                "op_kind": "",
                "shape_class": "",
                "M": "",
                "N": "",
                "K": "",
                "macs": "",
                "mac_fraction_of_workload": "",
                "rhs_weight_bytes": r["weight_bytes"],
                "best_tile_padding_waste": "",
                "is_tail_heavy": "",
                "region": r["region"],
                "avoidable_weight_reload": r["avoidable_weight_reload"],
            }
        )
    dominant = by_macs[0] if by_macs else None
    return {
        "n_ops": len(ops),
        "by_macs": by_macs,
        "by_weight_bytes": by_weight,
        "by_padding_waste": by_pad,
        "by_avoidable_reload": reload_rows,
        "dominant_op": dominant,
        "rows": flat,
    }


# proof-status conservatism: report the weakest status when an abstraction is referenced by several
# compiler-proof axes (unknown is worse than assumed, which is worse than proven).
_PROOF_RANK = {"unknown": 0, "unavailable": 0, "assumed": 1, "suggested": 1, "proven": 2, "": 3}


def abstraction_coverage(cs_dir: Path) -> list[dict]:
    """Replace the noisy ``n_X`` count facts with real coverage: for each candidate system
    abstraction, what fraction of the corpus (workloads / MACs / weight bytes / regions) implies it,
    its compiler-proof status, and an overfit/regret flag (single-workload support)."""
    cs_dir = Path(cs_dir)
    bm = _csv_rows(cs_dir / "hw_sw_boundary_matrix.csv")
    proofs = _csv_rows(cs_dir / "compiler_proof_matrix.csv")
    ops = _csv_rows(cs_dir / "operator_shape_table.csv")
    dm = _csv_rows(cs_dir / "data_movement_table.csv")
    all_wls = sorted({o["workload"] for o in ops})
    mac_by_wl: dict[str, int] = {}
    byte_by_wl: dict[str, int] = {}
    for o in ops:
        mac_by_wl[o["workload"]] = mac_by_wl.get(o["workload"], 0) + int(o["macs"])
        byte_by_wl[o["workload"]] = byte_by_wl.get(o["workload"], 0) + int(o["rhs_weight_bytes"])
    total_mac = sum(mac_by_wl.values()) or 1
    total_byte = sum(byte_by_wl.values()) or 1
    region_by_wl: dict[str, int] = {}
    for r in dm:
        region_by_wl[r["workload"]] = region_by_wl.get(r["workload"], 0) + 1
    total_regions = len(dm) or 1
    rows = []
    for b in bm:
        supp = [w.strip() for w in b["supporting_workloads"].split(";") if w.strip()]
        nwl = len(supp)
        # weakest proof status among compiler-proof axes that reference this atomic abstraction
        status = ""
        for p in proofs:
            if b["abstraction"] in p.get("system_abstraction", ""):
                if _PROOF_RANK.get(p["status"], 3) < _PROOF_RANK.get(status, 3):
                    status = p["status"]
        overfit = "high" if nwl <= 1 else "medium" if nwl < len(all_wls) else "low"
        rows.append(
            {
                "abstraction": b["abstraction"],
                "workloads_supporting": "; ".join(supp),
                "n_workloads": nwl,
                "workload_coverage": round(nwl / (len(all_wls) or 1), 4),
                "mac_coverage": round(sum(mac_by_wl.get(w, 0) for w in supp) / total_mac, 4),
                "byte_coverage": round(sum(byte_by_wl.get(w, 0) for w in supp) / total_byte, 4),
                "region_coverage": round(sum(region_by_wl.get(w, 0) for w in supp) / total_regions, 4),
                "boundary_pressure_score": int(b["boundary_pressure_score"]),
                "compiler_proof_status": status or "no_proof_axis",
                "overfit_risk": overfit,
            }
        )
    rows.sort(key=lambda r: (-r["mac_coverage"], -r["boundary_pressure_score"], r["abstraction"]))
    return rows


def _family_of(workload: str) -> str:
    from merlin.analysis.dse.guidance.models import MODEL_ARCH, _base_model

    base = _base_model(workload)
    if base and base in MODEL_ARCH:
        return MODEL_ARCH[base].family
    return "unknown"


def workload_family_summary(cs_dir: Path, findings: list[dict] | None = None) -> dict:
    """Group the recaptured workloads by architecture family and report the per-family signal
    profile, plus which findings are family-specific vs cross-family."""
    cs_dir = Path(cs_dir)
    ops = _csv_rows(cs_dir / "operator_shape_table.csv")
    shape = _csv_rows(cs_dir / "shape_summary_by_workload.csv")
    crit = {r["workload"]: r for r in _csv_rows(cs_dir / "critical_path_table.csv")}
    wls = sorted({o["workload"] for o in ops})
    from merlin.analysis.dse.guidance.models import MODEL_ARCH, _base_model

    fam: dict[str, dict] = {}
    for w in wls:
        f = _family_of(w)
        fam.setdefault(f, {"workloads": [], "total_macs": 0, "dominant_shape_class": "", "parallelism": [], "K": []})
        fam[f]["workloads"].append(w)
        fam[f]["total_macs"] += sum(int(o["macs"]) for o in ops if o["workload"] == w)
        if w in crit:
            fam[f]["parallelism"].append(float(crit[w]["available_parallelism"]))
        base = _base_model(w)
        if base and base in MODEL_ARCH:
            fam[f]["K"].append(MODEL_ARCH[base].loop_count)
    # dominant shape class per family = max mac_fraction-weighted class across its workloads
    for f, d in fam.items():
        agg: dict[str, float] = {}
        for r in shape:
            if r["workload"] in d["workloads"]:
                agg[r["shape_class"]] = agg.get(r["shape_class"], 0.0) + float(r["mac_fraction"])
        d["dominant_shape_class"] = max(agg, key=agg.get) if agg else ""
        d["parallelism_range"] = (
            [round(min(d["parallelism"]), 2), round(max(d["parallelism"]), 2)] if d["parallelism"] else []
        )
        d["K_range"] = [min(d["K"]), max(d["K"])] if d["K"] else []
        d.pop("parallelism")
    fam_specific, cross = [], []
    for fd in findings or []:
        if fd.get("presentation_placement") != "main":
            continue
        fams = {_family_of(w) for w in fd.get("supporting_workloads", []) if w not in ("ALL",)}
        fams.discard("unknown")
        rec = {"claim": fd.get("title", ""), "families": sorted(fams), "dse_question": fd.get("dse_question", "")}
        (fam_specific if len(fams) == 1 else cross).append(rec)
    return {"families": fam, "family_specific_findings": fam_specific, "cross_family_findings": cross}


# capture-fidelity asks that would most raise cross-workload confidence (what a recapture must keep).
_FIDELITY_ASKS = [
    "preserve the host K-loop (do not unroll the denoise/decode loop into a single pass)",
    "preserve packed low-bit weight layout + per-channel scales (do not dequantize to bf16)",
    "preserve the KV-cache / attention region (do not lower attention to dense matmuls)",
    "tag region roles (backbone-once vs repeated-head) in prov.fqn",
    "record the real loop count / control cadence (replace the assumed K reference)",
]


def corpus_expansion_plan(cs_dir: Path) -> dict:
    """Which registry families lack a committed recapture, grouped by family, plus the capture
    fidelity improvements that would most raise cross-workload confidence. A recommendation — no new
    data is ingested; this only states what additional captures the search-space study needs."""
    from merlin.analysis.dse.guidance.models import MODEL_ARCH, _base_model

    cs_dir = Path(cs_dir)
    ops = _csv_rows(cs_dir / "operator_shape_table.csv")
    captured = {b for b in (_base_model(w) for w in {o["workload"] for o in ops}) if b}
    missing = [m for m in MODEL_ARCH if m not in captured]
    by_family: dict[str, list[dict]] = {}
    for m in missing:
        a = MODEL_ARCH[m]
        by_family.setdefault(a.family, []).append(
            {"model": m, "loop_kind": a.loop_kind, "reference_K": a.loop_count, "note": a.note or ""}
        )
    return {
        "captured_models": sorted(captured),
        "captured_families": sorted({MODEL_ARCH[b].family for b in captured if b in MODEL_ARCH}),
        "missing_by_family": by_family,
        "n_missing": len(missing),
        "fidelity_asks": _FIDELITY_ASKS,
    }


def _signal_by_workload(cs_dir: Path):
    """Per-workload signal aggregates shared by the necessity classifier + robustness."""
    cs_dir = Path(cs_dir)
    ops = _csv_rows(cs_dir / "operator_shape_table.csv")
    shape = _csv_rows(cs_dir / "shape_summary_by_workload.csv")
    dm = _csv_rows(cs_dir / "data_movement_table.csv")
    shard = _csv_rows(cs_dir / "sharding_table.csv")
    epi = _csv_rows(cs_dir / "epilogue_pattern_table.csv")
    from merlin.analysis.dse.guidance.models import MODEL_ARCH, _base_model

    wls = sorted({o["workload"] for o in ops})
    sig = {}
    for w in wls:
        srows = [r for r in shape if r["workload"] == w]
        dense = sum(float(r["mac_fraction"]) for r in srows if r["shape_class"] == "squareish_gemm")
        true_gemv = sum(float(r["mac_fraction"]) for r in srows if r["shape_class"] == "gemv_like")
        skinny_gemm = sum(float(r["mac_fraction"]) for r in srows if r["shape_class"] in ("wide_skinny", "tall_skinny"))
        gemv = true_gemv + skinny_gemm
        rh = next((r for r in dm if r["workload"] == w and r["region"] == "repeated_head"), None)
        shrows = [r for r in shard if r["workload"] == w]
        erows = [r for r in epi if r["workload"] == w]
        base = _base_model(w)
        arch = MODEL_ARCH.get(base)
        sig[w] = {
            "family": arch.family if arch else "unknown",
            "K": arch.loop_count if arch else 1,
            "H": (arch.action_horizon or 0) if arch else 0,
            "control_rate": (arch.control_rate_hz if arch else None),
            "dense_mac": round(dense, 4),
            "gemv_mac": round(gemv, 4),
            "true_gemv_mac": round(true_gemv, 4),
            "skinny_gemm_mac": round(skinny_gemm, 4),
            "weight_bytes": int(rh["weight_bytes"]) if rh else 0,
            "avoidable_reload": int(rh["avoidable_weight_reload"]) if rh else 0,
            "invocations": int(rh["invocations"]) if rh else 0,
            "k_shard": any(
                r["axis"] == "K" and r["reduction_required"] == "True" and r.get("shardable_8") == "True"
                for r in shrows
            ),
            "mn_shard": any(
                r["axis"] in ("M", "N") and r["reduction_required"] == "False" and r.get("shardable_8") == "True"
                for r in shrows
            ),
            "epi_scale_act": any(r["has_scale"] == "True" and r["has_activation"] == "True" for r in erows),
            "epi_bias": any(r["has_bias"] == "True" for r in erows),
            "has_backbone": any(o["region_role"] == "backbone_once" for o in ops if o["workload"] == w),
            "has_repeated": any(o["region_role"] == "repeated_head" for o in ops if o["workload"] == w),
        }
    return sig, wls


_HOTSPOT_COLS = [
    "ranking",
    "rank",
    "workload",
    "op_index",
    "prov_fqn",
    "op_kind",
    "shape_class",
    "M",
    "N",
    "K",
    "macs",
    "mac_fraction_of_workload",
    "rhs_weight_bytes",
    "best_tile_padding_waste",
    "is_tail_heavy",
    "region",
    "avoidable_weight_reload",
]


_COVER_COLS = [
    "abstraction",
    "workloads_supporting",
    "n_workloads",
    "workload_coverage",
    "mac_coverage",
    "byte_coverage",
    "region_coverage",
    "boundary_pressure_score",
    "compiler_proof_status",
    "overfit_risk",
]


def _hotspots_md(h, scope) -> str:
    d = h.get("dominant_op")
    L = [
        f"# Per-operator hotspots ({scope})\n",
        "> Which few operators dominate the constraints DSE must size for. Structural quantities "
        "(MACs / weight bytes / tile padding waste / avoidable reload) recovered from the capture — "
        "no latency, throughput, or performance claim.\n",
        f"Total operators analyzed: **{h['n_ops']}**.\n",
    ]
    if d:
        L.append(
            f"**Dominant op (by MACs):** `{d['prov_fqn']}` in {d['workload']} — "
            f"{d['M']}x{d['N']}x{d['K']} = {d['macs']:,} MACs "
            f"({d['mac_fraction_of_workload']:.0%} of its workload), class {d['shape_class']}.\n"
        )
    L += [
        "## Top ops by MACs\n",
        "| workload | op | shape M×N×K | MACs | % of workload | class |",
        "|---|---|---|---|---|---|",
    ]
    for r in h["by_macs"]:
        L.append(
            f"| {r['workload']} | {r['prov_fqn']} | {r['M']}×{r['N']}×{r['K']} | "
            f"{r['macs']:,} | {r['mac_fraction_of_workload']:.0%} | {r['shape_class']} |"
        )
    L += [
        "\n## Top ops by tile padding waste (tile-hostility)\n",
        "| workload | op | shape M×N×K | best tile waste | class |",
        "|---|---|---|---|---|",
    ]
    for r in h["by_padding_waste"]:
        L.append(
            f"| {r['workload']} | {r['prov_fqn']} | {r['M']}×{r['N']}×{r['K']} | "
            f"{r['best_tile_padding_waste']:.3f} | {r['shape_class']} |"
        )
    L += [
        "\n## Regions by avoidable weight reload (residency target)\n",
        "| workload | region | avoidable reload (B) | weight bytes (B) |",
        "|---|---|---|---|",
    ]
    for r in h["by_avoidable_reload"]:
        L.append(f"| {r['workload']} | {r['region']} | {r['avoidable_weight_reload']:,} | {r['weight_bytes']:,} |")
    return "\n".join(L) + "\n"


def _coverage_md(rows, scope) -> str:
    L = [
        f"# Abstraction coverage ({scope})\n",
        "> Replaces raw support counts with corpus **coverage**: for each candidate system "
        "abstraction, the fraction of workloads / MACs / weight bytes / regions that imply it, its "
        "compiler-proof status, and an overfit flag (single-workload support is corpus-overfit "
        "risk). Support breadth, not a performance ranking.\n",
        "| abstraction | workloads | MAC cov | byte cov | proof | overfit |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        L.append(
            f"| {r['abstraction']} | {r['n_workloads']} ({r['workload_coverage']:.0%}) | "
            f"{r['mac_coverage']:.0%} | {r['byte_coverage']:.0%} | "
            f"{r['compiler_proof_status']} | {r['overfit_risk']} |"
        )
    return "\n".join(L) + "\n"


def _family_md(fs, scope) -> str:
    L = [
        f"# Workload-family summary ({scope})\n",
        "> The recaptured workloads grouped by architecture family, with each family's recovered "
        "signal profile and which findings are family-specific vs cross-family. Structural only.\n",
    ]
    for fam, d in sorted(fs["families"].items()):
        L.append(f"## {fam}\n")
        L.append(f"- workloads: {', '.join(d['workloads'])}")
        L.append(f"- total MACs (single-pass capture): {d['total_macs']:,}")
        L.append(f"- dominant shape class: {d['dominant_shape_class'] or '—'}")
        L.append(f"- inter-op parallelism (work/span) range: {d.get('parallelism_range') or '—'}")
        L.append(f"- reference loop count K range: {d.get('K_range') or '—'}\n")
    L.append("## Family-specific findings (one family only)\n")
    for f in fs["family_specific_findings"] or [{"claim": "—", "families": [], "dse_question": ""}]:
        L.append(f"- {f['claim']} _(families: {', '.join(f['families']) or '—'})_")
    L.append("\n## Cross-family findings (≥2 families)\n")
    for f in fs["cross_family_findings"] or [{"claim": "—", "families": [], "dse_question": ""}]:
        L.append(f"- {f['claim']} _(families: {', '.join(f['families']) or '—'})_")
    return "\n".join(L) + "\n"


def _corpus_md(cp, scope) -> str:
    L = [
        f"# Corpus-expansion plan ({scope})\n",
        "> A recommendation only — no new data is ingested here. Which registry model families lack "
        "a committed recapture, and the capture-fidelity improvements that would most raise "
        "cross-workload confidence before any quantitative DSE.\n",
        f"Captured models: {', '.join(cp['captured_models']) or '—'} "
        f"(families: {', '.join(cp['captured_families']) or '—'}).\n",
        f"**{cp['n_missing']} registry models lack a recapture.**\n",
        "## Missing captures by family\n",
        "| family | model | loop kind | reference K | note |",
        "|---|---|---|---|---|",
    ]
    for fam, models in sorted(cp["missing_by_family"].items()):
        for m in models:
            L.append(f"| {fam} | {m['model']} | {m['loop_kind']} | {m['reference_K']} | {m['note']} |")
    L.append("\n## Capture-fidelity asks (raise confidence on existing + new captures)\n")
    for a in cp["fidelity_asks"]:
        L.append(f"- {a}")
    return "\n".join(L) + "\n"
