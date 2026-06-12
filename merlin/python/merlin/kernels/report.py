"""Render the human-readable kernel-mining report (plain markdown, diff-friendly).

Sections: corpus summary, motif-frequency table (with the promotion verdict), promoted
abstraction candidates + policy rules, and a mandatory **Caveats** section that states the
limits of the evidence so the report never overclaims.
"""
from __future__ import annotations

import collections

from merlin.kernels.policy import CATALOG, MotifStat, PromotionResult, is_promotable


def _corpus_summary(records: list[dict]) -> str:
    by_source = collections.Counter(r.get("source", "?") for r in records)
    by_source_op = collections.Counter(
        (r.get("source", "?"), r.get("op", "?")) for r in records
    )
    lines = [f"- **Total kernels indexed:** {len(records)}"]
    for src, n in sorted(by_source.items()):
        lines.append(f"- **{src}:** {n}")
        ops = sorted([(op, c) for (s, op), c in by_source_op.items() if s == src],
                     key=lambda x: -x[1])
        op_str = ", ".join(f"{op}×{c}" for op, c in ops[:8])
        if op_str:
            lines.append(f"    - ops: {op_str}")
    return "\n".join(lines)


def _motif_table(stats: dict[str, MotifStat], promoted: set[str], min_kernels: int) -> str:
    header = ("| motif | kernels | sources | verdict |\n"
              "|---|---:|---|:--|")
    rows = []
    for motif, st in sorted(stats.items(), key=lambda kv: -kv[1].kernel_count):
        if motif in promoted:
            verdict = "✅ policy"
        elif motif not in CATALOG:
            verdict = "structural (no policy)" if is_promotable(st, min_kernels) else "—"
        else:
            verdict = "below gate"
        rows.append(f"| {motif} | {st.kernel_count} | {', '.join(sorted(st.sources))} | {verdict} |")
    note = ("\n\n_Promotion gate: ≥2 sources OR ≥%d kernels. 'structural' motifs clear the gate "
            "but are intentionally not mapped to a policy (too ubiquitous to be actionable)._" % min_kernels)
    return header + "\n" + "\n".join(rows) + note


def _artifacts(promo: PromotionResult) -> str:
    out = ["### Promoted abstraction candidates"]
    if promo.candidates:
        for c in promo.candidates:
            ev = ", ".join(c["evidence"])
            out.append(f"- **{c['name']}** ({c['kind']}) — {c['motivation']}")
            out.append(f"    - interface_features: {', '.join(c['interface_features'])}")
            out.append(f"    - evidence: {ev}")
    else:
        out.append("- _(none cleared the promotion gate)_")
    out.append("\n### Promoted policy rules")
    if promo.rules:
        for r in promo.rules:
            when = ", ".join(f"{k} {v}" for k, v in r["when"].items())
            out.append(f"- **{r['policy']}** — when: {when}")
            out.append(f"    - actions: {', '.join(r['actions'])}")
            out.append(f"    - evidence: {', '.join(r['evidence'])}")
    else:
        out.append("- _(none cleared the promotion gate)_")

    out.append("\n### Interface candidates (L5) — exposed via the 4 lowering variants")
    if promo.interfaces:
        for i in promo.interfaces:
            out.append(f"- **{i['name']}** — ops: {', '.join(i['interface_ops'])}; "
                       f"types: {', '.join(i['interface_types'])}")
            out.append(f"    - compiler must prove: {', '.join(i['compiler_must_prove'])}")
            out.append(f"    - hardware must provide: {', '.join(i['hardware_must_provide'])}")
            out.append(f"    - runtime must provide: {', '.join(i['runtime_must_provide'])}")
            out.append(f"    - lowering variants: {', '.join(i['lowering_variants'])}")
    else:
        out.append("- _(none)_")

    out.append("\n### Runtime candidates (L7)")
    if promo.runtime_candidates:
        for r in promo.runtime_candidates:
            obs = r.get("observed", {})
            obs_s = (f" (median {obs.get('median_dispatches_per_kernel')} dispatches/kernel, "
                     f"{obs.get('small_dispatch_fraction')} small)" if obs else "")
            out.append(f"- **{r['name']}**{obs_s}")
            out.append(f"    - compiler action: {', '.join(r['compiler_action'])}")
            out.append(f"    - runtime requirement: {', '.join(r['runtime_requirement'])}")
    else:
        out.append("- _(none)_")

    if promo.dialect_requirements:
        out.append("\n### Dialect requirements (L6 — input to TargetGen, status `proposed`)")
        for d in promo.dialect_requirements:
            out.append(f"- **{d['source_abstraction']}** @ {d['target']} — "
                       f"ops: {', '.join(d['required_ops'])}; "
                       f"types: {', '.join(d['required_types'])}; "
                       f"verifiers: {', '.join(d['required_verifiers'])}")
        out.append("\n### LLVM requirements (L8)")
        out.append(f"- All {len(promo.llvm_requirements)} emitted with "
                   "`requires_llvm_fork: false` — no machine-code change is justified until "
                   "Stage F (target lowering) and Stage G (exploitability) pass. Recorded "
                   "fork triggers name what *would* justify one.")
    return "\n".join(out)


def _regime_matrix(rm: dict) -> list[str]:
    """Render a shape_sweep result as a reuse × (K, tail) fires/silent table."""
    cells = {(c["reuse"], c["K"], c["tail_heavy"]): c["status"] for c in rm["cells"]}
    cols = sorted({(k, t) for (_, k, t) in cells}, key=lambda kt: (kt[0], kt[1]))
    mark = {"holds": "✓ fires", "fails": "· silent", "n/a": "n/a"}
    head = "| reuse | " + " | ".join(f"K={k}{' (tail)' if t else ''}" for k, t in cols) + " |"
    sep = "|---:|" + "|".join(["---"] * len(cols)) + "|"
    rows = []
    for r in sorted({r for (r, _, _) in cells}):
        rows.append(f"| {r} | " + " | ".join(mark.get(cells[(r, k, t)], "?")
                                             for k, t in cols) + " |")
    neg = ", ".join(f"{n}: **{st}**" for n, st in rm["negative_controls"].items())
    return ["    - regime matrix (fires {}/{} cells; negative controls — {}):".format(
                rm["fires"], len(rm["cells"]), neg),
            "", head, sep, *rows, ""]


def _validation(validation: dict | None) -> str:
    if not validation:
        return ""
    out = ["## Held-out validation (Stage D — symbolic, no execution)"]
    for policy_name, info in validation.items():
        verdicts = (", ".join(f"{wl}: **{st}**" for wl, st in info["workloads"].items())
                    or "_no benchmark workload mapped_")
        out.append(f"- **{policy_name}** — {verdicts}")
        for row in info.get("capacity_sweep", []):
            fit = "fits" if row["fits"] else "OVERFLOW"
            out.append(f"    - capacity @ {row['resident_store_bytes']}B: "
                       f"footprint {row['footprint_bytes']}B → {fit}")
        rm = info.get("regime_matrix")
        if rm == "shape_independent":
            out.append("    - regime matrix: shape-independent (`when` references no shape facts)")
        elif isinstance(rm, dict):
            out.extend(_regime_matrix(rm))
    return "\n".join(out)


# Which downstream Merlin construct each policy drives, and what would disprove it.
_POLICY_CONSUMERS = {
    "packed_rhs_policy": "`merlin.schedule.hoist_pack` → `merlin.interface.resident_pack`",
    "accumulator_commit_policy": "`merlin.interface.accumulator` + `commit`",
    "vl_agnostic_loop_policy": "`merlin.schedule` VL-polymorphic loop emission",
    "double_buffer_policy": "`merlin.schedule` overlap → `merlin.interface.async_copy`",
    "weight_stationary_dataflow_policy": "`target_contract.dataflow` + schedule placement",
}
_POLICY_FALSIFIERS = {
    "packed_rhs_policy": "no_reuse_matmul; mutable-RHS control",
    "accumulator_commit_policy": "no_reuse_matmul (no epilogue)",
    "vl_agnostic_loop_policy": "fixed-width-only target",
    "double_buffer_policy": "single-buffer-capacity target",
    "weight_stationary_dataflow_policy": "output-stationary target contract",
}


def _scorecard(stats: dict[str, MotifStat], promo: PromotionResult,
               validation: dict | None) -> str:
    """One actionability row per promoted policy: evidence breadth, Stage-D, consumer,
    falsifier, and the concrete next promotion step."""
    if not promo.rules:
        return ""
    motif_for_policy = {e["policy"]["policy"]: m for m, e in CATALOG.items()}
    iface_names = {i["name"] for i in promo.interfaces}
    iface_for_motif = {m: e["interface"]["name"] for m, e in CATALOG.items()
                       if e.get("interface")}
    out = ["## Actionability scorecard",
           "",
           "| policy | kernels | sources | op families | Stage-D | regime sweep | "
           "drives | falsifier | next step |",
           "|---|---:|---:|---:|---|---|---|---|---|"]
    for rule in promo.rules:
        name = rule["policy"]
        motif = motif_for_policy.get(name, "?")
        st = stats.get(motif, MotifStat())
        n_ops = len({eid.rsplit("_", 1)[-1] for eid in st.evidence_ids})
        v = (validation or {}).get(name, {})
        wl = v.get("workloads") or {}
        stage_d = ("; ".join(f"{k.split('_')[0]}:{s}" for k, s in wl.items())
                   if wl else "no workload mapped")
        rm = v.get("regime_matrix")
        regime = (f"fires {rm['fires']}/{len(rm['cells'])}, controls silent"
                  if isinstance(rm, dict) else "shape-independent")
        iface = iface_for_motif.get(motif)
        nxt = (f"Stage F: lower `{iface}` per dialect requirement (toy_npu)"
               if iface and iface in iface_names
               else "measure on real shapes (no HW/SW interface needed)")
        out.append(f"| {name} | {st.kernel_count} | {len(st.sources)} | {n_ops} | {stage_d} "
                   f"| {regime} | {_POLICY_CONSUMERS.get(name, '—')} "
                   f"| {_POLICY_FALSIFIERS.get(name, '—')} | {nxt} |")
    return "\n".join(out)


def _invariants(inv: dict | None) -> str:
    if not inv:
        return ""
    out = ["## Consistency invariants"]
    for c in inv["checks"]:
        mark = "✅" if c["status"] == "ok" else "⚠️"
        line = f"- {mark} {c['name']} ({c['violations']} violations)"
        out.append(line)
        for ex in c["examples"]:
            out.append(f"    - `{ex}`")
    if inv["surprises"]:
        out += ["", "### Surprises — motifs on unexpected op families "
                    "(marker bug *or* genuine insight; audit each)",
                "", "| motif | op | source | kernels | example |", "|---|---|---|---:|---|"]
        for s in inv["surprises"]:
            out.append(f"| {s['motif']} | {s['op']} | {s['source']} | {s['count']} "
                       f"| `{s['example']}` |")
    else:
        out.append("- ✅ no motif fired on an unexpected op family")
    return "\n".join(out)


def _plots(plot_paths: list | None) -> str:
    if not plot_paths:
        return ""
    out = ["## Plots", "",
           "_Evidence frequency only — no kernel was executed; nothing here is a speedup._",
           ""]
    for p in plot_paths:
        name = p.stem.replace("_", " ")
        out.append(f"### {name}")
        out.append(f"![{p.stem}](plots/{p.name})")
        out.append("")
    return "\n".join(out)


def _caveats(diagnostics: dict | None) -> str:
    exo = (diagnostics or {}).get("exo", {})
    exo_line = ""
    if exo:
        exo_line = (f"\n- **Exo:** compiled {exo.get('compiled', 0)} procs from "
                    f"{exo.get('specs', 0)} specs; {exo.get('skipped', 0)} skipped "
                    f"(import/compile failures, logged).")
    return (
        "## Caveats (read before trusting any policy)\n"
        "- Motifs are *decisions* extracted by deterministic markers, not measured speedups. "
        "**No kernel was executed or timed.** Policies are validated only by symbolic match "
        "against the benchmark workloads (positive fires / negative control silent)."
        "\n- **Autocomp:** shapes/dtypes are parsed from the `void test(...)` C signature; the "
        "Autocomp `score` is recorded in metadata only and is NOT treated as correctness."
        "\n- **Autocomp:** ~1700 of the 2637 manifest entries are 0-byte dedup placeholders and "
        "are skipped; counts reflect real, non-empty kernels only."
        f"{exo_line}"
        "\n- **Triton / triton-cpu:** the mined corpora are tutorial + shipped-kernel trees — "
        "pedagogical but real optimization decisions; verbatim copies across the two repos are "
        "deduplicated by content hash before counting."
        "\n- **OpenBLAS:** BLAS1/2 kernels are precision-generic via `DOUBLE` macros, so their "
        "dtype is recorded as `unknown`; scalar fallback files are skipped."
        "\n- Plots visualize evidence *frequency*, never speedup."
        "\n- A promoted motif is a *policy candidate*, not a proven compiler abstraction. "
        "Promotion to a dialect op/type requires held-out-shape and target-lowering validation "
        "(later sessions). This report does **not** claim automatic abstraction discovery."
    )


def _memory_behavior_highlight(records: list[dict]) -> str:
    """Show one fully-resolved per-tensor memory_behavior example (L2)."""
    for r in records:
        mb = (r.get("features", {}) or {}).get("memory_behavior")
        if mb and mb.get("rhs", {}).get("reuse_count", 0) >= 2:
            ev = r.get("evidence", {}).get("id", r.get("path", "?"))
            rhs, acc = mb["rhs"], mb["acc"]
            return ("## L2 memory roles (example)\n"
                    f"`{ev}` — op_sequence {r['features'].get('op_sequence')}:\n"
                    f"- **rhs**: {rhs['role']}, immutable={rhs['immutable']}, "
                    f"measured reuse_count={rhs['reuse_count']}, packed_once={rhs['packed_once']}\n"
                    f"- **acc**: {acc['role']}, widening={acc['widening']}, "
                    f"materialized_before_epilogue={acc['materialized_before_epilogue']}\n"
                    f"- **lhs**: streaming_activation  ·  **output**: committed_output")
    return ""


def write_report(
    records: list[dict],
    stats: dict[str, MotifStat],
    promo: PromotionResult,
    diagnostics: dict | None = None,
    min_kernels: int = 10,
    llm_summary: str | None = None,
    validation: dict | None = None,
    invariants: dict | None = None,
    plot_paths: list | None = None,
) -> str:
    """Return the full markdown report as a string."""
    dedup = (diagnostics or {}).get("dedup", {})
    dedup_line = (f"\n- _{dedup['duplicates_skipped']} kernels vendored verbatim across "
                  f"sources were deduplicated by content hash ({dedup['by_source']})._"
                  if dedup.get("duplicates_skipped") else "")
    parts = [
        "# Kernel mining report",
        "",
        "## Corpus summary",
        _corpus_summary(records) + dedup_line,
        "",
        "## Motifs",
        _motif_table(stats, promo.promoted, min_kernels),
        "",
        _artifacts(promo),
        "",
    ]
    sc = _scorecard(stats, promo, validation)
    if sc:
        parts += [sc, ""]
    mb = _memory_behavior_highlight(records)
    if mb:
        parts += [mb, ""]
    val = _validation(validation)
    if val:
        parts += [val, ""]
    inv = _invariants(invariants)
    if inv:
        parts += [inv, ""]
    pl = _plots(plot_paths)
    if pl:
        parts += [pl, ""]
    if llm_summary:
        parts += ["## LLM summary (advisory, non-authoritative)", llm_summary, ""]
    parts.append(_caveats(diagnostics))
    return "\n".join(parts) + "\n"
