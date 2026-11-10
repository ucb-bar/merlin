"""Render cross-analysis findings, evidence limits and presentation summaries.

Evidence labels, numerical caveats and artifact schemas retain their original
meaning; these analyses establish no hardware performance qualification.
"""

from __future__ import annotations

from pathlib import Path

from .capture import _capture_fidelity_md
from .evidence import DSE_QUESTIONS
from .findings import _scorecard16_md
from .necessity import _necessity_md
from .primitives import _frontier_md, _pareto_md, _robustness_md
from .workloads import _coverage_md, _hotspots_md


def _readme_md(bundle, rendered) -> str:
    s = bundle["evidence_strength"]
    npass = sum(1 for ok, _ in bundle["consistency_checks"] if ok)
    return (
        f"# Insight-mining run — scope: {bundle['scope']}\n\n"
        "Meta-analysis of the committed dse_guidance case-study package. Regeneratable; not "
        "committed. **No speedup / cycles / area / energy / performance claim.**\n\n"
        f"- normalized facts: {s['total_facts']} (tiers "
        f"A={s['by_tier'].get('A', 0)} B={s['by_tier'].get('B', 0)} C={s['by_tier'].get('C', 0)} "
        f"D={s['by_tier'].get('D', 0)})\n"
        f"- main findings: {sum(1 for f in bundle['findings'] if f['presentation_placement'] == 'main')}"
        f" · plots rendered: {len(rendered)}\n"
        f"- consistency checks: {npass}/{len(bundle['consistency_checks'])}\n"
        f"- **devil's-advocate gap audit: {len(bundle.get('open_avoidable_gaps', []))} open "
        f"avoidable gaps** (target 0); "
        f"{len(bundle.get('required_inputs', []))} inherent limits scoped to required inputs\n\n"
        "Files: artifact_inventory.{csv,md}, unified_fact_table.{csv,yaml}, "
        "evidence_strength_table.csv + evidence_strength_report.md, dse_usefulness_scorecard.md "
        "+ dse_usefulness_answers.yaml, presentation_candidate_findings.{md,csv}, "
        "plot_manifest.{yaml,md}, generated_plots/, consistency_checks.md, "
        "gap_audit_report.{md,csv}, required_inputs_manifest.{yaml,md}.\n"
    )


def _signal_report_md(bundle, scope) -> str:
    canon = bundle["canonical_signal"]
    findings = [f for f in bundle["findings"] if f["presentation_placement"] == "main"]
    by_q_find: dict[str, list] = {}
    for f in findings:
        by_q_find.setdefault(f.get("dse_question", ""), []).append(f)
    by_q_metric: dict[str, set] = {}
    for r in canon:
        by_q_metric.setdefault(r["dse_question"], set()).add(r["metric"])
    L = [
        f"# Signal findings report ({scope})\n",
        "> Given only workload artifacts: what changes the future DSE search space. Every finding "
        "is recovered/derived from the captures (or a host measurement); **no quantity is claimed "
        "for unbuilt hardware** (cycles / area / energy / throughput are refused, not estimated). "
        "Organized by the DSE question each metric answers.\n",
    ]
    for q in DSE_QUESTIONS:
        fs = by_q_find.get(q, [])
        ms = sorted(by_q_metric.get(q, set()))
        L.append(f"## {q}\n")
        if not fs and not ms:
            L.append("_No signal recovered for this question from the current corpus._\n")
            continue
        L.append(f"_Signal metrics: {', '.join(ms) or '—'}_\n")
        for f in fs:
            L.append(
                f"- **{f['title']}** [tier {f['evidence_tier']}] — {f['dse_implication']}  "
                f"_(workloads: {', '.join(f['supporting_workloads'])})_"
            )
        L.append("")
    # what remains unclaimed
    L.append("## What remains unclaimed (and the exact input needed)\n")
    for x in bundle.get("required_inputs", []):
        L.append(f"- {x['limit']}: {x['required_input']}")
    # closing devil's-advocate note: which signal is robust vs corpus-limited
    fam = bundle.get("family_summary", {})
    n_fam_specific = len(fam.get("family_specific_findings", []))
    L.append("\n## Devil's advocate — robust vs corpus-limited\n")
    L.append(
        "**Robust (structural, independent of magnitudes):** shape-class distribution, the "
        "recovered SSA data-dependency graph, the backbone/head role split, the dtype/epilogue "
        "numerical contract, and per-op MAC *fractions* (relative, not absolute). These hold "
        "regardless of weight magnitudes."
    )
    L.append("\n**Corpus-limited (treat as directional, not settled):**")
    L.append(
        "- The 4 recaptures are small, random-init f32 instances — structure and provenance "
        "are real, but absolute byte/MAC magnitudes are a small instance, so any finding that "
        "leans on absolute size is directional only."
    )
    L.append(
        "- low-bit / KV / attention structure is erased or lowered in the capture, so those "
        "candidates are blocked, not measured (see What remains unclaimed)."
    )
    if bundle.get("scope") == "all":
        L.append(
            f"- {n_fam_specific} family-specific findings recovered: the corpus is small and "
            "structurally homogeneous across families, so cross-family separation is weak — "
            "expanding the corpus (see corpus_expansion_plan.md) is the prerequisite before any "
            "cross-family DSE claim."
        )
    return "\n".join(L) + "\n"


def _slides_md(bundle, scope) -> str:
    nec = bundle.get("abstraction_necessity", {}).get("rollup", {})
    fr = bundle.get("primitive_frontier", {}).get("best_by_size", {})
    s1, s2 = fr.get(1), fr.get(2)
    L = [
        f"# Presentation slide candidates ({scope})\n",
        "> Show these, not the descriptive QA plots. Each slide = one DSE claim + its evidence + "
        "caveat. Structural only; no speedup/area/energy.\n",
    ]
    slides = [
        (
            "What Merlin does",
            "flat capture -> workload contract -> DSE search-space axes (it "
            "enumerates the search space; it does not choose a design)",
            "dse_search_space_knobs",
            "no performance claimed",
        ),
        (
            "Primitive set is a DSE axis",
            (
                f"one primitive worst-cov {s1['worst']:.2f}; {'+'.join(s2['set'])} -> {s2['worst']:.2f}"
                if s1 and s2
                else "best 2-set covers the corpus a single primitive cannot"
            ),
            "primitive_set_frontier",
            "set-union coverage, structural",
        ),
        (
            "Residency is a loop/rate abstraction",
            "weight bytes moved grows with K under reload; flat if resident; threshold set by dtype",
            "decision_weight_residency + decision_capacity_dtype",
            "K is configured/reference",
        ),
        (
            "Inter-op parallelism is low",
            "work/span ~1.1-1.6 -> pushes DSE to intra-op sharding / pipelining / specialized units",
            "critical_path_parallelism",
            "flattened capture may erase loop/pipeline parallelism",
        ),
        (
            "HW/SW boundary placement is a search axis",
            (
                f"abstraction necessity: {nec.get('necessary', 0)} necessary / {nec.get('blocked', 0)} "
                f"blocked -> DSE searches WHERE state/loops/layout/sync/reductions live"
            ),
            "boundary_necessity_matrix",
            "categorical, not a score",
        ),
        (
            "Capture fidelity is the limiting factor",
            "the flat capture erases K-loop / KV / packed-layout the loop & residency claims need",
            "capture_fidelity_matrix",
            "the central next-step result, not a side note",
        ),
    ]
    for i, (title, claim, art, cav) in enumerate(slides, 1):
        L.append(f"## Slide {i} — {title}\n")
        L.append(f"- **claim:** {claim}")
        L.append(f"- **show:** `{art}`")
        L.append(f"- **caveat:** {cav}\n")
    return "\n".join(L) + "\n"


def _findings_digest_md(bundle: dict, cs_dir: Path, rendered: list[str]) -> str:
    """ONE self-contained findings document per run: every metric table, the operator hotspots, the
    decision-impact plots, and the DSE-ingest knobs in a single file you can open and evaluate.

    Pure presentation of already-computed bundle data + the committed knob catalog — no new numbers.
    """
    cs_dir = Path(cs_dir)
    scope = bundle["scope"]
    canon = bundle["canonical_signal"]
    L = [
        f"# DSE findings digest — scope: {scope}\n",
        "Self-contained summary of the workload-contract analysis. Every number is recovered from "
        "the captures or a host measurement; **no quantity is claimed for unbuilt hardware**. "
        "Each metric carries an evidence tier (A/measured = IR or real measurement; B = "
        "recovered/derived + recompute check; C = config/assumed; D = unavailable) so you can "
        "weight it yourself. Source CSV/YAML are in this same folder; the full per-fact trace "
        "(metric -> source artifact -> check) is in `unified_fact_table.csv`.\n",
        "## 1. Headline metrics (canonical_signal_table.csv)\n",
        "Grouped by the DSE question each answers. `entity` is the thing the metric is about "
        "(workload / abstraction / region).\n",
    ]
    for q in DSE_QUESTIONS:
        qrows = [r for r in canon if r["dse_question"] == q]
        if not qrows:
            continue
        L.append(f"### {q}\n")
        L.append("| metric | entity | value | unit | tier | strength | implication |")
        L.append("|---|---|---|---|---|---|---|")
        for r in qrows:
            L.append(
                f"| {r['metric']} | {r['entity']} | {r['value']} | {r['unit']} | "
                f"{r['evidence_tier']} | {r['strength']} | {r['dse_implication']} |"
            )
        L.append("")
    # 2. operator hotspots (reuse the dedicated builder; replace its H1 with our section header)
    L.append("## 2. Per-operator hotspots\n")
    L.append(_hotspots_md(bundle["hotspots"], scope).split("\n", 1)[1])
    # 3. abstraction NECESSITY (strict; the decision-discriminating view) + frontier + pareto
    L.append("## 3. Abstraction necessity (strict — what DSE should commit to)\n")
    L.append(_necessity_md(bundle["abstraction_necessity"], scope).split("\n", 1)[1])
    L.append("## 4. Primitive-set frontier\n")
    L.append(_frontier_md(bundle["primitive_frontier"], scope).split("\n", 1)[1])
    L.append("## 5. Operator Pareto hotspots\n")
    L.append(_pareto_md(bundle["operator_pareto"], scope).split("\n", 1)[1])
    L.append("## 6. Capture fidelity (what the flat capture erased)\n")
    L.append(_capture_fidelity_md(bundle["capture_fidelity"], scope).split("\n", 1)[1])
    L.append("## 7. Decision-question scorecard\n")
    L.append(_scorecard16_md(bundle["decision_scorecard"], scope).split("\n", 1)[1])
    if bundle.get("robustness", {}).get("findings"):
        L.append("## 8. Leave-one-workload-out robustness\n")
        L.append(_robustness_md(bundle["robustness"], scope).split("\n", 1)[1])
    # the permissive coverage is kept only as a secondary 'possible-placement' reference
    if bundle["abstraction_coverage"]:
        L.append("## Appendix — abstraction support breadth (possible-placement view only)\n")
        L.append(_coverage_md(bundle["abstraction_coverage"], scope).split("\n", 1)[1])
    # decision-impact plots
    L.append("## Decision-impact plots (what changes if DSE picks differently)\n")
    L.append(
        "Structural what-if curves (bytes / coverage / counts — never latency or speedup). "
        "PNGs under `generated_plots/`.\n"
    )
    for p in bundle["plots"]:
        if p["plot_id"].startswith("decision_") and p["plot_id"] in rendered:
            L.append(f"- **{p['title']}** (`generated_plots/{p['plot_id']}.png`)  \n  {p.get('dse_caption', '')}")
    # 5. the DSE-ingest knob catalog (embed the committed consolidated table verbatim)
    L.append("\n## What a DSE tool ingests — knob catalog\n")
    knobs = cs_dir / "dse_search_space_knobs.md"
    if knobs.is_file():
        body = knobs.read_text().split("\n", 1)[1] if "\n" in knobs.read_text() else ""
        L.append(body.strip())
    else:
        L.append("_dse_search_space_knobs.md not found in the package._")
    # 6. how to read / reproduce
    L.append("\n## How to evaluate this yourself\n")
    L.append(
        "- Every headline row traces through `unified_fact_table.csv` "
        "(`metric_name -> source_artifact -> verifying_check`)."
    )
    L.append(
        "- The numbers are recomputed independently by "
        "`experiments/benchmarks/dse_guidance/verify_implementation.py` (run it; exit 0 = all checks "
        "pass)."
    )
    L.append(
        "- Regenerate this whole folder with `merlin-dse-guidance --insight-mining` "
        "(add `--workload <name>` for one network)."
    )
    return "\n".join(L) + "\n"


def _p17_findings_md(bundle, scope) -> str:
    aud = bundle.get("adversarial_audit", {}).get("rows", [])
    L = [
        f"# P17 final report — what is robust, what is blocked ({scope})\n",
        "> Per headline finding: a verdict from the adversarial audit (robust / winner-robust-but-"
        "magnitude-fragile / blocked-by-capture / descriptive), then the captures that would most "
        "improve the study. Structural only; timing rows are requirements, not measured performance.\n",
        "## Finding verdicts\n",
        "| conclusion | verdict | slide |",
        "|---|---|---|",
    ]
    for r in aud:
        v = r["verdict"]
        slide = "backup" if ("blocked" in v or "descriptive" in v) else "main"
        L.append(f"| {r['conclusion']} | {v} | {slide} |")
    L += [
        "\n## Most valuable next captures (to unblock the blocked findings)\n",
        "1. **Loop-preserving capture** — turns configured/reference K into IR-recovered loop "
        "structure; unblocks `resident_weight_object` / `bounded_loop_command` and grounds the "
        "timing-requirement envelope.",
        "2. **Loop/KV-preserving capture** — attention bmm / softmax / norm are already recovered "
        "(see `omitted_operator_accounting.md`); what remains is the KV *state* across the decode "
        "loop and the K-loop trip count.",
        "3. **Low-bit recapture** (packed weights + scales + per-format accuracy) — unblocks the "
        "low-bit abstractions (blocked today by the f32 fake-quant capture).",
        "4. **Measured dispatch counts** beyond small_llama — turns the command-rate envelope from "
        "proxy-only into a real requirement.\n",
        "## Headline for the talk\n",
        "Compiler-derived workload contracts tell a DSE tool which search axes to include before "
        "any hardware exists — primitive-SET coverage, loop/rate residency, operator concentration, "
        "HW/SW boundary placement — **and** reveal when the capture itself is too flat to decide "
        "(low-bit, KV, and the K-loop are erased). Capture fidelity is part of the methodology.\n",
    ]
    return "\n".join(L) + "\n"
