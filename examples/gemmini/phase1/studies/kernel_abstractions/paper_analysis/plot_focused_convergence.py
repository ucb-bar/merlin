"""One focused EL2/EL3 figure, from immutable CSV observations only.

Coverage belongs to one exact compiler source, accumulated over its grading
batches. The matching goal is every passing case of EL2's best observed source,
not a cardinality assembled from different compiler revisions. This is an
exploratory, one-author-per-condition comparison of the two authoring bundles.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from analyze_results import distinct_attempts, observed_failure, time_of

RUNS = ("el2", "el3")
COLORS = {"el2": "#777777", "el3": "#0072B2"}
LABELS = {"el2": "C++ dialect authoring (EL2)", "el3": "xDSL-assisted authoring (EL3)"}
TOKEN_FIELDS = ("tokens_total", "tokens_input", "tokens_fresh_input", "tokens_cache_read", "tokens_cache_write", "tokens_output", "tokens_reasoning")


def sources_and_curve(rows, case_ids=None):
    """Last explicit L2 outcome per source/case; no unrun-state pass erasure."""
    states, counts, curve, evidence = {}, {}, [(0.0, 0)], {}
    for r in sorted(rows, key=lambda r: (time_of(r), r["record_id"])):
        if r["record_type"] != "grade_case" or r["l2_status"] not in ("pass", "fail") or not r["source_sha256"]:
            continue
        if case_ids is not None and r["case_id"] not in case_ids:
            continue
        source, case = r["source_sha256"], r["case_id"]
        prior = states.setdefault(source, {}).get(case)
        states[source][case] = r["l2_status"]
        counts[source] = counts.get(source, 0) + (r["l2_status"] == "pass") - (prior == "pass")
        evidence.setdefault(source, {})[case] = r
        best = max(counts.values(), default=0)
        t = time_of(r)
        if curve[-1][0] == t:
            curve[-1] = t, best
        elif curve[-1][1] != best:
            curve.append((t, best))
    return states, curve, evidence


def first_source_reaching(rows, required):
    states = {}
    for r in sorted(rows, key=lambda r: (time_of(r), r["record_id"])):
        if r["record_type"] != "grade_case" or r["l2_status"] not in ("pass", "fail") or not r["source_sha256"]:
            continue
        source, case = r["source_sha256"], r["case_id"]
        states.setdefault(source, {})[case] = r["l2_status"]
        passing = {c for c, status in states[source].items() if status == "pass"}
        if required <= passing:
            return dict(at_s=time_of(r), source_sha256=source, required_case_ids=sorted(required),
                        passing_cases=len(passing), passing_case_ids=sorted(passing))
    return None


def usage_at(rows, t):
    usage = [r for r in rows if r["record_type"] == "provider_usage" and time_of(r) <= t]
    keys = [(r["thread_id"], r["response_id"]) for r in usage]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate provider response")
    return {k: sum(int(r[k]) for r in usage) for k in TOKEN_FIELDS}


def derive(rows):
    selected = {n: sorted([r for r in rows if r["experiment"] == n and time_of(r) is not None
                          and 0 <= time_of(r) <= 14400], key=lambda r: (time_of(r), r["record_id"])) for n in RUNS}
    baseline, _, _ = sources_and_curve(selected["el2"])
    baseline_source = max(baseline, key=lambda source: sum(v == "pass" for v in baseline[source].values()))
    goal = {c for c, status in baseline[baseline_source].items() if status == "pass"}
    result = dict(schema="merlin.focused_el23_convergence.v1", selection="EL2 versus EL3 authoring bundles; exploratory, one independent author each",
                  goal="All passing case IDs of the best observed EL2 source in the fixed four-hour window",
                  goal_cases=len(goal), goal_case_ids=sorted(goal), runs={})
    for n in RUNS:
        rr = selected[n]
        states, curve, evidence = sources_and_curve(rr)
        milestone = first_source_reaching(rr, goal)
        if milestone is None:
            raise ValueError(f"{n} does not reach the baseline case set")
        t = milestone["at_s"]
        # At a shared publication timestamp, all rows in that publication are
        # available together, even if CSV sort order crosses the goal mid-file.
        available = [r for r in rr if time_of(r) <= t]
        at_states, _, at_evidence = sources_and_curve(available)
        source = milestone["source_sha256"]
        passed = {c for c, state in at_states[source].items() if state == "pass"}
        milestone.update(passing_cases=len(passed), passing_case_ids=sorted(passed),
                         tokens=usage_at(rr, t),
                         failed_selfcheck_cases=sum(observed_failure(r) for r in distinct_attempts(available)),
                         selfcheck_case_attempts=len(distinct_attempts(available)))
        _, comparison_curve, _ = sources_and_curve(rr, case_ids=goal)
        result["runs"][n] = dict(milestone=milestone, curve=[list(p) for p in curve],
            comparison_curve=[list(p) for p in comparison_curve],
            best_observed_source_cases=max(sum(v == "pass" for v in s.values()) for s in states.values()),
            evidence=[{k: at_evidence[source][case].get(k) for k in
                ("record_id", "source_sha256", "case_id", "l2_status", "numeric_status", "observed_at", "publication_at", "evidence_path", "evidence_sha256", "evidence_locator", "elf_sha256")}
                for case in sorted(passed)])
    a, b = (result["runs"][n]["milestone"] for n in RUNS)
    result["reduction_vs_el2"] = dict(time=1-b["at_s"]/a["at_s"],
        total_tokens=1-b["tokens"]["tokens_total"]/a["tokens"]["tokens_total"],
        failed_selfcheck_cases=1-b["failed_selfcheck_cases"]/a["failed_selfcheck_cases"])
    result["uncached_input_plus_output_change"] = (
        (b["tokens"]["tokens_fresh_input"]+b["tokens"]["tokens_output"])/
        (a["tokens"]["tokens_fresh_input"]+a["tokens"]["tokens_output"])-1)
    result["limits"] = ["L2 execution evidence, not full L3 or hardware qualification",
        "Multiple immutable grading batches may contribute to one source; different sources are never pooled",
        "Published observations include validation delay and completed-response usage is a lower bound",
        "Recorded total tokens include cached input; uncached input plus output is slightly higher for EL3 at this milestone",
        "EL1 reaches the shared case set sooner; this focused comparison does not establish superiority over direct C++",
        "EL3 is a bundle of xDSL and authoring support, not an isolated language/dialect component toggle",
        "Exploratory one-run comparison; no significance or population effect claim"]
    return result, selected


def render(result, selected, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.6, "axes.titlesize": 9,
        "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "svg.fonttype": "none", "lines.linewidth": 1.65})
    fig, axes = plt.subplots(1, 3, figsize=(7.05, 3.4),
                             gridspec_kw={"width_ratios": [1.4, 1, 1]})
    fig.subplots_adjust(left=.075, right=.985, bottom=.28, top=.705, wspace=.48)
    goal = result["goal_cases"]
    baseline = result["runs"]["el2"]["milestone"]
    treatment = result["runs"]["el3"]["milestone"]
    for n in RUNS:
        data = result["runs"][n]
        m = data["milestone"]
        curve = [p for p in data["comparison_curve"] if p[0] <= m["at_s"]]
        curve.append([m["at_s"], goal])
        axes[0].step([t/60 for t,_ in curve], [v for _,v in curve], where="post", color=COLORS[n], lw=1.8)
        axes[0].plot(m["at_s"]/60, goal, "o", color=COLORS[n], ms=4.5)
        axes[0].vlines(m["at_s"]/60, 0, goal, color=COLORS[n], alpha=.4, lw=.75, ls=":")
        axes[0].annotate(f"{m['at_s']/60:.0f} min", (m["at_s"]/60, goal),
            xytext=(-4,8) if n=="el3" else (-2,-16), textcoords="offset points",
            ha="right", color=COLORS[n], fontsize=8, fontweight="bold")
    axes[0].axhline(goal, color="#aaaaaa", lw=.7, ls=":", zorder=0)
    axes[0].axvspan(treatment["at_s"]/60, baseline["at_s"]/60, color=COLORS["el3"], alpha=.055, zorder=0)
    axes[0].set(xlabel="Time since author start (min)", ylabel="Public tests passing Spike",
                xlim=(0,165), ylim=(0,145), xticks=[0,40,80,120,160], yticks=[0,32,64,96,128])
    reductions = result["reduction_vs_el2"]
    axes[0].set_title(f"(a) Time to the same coverage\n{reductions['time']:.0%} less time", loc="left", fontsize=8.3, fontweight="bold", color=COLORS["el3"])
    for ax, field, title, ylabel in (
        (axes[1], "tokens", f"(b) Total token usage\n{reductions['total_tokens']:.0%} fewer tokens", "Tokens including cache (M)"),
        (axes[2], "failures", f"(c) Failed checks\n{reductions['failed_selfcheck_cases']:.0%} fewer", "Failed self-check cases")):
        vals = []
        for i,n in enumerate(RUNS):
            m = result["runs"][n]["milestone"]
            value = m["tokens"]["tokens_total"]/1e6 if field=="tokens" else m["failed_selfcheck_cases"]
            vals.append(value)
            ax.bar(i, value, width=.57, color=COLORS[n])
            ax.text(i, value+max(vals)*.025, f"{value:.1f}M" if field=="tokens" else str(value),
                    ha="center", va="bottom", fontsize=8.5, color=COLORS[n], fontweight="bold")
        ax.set(xticks=[0,1], xticklabels=["C++", "xDSL"], ylabel=ylabel, ylim=(0,max(vals)*1.18))
        ax.set_title(title, loc="left", fontsize=8.3, fontweight="bold", color=COLORS["el3"])
    for ax in axes:
        ax.grid(axis="y", color="#e5e5e5", linewidth=.45);ax.set_axisbelow(True)
    fig.text(.5,.970,"xDSL authoring: same tested coverage, less time and lower total token use",
             fontsize=9.5, fontweight="bold", ha="center", va="center")
    fig.legend(handles=[Line2D([], [], color=COLORS[n], label=("C++ dialect scaffold (EL2)" if n=="el2" else "xDSL + authoring support (EL3)"), lw=2.5) for n in RUNS],
               loc="upper center", bbox_to_anchor=(.5,.937), frameon=False, ncol=2, fontsize=7.8)
    fig.text(.5,.846,"Both start with an empty target backend; same tests, references, model and 15-CPU budget.",
             ha="center", fontsize=6.7, color="#555555")
    extra = treatment["passing_cases"] - goal
    fresh_a = (baseline["tokens"]["tokens_fresh_input"] + baseline["tokens"]["tokens_output"]) / 1e6
    fresh_b = (treatment["tokens"]["tokens_fresh_input"] + treatment["tokens"]["tokens_output"]) / 1e6
    fig.text(.5,.154, f"All metrics at the same {goal}-test goal out of 149 (Spike / L2); xDSL passed {extra} extra tests.",
             ha="center", fontsize=7.1)
    fig.text(.5,.097,f"Total tokens include cache. Fresh input + output: {fresh_a:.2f}M (C++) vs {fresh_b:.2f}M (xDSL).",
             ha="center", fontsize=6.7, color="#555555")
    fig.text(.5,.040,"Compiler construction and Spike validation only. One author run per condition; recorded tokens are lower bounds.",
             ha="center", fontsize=6.3, color="#555555")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    from matplotlib.text import Text
    for artist in fig.findobj(match=Text):
        if not artist.get_visible() or not artist.get_text():
            continue
        bounds = artist.get_window_extent(renderer)
        if bounds.x0 < -1 or bounds.x1 > fig.bbox.width+1 or bounds.y0 < -1 or bounds.y1 > fig.bbox.height+1:
            raise ValueError(f"Figure text would be clipped: {artist.get_text()!r}")
    for ext in ("pdf", "svg", "png"):
        fig.savefig(out/f"tooling_convergence_el2_el3.{ext}", dpi=240, facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    with args.csv.open() as f:
        rows = list(csv.DictReader(f))
    result, selected = derive(rows)
    result["csv"] = str(args.csv.resolve())
    result["csv_sha256"] = hashlib.sha256(args.csv.read_bytes()).hexdigest()
    result["analysis_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.out.mkdir(parents=True, exist_ok=True)
    render(result, selected, args.out)
    (args.out/"tooling_convergence_el2_el3.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({"goal_cases": result["goal_cases"], "reduction_vs_el2": result["reduction_vs_el2"],
                      "uncached_input_plus_output_change":result["uncached_input_plus_output_change"]}, indent=2))


if __name__ == "__main__":
    main()
