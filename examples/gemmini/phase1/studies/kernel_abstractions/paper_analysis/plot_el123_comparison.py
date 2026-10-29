"""Fixed-budget EL1/EL2/EL3 comparison, preserving the faster EL1 baseline.

Coverage is the best observed single-source L2 coverage at each publication
time. It may combine grading batches for one exact source, never revisions.
The end-of-window best candidate is not necessarily the final edited source.
The default renderer discloses the subsequently discovered ABI-dependent
comparator confound; historical native verdicts are not relabeled as passes.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from analyze_results import distinct_attempts, observed_failure, time_of
from plot_focused_convergence import first_source_reaching, sources_and_curve, usage_at

RUNS = ("el1", "el2", "el3")
COLORS = {"el1": "#444444", "el2": "#999999", "el3": "#0072B2"}
LABELS = {"el1": "EL1: direct C++", "el2": "EL2: C++ dialect scaffold",
          "el3": "EL3: xDSL + authoring support"}
STYLES = {"el1": "--", "el2": ":", "el3": "-"}
WINDOW_S = 14400


def derive(rows):
    selected = {name: [r for r in rows if r["experiment"] == name
                      and time_of(r) is not None and 0 <= time_of(r) <= WINDOW_S]
                for name in RUNS}
    roster_sets = [{r["case_id"] for r in selected[n] if r["record_type"] == "grade_case"}
                   for n in RUNS]
    if not all(s == roster_sets[0] for s in roster_sets):
        raise ValueError("Background case rosters differ")
    result = dict(schema="merlin.el123_fixed_budget_figure.v1", window_s=WINDOW_S,
                  roster_case_ids=sorted(roster_sets[0]), runs={})
    passed_sets = {}
    for name in RUNS:
        rr = selected[name]
        states, curve, evidence = sources_and_curve(rr)
        source = max(states, key=lambda s: sum(v == "pass" for v in states[s].values()))
        passing = {c for c, status in states[source].items() if status == "pass"}
        passed_sets[name] = passing
        attempts = distinct_attempts(rr)
        tokens = usage_at(rr, WINDOW_S)
        result["runs"][name] = dict(
            best_source_sha256=source, best_source_pass_cases=len(passing),
            passing_case_ids=sorted(passing), curve=[list(p) for p in curve],
            tokens=tokens,
            fresh_input_plus_output=tokens["tokens_fresh_input"]+tokens["tokens_output"],
            failed_selfcheck_case_outcomes=sum(observed_failure(r) for r in attempts),
            selfcheck_case_attempts=len(attempts),
            evidence=[{k: evidence[source][case].get(k) for k in
                ("record_id", "case_id", "source_sha256", "l2_status", "numeric_status",
                 "observed_at", "publication_at", "evidence_path", "evidence_sha256",
                 "evidence_locator", "elf_sha256")} for case in sorted(passing)])
    common = set.intersection(*passed_sets.values())
    result["shared_case_ids"] = sorted(common)
    result["extra_el3_case_ids"] = sorted(passed_sets["el3"]-passed_sets["el1"])
    for name in RUNS:
        milestone = first_source_reaching(selected[name], common)
        milestone["tokens"] = usage_at(selected[name], milestone["at_s"])
        result["runs"][name]["shared_goal"] = milestone
    result["limits"] = [
        "One independent author per condition; descriptive, not a population effect",
        "Best observed compiler candidate, not fully graded final edited source",
        "Explicit L2 execution evidence, not complete L3 or hardware performance",
        "Same-source grading batches may accumulate; different sources are never pooled",
        "Whole four-hour token/check totals do not represent equal achieved coverage",
        "Total tokens include cached context; incomplete responses may be absent",
        "Failure and attempt counts have different case mixes, not matched error probabilities",
        "EL1 reaches the shared128-case goal sooner and with fewer tokens than EL3",
        "EL3 is an authoring bundle; the result does not isolate xDSL alone",
        "Six chain verdicts depend on ABI-selected output comparison; seventh baseline case was admission-deferred",
        "The seven-case native verdict gap does not establish broader compiler capability; common requalification is required",
    ]
    return result


def render_historical_three_panel(result, out):
    """Historical layout only; its capability interpretation was withdrawn."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.text import Text

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.2,
        "axes.labelsize": 7.2, "xtick.labelsize": 6.8, "ytick.labelsize": 6.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 3, figsize=(7.05, 3.7),
                             gridspec_kw={"width_ratios": [1.45, 1, 1]})
    fig.subplots_adjust(left=.075, right=.985, bottom=.305, top=.72, wspace=.48)
    for name in RUNS:
        r = result["runs"][name]
        curve = r["curve"] + [[WINDOW_S, r["curve"][-1][1]]]
        axes[0].step([p[0]/60 for p in curve], [p[1] for p in curve], where="post",
                     color=COLORS[name], ls=STYLES[name], lw=1.8 if name=="el3" else 1.5)
    axes[0].set(xlim=(0,250), ylim=(0,149), xticks=[0,60,120,180,240],
                yticks=[0,32,64,96,128], xlabel="Time since author start (min)",
                ylabel="Best compiler coverage (Spike / L2)")
    axes[0].axhline(128, color="#cccccc", lw=.6, ls=":", zorder=0)
    axes[0].annotate("135", (240,135), xytext=(-2,8), textcoords="offset points",
                     color=COLORS["el3"], fontsize=8.5, fontweight="bold", ha="right")
    axes[0].annotate("128 (EL1, EL2)", (240,128), xytext=(-2,-13), textcoords="offset points",
                     color=COLORS["el1"], fontsize=7.1, ha="right")
    axes[0].text(137,35,"Additional coverage:\n6 composed chains\n1 scale/readout case",
                 color=COLORS["el3"],fontsize=6.5,linespacing=1.35)
    axes[0].set_title("(a) Coverage over time\nEL3 passes 7 extra cases", loc="left",
                       fontsize=8.1, color=COLORS["el3"], fontweight="bold")
    for ax, field, title, ylabel in (
        (axes[1], "tokens", "(b) Token use\nFull four-hour window", "Total tokens including cache (M)"),
        (axes[2], "failures", "(c) Failed self-checks\nFull four-hour window", "Failed case outcomes")):
        vals = [result["runs"][n]["tokens"]["tokens_total"]/1e6 if field=="tokens"
                else result["runs"][n]["failed_selfcheck_case_outcomes"] for n in RUNS]
        for i, (name,value) in enumerate(zip(RUNS,vals)):
            ax.bar(i,value,width=.64,color=COLORS[name])
            ax.text(i,value+max(vals)*.025,f"{value:.1f}" if field=="tokens" else str(value),
                    ha="center", va="bottom", color=COLORS[name], fontsize=7.8, fontweight="bold")
        ax.set(xticks=[0,1,2], xticklabels=["EL1","EL2","EL3"], ylabel=ylabel,
                ylim=(0,max(vals)*1.2))
        ax.set_title(title, loc="left", color=COLORS["el3"], fontsize=8.1, fontweight="bold")
    axes[2].set_xticklabels([f"{n.upper()}\nn={result['runs'][n]['selfcheck_case_attempts']}" for n in RUNS])
    axes[2].set_xlabel("n = case checks attempted",fontsize=6.5,labelpad=2)
    for ax in axes:
        ax.grid(axis="y",color="#e5e5e5",lw=.45)
        ax.set_axisbelow(True)
    fig.text(.5,.963,"Broader compiler coverage with fewer total tokens after four hours",
             ha="center",va="center",fontsize=10,fontweight="bold")
    fig.legend(handles=[Line2D([],[],color=COLORS[n],ls=STYLES[n],lw=2,label=LABELS[n]) for n in RUNS],
               loc="upper center",bbox_to_anchor=(.5,.929),frameon=False,ncol=3,fontsize=7)
    fig.text(.5,.833,"Empty target backends; same 149 background tests, references, model and 15-CPU allowance.",
             ha="center",fontsize=6.8,color="#555555")
    fig.text(.5,.183,"EL1 reaches the shared 128 cases first: 91 min, versus 121 (EL3) and 148 (EL2).",
             ha="center",fontsize=7.1)
    fig.text(.5,.121,"Fresh input + output over four hours: 3.01M (EL1), 2.44M (EL2), 2.85M (EL3).",
             ha="center",fontsize=6.8,color="#555555")
    fig.text(.5,.059,"Best source-specific published evidence; one author per condition. Spike validation, not hardware performance.",
             ha="center",fontsize=6.3,color="#555555")
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    for artist in fig.findobj(match=Text):
        if artist.get_visible() and artist.get_text():
            b=artist.get_window_extent(renderer)
            if b.x0 < -1 or b.x1 > fig.bbox.width+1 or b.y0 < -1 or b.y1 > fig.bbox.height+1:
                raise ValueError(f"Figure text would be clipped: {artist.get_text()!r}")
    for ext in ("pdf","svg","png"):
        fig.savefig(out/f"el123_fixed_budget.{ext}",dpi=240,facecolor="white")
    plt.close(fig)


def render(result, out):
    from plot_el123_summary import render as render_with_comparator_notice
    render_with_comparator_notice(result, out)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()
    with args.csv.open() as f:
        rows=list(csv.DictReader(f))
    result=derive(rows)
    result["csv"] = str(args.csv.resolve())
    result["csv_sha256"] = hashlib.sha256(args.csv.read_bytes()).hexdigest()
    result["analysis_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.out.mkdir(parents=True,exist_ok=True)
    render(result,args.out)
    (args.out/"el123_fixed_budget.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({name:{k:v for k,v in result["runs"][name].items() if k in
        ("best_source_pass_cases","fresh_input_plus_output","failed_selfcheck_case_outcomes","selfcheck_case_attempts")}
        for name in RUNS},indent=2))


if __name__ == "__main__":
    main()
