"""Descriptive analysis/figures from all_results.csv, with no campaign access."""
from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

COLORS = {"el1": "#D55E00", "el2": "#E69F00", "el3": "#0072B2",
          "fresh10cca": "#009E73", "fresh10seed": "#CC79A7",
          "fresh10-tooling-infra": "#666666", "fresh10-tooling-scheduling": "#0072B2",
          "fresh10-tooling-no-reference-kernels": "#009E73"}
LABELS = {"el1": "EL1", "el2": "EL2", "el3": "EL3", "fresh10cca": "fresh10CCA", "fresh10seed": "fresh10seed",
          "fresh10-tooling-infra": "A: infrastructure", "fresh10-tooling-scheduling": "B: +scheduling",
          "fresh10-tooling-no-reference-kernels": "C: full, no references"}
HISTORICAL = ["el1", "el2", "el3", "fresh10cca", "fresh10seed"]
LIVE = ["fresh10-tooling-infra", "fresh10-tooling-scheduling", "fresh10-tooling-no-reference-kernels"]


def num(v):
    try:
        return float(v) if v not in (None, "") else None
    except (ValueError, TypeError):
        return None


def truth(v):
    return v is True or v == "True"


def time_of(r):
    # Historical per-row attribution can precede publication of the containing
    # grade. The result was observable only once both existed.
    values = [num(r.get(k)) for k in ("elapsed_s", "publication_elapsed_s")]
    return max((v for v in values if v is not None), default=None)


def observed_failure(r):
    if r.get("status") == "fail" or r.get("numeric_status") == "fail":
        return True
    if any(r.get(k) == "fail" for k in ("l0_status", "l1_status", "l2_status", "l3_status")):
        return True
    return r.get("barrier_status") == "fail" and r.get("failure_plane") not in ("oracle_unavailable", "unavailable")


def failure_group(r):
    plane, cat = (r.get("failure_plane") or "").lower(), (r.get("failure_category") or "").lower()
    if "numeric" in plane or "mismatch" in cat:
        return "Numerical"
    if "race" in plane or "ordering" in plane or "hazard" in cat:
        return "Ordering"
    if plane in ("oracle_unavailable", "unavailable"):
        return "Unavailable"
    if plane in ("spike", "gsim", "rtl") or "crash" in cat or "timeout" in cat:
        return "Execution/tool"
    return "Build/static"


def token_points(rows):
    t, total, fresh_output = [], [], []
    a = b = 0
    for r in sorted((r for r in rows if r["record_type"] == "provider_usage"), key=lambda r: time_of(r) or 0):
        if time_of(r) is None or time_of(r) < 0:
            continue
        a += num(r.get("tokens_total")) or 0
        b += (num(r.get("tokens_fresh_input")) or 0) + (num(r.get("tokens_output")) or 0)
        t.append(time_of(r)); total.append(a); fresh_output.append(b)
    return t, total, fresh_output


def coverage_points(rows, roster=None):
    selected = sorted((r for r in rows if r["record_type"] == "grade_case" and r["l2_status"] == "pass"),
                      key=lambda r: time_of(r) or 0)
    seen, points = set(), [(0, 0)]
    for r in selected:
        if time_of(r) is None or time_of(r) < 0 or (roster is not None and r["case_id"] not in roster):
            continue
        if r["case_id"] not in seen:
            seen.add(r["case_id"])
            points.append((time_of(r), len(seen)))
    return points


def distinct_attempts(rows, channel="selfcheck_case"):
    # Requests are observable author self-checks, distinct from background grades.
    # Keep the last visible state of each request/source/case, including failures.
    found = {}
    for r in sorted(rows, key=lambda r: time_of(r) or 0):
        if r["record_type"] != channel:
            continue
        key = r.get("attempt_id"), r.get("source_sha256"), r.get("case_id")
        found[key] = r
    return list(found.values())


def distribution(values):
    values = sorted(v for v in values if v is not None and v >= 0)
    def q(p):
        if not values:
            return None
        x = (len(values)-1)*p
        a, b = int(x), min(int(x)+1, len(values)-1)
        return values[a] + (values[b]-values[a])*(x-a)
    return dict(n=len(values), p50_s=q(.5), p95_s=q(.95), sum_s=sum(values) if values else None)


def repairs(attempts):
    active, closed, repeated = {}, [], 0
    for r in sorted(attempts, key=lambda r: time_of(r) or 0):
        case = r["case_id"]
        if observed_failure(r):
            signature = tuple(r.get(k) for k in ("failure_plane", "failure_category", "l2_status", "barrier_status"))
            a = active.setdefault(case, dict(start_s=time_of(r), signatures=set(), failures=0, last_source=None))
            repeated += signature in a["signatures"]
            a["signatures"].add(signature); a["failures"] += 1; a["last_source"] = r["source_sha256"]
        elif r.get("l2_status") == "pass" and case in active:
            a = active.pop(case)
            closed.append(dict(case_id=case, start_s=a["start_s"], end_s=time_of(r), elapsed_s=time_of(r)-a["start_s"],
                               failed_attempts=a["failures"], changed_source=a["last_source"] != r["source_sha256"]))
    return dict(closed_episodes=closed, open_episodes_censored=len(active), repeated_structured_failures=repeated,
                scope="Observed self-check failure to later L2 pass; source changes do not alone prove the root cause was fixed")


def analyze(rows, runs):
    by_run = defaultdict(list)
    for r in rows:
        by_run[r["experiment"]].append(r)
    ids = {name: {r["case_id"] for r in rr if r["record_type"] == "grade_case"} for name, rr in by_run.items()}
    background_roster = set.intersection(*(ids[n] for n in HISTORICAL if n in ids))
    summary = {"schema": "merlin.paper.descriptive_summary.v1", "background_common_case_ids": sorted(background_roster), "runs": {}}
    for meta in runs:
        name = meta["experiment"]
        if name not in HISTORICAL + LIVE:
            continue
        rr = by_run[name]
        four = [r for r in rr if time_of(r) is not None and 0 <= time_of(r) <= 14400]
        tokens = [r for r in four if r["record_type"] == "provider_usage"]
        case_rows = [r for r in four if r["record_type"] == "grade_case"]
        attempts = distinct_attempts(four)
        failures = [r for r in attempts if observed_failure(r)]
        passed = [r for r in attempts if r.get("l2_status") == "pass"]
        sums = {key: int(sum(num(r.get(key)) or 0 for r in tokens)) for key in
                ("tokens_total", "tokens_input", "tokens_fresh_input", "tokens_cache_read", "tokens_cache_write", "tokens_output", "tokens_reasoning")}
        l2 = {r["case_id"] for r in case_rows if r["l2_status"] == "pass"}
        grades = [r for r in four if r["record_type"] == "grade_snapshot"]
        latest = max(grades, key=lambda r: time_of(r), default={})
        retained = [r for r in rr if r["record_type"] == "source_snapshot" and num(r["scheduled_elapsed_s"]) == 14400]
        final_source = retained[0]["source_sha256"] if retained else None
        exact = [r for r in rr if r["record_type"] == "grade_case" and r["source_sha256"] == final_source] if final_source else []
        epass = {r["case_id"] for r in exact if r["l2_status"] == "pass"}
        efail = {r["case_id"] for r in exact if r["l2_status"] == "fail"}
        commands = [r for r in four if r["record_type"] == "author_tool" and r["tool_kind"] == "CommandExecution"]
        requests = [r for r in four if r["record_type"] == "request"]
        cov = coverage_points(four, background_roster)
        first = next((t for t, n in cov if n), None)
        target90 = next((t for t, n in cov if n >= .9 * len(background_roster)), None)
        snapshot_counts = {k: latest.get(k) for k in ("source_sha256", "n_cases", "n_l2_passed", "n_l3_passed", "n_failed", "n_pending", "n_refused", "coherent_snapshot")}
        summary["runs"][name] = dict(run_id=meta["run_id"], cohort=meta["cohort"], **sums,
            lead_tokens=int(sum(num(r["tokens_total"]) or 0 for r in tokens if r["agent_role"] == "lead")),
            helper_tokens=int(sum(num(r["tokens_total"]) or 0 for r in tokens if r["agent_role"] == "helper")),
            provider_responses=len(tokens), token_threads=len({r["thread_id"] for r in tokens}), tokens_are_lower_bounds=True,
            independent_author_runs=1, censored_before_four_hours=not meta.get("authoring_complete_at_cutoff", True),
            observed_window_s=min(14400, max((time_of(r) or 0 for r in rr), default=0)),
            background_cases_ever_l2_passed=len(l2), background_cases_observed=len({r["case_id"] for r in case_rows}),
            background_case_l2_pass_ids=sorted(l2), first_observed_background_l2_pass_s=first,
            first_90pct_ever_background_l2_s=target90,
            author_selfcheck_case_attempts=len(attempts), explicit_failed_selfcheck_case_attempts=len(failures),
            explicit_failed_selfcheck_by_plane=dict(Counter(failure_group(r) for r in failures)),
            l2_passed_selfcheck_case_attempts=len(passed),
            request_latency=distribution([num(r.get("latency_s")) for r in requests]),
            request_queue=distribution([num(r.get("queue_s")) for r in requests]),
            observed_execution_repair=repairs(attempts),
            latest_published_snapshot_before_four_hours=snapshot_counts,
            retained_four_hour_live_source_sha256=final_source,
            exact_four_hour_source_cases_observed_including_later_grades=len({r["case_id"] for r in exact}) if final_source else None,
            exact_four_hour_source_ever_l2_passes_including_later_grades=len(epass) if final_source else None,
            exact_four_hour_source_l2_failed_case_ids=sorted(efail) if final_source else None,
            command_attempts=len(commands) if meta.get("command_outcomes_available") else None,
            nonzero_command_exits=sum(num(r.get("exit_code")) not in (None, 0) for r in commands) if commands else None,
            note="Ever-observed coverage spans compiler revisions; exact-source observations can be partial and include later grading. No compiler final qualification implied.")
    summary["small_studies"] = {}
    for meta in runs:
        name = meta["experiment"]
        if not name.startswith("small-") or name.endswith("-oracle"):
            continue
        rr = by_run[name]
        tokens = [r for r in rr if r["record_type"] == "provider_usage" and time_of(r) is not None and 0 <= time_of(r) <= meta["author_budget_s"]]
        frozen = [r for r in rr if r["record_type"] == "small_case_result" and r["subject"].startswith("run-")]
        physical = [r for r in rr if r["record_type"] == "small_physical_cell" and r["subject"].startswith("run-")]
        baselines = {r["case_id"]: r for r in rr if r["record_type"] == "small_case_result" and r["subject"].startswith("baseline-")}
        matched = defaultdict(list)
        for r in frozen:
            b = baselines.get(r["case_id"])
            if not b or r["status"] != "pass" or b["status"] != "pass":
                continue
            keys = ("model_sha256", "comparison_domain_sha256", "scope_sha256", "capsule_sha256")
            if not all(r.get(k) and r[k] == b.get(k) for k in keys):
                continue
            bc, rc = num(b["estimated_cycles"]), num(r["estimated_cycles"])
            if bc and rc and truth(b["complete_event_accounting"]) and truth(r["complete_event_accounting"]):
                matched[r["split"]].append(bc/rc)
        outcomes = {}
        for split in ("train", "heldout"):
            a, p = [r for r in frozen if r["split"] == split], [r for r in physical if r["split"] == split]
            outcomes[split] = dict(model_cases=len(a), model_statuses=dict(Counter(r["status"] for r in a)),
                raw_physical_cells=len(p), raw_physical_statuses=dict(Counter(r["status"] for r in p)),
                modeled_speedup_vs_own_initial_geomean=math.exp(sum(math.log(x) for x in matched[split])/len(matched[split])) if matched[split] else None,
                modeled_speedup_matching_cases=len(matched[split]))
        summary["small_studies"][name] = dict(run_id=meta["run_id"], author_budget_s=meta["author_budget_s"], final_status=meta.get("final_status"),
            tokens_total=int(sum(num(r["tokens_total"]) or 0 for r in tokens)), provider_responses=len(tokens),
            outcome_by_split=outcomes, independent_author_runs=1, timing_qualified=False,
            note="30min cohort; raw FireSim slots lack consumed ELF/domain proof. Retain failed arm and baseline-missing model cells.")
    return summary, by_run


def plot(rows, runs, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.3, "axes.labelsize": 7.3,
                         "axes.titlesize": 8, "legend.fontsize": 6.8, "xtick.labelsize": 6.7,
                         "ytick.labelsize": 6.7, "pdf.fonttype": 42, "svg.fonttype": "none",
                         "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.35})
    summary, by_run = analyze(rows, runs)
    common = set(summary["background_common_case_ids"])
    out.mkdir(parents=True, exist_ok=True)

    def save(fig, name):
        for ext in ("pdf", "svg", "png"):
            fig.savefig(out / f"{name}.{ext}", dpi=220, facecolor="white")
        plt.close(fig)

    def clean(ax, xlabel=True):
        ax.grid(axis="y", color="#e5e5e5", linewidth=.45)
        ax.set_axisbelow(True)
        if xlabel:
            ax.set_xlabel("Authoring time (min)")
        ax.set_xlim(0, 240)
        ax.set_xticks([0, 60, 120, 180, 240])

    def curves(ax, names, kind):
        for name in names:
            if name not in by_run:
                continue
            rr = [r for r in by_run[name] if time_of(r) is not None and 0 <= time_of(r) <= 14400]
            style = "--" if name.startswith("fresh10") and name in HISTORICAL else "-"
            color = COLORS[name]
            if kind == "tokens":
                t, y, _ = token_points(rr)
                if t:
                    ax.step([0] + [x/60 for x in t] + [240], [0] + [v/1e6 for v in y] + [y[-1]/1e6], where="post", color=color, ls=style, label=LABELS[name])
            elif kind == "coverage":
                p = coverage_points(rr, common)
                ax.step([x/60 for x, _ in p] + [240], [y for _, y in p] + [p[-1][1]], where="post", color=color, ls=style, label=LABELS[name])
            else:
                fails = sorted((r for r in distinct_attempts(rr) if observed_failure(r)), key=time_of)
                t = [0] + [time_of(r)/60 for r in fails] + [240]
                y = [0] + list(range(1, len(fails)+1)) + [len(fails)]
                ax.step(t, y, where="post", color=color, ls=style, label=LABELS[name])

    # Controlled level cohort, single column. This is observed exploration, not
    # evidence that the final source passes the union of all earlier cases.
    fig, axes = plt.subplots(3, 1, figsize=(3.4, 5.15), constrained_layout=True)
    for ax, kind, title, ylabel in zip(axes, ["tokens", "coverage", "failures"],
        ["(a) Recorded provider usage", "(b) Ever-observed L2 passes", "(c) Failed author self-checks"],
        ["Total tokens (M)", "Distinct cases with L2 pass", "Failed case attempts"]):
        curves(ax, ["el1", "el2", "el3"], kind)
        clean(ax)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_ylabel(ylabel)
    axes[0].legend(loc="upper left", ncol=3, frameon=False)
    axes[1].set_ylim(0, len(common)+3)
    axes[1].axhline(len(common), color="#999999", lw=.6, ls=":")
    save(fig, "el123_progress_single_column")

    fig, axes = plt.subplots(2, 3, figsize=(7.05, 4.2), constrained_layout=True)
    for ax, kind, title, ylabel in zip(axes[0], ["tokens", "coverage", "failures"],
        ["(a) Total provider usage", "(b) Ever-observed L2 passes", "(c) Failed author self-checks"],
        ["Total tokens (M)", "Distinct public cases", "Explicit failures"]):
        curves(ax, HISTORICAL, kind); clean(ax); ax.set_title(title, loc="left"); ax.set_ylabel(ylabel)
    axes[0,0].legend(frameon=False, ncol=2)
    axes[0,1].set_ylim(0, len(common)+3)
    ax = axes[1,0]
    for name in HISTORICAL:
        rr = [r for r in by_run.get(name, []) if time_of(r) is not None and 0 <= time_of(r) <= 14400]
        t, tok, _ = token_points(rr)
        cov = coverage_points(rr, common)
        if t:
            xs = [tok[bisect.bisect_right(t, tm)-1]/1e6 if bisect.bisect_right(t, tm) else 0 for tm, _ in cov]
            ax.step(xs+[tok[-1]/1e6], [n for _, n in cov]+[cov[-1][1]], where="post", color=COLORS[name], ls="--" if name.startswith("fresh10") else "-", label=LABELS[name])
    ax.set(xlabel="Recorded total tokens (M)", ylabel="Distinct cases with L2 pass", title="(d) Progress versus usage", ylim=(0, len(common)+3))
    ax.grid(axis="y", color="#e5e5e5", linewidth=.45)
    ax = axes[1,1]
    positions = np.arange(len(HISTORICAL))
    fresh = [(summary["runs"][n]["tokens_fresh_input"] + summary["runs"][n]["tokens_output"])/1e6 for n in HISTORICAL]
    cached = [summary["runs"][n]["tokens_cache_read"]/1e6 for n in HISTORICAL]
    ax.barh(positions, fresh, color="#3F6C8A", label="Fresh input + output")
    ax.barh(positions, cached, left=fresh, color="#CBD6DD", label="Cached input")
    ax.set_yticks(positions, [LABELS[n] for n in HISTORICAL]); ax.invert_yaxis()
    ax.set(xlabel="Recorded tokens (M)", title="(e) Four-hour token breakdown")
    ax.legend(loc="lower right", frameon=False, fontsize=6)
    ax = axes[1,2]
    for i, name in enumerate(HISTORICAL):
        s = summary["runs"][name]
        ax.scatter(s["tokens_fresh_input"] / 1e6 + s["tokens_output"] / 1e6,
                   s["background_cases_ever_l2_passed"], color=COLORS[name], s=22)
        ax.annotate(LABELS[name], (s["tokens_fresh_input"]/1e6+s["tokens_output"]/1e6, s["background_cases_ever_l2_passed"]),
                    xytext=(3, (-1 if i%2 else 1)*7), textcoords="offset points", fontsize=6.2)
    ax.set(xlabel="Fresh input + output tokens (M)", ylabel="Distinct cases with L2 pass", title="(f) Four-hour observed endpoints")
    ax.margins(x=.24, y=.25); ax.grid(axis="y", color="#e5e5e5", linewidth=.45)
    save(fig, "campaign_comparison_double_column")

    # Censored current wave: stop each line at its observation cutoff, never
    # carry a 25-minute result horizontally through the remaining four hours.
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.25), constrained_layout=True)
    for name in LIVE:
        rr = [r for r in by_run.get(name, []) if time_of(r) is not None and 0 <= time_of(r) <= 14400]
        end = max((time_of(r) for r in rr), default=0)
        t, y, _ = token_points(rr)
        if t:
            axes[0].step([x/60 for x in t]+[end/60], [v/1e6 for v in y]+[y[-1]/1e6], where="post", color=COLORS[name], label=LABELS[name])
            axes[0].plot(end/60, y[-1]/1e6, marker=">", ms=4, color=COLORS[name])
        p = coverage_points(rr, common)
        axes[1].step([x/60 for x,_ in p]+[end/60], [v for _,v in p]+[p[-1][1]], where="post", color=COLORS[name], label=LABELS[name])
        axes[1].plot(end/60, p[-1][1], marker=">", ms=4, color=COLORS[name])
    for ax, title, ylabel in zip(axes, ["(a) Current tooling wave: recorded usage", "(b) Current tooling wave: observed L2 passes"], ["Total tokens (M)", "Distinct cases with L2 pass"]):
        ax.set(title=title, xlabel="Authoring time (min)", ylabel=ylabel)
        ax.grid(axis="y", color="#e5e5e5", linewidth=.45)
    axes[0].legend(frameon=False, loc="upper left")
    save(fig, "tooling_wave_censored")

    # Focused studies: workload families and 30-minute budgets are kept separate
    # from the joint compiler campaigns. Display failure retention and effort.
    fig, axes = plt.subplots(2, 3, figsize=(7.05, 4.0), constrained_layout=True)
    for col, study in enumerate(("cca", "library", "scheduling")):
        for arm, color in (("control", "#777777"), ("treatment", "#0072B2")):
            name = f"small-{study}-{arm}"
            rr = [r for r in by_run.get(name, []) if time_of(r) is not None and 0 <= time_of(r) <= 1800]
            t, y, _ = token_points(rr)
            if t:
                axes[0,col].step([0]+[v/60 for v in t], [0]+[v/1e6 for v in y], where="post", color=color, label=arm.title())
            s = summary["small_studies"].get(name, {})
            held = s.get("outcome_by_split", {}).get("heldout", {})
            for j, source in enumerate(("model", "raw_physical")):
                counts = held.get(source+"_statuses", {})
                total = sum(counts.values())
                success = counts.get("pass", 0)
                x = j + (-.18 if arm == "control" else .18)
                axes[1,col].bar(x, success, width=.33, color=color)
                axes[1,col].bar(x, total-success, bottom=success, width=.33, color="white", edgecolor=color, hatch="///", linewidth=.6)
                axes[1,col].text(x, total+.35, f"{success}/{total}", ha="center", fontsize=6.2)
        axes[0,col].set(title=study.upper() if study == "cca" else study.title(), xlabel="Authoring time (min)", ylabel="Recorded tokens (M)", xlim=(0,30))
        axes[0,col].grid(axis="y", color="#e5e5e5", linewidth=.45)
        axes[1,col].set(xticks=[0,1], xticklabels=["Model", "Raw FireSim\n(unqualified)"], ylabel="Held-out evaluator cells", ylim=(0,17.5), yticks=[0,4,8,12,16])
        axes[1,col].grid(axis="y", color="#e5e5e5", linewidth=.45)
        axes[1,col].set_axisbelow(True)
    axes[0,0].legend(frameon=False)
    save(fig, "small_studies_effort_and_transfer")

    # Exact 4h EL source snapshots; only complete estimates with matching
    # capsule/model/domain/scope are compared. These are not physical cycles.
    perf = {}
    for name in ("el1", "el2", "el3"):
        sha = summary["runs"][name]["retained_four_hour_live_source_sha256"]
        cases = {}
        for r in by_run[name]:
            if (r["record_type"] == "grade_case" and r["source_sha256"] == sha and r["l2_status"] == "pass"
                    and truth(r["complete_event_accounting"]) and num(r["estimated_cycles"])):
                key = tuple(r.get(k) for k in ("case_id", "capsule_sha256", "model_sha256", "comparison_domain_sha256", "scope_sha256"))
                if all(key):
                    cases[key] = r
        perf[name] = cases
    matched = sorted(set.intersection(*(set(v) for v in perf.values())))
    summary["exact_el_source_matched_modeled_cases"] = [dict(case_id=k[0], bindings=dict(zip(("capsule_sha256", "model_sha256", "comparison_domain_sha256", "scope_sha256"), k[1:])),
        cycles={n: num(perf[n][k]["estimated_cycles"]) for n in perf},
        source={n: perf[n][k]["source_sha256"] for n in perf}) for k in matched]
    if matched:
        fig, ax = plt.subplots(figsize=(7.05, 2.6), constrained_layout=True)
        x = np.arange(len(matched))
        for i, name in enumerate(("el1", "el2", "el3")):
            ax.scatter(x+(i-1)*.16, [num(perf[name][k]["estimated_cycles"])/num(perf["el1"][k]["estimated_cycles"]) for k in matched], s=22, color=COLORS[name], label=LABELS[name], marker=["o", "s", "^"][i], zorder=3)
        short = {"ZA_conv_3x3x28x28x128x128x4_native_i8": "Conv b4\n28², c128",
                 "CH_kv_heads_decoder_scale_model_q16kv4s96l96d52": "QK heads\nq16, kv4, s96, d52",
                 "JM_tiny_llama_decode_m1k2048n2048_raw": "Decode GEMM\n1×2048×2048",
                 "TL_mm_down_m16k5632n2048_raw": "Down GEMM\n16×5632×2048",
                 "TL_mm_gu_fused_m16k2048n11264_i8": "Gate/up GEMM\n16×2048×11264",
                 "TL_mm_qkv_fused_m16k2048n2560_i8": "QKV GEMM\n16×2048×2560"}
        ax.set_xticks(x, [short.get(k[0], k[0]) for k in matched], fontsize=6.5)
        ax.set(yscale="log", ylabel="Modeled cycles / EL1", title="Exact four-hour sources: six matched timing-model estimates")
        ax.axhline(1, color="#777777", lw=.6); ax.legend(frameon=False, ncol=3)
        ax.grid(axis="y", color="#e5e5e5", linewidth=.45); ax.set_axisbelow(True)
        save(fig, "el123_frozen_source_modeled_performance")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    with (args.data / "all_results.csv").open() as f:
        rows = list(csv.DictReader(f))
    runs = json.loads((args.data / "runs.json").read_text())
    summary = plot(rows, runs, args.out)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({n: {k: v for k,v in r.items() if k in ("tokens_total", "background_cases_ever_l2_passed", "explicit_failed_selfcheck_case_attempts", "exact_four_hour_source_ever_l2_passes_including_later_grades")}
                      for n,r in summary["runs"].items()}, indent=2))


if __name__ == "__main__":
    main()
