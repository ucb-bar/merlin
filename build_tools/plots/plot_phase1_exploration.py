#!/usr/bin/env python3
"""Evidence-backed Phase 1 paper prototypes; extract and render in separate envs.

Extract with the experiment environment, render with matplotlib installed::

  python build_tools/plots/plot_phase1_exploration.py --run RUN \
    --conclusion CAMPAIGN_CONCLUSION_JSON --out OUT --extract-only
  python build_tools/plots/plot_phase1_exploration.py --data OUT/data.json --out OUT

Only public capsules are exported. Historical records are never modified. Process
wall intervals are not CPU residency; response usage is not provider decode rate.
Rendering needs matplotlib, numpy and Pillow >= 10.1; extraction additionally
needs the installed experiment package. Render receipts record library versions.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import heapq
import html
import importlib.util
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def stamp(value):
    if value is None:
        return None
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def extract(run, conclusion, out):
    sys.path.insert(0, str(REPO / "packages/merlin-experiments/src"))
    from merlin_experiments.phase1.telemetry.author_agents import _thread_metrics, read_rollout
    from phase1_abstraction_trace import abstraction_markers

    sources = {}

    def read(path):
        raw = path.read_bytes()
        try:
            name = str(path.relative_to(run))
        except ValueError:
            name = str(path)
        sources[name] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    campaign = read(conclusion)
    start = stamp(campaign["campaign_started_at"])
    end = stamp(campaign["deadline"])

    def minute(value):
        return (stamp(value) - start) / 60 if value else None

    timing = read(run / "timing_detailed.json")
    grades = []
    excluded_grades = []
    for path in sorted((run / "qa_history").glob("verdict*.json")):
        row = read(path)
        if row.get("n_capsules") != campaign["latest_capsule_grade"]["n_capsules"]:
            excluded_grades.append(
                {
                    "source": path.name,
                    "n_capsules": row.get("n_capsules"),
                    "reason": "different cohort; not a comparable progress point",
                }
            )
            continue
        grades.append(
            {
                "minute": minute(row["graded_at"]),
                "source": path.name,
                "passed": row["n_passed"],
                "summary": row.get("qualification_summary", {}),
                "capsules": [
                    {
                        k: c.get(k)
                        for k in (
                            "capsule",
                            "status",
                            "numeric_status",
                            "trace_status",
                            "trace_requirement_status",
                            "tiers",
                            "failure_plane",
                        )
                    }
                    for c in row["per_capsule"]
                    if c.get("label") == "public"
                ],
            }
        )
    grades.sort(key=lambda g: g["minute"])
    public = {c["capsule"] for c in grades[-1]["capsules"]}
    assert len(public) == campaign["latest_capsule_grade"]["n_capsules"]
    for grade in grades:
        certified = sum(
            c["status"] == "pass"
            and c["numeric_status"] == "pass"
            and c["tiers"].get("L3") == "pass"
            and c["trace_requirement_status"] == "pass"
            for c in grade["capsules"]
        )
        assert certified == grade["summary"]["execution_passes_with_trace_evidence"]

    agents, tokens, abstractions = [], [], []
    seen_responses = set()
    snapshots = list((run / "rounds").glob("*.codex_rollout_snapshot/**/*.jsonl"))
    for summary in timing["author_agents"]["agents"]:
        paths = [p for p in snapshots if summary["thread_id"] in p.name]
        assert paths, f"No retained rollout for {summary['agent_path']}"
        rollouts = [(p, read_rollout(p)) for p in paths]
        path, rollout = max(rollouts, key=lambda pr: len(pr[1]["rows"]))
        sources[str(path.relative_to(run))] = hashlib.sha256(path.read_bytes()).hexdigest()
        metrics = _thread_metrics(rollout, None)
        for marker in abstraction_markers(metrics, summary["agent_path"]):
            abstractions.append({**marker, "a": (marker["start"] - start) / 60, "b": (marker["end"] - start) / 60})
        commands = []
        for command in metrics["commands"]:
            if command["start"] is not None and command["end"] is not None:
                commands.append(
                    {
                        "a": (command["start"] - start) / 60,
                        "b": (command["end"] - start) / 60,
                        "failed": bool(command["item"].get("exit_code")),
                    }
                )
        agents.append(
            {
                "name": summary["agent_path"],
                "role": summary["role"],
                "turns": [[(a - start) / 60, (b - start) / 60] for a, b in metrics["intervals"]],
                "commands": commands,
                "tokens": summary["tokens"],
                "observed_start": minute(summary["started_at"]),
                "observed_end": minute(summary["last_event_at"]),
            }
        )
        for row in rollout["rows"]:
            if row.get("type") != "token_usage_record":
                continue
            payload = row["payload"]
            key = (rollout["thread_id"], payload.get("response_id", row.get("ordinal")))
            if key in seen_responses:
                continue
            seen_responses.add(key)
            usage = payload["usage"]
            total_in = usage.get("input_tokens", 0)
            cached = usage.get("cached_input_tokens", 0)
            write = usage.get("cache_write_input_tokens", 0)
            tokens.append(
                {
                    "minute": minute(row["timestamp"]),
                    "agent": summary["agent_path"],
                    "fresh": max(total_in - cached - write, 0),
                    "cached": cached,
                    "cache_write": write,
                    "output": usage.get("output_tokens", 0),
                    "reasoning": usage.get("reasoning_output_tokens", 0),
                }
            )
    totals = timing["author_agents"]["totals"]["tokens_all"]
    assert sum(t["output"] for t in tokens) == totals["output"]
    assert sum(t["fresh"] + t["cached"] + t["cache_write"] for t in tokens) == totals["input_total"]

    tools = [
        {
            "a": minute(c["started_at"]),
            "b": minute(c.get("ended_at")),
            "name": c["name"],
            "completed": c["completed"],
            "error": c.get("error", False),
        }
        for c in timing["tools"]["calls"]
    ]
    invocations, seen = [], set()
    exclusions = collections.Counter()
    for path in sorted((run / "_qa_work").rglob("invocation.json")):
        record = read(path)
        if record.get("kind") != "subprocess":
            exclusions["non_subprocess"] += 1
            continue
        a, b = record.get("started_ns"), record.get("finished_ns")
        if not isinstance(a, int) or not isinstance(b, int) or b < a:
            exclusions["missing_or_invalid_interval"] += 1
            continue
        key = (a, b, tuple(record.get("argv", [])))
        if key in seen:
            exclusions["duplicate_execution_identity"] += 1
            continue
        seen.add(key)
        if a / 1e9 < start or b / 1e9 > end:
            exclusions["outside_campaign"] += 1
            continue
        parts = path.parts
        capsule = parts[parts.index("gemmini-capsule-bench") + 1] if "gemmini-capsule-bench" in parts else None
        if capsule not in public:
            exclusions["outside_public_cohort"] += 1
            continue
        exe = Path(record.get("executable", {}).get("path", "unknown")).name
        stage = record.get("stage")
        if stage == "selected_rocc_process":
            category = "L2 Spike" if exe == "spike" else "L3 gSIM" if "emulator" in exe else "Other"
        elif stage == "original_profile_package":
            category = "Compiler package"
        elif stage in ("harness_object", "object", "elf"):
            category = "Build / link"
        else:
            category = "Other"
        invocations.append(
            {
                "a": (a / 1e9 - start) / 60,
                "b": (b / 1e9 - start) / 60,
                "capsule": capsule,
                "category": category,
                "stage": stage,
                "tool": exe,
                "failed": record.get("returncode") != 0,
                "source": str(path.relative_to(run)),
            }
        )

    requests = []
    for agent in timing["author_agents"]["agents"]:
        for r in agent["selfcheck_requests"]:
            requests.append(
                {
                    "id": r["request_id"],
                    "agent": agent["agent_path"],
                    "a": minute(r["requested_at"]),
                    "start": minute(r.get("grade_started_at")),
                    "b": minute(r.get("graded_at")),
                    "count": r.get("n_capsules"),
                    "passed": r.get("n_passed"),
                    "rows": r.get("n_result_rows"),
                    "mode": r.get("mode"),
                }
            )
    requests.sort(key=lambda r: r["a"])
    data = {
        "schema": "merlin.phase1.paper_exploration.v1",
        "run": run.name,
        "start": campaign["campaign_started_at"],
        "end": campaign["deadline"],
        "duration": (end - start) / 60,
        "grades": grades,
        "agents": agents,
        "tokens": tokens,
        "token_totals": totals,
        "tools": tools,
        "abstractions": abstractions,
        "invocations": invocations,
        "requests": requests,
        "checks": campaign["native_compiler_workflow_checks"],
        "formal_complete": campaign["native_formal_complete"],
        "lead_span": [
            minute(timing["session_details"][0]["started_at"]),
            minute(timing["session_details"][0]["ended_at"]),
        ],
        "exclusions": dict(exclusions),
        "excluded_grades": excluded_grades,
        "telemetry_complete": campaign["telemetry"]["complete"],
        "limits": [
            "Observed response usage reconstructed across seven actor rollouts; final usage incomplete; not billing.",
            "Input/output rates are binned client-observed response traffic; no provider TTFT/decode measurement.",
            "No physical CPU residency data; overlapping process wall intervals are not CPU seconds.",
            "Lead activity uses paired tool intervals; client gaps include unobserved work and network delay.",
            "Global pass progress exists only at comparable whole-cohort verdict timestamps.",
            "Final capsule results do not imply complete full-model qualification or formal Phase 1 completion.",
            "Single-run abstraction usage is observational; causal effects require matched ablations.",
            (
                "Structural API markers are request witnesses, not dynamic call counts; shell "
                "success does not establish each nested subcommand result."
            ),
            "Invocation coverage is limited to retained public capsule subprocess records; not all campaign work.",
        ],
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "data.json").write_text(json.dumps(data, separators=(",", ":")) + "\n")
    manifest = {
        "schema": "merlin.paper_source_manifest.v1",
        "run": str(run),
        "sources": sources,
        "normalized_data_sha256": hashlib.sha256((out / "data.json").read_bytes()).hexdigest(),
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "abstraction_extractor_sha256": hashlib.sha256(
            Path(__file__).with_name("phase1_abstraction_trace.py").read_bytes()
        ).hexdigest(),
        "identities": {
            "public_capsules": len(public),
            "agents": len(agents),
            "usage_records": len(tokens),
            "unique_process_intervals": len(invocations),
            "requests": len(requests),
        },
        "limits": data["limits"],
    }
    (out / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["identities"]))
    return data


def render(data, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.collections import LineCollection
    from matplotlib.patches import Patch, PathPatch
    from matplotlib.path import Path as MPath
    from matplotlib.ticker import FuncFormatter

    style_path = REPO / "packages/merlin-analysis/src/merlin/plotting/merlin_plotstyle.py"
    spec = importlib.util.spec_from_file_location("merlin_plotstyle", style_path)
    style = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(style)
    style.use_merlin_style()
    from matplotlib import font_manager

    for family, fallback in ((style.SANS, "DejaVu Sans"), (style.SERIF, "DejaVu Serif")):
        try:
            font_manager.findfont(family, fallback_to_default=False)
        except ValueError:
            if family == style.SANS:
                plt.rcParams["font.family"] = fallback
            else:
                style.SERIF = fallback
    plt.rcParams.update(
        {"font.size": 10, "pdf.fonttype": 42, "axes.titlesize": 13, "legend.fontsize": 8, "axes.labelsize": 10}
    )
    C = {
        "Compiler package": style.GOLD,
        "Build / link": style.SLATE,
        "L2 Spike": style.NAVY,
        "L3 gSIM": style.MAUVE,
        "Other": style.SAGE,
    }
    states = ["Execution + trace", "Numeric only", "Refusal met", "Failed / incomplete"]
    SC = dict(zip(states, [style.NAVY, style.GOLD, style.SAGE, style.MAUVE]))
    T = data["duration"]
    lead_a, lead_b = data["lead_span"]
    events = data["invocations"]
    figures = []

    def base(ax, grid="y", clock=True):
        style.style_ax(ax, grid=grid)
        if clock:
            ax.set_xlim(0, T)
            ax.set_xticks(np.arange(0, T + 1, 30))
        ax.tick_params(labelsize=9)

    def section(ax, label):
        ax.set_title(label, loc="left", fontfamily=style.SERIF, fontsize=14, pad=10)

    def save(fig, name, label, description):
        for ext in ("svg", "pdf", "png"):
            fig.savefig(
                out / f"{name}.{ext}",
                dpi=160 if ext == "png" else 300,
                bbox_inches="tight",
                metadata={"Creator": "Merlin Phase 1 evidence visualization"},
            )
        figures.append({"name": name, "label": label, "description": description})
        plt.close(fig)

    def occupancy(intervals, edges, union=False):
        """Mean in-flight intervals; union=True measures occupied wall fraction."""
        merged = []
        for a, b in sorted(intervals):
            if b <= a:
                continue
            if union and merged and a <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        ans = np.zeros(len(edges) - 1)
        for a, b in merged:
            lo, hi = max(0, int(np.searchsorted(edges, a) - 1)), min(len(ans), int(np.searchsorted(edges, b) + 1))
            ans[lo:hi] += np.maximum(0, np.minimum(edges[lo + 1 : hi + 1], b) - np.maximum(edges[lo:hi], a))
        return ans / np.diff(edges)

    def state(row):
        if row["status"] == "pass" and row["numeric_status"] == "pass" and row["tiers"].get("L3") == "pass":
            return states[0] if row["trace_requirement_status"] == "pass" else states[1]
        return states[2] if row["status"] == "pass" else states[3]

    def progress(ax):
        gs = data["grades"]
        for field, label, color, marker in [
            ("passed", "Result passes (includes refusal)", style.GOLD, "o"),
            ("trace", "Execution + required trace", style.NAVY, "s"),
        ]:
            ys = [
                g["passed"] if field == "passed" else g["summary"]["execution_passes_with_trace_evidence"] for g in gs
            ]
            ax.step([g["minute"] for g in gs] + [T], ys + [ys[-1]], where="post", color=color, lw=2, label=label)
            ax.plot([g["minute"] for g in gs], ys, marker=marker, ls="", color=color, ms=5)
            for g, y in zip(gs[:2], ys[:2]):
                ax.annotate(
                    str(y),
                    (g["minute"], y),
                    xytext=(4, -13 if field == "trace" else 7),
                    textcoords="offset points",
                    color=color,
                    fontsize=9,
                    fontweight="bold",
                )
        ax.axvspan(0, gs[0]["minute"], color=style.SLATE, alpha=0.10)
        ax.text(
            gs[0]["minute"] * 0.48,
            73,
            "No comparable full-cohort\nverdict available yet",
            ha="center",
            va="center",
            color=style.SLATE,
            fontsize=9,
        )
        ax.set_ylim(0, 159)
        ax.set_ylabel("Capsules / 153")
        ax.legend(loc="lower right", ncol=2, frameon=False)

    # 1: Linked run anatomy. Observed response traffic is separated from wall activity.
    fig, axs = plt.subplots(
        5, 1, figsize=(13.3, 10.4), sharex=True, gridspec_kw={"height_ratios": [1.35, 1, 1, 1, 1.15], "hspace": 0.49}
    )
    fig.suptitle("One run, five synchronized views", fontfamily=style.SERIF, fontsize=23, x=0.10, ha="left", y=0.99)
    for ax in axs:
        base(ax)
        ax.axvline(lead_a, color=style.INK, lw=0.7, alpha=0.45, ls="--")
        ax.axvline(lead_b, color=style.INK, lw=0.7, alpha=0.45, ls="--")
    for a, b, label in ((0, lead_a, "Setup"), (lead_a, lead_b, "Authoring"), (lead_b, T, "Post-author validation")):
        axs[0].text(
            (a + b) / 2 / T,
            1.01,
            label,
            transform=axs[0].transAxes,
            ha="center",
            va="bottom",
            fontsize=8,
            color=style.SLATE,
        )
    section(axs[0], "a   Capsule progress · comparable full-cohort verdicts")
    progress(axs[0])
    edges = np.arange(0, T + 2, 2)
    x = (edges[:-1] + edges[1:]) / 2
    paired = [c for c in data["tools"] if c["completed"] and c["b"] is not None]
    tools_share = occupancy([(c["a"], c["b"]) for c in paired], edges, union=True)
    edits = occupancy([(c["a"], c["b"]) for c in paired if c["name"] != "Bash"], edges, union=True)
    observed = occupancy([(lead_a, lead_b)], edges, union=True)
    shares = [np.maximum(tools_share - edits, 0), edits, np.maximum(observed - tools_share, 0)]
    for i in range(len(shares)):
        shares[i] = np.divide(shares[i], observed, out=np.full_like(observed, np.nan), where=observed > 0)
    axs[1].stackplot(
        x,
        *shares,
        colors=[style.SLATE, style.SAGE, style.BLUE],
        labels=["Shell / tool interval", "Edit tool", "Client gap / model / network"],
        step="mid",
    )
    axs[1].set_ylim(0, 1)
    axs[1].yaxis.set_major_formatter(FuncFormatter(lambda n, _: f"{n:.0%}"))
    axs[1].set_ylabel("Lead wall share")
    section(axs[1], "b   Lead activity share · paired tool intervals, 2-minute bins")
    axs[1].legend(ncol=3, loc="upper right", frameon=False, fontsize=8)
    tokens = data["tokens"]
    rates = {}
    for key in ("fresh", "cached", "output", "reasoning"):
        rates[key] = np.histogram([t["minute"] for t in tokens], bins=edges, weights=[t[key] for t in tokens])[
            0
        ] / np.diff(edges)
    axs[2].stackplot(
        x,
        rates["cached"] / 1e6,
        rates["fresh"] / 1e6,
        colors=[style.SLATE, style.NAVY],
        labels=["Cached input", "Fresh input"],
        step="mid",
    )
    axs[2].set_ylabel("M input tokens/min")
    section(axs[2], "c   Observed input traffic · all seven actors")
    axs[2].legend(loc="upper right", ncol=2, frameon=False)
    axs[3].fill_between(x, rates["output"] / 1000, step="mid", color=style.SAGE, alpha=0.8, label="All output")
    axs[3].plot(x, rates["reasoning"] / 1000, color=style.BLUE, lw=1, label="Reasoning subset")
    axs[3].set_ylabel("k output tokens/min")
    section(axs[3], "d   Observed output traffic · binned response arrivals")
    axs[3].legend(loc="upper right", ncol=2, frameon=False)
    for category in C:
        sub = [(e["a"], e["b"]) for e in events if e["category"] == category]
        if sub:
            axs[4].plot(x, occupancy(sub, edges), color=C[category], lw=1.5, label=category)
    section(axs[4], "e   Concurrent recorded subprocesses · 2-minute averages")
    axs[4].set_ylabel("In flight")
    axs[4].legend(ncol=4, loc="upper right", frameon=False)
    axs[-1].set_xlabel("Wall time since campaign start (minutes)")
    fig.text(
        0.10,
        0.01,
        (
            "Dashed lines: observed author span. Historical run: formal completion failed; "
            "full-model qualification remained incomplete."
        ),
        fontsize=9,
        color=style.MAUVE,
    )
    fig.subplots_adjust(top=0.92, bottom=0.07, left=0.10, right=0.98)
    save(
        fig,
        "01_run_fingerprint",
        "Run fingerprint",
        (
            "Main-paper candidate: aligned progress, lead activity share, input/output traffic "
            "and actual tool occupancy. Cached input and reasoning subsets stay explicit. "
            "Unknown pre-verdict progress is not plotted as zero."
        ),
    )

    # 2: Agent turn lanes and interval-packed process lanes, explicitly not CPU cores.
    agents = sorted(data["agents"], key=lambda a: (a["role"] != "lead", a["observed_start"]))
    ordered = sorted(events, key=lambda e: (e["a"], e["b"]))
    heap, packed, nlanes = [], [], 0
    for event in ordered:
        if heap and heap[0][0] <= event["a"]:
            _, lane = heapq.heappop(heap)
        else:
            lane = nlanes
            nlanes += 1
        heapq.heappush(heap, (event["b"], lane))
        packed.append((event, lane))
    fig, axs = plt.subplots(
        2, 1, figsize=(13.3, 8), sharex=True, gridspec_kw={"height_ratios": [1, 1.9], "hspace": 0.36}
    )
    fig.suptitle("The parallel execution fabric", fontfamily=style.SERIF, fontsize=23, x=0.16, ha="left", y=0.99)
    for ax in axs:
        base(ax, "x")
    for i, agent in enumerate(agents):
        axs[0].plot([agent["observed_start"], agent["observed_end"]], [i, i], color=style.GOLD, lw=1, alpha=0.5)
        axs[0].broken_barh([(a, b - a) for a, b in agent["turns"]], (i - 0.28, 0.56), color=style.BLUE, alpha=0.55)
        axs[0].broken_barh([(c["a"], c["b"] - c["a"]) for c in agent["commands"]], (i - 0.28, 0.56), color=style.SAGE)
    axs[0].set_yticks(range(len(agents)), [a["name"].replace("/root/", "").replace("/root", "lead") for a in agents])
    axs[0].invert_yaxis()
    section(axs[0], "a   Agent turns with recorded command intervals")
    axs[0].legend(
        handles=[
            Patch(color=style.BLUE, alpha=0.55, label="Observed turn"),
            Patch(color=style.SAGE, label="Command wall interval"),
        ],
        loc="upper right",
        frameon=False,
    )
    for category, color in C.items():
        segs = [[(e["a"], lane), (e["b"], lane)] for e, lane in packed if e["category"] == category]
        axs[1].add_collection(LineCollection(segs, colors=color, linewidths=3, rasterized=False))
    axs[1].set_ylim(-1, nlanes)
    axs[1].set_ylabel("Packed process lanes")
    section(axs[1], f"b   {len(events):,} unique recorded invocations, packed into {nlanes} non-overlapping lanes")
    axs[1].legend(
        handles=[Patch(color=c, label=n) for n, c in C.items() if n != "Other"],
        ncol=4,
        frameon=False,
        loc="upper right",
    )
    axs[1].set_xlabel("Wall time since campaign start (minutes)")
    fig.text(
        0.16,
        0.015,
        (
            "Lanes are a compact packing of wall intervals. They do not identify physical "
            "cores; nested processes and waiting remain included."
        ),
        fontsize=9,
    )
    fig.subplots_adjust(left=0.16, right=0.98, top=0.91, bottom=0.09)
    save(
        fig,
        "02_execution_fabric",
        "Execution fabric",
        (
            "A truthful Gantt alternative: seven agent actors above actual invocation "
            "intervals. Process lanes are packed for readability, not assigned to the 18 CPU "
            "cores. This exposes overlapping compilation, Spike and gSIM work."
        ),
    )

    # 3: Capsule atlas and grade snapshots.
    names = sorted(c["capsule"] for c in data["grades"][-1]["capsules"])
    index = {name: i for i, name in enumerate(names)}
    fig = plt.figure(figsize=(13.3, 12))
    gs = fig.add_gridspec(1, 2, width_ratios=[6, 1.6], wspace=0.12)
    ax, snapshot = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    base(ax, "x")
    for category in ("L2 Spike", "L3 gSIM"):
        segs = [
            [
                (e["a"], index[e["capsule"]] + (-0.15 if category == "L2 Spike" else 0.15)),
                (e["b"], index[e["capsule"]] + (-0.15 if category == "L2 Spike" else 0.15)),
            ]
            for e in events
            if e["category"] == category
        ]
        ax.add_collection(LineCollection(segs, colors=C[category], linewidths=1.4))
        ax.scatter(
            [e["a"] for e in events if e["category"] == category],
            [
                index[e["capsule"]] + (-0.15 if category == "L2 Spike" else 0.15)
                for e in events
                if e["category"] == category
            ],
            s=2,
            color=C[category],
            linewidths=0,
        )
    for j, grade in enumerate(data["grades"]):
        for row in grade["capsules"]:
            snapshot.scatter(j, index[row["capsule"]], marker="s", s=19, color=SC[state(row)], linewidths=0)
    for axis in (ax, snapshot):
        axis.set_ylim(len(names), -1)
        axis.set_yticks([])
    for family in sorted({n.split("_")[0] for n in names}):
        ys = [i for n, i in index.items() if n.split("_")[0] == family]
        ax.text(
            -0.015,
            (min(ys) + max(ys)) / 2,
            f"{family}\n{len(ys)}",
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=10,
        )
        ax.axhline(min(ys) - 0.5, color=style.GOLD, lw=0.6)
        snapshot.axhline(min(ys) - 0.5, color=style.GOLD, lw=0.6)
    snapshot.set_xticks(range(len(data["grades"])), [f"{g['minute']:.0f}m" for g in data["grades"]])
    snapshot.tick_params(length=0)
    for spine in snapshot.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("Wall time since campaign start (minutes)")
    snapshot.set_xlabel("Verdict snapshots")
    section(ax, "Recorded simulation attempts · one row per capsule")
    section(snapshot, "Evidence status")
    fig.suptitle(
        "A capsule atlas: where work repeated, and where evidence changed",
        fontfamily=style.SERIF,
        fontsize=21,
        x=0.08,
        ha="left",
        y=0.99,
    )
    fig.legend(
        handles=[Patch(color=C[n], label=n) for n in ("L2 Spike", "L3 gSIM")]
        + [Patch(color=SC[n], label=n) for n in states],
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.55, 0.015),
    )
    fig.subplots_adjust(left=0.08, right=0.98, top=0.93, bottom=0.105)
    save(
        fig,
        "03_capsule_atlas",
        "Capsule atlas",
        (
            "All 153 public capsules as persistent rows: recorded Spike/gSIM attempts on the "
            "left, evidence states at the three comparable verdicts on the right. Dots mark "
            "starts so short runs remain visible. Family prefixes group rows; cache-carried "
            "verdicts do not invent fresh simulations."
        ),
    )

    # 4: Cost anatomy uses de-duplicated process intervals, never invented core seconds.
    fig = plt.figure(figsize=(13.3, 7.3))
    gs = fig.add_gridspec(2, 2, hspace=0.6, wspace=0.28)
    ax0, ax1, ax2, ax3 = [fig.add_subplot(gs[i, j]) for i, j in ((0, 0), (0, 1), (1, 0), (1, 1))]
    categories = [c for c in C if any(e["category"] == c for e in events)]
    times = [sum((e["b"] - e["a"]) for e in events if e["category"] == c) for c in categories]
    base(ax0, clock=False)
    style.hbars(ax0, np.arange(len(categories)), times, [C[c] for c in categories], height=0.57, shadow=False)
    ax0.set_yticks(range(len(categories)), categories)
    ax0.invert_yaxis()
    ax0.set_xlim(0, max(times) * 1.2)
    for i, value in enumerate(times):
        ax0.text(value + max(times) * 0.02, i, f"{value:.1f}", va="center", fontsize=9)
    ax0.set_xlabel("Sum of recorded process-wall minutes")
    section(ax0, "a   Where invocation time accumulates")
    base(ax1, clock=False)
    for category in categories:
        values = sorted((e["b"] - e["a"]) * 60 for e in events if e["category"] == category and e["b"] > e["a"])
        ax1.plot(values, np.arange(1, len(values) + 1) / len(values), color=C[category], label=category, lw=1.7)
    ax1.set_xscale("log")
    ax1.set_xlabel("Invocation wall duration (seconds, log scale)")
    ax1.set_ylabel("Fraction of invocations")
    ax1.legend(fontsize=7, frameon=False, loc="lower right")
    section(ax1, "b   The long tail, separated by stage")
    capsule_cost = collections.Counter()
    for e in events:
        capsule_cost[e["capsule"]] += e["b"] - e["a"]
    values = np.array(sorted(capsule_cost.values(), reverse=True))
    base(ax2, clock=False)
    ax2.plot(np.arange(1, len(values) + 1), np.cumsum(values) / sum(values), color=style.NAVY, lw=2)
    k80 = int(np.searchsorted(np.cumsum(values) / sum(values), 0.8) + 1)
    ax2.axvline(k80, color=style.GOLD, ls="--", lw=1)
    ax2.axhline(0.8, color=style.GOLD, ls="--", lw=1)
    ax2.text(
        0.98,
        0.13,
        f"{k80}/{len(values)} capsules account for 80%\nof recorded invocation time",
        transform=ax2.transAxes,
        ha="right",
        fontsize=10,
    )
    ax2.set_xlabel("Capsules, ranked by accumulated invocation time")
    ax2.set_ylabel("Cumulative time share")
    ax2.set_ylim(0, 1.02)
    section(ax2, "c   Concentration of repeated work")
    base(ax3, clock=False)
    counts = [sum(e["category"] == c for e in events) for c in categories]
    style.hbars(ax3, np.arange(len(categories)), counts, [C[c] for c in categories], height=0.57, shadow=False)
    ax3.set_yticks(range(len(categories)), categories)
    ax3.invert_yaxis()
    ax3.set_xlim(0, max(counts) * 1.2)
    for i, value in enumerate(counts):
        ax3.text(value + max(counts) * 0.02, i, f"{value:,}", va="center", fontsize=9)
    ax3.set_xlabel("Unique recorded subprocess invocations")
    section(ax3, "d   Frequency and duration are different costs")
    fig.suptitle("The cost anatomy of a run", fontfamily=style.SERIF, fontsize=23, x=0.11, ha="left", y=0.99)
    fig.text(
        0.11,
        0.012,
        (
            "Retained public-capsule invocations only. Wall intervals overlap and may nest; "
            "sums are neither campaign wall time nor CPU-core time."
        ),
        fontsize=9,
    )
    fig.subplots_adjust(top=0.88, left=0.13, right=0.98, bottom=0.12)
    save(
        fig,
        "04_cost_anatomy",
        "Cost anatomy",
        (
            "Stage totals, duration distributions and a capsule Pareto curve reveal whether "
            "frequency or long tails dominate. All costs derive from unique actual subprocess "
            "intervals; omitted/unlogged work stays unknown."
        ),
    )

    # 5: Self-check feedback queue; each row preserves its own subset denominator.
    reqs = [r for r in data["requests"] if r["b"] is not None and r["start"] is not None]
    fig, (ax, latency) = plt.subplots(1, 2, figsize=(13.3, 7.6), gridspec_kw={"width_ratios": [2.25, 1], "wspace": 0.4})
    base(ax, "x")
    base(latency, "x", clock=False)
    for i, r in enumerate(reqs):
        ax.broken_barh([(r["a"], r["start"] - r["a"])], (i - 0.3, 0.6), color=style.GOLD)
        ax.broken_barh([(r["start"], r["b"] - r["start"])], (i - 0.3, 0.6), color=style.NAVY)
        ax.scatter(r["b"], i, s=18, color=style.MAUVE, zorder=3)
        queue, grade = (r["start"] - r["a"]) * 60, (r["b"] - r["start"]) * 60
        latency.barh(i, queue, height=0.6, color=style.GOLD)
        latency.barh(i, grade, left=queue, height=0.6, color=style.NAVY)
        answer = f"{r['passed']}/{r['rows']}" if r["passed"] is not None and r["rows"] is not None else "no verdict"
        latency.text(queue + grade + 4, i, answer, va="center", fontsize=8)
    for axis in (ax, latency):
        axis.set_ylim(len(reqs) - 0.3, -0.7)
    ax.set_yticks(range(len(reqs)), [f"{i + 1:02d}  ({r['count']} requested)" for i, r in enumerate(reqs)])
    latency.set_yticks([])
    latency.set_xlim(0, max((r["b"] - r["a"]) * 60 for r in reqs) * 1.2)
    ax.set_xlabel("Wall time since campaign start (minutes)")
    latency.set_xlabel("Request-to-answer seconds")
    section(ax, "a   Feedback conveyor: requests can overlap")
    section(latency, "b   Queue + grading; label = subset passes")
    fig.suptitle(
        "How long does the agent wait to learn?", fontfamily=style.SERIF, fontsize=23, x=0.15, ha="left", y=0.99
    )
    fig.legend(
        handles=[
            Patch(color=style.GOLD, label="Queue"),
            Patch(color=style.NAVY, label="Grading"),
            Patch(color=style.MAUVE, label="Answer returned"),
        ],
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.035),
    )
    fig.subplots_adjust(left=0.15, right=0.98, top=0.88, bottom=0.15)
    save(
        fig,
        "05_feedback_conveyor",
        "Feedback conveyor",
        (
            "One row per self-check request makes asynchronous feedback visible. Requested and "
            "returned subset sizes remain separate from the global 153-capsule denominator. "
            "Queue and grading time are measured from request, start and answer timestamps."
        ),
    )

    # 6: Alluvial evidence flow across comparable whole-cohort snapshots.
    maps = [{c["capsule"]: state(c) for c in grade["capsules"]} for grade in data["grades"]]
    paths = collections.Counter(tuple(mapping[name] for mapping in maps) for name in sorted(maps[0]))
    fig, ax = plt.subplots(figsize=(12.2, 7.6))
    base(ax, None, clock=False)
    positions, counts = [], []
    gap = 5
    for mapping in maps:
        count = collections.Counter(mapping.values())
        counts.append(count)
        at, pos = 0, {}
        for s in states:
            pos[s] = at
            at += count[s] + gap
        positions.append(pos)
    offsets = [collections.Counter() for _ in maps]
    for seq, n in sorted(paths.items(), key=lambda p: tuple(states.index(s) for s in p[0])):
        starts = []
        for j, s in enumerate(seq):
            starts.append(positions[j][s] + offsets[j][s])
            offsets[j][s] += n
        for j in range(len(maps) - 1):
            left, right, a, b = j + 0.07, j + 1 - 0.07, starts[j], starts[j + 1]
            verts = [
                (left, a),
                (left + 0.36, a),
                (right - 0.36, b),
                (right, b),
                (right, b + n),
                (right - 0.36, b + n),
                (left + 0.36, a + n),
                (left, a + n),
                (left, a),
            ]
            codes = [MPath.MOVETO] + [MPath.CURVE4] * 3 + [MPath.LINETO] + [MPath.CURVE4] * 3 + [MPath.CLOSEPOLY]
            ax.add_patch(PathPatch(MPath(verts, codes), facecolor=SC[seq[j]], alpha=0.43, edgecolor="none"))
    for j, grade in enumerate(data["grades"]):
        for s in states:
            n = counts[j][s]
            if n:
                ax.bar(j, n, bottom=positions[j][s], width=0.13, color=SC[s], edgecolor=style.BG, lw=0.6)
                ax.text(
                    j if n >= 5 else j + 0.10,
                    positions[j][s] + n / 2,
                    str(n),
                    color=(style.BG if s != "Numeric only" else style.INK) if n >= 5 else SC[s],
                    ha="center" if n >= 5 else "left",
                    va="center",
                    fontsize=9,
                    fontweight="bold",
                )
    ax.set_xlim(-0.23, len(maps) - 0.7)
    ax.set_ylim(175, -5)
    ax.set_xticks(
        range(len(maps)),
        [f"{g['minute']:.1f} min\n{g['source'].replace('verdict_', '').replace('.json', '')}" for g in data["grades"]],
    )
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.suptitle(
        "From numerical success to evidence-backed execution",
        fontfamily=style.SERIF,
        fontsize=22,
        x=0.08,
        ha="left",
        y=0.98,
    )
    ax.set_title(
        "Each ribbon follows the same capsules between comparable verdicts; width = capsule count",
        loc="left",
        fontsize=11,
        pad=20,
    )
    fig.legend(
        handles=[Patch(color=SC[s], label=s) for s in states],
        ncol=4,
        loc="lower center",
        frameon=False,
        bbox_to_anchor=(0.5, 0.05),
    )
    fig.text(
        0.08,
        0.012,
        (
            "Execution + trace is a capsule evidence state. Full-model qualification is a "
            "separate gate; formal completion failed in this historical run."
        ),
        fontsize=9,
        color=style.MAUVE,
    )
    fig.subplots_adjust(left=0.08, right=0.98, top=0.85, bottom=0.18)
    save(
        fig,
        "06_evidence_river",
        "Evidence river",
        (
            "An alluvial diagram follows identical capsule identities across verdicts. It "
            "separates numerical-only passes, passes with trace evidence, expected refusal and "
            "unresolved cases. This makes qualification debt tangible without treating a new "
            "score as a new experiment."
        ),
    )

    # 7: Observational abstraction adoption, with zero-recognized-witness rows explicit.
    if data.get("abstractions"):
        families = {
            "direct_facts_file_inspection": "Facts file inspection",
            "cca_bijection": "CCA bijection",
            "cca_escalation_ladder": "Lever catalog",
            "isa_lint": "ISA lint",
            "isa_asm": "ISA assembly",
            "isa_disasm": "ISA disassembly",
            "isa_debug": "ISA debug",
            "rtl_facts": "Facts API",
            "derived_levers": "Derived levers API",
            "scaffold": "Scaffold generator",
        }
        fig, (ax, counts_ax) = plt.subplots(
            1, 2, figsize=(13.3, 6.5), gridspec_kw={"width_ratios": [2.3, 1], "wspace": 0.38}
        )
        base(ax, "x")
        base(counts_ax, "x", clock=False)
        counts = collections.Counter()
        failed = collections.Counter()
        for i, (family, label) in enumerate(families.items()):
            for marker in data["abstractions"]:
                if marker["category"] != family:
                    continue
                ok = marker["outer_command_succeeded"]
                (counts if ok else failed)[family] += 1
                ax.scatter(
                    marker["a"],
                    i,
                    marker="o" if ok else "x",
                    s=40,
                    color=style.NAVY if ok else style.MAUVE,
                    linewidths=1.1,
                )
            counts_ax.barh(i, counts[family], height=0.55, color=style.NAVY)
            if failed[family]:
                counts_ax.barh(i, failed[family], left=counts[family], height=0.55, color=style.MAUVE)
            counts_ax.text(
                counts[family] + failed[family] + 0.16,
                i,
                str(counts[family] + failed[family]),
                va="center",
                fontsize=10,
            )
        for axis in (ax, counts_ax):
            axis.set_ylim(len(families) - 0.4, -0.8)
        ax.set_yticks(range(len(families)), families.values())
        counts_ax.set_yticks([])
        ax.axvspan(lead_b, T, color=style.SLATE, alpha=0.10)
        ax.text(
            (lead_b + T) / 2,
            len(families) / 2,
            "Post-author\nvalidation",
            ha="center",
            va="center",
            color=style.SLATE,
            fontsize=10,
        )
        counts_ax.set_xlim(0, max(counts.values()) + 2)
        ax.set_xlabel("Wall time since campaign start (minutes)")
        counts_ax.set_xlabel("Recognized request witnesses")
        section(ax, "a   When abstraction requests appear in retained commands")
        section(counts_ax, "b   Recorded adoption")
        fig.suptitle(
            "Which abstractions did the author reach for?",
            fontfamily=style.SERIF,
            fontsize=23,
            x=0.16,
            ha="left",
            y=0.99,
        )
        fig.legend(
            handles=[
                Patch(color=style.NAVY, label="Successful outer shell"),
                Patch(color=style.MAUVE, label="Failed outer shell"),
            ],
            loc="lower center",
            ncol=2,
            frameon=False,
            bbox_to_anchor=(0.5, 0.055),
        )
        fig.text(
            0.16,
            0.015,
            (
                "Counts are structural request witnesses, not dynamic calls. Zero = no recognized "
                "witness. Causal benefits require matched experiment arms."
            ),
            fontsize=9,
        )
        fig.subplots_adjust(left=0.16, right=0.98, top=0.85, bottom=0.18)
        save(
            fig,
            "07_abstraction_adoption",
            "Abstraction adoption",
            (
                "A structural command/API audit separates exposure from observed adoption. "
                "ISA/CCA/facts-file inspection have recognized requests; several mandated APIs "
                "have no recognized witness. This is descriptive evidence. To measure abstraction "
                "effects, compare matched EL1–EL4 arms (and define any additional control "
                "explicitly), using the same capsules, gates, budget and hardware, with repeated "
                "fresh authors."
            ),
        )

    # Offline gallery with direct vector downloads and the full interpretation limits.
    cards = "\n".join(
        f'''<article><h2>{i + 1:02d} · {html.escape(f["label"])}</h2>
      <p>{html.escape(f["description"])}</p><a href="{f["name"]}.svg">
      <img loading="lazy" src="{f["name"]}.png" alt="{html.escape(f["label"])}"></a>
      <nav><a href="{f["name"]}.svg">Editable SVG</a>
      <a href="{f["name"]}.pdf">PDF</a><a href="{f["name"]}.png">PNG</a></nav></article>'''
        for i, f in enumerate(figures)
    )
    limits = "".join(f"<li>{html.escape(note)}</li>" for note in data["limits"])
    gallery = (Path(__file__).parent / "templates/phase1_gallery.html").read_text()
    replacements = {
        "BG": style.BG,
        "INK": style.INK,
        "NAVY": style.NAVY,
        "GOLD": style.GOLD,
        "MAUVE": style.MAUVE,
        "NFIGURES": str(len(figures)),
        "RUN": html.escape(data["run"]),
        "DURATION": f"{T:.0f}",
        "CARDS": cards,
        "LIMITS": limits,
    }
    for key, value in replacements.items():
        gallery = gallery.replace(f"@@{key}@@", value)
    (out / "index.html").write_text(gallery)
    (out / "figures.json").write_text(json.dumps(figures, indent=2) + "\n")
    capsule_explorer(data, out, C, SC)
    contact_sheet(out, figures, style)
    inputs = [Path(__file__), Path(__file__).with_name("phase1_abstraction_trace.py"), style_path]
    inputs += sorted((Path(__file__).parent / "templates").glob("phase1_*.html"))
    rendered = sorted(p for p in out.iterdir() if p.suffix in (".svg", ".pdf", ".png", ".html"))
    receipt = {
        "schema": "merlin.paper_render.v1",
        "normalized_data_sha256": hashlib.sha256((out / "data.json").read_bytes()).hexdigest(),
        "source_sha256": {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        "matplotlib_version": matplotlib.__version__,
        "numpy_version": np.__version__,
        "body_font": plt.rcParams["font.family"],
        "title_font": style.SERIF,
        "figure_families": len(figures),
        "outputs": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in rendered},
    }
    (out / "render_manifest.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"Rendered {len(figures)} figure families to {out}")


def capsule_explorer(data, out, colors, state_colors):
    """Small offline SVG explorer, no external JS/CSS or raw tool I/O."""
    payload = json.dumps(
        {
            "duration": data["duration"],
            "grades": data["grades"],
            "invocations": [
                {k: e[k] for k in ("a", "b", "capsule", "category", "failed")}
                for e in data["invocations"]
                if e["category"] in ("L2 Spike", "L3 gSIM")
            ],
            "colors": colors,
            "stateColors": state_colors,
        },
        separators=(",", ":"),
    ).replace("<", "\\u003c")
    template = (Path(__file__).parent / "templates/phase1_capsule_explorer.html").read_text()
    (out / "capsule_explorer.html").write_text(template.replace("@@PAYLOAD@@", payload))


def contact_sheet(out, figures, style):
    from PIL import Image, ImageDraw

    width, cell_w, cell_h = 1800, 900, 810
    canvas = Image.new("RGB", (width, cell_h * math.ceil(len(figures) / 2) + 90), style.BG)
    draw = ImageDraw.Draw(canvas)
    draw.text((30, 20), f"PHASE 1 / {len(figures)} EVIDENCE-BACKED VIEWS", fill=style.INK, font_size=28)
    for i, figure in enumerate(figures):
        im = Image.open(out / f"{figure['name']}.png").convert("RGB")
        im.thumbnail((cell_w - 30, cell_h - 65))
        x, y = (i % 2) * cell_w + 15, (i // 2) * cell_h + 90
        draw.text((x + 10, y), f"{i + 1:02d}  {figure['label']}", fill=style.NAVY, font_size=23)
        canvas.paste(im, (x + (cell_w - 30 - im.width) // 2, y + 40))
    canvas.save(out / "contact_sheet.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--conclusion", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--extract-only", action="store_true")
    args = parser.parse_args()
    if args.data:
        data = json.loads(args.data.read_text())
        args.out.mkdir(parents=True, exist_ok=True)
        destination = args.out / "data.json"
        if destination.resolve() != args.data.resolve():
            destination.write_bytes(args.data.read_bytes())
    elif args.run and args.conclusion:
        data = extract(args.run.resolve(), args.conclusion.resolve(), args.out.resolve())
    else:
        parser.error("Use --data or both --run and --conclusion")
    if not args.extract_only:
        render(data, args.out.resolve())


if __name__ == "__main__":
    main()
