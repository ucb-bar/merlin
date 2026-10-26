"""Measured performance over time: gSIM cycles per test case, speedups, utilization, movers and model gaps.

Read from the agent channel's gSIM simulation responses (``simresp_*.json`` with ``sim == "gsim"``,
``barrier_cycles`` per test case, timed by the file's modification time) of this run and, when a
campaign chain names one, of the previous attempt. Speedups compare each test case's latest measured
cycles with its first measurement in this run (and with the previous attempt's latest); families come
from the test case's declared operation; utilization is useful MACs over cycles × peak from the
:mod:`.perf` rows. Spike-model estimates are labelled ``modeled`` and compared with gSIM only as a gap
ratio. Operator-private reference targets (for example a study's ``pairs`` of agent test case and
reference kernel cycles) appear only with ``operator_private``. Display telemetry; it grades nothing.

Panel fields beyond the shared shape: ``families`` [{family, cases, speedup, util_median_pct}],
``movers`` {up, down}, ``model_gap`` {compared, median_ratio, over_2x}, ``targets`` [...].
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from ..io import clean, is_dir, mtime, num, obj, positive, rows, scan_dir
from .insight_common import (
    chain,
    line,
    out_root,
    panel,
    previous_workspace,
    private_rows,
    ratio_text,
    row,
    series,
    table,
)

MAX_NEW_FILES = 400
HOUR = 3600.0
#: Declared operations by family; anything else is ``other``. Operation names, not target names.
FAMILIES = (
    ("conv", ("conv",)),
    ("dense", ("matmul", "gemm", "gemv", "contraction", "linear", "dot", "resident_reuse")),
    ("epilogue", ("residual", "bias", "relu", "requant", "epilogue", "pointwise", "activation")),
    ("program", ("model", "component_program", "host_island")),
)


def family(op: Any) -> str:
    text = str(op or "").lower()
    for name, needles in FAMILIES:
        if any(n in text for n in needles):
            return name
    return "other"


def _scan(ctx, workspace: str | None, store: dict) -> dict[str, list[tuple[float, float]]]:
    """``{test case: [(t, cycles)]}`` of one channel, extended with responses not read before."""
    if not workspace:
        return store.setdefault("cases", {})
    channel = Path(workspace) / ".qa_channel"
    seen: set = store.setdefault("seen", set())
    cases: dict = store.setdefault("cases", {})
    if not is_dir(channel):
        return cases
    fresh = 0
    for entry in scan_dir(channel, 32768):
        name = entry.name
        if not (name.startswith("simresp_") and name.endswith(".json")) or name in seen:
            continue
        if fresh >= MAX_NEW_FILES:
            break
        fresh += 1
        document = ctx.reader.document(Path(entry.path), limit=8 << 20)
        if document is None:
            continue  # unreadable or over budget: retried next refresh
        seen.add(name)
        if document.get("sim") != "gsim":
            continue
        at = mtime(entry.path)
        for r in rows(document.get("per_capsule")):
            cycles = positive(r.get("barrier_cycles"))
            name_ = r.get("capsule")
            if cycles is not None and isinstance(name_, str) and at:
                cases.setdefault(name_, []).append((at, float(cycles)))
    for values in cases.values():
        values.sort()
    return cases


def _geomean(values: list[float]) -> float | None:
    values = [v for v in values if v and v > 0 and math.isfinite(v)]
    if not values:
        return None
    return math.exp(sum(math.log(v) for v in values) / len(values))


def _speedup_series(cases: dict) -> list[tuple[float, float]]:
    events = sorted((t, name, c) for name, values in cases.items() for t, c in values)
    first: dict[str, float] = {}
    latest: dict[str, float] = {}
    seen: dict[str, int] = {}
    points = []
    for i, (t, name, cycles) in enumerate(events):
        first.setdefault(name, cycles)
        latest[name] = cycles
        seen[name] = seen.get(name, 0) + 1
        if i % 8 == 7 or i == len(events) - 1:
            g = _geomean([first[n] / latest[n] for n in latest if seen[n] >= 2])
            if g is not None:
                points.append((t, round(g, 4)))
    return points


def _targets(ctx) -> list[dict[str, Any]]:
    """Operator-private reference pairs found under the run's ``out/artifacts/perf-studies``."""
    if not ctx.operator_private:
        return []
    root = out_root(ctx.experiment)
    if root is None:
        return []
    found = []
    studies = root / "artifacts" / "perf-studies"
    for study in scan_dir(studies, 64):
        for entry in scan_dir(Path(study.path), 64):
            if not entry.name.endswith(".json") or entry.name.startswith("."):
                continue
            document = ctx.reader.document(Path(entry.path), limit=4 << 20)
            for pair in rows(obj(document).get("pairs")):
                case = pair.get("agent_case")
                metric = pair.get("exo_metric") or "call1_cycles"
                scale = num(pair.get("exo_scale")) or 1
                ref = positive(obj(pair.get("exo_gsim")).get(metric))
                if isinstance(case, str) and ref is not None:
                    found.append(
                        {
                            "case": case,
                            "reference_cycles": ref * scale,
                            "label": clean(pair.get("exo_kernel") or pair.get("id"), 32),
                            # The study's own agent measurement (same instrument, an earlier compiler).
                            "study_agent_cycles": positive(obj(pair.get("agent_gsim")).get(metric)),
                            "study_compiler": clean(pair.get("agent_compiler_sha256"), 12) or None,
                            "source": entry.path,
                        }
                    )
    return found


def read(ctx) -> dict[str, Any]:
    now = ctx.now
    store = ctx.state.setdefault("current", {})
    cases = _scan(ctx, ctx.experiment.get("workspace"), store)
    previous = previous_workspace(ctx)
    prev_store = ctx.state.setdefault("previous", {})
    if prev_store.get("workspace") != previous:
        prev_store.clear()
        prev_store["workspace"] = previous
    prev_cases = _scan(ctx, previous, prev_store) if previous else {}

    capsule_rows = {r.get("name"): r for r in private_rows(ctx, "capsules")}
    perf_rows = {r.get("capsule"): r for r in private_rows(ctx, "perf")}

    latest = {n: v[-1][1] for n, v in cases.items() if v}
    first = {n: v[0][1] for n, v in cases.items() if v}
    multi = [n for n, v in cases.items() if len(v) >= 2]
    speedup = _geomean([first[n] / latest[n] for n in multi])
    prev_latest = {n: v[-1][1] for n, v in prev_cases.items() if v}
    common = [n for n in latest if n in prev_latest]
    vs_prev = _geomean([prev_latest[n] / latest[n] for n in common])

    def utilization(name: str) -> float | None:
        perf = perf_rows.get(name) or {}
        macs, peak = positive(perf.get("useful_macs")), positive(perf.get("peak_macs_per_cycle"))
        cycles = latest.get(name)
        if macs is None or peak is None or not cycles:
            return None
        return 100.0 * macs / (cycles * peak)

    utils = sorted(u for u in (utilization(n) for n in latest) if u is not None)
    median_util = utils[len(utils) // 2] if utils else None

    fam: dict[str, dict[str, list]] = {}
    for name in latest:
        key = family(obj(capsule_rows.get(name)).get("op"))
        bucket = fam.setdefault(key, {"cases": [], "ratios": [], "utils": []})
        bucket["cases"].append(name)
        if name in multi:
            bucket["ratios"].append(first[name] / latest[name])
        u = utilization(name)
        if u is not None:
            bucket["utils"].append(u)
    families = []
    for key in ("dense", "conv", "epilogue", "program", "other"):
        bucket = fam.get(key)
        if not bucket:
            continue
        u = sorted(bucket["utils"])
        families.append(
            {
                "family": key,
                "cases": len(bucket["cases"]),
                "speedup": _geomean(bucket["ratios"]),
                "util_median_pct": u[len(u) // 2] if u else None,
            }
        )

    movers = []
    for name, values in cases.items():
        before = [c for t, c in values if t <= now - HOUR]
        recent = [c for t, c in values if t > now - HOUR]
        if before and recent:
            movers.append((before[-1] / recent[-1], name))
    movers.sort()
    up = [m for m in reversed(movers) if m[0] > 1.02][:3]
    down = [m for m in movers if m[0] < 0.98][:3]

    gaps = []
    for name, perf in perf_rows.items():
        modeled = positive(obj(perf).get("spike_model_cycles"))
        measured = latest.get(name)
        if modeled is not None and measured:
            gaps.append(modeled / measured)
    gaps.sort()
    model_gap = {
        "compared": len(gaps),
        "median_ratio": gaps[len(gaps) // 2] if gaps else None,
        "over_2x": sum(1 for g in gaps if g > 2 or g < 0.5),
    }

    targets = []
    for target in _targets(ctx):
        agent, basis = latest.get(target["case"]), "this run"
        if agent is None and target["study_agent_cycles"]:
            agent, basis = target["study_agent_cycles"], "study"
        targets.append(
            {
                **target,
                "agent_cycles": agent,
                "basis": basis if agent else "not measured",
                "ratio": (agent / target["reference_cycles"]) if agent else None,
            }
        )

    private = ctx.operator_private
    summary = [
        line("test cases measured", len(latest)),
        line(
            f"speedup vs first (n={len(multi)})",
            ratio_text(speedup) if multi else "no case re-measured yet",
            "ok" if speedup and speedup > 1.05 else None,
            key="families",
        ),
        line(
            f"vs previous attempt (n={len(common)})",
            ratio_text(vs_prev) if previous else "no previous attempt",
            "warn" if vs_prev is not None and vs_prev < 0.98 else None,
        ),
        line(
            "median array utilization",
            f"{median_util:.1f}%" if median_util is not None else None,
            "warn" if median_util is not None and median_util < 10 else None,
        ),
        line(
            "moved in the last hour",
            f"{len(up)} faster · {len(down)} slower" if movers else "no repeat in the last hour",
        ),
        line(
            "modeled ÷ measured",
            f"median ×{model_gap['median_ratio']:.2f}, {model_gap['over_2x']} off >2×"
            if model_gap["median_ratio"]
            else None,
            "warn" if model_gap["over_2x"] else None,
            key="model_gap",
        ),
    ]
    table_rows = [
        row(
            [
                f["family"],
                f["cases"],
                ratio_text(f["speedup"]),
                f"{f['util_median_pct']:.1f}%" if f["util_median_pct"] is not None else None,
            ],
            key=f["family"],
        )
        for f in families
    ]
    for ratio, name in up:
        table_rows.append(
            row(["↑ " + (name if private else "case"), "", ratio_text(ratio), ""], "ok", key=name if private else None)
        )
    for ratio, name in down:
        table_rows.append(
            row(
                ["↓ " + (name if private else "case"), "", ratio_text(ratio), ""], "warn", key=name if private else None
            )
        )
    for target in targets[:6]:
        table_rows.append(
            row(
                [
                    "vs " + target["label"],
                    "",
                    ratio_text(target["ratio"]) if target["ratio"] else None,
                    target["basis"],
                ],
                "warn" if (target["ratio"] or 0) > 1.2 else None,
                key=target["case"],
            )
        )
    notes = ["cycles are measured gSIM barrier cycles; speedup = first ÷ latest (higher is faster)"]
    if not cases:
        notes.append("no gSIM measurement in this run's channel yet")
    if targets:
        notes.append(
            "vs rows: agent ÷ reference gSIM cycles (below ×1 beats it); basis 'study' is the study's own "
            "agent measurement of an earlier compiler; operator-only targets, not numerical goldens"
        )
    return panel(
        "PERFORMANCE",
        summary,
        table_=table(["family / case", "n", "speedup", "util"], table_rows),
        series_=[series("geomean speedup vs first", _speedup_series(cases))],
        notes=notes,
        families=families,
        movers={
            "up": [[r, n if private else None] for r, n in up],
            "down": [[r, n if private else None] for r, n in down],
        },
        model_gap=model_gap,
        targets=targets if private else [],
        speedup=speedup,
        vs_previous=vs_prev,
        median_utilization_pct=median_util,
        chain_manifest=chain(ctx).get("manifest") if private else None,
    )


def inspect(ctx, key: str) -> dict[str, Any]:
    section = obj(ctx.shared.get("performance"))
    if key == "model_gap":
        gap = obj(section.get("model_gap"))
        return {
            "title": "modeled vs measured cycles",
            "sections": [
                {
                    "title": "Spike model ÷ gSIM",
                    "lines": [
                        f"compared test cases: {gap.get('compared')}",
                        f"median ratio: {gap.get('median_ratio')}",
                        f"off by more than 2× either way: {gap.get('over_2x')}",
                        "A ratio far from 1 means the fast cycle model misleads tuning decisions on that case.",
                    ],
                }
            ],
            "paths": [],
        }
    cases = ctx.state.get("current", {}).get("cases", {})
    if key in cases and ctx.operator_private:
        values = cases[key]
        return {
            "title": f"gSIM cycles · {key}",
            "sections": [{"title": "measurements", "lines": [f"{v[1]:.0f} cycles" for v in values[-24:]]}],
            "paths": [],
        }
    return {
        "title": f"performance {key}",
        "sections": [{"title": "families", "lines": [json.dumps(f) for f in section.get("families") or []]}],
        "paths": [],
    }
