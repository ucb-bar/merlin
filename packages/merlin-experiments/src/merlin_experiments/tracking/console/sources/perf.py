"""Per-capsule performance: Spike-model estimate, gSIM cycles, structural floor, intensity and modeled activity.

The spike-model estimate is read only through :func:`..terminal_metadata.capsule_metric`, unchanged,
so an estimate whose comparison binding does not match stays ``None``. gSIM cycles are the barrier
cycles a native simulation response records for an L3 pass with the gSIM engine identity. A
divergence between the two is computed only when both were measured on the same full compiler
source digest; otherwise the row says why they are not comparable. Every number keeps its kind:
``spike_model_cycles`` is a model estimate, ``gsim_cycles`` an RTL-simulation count, ``floor_cycles``
a structural bound, and modeled resource activity is not measured utilization.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from ...terminal_metadata import capsule_metric
from ..io import clean, epoch, is_sha256, num, obj, positive, rows, scan_dir
from .insight_common import share_rows

MAX_RESPONSES = 160
MAX_SIM_RESPONSES = 320
GSIM_ENGINE = "gsim"


def _channels(experiment: dict) -> list[Path]:
    out = []
    if experiment.get("workspace"):
        out.append(Path(experiment["workspace"]) / ".qa_channel")
    out.append(Path(experiment["run_dir"]) / "agent_evidence_snapshot" / ".qa_channel")
    return out


def _responses(experiment: dict) -> tuple[list[tuple[float, Path]], list[tuple[float, Path]], bool]:
    """Newest response files by kind, deduplicated by name across the live and retained channels."""
    seen: dict[str, tuple[float, Path]] = {}
    capped = False
    for channel in _channels(experiment):
        for entry in scan_dir(channel, 16384):
            name = entry.name
            if not name.endswith(".json") or not (name.startswith("resp_") or name.startswith("simresp_")):
                continue
            if name in seen:
                continue
            try:
                seen[name] = (entry.stat().st_mtime, Path(entry.path))
            except OSError:
                continue
    responses = sorted((v for k, v in seen.items() if k.startswith("resp_")), reverse=True)
    sims = sorted((v for k, v in seen.items() if k.startswith("simresp_")), reverse=True)
    capped = len(responses) > MAX_RESPONSES or len(sims) > MAX_SIM_RESPONSES
    return responses[:MAX_RESPONSES], sims[:MAX_SIM_RESPONSES], capped


def _row_source(row: dict, declared: Any) -> str | None:
    common = obj(obj(row.get("tier_boundary")).get("common"))
    for value in (obj(row.get("compiler_source")).get("candidate_sha256"), common.get("compiler_sha256"), declared):
        if is_sha256(value):
            return value
    return None


def _roofline(perf: dict) -> dict[str, Any]:
    roof = obj(perf.get("structural_roofline"))
    service = obj(roof.get("logical_request_ideal_service"))
    requested = [positive(obj(service.get(k)).get("requested_bytes")) for k in ("read", "write")]
    total = sum(v for v in requested if v) or None
    macs = positive(roof.get("useful_macs"))
    return {
        "useful_macs": macs,
        "peak_macs_per_cycle": positive(roof.get("structural_peak_macs_per_cycle")),
        "requested_bytes": total,
        "intensity": round(macs / total, 4) if macs and total else None,
    }


def _raw_resources(perf: dict, cycles: float | None) -> list[dict[str, Any]]:
    """Whole-trace modeled bookings when no scoped feedback exists: busy unit-cycles per estimated cycle."""
    if not cycles:
        return []
    out = []
    for resource in rows(obj(perf.get("modeled_resource_activity")).get("resources"), 64):
        busy = num(resource.get("busy_unit_cycles"))
        if busy is not None:
            out.append(
                {
                    "name": clean(resource.get("name") or f"resource {resource.get('resource_id')}", 32),
                    "activity_pct": round(100 * busy / cycles, 2),
                    "basis": "busy unit-cycles / estimated cycles; capacity not recorded",
                }
            )
    return sorted(out, key=lambda r: -r["activity_pct"])[:3]


def _record(row: dict, source: str | None, path: Path, at: float) -> dict[str, Any] | None:
    perf = obj(row.get("performance"))
    if not perf:
        return None
    metric = capsule_metric(row, source_sha256=source, source={"path": str(path)})
    cycles = metric.get("cycles")
    resources = [
        {
            "name": clean(r.get("name"), 48),
            "activity_pct": round(r["activity_pct"], 2),
            "basis": "scoped modeled activity",
        }
        for r in metric.get("modeled_resource_activity") or []
    ]
    if not resources:
        resources = _raw_resources(perf, cycles)
    scoped = obj(perf.get("scoped_resource_feedback"))
    traffic = (
        [
            {"boundary": clean(t.get("boundary"), 40), "bytes": t.get("bytes")}
            for t in rows(scoped.get("traffic"), 16)
            if num(t.get("bytes")) is not None
        ]
        if resources and metric.get("modeled_resource_activity_status") == "modeled"
        else []
    )
    return {
        "at": at,
        "source_sha256": source,
        "path": str(path),
        "cycles": cycles,
        "floor": metric.get("roofline_cycles"),
        "efficiency": metric.get("roofline_efficiency_pct"),
        "model_status": metric.get("model_status"),
        "timing_mode": metric.get("timing_mode"),
        "binding": metric.get("binding_status"),
        "resources": resources,
        "traffic": traffic,
        **_roofline(perf),
    }


def collect(ctx) -> tuple[dict[str, dict], dict[str, dict], dict[str, dict], list[str]]:
    """``(newest record, newest modeled record, newest gSIM pass)`` per capsule, plus notes."""
    responses, sims, capped = _responses(ctx.experiment)
    newest: dict[str, dict] = {}
    modeled: dict[str, dict] = {}
    gsim: dict[str, dict] = {}
    notes = ["response inventory capped to the newest files"] if capped else []

    def consider(name: str, record: dict | None) -> None:
        if record is None:
            return
        if name not in newest or record["at"] > newest[name]["at"]:
            newest[name] = record
        if record["cycles"] is not None and (name not in modeled or record["at"] > modeled[name]["at"]):
            modeled[name] = record

    for mtime, path in responses:
        doc = ctx.reader.document(path, limit=16 << 20)
        if not doc:
            continue
        at = epoch(doc.get("graded_at_unix_ns")) or mtime
        declared = doc.get("submission_sha256") or doc.get("graded_source_sha256")
        for row in rows(doc.get("cases")) + rows(doc.get("per_capsule")):
            name = row.get("capsule") or row.get("name")
            if isinstance(name, str):
                consider(name, _record(row, _row_source(row, declared), path, at))
    for mtime, path in sims:
        doc = ctx.reader.document(path, limit=16 << 20)
        if not doc:
            continue
        at = epoch(doc.get("graded_at_unix_ns")) or mtime
        for row in rows(doc.get("per_capsule")):
            name = row.get("capsule")
            if not isinstance(name, str):
                continue
            source = _row_source(row, doc.get("submission_sha256"))
            consider(name, _record(row, source, path, at))
            engine = obj(row.get("barrier_timing_identity")).get("engine")
            cycles = positive(row.get("barrier_cycles"))
            if (
                cycles
                and row.get("barrier_tier") == "L3"
                and row.get("barrier_status") == "pass"
                and engine == GSIM_ENGINE
                and (name not in gsim or at > gsim[name]["at"])
            ):
                gsim[name] = {"at": at, "cycles": cycles, "source_sha256": source, "path": str(path)}
    return newest, modeled, gsim, notes


def read(ctx) -> dict[str, Any]:
    newest, modeled, gsim, notes = collect(ctx)
    catalog = {c["name"]: c for c in (obj(ctx.shared.get("capsules")).get("capsules") or []) if isinstance(c, dict)}
    out = []
    peaks: dict[float, int] = {}
    for name in sorted(set(newest) | set(gsim)):
        latest = newest.get(name)
        record = modeled.get(name) or latest or {}
        sim = gsim.get(name)
        cycles = record.get("cycles")
        basis = []
        if latest and record is not latest:
            basis.append(
                f"estimate from an older record ({str(record.get('source_sha256') or 'source unknown')[:12]});"
                " the newest record has no bound estimate"
            )
        elif latest and cycles is None:
            basis.append(f"model estimate {latest.get('model_status') or 'UNKNOWN'}; binding {latest.get('binding')}")
        divergence, comparable = None, False
        if sim and cycles:
            if sim["source_sha256"] and sim["source_sha256"] == record.get("source_sha256"):
                divergence, comparable = round(100 * (sim["cycles"] - cycles) / cycles, 2), True
            else:
                basis.append("gSIM and model estimate come from different compiler sources; not compared")
        intensity = record.get("intensity")
        achieved = round(record["useful_macs"] / cycles, 4) if record.get("useful_macs") and cycles else None
        peak = record.get("peak_macs_per_cycle")
        if peak:
            peaks[peak] = peaks.get(peak, 0) + 1
        resources = record.get("resources") or []
        out.append(
            {
                "capsule": name,
                "cohort": obj(catalog.get(name)).get("cohort"),
                "kind": obj(catalog.get(name)).get("kind"),
                "spike_model_cycles": cycles,
                "gsim_cycles": sim["cycles"] if sim else None,
                "l2_cycles": obj(obj(catalog.get(name)).get("cycles")).get("L2"),
                "floor_cycles": record.get("floor"),
                "efficiency_pct": record.get("efficiency"),
                "useful_macs": record.get("useful_macs"),
                "peak_macs_per_cycle": peak,
                "requested_bytes": record.get("requested_bytes"),
                "intensity": intensity,
                "achieved_macs_per_cycle": achieved,
                "model_status": record.get("model_status"),
                "timing_mode": record.get("timing_mode"),
                "top_resource": resources[0] if resources else None,
                "resources": resources,
                "traffic": record.get("traffic") or [],
                "divergence_pct": divergence,
                "comparable": comparable,
                "basis": "; ".join(basis) or None,
                "source": record.get("path") or (sim or {}).get("path"),
                "source_sha256": record.get("source_sha256") or (sim or {}).get("source_sha256"),
            }
        )
    efficiencies = [
        r["efficiency_pct"] for r in out if isinstance(r["efficiency_pct"], int | float) and r["efficiency_pct"] > 0
    ]
    summary = {
        "modeled": sum(1 for r in out if r["spike_model_cycles"] is not None),
        "provisional": sum(1 for r in out if r["model_status"] == "PROVISIONAL"),
        "unknown": sum(1 for r in out if r["spike_model_cycles"] is None and r["model_status"] in (None, "UNKNOWN")),
        "gsim": sum(1 for r in out if r["gsim_cycles"] is not None),
        "comparable": sum(1 for r in out if r["comparable"]),
        "best_eff": max(efficiencies) if efficiencies else None,
        "worst_eff": min(efficiencies) if efficiencies else None,
        "geo_eff": round(math.exp(sum(math.log(e) for e in efficiencies) / len(efficiencies)), 4)
        if efficiencies
        else None,
    }
    share_rows(ctx, "perf", out)
    section = {
        "rows": out if ctx.operator_private else [],
        "summary": summary,
        "peak_macs_per_cycle": max(peaks, key=peaks.get) if peaks else None,
        "notes": notes
        + ["efficiency = structural floor / model estimate; modeled activity is not measured utilization"],
    }
    if not ctx.operator_private:
        section["notes"].append("per-capsule performance rows require operator-private mode")
    return section


def inspect(ctx, key: str) -> dict[str, Any]:
    if not ctx.operator_private:
        return {
            "title": "performance detail",
            "sections": [],
            "paths": [],
            "notice": "performance detail requires operator-private mode",
        }
    newest, modeled, gsim, _ = collect(ctx)
    record, latest, sim = modeled.get(key), newest.get(key), gsim.get(key)
    if not (record or latest or sim):
        return {
            "title": f"performance {clean(key, 80)}",
            "sections": [],
            "paths": [],
            "notice": "no performance record",
        }

    def describe(r):
        if not r:
            return ["not recorded"]
        return [
            f"model estimate {r['cycles']:,} cycles" if r.get("cycles") else "model estimate not bound",
            f"status {r.get('model_status')}  timing {r.get('timing_mode')}  binding {r.get('binding')}",
            f"structural floor {r.get('floor')}  efficiency {r.get('efficiency')}",
            f"useful MACs {r.get('useful_macs')}  peak/cycle {r.get('peak_macs_per_cycle')}"
            f"  requested bytes {r.get('requested_bytes')}",
            f"source {r.get('source_sha256')}",
        ] + [f"modeled {x['name']}: {x['activity_pct']}% ({x['basis']})" for x in r.get("resources") or []]

    sections = [
        {"title": "Newest record", "lines": describe(latest)},
        {"title": "Newest bound model estimate", "lines": describe(record)},
        {
            "title": "gSIM (L3 barrier)",
            "lines": [f"{sim['cycles']:,} cycles  source {sim['source_sha256']}"] if sim else ["not recorded"],
        },
    ]
    paths = sorted({p for p in ((latest or {}).get("path"), (record or {}).get("path"), (sim or {}).get("path")) if p})
    return {"title": f"performance {key}", "sections": sections, "paths": paths}


__all__ = ["collect", "inspect", "read"]
