"""Compiler-construction stages of one experiment, from a stage ledger or derived from existing records.

The stage list, and how each stage is tied to evidence, is the reviewed contract
``merlin/contract/compiler_stages.yaml`` (resolved through ``merlin.common.paths.contract_dir``).
When the run wrote a ``merlin.compiler_stage_ledger.v1`` at one of the contract's ``ledger_locations``
every stage comes from it (basis ``ledger``). Otherwise each stage is derived (basis ``derived``) from:

* construction actions -- ``<workspace>/construction_study/events.jsonl`` (``merlin.construction_actions.v1``
  state changes), grouped by ``boundary`` onto the stage declaring it;
* capsules gated -- capsules whose ``pass_requirements`` carry a token the stage declares, and how many of
  them pass (from the ``capsules`` section, else the staged ``capsule.yaml`` files, pass state unknown);
* records -- fusion exploration (``fusion_public_results``), generated ISA packages
  (``merlin.isa_generation.v1`` manifests), the shared pass contract families (``passes`` section) and
  the performance summary (``perf`` section).

Derived state rules, first match wins:

* ``active``  -- an action is running, or (live run) the stage's evidence changed in the last 15 minutes;
* ``blocked`` -- every action failed, or capsules are gated, none passes and a later stage shows progress;
* ``done``    -- gated capsules all pass, or all actions completed with no capsule gated and none failing;
* ``pending`` -- evidence exists but the stage is not done;
* ``unknown`` -- no evidence at all (never shown as a failure or a zero).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..io import clean, is_file, mtime, num, obj, rows, scan_dir
from . import STAGE_STATES

LEDGER_SCHEMA = "merlin.compiler_stage_ledger.v1"
ISA_SCHEMA = "merlin.isa_generation.v1"
RECENT_S = 15 * 60
PASSING = ("pass", "passed")


def _ns(value: Any) -> float | None:
    """A recorded ``*_ns`` field as epoch seconds (the field name, not the magnitude, says nanoseconds)."""
    value = num(value)
    return value / 1e9 if value is not None and value > 0 else None


def contract(ctx) -> tuple[dict[str, Any] | None, str | None]:
    try:
        from merlin.common.paths import contract_dir

        path = contract_dir() / "compiler_stages.yaml"
    except Exception:  # noqa: BLE001 -- a console without an importable core still renders
        return None, None
    doc = ctx.reader.yaml(path, limit=1 << 20)
    return (doc if isinstance(doc, dict) and isinstance(doc.get("stages"), list) else None), str(path)


def _entries(doc: dict) -> list[dict[str, Any]]:
    out = []
    for kind in ("stages", "lanes"):
        for row in rows(doc.get(kind)):
            if isinstance(row.get("id"), str):
                out.append({**row, "lane": kind == "lanes"})
    return out


def _ledger(ctx, doc: dict) -> tuple[dict | None, str | None]:
    roots = {"workspace": ctx.experiment.get("workspace"), "run_dir": ctx.experiment.get("run_dir")}
    for location in rows(doc.get("ledger_locations")):
        for root_key, relative in location.items():
            root = roots.get(root_key)
            if root and isinstance(relative, str):
                path = Path(root) / relative
                ledger = ctx.reader.document(path, limit=4 << 20)
                if ledger and ledger.get("schema") == LEDGER_SCHEMA:
                    return ledger, str(path)
    return None, None


def _actions(ctx) -> dict[str, dict[str, Any]]:
    """Latest state of every construction action, incrementally tailed."""
    workspace = ctx.experiment.get("workspace")
    store = ctx.state.setdefault("actions", {})
    if not workspace:
        return store
    path = Path(workspace) / "construction_study" / "events.jsonl"

    def consume(row):
        action = row.get("action_id")
        if not isinstance(action, str):
            return
        entry = store.setdefault(
            action,
            {
                "id": action,
                "boundary": row.get("boundary"),
                "capability": row.get("capability"),
                "first": None,
                "last": None,
            },
        )
        t = _ns(row.get("time_ns"))
        entry["first"] = entry["first"] if entry["first"] is not None else t
        entry["last"] = t if t is not None else entry["last"]
        entry["state"] = clean(row.get("state"), 24)
        entry["returncode"] = row.get("returncode")
        entry["boundary"] = row.get("boundary") or entry["boundary"]

    ctx.reader.poll(path, consume, on_reset=store.clear)
    return store


def _capsule_rows(ctx) -> list[dict[str, Any]]:
    section = obj(ctx.shared.get("capsules"))
    found = [r for r in rows(section.get("capsules")) if isinstance(r.get("pass_requirements"), list)]
    if found:
        return found
    # Fallback: the staged definitions themselves; pass state is then unknown, never assumed.
    corpus = Path(ctx.experiment["run_dir"]) / "private_corpus_input"
    out = []
    for parent in (corpus / "public", corpus / "scalability" / "cases"):
        for entry in scan_dir(parent, 1024):
            doc = obj(ctx.reader.yaml(Path(entry.path) / "capsule.yaml", limit=1 << 20))
            if doc:
                out.append(
                    {"name": doc.get("name"), "pass_requirements": doc.get("pass_requirements") or [], "status": None}
                )
    return out


def _fusion(ctx) -> tuple[dict[str, Any], float | None]:
    workspace = ctx.experiment.get("workspace")
    if not workspace:
        return {}, None
    root = Path(workspace) / "fusion_public_results"
    summary = ctx.reader.document(root / "summary.json", limit=4 << 20)
    if summary is None:
        return {}, None
    patterns = rows(summary.get("patterns"))
    candidates, matched, unsupported = 0, 0, 0
    for pattern in patterns:
        statuses = pattern.get("statuses") or []
        unsupported += sum(1 for s in statuses if s == "unsupported")
        exploration = ctx.reader.json(root / clean(pattern.get("name"), 80) / "exploration.json", limit=16 << 20)
        for round_ in exploration if isinstance(exploration, list) else []:
            routes = obj(obj(obj(round_).get("page")).get("routes"))
            for route in routes.values():
                count = num(obj(route).get("candidates")) or 0
                candidates += count
                matched += 1 if count else 0
    return (
        {
            "fusion_patterns": len(patterns),
            "fusion_candidates": candidates,
            "fusion_routes_matched": matched,
            "fusion_unsupported": unsupported,
        },
        mtime(root / "summary.json"),
    )


def _isa(ctx) -> tuple[dict[str, Any], float | None]:
    """Generated ISA packages under the workspace (bounded walk, hidden directories skipped)."""
    workspace = ctx.experiment.get("workspace")
    if not workspace:
        return {}, None
    cached = ctx.state.get("isa")
    if cached and ctx.now - cached[0] < 60:
        return cached[1], cached[2]
    packages, instructions, latest = 0, None, None
    frontier, visited = [(Path(workspace), 0)], 0
    while frontier and visited < 600:
        directory, depth = frontier.pop()
        visited += 1
        for entry in scan_dir(directory, 2048):
            if entry.name.startswith(".") or entry.name == "__pycache__":
                continue
            if entry.name == "manifest.json" and depth > 0:
                manifest = ctx.reader.document(Path(entry.path), limit=1 << 20)
                if manifest and manifest.get("schema") == ISA_SCHEMA:
                    packages += 1
                    latest = max(latest or 0, mtime(entry.path) or 0)
                    resolved = ctx.reader.document(Path(entry.path).parent / "isa-resolved.json", limit=8 << 20)
                    count = len(obj(resolved).get("instructions") or []) if resolved else None
                    if count is not None:
                        instructions = max(instructions or 0, count)
            elif depth < 3 and entry.is_dir(follow_symlinks=False):
                frontier.append((Path(entry.path), depth + 1))
    metrics = {"generated_packages": packages, "instructions": instructions} if packages else {}
    ctx.state["isa"] = (ctx.now, metrics, latest)
    return metrics, latest


def _agent_tokens(ctx, stage: dict, ledger_threads: list) -> int | None:
    agents = rows(obj(ctx.shared.get("agents")).get("agents"))
    paths = set(stage.get("agent_paths") or [])
    total, seen = 0, False
    for agent in agents:
        if agent.get("stage") == stage["id"] or agent.get("path") in paths or agent.get("id") in ledger_threads:
            value = num(obj(agent.get("tokens")).get("total"))
            if value is not None:
                total, seen = total + value, True
    return int(total) if seen else None


def _derive_state(actions: dict, capsules: dict, evidence: bool, recent: bool, later_progress: bool) -> str:
    if actions["running"] or recent:
        return "active"
    if actions["total"] and actions["failed"] == actions["total"]:
        return "blocked"
    required, passing = capsules["required"], capsules["passing"]
    if required and passing == 0 and later_progress:
        return "blocked"
    if required and passing is not None and passing >= required:
        return "done"
    if actions["total"] and actions["done"] == actions["total"] and not required:
        return "done"
    return "pending" if evidence else "unknown"


def read(ctx) -> dict[str, Any]:
    doc, contract_path = contract(ctx)
    if doc is None:
        return {"stages": [], "contract": contract_path, "notes": ["Stage contract compiler_stages.yaml unavailable."]}
    entries = _entries(doc)
    ledger, ledger_path = _ledger(ctx, doc)
    if ledger is not None:
        by_id = {r.get("stage"): r for r in rows(ledger.get("stages"))}
        stages = []
        for entry in entries:
            row = obj(by_id.get(entry["id"]))
            state = row.get("state") if row.get("state") in STAGE_STATES else "unknown"
            start, end = _ns(row.get("started_ns")), _ns(row.get("finished_ns"))
            actions = {k: num(obj(row.get("actions")).get(k)) for k in ("done", "failed", "running", "total")}
            gated = row.get("capsules_gated") if isinstance(row.get("capsules_gated"), list) else []
            stages.append(
                {
                    "id": entry["id"],
                    "title": entry.get("title") or entry["id"],
                    "lane": entry["lane"],
                    "state": state,
                    "basis": "ledger",
                    "actions": actions,
                    "capsules": {"required": len(gated) if gated else None, "passing": None},
                    "wall_s": (end or ctx.now) - start if start else None,
                    "tokens": _agent_tokens(ctx, entry, row.get("agent_thread_ids") or []),
                    "metrics": {k: v for k, v in obj(row.get("metrics")).items() if isinstance(v, int | float | str)},
                    "items": [clean(row.get("note"), 160)] if row.get("note") else [],
                }
            )
        return {
            "stages": stages,
            "contract": contract_path,
            "ledger": ledger_path,
            "notes": ["Stages from the run's own stage ledger."],
        }

    boundary_stage = {b: e["id"] for e in entries for b in e.get("construction_boundaries") or []}
    token_stage = {t: e["id"] for e in entries for t in e.get("pass_requirement_tokens") or []}
    actions = _actions(ctx)
    capsules = _capsule_rows(ctx)
    fusion, fusion_at = _fusion(ctx)
    isa, isa_at = _isa(ctx)
    passes = rows(obj(ctx.shared.get("passes")).get("passes"))
    perf = obj(obj(ctx.shared.get("perf")).get("summary"))
    unmapped = sorted({str(a.get("boundary")) for a in actions.values() if a.get("boundary") not in boundary_stage})

    built = []
    for entry in entries:
        sid = entry["id"]
        mine = [a for a in actions.values() if boundary_stage.get(a.get("boundary")) == sid]
        counts = {
            "done": sum(a.get("state") == "completed" for a in mine),
            "failed": sum(
                a.get("state") == "failed" or (type(a.get("returncode")) is int and a["returncode"] != 0) for a in mine
            ),
            "running": sum(a.get("state") == "running" for a in mine),
            "total": len(mine),
        }
        gated = [c for c in capsules if any(token_stage.get(t) == sid for t in c.get("pass_requirements") or [])]
        known = [c for c in gated if c.get("status") is not None]
        passing = sum(1 for c in known if c.get("status") in PASSING) if known else None
        metrics: dict[str, Any] = {}
        items, evidence_times = [], [a["last"] for a in mine if a.get("last")]
        if gated:
            metrics.update(capsules_required=len(gated), capsules_passing=passing)
        if mine:
            metrics["actions"] = len(mine)
            items += [f"{a.get('state')}: {clean(a.get('id'), 40)} ({clean(a.get('capability'), 40)})" for a in mine][
                :4
            ]
        kinds = set(entry.get("record_kinds") or [])
        if kinds & {"merlin.fusion_workflow.v1"} and fusion:
            metrics.update(fusion)
            evidence_times.append(fusion_at)
        if kinds & {ISA_SCHEMA} and isa:
            metrics.update(isa)
            evidence_times.append(isa_at)
        families = [p for p in passes if p.get("kind") == "family" and boundary_stage.get(p.get("boundary")) == sid]
        if kinds & {"merlin.shared_pass_contract.v2", "host_pass_profile"} and passes:
            metrics.update(
                pass_families_declared=sum(1 for p in passes if p.get("declared")),
                pass_families_used=sum(1 for p in passes if p.get("used")),
                passes_profiled=sum(1 for p in passes if p.get("profile_wall_s") is not None),
            )
        if families:
            metrics["pass_families"] = len(families)
        if kinds & {"merlin.phase1.joint_performance.v1"} and perf:
            metrics.update(
                {
                    k: perf.get(src)
                    for k, src in (
                        ("modeled_capsules", "modeled"),
                        ("provisional", "provisional"),
                        ("geo_efficiency_pct", "geo_eff"),
                        ("worst_efficiency_pct", "worst_eff"),
                    )
                    if perf.get(src) is not None
                }
            )
        times = [t for t in evidence_times if t]
        start = min((a["first"] for a in mine if a.get("first")), default=None)
        end = max((a["last"] for a in mine if a.get("last")), default=None)
        evidence = bool(mine or gated or len(metrics) > (1 if gated else 0))
        built.append(
            {
                "id": sid,
                "title": entry.get("title") or sid,
                "lane": entry["lane"],
                "basis": "derived",
                "actions": counts,
                "capsules": {"required": len(gated) if gated else None, "passing": passing},
                "wall_s": (end - start) if start and end and end > start else None,
                "tokens": _agent_tokens(ctx, entry, []),
                "metrics": metrics,
                "items": items,
                "_evidence": evidence,
                "_recent": bool(ctx.live and times and ctx.now - max(times) < RECENT_S),
            }
        )
    ordered = [s for s in built if not s["lane"]]
    for stage in built:
        later = ordered[ordered.index(stage) + 1 :] if stage in ordered else []
        later_progress = any(s["actions"]["done"] or (s["capsules"]["passing"] or 0) > 0 for s in later)
        stage["state"] = _derive_state(
            stage["actions"], stage["capsules"], stage.pop("_evidence"), stage.pop("_recent"), later_progress
        )
    notes = ["Derived from existing records; no stage ledger was written by this run."]
    if any(c.get("status") is None for c in capsules) and capsules:
        notes.append("Capsule pass state unknown for capsules read from staged definitions.")
    if unmapped:
        notes.append("Construction boundaries not in the contract: " + ", ".join(unmapped[:6]))
    return {"stages": built, "contract": contract_path, "notes": notes}


def inspect(ctx, key: str) -> dict[str, Any]:
    section = read(ctx)
    stage = next((s for s in section["stages"] if s["id"] == key), None)
    if stage is None:
        return {"title": f"stage {key}", "sections": [], "paths": [], "notice": "unknown stage"}
    doc, path = contract(ctx)
    entry = next((e for e in _entries(doc or {}) if e["id"] == key), {})
    lines = [f"state {stage['state']} ({stage['basis']})", clean(entry.get("description"), 200)]
    lines += [f"{k}: {v}" for k, v in stage["metrics"].items()]
    evidence = [
        f"boundaries: {', '.join(entry.get('construction_boundaries') or []) or '-'}",
        f"pass requirement tokens: {', '.join(entry.get('pass_requirement_tokens') or []) or '-'}",
        f"record kinds: {', '.join(entry.get('record_kinds') or []) or '-'}",
        f"pipeline commands: {', '.join(entry.get('pipeline_commands') or []) or '-'}",
    ]
    actions = [
        f"{a.get('state')}  {clean(k, 40)}  {clean(a.get('capability'), 60)}"
        for k, a in _actions(ctx).items()
        if a.get("boundary") in (entry.get("construction_boundaries") or [])
    ]
    workspace = ctx.experiment.get("workspace")
    paths = [
        p
        for p in (
            path,
            section.get("ledger"),
            str(Path(workspace) / "construction_study" / "events.jsonl") if workspace else None,
        )
        if p
    ]
    return {
        "title": f"stage {stage['title']}",
        "sections": [
            {"title": "status", "lines": [x for x in lines if x]},
            {"title": "evidence declared", "lines": evidence},
            {"title": "construction actions", "lines": actions or ["none recorded"]},
            {"title": "recent items", "lines": stage["items"] or ["none"]},
        ],
        "paths": [p for p in paths if is_file(p)],
    }


__all__ = ["contract", "inspect", "read"]
