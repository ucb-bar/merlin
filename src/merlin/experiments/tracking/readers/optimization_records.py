"""Read stored optimization runs for dashboards and terminal summaries.

Candidate, measurement and liveness fields come from recorded observations.
Derived summaries use arithmetic over those fields; missing or foreign records
remain unavailable. Reading a run never measures, grades or builds it.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from merlin.experiments.tracking.readers.records import Inventory, StoreCache

# ---- records -----------------------------------------------------------------------------------


def phase2(
    run_dir: Path,
    inventory: Inventory,
    *,
    now: float,
    stall_hours: float,
    store: Path | None = None,
    cache: StoreCache | None = None,
    target: str | None = None,
) -> dict[str, Any]:
    """A phase-2 measured run: its candidates over time, failures, plateau, liveness, board and lineage."""
    from merlin.experiments.tracking.readers.backend_generation_records import _phase2_names
    from merlin.experiments.tracking.readers.records import (
        FAILURE_CLASSES,
        MEASURED,
        OPEN,
        RELAUNCHED,
        STALLED,
        STOPPED,
        SUPERSEDED,
        UNKNOWN,
        StoreCache,
        _age_state,
        _index_rows,
        _int,
        _lineage_chain,
        _relaunched_as,
        _text,
        epoch,
        load_store,
        run_started,
    )

    names = _phase2_names()
    R, L, S, J = names["R"], names["L"], names["S"], names["J"]
    cache = cache or StoreCache()
    run_dir = Path(run_dir)
    run = inventory.json("run.json", run_dir / "run.json", schema=R.RUN_SCHEMA)
    seed = inventory.json("resumed_seed.json", run_dir / "resumed_seed.json", schema=R.RESUMED_SEED_SCHEMA)
    config = inventory.json(R.CONFIG_NAME, run_dir / R.CONFIG_NAME)
    ledger = inventory.jsonl(L.ITERATIONS, run_dir / L.ITERATIONS)
    stage = run_dir / "stage"
    sessions_doc = inventory.json("stage/sessions.json", stage / "sessions.json", schema=S.SEQUENCE_SCHEMA)
    breaker = inventory.json("stage/infra_circuit_breaker.json", stage / "infra_circuit_breaker.json")
    host = inventory.json("stage/host_resource_telemetry.json", stage / "host_resource_telemetry.json")
    rounds = []
    rounds_dir = stage / "rounds"
    for path in sorted(rounds_dir.glob("round_*.round.json")) if rounds_dir.is_dir() else ():
        document = inventory.json(f"stage/rounds/{path.name}", path)
        if document is not None:
            rounds.append(
                {
                    "round": document.get("round"),
                    "status": document.get("status"),
                    "why": _text(document.get("why"), 200),
                    "wall_seconds": document.get("wall_seconds"),
                    "requested": len(document.get("requested") or ()),
                    "final_package": document.get("final_package_sha256"),
                    "final_timing_status": document.get("final_timing_status"),
                }
            )

    store_roots = (seed or {}).get("store_roots") if isinstance((seed or {}).get("store_roots"), Mapping) else {}
    store_root = (
        Path(store) if store is not None else (Path(store_roots["screen"]) if store_roots.get("screen") else None)
    )
    store_source = "--store" if store is not None else ("resumed_seed.json store_roots.screen" if store_root else None)
    store_doc = None
    if store_root is not None:
        if store_root.is_dir():
            store_doc = load_store(store_root, inventory, cache)
        else:
            inventory.note("measurement store", store_root, "absent")
    else:
        inventory.note("measurement store", run_dir / "resumed_seed.json", "absent", "no store_roots.screen recorded")

    screen_section = (config or {}).get("screen") if isinstance((config or {}).get("screen"), Mapping) else {}
    bar_path = screen_section.get("reference")
    bar_doc = cache.result(str(bar_path), inventory, "screen reference result.json") if bar_path else None
    from merlin.analysis.whole_model.grading import verdict as V

    bar = None
    if bar_doc is not None:
        bar = {
            "cycles": _int((bar_doc.get("verdict") or {}).get("whole_window_cycles")),
            "timing_status": bar_doc.get("timing_status"),
            "label": _text(bar_doc.get("label"), 120),
            "admissible_as_bar": bar_doc.get("timing_status") == V.TIMING_MEASURED,
        }
    orientation = [
        {
            "label": _text(row.get("label"), 160),
            "cycles": _int(row.get("whole_model_cycles")),
            "note": _text(row.get("note"), 200),
        }
        for row in (config or {}).get("orientation") or ()
        if isinstance(row, Mapping) and _int(row.get("whole_model_cycles"))
    ]

    started = run_started(run_dir, inventory)
    candidates = [dict(c) for c in (store_doc or {}).get("candidates") or ()]
    for row in candidates:
        row["in_run"] = None if started is None or row["requested"] is None else row["requested"] >= started
    measured = [c for c in candidates if c["class"] == MEASURED and c["cycles"] is not None]
    eligible = [c for c in measured if c["replicate"] == 0 and c["attributable"] and c["role"] != J.ROLE_REFERENCE]
    best_line, lowest = [], None
    for row in sorted(eligible, key=lambda c: c["finished"] or c["requested"] or 0.0):
        if lowest is None or row["cycles"] < lowest["cycles"]:
            lowest = row
        best_line.append({"at": row["finished"] or row["requested"], "cycles": lowest["cycles"], "key": lowest["key"]})
    counts = {name: 0 for name in (MEASURED, *FAILURE_CLASSES, OPEN, SUPERSEDED)}
    counts_in_run = dict(counts)
    for row in candidates:
        counts[row["class"]] = counts.get(row["class"], 0) + 1
        if row["in_run"]:
            counts_in_run[row["class"]] = counts_in_run.get(row["class"], 0) + 1

    landed = [
        c["finished"]
        for c in candidates
        if c["class"] in (MEASURED, "correctness")
        and c["whole_window"] is not None
        and c["finished"] is not None
        and (started is None or c["finished"] >= started)
    ]
    last_measured = max(landed, default=None)
    last_measured_store = max(
        (
            c["finished"]
            for c in candidates
            if c["class"] in (MEASURED, "correctness") and c["whole_window"] is not None and c["finished"]
        ),
        default=None,
    )
    stopped = (sessions_doc or {}).get("stopped") if isinstance((sessions_doc or {}).get("stopped"), Mapping) else None
    relaunched = _relaunched_as(run_dir, R.RUN_SCHEMA)
    if stopped:
        liveness = {
            "state": STOPPED,
            "hours": None,
            "detail": f"{stopped.get('kind')}: {_text(stopped.get('reason'), 240)}",
        }
    elif relaunched:
        liveness = {"state": RELAUNCHED, "hours": None, "detail": f"resumed as {Path(relaunched).name}"}
    else:
        what = (
            "the last measured candidate"
            if last_measured is not None
            else "the run start (no candidate measured since)"
        )
        state, hours, detail = _age_state(
            last_measured if last_measured is not None else started, now, stall_hours, what
        )
        if store_doc is None:
            # No store: nothing was ever requested from it, or it lives where the run does not record.
            state = STALLED if state == STALLED else UNKNOWN
            detail = f"no measurement store found; {detail}"
        liveness = {"state": state, "hours": hours, "detail": detail}
    liveness.update(
        last_measured=last_measured,
        last_measured_store=last_measured_store,
        run_started=started,
        stall_hours=stall_hours,
    )

    board = dict((store_doc or {}).get("board") or {})
    if store_doc is not None:
        waiting = [c for c in candidates if c["state"] == J.BOARD]
        board.update(
            queue=len(waiting),
            oldest_waiting=min((c["board_ready"] for c in waiting if c["board_ready"]), default=None),
            running=sum(1 for c in candidates if c["state"] == J.RUNNING),
            pending=sum(1 for c in candidates if c["state"] in (J.PENDING, J.SCREENING)),
        )

    ledger_rows = ledger or []
    ledger_view = None
    if ledger is not None:
        ledger_view = {
            "candidates": sum(1 for r in ledger_rows if r.get("kind") == "candidate"),
            "measured": [
                {
                    "n": r.get("n"),
                    "tag": r.get("tag"),
                    "timing_status": r.get("timing_status"),
                    "cycles": _int(r.get("objective_cycles")),
                    "at": epoch(r.get("at")),
                }
                for r in ledger_rows
                if r.get("kind") == "measured"
            ],
            "best": [
                {"n": r.get("n"), "package": r.get("package_sha256"), "at": epoch(r.get("at"))}
                for r in ledger_rows
                if r.get("kind") == "best"
            ],
        }
    target = target or (run or {}).get("target")
    roofline_best = (store_doc or {}).get("roofline_best")
    by_kind: dict[str, dict[str, Any]] = {}
    for gap in (roofline_best or {}).get("gaps") or ():
        entry = by_kind.setdefault(
            str(gap.get("kind")), {"kind": str(gap.get("kind")), "groups": 0, "ours": 0, "roofline": 0}
        )
        entry["groups"] += 1
        entry["ours"] += int(gap["ours"])
        entry["roofline"] += int(gap["roofline"])
    return {
        "run": {
            "target": (run or {}).get("target"),
            "method": (run or {}).get("method"),
            "prohibited_roles": list((run or {}).get("prohibited_instruction_roles") or []) if run else None,
            "config_sha256": (run or {}).get("config_sha256"),
            "resumed_from": (run or {}).get("resumed_from_run") or (seed or {}).get("resumed_from_run"),
            "started": started,
        },
        "store": {
            "root": str(store_root) if store_root else None,
            "source": store_source,
            "loaded": store_doc is not None,
        },
        "config": {
            "plateau_hours": (config or {}).get("plateau_hours"),
            "plateau_min_sessions": (config or {}).get("plateau_min_sessions"),
            "stop_at_bar": (config or {}).get("stop_at_bar"),
        }
        if config is not None
        else None,
        "bar": bar,
        "orientation": orientation,
        "candidates": candidates,
        "best_line": best_line,
        "best": lowest,
        "counts": counts,
        "counts_in_run": counts_in_run if started is not None else None,
        "roofline": {
            "key": roofline_best["key"],
            "gaps": roofline_best["gaps"][:20],
            "by_kind": sorted(by_kind.values(), key=lambda e: -(e["ours"] - e["roofline"])),
        }
        if roofline_best
        else None,
        "plateau": (store_doc or {}).get("plateau"),
        "board": board if store_doc is not None else None,
        "sessions": {
            "driver": sessions_doc.get("driver"),
            "model": sessions_doc.get("model"),
            "rows": list(sessions_doc.get("sessions") or []),
            "stopped": stopped,
        }
        if sessions_doc is not None
        else None,
        "rounds": rounds or None,
        "circuit_breaker": {"at": breaker.get("at"), "reason": _text(breaker.get("reason"), 300)} if breaker else None,
        "host_resource": {"status": host.get("status")} if host else None,
        "ledger": ledger_view,
        "lineage": {
            "seed_package": (seed or {}).get("seed_package_sha256"),
            "lineage_kind": (seed or {}).get("lineage_kind"),
            "origin_kind": (seed or {}).get("origin_kind"),
            "frozen": (seed or {}).get("frozen"),
            "why": _text((seed or {}).get("why"), 600),
            "imported": bool((seed or {}).get("imported")),
            "chain": _lineage_chain(seed),
            "index_rows": _index_rows(target, run_dir, inventory),
        }
        if seed is not None
        else None,
        "liveness": liveness,
    }
