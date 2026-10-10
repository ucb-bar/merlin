"""Host load samples and a monitor's notes: two operator-side records a run view can sit beside.

``--load PATH`` reads a tab-separated sample log with a header row (``utc  load1  cpu_busy_pct  gsim
spike  verilator  codex``, one row a minute); ``--monitor PATH`` reads a monitor's Markdown log, one
``## <utc>`` heading per check, whose first ``STATUS:`` line carries OK / WATCH / STUCK.  Both are read
as written: an absent file is "not recorded", a malformed row is counted and skipped, nothing is
inferred.  Neither file belongs to a run directory, and neither is ever written here.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from . import charts as C
from . import records as R
from .html import NR, details, esc, mono, table, tile, when

#: Columns of the sample log that count processes (each is drawn as its own line on one scale).
PROCESS_COLUMNS = ("gsim", "spike", "verilator", "codex")
MONITOR_STATES = ("OK", "WATCH", "STUCK")


def _number(text: str) -> float | None:
    try:
        return float(text)
    except ValueError:
        return None


def load_samples(path: Path | None, inventory: R.Inventory) -> dict[str, Any] | None:
    """The sample log's rows as ``{"columns", "rows": [{"at", <column>: float | None}], "bad"}``."""
    if path is None:
        return None
    text = inventory._text("load samples (TSV)", Path(path))
    if text is None:
        return None
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        inventory.note("load samples (TSV)", Path(path), "unreadable", "empty")
        return None
    header = [h.strip() for h in lines[0].split("\t")]
    if "utc" not in header:
        inventory.note("load samples (TSV)", Path(path), "unexpected schema", f"header {header!r} has no utc")
        return None
    rows, bad = [], 0
    for line in lines[1:]:
        cells = line.split("\t")
        if len(cells) != len(header):
            bad += 1
            continue
        record = dict(zip(header, cells, strict=True))
        at = R.epoch(record.pop("utc"))
        if at is None:
            bad += 1
            continue
        rows.append({"at": at, **{k: _number(v) for k, v in record.items()}})
    inventory.note(
        "load samples (TSV)", Path(path), "read", f"{len(rows)} rows" + (f", {bad} malformed" if bad else "")
    )
    return {"path": str(path), "columns": [h for h in header if h != "utc"], "rows": rows, "bad": bad}


def monitor_notes(path: Path | None, inventory: R.Inventory) -> dict[str, Any] | None:
    """A monitor log's checks, oldest first: ``{"at", "status", "lines"}`` per ``## <utc>`` heading."""
    if path is None:
        return None
    text = inventory._text("monitor notes", Path(path))
    if text is None:
        return None
    notes: list[dict[str, Any]] = []
    for line in text.splitlines():
        if line.startswith("## "):
            notes.append({"heading": line[3:].strip(), "at": R.epoch(line[3:].strip()), "status": None, "lines": []})
            continue
        if not notes:
            continue
        note = notes[-1]
        stripped = line.strip()
        if note["status"] is None and stripped.startswith("STATUS:"):
            word = stripped.partition(":")[2].strip().split(" ")[0].strip("*").upper() if stripped else ""
            note["status"] = word if word in MONITOR_STATES else (word or None)
            continue
        if stripped:
            note["lines"].append(stripped)
    inventory.note("monitor notes", Path(path), "read", f"{len(notes)} checks")
    counts = {state: sum(1 for n in notes if n["status"] == state) for state in MONITOR_STATES}
    return {"path": str(path), "notes": notes, "counts": counts}


# --------------------------------------------------------------------------- rendering
_MONITOR_BADGE = {"OK": "good", "WATCH": "warning", "STUCK": "critical"}


def resource_section(load: Mapping[str, Any] | None, now: float, *, requested: bool) -> str:
    if not requested:
        return ""
    out = '<section id="resources"><h2>Host resources</h2>'
    if not load or not load.get("rows"):
        return out + f"<p>Load samples: {NR}.</p></section>"
    rows = load["rows"]
    latest = rows[-1]
    busy = [
        (r["at"], r["cpu_busy_pct"], f"CPU busy {r['cpu_busy_pct']:g}% at {R.stamp(r['at'])}")
        for r in rows
        if r.get("cpu_busy_pct") is not None
    ]
    load1 = [
        (r["at"], r["load1"], f"load1 {r['load1']:g} at {R.stamp(r['at'])}") for r in rows if r.get("load1") is not None
    ]
    procs = {
        column: [
            (r["at"], r[column], f"{column} processes {r[column]:g} at {R.stamp(r['at'])}")
            for r in rows
            if r.get(column) is not None
        ]
        for column in PROCESS_COLUMNS
        if column in load["columns"]
    }

    def latest_value(column: str) -> str:
        value = latest.get(column)
        return f"{value:g}" if isinstance(value, float) else NR

    tiles = [
        tile("Last sample", when(latest["at"]), f"{len(rows)} samples"),
        tile("CPU busy", latest_value("cpu_busy_pct") + ("%" if latest.get("cpu_busy_pct") is not None else ""), ""),
        tile("load1", latest_value("load1"), ""),
    ] + [tile(f"{c} processes", latest_value(c), "") for c in procs]
    out += '<div class="tiles">' + "".join(tiles) + "</div>"
    out += "<h3>CPU busy (%)</h3>" + C.time_series({"CPU busy %": busy}, label="CPU busy", y_max=100, now=now)
    out += "<h3>1-minute load average</h3>" + C.time_series({"load1": load1}, label="load average", now=now)
    out += "<h3>Simulator and agent processes</h3>" + C.time_series(
        procs, label="process counts", step=True, now=now, y_fmt=lambda v: f"{v:g}"
    )
    out += f'<p class="sub">From {mono(load["path"])}' + (
        f"; {load['bad']} malformed rows skipped" if load["bad"] else ""
    )
    return out + ".</p></section>"


def monitor_section(monitor: Mapping[str, Any] | None, now: float, *, requested: bool) -> str:
    if not requested:
        return ""
    out = '<section id="monitor"><h2>Monitor notes</h2>'
    if not monitor or not monitor.get("notes"):
        return out + f"<p>Monitor notes: {NR}.</p></section>"
    notes = monitor["notes"]
    latest = notes[-1]
    counts = monitor["counts"]
    out += (
        f"<p>Latest check {esc(latest['heading'])}: "
        f'<span class="badge {_MONITOR_BADGE.get(latest["status"] or "", "")}">'
        f"{esc(latest['status'] or 'no STATUS line')}</span> &middot; {len(notes)} checks: "
        + ", ".join(f"{counts[s]} {s}" for s in MONITOR_STATES)
        + "</p>"
        + f'<pre class="note">{esc(chr(10).join(latest["lines"]))}</pre>'
    )
    lanes = [
        {
            "lane": "monitor STATUS",
            "points": [
                {
                    "at": n["at"],
                    "kind": n["status"] or "none",
                    "tip": f"{n['heading']} {n['status']}\n" + "\n".join(n["lines"][:6]),
                }
                for n in notes
            ],
        }
    ]
    colours = {"OK": "var(--good)", "WATCH": "var(--warning)", "STUCK": "var(--critical)", "none": "var(--muted)"}
    out += C.gantt(lanes, now=now, colours=colours, label="monitor checks")
    rows = [
        [
            esc(n["heading"]),
            esc(n["status"] or "not recorded"),
            f'<pre class="note">{esc(chr(10).join(n["lines"]))}</pre>',
        ]
        for n in reversed(notes[-60:])
    ]
    out += details(f"All checks, newest first ({len(rows)} shown)", table(["at", "status", "note"], rows))
    return out + f'<p class="sub">From {mono(monitor["path"])}.</p></section>'


__all__ = ["load_samples", "monitor_notes", "monitor_section", "resource_section"]
