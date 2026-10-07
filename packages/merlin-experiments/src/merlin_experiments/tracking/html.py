"""Render a tracking summary as ONE self-contained HTML page: inline CSS, inline SVG, a few lines of JS.

No network, no CDN, no external file: the page opens from disk or through an ssh port-forward, and a
copy mailed to someone renders the same.  Charts are SVG drawn here from the summary's numbers; the
only script shows a hover tooltip, so the page is complete with scripts disabled (every chart also
has a table view).  A value the records do not hold is printed as "not recorded" -- never a zero.
"""

from __future__ import annotations

import html
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from . import records as R

NR = '<span class="nr">not recorded</span>'

_STATE_BADGE = {
    R.LIVE: "good",
    R.FINISHED: "good",
    R.STALLED: "critical",
    R.STOPPED: "warning",
    R.ENDED: "serious",
    R.RELAUNCHED: "neutral",
    R.UNKNOWN: "neutral",
}

_DARK = """color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;--surface-2:#262624;--ink:#fff;--ink-2:#c3c2b7;
--grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);--s1:#3987e5;--s2:#d95926;
--band:rgba(57,135,229,.12);--t0:#184f95;--t1:#1c5cab;--t2:#256abf;--t3:#3987e5;--t4:#5598e7;--t5:#86b6ef;
--t6:#b7d3f6;--fail:#8a3434"""

CSS = (
    """
:root{color-scheme:light;--page:#f9f9f7;--surface:#fcfcfb;--surface-2:#f0efec;--ink:#0b0b0b;--ink-2:#52514e;
--muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;--border:rgba(11,11,11,.10);--s1:#2a78d6;--s2:#eb6834;
--band:rgba(42,120,214,.07);--good:#0ca30c;--warning:#fab219;--serious:#ec835a;--critical:#d03b3b;
--t0:#86b6ef;--t1:#5598e7;--t2:#3987e5;--t3:#256abf;--t4:#1c5cab;--t5:#104281;--t6:#0d366b;--fail:#e8a3a3}
"""
    + "@media (prefers-color-scheme:dark){:root:not([data-theme=light]){"
    + _DARK
    + "}}\n:root[data-theme=dark]{"
    + _DARK
    + """}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--ink);
font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1180px;margin:0 auto;padding:16px}
header{display:flex;flex-wrap:wrap;gap:8px 16px;align-items:baseline;margin-bottom:8px}
h1{font-size:20px;margin:0;word-break:break-all}h2{font-size:17px;margin:28px 0 8px}
h3{font-size:14px;margin:18px 0 6px}
.sub{color:var(--ink-2)}.nr{color:var(--muted);font-style:italic}
.mono{font-family:ui-monospace,Menlo,monospace;font-size:12px}
nav{display:flex;flex-wrap:wrap;gap:4px 14px;margin:6px 0 12px}nav a{color:var(--s1);text-decoration:none}
section{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:12px 16px;margin:12px 0}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px}
.tile{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:10px 12px}
.tile .k{color:var(--ink-2);font-size:12px}.tile .v{font-size:20px;margin-top:2px;word-break:break-word}
.tile .d{color:var(--muted);font-size:12px}
.badge{display:inline-flex;gap:6px;align-items:center;padding:2px 10px;border-radius:999px;font-weight:600;
font-size:13px;border:1px solid var(--border);background:var(--surface)}
.badge::before{content:"";width:10px;height:10px;border-radius:50%;background:var(--muted)}
.badge.good::before{background:var(--good)}.badge.warning::before{background:var(--warning)}
.badge.serious::before{background:var(--serious)}.badge.critical::before{background:var(--critical)}
table{border-collapse:collapse;width:100%;font-size:12.5px;font-variant-numeric:tabular-nums}
th,td{text-align:left;padding:3px 8px;border-bottom:1px solid var(--grid);vertical-align:top}
th{color:var(--ink-2);font-weight:600}td.n,th.n{text-align:right}
.scroll{overflow-x:auto}details{margin:6px 0}summary{cursor:pointer;color:var(--ink-2)}
svg{display:block;width:100%;height:auto}svg text{fill:var(--ink-2);font-size:11px}
.grid line{stroke:var(--grid);stroke-width:1}.axis{stroke:var(--axis);stroke-width:1}
.legend{display:flex;flex-wrap:wrap;gap:4px 16px;font-size:12px;color:var(--ink-2);margin:4px 0}
.sw{display:inline-block;width:12px;height:12px;border-radius:3px;vertical-align:-2px;margin-right:5px}
.cell-pass{background:color-mix(in srgb,var(--good) 22%,transparent)}
.cell-fail{background:color-mix(in srgb,var(--critical) 22%,transparent)}
.cell-other{background:var(--surface-2)}
.chain{display:flex;flex-wrap:wrap;align-items:stretch;gap:6px;margin:6px 0}
.node{border:1px solid var(--border);border-radius:6px;padding:6px 10px;background:var(--surface);min-width:150px}
.node .k{font-size:11px;color:var(--ink-2)}.arrow{align-self:center;color:var(--muted)}
#tip{position:fixed;z-index:9;max-width:420px;padding:6px 8px;border-radius:6px;background:var(--ink);
color:var(--surface);font-size:12px;white-space:pre-line;pointer-events:none}
footer{color:var(--muted);font-size:12px;margin:20px 0}
"""
)

JS = """(()=>{const t=document.getElementById('tip');document.addEventListener('mousemove',e=>{
const g=e.target.closest&&e.target.closest('[data-tip]');if(!g){t.hidden=true;return}
t.textContent=g.getAttribute('data-tip');t.hidden=false;
t.style.left=Math.max(4,Math.min(e.clientX+14,innerWidth-t.offsetWidth-8))+'px';
t.style.top=Math.min(e.clientY+16,innerHeight-t.offsetHeight-8)+'px'});})();"""


# --------------------------------------------------------------------------- formatting
def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def num(value: Any) -> str:
    return f"{value:,}" if isinstance(value, int) and not isinstance(value, bool) else NR


def short(value: float | None) -> str:
    if value is None:
        return "not recorded"
    value = float(value)
    for scale, suffix in ((1e9, "G"), (1e6, "M"), (1e3, "k")):
        if abs(value) >= scale:
            return f"{value / scale:.2f}{suffix}"
    return f"{value:g}"


def when(value: float | None) -> str:
    return esc(R.stamp(value)) if value is not None else NR


def hours(value: float | None) -> str:
    if value is None:
        return NR
    return f"{value:.1f} h" if value < 48 else f"{value / 24:.1f} d"


def digest(value: Any, n: int = 12) -> str:
    return f'<span class="mono">{esc(str(value)[:n])}</span>' if value else NR


def mono(value: Any) -> str:
    return f'<span class="mono">{esc(value)}</span>' if value not in (None, "") else NR


def text_or_nr(value: Any) -> str:
    return esc(value) if value not in (None, "", []) else NR


def share(value: Any) -> str:
    return f"{value * 100:.1f}%" if isinstance(value, int | float) and not isinstance(value, bool) else NR


def badge(state: str | None) -> str:
    kind = _STATE_BADGE.get(str(state), "neutral")
    return f'<span class="badge {kind}">{esc(state or R.UNKNOWN)}</span>'


def tile(key: str, value: str, detail: str = "") -> str:
    return (
        f'<div class="tile"><div class="k">{esc(key)}</div><div class="v">{value}</div>'
        f'<div class="d">{detail}</div></div>'
    )


def table(headers: Sequence[str], rows: Iterable[Sequence[str]], numeric: Iterable[int] = ()) -> str:
    numeric = set(numeric)

    def cell(tag: str, index: int, value: str) -> str:
        return f'<{tag} class="n">{value}</{tag}>' if index in numeric else f"<{tag}>{value}</{tag}>"

    head = "".join(cell("th", i, esc(h)) for i, h in enumerate(headers))
    body = "".join("<tr>" + "".join(cell("td", i, c) for i, c in enumerate(row)) + "</tr>" for row in rows)
    if not body:
        body = f'<tr><td colspan="{len(headers)}">{NR}</td></tr>'
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def details(label: str, inner: str) -> str:
    return f"<details><summary>{esc(label)}</summary>{inner}</details>"


def missing(what: str, record: str) -> str:
    return f'<p>{esc(what)} (<span class="mono">{esc(record)}</span>): {NR}.</p>'


def legend(entries: Sequence[tuple[str, str]]) -> str:
    items = "".join(
        f'<span><span class="sw" style="background:{colour}"></span>{label}</span>' for colour, label in entries
    )
    return f'<div class="legend">{items}</div>'


# --------------------------------------------------------------------------- SVG primitives
class Scale:
    def __init__(self, d0: float, d1: float, r0: float, r1: float) -> None:
        if d1 == d0:
            d0, d1 = d0 - 1.0, d1 + 1.0
        self.d0, self.d1, self.r0, self.r1 = d0, d1, r0, r1

    def __call__(self, value: float) -> float:
        return self.r0 + (value - self.d0) * (self.r1 - self.r0) / (self.d1 - self.d0)


def nice_ticks(lo: float, hi: float, count: int = 5) -> list[float]:
    if hi <= lo:
        return [lo]
    raw = (hi - lo) / max(1, count)
    magnitude = 10 ** math.floor(math.log10(raw))
    step = next(m * magnitude for m in (1, 2, 2.5, 5, 10) if m * magnitude >= raw)
    ticks, value = [], math.ceil(lo / step) * step
    while value <= hi + 1e-9 and len(ticks) < 40:
        ticks.append(value)
        value += step
    return ticks


_TIME_STEPS = tuple(h * 3600 for h in (1, 2, 3, 6, 12, 24, 48, 72, 168, 336, 720))


def time_ticks(t0: float, t1: float, count: int = 6) -> list[tuple[float, str]]:
    span = max(1.0, t1 - t0)
    step = next((s for s in _TIME_STEPS if span / s <= count), _TIME_STEPS[-1])
    fmt = "%m-%d %H:%M" if step < 86400 else "%m-%d"
    out, value = [], (t0 // step + 1) * step
    while value <= t1 and len(out) < 40:
        out.append((value, datetime.fromtimestamp(value, UTC).strftime(fmt)))
        value += step
    return out


def _svg(width: int, height: int, body: str, label: str) -> str:
    return f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{esc(label)}">{body}</svg>'


def _line(x1: float, y1: float, x2: float, y2: float, attributes: str) -> str:
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" {attributes}/>'


def _text(x: float, y: float, value: str, anchor: str = "start") -> str:
    return f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}">{esc(value)}</text>'


def _y_axis(scale: Scale, ticks: Sequence[float], left: float, right: float, fmt: Callable[[float], str]) -> str:
    grid = "".join(_line(left, scale(v), right, scale(v), "") for v in ticks)
    labels = "".join(_text(left - 6, scale(v) + 4, fmt(v), "end") for v in ticks)
    return f'<g class="grid">{grid}</g>{labels}'


def _x_time_axis(scale: Scale, t0: float, t1: float, bottom: float) -> str:
    parts = [_line(scale.r0, bottom, scale.r1, bottom, 'class="axis"')]
    for value, label in time_ticks(t0, t1):
        x = scale(value)
        parts.append(_line(x, bottom, x, bottom + 4, 'class="axis"'))
        parts.append(_text(x, bottom + 16, f"{label} UTC", "middle"))
    return "".join(parts)


def _mark(x: float, y: float, tip: str, kind: str) -> str:
    if kind == "invalid":
        path = f"M{x - 4:.1f} {y - 4:.1f}L{x + 4:.1f} {y + 4:.1f}M{x - 4:.1f} {y + 4:.1f}L{x + 4:.1f} {y - 4:.1f}"
        shape = f'<path d="{path}" stroke="var(--critical)" stroke-width="2" fill="none"/>'
    else:
        shape = f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="var(--s1)" stroke="var(--surface)" stroke-width="1.5"/>'
    hit = f'<circle cx="{x:.1f}" cy="{y:.1f}" r="9" fill="transparent"/>'
    return f'<g data-tip="{esc(tip)}">{hit}{shape}</g>'


def _off_scale_mark(x: float, top: float, tip: str, colour: str) -> str:
    hit = f'<circle cx="{x:.1f}" cy="{top + 6}" r="9" fill="transparent"/>'
    shape = f'<path d="M{x - 5:.1f} {top + 10}L{x:.1f} {top + 1}L{x + 5:.1f} {top + 10}Z" fill="{colour}"/>'
    return f'<g data-tip="{esc(tip)}">{hit}{shape}</g>'


def bars(rows: Sequence[tuple[str, int, str]], label: str) -> str:
    if not rows:
        return f"<p>{NR}</p>"
    peak = max(1, max(v for _, v, _ in rows))
    width, left, right = 960, 210, 900
    parts = []
    for i, (name, value, tip) in enumerate(rows):
        top = 4 + i * 22
        length = (right - left) * value / peak
        visible = max(length, 1.5 if value else 0)
        parts.append(
            f'<g data-tip="{esc(tip)}">{_text(left - 8, top + 13, name, "end")}'
            f'<rect x="{left}" y="{top + 2}" width="{visible:.1f}" height="14" rx="3" fill="var(--s1)"/>'
            f"{_text(left + length + 6, top + 13, f'{value:,}')}</g>"
        )
    return _svg(width, 22 * len(rows) + 8, "".join(parts), label)


# --------------------------------------------------------------------------- phase-2 charts
#: A reading this many times the median measured reading is drawn on the top edge, value on hover, so
#: one pathological candidate does not flatten every other point onto the axis.
OFF_SCALE = 4.0


def _plotted(p2: Mapping[str, Any]) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    candidates = p2.get("candidates") or []
    measured = [
        c
        for c in candidates
        if c["class"] == R.MEASURED and c.get("cycles") and (c.get("finished") or c.get("requested"))
    ]
    invalid = [c for c in candidates if c["class"] == "correctness" and c.get("whole_window") and c.get("finished")]
    return measured, invalid


def _at(c: Mapping[str, Any]) -> float:
    return c.get("finished") or c["requested"]


def time_domain(p2: Mapping[str, Any], now: float) -> tuple[float, float] | None:
    """The x domain both phase-2 charts share: every plotted candidate and this run's window."""
    measured, invalid = _plotted(p2)
    times = [_at(c) for c in measured + invalid]
    started = (p2.get("liveness") or {}).get("run_started")
    if started:
        times += [started, now]
    if not times:
        return None
    t0, t1 = min(times), max(times)
    pad = max(3600.0, (t1 - t0) * 0.03)
    return t0 - pad, t1 + pad


def candidate_tip(c: Mapping[str, Any]) -> str:
    coverage = (c.get("coverage") or {}).get("priced_share")
    if c.get("cycles"):
        reading = f"cycles {c['cycles']:,}"
    elif c.get("whole_window"):
        reading = f"whole window {c['whole_window']:,} (wrong output)"
    else:
        reading = ""
    lines = [
        str(c.get("label") or "candidate"),
        f"{str(c.get('key'))[:16]}  {c.get('timing_status')}",
        reading,
        f"finished {R.stamp(c.get('finished'))}",
        f"package-authored {share(coverage)} priced" if coverage is not None else "",
    ]
    return "\n".join(line for line in lines if line)


def _references(p2: Mapping[str, Any]) -> list[tuple[str, int, str]]:
    refs: list[tuple[str, int, str]] = []
    bar = p2.get("bar") or {}
    if bar.get("cycles"):
        status = "" if bar.get("admissible_as_bar") else f" ({bar.get('timing_status')}: orientation only)"
        refs.append((f"bar: reference on the same machine {short(bar['cycles'])}{status}", bar["cycles"], "6 4"))
    roofline = p2.get("roofline")
    if roofline and roofline.get("gaps"):
        total = sum(int(g["roofline"]) for g in roofline["gaps"])
        refs.append((f"derived roofline, {len(roofline['gaps'])} groups {short(total)}", total, "2 3"))
    for row in p2.get("orientation") or ():
        refs.append((f"{row['label']} {short(row['cycles'])} (context only)", row["cycles"], "1 4"))
    return refs


def _best_path(line: Sequence[Mapping[str, Any]], x: Scale, y: Scale, hi: float, end: float) -> str:
    points: list[str] = []
    previous = None
    for row in line:
        px, py = x(row["at"]), y(min(row["cycles"], hi))
        if previous is not None:
            points.append(f"L{px:.1f} {previous:.1f}")
        points.append(f"{'M' if not points else 'L'}{px:.1f} {py:.1f}")
        previous = py
    points.append(f"L{x(end):.1f} {previous:.1f}")
    return f'<path d="{" ".join(points)}" fill="none" stroke="var(--s2)" stroke-width="2"/>'


def cycles_chart(p2: Mapping[str, Any], now: float) -> str:
    measured, invalid = _plotted(p2)
    if not measured and not invalid:
        return f"<p>Measured candidates: {NR} (the store holds no landed reading).</p>"
    refs = _references(p2)
    t0, t1 = time_domain(p2, now)
    started = (p2.get("liveness") or {}).get("run_started")
    readings = sorted([c["cycles"] for c in measured] + [c["whole_window"] for c in invalid])
    median = readings[len(readings) // 2]
    values = [v for v in readings + [r[1] for r in refs] if v <= OFF_SCALE * median]
    lo, hi = min(values), max(values)
    span = max(1.0, hi - lo)
    lo, hi = lo - span * 0.06, hi + span * 0.1
    width, height, left, right, top, bottom = 960, 330, 64, 940, 14, 296
    x, y = Scale(t0, t1, left, right), Scale(lo, hi, bottom, top)
    parts = [_y_axis(y, nice_ticks(lo, hi, 6), left, right, short), _x_time_axis(x, t0, t1, bottom)]
    if started:
        x0, x1 = x(max(started, t0)), x(min(now, t1))
        parts.append(
            f'<rect x="{x0:.1f}" y="{top}" width="{max(0.0, x1 - x0):.1f}" height="{bottom - top}" fill="var(--band)"/>'
        )
        parts.append(_text(x0 + 4, bottom - 6, "this run"))
    shown = sorted((r for r in refs if lo <= r[1] <= hi), key=lambda r: r[1])
    for i, (label, value, dash) in enumerate(shown):
        yy = y(value)
        # A label sits above its line unless the next line up is too close; then below it.
        below = i + 1 < len(shown) and abs(y(shown[i + 1][1]) - yy) < 14
        parts.append(_line(left, yy, right, yy, f'stroke="var(--ink-2)" stroke-width="1" stroke-dasharray="{dash}"'))
        parts.append(_text(right - 4, yy + 13 if below else yy - 4, label, "end"))
    if p2.get("best_line"):
        parts.append(_best_path(p2["best_line"], x, y, hi, min(now, t1)))
    off = 0
    plotted = [(c, c["cycles"], "measured") for c in measured] + [(c, c["whole_window"], "invalid") for c in invalid]
    for c, value, kind in plotted:
        if value > hi:
            off += 1
            colour = "var(--s1)" if kind == "measured" else "var(--critical)"
            parts.append(_off_scale_mark(x(_at(c)), top, "off scale\n" + candidate_tip(c), colour))
        else:
            parts.append(_mark(x(_at(c)), y(value), candidate_tip(c), kind))
    entries = [
        ("var(--s1)", "measured candidate (valid)"),
        ("var(--critical)", "&#10005; measured, wrong output (cycles shown, never a result)"),
        ("var(--s2)", "best so far (lowest valid, attributable, first replicate)"),
        ("var(--band)", "this run's window"),
    ]
    if off:
        entries.append(("transparent", f"&#9650; {off} off scale (over {OFF_SCALE:g}x the median; value on hover)"))
    return legend(entries) + _svg(width, height, "".join(parts), "cycles per candidate over time")


def coverage_chart(p2: Mapping[str, Any], now: float) -> str:
    rows = [
        c
        for c in p2.get("candidates") or ()
        if c["class"] == R.MEASURED and isinstance((c.get("coverage") or {}).get("priced_share"), int | float)
    ]
    domain = time_domain(p2, now)
    if not rows or domain is None:
        return (
            f"<p>Package-authored priced share: {NR} "
            "(no measured result with a build route table and a priced reference).</p>"
        )
    t0, t1 = domain
    width, height, left, right, top, bottom = 960, 150, 64, 940, 10, 120
    x, y = Scale(t0, t1, left, right), Scale(0, 100, bottom, top)
    parts = [
        _y_axis(y, [0, 25, 50, 75, 100], left, right, lambda v: f"{v:g}%"),
        _x_time_axis(x, t0, t1, bottom),
    ]
    for c in rows:
        parts.append(_mark(x(_at(c)), y(c["coverage"]["priced_share"] * 100), candidate_tip(c), "measured"))
    return _svg(width, height, "".join(parts), "package-authored share per measured candidate")


# --------------------------------------------------------------------------- phase-2 section
def _phase2_tiles(p2: Mapping[str, Any]) -> str:
    live, best, bar = p2.get("liveness") or {}, p2.get("best") or {}, p2.get("bar") or {}
    counts, in_run = p2.get("counts") or {}, p2.get("counts_in_run")
    board = p2.get("board")
    ratio = (
        f" &middot; {best['cycles'] / bar['cycles']:.3f}x the bar" if best.get("cycles") and bar.get("cycles") else ""
    )
    if live.get("last_measured"):
        last = when(live["last_measured"])
    else:
        last = "none since the run started" if live.get("run_started") else NR
    failures = sum(counts.get(k, 0) for k in R.FAILURE_CLASSES)

    def this_run(value: int | None) -> str:
        return f"this run: {value}" if in_run is not None else "this run: not recorded"

    tiles = [
        tile("State", badge(live.get("state")), esc(live.get("detail") or "")),
        tile(
            "Last measured candidate (this run)",
            last,
            f"run started {when(live.get('run_started'))}; store's last {when(live.get('last_measured_store'))}",
        ),
        tile("Best so far (valid)", num(best.get("cycles")), digest(best.get("key")) + ratio),
        tile("Bar (reference, same machine)", num(bar.get("cycles")), text_or_nr(bar.get("label"))),
        tile("Measured", f"{counts.get(R.MEASURED, 0):,}", this_run((in_run or {}).get(R.MEASURED, 0))),
        tile("Failed or refused", f"{failures:,}", this_run(sum((in_run or {}).get(k, 0) for k in R.FAILURE_CLASSES))),
        tile(
            "Board queue",
            f"{board.get('queue', 0):,}" if board else NR,
            f"oldest waiting since {when(board.get('oldest_waiting'))}" if board else "",
        ),
    ]
    return '<div class="tiles">' + "".join(tiles) + "</div>"


def _store_line(p2: Mapping[str, Any]) -> str:
    store = p2.get("store") or {}
    if not store.get("root"):
        return (
            f'<p class="sub">Measurement store: {NR} (the run records no <span class="mono">store_roots.screen</span>;'
            ' pass <span class="mono">--store</span>).</p>'
        )
    line = f'<p class="sub">Measurement store {mono(store["root"])} (from {esc(store.get("source"))}).</p>'
    if not store.get("loaded"):
        line += '<p class="sub">The recorded store directory does not exist.</p>'
    return line


def _reading(c: Mapping[str, Any]) -> str:
    if c.get("cycles"):
        return num(c["cycles"])
    if c.get("whole_window"):
        return num(c["whole_window"]) + " (wrong)"
    return NR


def _in_run(c: Mapping[str, Any]) -> str:
    return {True: "yes", False: "no"}.get(c.get("in_run"), NR)


def _candidates_table(candidates: Sequence[Mapping[str, Any]]) -> str:
    rows = [
        [
            when(_at(c) if c.get("finished") or c.get("requested") else None),
            digest(c["key"], 14),
            text_or_nr(c.get("label")),
            esc(c["class"]),
            _reading(c),
            share((c.get("coverage") or {}).get("priced_share")),
            _in_run(c),
            text_or_nr(c.get("reason")),
        ]
        for c in reversed(candidates[-80:])
    ]
    headers = ["finished", "job", "label", "class", "cycles", "pkg-authored", "this run", "reason"]
    return details("Table view (newest 80 store jobs)", table(headers, rows, numeric=(4, 5)))


def _outcomes(p2: Mapping[str, Any]) -> str:
    candidates = p2.get("candidates") or []
    counts, in_run = p2.get("counts") or {}, p2.get("counts_in_run")
    rows = []
    for name in (R.MEASURED, *R.FAILURE_CLASSES, R.OPEN, R.SUPERSEDED):
        tip = f"{counts.get(name, 0)} of {len(candidates)} store jobs"
        if in_run is not None:
            tip += f"; {in_run.get(name, 0)} in this run"
        rows.append((name, counts.get(name, 0), tip))
    failures = [
        [when(_at(c)), digest(c["key"], 14), esc(c["class"]), text_or_nr(c.get("reason"))]
        for c in reversed([c for c in candidates if c["class"] in R.FAILURE_CLASSES][-30:])
    ]
    return bars(rows, "store jobs by outcome class") + details(
        "Latest failures with their recorded reason", table(["at", "job", "class", "reason"], failures)
    )


def _roofline(p2: Mapping[str, Any]) -> str:
    roofline = p2.get("roofline")
    if not roofline:
        return (
            f"<p>Per-form roofline gap: {NR} (no result.json in the store carries "
            '<span class="mono">diagnostics.per_group</span> roofline rows).</p>'
        )
    kinds = [
        [
            esc(k["kind"]),
            str(k["groups"]),
            num(k["ours"]),
            num(k["roofline"]),
            f"{k['ours'] / k['roofline']:.2f}x" if k["roofline"] else NR,
        ]
        for k in roofline["by_kind"]
    ]
    groups = [
        [
            esc(g["group"]),
            text_or_nr(g.get("kind")),
            num(g["ours"]),
            num(g["roofline"]),
            f"{g['over_roofline']}x",
            text_or_nr(g.get("limiter")),
        ]
        for g in roofline["gaps"]
    ]
    return (
        '<p class="sub">From the lowest measured candidate whose result records '
        f'<span class="mono">diagnostics.per_group</span> rooflines ({digest(roofline["key"])}).</p>'
        + table(["form (kind)", "groups", "ours", "roofline", "over roofline"], kinds, numeric=(1, 2, 3, 4))
        + details(
            "Per group (largest gap first)",
            table(["group", "kind", "ours", "roofline", "over", "limiter"], groups, numeric=(2, 3, 4)),
        )
    )


def _plateau(p2: Mapping[str, Any], now: float) -> str:
    plateau = p2.get("plateau")
    if not plateau:
        return missing("Plateau record", "plateau.json")
    rule = plateau.get("rule") or {}
    last = plateau.get("last_improvement") or {}
    anchor = R.epoch(last.get("epoch")) or R.epoch(last.get("at"))
    since_now = (now - anchor) / 3600.0 if anchor else None
    sessions = plateau.get("sessions") or []
    trace = [
        [
            when(s["at"]),
            text_or_nr(s.get("run")),
            text_or_nr(s.get("session")),
            num(s.get("cycles")),
            text_or_nr(s.get("model")),
            "reset" if s.get("reset") else text_or_nr(s.get("abandoned")),
        ]
        for s in reversed(sessions[-60:])
    ]
    return (
        f"<p>Rule: no eligible improvement for <b>{text_or_nr(rule.get('hours'))} h</b> over at least "
        f"<b>{text_or_nr(rule.get('min_sessions'))}</b> sessions. Last improvement {when(anchor)} "
        f"({num(last.get('cycles'))} cycles, {digest(last.get('package_sha256'))}); as recorded "
        f"{text_or_nr(last.get('hours_since'))} h / {text_or_nr(last.get('sessions_since'))} sessions since; "
        f"now {hours(since_now)}.</p>"
        + details(
            f"Session trace ({len(sessions)} rows, newest first)",
            table(["at", "run", "session", "best cycles at start", "model", "note"], trace, numeric=(3,)),
        )
    )


def _sessions(p2: Mapping[str, Any]) -> str:
    sessions = p2.get("sessions")
    if sessions:
        stopped = sessions.get("stopped") or {}
        out = (
            f"<p>Sessions: {len(sessions.get('rows') or [])} recorded; driver {text_or_nr(sessions.get('driver'))}, "
            f"model {text_or_nr(sessions.get('model'))}; stopped: {text_or_nr(stopped.get('kind'))} "
            f"{text_or_nr(stopped.get('reason'))}.</p>"
        )
    else:
        out = missing("Session record", "stage/sessions.json")
    breaker = p2.get("circuit_breaker")
    if breaker:
        out += (
            f"<p>Infra circuit breaker tripped {text_or_nr(breaker.get('at'))}: {text_or_nr(breaker.get('reason'))}</p>"
        )
    if p2.get("rounds"):
        rows = [
            [
                text_or_nr(r["round"]),
                text_or_nr(r["status"]),
                text_or_nr(r["why"]),
                text_or_nr(r["requested"]),
                digest(r["final_package"]),
                text_or_nr(r["final_timing_status"]),
            ]
            for r in p2["rounds"]
        ]
        headers = ["round", "status", "why", "requested", "final package", "final status"]
        out += details(f"Authoring rounds ({len(rows)})", table(headers, rows))
    return out


def _board(p2: Mapping[str, Any]) -> str:
    board = p2.get("board")
    if not board:
        return f"<p>Board queue and outages: {NR}.</p>"
    runner = board.get("batch_runner") or {}
    if runner:
        present = "process present now" if runner.get("process_present") else "no such process at generation time"
        runner_text = f"pid {text_or_nr(runner.get('pid'))} started {when(runner.get('started'))}; {present}"
    else:
        runner_text = NR
    summary = (
        f"<p>Waiting for the board: <b>{board.get('queue', 0)}</b> (oldest since {when(board.get('oldest_waiting'))});"
        f" running {board.get('running', 0)}; pending/screening {board.get('pending', 0)}. "
        f"Solo streak {num(board.get('solo_streak'))}. Batches run: {board.get('batches', 0)}. "
        f"Batch runner: {runner_text}.</p>"
    )
    outage = board.get("open_outage")
    if outage:
        summary += (
            f"<p><b>Board outage open</b> since {when(outage['opened'])}: {outage['failures']} failures; next try "
            f"{when(outage['retry_after'])}. Last: {text_or_nr(outage['last_reason'])}</p>"
        )
    else:
        summary += "<p>No board outage open.</p>"
    closed = [
        [
            when(o["opened"]),
            when(o["closed"]),
            hours((o["closed"] - o["opened"]) / 3600 if o["closed"] and o["opened"] else None),
            str(o["failures"]),
            text_or_nr(o["last_reason"]),
        ]
        for o in reversed(board.get("closed_outages") or [])
    ]
    batches = [
        [
            digest(b["id"], 40),
            str(b["jobs"]),
            text_or_nr(b["control_ok"]),
            text_or_nr(b["control_ratio"]),
            "yes" if b["board_unavailable"] else "no",
            text_or_nr(b["failure"]),
        ]
        for b in reversed(board.get("recent_batches") or [])
    ]
    return (
        summary
        + table(["outage opened", "closed", "duration", "failures", "last reason"], closed, numeric=(3,))
        + details(
            "Recent batches",
            table(["batch", "jobs", "control ok", "control ratio", "board unavailable", "failure"], batches),
        )
    )


def _node(key: str, value: str) -> str:
    return f'<div class="node"><div class="k">{esc(key)}</div>{value}</div>'


def _chain(nodes: Sequence[str]) -> str:
    return '<div class="chain">' + '<span class="arrow">&rarr;</span>'.join(nodes) + "</div>"


def _phase2_lineage(p2: Mapping[str, Any]) -> str:
    lineage, ledger = p2.get("lineage"), p2.get("ledger")
    parts = ['<section id="lineage"><h2>Champion lineage</h2>']
    if lineage:
        nodes = [
            _node(
                "resumed from",
                f"{mono(row['run'].rstrip('/').rpartition('/')[2])}<br>seed {digest(row.get('seed_package'))}",
            )
            for row in reversed(lineage["chain"])
        ]
        kind = lineage.get("lineage_kind") or "kind not recorded"
        origin = text_or_nr(lineage.get("origin_kind"))
        nodes.append(_node(f"this run's seed ({kind})", f"{digest(lineage.get('seed_package'))}<br>origin {origin}"))
        for event in (ledger or {}).get("best") or ():
            nodes.append(
                _node(
                    f"best moved (candidate {event.get('n')})",
                    f"{digest(event.get('package'))}<br>{when(event.get('at'))}",
                )
            )
        index_rows = lineage.get("index_rows")
        for row in (index_rows or {}).get("champions") or ():
            cycles = num((row.get("firesim") or {}).get("cycles"))
            nodes.append(_node("exported champion", f"{text_or_nr(row.get('package_id'))}<br>{cycles} cycles"))
        parts.append(_chain(nodes))
        parts.append(f'<p class="sub">Why this run exists: {text_or_nr(lineage.get("why"))}</p>')
        if index_rows is None:
            parts.append(f"<p>INDEX.yaml rows citing this run: {NR}.</p>")
    else:
        parts.append(missing("Seed lineage", "resumed_seed.json"))
    if ledger is None:
        parts.append(missing("OOT ledger", "iterations.jsonl"))
    else:
        parts.append(
            f"<p>OOT ledger: {ledger['candidates']} candidate commits, {len(ledger['measured'])} measured tags, "
            f"{len(ledger['best'])} best moves.</p>"
        )
    parts.append("</section>")
    return "".join(parts)


def phase2_section(summary: Mapping[str, Any], p2: Mapping[str, Any]) -> str:
    now = summary["generated"]
    return (
        '<section id="phase2"><h2>Phase 2 &middot; measured candidates</h2>'
        + _phase2_tiles(p2)
        + _store_line(p2)
        + "<h3>Cycles per candidate over time</h3>"
        + cycles_chart(p2, now)
        + _candidates_table(p2.get("candidates") or [])
        + "<h3>Package-authored coverage (priced share) beside the cycles</h3>"
        + coverage_chart(p2, now)
        + "<h3>Outcome by class</h3>"
        + _outcomes(p2)
        + "<h3>Distance to the roofline, per form</h3>"
        + _roofline(p2)
        + "<h3>Plateau and liveness</h3>"
        + _plateau(p2, now)
        + _sessions(p2)
        + "<h3>Board queue and outages</h3>"
        + _board(p2)
        + "</section>"
        + _phase2_lineage(p2)
    )


# --------------------------------------------------------------------------- phase 1
def _tier_colour(index: int, count: int) -> str:
    return "var(--t3)" if count <= 1 else f"var(--t{round(index * 6 / (count - 1))})"


def progression_chart(grades: Sequence[Mapping[str, Any]]) -> str:
    points = [g for g in grades if g.get("at") is not None and g.get("n_passed") is not None]
    if not points:
        return f"<p>Score progression: {NR}.</p>"
    t0, t1 = points[0]["at"], points[-1]["at"]
    pad = max(600.0, (t1 - t0) * 0.04)
    peak = max([g.get("n_capsules") or 0 for g in points] + [g["n_passed"] for g in points] + [1])
    width, height, left, right, top, bottom = 960, 230, 52, 940, 12, 196
    x, y = Scale(t0 - pad, t1 + pad, left, right), Scale(0, peak * 1.05, bottom, top)
    parts = [
        _y_axis(y, nice_ticks(0, peak, 5), left, right, lambda v: f"{v:g}"),
        _x_time_axis(x, t0 - pad, t1 + pad, bottom),
    ]
    totals = [g for g in points if g.get("n_capsules") is not None]
    if totals:
        path = " ".join(
            f"{'M' if i == 0 else 'L'}{x(g['at']):.1f} {y(g['n_capsules']):.1f}" for i, g in enumerate(totals)
        )
        parts.append(f'<path d="{path}" fill="none" stroke="var(--muted)" stroke-width="1.5" stroke-dasharray="4 3"/>')
    path = " ".join(f"{'M' if i == 0 else 'L'}{x(g['at']):.1f} {y(g['n_passed']):.1f}" for i, g in enumerate(points))
    parts.append(f'<path d="{path}" fill="none" stroke="var(--s1)" stroke-width="2"/>')
    for g in points:
        tip = (
            f"{g['name']}\n{R.stamp(g['at'])}\npassed {g['n_passed']} of {g.get('n_capsules')}\n"
            f"highest tier {g.get('highest_tier')}"
        )
        parts.append(_mark(x(g["at"]), y(g["n_passed"]), tip, "measured"))
    entries = [("var(--s1)", "capsules passed"), ("var(--muted)", "capsules graded")]
    return legend(entries) + _svg(width, height, "".join(parts), "capsules passed per grade")


def _heat_cell(name: str, index: int, grade: Mapping[str, Any], tiers: Sequence[str]) -> tuple[str, str]:
    row = next((c for c in grade["capsules"] if c["capsule"] == name), None)
    where = f"{name}\ngrade {index + 1} ({grade['name']})"
    if row is None:
        return "var(--surface-2)", f"{where}: not in this grade"
    if row["status"] == "pass":
        position = tiers.index(row["highest_pass"]) if row["highest_pass"] in tiers else 0
        return _tier_colour(position, len(tiers)), f"{where}: pass, highest tier {row['highest_pass']}"
    reason = f"{row.get('failure_plane') or ''} {row.get('failure_category') or ''}".strip()
    return "var(--fail)", f"{where}: {row['status']}, highest tier passed {row['highest_pass'] or 'none'}\n{reason}"


def heatmap(p1: Mapping[str, Any], limit: int = 40) -> str:
    grades = [g for g in p1.get("grades") or () if g.get("capsules")][-limit:]
    tiers = list(p1.get("tiers") or [])
    if not grades:
        return f"<p>Capsule history: {NR}.</p>"
    names = sorted({c["capsule"] for g in grades for c in g["capsules"]})
    width, left, top, cell_h = 960, 300, 18, 12
    cell_w = max(8, min(28, (width - left - 10) // len(grades)))
    parts = [_text(left + j * cell_w + cell_w / 2, 12, str(j + 1), "middle") for j in range(len(grades))]
    for i, name in enumerate(names):
        parts.append(_text(left - 6, top + i * cell_h + 10, name[:48], "end"))
        for j, grade in enumerate(grades):
            fill, tip = _heat_cell(name, j, grade, tiers)
            parts.append(
                f'<rect data-tip="{esc(tip)}" x="{left + j * cell_w + 1}" y="{top + i * cell_h + 1}" '
                f'width="{cell_w - 2}" height="{cell_h - 2}" rx="2" fill="{fill}"/>'
            )
    entries = [(_tier_colour(i, len(tiers)), f"pass, highest tier {esc(t)}") for i, t in enumerate(tiers)]
    entries += [("var(--fail)", "not passing (status shown on hover)"), ("var(--surface-2)", "not in that grade")]
    note = f'<p class="sub">Columns are grades in time order (the last {len(grades)}).</p>'
    return (
        legend(entries) + _svg(width, top + cell_h * len(names) + 6, "".join(parts), "capsule status per grade") + note
    )


def _tier_cell(status: str | None) -> str:
    if status is None:
        return '<td class="cell-other">&ndash;</td>'
    if status == "pass":
        return '<td class="cell-pass">&#10003; pass</td>'
    if status == "fail":
        return '<td class="cell-fail">&#10005; fail</td>'
    return f'<td class="cell-other">{esc(status)}</td>'


def tier_matrix(latest: Mapping[str, Any], tiers: Sequence[str]) -> str:
    if not latest.get("capsules"):
        return f"<p>Capsule &times; tier matrix: {NR}.</p>"
    rows = sorted(latest["capsules"], key=lambda c: (c["status"] == "pass", c["capsule"]))
    head = "".join(f"<th>{esc(t)}</th>" for t in tiers)
    body = "".join(
        f"<tr><td>{mono(c['capsule'])}</td><td>{text_or_nr(c.get('label'))}</td>"
        + "".join(_tier_cell(c["tiers"].get(t)) for t in tiers)
        + f"<td>{text_or_nr(c['status'])}</td><td>{text_or_nr(c.get('failure_plane'))}</td></tr>"
        for c in rows
    )
    header = f"<tr><th>capsule</th><th>set</th>{head}<th>status</th><th>failure plane</th></tr>"
    return f'<div class="scroll"><table><thead>{header}</thead><tbody>{body}</tbody></table></div>'


def _phase1_tiles(p1: Mapping[str, Any]) -> str:
    live, latest, plateau = p1.get("liveness") or {}, p1.get("latest") or {}, p1.get("plateau")
    if latest.get("n_passed") is not None:
        score = f"{latest['n_passed']} / {latest['n_capsules']}"
        when_graded = f"{esc(latest.get('name', ''))} &middot; {when(latest.get('at'))}"
    else:
        score, when_graded = NR, ""
    tiles = [
        tile("State", badge(live.get("state")), esc(live.get("detail") or "")),
        tile("Latest grade", score, when_graded),
        tile("Grades recorded", str(len(p1.get("grades") or [])), ""),
        tile("Highest tier (latest)", text_or_nr(latest.get("highest_tier")), ""),
        tile(
            "Plateau",
            ("stuck" if plateau.get("stuck") else "progressing") if plateau else NR,
            text_or_nr(plateau.get("sentence")) if plateau else "",
        ),
    ]
    return '<div class="tiles">' + "".join(tiles) + "</div>"


def _phase1_end(p1: Mapping[str, Any]) -> str:
    freeze, manifest, loop, commits = (
        p1.get("freeze"),
        p1.get("manifest"),
        p1.get("qa_loop_summary"),
        p1.get("oot_commits"),
    )
    out = "<h3>Freeze, formal grade and OOT history</h3>"
    if freeze:
        error = f"; OOT error {esc(freeze['oot_error'])}" if freeze.get("oot_error") else ""
        out += (
            f"<p>Frozen {when(freeze['at'])}: submission {digest(freeze.get('submission_sha256'))}, "
            f"commit {digest(freeze.get('frozen_commit'))}{error}.</p>"
        )
    else:
        out += missing("Freeze", "freeze.json")
    if manifest:
        out += (
            f"<p>Formal grade: public {esc(manifest['public'])}, hidden {esc(manifest['hidden'])}, complete "
            f"{text_or_nr(manifest['formal_grade_complete'])}; failures "
            f"{text_or_nr(', '.join(manifest['completion_failures']))}; model {text_or_nr(manifest.get('model'))}; "
            f"wall {text_or_nr(manifest.get('wall_time_seconds'))} s.</p>"
        )
    else:
        out += missing("Formal manifest", "run_manifest.yaml")
    if loop:
        out += (
            f"<p>QA loop: {text_or_nr(loop['n_rounds'])} rounds, converged {text_or_nr(loop['converged'])}, "
            f"formal complete {text_or_nr(loop['formal_complete'])}, feedback healthy "
            f"{text_or_nr(loop['feedback_healthy'])}.</p>"
        )
    else:
        out += missing("QA loop summary", "qa_loop_summary.yaml")
    if commits is None:
        return out + missing("OOT commit log", "oot_commits.jsonl")
    rows = [
        [
            when(r["at"]),
            text_or_nr(r["label"]),
            text_or_nr(r["key"]),
            digest(r["commit"]),
            f"{r['n_passed']}/{r['n_capsules']}" if r["n_passed"] is not None else NR,
            text_or_nr(r["error"]),
        ]
        for r in reversed(commits)
    ]
    return out + details(f"OOT commits ({len(rows)})", table(["at", "label", "key", "commit", "passed", "error"], rows))


def phase1_section(p1: Mapping[str, Any]) -> str:
    latest, tiers = p1.get("latest") or {}, p1.get("tiers") or []
    grades = [
        [esc(g["name"]), when(g["at"]), num(g["n_passed"]), num(g["n_capsules"]), text_or_nr(g["highest_tier"])]
        for g in reversed(p1.get("grades") or [])
    ]
    failing = [
        [
            mono(c["capsule"]),
            text_or_nr(c["status"]),
            text_or_nr(c.get("highest_pass")),
            text_or_nr(c.get("failure_plane")),
            text_or_nr(c.get("failure_category")),
            text_or_nr(c.get("failure_detail")),
        ]
        for c in p1.get("failing") or ()
    ]
    planes = latest.get("first_failure_planes") or {}
    plane_rows = [
        (str(k), int(v), f"{v} capsules first fail at {k}")
        for k, v in sorted(planes.items(), key=lambda kv: -int(kv[1]))
    ]
    return (
        '<section id="phase1"><h2>Phase 1 &middot; capsule grades</h2>'
        + _phase1_tiles(p1)
        + "<h3>Score progression</h3>"
        + progression_chart(p1.get("grades") or [])
        + details("Table view", table(["grade", "at", "passed", "graded", "highest tier"], grades, numeric=(2, 3)))
        + "<h3>Capsules over grades</h3>"
        + heatmap(p1)
        + "<h3>Capsule &times; tier, latest grade</h3>"
        + tier_matrix(latest, tiers)
        + "<h3>Where the latest grade fails first</h3>"
        + bars(plane_rows, "first failure planes")
        + "<h3>Failing capsules and their recorded reasons</h3>"
        + table(["capsule", "status", "highest tier passed", "plane", "category", "detail"], failing)
        + _phase1_end(p1)
        + "</section>"
    )


# --------------------------------------------------------------------------- orchestration, inventory, page
def orchestration_section(record: Mapping[str, Any] | None) -> str:
    if not record:
        return ""

    def pid(a: Mapping[str, Any]) -> str:
        present = {None: "", True: " (present now)", False: " (absent now)"}[a["process_present"]]
        return text_or_nr(a["pid"]) + present

    attempts = [
        [
            text_or_nr(a["phase"]),
            text_or_nr(a["adapter"]),
            text_or_nr(a["state"]),
            when(a["started"]),
            when(a["ended"]),
            text_or_nr(a["returncode"]),
            pid(a),
        ]
        for a in record.get("attempts") or ()
    ]
    phases = [
        [esc(n), text_or_nr(p["adapter"]), text_or_nr(p["state"]), mono(p["engine_output"])]
        for n, p in sorted((record.get("phases") or {}).items())
    ]
    return (
        '<section id="orchestration"><h2>Orchestration</h2>'
        f"<p>Experiment {text_or_nr(record.get('experiment'))}, state {text_or_nr(record.get('state'))}, "
        f"frozen {text_or_nr(record.get('frozen_at'))}; plan {esc(record.get('plan_binding'))}.</p>"
        + table(["phase", "adapter", "state", "engine output"], phases)
        + "<h3>Attempts</h3>"
        + table(["phase", "adapter", "state", "started", "ended", "exit", "pid"], attempts)
        + "</section>"
    )


def inventory_section(rows: Sequence[Mapping[str, Any]]) -> str:
    body = [[esc(r["record"]), esc(r["state"]), text_or_nr(r.get("detail")), mono(r["path"])] for r in rows]
    absent = sum(1 for r in rows if r["state"] != "read")
    return (
        '<section id="records"><h2>Records read</h2>'
        f'<p class="sub">{len(rows)} records consulted, {absent} absent, unreadable or of an unexpected schema. '
        "Nothing was measured, graded or rebuilt to draw this page.</p>"
        + details("Inventory", table(["record", "state", "detail", "path"], body))
        + "</section>"
    )


def page(title: str, heading: str, sub: str, nav: Sequence[tuple[str, str]], body: str, generated: float) -> str:
    links = "".join(f'<a href="#{esc(anchor)}">{esc(label)}</a>' for anchor, label in nav)
    head = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{esc(title)}</title><style>{CSS}</style></head><body><main>"
    )
    footer = (
        f'<footer>Generated {esc(R.stamp(generated))} by <span class="mono">merlin-experiment dashboard</span> '
        "from existing records only.</footer>"
    )
    return (
        f'{head}<header><h1>{heading}</h1><span class="sub">{sub}</span></header><nav>{links}</nav>{body}{footer}'
        f'</main><div id="tip" hidden></div><script>{JS}</script></body></html>\n'
    )


def run_body(summary: Mapping[str, Any]) -> tuple[list[str], list[tuple[str, str]]]:
    sections, nav = [], []
    if summary.get("phase2"):
        sections.append(phase2_section(summary, summary["phase2"]))
        nav += [("phase2", "Phase 2"), ("lineage", "Lineage")]
    if summary.get("phase1"):
        sections.append(phase1_section(summary["phase1"]))
        nav.append(("phase1", "Phase 1"))
    if summary.get("orchestration"):
        sections.append(orchestration_section(summary["orchestration"]))
        nav.append(("orchestration", "Orchestration"))
    for number, engine in sorted((summary.get("engines") or {}).items()):
        inner, _ = run_body({**engine, "orchestration": None, "engines": None})
        anchor = f"engine{number}"
        sections.append(
            f'<section id="{esc(anchor)}"><h2>Phase {esc(number)} engine output: {esc(engine["run_id"])}</h2>'
            + ("".join(inner) or f"<p>No phase records: {NR}.</p>")
            + "</section>"
        )
        nav.append((anchor, f"Phase {number} engine"))
    return sections, nav


def render_run(summary: Mapping[str, Any]) -> str:
    live = summary.get("liveness") or {}
    sections, nav = run_body(summary)
    if not sections:
        sections.append(f"<section><p>No phase records were found in this directory: {NR}.</p></section>")
    sections.append(inventory_section(summary.get("inventory") or []))
    nav.append(("records", "Records"))
    phases = ", ".join(summary.get("phases") or []) or "not recorded"
    sub = (
        f"target {text_or_nr(summary.get('target'))} &middot; phase {esc(phases)} &middot; "
        f"{badge(live.get('state'))} {esc(live.get('detail') or '')}"
    )
    title = f"Experiment dashboard {summary['run_id']}"
    return page(title, esc(summary["run_id"]), sub, nav, "".join(sections), summary["generated"])


# --------------------------------------------------------------------------- target view
def _champion_chain(row: Mapping[str, Any]) -> str:
    lineage, firesim = row.get("lineage") or {}, row.get("firesim") or {}
    control = text_or_nr(firesim.get("control_in_batch"))
    return _chain(
        [
            _node("phase-0 corpus seal", digest(lineage.get("corpus_seal_digest"))),
            _node(
                "phase-1 frozen", f"{text_or_nr(lineage.get('phase1_run'))}<br>{digest(lineage.get('frozen_commit'))}"
            ),
            _node("phase-2 best", f"{text_or_nr(lineage.get('phase2_run'))}<br>{digest(lineage.get('best_commit'))}"),
            _node(
                "champion",
                f"{text_or_nr(row.get('package_id'))}<br>{num(firesim.get('cycles'))} cycles "
                f"&middot; control in batch {control}",
            ),
        ]
    )


def _origin(row: Mapping[str, Any]) -> str:
    origin = row.get("origin")
    return digest(origin.get("commit")) if isinstance(origin, Mapping) else text_or_nr(origin)


def index_section(summary: Mapping[str, Any]) -> str:
    index = summary.get("index")
    if not index:
        return (
            f'<section id="lineage"><h2>Champion lineage</h2><p>INDEX.yaml: {NR} at {mono(summary.get("index_path"))} '
            f'(write it with <span class="mono">merlin-experiment index {esc(summary["target"])}</span>).</p></section>'
        )
    champions = "".join(_champion_chain(row) for row in index.get("champions") or ())
    best = [
        [
            text_or_nr(r.get("run")),
            _origin(r),
            digest(r.get("best_commit")),
            digest(r.get("package_digest")),
            str(len(r.get("measured") or [])),
        ]
        for r in index.get("phase2_best") or ()
    ]
    frozen = [
        [
            text_or_nr(r.get("run")),
            digest(r.get("frozen_commit")),
            digest(r.get("package_digest")),
            text_or_nr(r.get("rounds")),
            text_or_nr(r.get("frozen_at")),
        ]
        for r in index.get("phase1_frozen") or ()
    ]
    releases = [
        [
            text_or_nr(r.get("release")),
            text_or_nr(r.get("state")),
            digest(r.get("review_digest")),
            text_or_nr(r.get("source_run")),
            text_or_nr(r.get("sealed_at")),
        ]
        for r in index.get("phase0_releases") or ()
    ]
    problems = [[text_or_nr(p.get("path")), text_or_nr(p.get("problem"))] for p in index.get("problems") or ()]
    return (
        '<section id="lineage"><h2>Champion lineage</h2>'
        + (champions or f"<p>Exported champions: {NR}.</p>")
        + "<h3>Phase-2 bests</h3>"
        + table(["run", "origin", "best commit", "package", "measured tags"], best)
        + "<h3>Phase-1 frozen compilers</h3>"
        + table(["run", "frozen commit", "package", "rounds", "frozen at"], frozen)
        + "<h3>Phase-0 corpus releases</h3>"
        + table(["release", "state", "review digest", "source run", "sealed at"], releases)
        + (details(f"Index problems ({len(problems)})", table(["path", "problem"], problems)) if problems else "")
        + "</section>"
    )


def _run_row(phase: str, r: Mapping[str, Any]) -> list[str]:
    state = esc(r["problem"]) if r.get("problem") else badge(r.get("state"))
    if phase == "1":
        progress = (
            f"{r['latest_passed']}/{r['latest_capsules']} over {r['grades']} grades"
            if r.get("latest_passed") is not None
            else NR
        )
        identity, best = digest(r.get("frozen_commit")), ""
    else:
        progress, identity = "", text_or_nr(r.get("method"))
        best = num(r.get("best_cycles")) if phase == "2" else ""
    return [
        mono(r["run_id"]),
        state,
        text_or_nr(r.get("orchestration_state")),
        progress,
        identity,
        best,
        text_or_nr(r.get("detail")),
    ]


def render_target(summary: Mapping[str, Any]) -> str:
    sections = [index_section(summary)]
    for phase, rows in sorted((summary.get("phase_runs") or {}).items()):
        extra = (summary.get("truncated") or {}).get(phase) or 0
        title = f"Phase {phase} runs ({len(rows)}{f', {extra} older not shown' if extra else ''})"
        headers = ["run", "state", "orchestration", "latest grade", "frozen commit / method", "best cycles", "detail"]
        sections.append(
            f'<section id="phase{esc(phase)}"><h2>{esc(title)}</h2>'
            + table(headers, [_run_row(phase, r) for r in rows])
            + "</section>"
        )
    orchestrations = summary.get("orchestrations") or {}
    found = [
        [
            text_or_nr(r.get("experiment")),
            text_or_nr(r.get("state")),
            mono(r.get("run_dir")),
            esc(", ".join(f"{n}:{p.get('state')}" for n, p in sorted((r.get("phases") or {}).items()))),
        ]
        for r in orchestrations.get("runs") or ()
    ]
    problems = [
        [text_or_nr(p.get("run_dir")), text_or_nr(p.get("error"))] for p in orchestrations.get("problems") or ()
    ]
    sections.append(
        '<section id="orchestrations"><h2>Orchestrations</h2>'
        + table(["experiment", "state", "run", "phases"], found)
        + (details("Discovery problems", table(["run", "error"], problems)) if problems else "")
        + "</section>"
    )
    sections.append(inventory_section(summary.get("inventory") or []))
    nav = [("lineage", "Lineage")]
    nav += [(f"phase{p}", f"Phase {p}") for p in sorted(summary.get("phase_runs") or {})]
    nav += [("orchestrations", "Orchestrations"), ("records", "Records")]
    target = summary["target"]
    sub = "all runs across phases, from the target's records"
    return page(
        f"Target dashboard {target}", f"Target {esc(target)}", sub, nav, "".join(sections), summary["generated"]
    )


def render(summary: Mapping[str, Any]) -> str:
    return render_target(summary) if summary.get("kind") == "target" else render_run(summary)


__all__ = ["render", "render_run", "render_target"]
