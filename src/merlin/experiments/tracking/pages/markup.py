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

from merlin.experiments.tracking.readers import records as R

# ---- html --------------------------------------------------------------------------------------

NR = '<span class="nr">not recorded</span>'


_STATE_BADGE = {
    R.UNFINALIZED: "neutral",
    R.FROZEN: "neutral",
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


EXTRA_CSS = """
:root{--c1:#2a78d6;--c2:#eb6834;--c3:#1baf7a;--c4:#eda100;--c5:#e87ba4;--c6:#008300;--c7:#4a3aa7;--c8:#e34948}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--c1:#3987e5;--c2:#d95926;--c3:#199e70;
--c4:#c98500;--c5:#d55181;--c6:#008300;--c7:#9085e9;--c8:#e66767}}
:root[data-theme=dark]{--c1:#3987e5;--c2:#d95926;--c3:#199e70;--c4:#c98500;--c5:#d55181;--c6:#008300;
--c7:#9085e9;--c8:#e66767}
.mode{display:inline-block;padding:2px 10px;border-radius:6px;font-weight:600;font-size:12px;
border:1px dashed var(--border);background:var(--surface-2)}
.feed{list-style:none;margin:0;padding:0;font-size:12.5px}.feed li{padding:3px 0;border-bottom:1px solid var(--grid)}
.feed .t{color:var(--muted);font-family:ui-monospace,Menlo,monospace;font-size:11.5px;margin-right:8px}
.feed .kind{display:inline-block;min-width:74px;color:var(--ink-2);font-weight:600}
.feed .err{color:var(--critical)}
.now{border-left:4px solid var(--s1);padding:6px 10px;background:var(--surface-2);border-radius:4px}
.gap{background:color-mix(in srgb,var(--critical) 22%,transparent)}
.hit{background:color-mix(in srgb,var(--good) 22%,transparent)}
.cols{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:12px}
pre.note{white-space:pre-wrap;margin:4px 0;font-size:12px}
"""


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
    + EXTRA_CSS
)


JS = """(()=>{const t=document.getElementById('tip');document.addEventListener('mousemove',e=>{
const g=e.target.closest&&e.target.closest('[data-tip]');if(!g){t.hidden=true;return}
t.textContent=g.getAttribute('data-tip');t.hidden=false;
t.style.left=Math.max(4,Math.min(e.clientX+14,innerWidth-t.offsetWidth-8))+'px';
t.style.top=Math.min(e.clientY+16,innerHeight-t.offsetHeight-8)+'px'});
document.querySelectorAll('table.sortable th').forEach(h=>{h.style.cursor='pointer';h.title='sort';
h.addEventListener('click',()=>{const tb=h.closest('table').tBodies[0],i=h.cellIndex,up=h.dataset.up!=='1';
h.dataset.up=up?'1':'0';const k=r=>{const s=(r.cells[i]||{}).textContent||'',
n=parseFloat(s.split(',').join(''));return isNaN(n)?s:n};
[...tb.rows].sort((a,b)=>{const x=k(a),y=k(b);return(x>y?1:x<y?-1:0)*(up?1:-1)}).forEach(r=>tb.appendChild(r))})});
document.querySelectorAll('input[data-filter]').forEach(f=>f.addEventListener('input',()=>{
const q=f.value.toLowerCase();document.querySelectorAll('table.sortable tbody tr').forEach(r=>{
r.hidden=!!q&&!r.textContent.toLowerCase().includes(q)})}));
})();"""


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
