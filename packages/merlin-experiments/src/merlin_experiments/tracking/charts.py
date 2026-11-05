"""Inline-SVG chart primitives the richer views share: a Gantt, a treemap, a scatter, a time series.

Everything is drawn here from numbers a reader already took from records; nothing is fetched and no
script is needed to read a chart (hover text rides on ``data-tip`` for the page's one tooltip script).
Colours are the page's CSS tokens: categorical identity uses ``--c1`` .. ``--c8`` in a fixed order and
folds anything past the eighth entity into "other"; status colours stay reserved for status.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from . import records as R
from .html import NR, Scale, _line, _svg, _text, _x_time_axis, _y_axis, esc, legend, nice_ticks, short

#: Categorical slots in their fixed order (never cycled; the ninth entity is "other").
CATEGORICAL = tuple(f"var(--c{i})" for i in range(1, 9))
OTHER = "var(--muted)"


def palette(names: Sequence[str]) -> dict[str, str]:
    """Colour per entity in first-seen order; the ninth and later fold into :data:`OTHER`."""
    out: dict[str, str] = {}
    for name in names:
        if name not in out:
            out[name] = CATEGORICAL[len(out)] if len(out) < len(CATEGORICAL) else OTHER
    return out


def _tip(tip: str, inner: str) -> str:
    return f'<g data-tip="{esc(tip)}">{inner}</g>'


# --------------------------------------------------------------------------- Gantt
def gantt(
    lanes: Sequence[Mapping[str, Any]],
    *,
    now: float,
    colours: Mapping[str, str],
    label: str,
    window: tuple[float, float] | None = None,
    max_marks: int = 4000,
) -> str:
    """Lanes of spans and point events over time, with a "now" line.

    A lane is ``{"lane": name, "spans": [{"start", "end" | None, "kind", "tip"}], "points": [{"at",
    "kind", "tip"}]}``.  A span whose ``end`` is None is RUNNING: drawn to ``now`` with a dashed outline
    and no fill, so done and running read apart without colour.  ``window`` clips to a time range."""
    lanes = [lane for lane in lanes if lane.get("spans") or lane.get("points")]
    times = [s["start"] for lane in lanes for s in lane.get("spans") or () if s.get("start") is not None]
    times += [s["end"] for lane in lanes for s in lane.get("spans") or () if s.get("end") is not None]
    times += [p["at"] for lane in lanes for p in lane.get("points") or () if p.get("at") is not None]
    if not times:
        return f"<p>{esc(label)}: {NR}.</p>"
    t0, t1 = (min(times), max(max(times), now)) if window is None else window
    pad = max(60.0, (t1 - t0) * 0.01)
    t0, t1 = t0 - pad, t1 + pad
    width, left, right, top, lane_h = 960, 190, 944, 6, 18
    height = top + lane_h * len(lanes) + 28
    bottom = top + lane_h * len(lanes)
    x = Scale(t0, t1, left, right)
    parts = [_x_time_axis(x, t0, t1, bottom)]
    drawn = 0
    for i, lane in enumerate(lanes):
        y = top + i * lane_h
        parts.append(_line(left, y + lane_h, right, y + lane_h, 'stroke="var(--grid)"'))
        parts.append(_text(left - 6, y + 13, str(lane["lane"])[:30], "end"))
        for span in lane.get("spans") or ():
            start = span.get("start")
            if start is None or drawn >= max_marks:
                continue
            end = span.get("end")
            running = end is None
            end = now if running else end
            if end < t0 or start > t1:
                continue
            a, b = x(max(start, t0)), x(min(end, t1))
            colour = colours.get(span.get("kind"), OTHER)
            w = max(1.5, b - a)
            style = (
                f'fill="none" stroke="{colour}" stroke-width="1.5" stroke-dasharray="3 2"'
                if running
                else f'fill="{colour}"'
            )
            tip = (span.get("tip") or "") + ("\nRUNNING (no end recorded)" if running else "")
            rect = f'<rect x="{a:.1f}" y="{y + 3}" width="{w:.1f}" height="{lane_h - 6}" rx="2" {style}/>'
            parts.append(_tip(tip, rect))
            drawn += 1
        for point in lane.get("points") or ():
            at = point.get("at")
            if at is None or not t0 <= at <= t1 or drawn >= max_marks:
                continue
            px, colour = x(at), colours.get(point.get("kind"), OTHER)
            diamond = (
                f'<path d="M{px:.1f} {y + 2}L{px + 5:.1f} {y + 9}L{px:.1f} {y + 16}L{px - 5:.1f} {y + 9}Z" '
                f'fill="{colour}" stroke="var(--surface)" stroke-width="1"/>'
            )
            parts.append(_tip(point.get("tip") or "", diamond))
            drawn += 1
    if t0 <= now <= t1:
        parts.append(
            _line(x(now), top, x(now), bottom, 'stroke="var(--critical)" stroke-width="1.5" stroke-dasharray="4 3"')
        )
        parts.append(_text(x(now) - 3, top + 10, "now", "end"))
    note = f'<p class="sub">Drawn {drawn} marks (capped at {max_marks}).</p>' if drawn >= max_marks else ""
    return _svg(width, height, "".join(parts), label) + note


def gantt_legend(colours: Mapping[str, str], names: Mapping[str, str]) -> str:
    entries = [(colours[k], esc(names.get(k, k))) for k in colours if k in names]
    entries.append(("transparent", "dashed outline = running (no end recorded); &#9670; = an instant"))
    return legend(entries)


# --------------------------------------------------------------------------- treemap
def _squarify(values: list[float], x: float, y: float, w: float, h: float) -> list[tuple[float, float, float, float]]:
    """Squarified treemap layout of ``values`` (descending) into the rectangle; one box per value."""
    boxes: list[tuple[float, float, float, float]] = []
    total = sum(values)
    if total <= 0 or not values:
        return [(x, y, 0.0, 0.0) for _ in values]
    scale = w * h / total
    areas = [v * scale for v in values]
    row: list[float] = []

    def worst(row: list[float], side: float) -> float:
        s = sum(row)
        if s <= 0 or side <= 0:
            return float("inf")
        return max(max(side * side * r / (s * s), (s * s) / (side * side * r)) for r in row if r > 0)

    i = 0
    while i < len(areas):
        side = min(w, h)
        if not row or worst(row + [areas[i]], side) <= worst(row, side):
            row.append(areas[i])
            i += 1
            continue
        x, y, w, h = _place(row, x, y, w, h, boxes)
        row = []
    if row:
        _place(row, x, y, w, h, boxes)
    return boxes


def _place(row, x, y, w, h, boxes):
    s = sum(row)
    if w >= h:
        col = s / h if h else 0
        yy = y
        for r in row:
            hh = r / col if col else 0
            boxes.append((x, yy, col, hh))
            yy += hh
        return x + col, y, w - col, h
    rh = s / w if w else 0
    xx = x
    for r in row:
        ww = r / rh if rh else 0
        boxes.append((xx, y, ww, rh))
        xx += ww
    return x, y + rh, w, h - rh


def treemap(groups: Mapping[str, Mapping[str, int]], *, label: str, height: int = 380) -> str:
    """Two-level treemap: outer boxes are groups (coloured by identity), inner boxes their members.

    ``groups`` is ``{group: {member: count}}``.  Every box carries its count on hover and a label when
    there is room; the table beside it is the exact view."""
    flat = {g: {m: c for m, c in members.items() if c > 0} for g, members in groups.items()}
    flat = {g: m for g, m in flat.items() if m}
    if not flat:
        return f"<p>{esc(label)}: {NR}.</p>"
    order = sorted(flat, key=lambda g: -sum(flat[g].values()))
    colours = palette(order)
    width = 960
    outer = _squarify([float(sum(flat[g].values())) for g in order], 0, 0, width, height)
    parts = []
    for g, (gx, gy, gw, gh) in zip(order, outer, strict=True):
        members = sorted(flat[g].items(), key=lambda kv: -kv[1])
        inner = _squarify([float(c) for _, c in members], gx + 2, gy + 16, max(0, gw - 4), max(0, gh - 18))
        total = sum(c for _, c in members)
        parts.append(
            _tip(
                f"{g}: {total}",
                f'<rect x="{gx:.1f}" y="{gy:.1f}" width="{gw:.1f}" height="{gh:.1f}" fill="{colours[g]}" '
                f'fill-opacity=".18" stroke="var(--surface)" stroke-width="2"/>',
            )
        )
        if gw > 50 and gh > 14:
            parts.append(_text(gx + 4, gy + 12, f"{g} ({total})"[: int(gw / 6.5)]))
        for (m, c), (mx, my, mw, mh) in zip(members, inner, strict=True):
            box = (
                f'<rect x="{mx:.1f}" y="{my:.1f}" width="{max(0.0, mw):.1f}" height="{max(0.0, mh):.1f}" '
                f'fill="{colours[g]}" fill-opacity=".55" stroke="var(--surface)" stroke-width="1.5"/>'
            )
            text = _text(mx + 3, my + 12, f"{m} {c}"[: int(mw / 6.5)]) if mw > 36 and mh > 15 else ""
            parts.append(_tip(f"{g} / {m}: {c}", box + text))
    keys = [(colours[g], esc(g)) for g in order[: len(CATEGORICAL)]]
    if len(order) > len(CATEGORICAL):
        keys.append((OTHER, f"{len(order) - len(CATEGORICAL)} more (grey)"))
    return legend(keys) + _svg(width, height, "".join(parts), label)


# --------------------------------------------------------------------------- scatter and series
def scatter(
    points: Sequence[Mapping[str, Any]],
    *,
    x_label: str,
    y_label: str,
    label: str,
    categories: Sequence[str] = (),
    reference: float | None = None,
    reference_label: str = "",
    log_y: bool = False,
    log_x: bool = False,
) -> str:
    """Points ``{"x": number or category, "y", "series", "tip", "hollow"}``; categorical x when
    ``categories`` is given.  A hollow point is a single observation (no replicate)."""
    points = [p for p in points if isinstance(p.get("y"), int | float)]
    if not points:
        return f"<p>{esc(label)}: {NR}.</p>"
    series = palette([str(p.get("series") or "") for p in points])
    ys = [float(p["y"]) for p in points] + ([reference] if reference else [])
    if log_y:
        ys = [v for v in ys if v > 0] or [1.0]
    lo, hi = min(ys), max(ys)
    if log_y:
        lo, hi = math.log10(lo) - 0.05, math.log10(hi) + 0.05
    else:
        span = max(1e-9, hi - lo)
        lo, hi = min(0.0, lo - span * 0.05), hi + span * 0.08
    width, height, left, right, top, bottom = 960, 300, 64, 940, 12, 250
    if categories:
        x = Scale(-0.5, len(categories) - 0.5, left, right)
        index = {c: i for i, c in enumerate(categories)}
    else:
        xs = [float(p["x"]) for p in points if isinstance(p.get("x"), int | float)]
        if log_x:
            xs = [math.log10(v) for v in xs if v > 0] or [0.0]
        x = Scale(min(xs) - 0.05, max(xs) + 0.05, left, right)
        index = {}
    y = Scale(lo, hi, bottom, top)

    def ypos(v: float) -> float:
        if log_y:
            return y(math.log10(max(v, 1e-12)))
        return y(v)

    fmt: Callable[[float], str] = (lambda v: short(10**v)) if log_y else short
    parts = [_y_axis(y, nice_ticks(lo, hi, 5), left, right, fmt)]
    parts.append(_line(left, bottom, right, bottom, 'class="axis"'))
    if categories:
        for c, i in index.items():
            parts.append(_text(x(i), bottom + 14, c[:18], "middle"))
    else:
        for v in nice_ticks(x.d0, x.d1, 6):
            parts.append(_line(x(v), bottom, x(v), bottom + 4, 'class="axis"'))
            parts.append(_text(x(v), bottom + 16, short(10**v) if log_x else short(v), "middle"))
    parts.append(_text(left, top - 2 + 10, y_label))
    parts.append(_text(right, bottom + 30, x_label, "end"))
    if reference:
        yy = ypos(reference)
        parts.append(_line(left, yy, right, yy, 'stroke="var(--ink-2)" stroke-dasharray="6 4"'))
        parts.append(_text(right - 4, yy - 4, reference_label, "end"))
    for k, p in enumerate(points):
        if categories:
            i = index.get(str(p["x"]))
            if i is None:
                continue
            jitter = ((k * 7) % 11 - 5) * 2.0
            px = x(i) + jitter
        else:
            if not isinstance(p.get("x"), int | float) or (log_x and p["x"] <= 0):
                continue
            px = x(math.log10(p["x"])) if log_x else x(float(p["x"]))
        colour = series[str(p.get("series") or "")]
        fill = "var(--surface)" if p.get("hollow") else colour
        mark = (
            f'<circle cx="{px:.1f}" cy="{ypos(float(p["y"])):.1f}" r="4.5" fill="{fill}" stroke="{colour}" '
            'stroke-width="2"/>'
        )
        parts.append(_tip(p.get("tip") or "", mark))
    keys = [(c, esc(s or "value")) for s, c in series.items()]
    if any(p.get("hollow") for p in points):
        keys.append(("transparent", "hollow = single observation (no replicate)"))
    return legend(keys) + _svg(width, height + 20, "".join(parts), label)


def time_series(
    series: Mapping[str, Sequence[tuple[float, float, str]]],
    *,
    label: str,
    y_fmt: Callable[[float], str] = short,
    y_max: float | None = None,
    step: bool = False,
    now: float | None = None,
    height: int = 200,
) -> str:
    """One or more lines over time on ONE y scale: ``{name: [(epoch, value, tip), ...]}``."""
    series = {k: sorted(v) for k, v in series.items() if v}
    if not series:
        return f"<p>{esc(label)}: {NR}.</p>"
    colours = palette(list(series))
    times = [t for rows in series.values() for t, _, _ in rows] + ([now] if now else [])
    values = [v for rows in series.values() for _, v, _ in rows]
    t0, t1 = min(times), max(times)
    pad = max(60.0, (t1 - t0) * 0.02)
    t0, t1 = t0 - pad, t1 + pad
    hi = y_max if y_max is not None else max(values + [1e-9]) * 1.08
    lo = min(0.0, min(values))
    width, left, right, top = 960, 64, 940, 12
    bottom = height - 30
    x, y = Scale(t0, t1, left, right), Scale(lo, hi, bottom, top)
    parts = [_y_axis(y, nice_ticks(lo, hi, 5), left, right, y_fmt), _x_time_axis(x, t0, t1, bottom)]
    for name, rows in series.items():
        colour, path, prev = colours[name], [], None
        for t, v, _ in rows:
            if step and prev is not None:
                path.append(f"L{x(t):.1f} {y(prev):.1f}")
            path.append(f"{'M' if not path else 'L'}{x(t):.1f} {y(v):.1f}")
            prev = v
        parts.append(f'<path d="{" ".join(path)}" fill="none" stroke="{colour}" stroke-width="2"/>')
        if len(rows) <= 400:
            for t, v, tip in rows:
                hit = f'<circle cx="{x(t):.1f}" cy="{y(v):.1f}" r="6" fill="transparent"/>'
                dot = f'<circle cx="{x(t):.1f}" cy="{y(v):.1f}" r="2.5" fill="{colour}"/>'
                parts.append(_tip(tip or f"{name} {y_fmt(v)} at {R.stamp(t)}", hit + dot))
    if now is not None and t0 <= now <= t1:
        parts.append(_line(x(now), top, x(now), bottom, 'stroke="var(--critical)" stroke-dasharray="4 3"'))
    keys = legend([(colours[k], esc(k)) for k in series]) if len(series) > 1 else ""
    return keys + _svg(width, height, "".join(parts), label)


__all__ = ["CATEGORICAL", "OTHER", "gantt", "gantt_legend", "palette", "scatter", "time_series", "treemap"]
