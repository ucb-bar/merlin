"""Assemble the dashboard pages from the readers, with per-section reuse for ``--live``.

One :class:`Context` lives across a live view's refreshes: it keeps the JSONL tails (byte offsets), the
parsed compiler snapshots, and each section's rendered HTML keyed by the stat signature of the files it
reads.  A section whose inputs did not change is not re-read or re-rendered; a one-shot page uses a
fresh context and so reads everything once.  Sections whose drawing depends on the clock (a "now"
line, an age) also key on a coarse time bucket.

The page states its own freshness -- "data as of <newest record>, generated <now>, refreshed every N s"
-- so a stalled run reads differently from a stalled viewer.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import records as R
from .tail import Tails, signature

#: Seconds per time bucket for sections that draw the clock (a "now" line, an age).
CLOCK_BUCKET = 300.0


@dataclass
class Options:
    """What a dashboard page shows; one of ``run_dir``, ``target`` or ``phase0`` is the subject."""

    run_dir: Path | None = None
    target: str | None = None
    phase0: Path | None = None
    explorer: bool = False
    compare: tuple[Path, Path] | None = None
    monitor: Path | None = None
    load: Path | None = None
    corpus: tuple[Path, ...] = ()
    measurement_root: Path | None = None
    stage_root: Path | None = None
    store: Path | None = None
    operator_private: bool = False
    stall_hours: float = R.DEFAULT_STALL_HOURS
    refresh: float | None = None

    def inputs(self) -> list[Path]:
        """Directories this page reads (an output path must never be inside one)."""
        out = [self.run_dir, self.phase0, self.measurement_root, self.stage_root, self.store, *self.corpus]
        if self.explorer and self.target:
            from merlin.common import paths

            out += [paths.phase_runs_root(self.target, n) for n in (0, 1, 2)]
        if self.compare:
            out += list(self.compare)
        return [Path(p) for p in out if p is not None]


@dataclass
class Context:
    """State kept between refreshes of one live view (a one-shot page uses a fresh one)."""

    tails: Tails = field(default_factory=Tails)
    sections: dict[str, tuple[Any, str]] = field(default_factory=dict)
    newest: dict[str, float] = field(default_factory=dict)
    reused: int = 0
    rendered: int = 0

    def section(self, name: str, key: Any, build: Callable[[], str], paths: Iterable[Path] = ()) -> str:
        """``build()``'s HTML, reused while ``key`` is unchanged; ``paths`` feed the "data as of" stamp."""
        cached = self.sections.get(name)
        if cached is not None and cached[0] == key:
            self.reused += 1
            return cached[1]
        html = build()
        self.sections[name] = (key, html)
        self.rendered += 1
        newest = max((s[1] for s in signature(paths)), default=None)
        if newest is not None:
            self.newest[name] = newest / 1e9
        return html


def _listing(directory: Path, *, depth: int = 1) -> tuple:
    """A cheap signature of a directory: its entries' (name, mtime, size), ``depth`` levels down."""
    out = []
    try:
        entries = list(os.scandir(directory))
    except OSError:
        return ()
    for entry in entries:
        try:
            stat = entry.stat(follow_symlinks=False)
        except OSError:
            continue
        out.append((entry.name, stat.st_mtime_ns, stat.st_size))
        if depth > 1 and entry.is_dir(follow_symlinks=False):
            out.append((entry.name, _listing(Path(entry.path), depth=depth - 1)))
    return tuple(sorted(out, key=str))


def freshness(ctx: Context, now: float, refresh: float | None) -> str:
    from .html import esc

    newest = max(ctx.newest.values(), default=None)
    data = f"data as of {R.stamp(newest)} (newest record read)" if newest else "no record time read"
    text = f"{data}; page generated {R.stamp(now)}"
    if refresh:
        text += f"; refreshed every {refresh:g} s (live)"
    return f'<span class="mode">{esc(text)}</span>'


# --------------------------------------------------------------------------- run page
def run_page(opts: Options, ctx: Context, now: float) -> str:
    from . import activity, activity_html, compiler, phase1_detail, phase1_views, phase2_paired
    from . import html as H

    run_dir = Path(opts.run_dir).expanduser()
    bucket = int(now // CLOCK_BUCKET)
    inventory = R.Inventory()
    if phase2_paired.is_experiment_root(run_dir):
        return paired_page(opts, ctx, now)
    base_key = (_listing(run_dir), _listing(run_dir / "qa_history"), _listing(run_dir / "stage", depth=2), bucket)
    summary_holder: dict[str, Any] = {}

    def base() -> str:
        summary = R.run_summary(run_dir, now=now, stall_hours=opts.stall_hours, store=opts.store)
        summary_holder["summary"] = summary
        sections, nav = H.run_body(summary)
        ctx.tails.state["base_nav"] = nav
        ctx.tails.state["base_summary"] = summary
        return "".join(sections)

    base_html = ctx.section("base", base_key, base, paths=[run_dir / "qa_history", run_dir / "oot_commits.jsonl"])
    summary = summary_holder.get("summary") or ctx.tails.state["base_summary"]
    nav = list(ctx.tails.state["base_nav"])
    sections = [base_html]
    p1 = summary.get("phase1")
    streams = (
        sorted((run_dir / "rounds").glob("round_*.codex_events.timestamped.jsonl"))
        if (run_dir / "rounds").is_dir()
        else []
    )
    stream_key = signature(streams)
    if p1 is not None or streams:
        act_inventory = R.Inventory()
        acts = ctx.tails.state.get("activity_snapshot")
        if acts is None or ctx.tails.state.get("activity_key") != stream_key:
            acts = activity.read(run_dir, act_inventory, ctx.tails)
            ctx.tails.state["activity_snapshot"] = acts
            ctx.tails.state["activity_key"] = stream_key
            ctx.tails.state["activity_inventory"] = act_inventory.rows
        inventory.rows.extend(ctx.tails.state.get("activity_inventory") or [])
        sections.append(
            ctx.section("activity", (stream_key, bucket), lambda: activity_html.section(acts, now), paths=streams)
        )
        nav.append(("activity", "Agent activity"))
        detail_inventory = R.Inventory()
        env_ws = ctx.tails.state.get("workspace")
        channel_key = tuple(_listing(p) for p in phase1_detail.channel_dirs(run_dir, env_ws))
        detail_key = (
            base_key,
            channel_key,
            signature(
                [
                    run_dir / n
                    for n in (
                        "selfcheck_log.jsonl",
                        "cost_time_toolcalls.yaml",
                        "timing_detailed.json",
                        "qa_loop_state.yaml",
                        "environment.yaml",
                        "freeze.json",
                        "qa_loop_summary.yaml",
                    )
                ]
            ),
        )
        detail = ctx.tails.state.get("detail")
        if detail is None or ctx.tails.state.get("detail_key") != detail_key:
            detail = phase1_detail.read(run_dir, p1 or {}, detail_inventory, ctx.tails, now=now, corpus=opts.corpus)
            ctx.tails.state.update(
                detail=detail,
                detail_key=detail_key,
                detail_inventory=detail_inventory.rows,
                workspace=detail["environment"]["workspace"],
            )
        inventory.rows.extend(ctx.tails.state.get("detail_inventory") or [])
        sections.append(
            ctx.section(
                "timeline",
                (stream_key, detail_key, bucket),
                lambda: phase1_views.timeline_section(acts, detail, now),
                paths=[run_dir / "selfcheck_log.jsonl"],
            )
        )
        nav.append(("timeline", "Timeline"))
        commits_key = signature([run_dir / "oot_commits.jsonl", run_dir / "submission"])

        def evolution() -> str:
            cache = ctx.tails.state.setdefault("compiler_cache", {})
            evo_inventory = R.Inventory()
            evo = compiler.evolution(run_dir, (p1 or {}).get("oot_commits"), evo_inventory, cache)
            ctx.tails.state["compiler_inventory"] = evo_inventory.rows
            return phase1_views.compiler_section(evo)

        sections.append(ctx.section("compiler", commits_key, evolution, paths=[run_dir / "oot_commits.jsonl"]))
        inventory.rows.extend(ctx.tails.state.get("compiler_inventory") or [])
        nav.append(("compiler", "Compiler"))
        sections.append(
            ctx.section("families", (base_key, detail_key), lambda: phase1_views.family_section(detail, now))
        )
        nav.append(("families", "Families"))
        sections.append(ctx.section("cost", (stream_key, detail_key), lambda: phase1_views.cost_section(detail, acts)))
        nav.append(("cost", "Tokens and time"))
    sections += _operator_sections(opts, ctx, now, nav, inventory)
    sections.append(H.inventory_section(list(summary.get("inventory") or []) + inventory.rows))
    nav.append(("records", "Records"))
    live = summary.get("liveness") or {}
    phases = ", ".join(summary.get("phases") or []) or "not recorded"
    sub = (
        f"target {H.text_or_nr(summary.get('target'))} &middot; phase {H.esc(phases)} &middot; "
        f"{H.badge(live.get('state'))} {H.esc(live.get('detail') or '')} &middot; {freshness(ctx, now, opts.refresh)}"
    )
    return H.page(
        f"Experiment dashboard {summary['run_id']}", H.esc(summary["run_id"]), sub, nav, "".join(sections), now
    )


def _operator_sections(opts: Options, ctx: Context, now: float, nav: list, inventory: R.Inventory) -> list[str]:
    from . import resources as RS

    out = []
    if opts.monitor is not None:
        key = (signature([opts.monitor]), int(now // CLOCK_BUCKET))

        def monitor() -> str:
            inv = R.Inventory()
            notes = RS.monitor_notes(opts.monitor, inv)
            ctx.tails.state["monitor_inventory"] = inv.rows
            return RS.monitor_section(notes, now, requested=True)

        out.append(ctx.section("monitor", key, monitor, paths=[opts.monitor]))
        inventory.rows.extend(ctx.tails.state.get("monitor_inventory") or [])
        nav.append(("monitor", "Monitor"))
    if opts.load is not None:
        key = (signature([opts.load]), int(now // CLOCK_BUCKET))

        def load() -> str:
            inv = R.Inventory()
            samples = RS.load_samples(opts.load, inv)
            ctx.tails.state["load_inventory"] = inv.rows
            return RS.resource_section(samples, now, requested=True)

        out.append(ctx.section("resources", key, load, paths=[opts.load]))
        inventory.rows.extend(ctx.tails.state.get("load_inventory") or [])
        nav.append(("resources", "Resources"))
    return out


def paired_page(opts: Options, ctx: Context, now: float) -> str:
    from . import html as H
    from . import phase2_paired, phase2_views

    root = Path(opts.run_dir).expanduser()
    key = (
        _listing(root),
        _listing(root / "state"),
        _listing(opts.measurement_root, depth=2) if opts.measurement_root else (),
        _listing(opts.stage_root, depth=2) if opts.stage_root else (),
        int(now // CLOCK_BUCKET),
    )
    inventory = R.Inventory()
    nav = [("paired", "Phase 2 paired")]

    def build() -> str:
        inv = R.Inventory()
        paired = phase2_paired.summary(
            root,
            inv,
            ctx.tails,
            operator_private=opts.operator_private,
            measurement_root=opts.measurement_root,
            stage_root=opts.stage_root,
        )
        ctx.tails.state["paired_inventory"] = inv.rows
        return phase2_views.section(paired, now)

    sections = [ctx.section("paired", key, build, paths=[root / "state"])]
    inventory.rows.extend(ctx.tails.state.get("paired_inventory") or [])
    sections += _operator_sections(opts, ctx, now, nav, inventory)
    sections.append(H.inventory_section(inventory.rows))
    nav.append(("records", "Records"))
    sub = f"paired Phase 2 experiment &middot; {freshness(ctx, now, opts.refresh)}"
    return H.page(f"Phase 2 experiment {root.name}", H.esc(root.name), sub, nav, "".join(sections), now)


def phase0_page(opts: Options, ctx: Context, now: float) -> str:
    from . import phase0, phase0_html

    root = Path(opts.phase0).expanduser()
    key = (_listing(root, depth=3), opts.operator_private)
    html = ctx.section(
        "phase0",
        key,
        lambda: phase0_html.render(
            phase0.summary(root, operator_private=opts.operator_private, now=now), status="{status}"
        ),
        paths=[root],
    )
    return html.replace("{status}", freshness(ctx, now, opts.refresh))


def target_page(opts: Options, ctx: Context, now: float) -> str:
    from . import html as H

    summary = R.target_summary(opts.target, now=now, stall_hours=opts.stall_hours)
    return H.render_target(summary)


def render(opts: Options, ctx: Context | None = None, now: float | None = None) -> str:
    """The page ``opts`` asks for."""
    import time

    ctx = ctx or Context()
    now = time.time() if now is None else now
    if opts.compare is not None or opts.explorer:
        from . import explorer

        return explorer.page_html(opts, ctx, now)
    if opts.phase0 is not None:
        return phase0_page(opts, ctx, now)
    if opts.target is not None:
        return target_page(opts, ctx, now)
    return run_page(opts, ctx, now)


def render_pages(opts: Options, ctx: Context | None = None, now: float | None = None) -> dict[str, str]:
    """``{"": the page, <file name>: a linked page}``: the explorer also renders every run it links."""
    import time

    ctx = ctx or Context()
    now = time.time() if now is None else now
    if opts.explorer and opts.compare is None:
        from . import explorer

        return explorer.pages(opts, ctx, now)
    return {"": render(opts, ctx, now)}


__all__ = ["Context", "Options", "freshness", "render", "render_pages"]
