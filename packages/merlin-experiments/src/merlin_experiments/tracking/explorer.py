"""Explore a target's runs: an index of every phase's runs, the lineage between them, and comparisons.

``dashboard --target T --explorer`` reads:

* Phase 0 releases from the target index (:func:`merlin.targetgen.target_index.build_index`) plus each
  release's ``private/preparation.json`` admission counts and ``private/seal.json`` reviewer (the review
  note only in operator-private mode), and the Phase 0 run directories under ``out/runs/<target>/phase0``;
* every Phase 1 run under ``out/runs/<target>/phase1`` through :mod:`.records` and its ``environment.yaml``
  (driver, model, level, the corpus review it was admitted under);
* every Phase 2 run under ``out/runs/<target>/phase2``: a paired checkpointed experiment through
  :mod:`.phase2_paired`, a whole-model measured run through :mod:`.records`.

LINEAGE IS WHAT THE RECORDS BIND, CHECKED.  A Phase 1 run links to the release whose seal digest its
``environment.yaml`` ``corpus_review`` names; a paired Phase 2 experiment links to the Phase 1 run its
campaign manifests name (``functional_run_id``) and is flagged when the submission digest it was bound to
differs from that run's ``freeze.json``; a whole-model Phase 2 run links to the Phase 1 run whose frozen
commit its history repository started from (the index's ``origin``); champions follow the index's own
lineage.  A link the records do not make is "not recorded"; a link they make inconsistently is drawn red.

``dashboard --compare A B`` puts two runs of one phase side by side (corpus diff, grade overlay and
per-capsule tier deltas, per-member cycle and roofline deltas).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from . import records as R
from .html import (
    NR,
    Scale,
    _line,
    _svg,
    _text,
    details,
    digest,
    esc,
    legend,
    mono,
    nice_ticks,
    num,
    page,
    table,
    text_or_nr,
    tile,
    when,
)

MAX_RUNS = 80


def _dirs(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted((p for p in root.iterdir() if p.is_dir() and not p.is_symlink()), reverse=True)[:MAX_RUNS]


def page_name(phase: str, run_id: str) -> str:
    """The sibling page a run's node links to (the dashboard's own default name for a run page)."""
    if phase == "0" and not run_id.startswith("phase0-"):
        return f"phase0-{run_id}.html"
    return f"{run_id}.html"


# --------------------------------------------------------------------------- rows
def release_rows(index: Mapping[str, Any] | None, *, operator_private: bool) -> list[dict[str, Any]]:
    from merlin.common import paths

    rows = []
    for entry in (index or {}).get("phase0_releases") or ():
        root = Path(str(entry.get("release") or ""))
        if not root.is_absolute():
            root = paths.out_dir() / root
        prepared = R.Inventory().json("preparation.json", root / "private" / "preparation.json") or {}
        sealed = R.Inventory().json("seal.json", root / "private" / "seal.json") or {}
        admission = prepared.get("admission") if isinstance(prepared.get("admission"), Mapping) else {}
        lineage = prepared.get("generation_lineage") if isinstance(prepared.get("generation_lineage"), Mapping) else {}
        review = sealed.get("review") if isinstance(sealed.get("review"), Mapping) else {}
        capsules = lineage.get("capsules")
        rows.append(
            {
                "phase": "0",
                "run_id": root.name,
                "path": str(root),
                "page": page_name("0", root.name),
                "state": entry.get("state"),
                "review_digest": entry.get("review_digest"),
                "sealed_at": entry.get("sealed_at"),
                "reviewed_by": review.get("reviewed_by"),
                "note": review.get("note") if operator_private else None,
                "source_run": entry.get("source_run"),
                "counts": {k: v for k, v in admission.items() if isinstance(v, int) and not isinstance(v, bool)},
                "capsules": len(capsules) if isinstance(capsules, list) else capsules,
                "coverage": {k: (v or {}).get("status") for k, v in (lineage.get("coverage") or {}).items()}
                if isinstance(lineage.get("coverage"), Mapping)
                else {},
            }
        )
    return rows


def phase0_run_rows(root: Path) -> list[dict[str, Any]]:
    from . import phase0

    rows = []
    for run in _dirs(root):
        found = phase0.discover(run)
        rows.append(
            {
                "phase": "0",
                "run_id": run.name,
                "path": str(run),
                "page": page_name("0", run.name),
                "derivations": len(found["derivations"]),
                "corpus": bool(found["corpus"]),
                "coverage": bool(found["coverage"]),
            }
        )
    return rows


def _level(env: Mapping[str, Any]) -> str | None:
    config = env.get("run_config") if isinstance(env.get("run_config"), Mapping) else {}
    try:
        from ..phase1.levels import level_for_phase1

        return level_for_phase1(
            {
                **config,
                **{k: env[k] for k in ("arm", "treatment") if env.get(k)},
                **({"bundle": env["bundle_id"]} if env.get("bundle_id") else {}),
            }
        )
    except Exception:  # noqa: BLE001 -- an unlabelled level is "not recorded", never a guess
        return None


def phase1_row(run_dir: Path, now: float) -> dict[str, Any]:
    summary = R.run_summary(run_dir, now=now)
    p1 = summary.get("phase1") or {}
    env = R.Inventory().yaml("environment.yaml", run_dir / "environment.yaml") or {}
    review = env.get("corpus_review") if isinstance(env.get("corpus_review"), Mapping) else {}
    freeze_doc = R.Inventory().json("freeze.json", run_dir / "freeze.json") or {}
    latest = p1.get("latest") or {}
    manifest = p1.get("manifest") or {}
    grades = p1.get("grades") or []
    return {
        "phase": "1",
        "run_id": summary["run_id"],
        "path": str(run_dir),
        "page": page_name("1", summary["run_id"]),
        "state": (summary.get("liveness") or {}).get("state"),
        "level": _level(env),
        "driver": env.get("driver"),
        "model": env.get("model"),
        "started": R.epoch(env.get("started_at")) or R.run_started(run_dir, R.Inventory()),
        "ended": (p1.get("freeze") or {}).get("at"),
        "latest": f"{latest['n_passed']}/{latest['n_capsules']}" if latest.get("n_passed") is not None else None,
        "grades": grades,
        "public": (manifest.get("public") or {}).get("passed"),
        "hidden": (manifest.get("hidden") or {}).get("passed"),
        "frozen_commit": (p1.get("freeze") or {}).get("frozen_commit"),
        "submission_sha256": freeze_doc.get("submission_sha256"),
        "corpus_review_digest": review.get("review_digest"),
        "corpus_release": review.get("release"),
        "tokens": None,
    }


def phase2_row(run_dir: Path, now: float, *, operator_private: bool) -> dict[str, Any]:
    from . import phase2_paired
    from .tail import Tails

    if phase2_paired.is_experiment_root(run_dir):
        paired = phase2_paired.summary(run_dir, R.Inventory(), Tails(), operator_private=operator_private)
        ratios = [
            t.get("geometric_mean_speedup")
            for t in (paired.get("statistics") or {}).get("per_trial") or ()
            if isinstance(t, Mapping)
        ]
        if not ratios:
            # Not sealed yet: the geometric mean of the recorded pair ratios per trial, labelled as such.
            from .phase2_views import _geomean

            per_trial: dict[str, list[float]] = {}
            for key, cell in paired["cells"].items():
                for member in cell["members"].values():
                    per_trial.setdefault(key.partition(":")[0], []).extend(
                        p.get("baseline_over_candidate") for p in member.get("pairs") or ()
                    )
            ratios = [_geomean(v) for v in per_trial.values()]
        complete = sum(1 for c in paired["cells"].values() if (c.get("completion") or {}).get("complete"))
        return {
            "phase": "2",
            "kind": "paired",
            "run_id": run_dir.name,
            "path": str(run_dir),
            "page": page_name("2", run_dir.name),
            "state": "sealed" if paired["sealed"] else f"{len(paired['done'])}/{len(paired['expected'])} checkpoints",
            "trials": len(paired["trials"]),
            "cells": f"{complete}/{len(paired['trials']) * len(paired['labels'])}",
            "best_ratio": max((r for r in ratios if isinstance(r, int | float)), default=None),
            "sealed": paired["sealed"],
            "functional": paired.get("functional") or [],
            "paired": paired,
        }
    summary = R.run_summary(run_dir, now=now)
    p2 = summary.get("phase2") or {}
    return {
        "phase": "2",
        "kind": "whole_model",
        "run_id": summary["run_id"],
        "path": str(run_dir),
        "page": page_name("2", summary["run_id"]),
        "state": (summary.get("liveness") or {}).get("state"),
        "trials": None,
        "cells": None,
        "best_cycles": (p2.get("best") or {}).get("cycles"),
        "best_ratio": None,
        "sealed": None,
        "functional": [],
    }


def collect(target: str, now: float, *, operator_private: bool) -> dict[str, Any]:
    from merlin.common import paths
    from merlin.targetgen import target_index

    problems: list[dict[str, Any]] = []
    try:
        index = target_index.build_index(target)
    except Exception as exc:  # noqa: BLE001 -- an unreadable index is shown, never fatal
        index, problems = None, [{"path": "INDEX", "problem": f"{type(exc).__name__}: {exc}"}]
    rows: dict[str, list[dict[str, Any]]] = {"releases": release_rows(index, operator_private=operator_private)}
    rows["0"] = phase0_run_rows(paths.phase_runs_root(target, 0))
    for phase, reader in (("1", phase1_row), ("2", None)):
        out = []
        for run in _dirs(paths.phase_runs_root(target, phase)):
            try:
                out.append(phase1_row(run, now) if reader else phase2_row(run, now, operator_private=operator_private))
            except Exception as exc:  # noqa: BLE001 -- one unreadable run never hides the others
                out.append(
                    {
                        "phase": phase,
                        "run_id": run.name,
                        "path": str(run),
                        "page": None,
                        "problem": f"{type(exc).__name__}: {exc}",
                    }
                )
        rows[phase] = out
    return {
        "target": target,
        "index": index,
        "rows": rows,
        "problems": problems + list((index or {}).get("problems") or []),
    }


# --------------------------------------------------------------------------- lineage
def lineage(collected: Mapping[str, Any]) -> dict[str, Any]:
    """Nodes per column and the edges the records bind, each marked ok, mismatch or broken."""
    rows = collected["rows"]
    index = collected.get("index") or {}
    releases = rows.get("releases") or []
    p1 = [r for r in rows.get("1") or () if not r.get("problem")]
    p2 = [r for r in rows.get("2") or () if not r.get("problem")]
    edges: list[dict[str, Any]] = []
    notes: list[dict[str, Any]] = []
    by_digest = {r.get("review_digest"): r for r in releases if r.get("review_digest")}
    by_path = {str(Path(r["path"]).resolve()): r for r in releases}
    for run in p1:
        digest_value, release_path = run.get("corpus_review_digest"), run.get("corpus_release")
        if digest_value in by_digest:
            edges.append(
                {
                    "from": ("0", by_digest[digest_value]["run_id"]),
                    "to": ("1", run["run_id"]),
                    "state": "ok",
                    "label": f"seal {str(digest_value)[:12]}",
                }
            )
        elif release_path and str(Path(release_path).resolve()) in by_path:
            release = by_path[str(Path(release_path).resolve())]
            edges.append(
                {
                    "from": ("0", release["run_id"]),
                    "to": ("1", run["run_id"]),
                    "state": "mismatch",
                    "label": f"run names seal {str(digest_value)[:12]}, release has "
                    f"{str(release.get('review_digest'))[:12]}",
                }
            )
        else:
            notes.append(
                {
                    "run": run["run_id"],
                    "phase": "1",
                    "state": "not recorded" if not digest_value else "broken",
                    "detail": "no corpus seal recorded in environment.yaml"
                    if not digest_value
                    else f"seal {str(digest_value)[:12]} matches no release of this target",
                }
            )
    p1_by_id = {r["run_id"]: r for r in p1}
    frozen = {r.get("frozen_commit"): r for r in p1 if r.get("frozen_commit")}
    origins = {Path(str(b.get("run") or "")).name: (b.get("origin") or {}) for b in index.get("phase2_best") or ()}
    for run in p2:
        if run.get("kind") == "paired":
            if not run["functional"]:
                notes.append(
                    {
                        "run": run["run_id"],
                        "phase": "2",
                        "state": "not recorded",
                        "detail": "no campaign manifest names a functional run yet",
                    }
                )
            for binding in run["functional"]:
                source = p1_by_id.get(binding.get("run_id"))
                if source is None:
                    notes.append(
                        {
                            "run": run["run_id"],
                            "phase": "2",
                            "state": "broken",
                            "detail": f"functional run {binding.get('run_id')!r} is not a Phase 1 run of this target",
                        }
                    )
                    continue
                want, have = binding.get("submission_sha256"), source.get("submission_sha256")
                state = "ok" if want and want == have else "mismatch" if want and have else "not recorded"
                edges.append(
                    {
                        "from": ("1", source["run_id"]),
                        "to": ("2", run["run_id"]),
                        "state": state,
                        "label": f"submission {str(want)[:12]}"
                        + ("" if state == "ok" else f" vs freeze {str(have)[:12]}"),
                    }
                )
        else:
            origin = origins.get(run["run_id"]) or {}
            commit = origin.get("commit") if isinstance(origin, Mapping) else None
            if commit in frozen:
                edges.append(
                    {
                        "from": ("1", frozen[commit]["run_id"]),
                        "to": ("2", run["run_id"]),
                        "state": "ok",
                        "label": f"frozen {str(commit)[:12]}",
                    }
                )
            else:
                notes.append(
                    {
                        "run": run["run_id"],
                        "phase": "2",
                        "state": "not recorded",
                        "detail": "no recorded origin matches a Phase 1 frozen commit",
                    }
                )
    champions = []
    for row in index.get("champions") or ():
        name = str(row.get("package_id") or "champion")
        champions.append({"run_id": name, "page": None, "state": "champion"})
        source = Path(str((row.get("lineage") or {}).get("phase2_run") or "")).name
        if source and any(r["run_id"] == source for r in p2):
            edges.append({"from": ("2", source), "to": ("c", name), "state": "ok", "label": "exported"})
    columns = {"0": releases, "1": p1, "2": p2, "c": champions}
    return {"columns": columns, "edges": edges, "notes": notes}


_EDGE = {"ok": "var(--s1)", "mismatch": "var(--critical)", "not recorded": "var(--muted)"}


def lineage_svg(graph: Mapping[str, Any]) -> str:
    columns = graph["columns"]
    titles = {"0": "Phase 0 releases", "1": "Phase 1 runs", "2": "Phase 2 runs", "c": "Champions"}
    order = ["0", "1", "2", "c"]
    width, col_w, node_w, node_h, gap, top = 960, 240, 210, 34, 10, 26
    height = top + max(1, max(len(columns[c]) for c in order)) * (node_h + gap) + 10
    position: dict[tuple[str, str], tuple[float, float]] = {}
    parts = []
    for i, col in enumerate(order):
        x = 10 + i * col_w
        parts.append(_text(x, 14, titles[col]))
        for j, node in enumerate(columns[col]):
            y = top + j * (node_h + gap)
            position[(col, node["run_id"])] = (x, y)
            fill = "var(--surface-2)"
            label = esc(str(node["run_id"])[:30])
            sub = esc(str(node.get("state") or "")[:30])
            box = (
                f'<rect x="{x}" y="{y}" width="{node_w}" height="{node_h}" rx="5" fill="{fill}" '
                f'stroke="var(--border)"/><text x="{x + 6}" y="{y + 14}">{label}</text>'
                f'<text x="{x + 6}" y="{y + 28}">{sub}</text>'
            )
            tip = f"{node['run_id']}\n{node.get('path') or ''}"
            if node.get("page"):
                box = f'<a href="{esc(node["page"])}">{box}</a>'
            parts.append(f'<g data-tip="{esc(tip)}">{box}</g>')
    for edge in graph["edges"]:
        a, b = position.get(edge["from"]), position.get(edge["to"])
        if not a or not b:
            continue
        x1, y1, x2, y2 = a[0] + node_w, a[1] + node_h / 2, b[0], b[1] + node_h / 2
        colour = _EDGE.get(edge["state"], "var(--muted)")
        dash = "" if edge["state"] == "ok" else ' stroke-dasharray="5 3"'
        width_px = 3 if edge["state"] == "mismatch" else 1.8
        parts.append(
            f'<g data-tip="{esc(edge["state"] + ": " + edge["label"])}"><path d="M{x1} {y1}C{x1 + 20} {y1} {x2 - 20} '
            f'{y2} {x2} {y2}" fill="none" stroke="{colour}" stroke-width="{width_px}"{dash}/></g>'
        )
    keys = legend(
        [
            ("var(--s1)", "binding consistent"),
            ("var(--critical)", "MISMATCH: the records disagree"),
            ("var(--muted)", "binding not recorded"),
        ]
    )
    return keys + _svg(width, height, "".join(parts), "lineage")


# --------------------------------------------------------------------------- the explorer page
def _sortable(headers: Sequence[str], rows: Sequence[Sequence[str]], numeric: Sequence[int] = ()) -> str:
    return table(headers, rows, numeric).replace("<table>", '<table class="sortable">', 1)


def explorer_body(collected: Mapping[str, Any], graph: Mapping[str, Any], *, operator_private: bool) -> str:
    rows = collected["rows"]

    def link(row: Mapping[str, Any]) -> str:
        text = mono(row["run_id"])
        return f'<a href="{esc(row["page"])}">{text}</a>' if row.get("page") else text

    releases = [
        [
            link(r),
            text_or_nr(r.get("state")),
            digest(r.get("review_digest")),
            text_or_nr(r.get("reviewed_by")),
            text_or_nr(r.get("sealed_at")),
            text_or_nr(r.get("capsules")),
            esc(", ".join(f"{k} {v}" for k, v in (r.get("counts") or {}).items())) or NR,
            esc(", ".join(f"{k}: {v}" for k, v in (r.get("coverage") or {}).items())) or NR,
        ]
        + ([text_or_nr(r.get("note"))] if operator_private else [])
        for r in rows["releases"]
    ]
    p0 = [
        [link(r), str(r["derivations"]), "yes" if r["corpus"] else "no", "yes" if r["coverage"] else "no"]
        for r in rows["0"]
    ]
    p1 = [
        [
            link(r),
            text_or_nr(r.get("state") or r.get("problem")),
            text_or_nr(r.get("level")),
            text_or_nr(r.get("driver")),
            text_or_nr(r.get("model")),
            when(r.get("started")),
            when(r.get("ended")),
            text_or_nr(r.get("latest")),
            text_or_nr(r.get("public")),
            text_or_nr(r.get("hidden")),
            digest(r.get("frozen_commit")),
            digest(r.get("submission_sha256")),
            digest(r.get("corpus_review_digest")),
        ]
        for r in rows["1"]
    ]
    p2 = [
        [
            link(r),
            text_or_nr(r.get("kind")),
            text_or_nr(r.get("state") or r.get("problem")),
            text_or_nr(r.get("trials")),
            text_or_nr(r.get("cells")),
            f"{r['best_ratio']:.3f}x" if isinstance(r.get("best_ratio"), int | float) else NR,
            num(r.get("best_cycles")),
            {True: "yes", False: "no"}.get(r.get("sealed"), NR),
            esc(", ".join(str(b.get("run_id")) for b in r.get("functional") or ())) or NR,
        ]
        for r in rows["2"]
    ]
    notes = [[esc(n["phase"]), mono(n["run"]), esc(n["state"]), esc(n["detail"])] for n in graph["notes"]]
    mismatches = [e for e in graph["edges"] if e["state"] == "mismatch"]
    alarm = (
        f'<p class="badge critical">{len(mismatches)} lineage mismatch(es): '
        + esc("; ".join(f"{e['from'][1]} -> {e['to'][1]}: {e['label']}" for e in mismatches))
        + "</p>"
        if mismatches
        else ""
    )
    filter_box = (
        '<p><input type="search" data-filter placeholder="filter rows (any column)" '
        'style="width:100%;max-width:420px;padding:4px 8px"></p>'
    )
    return (
        '<section id="lineage"><h2>Lineage across phases</h2>'
        + alarm
        + lineage_svg(graph)
        + details(f"Links not recorded or broken ({len(notes)})", table(["phase", "run", "state", "detail"], notes))
        + "</section>"
        + '<section id="index"><h2>All runs</h2>'
        + filter_box
        + "<h3>Phase 0 releases</h3>"
        + _sortable(
            [
                "release",
                "state",
                "review digest",
                "reviewed by",
                "sealed at",
                "capsules",
                "admission counts",
                "coverage",
            ]
            + (["review note"] if operator_private else []),
            releases,
            (5,),
        )
        + "<h3>Phase 0 runs</h3>"
        + _sortable(["run", "derivations", "corpus", "coverage"], p0, (1,))
        + "<h3>Phase 1 runs</h3>"
        + _sortable(
            [
                "run",
                "state",
                "level",
                "driver",
                "model",
                "started",
                "ended",
                "latest grade",
                "public",
                "hidden",
                "frozen commit",
                "submission",
                "corpus seal",
            ],
            p1,
        )
        + "<h3>Phase 2 runs</h3>"
        + _sortable(
            ["run", "kind", "state", "trials", "cells", "best ratio", "best cycles", "sealed", "functional run"],
            p2,
            (3, 5, 6),
        )
        + "</section>"
    )


def pages(opts, ctx, now: float) -> dict[str, str]:
    """The explorer page (or the comparison page) and every linked run page, by file name."""
    from . import views

    if opts.compare is not None:
        return {"": compare_page(opts.compare, now, operator_private=opts.operator_private)}
    from merlin.common import paths

    roots = [paths.phase_runs_root(opts.target, n) for n in (0, 1, 2)]
    runs = [run for root in roots for run in _dirs(root)]
    key = (
        tuple(views._listing(run) for run in runs),
        tuple(views._listing(r) for r in roots),
        int(now // views.CLOCK_BUCKET),
        opts.operator_private,
    )
    cached = ctx.tails.state.get("explorer_collected")
    if cached is None or cached[0] != key:
        cached = (key, collect(opts.target, now, operator_private=opts.operator_private))
        ctx.tails.state["explorer_collected"] = cached
        stamps = [s[1] for listing in key[0] for s in listing if isinstance(s[1], int)]
        if stamps:
            ctx.newest["explorer"] = max(stamps) / 1e9
    collected = cached[1]
    graph = lineage(collected)
    mode = "operator-private" if opts.operator_private else "public"
    tiles = (
        '<div class="tiles">'
        + "".join(
            tile(label, str(len(collected["rows"][key])), "")
            for label, key in (
                ("Phase 0 releases", "releases"),
                ("Phase 0 runs", "0"),
                ("Phase 1 runs", "1"),
                ("Phase 2 runs", "2"),
            )
        )
        + "</div>"
    )
    body = tiles + explorer_body(collected, graph, operator_private=opts.operator_private)
    if collected["problems"]:
        body += details(
            "Index problems",
            table(
                ["path", "problem"],
                [[text_or_nr(p.get("path")), text_or_nr(p.get("problem"))] for p in collected["problems"]],
            ),
        )
    sub = f"run explorer &middot; {mode} mode &middot; {views.freshness(ctx, now, opts.refresh)}"
    out = {
        "": page(
            f"Run explorer {opts.target}",
            f"Runs of {esc(opts.target)}",
            sub,
            [("lineage", "Lineage"), ("index", "All runs")],
            body,
            now,
        )
    }
    sub_contexts = ctx.tails.state.setdefault("explorer_pages", {})
    for phase in ("releases", "0", "1", "2"):
        for row in collected["rows"][phase]:
            if not row.get("page") or row.get("problem"):
                continue
            path = Path(row["path"])
            sub_opts = views.Options(
                operator_private=opts.operator_private, refresh=opts.refresh, stall_hours=opts.stall_hours
            )
            if phase in ("releases", "0"):
                sub_opts.phase0 = path / "payload" if phase == "releases" and (path / "payload").is_dir() else path
            else:
                sub_opts.run_dir = path
            sub_ctx = sub_contexts.setdefault(str(path), views.Context())
            try:
                out[row["page"]] = views.render(sub_opts, sub_ctx, now)
            except Exception as exc:  # noqa: BLE001 -- one unreadable run never hides the others
                out[row["page"]] = page(
                    row["run_id"],
                    esc(row["run_id"]),
                    "unreadable",
                    [],
                    f"<section><p>{esc(type(exc).__name__)}: {esc(exc)}</p></section>",
                    now,
                )
    return out


def page_html(opts, ctx, now: float) -> str:
    return pages(opts, ctx, now)[""]


# --------------------------------------------------------------------------- comparison
def _phase_of(path: Path) -> str:
    from . import phase0, phase2_paired

    if phase2_paired.is_experiment_root(path):
        return "2p"
    found = phase0.discover(path)
    if found["derivations"] or found["corpus"]:
        if not any((path / m).exists() for m in R.PHASE1_MARKERS):
            return "0"
    phases = R.run_summary(path).get("phases") or []
    return phases[0] if phases else "?"


def _overlay(series: Mapping[str, Sequence[tuple[float, float, str]]], label: str) -> str:
    """Lines against hours since each run's own start, on one shared y scale."""
    series = {k: v for k, v in series.items() if v}
    if not series:
        return f"<p>{esc(label)}: {NR}.</p>"
    from .charts import palette

    colours = palette(list(series))
    xs = [x for rows in series.values() for x, _, _ in rows]
    ys = [y for rows in series.values() for _, y, _ in rows]
    width, height, left, right, top, bottom = 960, 230, 64, 940, 12, 196
    x, y = Scale(0, max(xs + [1.0]), left, right), Scale(0, max(ys + [1.0]) * 1.08, bottom, top)
    parts = [
        f'<g class="grid">{"".join(_line(left, y(v), right, y(v), "") for v in nice_ticks(0, max(ys + [1.0])))}</g>'
    ]
    parts += [_text(left - 6, y(v) + 4, f"{v:g}", "end") for v in nice_ticks(0, max(ys + [1.0]))]
    parts += [_text(x(v), bottom + 16, f"+{v:g} h", "middle") for v in nice_ticks(0, max(xs + [1.0]))]
    for name, rows in series.items():
        path = " ".join(f"{'M' if i == 0 else 'L'}{x(a):.1f} {y(b):.1f}" for i, (a, b, _) in enumerate(rows))
        parts.append(f'<path d="{path}" fill="none" stroke="{colours[name]}" stroke-width="2"/>')
        for a, b, tip in rows:
            parts.append(
                f'<g data-tip="{esc(tip)}"><circle cx="{x(a):.1f}" cy="{y(b):.1f}" r="4" fill="{colours[name]}"/></g>'
            )
    return legend([(colours[k], esc(k)) for k in series]) + _svg(width, height, "".join(parts), label)


def compare_phase1(a: Path, b: Path, now: float) -> str:
    rows = {}
    for side, path in (("A", a), ("B", b)):
        rows[side] = phase1_row(path, now)
    series = {}
    for side, row in rows.items():
        grades = [g for g in row["grades"] if g.get("at") and g.get("n_passed") is not None]
        start = row.get("started") or (grades[0]["at"] if grades else 0)
        series[f"{side}: {row['run_id']}"] = [
            ((g["at"] - start) / 3600.0, g["n_passed"], f"{side} {g['name']}: {g['n_passed']}/{g.get('n_capsules')}")
            for g in grades
        ]
    latest = {
        side: {c["capsule"]: c for c in ((row["grades"] or [{}])[-1].get("capsules") or ())}
        for side, row in rows.items()
    }
    names = sorted(set(latest["A"]) | set(latest["B"]))
    from .records import tier_key

    tier_rows = []
    for name in names:
        ca, cb = latest["A"].get(name), latest["B"].get(name)
        ta, tb = (ca or {}).get("highest_pass"), (cb or {}).get("highest_pass")
        if ta == tb and (ca or {}).get("status") == (cb or {}).get("status"):
            change = "same"
        elif ca is None or cb is None:
            change = "only in " + ("B" if ca is None else "A")
        else:
            ka, kb = tier_key(ta) if ta else ("", -2, ""), tier_key(tb) if tb else ("", -2, "")
            change = "B higher" if kb > ka else "A higher" if ka > kb else "status differs"
        cls = {"B higher": "hit", "A higher": "gap"}.get(change, "")
        tier_rows.append(
            f"<tr><td>{mono(name)}</td><td>{text_or_nr((ca or {}).get('status'))} {text_or_nr(ta)}</td>"
            f"<td>{text_or_nr((cb or {}).get('status'))} {text_or_nr(tb)}</td>"
            f'<td class="{cls}">{esc(change)}</td></tr>'
        )
    costs = {}
    for side, path in (("A", a), ("B", b)):
        costs[side] = R.Inventory().yaml("cost", path / "cost_time_toolcalls.yaml") or {}
    cost_rows = [
        [esc(k), text_or_nr(costs["A"].get(k)), text_or_nr(costs["B"].get(k))]
        for k in (
            "tokens_total",
            "tokens_output",
            "tool_calls",
            "wall_time_seconds",
            "active_wall_s",
            "subscription_notional_usd",
        )
    ]
    summary = [
        [esc(k), text_or_nr(rows["A"].get(k)), text_or_nr(rows["B"].get(k))]
        for k in (
            "state",
            "level",
            "driver",
            "model",
            "latest",
            "public",
            "hidden",
            "frozen_commit",
            "submission_sha256",
            "corpus_review_digest",
        )
    ]
    return (
        "<h3>Runs</h3>"
        + table(["", "A", "B"], summary)
        + "<h3>Capsules passed against hours since each run's start</h3>"
        + _overlay(series, "grade overlay")
        + "<h3>Per-capsule tier, latest grade</h3><div class='scroll'><table><thead><tr><th>capsule</th><th>A</th>"
        f"<th>B</th><th>change</th></tr></thead><tbody>{''.join(tier_rows) or f'<tr><td colspan=4>{NR}</td></tr>'}"
        "</tbody></table></div>" + "<h3>Tokens and time</h3>" + table(["", "A", "B"], cost_rows)
    )


def compare_phase0(a: Path, b: Path, *, operator_private: bool) -> str:
    from . import phase0

    sa, sb = (phase0.summary(p, operator_private=operator_private) for p in (a, b))

    def rows_of(s: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
        corpus = s.get("corpus") or {}
        rows = list(corpus.get("rows") or []) if corpus else []
        if not rows:
            for d in s.get("derivations") or ():
                rows += d.get("planned_rows") or []
        return {f"{r.get('category')}/{r['name']}": r for r in rows}

    ra, rb = rows_of(sa), rows_of(sb)
    keys = ("family", "op", "form", "stratum", "epilogue", "dtype", "tier", "M", "K", "N", "window")
    changed = [k for k in sorted(set(ra) & set(rb)) if any(ra[k].get(f) != rb[k].get(f) for f in keys)]
    added, removed = sorted(set(rb) - set(ra)), sorted(set(ra) - set(rb))
    change_rows = [
        [
            mono(k),
            esc(", ".join(f"{f}: {ra[k].get(f)} -> {rb[k].get(f)}" for f in keys if ra[k].get(f) != rb[k].get(f))),
        ]
        for k in changed
    ]
    cov = []
    for phase in ("phase1", "phase2"):
        ca, cb = ((s.get("coverage") or {}).get(phase) or {} for s in (sa, sb))
        cov.append([esc(phase), "status", text_or_nr(ca.get("status")), text_or_nr(cb.get("status"))])
        for axis in ("cells", "shape_geometry", "conv_geometry", "epilogue", "groups", "scope"):
            ga = (
                (ca.get("cells") or {}).get("uncovered")
                if axis == "cells"
                else ((ca.get("axes") or {}).get(axis) or {}).get("uncovered")
            )
            gb = (
                (cb.get("cells") or {}).get("uncovered")
                if axis == "cells"
                else ((cb.get("axes") or {}).get(axis) or {}).get("uncovered")
            )
            if ga is None and gb is None:
                continue
            closed = sorted(set(ga or []) - set(gb or []))
            opened = sorted(set(gb or []) - set(ga or []))
            cov.append(
                [
                    esc(phase),
                    esc(axis),
                    f"{len(ga or [])} gaps",
                    f"{len(gb or [])} gaps; closed {esc(closed) or 0}, opened {esc(opened) or 0}",
                ]
            )
    hidden = [[text_or_nr(((s.get("corpus") or {}).get("hidden") or {}).get("n"))] for s in (sa, sb)]
    return (
        f"<p>A: {mono(sa['root'])}<br>B: {mono(sb['root'])}</p>"
        + '<div class="tiles">'
        + tile("Added in B", str(len(added)), "")
        + tile("Removed in B", str(len(removed)), "")
        + tile("Changed", str(len(changed)), "")
        + tile("Hidden (counts)", f"{hidden[0][0]} / {hidden[1][0]}", "A / B")
        + "</div>"
        + details(f"Added ({len(added)})", "<br>".join(mono(k) for k in added) or NR)
        + details(f"Removed ({len(removed)})", "<br>".join(mono(k) for k in removed) or NR)
        + details(f"Changed ({len(changed)})", table(["capsule", "change"], change_rows))
        + "<h3>Coverage deltas</h3>"
        + table(["cohort", "axis", "A", "B"], cov)
    )


def compare_phase2(a: Path, b: Path, *, operator_private: bool) -> str:
    from . import phase2_views
    from .phase2_paired import summary as paired_summary
    from .tail import Tails

    sa, sb = (paired_summary(p, R.Inventory(), Tails(), operator_private=operator_private) for p in (a, b))

    def ratios(s: Mapping[str, Any]) -> dict[str, float]:
        out: dict[str, list[float]] = {}
        for key, cell in s["cells"].items():
            label = key.partition(":")[2]
            names = phase2_views._member_names(s, label)
            for member, row in cell["members"].items():
                for p in row.get("pairs") or ():
                    if isinstance(p.get("baseline_over_candidate"), int | float):
                        out.setdefault(f"{label}: {names.get(member, member)}", []).append(p["baseline_over_candidate"])
        return {k: phase2_views._geomean(v) for k, v in out.items()}

    def roofline(s: Mapping[str, Any]) -> dict[str, float]:
        out: dict[str, list[float]] = {}
        for stage in s["stages"].values():
            latest = {}
            for row in stage.get("feedback") or ():
                latest[row["member"]] = row
            for member, row in latest.items():
                if isinstance(row.get("candidate_over_roofline"), int | float):
                    out.setdefault(member, []).append(row["candidate_over_roofline"])
        return {k: sum(v) / len(v) for k, v in out.items()}

    ra, rb = ratios(sa), ratios(sb)
    rows = []
    for key in sorted(set(ra) | set(rb)):
        va, vb = ra.get(key), rb.get(key)
        delta = (vb / va - 1.0) if va and vb else None
        cls = "hit" if delta and delta > 0 else "gap" if delta and delta < 0 else ""
        rows.append(
            f"<tr><td>{mono(key)}</td><td>{f'{va:.3f}x' if va else NR}</td><td>{f'{vb:.3f}x' if vb else NR}</td>"
            f'<td class="{cls}">{f"{delta:+.1%}" if delta is not None else NR}</td></tr>'
        )
    fa, fb = roofline(sa), roofline(sb)
    roof = [
        [
            mono(k),
            f"{fa[k]:.3f}" if k in fa else NR,
            f"{fb[k]:.3f}" if k in fb else NR,
            f"{fb[k] - fa[k]:+.3f}" if k in fa and k in fb else NR,
        ]
        for k in sorted(set(fa) | set(fb))
    ]
    return (
        "<h3>Per member: baseline/candidate ratio (geometric mean over trials and replicates)</h3>"
        "<div class='scroll'><table><thead><tr><th>member</th><th>A</th><th>B</th><th>B vs A</th></tr></thead>"
        f"<tbody>{''.join(rows) or f'<tr><td colspan=4>{NR}</td></tr>'}</tbody></table></div>"
        "<h3>Roofline position (candidate / roofline, lower is closer)</h3>"
        + table(["member", "A", "B", "B - A"], roof, (1, 2, 3))
    )


def compare_page(paths_: tuple[Path, Path], now: float, *, operator_private: bool) -> str:
    from ..spec import SpecError

    a, b = (Path(p).expanduser() for p in paths_)
    for p in (a, b):
        if not p.is_dir():
            raise SpecError(f"not a run directory: {p}")
    pa, pb = _phase_of(a), _phase_of(b)
    if pa != pb or pa == "?":
        raise SpecError(f"--compare needs two runs of one phase; got phase {pa} and phase {pb}")
    if pa == "0":
        body = compare_phase0(a, b, operator_private=operator_private)
    elif pa == "1":
        body = compare_phase1(a, b, now)
    elif pa == "2p":
        body = compare_phase2(a, b, operator_private=operator_private)
    else:
        sa, sb = (R.run_summary(p, now=now) for p in (a, b))
        rows = [
            [
                esc(k),
                text_or_nr(((sa.get("phase2") or {}).get("best") or {}).get(k)),
                text_or_nr(((sb.get("phase2") or {}).get("best") or {}).get(k)),
            ]
            for k in ("cycles", "key")
        ]
        body = table(["best", "A", "B"], rows)
    mode = "operator-private" if operator_private else "public"
    section = f'<section id="compare"><h2>Compare {esc(a.name)} with {esc(b.name)}</h2>{body}</section>'
    return page(
        f"Compare {a.name} {b.name}",
        f"Compare: {esc(a.name)} vs {esc(b.name)}",
        f"phase {esc(pa.rstrip('p'))} &middot; {mode} mode",
        [("compare", "Compare")],
        section,
        now,
    )


__all__ = ["collect", "compare_page", "lineage", "lineage_svg", "page_html", "pages"]
