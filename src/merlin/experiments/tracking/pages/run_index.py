"""Explore a target's runs: an index of every phase's runs, the lineage between them, and comparisons.

``dashboard --target T --explorer`` reads:

* Phase 0 releases from the target index (:func:`merlin.development.targetgen.packaging.target_index.build_index`) plus each
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

from merlin.experiments.tracking.pages.markup import (
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
    table,
    text_or_nr,
    tile,
    when,
)
from merlin.experiments.tracking.pages.run_page import page
from merlin.experiments.tracking.readers import records as R

# ---- explorer ----------------------------------------------------------------------------------

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
    from merlin.experiments.tracking.readers import test_generation as phase0

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
        from merlin.development.backend.session.treatments import level_for_phase1

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
    from merlin.experiments.tracking.readers import phase2_paired
    from merlin.experiments.tracking.readers.records import Tails

    if phase2_paired.is_experiment_root(run_dir):
        paired = phase2_paired.summary(run_dir, R.Inventory(), Tails(), operator_private=operator_private)
        ratios = [
            t.get("geometric_mean_speedup")
            for t in (paired.get("statistics") or {}).get("per_trial") or ()
            if isinstance(t, Mapping)
        ]
        if not ratios:
            # Not sealed yet: the geometric mean of the recorded pair ratios per trial, labelled as such.
            from merlin.experiments.tracking.pages.optimization import _geomean

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
    from merlin.development.targetgen.packaging import target_index

    problems: list[dict[str, Any]] = []
    try:
        index = target_index.build_index(target)
    except Exception as exc:  # noqa: BLE001 -- an unreadable index is shown, never fatal
        index, problems = None, [{"path": "INDEX", "problem": f"{type(exc).__name__}: {exc}"}]
    rows: dict[str, list[dict[str, Any]]] = {"releases": release_rows(index, operator_private=operator_private)}
    from merlin.common.paths import generation_run_roots

    rows["0"] = [row for root in generation_run_roots(target, "0") for row in phase0_run_rows(root)]
    for phase, reader in (("1", phase1_row), ("2", None)):
        out = []
        discovered = [run for root in generation_run_roots(target, phase) for run in _dirs(root)]
        for run in discovered:
            if phase == "1" and run.name == "optimization":
                continue
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
