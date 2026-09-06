#!/usr/bin/env python3
"""Build the agentic-run report: index the runs, extract their facts, rescue what is at risk.

    build_report.py index    [--config roots.yaml] [--out PATH]
    build_report.py facts    [--index PATH] [--target T]... [--out PATH]
    build_report.py rescue   [--config roots.yaml] [--dry-run]
    build_report.py select   [--facts PATH]

Everything the figures and the written report cite comes out of ``run_facts.json``, so a number can
never be typed into a caption and drift from the run that produced it.

The library does the reading; this script owns the things the library may not know -- where the runs
are, what the arms are called, and which targets this particular report is about.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from merlin.agentreport.availability import Availability                      # noqa: E402
from merlin.agentreport.capsule_time import read_capsule_timings, summarize   # noqa: E402
from merlin.agentreport.index import ArmSpec, RunRef, build_index             # noqa: E402
from merlin.agentreport.passes import read_passes                             # noqa: E402
from merlin.agentreport.phase2 import read_phase2                             # noqa: E402
from merlin.agentreport.spans import concurrency, read_spans                  # noqa: E402
from merlin.agentreport.tokens import METERED, NOTIONAL, read_tokens          # noqa: E402
from merlin.common.artifacts import new_product                               # noqa: E402
from merlin.common.paths import artifacts_dir, repo_root                      # noqa: E402

HERE = Path(__file__).resolve().parent
CONCERN = "agentic-report"


# --------------------------------------------------------------------------- config

@dataclass
class Config:
    roots: list[dict]
    arms: tuple[ArmSpec, ...]
    arm_meta: dict[str, dict]
    phases: dict[str, str]
    rescue_globs: list[str]

    @classmethod
    def load(cls, path: Path) -> "Config":
        doc = yaml.safe_load(path.read_text()) or {}
        arms, meta = [], {}
        for entry in doc.get("arms") or []:
            arms.append(ArmSpec(arm_id=str(entry["id"]), name=str(entry["name"]),
                                prefix=str(entry["prefix"]),
                                bundle_ids=tuple(entry.get("bundles") or ())))
            meta[str(entry["id"])] = {"label": entry.get("label") or entry["id"],
                                      "adds": list(entry.get("adds") or []),
                                      "name": entry["name"]}
        return cls(roots=list(doc.get("roots") or []), arms=tuple(arms), arm_meta=meta,
                   phases=dict(doc.get("phases") or {}),
                   rescue_globs=list(doc.get("rescue_globs") or []))

    def resolved_roots(self) -> list[tuple[Path, dict]]:
        out = []
        for entry in self.roots:
            p = Path(entry["path"])
            if not p.is_absolute():
                p = (repo_root() / p).resolve()
            out.append((p, entry))
        return out


# --------------------------------------------------------------------------- facts

def run_facts(ref: RunRef, *, want_capsule_time: bool) -> dict:
    """Every number one run contributes, each beside how it was obtained."""
    avail = Availability(dict(ref.availability.fields))
    out: dict = {"key": ref.key, "run_id": ref.run_id, "target": ref.target, "arm": ref.arm,
                 "phase": ref.phase, "bench": ref.bench, "root": str(ref.root),
                 "path": str(ref.path), "arm_conflict": ref.arm_conflict,
                 "bundle_id": ref.bundle_id, "driver": ref.driver,
                 "model_declared": ref.model, "started_at": ref.started_at,
                 "repo_sha": ref.repo_sha, "n_rounds": ref.n_rounds}

    tok = read_tokens(ref.path)
    avail.fields.update(tok.availability.fields)
    out.update({
        "model": tok.model or ref.model, "model_is_family_only": tok.model_is_family_only,
        "input_tokens": tok.input_tokens, "output_tokens": tok.output_tokens,
        "cache_read_tokens": tok.cache_read_tokens,
        "cache_creation_tokens": tok.cache_creation_tokens,
        "reasoning_tokens": tok.reasoning_tokens, "total_tokens": tok.total_tokens,
        "cached_share": tok.cached_share, "tool_calls": tok.tool_calls,
        "wall_s": tok.wall_s, "active_wall_s": tok.active_wall_s,
        "rate_limit_wait_s": tok.rate_limit_wait_s,
        "cost_kind": tok.cost_kind, "cost_usd": tok.cost_usd,
        "notional_usd": tok.notional_usd, "cost_reason": tok.cost_reason})

    series = read_passes(ref.path)
    avail.fields.update(series.availability.fields)
    best = series.best
    out.update({
        "passed": best[0] if best else None, "capsules": best[1] if best else None,
        "n_selfcheck_rows": series.n_rows, "n_no_denominator": series.n_no_denominator,
        "n_regressions": series.n_regressions,
        "pass_milestones": [{"t_s": round(p.t_s, 1), "n_passed": p.n_passed,
                             "n_capsules": p.n_capsules} for p in series.milestones()],
        "pass_wall_s": series.wall_s})

    if ref.phase == "phase2":
        p2 = read_phase2(ref.path)
        avail.fields.update(p2.availability.fields)
        spanset = p2.spanset
        out["broker_actions"] = p2.broker_actions
        out["broker_totals"] = {a: {"calls": n, "seconds": round(s, 2)}
                                for a, (n, s) in sorted(p2.action_totals().items())}
        out["n_point_events"] = p2.n_point_events
    else:
        spanset = read_spans(ref.path)
        avail.fields.update(spanset.availability.fields)

    conc = concurrency(spanset)
    avail.fields.update(conc.availability.fields)
    out.update({
        "span_source": spanset.source, "n_spans": len(spanset.spans),
        "span_wall_s": spanset.wall_s,
        "flush_collapsed_fraction": round(spanset.flush_collapsed_fraction, 3),
        "tool_seconds": round(sum(s.duration_s for s in spanset.spans), 1),
        "overlap_s": round(conc.overlap_s, 1), "overlap_share": round(conc.overlap_share, 4),
        "max_concurrent": conc.max_concurrent})

    if want_capsule_time:
        rows = read_capsule_timings(ref.path)
        tiers: dict = {}
        for tier in sorted({r.tier for r in rows}):
            for status in sorted({r.status for r in rows if r.status}):
                s = summarize(rows, tier=tier, status=status)
                if s.n == 0:
                    continue
                tiers[f"{tier}/{status}"] = {
                    "n": s.n, "n_carried": s.n_carried, "n_no_timing": s.n_no_timing,
                    "median_active_s": s.median_active_s, "p90_active_s": s.p90_active_s,
                    "max_active_s": s.max_active_s, "total_active_s": round(s.total_active_s, 1),
                    "wall_inconsistent": s.wall_inconsistent,
                    "status": s.availability.get("tier_cost").kind,
                    "note": s.availability.get("tier_cost").reason}
        out["tier_cost"] = tiers
        out["capsule_workers"] = sorted({r.workers for r in rows if r.workers}) or None

    out["availability"] = avail.to_dict()
    out["availability_score"] = round(avail.score, 3)
    return out


# --------------------------------------------------------------------------- selection

def select(facts: list[dict], *, per_cell: int = 1) -> list[dict]:
    """Mark the runs a report leads with, and say why each was or was not chosen.

    Ranked inside every ``(target, arm, phase)`` cell by score, then by how much of the run is
    actually readable, then by recency. Every run keeps a ``selection_reason``, so the exclusions are
    auditable rather than implicit -- the denominator is part of the result."""
    def score(f: dict) -> float:
        return (f["passed"] / f["capsules"]) if f.get("passed") is not None and f.get("capsules") else -1.0

    cells: dict[tuple, list[dict]] = {}
    for f in facts:
        cells.setdefault((f["target"], f["arm"], f["phase"]), []).append(f)

    for key, group in cells.items():
        group.sort(key=lambda f: (score(f), f.get("availability_score", 0.0),
                                  f.get("started_at") or ""), reverse=True)
        for rank, f in enumerate(group):
            if score(f) < 0:
                f["selected"] = False
                f["selection_reason"] = "no graded verdict on a sized corpus"
            elif rank < per_cell:
                f["selected"] = True
                f["selection_reason"] = (
                    f"best of {len(group)} in {key[0]}/{key[1]}/{key[2]} by score "
                    f"{f['passed']}/{f['capsules']}")
            else:
                f["selected"] = False
                f["selection_reason"] = (
                    f"rank {rank + 1} of {len(group)} in {key[0]}/{key[1]}/{key[2]} "
                    f"({f['passed']}/{f['capsules']})")

    # A complete ladder is worth more than four unrelated bests: it is the only like-for-like
    # comparison the study supports, so every member is kept even when it is not its cell's best.
    by_tag: dict[tuple, dict[str, dict]] = {}
    for f in facts:
        if score(f) < 0:
            continue
        prefix = f["run_id"].split("_", 1)
        tag = prefix[1] if len(prefix) == 2 else f["run_id"]
        by_tag.setdefault((f["target"], f["phase"], tag), {})[f["arm"]] = f
    for (target, phase, tag), members in by_tag.items():
        if len(members) < 3:
            continue
        for arm, f in members.items():
            f["selected"] = True
            f["ladder"] = f"{target}/{phase}/{tag}"
            f["selection_reason"] = (
                f"member of the {len(members)}-arm ladder {tag!r} on {target} "
                f"({sorted(members)}), which is a like-for-like comparison")
    return facts


# --------------------------------------------------------------------------- commands

def cmd_index(a) -> int:
    cfg = Config.load(a.config)
    refs: list[RunRef] = []
    for root, entry in cfg.resolved_roots():
        if not root.is_dir():
            print(f"  [skip] {entry['label']}: {root} does not exist", file=sys.stderr)
            continue
        found = build_index([root], cfg.arms, cfg.phases)
        print(f"  [{entry['label']}] {len(found)} run(s) under {root}")
        refs.extend(found)
    out = a.out or (artifacts_dir() / CONCERN / "index.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps([r.to_dict() for r in refs], indent=2) + "\n")
    conflicts = [r for r in refs if r.arm_conflict]
    print(f"\nindexed {len(refs)} run(s) -> {out}")
    if conflicts:
        print(f"{len(conflicts)} run(s) whose bundle and run-id disagree about their arm:")
        for r in conflicts:
            print(f"  {r.target}/{r.run_id}: {r.arm_conflict} (resolved to {r.arm})")
    return 0


def cmd_facts(a) -> int:
    cfg = Config.load(a.config)
    index_path = a.index or (artifacts_dir() / CONCERN / "index.json")
    refs = [RunRef(root=Path(d["root"]), path=Path(d["path"]), target=d["target"],
                   bench=d["bench"], phase=d["phase"], run_id=d["run_id"], arm=d["arm"],
                   arm_source=d["arm_source"], arm_conflict=d["arm_conflict"],
                   bundle_id=d["bundle_id"], driver=d["driver"], model=d["model"],
                   provider=d["provider"], started_at=d["started_at"], repo_sha=d["repo_sha"],
                   n_rounds=d["n_rounds"],
                   availability=Availability.from_dict(d.get("availability") or {}))
            for d in json.loads(index_path.read_text())]
    if a.target:
        refs = [r for r in refs if r.target in set(a.target)]
    facts = []
    for i, ref in enumerate(refs, 1):
        if i % 25 == 0 or i == len(refs):
            print(f"  {i}/{len(refs)} …", file=sys.stderr)
        try:
            facts.append(run_facts(ref, want_capsule_time=not a.no_capsule_time))
        except Exception as exc:  # noqa: BLE001 - one unreadable run must not lose the report
            print(f"  [warn] {ref.key}: {type(exc).__name__}: {exc}", file=sys.stderr)
    select(facts, per_cell=a.per_cell)
    out = a.out or (artifacts_dir() / CONCERN / "run_facts.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(facts, indent=2) + "\n")
    sel = [f for f in facts if f.get("selected")]
    print(f"\n{len(facts)} run(s) read, {len(sel)} selected -> {out}")
    return 0


def cmd_rescue(a) -> int:
    """Copy the light telemetry out of any root that holds the only copy of its runs."""
    cfg = Config.load(a.config)
    dest_root = artifacts_dir() / CONCERN / "rescued"
    n_runs = n_files = 0
    total_bytes = 0
    for root, entry in cfg.resolved_roots():
        if not entry.get("fragile") or not root.is_dir():
            continue
        for ref in build_index([root], cfg.arms, cfg.phases):
            dest = dest_root / ref.target / ref.bench / ref.run_id
            copied = 0
            for pattern in cfg.rescue_globs:
                for src in sorted(ref.path.glob(pattern)):
                    if not src.is_file():
                        continue
                    rel = src.relative_to(ref.path)
                    tgt = dest / rel
                    if not a.dry_run:
                        tgt.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(src, tgt)
                    copied += 1
                    total_bytes += src.stat().st_size
            if copied:
                n_runs += 1
                n_files += copied
        print(f"  [{entry['label']}] rescued through {root}")
    verb = "would copy" if a.dry_run else "copied"
    print(f"\n{verb} {n_files} file(s) from {n_runs} run(s), {total_bytes / 1e6:.1f} MB -> {dest_root}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=HERE / "roots.yaml")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("index", help="identify every run under every configured root")
    p.add_argument("--out", type=Path)
    p.set_defaults(fn=cmd_index)

    p = sub.add_parser("facts", help="extract every number the report cites")
    p.add_argument("--index", type=Path)
    p.add_argument("--target", action="append", default=[])
    p.add_argument("--per-cell", type=int, default=1)
    p.add_argument("--no-capsule-time", action="store_true",
                   help="skip the per-capsule tier scan (much faster; drops the cost figures)")
    p.add_argument("--out", type=Path)
    p.set_defaults(fn=cmd_facts)

    p = sub.add_parser("rescue", help="copy light telemetry out of the fragile roots")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(fn=cmd_rescue)

    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
