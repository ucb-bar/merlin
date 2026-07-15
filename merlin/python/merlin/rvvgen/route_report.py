"""``merlin-cca-route`` — tell me WHICH section of the compiler to modify for a given divergence.

Given a CCA divergence (or a whole ``divergences.yaml`` from ``merlin-rvv-mine``), this prints, per
axis: the cheapest applicable action CLASS (FLAG / KNOB / HEURISTIC / PASS / CODEGEN), the CONCRETE
seam file to edit, whether a fork can express it today (``forkable_now``) or it needs new code, and the
full escalation ladder (the next-stronger lever if the cheap one doesn't land the intended facet). It
is a thin, read-only view over ``action_catalog`` (``route`` / ``escalation_ladder`` / ``seam_location``)
— the single source of truth for "what can we change in the compiler, and where".

Usage:
  merlin-cca-route                         # the whole rvv route table (every axis, cheapest lever + seam)
  merlin-cca-route --axis compute.accumulator_resident   # the full ladder for one axis
  merlin-cca-route --divergences out/artifacts/kernel-mining/rvv/<run>/divergences.yaml
  merlin-cca-route --json                  # machine-readable
"""
from __future__ import annotations

import argparse
import json
import sys

from merlin.kernels import action_catalog as ac
from merlin.kernels.cca_compare import Divergence


def _all_axes(backend: str) -> list[str]:
    return sorted({r.axis for r in ac._ROUTES.get(backend, [])})


def _ladder_for(axis: str, backend: str) -> dict:
    return {"axis": axis, "ladder": ac.escalation_ladder(axis, backend)}


def _load_divergences(path: str) -> list[Divergence]:
    import yaml
    doc = yaml.safe_load(open(path, encoding="utf-8")) or []
    rows = doc.get("divergences", doc) if isinstance(doc, dict) else doc
    out = []
    for r in rows:
        out.append(Divergence(axis=r["axis"], expert=r.get("expert"), ours=r.get("ours"),
                              backend=r.get("backend", "rvv"), evidence=list(r.get("evidence", []))))
    return out


def _route_view(d: Divergence) -> dict:
    a = ac.route(d)
    view = {"axis": d.axis, "expert": d.expert, "ours": d.ours, "routed": a is not None}
    if a is not None:
        loc = ac.seam_location(a.target_seam)
        view.update({"action_class": a.action_class, "target_seam": a.target_seam,
                     "seam_file": loc["seam_file"], "seam_kind": loc["seam_kind"],
                     "needs_new_code": loc["needs_new_code"], "forkable_now": a.forkable_now})
    view["ladder"] = ac.escalation_ladder(d.axis, d.backend)
    return view


def _print_axis(view: dict) -> None:
    axis = view["axis"]
    print(f"\n{axis}")
    if "expert" in view:
        print(f"  divergence: expert={view['expert']!r} ours={view['ours']!r}")
    if not view.get("ladder"):
        print("  (no route registered — surfaced as unrouted, never silently dropped)")
        return
    for i, step in enumerate(view["ladder"]):
        mark = "*" if (view.get("routed") and step["action_class"] == view.get("action_class")) else " "
        fork = "forkable now" if step["forkable_now"] else "NEEDS NEW CODE" if step["needs_new_code"] else "deferred"
        print(f"  [{mark}] {step['action_class']:9s} {fork:14s} {step['seam_file']}")
        print(f"        seam: {step['target_seam']}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", default="rvv", help="target backend (default: rvv)")
    ap.add_argument("--axis", help="show the full escalation ladder for one axis (e.g. compute.epilogue)")
    ap.add_argument("--divergences", help="a divergences.yaml (list, or {divergences: [...]}) to route")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args(argv)

    if args.divergences:
        views = [_route_view(d) for d in _load_divergences(args.divergences)]
    elif args.axis:
        views = [_ladder_for(args.axis, args.backend)]
    else:
        views = [_ladder_for(ax, args.backend) for ax in _all_axes(args.backend)]

    if args.json:
        print(json.dumps({"backend": args.backend, "axes": views}, indent=2))
        return 0

    print(f"== compiler-modification routes for backend '{args.backend}' "
          "(* = cheapest that fires; ladder = weakest -> strongest) ==")
    for v in views:
        _print_axis(v)
    return 0


if __name__ == "__main__":
    sys.exit(main())
