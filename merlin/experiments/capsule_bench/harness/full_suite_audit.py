#!/usr/bin/env python3
"""Full-suite cycle + correctness audit of frozen baseline submissions against ALL capsules.

The pilot iterate-to-pass loop only grades the 4 pilot capsules. This audit answers the broader
question — "how do the baselines do on *every* test, and how many cycles does each capsule take" —
by re-grading each frozen submission against the **entire 25-capsule corpus** (public + hidden) on
the real RTL oracle (L3 verilator; opportunistically L4 VCS), in PARALLEL. No agent re-run, $0 API.

It is honest about partial pass: backends built only against the pilot are expected to fail capsule
classes they never implemented (conv/im2col, attention) — those show as failures with their first
failure plane, never hidden. Per-capsule L3 cycles and an active-vs-waiting timing rollup
(sim_active vs oracle/queue wait, parallel speedup) are reported per the user's directive.

Usage:
  full_suite_audit.py [--backends rb_pilot_cpp_01,rb_pilot_0002] [--workers 8] [--tiers L2,L3]
                      [--timeout 900]
Writes reports/full_suite_audit.{md,json}.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

from merlin.common.artifacts import cache_dir  # noqa: E402 — purgeable work trees
import _common as C

sys.path.insert(0, str(C.REPO / "merlin" / "python"))
from merlin.targetgen import capsule_grade as CG          # noqa: E402
from merlin.targetgen import capsule_runner as CR          # noqa: E402
from merlin.targetgen import heavy_oracles as HO           # noqa: E402

CORPUS = C.REPO / "merlin/contract" / "capsules"
CONTRACT = C.REPO / "merlin/contract"


def _capsule_class(cap_dir: Path) -> str:
    """Primary workload class for coverage rollup (derived from the interface MLIR + capsule.yaml)."""
    iface = (cap_dir / "capsule.interface.mlir")
    txt = iface.read_text() if iface.is_file() else ""
    y = {}
    cy = cap_dir / "capsule.yaml"
    if cy.is_file():
        try:
            y = yaml.safe_load(cy.read_text()) or {}
        except Exception:
            y = {}
    name = (y.get("name") or cap_dir.name)
    if "merlin_iface.conv" in txt:
        return "conv"
    if "merlin_iface.movement" in txt and "merlin_iface.matmul" not in txt:
        return "movement"
    if y.get("kind") == "model_slice" or name[:1] == "C":
        return "attention" if "attention" in str(y.get("source_reference", "")).lower() \
            or name[:2] in ("C2", "C3", "C4", "C5", "C6") else "mlp"
    # modes live under expected.modes (booleans); check the VALUE, not key presence
    modes = (y.get("expected", {}) or {}).get("modes", {}) or y.get("modes", {}) or {}
    if modes.get("relu"):
        return "matmul+relu"
    if modes.get("acc_scale"):
        return "matmul+acc_scale"
    return "matmul"


def _sim_via() -> str | None:
    """The target's declared bespoke sim (chipyard for gemmini), from the descriptor — so the audit
    resolves atlas/arc targets' oracle tiers from their contract instead of the gemmini spike/verilator."""
    try:
        from merlin.targetgen.target_experiment import load_target_experiment
        return load_target_experiment(C.EXP / "target_experiment.yaml").sim_via
    except Exception:  # noqa: BLE001
        return "chipyard"


def _adapters_for(tiers: list[str]) -> dict:
    """Resolve the requested audit tiers target-awarely. A chipyard target (gemmini) maps tiers onto its
    spike/verilator/vcs ladder; any other target uses its contract-derived tiers (atlas external_backend
    -> the program oracle) filtered to those requested — routing atlas through the hardcoded
    spike/verilator adapters ran the gemmini/RVV lowering path and crashed (AW4)."""
    if _sim_via() == "chipyard":
        ad = {}
        if "L2" in tiers:
            ad["L2"] = CR._spike_verilator_adapter("spike", C.TARGET)
        if "L3" in tiers:
            ad["L3"] = CR._spike_verilator_adapter("verilator", C.TARGET)
        if "L4" in tiers and HO.vcs_available():
            ad["L4"] = HO.vcs_adapter(C.TARGET)
        return ad
    full = CR.oracle_adapters(C.TARGET, _sim_via())
    sel = {t: a for t, a in full.items() if t in tiers}
    return sel or full          # fall back to the target's real tier(s) if none of `tiers` apply


def audit_backend(run_id: str, *, workers: int, tiers: list[str], timeout: int) -> dict | None:
    sub = C.RUNS / "raw_baseline" / run_id / "submission"
    if not (sub / "manifest.yaml").is_file():
        print(f"  !! {run_id}: no submission/manifest.yaml — skipping")
        return None
    # A grading WORK tree, not a run: keep it out of the runs root so run-enumerating tools
    # (gen_reports / gen_fullsuite_report glob "*/*/run_manifest.yaml") cannot mistake it for one.
    runs_root = cache_dir("capsule_bench_audit") / run_id
    runs_root.mkdir(parents=True, exist_ok=True)
    adapters = _adapters_for(tiers)
    t0 = time.perf_counter()
    score = CG.grade(sub, capsules_root=CORPUS, runs_root=runs_root,
                     labels={"public", "dev", "hidden"}, contract=CONTRACT,
                     oracle_adapters=adapters, timeout=timeout, max_workers=workers, target=C.TARGET)
    score["_audit_wall_s"] = round(time.perf_counter() - t0, 1)
    score["_lang"] = yaml.safe_load((sub / "manifest.yaml").read_text()).get("language", "?")
    (runs_root / "score_full.json").write_text(json.dumps(score, indent=2))
    return score


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backends", default="rb_pilot_cpp_01,rb_pilot_0002")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--tiers", default="L2,L3", help="oracle tiers to run (e.g. L2,L3 or L2,L3,L4)")
    ap.add_argument("--timeout", type=int, default=900)
    a = ap.parse_args(argv)
    tiers = [t.strip() for t in a.tiers.split(",") if t.strip()]
    backends = [b.strip() for b in a.backends.split(",") if b.strip()]

    caps = CR.discover_capsules(CORPUS, labels={"public", "dev", "hidden"}, contract=CONTRACT)
    cap_class = {c["name"]: _capsule_class(Path(c["__dir__"])) for c in caps}
    cap_label = {c["name"]: c.get("label") for c in caps}
    cap_names = [c["name"] for c in caps]

    scores: dict[str, dict] = {}
    for run_id in backends:
        print(f"== auditing {run_id} on {len(cap_names)} capsules (workers={a.workers}, tiers={tiers}) ==")
        s = audit_backend(run_id, workers=a.workers, tiers=tiers, timeout=a.timeout)
        if s is not None:
            scores[run_id] = s
            print(f"   {run_id}: {s.get('headline')} wall={s['_audit_wall_s']}s "
                  f"speedup={s.get('timing_rollup', {}).get('parallel_speedup')}")

    # per-capsule status/cycles per backend
    pc = {rid: {p["capsule"]: p for p in s.get("per_capsule", [])} for rid, s in scores.items()}
    cyc = {rid: s.get("cycles_diagnostic", {}) for rid, s in scores.items()}

    out = {"corpus": str(CORPUS), "n_capsules": len(cap_names), "tiers": tiers,
           "workers": a.workers, "backends": {}, "matrix": [], "class_coverage": {}}
    for rid, s in scores.items():
        out["backends"][rid] = {
            "language": s.get("_lang"), "passed": f"{s['n_passed']}/{s['n_capsules']}",
            "public_passed": s.get("public_passed"), "hidden_passed": s.get("hidden_passed"),
            # the qualified form, so a reader of this artifact cannot quote the fraction alone
            "headline": s.get("headline"), "pass_evidence": s.get("pass_evidence"),
            "highest_tier": s.get("highest_tier"), "audit_wall_s": s.get("_audit_wall_s"),
            "timing_rollup": s.get("timing_rollup"),
            "first_failure_planes": s.get("first_failure_planes"),
        }
    # class coverage: per class, how many capsules each backend passed
    classes: dict[str, list[str]] = {}
    for n in cap_names:
        classes.setdefault(cap_class[n], []).append(n)
    for cls, names in sorted(classes.items()):
        out["class_coverage"][cls] = {
            "n": len(names),
            **{rid: sum(1 for n in names if pc.get(rid, {}).get(n, {}).get("status") == "pass")
               for rid in scores},
        }
    for n in cap_names:
        row = {"capsule": n, "label": cap_label[n], "class": cap_class[n]}
        for rid in scores:
            p = pc.get(rid, {}).get(n, {})
            row[f"{rid}__status"] = p.get("status")
            row[f"{rid}__cycles"] = cyc.get(rid, {}).get(n)
        out["matrix"].append(row)

    (C.REPORTS / "full_suite_audit.json").write_text(json.dumps(out, indent=2))
    _write_md(out, scores)
    print(f"\nwrote reports/full_suite_audit.{{md,json}} ({len(cap_names)} capsules x {len(scores)} backends)")
    return 0


def _write_md(out: dict, scores: dict) -> None:
    rids = list(scores)
    md = ["# Full-suite audit (capsule_bench_v0) — all 25 capsules, RTL oracle", "",
          f"Corpus: `{out['corpus']}` · {out['n_capsules']} capsules · tiers {out['tiers']} · "
          f"{out['workers']} parallel workers. Cycle counts are **L3 verilator (cycle-accurate RTL)**. "
          "Backends were built against the 4-capsule pilot only — failures on unimplemented classes "
          "(conv, attention) are expected and reported honestly, not hidden.", ""]
    # `rtl-backed` sits beside the counts on purpose: a table with `public` and `tier` in separate
    # columns still lets the eye read "20/20" and stop, and the tier column reports the tier EVERY
    # capsule cleared -- which says nothing about how many cleared the RTL one above it.
    md += ["## Headline", "", "| backend | lang | passed (all) | public | hidden | tier | rtl-backed | "
           "audit wall(s) | sim_active(s) | oracle_wait(s) | speedup |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for rid in rids:
        b = out["backends"][rid]
        tr = b.get("timing_rollup") or {}
        _ev = b.get("pass_evidence") or {}
        _rtl = ("n/a" if _ev.get("rtl_backed") is None
                else f"{_ev['rtl_backed']}/{_ev.get('n_passed', '?')}")
        md.append(f"| {rid} | {b['language']} | {b['passed']} | {b['public_passed']} | "
                  f"{b['hidden_passed']} | {b['highest_tier']} | {_rtl} | {b['audit_wall_s']} | "
                  f"{tr.get('sim_active_s')} | {tr.get('oracle_wait_s')} | {tr.get('parallel_speedup')} |")
    md += ["", "## Coverage by workload class", "",
           "| class | n | " + " | ".join(rids) + " |",
           "|---|---|" + "|".join(["---"] * len(rids)) + "|"]
    for cls, cc in out["class_coverage"].items():
        md.append(f"| {cls} | {cc['n']} | " + " | ".join(f"{cc.get(r,0)}/{cc['n']}" for r in rids) + " |")
    md += ["", "## Per-capsule matrix (status · L3 cycles)", "",
           "| capsule | label | class | " + " | ".join(rids) + " |",
           "|---|---|---|" + "|".join(["---"] * len(rids)) + "|"]
    for row in out["matrix"]:
        cells = []
        for rid in rids:
            st = row.get(f"{rid}__status")
            cy = row.get(f"{rid}__cycles")
            cells.append(f"{st}" + (f" · {cy}cyc" if cy is not None else ""))
        md.append(f"| {row['capsule']} | {row['label']} | {row['class']} | " + " | ".join(cells) + " |")
    md += ["", "_Legend: cycles = L3 verilator RTL cycles (rdcycle-bracketed). oracle_wait(s) is time "
           "blocked on a queue/FPGA slot (≈0 for local verilator; nonzero only for queued VCS/FireSim). "
           "speedup = sum(active_sim)/wall under parallel workers._"]
    (C.REPORTS / "full_suite_audit.md").write_text("\n".join(md) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
