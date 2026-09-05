#!/usr/bin/env python3
"""CAMPAIGN: ours vs ExecuTorch (qd8) across every int8 model, one row per cell, guards live.

This is the driver that produces the headline evidence table. It owns everything AROUND the per-cell
instrument (``k1_int8_fair_compare.py``) and nothing inside it:

  * resolve each cell through ``merlin.baselines.bundle.resolve`` -- never a hardcoded path. A stale
    hardcoded map in another driver had all eight of its entries pointing at directories that do not
    exist and returned ``not_run`` for every cell instead of failing loudly;
  * refuse offline what can be refused offline (missing bundle, missing golden, a model that cannot
    fit the board) so a doomed cell never costs a build, a transfer and a board slot;
  * run the instrument as a SUBPROCESS per cell, so one cell blowing up cannot take the campaign
    with it and its stderr survives into the row;
  * append one row per cell to a JSONL ledger and rewrite the summary after EVERY cell, so a board
    session that dies mid-campaign is not wasted -- re-running with the same ``--out-dir`` skips
    what is already recorded;
  * end with a summary that states, unmissably, how many cells produced a verdict at all. The
    project claim is a win on a MAJORITY of a diverse set; a summary that reported only the wins
    would make that claim unfalsifiable.

MOST CELLS ARE EXPECTED TO REFUSE TODAY, and that is the honest state, not a bug. Recording each
refusal precisely -- with the bundle it refused, the guard that fired, and the reason -- is the
product.

Usage::

  # sanity-check what would run, spend no board time
  PYTHONPATH=merlin/python .venv/bin/python build_tools/scripts/k1_int8_et_campaign.py --dry-run

  # run it (board serialized: only one campaign at a time)
  MERLIN_K1_HOST=root@<board-ip> PYTHONPATH=merlin/python .venv/bin/python \
      build_tools/scripts/k1_int8_et_campaign.py

  # resume a campaign whose session died
  ... k1_int8_et_campaign.py --out-dir out/artifacts/compare/v1/compare_v1_<TS>_<sha>
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "merlin" / "python"))

from merlin.common import provenance as _prov          # noqa: E402
from merlin.common.artifacts import new_product, utc_stamp  # noqa: E402
from merlin.common.paths import build_dir, repo_root    # noqa: E402
from merlin.compare import et_campaign as ec            # noqa: E402

#: The diverse set the headline table is claimed over, cheapest-first so a session that dies late
#: still holds the cells that were affordable. A CLI default, overridable with ``--models``; the
#: campaign itself is model-agnostic and reads whatever it is given.
#:
#: Chosen for ARCHITECTURAL diversity rather than convenience -- a vision-language-action policy, a
#: recurrent/attention hybrid, a pure CNN, and a decoder-only LM -- so a win is not a win on one
#: shape of graph. All four resolve to a bundle and plan as runnable; the previous set was retired
#: because two of its members (``spectformer``, ``gemma2_2b``) have upstream reference blockers and
#: ``small_llama`` is small enough that its result generalises poorly.
DEFAULT_MODELS = ("lstmnetvit", "resnet50_v1_5", "smolvla", "tiny_llama")

#: Our int8 codegen package. A path under the codegen-package home, overridable with ``--package``.
DEFAULT_PACKAGE = "out/artifacts/targets/rvv/hand_v0_int8"

#: The instrument. READ-ONLY from here: this driver never edits it and never re-implements its
#: guards -- it runs it and records what it wrote.
INSTRUMENT = "build_tools/scripts/k1_int8_fair_compare.py"

#: Sources whose bytes decide what this campaign measured, digested into every summary so a result
#: produced from an uncommitted edit is identifiable instead of looking pinned.
_CAMPAIGN_SOURCES = ("build_tools/scripts/k1_int8_et_campaign.py",
                     INSTRUMENT,
                     "merlin/python/merlin/compare/et_campaign.py",
                     "merlin/python/merlin/compare/executorch_column.py")


def _instrument_command(plan, a) -> list:
    cmd = [sys.executable, str(_ROOT / INSTRUMENT),
           "--model", plan.model,
           "--model-dir", str(plan.ours_bundle_root),
           "--baseline", str(_ROOT / a.package),
           "--n", str(a.n), "--warmup", str(a.warmup), "--iters", str(a.iters),
           "--et-n-lo", str(a.et_n_lo), "--et-n-hi", str(a.et_n_hi),
           "--compile-timeout-s", str(a.compile_timeout_s)]
    if a.parallel_harts:
        cmd += ["--parallel-harts", str(a.parallel_harts)]
    if a.ref_cpu_threads:
        cmd += ["--ref-cpu-threads", str(a.ref_cpu_threads)]
    if a.features is not None:
        cmd += ["--features", a.features]
    if a.also_weight_only:
        cmd.append("--also-weight-only")
    return cmd


def _dirty(paths) -> list:
    out = []
    for rel in paths:
        got = subprocess.run(["git", "status", "--porcelain", "--", rel],
                             cwd=str(repo_root()), capture_output=True, text=True)
        if got.stdout.strip():
            out.append(rel)
    return sorted(out)


def _write_manifest(outdir: Path, product) -> None:
    """Keep manifest.yaml current in BOTH the fresh and the resumed case.

    A product dir under out/artifacts/<topic>/v*/ without a manifest fails the layout gate for
    everyone on the tree, so it is rewritten after every cell rather than once at the end. On a
    resume the existing manifest's identity fields (run_id / timestamp / git_sha) are PRESERVED --
    they name the campaign, and re-stamping them would silently re-date somebody's cited result.
    """
    from merlin.common.yaml import dump_yaml, load_yaml

    files = sorted(p.name for p in outdir.iterdir() if p.is_file() and p.name != "manifest.yaml")
    cells = sorted(f"cells/{p.name}" for p in (outdir / "cells").iterdir()) \
        if (outdir / "cells").is_dir() else []
    mf = outdir / "manifest.yaml"
    if mf.is_file():
        existing = load_yaml(mf) or {}
        if isinstance(existing, dict):
            existing["artifacts"] = files + cells
            mf.write_text(dump_yaml(existing), encoding="utf-8")
            return
    if product is not None:
        product._artifacts = files + cells
        product.write_manifest()


def _write_summary(outdir: Path, ledger: Path, product) -> dict:
    """Rewrite summary.json / summary.txt / manifest.yaml from the ledger. Called after EVERY cell so
    a killed session still leaves a readable, layout-valid product directory behind."""
    rows = ec.read_ledger(ledger)
    summary = ec.summarize(rows)
    summary["generated"] = utc_stamp()
    summary["ledger"] = ledger.name
    try:
        summary["provenance"] = _prov.record(
            sources=[str(repo_root() / s) for s in _CAMPAIGN_SOURCES])
    except Exception as e:                       # a provenance stamp must not break a campaign
        summary["provenance"] = {"error": f"{type(e).__name__}: {e}"}
    summary["source_dirty"] = _dirty(_CAMPAIGN_SOURCES)
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    text = ec.format_summary(summary)
    (outdir / "summary.txt").write_text(text + "\n", encoding="utf-8")
    _write_manifest(outdir, product)
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS),
                    help="comma-separated cells, cheapest-first")
    ap.add_argument("--variant", default="int8")
    ap.add_argument("--package", default=DEFAULT_PACKAGE, help="our codegen package (repo-relative)")
    ap.add_argument("--out-dir", default=None,
                    help="resume into an existing campaign product dir; omitted = create a new one")
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve and price every cell, print what WOULD run, touch no board and "
                         "write no artifact")
    ap.add_argument("--force", action="store_true", help="re-run cells already recorded")
    ap.add_argument("--retry-refused", action="store_true",
                    help="on a resume, re-run the cells whose recorded outcome was a refusal (a "
                         "board that went away mid-campaign refuses every remaining cell, and those "
                         "refusals are about the session, not the model)")
    ap.add_argument("--no-board-preflight", action="store_true",
                    help="skip the board reachability check. Do not use casually: without it an "
                         "unreachable board records a refusal on EVERY cell, and a resume then "
                         "skips them forever as settled outcomes")
    ap.add_argument("--board-usable-bytes", type=int, default=ec.DEFAULT_BOARD_USABLE_BYTES,
                    help="RAM a whole-model run may use; DECLARED, and recorded as such on every row")
    ap.add_argument("--prefer-rewritten", action="store_true",
                    help="measure ours on a declared LAYOUT-ONLY derivative of the resolved bundle "
                         "(read from its own bundle.rewrites.json, never from a name rule)")
    ap.add_argument("--features", default=None, help="compiler features for OUR arm")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--et-n-lo", type=int, default=1)
    ap.add_argument("--et-n-hi", type=int, default=6)
    ap.add_argument("--also-weight-only", action="store_true")
    ap.add_argument("--cell-timeout", type=int, default=7200, help="seconds per cell")
    ap.add_argument("--ref-cpu-threads", type=int, default=None,
                    help="cores the REFERENCE arm may use; unset means its threadpool takes every "
                         "online CPU (8 here). Pass 1 for a per-core comparison.")
    ap.add_argument("--parallel-harts", type=int, default=None,
                    help="cores OUR arm may use (default 1). The board has 8 and the reference's "
                         "runner links pthreadpool, so the default compares our one core against "
                         "however many ExecuTorch takes. Recorded on every cell either way.")
    ap.add_argument("--compile-timeout-s", type=int, default=3600,
                    help="ceiling on any single build command inside a cell (see the instrument). "
                         "Must be under --cell-timeout: a compile allowed to outlive its own cell "
                         "cannot be what stops the cell.")
    ap.add_argument("--clean-work", action="store_true",
                    help="delete the instrument's build tree after each cell (bounds host disk; the "
                         "emitted object is then no longer available to explain the result)")
    a = ap.parse_args()
    if a.compile_timeout_s >= a.cell_timeout:
        ap.error(f"--compile-timeout-s={a.compile_timeout_s} is not under "
                 f"--cell-timeout={a.cell_timeout}: a build allowed to outlive its own cell "
                 "can never be the thing that stops the cell, and the cell would time out "
                 "with no record instead of reporting which command ran long.")

    models = [m.strip() for m in a.models.split(",") if m.strip()]
    plans = ec.plan_campaign(models, variant=a.variant, int8=True,
                             budget_bytes=a.board_usable_bytes,
                             budget_source=("declared via --board-usable-bytes"
                                            if a.board_usable_bytes != ec.DEFAULT_BOARD_USABLE_BYTES
                                            else "declared default (et_campaign)"),
                             prefer_rewritten=a.prefer_rewritten)

    if a.dry_run:
        print(f"[dry-run] {len(plans)} cells; package={a.package}; NO board time will be spent\n")
        for p in plans:
            print(f"--- {p.model} ({p.variant})")
            print(f"    resolved bundle : {p.reference_bundle_root}")
            print(f"    ours bundle     : {p.ours_bundle_id}"
                  + (f"  [layout-only rewrite of {p.reference_bundle_id}]"
                     if p.layout_equivalence else ""))
            print(f"    goldens         : {p.goldens}")
            print(f"    w8a8 reference  : {p.w8a8_reference['status']} "
                  f"(independent={p.w8a8_reference['independent']})")
            fp = p.footprint
            print(f"    resident >= {fp['resident_lower_bound_bytes'] / 1e9:.2f} GB vs budget "
                  f"{fp['budget_bytes'] / 1e9:.2f} GB [{fp['budget_source']}] -> "
                  f"fits={fp['fits']}")
            if p.refusals:
                print("    WOULD REFUSE (no board time):")
                for r in p.refusals:
                    print(f"      - {r}")
            else:
                print("    WOULD RUN: " + " ".join(_instrument_command(p, a)))
            for n in p.notes:
                print(f"    note: {n}")
            print()
        runnable = [p.model for p in plans if p.runnable]
        print(f"[dry-run] would run {len(runnable)}/{len(plans)}: {runnable}")
        print(f"[dry-run] would refuse offline: {[p.model for p in plans if not p.runnable]}")
        return 0

    # The board, ONCE, before any row is written. Without this an unreachable board produces a
    # refusal on every cell and a resume then skips all four forever -- recording an outage as a
    # property of four models. Checked here so it is one loud failure with no ledger side effects.
    if not a.no_board_preflight:
        from merlin.mining import k1 as k1mod
        if not k1mod.available():
            print("ERROR: the board is not reachable (host="
                  f"{k1mod.K1_HOST!r}, toolchain={k1mod.toolchain_cc()}). Nothing recorded: a "
                  "campaign run now would write a refusal on every cell and a later resume would "
                  "skip them as settled outcomes. Set MERLIN_K1_HOST and retry.")
            return 2

    if a.out_dir:
        outdir = Path(a.out_dir)
        if not outdir.is_dir():
            print(f"ERROR: --out-dir {outdir} does not exist; omit it to create a new campaign")
            return 2
        if not (outdir / "manifest.yaml").is_file():
            print(f"ERROR: --out-dir {outdir} carries no manifest.yaml, so it is not a campaign "
                  "product dir. Resuming into it would leave a product directory that fails the "
                  "artifact-layout gate for everyone on the tree; omit --out-dir to create one.")
            return 2
        product = None
        print(f"[resume] {outdir}")
    else:
        product = new_product("compare", version=1,
                              sources=[str(repo_root() / s) for s in _CAMPAIGN_SOURCES],
                              notes="ours vs ExecuTorch qd8 int8 campaign")
        outdir = product.path
        print(f"[campaign] {outdir}")
    (outdir / "cells").mkdir(parents=True, exist_ok=True)
    ledger = outdir / "ledger.jsonl"
    done = set() if a.force else ec.completed_models(ec.read_ledger(ledger),
                                                     retry_refused=a.retry_refused)
    if done:
        print(f"[resume] already recorded, skipping: {sorted(done)}")

    for plan in plans:
        if plan.model in done:
            continue
        print(f"\n===== {plan.model} =====", flush=True)
        if not plan.runnable:
            reason = " | ".join(plan.refusals)
            print(f"REFUSED offline: {reason}", flush=True)
            ec.append_row(ledger, ec.campaign_row(plan, None, refusal=reason))
            _write_summary(outdir, ledger, product)
            continue
        cmd = _instrument_command(plan, a)
        cell_json = outdir / "cells" / f"{plan.model}.json"
        log = outdir / "cells" / f"{plan.model}.log"
        full = cmd + ["--out", str(cell_json)]
        print("$ " + " ".join(full), flush=True)
        t0 = time.time()
        record, refusal = None, ""
        try:
            with log.open("w", encoding="utf-8") as lf:
                got = subprocess.run(full, cwd=str(repo_root()), stdout=lf,
                                     stderr=subprocess.STDOUT, timeout=a.cell_timeout)
            rc = got.returncode
        except subprocess.TimeoutExpired:
            rc = None
            refusal = (f"the instrument did not finish within --cell-timeout={a.cell_timeout}s; no "
                       f"record was written. See {log.name}.")
        if not refusal:
            if cell_json.is_file():
                try:
                    record = json.loads(cell_json.read_text(encoding="utf-8"))
                except ValueError as e:
                    refusal = f"the instrument's record at {cell_json.name} is unreadable: {e}"
            else:
                tail = ""
                if log.is_file():
                    tail = "\n".join(log.read_text(encoding="utf-8",
                                                   errors="replace").splitlines()[-12:])
                refusal = (f"the instrument exited {rc} without writing a record; nothing to read a "
                           f"ratio from. Tail of {log.name}:\n{tail}")
        elapsed = round(time.time() - t0, 1)
        row = ec.campaign_row(plan, record, refusal=refusal, command=full, elapsed_s=elapsed)
        ec.append_row(ledger, row)
        v = row["verdict"]
        if v.get("status") == "measured":
            print(f"MEASURED {plan.model}: ours={v['ours_ns'] / 1e6:.3f} ms  et_warm="
                  f"{v['executorch_warm_ns'] / 1e6:.3f} ms  speedup="
                  f"{v['speedup_vs_executorch']:.3f}x  beats={v['beats_executorch']}", flush=True)
        else:
            print(f"REFUSED {plan.model}: {v.get('reason', '')[:400]}", flush=True)
        _write_summary(outdir, ledger, product)
        if a.clean_work:
            shutil.rmtree(build_dir() / "fair_compare" / plan.ours_bundle_id, ignore_errors=True)

    summary = _write_summary(outdir, ledger, product)
    print("\n" + ec.format_summary(summary))
    print(f"\n[out] {outdir}")
    print(f"[resume] re-run with --out-dir {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
