#!/usr/bin/env python3
"""Ingest a user's exported PyTorch applications and report the capsules they imply.

    build_tools/scripts/ingest_applications.py --target gemmini --applications path/to/pt2s/
    build_tools/scripts/ingest_applications.py --target gemmini --report-only

Point it at a directory of ``torch.export`` archives -- the models this target is actually FOR --
and it captures each one into the recapture store, groups their regions by what the compiler must do
with them, and sizes each behavioural class against what a certification costs on that target.

WHY A CLI RATHER THAN A GENERATION STEP. Capturing a model is minutes of work in another virtualenv
and the result is a large binary bundle; folding that into corpus generation would make every
regeneration pay for it. The bundles land in the store the requirement already reads, so ingesting
is a one-off and everything downstream picks them up.

The report is the honest part. It names, per class, the shape the application really contains, the
size that class was clamped to for cycle-accurate certification, and every class that could not be
sized at all -- because an unaffordable behaviour must look different from an absent one.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "merlin" / "python") not in sys.path:
    sys.path.insert(0, str(REPO / "merlin" / "python"))


def _worker_env():
    """The m2m interpreter and checkout, or a reason they are unusable."""
    from merlin.targetgen.capsule_source import _m2m_dir, _m2m_python  # noqa: PLC2701

    m2m, python = Path(_m2m_dir()), Path(_m2m_python())
    if not python.exists():
        return None, None, f"no m2m interpreter at {python} (set MERLIN_M2M_PYTHON)"
    if not m2m.is_dir():
        return None, None, f"no model2MLIR checkout at {m2m} (set MERLIN_M2M_DIR)"
    return m2m, python, None


def ingest(exported: Path, *, quant: str | None = None, timeout: int = 1800) -> tuple[Path | None, str]:
    """Capture one ``.pt2`` into the recapture store. ``(bundle_dir, message)``."""
    from merlin.common.artifacts import recaptures_dir

    m2m, python, why = _worker_env()
    if why:
        return None, why
    worker = REPO / "merlin" / "python" / "merlin" / "targetgen" / "_pt2_capture_worker.py"
    # The bundle name carries the format, matching the store's own convention so the existing
    # name-normalisation reads it the way it reads every other capture.
    out = Path(recaptures_dir()) / f"{exported.stem}_{quant or 'fp32'}_app"
    cmd = [str(python), str(worker), "--exported", str(exported), "--out", str(out),
           "--m2m-dir", str(m2m)]
    if quant:
        cmd += ["--quant", quant]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0 or not (out / "model.mlir").is_file():
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()
        return None, f"capture failed (rc={proc.returncode}): {tail[-1] if tail else 'no output'}"
    return out, "ok"


def report(target: str, captures: dict, *, budget_s: float | None = None) -> dict:
    """The classes these applications imply for ``target``, sized to a certification budget."""
    from merlin.targetgen.conformance import _application_axis  # noqa: PLC2701

    return _application_axis(target, captures=captures, budget_s=budget_s)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", required=True)
    ap.add_argument("--applications", type=Path,
                    help="directory of torch.export .pt2 archives to capture")
    ap.add_argument("--quant", default="", help="torchAO scheme for the capture (default: none)")
    ap.add_argument("--budget-s", type=float, default=None,
                    help="seconds a single cycle-accurate certification may take")
    ap.add_argument("--report-only", action="store_true",
                    help="skip capture; analyse the application bundles already in the store")
    a = ap.parse_args(argv)

    from merlin.common.artifacts import recaptures_dir

    captures: dict[str, Path] = {}
    if a.applications and not a.report_only:
        pt2s = sorted(a.applications.glob("*.pt2")) if a.applications.is_dir() else []
        if not pt2s:
            print(f"no .pt2 archives under {a.applications}", file=sys.stderr)
            return 2
        for src in pt2s:
            bundle, msg = ingest(src, quant=(a.quant or None))
            if bundle is None:
                print(f"  [FAIL] {src.name}: {msg}", file=sys.stderr)
                continue
            print(f"  [ok]   {src.name} -> {bundle}")
            captures[bundle.name] = bundle / "model.mlir"

    # Whatever was just captured, plus any application bundle already in the store.
    for d in sorted(Path(recaptures_dir()).glob("*_app")):
        if (d / "model.mlir").is_file():
            captures.setdefault(d.name, d / "model.mlir")
    if not captures:
        print("no application bundles found; pass --applications to capture some", file=sys.stderr)
        return 2

    axis = report(a.target, captures, budget_s=a.budget_s)
    print(f"\n== {a.target}: {axis['n_classes']} behavioural class(es) over {axis['n_regions']} "
          f"region(s) from {len(captures)} application(s)")
    fit = axis.get("cost_model")
    print(f"   budget {axis['cert_budget_s']}s ({axis['budget_source']}); cost model: "
          + (f"{fit['n_samples']} measured runs, R^2 {fit['r2']}" if fit else
             "NONE — no certification history, so no class can be sized"))
    for cap in axis.get("required") or ():
        basis = cap.get("basis") or {}
        extends = f"  extends {cap['extends']}" if cap.get("extends") else ""
        print(f"   {cap['tier']:3s} {cap['class']:58s} ({cap['M']},{cap['K']},{cap['N']})"
              f"  [{basis.get('sized_by')}]{extends}")
    for refusal in axis.get("refused") or ():
        print(f"   ---  {refusal}")
    if axis.get("captures_unreadable"):
        print(f"   unreadable: {axis['captures_unreadable']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
