"""``merlin-compare`` CLI entry — spec-driven, versioned comparison driver.

Usage:
  merlin-compare --spec spec.yaml
  merlin-compare --configs baseline,ours_wholemodel,ours_wholemodel_vf,xnnpack,openblas \
                 --workloads openvla,rdt2,bitvla,gemm:64 [--target k1] [--metric wall] [--reps 5]

v1 INGESTS cached measurements (HOST + cached board JSON; no new board run). ``--run`` is a declared
seam and is not implemented (it raises).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .driver import run
from .spec import Spec


def _build_spec(args) -> Spec:
    if args.spec:
        return Spec.from_yaml(args.spec)
    if not (args.configs and args.workloads):
        raise SystemExit("provide --spec, or both --configs and --workloads")
    raw = {
        "label": args.label,
        "target": args.target,
        "metric": args.metric,
        "reps": args.reps,
        "configs": [c.strip() for c in args.configs.split(",") if c.strip()],
        "workloads": [w.strip() for w in args.workloads.split(",") if w.strip()],
    }
    return Spec.parse(raw)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="merlin-compare",
                                 description="Spec-driven baseline-vs-ours-vs-experts comparison.")
    ap.add_argument("--spec", type=Path, help="YAML spec file")
    ap.add_argument("--configs", help="comma list, e.g. baseline,ours_wholemodel,xnnpack,openblas")
    ap.add_argument("--workloads", help="comma list, e.g. openvla,rdt2,bitvla,gemm:64")
    ap.add_argument("--target", default="k1", help="k1 (impl); spike/gemmini/npu are seams")
    ap.add_argument("--metric", default="wall", choices=["wall", "instret"])
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--label", default="compare")
    ap.add_argument("--out-root", type=Path, default=None,
                    help="artifact root (default mined_knowledge/rvv)")
    ap.add_argument("--run", action="store_true",
                    help="LIVE board measurement (declared seam; not implemented in v1)")
    args = ap.parse_args(argv)

    spec = _build_spec(args)
    out_dir = run(spec, out_root=args.out_root, run_board=args.run)
    print(f"merlin-compare: wrote {out_dir}")
    print(f"  {out_dir / 'compare.md'}")
    print(f"  {out_dir / 'manifest.yaml'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
