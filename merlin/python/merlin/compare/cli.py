"""``merlin-compare`` CLI entry — spec-driven, versioned comparison driver.

Usage:
  merlin-compare --spec spec.yaml
  merlin-compare --configs baseline,ours_wholemodel,ours_wholemodel_vf,xnnpack,openblas \
                 --workloads openvla,rdt2,bitvla,gemm:64 [--target k1] [--metric wall] [--reps 5]

Version 1 preserves the historical cache/live comparison. Version 2 implements the frozen-compiler
paper protocol: plan/preflight by default, content-addressed freeze, and strict live matrix runs.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

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


def _load_versioned(path: Path):
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and int(raw.get("version", 1)) == 2:
        from .paper import PaperStudySpec
        return PaperStudySpec.parse(raw, source_path=path.resolve())
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
                    help="explicit output directory/root (default canonical out/ layout)")
    ap.add_argument("--run", action="store_true",
                    help="perform live board measurement (v2 requires a clean preflight)")
    ap.add_argument("--audit-environment", action="store_true",
                    help="exhaustively audit captures, adapters, packages, tools, and board without "
                         "compiling or running a matrix cell")
    ap.add_argument("--freeze", action="store_true",
                    help="resolve capture/package/runtime digests into a new frozen v2 spec")
    ap.add_argument("--policy", type=Path,
                    help="compiler policy/package directory to freeze (required with --freeze)")
    ap.add_argument("--runtime-path", type=Path, action="append", default=[],
                    help="runtime source path to hash; repeatable (defaults to Merlin runtime sources)")
    ap.add_argument("--toolchain-authority", type=Path,
                    help="externally reviewed paper toolchain authority JSON (required with --freeze)")
    ap.add_argument("--frozen-out", type=Path,
                    help="where to write the frozen spec (default canonical paper-study artifact)")
    args = ap.parse_args(argv)

    spec = _load_versioned(args.spec) if args.spec else _build_spec(args)
    from .paper import PaperStudySpec
    if isinstance(spec, PaperStudySpec):
        if args.run and args.audit_environment:
            ap.error("--run and --audit-environment are mutually exclusive")
        if args.freeze:
            if args.policy is None:
                ap.error("--freeze requires --policy")
            if args.toolchain_authority is None:
                ap.error("--freeze requires --toolchain-authority")
            from merlin.common.artifacts import new_product
            from merlin.common.paths import merlin_dir
            from .freeze import freeze_study
            product = None
            frozen_out = args.frozen_out
            if frozen_out is None:
                product = new_product("paper-study", version=2, target=spec.target,
                                      sources=[str(spec.source_path)] if spec.source_path else [])
                frozen_out = product.add_artifact("frozen-study.yaml")
            runtime_paths = args.runtime_path or [
                merlin_dir() / "runtime",
                merlin_dir() / "python" / "merlin" / "runtime",
                merlin_dir() / "python" / "merlin" / "llvmlower" / "c_runtime.py",
                merlin_dir() / "python" / "merlin" / "mining" / "k1.py",
            ]
            frozen = freeze_study(
                spec, policy_path=args.policy, runtime_paths=runtime_paths,
                toolchain_authority_path=args.toolchain_authority, output_path=frozen_out)
            if product:
                product.notes = f"frozen study sha256={frozen.sha256()}"
                product.write_manifest()
            print(f"merlin-compare: wrote frozen study {frozen_out}")
            return 0
        from . import study
        try:
            out_dir = study.run(spec, live=args.run, environment_audit=args.audit_environment,
                                out_dir=args.out_root)
        except study.StudyNotReady as exc:
            print(f"merlin-compare: BLOCKED — {exc}; see {exc.output_dir / 'preflight.yaml'}",
                  file=sys.stderr)
            return 2
        print(f"merlin-compare: wrote {out_dir}")
        print(f"  {out_dir / 'report.md'}")
        print(f"  {out_dir / 'preflight.yaml'}")
        return 0
    if args.freeze:
        ap.error("--freeze requires a version: 2 paper study spec")
    out_dir = run(spec, out_root=args.out_root, run_board=args.run)
    print(f"merlin-compare: wrote {out_dir}")
    print(f"  {out_dir / 'compare.md'}")
    print(f"  {out_dir / 'manifest.yaml'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
