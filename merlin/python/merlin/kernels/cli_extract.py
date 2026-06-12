"""``kernel-extract`` CLI: aggregate kernel indexes into abstractions + policies + report.

Reads one or more index files (from ``kernel-index``), writes a flat feature table
(``kernel_features.jsonl``, optional ``.parquet``), the promoted ``abstraction_candidates``
and ``policy_rules`` YAML, and the markdown report. Thin wrapper over
``merlin.kernels.{policy,report}``.
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import sys
from pathlib import Path

import yaml

from merlin.kernels import invariants as invariants_mod
from merlin.kernels import policy, report, validate


def _load_indexes(patterns: list[str]) -> tuple[list[dict], dict]:
    records: list[dict] = []
    diagnostics: dict = {}
    paths: list[str] = []
    for pat in patterns:
        paths.extend(sorted(glob.glob(pat)))
    if not paths:
        raise SystemExit(f"no index files matched: {patterns}")
    for p in paths:
        data = json.loads(Path(p).read_text(encoding="utf-8"))
        if isinstance(data, dict):
            records.extend(data.get("records", []))
            for k, v in (data.get("diagnostics") or {}).items():
                diagnostics[k] = v
        elif isinstance(data, list):  # tolerate a bare list of records
            records.extend(data)
    return records, diagnostics


def _flat_row(rec: dict) -> dict:
    row = {k: rec.get(k) for k in ("source", "target", "op", "dtype", "shape_family", "path")}
    row["motifs"] = (rec.get("evidence", {}) or {}).get("motifs", [])
    for fk, fv in (rec.get("features", {}) or {}).items():
        row[f"feat_{fk}"] = fv
    return row


def _write_jsonl(rows: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, default=str) + "\n")


def _maybe_parquet(rows: list[dict], path: Path) -> None:
    try:
        import pyarrow as pa  # noqa
        import pyarrow.parquet as pq
    except Exception as e:
        logging.warning("parquet requested but pyarrow unavailable (%s); wrote JSONL only. "
                        "Install with `pip install -e .[kernels-parquet]`.", e)
        return
    # normalize: stringify list/dict columns for a stable schema
    norm = [{k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in r.items()}
            for r in rows]
    table = pa.Table.from_pylist(norm)
    pq.write_table(table, str(path))
    logging.info("wrote parquet -> %s", path)


def _llm_summary(stats, promo) -> str | None:
    """One-shot advisory summary over the *aggregated* motif table (never per kernel).

    Uses :func:`merlin.common.llm.summarize`, which synthesizes a narrative deterministically
    and upgrades to a real LLM call when an Anthropic SDK + key are present. Advisory only —
    the deterministic artifacts remain the source of truth.
    """
    from merlin.common.llm import summarize
    table = {m: {"kernels": s.kernel_count, "sources": sorted(s.sources)}
             for m, s in stats.items()}
    return summarize(table, [r["policy"] for r in promo.rules])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="kernel-extract", description=__doc__)
    ap.add_argument("--inputs", nargs="+", required=True, help="index json files or globs")
    ap.add_argument("--out", required=True, help="abstraction_candidates.yaml output")
    ap.add_argument("--policies", required=True, help="policy_rules.yaml output")
    ap.add_argument("--interfaces", default=None, help="interface_candidates.yaml output")
    ap.add_argument("--runtime", default=None, help="runtime_candidates.yaml output")
    ap.add_argument("--dialect-reqs", default=None, help="dialect_requirements.yaml output (L6)")
    ap.add_argument("--llvm-reqs", default=None, help="llvm_requirements.yaml output (L8)")
    ap.add_argument("--report", default=None, help="kernel_mining_report.md output")
    ap.add_argument("--features", default=None, help="kernel_features.jsonl output")
    ap.add_argument("--min-kernels", type=int, default=10, help="single-source promotion gate")
    ap.add_argument("--parquet", action="store_true", help="also write a parquet feature table")
    ap.add_argument("--llm-summary", action="store_true", help="advisory LLM pass over the motif table")
    ap.add_argument("--plots", action="store_true",
                    help="write evaluation PNGs under <out_dir>/plots (needs matplotlib)")
    ap.add_argument("--json", action="store_true",
                    help="print a machine-readable summary JSON to stdout")
    ap.add_argument("--strict", action="store_true",
                    help="exit 2 when a consistency invariant is violated")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")

    records, diagnostics = _load_indexes(args.inputs)
    records, dedup_diag = policy.dedupe_records(records)
    if dedup_diag["duplicates_skipped"]:
        diagnostics["dedup"] = dedup_diag
        logging.info("deduplicated %d kernels vendored across sources: %s",
                     dedup_diag["duplicates_skipped"], dedup_diag["by_source"])
    stats = policy.aggregate(records)
    promo = policy.promote(stats, min_kernels=args.min_kernels, records=records)
    validation = validate.validate_policies(promo.rules)
    inv = invariants_mod.check_invariants(records, stats, promo)

    out_dir = Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # feature table
    rows = [_flat_row(r) for r in records]
    feat_path = Path(args.features) if args.features else out_dir / "kernel_features.jsonl"
    _write_jsonl(rows, feat_path)
    if args.parquet:
        _maybe_parquet(rows, feat_path.with_suffix(".parquet"))

    # artifacts
    Path(args.out).write_text(yaml.safe_dump(promo.candidates, sort_keys=False), encoding="utf-8")
    Path(args.policies).write_text(yaml.safe_dump(promo.rules, sort_keys=False), encoding="utf-8")
    iface_path = Path(args.interfaces) if args.interfaces else out_dir / "interface_candidates.yaml"
    iface_path.write_text(yaml.safe_dump(promo.interfaces, sort_keys=False), encoding="utf-8")
    rt_path = Path(args.runtime) if args.runtime else out_dir / "runtime_candidates.yaml"
    rt_path.write_text(yaml.safe_dump(promo.runtime_candidates, sort_keys=False), encoding="utf-8")
    dreq_path = Path(args.dialect_reqs) if args.dialect_reqs else out_dir / "dialect_requirements.yaml"
    dreq_path.write_text(yaml.safe_dump(promo.dialect_requirements, sort_keys=False), encoding="utf-8")
    lreq_path = Path(args.llvm_reqs) if args.llvm_reqs else out_dir / "llvm_requirements.yaml"
    lreq_path.write_text(yaml.safe_dump(promo.llvm_requirements, sort_keys=False), encoding="utf-8")

    plot_paths: list = []
    if args.plots:
        from merlin.kernels import plots as plots_mod
        plot_dir = (Path(args.report).parent if args.report else out_dir) / "plots"
        plot_paths = plots_mod.generate_all(records, stats, promo, validation, plot_dir,
                                            min_kernels=args.min_kernels)

    summary = _llm_summary(stats, promo) if args.llm_summary else None
    if args.report:
        md = report.write_report(records, stats, promo, diagnostics=diagnostics,
                                 min_kernels=args.min_kernels, llm_summary=summary,
                                 validation=validation, invariants=inv,
                                 plot_paths=plot_paths)
        Path(args.report).write_text(md, encoding="utf-8")

    human = (
        f"aggregated {len(records)} kernels -> "
        f"{len(promo.candidates)} abstractions, {len(promo.interfaces)} interfaces, "
        f"{len(promo.rules)} policies, {len(promo.runtime_candidates)} runtime candidates, "
        f"{len(promo.dialect_requirements)} dialect reqs (L6), "
        f"{len(promo.llvm_requirements)} llvm reqs (L8)\n"
        f"  features: {feat_path}\n"
        f"  candidates: {args.out}  | interfaces: {iface_path}  | runtime: {rt_path}\n"
        f"  dialect reqs: {dreq_path}  | llvm reqs: {lreq_path}\n"
        f"  policies: {args.policies}"
        + (f"\n  report: {args.report}" if args.report else "")
        + (f"\n  plots: {len(plot_paths)} PNGs in {plot_paths[0].parent}" if plot_paths else "")
        + (f"\n  INVARIANT VIOLATIONS: {inv['total_violations']}"
           if inv["total_violations"] else ""))
    if args.json:
        print(json.dumps({
            "kernels": len(records),
            "sources": sorted({r.get("source", "?") for r in records}),
            "motifs": {m: {"kernels": s.kernel_count, "sources": sorted(s.sources)}
                       for m, s in stats.items()},
            "promoted": sorted(promo.promoted),
            "policies": [r["policy"] for r in promo.rules],
            "abstractions": [c["name"] for c in promo.candidates],
            "interfaces": [i["name"] for i in promo.interfaces],
            "runtime_candidates": [r["name"] for r in promo.runtime_candidates],
            "validation": validation,
            "invariants": {"total_violations": inv["total_violations"],
                           "surprises": inv["surprises"]},
            "artifacts": {"features": str(feat_path), "candidates": args.out,
                          "policies": args.policies, "interfaces": str(iface_path),
                          "runtime": str(rt_path), "dialect_requirements": str(dreq_path),
                          "llvm_requirements": str(lreq_path),
                          "report": args.report,
                          "plots": [str(p) for p in plot_paths]},
        }, indent=1, default=str))
        print(human, file=sys.stderr)
    else:
        print(human)
    if args.strict and inv["total_violations"]:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
