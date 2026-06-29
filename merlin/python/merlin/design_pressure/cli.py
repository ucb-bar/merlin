"""``merlin-design-pressure`` CLI: workload region -> pressure vector -> contracts.

Thin wrapper — logic lives in ``merlin.design_pressure``. Reads a workload region (a named
synthetic ``vla_action_chunk_decode``, a benchmark name, or an explicit YAML path) and writes
``design_pressure.json`` + ``candidate_contracts.yaml`` under ``output/dse/<workload>/``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from merlin.common import paths
from merlin.common.artifacts import Artifact
from merlin.common.yaml import dump_yaml, load_yaml
from merlin.design_pressure import synthesize as S
from merlin.design_pressure.emit.candidate_contracts import emit_candidate_contracts
from merlin.design_pressure.emit.design_pressure import emit_design_pressure
from merlin.design_pressure.pressure_vector import compute_rpv
from merlin.design_pressure.workloads.vla_action_chunk_decode import build_region


def _load_region(args) -> dict:
    if args.from_mlir:
        from merlin.design_pressure.ingest import mlir_m2m
        if not mlir_m2m.available():
            raise SystemExit("--from-mlir needs xdsl: uv sync --extra xdsl")
        return mlir_m2m.region_from_mlir(args.from_mlir, region_id=args.region_id, H=args.H)
    if args.region_yaml:
        return load_yaml(args.region_yaml)
    if args.workload == "vla_action_chunk_decode":
        return build_region(H=args.H, reuse_count=args.reuse, dtype=args.dtype,
                            epilogue=not args.no_epilogue, K=args.K)
    # Otherwise treat --workload as a benchmark name under semantic_memory.
    bench = paths.merlin_dir() / "benchmarks" / "semantic_memory" / f"{args.workload}.yaml"
    if bench.is_file():
        return load_yaml(bench)
    raise SystemExit(f"unknown workload '{args.workload}' (no benchmark {bench})")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="merlin-design-pressure", description=__doc__)
    ap.add_argument("--workload", default="vla_action_chunk_decode",
                    help="synthetic name, benchmark name, or use --region-yaml")
    ap.add_argument("--region-yaml", default=None, help="explicit workload_region YAML path")
    ap.add_argument("--from-mlir", default=None, help="model2MLIR file to extract a region from")
    ap.add_argument("--region-id", default=None, help="m2m.region_id to select (with --from-mlir)")
    ap.add_argument("--H", type=int, default=16, help="action horizon (synthetic)")
    ap.add_argument("--reuse", type=int, default=None, help="weight reuse (default = H)")
    ap.add_argument("--dtype", default="i8")
    ap.add_argument("--K", type=int, default=256, help="contraction depth (synthetic)")
    ap.add_argument("--no-epilogue", action="store_true")
    ap.add_argument("--resident-store-bytes", type=int, default=None)
    ap.add_argument("--out", default=None, help="output dir (default output/dse/<workload>)")
    args = ap.parse_args(argv)

    region = _load_region(args)
    workload = region.get("name", args.workload)
    rpv = compute_rpv(region)
    pol = S.load_policies()
    feats = S.recommended_features(rpv, pol, resident_store_bytes=args.resident_store_bytes)
    contracts = S.legal_contracts(rpv, pol, resident_store_bytes=args.resident_store_bytes)

    dp = emit_design_pressure(workload, rpv["cutpoints"], rpv, feats)
    cc = emit_candidate_contracts(workload, contracts)

    out = Path(args.out) if args.out else paths.repo_root() / "artifacts" / "design-pressure" / workload
    Artifact("design_pressure.json", json.dumps(dp, indent=2, sort_keys=True)).write(out)
    Artifact("candidate_contracts.yaml", dump_yaml(cc)).write(out)

    print(f"workload={workload}  recommended_features={feats}")
    print(f"legal_contracts={[c['name'] for c in contracts if c['legal']]}")
    print(f"artifacts -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
