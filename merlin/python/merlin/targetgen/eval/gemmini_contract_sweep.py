"""Migrate the Gemmini conformance battery THROUGH the experiment-ABI contract runner.

For each rung (C0..Q1t) this emits the rung's interface.mlir (from its command buffer, via the
merlin_iface grammar), then certifies it through merlin.targetgen.oot_runner against the
`merlin_native_v0` package — so the headline battery now flows through the package contract
(subprocess + file boundary) and lands as recorded aet runs. Resumable: a run with an existing
run_manifest.yaml is skipped unless --force.

The recorded runs are summarizable by the SAME manifest reader as the original battery
(merlin.targetgen.eval.gemmini_dispatcher.summarize_from_manifests), which is the acceptance test: the
headline table regenerates from contract-runner manifests alone.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from merlin.targetgen.eval.gemmini_conformance import build, RUNGS, QUANT_RUNGS
from merlin.targetgen.contract.interface_emit import emit_interface_mlir
from merlin.targetgen.oot_runner import certify

ALL_RUNGS = list(RUNGS) + list(QUANT_RUNGS)
NATIVE_PKG = "out/artifacts/targets/gemmini/merlin_native_v0"


def emit_rung_interface(rung: str, out_dir: str | Path) -> Path:
    """Write the rung's interface.mlir (grammar text) and return its path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cb = build(rung)
    path = out_dir / f"{rung}.interface.mlir"
    path.write_text(emit_interface_mlir(cb), encoding="utf-8")
    return path


def run_sweep(rungs: list[str], simulators: list[str], *, runs_root: str | Path,
              package: str | Path = NATIVE_PKG, contract: str = "merlin/contract",
              force: bool = False, timeout: int = 600) -> list[dict[str, Any]]:
    """Cartesian rung x simulator sweep through oot_runner. Returns per-run summaries."""
    runs_root = Path(runs_root)
    inputs_dir = runs_root / "inputs"
    out: list[dict[str, Any]] = []
    for rung in rungs:
        iface = emit_rung_interface(rung, inputs_dir)
        for sim in simulators:
            run_id = f"{rung}_{sim}_oot_native"
            existing = runs_root / "runs" / "gemmini-contract" / run_id / "run_manifest.yaml"
            if existing.is_file() and not force:
                out.append({"run_id": run_id, "skipped": True})
                continue
            res = certify(package, iface, runs_root=runs_root, run_id=run_id,
                          simulator=sim, contract=contract, timeout=timeout)
            out.append({"run_id": run_id, "rung": rung, "simulator": sim,
                        "status": res["status"], "oracle": res["oracle"]})
            print(f"{run_id:34s} status={res['status']:5s} "
                  f"oracle={res['oracle'].get('kind')} cycles={res['oracle'].get('cycles')}")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Migrate the Gemmini battery through the contract runner")
    ap.add_argument("--rungs", default=",".join(ALL_RUNGS))
    ap.add_argument("--simulators", default="spike,verilator")
    ap.add_argument("--package", default=NATIVE_PKG)
    ap.add_argument("--runs-root", default="out/runs/gemmini_cert_oot")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--timeout", type=int, default=600)
    args = ap.parse_args(argv)
    run_sweep(args.rungs.split(","), args.simulators.split(","), runs_root=args.runs_root,
              package=args.package, force=args.force, timeout=args.timeout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
