"""``kernel-index`` CLI: scan one source repo and emit a kernel-record index.

Thin wrapper — all logic lives in ``merlin.kernels.{ingest,emit}``. Writes one index file
(a JSON object with ``records`` + ``diagnostics``) per source/target. Artifacts go under
``output/`` (gitignored).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import itertools

from merlin.kernels.emit.kernel_record import emit_kernel_record
from merlin.kernels.ingest.autocomp import ingest_autocomp
from merlin.kernels.ingest.exo import ingest_exo, ingest_exo_schedules
from merlin.kernels.ingest.openblas import ingest_openblas
from merlin.kernels.ingest.triton import ingest_triton
from merlin.kernels.ingest.xnnpack import ingest_xnnpack

_DEFAULT_TARGET = {"xnnpack": "rvv", "autocomp": "gemmini", "exo": None,
                   "triton": "triton", "triton_cpu": "triton", "openblas": "rvv"}


def _resolve_repo(source: str, repo: str | None) -> str:
    if repo:
        return repo
    env = os.environ.get(f"MERLIN_{source.upper()}_REPO")
    if env:
        return env
    raise SystemExit(f"--repo not given and MERLIN_{source.upper()}_REPO is unset")


def _ingest(source: str, repo: str, target: str | None, limit: int | None, out_path: Path):
    diagnostics: dict = {}
    if source == "xnnpack":
        gen = ingest_xnnpack(repo, target=target or "rvv", limit=limit)
    elif source == "autocomp":
        gen = ingest_autocomp(repo, target=target, limit=limit)
    elif source == "exo":
        exo_diag: dict = {}
        diagnostics["exo"] = exo_diag
        # Dual mining: compile specs to C (breadth) AND mine the schedule .py (rich,
        # explicit decisions). Schedules need no Exo install, so they always run.
        compiled = ingest_exo(repo, target=target,
                              out_dir=str(out_path.parent / "exo_generated"),
                              limit=limit, diagnostics=exo_diag)
        schedules = ingest_exo_schedules(repo, limit=limit)
        gen = itertools.chain(compiled, schedules)
    elif source in ("triton", "triton_cpu"):
        gen = ingest_triton(repo, target=target or "triton", limit=limit, source=source)
    elif source == "openblas":
        ob_diag: dict = {}
        diagnostics["openblas"] = ob_diag
        gen = ingest_openblas(repo, target=target or "rvv", limit=limit, diagnostics=ob_diag)
    else:
        raise SystemExit(f"unknown source: {source}")
    return gen, diagnostics


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="kernel-index", description=__doc__)
    ap.add_argument("--source", required=True,
                    choices=["xnnpack", "autocomp", "exo", "triton", "triton_cpu", "openblas"])
    ap.add_argument("--repo", default=None, help="path to source repo (or MERLIN_<SRC>_REPO)")
    ap.add_argument("--target", default=None, help="ISA target (default per source)")
    ap.add_argument("--out", required=True, help="output index json path")
    ap.add_argument("--limit", type=int, default=None, help="cap kernels (dev runs)")
    ap.add_argument("--json", action="store_true",
                    help="print a machine-readable summary JSON to stdout")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")

    source = args.source
    repo = _resolve_repo(source, args.repo)
    target = args.target if args.target is not None else _DEFAULT_TARGET.get(source)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    gen, diagnostics = _ingest(source, repo, target, args.limit, out_path)
    records = []
    errors = 0
    for nk in gen:
        try:
            records.append(emit_kernel_record(nk))
        except Exception as e:  # never let one bad kernel kill the run
            errors += 1
            logging.warning("skip record for %s (%s)", nk.path, e)
    payload = {
        "source": source,
        "target": target,
        "repo": str(Path(repo).resolve()),  # lets kernel-audit re-read source context
        "count": len(records),
        "errors": errors,
        "diagnostics": diagnostics,
        "records": records,
    }
    out_path.write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")
    human = (f"indexed {len(records)} kernels ({source}/{target}) -> {out_path}"
             + (f"  [{errors} skipped]" if errors else ""))
    if args.json:
        print(json.dumps({"source": source, "target": target, "count": len(records),
                          "errors": errors, "diagnostics": diagnostics,
                          "out": str(out_path)}, indent=1, default=str))
        print(human, file=sys.stderr)
    else:
        print(human)
    return 0


if __name__ == "__main__":
    sys.exit(main())
