"""Sweep {op} x {dtype} x {model} and report the RUNTIME ESCAPES in each emitted compute region.

Generalizes a point fix into an audit. One escape -- a per-tile ``memrefCopy`` that bufferization
left behind as a self-copy -- was 77% of the f32 GEMM's retired instructions while being invisible to
every existing diagnostic, and enabling the fix on the int8 path moved it only 0.03%. So "is there
another one?" is a question per (op, dtype, model) cell, not a question with one answer.

Running the matrix immediately corrected the story that motivated it. The int8 no-op had been read as
"int8 lowers differently and has no self-copy"; in fact the int8 matmul DOES emit the in-loop
``memrefCopy`` and ``erase_self_copy`` DOES remove it from the emitted code. Whatever explains the
0.03% is not the absence of the escape. That is the value of sweeping rather than generalizing from
one measurement.

Each cell BUILDS (host-only cross-compile; no board, no measurement) and lifts
:func:`merlin.kernels.escape_audit.audit`. Cells are independent, so they fan out across threads.

What the sweep reports per cell is the *screening* signal:
  * ``in_loop``  -- escapes enclosed by a back-edge. Per-iteration overhead, so cost scales with the
                   problem. This is the only column that has ever indicated a real defect.
  * ``prologue`` -- escapes not enclosed by any loop. Once per inference: allocation, argument
                   marshalling, result copy-out. Real work; erasing these would be a correctness bug.
  * ``depths_reliable`` -- whether the depth NUMBERS can be believed (see ``_is_containment_chain``).
                   ``in_loop`` still stands when this is False; only the count is untrustworthy.

Screening only ranks candidates. An in-loop escape is a *suspect*, not a finding: confirming one
means proving the call is redundant at the IR level and then measuring it. A cell whose artifacts
could not be read is reported ``unknown`` and never as clean.
"""
from __future__ import annotations

import json
import tempfile
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

F32_PKG = "out/artifacts/targets/rvv/hand_v0"
INT8_PKG = "out/artifacts/targets/rvv/hand_v0_int8"


@dataclass
class EscapeCell:
    """One (workload, dtype, compiler-feature) build to audit."""

    name: str                                  # cell label, e.g. "matmul:f32:baseline"
    kind: str                                  # "op" | "model"
    dtype: str                                 # "f32" | "int8"
    features: tuple[str, ...] = ()             # default-off compiler features to enable
    op: str | None = None                      # op generator name, for kind="op"
    shape: dict[str, int] = field(default_factory=dict)
    model_dir: str | None = None               # bundle path, for kind="model"

    @property
    def pkg_dir(self) -> str:
        return INT8_PKG if self.dtype == "int8" else F32_PKG


@dataclass
class EscapeResult:
    cell: str
    kind: str
    dtype: str
    features: tuple[str, ...]
    status: str                                # "ok" | "unknown" | "build_failed"
    in_loop: dict[str, int] | None = None
    prologue: dict[str, int] | None = None
    max_depth: int | None = None
    depths_reliable: bool | None = None
    sites: list[dict[str, Any]] = field(default_factory=list)
    scope: tuple[str, ...] | None = None
    note: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "cell": self.cell, "kind": self.kind, "dtype": self.dtype,
            "features": list(self.features), "status": self.status,
            "in_loop": self.in_loop, "prologue": self.prologue,
            "max_depth": self.max_depth, "depths_reliable": self.depths_reliable,
            "sites": self.sites,
            "scope": list(self.scope) if self.scope else None, "note": self.note,
        }


def _build_workload(cell: EscapeCell, out: Path) -> Path:
    """Materialize the cell's model bundle (generating an op workload when kind='op')."""
    if cell.kind == "model":
        return Path(cell.model_dir)
    from . import workloads
    gen = getattr(workloads, f"gen_{cell.op}_f32")
    return Path(gen(out, **cell.shape))


def run_cell(cell: EscapeCell, *, timeout: int = 3600) -> EscapeResult:
    """Build ONE cell host-side and lift its escape report. Never raises: a failed build is a
    reported ``build_failed`` cell, so one bad cell cannot silently shrink the matrix."""
    from ..kernels.escape_audit import audit
    from .registry import load_rvv_package

    base = EscapeResult(cell=cell.name, kind=cell.kind, dtype=cell.dtype,
                        features=cell.features, status="build_failed")
    work = Path(tempfile.mkdtemp(prefix=f"esc_{cell.kind}_"))
    try:
        from . import k1
        bundle = _build_workload(cell, work)
        pkg = load_rvv_package(cell.pkg_dir)
        if cell.features:
            pkg = replace(pkg, run_id=f"esc_{cell.name}",
                          compiler_features=list(cell.features))
        bwork = work / "build"
        elf = k1.build_k1_binary(bundle, bwork, pkg)
        rep = audit(bwork / "model.o", elf)
        if not rep.readable:
            return replace(base, status="unknown",
                           note="object or ELF unreadable (disassembly/symbol table)")
        if rep.loop_structure_suspect:
            # No back-edge anywhere in the compute scope: the loop structure was almost certainly
            # unreadable, so every depth read 0. Reporting "no in-loop escapes" here would be the
            # exact false negative this sweep exists to prevent.
            return replace(base, status="unknown",
                           note="no loops found in compute scope -- loop structure unreadable, "
                                "depths untrustworthy (audit the LINKED ELF, not the object)")
        return EscapeResult(
            cell=cell.name, kind=cell.kind, dtype=cell.dtype, features=cell.features,
            status="ok", in_loop=rep.in_loop_counts(),
            prologue={k: v for k, v in (rep.site_counts() or {}).items()
                      if v - (rep.in_loop_counts() or {}).get(k, 0) > 0},
            max_depth=rep.max_depth(), scope=rep.scope,
            depths_reliable=rep.depths_reliable,
            sites=[{"helper": s.helper, "caller": s.caller, "addr": s.addr,
                    "loop_depth": s.loop_depth, "depth_reliable": s.depth_reliable}
                   for s in rep.sites],
        )
    except Exception as exc:                    # noqa: BLE001 -- a failed cell must be reported
        return replace(base, note=f"{type(exc).__name__}: {exc}",
                       sites=[{"traceback": traceback.format_exc()[-1200:]}])


def run_sweep(cells: list[EscapeCell], *, workers: int = 4, retry_serial: bool = True
              ) -> list[EscapeResult]:
    """Audit every cell. Builds are independent and host-only, so they fan out.

    Failed cells are retried ONCE serially. Parts of the lowering front-end touch a shared MLIR
    context and are not thread-safe under fan-out (observed: a whole-model cell dying with "Can't add
    to a block an operation already attached to a block", which then succeeded alone). Without the
    retry that race silently removes cells from the matrix -- and a missing cell in an audit reads as
    "nothing found there", which is the one conclusion this tool must never fabricate.
    """
    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(run_cell, cells))
    if not retry_serial:
        return results
    by_name = {c.name: c for c in cells}
    for i, r in enumerate(results):
        if r.status == "build_failed" and r.cell in by_name:
            again = run_cell(by_name[r.cell])
            if again.status != "build_failed":
                again.note = f"(recovered on serial retry) {again.note}".strip()
                results[i] = again
    return results


def default_cells(*, models: dict[str, list[str]] | None = None,
                  dtypes: tuple[str, ...] = ("f32", "int8"),
                  features: tuple[tuple[str, ...], ...] = ((), ("erase_self_copy",)),
                  ) -> list[EscapeCell]:
    """The standard matrix: the op family x dtypes x {baseline, erase_self_copy}, plus whole models.

    ``erase_self_copy`` is swept as a second arm on every cell so the report shows not just where
    escapes are but which ones the existing fix already reaches -- the int8 no-op is a RESULT, and it
    should be visible in the matrix rather than assumed.
    """
    ops = [
        ("matmul", {"M": 128, "N": 128, "K": 128}),
        ("batch_matmul", {"B": 4, "M": 32, "N": 8, "K": 32}),
        ("conv2d_as_matmul", {"M": 64, "N": 16, "K": 27}),
        ("softmax", {"M": 64, "N": 64}),
        ("gelu", {"N": 16384}),
        ("sigmoid", {"N": 16384}),
        ("silu", {"N": 16384}),
    ]
    cells: list[EscapeCell] = []
    for dtype in dtypes:
        for feats in features:
            tag = "+".join(feats) if feats else "baseline"
            for op, shape in ops:
                cells.append(EscapeCell(name=f"{op}:{dtype}:{tag}", kind="op", dtype=dtype,
                                        features=feats, op=op, shape=shape))
            # Whole models are dtype-matched: the int8 arm gets the int8 recapture bundles, since a
            # dtype is a property of the bundle as much as of the package.
            for md in (models or {}).get(dtype, []):
                cells.append(EscapeCell(name=f"{Path(md).name}:{dtype}:{tag}", kind="model",
                                        dtype=dtype, features=feats, model_dir=md))
    return cells


def summarize(results: list[EscapeResult]) -> dict[str, Any]:
    """Roll the matrix up to the question it exists to answer: which cells have an in-loop escape."""
    suspects = [r.cell for r in results if r.status == "ok" and r.in_loop]
    unknown = [r.cell for r in results if r.status == "unknown"]
    failed = [r.cell for r in results if r.status == "build_failed"]
    helpers: dict[str, int] = {}
    for r in results:
        for h, n in (r.in_loop or {}).items():
            helpers[h] = helpers.get(h, 0) + n
    return {
        "n_cells": len(results), "n_ok": sum(1 for r in results if r.status == "ok"),
        "n_suspect": len(suspects), "suspect_cells": suspects,
        "in_loop_helpers": helpers, "unknown_cells": unknown, "failed_cells": failed,
    }


def main(argv: list[str] | None = None) -> int:
    import argparse

    from ..common.artifacts import new_product

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models-f32", default="", help="comma-separated f32 model bundle dirs")
    ap.add_argument("--models-int8", default="", help="comma-separated int8 model bundle dirs")
    ap.add_argument("--dtypes", default="f32,int8")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="", help="write the report here instead of a new product dir")
    a = ap.parse_args(argv)

    models = {"f32": [m for m in a.models_f32.split(",") if m],
              "int8": [m for m in a.models_int8.split(",") if m]}
    cells = default_cells(models=models, dtypes=tuple(a.dtypes.split(",")))
    results = run_sweep(cells, workers=a.workers)
    report = {"summary": summarize(results), "cells": [r.to_json() for r in results]}

    if a.out:
        dest = Path(a.out)
        dest.parent.mkdir(parents=True, exist_ok=True)
    else:
        dest = Path(new_product("runtime-escapes", target="rvv", version=1)) / "escapes.json"
    dest.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
