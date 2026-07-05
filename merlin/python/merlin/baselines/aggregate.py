"""Collect per-framework results into the cross-framework matrix (merlin vs the 5 baselines).

Reads ``baseline_result.json`` files (written via :meth:`BaselineResult.write`) and renders a
framework × model matrix — status + E2E cycles + RVV% per cell — as markdown and CSV. Rendering is a
pure function of a result list so it is unit-testable; :func:`collect_dir` walks a measurements tree.

The matrix is deliberately honest: ``not_built`` / ``not_run`` cells show the gap, never a blank that
could read as "fine", and every cell carries its RVV coverage so a fast-but-scalar result can't be
mistaken for a fast-and-vectorized one.
"""
from __future__ import annotations

from pathlib import Path

from merlin.baselines.contract import BaselineResult


def collect_dir(root: str | Path) -> list[BaselineResult]:
    """Load every ``baseline_result.json`` under a directory tree."""
    root = Path(root)
    out: list[BaselineResult] = []
    for p in sorted(root.rglob("baseline_result.json")):
        try:
            out.append(BaselineResult.load(p))
        except Exception:  # noqa: BLE001 - skip malformed, don't crash the whole matrix
            continue
    return out


def _cell(r: BaselineResult | None) -> str:
    if r is None:
        return "—"
    st = r.status()
    if st != "pass":
        return st
    cyc = f"{r.e2e_cycles/1e6:.1f}M" if r.e2e_cycles else "?"
    cov = f"{100*r.rvv_coverage_overall:.0f}%RVV" if r.rvv_coverage_overall is not None else "?RVV"
    return f"pass {cyc} {cov}"


def render_markdown(results: list[BaselineResult]) -> str:
    """framework × model matrix as a GitHub-flavored markdown table."""
    frameworks = sorted({r.framework for r in results})
    modelvars = sorted({(r.model, r.variant) for r in results})
    idx = {(r.framework, r.model, r.variant): r for r in results}

    header = "| model | " + " | ".join(frameworks) + " |"
    sep = "|" + "---|" * (len(frameworks) + 1)
    lines = [header, sep]
    for model, variant in modelvars:
        row = [f"{model}/{variant}"]
        for fw in frameworks:
            row.append(_cell(idx.get((fw, model, variant))))
        lines.append("| " + " | ".join(row) + " |")

    # honesty footer: count gaps + scalar fallbacks
    n_pass = sum(1 for r in results if r.passed)
    n_gap = sum(1 for r in results if r.status() in ("not_built", "not_run"))
    n_fallback = sum(len(r.scalar_fallbacks) for r in results)
    lines += [
        "",
        f"**{n_pass}/{len(results)} pass** · {n_gap} not-built/not-run gaps · "
        f"{n_fallback} labeled scalar fallbacks across all cells.",
        "",
        "_Legend: `pass <e2e-cycles> <RVV%>` | `fail` (ran, missed tolerance) | "
        "`not_run`/`not_built` (explicit gap). Cycles are K1 rdtime estimates, not cycle-accurate._",
    ]
    return "\n".join(lines)


def render_csv(results: list[BaselineResult]) -> str:
    cols = ["framework", "model", "variant", "status", "e2e_cycles", "e2e_wall_ns",
            "cos", "rel", "rvv_coverage_overall", "n_scalar_fallbacks", "gap_reason"]
    rows = [",".join(cols)]
    for r in sorted(results, key=lambda x: (x.model, x.variant, x.framework)):
        rows.append(",".join(str(v) for v in [
            r.framework, r.model, r.variant, r.status(),
            r.e2e_cycles if r.e2e_cycles is not None else "",
            r.e2e_wall_ns if r.e2e_wall_ns is not None else "",
            r.cos if r.cos is not None else "",
            r.rel if r.rel is not None else "",
            r.rvv_coverage_overall if r.rvv_coverage_overall is not None else "",
            len(r.scalar_fallbacks),
            (r.gap_reason or "").replace(",", ";"),
        ]))
    return "\n".join(rows)
