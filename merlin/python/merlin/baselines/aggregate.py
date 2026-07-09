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


def dedupe_latest(results: list[BaselineResult]) -> list[BaselineResult]:
    """Collapse many historical measurements to ONE per (framework, model, variant) cell.

    Honest rule: a result that actually **executed** (``ran=True``: pass OR fail) always beats one
    that did not (``not_run`` / ``not_built``). ``not_run`` is the *absence* of a run on that pass —
    a timed-out or skipped re-verification must never erase an earlier genuine on-board pass/fail.
    Among executed results the latest timestamp wins (so a real later ``fail`` correctly supersedes a
    stale pass); among non-executed results the latest wins too. Rank key = ``(ran, timestamp)``.
    """
    best: dict[tuple[str, str, str], BaselineResult] = {}
    for r in results:
        key = (r.framework, r.model, r.variant)
        cur = best.get(key)
        rank = (1 if r.ran else 0, r.timestamp or "")
        crank = (1 if cur.ran else 0, cur.timestamp or "") if cur is not None else (-1, "")
        if rank >= crank:
            best[key] = r
    return list(best.values())


def _cell(r: BaselineResult | None) -> str:
    if r is None:
        return "—"
    st = r.status()
    if st != "pass":
        return st
    # Prefer wall-clock (uniform ms/s across arms → apples-to-apples), fall back to the rdtime
    # cycle-estimate, so a passing cell always shows a real latency rather than "?".
    if r.e2e_wall_ns:
        ms = r.e2e_wall_ns / 1e6
        lat = f"{ms:.0f}ms" if ms < 10000 else f"{ms/1000:.1f}s"
    elif r.e2e_cycles:
        lat = f"{r.e2e_cycles/1e6:.1f}Mc"
    else:
        lat = "?"
    cov = f"{100*r.rvv_coverage_overall:.0f}%RVV" if r.rvv_coverage_overall is not None else "?RVV"
    return f"pass {lat} {cov}"


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


def main() -> None:
    """Regenerate the cross-framework K1 matrix product from the measurements tree.

    Reproducible: ``.venv/bin/python -m merlin.baselines.aggregate`` collects every
    ``baseline_result.json`` under ``artifacts/measurements/k1_spacemit/``, dedupes to the latest
    executed result per cell, and writes ``matrix.{md,csv}`` as a versioned ``compare`` product.
    """
    import argparse

    from merlin.common.artifacts import new_product
    from merlin.common.paths import repo_root

    ap = argparse.ArgumentParser(description="Cross-framework K1-RVV matrix (merlin vs baselines)")
    ap.add_argument("--measurements", default=None,
                    help="measurements root (default artifacts/measurements/k1_spacemit)")
    args = ap.parse_args()

    root = Path(args.measurements) if args.measurements else \
        repo_root() / "artifacts" / "measurements" / "k1_spacemit"
    results = dedupe_latest(collect_dir(root))
    md, csv = render_markdown(results), render_csv(results)

    prod = new_product("compare", version=1,
                       notes="cross-framework K1-RVV matrix (deduped latest executed per cell)")
    (prod.path / "matrix.md").write_text(md)
    (prod.path / "matrix.csv").write_text(csv)
    prod.add_artifact("matrix.md")
    prod.add_artifact("matrix.csv")
    prod.write_manifest()
    print(prod.path)
    print(md)


if __name__ == "__main__":
    main()
