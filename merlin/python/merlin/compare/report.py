"""LAYER 5 — REPORT + MANIFEST: the dashboard ``compare.md`` and the deterministic ``manifest.yaml``.

``compare.md`` = measured table + per-config CCA + attribution (measured gap paired with structural
divergences + routed compiler actions) + figure references.
``manifest.yaml`` = spec + git commit + source JSONs + figure list (deterministic; a re-run over the
same cached sources at the same commit reproduces the same manifest sans timestamp).
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from .attribution import Attribution
from .empirical import Measurement


def _git_commit(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def _fmt_wall(m: Measurement) -> str:
    if m.status != "measured" or m.value is None:
        return "not_measured"
    return f"{m.value / 1e9:.4f}s"


def measured_table_md(spec, measurements: dict[tuple[str, str], Measurement]) -> str:
    cfgs = [c.name for c in spec.configs]
    lines = ["| workload | " + " | ".join(cfgs) + " |",
             "|" + "---|" * (len(cfgs) + 1)]
    for w in spec.workloads:
        cells = []
        for cn in cfgs:
            m = measurements.get((cn, w.name))
            if m is None:
                cells.append("—")
            elif m.status != "measured":
                cells.append("not_measured")
            else:
                extra = f" (cos {m.cos:.4f})" if m.cos is not None else ""
                sp = f" ±{m.spread_pct:.1f}%" if m.spread_pct is not None else ""
                cells.append(f"{_fmt_wall(m)}{sp}{extra}")
        lines.append(f"| {w.name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def cca_table_md(ccas: dict[str, Any]) -> str:
    lines = ["| config | contraction | acc_resident | nr_is_vsetvlmax | sew/lmul | vfmacc(.vf/.vv) |",
             "|---|---|---|---|---|---|"]
    for name, cca in ccas.items():
        if cca is None:
            lines.append(f"| {name} | (no vector matmul decode — scalar/baseline) | | | | |")
            continue
        c = cca.compute
        v = cca.vector
        vf = cca.provenance.get("fma_loop_vfmacc_vf")
        vv = cca.provenance.get("fma_loop_vfmacc_vv")
        lines.append(
            f"| {name} | {c.contraction_form} | {c.accumulator_resident} | {c.nr_is_vsetvlmax} | "
            f"{v.sew if v else '?'}/{v.lmul if v else '?'} | vf={vf}, vv={vv} |")
    return "\n".join(lines)


def attribution_md(attrs: list[Attribution]) -> str:
    if not attrs:
        return "_No (ours, expert) pair had both cells measured; no attribution._"
    blocks = []
    for a in attrs:
        m = a.measured
        verdict = ("BEATS" if m["ours_faster"] else f"{m['pct_of_expert']}% of")
        head = (f"### {a.workload}: `{a.ours_config}` vs `{a.expert_config}` — "
                f"ours {verdict} expert")
        rows = [head, "",
                f"- measured: ours={m['ours_value']/1e9:.4f}s, expert={m['expert_value']/1e9:.4f}s, "
                f"ratio(ours/expert)={m['ratio_ours_over_expert']:.2f}x"]
        if a.divergences:
            rows.append("- structural divergences (expert vs ours):")
            for d in a.divergences:
                rows.append(f"    - `{d.axis}`: expert={d.expert!r} vs ours={d.ours!r}")
        if a.actions:
            rows.append("- routed compiler actions:")
            for act in a.actions:
                fk = "forkable-now" if act.forkable_now else "deferred"
                rows.append(f"    - [{act.action_class}] `{act.target_seam}` ({fk}) — "
                            f"{act.expected_effect}")
        if a.unrouted:
            rows.append("- unrouted divergences (surfaced, not dropped): "
                        + ", ".join(f"`{d.axis}`" for d in a.unrouted))
        for n in a.notes:
            rows.append(f"- note: {n}")
        blocks.append("\n".join(rows))
    return "\n\n".join(blocks)


def write_report(out_dir: Path, *, spec, measurements, ccas, attrs, figures,
                 root: Path, gap_axes: set[str]) -> Path:
    out_dir = Path(out_dir)
    parts = [
        f"# merlin-compare — {spec.label}",
        "",
        f"target=`{spec.target}` · metric=`{spec.metric}` · reps={spec.reps} · "
        f"commit=`{_git_commit(root)[:12]}`",
        "",
        "> v1 INGESTS already-measured host/board data (no new board run). "
        "Static CCA decode gives the RANKING of structural factors, not exact cycle fractions "
        "(no K1 perf counters).",
        "",
        "## 1. Empirical (measured table)",
        "",
        measured_table_md(spec, measurements),
        "",
        "## 2. Structural (per-config CCA)",
        "",
        cca_table_md(ccas),
        "",
        "## 3. Attribution (measured gap × structural divergence × routed action)",
        "",
        attribution_md(attrs),
        "",
        "### Gap-driver axes (union across trailing attributions)",
        "",
        ("- " + "\n- ".join(sorted(gap_axes))) if gap_axes else "_none (ours not trailing)_",
        "",
        "## 4. Figures",
        "",
    ]
    if figures:
        parts += [f"![{f}]({f})" for f in figures]
    else:
        parts.append("_figures skipped (matplotlib unavailable) or no measured cells_")
    parts += ["", "## 5. Manifest", "", "See `manifest.yaml` (spec + git + source provenance)."]
    p = out_dir / "compare.md"
    p.write_text("\n".join(parts) + "\n")
    return p


def write_manifest(out_dir: Path, *, spec, measurements, ccas, figures, root: Path) -> Path:
    import yaml
    out_dir = Path(out_dir)
    sources = sorted({m.source for m in measurements.values() if m.source})
    decode_src = "output/kernels/ceiling/kernel_breakdown_decode.json"
    cells = {}
    for (cn, wn), m in measurements.items():
        cells[f"{cn}|{wn}"] = {
            "status": m.status,
            "value_ns": m.value,
            "spread_pct": m.spread_pct,
            "cos": m.cos,
            "source": m.source,
        }
    cca_provenance = {
        name: (None if cca is None
               else {"decode_kernel": cca.provenance.get("decode_kernel"),
                     "decode_shape": cca.provenance.get("decode_shape"),
                     "source": cca.provenance.get("source")})
        for name, cca in ccas.items()
    }
    manifest = {
        "tool": "merlin-compare",
        "version": 1,
        "git_commit": _git_commit(root),
        "spec": spec.to_dict(),
        "empirical_sources": sources,
        "structural_source": decode_src,
        "measured_cells": cells,
        "cca_provenance": cca_provenance,
        "figures": figures,
        "deterministic": True,
    }
    p = out_dir / "manifest.yaml"
    p.write_text(yaml.safe_dump(manifest, sort_keys=True, default_flow_style=False))
    return p
