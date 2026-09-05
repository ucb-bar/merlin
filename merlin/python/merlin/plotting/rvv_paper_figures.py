"""Generate claim-safe RVV paper figures from a frozen ``paper-results.yaml``.

The figures deliberately consume the structured paper report rather than raw benchmark logs.  That
keeps diagnostic stage timings out of end-to-end plots and preserves missing, failed, and unsupported
cells as visible marks instead of silently dropping them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml

from merlin.common.paths import artifacts_dir
from merlin.compare.paper import PaperStudySpec
from merlin.compare.paper_report import build_paper_report, load_issuance_notary


_BACKEND_ORDER = (
    "hand_v0_int8",
    "merlin_frozen",
    "merlin_xnnpack",
    "merlin_openblas",
    "executorch_xnnpack",
)
_BACKEND_LABEL = {
    "hand_v0_int8": "hand v0",
    "merlin_frozen": "Merlin",
    "merlin_xnnpack": "Merlin + XNNPACK kernels",
    "merlin_openblas": "Merlin + OpenBLAS kernels",
    "executorch_xnnpack": "ExecuTorch + XNNPACK",
}
_MODEL_LABEL = {
    "gemma2_2b": "Gemma 2 2B",
    "tinyllama_1_1b": "TinyLlama 1.1B",
    "smolvla": "SmolVLA",
    "resnet50_v1_5": "ResNet-50 v1.5",
    "lstmnetvit": "LSTMNetViT",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_report(path: Path, study_path: Path,
                 results_path: Path, *, trusted_issuance_fingerprints: Mapping[str, str] | None
                 ) -> tuple[dict[str, Any], PaperStudySpec, dict[str, Any]]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") != 2:
        raise ValueError("paper report must be a schema-version-2 mapping")
    primary = raw.get("primary_end_to_end")
    coverage = raw.get("coverage")
    if not isinstance(primary, Mapping) or not isinstance(primary.get("rows"), list):
        raise ValueError("paper report has no primary_end_to_end.rows")
    if not isinstance(coverage, Mapping):
        raise ValueError("paper report has no coverage mapping")
    claimed_sha = raw.get("study_sha256")
    if (not isinstance(claimed_sha, str) or len(claimed_sha) != 64
            or any(character not in "0123456789abcdef" for character in claimed_sha)):
        raise ValueError("paper report has no frozen study SHA-256")
    study = PaperStudySpec.from_yaml(study_path)
    if study.status != "frozen":
        raise ValueError("paper figures require the supplied study to have status=frozen")
    preflight = study.preflight()
    if not preflight.ready:
        raise ValueError(f"paper figures require a ready frozen study: {preflight.to_dict()}")
    if study.sha256() != claimed_sha:
        raise ValueError("paper report study_sha256 does not match the supplied frozen study")
    results = yaml.safe_load(results_path.read_text(encoding="utf-8"))
    if not isinstance(results, dict):
        raise ValueError("retained results document must be a mapping")
    report_seal = raw.get("results_content_seal")
    if not isinstance(report_seal, Mapping):
        raise ValueError("paper report has no results_content_seal")
    if report_seal != results.get("content_seal"):
        raise ValueError("paper report results_content_seal differs from retained results")
    derived = (build_paper_report(study, results)
               if trusted_issuance_fingerprints is None else
               build_paper_report(
                   study, results,
                   trusted_issuance_fingerprints=trusted_issuance_fingerprints))
    if raw != derived:
        raise ValueError("paper report differs from the report re-derived from retained results")
    return raw, study, results


def _cells(report: Mapping[str, Any]) -> dict[tuple[str, str, int], dict[str, dict[str, Any]]]:
    """Return ``(model, precision, cores) -> backend -> honest cell summary``."""
    cells: dict[tuple[str, str, int], dict[str, dict[str, Any]]] = {}
    for row in report["primary_end_to_end"]["rows"]:
        key = (str(row["model"]), str(row["precision"]), int(row["core_count"]))
        by_backend = cells.setdefault(key, {})
        by_backend[str(row["ours_backend"])] = dict(row["ours"])
        for comparison in row.get("comparisons", []):
            by_backend[str(comparison["comparator"])] = {
                "status": comparison.get("comparator_status", "missing"),
                "median_ns": comparison.get("comparator_median_ns"),
                "p05_ns": (comparison.get("comparator_observed_range_p05_p95_ns") or [None])[0],
                "p95_ns": (comparison.get("comparator_observed_range_p05_p95_ns") or [None, None])[1],
            }
    return cells


def _unsupported_backends(report: Mapping[str, Any], precision: str) -> set[str]:
    return {str(value.get("backend")) for value in report.get("unsupported_comparisons", [])
            if isinstance(value, Mapping) and value.get("precision") == precision}


def _comparison_gap(row: Mapping[str, Any], comparison: Mapping[str, Any] | None,
                    *, backend: str, unsupported: set[str]) -> str:
    """Keep each lifecycle state and its owner visible in relative plots."""
    if backend in unsupported:
        return "UNSUPPORTED"
    ours_status = str((row.get("ours") or {}).get("status", "missing")).upper()
    comparator_status = ("MISSING" if comparison is None else
                         str(comparison.get("comparator_status", "missing")).upper())
    gaps = []
    if ours_status != "PASS":
        gaps.append(f"MERLIN {ours_status}")
    if comparator_status != "PASS":
        gaps.append(f"COMPARATOR {comparator_status}")
    return " / ".join(gaps) or "NOT COMPARABLE"


def _claim_rows(report: Mapping[str, Any], precision: str, cores: int) -> list[dict[str, Any]]:
    return [dict(row) for row in report["primary_end_to_end"]["rows"]
            if row["precision"] == precision and int(row["core_count"]) == cores]


def _backend_color(name: str, style: Any) -> str:
    return {
        "hand_v0_int8": style.MAUVE,
        "merlin_frozen": style.NAVY,
        "merlin_xnnpack": style.SLATE,
        "merlin_openblas": style.SAGE,
        "executorch_xnnpack": style.BLUE,
    }.get(name, style.GOLD)


def _save(fig: Any, stem: Path) -> list[Path]:
    written = []
    for suffix, kwargs in ((".png", {"dpi": 180}), (".svg", {})):
        path = stem.with_suffix(suffix)
        fig.savefig(path, bbox_inches="tight", facecolor="#FDF7EF", **kwargs)
        written.append(path)
    return written


def _observed_range(ax: Any, xpos: float, cell: Mapping[str, Any], *, scale: float = 1.0,
                    color: str) -> None:
    """Draw the report's observed p05--p95 range around its median.

    These are descriptive observed ranges, not confidence intervals.  The report is independently
    re-derived before plotting, so this helper never estimates or fills in absent endpoints.
    """
    median, p05, p95 = (cell.get("median_ns"), cell.get("p05_ns"), cell.get("p95_ns"))
    if not all(isinstance(value, (int, float)) and not isinstance(value, bool)
               for value in (median, p05, p95)):
        return
    median, p05, p95 = float(median) * scale, float(p05) * scale, float(p95) * scale
    if not 0 < p05 <= median <= p95:
        raise ValueError("paper report contains an invalid observed p05/median/p95 range")
    ax.errorbar([xpos], [median], yerr=[[median - p05], [p95 - median]], fmt="none",
                ecolor=color, elinewidth=1.4, capsize=3.0, capthick=1.4, zorder=6)


def _head_to_head(report: Mapping[str, Any], *, precision: str, cores: int,
                  output: Path, plt: Any, np: Any, style: Any) -> list[Path]:
    cells = _cells(report)
    models = [str(row["model"]) for row in _claim_rows(report, precision, cores)]
    models = list(dict.fromkeys(models))
    backends = [backend for backend in _BACKEND_ORDER
                if (any(backend in cells.get((model, precision, cores), {}) for model in models)
                    or backend in _unsupported_backends(report, precision))]
    if not models or not backends:
        return []
    passing = [cell for model in models for backend in backends
               if (cell := cells.get((model, precision, cores), {}).get(backend, {})).get(
                   "status") == "pass" and cell.get("median_ns")]
    values = [float(cell["median_ns"]) / 1e6 for cell in passing]
    observed_lows = [float(cell.get("p05_ns") or cell["median_ns"]) / 1e6 for cell in passing]
    observed_highs = [float(cell.get("p95_ns") or cell["median_ns"]) / 1e6 for cell in passing]
    fig, ax = plt.subplots(figsize=(max(9.5, len(models) * 2.25), 5.8))
    style.style_ax(ax)
    x = np.arange(len(models), dtype=float)
    width = 0.78 / max(len(backends), 1)
    if values:
        low = min(observed_lows)
        marker_y = low / 1.55
        ax.set_yscale("log")
        ax.set_ylim(bottom=max(low / 2.2, 1e-9), top=max(observed_highs) * 1.7)
    else:
        marker_y = 1.0
        ax.set_ylim(0.5, 1.5)
    from matplotlib.patches import Patch
    legend = []
    for index, backend in enumerate(backends):
        offset = (index - (len(backends) - 1) / 2) * width
        for model_index, model in enumerate(models):
            cell = cells.get((model, precision, cores), {}).get(backend)
            xpos = x[model_index] + offset
            if cell and cell.get("status") == "pass" and cell.get("median_ns"):
                style.vbars(ax, [xpos], [float(cell["median_ns"]) / 1e6],
                            _backend_color(backend, style), width=width * 0.86)
                _observed_range(ax, xpos, cell, scale=1e-6, color=style.INK)
            else:
                status = ("UNSUPPORTED" if backend in _unsupported_backends(report, precision)
                          else "MISSING" if cell is None
                          else str(cell.get("status", "missing")).upper())
                ax.scatter([xpos], [marker_y], marker="x", s=42, linewidth=1.8,
                           color=_backend_color(backend, style), zorder=5)
                ax.text(xpos, marker_y, status, rotation=90, ha="center", va="top",
                        fontsize=6.5, color=style.INK)
        legend.append(Patch(facecolor=_backend_color(backend, style), edgecolor=style.INK,
                            label=_BACKEND_LABEL.get(backend, backend)))
    ax.set_xticks(x)
    ax.set_xticklabels([_MODEL_LABEL.get(model, model) for model in models])
    ax.set_ylabel("continuous-session latency (ms, log; p05–p95 whiskers) — lower is faster")
    style.title(ax, f"All backends — {precision.upper()}, {cores} core{'s' if cores != 1 else ''}")
    ax.legend(handles=legend, fontsize=8.3, ncol=min(3, len(legend)), loc="upper left")
    fig.tight_layout()
    written = _save(fig, output / f"latency_{precision}_{cores}c")
    plt.close(fig)
    return written


def _relative(report: Mapping[str, Any], *, precision: str, cores: int,
              output: Path, plt: Any, np: Any, style: Any) -> list[Path]:
    rows = _claim_rows(report, precision, cores)
    models = [str(row["model"]) for row in rows]
    unsupported = _unsupported_backends(report, precision)
    comparators = [backend for backend in _BACKEND_ORDER if backend != "merlin_frozen" and (
        any(comparison["comparator"] == backend for row in rows for comparison in row["comparisons"])
        or backend in unsupported)]
    if not rows or not comparators:
        return []
    fig, ax = plt.subplots(figsize=(max(9.5, len(models) * 2.25), 5.6))
    style.style_ax(ax)
    x = np.arange(len(models), dtype=float)
    width = 0.78 / max(len(comparators), 1)
    finite = []
    envelope_highs = []
    by_model = {str(row["model"]): {str(c["comparator"]): c for c in row["comparisons"]}
                for row in rows}
    from matplotlib.patches import Patch
    legend = []
    for index, backend in enumerate(comparators):
        offset = (index - (len(comparators) - 1) / 2) * width
        for model_index, model in enumerate(models):
            comparison = by_model[model].get(backend)
            ratio = None if comparison is None else comparison.get("ratio_ours_over_comparator")
            xpos = x[model_index] + offset
            if isinstance(ratio, (int, float)) and ratio > 0:
                speedup = 1.0 / float(ratio)
                finite.append(speedup)
                style.vbars(ax, [xpos], [speedup], _backend_color(backend, style),
                            width=width * 0.86)
                ours_range = comparison.get("ours_observed_range_p05_p95_ns") or []
                comparator_range = comparison.get("comparator_observed_range_p05_p95_ns") or []
                if (len(ours_range) == len(comparator_range) == 2
                        and all(isinstance(value, (int, float)) and not isinstance(value, bool)
                                and value > 0 for value in (*ours_range, *comparator_range))):
                    low = float(comparator_range[0]) / float(ours_range[1])
                    high = float(comparator_range[1]) / float(ours_range[0])
                    if not 0 < low <= speedup <= high:
                        raise ValueError("paper comparison has an invalid observed speedup range")
                    envelope_highs.append(high)
                    ax.errorbar([xpos], [speedup], yerr=[[speedup - low], [high - speedup]],
                                fmt="none", ecolor=style.INK, elinewidth=1.2, capsize=2.5,
                                capthick=1.2, zorder=6)
                ax.text(xpos, speedup, f"{speedup:.2f}×\n{comparison['label']}",
                        ha="center", va="bottom", fontsize=7.0, color=style.INK)
            else:
                ax.scatter([xpos], [0.05], marker="x", s=42, linewidth=1.8,
                           color=_backend_color(backend, style), zorder=5)
                ax.text(xpos, 0.05, _comparison_gap(
                            next(row for row in rows if row["model"] == model), comparison,
                            backend=backend, unsupported=unsupported),
                        rotation=90, ha="center", va="bottom", fontsize=6.3, color=style.INK)
        legend.append(Patch(facecolor=_backend_color(backend, style), edgecolor=style.INK,
                            label=f"vs {_BACKEND_LABEL.get(backend, backend)}"))
    ax.axhline(1.0, color=style.INK, ls="--", lw=1.2, alpha=0.7)
    ax.set_ylim(0, max([1.35, *(value * 1.3 for value in finite),
                        *(value * 1.15 for value in envelope_highs)]))
    ax.set_xticks(x)
    ax.set_xticklabels([_MODEL_LABEL.get(model, model) for model in models])
    ax.set_ylabel("comparator / Merlin latency (×; p05–p95 envelope) — higher is faster")
    style.title(ax, f"Head-to-head comparison — {precision.upper()}, {cores} core{'s' if cores != 1 else ''}")
    ax.legend(handles=legend, fontsize=8.3, ncol=min(2, len(legend)), loc="upper left")
    fig.tight_layout()
    written = _save(fig, output / f"head_to_head_{precision}_{cores}c")
    plt.close(fig)
    return written


def _causal_why_how(report: Mapping[str, Any], *, precision: str, cores: int,
                    output: Path, plt: Any, style: Any) -> list[Path]:
    """Render explicit why/how cards only for claim-eligible wins in the sealed report."""
    wins: list[dict[str, Any]] = []
    for row in _claim_rows(report, precision, cores):
        for comparison in row.get("comparisons", []):
            if comparison.get("e2e_win_claim") is not True:
                continue
            causal = comparison.get("causal_attribution")
            if not isinstance(causal, Mapping):
                raise ValueError("an end-to-end win has no sealed causal attribution")
            why, how, evidence = causal.get("why"), causal.get("how"), causal.get("evidence")
            if (not isinstance(why, str) or not why.strip() or not isinstance(how, str)
                    or not how.strip() or not isinstance(evidence, Mapping)
                    or not isinstance(evidence.get("binding_sha256"), str)):
                raise ValueError("an end-to-end win has malformed sealed why/how evidence")
            wins.append({
                "model": str(row["model"]), "comparator": str(comparison["comparator"]),
                "why": why.strip(), "how": how.strip(),
                "binding": str(evidence["binding_sha256"]),
            })
    if not wins:
        return []

    from matplotlib.patches import FancyBboxPatch
    written: list[Path] = []
    per_page = 5
    for page_start in range(0, len(wins), per_page):
        page = wins[page_start:page_start + per_page]
        fig, ax = plt.subplots(figsize=(11.2, 1.68 + 1.55 * len(page)))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, len(page))
        ax.axis("off")
        style.title(
            ax, f"Why the claimed wins occur — {precision.upper()}, "
            f"{cores} core{'s' if cores != 1 else ''}")
        for local_index, win in enumerate(page):
            claim_index = page_start + local_index + 1
            y = len(page) - local_index - 0.93
            card = FancyBboxPatch(
                (0.015, y), 0.97, 0.82, boxstyle="round,pad=0.012,rounding_size=0.018",
                facecolor="#F5EEE4", edgecolor=style.GOLD, linewidth=1.2,
                path_effects=[style.SHADOW], zorder=1)
            ax.add_patch(card)
            heading = (
                f"W{claim_index}  {_MODEL_LABEL.get(win['model'], win['model'])} vs "
                f"{_BACKEND_LABEL.get(win['comparator'], win['comparator'])}")
            ax.text(0.04, y + 0.66, heading, fontsize=10.0, fontweight="bold",
                    color=style.NAVY, va="center", zorder=2)
            ax.text(0.04, y + 0.42, "WHY  " + textwrap.fill(win["why"], width=112),
                    fontsize=8.6, color=style.INK, va="center", zorder=2)
            ax.text(0.04, y + 0.19, "HOW   " + textwrap.fill(win["how"], width=112),
                    fontsize=8.6, color=style.INK, va="center", zorder=2)
            ax.text(0.965, y + 0.08, f"sealed binding {win['binding'][:12]}…",
                    fontsize=6.7, color=style.GOLD, ha="right", va="bottom", zorder=2)
        fig.tight_layout()
        page_number = page_start // per_page + 1
        written += _save(
            fig, output / f"causal_why_how_{precision}_{cores}c_p{page_number:02d}")
        plt.close(fig)
    return written


def _core_scaling(report: Mapping[str, Any], *, precision: str, output: Path,
                  plt: Any, np: Any, style: Any) -> list[Path]:
    cells = _cells(report)
    models = list(dict.fromkeys(str(row["model"]) for row in report["primary_end_to_end"]["rows"]
                                if row["precision"] == precision))
    if not models:
        return []
    unsupported = _unsupported_backends(report, precision)
    backends = [backend for backend in _BACKEND_ORDER
                if (any(backend in cells.get((model, precision, cores), {})
                        for model in models for cores in (1, 8)) or backend in unsupported)]
    fig, ax = plt.subplots(figsize=(max(9.5, len(models) * 2.25), 5.4))
    style.style_ax(ax)
    from matplotlib.patches import Patch
    x = np.arange(len(models), dtype=float)
    width = 0.78 / max(len(backends), 1)
    finite: list[float] = []
    envelope_highs: list[float] = []
    legend = []
    for backend_index, backend in enumerate(backends):
        offset = (backend_index - (len(backends) - 1) / 2) * width
        color = _backend_color(backend, style)
        for model_index, model in enumerate(models):
            xpos = x[model_index] + offset
            one = cells.get((model, precision, 1), {}).get(backend, {})
            eight = cells.get((model, precision, 8), {}).get(backend, {})
            if (one.get("status") == eight.get("status") == "pass" and one.get("median_ns")
                    and eight.get("median_ns")):
                speedup = float(one["median_ns"]) / float(eight["median_ns"])
                finite.append(speedup)
                style.vbars(ax, [xpos], [speedup], color, width=width * 0.86)
                one_p05, one_p95 = one.get("p05_ns"), one.get("p95_ns")
                eight_p05, eight_p95 = eight.get("p05_ns"), eight.get("p95_ns")
                if all(isinstance(value, (int, float)) and not isinstance(value, bool)
                       and value > 0 for value in (one_p05, one_p95, eight_p05, eight_p95)):
                    low = float(one_p05) / float(eight_p95)
                    high = float(one_p95) / float(eight_p05)
                    if not 0 < low <= speedup <= high:
                        raise ValueError("paper report has an invalid core-scaling observed range")
                    envelope_highs.append(high)
                    ax.errorbar([xpos], [speedup], yerr=[[speedup - low], [high - speedup]],
                                fmt="none", ecolor=style.INK, elinewidth=1.2, capsize=2.5,
                                capthick=1.2, zorder=6)
                ax.text(xpos, speedup, f"{speedup:.2f}×", ha="center", va="bottom",
                        fontsize=7.0, fontweight="bold", color=style.GOLD)
            else:
                status = ("UNSUPPORTED" if backend in unsupported else
                          f"1C {str(one.get('status', 'missing')).upper()} / "
                          f"8C {str(eight.get('status', 'missing')).upper()}")
                ax.scatter([xpos], [0.05], marker="x", s=42, linewidth=1.8,
                           color=color, zorder=5)
                ax.text(xpos, 0.05, status, rotation=90, va="bottom", ha="center",
                        fontsize=5.9, color=color)
        legend.append(Patch(facecolor=color, edgecolor=style.INK,
                            label=_BACKEND_LABEL.get(backend, backend)))
    ax.axhline(1.0, color=style.INK, ls="--", lw=1.2, alpha=0.7)
    ax.set_ylim(0, max([1.25, *(value * 1.3 for value in finite),
                        *(value * 1.15 for value in envelope_highs)]))
    ax.set_xticks(x)
    ax.set_xticklabels([_MODEL_LABEL.get(model, model) for model in models])
    ax.set_ylabel("1-core latency / 8-core latency (×; p05–p95 envelope)")
    style.title(ax, f"CPU host scaling by backend — {precision.upper()}")
    ax.legend(handles=legend, fontsize=8.0, ncol=min(3, len(legend)), loc="upper left")
    fig.tight_layout()
    written = _save(fig, output / f"core_scaling_{precision}")
    plt.close(fig)
    return written


def generate_paper_figures(report_path: str | Path, study_path: str | Path, *,
                           results_path: str | Path,
                           output_dir: str | Path | None = None,
                           trusted_issuance_fingerprints: Mapping[str, str] | None = None) -> Path:
    """Render timestamped PNG/SVG figures plus a provenance manifest; never overwrite a run."""
    report_path = Path(report_path).resolve()
    study_path = Path(study_path).resolve()
    results_path = Path(results_path).resolve()
    report, _study, results = _load_report(
        report_path, study_path, results_path,
        trusted_issuance_fingerprints=trusted_issuance_fingerprints)
    report_sha = _sha256(report_path)
    if output_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = artifacts_dir() / "paper-figures" / "k1" / f"{stamp}_{report_sha[:8]}"
    else:
        output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from merlin.plotting import merlin_plotstyle as style
    style.use_merlin_style()

    written: list[Path] = []
    configurations = sorted({(str(row["precision"]), int(row["core_count"]))
                             for row in report["primary_end_to_end"]["rows"]})
    for precision, cores in configurations:
        written += _head_to_head(report, precision=precision, cores=cores, output=output,
                                 plt=plt, np=np, style=style)
        written += _relative(report, precision=precision, cores=cores, output=output,
                             plt=plt, np=np, style=style)
        written += _causal_why_how(report, precision=precision, cores=cores, output=output,
                                   plt=plt, style=style)
    for precision in sorted({precision for precision, _ in configurations}):
        written += _core_scaling(report, precision=precision, output=output,
                                 plt=plt, np=np, style=style)

    manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": {"paper_report": str(report_path), "paper_report_sha256": report_sha,
                  "study": str(study_path), "study_file_sha256": _sha256(study_path),
                  "study_sha256": report["study_sha256"],
                  "results": str(results_path), "results_file_sha256": _sha256(results_path),
                  "results_content_seal": results["content_seal"]},
        "claim_scope": "primary_end_to_end_only",
        "coverage": dict(report["coverage"]),
        "kernel_swap_coverage": dict(report.get("kernel_swap_coverage", {})),
        "unsupported_comparisons": list(report.get("unsupported_comparisons", [])),
        "figures": [{"path": path.name, "sha256": _sha256(path)} for path in written],
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True,
                        help="frozen paper-results.yaml generated by merlin.compare.paper_report")
    parser.add_argument("--study", type=Path, required=True,
                        help="the exact ready status=frozen paper study bound by the report")
    parser.add_argument("--results", type=Path, required=True,
                        help="the exact content-sealed results.yaml used to derive the report")
    parser.add_argument("--issuance-notary", type=Path, required=True,
                        help="externally retained run_id -> issuance fingerprint manifest")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    study = PaperStudySpec.from_yaml(args.study)
    fingerprints = load_issuance_notary(
        args.issuance_notary, expected_study_sha256=study.sha256())
    output = generate_paper_figures(
        args.report, args.study, results_path=args.results, output_dir=args.output_dir,
        trusted_issuance_fingerprints=fingerprints)
    print(json.dumps({"output_dir": str(output), "manifest": str(output / "manifest.json")},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
