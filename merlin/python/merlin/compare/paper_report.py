"""Honest result analysis for a frozen version-2 paper study.

The primary table in this module contains only complete continuous sessions.  Per-stage timings are
kept in a separate diagnostic table and are never used to make an end-to-end claim.  In particular,
a low median is not, by itself, called a win: the frozen reporting policy, non-overlapping observed
ranges, passing lifecycle gates, and a comparator-specific causal explanation must all agree.
"""
from __future__ import annotations

import copy
import hashlib
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from .paper_attribution import causal_record as _evidence_causal_record

from .paper import PaperStudySpec, validate_paper_result


def _is_sha256(value: object) -> bool:
    text = str(value)
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _closed(value: object, fields: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    extra, missing = sorted(set(value) - fields), sorted(fields - set(value))
    if extra or missing:
        raise ValueError(f"{label} is closed; unrecognized={extra} missing={missing}")
    return value


def _retained(path_value: object, digest: object, *, root: Path | None = None,
              label: str) -> Path:
    path = Path(str(path_value))
    if root is not None:
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"{label} must be a contained relative path")
        path = (root / path).resolve()
        try:
            path.relative_to(root.resolve())
        except ValueError as error:
            raise ValueError(f"{label} escapes its receipt directory") from error
    else:
        path = path.resolve()
    if (not path.is_file() or not _is_sha256(digest)
            or hashlib.sha256(path.read_bytes()).hexdigest() != digest):
        raise ValueError(f"{label} digest differs from retained file")
    return path


def _measurement_roots(spec: "PaperStudySpec",
                       results: list[dict[str, Any]], *,
                       trusted_issuance_fingerprints: Mapping[str, str] | None = None
                       ) -> list[dict[str, Any]]:
    """Replay controller-owned receipts; arbitrary raw/AET documents have no authority."""
    from .paper_measurement_controller import verify_receipt

    if trusted_issuance_fingerprints is not None:
        expected_run_ids = {str(result.get("run_id")) for result in results}
        if set(trusted_issuance_fingerprints) != expected_run_ids:
            raise ValueError(
                "external issuance notary must contain exactly every result run_id")
    roots: list[dict[str, Any]] = []
    for index, result in enumerate(results):
        ref = _closed(result.get("measurement_receipt"),
                      {"path", "sha256", "aet_run_id", "command_sha256"},
                      f"results[{index}].measurement receipt")
        receipt_path = _retained(ref["path"], ref["sha256"],
                                 label=f"results[{index}].measurement receipt")
        if ref["aet_run_id"] != result.get("run_id"):
            raise ValueError(f"results[{index}] measurement receipt identity differs")
        root = verify_receipt(
            receipt_path, expected_result=result, expected_study_sha256=spec.sha256(),
            trusted_issuance_fingerprint=(
                trusted_issuance_fingerprints.get(str(result.get("run_id")))
                if trusted_issuance_fingerprints is not None else None))
        if (root["receipt_sha256"] != ref["sha256"]
                or root["command_sha256"] != ref["command_sha256"]):
            raise ValueError(f"results[{index}] controller receipt reference differs")
        roots.append({"index": index, **root})
    return roots


def _results_seal(study_sha256: str, results: list[dict[str, Any]],
                  measurement_roots: list[dict[str, Any]]) -> dict[str, Any]:
    measurements = [{
        "index": index,
        "cell": _cell_key(*_result_key(result)),
        "sha256": _canonical_sha256(result),
    } for index, result in enumerate(results)]
    body = {
        "schema_version": 3,
        "algorithm": "sha256",
        "study_sha256": study_sha256,
        "results_sha256": _canonical_sha256(results),
        "measurement_roots_sha256": _canonical_sha256(measurement_roots),
        "measurement_roots": measurement_roots,
        "measurements": measurements,
    }
    return {**body, "seal_sha256": _canonical_sha256(body)}


def seal_results_document(
        spec: "PaperStudySpec", results: list[dict[str, Any]], *,
        trusted_issuance_fingerprints: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Seal results to reconciled raw measurements and finalized native AET lifecycles."""
    retained = copy.deepcopy(results)
    study_sha256 = spec.sha256()
    roots = (_measurement_roots(spec, retained)
             if trusted_issuance_fingerprints is None else
             _measurement_roots(
                 spec, retained,
                 trusted_issuance_fingerprints=trusted_issuance_fingerprints))
    return {
        "schema_version": 3,
        "study_sha256": study_sha256,
        "results": retained,
        "measurement_roots": copy.deepcopy(roots),
        "content_seal": _results_seal(study_sha256, retained, roots),
    }


def _verify_results_seal(
        spec: "PaperStudySpec", document: Mapping[str, Any], *,
        trusted_issuance_fingerprints: Mapping[str, str] | None = None) -> None:
    _closed(document, {"schema_version", "study_sha256", "results", "measurement_roots",
                       "content_seal"}, "results document")
    if document.get("schema_version") != 3:
        raise ValueError("results.yaml schema_version must be 3")
    results = document.get("results")
    if not isinstance(results, list) or not all(isinstance(result, dict) for result in results):
        raise ValueError("results.yaml results must be a list of mappings")
    study_sha256 = spec.sha256()
    if document.get("study_sha256") != study_sha256:
        raise ValueError("results content seal study digest differs from the frozen study")
    roots = (_measurement_roots(spec, results)
             if trusted_issuance_fingerprints is None else
             _measurement_roots(
                 spec, results,
                 trusted_issuance_fingerprints=trusted_issuance_fingerprints))
    if document.get("measurement_roots") != roots:
        raise ValueError("results measurement roots differ from retained run receipts")
    _closed(document.get("content_seal"), {
        "schema_version", "algorithm", "study_sha256", "results_sha256",
        "measurement_roots_sha256", "measurement_roots", "measurements", "seal_sha256",
    }, "results content seal")
    expected = _results_seal(study_sha256, results, roots)
    if document.get("content_seal") != expected:
        raise ValueError("results content seal does not match retained measurement bytes")


@dataclass(frozen=True)
class PerformanceClaims:
    parity_low: float
    parity_high: float
    win_max: float
    require_nonoverlap: bool
    require_attribution: bool
    language: str

    @staticmethod
    def parse(reporting: Mapping[str, Any]) -> "PerformanceClaims":
        raw = reporting.get("performance_claims")
        if not isinstance(raw, Mapping):
            raise ValueError("reporting.performance_claims must be an explicit mapping")
        band = raw.get("parity_median_ratio_band")
        if (not isinstance(band, (list, tuple)) or len(band) != 2
                or any(isinstance(value, bool) or not isinstance(value, (int, float))
                       for value in band)):
            raise ValueError(
                "reporting.performance_claims.parity_median_ratio_band must contain two numbers")
        low, high = float(band[0]), float(band[1])
        win_max = raw.get("win_median_ratio_max")
        if isinstance(win_max, bool) or not isinstance(win_max, (int, float)):
            raise ValueError(
                "reporting.performance_claims.win_median_ratio_max must be numeric")
        win_max = float(win_max)
        if not (0.0 < win_max <= low <= 1.0 <= high):
            raise ValueError(
                "performance claim ratios must satisfy 0 < win_max <= parity_low <= 1 <= parity_high")
        if raw.get("win_requires_nonoverlapping_observed_ranges") is not True:
            raise ValueError("paper wins must require non-overlapping observed ranges")
        if raw.get("win_requires_causal_attribution") is not True:
            raise ValueError("paper wins must require causal attribution")
        language = str(raw.get("language", ""))
        if language != "descriptive_ratio_not_statistical_significance":
            raise ValueError(
                "paper comparison language must be descriptive_ratio_not_statistical_significance")
        return PerformanceClaims(low, high, win_max, True, True, language)

    def to_dict(self) -> dict[str, Any]:
        return {
            "parity_median_ratio_band": [self.parity_low, self.parity_high],
            "win_median_ratio_max": self.win_max,
            "win_requires_nonoverlapping_observed_ranges": self.require_nonoverlap,
            "win_requires_causal_attribution": self.require_attribution,
            "language": self.language,
        }


def _result_key(result: Mapping[str, Any]) -> tuple[str, str, str, int]:
    return (str(result.get("model", "")), str(result.get("backend", "")),
            str(result.get("precision", "")), int(result.get("core_count") or 0))


def _cell_key(model: str, backend: str, precision: str, cores: int) -> str:
    return f"{model}/{backend}/{precision}/{cores}c"


def _quantile(samples: Iterable[int], fraction: float) -> int:
    """Use the same nearest-observation convention as the v2 result validator."""
    ordered = sorted(int(value) for value in samples)
    if not ordered:
        raise ValueError("cannot compute a quantile of an empty sample set")
    return ordered[min(len(ordered) - 1,
                       max(0, int(round(fraction * (len(ordered) - 1)))))]


def _median(samples: Iterable[int]) -> int | float:
    value = statistics.median(list(samples))
    return int(value) if float(value).is_integer() else float(value)


def _is_end_to_end(result: Mapping[str, Any]) -> bool:
    timing = result.get("timing", {}) or {}
    session = result.get("session", {}) or {}
    return (timing.get("scope") == "end_to_end"
            and list(timing.get("timed_stages", ()) or ())
            == list(session.get("stages", ()) or ())
            and timing.get("sample_unit") == "complete_session")


def _cell_summary(result: Mapping[str, Any] | None) -> dict[str, Any]:
    if result is None:
        return {"status": "missing", "reason": "no end-to-end result in results.yaml",
                "median_ns": None, "p05_ns": None, "p95_ns": None}
    lifecycle = result.get("lifecycle", {}) or {}
    status = str(lifecycle.get("status", "error"))
    samples = list((result.get("timing", {}) or {}).get("samples", ()) or ())
    # Failed and not-run cells remain visible, but their timings are not comparison evidence.
    usable = status == "pass" and bool(samples)
    return {
        "status": status,
        "reason": lifecycle.get("reason"),
        "median_ns": _median(samples) if usable else None,
        "p05_ns": _quantile(samples, 0.05) if usable else None,
        "p95_ns": _quantile(samples, 0.95) if usable else None,
    }


def _attribution_records(result: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw = result.get("causal_attribution")
    if isinstance(raw, list):
        return [record for record in raw if isinstance(record, Mapping)]
    if not isinstance(raw, Mapping):
        return []
    records = raw.get("records")
    if isinstance(records, list):
        return [record for record in records if isinstance(record, Mapping)]
    # Accept either one record or a comparator-keyed mapping.  Both serialize cleanly in YAML while
    # retaining an unambiguous comparator for each claimed win.
    if "comparator" in raw:
        return [raw]
    out: list[Mapping[str, Any]] = []
    for comparator, value in raw.items():
        if isinstance(value, Mapping):
            out.append({"comparator": str(comparator), **dict(value)})
    return out


def _causal_record(spec: PaperStudySpec, result: Mapping[str, Any] | None,
                   comparator_result: Mapping[str, Any] | None, comparator: str) -> dict[str, Any] | None:
    """Return only an evidence-derived explanation, never free-form result annotation.

    ``causal_attribution`` in a result is retained as an audit trail, but is deliberately not
    trusted as the source of a paper claim.  Re-deriving from the frozen manifest catches a later
    result edit and guarantees that timing alone cannot manufacture why/how text.
    """
    if result is None or comparator_result is None:
        return None
    record = _evidence_causal_record(spec, result, comparator_result)
    if (record.get("comparator") != comparator or record.get("status") != "available"):
        return None
    why, how = str(record.get("why", "")).strip(), str(record.get("how", "")).strip()
    evidence = record.get("evidence")
    if not why or not how or not isinstance(evidence, Mapping):
        return None
    if not all(_is_sha256(evidence.get(key)) for key in
               ("binding_sha256", "ablation_sha256", "structural_sha256")):
        return None
    return dict(record)


def _comparison(spec: PaperStudySpec, ours_result: Mapping[str, Any] | None,
                comparator_result: Mapping[str, Any] | None, *, comparator: str,
                comparator_kind: str, policy: PerformanceClaims) -> dict[str, Any]:
    ours = _cell_summary(ours_result)
    other = _cell_summary(comparator_result)
    row: dict[str, Any] = {
        "comparator": comparator,
        "comparator_kind": comparator_kind,
        "ours_status": ours["status"],
        "comparator_status": other["status"],
        "ours_median_ns": ours["median_ns"],
        "comparator_median_ns": other["median_ns"],
        "ours_observed_range_p05_p95_ns": [ours["p05_ns"], ours["p95_ns"]],
        "comparator_observed_range_p05_p95_ns": [other["p05_ns"], other["p95_ns"]],
        "ratio_ours_over_comparator": None,
        "observed_ranges_nonoverlap": None,
        "label": "not_comparable",
        "e2e_win_claim": False,
        "claim_notes": [],
    }
    if ours["status"] != "pass" or other["status"] != "pass":
        row["claim_notes"] = ["both end-to-end cells must pass before comparing performance"]
        return row

    assert ours["median_ns"] and other["median_ns"]
    ratio = float(ours["median_ns"]) / float(other["median_ns"])
    row["ratio_ours_over_comparator"] = ratio
    if ratio > policy.parity_high:
        row["label"] = "loss"
        row["claim_notes"] = ["median ratio is above the frozen parity band"]
        return row
    if ratio > policy.win_max:
        if ratio >= policy.parity_low:
            row["label"] = "parity"
            row["claim_notes"] = ["median ratio is inside the frozen parity band"]
            return row
        # This is possible when a methodology intentionally leaves a gap between its parity and win
        # thresholds.  Do not silently round it into either claim.
        row["label"] = "advantage_not_claimable"
        row["claim_notes"] = ["median advantage is outside parity but does not reach win threshold"]
        return row

    nonoverlap = (ours["p95_ns"] is not None and other["p05_ns"] is not None
                  and ours["p95_ns"] < other["p05_ns"])
    row["observed_ranges_nonoverlap"] = nonoverlap
    causal = _causal_record(spec, ours_result, comparator_result, comparator)
    blockers = []
    if not nonoverlap:
        blockers.append("observed ranges overlap (requires ours p95 < comparator p05)")
    if causal is None:
        blockers.append("missing comparator-specific causal attribution with nonempty why/how")
    if blockers:
        # If the two configured bands share their boundary, a threshold-equal result that misses a
        # win gate can still make the weaker parity claim.  Ratios below the parity band cannot.
        row["label"] = ("parity" if ratio >= policy.parity_low
                        else "advantage_not_claimable")
        row["claim_notes"] = blockers
        return row
    row["label"] = "win"
    row["e2e_win_claim"] = True
    row["causal_attribution"] = causal
    row["claim_notes"] = [
        "median ratio reaches the win threshold",
        "ours p95 is below comparator p05",
        "winning end-to-end cell passes and provides causal why/how",
    ]
    return row


def _expected_keys(spec: PaperStudySpec) -> set[tuple[str, str, str, int]]:
    return {(cell.model.name, cell.backend.name, cell.precision, cell.core_count)
            for cell in spec.matrix()}


def build_paper_report(spec: PaperStudySpec,
                       results_document: Mapping[str, Any], *,
                       trusted_issuance_fingerprints: Mapping[str, str] | None = None
                       ) -> dict[str, Any]:
    """Build a deterministic, machine-readable report from a frozen study and ``results.yaml``."""
    if spec.status != "frozen":
        raise ValueError("paper result reporting requires a frozen study")
    if int(results_document.get("schema_version", 0)) != 3:
        raise ValueError("results.yaml schema_version must be 3")
    _verify_results_seal(
        spec, results_document,
        trusted_issuance_fingerprints=trusted_issuance_fingerprints)
    policy = PerformanceClaims.parse(spec.reporting)
    raw_results = results_document.get("results")
    if not isinstance(raw_results, list):
        raise ValueError("results.yaml results must be a list")
    for index, result in enumerate(raw_results):
        if not isinstance(result, dict):
            raise ValueError(f"results[{index}] must be a mapping")
        validate_paper_result(result)

    primary: dict[tuple[str, str, str, int], Mapping[str, Any]] = {}
    diagnostic_results: list[Mapping[str, Any]] = []
    for result in raw_results:
        if _is_end_to_end(result):
            key = _result_key(result)
            if key in primary:
                raise ValueError(f"duplicate end-to-end result for {_cell_key(*key)}")
            primary[key] = result
        else:
            diagnostic_results.append(result)

    expected = _expected_keys(spec)
    unexpected = sorted(_cell_key(*key) for key in set(primary) - expected)
    compiler_backends = [backend for backend in spec.backends if backend.kind == "compiler"]
    if len(compiler_backends) != 1:
        raise ValueError("paper report requires exactly one compiler backend as 'ours'")
    ours_backend = compiler_backends[0]
    comparators = [backend for backend in spec.backends if backend.kind != "compiler"]

    rows: list[dict[str, Any]] = []
    for model in spec.models:
        for precision in (spec.primary_precision, spec.control_precision):
            if precision not in model.precisions or precision not in ours_backend.precisions:
                continue
            for cores in spec.core_counts:
                ours_key = (model.name, ours_backend.name, precision, cores)
                ours_result = primary.get(ours_key)
                applicable = [backend for backend in comparators
                              if precision in backend.precisions
                              and (model.name, backend.name, precision, cores) in expected]
                rows.append({
                    "model": model.name,
                    "precision": precision,
                    "core_count": cores,
                    "ours_backend": ours_backend.name,
                    "ours": _cell_summary(ours_result),
                    "comparisons": [
                        _comparison(spec, ours_result, primary.get(
                            (model.name, backend.name, precision, cores)),
                                    comparator=backend.name, comparator_kind=backend.kind,
                                    policy=policy)
                        for backend in applicable
                    ],
                })

    stage_rows: list[dict[str, Any]] = []
    for result in raw_results:
        timing = result.get("timing", {}) or {}
        lifecycle = result.get("lifecycle", {}) or {}
        for stage, samples in (timing.get("stage_samples", {}) or {}).items():
            stage_rows.append({
                "model": result["model"], "backend": result["backend"],
                "precision": result["precision"], "core_count": result["core_count"],
                "stage": str(stage), "median_ns": _median(samples),
                "source_scope": timing.get("scope"), "status": lifecycle.get("status"),
                "claim_eligible": False,
            })
    for result in diagnostic_results:
        timing = result.get("timing", {}) or {}
        stage_rows.append({
            "model": result["model"], "backend": result["backend"],
            "precision": result["precision"], "core_count": result["core_count"],
            "stage": "+".join(timing.get("timed_stages", ()) or ()),
            "median_ns": timing.get("median"), "source_scope": timing.get("scope"),
            "status": (result.get("lifecycle", {}) or {}).get("status"),
            "claim_eligible": False,
        })

    missing = sorted(_cell_key(*key) for key in expected if key not in primary)
    not_run = sorted(_cell_key(*key) for key, result in primary.items()
                     if key in expected
                     and (result.get("lifecycle", {}) or {}).get("status") == "not_run")
    failed = sorted(_cell_key(*key) for key, result in primary.items()
                    if key in expected
                    and (result.get("lifecycle", {}) or {}).get("status") in {"fail", "error"})
    kernel_swap_names = {
        backend.name for backend in spec.backends if backend.kind == "kernel_swap"}
    kernel_swap_rows = []
    for key in sorted(set(primary) & expected):
        result = primary[key]
        if result["backend"] not in kernel_swap_names:
            continue
        execution = result.get("execution", {}) or {}
        routed = execution.get("n_routed")
        eligible = execution.get("n_eligible")
        candidates = execution.get("n_candidates")
        integer_counts = all(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0
            for value in (routed, eligible, candidates))
        complete = bool(
            integer_counts and eligible > 0 and routed == eligible and candidates >= eligible)
        kernel_swap_rows.append({
            "model": result["model"], "backend": result["backend"],
            "precision": result["precision"], "core_count": result["core_count"],
            "candidates": candidates, "eligible": eligible, "routed": routed,
            "eligible_coverage": ((float(routed) / float(eligible))
                                  if integer_counts and eligible > 0 else None),
            "complete_eligible_coverage": complete,
            "status": (result.get("lifecycle", {}) or {}).get("status"),
        })
    return {
        "schema_version": 2,
        "study_label": spec.label,
        "study_sha256": spec.sha256(),
        # Downstream figures and paper tables retain the exact verified root of the measurements
        # from which this report was computed.  This is copied only after ``_verify_results_seal``.
        "results_content_seal": copy.deepcopy(results_document["content_seal"]),
        "performance_claims": policy.to_dict(),
        # Unsupported cells are methodology results, not absent data.  Carry the frozen declaration
        # into every downstream report/figure so a renderer cannot silently make them disappear.
        "unsupported_comparisons": [dict(value) for value in
                                    (spec.reporting.get("unsupported_comparisons", []) or [])],
        "primary_end_to_end": {
            "scope": "complete continuous session; lower latency is better",
            "rows": rows,
        },
        "stage_diagnostics": {
            "scope": "diagnostic only; never used for an end-to-end win claim",
            "rows": stage_rows,
        },
        "kernel_swap_coverage": {
            "scope": ("candidate and eligible counts are defined by each frozen backend's exact "
                      "canonical linalg.matmul classifier; passing cells require routed=eligible>0"),
            "rows": kernel_swap_rows,
        },
        "coverage": {"expected_cells": len(expected), "observed_end_to_end_cells":
                     len(set(primary) & expected), "missing_cells": missing,
                     "not_run_cells": not_run, "failed_cells": failed,
                     "unexpected_cells": unexpected},
    }


def _fmt_latency(value: int | float | None) -> str:
    return "—" if value is None else f"{float(value) / 1e6:.3f} ms"


def render_markdown(report: Mapping[str, Any]) -> str:
    """Render the structured report without promoting diagnostic timings into the primary table."""
    lines = [f"# Paper results — {report['study_label']}", "",
             "> Ratios are descriptive, not claims of statistical significance. A `win` additionally "
             "requires passing cells, Merlin p95 below comparator p05, and causal why/how.", "",
             "## Primary: end-to-end continuous sessions", "",
             "| model | precision | cores | Merlin | comparator | comparator latency | ours / comparator | label |",
             "|---|---:|---:|---:|---|---:|---:|---|"]
    for row in report["primary_end_to_end"]["rows"]:
        comparisons = row["comparisons"] or [{"comparator": "—", "comparator_median_ns": None,
                                                "ratio_ours_over_comparator": None,
                                                "label": "not_applicable"}]
        for comparison in comparisons:
            ratio = comparison["ratio_ours_over_comparator"]
            lines.append(
                f"| {row['model']} | {row['precision']} | {row['core_count']} | "
                f"{_fmt_latency(row['ours']['median_ns'])} ({row['ours']['status']}) | "
                f"{comparison['comparator']} | "
                f"{_fmt_latency(comparison['comparator_median_ns'])} "
                f"({comparison.get('comparator_status', 'n/a')}) | "
                f"{'—' if ratio is None else f'{ratio:.3f}x'} | {comparison['label']} |")

    coverage = report["coverage"]
    lines += ["", "### Coverage", "",
              f"Expected {coverage['expected_cells']} cells; observed "
              f"{coverage['observed_end_to_end_cells']} end-to-end cells."]
    for label, key in (("Missing", "missing_cells"), ("Not run", "not_run_cells"),
                       ("Failed/error", "failed_cells"), ("Unexpected", "unexpected_cells")):
        values = coverage[key]
        lines.append(f"- {label}: {', '.join(f'`{value}`' for value in values) if values else 'none'}")

    unsupported = report.get("unsupported_comparisons", []) or []
    lines += ["", "### Unsupported comparisons", ""]
    if unsupported:
        for cell in unsupported:
            lines.append(
                f"- `{cell.get('backend', 'unknown')}/{cell.get('precision', 'unknown')}`: "
                f"{cell.get('status', 'unsupported')} — {cell.get('reason', 'no reason recorded')}")
    else:
        lines.append("- none")

    lines += ["", "### Kernel-swap routing coverage", "",
              "| model | backend | precision | cores | candidates | eligible | routed | eligible coverage | status |",
              "|---|---|---:|---:|---:|---:|---:|---:|---|"]
    routing_rows = (report.get("kernel_swap_coverage", {}) or {}).get("rows", []) or []
    if routing_rows:
        for row in routing_rows:
            ratio = row.get("eligible_coverage")
            lines.append(
                f"| {row['model']} | {row['backend']} | {row['precision']} | "
                f"{row['core_count']} | {row.get('candidates', '—')} | "
                f"{row.get('eligible', '—')} | {row.get('routed', '—')} | "
                f"{'—' if ratio is None else f'{100.0 * float(ratio):.1f}%'} | "
                f"{row.get('status', 'unknown')} |")
    else:
        lines.append("| — | — | — | — | — | — | — | — | no kernel-swap cells |")

    lines += ["", "## Stage diagnostics (not end-to-end claims)", "",
              "| model | backend | precision | cores | stage/subset | median | status |",
              "|---|---|---:|---:|---|---:|---|"]
    stage_rows = report["stage_diagnostics"]["rows"]
    if stage_rows:
        for row in stage_rows:
            lines.append(f"| {row['model']} | {row['backend']} | {row['precision']} | "
                         f"{row['core_count']} | {row['stage']} | "
                         f"{_fmt_latency(row['median_ns'])} | {row['status']} |")
    else:
        lines.append("| — | — | — | — | no stage diagnostics recorded | — | — |")
    return "\n".join(lines) + "\n"


def generate_paper_report(study_path: str | Path, results_path: str | Path, *,
                          output_dir: str | Path | None = None,
                          trusted_issuance_fingerprints: Mapping[str, str] | None = None
                          ) -> tuple[Path, Path]:
    """Load frozen inputs and write ``paper-results.yaml`` plus ``paper-results.md``."""
    study_path, results_path = Path(study_path), Path(results_path)
    spec = PaperStudySpec.from_yaml(study_path)
    document = yaml.safe_load(results_path.read_text(encoding="utf-8"))
    if not isinstance(document, Mapping):
        raise ValueError("results.yaml must contain a mapping")
    report = build_paper_report(
        spec, document, trusted_issuance_fingerprints=trusted_issuance_fingerprints)
    out = Path(output_dir) if output_dir is not None else results_path.parent
    out.mkdir(parents=True, exist_ok=True)
    yaml_path, md_path = out / "paper-results.yaml", out / "paper-results.md"
    yaml_path.write_text(yaml.safe_dump(report, sort_keys=False), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return yaml_path, md_path


def load_issuance_notary(path: str | Path, *, expected_study_sha256: str) -> dict[str, str]:
    """Load an explicitly supplied external issuance-fingerprint manifest."""
    value = _closed(yaml.safe_load(Path(path).read_text(encoding="utf-8")), {
        "schema_version", "kind", "study_sha256", "fingerprints",
    }, "external issuance notary")
    fingerprints = value["fingerprints"]
    if (value["schema_version"] != 1
            or value["kind"] != "paper_external_issuance_notary_v1"
            or value["study_sha256"] != expected_study_sha256
            or not isinstance(fingerprints, Mapping) or not fingerprints
            or any(not isinstance(run_id, str) or not run_id or not _is_sha256(fingerprint)
                   for run_id, fingerprint in fingerprints.items())):
        raise ValueError("external issuance notary identity/fingerprints are invalid")
    return {str(run_id): str(fingerprint) for run_id, fingerprint in fingerprints.items()}


def main(argv: list[str] | None = None) -> int:
    """Re-derive a production report; fresh-process authority always needs a notary."""
    import argparse
    parser = argparse.ArgumentParser(prog="python -m merlin.compare.paper_report")
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--issuance-notary", type=Path, required=True,
                        help="externally retained run_id -> issuance fingerprint manifest")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    spec = PaperStudySpec.from_yaml(args.study)
    fingerprints = load_issuance_notary(
        args.issuance_notary, expected_study_sha256=spec.sha256())
    generated = generate_paper_report(
        args.study, args.results, output_dir=args.output_dir,
        trusted_issuance_fingerprints=fingerprints)
    print("\n".join(str(path) for path in generated))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "PerformanceClaims", "build_paper_report", "generate_paper_report",
    "load_issuance_notary", "main", "render_markdown",
]
