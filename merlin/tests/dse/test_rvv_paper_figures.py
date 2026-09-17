from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from merlin.plotting import rvv_paper_figures as figures
from merlin.plotting.rvv_paper_figures import generate_paper_figures


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("matplotlib") is None, reason="matplotlib optional extra is absent")


def _report() -> dict:
    rows = []
    for model, latency in (("gemma2_2b", 100_000_000), ("resnet50_v1_5", 40_000_000)):
        for precision in ("w8a8", "fp32"):
            for cores in (1, 8):
                ours = latency // cores
                ours_gap = model == "gemma2_2b" and precision == "w8a8" and cores == 8
                comparators = (["hand_v0_int8"] if precision == "w8a8" else
                               ["merlin_xnnpack", "merlin_openblas", "executorch_xnnpack"])
                comparisons = []
                for index, comparator in enumerate(comparators, 2):
                    # Exercise a visible missing external-runtime cell instead of silently eliding it.
                    missing = comparator == "executorch_xnnpack" and model == "gemma2_2b"
                    failed = comparator == "merlin_openblas" and model == "resnet50_v1_5"
                    comparisons.append({
                        "comparator": comparator,
                        "comparator_status": "not_run" if missing else "fail" if failed else "pass",
                        "comparator_median_ns": None if (missing or failed) else ours * index,
                        "ours_observed_range_p05_p95_ns": (
                            [None, None] if ours_gap else [ours - 1, ours + 1]),
                        "comparator_observed_range_p05_p95_ns": (
                            [None, None] if (missing or failed)
                            else [ours * index - 1, ours * index + 1]),
                        "ratio_ours_over_comparator": None if (missing or failed or ours_gap) else 1 / index,
                        "label": "not_comparable" if (missing or failed or ours_gap) else "win",
                    })
                rows.append({
                    "model": model, "precision": precision, "core_count": cores,
                    "ours_backend": "merlin_frozen",
                    "ours": {"status": "not_run" if ours_gap else "pass",
                             "median_ns": None if ours_gap else ours,
                             "p05_ns": None if ours_gap else ours - 1,
                             "p95_ns": None if ours_gap else ours + 1},
                    "comparisons": comparisons,
                })
    return {
        "schema_version": 2, "study_label": "test", "study_sha256": "a" * 64,
        "results_content_seal": {"schema_version": 1, "seal_sha256": "b" * 64},
        "primary_end_to_end": {
            "scope": "complete continuous session; lower latency is better", "rows": rows},
        "unsupported_comparisons": [{"backend": "executorch_xnnpack", "precision": "w8a8",
                                     "status": "not_implemented", "reason": "no W8A8 path"}],
        "kernel_swap_coverage": {"scope": "exact classifier", "rows": [{
            "model": "gemma2_2b", "backend": "merlin_xnnpack", "precision": "fp32",
            "core_count": 1, "candidates": 7, "eligible": 5, "routed": 5,
            "eligible_coverage": 1.0, "complete_eligible_coverage": True, "status": "pass",
        }]},
        "coverage": {"expected_cells": 16, "observed_end_to_end_cells": 14,
                     "missing_cells": [], "not_run_cells": ["gemma2_2b/executorch_xnnpack/fp32"],
                     "failed_cells": [], "unexpected_cells": []},
    }


def _results(tmp_path: Path) -> Path:
    path = tmp_path / "results.yaml"
    path.write_text(yaml.safe_dump({
        "schema_version": 2, "content_seal": _report()["results_content_seal"]}),
        encoding="utf-8")
    return path


def _study(monkeypatch, tmp_path: Path, *, sha: str = "a" * 64, status: str = "frozen",
           ready: bool = True) -> Path:
    path = tmp_path / "study.frozen.yaml"
    path.write_text("exact frozen study bytes\n", encoding="utf-8")
    check = SimpleNamespace(ready=ready, to_dict=lambda: {
        "ready": ready, "errors": [] if ready else ["unresolved"], "blockers": [], "warnings": []})
    spec = SimpleNamespace(status=status, sha256=lambda: sha, preflight=lambda: check)
    monkeypatch.setattr(figures.PaperStudySpec, "from_yaml", lambda _path: spec)
    return path


def test_paper_figures_are_timestamp_run_compatible_and_provenance_bound(
        tmp_path: Path, monkeypatch) -> None:
    report_path = tmp_path / "paper-results.yaml"
    report_path.write_text(yaml.safe_dump(_report(), sort_keys=False), encoding="utf-8")
    study_path = _study(monkeypatch, tmp_path)
    results_path = _results(tmp_path)
    monkeypatch.setattr(figures, "build_paper_report", lambda *_args: _report())
    output = generate_paper_figures(
        report_path, study_path, results_path=results_path,
        output_dir=tmp_path / "figures")

    pngs, svgs = sorted(output.glob("*.png")), sorted(output.glob("*.svg"))
    assert len(pngs) == len(svgs) == 10
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["claim_scope"] == "primary_end_to_end_only"
    assert manifest["input"]["study_sha256"] == "a" * 64
    assert manifest["input"]["study_file_sha256"]
    assert manifest["coverage"]["not_run_cells"]
    assert manifest["kernel_swap_coverage"]["rows"][0]["routed"] == 5
    assert manifest["unsupported_comparisons"][0]["backend"] == "executorch_xnnpack"
    assert len(manifest["figures"]) == 20

    svg = "\n".join(path.read_text(encoding="utf-8") for path in svgs)
    assert "UNSUPPORTED" in svg
    assert "COMPARATOR NOT_RUN" in svg
    assert "COMPARATOR FAIL" in svg
    assert "1C PASS / 8C NOT_RUN" in svg
    assert "2.00×" in svg  # report ratio 0.5 is rendered in the correct comparator/ours direction
    assert "ExecuTorch + XNNPACK" in svg
    assert "CPU host scaling by backend" in svg
    assert "p05–p95 whiskers" in svg
    assert "#fdf7ef" in svg.lower()
    assert "#0f3759" in svg.lower()  # Merlin's house-series navy

    with pytest.raises(FileExistsError):
        generate_paper_figures(
            report_path, study_path, results_path=results_path, output_dir=output)


def test_paper_figures_reject_unbound_report_before_creating_output(tmp_path: Path) -> None:
    report = _report()
    del report["study_sha256"]
    report_path = tmp_path / "paper-results.yaml"
    report_path.write_text(yaml.safe_dump(report), encoding="utf-8")
    output = tmp_path / "figures"
    with pytest.raises(ValueError, match="frozen study SHA-256"):
        generate_paper_figures(
            report_path, tmp_path / "absent-study.yaml", results_path=tmp_path / "results.yaml",
            output_dir=output)
    assert not output.exists()


def test_paper_figures_reject_report_without_retained_results_seal(
        tmp_path: Path, monkeypatch) -> None:
    report_path = tmp_path / "paper-results.yaml"
    unsealed = _report()
    del unsealed["results_content_seal"]
    report_path.write_text(yaml.safe_dump(unsealed, sort_keys=False), encoding="utf-8")
    results_path = tmp_path / "results.yaml"
    results_path.write_text(yaml.safe_dump({"schema_version": 2}), encoding="utf-8")
    study_path = _study(monkeypatch, tmp_path)

    with pytest.raises(ValueError, match="results_content_seal"):
        generate_paper_figures(
            report_path, study_path, results_path=results_path,
            output_dir=tmp_path / "figures")


def test_paper_figures_reject_rows_changed_after_report_derivation(
        tmp_path: Path, monkeypatch) -> None:
    seal = {"schema_version": 1, "seal_sha256": "b" * 64}
    expected = _report()
    expected["results_content_seal"] = seal
    forged = copy.deepcopy(expected)
    forged["primary_end_to_end"]["rows"][0]["comparisons"][0][
        "ratio_ours_over_comparator"] = 0.000001
    report_path = tmp_path / "paper-results.yaml"
    report_path.write_text(yaml.safe_dump(forged, sort_keys=False), encoding="utf-8")
    results_path = tmp_path / "results.yaml"
    results_path.write_text(yaml.safe_dump({"schema_version": 2, "content_seal": seal}),
                            encoding="utf-8")
    study_path = _study(monkeypatch, tmp_path)
    monkeypatch.setattr(figures, "build_paper_report", lambda *_args: expected)

    with pytest.raises(ValueError, match="re-derived"):
        generate_paper_figures(
            report_path, study_path, results_path=results_path,
            output_dir=tmp_path / "figures")


@pytest.mark.parametrize("status, ready, sha, message", [
    ("draft", True, "a" * 64, "status=frozen"),
    ("frozen", False, "a" * 64, "ready frozen study"),
    ("frozen", True, "b" * 64, "does not match"),
])
def test_paper_figures_reject_nonfrozen_unready_or_mismatched_study(
        tmp_path: Path, monkeypatch, status: str, ready: bool, sha: str, message: str) -> None:
    report_path = tmp_path / "paper-results.yaml"
    report_path.write_text(yaml.safe_dump(_report(), sort_keys=False), encoding="utf-8")
    study_path = _study(monkeypatch, tmp_path, sha=sha, status=status, ready=ready)
    with pytest.raises(ValueError, match=message):
        generate_paper_figures(
            report_path, study_path, results_path=tmp_path / "results.yaml",
            output_dir=tmp_path / "figures")


def test_default_output_uses_timestamped_artifact_tree(tmp_path: Path, monkeypatch) -> None:
    report_path = tmp_path / "paper-results.yaml"
    report_path.write_text(yaml.safe_dump(_report(), sort_keys=False), encoding="utf-8")
    study_path = _study(monkeypatch, tmp_path)
    results_path = _results(tmp_path)
    monkeypatch.setattr(figures, "build_paper_report", lambda *_args: _report())
    artifact_root = tmp_path / "out" / "artifacts"
    monkeypatch.setattr(figures, "artifacts_dir", lambda: artifact_root)
    output = generate_paper_figures(report_path, study_path, results_path=results_path)
    assert output.parent == artifact_root / "paper-figures" / "k1"
    assert output.name.endswith("_" + figures._sha256(report_path)[:8])


def test_claimed_win_gets_explicit_sealed_why_how_figure(
        tmp_path: Path, monkeypatch) -> None:
    report = _report()
    row = next(row for row in report["primary_end_to_end"]["rows"]
               if row["model"] == "gemma2_2b" and row["precision"] == "fp32"
               and row["core_count"] == 1)
    comparison = next(value for value in row["comparisons"]
                      if value["comparator"] == "merlin_xnnpack")
    comparison.update({
        "e2e_win_claim": True,
        "causal_attribution": {
            "why": "whole-model visibility removes a materialization boundary",
            "how": "the frozen fusion pass retains producer and consumer in one region",
            "evidence": {"binding_sha256": "c" * 64},
        },
    })
    report_path = tmp_path / "paper-results.yaml"
    report_path.write_text(yaml.safe_dump(report, sort_keys=False), encoding="utf-8")
    study_path = _study(monkeypatch, tmp_path)
    results_path = _results(tmp_path)
    monkeypatch.setattr(figures, "build_paper_report", lambda *_args: report)

    output = generate_paper_figures(
        report_path, study_path, results_path=results_path,
        output_dir=tmp_path / "figures-with-causal")

    causal = output / "causal_why_how_fp32_1c_p01.svg"
    assert causal.is_file()
    svg = causal.read_text(encoding="utf-8")
    assert "WHY" in svg
    assert "whole-model visibility removes a materialization boundary" in svg
    assert "HOW" in svg
    assert "the frozen fusion pass retains producer and consumer in one region" in svg
    assert "sealed binding cccccccccccc" in svg
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert any(row["path"] == causal.name for row in manifest["figures"])
