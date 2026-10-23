#!/usr/bin/env python3
"""Build the auditable exact Core ATen capsule-cover artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from merlin.common.paths import artifacts_dir, merlin_dir
from merlin.targetgen import aten_coverage, core_aten_cases
from merlin.targetgen import core_aten_cover as cover
from merlin.targetgen.core_aten_samples import capture_gap_fillers, gap_reason


def _observed_gap_reasons(capsule_roots: list[Path], uncovered: tuple[str, ...]) -> dict[str, str]:
    observed: dict[str, str] = {}
    for root in capsule_roots:
        for path in sorted(root.rglob("capture.json")):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                continue
            overload = record.get("overload")
            if overload not in uncovered:
                continue
            status = record.get("status")
            if status == "captured_lost_exact_provenance":
                captured = ", ".join(record.get("captured_overloads") or ()) or "none"
                observed[overload] = f"capture succeeded but exact provenance was absent (observed: {captured})"
                continue
            if status != "capture_failed":
                continue
            reason = str(record.get("reason") or "capture failed without a recorded reason")
            if "Opaque ops:" in reason:
                detail = reason.rsplit("Opaque ops:", 1)[1].splitlines()[0].strip()
                if detail != "(none reported)":
                    observed[overload] = f"model2MLIR emitted unsupported opaque operation(s): {detail}"
                else:
                    observed[overload] = "model2MLIR conversion did not produce a clean standard-dialect program"
            elif "normalization failed" in reason:
                observed[overload] = "model2MLIR normalization failed on the captured dynamic-shape MLIR"
            elif "non-clean program" in reason:
                observed[overload] = "model2MLIR conversion did not produce a clean standard-dialect program"
            else:
                observed[overload] = reason.splitlines()[0][:400]
    return {op: observed.get(op, gap_reason(op)) for op in uncovered}


def _documents(capsule_roots: list[Path]) -> dict[str, bytes]:
    denominator = cover.denominator_document(aten_coverage.core_opset(refresh=True))
    case_corpus = core_aten_cases.corpus_document(
        denominator["overloads"], pytorch_version=denominator["pytorch_version"]
    )
    case_corpus["denominator_sha256"] = denominator["overload_sha256"]
    matrix = cover.capsule_operator_matrix(capsule_roots, denominator["overloads"])
    matrix["denominator_count"] = denominator["overload_count"]
    matrix["denominator_sha256"] = denominator["overload_sha256"]
    candidate_map = {name: matrix["capsules"][name] for name in matrix["eligible_candidates"]}
    result = cover.exact_minimum_cover(denominator["overloads"], candidate_map, allow_partial=True)
    minimum = result.to_dict()
    gap_reasons = _observed_gap_reasons(capsule_roots, result.uncovered_overloads)
    minimum["uncovered_reasons"] = gap_reasons
    return {
        "core_aten_ops.json": cover.json_bytes(denominator),
        "core_aten_test_cases.json": cover.json_bytes(case_corpus),
        "capsule_operator_matrix.json": cover.json_bytes(matrix),
        "minimum_capsule_cover.json": cover.json_bytes(minimum),
        "core_aten_capsule_coverage.md": cover.markdown_summary(
            denominator, matrix, result, case_corpus=case_corpus, gap_reasons=gap_reasons
        ).encode(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--capsule-root",
        action="append",
        type=Path,
        default=[],
        help="capsule tree to inventory (repeatable; defaults to the shared contract tree)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="artifact directory (defaults beneath the configured generated-artifact root)",
    )
    parser.add_argument("--require-full", action="store_true", help="return nonzero if any overload is uncovered")
    parser.add_argument(
        "--capture-gaps",
        action="store_true",
        help="capture the bundled target-agnostic gap fillers before solving",
    )
    parser.add_argument("--m2m-dir", type=Path, default=None, help="model2MLIR checkout used by --capture-gaps")
    parser.add_argument("--m2m-python", type=Path, default=None, help="PyTorch interpreter used by --capture-gaps")
    args = parser.parse_args(argv)

    roots = args.capsule_root or [merlin_dir() / "contract" / "capsules"]
    destination = args.output_dir or artifacts_dir() / "verification" / "core-aten"
    if args.capture_gaps:
        generated = destination / "generated-candidates"
        capture_gap_fillers(generated, m2m_dir=args.m2m_dir, python=args.m2m_python)
        roots = [*roots, generated]
    first = _documents(roots)
    second = _documents(roots)
    if first != second:
        print("FAIL: Core ATen coverage artifacts changed across identical reruns", file=sys.stderr)
        return 2
    destination.mkdir(parents=True, exist_ok=True)
    for name, payload in sorted(first.items()):
        (destination / name).write_bytes(payload)

    result = json.loads(first["minimum_capsule_cover.json"])
    print(
        f"{result['coverable_count']}/{result['denominator_count']} Core ATen overloads covered by "
        f"{result['selected_count']}/{result['candidate_count']} capsules"
    )
    print(f"artifacts: {destination}")
    if args.require_full and not result["selected_union_matches_denominator"]:
        print(f"FAIL: {len(result['uncovered_overloads'])} overload(s) remain uncovered", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
