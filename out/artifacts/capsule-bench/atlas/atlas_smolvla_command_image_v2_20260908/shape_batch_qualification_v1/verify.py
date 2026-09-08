#!/usr/bin/env python3
"""Verify the saved batch qualification without compiling or running RTL."""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import qualify_shapes as q


HERE = Path(__file__).resolve().parent
EVIDENCE = HERE / "evidence"


def verify() -> dict:
    summary = q.load_json(EVIDENCE / "qualification.json")
    mapping = q.load_json(EVIDENCE / summary["partition_receipt_map"])
    plan = q.load_json(q.PLAN_PATH)
    kernel_by_id, partition_by_id = q.validate_plan(plan)
    errors: list[str] = []
    if summary.get("schema") != "atlas_smolvla_shape_batch_qualification_v1":
        errors.append("unexpected qualification schema")
    if q.sha256_file(q.PLAN_PATH) != summary.get("source_plan_sha256"):
        errors.append("source plan hash drift")
    compile_receipts = {}
    for kernel_id, relative in summary.get("compile_receipts", {}).items():
        path = EVIDENCE / relative
        receipt = q.load_json(path)
        compile_receipts[kernel_id] = receipt
        if receipt.get("kernel_id") != kernel_id or not receipt.get("qualified"):
            errors.append(f"compile receipt is not qualified: {kernel_id}")
        kernel = kernel_by_id.get(kernel_id, {})
        if receipt.get("source_interface_sha256") != kernel.get("interface_sha256"):
            errors.append(f"compile receipt interface mismatch: {kernel_id}")
        if receipt.get("assembly_sha256") != kernel.get("assembly_sha256"):
            errors.append(f"compile receipt assembly mismatch: {kernel_id}")
        if receipt.get("instruction_words") != kernel.get("instruction_words"):
            errors.append(f"compile receipt word-count mismatch: {kernel_id}")
    if set(compile_receipts) != set(kernel_by_id):
        errors.append("compile receipt set differs from 28-shape source library")

    numeric_receipts = {}
    expected_numeric = {
        "matmul_1_32_960_bias": True,
        "matmul_50_32_720_bias": True,
        "matmul_50_720_32_bias": True,
        "matmul_batched_15_50_64_113": False,
    }
    for kernel_id, relative in summary.get("rtl_numeric_receipts", {}).items():
        path = EVIDENCE / relative
        receipt = q.load_json(path)
        numeric_receipts[kernel_id] = receipt
        if kernel_id not in q.NUMERIC_CASES:
            errors.append(f"unexpected RTL numeric receipt: {kernel_id}")
        if receipt.get("qualified") is not expected_numeric.get(kernel_id):
            errors.append(f"RTL numeric outcome drift: {kernel_id}")
        if receipt.get("engine_kind") != "assertion-enabled elaborated RTL GSIM":
            errors.append(f"non-authoritative engine: {kernel_id}")
        stderr_path = HERE / "evidence/numeric" / kernel_id / "raw_gsim_stderr.txt"
        if not stderr_path.is_file() or q.sha256_file(stderr_path) != receipt.get("stderr_sha256"):
            errors.append(f"GSIM stderr hash mismatch: {kernel_id}")
        if expected_numeric.get(kernel_id):
            if receipt.get("stderr_sha256") != q.EMPTY_SHA256:
                errors.append(f"passing GSIM receipt has non-empty stderr: {kernel_id}")
        else:
            stderr = stderr_path.read_text(encoding="utf-8") if stderr_path.is_file() else ""
            if (
                receipt.get("returncode") != -6
                or receipt.get("assertion_clean") is not False
                or "DMA VMEM transfer range exceeds VMEM capacity" not in stderr
            ):
                errors.append(f"expected RTL VMEM assertion is absent: {kernel_id}")
        raw_output = EVIDENCE / receipt.get("raw_output", "")
        if not raw_output.is_file() or q.sha256_file(raw_output) != receipt.get("raw_output_sha256"):
            errors.append(f"numeric raw-output hash mismatch: {kernel_id}")
        compile_path = HERE / receipt.get("compile_receipt", "")
        if not compile_path.is_file() or q.sha256_file(compile_path) != receipt.get("compile_receipt_sha256"):
            errors.append(f"numeric-to-compile receipt link mismatch: {kernel_id}")
        spec_path = EVIDENCE / receipt.get("spec", "")
        if not spec_path.is_file():
            errors.append(f"numeric spec is missing: {kernel_id}")
        else:
            with gzip.open(spec_path, "rb") as source:
                if q.sha256_bytes(source.read()) != receipt.get("spec_sha256"):
                    errors.append(f"numeric spec hash mismatch: {kernel_id}")
        stdout_path = HERE / "evidence/numeric" / kernel_id / "raw_gsim_stdout.txt.gz"
        if not stdout_path.is_file():
            errors.append(f"GSIM stdout is missing: {kernel_id}")
        else:
            with gzip.open(stdout_path, "rb") as source:
                if q.sha256_bytes(source.read()) != receipt.get("stdout_sha256"):
                    errors.append(f"GSIM stdout hash mismatch: {kernel_id}")
    if set(numeric_receipts) != set(q.NUMERIC_CASES):
        errors.append("representative RTL numeric case set changed")

    rows = mapping.get("partitions", [])
    if len(rows) != len(partition_by_id) or {row.get("partition_id") for row in rows} != set(partition_by_id):
        errors.append("partition map is not a one-to-one mapping of all 391 partitions")
    direct = q.direct_capture_qualifications(partition_by_id, compile_receipts)
    if direct != summary.get("direct_physical_qualifications", {}):
        errors.append("direct capture-bound qualifications drifted")
    rebuilt = q.build_partition_map(plan, compile_receipts, numeric_receipts, direct)
    if mapping != rebuilt:
        errors.append("saved partition map differs from fail-closed reconstruction")
    counts = mapping.get("counts", {})
    expected_counts = {
        "capture_partitions_total": 391,
        "compile_qualified_partitions": 391,
        "shape_rtl_numeric_tested_partitions": 11,
        "shape_rtl_numeric_qualified_partitions": 3,
        "physical_partitions_qualified": 3,
        "physical_partitions_unqualified": 388,
    }
    if counts != expected_counts:
        errors.append(f"partition counts changed: {counts!r}")
    qualified = [row for row in rows if row.get("physical_partition_qualified")]
    if {row["partition_id"] for row in qualified} != {"atlas_p0098", "atlas_p0243", "atlas_p0244"}:
        errors.append("physical qualification set changed")
    for row in rows:
        if row.get("physical_partition_qualified") and not row.get("physical_qualification_source"):
            errors.append(f"physical qualification lacks direct source: {row.get('partition_id')}")
        if not row.get("physical_partition_qualified") and row.get("physical_qualification_source"):
            errors.append(f"unqualified partition has a qualification source: {row.get('partition_id')}")
    if summary.get("fail_closed") != {
        "shape_compile_does_not_imply_shape_numeric": True,
        "shape_numeric_does_not_imply_physical_partition_numeric": True,
        "whole_model_runnable": False,
        "whole_model_numerically_qualified": False,
    }:
        errors.append("fail-closed declaration changed")
    summary_counts = summary.get("counts", {})
    for key, wanted in {
        "unique_shapes_total": 28,
        "unique_shapes_compile_tested": 28,
        "unique_shapes_compile_qualified": 28,
        "unique_shapes_compile_unqualified": 0,
        "unique_shapes_rtl_numeric_tested": 4,
        "unique_shapes_rtl_numeric_qualified": 3,
        "unique_shapes_rtl_numeric_unqualified": 1,
        "physical_partitions_qualified": 3,
        "physical_partitions_unqualified": 388,
    }.items():
        if summary_counts.get(key) != wanted:
            errors.append(f"summary count drift: {key}={summary_counts.get(key)!r}")
    if summary.get("status") != "bounded_progress_with_detected_rtl_failure":
        errors.append("expected fail-closed RTL-rejection status is absent")
    if errors:
        raise AssertionError("\n".join(errors))
    return {"ok": True, "counts": summary["counts"], "status": summary["status"]}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2, sort_keys=True))
