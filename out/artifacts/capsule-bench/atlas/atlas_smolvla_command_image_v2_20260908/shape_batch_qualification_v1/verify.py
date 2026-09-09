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
        expected_change = kernel_id in q.EXPECTED_ASSEMBLY_CHANGES
        if receipt.get("hardware_bounds_change_expected") is not expected_change:
            errors.append(f"compile receipt change scope mismatch: {kernel_id}")
        if receipt.get("baseline_assembly_match") is expected_change:
            errors.append(f"compile receipt assembly change mismatch: {kernel_id}")
        if expected_change and receipt.get("assembly_sha256") == kernel.get("assembly_sha256"):
            errors.append(f"hardware-bounds fix did not change image: {kernel_id}")
        words = receipt.get("instruction_words")
        if not isinstance(words, int) or not 0 < words <= q.IMEM_WORDS:
            errors.append(f"compile receipt violates IMEM bound: {kernel_id}")
        compiler = Path(str(receipt.get("compiler", "")))
        if compiler != q.ATLAS_OPT or not compiler.is_file():
            errors.append(f"compile receipt does not use isolated fixed backend: {kernel_id}")
        elif q.sha256_file(compiler) != receipt.get("compiler_sha256"):
            errors.append(f"compiler hash mismatch: {kernel_id}")
    if set(compile_receipts) != set(kernel_by_id):
        errors.append("compile receipt set differs from 28-shape source library")

    numeric_receipts = {}
    expected_numeric = set(q.NUMERIC_CASES)
    for kernel_id, relative in summary.get("rtl_numeric_receipts", {}).items():
        path = EVIDENCE / relative
        receipt = q.load_json(path)
        numeric_receipts[kernel_id] = receipt
        if kernel_id not in q.NUMERIC_CASES:
            errors.append(f"unexpected RTL numeric receipt: {kernel_id}")
        if receipt.get("qualified") is not True:
            errors.append(f"RTL numeric outcome drift: {kernel_id}")
        if receipt.get("engine_kind") != "assertion-enabled elaborated RTL GSIM":
            errors.append(f"non-authoritative engine: {kernel_id}")
        engine = Path(str(receipt.get("engine_binary", "")))
        if not engine.is_file() or q.sha256_file(engine) != receipt.get("engine_sha256"):
            errors.append(f"RTL engine hash mismatch: {kernel_id}")
        stderr_path = HERE / "evidence/numeric" / kernel_id / "raw_gsim_stderr.txt"
        if not stderr_path.is_file() or q.sha256_file(stderr_path) != receipt.get("stderr_sha256"):
            errors.append(f"GSIM stderr hash mismatch: {kernel_id}")
        if (
            receipt.get("returncode") != 0
            or receipt.get("halted") is not True
            or receipt.get("assertion_clean") is not True
            or receipt.get("stderr_sha256") != q.EMPTY_SHA256
        ):
            errors.append(f"passing GSIM receipt is not assertion-clean: {kernel_id}")
        comparison = receipt.get("comparison") or {}
        if comparison.get("mismatches") != 0 or comparison.get("max_abs_error") != 0.0:
            errors.append(f"passing GSIM receipt is not bit-exact: {kernel_id}")
        raw_output = EVIDENCE / receipt.get("raw_output", "")
        if not raw_output.is_file() or q.sha256_file(raw_output) != receipt.get("raw_output_sha256"):
            errors.append(f"numeric raw-output hash mismatch: {kernel_id}")
        elif receipt.get("raw_output_sha256") != comparison.get("expected_sha256"):
            errors.append(f"numeric output is not the exact expected BF16 image: {kernel_id}")
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
    if set(numeric_receipts) != expected_numeric:
        errors.append("representative RTL numeric case set changed")

    negative_summary = summary.get("rtl_negative_control", {})
    negative_path = EVIDENCE / str(negative_summary.get("receipt", ""))
    if not negative_path.is_file():
        errors.append("pre-fix negative-control receipt is missing")
        negative = {}
    else:
        negative = q.load_json(negative_path)
    negative_dir = negative_path.parent
    negative_compile_path = HERE / str(negative.get("compile_receipt", ""))
    if not negative_compile_path.is_file():
        errors.append("pre-fix compile receipt is missing")
    else:
        negative_compile = q.load_json(negative_compile_path)
        kernel = kernel_by_id["matmul_batched_15_50_64_113"]
        if (
            not negative_compile.get("qualified")
            or negative_compile.get("assembly_sha256") != kernel.get("assembly_sha256")
            or negative_compile.get("instruction_words") != kernel.get("instruction_words")
            or not negative_compile.get("baseline_assembly_match")
            or q.sha256_file(negative_compile_path) != negative.get("compile_receipt_sha256")
        ):
            errors.append("pre-fix compile receipt no longer reproduces exact baseline")
    negative_stderr_path = negative_dir / "raw_gsim_stderr.txt"
    negative_stderr = (
        negative_stderr_path.read_text(encoding="utf-8")
        if negative_stderr_path.is_file() else ""
    )
    if (
        negative.get("schema") != "atlas_shape_rtl_negative_control_v1"
        or negative.get("control_passed") is not True
        or negative_summary.get("control_passed") is not True
        or negative.get("returncode") != -6
        or "Assertion failed" not in negative_stderr
        or negative.get("expected_assertion", "") not in negative_stderr
        or not negative_stderr_path.is_file()
        or q.sha256_file(negative_stderr_path) != negative.get("stderr_sha256")
    ):
        errors.append("pre-fix negative control did not preserve the expected assertion")
    for name, compressed, hash_key in (
        ("raw_gsim_spec.json.gz", True, "spec_sha256"),
        ("raw_gsim_stdout.txt.gz", True, "stdout_sha256"),
    ):
        path = negative_dir / name
        if not path.is_file():
            errors.append(f"pre-fix negative-control artifact missing: {name}")
            continue
        raw = gzip.open(path, "rb").read() if compressed else path.read_bytes()
        if q.sha256_bytes(raw) != negative.get(hash_key):
            errors.append(f"pre-fix negative-control hash mismatch: {name}")
    engine = Path(str(negative.get("engine_binary", "")))
    if not engine.is_file() or q.sha256_file(engine) != negative.get("engine_sha256"):
        errors.append("pre-fix negative-control engine hash mismatch")

    positive_spec_path = EVIDENCE / numeric_receipts[
        "matmul_batched_15_50_64_113"
    ].get("spec", "")
    negative_spec_path = negative_dir / "raw_gsim_spec.json.gz"
    if positive_spec_path.is_file() and negative_spec_path.is_file():
        with gzip.open(positive_spec_path, "rt", encoding="utf-8") as source:
            positive_spec = json.load(source)
        with gzip.open(negative_spec_path, "rt", encoding="utf-8") as source:
            negative_spec = json.load(source)
        positive_spec.pop("words", None)
        negative_spec.pop("words", None)
        if positive_spec != negative_spec:
            errors.append("positive and pre-fix control workloads/stimuli differ")

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
        "shape_rtl_numeric_qualified_partitions": 11,
        "physical_partitions_qualified": 4,
        "physical_partitions_unqualified": 387,
    }
    if counts != expected_counts:
        errors.append(f"partition counts changed: {counts!r}")
    qualified = [row for row in rows if row.get("physical_partition_qualified")]
    if {row["partition_id"] for row in qualified} != {
        "atlas_p0098", "atlas_p0102", "atlas_p0243", "atlas_p0244"
    }:
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
    isolated = summary.get("isolated_backend", {})
    baseline_digest, baseline_files = q.tree_digest(q.ISOLATED_BASELINE)
    fixed_digest, fixed_files = q.tree_digest(q.ISOLATED_FIXED)
    changed_files = sorted(
        name for name in set(baseline_files) | set(fixed_files)
        if baseline_files.get(name) != fixed_files.get(name)
    )
    if baseline_digest != isolated.get("baseline_tree_sha256"):
        errors.append("isolated baseline backend tree hash drift")
    if fixed_digest != isolated.get("fixed_tree_sha256"):
        errors.append("isolated fixed backend tree hash drift")
    if changed_files != ["codegen.py"] or isolated.get("changed_files") != changed_files:
        errors.append(f"isolated backend change scope drift: {changed_files!r}")
    if set(q.EXPECTED_ASSEMBLY_CHANGES) != set(kernel_by_id):
        errors.append("shared compact-loop fix does not explicitly cover all 28 shapes")
    if isolated.get("hardware_bounds") != {
        "dma_beat_bytes": 32,
        "vmem_dma_line_capacity": 49152,
        "source": "AtlasCore24 DMA launch assertions: size>>5 and final line < 0xc000",
    }:
        errors.append("hardware-derived DMA bounds changed")
    summary_counts = summary.get("counts", {})
    for key, wanted in {
        "unique_shapes_total": 28,
        "unique_shapes_compile_tested": 28,
        "unique_shapes_compile_qualified": 28,
        "unique_shapes_compile_unqualified": 0,
        "unique_shapes_rtl_numeric_tested": 4,
        "unique_shapes_rtl_numeric_qualified": 4,
        "unique_shapes_rtl_numeric_unqualified": 0,
        "rtl_negative_controls_tested": 1,
        "rtl_negative_controls_passed": 1,
        "shape_rtl_numeric_tested_partitions": 11,
        "shape_rtl_numeric_qualified_partitions": 11,
        "physical_partitions_qualified": 4,
        "physical_partitions_unqualified": 387,
    }.items():
        if summary_counts.get(key) != wanted:
            errors.append(f"summary count drift: {key}={summary_counts.get(key)!r}")
    if summary.get("status") != "bounded_progress_fail_closed":
        errors.append("expected fail-closed bounded-progress status is absent")
    if errors:
        raise AssertionError("\n".join(errors))
    return {"ok": True, "counts": summary["counts"], "status": summary["status"]}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2, sort_keys=True))
