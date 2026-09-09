from __future__ import annotations

import copy
import gzip
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import qualify_shapes as q  # noqa: E402
import verify  # noqa: E402
from mlir_oot import codegen  # noqa: E402


def test_saved_qualification_is_complete_and_fail_closed() -> None:
    result = verify.verify()
    assert result["ok"]
    assert result["counts"]["unique_shapes_rtl_numeric_qualified"] == 4
    assert result["counts"]["unique_shapes_rtl_numeric_unqualified"] == 0
    assert result["counts"]["rtl_negative_controls_passed"] == 1
    assert result["counts"]["physical_partitions_qualified"] == 4
    assert result["counts"]["physical_partitions_unqualified"] == 387


def test_shape_numeric_receipt_cannot_promote_a_physical_occurrence() -> None:
    plan = q.load_json(q.PLAN_PATH)
    kernel_by_id, _ = q.validate_plan(plan)
    compile_receipts = {kernel_id: {"qualified": True} for kernel_id in kernel_by_id}
    numeric_receipts = {kernel_id: {"qualified": True} for kernel_id in kernel_by_id}
    mapping = q.build_partition_map(plan, compile_receipts, numeric_receipts, {})
    assert mapping["counts"]["shape_rtl_numeric_qualified_partitions"] == 391
    assert mapping["counts"]["physical_partitions_qualified"] == 0
    assert mapping["counts"]["physical_partitions_unqualified"] == 391


def test_compile_failure_invalidates_direct_physical_receipt() -> None:
    summary = q.load_json(HERE / "evidence/qualification.json")
    plan = q.load_json(q.PLAN_PATH)
    compile_receipts = {
        kernel_id: q.load_json(HERE / "evidence" / relative)
        for kernel_id, relative in summary["compile_receipts"].items()
    }
    failed = copy.deepcopy(compile_receipts)
    failed["matmul_1_32_960_bias"]["qualified"] = False
    mapping = q.build_partition_map(
        plan, failed, {}, summary["direct_physical_qualifications"]
    )
    state = next(row for row in mapping["partitions"] if row["partition_id"] == "atlas_p0098")
    assert not state["physical_partition_qualified"]
    assert state["physical_unqualified_reason"] == "fresh shape compilation failed"


def test_saved_batched_shape_is_assertion_clean_and_bit_exact() -> None:
    receipt = q.load_json(
        HERE / "evidence/numeric/matmul_batched_15_50_64_113/receipt.json"
    )
    stderr = (
        HERE / "evidence/numeric/matmul_batched_15_50_64_113/raw_gsim_stderr.txt"
    ).read_text(encoding="utf-8")
    assert receipt["qualified"]
    assert receipt["returncode"] == 0
    assert receipt["assertion_clean"]
    assert not stderr
    assert receipt["comparison"] == {
        "elements": 84750,
        "mismatches": 0,
        "max_abs_error": 0.0,
        "expected_sha256": receipt["raw_output_sha256"],
    }


def test_pre_fix_image_is_an_exact_assertion_negative_control() -> None:
    case = HERE / "evidence/negative/pre_fix_batched_15_50_64_113"
    receipt = q.load_json(case / "receipt.json")
    compile_receipt = q.load_json(case / "matmul_batched_15_50_64_113.json")
    plan = q.load_json(q.PLAN_PATH)
    kernels, _ = q.validate_plan(plan)
    planned = kernels["matmul_batched_15_50_64_113"]
    stderr = (case / "raw_gsim_stderr.txt").read_text(encoding="utf-8")
    assert receipt["control_passed"]
    assert receipt["returncode"] == -6
    assert "Assertion failed" in stderr
    assert receipt["expected_assertion"] in stderr
    assert compile_receipt["assembly_sha256"] == planned["assembly_sha256"]
    assert compile_receipt["instruction_words"] == planned["instruction_words"]

    with gzip.open(
        HERE / "evidence/numeric/matmul_batched_15_50_64_113/raw_gsim_spec.json.gz",
        "rt", encoding="utf-8",
    ) as source:
        positive_spec = json.load(source)
    with gzip.open(case / "raw_gsim_spec.json.gz", "rt", encoding="utf-8") as source:
        negative_spec = json.load(source)
    positive_spec.pop("words")
    negative_spec.pop("words")
    assert positive_spec == negative_spec


def test_dma_window_guard_matches_atlas_rtl_line_bound() -> None:
    last_valid_line_base_words = (codegen.VMEM_DMA_LINE_CAPACITY - 1) << 3
    codegen._check_dma_window(last_valid_line_base_words, codegen.DMA_BEAT_BYTES)
    try:
        codegen._check_dma_window(1024, 2)
    except ValueError as error:
        assert "positive 32-byte multiple" in str(error)
    else:
        raise AssertionError("sub-beat DMA was accepted")
    try:
        codegen._check_dma_window(
            codegen.VMEM_DMA_LINE_CAPACITY << 3, codegen.DMA_BEAT_BYTES
        )
    except ValueError as error:
        assert "exceeds hardware capacity" in str(error)
    else:
        raise AssertionError("out-of-VMEM DMA was accepted")


def test_fixed_batched_image_stays_inside_imem_and_differs_from_pre_fix() -> None:
    fixed = q.load_json(
        HERE / "evidence/receipts/compile/matmul_batched_15_50_64_113.json"
    )
    baseline = q.load_json(
        HERE / "evidence/negative/pre_fix_batched_15_50_64_113/"
        "matmul_batched_15_50_64_113.json"
    )
    assert fixed["instruction_words"] == 31130
    assert fixed["instruction_words"] <= q.IMEM_WORDS
    assert fixed["control_flow"]["backward_edges"] > baseline["control_flow"]["backward_edges"]
    assert fixed["assembly_sha256"] != baseline["assembly_sha256"]


def test_plan_cardinality_and_occurrences_are_exact() -> None:
    plan = q.load_json(q.PLAN_PATH)
    kernels, partitions = q.validate_plan(plan)
    assert len(kernels) == 28
    assert len(partitions) == 391
    assert sum(row["partition_occurrences"] for row in kernels.values()) == 391


if __name__ == "__main__":
    test_saved_qualification_is_complete_and_fail_closed()
    test_shape_numeric_receipt_cannot_promote_a_physical_occurrence()
    test_compile_failure_invalidates_direct_physical_receipt()
    test_saved_batched_shape_is_assertion_clean_and_bit_exact()
    test_pre_fix_image_is_an_exact_assertion_negative_control()
    test_dma_window_guard_matches_atlas_rtl_line_bound()
    test_fixed_batched_image_stays_inside_imem_and_differs_from_pre_fix()
    test_plan_cardinality_and_occurrences_are_exact()
    print("ok: 8 qualification/verifier tests")
