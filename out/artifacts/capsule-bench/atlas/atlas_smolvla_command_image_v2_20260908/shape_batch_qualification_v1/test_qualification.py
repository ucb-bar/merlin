from __future__ import annotations

import copy
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import qualify_shapes as q  # noqa: E402
import verify  # noqa: E402


def test_saved_qualification_is_complete_and_fail_closed() -> None:
    result = verify.verify()
    assert result["ok"]
    assert result["counts"]["unique_shapes_rtl_numeric_qualified"] == 3
    assert result["counts"]["unique_shapes_rtl_numeric_unqualified"] == 1
    assert result["counts"]["physical_partitions_qualified"] == 3
    assert result["counts"]["physical_partitions_unqualified"] == 388


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


def test_saved_batched_shape_is_rejected_by_rtl_vmem_assertion() -> None:
    receipt = q.load_json(
        HERE / "evidence/numeric/matmul_batched_15_50_64_113/receipt.json"
    )
    stderr = (
        HERE / "evidence/numeric/matmul_batched_15_50_64_113/raw_gsim_stderr.txt"
    ).read_text(encoding="utf-8")
    assert not receipt["qualified"]
    assert receipt["returncode"] == -6
    assert not receipt["assertion_clean"]
    assert "DMA VMEM transfer range exceeds VMEM capacity" in stderr


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
    test_saved_batched_shape_is_rejected_by_rtl_vmem_assertion()
    test_plan_cardinality_and_occurrences_are_exact()
    print("ok: 5 qualification/verifier tests")
