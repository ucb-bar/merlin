"""Compile-only guards for the target-neutral weight-only contraction bridge."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

from mlir_oot.frontend.integer_prepare import prepare_dynamic_weight_only_text
from mlir_oot.gemmini_opt import Pipeline
from mlir_oot.lowering.plan import Buffer


CANDIDATE = Path(__file__).resolve().parents[1]
PROBE = CANDIDATE / "tests/fixtures/dynamic_weight_linear.mlir"
QDQ_PROBE = CANDIDATE / "tests/fixtures/qdq_weight_linear.mlir"


def _prepare(text: str) -> tuple[str, dict]:
    return prepare_dynamic_weight_only_text(text)


def test_structural_weight_only_linear_becomes_one_mesh_contraction() -> None:
    prepared, receipt = _prepare(PROBE.read_text(encoding="utf-8"))

    assert receipt["integer_pass_counts"] == {"contraction_int8": 1}
    assert receipt["selection"] == {
        "candidate_f32_contractions": 1,
        "eligible": 1,
        "eligible_dynamic_activation": 1,
        "refused": {},
    }
    # Exact generated activation conversion: division, nearest-even rounding,
    # symmetric saturation. This is part of the named numeric contract.
    assert prepared.count("math.roundeven") == 1
    assert "arith.divf" in prepared
    assert "arith.minimumf" in prepared
    assert "arith.maximumf" in prepared

    pipe = Pipeline(prepared, enable_source_conv=True).run()
    assert pipe.declined is None
    placements = pipe.plan.command_buffer["params"]["lane_placement"]
    assert sum(p["lane"] == "on_mesh" for p in placements) == 1
    assert [command["opcode"] for command in pipe.plan.command_buffer["commands"]] == [
        "RES_PACK", "MATMUL_RESIDENT", "COMMIT"]


def test_nonzero_weight_zero_point_is_an_audited_refusal() -> None:
    source = PROBE.read_text(encoding="utf-8").replace(
        "%c0_i32 = arith.constant 0 : i32",
        "%c0_i32 = arith.constant 1 : i32",
    )
    prepared, receipt = _prepare(source)

    assert receipt["integer_pass_counts"] == {"contraction_int8": 0}
    assert receipt["selection"]["eligible"] == 0
    assert receipt["selection"]["refused"] == {
        "weight_zero_point_not_proven_zero": 1,
    }
    assert prepared == source
    pipe = Pipeline(prepared, enable_source_conv=True).run()
    assert pipe.declined is None
    assert not pipe.plan.command_buffer["commands"]


def test_explicit_activation_qdq_is_reused_without_dynamic_requantization() -> None:
    prepared, receipt = _prepare(QDQ_PROBE.read_text(encoding="utf-8"))

    assert receipt["integer_pass_counts"] == {"contraction_int8": 1}
    assert receipt["integer_pass_report"]["contraction_int8"] == {
        "static_activation_reused": 1,
    }
    assert receipt["selection"] == {
        "candidate_f32_contractions": 1,
        "eligible": 1,
        "eligible_static_activation_qdq": 1,
        "refused": {},
    }
    # The one roundeven is the source quantize boundary lowered to upstream
    # linalg. The bridge emits no second activation quantizer/amax scan.
    assert prepared.count("math.roundeven") == 1
    pipe = Pipeline(prepared, enable_source_conv=True).run()
    assert pipe.declined is None
    assert len(pipe.plan.command_buffer["params"]["mesh_regions"]) == 1
    assert len(pipe.plan.command_buffer["commands"]) == 3


def test_scale_on_reduction_dimension_is_refused() -> None:
    source = (PROBE.read_text(encoding="utf-8")
              .replace("tensor<7xf32>", "tensor<5xf32>")
              .replace("tensor<7xi32>", "tensor<5xi32>")
              .replace("axis = 1 : i64", "axis = 0 : i64"))
    _prepared, receipt = _prepare(source)

    assert receipt["integer_pass_counts"] == {"contraction_int8": 0}
    assert receipt["selection"]["refused"] == {
        "weight_scale_varies_along_reduction": 1,
    }


def test_i1_storage_uses_one_addressable_byte_per_element() -> None:
    # The whole-program arena is byte addressed. Sub-byte packing is a separate
    # encoding decision; absent such an encoding, i1 has one byte of storage.
    assert Buffer("mask", [2, 17], "i1", "scratch").nbytes == 64


def test_cli_records_explicit_non_equivalence_contract(tmp_path: Path) -> None:
    command_buffer = tmp_path / "command_buffer.json"
    env = dict(os.environ, MERLIN_PYTHON=sys.executable)
    subprocess.run([
        str(CANDIDATE / "run-gemmini-opt"),
        "--dynamic-weight-only-contract",
        "symmetric_per_output_channel_roundeven_v1",
        f"--emit-command-buffer={command_buffer}",
        str(PROBE),
    ], check=True, env=env, capture_output=True, text=True)
    contract = json.loads(command_buffer.read_text(encoding="utf-8"))["params"][
        "dynamic_weight_only_contract"]

    assert contract["source_f32_bit_equivalence"] == "NOT_CLAIMED"
    assert contract["activation_contract"]["rounding"] == "round_nearest_even"
    assert contract["preparation"]["selection"]["eligible"] == 1
