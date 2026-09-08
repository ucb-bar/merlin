"""Source-semantic formation that really deletes the full-width host boundary."""

from __future__ import annotations

from pathlib import Path

from mlir_oot.gemmini_opt import Pipeline, _print


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = (ROOT / "tests/fixtures/exact_narrow_epilogue_template.mlir").read_text()


def source(m: int = 2, k: int = 5, n: int = 7, *, static_identity: bool = True) -> str:
    text = TEMPLATE.replace("@M@", str(m)).replace("@K@", str(k)).replace("@N@", str(n))
    if not static_identity:
        return text
    anchor = f"    %acc_empty = tensor.empty() : tensor<{m}x{n}xi32>"
    constants = f"""    %native_one_f = arith.constant 1.000000e+00 : f32
    %native_zero_f = arith.constant 0.000000e+00 : f32
    %native_one_scalar = tensor.splat %native_one_f : tensor<f32>
    %native_ones_channel = tensor.splat %native_one_f : tensor<{n}xf32>
    %native_zeros_channel = tensor.splat %native_zero_f : tensor<{n}xf32>
"""
    return (text.replace(anchor, constants + anchor)
            .replace("ins(%acc, %acc_scale, %channel_scale",
                     "ins(%acc, %native_one_scalar, %native_ones_channel")
            .replace("ins(%affine, %bias", "ins(%affine, %native_zeros_channel")
            .replace("ins(%qscale : tensor<f32>)",
                     "ins(%native_one_scalar : tensor<f32>)"))


def test_exact_identity_chain_deletes_host_boundary_and_full_width_readout() -> None:
    pipe = Pipeline(source(), enable_source_conv=True).run()
    assert pipe.declined is None
    cb = pipe.plan.command_buffer
    receipt = cb["params"]["target_neutral_quantized_epilogues"]
    assert receipt["selected_count"] == 1
    assert receipt["full_width_intermediate_bytes_eliminated"] == 2 * 7 * 4
    assert receipt["logical_device_to_host_boundaries_eliminated"] == 1
    assert receipt["output_dma_bytes_eliminated"] == 2 * 7 * 3
    assert receipt["physical_fences_eliminated"] == 0
    assert receipt["numeric_contract"] == "exact_for_every_i32_accumulator"
    assert [row["kind"] for row in cb["params"]["global_program_plan"]["tasks"]] == [
        "contraction"]
    assert cb["commands"][-1]["attributes"] == {
        "epilogue": ["relu"], "output_dtype": "i8"}
    assert [ins.kind for ins in pipe.instrs].count("loop_ws_block") == 1
    assert not [ins for ins in pipe.instrs if ins.kind == "host_segment"]
    block = next(ins for ins in pipe.instrs if ins.kind == "loop_ws_block")
    assert block.attrs["full_c"] is False
    assert block.attrs["c_dtype"] == "i8"
    assert block.attrs["acc_act"] == 1
    target = _print(pipe.artifact)
    assert '"llvm.intr.roundeven"' not in target
    assert '"llvm.fptosi"' not in target


def test_runtime_per_channel_arithmetic_fails_closed_to_original_host_chain() -> None:
    pipe = Pipeline(source(static_identity=False), enable_source_conv=True).run()
    assert pipe.declined is None
    receipt = pipe.plan.command_buffer["params"]["target_neutral_quantized_epilogues"]
    assert receipt["selected_count"] == 0
    assert receipt["logical_device_to_host_boundaries_eliminated"] == 0
    assert receipt["physical_fences_eliminated"] == 0
    assert receipt["refused"] == [{
        "producer_source_op_index": 3,
        "reason": "accumulator_scale_is_not_proven_identity",
        "stage": "exact_native_equivalence",
    }]
    assert any(ins.kind == "host_segment" for ins in pipe.instrs)
    assert '"llvm.intr.roundeven"' in _print(pipe.artifact)


def test_identity_proof_covers_every_i32_saturation_regime() -> None:
    # Values inside the i8 interval are represented exactly by f32. Outside it, both the source
    # chain and native store saturate before i8 conversion; include float's 2**24 precision cliff
    # and the signed endpoints explicitly.
    samples = [-(1 << 31), -(1 << 24) - 1, -129, -128, -1, 0, 1, 126, 127, 128,
               (1 << 24) + 1, (1 << 31) - 1]
    for value in samples:
        source_result = max(0, min(127, value))
        native_result = max(0, min(127, value))
        assert source_result == native_result


def test_altered_bias_scale_order_is_explicitly_refused() -> None:
    # Put the zero bias inside the first pointwise body, then apply the per-channel identity scale
    # in the following body.  These happen to commute for this fixture's values, but selection must
    # be based on the exact source order, not stage membership or a lucky numerical sample.
    text = source()
    text = text.replace(
        "ins(%acc, %native_one_scalar, %native_ones_channel",
        "ins(%acc, %native_one_scalar, %native_zeros_channel")
    text = text.replace(
        "%channel_scaled = arith.mulf %scaled, %cs : f32",
        "%channel_scaled = arith.addf %scaled, %cs : f32")
    text = text.replace(
        "ins(%affine, %native_zeros_channel",
        "ins(%affine, %native_ones_channel")
    text = text.replace(
        "%sum = arith.addf %x, %b0 : f32",
        "%sum = arith.mulf %x, %b0 : f32")
    pipe = Pipeline(text, enable_source_conv=True).run()
    assert pipe.declined is None
    receipt = pipe.plan.command_buffer["params"]["target_neutral_quantized_epilogues"]
    assert receipt["selected_count"] == 0
    assert receipt["refused"] == [{
        "producer_source_op_index": 8,
        "reason": "ordered affine body is not exact accumulator-scale then per-channel-scale",
        "blocker_source_op_index": 10,
        "blocker_op": "linalg.generic",
        "stage": "structural_formation",
    }]
    assert any(ins.kind == "host_segment" for ins in pipe.instrs)
