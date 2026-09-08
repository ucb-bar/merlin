"""Capability/semantics selector for the optional native convolution engine.

This is deliberately a pure structural decision: it sees target facts, tensor encodings and
operation semantics, never model or layer names.  Callers retain their existing generic route
when this function returns false.
"""
from __future__ import annotations

from ..tables import rtl_facts as F


def select_native_loop_conv(c, activation, weight, dst, *, epilogue_stages=(), bias=None,
                            bias_buffer=None, acc_scale=1.0
                            ) -> tuple[bool, str]:
    """Return whether ``c`` is exactly representable by the pinned LOOP_CONV_WS contract."""
    if not F.HAS_LOOP_CONV:
        return False, "target_has_no_loop_conv"
    if (c.output_dtype != F.OPERAND_DTYPE
            or activation.dtype != F.OPERAND_DTYPE
            or weight.dtype != F.OPERAND_DTYPE
            or dst.dtype != F.OPERAND_DTYPE):
        # LoopConvSt hardwires read_full=false; accepting i32 would silently saturate.
        return False, "loop_conv_store_is_narrow_only"
    stages = list(epilogue_stages)
    allowed = {"bias", "bias_add", "acc_scale", "relu"}
    if any(stage not in allowed for stage in stages):
        return False, "native_loop_conv_epilogue_not_representable"
    ordered = ["bias" if stage == "bias_add" else stage for stage in stages]
    rank = {"bias": 0, "acc_scale": 1, "relu": 2}
    if len(set(ordered)) != len(ordered) or ordered != sorted(ordered, key=rank.__getitem__):
        # LOOP_CONV initializes accumulator rows from LOAD3, then the store path scales and finally
        # applies activation. Membership in the stage set is insufficient: silently accepting a
        # different source order changes floating-point rounding and activation semantics.
        return False, "native_loop_conv_epilogue_order_must_be_bias_scale_activation"
    bias_stages = [i for i, stage in enumerate(stages) if stage in {"bias", "bias_add"}]
    if bias_stages and (len(bias_stages) != 1 or bias_stages[0] != 0):
        return False, "native_loop_conv_bias_must_initialize_accumulator"
    if bool(bias_stages) != (bias is not None):
        return False, "native_loop_conv_bias_stage_and_operand_disagree"
    if bias is not None:
        if bias_buffer is None or bias_buffer.dtype != F.ACCUMULATOR_DTYPE:
            return False, "native_loop_conv_bias_must_be_i32_accumulator_units"
        if bias_buffer.shape != [c.co]:
            return False, "native_loop_conv_bias_shape_mismatch"
    if "acc_scale" not in stages and float(acc_scale) != 1.0:
        return False, "native_loop_conv_scale_without_acc_scale_stage"
    if float(acc_scale) <= 0.0:
        return False, "native_loop_conv_acc_scale_must_be_positive"
    if (c.input_layout, c.weight_layout, c.output_layout) != ("NHWC", "HWIO", "NHWC"):
        return False, "native_layout_contract_not_satisfied"
    if c.kh != c.kw or c.stride_h != c.stride_w or c.dilation_h != c.dilation_w:
        return False, "loop_conv_requires_uniform_2d_geometry"
    if not (c.pad_top == c.pad_left == c.pad_bottom == c.pad_right):
        return False, "loop_conv_requires_uniform_padding"
    if c.pad_top < 0 or c.pad_top >= c.kh:
        return False, "native_loop_conv_requires_0_le_padding_lt_kernel"
    if min(c.batch, c.ci, c.hi, c.wi, c.co, c.kh, c.kw, c.ho, c.wo,
           c.stride_h, c.dilation_h) < 1:
        return False, "loop_conv_requires_positive_static_geometry"
    expected_activation = [c.batch, c.hi, c.wi, c.ci]
    expected_weights = ([c.kh, c.kw, c.ci, c.co], [c.kh * c.kw * c.ci, c.co])
    expected_outputs = ([c.batch, c.ho, c.wo, c.co], [c.batch * c.ho * c.wo, c.co])
    if (activation.shape != expected_activation
            or weight.shape not in expected_weights or dst.shape not in expected_outputs):
        return False, "buffer_shape_does_not_match_native_layout"
    # Widths come directly from gemmini_loop_conv_ws.  Checking the physical row strides here is
    # essential: channel padding can make them differ from logical C/CO.
    fields16 = (c.batch, c.ci, c.hi, c.wi, c.co, c.kh, c.kw, c.ho, c.wo,
                activation.pitch, weight.pitch, dst.pitch)
    if any(value >= (1 << 16) for value in fields16):
        return False, "native_loop_conv_16bit_field_overflow"
    if c.stride_h >= (1 << 8) or c.pad_top >= (1 << 8):
        return False, "native_loop_conv_8bit_field_overflow"
    if c.dilation_h >= (1 << 10):
        return False, "kernel_dilation_not_representable"
    if c.kh * c.kw * c.ci * 128 * 128 > (1 << 31) - 1:
        return False, "native_loop_conv_accumulator_may_overflow_i32"
    return True, "selected"


def select_native_loop_conv_compute_only(c, activation, weight, dst) -> tuple[bool, str]:
    """Admit LOOP_CONV as an i32 dot-product producer, without using its narrow store.

    The universal RTL's LoopConv store is narrow-only.  This route deliberately leaves the
    output pointer null, fences, and performs explicit full-width accumulator MVOUTs.  It is
    consequently independent of the native scale/bias/activation path and preserves the
    canonical PT2E epilogue on the host lane.

    Batch-one NCHW is representable by ``trans_input_3120``: with N=1, NCHW and CHWN have the
    same byte order.  We do not generalise that identity to N>1.  Weights are explicitly
    prepacked from OIHW to HWIO, and the i32 staging buffer has an explicit logical-NCHW /
    physical-NHWC encoding.
    """
    if not F.HAS_LOOP_CONV:
        return False, "target_has_no_loop_conv"
    if not F.HAS_TRAINING_CONVS:
        return False, "target_has_no_loop_conv_layout_transforms"
    if (activation.dtype != F.OPERAND_DTYPE or weight.dtype != F.OPERAND_DTYPE
            or c.output_dtype != F.ACCUMULATOR_DTYPE or dst.dtype != F.ACCUMULATOR_DTYPE):
        return False, "compute_only_requires_i8_i8_to_i32"
    if dst.role == "output":
        return False, "compute_only_physical_nhwc_must_not_escape_output_abi"
    if c.batch != 1:
        return False, "nchw_trans_input_3120_requires_batch_one"
    if (c.input_layout, c.weight_layout, c.output_layout) != (
            "NCHW_batch1", "HWIO", "NHWC_accumulator"):
        return False, "compute_only_layout_contract_not_satisfied"
    if c.bias is not None or c.relu or float(c.acc_scale) != 1.0:
        return False, "compute_only_must_not_own_epilogue"
    if c.kh != c.kw or c.stride_h != c.stride_w or c.dilation_h != c.dilation_w:
        return False, "loop_conv_requires_uniform_2d_geometry"
    if not (c.pad_top == c.pad_left == c.pad_bottom == c.pad_right):
        return False, "loop_conv_requires_uniform_padding"
    if c.pad_top < 0 or c.pad_top >= c.kh:
        return False, "native_loop_conv_requires_0_le_padding_lt_kernel"
    if min(c.batch, c.ci, c.hi, c.wi, c.co, c.kh, c.kw, c.ho, c.wo,
           c.stride_h, c.dilation_h) < 1:
        return False, "loop_conv_requires_positive_static_geometry"
    if activation.shape != [1, c.ci, c.hi, c.wi]:
        return False, "activation_shape_does_not_match_batch1_nchw"
    if activation.pitch != c.wi:
        return False, "trans_input_3120_requires_dense_nchw_rows"
    if weight.shape not in ([c.kh, c.kw, c.ci, c.co], [c.kh * c.kw * c.ci, c.co]):
        return False, "weight_shape_does_not_match_hwio"
    encoding = dst.storage_encoding or {}
    if (dst.shape != [1, c.co, c.ho, c.wo]
            or encoding.get("schema") != "permuted_axes_storage_v1"
            or encoding.get("permutation") != [0, 2, 3, 1]
            or encoding.get("physical_shape") != [1, c.ho, c.wo, c.co]):
        return False, "output_is_not_logical_nchw_physical_nhwc"
    fields16 = (c.batch, c.ci, c.hi, c.wi, c.co, c.kh, c.kw, c.ho, c.wo,
                activation.pitch, weight.pitch, dst.pitch)
    if any(value >= (1 << 16) for value in fields16):
        return False, "native_loop_conv_16bit_field_overflow"
    if c.stride_h >= (1 << 8) or c.pad_top >= (1 << 8):
        return False, "native_loop_conv_8bit_field_overflow"
    if c.dilation_h >= (1 << 10):
        return False, "kernel_dilation_not_representable"
    if c.kh * c.kw * c.ci * 128 * 128 > (1 << 31) - 1:
        return False, "native_loop_conv_accumulator_may_overflow_i32"
    return True, "selected_compute_only_full_width_mvout"
