"""A fake-quantized convolution in the form a capture has: host gather, reshape, weight on the left.

``module`` prints dq(x) -> pad -> windowed gather -> reshape, dq(w) -> reshape, ``W @ patches``,
reshape back to NCHW, bias over the channel axis, relu, quantize. The stored tensor is the LEFT
operand and the bias runs along the result's ROWS, which is the transpose of the form a unit runs.
"""

from __future__ import annotations


def module(
    *,
    channels: int = 2,
    out_channels: int = 4,
    image: int = 4,
    taps: int = 3,
    stride: int = 1,
    pad: int = 1,
    bias_axis: int = 1,
) -> str:
    padded = image + 2 * pad
    out = (padded - taps) // stride + 1
    reduced, positions = channels * taps * taps, out * out
    step = f"d4 * {stride} + d1" if stride != 1 else "d4 + d1"
    step_w = f"d5 * {stride} + d2" if stride != 1 else "d5 + d2"
    six = "d0, d1, d2, d3, d4, d5"
    nchw = f"1x{out_channels}x{out}x{out}"
    bias_extent = out_channels if bias_axis == 1 else out
    lines = [
        "builtin.module {",
        f"  func.func @forward(%x: tensor<1x{channels}x{image}x{image}xi8>, "
        f"%w: tensor<{out_channels}x{channels}x{taps}x{taps}xi8>, %b: tensor<{bias_extent}xf32>) "
        f"-> tensor<{nchw}xi8> {{",
        "    %s = arith.constant dense<5.000000e-01> : tensor<f32>",
        "    %z = arith.constant dense<0> : tensor<i64>",
        '    %xd = "quant_ext.dequantize_per_tensor"(%x, %s, %z) <{quant_min = -128 : i64, quant_max = 127 : i64}> : '
        f"(tensor<1x{channels}x{image}x{image}xi8>, tensor<f32>, tensor<i64>) -> tensor<1x{channels}x{image}x{image}xf32>",
        '    %wd = "quant_ext.dequantize_per_tensor"(%w, %s, %z) <{quant_min = -127 : i64, quant_max = 127 : i64}> : '
        f"(tensor<{out_channels}x{channels}x{taps}x{taps}xi8>, tensor<f32>, tensor<i64>) -> "
        f"tensor<{out_channels}x{channels}x{taps}x{taps}xf32>",
        f"    %pe = tensor.empty() : tensor<1x{channels}x{padded}x{padded}xf32>",
        '    %px = "tensor.insert_slice"(%xd, %pe) <{static_offsets = array<i64: 0, 0, '
        f"{pad}, {pad}>, static_sizes = array<i64: 1, {channels}, {image}, {image}>, static_strides = "
        "array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> : "
        f"(tensor<1x{channels}x{image}x{image}xf32>, tensor<1x{channels}x{padded}x{padded}xf32>) -> "
        f"tensor<1x{channels}x{padded}x{padded}xf32>",
        f"    %ge = tensor.empty() : tensor<{channels}x{taps}x{taps}x1x{out}x{out}xf32>",
        f"    %g = linalg.generic {{indexing_maps = [affine_map<({six}) -> (d3, d0, {step}, {step_w})>, "
        f'affine_map<({six}) -> ({six})>], iterator_types = ["parallel", "parallel", "parallel", "parallel", '
        f'"parallel", "parallel"]}} ins(%px : tensor<1x{channels}x{padded}x{padded}xf32>) '
        f"outs(%ge : tensor<{channels}x{taps}x{taps}x1x{out}x{out}xf32>) {{",
        "    ^bb0(%p: f32, %o: f32):",
        "      linalg.yield %p : f32",
        f"    }} -> tensor<{channels}x{taps}x{taps}x1x{out}x{out}xf32>",
        f"    %gf = tensor.collapse_shape %g [[0, 1, 2, 3, 4, 5]] : tensor<{channels}x{taps}x{taps}x1x{out}x{out}xf32> "
        f"into tensor<{reduced * positions}xf32>",
        f"    %gm = tensor.expand_shape %gf [[0, 1]] output_shape [{reduced}, {positions}] : "
        f"tensor<{reduced * positions}xf32> into tensor<{reduced}x{positions}xf32>",
        f"    %wf = tensor.collapse_shape %wd [[0, 1, 2, 3]] : tensor<{out_channels}x{channels}x{taps}x{taps}xf32> "
        f"into tensor<{out_channels * reduced}xf32>",
        f"    %wm = tensor.expand_shape %wf [[0, 1]] output_shape [{out_channels}, {reduced}] : "
        f"tensor<{out_channels * reduced}xf32> into tensor<{out_channels}x{reduced}xf32>",
        f"    %e0 = tensor.empty() : tensor<{out_channels}x{positions}xf32>",
        "    %c0 = arith.constant 0.000000e+00 : f32",
        f"    %f = linalg.fill ins(%c0 : f32) outs(%e0 : tensor<{out_channels}x{positions}xf32>) -> "
        f"tensor<{out_channels}x{positions}xf32>",
        f"    %mm = linalg.matmul ins(%wm, %gm : tensor<{out_channels}x{reduced}xf32>, "
        f"tensor<{reduced}x{positions}xf32>) outs(%f : tensor<{out_channels}x{positions}xf32>) -> "
        f"tensor<{out_channels}x{positions}xf32>",
        f"    %yf = tensor.collapse_shape %mm [[0, 1]] : tensor<{out_channels}x{positions}xf32> into "
        f"tensor<{out_channels * positions}xf32>",
        f"    %y = tensor.expand_shape %yf [[0, 1, 2, 3]] output_shape [1, {out_channels}, {out}, {out}] : "
        f"tensor<{out_channels * positions}xf32> into tensor<{nchw}xf32>",
        f"    %e1 = tensor.empty() : tensor<{nchw}xf32>",
        "    %ba = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, "
        f"affine_map<(d0, d1, d2, d3) -> (d{bias_axis})>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], "
        'iterator_types = ["parallel", "parallel", "parallel", "parallel"]} '
        f"ins(%y, %b : tensor<{nchw}xf32>, tensor<{bias_extent}xf32>) outs(%e1 : tensor<{nchw}xf32>) {{",
        "    ^bb0(%p: f32, %q0: f32, %o: f32):",
        "      %r = arith.addf %p, %q0 : f32",
        "      linalg.yield %r : f32",
        f"    }} -> tensor<{nchw}xf32>",
        f"    %e2 = tensor.empty() : tensor<{nchw}xf32>",
        "    %relu = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, "
        "affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = "
        f'["parallel", "parallel", "parallel", "parallel"]}} ins(%ba : tensor<{nchw}xf32>) '
        f"outs(%e2 : tensor<{nchw}xf32>) {{",
        "    ^bb0(%p: f32, %o: f32):",
        "      %zero = arith.constant 0.000000e+00 : f32",
        "      %r = arith.maximumf %p, %zero : f32",
        "      linalg.yield %r : f32",
        f"    }} -> tensor<{nchw}xf32>",
        '    %out = "quant_ext.quantize_per_tensor"(%relu, %s, %z) <{quant_min = -128 : i64, quant_max = 127 : i64, '
        f'output_dtype = "int8"}}> : (tensor<{nchw}xf32>, tensor<f32>, tensor<i64>) -> tensor<{nchw}xi8>',
        f"    func.return %out : tensor<{nchw}xi8>",
        "  }",
        "}",
    ]
    return "\n".join(lines)
