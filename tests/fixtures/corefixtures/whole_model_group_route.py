from __future__ import annotations

_ID = "affine_map<(d0, d1) -> (d0, d1)>"
_COL = "affine_map<(d0, d1) -> (d1)>"
_ROW = "affine_map<(d0, d1) -> (d0)>"


def _layer(tag: str, activation: str, m: int, k: int, n: int, *, bias: str, bias_map: str = _COL) -> list[str]:
    """dq(x), dq(w) -> matmul -> bias -> relu -> quantize, in the fake-quantized form a capture has."""
    t = tag
    return [
        f'    %wd{t} = "quant_ext.dequantize_per_tensor"(%w{t}, %s, %z) <{{quant_min = -127 : i64, quant_max = 127 : i64}}> : (tensor<{k}x{n}xi8>, tensor<f32>, tensor<i64>) -> tensor<{k}x{n}xf32>',
        f'    %xd{t} = "quant_ext.dequantize_per_tensor"({activation}, %s, %z) <{{quant_min = -128 : i64, quant_max = 127 : i64}}> : (tensor<{m}x{k}xi8>, tensor<f32>, tensor<i64>) -> tensor<{m}x{k}xf32>',
        f"    %e0{t} = tensor.empty() : tensor<{m}x{n}xf32>",
        f"    %f{t} = linalg.fill ins(%c0 : f32) outs(%e0{t} : tensor<{m}x{n}xf32>) -> tensor<{m}x{n}xf32>",
        f'    %mm{t} = linalg.matmul {{prov.region_id = "matmul_{t}", prov.op = "matmul", prov.family = "contraction", prov.orig_dtype = "int8"}} ins(%xd{t}, %wd{t} : tensor<{m}x{k}xf32>, tensor<{k}x{n}xf32>) outs(%f{t} : tensor<{m}x{n}xf32>) -> tensor<{m}x{n}xf32>',
        f"    %e1{t} = tensor.empty() : tensor<{m}x{n}xf32>",
        f'    %ba{t} = linalg.generic {{indexing_maps = [{_ID}, {bias_map}, {_ID}], iterator_types = ["parallel", "parallel"]}} ins(%mm{t}, {bias} : tensor<{m}x{n}xf32>, tensor<{(n if bias_map == _COL else m)}xf32>) outs(%e1{t} : tensor<{m}x{n}xf32>) {{',
        "    ^bb0(%p: f32, %q0: f32, %o: f32):",
        "      %r = arith.addf %p, %q0 : f32",
        "      linalg.yield %r : f32",
        f"    }} -> tensor<{m}x{n}xf32>",
        f"    %e2{t} = tensor.empty() : tensor<{m}x{n}xf32>",
        f'    %relu{t} = linalg.generic {{indexing_maps = [{_ID}, {_ID}], iterator_types = ["parallel", "parallel"]}} ins(%ba{t} : tensor<{m}x{n}xf32>) outs(%e2{t} : tensor<{m}x{n}xf32>) {{',
        "    ^bb0(%p: f32, %o: f32):",
        "      %zero = arith.constant 0.000000e+00 : f32",
        "      %r = arith.maximumf %p, %zero : f32",
        "      linalg.yield %r : f32",
        f"    }} -> tensor<{m}x{n}xf32>",
        f'    %q{t} = "quant_ext.quantize_per_tensor"(%relu{t}, %s, %z) <{{quant_min = -128 : i64, quant_max = 127 : i64, output_dtype = "int8"}}> : (tensor<{m}x{n}xf32>, tensor<f32>, tensor<i64>) -> tensor<{m}x{n}xi8>',
    ]


def _two_layers(*, second_bias_map: str = _COL) -> str:
    """Two chained quantized linear layers: the smallest module that can tell one call from two."""
    rows = second_bias_map == _ROW
    return "\n".join(
        [
            "builtin.module {",
            "  func.func @forward(%x: tensor<4x8xi8>, %wa: tensor<8x16xi8>, %biasa: tensor<16xf32>, %wb: tensor<16x32xi8>, %biasb: tensor<32xf32>, %biasr: tensor<4xf32>) -> tensor<4x32xi8> {",
            "    %s = arith.constant dense<5.000000e-01> : tensor<f32>",
            "    %z = arith.constant dense<0> : tensor<i64>",
            "    %c0 = arith.constant 0.000000e+00 : f32",
            *_layer("a", "%x", 4, 8, 16, bias="%biasa"),
            *_layer("b", "%qa", 4, 16, 32, bias="%biasr" if rows else "%biasb", bias_map=second_bias_map),
            "    func.return %qb : tensor<4x32xi8>",
            "  }",
            "}",
        ]
    )
