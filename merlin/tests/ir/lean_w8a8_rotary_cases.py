"""Independent geometries for source-derived rotary-bound qualification."""

from merlin.frontends.linalg_mlir import parse_mlir_text


def source(*, rows=8, channels=3, step=1, arithmetic="arith.mulf"):
    positions = f"tensor<{rows}xi64>"
    floats = f"tensor<{rows}xf32>"
    frequencies = f"tensor<{channels}xf32>"
    angles = f"tensor<{rows}x{channels}xf32>"
    transposed = f"tensor<{channels}x{rows}xf32>"
    joined = f"tensor<{channels}x{2 * rows}xf32>"
    weight = f"tensor<{channels}x2xf32>"
    result = f"tensor<{rows}x2xf32>"
    return parse_mlir_text(f"""func.func @forward(%frequency: {frequencies}, %weight: {weight}) -> {result} {{
      %position_empty = tensor.empty() : {positions}
      %positions = linalg.generic {{indexing_maps = [affine_map<(d0)->(d0)>], iterator_types = ["parallel"]}}
          outs(%position_empty : {positions}) {{
        ^bb0(%old: i64):
          %index = linalg.index 0 : index
          %integer = arith.index_cast %index : index to i64
          %step = arith.constant {step} : i64
          %scaled = arith.muli %integer, %step : i64
          %start = arith.constant 0 : i64
          %position = arith.addi %scaled, %start : i64
          linalg.yield %position : i64
      }} -> {positions}
      %float_empty = tensor.empty() : {floats}
      %floats = linalg.generic {{indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>],
          iterator_types = ["parallel"]}} ins(%positions : {positions}) outs(%float_empty : {floats}) {{
        ^bb0(%position: i64, %old: f32):
          %converted = arith.sitofp %position : i64 to f32
          linalg.yield %converted : f32
      }} -> {floats}
      %zero = arith.constant 0.0 : f32
      %initial = tensor.splat %zero : {angles}
      %angles = linalg.generic {{indexing_maps = [affine_map<(d0,d1)->(d1)>, affine_map<(d0,d1)->(d0)>,
          affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel", "parallel"]}}
          ins(%frequency, %floats : {frequencies}, {floats}) outs(%initial : {angles})
          attrs = {{prov.bound = "angles"}} {{
        ^bb0(%angle_frequency: f32, %angle_position: f32, %angle_initial: f32):
          %product = {arithmetic} %angle_frequency, %angle_position : f32
          %sum = arith.addf %angle_initial, %product : f32
          linalg.yield %sum : f32
      }} -> {angles}
      %transpose_empty = tensor.empty() : {transposed}
      %transposed = linalg.transpose ins(%angles : {angles}) outs(%transpose_empty : {transposed})
          permutation = [1, 0]
      %joined = tensor.concat dim(1) %transposed, %transposed : ({transposed}, {transposed}) -> {joined}
      %empty = tensor.empty() : {result}
      %filled = linalg.fill ins(%zero : f32) outs(%empty : {result}) -> {result}
      %scale_value = arith.constant 0.01 : f32
      %scale = tensor.splat %scale_value : tensor<f32>
      %zero_value = arith.constant 0 : i64
      %point = tensor.splat %zero_value : tensor<i64>
      %quantized = "quant_ext.quantize_per_tensor"(%angles, %scale, %point)
        <{{quant_min = -128 : i64, quant_max = 127 : i64, output_dtype = "int8"}}>
        : ({angles}, tensor<f32>, tensor<i64>) -> tensor<{rows}x{channels}xi8>
      %dequantized = "quant_ext.dequantize_per_tensor"(%quantized, %scale, %point)
        <{{quant_min = -128 : i64, quant_max = 127 : i64, input_dtype = "int8"}}>
        : (tensor<{rows}x{channels}xi8>, tensor<f32>, tensor<i64>) -> {angles}
      %result = linalg.matmul ins(%dequantized, %weight : {angles}, {weight}) outs(%filled : {result}) -> {result}
      return %result : {result}
    }}""")
