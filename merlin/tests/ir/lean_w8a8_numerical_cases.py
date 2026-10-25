"""Source-derived normalization and quantization fragments for candidate qualification."""

from __future__ import annotations

from merlin.frontends.linalg_mlir import parse_mlir_text


def reduction_source(shape, dimensions, *, variadic=False):
    """Named-reduction rule probes; these have no contraction rewrite sites."""

    def tensor(extents):
        return "tensor<" + "".join(f"{extent}x" for extent in extents) + "f32>"

    data = tensor(shape)
    output = tensor([extent for axis, extent in enumerate(shape) if axis not in dimensions])
    coordinates = ", ".join(f"d{axis}" for axis in range(len(shape)))
    identity = f"affine_map<({coordinates})->({coordinates})>"
    kinds = ", ".join('"parallel"' for _ in shape)
    if variadic:
        signature = f"({output}, {output})"
        declaration = f"%reduced:2 = linalg.reduce ins(%square, %square : {data}, {data})"
        destinations = f"outs(%initial, %initial : {output}, {output})"
        body = """(%x: f32, %y: f32, %acc: f32, %other: f32) {
          %sum = arith.addf %x, %acc : f32
          %second = arith.addf %y, %other : f32
          linalg.yield %sum, %second : f32, f32
        }"""
        returned = f"%reduced#0, %reduced#1 : {output}, {output}"
    else:
        signature = output
        declaration = f"%reduced = linalg.reduce ins(%square : {data})"
        destinations = f"outs(%initial : {output})"
        body = """(%x: f32, %acc: f32) {
          %sum = arith.addf %x, %acc : f32
          linalg.yield %sum : f32
        }"""
        returned = f"%reduced : {output}"
    # Keep this text for the authoritative MLIR parser: the installed xDSL ReduceOp
    # exposes a single input/init pair and cannot represent the variadic probe.
    return f"""module {{ func.func @forward(%input: {data}) -> {signature} {{
      %empty = tensor.empty() : {data}
      %square = linalg.generic {{indexing_maps = [{identity}, {identity}], iterator_types = [{kinds}]}}
        ins(%input : {data}) outs(%empty : {data}) {{
        ^bb0(%x: f32, %dest: f32):
          %product = arith.mulf %x, %x : f32
          linalg.yield %product : f32
        }} -> {data}
      %zero = arith.constant 0.0 : f32
      %initial = tensor.splat %zero : {output}
      {declaration} {destinations} dimensions = [{", ".join(map(str, dimensions))}] {body}
      return {returned}
    }} }}"""


def normalization_source(*, rows=2, width=3, epsilon="0.00001", scale="0.01", zero=0, zero_width=64):
    """Keep the retained RMSnorm operation order; vary dimensions independently of model names."""
    data = f"tensor<{rows}x{width}xf32>"
    reduced = f"tensor<{rows}xf32>"
    weight = f"tensor<{width}x2xf32>"
    output = f"tensor<{rows}x2xf32>"
    return parse_mlir_text(f"""func.func @forward(%input: {data}, %gain: tensor<{width}xf32>, %weight: {weight})
        -> {output} {{
      %zero = arith.constant 0.0 : f32
      %two = arith.constant 2.0 : f32
      %sqempty = tensor.empty() : {data}
      %square = linalg.generic {{indexing_maps = [affine_map<(d0,d1)->(d0,d1)>,
          affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel", "parallel"]}}
        ins(%input : {data}) outs(%sqempty : {data}) {{
        ^bb0(%x: f32, %dest: f32):
          %unused = arith.constant 2.0 : f32
          %squared = arith.mulf %x, %x : f32
          linalg.yield %squared : f32
      }} -> {data}
      %initial = tensor.splat %zero : {reduced}
      %sum = linalg.reduce ins(%square : {data}) outs(%initial : {reduced}) dimensions = [1]
        (%x: f32, %acc: f32) {{
          %next = arith.addf %x, %acc : f32
          linalg.yield %next : f32
        }}
      %denom = arith.constant {float(width)} : f32
      %denoms = tensor.splat %denom : {reduced}
      %meanempty = tensor.empty() : {reduced}
      %mean = linalg.generic {{indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>,
          affine_map<(d0)->(d0)>], iterator_types = ["parallel"]}}
        ins(%sum, %denoms : {reduced}, {reduced}) outs(%meanempty : {reduced}) {{
        ^bb0(%x: f32, %y: f32, %dest: f32):
          %quotient = arith.divf %x, %y : f32
          linalg.yield %quotient : f32
      }} -> {reduced}
      %eps = arith.constant {epsilon} : f32
      %epss = tensor.splat %eps : {reduced}
      %epsempty = tensor.empty() : {reduced}
      %shifted = linalg.generic {{indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>,
          affine_map<(d0)->(d0)>], iterator_types = ["parallel"]}}
        ins(%mean, %epss : {reduced}, {reduced}) outs(%epsempty : {reduced}) {{
        ^bb0(%x: f32, %y: f32, %dest: f32):
          %plus = arith.addf %x, %y : f32
          linalg.yield %plus : f32
      }} -> {reduced}
      %invempty = tensor.empty() : {reduced}
      %inverse = linalg.generic {{indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>],
          iterator_types = ["parallel"]}} ins(%shifted : {reduced}) outs(%invempty : {reduced}) {{
        ^bb0(%x: f32, %dest: f32):
          %rsqrt = math.rsqrt %x : f32
          linalg.yield %rsqrt : f32
      }} -> {reduced}
      %normempty = tensor.empty() : {data}
      %normalized = linalg.generic {{indexing_maps = [affine_map<(d0,d1)->(d0,d1)>,
          affine_map<(d0,d1)->(d0)>, affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel", "parallel"]}}
        ins(%input, %inverse : {data}, {reduced}) outs(%normempty : {data}) {{
        ^bb0(%x: f32, %y: f32, %dest: f32):
          %product = arith.mulf %x, %y : f32
          linalg.yield %product : f32
      }} -> {data}
      %gainempty = tensor.empty() : {data}
      %scaled = linalg.generic {{indexing_maps = [affine_map<(d0,d1)->(d1)>,
          affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel", "parallel"]}}
        ins(%gain, %normalized : tensor<{width}xf32>, {data}) outs(%gainempty : {data}) {{
        ^bb0(%x: f32, %y: f32, %dest: f32):
          %product = arith.mulf %x, %y : f32
          linalg.yield %product : f32
      }} -> {data}
      %scalev = arith.constant {scale} : f32
      %scale = tensor.splat %scalev : tensor<f32>
      %zerov = arith.constant {zero} : i{zero_width}
      %zerop = tensor.splat %zerov : tensor<i{zero_width}>
      %quantized = "quant_ext.quantize_per_tensor"(%scaled, %scale, %zerop)
        <{{quant_min = -128 : i64, quant_max = 127 : i64, output_dtype = "int8"}}>
        : ({data}, tensor<f32>, tensor<i{zero_width}>) -> tensor<{rows}x{width}xi8>
      %dequantized = "quant_ext.dequantize_per_tensor"(%quantized, %scale, %zerop)
        <{{quant_min = -128 : i64, quant_max = 127 : i64, input_dtype = "int8"}}>
        : (tensor<{rows}x{width}xi8>, tensor<f32>, tensor<i{zero_width}>) -> {data}
      %empty = tensor.empty() : {output}
      %filled = linalg.fill ins(%zero : f32) outs(%empty : {output}) -> {output}
      %result = linalg.matmul ins(%dequantized, %weight : {data}, {weight})
        outs(%filled : {output}) -> {output}
      return %result : {output}
    }}""")
