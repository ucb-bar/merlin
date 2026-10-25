"""Independent floating contraction geometries for ordered bound checks."""

from merlin.frontends.linalg_mlir import parse_mlir_text


def source(*, kind="matmul", m=2, k=3, n=4, batch=2, accumulator_first=False, initial=0.0):
    prefix = f"{batch}x" if kind == "batch_matmul" else ""
    left, right, result = (f"tensor<{prefix}{rows}x{cols}xf32>" for rows, cols in ((m, k), (k, n), (m, n)))
    if kind == "generic":
        addition = "%acc, %product" if accumulator_first else "%product, %acc"
        contraction = f"""linalg.generic {{
          indexing_maps = [affine_map<(d0,d1,d2)->(d0,d2)>, affine_map<(d0,d1,d2)->(d2,d1)>,
                           affine_map<(d0,d1,d2)->(d0,d1)>],
          iterator_types = ["parallel", "parallel", "reduction"]}}
          ins(%a, %b : {left}, {right}) outs(%filled : {result}) {{
        ^bb0(%x: f32, %y: f32, %acc: f32):
          %product = arith.mulf %x, %y : f32
          %sum = arith.addf {addition} : f32
          linalg.yield %sum : f32
        }} -> {result}"""
    elif kind == "batch_matmul":
        # xDSL has no registered custom parser for this upstream named operation.
        # Its generic form retains the actual body and exact operand segments.
        contraction = f""""linalg.batch_matmul"(%a, %b, %filled) <{{operandSegmentSizes = array<i32: 2, 1>}}> ({{
        ^bb0(%x: f32, %y: f32, %acc: f32):
          %product = arith.mulf %x, %y : f32
          %sum = arith.addf %product, %acc : f32
          linalg.yield %sum : f32
        }}) : ({left}, {right}, {result}) -> {result}"""
    else:
        assert kind == "matmul"
        contraction = f"linalg.{kind} ins(%a, %b : {left}, {right}) outs(%filled : {result}) -> {result}"
    return parse_mlir_text(f"""func.func @forward(%a: {left}, %b: {right}) -> {result} {{
      %empty = tensor.empty() : {result}
      %zero = arith.constant {initial} : f32
      %filled = linalg.fill ins(%zero : f32) outs(%empty : {result}) -> {result}
      %result = {contraction}
      return %result : {result}
    }}""")
