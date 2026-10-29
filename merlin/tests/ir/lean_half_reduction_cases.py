"""Ordinary pure reductions outside the f32 multiply-add policy."""


def reduction(fmt="bf16", operation="addf"):
    initializer = "0.0" if operation == "addf" else "1.0"
    return f""""builtin.module"() ({{
  "func.func"() <{{function_type = (tensor<4x{fmt}>) -> tensor<{fmt}>, sym_name = "reduce"}}> ({{
  ^bb0(%arg: tensor<4x{fmt}>):
    %zero = "arith.constant"() <{{value = dense<{initializer}> : tensor<{fmt}>}}> : () -> tensor<{fmt}>
    %out = "linalg.generic"(%arg, %zero) <{{
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>],
      iterator_types = [#linalg.iterator_type<reduction>], operandSegmentSizes = array<i32: 1, 1>}}> ({{
    ^bb0(%value: {fmt}, %acc: {fmt}):
      %sum = "arith.{operation}"(%value, %acc) <{{fastmath = #arith.fastmath<none>}}> : ({fmt}, {fmt}) -> {fmt}
      "linalg.yield"(%sum) : ({fmt}) -> ()
    }}) : (tensor<4x{fmt}>, tensor<{fmt}>) -> tensor<{fmt}>
    "func.return"(%out) : (tensor<{fmt}>) -> ()
  }}) : () -> ()
}}) : () -> ()
"""
