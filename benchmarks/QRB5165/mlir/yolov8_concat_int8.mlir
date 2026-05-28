// Standalone NCHW int8 concat dispatch — yolov8 has 17 of these
// (mostly along the channel dim, dim=1 in NCHW).
//
// Pattern: 2+ i8 NCHW inputs → dequant generics → tensor.concat (f32)
// → return f32. The QNN lowering uses Concat directly on the i8
// tensors with a Dequantize tail.

module {
  func.func @yolov8_concat_int8(
      %a: tensor<1x16x80x80xi8>,
      %b: tensor<1x16x80x80xi8>)
      -> tensor<1x32x80x80xf32> {
    %cst_in_scale = arith.constant 1.000000e-01 : f32

    %deq_init = tensor.empty() : tensor<1x16x80x80xf32>
    %deq_a = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
        ],
        iterator_types = ["parallel", "parallel", "parallel", "parallel"]
      } ins(%a : tensor<1x16x80x80xi8>)
        outs(%deq_init : tensor<1x16x80x80xf32>) {
      ^bb0(%in: i8, %y: f32):
        %f = arith.sitofp %in : i8 to f32
        %s = arith.mulf %f, %cst_in_scale : f32
        linalg.yield %s : f32
    } -> tensor<1x16x80x80xf32>

    %deq_b = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
        ],
        iterator_types = ["parallel", "parallel", "parallel", "parallel"]
      } ins(%b : tensor<1x16x80x80xi8>)
        outs(%deq_init : tensor<1x16x80x80xf32>) {
      ^bb0(%in: i8, %y: f32):
        %f = arith.sitofp %in : i8 to f32
        %s = arith.mulf %f, %cst_in_scale : f32
        linalg.yield %s : f32
    } -> tensor<1x16x80x80xf32>

    %concat = tensor.concat dim(1) %deq_a, %deq_b
        : (tensor<1x16x80x80xf32>, tensor<1x16x80x80xf32>)
        -> tensor<1x32x80x80xf32>

    return %concat : tensor<1x32x80x80xf32>
  }
}
