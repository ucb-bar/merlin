module {
func.func @padded(%input: tensor<2x3xf32>) -> tensor<4x5xi8> {
 %zero = arith.constant 0 : i8
 %empty = tensor.empty() : tensor<2x3xi8>
 %pad = tensor.splat %zero : tensor<4x5xi8>
 %q = linalg.generic {indexing_maps=[affine_map<(i,j)->(i,j)>,affine_map<(i,j)->(i,j)>],iterator_types=["parallel","parallel"]} ins(%input : tensor<2x3xf32>) outs(%empty : tensor<2x3xi8>) {
 ^bb0(%x: f32,%unused: i8):
  %y = arith.fptosi %x : f32 to i8
  linalg.yield %y : i8
 } -> tensor<2x3xi8>
 %result = "tensor.insert_slice"(%q,%pad) <{static_offsets=array<i64:1,1>,static_sizes=array<i64:2,3>,static_strides=array<i64:1,1>,operandSegmentSizes=array<i32:1,1,0,0,0>}> : (tensor<2x3xi8>,tensor<4x5xi8>) -> tensor<4x5xi8>
 return %result : tensor<4x5xi8>
}
}
