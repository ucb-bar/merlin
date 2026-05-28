// uint8 NHWC Conv2D fixture — IREE-style quantized conv body, the smallest
// non-trivial input the emitter needs to recognise to support real yolov8
// (which uses signed int8, but the structural pattern is the same).
//
// Shape: input [1,8,8,3] u8, weight [3,3,3,4] u8, bias [4] i32,
//        output [1,6,6,4] u8.
//
// This MLIR encodes per-tensor quant params via op attributes that the
// recogniser parses out — `merlin.qnn.input_qparams = {scale=0.05,
// offset=128}` etc. (Real IREE-emitted IR encodes them via
// `iree_encoding.set_encoding` + `arith.mulf` chains; the recogniser
// support for that lowering is tracked separately in tasks #102.)

module {
  func.func @conv2d_uint8(%input: tensor<1x8x8x3xui8>)
      -> tensor<1x6x6x4xui8> attributes {
        merlin.qnn.input_qparams  = {scale = 0.05 : f32, offset = 128 : i32},
        merlin.qnn.weight_qparams = {scale = 0.025 : f32, offset = 128 : i32},
        merlin.qnn.bias_qparams   = {scale = 0.00125 : f32, offset = 0 : i32},
        merlin.qnn.output_qparams = {scale = 0.10 : f32, offset = 128 : i32}
      } {
    %weight = arith.constant dense<129> : tensor<3x3x3x4xui8>
    %bias = arith.constant dense<0> : tensor<4xi32>

    %init = tensor.empty() : tensor<1x6x6x4xui8>
    %out = "merlin.qnn.conv2d_uint8"(%input, %weight, %bias, %init) {
        strides = dense<1> : tensor<2xi64>,
        dilations = dense<1> : tensor<2xi64>
      } : (tensor<1x8x8x3xui8>, tensor<3x3x3x4xui8>,
           tensor<4xi32>, tensor<1x6x6x4xui8>) -> tensor<1x6x6x4xui8>
    return %out : tensor<1x6x6x4xui8>
  }
}
