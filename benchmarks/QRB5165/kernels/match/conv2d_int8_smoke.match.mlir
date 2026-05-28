// Linalg-DAG match pattern for the uint8 NHWC Conv2D smoke kernel.
//
// Recognises the canonical IREE form for `merlin.qnn.conv2d_uint8`
// (the custom op we use to mark a fully-baked uint8 Conv2D). The QNN
// passthrough plugin substitutes the matched op with the prebuilt
// uint8 ctxbin authored in `kernels/abi/conv2d_int8_smoke.qnn.cpp`.
//
// Kernel signature (per manifest): single uint8 input → single uint8
// output. Weight + bias are baked into the ctxbin.

^bb0(%arg0: tensor<1x8x8x3xui8>, %weight: tensor<3x3x3x4xui8>, %bias: tensor<4xi32>):
%init = tensor.empty() {"match.operation_name_only"} : tensor<1x6x6x4xui8>
%out = "merlin.qnn.conv2d_uint8"(%arg0, %weight, %bias, %init) {
    strides = dense<1> : tensor<2xi64>,
    dilations = dense<1> : tensor<2xi64>
  } : (tensor<1x8x8x3xui8>, tensor<3x3x3x4xui8>, tensor<4xi32>, tensor<1x6x6x4xui8>) -> tensor<1x6x6x4xui8>
