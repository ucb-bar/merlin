// RUN: iree-opt %s -split-input-file | FileCheck %s

// CHECK-LABEL: @qnn_conv2d_basic
func.func @qnn_conv2d_basic(%input: tensor<1x320x320x3xi8>,
                            %weight: tensor<3x3x3x16xi8>,
                            %bias: tensor<16xi32>) -> tensor<1x160x160x16xi32> {
  // CHECK: qnn.conv2d
  // CHECK-SAME: stride = [2 : i32, 2 : i32]
  // CHECK-SAME: pad_amount = [1 : i32, 1 : i32, 1 : i32, 1 : i32]
  // CHECK-SAME: dilation = [1 : i32, 1 : i32]
  // CHECK-SAME: group = 1
  %0 = qnn.conv2d ins(%input, %weight, %bias)
      attrs = {stride = [2 : i32, 2 : i32],
               pad_amount = [1 : i32, 1 : i32, 1 : i32, 1 : i32],
               dilation = [1 : i32, 1 : i32],
               group = 1 : i32}
      : (tensor<1x320x320x3xi8>, tensor<3x3x3x16xi8>, tensor<16xi32>)
        -> tensor<1x160x160x16xi32>
  return %0 : tensor<1x160x160x16xi32>
}

// -----

// CHECK-LABEL: @qnn_element_wise_neuron_relu
func.func @qnn_element_wise_neuron_relu(%input: tensor<1x10x10x16xi8>)
    -> tensor<1x10x10x16xi8> {
  // CHECK: qnn.element_wise_neuron
  // CHECK-SAME: op_kind = 1
  %0 = qnn.element_wise_neuron ins(%input) {op_kind = 1 : i32}
      : tensor<1x10x10x16xi8> -> tensor<1x10x10x16xi8>
  return %0 : tensor<1x10x10x16xi8>
}

// -----

// Verifier: NCHW input must be rejected.
// expected-error @+1 {{input must be rank 4 (NHWC)}}
func.func @qnn_conv2d_wrong_rank(%input: tensor<3x320x320xi8>,
                                  %weight: tensor<3x3x3x16xi8>)
    -> tensor<1x160x160x16xi32> {
  %0 = qnn.conv2d ins(%input, %weight)
      attrs = {stride = [2 : i32, 2 : i32],
               pad_amount = [1 : i32, 1 : i32, 1 : i32, 1 : i32],
               dilation = [1 : i32, 1 : i32],
               group = 1 : i32}
      : (tensor<3x320x320xi8>, tensor<3x3x3x16xi8>)
        -> tensor<1x160x160x16xi32>
  return %0 : tensor<1x160x160x16xi32>
}
