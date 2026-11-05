module attributes {merlin_iface.version = "0.1", merlin_iface.target = "gemmini", merlin_iface.abi_version = "0.1"} {
  %X0 = merlin_iface.tensor {name = "X0", role = "input"} : tensor<1x4x4xi8>
  %Y0 = merlin_iface.window_sum %X0 {name = "Y0", mode = "fold", kernel = [2, 2], stride = [1, 1], padding = [0, 0], dilation = [1, 1], layout = "nchw", readout_scale = 1.0 : f32, output_dtype = "i32"} : (tensor<1x4x4xi8>) -> tensor<1x1x3x3xi32>
}
