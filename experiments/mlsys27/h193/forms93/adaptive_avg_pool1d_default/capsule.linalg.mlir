module attributes {merlin_iface.version = "0.1", merlin_iface.target = "gemmini", merlin_iface.abi_version = "0.1"} {
  %X0 = merlin_iface.tensor {name = "X0", role = "input"} : tensor<2x3x6xi8>
  %Y0 = merlin_iface.window_sum %X0 {name = "Y0", mode = "adaptive", layout = "nchw", readout_scale = 1.0 : f32, output_dtype = "i8"} : (tensor<2x3x6xi8>) -> tensor<2x3x4xi8>
}
