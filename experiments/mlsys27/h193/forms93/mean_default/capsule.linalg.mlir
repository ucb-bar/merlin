module attributes {merlin_iface.version = "0.1", merlin_iface.target = "gemmini", merlin_iface.abi_version = "0.1"} {
  %X0 = merlin_iface.tensor {name = "X0", role = "input"} : tensor<4x8xi8>
  %Y0 = merlin_iface.window_sum %X0 {name = "Y0", mode = "reduce", dims = [0, 1], average = true, layout = "nchw", readout_scale = 1.0 : f32, output_dtype = "i8"} : (tensor<4x8xi8>) -> tensor<1x1xi8>
}
