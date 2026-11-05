module attributes {merlin_iface.version = "0.1", merlin_iface.target = "gemmini", merlin_iface.abi_version = "0.1"} {
  %X0 = merlin_iface.tensor {name = "X0", role = "input"} : tensor<1x2x7x7xi8>
  %Y0 = merlin_iface.window_sum %X0 {name = "Y0", mode = "window", kernel = [2, 2], stride = [2, 2], padding = [0, 0], count_include_pad = false, ceil_mode = true, layout = "nchw", readout_scale = 1.0 : f32, output_dtype = "i8"} : (tensor<1x2x7x7xi8>) -> tensor<1x2x4x4xi8>
}
