module attributes {merlin_iface.version = "0.1", merlin_iface.target = "gemmini", merlin_iface.abi_version = "0.1"} {
  %X = merlin_iface.tensor {name = "X", role = "input"} : tensor<1x1x3x3x3xi8>
  %Y0 = merlin_iface.movement %X {name = "Y0", form = "pad", low = [0, 0, 1, 1, 1], high = [0, 0, 1, 1, 1], mode = "reflect", output_dtype = "i8"} : (tensor<1x1x3x3x3xi8>) -> tensor<1x1x5x5x5xi8>
}
